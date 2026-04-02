#!/usr/bin/env python3
"""현물 일봉 Risk-On + 선물 all OFF일 때만 숏 오버레이 테스트.

목적:
- 업비트 현물 롱이 유지되는 동안
- 바이낸스 선물 4h1/4h2/1h1 이 모두 OFF면
- 선물 숏 sleeve가 의미가 있는지 확인

주의:
- 이 스크립트는 '숏 sleeve 자체'를 테스트한다.
- 현물 포트폴리오 수익률까지 합쳐서 총합 포트폴리오를 만들지는 않는다.
- 즉 "이 조건에서 숏을 켜는 것이 유효한가"를 먼저 보는 단계다.
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import get_mcap, load_data
from coin_engine import load_data as load_coin_daily_data
from run_short_per_canary_off_test import STRATEGIES, ShortOnlyEngine
from run_stoploss_test import START, END


CANDIDATES = [
    dict(name="btc_25", mode="btc", short_weight=0.25),
    dict(name="btc_50", mode="btc", short_weight=0.50),
    dict(name="top3_donchian50_25", mode="dynamic", short_weight=0.25, topn=3, rule="donchian50"),
    dict(name="top3_donchian50_50", mode="dynamic", short_weight=0.50, topn=3, rule="donchian50"),
    dict(name="top5_sma200_25", mode="dynamic", short_weight=0.25, topn=5, rule="sma200"),
    dict(name="top5_sma200_50", mode="dynamic", short_weight=0.50, topn=5, rule="sma200"),
]


def _coin_spot_risk_on_daily():
    prices, _ = load_coin_daily_data(top_n=50)
    btc = prices["BTC-USD"]["Close"].loc[START:END].copy()
    sma50 = btc.rolling(50).mean()
    upper = 1.015
    lower = 0.985
    prev = None
    out = {}
    for dt in btc.index:
        cur = float(btc.loc[dt])
        sma = float(sma50.loc[dt]) if pd.notna(sma50.loc[dt]) else np.nan
        if not np.isfinite(sma):
            out[dt.normalize()] = False
            continue

        # crash breaker: worst daily return in last 3 days <= -10%
        loc = btc.index.get_loc(dt)
        if loc >= 3:
            rets = btc.iloc[loc - 3 : loc + 1].pct_change().dropna()
            if len(rets) and float(rets.min()) <= -0.10:
                prev = False
                out[dt.normalize()] = False
                continue

        if prev is None:
            risk_on = cur > sma
        elif prev:
            risk_on = not (cur < sma * lower)
        else:
            risk_on = cur > sma * upper
        out[dt.normalize()] = risk_on
        prev = risk_on
    return out


def _build_canary_off_series(data, strat_name, base_index):
    cfg = STRATEGIES[strat_name]
    bars, _ = data[cfg["interval"]]
    btc = bars["BTC"]
    dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    close = btc["Close"].values
    prev_canary = False
    vals = {}
    for date in dates:
        ci = btc.index.get_loc(date)
        if ci < cfg["sma_bars"]:
            canary = False
        else:
            sma = float(np.mean(close[ci - cfg["sma_bars"] + 1 : ci + 1]))
            ratio = float(close[ci] / sma) if sma > 0 else np.nan
            if prev_canary:
                canary = ratio >= (1.0 - cfg["canary_hyst"])
            else:
                canary = ratio > (1.0 + cfg["canary_hyst"])
        vals[date] = not canary
        prev_canary = canary
    s = pd.Series(vals, dtype=bool).sort_index()
    return s.reindex(base_index, method="ffill").fillna(False)


def _build_rule_cache(bars, interval):
    bpd = {"4h": 6, "1h": 24}[interval]
    cache = {}
    for coin, df in bars.items():
        close = df["Close"].astype(float)
        low = df["Low"].astype(float)
        cache[coin] = pd.DataFrame(
            {
                "close": close,
                "sma200": close.rolling(200 * bpd).mean(),
                "donchian50": low.shift(1).rolling(50 * bpd).min(),
            }
        )
    return cache


def _dynamic_short_target(date, topn, rule, rule_cache):
    picks = []
    for coin in get_mcap(date)[:topn]:
        ind = rule_cache.get(coin)
        if ind is None:
            continue
        ci = ind.index.get_indexer([date], method="ffill")[0]
        if ci < 0:
            continue
        row = ind.iloc[ci]
        px = float(row["close"])
        if rule == "sma200":
            sma = row["sma200"]
            if pd.notna(sma) and px < float(sma):
                picks.append(coin)
        elif rule == "donchian50":
            dc = row["donchian50"]
            if pd.notna(dc) and px < float(dc):
                picks.append(coin)
    return picks


def build_target_series(data, candidate):
    bars, _ = data["1h"]
    base_index = bars["BTC"].index[(bars["BTC"].index >= START) & (bars["BTC"].index <= END)]
    spot_on_daily = _coin_spot_risk_on_daily()
    off_4h1 = _build_canary_off_series(data, "4h1", base_index)
    off_4h2 = _build_canary_off_series(data, "4h2", base_index)
    off_1h1 = _build_canary_off_series(data, "1h1", base_index)
    all_off = off_4h1 & off_4h2 & off_1h1
    rule_cache = _build_rule_cache(bars, "1h")
    out = []

    for date in base_index:
        spot_on = bool(spot_on_daily.get(date.normalize(), False))
        if not (spot_on and bool(all_off.loc[date])):
            out.append((date, {"CASH": 1.0}))
            continue

        if candidate["mode"] == "btc":
            out.append((date, {"BTC": candidate["short_weight"], "CASH": 1.0 - candidate["short_weight"]}))
            continue

        picks = _dynamic_short_target(date, candidate["topn"], candidate["rule"], rule_cache)
        if not picks:
            out.append((date, {"CASH": 1.0}))
            continue
        w = candidate["short_weight"] / len(picks)
        target = {coin: w for coin in picks}
        target["CASH"] = 1.0 - candidate["short_weight"]
        out.append((date, target))
    return out


def main():
    t0 = time.time()
    print("Loading futures data...")
    data = {iv: load_data(iv) for iv in ["4h", "1h"]}
    bars, funding = data["1h"]
    engine = ShortOnlyEngine(
        bars,
        funding,
        leverage=1.0,
        tx_cost=0.0004,
        maint_rate=0.004,
        initial_capital=10000.0,
    )

    rows = []
    for cand in CANDIDATES:
        target_series = build_target_series(data, cand)
        m = engine.run(target_series)
        row = {
            "name": cand["name"],
            "Cal": m.get("Cal", 0),
            "CAGR": m.get("CAGR", 0),
            "MDD": m.get("MDD", 0),
            "Sharpe": m.get("Sharpe", 0),
            "Liq": m.get("Liq", 0),
            "Rebal": m.get("Rebal", 0),
        }
        rows.append(row)
        print(
            f"{row['name']:<22} Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
            f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Rebal={row['Rebal']}"
        )

    rows.sort(key=lambda r: (-r["Cal"], -r["Sharpe"], r["name"]))
    print("\nTop candidates")
    for row in rows:
        print(
            f"- {row['name']}: Cal={row['Cal']:.2f}, "
            f"CAGR={row['CAGR']:+.1%}, MDD={row['MDD']:+.1%}"
        )
    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
