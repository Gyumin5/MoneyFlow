#!/usr/bin/env python3
"""공개 아이디어 확장판 숏 benchmark.

원칙:
- 4h1 / 4h2 / 1h1 OFF 를 각각 독립적으로 사용
- 각 전략은 자기 시간축에서 평가
- 유니버스는 고정 코인이 아니라 매 시점 시총 상위 N개
- 공개 전략 개념은 bar-based로 환산해서 사용
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import get_mcap, load_data
from run_short_per_canary_off_test import STRATEGIES, ShortOnlyEngine
from run_stoploss_test import END, START


UNIVERSE_SIZES = [3, 5, 8]
RULES = [
    dict(name="sma200", desc="Close < SMA200d"),
    dict(name="tsmom252", desc="252d TSMOM < 0"),
    dict(name="donchian50", desc="50d 저점 이탈"),
    dict(name="sma200_and_tsmom252", desc="Close < SMA200d and 252d TSMOM < 0"),
    dict(name="ema50_200_macd", desc="EMA50d<EMA200d and MACD hist<0"),
    dict(name="bear_rsi60", desc="하락장 + RSI14>60"),
    dict(name="bear_rsi65", desc="하락장 + RSI14>65"),
    dict(name="bear_dyn_resist", desc="하락장 + EMA10/SMA20 저항"),
    dict(name="bear_bb_upper", desc="하락장 + 볼밴 상단"),
]


def _parse_universe_sizes():
    raw = os.getenv("SHORT_TOPS", "").strip()
    if not raw:
        return UNIVERSE_SIZES
    return [int(x) for x in raw.split(",") if x.strip()]


def _parse_rules():
    raw = os.getenv("SHORT_RULES", "").strip()
    if not raw:
        return RULES
    wanted = {x.strip() for x in raw.split(",") if x.strip()}
    return [r for r in RULES if r["name"] in wanted]


def _bars_per_day(interval):
    return {"4h": 6, "1h": 24}[interval]


def _ema(arr, span):
    return pd.Series(arr).ewm(span=span, adjust=False).mean().values


def _rsi(arr, length=14):
    s = pd.Series(arr)
    delta = s.diff()
    up = delta.clip(lower=0.0)
    down = -delta.clip(upper=0.0)
    avg_up = up.ewm(alpha=1 / length, adjust=False).mean()
    avg_down = down.ewm(alpha=1 / length, adjust=False).mean()
    rs = avg_up / avg_down.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(50.0).values


def _build_indicator_cache(bars, bpd):
    cache = {}
    for coin, df in bars.items():
        close_s = df["Close"].astype(float)
        low_s = df["Low"].astype(float)

        sma50 = close_s.rolling(50 * bpd).mean()
        sma200 = close_s.rolling(200 * bpd).mean()
        ema50d = close_s.ewm(span=50 * bpd, adjust=False).mean()
        ema200d = close_s.ewm(span=200 * bpd, adjust=False).mean()
        ema10 = close_s.ewm(span=10, adjust=False).mean()
        sma20 = close_s.rolling(20).mean()
        std20 = close_s.rolling(20).std(ddof=0)
        bb_upper = sma20 + 2.0 * std20
        donchian50 = low_s.shift(1).rolling(50 * bpd).min()
        tsmom252 = close_s / close_s.shift(252 * bpd) - 1.0

        ema12 = close_s.ewm(span=12, adjust=False).mean()
        ema26 = close_s.ewm(span=26, adjust=False).mean()
        macd = ema12 - ema26
        signal = macd.ewm(span=9, adjust=False).mean()
        macd_hist = macd - signal
        rsi14 = pd.Series(_rsi(close_s.values, 14), index=close_s.index)

        cache[coin] = pd.DataFrame(
            {
                "close": close_s,
                "sma50": sma50,
                "sma200": sma200,
                "ema50d": ema50d,
                "ema200d": ema200d,
                "ema10": ema10,
                "sma20": sma20,
                "bb_upper": bb_upper,
                "donchian50": donchian50,
                "tsmom252": tsmom252,
                "macd_hist": macd_hist,
                "rsi14": rsi14,
            }
        )
    return cache


def _build_gate_series(data, strat_name):
    cfg = STRATEGIES[strat_name]
    bars, _ = data[cfg["interval"]]
    btc = bars["BTC"]
    dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    close = btc["Close"].values
    prev_canary = False
    out = {}
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
        out[date] = not canary
        prev_canary = canary
    return out


def _signal_for_coin(ind, ci, rule_name):
    row = ind.iloc[ci]
    price = float(row["close"])
    sma50 = None if pd.isna(row["sma50"]) else float(row["sma50"])
    sma200 = None if pd.isna(row["sma200"]) else float(row["sma200"])
    ema50d = None if pd.isna(row["ema50d"]) else float(row["ema50d"])
    ema200d = None if pd.isna(row["ema200d"]) else float(row["ema200d"])
    ema10 = None if pd.isna(row["ema10"]) else float(row["ema10"])
    sma20 = None if pd.isna(row["sma20"]) else float(row["sma20"])
    bb_upper = None if pd.isna(row["bb_upper"]) else float(row["bb_upper"])
    donchian50 = None if pd.isna(row["donchian50"]) else float(row["donchian50"])
    tsmom252 = None if pd.isna(row["tsmom252"]) else float(row["tsmom252"])
    macd_hist = None if pd.isna(row["macd_hist"]) else float(row["macd_hist"])
    rsi14 = None if pd.isna(row["rsi14"]) else float(row["rsi14"])

    if rule_name == "sma200":
        return sma200 is not None and price < sma200
    if rule_name == "tsmom252":
        return tsmom252 is not None and tsmom252 < 0
    if rule_name == "donchian50":
        return donchian50 is not None and price < donchian50
    if rule_name == "sma200_and_tsmom252":
        return sma200 is not None and tsmom252 is not None and price < sma200 and tsmom252 < 0
    if rule_name == "ema50_200_macd":
        return ema50d is not None and ema200d is not None and macd_hist is not None and ema50d < ema200d and macd_hist < 0
    if rule_name == "bear_rsi60":
        return sma50 is not None and rsi14 is not None and price < sma50 and rsi14 > 60
    if rule_name == "bear_rsi65":
        return sma50 is not None and rsi14 is not None and price < sma50 and rsi14 > 65
    if rule_name == "bear_dyn_resist":
        if sma50 is None or ema10 is None or sma20 is None:
            return False
        if price >= sma50:
            return False
        near_ema10 = abs(price / ema10 - 1.0) <= 0.01
        near_sma20 = abs(price / sma20 - 1.0) <= 0.01
        return near_ema10 or near_sma20
    if rule_name == "bear_bb_upper":
        return sma50 is not None and bb_upper is not None and price < sma50 and price >= bb_upper
    return False


def build_target_series(data, strat_name, universe_size, rule):
    cfg = STRATEGIES[strat_name]
    bars, _ = data[cfg["interval"]]
    dates = bars["BTC"].index[(bars["BTC"].index >= START) & (bars["BTC"].index <= END)]
    bpd = _bars_per_day(cfg["interval"])
    gate = _build_gate_series(data, strat_name)
    ind_cache = _build_indicator_cache(bars, bpd)
    out = []

    for date in dates:
        if not gate.get(date, False):
            out.append((date, {"CASH": 1.0}))
            continue

        picks = []
        for coin in get_mcap(date)[:universe_size]:
            ind = ind_cache.get(coin)
            if ind is None:
                continue
            ci = ind.index.get_indexer([date], method="ffill")[0]
            if ci < 0:
                continue
            if _signal_for_coin(ind, ci, rule["name"]):
                picks.append(coin)

        if not picks:
            out.append((date, {"CASH": 1.0}))
            continue

        w = 1.0 / len(picks)
        out.append((date, {coin: w for coin in picks}))

    return out


def main():
    t0 = time.time()
    print("Loading data...")
    data = {iv: load_data(iv) for iv in ["4h", "1h"]}
    universe_sizes = _parse_universe_sizes()
    rules = _parse_rules()

    rows = []
    for strat_name, cfg in STRATEGIES.items():
        bars, funding = data[cfg["interval"]]
        engine = ShortOnlyEngine(
            bars,
            funding,
            leverage=1.0,
            tx_cost=0.0004,
            maint_rate=0.004,
            initial_capital=10000.0,
        )
        print(f"\n== {strat_name} ({cfg['interval']}) ==")
        for universe_size in universe_sizes:
            for rule in rules:
                target_series = build_target_series(data, strat_name, universe_size, rule)
                m = engine.run(target_series)
                row = {
                    "strategy": strat_name,
                    "interval": cfg["interval"],
                    "universe_size": universe_size,
                    "rule": rule["name"],
                    "desc": f"top{universe_size} dynamic / {rule['desc']}",
                    "Cal": m.get("Cal", 0),
                    "CAGR": m.get("CAGR", 0),
                    "MDD": m.get("MDD", 0),
                    "Sharpe": m.get("Sharpe", 0),
                    "Liq": m.get("Liq", 0),
                    "Rebal": m.get("Rebal", 0),
                }
                rows.append(row)
                print(
                    f"top{universe_size:<2} {rule['name']:<22} "
                    f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
                    f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Rebal={row['Rebal']}"
                )

    rows.sort(key=lambda r: (-r["Cal"], -r["Sharpe"], r["strategy"], r["universe_size"], r["rule"]))
    print("\nTop candidates")
    for row in rows[:20]:
        print(
            f"- {row['strategy']} / top{row['universe_size']} / {row['rule']}: "
            f"Cal={row['Cal']:.2f}, CAGR={row['CAGR']:+.1%}, "
            f"MDD={row['MDD']:+.1%}, {row['desc']}"
        )
    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
