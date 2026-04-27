#!/usr/bin/env python3
"""현재 현물 V20 전략 전용 백테스트.

라이브 엔진(trade/coin_live_engine.py)의 멤버 로직을 그대로 재사용하되,
네트워크 없이 로컬 데이터로 재현 가능하게 만든 전용 러너.

가정:
- 신호 로직: V20 라이브 엔진과 동일
- 유니버스: data/historical_universe.json 월별 Top40 사용
- 가격 데이터: data/futures/*_1h.csv 를 Binance 현물 대용으로 사용
- 체결: 4h 봉 시가에서 exact rebalance, 수수료 0.4%
- 실행 버퍼: executor_coin.py 와 동일하게 Cash buffer 2% 적용
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import timezone
from typing import Dict, List

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trade"))

from backtest_futures_full import TICKER_MAP
from coin_live_engine import (
    ENSEMBLE_WEIGHTS,
    MEMBERS,
    MemberState,
    combine_ensemble,
    compute_member_target,
    slice_to_last_closed,
)


# V21 (2026-04-21) 에서 가드/exclusion 제거됨. 기존 시그널 호출자 호환 no-op.
def prune_expired_exclusions(*args, **kwargs):
    return None


def update_excluded_after_gap(*args, **kwargs):
    return None


DATA_DIR = os.path.join(ROOT, "data", "futures")
UNIVERSE_FILE = os.path.join(ROOT, "data", "historical_universe.json")

TX_COST = 0.004
INITIAL_CAPITAL = 10000.0
CASH_BUFFER_PCT = 0.02


def _resample(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    freq_map = {"4h": "4h", "D": "D"}
    return (
        df.resample(freq_map[interval])
        .agg({
            "Open": "first",
            "High": "max",
            "Low": "min",
            "Close": "last",
            "Volume": "sum",
        })
        .dropna(subset=["Close"])
    )


def load_universe(top_n: int = 40) -> Dict[str, List[str]]:
    with open(UNIVERSE_FILE) as f:
        raw = json.load(f)
    out = {}
    for ds, tickers in raw.items():
        bare = []
        for ticker in tickers:
            coin = ticker.replace("-USD", "")
            if coin in TICKER_MAP:
                bare.append(coin)
        out[ds] = bare[:top_n]
    return out


def get_universe_for_date(universe_map: Dict[str, List[str]], date: pd.Timestamp) -> List[str]:
    mk = date.strftime("%Y-%m") + "-01"
    if mk in universe_map:
        return list(universe_map[mk])
    keys = sorted(k for k in universe_map if k <= mk)
    return list(universe_map[keys[-1]]) if keys else []


def load_price_bars(universe_map: Dict[str, List[str]]) -> Dict[str, Dict[str, pd.DataFrame]]:
    needed = {"BTC"}
    for tickers in universe_map.values():
        needed.update(tickers)

    bars_1h: Dict[str, pd.DataFrame] = {}
    bars_4h: Dict[str, pd.DataFrame] = {}
    bars_d: Dict[str, pd.DataFrame] = {}
    for coin in sorted(needed):
        sym = TICKER_MAP.get(coin)
        if not sym:
            continue
        path = os.path.join(DATA_DIR, f"{sym}_1h.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path, parse_dates=["Date"], index_col="Date")
        if len(df) < 300:
            continue
        if df.index.tz is None:
            df.index = df.index.tz_localize("UTC")
        bars_1h[coin] = df
        bars_4h[coin] = _resample(df, "4h")
        bars_d[coin] = _resample(df, "D")
    return {"1h": bars_1h, "4h": bars_4h, "D": bars_d}


def apply_cash_buffer(target: Dict[str, float], buffer_pct: float) -> Dict[str, float]:
    if buffer_pct <= 0:
        return dict(target)
    out = {}
    for key, value in target.items():
        if key == "CASH":
            continue
        out[key] = value * (1.0 - buffer_pct)
    out["CASH"] = target.get("CASH", 0.0) * (1.0 - buffer_pct) + buffer_pct
    total = sum(out.values())
    if total > 0:
        out = {k: v / total for k, v in out.items()}
    return out


def get_bar_price(bars_4h: Dict[str, pd.DataFrame], coin: str, date: pd.Timestamp, field: str) -> float:
    df = bars_4h.get(coin)
    if df is None or date not in df.index:
        return 0.0
    return float(df.at[date, field])


def port_value(
    holdings: Dict[str, float],
    cash: float,
    bars_4h: Dict[str, pd.DataFrame],
    date: pd.Timestamp,
    field: str,
) -> float:
    total = cash
    for coin, qty in holdings.items():
        px = get_bar_price(bars_4h, coin, date, field)
        if px > 0:
            total += qty * px
    return total


def rebalance_spot(
    holdings: Dict[str, float],
    cash: float,
    target: Dict[str, float],
    bars_4h: Dict[str, pd.DataFrame],
    date: pd.Timestamp,
    tx_cost: float,
) -> tuple[Dict[str, float], float]:
    pv = port_value(holdings, cash, bars_4h, date, "Open")
    if pv <= 0:
        return holdings, cash

    target_qty = {}
    for coin, weight in target.items():
        if coin == "CASH" or weight <= 0:
            continue
        px = get_bar_price(bars_4h, coin, date, "Open")
        if px <= 0:
            continue
        target_qty[coin] = pv * weight / px

    for coin in list(holdings.keys()):
        px = get_bar_price(bars_4h, coin, date, "Open")
        if px <= 0:
            continue
        cur_qty = holdings.get(coin, 0.0)
        tgt_qty = target_qty.get(coin, 0.0)
        if tgt_qty >= cur_qty:
            continue
        sell_qty = cur_qty - tgt_qty
        cash += sell_qty * px * (1.0 - tx_cost)
        if tgt_qty <= 1e-12:
            del holdings[coin]
        else:
            holdings[coin] = tgt_qty

    for coin, tgt_qty in target_qty.items():
        px = get_bar_price(bars_4h, coin, date, "Open")
        if px <= 0:
            continue
        cur_qty = holdings.get(coin, 0.0)
        if tgt_qty <= cur_qty:
            continue
        buy_qty = tgt_qty - cur_qty
        max_qty = cash / (px * (1.0 + tx_cost))
        buy_qty = min(buy_qty, max_qty)
        if buy_qty <= 1e-12:
            continue
        cash -= buy_qty * px * (1.0 + tx_cost)
        holdings[coin] = cur_qty + buy_qty

    return holdings, cash


def calc_metrics(eq: pd.Series) -> Dict[str, float]:
    eq = eq.dropna()
    if len(eq) < 2:
        return {"Sharpe": 0.0, "CAGR": 0.0, "MDD": 0.0, "Cal": 0.0, "Final": 0.0}

    eq_daily = eq.resample("D").last().dropna()
    years = (eq_daily.index[-1] - eq_daily.index[0]).days / 365.25
    if years <= 0 or eq_daily.iloc[0] <= 0:
        return {"Sharpe": 0.0, "CAGR": 0.0, "MDD": 0.0, "Cal": 0.0, "Final": float(eq.iloc[-1])}

    cagr = (eq_daily.iloc[-1] / eq_daily.iloc[0]) ** (1.0 / years) - 1.0
    dr = eq_daily.pct_change().dropna()
    sharpe = float(dr.mean() / dr.std() * math.sqrt(365)) if dr.std() > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1.0).min())
    cal = cagr / abs(mdd) if mdd != 0 else 0.0
    return {
        "Sharpe": sharpe,
        "CAGR": cagr,
        "MDD": mdd,
        "Cal": cal,
        "Final": float(eq.iloc[-1]),
    }


def run_backtest(
    start: str,
    end: str,
    initial_capital: float = INITIAL_CAPITAL,
    tx_cost: float = TX_COST,
    buffer_pct: float = CASH_BUFFER_PCT,
    top_n: int = 40,
    initial_phases: Dict[str, int] = None,
):
    universe_map = load_universe(top_n=top_n)
    bars = load_price_bars(universe_map)
    bars_4h = bars["4h"]
    bars_d = bars["D"]

    btc_4h = bars_4h.get("BTC")
    if btc_4h is None or btc_4h.empty:
        raise RuntimeError("BTC 4h data missing")

    start_ts = pd.Timestamp(start, tz="UTC")
    end_ts = pd.Timestamp(end, tz="UTC")
    exec_dates = btc_4h.index[(btc_4h.index >= start_ts) & (btc_4h.index <= end_ts)]
    _phases = initial_phases or {}
    member_states = {name: MemberState() for name in MEMBERS}
    for _mname, _phase in _phases.items():
        if _mname in member_states:
            member_states[_mname].bar_counter = int(_phase)
    excluded_all = {name: {} for name in MEMBERS}
    holdings: Dict[str, float] = {}
    cash = float(initial_capital)
    rebal_count = 0
    prev_target = {"CASH": 1.0}
    records = []

    for date in exec_dates:
        now_utc = date.to_pydatetime().replace(tzinfo=timezone.utc)
        universe = [c for c in get_universe_for_date(universe_map, date) if c in bars_4h]

        member_targets = {}
        for mname, cfg in MEMBERS.items():
            interval = cfg["interval"]
            member_bars = bars_d if interval == "D" else bars_4h
            sliced = slice_to_last_closed(member_bars, interval, now_utc)
            if "BTC" not in sliced:
                member_targets[mname] = {"CASH": 1.0}
                continue

            mem_state = member_states[mname]

            # V21 (2026-04-21): compute_member_target 시그니처 변경 — mem_excluded 제거.
            res = compute_member_target(
                mname,
                cfg,
                sliced,
                universe,
                mem_state,
                now_utc,
            )

            member_states[mname] = res.new_state
            member_targets[mname] = res.target

        combined = combine_ensemble(member_targets, ENSEMBLE_WEIGHTS)
        combined = apply_cash_buffer(combined, buffer_pct)

        if combined != prev_target:
            holdings, cash = rebalance_spot(holdings, cash, combined, bars_4h, date, tx_cost)
            rebal_count += 1
            prev_target = dict(combined)

        equity = port_value(holdings, cash, bars_4h, date, "Close")
        rec = {"Date": date, "Value": equity, "CASH": combined.get("CASH", 0.0)}
        for name, target in member_targets.items():
            rec[f"{name}_CASH"] = target.get("CASH", 0.0)
        records.append(rec)

    eq = pd.DataFrame(records).set_index("Date")
    metrics = calc_metrics(eq["Value"])
    return {"metrics": metrics, "equity": eq["Value"], "rebal_count": rebal_count, "detail": eq}


def main():
    parser = argparse.ArgumentParser(description="Current V20 spot-only backtest")
    parser.add_argument("--start", default="2020-10-01")
    parser.add_argument("--end", default="2026-03-31")
    parser.add_argument("--initial-capital", type=float, default=INITIAL_CAPITAL)
    parser.add_argument("--tx-cost", type=float, default=TX_COST)
    parser.add_argument("--buffer-pct", type=float, default=CASH_BUFFER_PCT)
    parser.add_argument("--top-n", type=int, default=40)
    args = parser.parse_args()

    t0 = time.time()
    result = run_backtest(
        start=args.start,
        end=args.end,
        initial_capital=args.initial_capital,
        tx_cost=args.tx_cost,
        buffer_pct=args.buffer_pct,
        top_n=args.top_n,
    )
    m = result["metrics"]
    print("V20 spot backtest")
    print(f"  Period:  {args.start} ~ {args.end}")
    print(f"  Sharpe:  {m['Sharpe']:.3f}")
    print(f"  CAGR:    {m['CAGR']:+.1%}")
    print(f"  MDD:     {m['MDD']:+.1%}")
    print(f"  Calmar:  {m['Cal']:.2f}")
    print(f"  Rebal:   {result['rebal_count']}")
    print(f"  Final:   {m['Final']:.2f}")
    print(f"  Elapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
