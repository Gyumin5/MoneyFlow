#!/usr/bin/env python3
"""현물식 gap exclusion을 선물에 적용하면?

기존 futures 엔진의 blacklist 기능 활용 (bl_drop + bl_days).
L3_4h 단독과 L3 ENS로 비교.
"""
from __future__ import annotations
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from backtest_futures_full import load_data, run as bt_run
from futures_ensemble_engine import SingleAccountEngine, combine_targets
from run_futures_fixedlev_search import FIXED_CFG

START = "2020-10-01"
END = "2026-04-13"
WINNERS = os.path.join(HERE, "guard_search_runs/guard_v3_floor_mmdd/winners/winners.json")
PHASE_B = os.path.join(HERE, "guard_search_runs/guard_v3_floor_mmdd/phase_b_results.csv")


def build_trace(data, interval, params, start, end):
    bars, funding = data[interval]
    trace = []
    bt_run(bars, funding, interval=interval, leverage=1.0,
           start_date=start, end_date=end, _trace=trace, **params)
    return trace


def run_variant(members, data, bars_1h, funding_1h, all_dates, lev, bl_drop, bl_days, bl_lb):
    traces, weights = {}, {}
    w = 1.0 / len(members)
    for m in members:
        params = dict(FIXED_CFG)
        params.update({
            "sma_bars": int(m["sma_bars"]),
            "mom_short_bars": int(m["mom_short_bars"]),
            "mom_long_bars": int(m["mom_long_bars"]),
            "vol_threshold": float(m["vol_threshold"]),
            "snap_interval_bars": int(m["snap_interval_bars"]),
            "bl_drop": bl_drop,
            "bl_days": bl_days,
            "bl_lookback_bars": bl_lb,
        })
        tr = build_trace(data, m["interval"], params, START, END)
        traces[m["case_id"]] = tr
        weights[m["case_id"]] = w
    dates = all_dates[(all_dates >= START) & (all_dates <= END)]
    combined = combine_targets(traces, weights, dates)
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=float(lev), leverage_mode="fixed", per_coin_leverage_mode="none",
        stop_kind="none", stop_pct=0.0, stop_lookback_bars=0, stop_gate="always",
    )
    return engine.run(combined)


def main():
    ws = json.load(open(WINNERS))["winners"]
    l3_singles = sorted([w for w in ws if w["lev"]==3.0 and not w.get("is_ensemble")],
                        key=lambda x: -x["Cal"])[:3]
    df = pd.read_csv(PHASE_B, on_bad_lines="skip")
    df = df[df["error"].fillna("") == ""]

    # Enrich member params
    members = []
    for t in l3_singles:
        sub = df[df["case_id"] == t["case_id"]]
        if not sub.empty:
            row = sub.iloc[0].to_dict()
            t.update({
                "sma_bars": row["sma_bars"],
                "mom_short_bars": row["mom_short_bars"],
                "mom_long_bars": row["mom_long_bars"],
                "vol_threshold": row["vol_threshold"],
                "snap_interval_bars": row["snap_interval_bars"],
            })
            members.append(t)
    print(f"L3 members: {[m['label'][:30] for m in members]}")

    print("Loading data...")
    data = {iv: load_data(iv) for iv in ["1h", "2h", "4h", "D"]}
    bars_1h, funding_1h = data["1h"]
    all_dates = bars_1h["BTC"].index

    # Variants: (label, bl_drop, bl_days, bl_lookback_bars)
    variants = [
        ("baseline (no guard)", 0, 0, 0),
        ("gap-15%/30d (D식)",  -0.15, 30, 0),   # bl_lookback=0 → bpd(일간)
        ("gap-10%/10d (4h식)", -0.10, 10, 0),
        ("gap-20%/14d",         -0.20, 14, 0),
        ("gap-15%/14d",         -0.15, 14, 0),
    ]

    rows = []
    for label, bld, bdy, blb in variants:
        # L3 ENS
        m_ens = run_variant(members, data, bars_1h, funding_1h, all_dates, 3.0, bld, bdy, blb)
        # L3 single (4h top)
        m_single = run_variant([members[0]], data, bars_1h, funding_1h, all_dates, 3.0, bld, bdy, blb)
        rows.append({
            "variant": label,
            "single_Cal": m_single.get("Cal", 0),
            "single_CAGR": m_single.get("CAGR", 0),
            "single_MDD": m_single.get("MDD", 0),
            "single_Sh": m_single.get("Sharpe", 0),
            "single_Liq": int(m_single.get("Liq", 0)),
            "ens_Cal": m_ens.get("Cal", 0),
            "ens_CAGR": m_ens.get("CAGR", 0),
            "ens_MDD": m_ens.get("MDD", 0),
            "ens_Sh": m_ens.get("Sharpe", 0),
            "ens_Liq": int(m_ens.get("Liq", 0)),
        })
        print(f"{label:<22s} | SINGLE Cal={m_single.get('Cal',0):.2f} CAGR={m_single.get('CAGR',0):+.1%} MDD={m_single.get('MDD',0):+.1%} Liq={int(m_single.get('Liq',0))}"
              f" | ENS Cal={m_ens.get('Cal',0):.2f} CAGR={m_ens.get('CAGR',0):+.1%} MDD={m_ens.get('MDD',0):+.1%} Liq={int(m_ens.get('Liq',0))}")

    out = pd.DataFrame(rows)
    out.to_csv(os.path.join(HERE, "futures_gap_exclusion_test.csv"), index=False)


if __name__ == "__main__":
    main()
