#!/usr/bin/env python3
"""OOS holdout validation.

IS: 2020-10-01 ~ 2025-10-01
OOS: 2025-10-01 ~ 2026-04-13

대상:
- L2 top candidates from winners.json (4h single, D single, 2h single, ENS_none)
- Spot V20 (D+4h 50:50)

출력: oos_validation.csv (case_id, label, IS_metrics, OOS_metrics)
"""
from __future__ import annotations
import json
import os
import sys

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from backtest_futures_full import load_data, run as bt_run
from futures_ensemble_engine import SingleAccountEngine, combine_targets
from run_futures_fixedlev_search import FIXED_CFG

IS_START = "2020-10-01"
IS_END = "2025-10-01"
OOS_START = "2025-10-01"
OOS_END = "2026-04-13"


def build_trace(data, interval, params, start, end):
    bars, funding = data[interval]
    trace: list = []
    bt_run(bars, funding, interval=interval, leverage=1.0,
           start_date=start, end_date=end, _trace=trace, **params)
    return trace


def run_single(w, data, bars_1h, funding_1h, all_dates_1h, start, end):
    params = dict(FIXED_CFG)
    params.update({
        "sma_bars": int(w["sma_bars"]),
        "mom_short_bars": int(w["mom_short_bars"]),
        "mom_long_bars": int(w["mom_long_bars"]),
        "vol_threshold": float(w["vol_threshold"]),
        "snap_interval_bars": int(w["snap_interval_bars"]),
    })
    tr = build_trace(data, w["interval"], params, start, end)
    dates = all_dates_1h[(all_dates_1h >= start) & (all_dates_1h <= end)]
    combined = combine_targets({w["case_id"]: tr}, {w["case_id"]: 1.0}, dates)
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=float(w.get("lev", w.get("leverage"))), leverage_mode="fixed", per_coin_leverage_mode="none",
        stop_kind=str(w["stop_kind"]), stop_pct=float(w["stop_pct_actual"]),
        stop_lookback_bars=int(w["stop_lookback_bars"]), stop_gate="always",
    )
    return engine.run(combined)


def run_ens(members_rows, data, bars_1h, funding_1h, all_dates_1h, lev, stop_kind, stop_pct, lb, start, end):
    traces, weights = {}, {}
    w = 1.0 / len(members_rows)
    for m in members_rows:
        params = dict(FIXED_CFG)
        params.update({
            "sma_bars": int(m["sma_bars"]),
            "mom_short_bars": int(m["mom_short_bars"]),
            "mom_long_bars": int(m["mom_long_bars"]),
            "vol_threshold": float(m["vol_threshold"]),
            "snap_interval_bars": int(m["snap_interval_bars"]),
        })
        tr = build_trace(data, m["interval"], params, start, end)
        traces[m["case_id"]] = tr
        weights[m["case_id"]] = w
    dates = all_dates_1h[(all_dates_1h >= start) & (all_dates_1h <= end)]
    combined = combine_targets(traces, weights, dates)
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=float(lev), leverage_mode="fixed", per_coin_leverage_mode="none",
        stop_kind=stop_kind, stop_pct=stop_pct, stop_lookback_bars=lb, stop_gate="always",
    )
    return engine.run(combined)


def pick_targets(winners_json_path, phase_b_csv_path):
    d = json.load(open(winners_json_path))
    ws = d["winners"]
    # L2 singles: top 3 by Cal
    l2_singles = sorted(
        [w for w in ws if w["lev"] == 2.0 and not w.get("is_ensemble")],
        key=lambda x: -x["Cal"])[:3]
    # ENS_L2_none
    l2_ens_none = [w for w in ws if w["lev"] == 2.0 and w.get("is_ensemble") and w["guard_name"] == "none"]
    targets = l2_singles + l2_ens_none

    # Get full param rows from phase_b csv (for singles)
    df = pd.read_csv(phase_b_csv_path, on_bad_lines="skip")
    df = df[df["error"].fillna("") == ""]
    enriched = []
    for t in targets:
        if t.get("is_ensemble"):
            # Need to enrich member case_ids
            member_ids = t["members"]
            mem_rows = []
            for mid in member_ids:
                sub = df[df["case_id"] == mid]
                if not sub.empty:
                    mem_rows.append(sub.iloc[0].to_dict())
            t["_member_rows"] = mem_rows
            enriched.append(t)
        else:
            sub = df[df["case_id"] == t["case_id"]]
            if not sub.empty:
                row = sub.iloc[0].to_dict()
                t["sma_bars"] = row["sma_bars"]
                t["mom_short_bars"] = row["mom_short_bars"]
                t["mom_long_bars"] = row["mom_long_bars"]
                t["vol_threshold"] = row["vol_threshold"]
                t["snap_interval_bars"] = row["snap_interval_bars"]
                enriched.append(t)
    return enriched


def fmt(m):
    return (f"Cal={m.get('Cal',0):.2f} CAGR={m.get('CAGR',0):+.1%} "
            f"MDD={m.get('MDD',0):+.1%} Sh={m.get('Sharpe',0):.2f} "
            f"Liq={int(m.get('Liq',0))} Stops={int(m.get('Stops',0))}")


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--winners", default=os.path.join(
        HERE, "guard_search_runs/guard_v3_floor_mmdd/winners/winners.json"))
    p.add_argument("--phase-b-csv", default=os.path.join(
        HERE, "guard_search_runs/guard_v3_floor_mmdd/phase_b_results.csv"))
    p.add_argument("--out-csv", default=os.path.join(HERE, "oos_validation.csv"))
    args = p.parse_args()

    targets = pick_targets(args.winners, args.phase_b_csv)
    print(f"Selected {len(targets)} futures targets (L2)")
    for t in targets:
        print(f"  {t['case_id']:<22s} lev={t['lev']} {t['label'][:50]}")

    print("\nLoading data...")
    data = {iv: load_data(iv) for iv in ["1h", "2h", "4h", "D"]}
    bars_1h, funding_1h = data["1h"]
    all_dates_1h = bars_1h["BTC"].index

    rows = []
    for t in targets:
        print(f"\n=== {t['case_id']} {t['label'][:60]} ===")
        for label, start, end in [("IS", IS_START, IS_END), ("OOS", OOS_START, OOS_END)]:
            try:
                if t.get("is_ensemble"):
                    m = run_ens(t["_member_rows"], data, bars_1h, funding_1h, all_dates_1h,
                                t["lev"], t["stop_kind"], float(t["stop_pct_actual"]),
                                int(t["stop_lookback_bars"]), start, end)
                else:
                    m = run_single(t, data, bars_1h, funding_1h, all_dates_1h, start, end)
                print(f"  {label} {start}~{end}: {fmt(m)}")
                rows.append({
                    "case_id": t["case_id"], "label": t["label"], "period": label,
                    "start": start, "end": end,
                    "Cal": m.get("Cal", 0), "CAGR": m.get("CAGR", 0),
                    "MDD": m.get("MDD", 0), "Sharpe": m.get("Sharpe", 0),
                    "Liq": int(m.get("Liq", 0)), "Stops": int(m.get("Stops", 0)),
                })
            except Exception as e:
                print(f"  {label} FAIL: {e}")
                rows.append({"case_id": t["case_id"], "label": t["label"], "period": label,
                             "error": str(e)})

    pd.DataFrame(rows).to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv}")


if __name__ == "__main__":
    main()
