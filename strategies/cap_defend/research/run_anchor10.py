#!/usr/bin/env python3
"""10-anchor multi-anchor robustness for futures candidates.

Anchors: 2020-10, 2021-04, 2021-10, 2022-04, 2022-10, 2023-04,
         2023-10, 2024-04, 2024-10, 2025-04 (all -> FULL_END).

Candidates:
- d005 ENS (L3): current 4-member EW from futures_live_config
- 4h_S240 (L2), 1D_S40 (L2), 2h_S240 (L2): top-3 singles (from winners)
- ensemble_top3_guard=none (L2)

Output: research/anchor10/{anchor_summary.csv, raw.csv}
robust_ok = std_Sh <= 0.1
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from backtest_futures_full import load_data, run as bt_run
from futures_ensemble_engine import SingleAccountEngine, combine_targets
from run_futures_fixedlev_search import FIXED_CFG
from futures_live_config import CURRENT_STRATEGIES

FULL_END = "2026-04-13"

ANCHORS = [
    "2020-10-01", "2021-04-01", "2021-10-01", "2022-04-01", "2022-10-01",
    "2023-04-01", "2023-10-01", "2024-04-01", "2024-10-01", "2025-04-01",
]


def build_trace(data, interval, params, start, end):
    bars, funding = data[interval]
    trace: list = []
    bt_run(bars, funding, interval=interval, leverage=1.0,
           start_date=start, end_date=end, _trace=trace, **params)
    return trace


def run_single(cand, data, bars_1h, funding_1h, all_dates_1h, start, end):
    params = dict(FIXED_CFG)
    params.update({
        "sma_bars": int(cand["sma_bars"]),
        "mom_short_bars": int(cand["mom_short_bars"]),
        "mom_long_bars": int(cand["mom_long_bars"]),
        "vol_threshold": float(cand["vol_threshold"]),
        "snap_interval_bars": int(cand["snap_interval_bars"]),
    })
    tr = build_trace(data, cand["interval"], params, start, end)
    dates = all_dates_1h[(all_dates_1h >= start) & (all_dates_1h <= end)]
    combined = combine_targets({cand["case_id"]: tr}, {cand["case_id"]: 1.0}, dates)
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=float(cand["lev"]), leverage_mode="fixed", per_coin_leverage_mode="none",
        stop_kind=str(cand["stop_kind"]), stop_pct=float(cand["stop_pct_actual"]),
        stop_lookback_bars=int(cand["stop_lookback_bars"]), stop_gate="always",
    )
    return engine.run(combined)


def run_ensemble(cand, data, bars_1h, funding_1h, all_dates_1h, start, end):
    traces, weights = {}, {}
    members = cand["_member_rows"]
    w = 1.0 / len(members)
    for idx, m in enumerate(members):
        params = dict(FIXED_CFG)
        # override vol_mode if provided (d005 has mixed daily/bar)
        if "vol_mode" in m:
            params["vol_mode"] = m["vol_mode"]
        params.update({
            "sma_bars": int(m["sma_bars"]),
            "mom_short_bars": int(m["mom_short_bars"]),
            "mom_long_bars": int(m["mom_long_bars"]),
            "vol_threshold": float(m["vol_threshold"]),
            "snap_interval_bars": int(m["snap_interval_bars"]),
        })
        mid = m.get("case_id") or f"{cand['case_id']}_m{idx}"
        tr = build_trace(data, m["interval"], params, start, end)
        traces[mid] = tr
        weights[mid] = w
    dates = all_dates_1h[(all_dates_1h >= start) & (all_dates_1h <= end)]
    combined = combine_targets(traces, weights, dates)
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=float(cand["lev"]), leverage_mode="fixed", per_coin_leverage_mode="none",
        stop_kind=str(cand["stop_kind"]), stop_pct=float(cand["stop_pct_actual"]),
        stop_lookback_bars=int(cand["stop_lookback_bars"]), stop_gate="always",
    )
    return engine.run(combined)


def build_d005_candidate():
    member_rows = []
    for name, cfg in CURRENT_STRATEGIES.items():
        row = dict(cfg)
        row["case_id"] = f"d005_{name}"
        member_rows.append(row)
    return {
        "case_id": "d005_ENS_L3",
        "label": "d005_ENS(4mem)_L3",
        "lev": 3.0,
        "is_ensemble": True,
        "stop_kind": "prev_close_pct",
        "stop_pct_actual": 0.15,
        "stop_lookback_bars": 0,
        "_member_rows": member_rows,
    }


def pick_legacy_candidates(winners_json, phase_b_csv):
    d = json.load(open(winners_json))
    ws = d["winners"]
    l2_singles = sorted(
        [w for w in ws if w["lev"] == 2.0 and not w.get("is_ensemble")],
        key=lambda x: -x["Cal"])[:3]
    l2_ens_none = [w for w in ws if w["lev"] == 2.0 and w.get("is_ensemble") and w["guard_name"] == "none"]
    targets = l2_singles + l2_ens_none

    df = pd.read_csv(phase_b_csv, on_bad_lines="skip")
    df = df[df["error"].fillna("") == ""]
    out = []
    for t in targets:
        if t.get("is_ensemble"):
            rows = []
            for mid in t["members"]:
                sub = df[df["case_id"] == mid]
                if not sub.empty:
                    rows.append(sub.iloc[0].to_dict())
            t["_member_rows"] = rows
            out.append(t)
        else:
            sub = df[df["case_id"] == t["case_id"]]
            if not sub.empty:
                row = sub.iloc[0].to_dict()
                t["sma_bars"] = row["sma_bars"]
                t["mom_short_bars"] = row["mom_short_bars"]
                t["mom_long_bars"] = row["mom_long_bars"]
                t["vol_threshold"] = row["vol_threshold"]
                t["snap_interval_bars"] = row["snap_interval_bars"]
                out.append(t)
    return out


def summarize(label, metrics_list):
    if not metrics_list:
        return {"label": label, "n_anchors": 0, "robust_ok": False}
    sh = np.array([m["Sharpe"] for m in metrics_list])
    cal = np.array([m["Cal"] for m in metrics_list])
    cagr = np.array([m["CAGR"] for m in metrics_list])
    mdd = np.array([m["MDD"] for m in metrics_list])
    return {
        "label": label,
        "n_anchors": len(metrics_list),
        "mean_Sh": float(sh.mean()), "std_Sh": float(sh.std()),
        "mean_Cal": float(cal.mean()), "std_Cal": float(cal.std()),
        "mean_CAGR": float(cagr.mean()),
        "worst_MDD": float(mdd.min()),
        "robust_ok": bool(sh.std() <= 0.1),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--winners", default=os.path.join(
        HERE, "guard_search_runs/guard_v3_floor_mmdd/winners/winners.json"))
    p.add_argument("--phase-b-csv", default=os.path.join(
        HERE, "guard_search_runs/guard_v3_floor_mmdd/phase_b_results.csv"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "anchor10"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    candidates = pick_legacy_candidates(args.winners, args.phase_b_csv)
    candidates.insert(0, build_d005_candidate())
    print(f"Candidates: {len(candidates)}")
    for c in candidates:
        print(f"  {c['case_id']:<22s} L{c['lev']} {c['label'][:50]}")

    print("\nLoading data...")
    data = {iv: load_data(iv) for iv in ["1h", "2h", "4h", "D"]}
    bars_1h, funding_1h = data["1h"]
    all_dates_1h = bars_1h["BTC"].index

    print(f"\nAnchors: {len(ANCHORS)}")

    anchor_summary = []
    all_rows = []

    for c in candidates:
        label = c["label"][:40]
        print(f"\n=== {label} (L{c['lev']}) ===")
        an_metrics = []
        for a in ANCHORS:
            try:
                if c.get("is_ensemble"):
                    m = run_ensemble(c, data, bars_1h, funding_1h, all_dates_1h, a, FULL_END)
                else:
                    m = run_single(c, data, bars_1h, funding_1h, all_dates_1h, a, FULL_END)
                an_metrics.append(m)
                all_rows.append({
                    "case_id": c["case_id"], "label": label,
                    "lev": c["lev"], "anchor": a, "end": FULL_END,
                    "Cal": m.get("Cal", 0), "CAGR": m.get("CAGR", 0),
                    "MDD": m.get("MDD", 0), "Sharpe": m.get("Sharpe", 0),
                })
                print(f"  {a}: Cal={m.get('Cal',0):.2f} Sh={m.get('Sharpe',0):.2f} "
                      f"CAGR={m.get('CAGR',0):+.1%} MDD={m.get('MDD',0):+.1%}")
            except Exception as e:
                print(f"  {a} FAIL: {e}")
        anchor_summary.append({"case_id": c["case_id"], "lev": c["lev"],
                               **summarize(label, an_metrics)})

    pd.DataFrame(all_rows).to_csv(os.path.join(args.out_dir, "raw.csv"), index=False)
    pd.DataFrame(anchor_summary).to_csv(os.path.join(args.out_dir, "anchor_summary.csv"), index=False)

    print("\n\n=== 10-ANCHOR SUMMARY ===")
    print(f"{'label':<42s} {'L':>3s} {'mSh':>5s} {'sSh':>5s} {'mCal':>5s} {'sCal':>5s} {'wMDD':>8s} {'robust':>7s}")
    for s in anchor_summary:
        if s.get("n_anchors", 0) == 0:
            continue
        print(f"{s['label']:<42s} {s['lev']:>3.0f} {s['mean_Sh']:>5.2f} {s['std_Sh']:>5.2f} "
              f"{s['mean_Cal']:>5.2f} {s['std_Cal']:>5.2f} {s['worst_MDD']:>+8.1%} "
              f"{'YES' if s['robust_ok'] else 'no':>7s}")


if __name__ == "__main__":
    main()
