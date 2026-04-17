#!/usr/bin/env python3
"""Extended 10-anchor for L3/L4 single + L4 ENS candidates missing from run_anchor10.

Adds: 4h_S240 L3 single, 4h_S240 L4 single, 1D_S40 L4 single,
      ensemble_top3_none L3, ensemble_top3_none L4.

Output: research/anchor10_ext/{anchor_summary.csv, raw.csv}
"""
from __future__ import annotations
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from run_anchor10 import (
    ANCHORS, FULL_END, run_single, run_ensemble, summarize,
)
from backtest_futures_full import load_data


def pick_by_spec(winners_json, phase_b_csv, specs):
    d = json.load(open(winners_json))
    ws = d["winners"]
    df = pd.read_csv(phase_b_csv, on_bad_lines="skip")
    df = df[df["error"].fillna("") == ""]

    out = []
    for spec in specs:
        lev = spec["lev"]
        if spec["is_ensemble"]:
            match = [w for w in ws if w["lev"] == lev and w.get("is_ensemble")
                     and w.get("guard_name") == spec.get("guard_name", "none")]
            if not match:
                print(f"SKIP missing ensemble L{lev} {spec}")
                continue
            t = dict(match[0])
            rows = []
            for mid in t["members"]:
                sub = df[df["case_id"] == mid]
                if len(sub):
                    rows.append(sub.iloc[0].to_dict())
            t["_member_rows"] = rows
            out.append(t)
        else:
            match = [w for w in ws if w["lev"] == lev and not w.get("is_ensemble")
                     and w["case_id"] == spec["case_id"]]
            if not match:
                print(f"SKIP missing single L{lev} {spec['case_id']}")
                continue
            t = dict(match[0])
            sub = df[df["case_id"] == t["case_id"]]
            if len(sub):
                row = sub.iloc[0].to_dict()
                t.update({
                    "sma_bars": row["sma_bars"],
                    "mom_short_bars": row["mom_short_bars"],
                    "mom_long_bars": row["mom_long_bars"],
                    "vol_threshold": row["vol_threshold"],
                    "snap_interval_bars": row["snap_interval_bars"],
                })
                out.append(t)
    return out


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--winners", default=os.path.join(
        HERE, "guard_search_runs/guard_v3_floor_mmdd/winners/winners.json"))
    p.add_argument("--phase-b-csv", default=os.path.join(
        HERE, "guard_search_runs/guard_v3_floor_mmdd/phase_b_results.csv"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "anchor10_ext"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    # L3/L4 top singles from HTML (4h_S240 is top Cal at both L3 and L4)
    # L4 best single = 4h_S240_MS40_ML480_V05_SN84 per winners
    specs = [
        {"lev": 3.0, "is_ensemble": False, "case_id": "590f6ac76a1acb4f"},  # 4h_S240_MS40_ML720 L3
        {"lev": 3.0, "is_ensemble": False, "case_id": "8a34a940aeba307b"},  # 1D_S40 L3
        {"lev": 4.0, "is_ensemble": False, "case_id": None},  # set below
        {"lev": 3.0, "is_ensemble": True, "guard_name": "none"},
        {"lev": 4.0, "is_ensemble": True, "guard_name": "none"},
    ]
    # Fix: L4 best single is actually 4h_S240_MS40_ML480 (case id differs).
    # Find it from winners
    with open(args.winners) as f:
        w_all = json.load(f)["winners"]
    for w in w_all:
        if w["lev"] == 4.0 and not w.get("is_ensemble"):
            print("L4 single top candidate from winners:", w["label"], w["case_id"])
            specs[2] = {"lev": 4.0, "is_ensemble": False, "case_id": w["case_id"]}
            break

    candidates = pick_by_spec(args.winners, args.phase_b_csv, specs)
    print(f"Candidates: {len(candidates)}")
    for c in candidates:
        tag = "ENS" if c.get("is_ensemble") else "S"
        print(f"  {c['case_id']:<22s} L{c['lev']} {tag} {c['label'][:50]}")

    print("\nLoading data...")
    data = {iv: load_data(iv) for iv in ["1h", "2h", "4h", "D"]}
    bars_1h, funding_1h = data["1h"]
    all_dates_1h = bars_1h["BTC"].index

    anchor_summary = []
    all_rows = []
    for c in candidates:
        label = c["label"][:40]
        tag = "ENS" if c.get("is_ensemble") else "S"
        lab_full = f"{label}_L{int(c['lev'])}_{tag}"
        print(f"\n=== {lab_full} ===")
        an_metrics = []
        for a in ANCHORS:
            try:
                if c.get("is_ensemble"):
                    m = run_ensemble(c, data, bars_1h, funding_1h, all_dates_1h, a, FULL_END)
                else:
                    m = run_single(c, data, bars_1h, funding_1h, all_dates_1h, a, FULL_END)
                an_metrics.append(m)
                all_rows.append({
                    "case_id": c["case_id"], "label": lab_full, "lev": c["lev"],
                    "anchor": a, "end": FULL_END,
                    "Cal": m.get("Cal", 0), "CAGR": m.get("CAGR", 0),
                    "MDD": m.get("MDD", 0), "Sharpe": m.get("Sharpe", 0),
                })
                print(f"  {a}: Cal={m.get('Cal',0):.2f} Sh={m.get('Sharpe',0):.2f} "
                      f"CAGR={m.get('CAGR',0):+.1%} MDD={m.get('MDD',0):+.1%}")
            except Exception as e:
                print(f"  {a} FAIL: {e}")
        anchor_summary.append({"case_id": c["case_id"], "lev": c["lev"],
                               **summarize(lab_full, an_metrics)})

    pd.DataFrame(all_rows).to_csv(os.path.join(args.out_dir, "raw.csv"), index=False)
    pd.DataFrame(anchor_summary).to_csv(os.path.join(args.out_dir, "anchor_summary.csv"), index=False)

    print("\n=== EXT 10-ANCHOR SUMMARY ===")
    print(f"{'label':<44s} {'L':>3s} {'mSh':>5s} {'sSh':>5s} {'mCal':>5s} {'wMDD':>8s}")
    for s in anchor_summary:
        if s.get("n_anchors", 0) == 0:
            continue
        print(f"{s['label']:<44s} {s['lev']:>3.0f} {s['mean_Sh']:>5.2f} "
              f"{s['std_Sh']:>5.2f} {s['mean_Cal']:>5.2f} {s['worst_MDD']:>+8.1%}")


if __name__ == "__main__":
    main()
