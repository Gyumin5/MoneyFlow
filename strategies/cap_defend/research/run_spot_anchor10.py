#!/usr/bin/env python3
"""10-anchor robustness for spot candidates: V20 ensemble vs spot_4h single vs spot_D single.

Monkey-patches trade.coin_live_engine MEMBERS/ENSEMBLE_WEIGHTS to isolate members.
Runs each candidate across 10 anchor starts to FULL_END.

Output: research/spot_anchor10/{summary.csv, raw.csv, run.log}
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
CAP_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, CAP_DIR)
sys.path.insert(0, ROOT)

from trade import coin_live_engine as engine_mod

FULL_END = "2026-04-13"

ANCHORS = [
    "2020-10-01", "2021-04-01", "2021-10-01", "2022-04-01", "2022-10-01",
    "2023-04-01", "2023-10-01", "2024-04-01", "2024-10-01", "2025-04-01",
]


def patch_members(config_name):
    """config_name: 'V20' | 'spot_4h' | 'spot_D'"""
    full = {
        'D_SMA50': engine_mod.MEMBER_D_SMA50,
        '4h_SMA240': engine_mod.MEMBER_4H_SMA240,
    }
    if config_name == 'V20':
        members = full
        weights = {'D_SMA50': 0.5, '4h_SMA240': 0.5}
    elif config_name == 'spot_4h':
        members = {'4h_SMA240': full['4h_SMA240']}
        weights = {'4h_SMA240': 1.0}
    elif config_name == 'spot_D':
        members = {'D_SMA50': full['D_SMA50']}
        weights = {'D_SMA50': 1.0}
    else:
        raise ValueError(config_name)

    # Patch at import site for run_current_coin_v20_backtest
    import strategies.cap_defend.run_current_coin_v20_backtest as bt
    bt.MEMBERS = members
    bt.ENSEMBLE_WEIGHTS = weights
    return bt


def run_config(config_name, start, end):
    bt = patch_members(config_name)
    res = bt.run_backtest(start=start, end=end)
    m = res["metrics"]
    return {
        "Sharpe": m.get("Sharpe", 0),
        "Cal": m.get("Cal", 0),
        "CAGR": m.get("CAGR", 0),
        "MDD": m.get("MDD", 0),
    }


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
    out_dir = os.path.join(os.path.dirname(__file__), "spot_anchor10")
    os.makedirs(out_dir, exist_ok=True)

    configs = ["V20", "spot_4h", "spot_D"]
    all_rows = []
    summary_rows = []

    for cfg in configs:
        print(f"\n=== {cfg} ===", flush=True)
        an_metrics = []
        for a in ANCHORS:
            try:
                m = run_config(cfg, a, FULL_END)
                an_metrics.append(m)
                all_rows.append({
                    "config": cfg, "anchor": a, "end": FULL_END,
                    **m,
                })
                print(f"  {a}: Cal={m['Cal']:.2f} Sh={m['Sharpe']:.2f} "
                      f"CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%}", flush=True)
            except Exception as e:
                print(f"  {a} FAIL: {e}", flush=True)
        summary_rows.append(summarize(cfg, an_metrics))

    pd.DataFrame(all_rows).to_csv(os.path.join(out_dir, "raw.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(os.path.join(out_dir, "summary.csv"), index=False)

    print("\n=== SPOT 10-ANCHOR SUMMARY ===")
    print(f"{'label':<10s} {'mSh':>5s} {'sSh':>5s} {'mCal':>5s} {'sCal':>5s} {'mCAGR':>8s} {'wMDD':>8s} {'robust':>7s}")
    for s in summary_rows:
        if s.get("n_anchors", 0) == 0:
            continue
        print(f"{s['label']:<10s} {s['mean_Sh']:>5.2f} {s['std_Sh']:>5.2f} "
              f"{s['mean_Cal']:>5.2f} {s['std_Cal']:>5.2f} "
              f"{s['mean_CAGR']:>+7.1%} {s['worst_MDD']:>+8.1%} "
              f"{'YES' if s['robust_ok'] else 'no':>7s}")


if __name__ == "__main__":
    main()
