#!/usr/bin/env python3
"""Walk-forward + multi-anchor robustness 검증.

1. Walk-forward: 6-month rolling OOS windows across 2021.01~2026.04
2. Multi-anchor: 5 different start dates (2020-10, 2021-04, 2021-10, 2022-04, 2022-10)
   → 전 구간 백테스트 & sigma(Sharpe) / sigma(Cal) 계산

대상: L2 후보 4개 (4h_S240, 1D_S40, 2h_S240, ENS_none)
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

FULL_END = "2026-04-13"


def build_trace(data, interval, params, start, end):
    bars, funding = data[interval]
    trace: list = []
    bt_run(bars, funding, interval=interval, leverage=1.0,
           start_date=start, end_date=end, _trace=trace, **params)
    return trace


def run_candidate(cand, data, bars_1h, funding_1h, all_dates_1h, start, end):
    if cand.get("is_ensemble"):
        traces, weights = {}, {}
        w = 1.0 / len(cand["_member_rows"])
        for m in cand["_member_rows"]:
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
    else:
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


def pick_candidates(winners_json, phase_b_csv):
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


def walk_forward_windows(global_start="2020-10-01", global_end=FULL_END, win_months=6):
    """6-month non-overlapping windows starting from global_start."""
    wins = []
    cur = pd.Timestamp(global_start)
    end_dt = pd.Timestamp(global_end)
    while cur < end_dt:
        nxt = cur + pd.DateOffset(months=win_months)
        if nxt > end_dt:
            nxt = end_dt
        wins.append((str(cur.date()), str(nxt.date())))
        cur = nxt
    return wins


def multi_anchor_starts():
    return ["2020-10-01", "2021-04-01", "2021-10-01", "2022-04-01", "2022-10-01"]


def summarize_wf(candidate_label, per_window_metrics):
    sh = np.array([m["Sharpe"] for m in per_window_metrics])
    cal = np.array([m["Cal"] for m in per_window_metrics])
    cagr = np.array([m["CAGR"] for m in per_window_metrics])
    mdd = np.array([m["MDD"] for m in per_window_metrics])
    pos_share = (cagr > 0).sum() / len(cagr) if len(cagr) else 0
    return {
        "label": candidate_label, "n_windows": len(per_window_metrics),
        "mean_Sh": float(sh.mean()), "std_Sh": float(sh.std()),
        "mean_Cal": float(cal.mean()), "std_Cal": float(cal.std()),
        "mean_CAGR": float(cagr.mean()), "worst_MDD": float(mdd.min()),
        "pos_windows_pct": float(pos_share),
    }


def summarize_anchor(candidate_label, per_anchor_metrics):
    sh = np.array([m["Sharpe"] for m in per_anchor_metrics])
    cal = np.array([m["Cal"] for m in per_anchor_metrics])
    cagr = np.array([m["CAGR"] for m in per_anchor_metrics])
    mdd = np.array([m["MDD"] for m in per_anchor_metrics])
    return {
        "label": candidate_label, "n_anchors": len(per_anchor_metrics),
        "mean_Sh": float(sh.mean()), "std_Sh": float(sh.std()),
        "mean_Cal": float(cal.mean()), "std_Cal": float(cal.std()),
        "mean_CAGR": float(cagr.mean()), "worst_MDD": float(mdd.min()),
        "robust_ok": bool(sh.std() <= 0.1),
    }


def main():
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--winners", default=os.path.join(
        HERE, "guard_search_runs/guard_v3_floor_mmdd/winners/winners.json"))
    p.add_argument("--phase-b-csv", default=os.path.join(
        HERE, "guard_search_runs/guard_v3_floor_mmdd/phase_b_results.csv"))
    p.add_argument("--out-dir", default=os.path.join(HERE, "wf_multianchor"))
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    candidates = pick_candidates(args.winners, args.phase_b_csv)
    print(f"Candidates: {len(candidates)}")
    for c in candidates:
        print(f"  {c['case_id']:<22s} {c['label'][:50]}")

    print("\nLoading data...")
    data = {iv: load_data(iv) for iv in ["1h", "2h", "4h", "D"]}
    bars_1h, funding_1h = data["1h"]
    all_dates_1h = bars_1h["BTC"].index

    wins = walk_forward_windows()
    anchors = multi_anchor_starts()
    print(f"\nWF windows: {len(wins)} / Anchors: {len(anchors)}")

    wf_summary = []
    anchor_summary = []
    all_rows = []

    for c in candidates:
        label = c["label"][:40]
        print(f"\n=== {label} ===")
        # Walk-forward
        wf_metrics = []
        for s, e in wins:
            try:
                m = run_candidate(c, data, bars_1h, funding_1h, all_dates_1h, s, e)
                wf_metrics.append(m)
                all_rows.append({"case_id": c["case_id"], "label": label, "mode": "wf",
                                 "start": s, "end": e,
                                 "Cal": m.get("Cal", 0), "CAGR": m.get("CAGR", 0),
                                 "MDD": m.get("MDD", 0), "Sharpe": m.get("Sharpe", 0)})
                print(f"  WF {s}~{e}: Cal={m.get('Cal',0):.2f} CAGR={m.get('CAGR',0):+.1%} MDD={m.get('MDD',0):+.1%}")
            except Exception as e2:
                print(f"  WF {s}~{e} FAIL: {e2}")
        wf_summary.append(summarize_wf(label, wf_metrics))

        # Multi-anchor (start different dates → run to FULL_END)
        an_metrics = []
        for anchor_start in anchors:
            try:
                m = run_candidate(c, data, bars_1h, funding_1h, all_dates_1h, anchor_start, FULL_END)
                an_metrics.append(m)
                all_rows.append({"case_id": c["case_id"], "label": label, "mode": "anchor",
                                 "start": anchor_start, "end": FULL_END,
                                 "Cal": m.get("Cal", 0), "CAGR": m.get("CAGR", 0),
                                 "MDD": m.get("MDD", 0), "Sharpe": m.get("Sharpe", 0)})
                print(f"  Anchor {anchor_start}: Cal={m.get('Cal',0):.2f} Sh={m.get('Sharpe',0):.2f}")
            except Exception as e2:
                print(f"  Anchor {anchor_start} FAIL: {e2}")
        anchor_summary.append(summarize_anchor(label, an_metrics))

    pd.DataFrame(all_rows).to_csv(os.path.join(args.out_dir, "wf_multianchor_raw.csv"), index=False)
    pd.DataFrame(wf_summary).to_csv(os.path.join(args.out_dir, "wf_summary.csv"), index=False)
    pd.DataFrame(anchor_summary).to_csv(os.path.join(args.out_dir, "anchor_summary.csv"), index=False)

    print("\n\n=== WALK-FORWARD SUMMARY ===")
    print(f"{'label':<42s} {'win%':>5s} {'mSh':>5s} {'sSh':>5s} {'mCal':>5s} {'worstDD':>8s}")
    for s in wf_summary:
        print(f"{s['label']:<42s} {s['pos_windows_pct']*100:>4.0f}% {s['mean_Sh']:>5.2f} "
              f"{s['std_Sh']:>5.2f} {s['mean_Cal']:>5.2f} {s['worst_MDD']:>+8.1%}")

    print("\n=== MULTI-ANCHOR SUMMARY ===")
    print(f"{'label':<42s} {'mSh':>5s} {'sSh':>5s} {'mCal':>5s} {'sCal':>5s} {'robust':>7s}")
    for s in anchor_summary:
        print(f"{s['label']:<42s} {s['mean_Sh']:>5.2f} {s['std_Sh']:>5.2f} "
              f"{s['mean_Cal']:>5.2f} {s['std_Cal']:>5.2f} {'YES' if s['robust_ok'] else 'no':>7s}")


if __name__ == "__main__":
    main()
