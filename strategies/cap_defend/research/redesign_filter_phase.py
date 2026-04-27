"""Step 3 필터 — phase sweep survivor 산출 (명시 CSV).

Plan v3 Step 3 필터 복합 조건:
  - med_Cal ≥ top500 자산별 median
  - p20_Cal ≥ 자산별 p30
  - phase_CV ≤ 자산별 p50

입력: redesign_rerank_phase_{asset}.csv
출력: redesign_phase_survivors_{asset}.csv (tag 외 phase_med/p20/CV/pass_reason)

snap_nudge / yearly 는 survivor CSV 만 읽도록 업데이트 필요.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))


def compute_stats(asset: str) -> pd.DataFrame:
    path = os.path.join(HERE, f"redesign_rerank_phase_{asset}.csv")
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df.dropna(subset=["Cal"])
    if "MDD" in df.columns:
        df = df.dropna(subset=["MDD"])
    rows = []
    for tag, g in df.groupby("tag"):
        cal = g["Cal"].astype(float)
        med = cal.median()
        p20 = np.percentile(cal, 20)
        mean = cal.mean()
        std = cal.std()
        cv = (std / abs(mean)) if abs(mean) > 1e-9 else 99.0
        rows.append({"tag": tag, "phase_med": med, "phase_p20": p20,
                     "phase_mean": mean, "phase_std": std, "phase_CV": cv,
                     "n_phase": len(cal)})
    return pd.DataFrame(rows)


def apply_filter(stats: pd.DataFrame, top500_cal_median: float = None) -> pd.DataFrame:
    # plan 기준: med_Cal ≥ top500 원본 median. 원본 median 제공 안되면 stats 내부 median fallback.
    med_gate = float(top500_cal_median) if top500_cal_median is not None else float(stats["phase_med"].median())
    p30 = np.percentile(stats["phase_p20"].dropna(), 30)
    cv_p50 = np.percentile(stats["phase_CV"].dropna(), 50)
    mask_med = stats["phase_med"] >= med_gate
    mask_p20 = stats["phase_p20"] >= p30
    mask_cv = stats["phase_CV"] <= cv_p50
    stats["pass_med"] = mask_med
    stats["pass_p20"] = mask_p20
    stats["pass_cv"] = mask_cv
    stats["pass_all"] = mask_med & mask_p20 & mask_cv
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=["fut", "spot", "stock"])
    args = ap.parse_args()
    stats = compute_stats(args.asset)
    if stats.empty:
        print(f"[{args.asset}] no phase data")
        return
    top500_path = os.path.join(HERE, f"redesign_top500_{args.asset}_k1.csv")
    top500_cal_median = None
    if os.path.exists(top500_path):
        top500 = pd.read_csv(top500_path)
        if "Cal" in top500.columns:
            top500_cal_median = float(top500["Cal"].median())
            print(f"[{args.asset}] top500 Cal median: {top500_cal_median:.3f}")
    stats = apply_filter(stats, top500_cal_median)
    out = os.path.join(HERE, f"redesign_phase_survivors_{args.asset}.csv")
    stats.to_csv(out, index=False)
    n_pass = int(stats["pass_all"].sum())
    print(f"[{args.asset}] total {len(stats)} configs, survivors {n_pass} → {out}")


if __name__ == "__main__":
    main()
