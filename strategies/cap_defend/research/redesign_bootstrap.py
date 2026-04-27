"""Step 4.5 — Yearly block bootstrap (보조 진단).

각 해를 block 단위로 재순열 N=200회 해서 Cal/MDD 분포 계산.
순서 의존성이 큰 전략 식별 (bootstrap_Cal_p10 낮음).

fixed seed 재현성 보장.

입력: redesign_yearly_{asset}.csv (Step 4 결과)
출력: redesign_bootstrap_{asset}.csv
  tag, bootstrap_Cal_med, bootstrap_Cal_p10, bootstrap_MDD_p10, n_samples
"""
from __future__ import annotations
import argparse
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)


def compute_bootstrap(yearly_df, n_samples=200, seed=42):
    """yearly_df: tag, year, Cal, CAGR, MDD columns.
    year CAGR 로부터 random permutation 으로 누적 equity 재구성.
    MDD 는 reorder 후 cummax 기반 재계산 (근사치).
    """
    rng = np.random.default_rng(seed)
    rows = []
    for tag, g in yearly_df.groupby("tag"):
        if len(g) < 2 or "CAGR" not in g.columns or "MDD" not in g.columns:
            continue
        # CAGR from Cal 값이 없으면 skip. yearly csv 에 CAGR/MDD 필수
        cagr = g["CAGR"].dropna().astype(float).values
        mdd_years = g["MDD"].dropna().astype(float).values
        if len(cagr) < 2:
            continue
        cal_samples = []
        mdd_samples = []
        for _ in range(n_samples):
            perm = rng.permutation(len(cagr))
            cagr_p = cagr[perm]
            # 누적 equity
            eq = np.cumprod(1.0 + cagr_p)
            if eq[-1] <= 0:
                continue
            yrs = len(cagr)
            total_cagr = eq[-1] ** (1.0 / yrs) - 1.0
            # MDD: year-level path 에서 cummax 기반 (conservative: per-year MDD 연결 불가하므로 CAGR eq path MDD 만)
            mdd = float((eq / np.maximum.accumulate(eq) - 1).min())
            cal = total_cagr / abs(mdd) if mdd < 0 else 0.0
            cal_samples.append(cal)
            mdd_samples.append(mdd)
        if not cal_samples:
            continue
        rows.append({
            "tag": tag,
            "n_samples": len(cal_samples),
            "bootstrap_Cal_med": float(np.median(cal_samples)),
            "bootstrap_Cal_p10": float(np.percentile(cal_samples, 10)),
            "bootstrap_Cal_p90": float(np.percentile(cal_samples, 90)),
            "bootstrap_MDD_p10": float(np.percentile(mdd_samples, 10)),
            "bootstrap_MDD_med": float(np.median(mdd_samples)),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=["fut", "spot", "stock"])
    ap.add_argument("--n_samples", type=int, default=200)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    yearly_csv = os.path.join(HERE, f"redesign_yearly_{args.asset}.csv")
    if not os.path.exists(yearly_csv):
        print(f"[{args.asset}] missing: {yearly_csv}")
        return
    yearly = pd.read_csv(yearly_csv)
    if "status" in yearly.columns:
        yearly = yearly[yearly["status"] == "ok"]
    res = compute_bootstrap(yearly, n_samples=args.n_samples, seed=args.seed)
    out = os.path.join(HERE, f"redesign_bootstrap_{args.asset}.csv")
    res.to_csv(out, index=False)
    print(f"[{args.asset}] wrote {out} ({len(res)} rows)")


if __name__ == "__main__":
    main()
