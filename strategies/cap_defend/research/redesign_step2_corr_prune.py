#!/usr/bin/env python3
"""Redesign Step 2: robust 후보에서 realized corr 기반 greedy pruning.

- 입력: redesign_step1/holdout_eval.csv + pairwise_corr.csv
- Filter: Cal_train>=2.0 AND Cal_holdout>=0.5 (robust)
- Rank: holdout Cal desc
- Greedy pick: 상위부터, 이미 선정된 멤버와 realized daily-ret corr < 0.85인 경우만 추가
- 자산별 (spot/fut_L2/L3/L4) 분리 처리
- 출력: redesign_step2/final_candidates.csv
"""
from __future__ import annotations
import os, sys
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
IN = os.path.join(HERE, "redesign_step1")
OUT = os.path.join(HERE, "redesign_step2")
os.makedirs(OUT, exist_ok=True)

CORR_CUTOFF = 0.85
TRAIN_FLOOR = 2.0
HOLDOUT_FLOOR = 0.5


def main():
    df = pd.read_csv(os.path.join(IN, "holdout_eval.csv"))
    corr = pd.read_csv(os.path.join(IN, "pairwise_corr.csv"), index_col=0)
    df = df.dropna(subset=["Cal_holdout"]).copy()

    # Robust
    rob = df[(df["Cal_train"] >= TRAIN_FLOOR) & (df["Cal_holdout"] >= HOLDOUT_FLOOR)].copy()
    print(f"Robust: {len(rob)}/{len(df)}")
    rob = rob.sort_values("Cal_holdout", ascending=False)

    # Asset bucket
    def bucket(r):
        if r["asset"] == "spot":
            return "spot"
        return f"fut_L{int(r['lev'])}"
    rob["bucket"] = rob.apply(bucket, axis=1)

    picks_all = []
    for bkt, sub in rob.groupby("bucket"):
        kept = []  # list of ensemble_tag
        for _, r in sub.iterrows():
            tag = r["ensemble_tag"]
            if tag not in corr.columns:
                continue
            ok = True
            for k in kept:
                if k in corr.columns:
                    c = float(corr.loc[tag, k])
                    if c >= CORR_CUTOFF:
                        ok = False
                        break
            if ok:
                kept.append(tag)
                picks_all.append(r)
        print(f"  {bkt}: {len(sub)} robust → {len([p for p in picks_all if p['bucket']==bkt])} after corr prune")

    final = pd.DataFrame(picks_all)
    final.to_csv(os.path.join(OUT, "final_candidates.csv"), index=False)
    print(f"\n=== 최종 후보 ({len(final)}개, corr<{CORR_CUTOFF}) ===")
    cols = ["bucket", "ensemble_tag", "k", "Cal_train", "Cal_holdout",
            "CAGR_holdout", "MDD_holdout"]
    print(final[cols].to_string(index=False))

    # Save by bucket summary
    print("\n=== bucket별 best 후보 ===")
    for bkt, sub in final.groupby("bucket"):
        best = sub.iloc[0]
        print(f"  {bkt}: {best['ensemble_tag']} | train={best['Cal_train']:.2f} "
              f"holdout={best['Cal_holdout']:.2f} CAGR_h={best['CAGR_holdout']:.2%} "
              f"MDD_h={best['MDD_holdout']:.2%} (k={best['k']})")


if __name__ == "__main__":
    main()
