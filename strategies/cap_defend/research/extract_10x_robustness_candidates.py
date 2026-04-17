#!/usr/bin/env python3
"""phase4_10x/raw.csv → robustness 후보 CSV 추출.

각 (alloc, band_mode) 그룹에서 Cal top-1을 선택.
최소 후보: 60/25/15, 60/30/10, 60/35/5, 60/40/0 × {abs, sleeve}
"""
from __future__ import annotations
import os
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
SRC = os.path.join(HERE, "phase4_10x/raw.csv")
OUT = os.path.join(HERE, "phase4_10x_robustness_candidates.csv")

ALLOCS = ["60/25/15", "60/30/10", "60/35/5", "60/40/0"]


def main():
    df = pd.read_csv(SRC)
    df["alloc"] = df.apply(
        lambda r: f"60/{int(round(r['sp_w']*100))}/{int(round(r['fu_w']*100))}",
        axis=1)
    cands = []
    for alloc in ALLOCS:
        sub = df[df["alloc"] == alloc]
        if sub.empty:
            continue
        # pick Cal top-1 absolute band and Cal top-1 sleeve-relative band
        for mode in ["abs", "sleeve"]:
            msub = sub[sub["band_mode"] == mode]
            if msub.empty:
                continue
            best = msub.sort_values("Cal", ascending=False).iloc[0]
            label = f"10x_{alloc.replace('/','_')}_{mode}"
            cands.append({
                "label": label,
                "spot": best["spot"],
                "fut": best.get("fut"),
                "fut_lev": best.get("fut_lev", 0),
                "st_w": best["st_w"],
                "sp_w": best["sp_w"],
                "fu_w": best["fu_w"],
                "band": best["band"],
            })
    out = pd.DataFrame(cands)
    out.to_csv(OUT, index=False)
    print(f"Wrote {len(out)} candidates to {OUT}")
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
