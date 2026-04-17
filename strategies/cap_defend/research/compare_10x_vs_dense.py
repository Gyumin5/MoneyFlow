#!/usr/bin/env python3
"""10배수(phase4_10x) vs dense(phase4_3asset) 결과 나란히 비교.

출력:
- compare_10x/overall.csv
- compare_10x/by_allocation.csv  (60/25/15, 60/30/10, 60/35/5, 60/40/0 각 best)
- compare_10x/report.txt
"""
from __future__ import annotations
import json
import os
import sys
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
OUT = os.path.join(HERE, "compare_10x")
os.makedirs(OUT, exist_ok=True)

DENSE = os.path.join(HERE, "phase4_3asset/raw.csv")
LEAN = os.path.join(HERE, "phase4_10x/raw.csv")


def load(path, tag):
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    df["grid"] = tag
    # allocation label: 60/{sp*100}/{fu*100}
    df["alloc"] = df.apply(
        lambda r: f"60/{int(round(r['sp_w']*100))}/{int(round(r['fu_w']*100))}",
        axis=1)
    return df


def top_by_alloc(df, n=3):
    if df.empty:
        return df
    return df.sort_values("Cal", ascending=False).groupby("alloc").head(n)


def main():
    dense = load(DENSE, "dense")
    lean = load(LEAN, "10x")
    print(f"dense rows={len(dense)}, 10x rows={len(lean)}")

    # 1. Overall top-10 each grid
    cols = ["grid", "alloc", "spot", "fut", "fut_lev", "band", "Cal", "Sh", "CAGR", "MDD"]
    overall = []
    for tag, df in [("dense", dense), ("10x", lean)]:
        if df.empty:
            continue
        top = df.sort_values("Cal", ascending=False).head(10).copy()
        for c in cols:
            if c not in top.columns:
                top[c] = None
        overall.append(top[cols])
    overall_df = pd.concat(overall) if overall else pd.DataFrame()
    overall_df.to_csv(os.path.join(OUT, "overall.csv"), index=False)

    # 2. By-allocation best
    rows = []
    for alloc in ["60/25/15", "60/30/10", "60/35/5", "60/40/0"]:
        for tag, df in [("dense", dense), ("10x", lean)]:
            if df.empty:
                continue
            sub = df[df["alloc"] == alloc]
            if sub.empty:
                rows.append({"alloc": alloc, "grid": tag, "status": "no_data"})
                continue
            best = sub.sort_values("Cal", ascending=False).iloc[0]
            rows.append({
                "alloc": alloc, "grid": tag, "status": "ok",
                "spot": best.get("spot"), "fut": best.get("fut"),
                "fut_lev": best.get("fut_lev"), "band": best.get("band"),
                "Cal": best["Cal"], "Sh": best["Sh"],
                "CAGR": best["CAGR"], "MDD": best["MDD"],
            })
    ba = pd.DataFrame(rows)
    ba.to_csv(os.path.join(OUT, "by_allocation.csv"), index=False)

    # 3. Report
    with open(os.path.join(OUT, "report.txt"), "w") as f:
        f.write(f"Rows: dense={len(dense)}, 10x={len(lean)}\n\n")
        f.write("=== Overall Cal top-5 ===\n")
        for tag, df in [("dense", dense), ("10x", lean)]:
            if df.empty:
                f.write(f"\n-- {tag}: EMPTY --\n")
                continue
            top = df.sort_values("Cal", ascending=False).head(5)
            f.write(f"\n-- {tag} --\n")
            f.write(top[["alloc", "fut_lev", "band", "Cal", "Sh", "CAGR", "MDD"]]
                    .to_string(index=False))
            f.write("\n")
        f.write("\n\n=== By allocation (Cal best) ===\n")
        f.write(ba.to_string(index=False))
        f.write("\n")
    print(f"compare_10x/ written: overall.csv, by_allocation.csv, report.txt")


if __name__ == "__main__":
    main()
