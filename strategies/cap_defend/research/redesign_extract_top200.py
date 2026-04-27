"""Step 1: 기존 그리드에서 top 200 추출.

Sources
-------
- iter_refine/raw_combined.csv: single-strategy grid, asset {spot,fut} × iv {4h,D} × lev
- phase1_v2_sweep/summary.csv : multi-anchor summary (mCal, wMDD, ...)
- grid_results/topcands_proper_FUT_1x_k{2,3,4}_partial.csv: ensemble combos (k=2/3/4)
- grid_results/ensemble_spot_triple_sweep.csv, topcands_proper_SPOT_1x_k*.csv

Output
------
- redesign_top200_fut.csv
- redesign_top200_spot.csv

각 자산별로 k=1 (single) / k=2 / k=3 세 그룹씩 top 200 Cal 추출. k=1 은
iter_refine 전체 iv/lev 통합 후 Cal 기준 정렬.
"""

import os
import sys
import pandas as pd

RESEARCH = os.path.dirname(os.path.abspath(__file__))


def _read(rel):
    p = os.path.join(RESEARCH, rel)
    if not os.path.exists(p):
        print(f"[skip] {rel} not found", file=sys.stderr)
        return None
    return pd.read_csv(p)


def extract_k1(asset):
    d = _read("iter_refine/raw_combined.csv")
    if d is None:
        return pd.DataFrame()
    d = d[d["asset"] == asset].dropna(subset=["Cal"]).copy()
    if "MDD" in d.columns:
        d = d.dropna(subset=["MDD"])
    d["k"] = 1
    d["source"] = "iter_refine"
    return d.sort_values("Cal", ascending=False).head(200).reset_index(drop=True)


def extract_ensemble(asset, k):
    tag = "FUT" if asset == "fut" else "SPOT"
    candidates = [
        f"grid_results/topcands_proper_{tag}_1x_k{k}.csv",
        f"grid_results/topcands_proper_{tag}_1x_k{k}_partial.csv",
    ]
    frames = []
    for rel in candidates:
        d = _read(rel)
        if d is not None:
            d["k"] = k
            d["source"] = rel
            frames.append(d)
    if not frames:
        return pd.DataFrame()
    d = pd.concat(frames, ignore_index=True).drop_duplicates(
        subset=[c for c in ["combo_id", "members"] if c in frames[0].columns]
    )
    d = d.dropna(subset=["Cal"])
    if "MDD" in d.columns:
        d = d.dropna(subset=["MDD"])
    return d.sort_values("Cal", ascending=False).head(200).reset_index(drop=True)


def build_asset(asset):
    parts = []
    for k, fn in [(1, extract_k1), (2, lambda a: extract_ensemble(a, 2)),
                   (3, lambda a: extract_ensemble(a, 3))]:
        df = fn(asset)
        print(f"{asset} k={k}: {len(df)} rows", file=sys.stderr)
        parts.append(df)
    return pd.concat(parts, ignore_index=True, sort=False)


def main():
    os.chdir(RESEARCH)
    for asset, out in [("fut", "redesign_top200_fut.csv"),
                        ("spot", "redesign_top200_spot.csv")]:
        df = build_asset(asset)
        df.to_csv(out, index=False)
        print(f"wrote {out} ({len(df)} rows)")
        print(df.groupby("k").size().to_string())
        print()


if __name__ == "__main__":
    main()
