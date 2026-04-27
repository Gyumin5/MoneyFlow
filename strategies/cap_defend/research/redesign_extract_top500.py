"""Step 2 — 자산별 Top 500 k=1 추출.

입력 (우선순위):
  1. redesign_univ3_raw.csv (Step 1 재평가 결과 — 완료 후)
  2. iter_refine/raw_combined.csv (재평가 전 임시)
  3. v17_snap_v2_out/raw_combined.csv (주식, 재실행 후)

출력:
  redesign_top500_fut_k1.csv
  redesign_top500_spot_k1.csv
  redesign_top500_stock_k1.csv
"""
from __future__ import annotations
import argparse
import os
import sys

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))

SOURCES = {
    "fut":   ["redesign_univ3_raw_fut.csv", "redesign_univ3_raw.csv", "iter_refine/raw_combined.csv"],
    "spot":  ["redesign_univ3_raw_spot.csv", "redesign_univ3_raw.csv", "iter_refine/raw_combined.csv"],
    "stock": ["redesign_univ3_raw_stock.csv", "v17_snap_v2_out/raw_combined.csv"],
}


def load_for(asset, top_n):
    for rel in SOURCES[asset]:
        path = os.path.join(HERE, rel)
        if os.path.exists(path):
            df = pd.read_csv(path)
            # asset 컬럼이 있으면 필터 (iter_refine/raw_combined)
            if "asset" in df.columns:
                df = df[df["asset"] == asset]
            df = df.dropna(subset=["Cal"]).copy()
            if "MDD" in df.columns:
                df = df.dropna(subset=["MDD"])
            df["source"] = rel
            # dedup by tag (raw_combined 에 iter 중복 가능 — v17_snap_iter_v2 등)
            if "tag" in df.columns:
                df = df.sort_values("Cal", ascending=False).drop_duplicates(subset=["tag"], keep="first")
            df = df.sort_values("Cal", ascending=False).head(top_n).reset_index(drop=True)
            df["k"] = 1
            return df
    return pd.DataFrame()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--top", type=int, default=500)
    args = ap.parse_args()
    for asset in ("fut", "spot", "stock"):
        df = load_for(asset, args.top)
        out = os.path.join(HERE, f"redesign_top{args.top}_{asset}_k1.csv")
        df.to_csv(out, index=False)
        print(f"{asset}: wrote {out} ({len(df)} rows)")


if __name__ == "__main__":
    main()
