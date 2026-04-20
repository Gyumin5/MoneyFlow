#!/usr/bin/env python3
"""Redesign Step 6: Walk-forward (weight × band) 안정성 검증.

5개 rolling split × weight grid × band grid → 각 split holdout 최고 config 계수.
60/25/15 + abs 8pp가 진짜 robust인지 확인.
"""
from __future__ import annotations
import os, sys
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from phase_common import (parse_tag, run_single_target, run_spot_ensemble,
                          equity_metrics, FULL_END)
from phase4_3asset import mix_eq
import run_3asset_grid as r3

OUT = os.path.join(HERE, "redesign_step6")
os.makedirs(OUT, exist_ok=True)

START = "2020-10-01"


def slice_cal(eq, start=None, end=None):
    eq = eq.dropna()
    if not isinstance(eq.index, pd.DatetimeIndex):
        eq.index = pd.to_datetime(eq.index)
    if getattr(eq.index, "tz", None) is not None:
        eq.index = eq.index.tz_localize(None)
    if start:
        eq = eq[eq.index >= start]
    if end:
        eq = eq[eq.index <= end]
    if len(eq) < 30 or eq.iloc[0] <= 0:
        return {"Cal": 0.0, "CAGR": 0.0, "MDD": 0.0}
    return equity_metrics(eq / eq.iloc[0])


def build_spot_k2():
    members = ["spot_4h_S240_M42_488_d0.05_SN360_L1",
               "spot_4h_S240_M20_720_b0.70_SN96_L1"]
    cfgs = {m: {k: parse_tag(m)[k] for k in ("interval","sma","ms","ml","vol_mode","vol_thr","snap")}
            for m in members}
    r = run_spot_ensemble(cfgs, {m: 0.5 for m in members}, START, end=FULL_END, want_equity=True)
    return r.get("_equity")


def build_fut(tag):
    meta = parse_tag(tag)
    cfg = {k: meta[k] for k in ("interval","sma","ms","ml","vol_mode","vol_thr","snap")}
    return run_single_target("fut", cfg, lev=float(meta["lev"]), anchor=START,
                             end=FULL_END, want_equity=True).get("_equity")


def main():
    print("Loading equities...")
    stock_eq = r3.load_stock_v17()
    spot_eq = build_spot_k2()
    fut_eq = build_fut("fut_1D_S44_M28_127_d0.05_SN24_L3")

    weight_grid = [
        (0.60, 0.35, 0.05),
        (0.60, 0.30, 0.10),
        (0.60, 0.25, 0.15),
        (0.60, 0.22, 0.18),
        (0.60, 0.20, 0.20),
        (0.60, 0.18, 0.22),
        (0.60, 0.15, 0.25),
    ]
    band_grid = [0.03, 0.05, 0.08, 0.10, 0.12]
    splits = [
        ("2023-01", "2022-12-31", "2023-01-01"),
        ("2023-07", "2023-06-30", "2023-07-01"),
        ("2024-01", "2023-12-31", "2024-01-01"),
        ("2024-07", "2024-06-30", "2024-07-01"),
        ("2025-01", "2024-12-31", "2025-01-01"),
    ]

    rows = []
    for w in weight_grid:
        st_w, sp_w, fu_w = w
        for bp in band_grid:
            mx = mix_eq({"st": stock_eq, "sp": spot_eq, "fut": fut_eq},
                        {"st": st_w, "sp": sp_w, "fut": fu_w},
                        {"st": bp, "sp": bp, "fut": bp})
            for sname, te, hs in splits:
                mh = slice_cal(mx, hs, FULL_END)
                rows.append({
                    "split": sname, "sp_w": sp_w, "fu_w": fu_w,
                    "band": bp,
                    "Cal_h": round(mh["Cal"], 4),
                    "CAGR_h": round(mh["CAGR"], 4),
                    "MDD_h": round(mh["MDD"], 4),
                })
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "walkforward.csv"), index=False)

    # Per split top-5
    print("\n=== Per-split Top-5 (by Cal_h) ===")
    for s in df["split"].unique():
        sub = df[df["split"] == s].sort_values("Cal_h", ascending=False).head(5)
        print(f"\n-- {s} --")
        print(sub[["sp_w","fu_w","band","Cal_h","CAGR_h","MDD_h"]].to_string(index=False))

    # 60/25/15 × 8pp vs others: rank across splits
    print("\n=== 60/25/15 × 8pp rank across splits ===")
    ranks = []
    for s in df["split"].unique():
        sub = df[df["split"] == s].sort_values("Cal_h", ascending=False).reset_index(drop=True)
        hit = sub[(sub["sp_w"]==0.25) & (sub["fu_w"]==0.15) & (sub["band"]==0.08)]
        if len(hit):
            r = hit.index[0] + 1
            ranks.append((s, r, float(hit["Cal_h"].iloc[0])))
    for s, r, c in ranks:
        print(f"  {s}: rank {r}/{len(df['split'].unique())*len(weight_grid)*len(band_grid)} Cal_h={c:.2f}")

    # Aggregate: mean holdout Cal per (weight, band)
    print("\n=== 평균 holdout Cal (weight × band, 5 splits) ===")
    agg = df.groupby(["sp_w","fu_w","band"]).agg(
        mean_Cal=("Cal_h","mean"),
        min_Cal=("Cal_h","min"),
        mean_CAGR=("CAGR_h","mean"),
        worst_MDD=("MDD_h","min"),
    ).round(3).sort_values("mean_Cal", ascending=False)
    print(agg.head(15))
    agg.to_csv(os.path.join(OUT, "walkforward_agg.csv"))

    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
