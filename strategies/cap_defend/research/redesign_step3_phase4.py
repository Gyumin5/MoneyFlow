#!/usr/bin/env python3
"""Redesign Step 3: 7 final 후보 × stock V17 × 3자산 mix + train/holdout 분리 테스트.

핵심: abs 8pp vs sleeve-relative (r 0.2, 0.3) band 비교.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from phase_common import (parse_tag, run_single_target, run_spot_ensemble, build_trace,
                          preload_futures, equity_metrics, FULL_END)
from phase4_3asset import build_ensemble_full_equity, mix_eq
import run_3asset_grid as r3

OUT = os.path.join(HERE, "redesign_step3")
os.makedirs(OUT, exist_ok=True)

START = "2020-10-01"
TRAIN_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"


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
    eq_n = eq / eq.iloc[0]
    return equity_metrics(eq_n)


def main():
    t0 = time.time()
    cands = pd.read_csv(os.path.join(HERE, "redesign_step2", "final_candidates.csv"))
    spot_c = cands[cands["asset"] == "spot"]
    fut_c = cands[cands["asset"] == "fut"]
    print(f"Candidates: spot={len(spot_c)} fut={len(fut_c)}")

    # Stock V17
    print("Loading V17 stock equity...")
    stock_eq = r3.load_stock_v17()

    # Ensemble equities
    print("Building ensemble equities...")
    spot_eqs = {}
    for _, r in spot_c.iterrows():
        print(f"  spot {r['ensemble_tag']}")
        spot_eqs[r["ensemble_tag"]] = build_ensemble_full_equity(r)
    fut_eqs = {}
    fut_levs = {}
    for _, r in fut_c.iterrows():
        print(f"  fut {r['ensemble_tag']} L{int(r['lev'])}")
        fut_eqs[r["ensemble_tag"]] = build_ensemble_full_equity(r)
        fut_levs[r["ensemble_tag"]] = float(r["lev"])

    # Weight grid
    weights_list = [
        (0.60, 0.35, 0.05),   # baseline
        (0.60, 0.30, 0.10),
        (0.60, 0.25, 0.15),
        (0.60, 0.20, 0.20),
        (0.60, 0.15, 0.25),
        (0.60, 0.10, 0.30),
    ]

    # Band specs
    band_specs = []
    for pp in (0.05, 0.08, 0.10):
        band_specs.append(("abs", pp, {"st": pp, "sp": pp, "fut": pp}))
    for ratio in (0.15, 0.20, 0.30):
        band_specs.append(("sleeve", ratio, None))  # filled per weight

    rows = []
    n = 0
    for sp_id, sp_eq in spot_eqs.items():
        for fu_id, fu_eq in fut_eqs.items():
            lev = fut_levs[fu_id]
            for w in weights_list:
                st_w, sp_w, fu_w = w
                for mode, param, band_d in band_specs:
                    if mode == "sleeve":
                        bd = {"st": round(st_w * param, 4),
                              "sp": round(sp_w * param, 4),
                              "fut": round(fu_w * param, 4)}
                    else:
                        bd = {"st": param, "sp": param, "fut": param}
                    try:
                        mix = mix_eq({"st": stock_eq, "sp": sp_eq, "fut": fu_eq},
                                     {"st": st_w, "sp": sp_w, "fut": fu_w}, bd)
                    except Exception as e:
                        print(f"FAIL {sp_id}/{fu_id}/{w}/{mode}/{param}: {e}")
                        continue
                    m_full = slice_cal(mix, START, FULL_END)
                    m_train = slice_cal(mix, START, TRAIN_END)
                    m_holdout = slice_cal(mix, HOLDOUT_START, FULL_END)
                    rows.append({
                        "spot": sp_id, "fut": fu_id, "fut_lev": lev,
                        "st_w": st_w, "sp_w": sp_w, "fu_w": fu_w,
                        "band_mode": mode, "band_param": param,
                        "band_st": bd["st"], "band_sp": bd["sp"], "band_fut": bd["fut"],
                        "Cal_full": round(m_full["Cal"], 4),
                        "Cal_train": round(m_train["Cal"], 4),
                        "Cal_holdout": round(m_holdout["Cal"], 4),
                        "CAGR_full": round(m_full["CAGR"], 4),
                        "CAGR_holdout": round(m_holdout["CAGR"], 4),
                        "MDD_full": round(m_full["MDD"], 4),
                        "MDD_holdout": round(m_holdout["MDD"], 4),
                    })
                    n += 1
    print(f"Total configs: {n} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "mix_eval.csv"), index=False)

    # Top 20 by holdout Cal
    print("\n=== Top 20 by Cal_holdout (전체) ===")
    cols = ["spot","fut","fut_lev","st_w","sp_w","fu_w","band_mode","band_param",
            "Cal_full","Cal_train","Cal_holdout","CAGR_holdout","MDD_holdout"]
    print(df.sort_values("Cal_holdout", ascending=False).head(20)[cols].to_string(index=False))

    # Per fut_lev best
    print("\n=== 각 레버리지별 best (by Cal_holdout) ===")
    for lev in sorted(df["fut_lev"].unique()):
        sub = df[df["fut_lev"] == lev].sort_values("Cal_holdout", ascending=False).head(3)
        print(f"\n-- L{int(lev)} --")
        print(sub[cols].to_string(index=False))

    # Abs vs Sleeve 비교
    print("\n=== abs vs sleeve 비교 (같은 weight, same spot/fut) ===")
    pivot = df.groupby(["band_mode"]).agg(
        n=("Cal_holdout","count"),
        Cal_holdout_mean=("Cal_holdout","mean"),
        Cal_holdout_med=("Cal_holdout","median"),
        MDD_holdout_mean=("MDD_holdout","mean"),
        MDD_holdout_worst=("MDD_holdout","min"),
    ).round(3)
    print(pivot)

    print(f"\n저장: {OUT}/mix_eval.csv")


if __name__ == "__main__":
    main()
