#!/usr/bin/env python3
"""Redesign Step 5: TX/delay/slippage 감도 + rolling holdout split + L2 대안 비교."""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from phase_common import (parse_tag, run_single_target, run_spot_ensemble,
                          equity_metrics, FULL_END)
from phase4_3asset import mix_eq
import run_3asset_grid as r3

OUT = os.path.join(HERE, "redesign_step5")
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
        return {"Cal": 0.0, "CAGR": 0.0, "MDD": 0.0, "Sh": 0.0}
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
    return run_single_target("fut", cfg, lev=float(meta["lev"]),
                             anchor=START, end=FULL_END, want_equity=True).get("_equity")


def main():
    print("Loading stock V17 + spot k2 + fut L3 + fut L2...")
    stock_eq = r3.load_stock_v17()
    spot_eq = build_spot_k2()
    fut_L3 = build_fut("fut_1D_S44_M28_127_d0.05_SN24_L3")
    fut_L2 = build_fut("fut_1D_S44_M14_127_d0.05_SN24_L2")  # ENS_fut_L2_k1_6eff4b1a

    BAND = {"st": 0.08, "sp": 0.08, "fut": 0.08}

    # ── Rolling holdout split comparison ──
    print("\n=== Rolling holdout split ===")
    mix_L3 = mix_eq({"st": stock_eq, "sp": spot_eq, "fut": fut_L3},
                    {"st": 0.60, "sp": 0.25, "fut": 0.15}, BAND)
    mix_L2 = mix_eq({"st": stock_eq, "sp": spot_eq, "fut": fut_L2},
                    {"st": 0.60, "sp": 0.35, "fut": 0.05}, BAND)
    # 60/35/5 fut_L2 (현재 live config과 유사)
    mix_L2_live = mix_eq({"st": stock_eq, "sp": spot_eq, "fut": fut_L2},
                         {"st": 0.60, "sp": 0.35, "fut": 0.05}, BAND)

    splits = [
        ("train~2022-12 / holdout 2023+", "2022-12-31", "2023-01-01"),
        ("train~2023-06 / holdout 2023-07+", "2023-06-30", "2023-07-01"),
        ("train~2023-12 / holdout 2024+", "2023-12-31", "2024-01-01"),
        ("train~2024-06 / holdout 2024-07+", "2024-06-30", "2024-07-01"),
        ("train~2024-12 / holdout 2025+", "2024-12-31", "2025-01-01"),
    ]
    rows = []
    for name, te, hs in splits:
        for lab, mx in [("L3-60/25/15", mix_L3), ("L2-60/35/5", mix_L2_live)]:
            mt = slice_cal(mx, START, te)
            mh = slice_cal(mx, hs, FULL_END)
            print(f"  {name:<40}  {lab:<14}  train Cal={mt['Cal']:.2f}  "
                  f"holdout Cal={mh['Cal']:.2f}  ret_h={mh['CAGR']:.2%}")
            rows.append({"split": name, "config": lab,
                         "train_Cal": round(mt["Cal"],4),
                         "holdout_Cal": round(mh["Cal"],4),
                         "holdout_CAGR": round(mh["CAGR"],4),
                         "holdout_MDD": round(mh["MDD"],4)})
    pd.DataFrame(rows).to_csv(os.path.join(OUT, "rolling_holdout.csv"), index=False)

    # ── L2 vs L3 holdout 직접 비교 (동일 weight) ──
    print("\n=== L2 vs L3 동일 weight 비교 ===")
    print("weight=60/25/15, holdout=2024+, band abs 8pp")
    mix_L2_eq = mix_eq({"st": stock_eq, "sp": spot_eq, "fut": fut_L2},
                       {"st": 0.60, "sp": 0.25, "fut": 0.15}, BAND)
    for lab, mx in [("L2", mix_L2_eq), ("L3", mix_L3)]:
        m_f = slice_cal(mx, START, FULL_END)
        m_h = slice_cal(mx, "2024-01-01", FULL_END)
        print(f"  {lab}:  full Cal={m_f['Cal']:.2f}  holdout Cal={m_h['Cal']:.2f}  "
              f"CAGR_h={m_h['CAGR']:.2%}  MDD_h={m_h['MDD']:.2%}")

    # ── 비중 민감도: spot vs fut 비율 변화 ──
    print("\n=== Weight 민감도 (fut L3 기준) ===")
    for fu_w in [0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        sp_w = round(0.40 - fu_w, 4)
        if sp_w < 0: continue
        mx = mix_eq({"st": stock_eq, "sp": spot_eq, "fut": fut_L3},
                    {"st": 0.60, "sp": sp_w, "fut": fu_w}, BAND)
        m_f = slice_cal(mx, START, FULL_END)
        m_h = slice_cal(mx, "2024-01-01", FULL_END)
        print(f"  60/{int(sp_w*100):02d}/{int(fu_w*100):02d}:  full Cal={m_f['Cal']:.2f}  "
              f"holdout Cal={m_h['Cal']:.2f}  CAGR_h={m_h['CAGR']:.2%}  MDD_h={m_h['MDD']:.2%}")

    # ── Band 재확인: 다양한 band ──
    print("\n=== Band 민감도 (60/25/15 × L3) ===")
    for bp in [0.03, 0.05, 0.08, 0.10, 0.12, 0.15]:
        mx = mix_eq({"st": stock_eq, "sp": spot_eq, "fut": fut_L3},
                    {"st": 0.60, "sp": 0.25, "fut": 0.15},
                    {"st": bp, "sp": bp, "fut": bp})
        m_h = slice_cal(mx, "2024-01-01", FULL_END)
        m_f = slice_cal(mx, START, FULL_END)
        print(f"  band {bp*100:4.1f}pp:  full Cal={m_f['Cal']:.2f}  "
              f"holdout Cal={m_h['Cal']:.2f}  MDD_h={m_h['MDD']:.2%}")

    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
