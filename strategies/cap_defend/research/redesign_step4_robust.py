#!/usr/bin/env python3
"""Redesign Step 4: 베스트 config 심층 robustness.

Config: Stock V17 60% + spot_k2_dbaf3f9c 25% + fut_L3_6bdcbc78 15%, abs 8pp band.

검증:
1. 연도별 분해
2. Block bootstrap (holdout 기간)
3. Single-member ablation (spot k=2 멤버 각자 제거)
4. Anchor shift (2020-10, 2021-01, 2021-04 시작점)
5. 주식 제거 / 현물 제거 / 선물 제거 각 조합
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from phase_common import (parse_tag, run_single_target, run_spot_ensemble, build_trace,
                          preload_futures, equity_metrics, FULL_END)
from phase4_3asset import build_ensemble_full_equity, mix_eq
import run_3asset_grid as r3

OUT = os.path.join(HERE, "redesign_step4")
os.makedirs(OUT, exist_ok=True)

START = "2020-10-01"
TRAIN_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"

BEST_SPOT = "ENS_spot_k2_dbaf3f9c"
BEST_FUT = "ENS_fut_L3_k1_6bdcbc78"
BEST_WEIGHTS = {"st": 0.60, "sp": 0.25, "fut": 0.15}
BEST_BAND = {"st": 0.08, "sp": 0.08, "fut": 0.08}


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
    eq_n = eq / eq.iloc[0]
    return equity_metrics(eq_n)


def build_spot_k2():
    """spot_4h_S240_M42_488_d0.05_SN360_L1 + spot_4h_S240_M20_720_b0.70_SN96_L1 EW."""
    members = [
        "spot_4h_S240_M42_488_d0.05_SN360_L1",
        "spot_4h_S240_M20_720_b0.70_SN96_L1",
    ]
    cfgs = {}
    for m in members:
        meta = parse_tag(m)
        cfgs[m] = {k: meta[k] for k in ("interval","sma","ms","ml","vol_mode","vol_thr","snap")}
    r = run_spot_ensemble(cfgs, {m: 0.5 for m in members}, START, end=FULL_END, want_equity=True)
    return r.get("_equity")


def build_spot_k1(tag):
    meta = parse_tag(tag)
    cfg = {k: meta[k] for k in ("interval","sma","ms","ml","vol_mode","vol_thr","snap")}
    r = run_single_target("spot", cfg, lev=1.0, anchor=START, end=FULL_END, want_equity=True)
    return r.get("_equity")


def build_fut_single(tag):
    meta = parse_tag(tag)
    cfg = {k: meta[k] for k in ("interval","sma","ms","ml","vol_mode","vol_thr","snap")}
    r = run_single_target("fut", cfg, lev=float(meta["lev"]), anchor=START, end=FULL_END, want_equity=True)
    return r.get("_equity")


def main():
    print("Loading V17 stock equity...")
    stock_eq = r3.load_stock_v17()

    print("Building spot k=2 ensemble equity...")
    spot_eq = build_spot_k2()

    print("Building fut L3 equity...")
    fut_eq = build_fut_single("fut_1D_S44_M28_127_d0.05_SN24_L3")

    print("Building individual spot k=2 members (ablation)...")
    spot_m1 = build_spot_k1("spot_4h_S240_M42_488_d0.05_SN360_L1")
    spot_m2 = build_spot_k1("spot_4h_S240_M20_720_b0.70_SN96_L1")

    # ── Baseline mix ──
    print("\n=== Baseline (60/25/15, abs 8pp) ===")
    mix = mix_eq({"st": stock_eq, "sp": spot_eq, "fut": fut_eq},
                 BEST_WEIGHTS, BEST_BAND)
    m_full = slice_cal(mix, START, FULL_END)
    m_train = slice_cal(mix, START, TRAIN_END)
    m_holdout = slice_cal(mix, HOLDOUT_START, FULL_END)
    print(f"full:    Cal={m_full['Cal']:.2f} CAGR={m_full['CAGR']:.2%} MDD={m_full['MDD']:.2%} Sh={m_full['Sh']:.2f}")
    print(f"train:   Cal={m_train['Cal']:.2f} CAGR={m_train['CAGR']:.2%} MDD={m_train['MDD']:.2%}")
    print(f"holdout: Cal={m_holdout['Cal']:.2f} CAGR={m_holdout['CAGR']:.2%} MDD={m_holdout['MDD']:.2%}")

    # ── 연도별 ──
    print("\n=== 연도별 분해 ===")
    rows_y = []
    if not isinstance(mix.index, pd.DatetimeIndex):
        mix.index = pd.to_datetime(mix.index)
    for y in sorted(set(mix.index.year)):
        sub = mix[mix.index.year == y]
        if len(sub) < 30 or sub.iloc[0] <= 0:
            continue
        sub_n = sub / sub.iloc[0]
        m = equity_metrics(sub_n)
        ret = float(sub.iloc[-1]/sub.iloc[0] - 1)
        print(f"  {y}: ret={ret:+.2%}  Cal={m['Cal']:.2f}  MDD={m['MDD']:.2%}")
        rows_y.append({"year": y, "ret": round(ret,4), "Cal": round(m["Cal"],4),
                       "MDD": round(m["MDD"],4)})
    pd.DataFrame(rows_y).to_csv(os.path.join(OUT, "yearly.csv"), index=False)

    # ── 자산 ablation ──
    print("\n=== Asset ablation ===")
    ablations = [
        ("stock only",      {"st": 1.00}, {"st": stock_eq}),
        ("no stock (40/60 spot/fut)", {"sp": 0.625, "fut": 0.375},
            {"sp": spot_eq, "fut": fut_eq}),
        ("no spot (70/30 st/fut)", {"st": 0.70, "fut": 0.30},
            {"st": stock_eq, "fut": fut_eq}),
        ("no fut (70/30 st/sp)", {"st": 0.70, "sp": 0.30},
            {"st": stock_eq, "sp": spot_eq}),
        ("baseline", BEST_WEIGHTS, {"st": stock_eq, "sp": spot_eq, "fut": fut_eq}),
    ]
    for name, w, eqs in ablations:
        band = {k: 0.08 for k in w}
        mx = mix_eq(eqs, w, band)
        m_h = slice_cal(mx, HOLDOUT_START, FULL_END)
        m_f = slice_cal(mx, START, FULL_END)
        print(f"  {name:<40}  full Cal={m_f['Cal']:.2f} / holdout Cal={m_h['Cal']:.2f} "
              f"CAGR_h={m_h['CAGR']:.2%} MDD_h={m_h['MDD']:.2%}")

    # ── Spot member ablation ──
    print("\n=== Spot k=2 member ablation ===")
    member_tests = [
        ("M42_488 only (drop M20_720)", spot_m1),
        ("M20_720 only (drop M42_488)", spot_m2),
        ("both (baseline)", spot_eq),
    ]
    for name, sp in member_tests:
        mx = mix_eq({"st": stock_eq, "sp": sp, "fut": fut_eq}, BEST_WEIGHTS, BEST_BAND)
        m_h = slice_cal(mx, HOLDOUT_START, FULL_END)
        m_f = slice_cal(mx, START, FULL_END)
        print(f"  {name:<40}  full Cal={m_f['Cal']:.2f} / holdout Cal={m_h['Cal']:.2f} "
              f"MDD_h={m_h['MDD']:.2%}")

    # ── Block bootstrap on holdout ──
    print("\n=== Block bootstrap (holdout 60일 블록 × 300) ===")
    holdout = mix[(mix.index >= HOLDOUT_START) & (mix.index <= FULL_END)]
    rets = holdout.pct_change().dropna().values
    n_r = len(rets)
    if n_r > 120:
        rng = np.random.default_rng(42)
        cagr_l, mdd_l = [], []
        for _ in range(300):
            starts = rng.integers(0, n_r-60, size=(n_r//60)+1)
            blocks = np.concatenate([rets[s:s+60] for s in starts])[:n_r]
            eq_b = (1 + blocks).cumprod()
            if eq_b[-1] <= 0: continue
            yrs = len(blocks) / 365.25
            cagr_l.append(eq_b[-1]**(1/yrs) - 1)
            mx = np.maximum.accumulate(eq_b)
            mdd_l.append(float(((eq_b-mx)/mx).min()))
        print(f"  CAGR p5={np.percentile(cagr_l,5):.2%} p50={np.percentile(cagr_l,50):.2%} "
              f"p95={np.percentile(cagr_l,95):.2%}")
        print(f"  MDD  p5={np.percentile(mdd_l,5):.2%} p50={np.percentile(mdd_l,50):.2%} "
              f"worst={min(mdd_l):.2%}")

    # ── Save baseline equity ──
    mix.to_csv(os.path.join(OUT, "baseline_mix_equity.csv"))
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
