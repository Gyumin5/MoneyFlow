#!/usr/bin/env python3
"""Test 2: Regime decomposition — full-run 후 daily return을 regime label에 매핑.

재설계 배경 (Gemini + Codex 둘 다 지적):
이전 버전은 regime 날짜만 필터하여 비연속 구간을 simulate에 넘겨
시계열 단절 (pct_change, shift) 때문에 CAGR/MDD 왜곡.

올바른 접근:
1. 전체 기간 V21 단독 + V21+C 각각 1회 시뮬레이션 → port_equity 시계열
2. 각 날짜를 BTC 90d regime으로 태깅
3. regime별 group에서 daily return aggregate → 연환산 수익/sharpe-유사 지표
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (P_SPOT, P_FUT, CAP_SPOT, CAP_FUT_OPTS, FULL_END,
                     load_all, load_cached_events, slice_v21,
                     run_spot_combo, run_fut_combo)
from m3_engine_final import metrics as metrics_spot
from m3_engine_futures import metrics as metrics_fut

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def classify_regime(ret_90d: float) -> str:
    if ret_90d > 0.30: return "StrongBull"
    if ret_90d > 0.10: return "WeakBull"
    if ret_90d > -0.10: return "Sideways"
    if ret_90d > -0.30: return "WeakBear"
    return "StrongBear"


def build_regime_labels(btc_daily: pd.Series) -> pd.Series:
    """BTC 종가 시계열 → 90d rolling return → regime label (NaN warmup 제외)."""
    ret90 = btc_daily.pct_change(90).shift(1)
    return ret90.apply(lambda x: classify_regime(float(x)) if pd.notna(x) else np.nan)


def decompose(port_eq: pd.Series, v21_eq: pd.Series, regimes: pd.Series,
               label: str) -> list[dict]:
    """port_eq (V21+C) vs v21_eq (V21 단독) 일별 수익률을 regime 그룹에 귀속.

    연환산 지표:
      annualized_ret = mean_daily_ret × 365.25
      (정확 CAGR은 cumulative 곱 필요하지만 regime별로는 sampling이므로 근사)
    """
    def _norm_idx(s):
        s = s.copy()
        s.index = pd.to_datetime(s.index).tz_localize(None) if getattr(s.index, "tz", None) else pd.to_datetime(s.index)
        s.index = s.index.normalize()
        return s

    v21_d = _norm_idx(v21_eq.astype(float))
    port_d = _norm_idx(port_eq.astype(float))
    reg_d = _norm_idx(regimes.dropna())

    common_idx = v21_d.index.intersection(port_d.index).intersection(reg_d.index)
    v21_d = v21_d.loc[common_idx]
    port_d = port_d.loc[common_idx]
    reg_d = reg_d.loc[common_idx]

    v21_ret = v21_d.pct_change()
    port_ret = port_d.pct_change()

    out = []
    for reg in ["StrongBull", "WeakBull", "Sideways", "WeakBear", "StrongBear"]:
        mask = reg_d == reg
        n_days = int(mask.sum())
        if n_days < 30:
            continue
        vr = v21_ret[mask].dropna()
        pr = port_ret[mask].dropna()
        v21_ann = float(vr.mean() * 365.25)
        port_ann = float(pr.mean() * 365.25)
        v21_vol = float(vr.std() * np.sqrt(365.25)) if vr.std() > 0 else 0.0
        port_vol = float(pr.std() * np.sqrt(365.25)) if pr.std() > 0 else 0.0
        out.append({
            "label": label, "regime": reg, "n_days": n_days,
            "V21_ann_ret": round(v21_ann, 4),
            "V21_ann_vol": round(v21_vol, 4),
            "C_ann_ret": round(port_ann, 4),
            "C_ann_vol": round(port_vol, 4),
            "d_ann_ret": round(port_ann - v21_ann, 4),
            "d_vol": round(port_vol - v21_vol, 4),
        })
    return out


def main():
    v21_s, v21_f, hist, avail, cd = load_all()

    # BTC regime
    btc = cd.get("BTC")
    if btc is None:
        raise RuntimeError("BTC daily not available")
    regimes = build_regime_labels(btc)
    regime_count = regimes.value_counts().to_dict()
    print("=== Regime day count (BTC 90d) ===")
    for k, v in regime_count.items():
        print(f"  {k}: {int(v)} days")

    # 공통 이벤트
    ev_s = load_cached_events("spot")
    ev_f = load_cached_events("fut")

    # 전체 기간 full run
    v21_s_full = slice_v21(v21_s, v21_s.index[0], FULL_END)
    v21_f_full = slice_v21(v21_f, v21_f.index[0], FULL_END)

    rows = []

    # SPOT V21+C full run
    ev_s_full = ev_s[(ev_s["entry_ts"] >= v21_s_full.index[0])
                     & (ev_s["entry_ts"] <= v21_s_full.index[-1])]
    port_s, _ = run_spot_combo(ev_s_full, cd, v21_s_full, hist, CAP_SPOT)
    rows += decompose(port_s, v21_s_full["equity"], regimes,
                       f"spot_cap{CAP_SPOT}")

    # FUT V21+C full run (cap 3종)
    ev_f_full = ev_f[(ev_f["entry_ts"] >= v21_f_full.index[0])
                     & (ev_f["entry_ts"] <= v21_f_full.index[-1])]
    for cap in CAP_FUT_OPTS:
        port_f, _ = run_fut_combo(ev_f_full, cd, v21_f_full, hist, cap)
        rows += decompose(port_f, v21_f_full["equity"], regimes,
                           f"fut_cap{cap}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test2_regime.csv"), index=False)

    # 연도별 분해 (기존 방식: slice 후 재시뮬, 연속 구간이라 안전)
    year_rows = []
    for asset_label, v21, ev, sim, cap, metric_fn in [
        ("spot", v21_s, ev_s, run_spot_combo, CAP_SPOT, metrics_spot),
        ("fut_cap0.12", v21_f, ev_f, run_fut_combo, 0.12, metrics_fut),
        ("fut_cap0.25", v21_f, ev_f, run_fut_combo, 0.25, metrics_fut),
        ("fut_cap0.30", v21_f, ev_f, run_fut_combo, 0.30, metrics_fut),
    ]:
        for yr in [2020, 2021, 2022, 2023, 2024, 2025, 2026]:
            v21_sub = slice_v21(v21, pd.Timestamp(f"{yr}-01-01"),
                                  pd.Timestamp(f"{yr}-12-31"))
            if len(v21_sub) < 30: continue
            ev_sub = ev[(ev["entry_ts"] >= v21_sub.index[0])
                        & (ev["entry_ts"] <= v21_sub.index[-1])]
            m_alone = metric_fn(v21_sub["equity"])
            _, st = sim(ev_sub, cd, v21_sub, hist, cap)
            year_rows.append({
                "asset": asset_label, "year": yr, "n_days": len(v21_sub),
                "V21_Cal": round(m_alone["Cal"], 3),
                "V21_CAGR": round(m_alone["CAGR"], 4),
                "V21_MDD": round(m_alone["MDD"], 4),
                "C_Cal": round(st["Cal"], 3),
                "C_CAGR": round(st["CAGR"], 4),
                "C_MDD": round(st["MDD"], 4),
                "d_Cal": round(st["Cal"] - m_alone["Cal"], 3),
                "d_MDD": round(st["MDD"] - m_alone["MDD"], 4),
                "n_entries": st["n_entries"],
            })
    pd.DataFrame(year_rows).to_csv(os.path.join(OUT, "test2_yearly.csv"), index=False)

    print("\n=== Regime 분해 (full run → daily return 매핑) ===")
    print(df.to_string(index=False))
    print("\n=== 연도별 분해 (독립 구간 재시뮬) ===")
    print(pd.DataFrame(year_rows).to_string(index=False))
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
