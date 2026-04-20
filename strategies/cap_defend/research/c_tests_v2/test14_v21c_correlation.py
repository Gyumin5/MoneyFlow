#!/usr/bin/env python3
"""Test 14: V21 daily return ↔ C contribution daily return correlation.

C가 V21의 독립 알파인지 확인.
- V21 단독 일별 return 시계열
- V21+C 일별 return 시계열
- C 기여분 = V21+C ret - V21 ret
- 두 시계열 correlation
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CAP_SPOT, CAP_FUT_OPTS, HOLDOUT_START, FULL_END,
                     load_all, load_cached_events, slice_v21,
                     run_spot_combo, run_fut_combo)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def _norm_idx(s: pd.Series) -> pd.Series:
    s = s.copy()
    idx = pd.to_datetime(s.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    s.index = idx
    return s


def main():
    v21_s, v21_f, hist, avail, cd = load_all()
    ev_s = load_cached_events("spot")
    ev_f = load_cached_events("fut")

    rows = []
    for span, s, e in [("full", v21_s.index[0], FULL_END),
                        ("holdout", HOLDOUT_START, FULL_END)]:
        v21_sub_s = slice_v21(v21_s, s, e)
        v21_sub_f = slice_v21(v21_f, s, e)
        ev_s_sub = ev_s[(ev_s["entry_ts"] >= s) & (ev_s["entry_ts"] <= e)].copy()
        ev_f_sub = ev_f[(ev_f["entry_ts"] >= s) & (ev_f["entry_ts"] <= e)].copy()

        # SPOT
        port_s, _ = run_spot_combo(ev_s_sub, cd, v21_sub_s, hist, CAP_SPOT)
        v21_ret = _norm_idx(v21_sub_s["equity"]).pct_change().dropna()
        port_ret = _norm_idx(port_s).pct_change().dropna()
        common = v21_ret.index.intersection(port_ret.index)
        vr, pr = v21_ret.loc[common], port_ret.loc[common]
        c_contrib = pr - vr
        corr_with_v21 = float(c_contrib.corr(vr))
        corr_port_v21 = float(pr.corr(vr))
        rows.append({"asset":"spot", "span":span, "cap":CAP_SPOT,
                     "corr(C_contrib, V21)": round(corr_with_v21, 3),
                     "corr(V21+C, V21)": round(corr_port_v21, 3),
                     "C_contrib_mean_bps_per_day": round(float(c_contrib.mean() * 10000), 2),
                     "C_contrib_vol_bps_per_day": round(float(c_contrib.std() * 10000), 2)})

        # FUT 3caps
        for cap in CAP_FUT_OPTS:
            port_f, _ = run_fut_combo(ev_f_sub, cd, v21_sub_f, hist, cap)
            v21_ret = _norm_idx(v21_sub_f["equity"]).pct_change().dropna()
            port_ret = _norm_idx(port_f).pct_change().dropna()
            common = v21_ret.index.intersection(port_ret.index)
            vr, pr = v21_ret.loc[common], port_ret.loc[common]
            c_contrib = pr - vr
            corr_with_v21 = float(c_contrib.corr(vr))
            corr_port_v21 = float(pr.corr(vr))
            rows.append({"asset":f"fut_cap{cap}", "span":span, "cap":cap,
                         "corr(C_contrib, V21)": round(corr_with_v21, 3),
                         "corr(V21+C, V21)": round(corr_port_v21, 3),
                         "C_contrib_mean_bps_per_day": round(float(c_contrib.mean() * 10000), 2),
                         "C_contrib_vol_bps_per_day": round(float(c_contrib.std() * 10000), 2)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test14_v21c_correlation.csv"), index=False)
    print("=== V21 vs C contribution correlation ===")
    print("corr(C_contrib, V21) < 0.3 → 독립 알파")
    print("corr(C_contrib, V21) > 0.7 → V21과 실질 동일 (diversification 없음)")
    print(df.to_string(index=False))
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
