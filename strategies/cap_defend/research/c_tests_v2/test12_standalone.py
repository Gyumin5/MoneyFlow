#!/usr/bin/env python3
"""Test 12: C standalone (V21 없이, 단독으로 돈 버는 전략인가?).

"단독 실패하면 섞어도 의미 없다" 원칙.
V21 없이 C만 돌리면 어떤 성과가 나오는지 측정.
fake V21 = 현금 100% 유지 (equity 상수 + cash_ratio=1.0).
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (P_SPOT, P_FUT, CAP_SPOT, CAP_FUT_OPTS,
                     HOLDOUT_START, FULL_END,
                     load_all, load_cached_events)
from m3_engine_final import simulate
from m3_engine_futures import simulate_fut

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def fake_v21(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """V21이 아무것도 안 하는 가정. equity = 1.0 상수, cash_ratio = 1.0."""
    df = pd.DataFrame({
        "equity": 1.0,
        "cash_ratio": 1.0,
        "v21_ret": 0.0,
        "prev_cash": 1.0,
    }, index=dates)
    return df


def main():
    v21_s, v21_f, hist, avail, cd = load_all()
    ev_s = load_cached_events("spot")
    ev_f = load_cached_events("fut")

    # V21 일자 기준 fake v21 생성
    fv_s = fake_v21(v21_s.index)
    fv_f = fake_v21(v21_f.index)

    rows = []
    # Full + Holdout 구간별
    for span, s, e in [("full", v21_s.index[0], FULL_END),
                        ("holdout", HOLDOUT_START, FULL_END)]:
        fv_s_sub = fv_s[(fv_s.index >= s) & (fv_s.index <= e)]
        fv_f_sub = fv_f[(fv_f.index >= s) & (fv_f.index <= e)]
        ev_s_sub = ev_s[(ev_s["entry_ts"] >= s) & (ev_s["entry_ts"] <= e)].copy()
        ev_f_sub = ev_f[(ev_f["entry_ts"] >= s) & (ev_f["entry_ts"] <= e)].copy()

        # spot standalone
        _, st = simulate(ev_s_sub, cd, fv_s_sub.copy(), hist,
                         n_pick=1, cap_per_slot=CAP_SPOT, universe_size=15,
                         tx_cost=0.003, swap_edge_threshold=1)
        rows.append({"asset":"spot_standalone", "span":span, "cap":CAP_SPOT,
                     "Cal":round(st["Cal"],3), "CAGR":round(st["CAGR"],4),
                     "MDD":round(st["MDD"],4), "entries":st["n_entries"]})

        # fut standalone (cap 3종)
        for cap in CAP_FUT_OPTS:
            _, st = simulate_fut(ev_f_sub, cd, fv_f_sub.copy(), hist,
                                  n_pick=1, cap_per_slot=cap, universe_size=15,
                                  tx_cost=0.003, swap_edge_threshold=1, leverage=3.0)
            rows.append({"asset":f"fut_standalone_cap{cap}", "span":span, "cap":cap,
                         "Cal":round(st["Cal"],3), "CAGR":round(st["CAGR"],4),
                         "MDD":round(st["MDD"],4), "entries":st["n_entries"],
                         "liq":st.get("n_liquidations",0)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test12_standalone.csv"), index=False)
    print("=== C Standalone (V21 없이) ===")
    print(df.to_string(index=False))
    print("\n해석:")
    print("- holdout Cal >= 0.5: 단독 엣지 존재")
    print("- holdout Cal < 0.5: V21 없으면 무의미 (보조 전략)")
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
