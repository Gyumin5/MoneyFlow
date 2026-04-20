#!/usr/bin/env python3
"""Test 15: Cross-anchor — 다른 시작일에서 V21+C 성과 변동성.

시작일 2020-10 / 2021-01 / 2021-04 / 2021-07 로 이동하며 성과 차이 확인.
단일 앵커 의존성 방지.
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CAP_SPOT, FULL_END, load_all, load_cached_events,
                     slice_v21, run_spot_combo, run_fut_combo)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def main():
    v21_s, v21_f, hist, avail, cd = load_all()
    ev_s = load_cached_events("spot")
    ev_f = load_cached_events("fut")

    anchors = ["2020-10-01", "2021-01-01", "2021-04-01", "2021-07-01", "2022-01-01"]
    rows = []
    for a in anchors:
        astart = pd.Timestamp(a)
        v21_sub_s = slice_v21(v21_s, astart, FULL_END)
        v21_sub_f = slice_v21(v21_f, astart, FULL_END)
        ev_s_sub = ev_s[(ev_s["entry_ts"] >= astart) & (ev_s["entry_ts"] <= FULL_END)].copy()
        ev_f_sub = ev_f[(ev_f["entry_ts"] >= astart) & (ev_f["entry_ts"] <= FULL_END)].copy()

        _, st_s = run_spot_combo(ev_s_sub, cd, v21_sub_s, hist, CAP_SPOT)
        rows.append({"anchor":a, "asset":"spot", "cap":CAP_SPOT,
                     "Cal":round(st_s["Cal"],3), "CAGR":round(st_s["CAGR"],4),
                     "MDD":round(st_s["MDD"],4), "entries":st_s["n_entries"]})

        for cap in [0.12, 0.25, 0.30]:
            _, st_f = run_fut_combo(ev_f_sub, cd, v21_sub_f, hist, cap)
            rows.append({"anchor":a, "asset":f"fut_cap{cap}", "cap":cap,
                         "Cal":round(st_f["Cal"],3), "CAGR":round(st_f["CAGR"],4),
                         "MDD":round(st_f["MDD"],4), "entries":st_f["n_entries"]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test15_cross_anchor.csv"), index=False)
    print(df.to_string(index=False))
    print("\n해석: anchor 바뀌어도 Cal 분산 작으면 robust.")
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
