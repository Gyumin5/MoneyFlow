#!/usr/bin/env python3
"""Test 11: Universe / n_pick / swap_edge 민감도."""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CAP_SPOT, HOLDOUT_START, FULL_END,
                     load_all, load_cached_events, slice_v21)
from m3_engine_final import simulate
from m3_engine_futures import simulate_fut

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def main():
    v21_s, v21_f, hist, avail, cd = load_all()
    ev_s = load_cached_events("spot")
    ev_f = load_cached_events("fut")
    v21_sH = slice_v21(v21_s, HOLDOUT_START, FULL_END)
    v21_fH = slice_v21(v21_f, HOLDOUT_START, FULL_END)
    ev_sH = ev_s[(ev_s["entry_ts"] >= HOLDOUT_START) & (ev_s["entry_ts"] <= FULL_END)].copy()
    ev_fH = ev_f[(ev_f["entry_ts"] >= HOLDOUT_START) & (ev_f["entry_ts"] <= FULL_END)].copy()

    rows = []
    for uni in [10, 15, 20, 30]:
        for np_ in [1, 2, 3]:
            for swap in [0, 1, 2, 3]:
                # spot
                _, st_s = simulate(ev_sH, cd, v21_sH.copy(), hist,
                                    n_pick=np_, cap_per_slot=CAP_SPOT,
                                    universe_size=uni, tx_cost=0.003,
                                    swap_edge_threshold=swap)
                rows.append({"asset":"spot", "uni":uni, "n_pick":np_, "swap":swap,
                             "Cal":round(st_s["Cal"],3), "CAGR":round(st_s["CAGR"],4),
                             "MDD":round(st_s["MDD"],4), "entries":st_s["n_entries"]})
                # fut cap0.25
                _, st_f = simulate_fut(ev_fH, cd, v21_fH.copy(), hist,
                                        n_pick=np_, cap_per_slot=0.25,
                                        universe_size=uni, tx_cost=0.003,
                                        swap_edge_threshold=swap, leverage=3.0)
                rows.append({"asset":"fut_cap0.25", "uni":uni, "n_pick":np_, "swap":swap,
                             "Cal":round(st_f["Cal"],3), "CAGR":round(st_f["CAGR"],4),
                             "MDD":round(st_f["MDD"],4), "entries":st_f["n_entries"]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test11_universe_npick.csv"), index=False)
    for asset_val in df["asset"].unique():
        sub = df[df["asset"] == asset_val].sort_values("Cal", ascending=False)
        print(f"\n=== {asset_val} top 10 (by Cal_holdout) ===")
        print(sub.head(10).to_string(index=False))
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
