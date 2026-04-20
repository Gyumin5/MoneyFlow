#!/usr/bin/env python3
"""Test 16: Top N event removal — 희소 수익 의존도.

Holdout에서 pnl 상위 N event 제거했을 때 성과 변화.
Top 10 제거해도 Cal >= 0.5 유지되면 robust.
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CAP_SPOT, CAP_FUT_OPTS, HOLDOUT_START, FULL_END,
                     load_all, load_cached_events, slice_v21,
                     run_spot_combo, run_fut_combo)

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
    for n in [0, 1, 3, 5, 10, 20]:
        # spot
        ev_keep = ev_sH.sort_values("pnl_pct", ascending=False).iloc[n:] if n else ev_sH
        _, st_s = run_spot_combo(ev_keep, cd, v21_sH, hist, CAP_SPOT)
        rows.append({"asset":"spot", "cap":CAP_SPOT, "top_n_removed":n,
                     "Cal":round(st_s["Cal"],3), "CAGR":round(st_s["CAGR"],4),
                     "MDD":round(st_s["MDD"],4)})
        # fut 3caps
        for cap in CAP_FUT_OPTS:
            ev_keep = ev_fH.sort_values("pnl_pct", ascending=False).iloc[n:] if n else ev_fH
            _, st_f = run_fut_combo(ev_keep, cd, v21_fH, hist, cap)
            rows.append({"asset":f"fut_cap{cap}", "cap":cap, "top_n_removed":n,
                         "Cal":round(st_f["Cal"],3), "CAGR":round(st_f["CAGR"],4),
                         "MDD":round(st_f["MDD"],4)})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test16_top_removal.csv"), index=False)
    for asset_val in df["asset"].unique():
        print(f"\n=== {asset_val} Cal by top_n_removed ===")
        sub = df[df["asset"] == asset_val][["top_n_removed","Cal","CAGR","MDD"]]
        print(sub.to_string(index=False))
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
