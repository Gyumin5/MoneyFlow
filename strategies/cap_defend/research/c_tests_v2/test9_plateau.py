#!/usr/bin/env python3
"""Test 9: Parameter plateau — 최적값 인접 grid 성능 확인.

확정 파라미터 주변 ±1 step으로 sweep. plateau가 평탄할수록 robust.
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (P_SPOT, P_FUT, CAP_SPOT, HOLDOUT_START, FULL_END,
                     load_all, extract_events, slice_v21,
                     run_spot_combo, run_fut_combo)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def main():
    v21_s, v21_f, hist, avail, cd = load_all()
    v21_sH = slice_v21(v21_s, HOLDOUT_START, FULL_END)
    v21_fH = slice_v21(v21_f, HOLDOUT_START, FULL_END)

    # SPOT plateau: ±변형
    spot_grid = []
    for db in [18, 24, 30]:
        for dt in [-0.18, -0.20, -0.22]:
            for tp in [0.03, 0.04, 0.05, 0.06]:
                for ts in [18, 24, 30, 36]:
                    spot_grid.append({"dip_bars":db,"dip_thr":dt,"tp":tp,"tstop":ts})

    fut_grid = []
    for db in [18, 24, 30]:
        for dt in [-0.15, -0.18, -0.20]:
            for tp in [0.06, 0.08, 0.10]:
                for ts in [36, 48, 60, 72]:
                    fut_grid.append({"dip_bars":db,"dip_thr":dt,"tp":tp,"tstop":ts})

    print(f"SPOT grid: {len(spot_grid)}, FUT grid: {len(fut_grid)}")

    rows = []
    for P in spot_grid:
        ev = extract_events(avail, P)
        ev_sub = ev[(ev["entry_ts"] >= HOLDOUT_START) & (ev["entry_ts"] <= FULL_END)].copy()
        _, st = run_spot_combo(ev_sub, cd, v21_sH, hist, CAP_SPOT)
        rows.append({"asset":"spot", **P, "Cal_h":round(st["Cal"],3),
                     "CAGR_h":round(st["CAGR"],4), "MDD_h":round(st["MDD"],4),
                     "n_entries":st["n_entries"]})

    for P in fut_grid:
        ev = extract_events(avail, P)
        ev_sub = ev[(ev["entry_ts"] >= HOLDOUT_START) & (ev["entry_ts"] <= FULL_END)].copy()
        _, st = run_fut_combo(ev_sub, cd, v21_fH, hist, 0.25)
        rows.append({"asset":"fut_cap0.25", **P, "Cal_h":round(st["Cal"],3),
                     "CAGR_h":round(st["CAGR"],4), "MDD_h":round(st["MDD"],4),
                     "n_entries":st["n_entries"]})

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test9_plateau.csv"), index=False)

    # 확정 파라미터 찾아서 주변 이웃과 비교
    conf_spot = {"dip_bars":24,"dip_thr":-0.20,"tp":0.04,"tstop":24}
    conf_fut = {"dip_bars":24,"dip_thr":-0.18,"tp":0.08,"tstop":48}

    print("\n=== SPOT confirmed vs neighbors (Cal_h) ===")
    sub = df[df["asset"] == "spot"]
    print(sub.sort_values("Cal_h", ascending=False).head(10).to_string(index=False))
    print(f"\n  Confirmed ({conf_spot}):")
    c = sub[(sub["dip_bars"] == conf_spot["dip_bars"]) & (sub["dip_thr"] == conf_spot["dip_thr"])
             & (sub["tp"] == conf_spot["tp"]) & (sub["tstop"] == conf_spot["tstop"])]
    print(f"  rank = {sub['Cal_h'].rank(ascending=False)[c.index].values if len(c) else '?'}/ {len(sub)}")
    print(f"  Cal_h = {c['Cal_h'].values if len(c) else '?'}")

    print("\n=== FUT cap0.25 confirmed vs neighbors (Cal_h) ===")
    sub = df[df["asset"] == "fut_cap0.25"]
    print(sub.sort_values("Cal_h", ascending=False).head(10).to_string(index=False))
    c = sub[(sub["dip_bars"] == conf_fut["dip_bars"]) & (sub["dip_thr"] == conf_fut["dip_thr"])
             & (sub["tp"] == conf_fut["tp"]) & (sub["tstop"] == conf_fut["tstop"])]
    print(f"\n  Confirmed rank = {sub['Cal_h'].rank(ascending=False)[c.index].values if len(c) else '?'}/ {len(sub)}")

    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
