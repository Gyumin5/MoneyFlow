#!/usr/bin/env python3
"""Group E — 스탑로스 세밀화.

E1. 빠른 fixed stop: -3/-5/-7% (기존 addl_v3는 -10 이상만 봤음, 공백 구간)
E3. Time-conditioned stop: 진입 후 min_hours(4/8/12h) 경과한 뒤에만 stop 활성
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common3 import (
    load_all, load_cached_events, run_splits,
    apply_intrabar_stop, apply_time_conditioned_stop,
    CAP_SPOT, CAP_FUT_OPTS, OUT,
)
import pandas as pd


def main():
    v21_spot, v21_fut, hist, _, cd = load_all()
    ev_spot = load_cached_events("spot")
    ev_fut  = load_cached_events("fut")

    rows = []
    rows += run_splits("baseline", "spot", ev_spot, v21_spot, cd, hist, [CAP_SPOT])
    rows += run_splits("baseline", "fut",  ev_fut,  v21_fut,  cd, hist, CAP_FUT_OPTS)

    # E1 fast stop
    for s in [-0.03, -0.05, -0.07]:
        for kind, ev, v21, caps in [
            ("spot", ev_spot, v21_spot, [CAP_SPOT]),
            ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS),
        ]:
            ev2 = apply_intrabar_stop(ev, s)
            lab = f"E1_stop_{s:+.0%}"
            n_stop = int((ev2.get("reason") == "stop").sum()) if "reason" in ev2 else 0
            print(f"  {lab} {kind}: n_stop={n_stop}")
            rows += run_splits(lab, kind, ev2, v21, cd, hist, caps)

    # E3 time-conditioned stop
    for min_h in [4, 8, 12]:
        for s in [-0.10, -0.15, -0.20]:
            for kind, ev, v21, caps in [
                ("spot", ev_spot, v21_spot, [CAP_SPOT]),
                ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS),
            ]:
                ev2 = apply_time_conditioned_stop(ev, s, min_h)
                lab = f"E3_tstop_{s:+.0%}_min{min_h}h"
                rows += run_splits(lab, kind, ev2, v21, cd, hist, caps)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "test_stop_fine.csv")
    df.to_csv(path, index=False)
    print(f"\n저장: {path} ({len(df)} rows)")
    piv = df[df["span"] == "holdout"].pivot_table(
        index="label", columns=["kind", "cap"], values="Cal")
    print("\n=== Holdout Cal ===")
    print(piv.to_string())


if __name__ == "__main__":
    main()
