#!/usr/bin/env python3
"""Group A — 진입 필터 (falling knife 회피).

A1. 연속 dip 필터 — 시그널 봉 다음 봉 수익률 ≤ threshold 이면 진입 스킵
A2. 반등 확인 — 시그널 봉 이후 window_h 시간 내 양봉 없으면 진입 스킵

각 변형 × (현물 cap 0.333, 선물 cap [0.12, 0.25, 0.30]) × 3 split.
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common3 import (
    load_all, load_cached_events, run_splits,
    filter_next_bar_dip, filter_bounce_confirm,
    CAP_SPOT, CAP_FUT_OPTS, OUT,
)
import pandas as pd


def main():
    v21_spot, v21_fut, hist, _, cd = load_all()
    ev_spot = load_cached_events("spot")
    ev_fut  = load_cached_events("fut")
    print(f"base events: spot={len(ev_spot)} fut={len(ev_fut)}")

    rows = []
    # Baseline
    rows += run_splits("baseline", "spot", ev_spot, v21_spot, cd, hist, [CAP_SPOT])
    rows += run_splits("baseline", "fut",  ev_fut,  v21_fut,  cd, hist, CAP_FUT_OPTS)

    # A1 next-bar dip filter
    for drop in [-0.01, -0.02, -0.03]:
        for kind, ev, v21, caps in [
            ("spot", ev_spot, v21_spot, [CAP_SPOT]),
            ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS),
        ]:
            ev2 = filter_next_bar_dip(ev, drop)
            label = f"A1_nextdip_{drop:+.0%}"
            print(f"  {label} {kind}: kept {len(ev2)}/{len(ev)}")
            rows += run_splits(label, kind, ev2, v21, cd, hist, caps)

    # A2 bounce confirm (양봉 window)
    for w in [1, 2, 3]:
        for kind, ev, v21, caps in [
            ("spot", ev_spot, v21_spot, [CAP_SPOT]),
            ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS),
        ]:
            ev2 = filter_bounce_confirm(ev, w)
            label = f"A2_bounce_w{w}"
            print(f"  {label} {kind}: kept {len(ev2)}/{len(ev)}")
            rows += run_splits(label, kind, ev2, v21, cd, hist, caps)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "test_entry_filters.csv")
    df.to_csv(path, index=False)
    print(f"\n저장: {path} ({len(df)} rows)")

    # 요약: holdout Cal
    piv = df[df["span"] == "holdout"].pivot_table(
        index="label", columns=["kind", "cap"], values="Cal")
    print("\n=== Holdout Cal (pivot) ===")
    print(piv.to_string())


if __name__ == "__main__":
    main()
