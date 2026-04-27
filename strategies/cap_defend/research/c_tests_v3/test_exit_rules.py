#!/usr/bin/env python3
"""Group B — 청산 규칙 (하락 지속 리스크 대응).

B1. Trailing stop: entry 후 최고점 대비 -5/-7/-10%
B2. 모멘텀 exit: 진입 후 window_h 내 양봉 없으면 즉시 청산 (1/2/3h)
B3. tstop 단축: 3/6/12/24/48h (현물 baseline 24h / 선물 48h)
B4. 부분 TP: tp_full * 0.5 에서 절반, 나머지 tp에서 전부 (split 0.5)
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common3 import (
    load_all, load_cached_events, run_splits,
    apply_trailing_stop, apply_momentum_exit, apply_tstop, apply_partial_tp,
    CAP_SPOT, CAP_FUT_OPTS, P_SPOT, P_FUT, OUT,
)
import pandas as pd


def main():
    v21_spot, v21_fut, hist, _, cd = load_all()
    ev_spot = load_cached_events("spot")
    ev_fut  = load_cached_events("fut")

    rows = []
    rows += run_splits("baseline", "spot", ev_spot, v21_spot, cd, hist, [CAP_SPOT])
    rows += run_splits("baseline", "fut",  ev_fut,  v21_fut,  cd, hist, CAP_FUT_OPTS)

    # B1 trailing stop
    for t in [-0.05, -0.07, -0.10]:
        for kind, ev, v21, caps in [
            ("spot", ev_spot, v21_spot, [CAP_SPOT]),
            ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS),
        ]:
            ev2 = apply_trailing_stop(ev, t)
            lab = f"B1_trail_{t:+.0%}"
            n_trail = int((ev2.get("reason") == "trail").sum()) if "reason" in ev2 else 0
            print(f"  {lab} {kind}: n_trail={n_trail}")
            rows += run_splits(lab, kind, ev2, v21, cd, hist, caps)

    # B2 momentum exit (양봉 없으면 청산)
    for w in [1, 2, 3]:
        for kind, ev, v21, caps in [
            ("spot", ev_spot, v21_spot, [CAP_SPOT]),
            ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS),
        ]:
            ev2 = apply_momentum_exit(ev, w)
            lab = f"B2_momexit_w{w}"
            n_cut = int((ev2.get("reason") == "momentum_exit").sum()) if "reason" in ev2 else 0
            print(f"  {lab} {kind}: n_cut={n_cut}")
            rows += run_splits(lab, kind, ev2, v21, cd, hist, caps)

    # B3 tstop 단축
    for tstop in [3, 6, 12, 24, 48]:
        for kind, ev, v21, caps, orig in [
            ("spot", ev_spot, v21_spot, [CAP_SPOT], P_SPOT["tstop"]),
            ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS, P_FUT["tstop"]),
        ]:
            if tstop >= orig:
                # 단축이 아니라 확장 — baseline과 사실상 동일하므로 스킵
                continue
            ev2 = apply_tstop(ev, tstop)
            lab = f"B3_tstop_{tstop}h"
            n_cut = int((ev2.get("reason") == "tstop_cut").sum()) if "reason" in ev2 else 0
            print(f"  {lab} {kind}: n_cut={n_cut}")
            rows += run_splits(lab, kind, ev2, v21, cd, hist, caps)

    # B4 부분 TP
    for split in [0.3, 0.5, 0.7]:
        for kind, ev, v21, caps, tp_full in [
            ("spot", ev_spot, v21_spot, [CAP_SPOT], P_SPOT["tp"]),
            ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS, P_FUT["tp"]),
        ]:
            ev2 = apply_partial_tp(ev, tp_full, split=split)
            lab = f"B4_partialtp_s{split:.1f}"
            rows += run_splits(lab, kind, ev2, v21, cd, hist, caps)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "test_exit_rules.csv")
    df.to_csv(path, index=False)
    print(f"\n저장: {path} ({len(df)} rows)")
    piv = df[df["span"] == "holdout"].pivot_table(
        index="label", columns=["kind", "cap"], values="Cal")
    print("\n=== Holdout Cal ===")
    print(piv.to_string())


if __name__ == "__main__":
    main()
