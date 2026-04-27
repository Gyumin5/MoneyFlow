#!/usr/bin/env python3
"""Phase C2 — dip_thr × 가드 2×2 ablation.

champion 조합에서 어느 축이 얼마를 개선했는지 분리:
- 현물 champion: s_dthr12_tp3 (dthr=-0.12, tp=0.03) + G1 (A2_bounce_w1)
- 선물 champion: f_dthr14 (dthr=-0.14) + G3 (A2+B2)

2×2 cells:
- (old_thr, no_guard): 기존 baseline
- (old_thr, guard): 가드만 단독
- (new_thr, no_guard): dthr만 단독
- (new_thr, guard): champion (결합)
"""
from __future__ import annotations
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "next_strategies")))

from _common3 import (
    load_all, run_splits, CAP_SPOT, CAP_FUT_OPTS, OUT,
    filter_bounce_confirm, apply_momentum_exit,
)
from joblib import Parallel, delayed
import pandas as pd

N_JOBS = 24


def _extract_one(coin: str, params: dict) -> list[dict]:
    import sys, os
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__),
                                                     "..", "next_strategies")))
    from c_engine_v5 import load_coin, run_c_v5
    df = load_coin(coin + "USDT")
    if df is None:
        return []
    _, evs = run_c_v5(df, **params)
    for e in evs:
        e["coin"] = coin
    return evs


def extract(avail, params):
    res = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_extract_one)(c, params) for c in sorted(avail))
    return pd.DataFrame([e for b in res for e in b])


def main():
    v21_spot, v21_fut, hist, avail, cd = load_all()
    rows = []

    # SPOT 2x2
    spot_cells = [
        # (dthr, tp, guard_fn, guard_name, label)
        (-0.20, 0.04, None, "no_guard",     "SPOT_old_thr_no_guard"),
        (-0.20, 0.04, "G1",  "A2_bounce_w1", "SPOT_old_thr_guard"),
        (-0.12, 0.03, None, "no_guard",     "SPOT_new_thr_no_guard"),
        (-0.12, 0.03, "G1",  "A2_bounce_w1", "SPOT_new_thr_guard"),
    ]
    for (dthr, tp, gkey, gname, lab) in spot_cells:
        t0 = time.time()
        ev = extract(avail, dict(dip_bars=24, dip_thr=dthr, tp=tp, tstop=24))
        if gkey == "G1":
            ev = filter_bounce_confirm(ev, 1)
        print(f"  {lab}: {len(ev)} events ({time.time()-t0:.0f}s)")
        sub = run_splits(lab, "spot", ev, v21_spot, cd, hist, [CAP_SPOT])
        rows += [{"cell": lab, "axis": f"dthr={dthr}, tp={tp}, guard={gname}", **r}
                 for r in sub]

    # FUT 2x2
    fut_cells = [
        (-0.18, 0.08, None, "no_guard", "FUT_old_thr_no_guard"),
        (-0.18, 0.08, "G3",  "A2+B2",   "FUT_old_thr_guard"),
        (-0.14, 0.08, None, "no_guard", "FUT_new_thr_no_guard"),
        (-0.14, 0.08, "G3",  "A2+B2",   "FUT_new_thr_guard"),
    ]
    for (dthr, tp, gkey, gname, lab) in fut_cells:
        t0 = time.time()
        ev = extract(avail, dict(dip_bars=24, dip_thr=dthr, tp=tp, tstop=48))
        if gkey == "G3":
            ev = filter_bounce_confirm(ev, 1)
            ev = apply_momentum_exit(ev, 2)
        print(f"  {lab}: {len(ev)} events ({time.time()-t0:.0f}s)")
        sub = run_splits(lab, "fut", ev, v21_fut, cd, hist, [0.30])
        rows += [{"cell": lab, "axis": f"dthr={dthr}, tp={tp}, guard={gname}", **r}
                 for r in sub]

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "test_c2_ablation.csv")
    df.to_csv(path, index=False)
    print(f"\n저장: {path}")

    # 요약: Holdout Cal 비교
    hd = df[df["span"] == "holdout"]
    print("\n=== Holdout Cal + MDD 2x2 ===")
    print(hd[["cell","axis","cap","Cal","CAGR","MDD","n_entries"]].to_string(index=False))


if __name__ == "__main__":
    main()
