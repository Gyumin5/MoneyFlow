#!/usr/bin/env python3
"""Phase C2 — Walk-forward 5-fold 재검증.

champion 조합의 안정성을 timing 민감도로 확인:
- 2021-01-01 ~ 2026-03-30 구간을 5 fold로 나눠 각 구간 성과 평가
- fold별 Cal/CAGR/MDD 분포 확인 (baseline도 동시)
"""
from __future__ import annotations
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "next_strategies")))

from _common3 import (
    load_all, slice_v21, CAP_SPOT, OUT,
    filter_bounce_confirm, apply_momentum_exit,
    FULL_END,
)
from m3_engine_final import simulate
from m3_engine_futures import simulate_fut
from joblib import Parallel, delayed
import pandas as pd

N_JOBS = 24
WF_START = pd.Timestamp("2021-01-01")
FOLDS = 5


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


def fold_bounds():
    total_days = (FULL_END - WF_START).days
    per = total_days / FOLDS
    out = []
    for i in range(FOLDS):
        s = WF_START + pd.Timedelta(days=int(per * i))
        e = WF_START + pd.Timedelta(days=int(per * (i + 1))) if i < FOLDS - 1 else FULL_END
        out.append((f"fold{i+1}", s, e))
    return out


def run_fold(kind, label, ev, v21, cd, hist, cap, start, end, uni=15, lev=3.0):
    runner = simulate_fut if kind == "fut" else simulate
    v21s = slice_v21(v21, start, end)
    if v21s is None or len(v21s) < 30:
        return {"kind": kind, "label": label, "start": str(start.date()), "end": str(end.date()),
                "cap": cap, "Cal": None, "CAGR": None, "MDD": None, "n_entries": 0}
    ev_s = ev[(pd.to_datetime(ev["entry_ts"]) >= v21s.index[0]) &
               (pd.to_datetime(ev["entry_ts"]) <= v21s.index[-1])] if len(ev) else ev
    kwargs = dict(n_pick=1, cap_per_slot=cap, universe_size=uni,
                  tx_cost=0.003, swap_edge_threshold=1)
    if kind == "fut": kwargs["leverage"] = lev
    if len(ev_s) == 0:
        return {"kind": kind, "label": label, "start": str(start.date()), "end": str(end.date()),
                "cap": cap, "Cal": 0, "CAGR": 0, "MDD": 0, "n_entries": 0}
    _, st = runner(ev_s, cd, v21s.copy(), hist, **kwargs)
    return {"kind": kind, "label": label, "start": str(start.date()), "end": str(end.date()),
            "cap": cap,
            "Cal": round(st.get("Cal", 0), 3),
            "CAGR": round(st.get("CAGR", 0), 4),
            "MDD": round(st.get("MDD", 0), 4),
            "n_entries": int(st.get("n_entries", 0))}


def main():
    v21_spot, v21_fut, hist, avail, cd = load_all()

    # baseline + champion 조합 각각
    configs_spot = [
        ("baseline_old",  dict(dip_bars=24, dip_thr=-0.20, tp=0.04, tstop=24), "G0"),
        ("champion",      dict(dip_bars=24, dip_thr=-0.12, tp=0.03, tstop=24), "G1"),
    ]
    configs_fut = [
        ("baseline_old", dict(dip_bars=24, dip_thr=-0.18, tp=0.08, tstop=48), "G0"),
        ("champion",     dict(dip_bars=24, dip_thr=-0.14, tp=0.08, tstop=48), "G3"),
    ]

    rows = []
    folds = fold_bounds()

    for (name, params, gkey) in configs_spot:
        t0 = time.time()
        ev = extract(avail, params)
        if gkey == "G1": ev = filter_bounce_confirm(ev, 1)
        print(f"SPOT {name}: {len(ev)} events ({time.time()-t0:.0f}s)")
        for flab, s, e in folds:
            r = run_fold("spot", f"{name}", ev, v21_spot, cd, hist, CAP_SPOT, s, e)
            r["fold"] = flab
            rows.append(r)

    for (name, params, gkey) in configs_fut:
        t0 = time.time()
        ev = extract(avail, params)
        if gkey == "G3":
            ev = filter_bounce_confirm(ev, 1)
            ev = apply_momentum_exit(ev, 2)
        print(f"FUT {name}: {len(ev)} events ({time.time()-t0:.0f}s)")
        for flab, s, e in folds:
            r = run_fold("fut", f"{name}", ev, v21_fut, cd, hist, 0.30, s, e)
            r["fold"] = flab
            rows.append(r)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "test_c2_walkforward.csv")
    df.to_csv(path, index=False)
    print(f"\n저장: {path}")

    # 요약
    piv = df.pivot_table(index=["kind","label"], columns="fold", values="Cal")
    print("\n=== Cal per fold ===")
    print(piv.to_string())
    piv_mdd = df.pivot_table(index=["kind","label"], columns="fold", values="MDD")
    print("\n=== MDD per fold ===")
    print(piv_mdd.to_string())

    print("\n=== Summary (mean, std, min) ===")
    agg = df.groupby(["kind","label"]).agg(
        Cal_mean=("Cal","mean"), Cal_std=("Cal","std"), Cal_min=("Cal","min"),
        MDD_mean=("MDD","mean"), MDD_worst=("MDD","min"),
    )
    print(agg.to_string())


if __name__ == "__main__":
    main()
