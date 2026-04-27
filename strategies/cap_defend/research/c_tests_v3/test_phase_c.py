#!/usr/bin/env python3
"""Phase C — 상위 가드 결합 × 신호 파라미터 재최적화.

Phase A/B에서 유의미했던 가드:
- A2_bounce_w1: 진입 후 1h 내 양봉 확인해야 진입 (entry 지연)
- B2_momexit_w2: 진입 후 2h 내 양봉 없으면 청산
- F3_uni10: universe Top10 (m3 엔진 universe_size=10)

방식:
1. 신호 파라미터 grid로 events 재추출 (c_engine_v5) — joblib 24
2. 각 events 위에 가드 조합 5가지 적용:
   G0 = no guard (baseline)
   G1 = A2 only
   G2 = B2 only
   G3 = A2 + B2
   G4 = F3 (universe 10)
3. 각 조합 × 3 split × cap.
"""
from __future__ import annotations
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..", "next_strategies")))

from _common3 import (
    load_all, CAP_SPOT, CAP_FUT_OPTS, OUT,
    slice_v21, TRAIN_END, HOLDOUT_START, FULL_END,
    filter_bounce_confirm, apply_momentum_exit,
)
from c_engine_v5 import run_c_v5
from m3_engine_final import simulate
from m3_engine_futures import simulate_fut
from joblib import Parallel, delayed
import pandas as pd

N_JOBS = 24

# 신호 파라미터 grid (focused around Phase B 유의미 구간)
# Phase B: spot dthr-12 best Holdout 2.98; fut dthr-14/25 best.
SPOT_GRID = [
    # (dip_bars, dip_thr, tp, tstop, tag)
    (24, -0.20, 0.04, 24, "s_base"),         # baseline
    (24, -0.12, 0.04, 24, "s_dthr12"),
    (24, -0.15, 0.04, 24, "s_dthr15"),
    (24, -0.22, 0.04, 24, "s_dthr22"),
    (12, -0.12, 0.04, 24, "s_dbars12_dthr12"),
    (12, -0.15, 0.04, 24, "s_dbars12_dthr15"),
    (24, -0.12, 0.03, 24, "s_dthr12_tp3"),
    (24, -0.12, 0.05, 24, "s_dthr12_tp5"),
    (24, -0.12, 0.04, 36, "s_dthr12_tstop36"),
    (24, -0.15, 0.05, 24, "s_dthr15_tp5"),
]

FUT_GRID = [
    (24, -0.18, 0.08, 48, "f_base"),        # baseline
    (24, -0.14, 0.08, 48, "f_dthr14"),
    (24, -0.22, 0.08, 48, "f_dthr22"),
    (24, -0.25, 0.08, 48, "f_dthr25"),
    (24, -0.14, 0.08, 36, "f_dthr14_tstop36"),
    (24, -0.14, 0.10, 48, "f_dthr14_tp10"),
    (24, -0.14, 0.06, 48, "f_dthr14_tp6"),
    (36, -0.18, 0.08, 48, "f_dbars36"),
    (24, -0.25, 0.10, 48, "f_dthr25_tp10"),
    (24, -0.22, 0.10, 48, "f_dthr22_tp10"),
]


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


def extract_parallel(avail, params) -> pd.DataFrame:
    res = Parallel(n_jobs=N_JOBS, prefer="threads")(
        delayed(_extract_one)(c, params) for c in sorted(avail))
    return pd.DataFrame([e for b in res for e in b])


def apply_guard(ev: pd.DataFrame, guard: str) -> pd.DataFrame:
    if guard == "G0":
        return ev
    if guard == "G1":
        return filter_bounce_confirm(ev, 1)
    if guard == "G2":
        return apply_momentum_exit(ev, 2)
    if guard == "G3":
        ev2 = filter_bounce_confirm(ev, 1)
        return apply_momentum_exit(ev2, 2)
    if guard == "G4":
        return ev  # universe_size는 sim에서 처리
    raise ValueError(guard)


GUARDS = [
    ("G0", "baseline"),
    ("G1", "A2_bounce_w1"),
    ("G2", "B2_momexit_w2"),
    ("G3", "A2+B2"),
    ("G4", "F3_uni10"),
]


def run_one(kind, ev, v21, cd, hist, cap, universe_size=15, lev=3.0):
    runner = simulate_fut if kind == "fut" else simulate
    out = []
    for span, start, end in [
        ("full",    v21.index[0], FULL_END),
        ("train",   v21.index[0], TRAIN_END),
        ("holdout", HOLDOUT_START, FULL_END),
    ]:
        v21s = slice_v21(v21, start, end)
        if v21s is None: continue
        ev_s = ev[(pd.to_datetime(ev["entry_ts"]) >= v21s.index[0]) &
                   (pd.to_datetime(ev["entry_ts"]) <= v21s.index[-1])] if len(ev) else ev
        kwargs = dict(n_pick=1, cap_per_slot=cap, universe_size=universe_size,
                      tx_cost=0.003, swap_edge_threshold=1)
        if kind == "fut":
            kwargs["leverage"] = lev
        if len(ev_s) == 0:
            out.append({"span": span, "cap": cap, "uni": universe_size,
                        "Cal": 0, "CAGR": 0, "MDD": 0, "n_entries": 0})
            continue
        _, st = runner(ev_s, cd, v21s.copy(), hist, **kwargs)
        out.append({"span": span, "cap": cap, "uni": universe_size,
                    "Cal": round(st.get("Cal", 0), 3),
                    "CAGR": round(st.get("CAGR", 0), 4),
                    "MDD": round(st.get("MDD", 0), 4),
                    "Sharpe": round(st.get("Sharpe", 0), 3),
                    "n_entries": int(st.get("n_entries", 0))})
    return out


def main():
    v21_spot, v21_fut, hist, avail, cd = load_all()
    rows = []

    # SPOT
    print("=== SPOT grid ===")
    for (db, dt, tp, ts_, tag) in SPOT_GRID:
        t0 = time.time()
        ev = extract_parallel(avail, dict(dip_bars=db, dip_thr=dt, tp=tp, tstop=ts_))
        print(f"  extract {tag}: {len(ev)} events ({time.time()-t0:.0f}s)")
        for (gkey, gname) in GUARDS:
            ev_g = apply_guard(ev, gkey) if gkey != "G4" else ev
            uni = 10 if gkey == "G4" else 15
            sub = run_one("spot", ev_g, v21_spot, cd, hist, CAP_SPOT,
                          universe_size=uni)
            for r in sub:
                rows.append({"signal": tag, "guard": gkey, "guard_name": gname,
                             "kind": "spot", **r})

    # FUT
    print("=== FUT grid ===")
    for (db, dt, tp, ts_, tag) in FUT_GRID:
        t0 = time.time()
        ev = extract_parallel(avail, dict(dip_bars=db, dip_thr=dt, tp=tp, tstop=ts_))
        print(f"  extract {tag}: {len(ev)} events ({time.time()-t0:.0f}s)")
        for (gkey, gname) in GUARDS:
            ev_g = apply_guard(ev, gkey) if gkey != "G4" else ev
            uni = 10 if gkey == "G4" else 15
            for cap in CAP_FUT_OPTS:
                sub = run_one("fut", ev_g, v21_fut, cd, hist, cap,
                              universe_size=uni, lev=3.0)
                for r in sub:
                    rows.append({"signal": tag, "guard": gkey,
                                 "guard_name": gname, "kind": "fut", **r})

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "test_phase_c.csv")
    df.to_csv(path, index=False)
    print(f"\n저장: {path} ({len(df)} rows)")

    # 요약
    hd = df[df["span"] == "holdout"].copy()
    hd["score_3"] = hd.groupby(["kind"])["Cal"].transform(lambda s: s)
    print("\n=== Holdout top 15 ===")
    print(hd.sort_values("Cal", ascending=False)[
        ["signal","guard","guard_name","kind","cap","Cal","CAGR","MDD","n_entries"]
    ].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
