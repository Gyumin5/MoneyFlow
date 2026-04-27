#!/usr/bin/env python3
"""Phase C2 — 2022 하락장 adverse excursion 단독 평가.

champion 조합이 하락 지속 구간에서도 방어하는지 확인.
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
)
from m3_engine_final import simulate
from m3_engine_futures import simulate_fut
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


def run_adverse(kind, label, ev, v21, cd, hist, cap, start, end, uni=15, lev=3.0):
    runner = simulate_fut if kind == "fut" else simulate
    v21s = slice_v21(v21, start, end)
    if v21s is None:
        return None
    ev_s = ev[(pd.to_datetime(ev["entry_ts"]) >= v21s.index[0]) &
               (pd.to_datetime(ev["entry_ts"]) <= v21s.index[-1])] if len(ev) else ev
    kwargs = dict(n_pick=1, cap_per_slot=cap, universe_size=uni,
                  tx_cost=0.003, swap_edge_threshold=1)
    if kind == "fut": kwargs["leverage"] = lev
    if len(ev_s) == 0:
        return {"kind": kind, "label": label, "period": f"{start.date()}~{end.date()}",
                "cap": cap, "Cal": 0, "CAGR": 0, "MDD": 0, "n_entries": 0}
    _, st = runner(ev_s, cd, v21s.copy(), hist, **kwargs)
    return {"kind": kind, "label": label, "period": f"{start.date()}~{end.date()}",
            "cap": cap,
            "Cal": round(st.get("Cal",0),3),
            "CAGR": round(st.get("CAGR",0),4),
            "MDD": round(st.get("MDD",0),4),
            "n_entries": int(st.get("n_entries",0))}


def main():
    v21_spot, v21_fut, hist, avail, cd = load_all()

    # 주요 하락 구간들
    periods = [
        ("2022_bear", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
        ("2022_H1_crash", pd.Timestamp("2022-01-01"), pd.Timestamp("2022-06-30")),
        ("2022_H2_FTX", pd.Timestamp("2022-07-01"), pd.Timestamp("2022-12-31")),
        ("2025_q1_drop", pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31")),
    ]

    rows = []
    for name, params, gkey, asset in [
        ("spot_baseline", dict(dip_bars=24, dip_thr=-0.20, tp=0.04, tstop=24), "G0", "spot"),
        ("spot_champion", dict(dip_bars=24, dip_thr=-0.12, tp=0.03, tstop=24), "G1", "spot"),
        ("fut_baseline",  dict(dip_bars=24, dip_thr=-0.18, tp=0.08, tstop=48), "G0", "fut"),
        ("fut_champion",  dict(dip_bars=24, dip_thr=-0.14, tp=0.08, tstop=48), "G3", "fut"),
    ]:
        t0 = time.time()
        ev = extract(avail, params)
        if gkey == "G1":
            ev = filter_bounce_confirm(ev, 1)
        elif gkey == "G3":
            ev = filter_bounce_confirm(ev, 1)
            ev = apply_momentum_exit(ev, 2)
        print(f"{name}: {len(ev)} events ({time.time()-t0:.0f}s)")
        for (pname, s, e) in periods:
            v21 = v21_spot if asset == "spot" else v21_fut
            cap = CAP_SPOT if asset == "spot" else 0.30
            r = run_adverse(asset, name, ev, v21, cd, hist, cap, s, e)
            if r is None: continue
            r["period_label"] = pname
            rows.append(r)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "test_c2_adverse.csv")
    df.to_csv(path, index=False)
    print(f"\n저장: {path}")

    piv_cal = df.pivot_table(index=["kind","label"], columns="period_label", values="Cal")
    piv_mdd = df.pivot_table(index=["kind","label"], columns="period_label", values="MDD")
    print("\n=== Cal per adverse period ===")
    print(piv_cal.to_string())
    print("\n=== MDD per adverse period ===")
    print(piv_mdd.to_string())


if __name__ == "__main__":
    main()
