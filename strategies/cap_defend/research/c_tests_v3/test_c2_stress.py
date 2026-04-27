#!/usr/bin/env python3
"""Phase C2 — 거래비용/슬리피지 stress test.

champion 조합에 대해:
- TX cost: 0.003 (baseline) / 0.0045 (1.5배) / 0.006 (2배) / 0.009 (3배)
- Slippage: entry_px *= (1 + slip_bps), exit_px *= (1 - slip_bps)
  slip_bps ∈ {0, 10, 20, 30} bp (=0.1% ~ 0.3%)
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
    TRAIN_END, HOLDOUT_START, FULL_END,
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


def apply_slippage(ev: pd.DataFrame, slip_bps: float) -> pd.DataFrame:
    """entry_px *= (1 + slip), exit_px *= (1 - slip). pnl_pct 재계산."""
    if len(ev) == 0 or slip_bps <= 0:
        return ev
    slip = slip_bps / 10000.0
    ev = ev.copy()
    ev["entry_px"] = ev["entry_px"] * (1 + slip)
    ev["exit_px"] = ev["exit_px"] * (1 - slip)
    ev["pnl_pct"] = (ev["exit_px"] / ev["entry_px"] - 1.0) * 100.0
    return ev


def run_one(kind, ev, v21, cd, hist, cap, tx, uni=15, lev=3.0):
    runner = simulate_fut if kind == "fut" else simulate
    rows = []
    for span, start, end in [
        ("full",    v21.index[0], FULL_END),
        ("train",   v21.index[0], TRAIN_END),
        ("holdout", HOLDOUT_START, FULL_END),
    ]:
        v21s = slice_v21(v21, start, end)
        if v21s is None: continue
        ev_s = ev[(pd.to_datetime(ev["entry_ts"]) >= v21s.index[0]) &
                   (pd.to_datetime(ev["entry_ts"]) <= v21s.index[-1])] if len(ev) else ev
        kwargs = dict(n_pick=1, cap_per_slot=cap, universe_size=uni,
                      tx_cost=tx, swap_edge_threshold=1)
        if kind == "fut": kwargs["leverage"] = lev
        if len(ev_s) == 0:
            rows.append({"span": span, "Cal":0, "CAGR":0, "MDD":0, "n_entries":0})
            continue
        _, st = runner(ev_s, cd, v21s.copy(), hist, **kwargs)
        rows.append({"span": span,
                     "Cal": round(st.get("Cal",0),3),
                     "CAGR": round(st.get("CAGR",0),4),
                     "MDD": round(st.get("MDD",0),4),
                     "n_entries": int(st.get("n_entries",0))})
    return rows


def main():
    v21_spot, v21_fut, hist, avail, cd = load_all()
    rows = []

    # SPOT champion: s_dthr12_tp3 + G1
    t0 = time.time()
    ev_spot = extract(avail, dict(dip_bars=24, dip_thr=-0.12, tp=0.03, tstop=24))
    ev_spot = filter_bounce_confirm(ev_spot, 1)
    print(f"spot champion events: {len(ev_spot)} ({time.time()-t0:.0f}s)")

    # FUT champion: f_dthr14 + G3
    t0 = time.time()
    ev_fut = extract(avail, dict(dip_bars=24, dip_thr=-0.14, tp=0.08, tstop=48))
    ev_fut = filter_bounce_confirm(ev_fut, 1)
    ev_fut = apply_momentum_exit(ev_fut, 2)
    print(f"fut champion events: {len(ev_fut)} ({time.time()-t0:.0f}s)")

    for tx in [0.003, 0.0045, 0.006, 0.009]:
        for slip in [0, 10, 20, 30]:
            for kind, ev, v21, cap, uni in [
                ("spot", ev_spot, v21_spot, CAP_SPOT, 15),
                ("fut",  ev_fut,  v21_fut,  0.30, 15),
            ]:
                ev_s = apply_slippage(ev, slip)
                sub = run_one(kind, ev_s, v21, cd, hist, cap, tx, uni)
                for r in sub:
                    rows.append({"kind": kind, "tx": tx, "slip_bps": slip, **r})

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "test_c2_stress.csv")
    df.to_csv(path, index=False)
    print(f"\n저장: {path}")

    # Holdout 요약
    hd = df[df["span"]=="holdout"]
    print("\n=== SPOT Holdout under stress (Cal) ===")
    piv = hd[hd["kind"]=="spot"].pivot_table(index="tx", columns="slip_bps", values="Cal")
    print(piv.to_string())
    print("\n=== FUT Holdout under stress (Cal) ===")
    piv = hd[hd["kind"]=="fut"].pivot_table(index="tx", columns="slip_bps", values="Cal")
    print(piv.to_string())
    print("\n=== FUT Holdout MDD ===")
    piv = hd[hd["kind"]=="fut"].pivot_table(index="tx", columns="slip_bps", values="MDD")
    print(piv.to_string())


if __name__ == "__main__":
    main()
