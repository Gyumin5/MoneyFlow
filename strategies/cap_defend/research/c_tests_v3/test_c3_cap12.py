#!/usr/bin/env python3
"""Phase C3 — cap 0.12 기준 재검증.

champion (cap 0.12):
- 현물: s_dthr12_tp3 + G1 (A2_bounce_w1) — 기존 cap 0.333 유지 (현물 cap은 합의)
- 선물: f_dthr14_tp10 + G1 (A2_bounce_w1) — cap 0.12

검증:
  (A) ablation 2×2 at cap 0.12
  (B) cap sensitivity 0.08/0.10/0.12/0.15/0.18/0.20 × (baseline vs champion)
  (C) stress (tx, slip) at cap 0.12
  (D) walk-forward 5 fold at cap 0.12
  (E) adverse periods at cap 0.12
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
    FULL_END, TRAIN_END, HOLDOUT_START,
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
    if len(ev) == 0 or slip_bps <= 0:
        return ev
    slip = slip_bps / 10000.0
    ev = ev.copy()
    ev["entry_px"] = ev["entry_px"] * (1 + slip)
    ev["exit_px"] = ev["exit_px"] * (1 - slip)
    ev["pnl_pct"] = (ev["exit_px"] / ev["entry_px"] - 1.0) * 100.0
    return ev


def run_one(kind, ev, v21, cd, hist, cap, tx=0.003, uni=15, lev=3.0,
             start=None, end=None):
    runner = simulate_fut if kind == "fut" else simulate
    rows = []
    if start is not None:
        segs = [("custom", start, end)]
    else:
        segs = [("full", v21.index[0], FULL_END),
                ("train", v21.index[0], TRAIN_END),
                ("holdout", HOLDOUT_START, FULL_END)]
    for span, s, e in segs:
        v21s = slice_v21(v21, s, e)
        if v21s is None: continue
        ev_s = ev[(pd.to_datetime(ev["entry_ts"]) >= v21s.index[0]) &
                   (pd.to_datetime(ev["entry_ts"]) <= v21s.index[-1])] if len(ev) else ev
        kwargs = dict(n_pick=1, cap_per_slot=cap, universe_size=uni,
                      tx_cost=tx, swap_edge_threshold=1)
        if kind == "fut": kwargs["leverage"] = lev
        if len(ev_s) == 0:
            rows.append({"span": span, "Cal":0, "CAGR":0, "MDD":0, "n_entries":0}); continue
        _, st = runner(ev_s, cd, v21s.copy(), hist, **kwargs)
        rows.append({"span": span,
                     "Cal": round(st.get("Cal",0),3),
                     "CAGR": round(st.get("CAGR",0),4),
                     "MDD": round(st.get("MDD",0),4),
                     "n_entries": int(st.get("n_entries",0))})
    return rows


def main():
    v21_spot, v21_fut, hist, avail, cd = load_all()
    all_rows = []

    # FUT configs: baseline + champion
    print("=== extracting fut sets ===")
    t0 = time.time()
    ev_fut_base = extract(avail, dict(dip_bars=24, dip_thr=-0.18, tp=0.08, tstop=48))
    print(f"  fut baseline: {len(ev_fut_base)} events ({time.time()-t0:.0f}s)")
    t0 = time.time()
    ev_fut_champ_raw = extract(avail, dict(dip_bars=24, dip_thr=-0.14, tp=0.10, tstop=48))
    ev_fut_champ = filter_bounce_confirm(ev_fut_champ_raw, 1)
    print(f"  fut champion (dthr14_tp10 raw): {len(ev_fut_champ_raw)}, after G1: {len(ev_fut_champ)} ({time.time()-t0:.0f}s)")

    t0 = time.time()
    ev_spot_base = extract(avail, dict(dip_bars=24, dip_thr=-0.20, tp=0.04, tstop=24))
    print(f"  spot baseline: {len(ev_spot_base)} events ({time.time()-t0:.0f}s)")
    t0 = time.time()
    ev_spot_champ_raw = extract(avail, dict(dip_bars=24, dip_thr=-0.12, tp=0.03, tstop=24))
    ev_spot_champ = filter_bounce_confirm(ev_spot_champ_raw, 1)
    print(f"  spot champion (dthr12_tp3 raw): {len(ev_spot_champ_raw)}, after G1: {len(ev_spot_champ)} ({time.time()-t0:.0f}s)")

    # ============ (A) ablation 2x2 at cap 0.12 for fut ============
    print("\n=== (A) Ablation 2x2 FUT cap 0.12 ===")
    cells = [
        ("old_thr_no_guard", ev_fut_base, None),
        ("old_thr_guard",    ev_fut_base, "G1"),
        ("new_thr_no_guard", ev_fut_champ_raw, None),
        ("new_thr_guard",    ev_fut_champ_raw, "G1"),
    ]
    for lab, ev, gkey in cells:
        ev2 = filter_bounce_confirm(ev, 1) if gkey == "G1" else ev
        sub = run_one("fut", ev2, v21_fut, cd, hist, 0.12)
        for r in sub:
            all_rows.append({"test":"ablation_cap12","label":lab, "kind":"fut", "cap":0.12, **r})

    # ============ (B) cap sensitivity ============
    print("\n=== (B) Cap sensitivity FUT ===")
    for cap in [0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25, 0.30]:
        for name, ev in [("baseline", ev_fut_base), ("champion", ev_fut_champ)]:
            sub = run_one("fut", ev, v21_fut, cd, hist, cap)
            for r in sub:
                all_rows.append({"test":"cap_sens","label":name,"kind":"fut","cap":cap, **r})

    # ============ (C) stress at cap 0.12 ============
    print("\n=== (C) Stress cap 0.12 ===")
    for tx in [0.003, 0.0045, 0.006, 0.009]:
        for slip in [0, 10, 20, 30]:
            for kind, ev, v21, cap in [
                ("fut",  ev_fut_champ,  v21_fut,  0.12),
                ("spot", ev_spot_champ, v21_spot, CAP_SPOT),
            ]:
                ev_s = apply_slippage(ev, slip)
                sub = run_one(kind, ev_s, v21, cd, hist, cap, tx=tx)
                for r in sub:
                    all_rows.append({"test":"stress","label":f"tx{tx}_slip{slip}",
                                      "kind":kind,"cap":cap,"tx":tx,"slip":slip, **r})

    # ============ (D) walk-forward 5 fold ============
    print("\n=== (D) Walk-forward 5 fold ===")
    WF_START = pd.Timestamp("2021-01-01")
    per = (FULL_END - WF_START).days / 5
    for i in range(5):
        s = WF_START + pd.Timedelta(days=int(per * i))
        e = WF_START + pd.Timedelta(days=int(per * (i + 1))) if i < 4 else FULL_END
        for name, ev, v21, kind, cap in [
            ("baseline", ev_fut_base,  v21_fut,  "fut",  0.12),
            ("champion", ev_fut_champ, v21_fut,  "fut",  0.12),
            ("baseline", ev_spot_base, v21_spot, "spot", CAP_SPOT),
            ("champion", ev_spot_champ,v21_spot, "spot", CAP_SPOT),
        ]:
            sub = run_one(kind, ev, v21, cd, hist, cap, start=s, end=e)
            for r in sub:
                all_rows.append({"test":"wf5","label":name,"kind":kind,"cap":cap,
                                  "fold":f"fold{i+1}","period":f"{s.date()}~{e.date()}", **r})

    # ============ (E) adverse periods ============
    print("\n=== (E) Adverse ===")
    periods = [
        ("2022_bear",      pd.Timestamp("2022-01-01"), pd.Timestamp("2022-12-31")),
        ("2022_H1_crash",  pd.Timestamp("2022-01-01"), pd.Timestamp("2022-06-30")),
        ("2022_H2_FTX",    pd.Timestamp("2022-07-01"), pd.Timestamp("2022-12-31")),
        ("2025_q1_drop",   pd.Timestamp("2025-01-01"), pd.Timestamp("2025-03-31")),
    ]
    for pname, s, e in periods:
        for name, ev, v21, kind, cap in [
            ("baseline", ev_fut_base,  v21_fut,  "fut",  0.12),
            ("champion", ev_fut_champ, v21_fut,  "fut",  0.12),
            ("baseline", ev_spot_base, v21_spot, "spot", CAP_SPOT),
            ("champion", ev_spot_champ,v21_spot, "spot", CAP_SPOT),
        ]:
            sub = run_one(kind, ev, v21, cd, hist, cap, start=s, end=e)
            for r in sub:
                all_rows.append({"test":"adverse","label":name,"kind":kind,"cap":cap,
                                  "period":pname, **r})

    df = pd.DataFrame(all_rows)
    path = os.path.join(OUT, "test_c3_cap12.csv")
    df.to_csv(path, index=False)
    print(f"\n저장: {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
