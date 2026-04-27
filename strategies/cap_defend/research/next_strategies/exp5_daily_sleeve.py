#!/usr/bin/env python3
"""실험 5: Daily timeframe 전략 2개를 V21 sleeve로 편입.

신규:
- weekly_rotation (7d momentum)
- monthly_reversal (30d low bounce)
둘 다 BTC 단독 양수 (Weekly +160%, Monthly +18%)

V21 + 전략 구조 (C와 동일, cap 0.05/0.10):
- V21 우선, 남는 cash에서 전략 작동 (m3 엔진)
"""
from __future__ import annotations
import os, sys, time
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c_engine_v5 import load_coin
from engine_weekly_rotation import run_weekly_rotation
from engine_monthly_reversal import run_monthly_reversal
from m3_engine_futures import (load_v21_futures, simulate_fut, metrics,
                                load_universe_hist, list_available_futures,
                                load_coin_daily)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

TRAIN_END = pd.Timestamp("2023-12-31")
HOLDOUT_START = pd.Timestamp("2024-01-01")
FULL_END = pd.Timestamp("2026-03-30")


def extract_par(avail, fn, params, n_jobs=24):
    def _one(c):
        df = load_coin(c + "USDT")
        if df is None: return []
        _, evs = fn(df, buy_at="open", tx=0.0004, **params)
        for e in evs:
            e["coin"] = c
        return evs
    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_one)(c) for c in avail)
    return pd.DataFrame([e for b in results for e in b])


def slice_v21(v21_raw, start, end):
    sub = v21_raw[(v21_raw.index >= start) & (v21_raw.index <= end)].copy()
    if len(sub) < 30: return None
    sub["equity"] = sub["equity"].astype(float) / float(sub["equity"].iloc[0])
    sub["v21_ret"] = sub["equity"].pct_change().fillna(0)
    sub["prev_cash"] = sub["cash_ratio"].shift(1).fillna(sub["cash_ratio"].iloc[0])
    return sub


def run_splits(ev, v21_raw, cd, hist, cap, label):
    rows = []
    for span, s, e in [("full", v21_raw.index[0], FULL_END),
                        ("train", v21_raw.index[0], TRAIN_END),
                        ("holdout", HOLDOUT_START, FULL_END)]:
        v21s = slice_v21(v21_raw, s, e)
        if v21s is None: continue
        ev_s = ev[(ev["entry_ts"] >= v21s.index[0]) & (ev["entry_ts"] <= v21s.index[-1])].copy() if len(ev) else ev
        if len(ev_s) == 0:
            rows.append({"label": label, "span": span, "cap": cap,
                         "Cal": 0, "CAGR": 0, "MDD": 0, "n_entries": 0})
            continue
        _, st = simulate_fut(ev_s, cd, v21s.copy(), hist,
                             n_pick=1, cap_per_slot=cap, universe_size=15,
                             tx_cost=0.0004, swap_edge_threshold=1, leverage=3.0)
        rows.append({"label": label, "span": span, "cap": cap,
                     "Cal": round(st.get("Cal",0),3),
                     "CAGR": round(st.get("CAGR",0),4),
                     "MDD": round(st.get("MDD",0),4),
                     "n_entries": st.get("n_entries",0)})
    return rows


def main():
    print("Loading...")
    avail = sorted(list_available_futures())
    v21_fut = load_v21_futures()
    hist = load_universe_hist()
    cd = load_coin_daily(avail)

    rows = []
    # V21 baseline
    for span, s, e in [("full", v21_fut.index[0], FULL_END),
                        ("train", v21_fut.index[0], TRAIN_END),
                        ("holdout", HOLDOUT_START, FULL_END)]:
        v21s = slice_v21(v21_fut, s, e)
        m = metrics(v21s["equity"])
        rows.append({"label": "V21_alone", "span": span, "cap": 0,
                     "Cal": round(m["Cal"],3), "CAGR": round(m["CAGR"],4),
                     "MDD": round(m["MDD"],4), "n_entries": 0})

    engines = [
        ("weekly_rot",  run_weekly_rotation, {"mom_days":7, "mom_thr":0.10, "exit_thr":0.02,
                                               "sma_days":30, "hold_days":5, "stop_loss":0.06}),
        ("monthly_rev", run_monthly_reversal, {"low_days":30, "near_low_pct":0.03,
                                                "sma_days":90, "tp":0.08, "tstop":15, "stop_loss":0.06}),
    ]
    for name, fn, params in engines:
        t0 = time.time()
        print(f"\n[{name}] extracting...")
        ev = extract_par(avail, fn, params)
        print(f"  {len(ev)} events ({time.time()-t0:.0f}s)")
        for cap in [0.05, 0.10, 0.15, 0.20]:
            rows += run_splits(ev, v21_fut, cd, hist, cap, f"V21+{name}_cap{cap}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp5_daily_sleeve.csv"), index=False)

    v21_base = {r["span"]: r["Cal"] for r in rows if r["label"] == "V21_alone"}
    print("\n=== vs V21 단독 ===")
    print(f"V21 단독: full {v21_base['full']}, train {v21_base['train']}, holdout {v21_base['holdout']}")
    print()
    for lab in sorted(set(r["label"] for r in rows if r["label"] != "V21_alone")):
        subset = [r for r in rows if r["label"] == lab]
        full = next(r["Cal"] for r in subset if r["span"] == "full")
        train = next(r["Cal"] for r in subset if r["span"] == "train")
        hout = next(r["Cal"] for r in subset if r["span"] == "holdout")
        d_f = full - v21_base["full"]
        d_t = train - v21_base["train"]
        d_h = hout - v21_base["holdout"]
        flag = "✓" if (d_f > 0 and d_h > 0) else ("△" if (d_f > 0 or d_h > 0) else "✗")
        print(f"  {flag} {lab}: full {full} (Δ{d_f:+.2f}) train {train} (Δ{d_t:+.2f}) holdout {hout} (Δ{d_h:+.2f})")

    print(f"\n저장: {OUT}/exp5_daily_sleeve.csv")


if __name__ == "__main__":
    main()
