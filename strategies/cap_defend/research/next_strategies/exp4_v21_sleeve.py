#!/usr/bin/env python3
"""실험 4: V21 남은 cash에서 추가 전략 작동 (C와 동일 구조).

아키텍처:
- V21 fut daily equity + cash_ratio (기존 baseline)
- 각 신규 전략의 events → m3_engine_futures.simulate_fut에 투입
- cap_per_slot = X% (V21 남은 cash 한도 내에서)
- 결과: V21 단독 vs V21 + 전략, V21 + C + 전략 비교

V21 파이 침해 없음 (기존 C와 동일 원리).
"""
from __future__ import annotations
import os, sys, time
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from c_engine_v5 import load_coin
from engine_pullback import run_pullback
from engine_vbo import run_vbo
from engine_weekly_low import run_weekly_low
from engine_macd_cross import run_macd_cross
from engine_momentum_rotation import run_momentum_rotation
from m3_engine_futures import (load_v21_futures, simulate_fut, metrics,
                                load_universe_hist, list_available_futures,
                                load_coin_daily)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)

TRAIN_END = pd.Timestamp("2023-12-31")
HOLDOUT_START = pd.Timestamp("2024-01-01")
FULL_END = pd.Timestamp("2026-03-30")


def extract_events_par(avail, engine_fn, params, n_jobs=24):
    """events 리스트 추출. buy_at=open, tx=0.0004 실전 가정."""
    def _one(c):
        df = load_coin(c + "USDT")
        if df is None: return []
        _, evs = engine_fn(df, buy_at="open", tx=0.0004, **params)
        for e in evs:
            e["coin"] = c
        return evs
    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_one)(c) for c in avail)
    rows = [e for batch in results for e in batch]
    return pd.DataFrame(rows)


def slice_v21(v21_raw, start, end):
    sub = v21_raw[(v21_raw.index >= start) & (v21_raw.index <= end)].copy()
    if len(sub) < 30: return None
    sub["equity"] = sub["equity"].astype(float) / float(sub["equity"].iloc[0])
    sub["v21_ret"] = sub["equity"].pct_change().fillna(0)
    sub["prev_cash"] = sub["cash_ratio"].shift(1).fillna(sub["cash_ratio"].iloc[0])
    return sub


def eval_combo(events, coin_daily, v21_slice, hist, cap):
    if events is None or len(events) == 0:
        return {"Cal":0, "CAGR":0, "MDD":0, "n_entries":0, "n_liq":0}
    ev = events[(events["entry_ts"] >= v21_slice.index[0])
                 & (events["entry_ts"] <= v21_slice.index[-1])].copy()
    _, st = simulate_fut(ev, coin_daily, v21_slice.copy(), hist,
                          n_pick=1, cap_per_slot=cap, universe_size=15,
                          tx_cost=0.0004, swap_edge_threshold=1, leverage=3.0)
    return st


def run_splits(ev, v21_raw, cd, hist, cap, label):
    rows = []
    for span, s, e in [("full", v21_raw.index[0], FULL_END),
                        ("train", v21_raw.index[0], TRAIN_END),
                        ("holdout", HOLDOUT_START, FULL_END)]:
        v21s = slice_v21(v21_raw, s, e)
        if v21s is None: continue
        st = eval_combo(ev, cd, v21s, hist, cap)
        rows.append({"label": label, "span": span, "cap": cap,
                     "Cal": round(st.get("Cal",0),3),
                     "CAGR": round(st.get("CAGR",0),4),
                     "MDD": round(st.get("MDD",0),4),
                     "n_entries": st.get("n_entries",0),
                     "n_liq": st.get("n_liquidations",0)})
    return rows


def main():
    print("Loading data...")
    avail = sorted(list_available_futures())
    v21_fut = load_v21_futures()
    hist = load_universe_hist()
    cd = load_coin_daily(avail)
    print(f"Coins: {len(avail)}")

    # V21 단독 baseline
    rows = []
    for span, s, e in [("full", v21_fut.index[0], FULL_END),
                        ("train", v21_fut.index[0], TRAIN_END),
                        ("holdout", HOLDOUT_START, FULL_END)]:
        v21s = slice_v21(v21_fut, s, e)
        if v21s is None: continue
        m = metrics(v21s["equity"])
        rows.append({"label": "V21_alone", "span": span, "cap": 0,
                     "Cal": round(m["Cal"],3),
                     "CAGR": round(m["CAGR"],4),
                     "MDD": round(m["MDD"],4),
                     "n_entries": 0, "n_liq": 0})

    engines = [
        ("weekly_low", run_weekly_low, {"week_hours":168, "near_low_pct":0.02,
                                         "sma_regime":720, "tp":0.05, "tstop":96, "stop_loss":0.04}),
        ("mom_rot",    run_momentum_rotation, {"mom_lookback":168, "mom_thr":0.05,
                                                "tp":0.15, "tstop":168, "stop_loss":0.05}),
        ("pullback",   run_pullback, {"ema_fast":50, "ema_slow":200,
                                       "pullback_min":0.015, "pullback_max":0.08,
                                       "tp":0.06, "tstop":72, "trail_drop":0.02}),
        ("vbo",        run_vbo, {"donch_window":48, "atr_window":14, "sma_regime":168,
                                  "trail_atr_mult":2.0, "tp":0.08, "tstop":96,
                                  "vol_filter_max":0.035}),
        ("macd",       run_macd_cross, {"fast":12, "slow":26, "signal":9, "sma_regime":240,
                                         "tp":0.06, "tstop":72, "stop_loss":0.04}),
    ]

    for name, fn, params in engines:
        t0 = time.time()
        print(f"\n[{name}] extracting events...")
        ev = extract_events_par(avail, fn, params)
        print(f"  {len(ev)} events ({time.time()-t0:.0f}s)")
        if len(ev) == 0:
            continue
        # cap sensitivity
        for cap in [0.05, 0.10, 0.20]:
            rows += run_splits(ev, v21_fut, cd, hist, cap, f"V21+{name}_cap{cap}")
        print(f"  {name} cap=0.10 full/train/holdout Cal:")
        sub = [r for r in rows if r["label"] == f"V21+{name}_cap0.1"]
        for r in sub:
            print(f"    {r['span']}: Cal={r['Cal']:.2f} CAGR={r['CAGR']:.2%} MDD={r['MDD']:.2%}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp4_v21_sleeve.csv"), index=False)

    print("\n=== Cal by config × span ===")
    pv = df.pivot_table(index="label", columns="span", values="Cal", aggfunc="first")
    print(pv.to_string())

    # 개선 여부 요약
    v21_base = {r["span"]: r["Cal"] for r in rows if r["label"] == "V21_alone"}
    print("\n=== vs V21 단독 개선 (holdout) ===")
    for lab in sorted(set(r["label"] for r in rows if r["label"] != "V21_alone")):
        h = next((r["Cal"] for r in rows if r["label"] == lab and r["span"] == "holdout"), None)
        f = next((r["Cal"] for r in rows if r["label"] == lab and r["span"] == "full"), None)
        if h is not None and f is not None:
            d_h = h - v21_base.get("holdout", 0)
            d_f = f - v21_base.get("full", 0)
            flag = "✓" if d_h > 0 and d_f > 0 else "✗"
            print(f"  {flag} {lab}: Δhout={d_h:+.2f}, Δfull={d_f:+.2f}")


if __name__ == "__main__":
    main()
