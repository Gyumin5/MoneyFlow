#!/usr/bin/env python3
"""실험 2: 생존 5 전략 × 38 coins EW aggregate 재검증.

수정 사항 (Exp1에서 확인):
- buy_at="open" (실전 t+1 open 체결 가정)
- tx=0.0004 (실전 Binance fut 수수료)
- 기본 설정만 각 1개 (BTC sanity에서 양수 gross edge 확인된 best 파라미터)
"""
from __future__ import annotations
import os, sys, time
import pandas as pd
from joblib import Parallel, delayed

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common_next import aggregate_ew_portfolio, TRAIN_END, HOLDOUT_START, FULL_END
from c_engine_v5 import load_coin
from engine_pullback import run_pullback
from engine_vbo import run_vbo
from engine_weekly_low import run_weekly_low
from engine_macd_cross import run_macd_cross
from engine_momentum_rotation import run_momentum_rotation
from m3_engine_final import (load_v21, load_universe_hist,
                              list_available_futures, load_coin_daily, metrics as m_spot)
from m3_engine_futures import metrics as m_fut

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def extract_par(avail, engine_fn, params, n_jobs=24):
    def _one(c):
        df = load_coin(c + "USDT")
        if df is None: return None
        eq, _ = engine_fn(df, buy_at="open", tx=0.0004, **params)
        return c, eq
    results = Parallel(n_jobs=n_jobs, prefer="threads")(delayed(_one)(c) for c in avail)
    return {c: eq for c, eq in results if eq is not None}


def split_metrics(eq):
    idx = pd.to_datetime(eq.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    eq.index = idx
    rows = {}
    for span, s, e in [("full", eq.index[0], FULL_END),
                        ("train", eq.index[0], TRAIN_END),
                        ("holdout", HOLDOUT_START, FULL_END)]:
        sub = eq[(eq.index >= s) & (eq.index <= e)]
        if len(sub) < 30 or sub.iloc[0] <= 0:
            rows[span] = {"Cal": 0, "CAGR": 0, "MDD": 0}
            continue
        sub_n = sub / sub.iloc[0]
        rows[span] = m_spot(sub_n)
    return rows


def main():
    print("Loading universe...")
    avail = sorted(list_available_futures())
    print(f"Coins: {len(avail)}")

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

    rows = []
    for name, fn, params in engines:
        t0 = time.time()
        print(f"\n[{name}] extracting...")
        eq_map = extract_par(avail, fn, params)
        print(f"  {len(eq_map)} coins OK ({time.time()-t0:.1f}s)")
        port = aggregate_ew_portfolio(eq_map)
        m = split_metrics(port)
        for span, metrics in m.items():
            rows.append({
                "engine": name, "span": span,
                "Cal": round(metrics.get("Cal",0), 3),
                "CAGR": round(metrics.get("CAGR",0), 4),
                "MDD": round(metrics.get("MDD",0), 4),
                "Sh": round(metrics.get("Sh",0), 2),
            })
        print(f"  {name} full Cal={m['full'].get('Cal',0):.2f} CAGR={m['full'].get('CAGR',0):.2%} MDD={m['full'].get('MDD',0):.2%}")
        print(f"  {name} train Cal={m['train'].get('Cal',0):.2f}")
        print(f"  {name} hout  Cal={m['holdout'].get('Cal',0):.2f} CAGR={m['holdout'].get('CAGR',0):.2%}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "exp2_multicoin_fixed.csv"), index=False)

    print("\n=== Cal by engine × span ===")
    print(df.pivot_table(index="engine", columns="span", values="Cal", aggfunc="first").to_string())
    print(f"\n저장: {OUT}/exp2_multicoin_fixed.csv")


if __name__ == "__main__":
    main()
