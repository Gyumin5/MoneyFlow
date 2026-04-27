#!/usr/bin/env python3
"""실험 3: V21 + 전략 조합 개선 효과 측정.

관점: standalone 약해도 V21과 상관 낮고 양의 return이면 조합 시 Cal↑

Steps:
1. 각 전략 38 coins EW equity (buy_at=open, tx=0.04%) 이미 계산됨 (Exp2 방식)
2. V21 선물 daily equity vs 전략 daily return
3. corr 계산
4. V21 + w × 전략 portfolio (w = 0.05/0.10/0.20) Cal 비교
5. V21 + C + w × 전략도 (3-way)
"""
from __future__ import annotations
import os, sys, time
import pandas as pd
import numpy as np
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
from m3_engine_final import (load_v21, load_universe_hist, list_available_futures,
                              load_coin_daily, simulate, metrics as m_spot)
from m3_engine_futures import (load_v21_futures, simulate_fut, metrics as m_fut)

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


def strat_daily_ret(eq_map):
    port = aggregate_ew_portfolio(eq_map)
    idx = pd.to_datetime(port.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    port.index = idx
    return port.pct_change().dropna()


def metrics_from_ret(ret, bpy=365):
    if len(ret) < 30:
        return {"CAGR":0, "MDD":0, "Cal":0}
    eq = (1 + ret).cumprod()
    days = (eq.index[-1] - eq.index[0]).days
    yrs = days / 365.25 if days > 0 else 0.001
    cagr = float(eq.iloc[-1] ** (1/yrs) - 1)
    mdd = float((eq / eq.cummax() - 1).min())
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {"CAGR": round(cagr, 4), "MDD": round(mdd, 4), "Cal": round(cal, 3)}


def split_eval(ret_combined, label):
    rows = []
    for span, s, e in [("full", ret_combined.index[0], FULL_END),
                        ("train", ret_combined.index[0], TRAIN_END),
                        ("holdout", HOLDOUT_START, FULL_END)]:
        sub = ret_combined[(ret_combined.index >= s) & (ret_combined.index <= e)]
        m = metrics_from_ret(sub)
        rows.append({"label": label, "span": span, **m, "n_days": len(sub)})
    return rows


def main():
    print("Loading data...")
    avail = sorted(list_available_futures())
    v21_fut = load_v21_futures()

    # V21 선물 daily return (already normalized equity)
    v21_ret = v21_fut["equity"].pct_change().fillna(0)
    v21_idx = pd.to_datetime(v21_ret.index)
    if getattr(v21_idx, "tz", None) is not None:
        v21_idx = v21_idx.tz_localize(None)
    v21_ret.index = v21_idx

    # V21 metrics baseline
    base_rows = split_eval(v21_ret, "V21_alone")

    # C 기여 (기존 cap 0.12)
    # 단순 proxy: V21+C daily return 필요. 여기선 이미 검증된 수치 사용 대신 V21 단독 base만 비교.
    # 대신 C daily return은 events 기반 근사:
    # 시간 절약: V21 ret만 기준으로 각 전략 mixing

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

    all_rows = base_rows.copy()
    corr_rows = []

    for name, fn, params in engines:
        t0 = time.time()
        print(f"\n[{name}] extracting...")
        eq_map = extract_par(avail, fn, params)
        s_ret = strat_daily_ret(eq_map)
        print(f"  {name} daily ret: {len(s_ret)} days ({time.time()-t0:.0f}s)")

        # align
        common = v21_ret.index.intersection(s_ret.index)
        v21r = v21_ret.loc[common]
        sr = s_ret.loc[common]

        corr_full = float(v21r.corr(sr))
        corr_h = float(v21r[v21r.index >= HOLDOUT_START].corr(sr[sr.index >= HOLDOUT_START]))
        corr_rows.append({"engine": name, "corr_full": round(corr_full, 3),
                          "corr_holdout": round(corr_h, 3)})

        # weights
        for w in [0.05, 0.10, 0.20, 0.30]:
            combined = (1 - w) * v21r + w * sr
            all_rows += split_eval(combined, f"V21+{name}_w{w:.2f}")

        print(f"  corr(V21, {name}) full={corr_full:.3f} holdout={corr_h:.3f}")
        for w in [0.10, 0.20]:
            combined = (1 - w) * v21r + w * sr
            m = metrics_from_ret(combined)
            print(f"  V21+{name} w={w}: Cal={m['Cal']:.2f} CAGR={m['CAGR']:.2%} MDD={m['MDD']:.2%}")

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, "exp3_combo_v21.csv"), index=False)
    corr_df = pd.DataFrame(corr_rows)
    corr_df.to_csv(os.path.join(OUT, "exp3_corr.csv"), index=False)

    print("\n=== Cal by config × span ===")
    pv = df.pivot_table(index="label", columns="span", values="Cal", aggfunc="first")
    print(pv.to_string())

    print("\n=== Correlations with V21 ===")
    print(corr_df.to_string(index=False))


if __name__ == "__main__":
    main()
