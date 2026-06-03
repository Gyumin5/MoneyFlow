"""V25 cap 분산 BT — cap=1/N + universe_size=N (N=3..7).

baseline N=3 (현행) vs N=4/5/6/7. spot + fut. BTC 항상 universe 유지.
"""
from __future__ import annotations
import os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

START = "2020-10-01"
END = "2026-05-29"


def run_spot(N):
    from unified_backtest import run as bt_run, load_data
    bars, _ = load_data('D')
    m = bt_run(
        bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=20, mom_long_bars=127,
        vol_threshold=0.05, vol_mode='daily',
        n_snapshots=7, snap_interval_bars=217,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=N, cap=1/N, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol',
    )
    return m.get('_equity') if m else None


def run_fut(N):
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    bars, funding = load_data('D')
    k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    m = fbt_run(
        bars, funding, interval='D', leverage=k2, universe_size=N, cap=1/N,
        tx_cost=0.0006, maint_rate=0.004,
        sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
        canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=5, snap_interval_bars=95,
        start_date=START, end_date=END,
    )
    return m.get('_equity') if m else None


def metrics(eq):
    if eq is None or len(eq.dropna()) < 30:
        return None
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1]/eq.iloc[0]) ** (1/yrs) - 1
    peak = eq.cummax(); mdd = (eq/peak - 1).min()
    rets = eq.pct_change().dropna()
    sh = rets.mean()/rets.std()*np.sqrt(252) if rets.std() > 0 else 0
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return dict(cagr=cagr*100, mdd=mdd*100, sharpe=sh, cal=cal)


def window_rs(eq_dict):
    WIN = [504, 756, 1008]; STR = [63, 126, 252]
    common = None
    for s in eq_dict.values():
        if s is None: continue
        if common is None: common = s.index
        else: common = common.intersection(s.index)
    if common is None: return None
    common = sorted(common)
    if len(common) < max(WIN) + max(STR): return None
    sums = defaultdict(float); wins = defaultdict(int); n = 0
    for size in WIN:
        for stride in STR:
            for i in range(0, len(common) - size, stride):
                d0 = common[i]; d1 = common[i + size - 1]
                cals = {}
                for k, s in eq_dict.items():
                    if s is None: cals[k] = np.nan; continue
                    seg = s.loc[d0:d1].dropna()
                    if len(seg) < 30: cals[k] = np.nan; continue
                    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                    if yrs <= 0: cals[k] = np.nan; continue
                    cagr = (seg.iloc[-1]/seg.iloc[0]) ** (1/yrs) - 1
                    peak = seg.cummax(); mdd = float((seg/peak - 1).min())
                    cals[k] = cagr/abs(mdd) if mdd < 0 else 0
                if any(np.isnan(v) for v in cals.values()): continue
                ranked = sorted(cals.items(), key=lambda x: -x[1])
                for r, (mk, _) in enumerate(ranked, 1): sums[mk] += r
                wins[ranked[0][0]] += 1; n += 1
    return sums, wins, n


def main():
    t0 = time.time()
    Ns = [3, 4, 5, 6, 7]
    eq_sp = {}; eq_fu = {}
    print("=== SPOT ===")
    print(f"  {'cfg':<10} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'Cal':>6}")
    for N in Ns:
        eq = run_spot(N); eq_sp[f'N={N}'] = eq
        m = metrics(eq)
        if m is None: print(f"  N={N}        FAIL"); continue
        tag = f"N={N}{' (현행)' if N==3 else ''}"
        print(f"  {tag:<10} {m['cagr']:>7.1f}% {m['mdd']:>+7.1f}% {m['sharpe']:>7.2f} {m['cal']:>6.2f}")
    print("\nwindow rank-sum (spot):")
    rs = window_rs(eq_sp)
    if rs:
        sums, wins, n = rs
        for k, v in sorted(sums.items(), key=lambda x: x[1]):
            print(f"  {k:<10} avg_rank={v/n:.3f} win={wins[k]/n*100:.1f}%")

    print("\n=== FUT ===")
    print(f"  {'cfg':<10} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'Cal':>6}")
    for N in Ns:
        eq = run_fut(N); eq_fu[f'N={N}'] = eq
        m = metrics(eq)
        if m is None: print(f"  N={N}        FAIL"); continue
        tag = f"N={N}{' (현행)' if N==3 else ''}"
        print(f"  {tag:<10} {m['cagr']:>7.1f}% {m['mdd']:>+7.1f}% {m['sharpe']:>7.2f} {m['cal']:>6.2f}")
    print("\nwindow rank-sum (fut):")
    rs = window_rs(eq_fu)
    if rs:
        sums, wins, n = rs
        for k, v in sorted(sums.items(), key=lambda x: x[1]):
            print(f"  {k:<10} avg_rank={v/n:.3f} win={wins[k]/n*100:.1f}%")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
