"""V25 아웃라이어 제외 BT — spot/fut 각각 top contributor 제외 시 성과 변화.

baseline / -BNB / -BNB-BTC / -BNB-BTC-SOL / -top4 / -top5 / -all_top_alts(BNB+SOL+XRP+DOGE)
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

START = "2020-10-01"
END = "2026-05-29"


def run_spot(exclude):
    from unified_backtest import run as bt_run, load_data
    bars, _ = load_data('D')
    m = bt_run(
        bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=20, mom_long_bars=127,
        vol_threshold=0.05, vol_mode='daily',
        n_snapshots=7, snap_interval_bars=217,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol',
        exclude_assets=frozenset(exclude) if exclude else None,
    )
    return m.get('_equity') if m else None


def run_fut(exclude):
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    bars_full, funding = load_data('D')
    # K2 는 full bars (BTC 포함) 로 계산 — BTC_cap 신호용 BTC 필요
    k2 = build_K2_signal(bars_full, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    # 단 BTC 자체는 universe 에 남겨야 K2 가 정상 작동 (BTC 가 universe 에 없으면 K2 신호는 있어도 picks 안 됨)
    # 따라서 BTC 는 항상 universe 유지. BTC 제외 시는 별도 처리 필요 (현재는 안 함)
    if exclude:
        bars = {c: df for c, df in bars_full.items() if c not in exclude or c == 'BTC' and 'BTC' in exclude}
        # 다만 BTC 가 exclude 에 있으면 universe 에서 제외 (K2 신호는 살아있음)
        bars = {c: df for c, df in bars_full.items() if c not in exclude}
    else:
        bars = bars_full
    m = fbt_run(
        bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
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


def main():
    t0 = time.time()
    cfgs = [
        ('baseline', []),
        ('-BNB', ['BNB']),
        ('-BNB-SOL', ['BNB', 'SOL']),
        ('-BNB-SOL-XRP', ['BNB', 'SOL', 'XRP']),
        ('-BNB-SOL-XRP-DOGE', ['BNB', 'SOL', 'XRP', 'DOGE']),
        ('-BNB-SOL-XRP-AAVE-LINK', ['BNB', 'SOL', 'XRP', 'AAVE', 'LINK']),
        ('-BNB-SOL-XRP-DOGE-AAVE-LINK', ['BNB', 'SOL', 'XRP', 'DOGE', 'AAVE', 'LINK']),
    ]
    print("=== SPOT ===")
    print(f"  {'cfg':<32} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'Cal':>6}")
    for tag, excl in cfgs:
        eq = run_spot(excl)
        m = metrics(eq)
        if m is None: print(f"  {tag:<32} FAIL"); continue
        print(f"  {tag:<32} {m['cagr']:>7.1f}% {m['mdd']:>+7.1f}% {m['sharpe']:>7.2f} {m['cal']:>6.2f}")

    print("\n=== FUT ===")
    print(f"  {'cfg':<32} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'Cal':>6}")
    for tag, excl in cfgs:
        eq = run_fut(excl)
        m = metrics(eq)
        if m is None: print(f"  {tag:<32} FAIL"); continue
        print(f"  {tag:<32} {m['cagr']:>7.1f}% {m['mdd']:>+7.1f}% {m['sharpe']:>7.2f} {m['cal']:>6.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
