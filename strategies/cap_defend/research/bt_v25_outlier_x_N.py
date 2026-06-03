"""V25 outlier-drop × universe_size 2D 그리드 BT.

drop 단계 × N. BTC 항상 universe 유지.
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


def run_spot(exclude, N):
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
        exclude_assets=frozenset(exclude) if exclude else None,
    )
    return m.get('_equity') if m else None


def run_fut(exclude, N):
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    bars_full, funding = load_data('D')
    k2 = build_K2_signal(bars_full, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    if exclude:
        bars = {c: df for c, df in bars_full.items() if c not in exclude}
    else:
        bars = bars_full
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
    if eq is None or len(eq.dropna()) < 30: return None
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
    drop_levels = [
        ('drop0',        []),
        ('drop1 -BNB',   ['BNB']),
        ('drop2 -BNB,SOL', ['BNB', 'SOL']),
        ('drop3 +XRP',     ['BNB', 'SOL', 'XRP']),
        ('drop4 +DOGE',    ['BNB', 'SOL', 'XRP', 'DOGE']),
    ]
    Ns = [3, 4, 5, 6]

    for asset, runner in [('SPOT', run_spot), ('FUT', run_fut)]:
        print(f"\n========== {asset} ==========")
        # 표 헤더
        print(f"  {'drop':<22} " + ' '.join(f"{'N='+str(n):>12}" for n in Ns))
        for tag, excl in drop_levels:
            row_strs = []
            for N in Ns:
                eq = runner(excl, N)
                m = metrics(eq)
                if m is None:
                    row_strs.append(f"{'FAIL':>12}")
                else:
                    row_strs.append(f"{m['cagr']:>5.0f}%/{m['cal']:>4.2f}")
            print(f"  {tag:<22} " + ' '.join(f"{s:>12}" for s in row_strs))
        print(f"  (셀 표기: CAGR% / Calmar)")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
