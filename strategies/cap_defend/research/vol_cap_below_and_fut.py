"""vol_cap < 0.05 및 현물 vs 선물 최적 vol_cap 비교.

(1) 현물(spot) vol_threshold 를 0.03~0.06 로 세밀 스윕 — 0.05보다 타이트하게가 나은지.
(2) 선물(fut, V25 동적 L) vol_threshold 를 0.03~0.10 스윕 — 최적이 현물과 다른지.

read-only. 단일 전체기간 Cal/MDD (방향성 지표; 채택은 별도 window rank-sum 필요).
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import numpy as np
import pandas as pd
import unified_backtest as ub

SPOT_KW = dict(
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, canary_hyst=0.015, n_snapshots=7,
    universe_size=3, cap=1/3, tx_cost=0.004,
    health_mode='mom2vol', vol_mode='daily', drift_threshold=0.10,
    snap_interval_bars=217,
)
START, END = '2020-10-01', '2026-05-13'


def main():
    t0 = time.time()
    bars, funding = ub.load_data('D')

    print("===== (1) 현물 SPOT vol_cap 세밀 스윕 (0.05 위·아래) =====")
    print(f"  {'vthr':>6s} {'Cal':>7s} {'CAGR':>8s} {'MDD':>8s} {'Trades':>7s} {'코인수':>6s}")
    for vthr in [0.03, 0.035, 0.04, 0.045, 0.05, 0.055, 0.06]:
        tr = []
        kw = dict(SPOT_KW); kw.update(start_date=START, end_date=END, vol_threshold=vthr, _trace=tr)
        m = ub.run(bars, funding, **kw)
        ncoin = len(set(k for t in tr for k in (t.get('target') or {}) if str(k).upper() != 'CASH'))
        star = " *현행" if abs(vthr - 0.05) < 1e-9 else ""
        print(f"  {vthr:>6.3f} {m['Cal']:>7.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Trades']:>7d} {ncoin:>6d}{star}")

    print("\n===== (2) 선물 FUT (V25 동적 L) vol_cap 스윕 =====")
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  {'vthr':>6s} {'Cal':>7s} {'CAGR':>8s} {'MDD':>8s} {'Trades':>7s}")
    for vthr in [0.03, 0.04, 0.05, 0.06, 0.07, 0.10]:
        m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
                    tx_cost=0.0006, maint_rate=0.004,
                    sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
                    canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
                    health_mode='mom2vol', vol_mode='daily', vol_threshold=vthr,
                    n_snapshots=5, snap_interval_bars=95,
                    start_date='2020-10-01', end_date=END)
        if not m:
            print(f"  {vthr:>6.3f}  NO DATA"); continue
        star = " *현행" if abs(vthr - 0.05) < 1e-9 else ""
        print(f"  {vthr:>6.3f} {m['Cal']:>7.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Rebal']:>7d}{star}")

    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
