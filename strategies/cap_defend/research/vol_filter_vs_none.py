"""vol 필터(0.05) vs 필터 없음 비교 — 현물·선물 각각.
필터 없음 = vol_threshold=1.0 (일간 vol>100% 코인 없으므로 사실상 무필터).
read-only.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
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

    print("===== 현물 SPOT: vol 0.05 vs 무필터 =====")
    for label, vthr in [('vol 0.05 (현행)', 0.05), ('무필터(vol=1.0)', 1.0)]:
        kw = dict(SPOT_KW); kw.update(start_date=START, end_date=END, vol_threshold=vthr)
        m = ub.run(bars, funding, **kw)
        print(f"  {label:>16s}: Cal={m['Cal']:.2f} CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%} Trades={m['Trades']}")

    print("\n===== 선물 FUT (V25 동적 L): vol 0.05 vs 무필터 =====")
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    for label, vthr in [('vol 0.05 (현행)', 0.05), ('무필터(vol=1.0)', 1.0)]:
        m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
                    tx_cost=0.0006, maint_rate=0.004,
                    sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
                    canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
                    health_mode='mom2vol', vol_mode='daily', vol_threshold=vthr,
                    n_snapshots=5, snap_interval_bars=95,
                    start_date='2020-10-01', end_date=END)
        print(f"  {label:>16s}: Cal={m['Cal']:.2f} CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%} Rebal={m['Rebal']}")

    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
