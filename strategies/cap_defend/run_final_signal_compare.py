#!/usr/bin/env python3
"""기존 실거래 조합 vs 최종 후보 조합 직접 비교.

실행층 고정:
- per_coin_leverage_mode = cap_mom_blend_543_cash
- stop = prev_close15 + cash_guard(34%)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data, run
from run_ensemble import SingleAccountEngine, combine_targets
from run_stoploss_test import START, END


STRATEGIES = {
    'old_4h1': dict(
        interval='4h', sma_bars=240, mom_short_bars=10, mom_long_bars=30,
        canary_hyst=0.015, drift_threshold=0.0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode='mom1vol', vol_mode='daily',
        vol_threshold=0.05, n_snapshots=3, snap_interval_bars=120,
    ),
    'old_4h2': dict(
        interval='4h', sma_bars=120, mom_short_bars=20, mom_long_bars=120,
        canary_hyst=0.015, drift_threshold=0.0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
        vol_threshold=0.80, n_snapshots=3, snap_interval_bars=18,
    ),
    'old_1h1': dict(
        interval='1h', sma_bars=200, mom_short_bars=200, mom_long_bars=1200,
        canary_hyst=0.015, drift_threshold=0.0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
        vol_threshold=0.80, n_snapshots=3, snap_interval_bars=336,
    ),
    'new_1h_09': dict(
        interval='1h', sma_bars=168, mom_short_bars=36, mom_long_bars=720,
        canary_hyst=0.015, drift_threshold=0.0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
        vol_threshold=0.80, n_snapshots=3, snap_interval_bars=27,
    ),
    'new_4h_01': dict(
        interval='4h', sma_bars=240, mom_short_bars=10, mom_long_bars=30,
        canary_hyst=0.015, drift_threshold=0.0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode='mom1vol', vol_mode='daily',
        vol_threshold=0.05, n_snapshots=3, snap_interval_bars=120,
    ),
    'new_4h_09': dict(
        interval='4h', sma_bars=120, mom_short_bars=20, mom_long_bars=120,
        canary_hyst=0.015, drift_threshold=0.0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
        vol_threshold=0.60, n_snapshots=3, snap_interval_bars=21,
    ),
}

COMBOS = [
    ('current_live', {'old_4h1': 1 / 3, 'old_4h2': 1 / 3, 'old_1h1': 1 / 3}),
    ('final_candidate', {'new_1h_09': 1 / 3, 'new_4h_01': 1 / 3, 'new_4h_09': 1 / 3}),
]


def generate_trace(data, cfg):
    run_cfg = dict(cfg)
    interval = run_cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(
        bars, funding,
        interval=interval,
        leverage=1.0,
        start_date=START,
        end_date=END,
        _trace=trace,
        **run_cfg,
    )
    return trace


def main():
    t0 = time.time()
    print('Loading data...')
    data = {iv: load_data(iv) for iv in ['4h', '1h']}
    bars_1h, funding_1h = data['1h']
    all_dates = bars_1h['BTC'].index[(bars_1h['BTC'].index >= START) & (bars_1h['BTC'].index <= END)]

    print('Generating traces...')
    traces = {name: generate_trace(data, cfg) for name, cfg in STRATEGIES.items()}

    print('\nResults')
    for name, weights in COMBOS:
        selected = {k: traces[k] for k in weights}
        combined = combine_targets(selected, weights, all_dates)
        engine = SingleAccountEngine(
            bars_1h,
            funding_1h,
            leverage=5.0,
            stop_kind='prev_close_pct',
            stop_pct=0.15,
            stop_gate='cash_guard',
            stop_gate_cash_threshold=0.34,
            per_coin_leverage_mode='cap_mom_blend_543_cash',
            leverage_floor=3.0,
            leverage_mid=4.0,
            leverage_ceiling=5.0,
            leverage_cash_threshold=0.34,
            leverage_partial_cash_threshold=0.0,
            leverage_count_floor_max=2,
            leverage_count_mid_max=4,
            leverage_canary_floor_gap=0.015,
            leverage_canary_mid_gap=0.04,
            leverage_canary_high_gap=0.08,
            leverage_canary_sma_bars=1200,
            leverage_mom_lookback_bars=24 * 30,
            leverage_vol_lookback_bars=24 * 90,
        )
        m = engine.run(combined)
        print(
            f"{name:<16} "
            f"Cal={m['Cal']:.2f} "
            f"CAGR={m['CAGR']:+.1%} "
            f"MDD={m['MDD']:+.1%} "
            f"Liq={m['Liq']} "
            f"Stops={m.get('Stops', 0)} "
            f"Rebal={m.get('Rebal', 0)}"
        )
    print(f'\nElapsed: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
