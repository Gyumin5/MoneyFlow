#!/usr/bin/env python3
"""신규 앙상블 후보(D+D+4h+2h)를 d005와 동일 파이프라인(SingleAccountEngine)으로 비교.

비교 케이스:
  1) fixed_3x      + stop=none           (raw signal, 3x, 가드 OFF)
  2) fixed_5x      + stop=none           (raw signal, 5x, 가드 OFF)
  3) capmom_543    + stop=none           (동적 레버리지, 가드 부분)
  4) capmom_543    + prev_close15+cash_guard34  (production 가드 ON)

사용법: python3 compare_new_ensemble_guards.py
"""
import csv
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from backtest_futures_full import load_data, run
from run_ensemble import SingleAccountEngine, combine_targets

START = '2020-10-01'
END = '2026-03-28'

# 신규 추천 앙상블 (futures Top1)
NEW_STRATEGIES = {
    'D_M20': dict(
        interval='D',
        sma_bars=50, mom_short_bars=20, mom_long_bars=90,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=120,
    ),
    'D_M80': dict(
        interval='D',
        sma_bars=50, mom_short_bars=80, mom_long_bars=240,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=120,
    ),
    '4h_S240': dict(
        interval='4h',
        sma_bars=240, mom_short_bars=30, mom_long_bars=720,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=120,
    ),
    '2h_S480': dict(
        interval='2h',
        sma_bars=480, mom_short_bars=10, mom_long_bars=120,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=120,
    ),
}
WEIGHTS = {k: 1/4 for k in NEW_STRATEGIES}

CASES = [
    dict(name='fixed_3x_no_guard',  leverage=3.0, leverage_mode='fixed',
         per_coin_leverage_mode='none', stop_kind='none'),
    dict(name='fixed_5x_no_guard',  leverage=5.0, leverage_mode='fixed',
         per_coin_leverage_mode='none', stop_kind='none'),
    dict(name='capmom_543_no_stop', leverage=4.0, leverage_mode='fixed',
         per_coin_leverage_mode='cap_mom_blend_543_cash', stop_kind='none'),
    dict(name='capmom_543_full',    leverage=4.0, leverage_mode='fixed',
         per_coin_leverage_mode='cap_mom_blend_543_cash',
         stop_kind='prev_close_pct', stop_pct=0.15,
         stop_gate='cash_guard', stop_gate_cash_threshold=0.34),
]


def generate_trace(data, cfg):
    cfg = dict(cfg)
    interval = cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=START, end_date=END, _trace=trace, **cfg)
    return trace


def run_case(bars_1h, funding_1h, combined, case):
    eng = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=case.get('leverage', 4.0),
        stop_kind=case.get('stop_kind', 'none'),
        stop_pct=case.get('stop_pct', 0.0),
        stop_gate=case.get('stop_gate', 'always'),
        stop_gate_cash_threshold=case.get('stop_gate_cash_threshold', 0.0),
        leverage_mode=case.get('leverage_mode', 'fixed'),
        per_coin_leverage_mode=case.get('per_coin_leverage_mode', 'none'),
        leverage_floor=3.0, leverage_mid=4.0, leverage_ceiling=5.0,
        leverage_cash_threshold=0.34,
        leverage_partial_cash_threshold=0.0,
        leverage_count_floor_max=2, leverage_count_mid_max=4,
        leverage_canary_floor_gap=0.015,
        leverage_canary_mid_gap=0.04,
        leverage_canary_high_gap=0.08,
        leverage_canary_sma_bars=1200,
        leverage_mom_lookback_bars=24*30,
        leverage_vol_lookback_bars=24*90,
    )
    return eng.run(combined)


def main():
    t0 = time.time()
    print('Loading data: D, 4h, 2h, 1h ...')
    data = {iv: load_data(iv) for iv in ['D', '4h', '2h', '1h']}

    traces = {}
    for k, cfg in NEW_STRATEGIES.items():
        print(f'Generating trace: {k}')
        traces[k] = generate_trace(data, cfg)

    bars_1h, funding_1h = data['1h']
    btc = bars_1h['BTC']
    all_dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    combined = combine_targets(traces, WEIGHTS, all_dates)

    rows = []
    for c in CASES:
        m = run_case(bars_1h, funding_1h, combined, c)
        row = dict(name=c['name'],
                   Sharpe=m.get('Sharpe', 0), CAGR=m.get('CAGR', 0),
                   MDD=m.get('MDD', 0), Cal=m.get('Cal', 0),
                   Liq=m.get('Liq', 0), Stops=m.get('Stops', 0))
        rows.append(row)
        print(f"  {c['name']:<22} Sh{row['Sharpe']:.2f} CAGR{row['CAGR']:+.1%} "
              f"MDD{row['MDD']:+.1%} Cal{row['Cal']:.2f} Liq{row['Liq']} Stops{row['Stops']}")

    out = os.path.join(_here, 'new_ensemble_guard_compare.csv')
    with open(out, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved: {out}  ({time.time()-t0:.1f}s)')


if __name__ == '__main__':
    main()
