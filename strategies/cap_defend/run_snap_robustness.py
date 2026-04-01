#!/usr/bin/env python3
"""최종 후보 조합의 snap_interval_bars robustness 체크.

대상 조합:
- 1h_09 + 4h_01 + 4h_09

검증 축:
- 1h_09 snap: 21, 24, 27, 30, 33
- 4h_01 snap: 60, 66, 72, 78, 84
- 4h_09 snap: 15, 18, 21, 24, 27
"""
import argparse
import csv
import multiprocessing as mp
import os
import sys
import time

sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data, run
from run_ensemble import SingleAccountEngine, combine_targets
from run_stoploss_test import START, END

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_CSV = os.path.join(HERE, 'snap_robustness_results.csv')

BASE = {
    '1h_09': dict(interval='1h', sma_bars=168, mom_short_bars=36, mom_long_bars=720,
                  canary_hyst=0.015, drift_threshold=0.0, dd_threshold=0, dd_lookback=0,
                  bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
                  vol_threshold=0.80, n_snapshots=3, snap_interval_bars=24),
    '4h_01': dict(interval='4h', sma_bars=240, mom_short_bars=10, mom_long_bars=30,
                  canary_hyst=0.015, drift_threshold=0.0, dd_threshold=0, dd_lookback=0,
                  bl_drop=0, bl_days=0, health_mode='mom1vol', vol_mode='daily',
                  vol_threshold=0.05, n_snapshots=3, snap_interval_bars=120),
    '4h_09': dict(interval='4h', sma_bars=120, mom_short_bars=20, mom_long_bars=120,
                  canary_hyst=0.015, drift_threshold=0.0, dd_threshold=0, dd_lookback=0,
                  bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
                  vol_threshold=0.60, n_snapshots=3, snap_interval_bars=18),
}

GRID = {
    '1h_09': [21, 24, 27, 30, 33],
    '4h_01': [60, 66, 72, 78, 84, 120],
    '4h_09': [15, 18, 21, 24, 27],
}

_WORK_BARS_1H = None
_WORK_FUNDING_1H = None
_WORK_ALL_DATES = None
_WORK_TRACE_MAP = None


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


def build_cases():
    cases = []
    for snap in GRID['1h_09']:
        cases.append({'changed': '1h_09', 'snap': snap})
    for snap in GRID['4h_01']:
        cases.append({'changed': '4h_01', 'snap': snap})
    for snap in GRID['4h_09']:
        cases.append({'changed': '4h_09', 'snap': snap})
    return cases


def _init_worker(bars_1h, funding_1h, all_dates, trace_map):
    global _WORK_BARS_1H, _WORK_FUNDING_1H, _WORK_ALL_DATES, _WORK_TRACE_MAP
    _WORK_BARS_1H = bars_1h
    _WORK_FUNDING_1H = funding_1h
    _WORK_ALL_DATES = all_dates
    _WORK_TRACE_MAP = trace_map


def _run_case(item):
    run_idx, total_runs, case = item
    traces = {}
    for name in BASE.keys():
        if name == case['changed']:
            if case['snap'] == BASE[name]['snap_interval_bars']:
                traces[name] = _WORK_TRACE_MAP[name]
            else:
                traces[name] = _WORK_TRACE_MAP[f'{name}@{case["snap"]}']
        else:
            traces[name] = _WORK_TRACE_MAP[name]
    weights = {name: 1 / 3 for name in BASE.keys()}
    combined = combine_targets(traces, weights, _WORK_ALL_DATES)
    engine = SingleAccountEngine(
        _WORK_BARS_1H,
        _WORK_FUNDING_1H,
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
    row = {
        'changed': case['changed'],
        'snap': case['snap'],
        'Sharpe': m.get('Sharpe', 0),
        'CAGR': m.get('CAGR', 0),
        'MDD': m.get('MDD', 0),
        'Cal': m.get('Cal', 0),
        'Liq': m.get('Liq', 0),
        'Stops': m.get('Stops', 0),
        'Rebal': m.get('Rebal', 0),
    }
    progress = (
        f"[{run_idx:02d}/{total_runs}] {row['changed']}->{row['snap']} "
        f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} MDD={row['MDD']:+.1%}"
    )
    return row, progress


def main():
    parser = argparse.ArgumentParser(description='최종 snap robustness 체크')
    parser.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 1) - 1)))
    args = parser.parse_args()

    t0 = time.time()
    print('Loading data...')
    data = {iv: load_data(iv) for iv in ['4h', '1h']}
    bars_1h, funding_1h = data['1h']
    all_dates = bars_1h['BTC'].index[(bars_1h['BTC'].index >= START) & (bars_1h['BTC'].index <= END)]

    trace_map = {}
    for name, cfg in BASE.items():
        trace_map[name] = generate_trace(data, cfg)
        for snap in GRID[name]:
            if snap == cfg['snap_interval_bars']:
                continue
            mod = dict(cfg)
            mod['snap_interval_bars'] = snap
            trace_map[f'{name}@{snap}'] = generate_trace(data, mod)

    cases = build_cases()
    work_items = [(idx + 1, len(cases), case) for idx, case in enumerate(cases)]

    rows = []
    if args.workers <= 1:
        _init_worker(bars_1h, funding_1h, all_dates, trace_map)
        for item in work_items:
            row, progress = _run_case(item)
            rows.append(row)
            print(progress)
    else:
        ctx = mp.get_context('fork')
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(bars_1h, funding_1h, all_dates, trace_map),
        ) as pool:
            for row, progress in pool.imap_unordered(_run_case, work_items, chunksize=1):
                rows.append(row)
                print(progress)

    rows.sort(key=lambda r: (-r['Cal'], r['changed'], r['snap']))
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['changed', 'snap', 'Sharpe', 'CAGR', 'MDD', 'Cal', 'Liq', 'Stops', 'Rebal'])
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved: {OUT_CSV}')
    print('\nAll results')
    for row in rows:
        print(
            f"{row['changed']:<6} {row['snap']:>3} "
            f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} MDD={row['MDD']:+.1%}"
        )
    print(f'Elapsed: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
