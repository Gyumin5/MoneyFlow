#!/usr/bin/env python3
"""앙상블 스탑로스 배치 테스트.

기본 전략: 4h1 + 4h2 + 1h1
리스크 옵션: DD/Blacklist 비활성
실행 엔진: SingleAccountEngine (2x~5x)
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

START = '2020-10-01'
END = '2026-03-28'
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stoploss_results.csv')

BASE_STRATEGIES = {
    '4h1': dict(
        interval='4h',
        sma_bars=240, mom_short_bars=10, mom_long_bars=30,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode='mom1vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=120,
    ),
    '4h2': dict(
        interval='4h',
        sma_bars=120, mom_short_bars=20, mom_long_bars=120,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode='mom2vol', vol_mode='bar', vol_threshold=0.80,
        n_snapshots=3, snap_interval_bars=18,
    ),
    '1h1': dict(
        interval='1h',
        sma_bars=200, mom_short_bars=200, mom_long_bars=1200,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode='mom2vol', vol_mode='bar', vol_threshold=0.80,
        n_snapshots=3, snap_interval_bars=336,
    ),
}

WEIGHTS = {'4h1': 1/3, '4h2': 1/3, '1h1': 1/3}
LEVERAGES = [2.0, 3.0, 4.0, 5.0]
STOP_PCTS = [0.03, 0.05, 0.07, 0.10, 0.12, 0.15]
LOOKBACKS = [3, 6, 12, 24]

_WORK_BARS_1H = None
_WORK_FUNDING_1H = None
_WORK_COMBINED = None


def build_cases():
    cases = [{'stop_kind': 'none', 'stop_pct': 0.0, 'stop_lookback_bars': 0}]

    for kind in [
        'prev_close_pct',
        'highest_close_since_entry_pct',
        'highest_high_since_entry_pct',
    ]:
        for pct in STOP_PCTS:
            cases.append({'stop_kind': kind, 'stop_pct': pct, 'stop_lookback_bars': 0})

    for kind in ['rolling_high_close_pct', 'rolling_high_high_pct']:
        for lookback in LOOKBACKS:
            for pct in STOP_PCTS:
                cases.append({'stop_kind': kind, 'stop_pct': pct, 'stop_lookback_bars': lookback})
    return cases


def generate_trace(data, strat_cfg):
    cfg = dict(strat_cfg)
    interval = cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(
        bars, funding,
        interval=interval,
        leverage=1.0,
        start_date=START,
        end_date=END,
        _trace=trace,
        **cfg,
    )
    return trace


def _init_worker(bars_1h, funding_1h, combined):
    global _WORK_BARS_1H, _WORK_FUNDING_1H, _WORK_COMBINED
    _WORK_BARS_1H = bars_1h
    _WORK_FUNDING_1H = funding_1h
    _WORK_COMBINED = combined


def _run_case(item):
    run_idx, total_runs, lev, case = item
    engine = SingleAccountEngine(
        _WORK_BARS_1H, _WORK_FUNDING_1H,
        leverage=lev,
        stop_kind=case['stop_kind'],
        stop_pct=case['stop_pct'],
        stop_lookback_bars=case['stop_lookback_bars'],
    )
    m = engine.run(_WORK_COMBINED)
    row = {
        'leverage': lev,
        'stop_kind': case['stop_kind'],
        'stop_pct': case['stop_pct'],
        'stop_lookback_bars': case['stop_lookback_bars'],
        'Sharpe': m.get('Sharpe', 0),
        'CAGR': m.get('CAGR', 0),
        'MDD': m.get('MDD', 0),
        'Cal': m.get('Cal', 0),
        'Liq': m.get('Liq', 0),
        'Stops': m.get('Stops', 0),
        'Rebal': m.get('Rebal', 0),
    }
    progress = (
        f"[{run_idx:03d}/{total_runs}] "
        f"{lev:.1f}x {case['stop_kind']} pct={case['stop_pct']:.0%} lb={case['stop_lookback_bars']} "
        f"Cal={row['Cal']:.2f} MDD={row['MDD']:+.1%} Liq={row['Liq']} Stops={row['Stops']}"
    )
    return row, progress


def main():
    parser = argparse.ArgumentParser(description='앙상블 스탑로스 배치 테스트')
    parser.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 1) - 1)))
    args = parser.parse_args()

    t0 = time.time()
    print('Loading data...')
    data = {iv: load_data(iv) for iv in ['4h', '1h']}
    traces = {}
    for key, cfg in BASE_STRATEGIES.items():
        print(f'Generating trace: {key}')
        traces[key] = generate_trace(data, cfg)

    bars_1h, funding_1h = data['1h']
    btc_1h = bars_1h['BTC']
    all_dates = btc_1h.index[(btc_1h.index >= START) & (btc_1h.index <= END)]
    combined = combine_targets(traces, WEIGHTS, all_dates)

    cases = build_cases()
    print(f'Total stop cases: {len(cases)}')
    total_runs = len(cases) * len(LEVERAGES)
    work_items = []
    run_idx = 0
    for lev in LEVERAGES:
        for case in cases:
            run_idx += 1
            work_items.append((run_idx, total_runs, lev, case))

    print(f'Workers: {args.workers}')
    rows = []
    if args.workers <= 1:
        _init_worker(bars_1h, funding_1h, combined)
        for item in work_items:
            row, progress = _run_case(item)
            rows.append(row)
            print(progress)
    else:
        ctx = mp.get_context('fork')
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(bars_1h, funding_1h, combined),
        ) as pool:
            for row, progress in pool.imap_unordered(_run_case, work_items, chunksize=1):
                rows.append(row)
                print(progress)

    rows.sort(key=lambda r: (r['leverage'], -r['Cal'], -r['Sharpe']))
    fieldnames = [
        'leverage', 'stop_kind', 'stop_pct', 'stop_lookback_bars',
        'Sharpe', 'CAGR', 'MDD', 'Cal', 'Liq', 'Stops', 'Rebal',
    ]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nSaved: {OUTPUT_CSV}')
    print(f'Elapsed: {time.time() - t0:.1f}s')

    print('\nTop 5 by leverage')
    for lev in LEVERAGES:
        print(f'\n== {lev:.1f}x ==')
        top = [r for r in rows if r['leverage'] == lev][:5]
        for r in top:
            print(
                f"{r['stop_kind']:<30} pct={r['stop_pct']:.0%} lb={r['stop_lookback_bars']:>2} "
                f"Cal={r['Cal']:.2f} CAGR={r['CAGR']:+.1%} MDD={r['MDD']:+.1%} "
                f"Liq={r['Liq']} Stops={r['Stops']}"
            )


if __name__ == '__main__':
    main()
