#!/usr/bin/env python3
"""시총순 Top5 EW 앙상블 비교.

비교 대상:
- none
- prev_close 15% + cash_guard(20/34/40%)

전략 파라미터는 기존 4h1 + 4h2 + 1h1을 유지하되,
selection='baseline', cap=1.0 으로 바꿔 Greedy 없이 Top5 EW를 만든다.
"""
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

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'top5ew_compare_results.csv')

BASE_STRATEGIES_TOP5EW = {
    '4h1': dict(
        interval='4h',
        sma_bars=240, mom_short_bars=10, mom_long_bars=30,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode='mom1vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=120,
        universe_size=5, selection='baseline', cap=1.0,
    ),
    '4h2': dict(
        interval='4h',
        sma_bars=120, mom_short_bars=20, mom_long_bars=120,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode='mom2vol', vol_mode='bar', vol_threshold=0.80,
        n_snapshots=3, snap_interval_bars=18,
        universe_size=5, selection='baseline', cap=1.0,
    ),
    '1h1': dict(
        interval='1h',
        sma_bars=200, mom_short_bars=200, mom_long_bars=1200,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode='mom2vol', vol_mode='bar', vol_threshold=0.80,
        n_snapshots=3, snap_interval_bars=336,
        universe_size=5, selection='baseline', cap=1.0,
    ),
}

WEIGHTS = {'4h1': 1/3, '4h2': 1/3, '1h1': 1/3}
LEVERAGES = [2.0, 3.0, 4.0, 5.0]

CASES = [
    dict(name='none', stop_kind='none'),
    dict(name='prev_close15_cash20', stop_kind='prev_close_pct', stop_pct=0.15,
         stop_gate='cash_guard', stop_gate_cash_threshold=0.20),
    dict(name='prev_close15_cash34', stop_kind='prev_close_pct', stop_pct=0.15,
         stop_gate='cash_guard', stop_gate_cash_threshold=0.34),
    dict(name='prev_close15_cash40', stop_kind='prev_close_pct', stop_pct=0.15,
         stop_gate='cash_guard', stop_gate_cash_threshold=0.40),
]

_WORK_BARS_1H = None
_WORK_FUNDING_1H = None
_WORK_COMBINED = None


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
        stop_kind=case.get('stop_kind', 'none'),
        stop_pct=case.get('stop_pct', 0.0),
        stop_gate=case.get('stop_gate', 'always'),
        stop_gate_cash_threshold=case.get('stop_gate_cash_threshold', 0.0),
    )
    m = engine.run(_WORK_COMBINED)
    row = {
        'leverage': lev,
        'name': case['name'],
        'stop_kind': case.get('stop_kind', 'none'),
        'stop_pct': case.get('stop_pct', 0.0),
        'stop_gate': case.get('stop_gate', 'always'),
        'stop_gate_cash_threshold': case.get('stop_gate_cash_threshold', 0.0),
        'Sharpe': m.get('Sharpe', 0),
        'CAGR': m.get('CAGR', 0),
        'MDD': m.get('MDD', 0),
        'Cal': m.get('Cal', 0),
        'Liq': m.get('Liq', 0),
        'Stops': m.get('Stops', 0),
        'Rebal': m.get('Rebal', 0),
    }
    progress = (
        f"[{run_idx:02d}/{total_runs}] {lev:.1f}x {case['name']} "
        f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
        f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Stops={row['Stops']}"
    )
    return row, progress


def main():
    t0 = time.time()
    print('Loading data...')
    data = {iv: load_data(iv) for iv in ['4h', '1h']}
    traces = {}
    for key, cfg in BASE_STRATEGIES_TOP5EW.items():
        print(f'Generating trace: {key}')
        traces[key] = generate_trace(data, cfg)

    bars_1h, funding_1h = data['1h']
    btc_1h = bars_1h['BTC']
    all_dates = btc_1h.index[(btc_1h.index >= START) & (btc_1h.index <= END)]
    combined = combine_targets(traces, WEIGHTS, all_dates)

    work_items = []
    run_idx = 0
    total_runs = len(LEVERAGES) * len(CASES)
    for lev in LEVERAGES:
        for case in CASES:
            run_idx += 1
            work_items.append((run_idx, total_runs, lev, case))

    workers = max(1, min(8, (os.cpu_count() or 1) - 1))
    print(f'Workers: {workers}')
    rows = []
    ctx = mp.get_context('fork')
    with ctx.Pool(
        processes=workers,
        initializer=_init_worker,
        initargs=(bars_1h, funding_1h, combined),
    ) as pool:
        for row, progress in pool.imap_unordered(_run_case, work_items, chunksize=1):
            rows.append(row)
            print(progress)

    rows.sort(key=lambda r: (r['leverage'], -r['Cal'], r['name']))
    fieldnames = [
        'leverage', 'name', 'stop_kind', 'stop_pct',
        'stop_gate', 'stop_gate_cash_threshold',
        'Sharpe', 'CAGR', 'MDD', 'Cal', 'Liq', 'Stops', 'Rebal',
    ]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nSaved: {OUTPUT_CSV}')
    print(f'Elapsed: {time.time() - t0:.1f}s')

    for lev in LEVERAGES:
        print(f'\n== {lev:.1f}x ==')
        top = [r for r in rows if r['leverage'] == lev]
        for r in top:
            print(
                f"{r['name']:<22} Cal={r['Cal']:.2f} CAGR={r['CAGR']:+.1%} "
                f"MDD={r['MDD']:+.1%} Liq={r['Liq']} Stops={r['Stops']}"
            )


if __name__ == '__main__':
    main()
