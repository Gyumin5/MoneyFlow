#!/usr/bin/env python3
"""최종 후보 전략 위에서 전략별 PFD bars 튜닝.

대상:
- coin_capmom_543_cash + prev_close15 + cash_guard(34%)
- coin_mom_543_cash + prev_close15 + cash_guard(34%)

비교 축:
- 4h1 pfd_bars_override
- 4h2 pfd_bars_override
- 1h1 pfd_bars_override

주의:
- pfd_bars_override=0 은 기존 기본값 유지
  (post_flip_delay=5일 * bars_per_day)
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
from run_stoploss_test import BASE_STRATEGIES, WEIGHTS, START, END

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'pfd_tuning_results.csv')

BASE_CASES = [
    dict(name='coin_capmom_543_cash', per_coin_leverage_mode='cap_mom_blend_543_cash'),
    dict(name='coin_mom_543_cash', per_coin_leverage_mode='momrank_543_cash'),
]

PFD_GRID = {
    '4h1': [0, 2, 3, 6],
    '4h2': [0, 2, 3, 6],
    '1h1': [0, 6, 12, 24],
}

_WORK_BARS_1H = None
_WORK_FUNDING_1H = None
_WORK_TRACES = None


def generate_trace(data, strat_name, strat_cfg, pfd_bars_override):
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
        pfd_bars_override=pfd_bars_override,
        _trace=trace,
        **cfg,
    )
    return strat_name, pfd_bars_override, trace


def build_cases():
    cases = []
    for base in BASE_CASES:
        for pfd_4h1 in PFD_GRID['4h1']:
            for pfd_4h2 in PFD_GRID['4h2']:
                for pfd_1h1 in PFD_GRID['1h1']:
                    cases.append({
                        'name': base['name'],
                        'per_coin_leverage_mode': base['per_coin_leverage_mode'],
                        'pfd_4h1': pfd_4h1,
                        'pfd_4h2': pfd_4h2,
                        'pfd_1h1': pfd_1h1,
                    })
    return cases


def _init_worker(bars_1h, funding_1h, traces):
    global _WORK_BARS_1H, _WORK_FUNDING_1H, _WORK_TRACES
    _WORK_BARS_1H = bars_1h
    _WORK_FUNDING_1H = funding_1h
    _WORK_TRACES = traces


def _run_case(item):
    run_idx, total_runs, case = item
    traces = {
        '4h1': _WORK_TRACES[('4h1', case['pfd_4h1'])],
        '4h2': _WORK_TRACES[('4h2', case['pfd_4h2'])],
        '1h1': _WORK_TRACES[('1h1', case['pfd_1h1'])],
    }
    btc_1h = _WORK_BARS_1H['BTC']
    all_dates = btc_1h.index[(btc_1h.index >= START) & (btc_1h.index <= END)]
    combined = combine_targets(traces, WEIGHTS, all_dates)

    engine = SingleAccountEngine(
        _WORK_BARS_1H,
        _WORK_FUNDING_1H,
        leverage=5.0,
        stop_kind='prev_close_pct',
        stop_pct=0.15,
        stop_gate='cash_guard',
        stop_gate_cash_threshold=0.34,
        leverage_mode='fixed',
        per_coin_leverage_mode=case['per_coin_leverage_mode'],
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
        'name': case['name'],
        'per_coin_leverage_mode': case['per_coin_leverage_mode'],
        'pfd_4h1': case['pfd_4h1'],
        'pfd_4h2': case['pfd_4h2'],
        'pfd_1h1': case['pfd_1h1'],
        'Sharpe': m.get('Sharpe', 0),
        'CAGR': m.get('CAGR', 0),
        'MDD': m.get('MDD', 0),
        'Cal': m.get('Cal', 0),
        'Liq': m.get('Liq', 0),
        'Stops': m.get('Stops', 0),
        'Rebal': m.get('Rebal', 0),
    }
    progress = (
        f"[{run_idx:03d}/{total_runs}] {row['name']} "
        f"PFD(4h1={row['pfd_4h1']},4h2={row['pfd_4h2']},1h1={row['pfd_1h1']}) "
        f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
        f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Stops={row['Stops']}"
    )
    return row, progress


def main():
    parser = argparse.ArgumentParser(description='전략별 PFD bars 튜닝')
    parser.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 1) - 1)))
    args = parser.parse_args()

    t0 = time.time()
    print('Loading data...')
    data = {iv: load_data(iv) for iv in ['4h', '1h']}
    bars_1h, funding_1h = data['1h']

    trace_jobs = []
    for strat_name, cfg in BASE_STRATEGIES.items():
        for pfd_bars in PFD_GRID[strat_name]:
            trace_jobs.append((strat_name, cfg, pfd_bars))

    print(f'Generating {len(trace_jobs)} traces...')
    traces = {}
    for strat_name, cfg, pfd_bars in trace_jobs:
        print(f'  trace {strat_name} pfd={pfd_bars}')
        key = (strat_name, pfd_bars)
        traces[key] = generate_trace(data, strat_name, cfg, pfd_bars)[2]

    cases = build_cases()
    total_runs = len(cases)
    work_items = [(idx + 1, total_runs, case) for idx, case in enumerate(cases)]
    print(f'Total runs: {total_runs}')
    print(f'Workers: {args.workers}')

    rows = []
    if args.workers <= 1:
        _init_worker(bars_1h, funding_1h, traces)
        for item in work_items:
            row, progress = _run_case(item)
            rows.append(row)
            print(progress)
    else:
        ctx = mp.get_context('fork')
        with ctx.Pool(
            processes=args.workers,
            initializer=_init_worker,
            initargs=(bars_1h, funding_1h, traces),
        ) as pool:
            for row, progress in pool.imap_unordered(_run_case, work_items, chunksize=1):
                rows.append(row)
                print(progress)

    rows.sort(key=lambda r: (-r['Cal'], r['name'], r['pfd_4h1'], r['pfd_4h2'], r['pfd_1h1']))
    fieldnames = [
        'name', 'per_coin_leverage_mode', 'pfd_4h1', 'pfd_4h2', 'pfd_1h1',
        'Sharpe', 'CAGR', 'MDD', 'Cal', 'Liq', 'Stops', 'Rebal',
    ]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nSaved: {OUTPUT_CSV}')
    print(f'Elapsed: {time.time() - t0:.1f}s')
    print('\nTop 10')
    for row in rows[:10]:
        print(
            f"{row['name']:<24} "
            f"PFD({row['pfd_4h1']:>2},{row['pfd_4h2']:>2},{row['pfd_1h1']:>2}) "
            f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
            f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Stops={row['Stops']}"
        )


if __name__ == '__main__':
    main()
