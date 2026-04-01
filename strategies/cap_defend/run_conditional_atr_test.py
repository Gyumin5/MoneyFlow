#!/usr/bin/env python3
"""조건부 스탑 + ATR 스탑 비교 테스트."""
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

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'conditional_atr_results.csv')
LEVERAGES = [3.0, 4.0, 5.0]

_WORK_BARS_1H = None
_WORK_FUNDING_1H = None
_WORK_COMBINED = None


CASES = [
    dict(name='none', stop_kind='none'),

    dict(name='prev_close15', stop_kind='prev_close_pct', stop_pct=0.15),
    dict(name='prev_close15_target_exit', stop_kind='prev_close_pct', stop_pct=0.15, stop_gate='target_exit_only'),
    dict(name='prev_close15_cash_guard', stop_kind='prev_close_pct', stop_pct=0.15, stop_gate='cash_guard', stop_gate_cash_threshold=0.34),
    dict(name='prev_close15_exit_or_cash', stop_kind='prev_close_pct', stop_pct=0.15, stop_gate='target_exit_or_cash_guard', stop_gate_cash_threshold=0.34),

    dict(name='rolling_hh15_lb24', stop_kind='rolling_high_high_pct', stop_pct=0.15, stop_lookback_bars=24),
    dict(name='rolling_hh15_lb24_target_exit', stop_kind='rolling_high_high_pct', stop_pct=0.15, stop_lookback_bars=24, stop_gate='target_exit_only'),
    dict(name='rolling_hh15_lb24_cash_guard', stop_kind='rolling_high_high_pct', stop_pct=0.15, stop_lookback_bars=24, stop_gate='cash_guard', stop_gate_cash_threshold=0.34),
    dict(name='rolling_hh15_lb24_exit_or_cash', stop_kind='rolling_high_high_pct', stop_pct=0.15, stop_lookback_bars=24, stop_gate='target_exit_or_cash_guard', stop_gate_cash_threshold=0.34),

    dict(name='atr_hh_atr24_x2.5', stop_kind='atr_highest_high_since_entry', stop_atr_lookback_bars=24, stop_atr_mult=2.5),
    dict(name='atr_hh_atr24_x3.0', stop_kind='atr_highest_high_since_entry', stop_atr_lookback_bars=24, stop_atr_mult=3.0),
    dict(name='atr_hh_atr48_x3.0', stop_kind='atr_highest_high_since_entry', stop_atr_lookback_bars=48, stop_atr_mult=3.0),
    dict(name='atr_hh_atr24_x3.0_exit_or_cash', stop_kind='atr_highest_high_since_entry', stop_atr_lookback_bars=24, stop_atr_mult=3.0, stop_gate='target_exit_or_cash_guard', stop_gate_cash_threshold=0.34),

    dict(name='atr_roll_hh_lb24_atr24_x2.5', stop_kind='atr_rolling_high_high', stop_lookback_bars=24, stop_atr_lookback_bars=24, stop_atr_mult=2.5),
    dict(name='atr_roll_hh_lb24_atr24_x3.0', stop_kind='atr_rolling_high_high', stop_lookback_bars=24, stop_atr_lookback_bars=24, stop_atr_mult=3.0),
    dict(name='atr_roll_hh_lb24_atr48_x3.0', stop_kind='atr_rolling_high_high', stop_lookback_bars=24, stop_atr_lookback_bars=48, stop_atr_mult=3.0),
    dict(name='atr_roll_hh_lb24_atr24_x4.0', stop_kind='atr_rolling_high_high', stop_lookback_bars=24, stop_atr_lookback_bars=24, stop_atr_mult=4.0),
    dict(name='atr_roll_hh_lb24_atr24_x3.0_exit_or_cash', stop_kind='atr_rolling_high_high', stop_lookback_bars=24, stop_atr_lookback_bars=24, stop_atr_mult=3.0, stop_gate='target_exit_or_cash_guard', stop_gate_cash_threshold=0.34),
]


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
        stop_lookback_bars=case.get('stop_lookback_bars', 0),
        reentry_cooldown_bars=case.get('reentry_cooldown_bars', 0),
        stop_gate=case.get('stop_gate', 'always'),
        stop_gate_cash_threshold=case.get('stop_gate_cash_threshold', 0.0),
        stop_atr_lookback_bars=case.get('stop_atr_lookback_bars', 0),
        stop_atr_mult=case.get('stop_atr_mult', 0.0),
    )
    m = engine.run(_WORK_COMBINED)
    row = {
        'leverage': lev,
        'name': case['name'],
        'stop_kind': case.get('stop_kind', 'none'),
        'stop_pct': case.get('stop_pct', 0.0),
        'stop_lookback_bars': case.get('stop_lookback_bars', 0),
        'stop_gate': case.get('stop_gate', 'always'),
        'stop_gate_cash_threshold': case.get('stop_gate_cash_threshold', 0.0),
        'stop_atr_lookback_bars': case.get('stop_atr_lookback_bars', 0),
        'stop_atr_mult': case.get('stop_atr_mult', 0.0),
        'Sharpe': m.get('Sharpe', 0),
        'CAGR': m.get('CAGR', 0),
        'MDD': m.get('MDD', 0),
        'Cal': m.get('Cal', 0),
        'Liq': m.get('Liq', 0),
        'Stops': m.get('Stops', 0),
        'Rebal': m.get('Rebal', 0),
    }
    progress = (
        f"[{run_idx:03d}/{total_runs}] {lev:.1f}x {case['name']} "
        f"Cal={row['Cal']:.2f} MDD={row['MDD']:+.1%} Liq={row['Liq']} Stops={row['Stops']}"
    )
    return row, progress


def main():
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
        'leverage', 'name', 'stop_kind', 'stop_pct', 'stop_lookback_bars',
        'stop_gate', 'stop_gate_cash_threshold', 'stop_atr_lookback_bars', 'stop_atr_mult',
        'Sharpe', 'CAGR', 'MDD', 'Cal', 'Liq', 'Stops', 'Rebal',
    ]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nSaved: {OUTPUT_CSV}')
    print(f'Elapsed: {time.time() - t0:.1f}s')

    print('\nTop 8 by leverage')
    for lev in LEVERAGES:
        print(f'\n== {lev:.1f}x ==')
        top = [r for r in rows if r['leverage'] == lev][:8]
        for r in top:
            print(
                f"{r['name']:<34} Cal={r['Cal']:.2f} CAGR={r['CAGR']:+.1%} "
                f"MDD={r['MDD']:+.1%} Liq={r['Liq']} Stops={r['Stops']}"
            )


if __name__ == '__main__':
    main()
