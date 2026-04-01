#!/usr/bin/env python3
"""동적 레버리지 방법론 비교.

비교 범위:
- 고정 3x/4x/5x
- 계정 단위 동적 레버리지 4종
- 종목 단위 동적 레버리지 4종
- stop: none / prev_close15 + cash_guard(34%)
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

OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'dynamic_leverage_results.csv')

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


BASE_CASES = [
    dict(name='fixed_3x', leverage=3.0),
    dict(name='fixed_4x', leverage=4.0),
    dict(name='fixed_5x', leverage=5.0),
    dict(name='acct_cash_543', leverage_mode='cash_based_543'),
    dict(name='acct_count_543', leverage_mode='count_based_543'),
    dict(name='acct_canary_543', leverage_mode='canary_based_543'),
    dict(name='acct_mixed_543', leverage_mode='mixed_score_543'),
    dict(name='coin_rank_543_cash', per_coin_leverage_mode='rank_543_cash'),
    dict(name='coin_mom_543_cash', per_coin_leverage_mode='momrank_543_cash'),
    dict(name='coin_lowvol_543_cash', per_coin_leverage_mode='lowvol_543_cash'),
    dict(name='coin_capmom_543_cash', per_coin_leverage_mode='cap_mom_blend_543_cash'),
]

STOP_CASES = [
    dict(stop_name='none', stop_kind='none'),
    dict(
        stop_name='prev_close15_cash_guard34',
        stop_kind='prev_close_pct',
        stop_pct=0.15,
        stop_gate='cash_guard',
        stop_gate_cash_threshold=0.34,
    ),
]


def build_cases():
    cases = []
    for base in BASE_CASES:
        for stop in STOP_CASES:
            merged = dict(base)
            merged.update(stop)
            cases.append(merged)
    return cases


def _init_worker(bars_1h, funding_1h, combined):
    global _WORK_BARS_1H, _WORK_FUNDING_1H, _WORK_COMBINED
    _WORK_BARS_1H = bars_1h
    _WORK_FUNDING_1H = funding_1h
    _WORK_COMBINED = combined


def _run_case(item):
    run_idx, total_runs, case = item
    engine = SingleAccountEngine(
        _WORK_BARS_1H, _WORK_FUNDING_1H,
        leverage=case.get('leverage', 4.0),
        stop_kind=case.get('stop_kind', 'none'),
        stop_pct=case.get('stop_pct', 0.0),
        stop_gate=case.get('stop_gate', 'always'),
        stop_gate_cash_threshold=case.get('stop_gate_cash_threshold', 0.0),
        leverage_mode=case.get('leverage_mode', 'fixed'),
        per_coin_leverage_mode=case.get('per_coin_leverage_mode', 'none'),
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
    m = engine.run(_WORK_COMBINED)
    row = {
        'name': case['name'],
        'stop_name': case['stop_name'],
        'leverage': case.get('leverage', 4.0),
        'leverage_mode': case.get('leverage_mode', 'fixed'),
        'per_coin_leverage_mode': case.get('per_coin_leverage_mode', 'none'),
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
        f"[{run_idx:02d}/{total_runs}] {row['name']} + {row['stop_name']} "
        f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
        f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Stops={row['Stops']}"
    )
    return row, progress


def main():
    parser = argparse.ArgumentParser(description='동적 레버리지 비교')
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
    work_items = [(idx + 1, len(cases), case) for idx, case in enumerate(cases)]

    workers = args.workers
    print(f'Workers: {workers}')
    rows = []
    if workers <= 1:
        _init_worker(bars_1h, funding_1h, combined)
        for item in work_items:
            row, progress = _run_case(item)
            rows.append(row)
            print(progress)
    else:
        ctx = mp.get_context('fork')
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(bars_1h, funding_1h, combined),
        ) as pool:
            for row, progress in pool.imap_unordered(_run_case, work_items, chunksize=1):
                rows.append(row)
                print(progress)

    rows.sort(key=lambda r: (-r['Cal'], r['name'], r['stop_name']))
    fieldnames = [
        'name', 'stop_name', 'leverage', 'leverage_mode', 'per_coin_leverage_mode',
        'stop_kind', 'stop_pct', 'stop_gate', 'stop_gate_cash_threshold',
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
            f"{row['name']:<20} {row['stop_name']:<26} "
            f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
            f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Stops={row['Stops']}"
        )


if __name__ == '__main__':
    main()
