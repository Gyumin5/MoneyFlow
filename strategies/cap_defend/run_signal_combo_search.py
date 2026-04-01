#!/usr/bin/env python3
"""상위 1h/4h 후보 조합 탐색.

입력:
- signal_screen_1h.csv 상위 4개
- signal_screen_4h.csv 상위 4개

탐색:
- 2개 조합: 1h1+4h1, 1h2, 4h2
- 3개 조합: 1h2+4h1, 1h1+4h2
- 4개 조합: 1h2+4h2

실행층:
- coin_capmom_543_cash + prev_close15 + cash_guard(34%)
"""
import argparse
import csv
import itertools
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
SCREEN_1H = os.path.join(HERE, 'signal_screen_1h.csv')
SCREEN_4H = os.path.join(HERE, 'signal_screen_4h.csv')
OUT_CSV = os.path.join(HERE, 'signal_combo_search.csv')

TOP_K = 4
_WORK_BARS_1H = None
_WORK_FUNDING_1H = None
_WORK_ALL_DATES = None
_WORK_TRACE_MAP = None


def load_top_candidates(path, top_k):
    rows = []
    with open(path, newline='') as f:
        for row in csv.DictReader(f):
            rows.append(row)
    rows = rows[:top_k]
    cands = []
    for row in rows:
        cands.append({
            'name': row['name'],
            'interval': row['stage'],
            'sma_bars': int(row['sma_bars']),
            'mom_short_bars': int(row['mom_short_bars']),
            'mom_long_bars': int(row['mom_long_bars']),
            'canary_hyst': 0.015,
            'drift_threshold': 0.0,
            'dd_threshold': 0,
            'dd_lookback': 0,
            'bl_drop': 0,
            'bl_days': 0,
            'health_mode': row['health_mode'],
            'vol_mode': row['vol_mode'],
            'vol_threshold': float(row['vol_threshold']),
            'n_snapshots': 3,
            'snap_interval_bars': int(row['snap_interval_bars']),
        })
    return cands


def generate_trace(data, cfg):
    run_cfg = dict(cfg)
    run_cfg.pop('name', None)
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


def build_cases(one_h, four_h):
    cases = []

    for a in one_h:
        for b in four_h:
            cases.append({'kind': '2mix', 'members': [a['name'], b['name']]})
    for a, b in itertools.combinations(one_h, 2):
        cases.append({'kind': '2x1h', 'members': [a['name'], b['name']]})
    for a, b in itertools.combinations(four_h, 2):
        cases.append({'kind': '2x4h', 'members': [a['name'], b['name']]})

    for a, b in itertools.combinations(one_h, 2):
        for c in four_h:
            cases.append({'kind': '3_1h1h4h', 'members': [a['name'], b['name'], c['name']]})
    for a in one_h:
        for b, c in itertools.combinations(four_h, 2):
            cases.append({'kind': '3_1h4h4h', 'members': [a['name'], b['name'], c['name']]})

    for a, b in itertools.combinations(one_h, 2):
        for c, d in itertools.combinations(four_h, 2):
            cases.append({'kind': '4_1h1h4h4h', 'members': [a['name'], b['name'], c['name'], d['name']]})
    return cases


def _init_worker(bars_1h, funding_1h, all_dates, trace_map):
    global _WORK_BARS_1H, _WORK_FUNDING_1H, _WORK_ALL_DATES, _WORK_TRACE_MAP
    _WORK_BARS_1H = bars_1h
    _WORK_FUNDING_1H = funding_1h
    _WORK_ALL_DATES = all_dates
    _WORK_TRACE_MAP = trace_map


def _run_case(item):
    run_idx, total_runs, case = item
    weights = {name: 1.0 / len(case['members']) for name in case['members']}
    traces = {name: _WORK_TRACE_MAP[name] for name in case['members']}
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
        'kind': case['kind'],
        'members': '+'.join(case['members']),
        'n_members': len(case['members']),
        'Sharpe': m.get('Sharpe', 0),
        'CAGR': m.get('CAGR', 0),
        'MDD': m.get('MDD', 0),
        'Cal': m.get('Cal', 0),
        'Liq': m.get('Liq', 0),
        'Stops': m.get('Stops', 0),
        'Rebal': m.get('Rebal', 0),
    }
    progress = (
        f"[{run_idx:03d}/{total_runs}] {row['kind']} {row['members']} "
        f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
        f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Stops={row['Stops']}"
    )
    return row, progress


def main():
    parser = argparse.ArgumentParser(description='상위 신호 조합 탐색')
    parser.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 1) - 1)))
    args = parser.parse_args()

    t0 = time.time()
    one_h = load_top_candidates(SCREEN_1H, TOP_K)
    four_h = load_top_candidates(SCREEN_4H, TOP_K)
    print(f'Loaded top candidates: 1h={len(one_h)} 4h={len(four_h)}')

    data = {iv: load_data(iv) for iv in ['4h', '1h']}
    bars_1h, funding_1h = data['1h']
    all_dates = bars_1h['BTC'].index[(bars_1h['BTC'].index >= START) & (bars_1h['BTC'].index <= END)]

    trace_map = {}
    for cand in one_h + four_h:
        print(f"Generating trace: {cand['name']}")
        trace_map[cand['name']] = generate_trace(data, cand)

    cases = build_cases(one_h, four_h)
    print(f'Total combo cases: {len(cases)}')
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

    rows.sort(key=lambda r: (-r['Cal'], -r['Sharpe'], r['members']))
    fieldnames = ['kind', 'members', 'n_members', 'Sharpe', 'CAGR', 'MDD', 'Cal', 'Liq', 'Stops', 'Rebal']
    with open(OUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f'Saved: {OUT_CSV}')
    print('\nTop 15')
    for row in rows[:15]:
        print(
            f"{row['kind']:<12} {row['members']:<40} "
            f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} MDD={row['MDD']:+.1%}"
        )
    print(f'Elapsed: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
