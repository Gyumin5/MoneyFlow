#!/usr/bin/env python3
"""1h/4h 전략군 스크리닝.

병렬 실행 전제:
- 1h 스크리닝: 후보 1h + 기준 4h1 + 기준 4h2
- 4h 스크리닝: 후보 4h + 기준 1h1

실행 엔진은 현재 최종 후보 실행층을 유지한다.
  coin_capmom_543_cash + prev_close15 + cash_guard(34%)
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
from run_stoploss_test import BASE_STRATEGIES, START, END

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_1H = os.path.join(HERE, 'signal_screen_1h.csv')
OUT_4H = os.path.join(HERE, 'signal_screen_4h.csv')

WEIGHTS_1H = {'4h1_base': 1 / 3, '4h2_base': 1 / 3, 'cand': 1 / 3}
WEIGHTS_4H = {'1h1_base': 0.5, 'cand': 0.5}

_WORK_BARS_1H = None
_WORK_FUNDING_1H = None
_WORK_ALL_DATES = None
_WORK_BASE_TRACES = None
_TRACE_DATA = None


def build_1h_candidates():
    candidates = []
    idx = 0
    specs = [
        (120, 12, 240, 'mom2vol', 0.80, 84),
        (120, 24, 240, 'mom2vol', 0.80, 84),
        (120, 24, 480, 'mom2vol', 0.80, 84),
        (120, 36, 480, 'mom2vol', 0.80, 168),
        (120, 48, 720, 'mom2vol', 0.60, 84),
        (168, 12, 240, 'mom2vol', 0.80, 84),
        (168, 24, 480, 'mom2vol', 0.80, 168),
        (168, 24, 480, 'mom1vol', 0.80, 168),
        (168, 36, 720, 'mom2vol', 0.80, 168),
        (168, 48, 960, 'mom2vol', 0.80, 336),
        (200, 24, 480, 'mom2vol', 0.80, 168),
        (200, 36, 720, 'mom2vol', 0.80, 336),
        (200, 36, 720, 'mom1vol', 0.80, 168),
        (200, 48, 960, 'mom2vol', 0.60, 336),
        (200, 72, 960, 'mom2vol', 0.80, 336),
    ]
    for sma, mshort, mlong, hmode, vth, snap in specs:
        idx += 1
        candidates.append({
            'name': f'1h_{idx:02d}',
            'interval': '1h',
            'sma_bars': sma,
            'mom_short_bars': mshort,
            'mom_long_bars': mlong,
            'canary_hyst': 0.015,
            'drift_threshold': 0.0,
            'dd_threshold': 0,
            'dd_lookback': 0,
            'bl_drop': 0,
            'bl_days': 0,
            'health_mode': hmode,
            'vol_mode': 'bar',
            'vol_threshold': vth,
            'n_snapshots': 3,
            'snap_interval_bars': snap,
        })
    return candidates


def build_4h_candidates():
    candidates = []
    idx = 0
    specs = [
        (240, 10, 30, 'mom1vol', 'daily', 0.05, 120),
        (240, 10, 60, 'mom1vol', 'daily', 0.05, 72),
        (180, 10, 30, 'mom1vol', 'daily', 0.05, 120),
        (180, 20, 60, 'mom2vol', 'bar', 0.80, 36),
        (180, 20, 120, 'mom2vol', 'bar', 0.80, 72),
        (120, 10, 30, 'mom1vol', 'daily', 0.04, 72),
        (120, 20, 60, 'mom2vol', 'bar', 0.80, 18),
        (120, 20, 120, 'mom2vol', 'bar', 0.80, 18),
        (120, 20, 120, 'mom2vol', 'bar', 0.60, 18),
        (120, 30, 120, 'mom2vol', 'bar', 0.80, 36),
        (90, 10, 30, 'mom1vol', 'daily', 0.05, 72),
        (90, 20, 60, 'mom2vol', 'bar', 0.80, 18),
        (90, 30, 120, 'mom2vol', 'bar', 0.60, 18),
        (240, 20, 120, 'mom2vol', 'bar', 0.80, 36),
        (180, 30, 120, 'mom2vol', 'bar', 0.80, 18),
    ]
    for sma, mshort, mlong, hmode, vmode, vth, snap in specs:
        idx += 1
        candidates.append({
            'name': f'4h_{idx:02d}',
            'interval': '4h',
            'sma_bars': sma,
            'mom_short_bars': mshort,
            'mom_long_bars': mlong,
            'canary_hyst': 0.015,
            'drift_threshold': 0.0,
            'dd_threshold': 0,
            'dd_lookback': 0,
            'bl_drop': 0,
            'bl_days': 0,
            'health_mode': hmode,
            'vol_mode': vmode,
            'vol_threshold': vth,
            'n_snapshots': 3,
            'snap_interval_bars': snap,
        })
    return candidates


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


def _init_trace_worker(data):
    global _TRACE_DATA
    _TRACE_DATA = data


def _generate_trace_item(item):
    idx, total, cand = item
    trace = generate_trace(_TRACE_DATA, cand)
    progress = f"[trace {idx:02d}/{total}] {cand['name']}"
    return cand, trace, progress


def _init_worker(bars_1h, funding_1h, all_dates, base_traces):
    global _WORK_BARS_1H, _WORK_FUNDING_1H, _WORK_ALL_DATES, _WORK_BASE_TRACES
    _WORK_BARS_1H = bars_1h
    _WORK_FUNDING_1H = funding_1h
    _WORK_ALL_DATES = all_dates
    _WORK_BASE_TRACES = base_traces


def _run_case(item):
    run_idx, total_runs, stage, cand, cand_trace = item
    if stage == '1h':
        traces = {
            '4h1_base': _WORK_BASE_TRACES['4h1_base'],
            '4h2_base': _WORK_BASE_TRACES['4h2_base'],
            'cand': cand_trace,
        }
        weights = WEIGHTS_1H
    else:
        traces = {
            '1h1_base': _WORK_BASE_TRACES['1h1_base'],
            'cand': cand_trace,
        }
        weights = WEIGHTS_4H

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
        'stage': stage,
        'name': cand['name'],
        'sma_bars': cand['sma_bars'],
        'mom_short_bars': cand['mom_short_bars'],
        'mom_long_bars': cand['mom_long_bars'],
        'health_mode': cand['health_mode'],
        'vol_mode': cand['vol_mode'],
        'vol_threshold': cand['vol_threshold'],
        'snap_interval_bars': cand['snap_interval_bars'],
        'Sharpe': m.get('Sharpe', 0),
        'CAGR': m.get('CAGR', 0),
        'MDD': m.get('MDD', 0),
        'Cal': m.get('Cal', 0),
        'Liq': m.get('Liq', 0),
        'Stops': m.get('Stops', 0),
        'Rebal': m.get('Rebal', 0),
    }
    progress = (
        f"[{run_idx:02d}/{total_runs}] {stage} {row['name']} "
        f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
        f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Stops={row['Stops']}"
    )
    return row, progress


def run_stage(stage, workers):
    print(f'Loading data for stage={stage}...')
    data = {iv: load_data(iv) for iv in ['4h', '1h']}
    bars_1h, funding_1h = data['1h']
    all_dates = bars_1h['BTC'].index[(bars_1h['BTC'].index >= START) & (bars_1h['BTC'].index <= END)]

    if stage == '1h':
        candidates = build_1h_candidates()
        base_traces = {
            '4h1_base': generate_trace(data, BASE_STRATEGIES['4h1']),
            '4h2_base': generate_trace(data, BASE_STRATEGIES['4h2']),
        }
        out_csv = OUT_1H
    else:
        candidates = build_4h_candidates()
        base_traces = {
            '1h1_base': generate_trace(data, BASE_STRATEGIES['1h1']),
        }
        out_csv = OUT_4H

    print(f'Generating candidate traces: {len(candidates)}')
    trace_jobs = [(idx + 1, len(candidates), cand) for idx, cand in enumerate(candidates)]
    trace_items = []
    if workers <= 1:
        _init_trace_worker(data)
        for job in trace_jobs:
            cand, trace, progress = _generate_trace_item(job)
            trace_items.append((cand, trace))
            print(progress)
    else:
        ctx = mp.get_context('fork')
        with ctx.Pool(processes=workers, initializer=_init_trace_worker, initargs=(data,)) as pool:
            for cand, trace, progress in pool.imap_unordered(_generate_trace_item, trace_jobs, chunksize=1):
                trace_items.append((cand, trace))
                print(progress)
    work_items = [(idx + 1, len(trace_items), stage, cand, trace) for idx, (cand, trace) in enumerate(trace_items)]
    print(f'Running screen cases: {len(work_items)}')

    rows = []
    if workers <= 1:
        _init_worker(bars_1h, funding_1h, all_dates, base_traces)
        for item in work_items:
            row, progress = _run_case(item)
            rows.append(row)
            print(progress)
    else:
        ctx = mp.get_context('fork')
        with ctx.Pool(
            processes=workers,
            initializer=_init_worker,
            initargs=(bars_1h, funding_1h, all_dates, base_traces),
        ) as pool:
            for row, progress in pool.imap_unordered(_run_case, work_items, chunksize=1):
                rows.append(row)
                print(progress)

    rows.sort(key=lambda r: (-r['Cal'], -r['Sharpe'], r['name']))
    fieldnames = [
        'stage', 'name', 'sma_bars', 'mom_short_bars', 'mom_long_bars',
        'health_mode', 'vol_mode', 'vol_threshold', 'snap_interval_bars',
        'Sharpe', 'CAGR', 'MDD', 'Cal', 'Liq', 'Stops', 'Rebal',
    ]
    with open(out_csv, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved: {out_csv}")
    print("Top 10")
    for row in rows[:10]:
        print(
            f"{row['name']:<8} sma={row['sma_bars']:>3} ms={row['mom_short_bars']:>3} "
            f"ml={row['mom_long_bars']:>4} {row['health_mode']:<8} "
            f"vol={row['vol_mode']}:{row['vol_threshold']:.2f} snap={row['snap_interval_bars']:>3} "
            f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} MDD={row['MDD']:+.1%}"
        )


def main():
    parser = argparse.ArgumentParser(description='1h/4h 전략군 스크리닝')
    parser.add_argument('--stage', choices=['1h', '4h'], required=True)
    parser.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 1) - 1)))
    args = parser.parse_args()
    t0 = time.time()
    run_stage(args.stage, args.workers)
    print(f'Elapsed: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
