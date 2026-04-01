#!/usr/bin/env python3
"""상위 스탑 조합 대상 재진입 쿨다운 테스트."""
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

INPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stoploss_results.csv')
OUTPUT_CSV = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'stoploss_cooldown_results.csv')
COOLDOWN_BARS = [0, 6, 12, 24, 48, 72, 168]

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


def load_candidates(top_n):
    rows = []
    with open(INPUT_CSV, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row['leverage'] = float(row['leverage'])
            row['stop_pct'] = float(row['stop_pct'])
            row['stop_lookback_bars'] = int(float(row['stop_lookback_bars']))
            row['Cal'] = float(row['Cal'])
            if row['stop_kind'] == 'none':
                continue
            rows.append(row)

    candidates = []
    for lev in sorted({r['leverage'] for r in rows}):
        sub = [r for r in rows if r['leverage'] == lev]
        sub.sort(key=lambda r: (-r['Cal'], r['stop_kind'], r['stop_pct'], r['stop_lookback_bars']))
        seen = set()
        count = 0
        for r in sub:
            key = (r['leverage'], r['stop_kind'], r['stop_pct'], r['stop_lookback_bars'])
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                'leverage': r['leverage'],
                'stop_kind': r['stop_kind'],
                'stop_pct': r['stop_pct'],
                'stop_lookback_bars': r['stop_lookback_bars'],
            })
            count += 1
            if count >= top_n:
                break
    return candidates


def _init_worker(bars_1h, funding_1h, combined):
    global _WORK_BARS_1H, _WORK_FUNDING_1H, _WORK_COMBINED
    _WORK_BARS_1H = bars_1h
    _WORK_FUNDING_1H = funding_1h
    _WORK_COMBINED = combined


def _run_case(item):
    run_idx, total_runs, cand, cooldown = item
    engine = SingleAccountEngine(
        _WORK_BARS_1H, _WORK_FUNDING_1H,
        leverage=cand['leverage'],
        stop_kind=cand['stop_kind'],
        stop_pct=cand['stop_pct'],
        stop_lookback_bars=cand['stop_lookback_bars'],
        reentry_cooldown_bars=cooldown,
    )
    m = engine.run(_WORK_COMBINED)
    row = {
        'leverage': cand['leverage'],
        'stop_kind': cand['stop_kind'],
        'stop_pct': cand['stop_pct'],
        'stop_lookback_bars': cand['stop_lookback_bars'],
        'reentry_cooldown_bars': cooldown,
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
        f"{cand['leverage']:.1f}x {cand['stop_kind']} pct={cand['stop_pct']:.0%} "
        f"lb={cand['stop_lookback_bars']} cd={cooldown} "
        f"Cal={row['Cal']:.2f} MDD={row['MDD']:+.1%} Liq={row['Liq']} Stops={row['Stops']}"
    )
    return row, progress


def main():
    parser = argparse.ArgumentParser(description='상위 스탑 조합 대상 재진입 쿨다운 테스트')
    parser.add_argument('--workers', type=int, default=max(1, min(8, (os.cpu_count() or 1) - 1)))
    parser.add_argument('--top-n', type=int, default=5)
    args = parser.parse_args()

    if not os.path.exists(INPUT_CSV):
        raise SystemExit(f'missing input: {INPUT_CSV}')

    t0 = time.time()
    candidates = load_candidates(args.top_n)
    print(f'Loaded candidates: {len(candidates)}')

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
    total_runs = len(candidates) * len(COOLDOWN_BARS)
    for cand in candidates:
        for cooldown in COOLDOWN_BARS:
            run_idx += 1
            work_items.append((run_idx, total_runs, cand, cooldown))

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

    rows.sort(key=lambda r: (r['leverage'], -r['Cal'], r['reentry_cooldown_bars']))
    fieldnames = [
        'leverage', 'stop_kind', 'stop_pct', 'stop_lookback_bars', 'reentry_cooldown_bars',
        'Sharpe', 'CAGR', 'MDD', 'Cal', 'Liq', 'Stops', 'Rebal',
    ]
    with open(OUTPUT_CSV, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'\nSaved: {OUTPUT_CSV}')
    print(f'Elapsed: {time.time() - t0:.1f}s')

    print('\nTop 5 by leverage')
    for lev in sorted({r['leverage'] for r in rows}):
        print(f'\n== {lev:.1f}x ==')
        top = [r for r in rows if r['leverage'] == lev][:5]
        for r in top:
            print(
                f"{r['stop_kind']:<30} pct={r['stop_pct']:.0%} lb={r['stop_lookback_bars']:>3} "
                f"cd={r['reentry_cooldown_bars']:>3} Cal={r['Cal']:.2f} "
                f"CAGR={r['CAGR']:+.1%} MDD={r['MDD']:+.1%} Liq={r['Liq']} Stops={r['Stops']}"
            )


if __name__ == '__main__':
    main()
