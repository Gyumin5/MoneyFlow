#!/usr/bin/env python3
"""Health filter × multi-anchor-day comparison.
Tests if Mom21 health benefit persists across all rebalancing days."""

import sys, os, numpy as np
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from test_stock_improve import (
    SP, load_prices, precompute, run_bt, metrics, _init, ALL_TICKERS
)

# Config
DAYS = [1, 2, 3, 4, 5, 8, 10, 12, 15, 18, 20]
HEALTHS = [
    ('None',     'none'),
    ('SMA200',   'sma200'),
    ('Mom21',    'mom21'),
    ('Mom42',    'mom42'),
    ('Mom63',    'mom63'),
    ('Mom126',   'mom126'),
    ('Mom21+63', 'mom21_63'),
]

R8 = ('SPY','QQQ','VGK','EWJ','EEM','VWO','GLD','PDBC')
DEF = ('IEF','BIL','BNDX','GLD','PDBC')


def make_task(h_val, day):
    return SP(offensive=R8, defensive=DEF,
              canary_assets=('EEM',), canary_hyst=0.005,
              health=h_val, defense='top3', mom_style='12m', weight='ew',
              _anchor=day)


def worker(args):
    """Must be top-level for pickle."""
    h_name, day, p = args
    from test_stock_improve import _g_prices, _g_ind
    df = run_bt(_g_prices, _g_ind, p)
    m = metrics(df) if df is not None else None
    return h_name, day, m


def main():
    print('Loading prices...')
    prices = load_prices(ALL_TICKERS)
    print(f'  {len(prices)} tickers loaded')

    print('Precomputing indicators...')
    ind = precompute(prices)
    print('  Done')

    # Build tasks
    all_tasks = []
    for h_name, h_val in HEALTHS:
        for day in DAYS:
            p = make_task(h_val, day)
            all_tasks.append((h_name, day, p))

    print(f'Running {len(all_tasks)} configs with 24 workers...\n')

    with Pool(24, initializer=_init, initargs=(prices, ind)) as pool:
        results = pool.map(worker, all_tasks)

    # Aggregate
    by_health = defaultdict(list)
    per_day = {}
    for h_name, day, m in results:
        if m:
            by_health[h_name].append(m)
            per_day[(h_name, day)] = m

    # Table 1: Aggregated
    rows = []
    for h_name, _ in HEALTHS:
        ms = by_health[h_name]
        if not ms: continue
        sharpes = [m['Sharpe'] for m in ms]
        cagrs = [m['CAGR'] for m in ms]
        mdds = [m['MDD'] for m in ms]
        rows.append((h_name, np.mean(sharpes), np.std(sharpes),
                      np.mean(cagrs), np.mean(mdds),
                      min(sharpes), max(sharpes)))

    rows.sort(key=lambda x: x[1], reverse=True)

    print(f"{'Health':<12} {'Avg Sh':>8} {'σ(Sh)':>7} {'Avg CAGR':>10} {'Avg MDD':>9} {'Min Sh':>8} {'Max Sh':>8}")
    print('=' * 72)
    for h_name, avg_sh, std_sh, avg_cagr, avg_mdd, min_sh, max_sh in rows:
        mark = ' ★' if h_name == rows[0][0] else ''
        print(f'{h_name:<12} {avg_sh:>8.3f} {std_sh:>7.3f} {avg_cagr:>+9.1%} {avg_mdd:>8.1%} {min_sh:>8.3f} {max_sh:>8.3f}{mark}')

    # Table 2: Per-day Sharpe
    top_h = [r[0] for r in rows]
    print(f"\n{'Day':>4}", end='')
    for h in top_h:
        print(f'  {h:>8}', end='')
    print()
    print('-' * (4 + 10 * len(top_h)))

    for day in DAYS:
        print(f'{day:>4}', end='')
        for h in top_h:
            m = per_day.get((h, day))
            if m:
                print(f'  {m["Sharpe"]:>8.3f}', end='')
            else:
                print(f'  {"N/A":>8}', end='')
        print()

    # Table 3: Winner per day
    print(f'\nWinner per anchor day:')
    none_wins = 0
    mom21_wins = 0
    for day in DAYS:
        best_h, best_sh = None, -999
        for h_name, _ in HEALTHS:
            m = per_day.get((h_name, day))
            if m and m['Sharpe'] > best_sh:
                best_sh = m['Sharpe']
                best_h = h_name
        none_m = per_day.get(('None', day))
        mom21_m = per_day.get(('Mom21', day))
        delta = (mom21_m['Sharpe'] - none_m['Sharpe']) if mom21_m and none_m else 0
        print(f'  Day {day:>2}: {best_h:<10} ({best_sh:.3f})  Mom21-None: {delta:+.3f}')
        if best_h == 'None': none_wins += 1
        if best_h == 'Mom21': mom21_wins += 1

    print(f'\nMom21 wins: {mom21_wins}/{len(DAYS)}, None wins: {none_wins}/{len(DAYS)}')


if __name__ == '__main__':
    main()
