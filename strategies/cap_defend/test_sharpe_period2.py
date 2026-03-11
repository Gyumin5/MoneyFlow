#!/usr/bin/env python3
"""Test shorter Sharpe lookback periods for stock strategy.
Sh63 > Sh126 > Sh252 confirmed. Now test even shorter: 21, 42, 63."""

import sys, os, numpy as np
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from test_stock_improve import (
    SP, load_prices, precompute, metrics, _init, ALL_TICKERS,
    resolve_canary, get_price, filter_healthy,
    select_offensive, select_defensive, check_crash, check_dd_exit
)
from test_sharpe_period import run_bt_flip


def worker(args):
    name, day, p = args
    from test_stock_improve import _g_prices, _g_ind
    df = run_bt_flip(_g_prices, _g_ind, p)
    m = metrics(df) if df is not None else None
    return name, day, m


def main():
    print('Loading prices...')
    prices = load_prices(ALL_TICKERS)
    print(f'  {len(prices)} tickers loaded')

    print('Precomputing indicators...')
    ind = precompute(prices)
    print('  Done')

    DAYS = [1, 2, 3, 4, 5, 8, 10, 12, 15, 18, 20]
    R8 = ('SPY','QQQ','VGK','EWJ','EEM','VWO','GLD','PDBC')
    DEF = ('IEF','BIL','BNDX','GLD','PDBC')

    # Need to add sharpe21/sharpe42 to precompute
    # Patch: add missing sharpe columns
    for t, df in ind.items():
        if 'ret' not in df.columns:
            continue
        for lb in (21, 42):
            col = f'sharpe{lb}'
            if col not in df.columns:
                rlb = df['ret'].rolling(lb)
                df[col] = rlb.mean() / rlb.std() * np.sqrt(252)
                # Also need sortino
                neg_std = df['ret'].clip(upper=0).rolling(lb).std()
                df[f'sortino{lb}'] = rlb.mean() / neg_std * np.sqrt(252)

    # Re-init with patched ind
    print('Patched sharpe21/42 into indicators')

    CONFIGS = [
        ('12M+Sh21',   21),
        ('12M+Sh42',   42),
        ('12M+Sh63',   63),
        ('12M+Sh126',  126),
        ('12M+Sh252',  252),
    ]

    all_tasks = []
    for name, sh_lb in CONFIGS:
        for day in DAYS:
            p = SP(offensive=R8, defensive=DEF,
                   canary_assets=('EEM',), canary_hyst=0.005,
                   health='none', defense='top3',
                   mom_style='12m', weight='ew',
                   sharpe_lookback=sh_lb, select='mom3_sh3',
                   _anchor=day)
            all_tasks.append((name, day, p))

    print(f'Running {len(all_tasks)} configs...\n')

    with Pool(24, initializer=_init, initargs=(prices, ind)) as pool:
        results = pool.map(worker, all_tasks)

    by_config = defaultdict(list)
    for name, day, m in results:
        if m:
            by_config[name].append(m)

    print(f"{'Config':<15} {'Avg Sh':>8} {'σ(Sh)':>7} {'CAGR':>8} {'MDD':>8} {'Min Sh':>8} {'Max Sh':>8}")
    print('=' * 70)

    best_sh = -999
    rows = []
    for name, sh_lb in CONFIGS:
        ms = by_config[name]
        if not ms: continue
        sharpes = [m['Sharpe'] for m in ms]
        cagrs = [m['CAGR'] for m in ms]
        mdds = [m['MDD'] for m in ms]
        avg = np.mean(sharpes)
        if avg > best_sh:
            best_sh = avg
        rows.append((name, avg, np.std(sharpes),
                      np.mean(cagrs), np.mean(mdds),
                      min(sharpes), max(sharpes)))

    for name, avg_sh, std_sh, avg_cagr, avg_mdd, min_sh, max_sh in rows:
        mark = ' ★' if avg_sh == best_sh else ''
        print(f'{name:<15} {avg_sh:>8.3f} {std_sh:>7.3f} {avg_cagr:>+7.1%} {avg_mdd:>7.1%} {min_sh:>8.3f} {max_sh:>8.3f}{mark}')


if __name__ == '__main__':
    main()
