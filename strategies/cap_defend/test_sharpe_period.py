#!/usr/bin/env python3
"""Test Sharpe lookback period for stock strategy.

Current V14: Mom 12M (252d) + Sharpe 126d.
Question: Was Sharpe 126d validated vs 63d, 252d?
"""

import sys, os, numpy as np
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from test_stock_improve import (
    SP, load_prices, precompute, metrics, _init, ALL_TICKERS,
    resolve_canary, get_price, get_val, filter_healthy,
    select_offensive, select_defensive, check_crash, check_dd_exit
)


def run_bt_flip(prices_dict, ind, params):
    """run_bt with canary flip immediate rebalancing."""
    spy = ind.get('SPY')
    if spy is None:
        return None

    dates = spy.index[(spy.index >= params.start) & (spy.index <= params.end)]
    if len(dates) < 2:
        return None

    anchor = params._anchor
    holdings = {}
    cash = params.capital
    prev_month = None
    prev_risk_on = None
    history = []
    rebal_count = 0
    crash_cooldown = 0
    rebalanced_this_month = False
    prev_trading_date = None

    for date in dates:
        cur_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and cur_month != prev_month)
        is_first = (prev_month is None)

        if is_month_change:
            rebalanced_this_month = False

        pv = cash
        for t, shares in holdings.items():
            p = get_price(ind, t, date)
            if not np.isnan(p):
                pv += shares * p

        # Crash breaker
        crash_just_ended = False
        if crash_cooldown > 0:
            crash_cooldown -= 1
            if crash_cooldown == 0:
                if check_crash(params, ind, date):
                    crash_cooldown = params.crash_cool
                else:
                    crash_just_ended = True
        elif check_crash(params, ind, date):
            for t in list(holdings.keys()):
                if t in ('IEF','BIL','BNDX','GLD','PDBC','TLT','SHY','AGG','TIP','LQD'):
                    continue
                p = get_price(ind, t, date)
                shares = holdings.pop(t, 0)
                if shares > 0 and not np.isnan(p):
                    cash += shares * p * (1 - params.tx_cost)
            crash_cooldown = params.crash_cool
            pv = cash
            for t, shares in holdings.items():
                p = get_price(ind, t, date)
                if not np.isnan(p):
                    pv += shares * p

        # DD exit
        dd_triggered = False
        if crash_cooldown <= 0:
            dd_exits = check_dd_exit(params, ind, date, holdings)
            if dd_exits:
                dd_triggered = True
            for t in dd_exits:
                p = get_price(ind, t, date)
                shares = holdings.pop(t, 0)
                if shares > 0 and not np.isnan(p):
                    cash += shares * p * (1 - params.tx_cost)

        is_rebal = False

        if is_first:
            is_rebal = True
        elif not rebalanced_this_month and date.day >= anchor:
            is_rebal = True
        if crash_just_ended and not holdings:
            is_rebal = True
        if dd_triggered:
            is_rebal = True

        # Canary flip: always immediate
        if (not is_rebal and not is_first and crash_cooldown <= 0
            and prev_trading_date is not None):
            sig_date = prev_trading_date
            daily_risk_on = resolve_canary(params, ind, sig_date, prev_risk_on)
            if prev_risk_on is not None and daily_risk_on != prev_risk_on:
                is_rebal = True

        if crash_cooldown > 0:
            is_rebal = False

        if is_rebal:
            rebalanced_this_month = True
            rebal_count += 1

            pv = cash
            for t, shares in holdings.items():
                p = get_price(ind, t, date)
                if not np.isnan(p):
                    pv += shares * p

            sig_date = prev_trading_date if prev_trading_date is not None else date
            risk_on = resolve_canary(params, ind, sig_date, prev_risk_on)
            prev_risk_on = risk_on

            if risk_on:
                candidates = filter_healthy(params, ind, sig_date, params.offensive)
                if not candidates:
                    candidates = list(params.offensive)
                weights = select_offensive(params, ind, sig_date, candidates)
            else:
                weights = select_defensive(params, ind, sig_date)

            for t in list(holdings.keys()):
                p = get_price(ind, t, date)
                shares = holdings.pop(t, 0)
                if shares > 0 and not np.isnan(p):
                    cash += shares * p * (1 - params.tx_cost)

            holdings = {}
            if weights:
                invest = cash
                for t, w in weights.items():
                    p = get_price(ind, t, date)
                    if np.isnan(p) or p <= 0:
                        continue
                    alloc = invest * w
                    shares = alloc / p
                    cost = alloc * params.tx_cost
                    holdings[t] = shares
                    cash -= (alloc + cost)

            pv = cash
            for t, shares in holdings.items():
                p = get_price(ind, t, date)
                if not np.isnan(p):
                    pv += shares * p

        history.append({'Date': date, 'Value': pv})
        prev_month = cur_month
        prev_trading_date = date

    if not history:
        return None

    import pandas as pd
    df = pd.DataFrame(history).set_index('Date')
    df.attrs['rebal_count'] = rebal_count
    return df


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

    # (name, sharpe_lookback, mom_style, select)
    CONFIGS = [
        # Sharpe period sweep (with 12M momentum)
        ('12M+Sh63',   63,  '12m', 'mom3_sh3'),
        ('12M+Sh126',  126, '12m', 'mom3_sh3'),
        ('12M+Sh252',  252, '12m', 'mom3_sh3'),
        # Sharpe period sweep (with lh momentum 20/30/50)
        ('lh+Sh63',    63,  'lh',  'mom3_sh3'),
        ('lh+Sh126',   126, 'lh',  'mom3_sh3'),
        ('lh+Sh252',   252, 'lh',  'mom3_sh3'),
        # Mom-only (no Sharpe) for comparison
        ('12M+Mom3',   126, '12m', 'mom3'),
        ('12M+Mom5',   126, '12m', 'mom5'),
        # Sharpe-only for comparison
        ('Sh63 only',  63,  '12m', 'sh3'),
        ('Sh126 only', 126, '12m', 'sh3'),
        ('Sh252 only', 252, '12m', 'sh3'),
    ]

    all_tasks = []
    for name, sh_lb, mom_st, sel in CONFIGS:
        for day in DAYS:
            p = SP(offensive=R8, defensive=DEF,
                   canary_assets=('EEM',), canary_hyst=0.005,
                   health='none', defense='top3',
                   mom_style=mom_st, weight='ew',
                   sharpe_lookback=sh_lb, select=sel,
                   _anchor=day)
            all_tasks.append((name, day, p))

    print(f'Running {len(all_tasks)} configs (canary flip ON)...\n')

    with Pool(24, initializer=_init, initargs=(prices, ind)) as pool:
        results = pool.map(worker, all_tasks)

    by_config = defaultdict(list)
    per_day = {}
    for name, day, m in results:
        if m:
            by_config[name].append(m)
            per_day[(name, day)] = m

    print(f"{'Config':<15} {'Avg Sh':>8} {'σ(Sh)':>7} {'Avg CAGR':>10} {'Avg MDD':>9} {'Min Sh':>8} {'Max Sh':>8}")
    print('=' * 74)

    rows = []
    for name, _, _, _ in CONFIGS:
        ms = by_config[name]
        if not ms: continue
        sharpes = [m['Sharpe'] for m in ms]
        cagrs = [m['CAGR'] for m in ms]
        mdds = [m['MDD'] for m in ms]
        rows.append((name, np.mean(sharpes), np.std(sharpes),
                      np.mean(cagrs), np.mean(mdds),
                      min(sharpes), max(sharpes)))

    best_sh = max(r[1] for r in rows) if rows else 0
    for name, avg_sh, std_sh, avg_cagr, avg_mdd, min_sh, max_sh in rows:
        mark = ' ★' if avg_sh == best_sh else ''
        print(f'{name:<15} {avg_sh:>8.3f} {std_sh:>7.3f} {avg_cagr:>+9.1%} {avg_mdd:>8.1%} {min_sh:>8.3f} {max_sh:>8.3f}{mark}')

    # Grouped summary
    print('\n--- Sharpe Period Summary (12M Mom) ---')
    for lb in [63, 126, 252]:
        name = f'12M+Sh{lb}'
        ms = by_config[name]
        if ms:
            sharpes = [m['Sharpe'] for m in ms]
            print(f'  Sharpe {lb:>3}d: Avg {np.mean(sharpes):.3f}, σ {np.std(sharpes):.3f}')

    print('\n--- Mom-only vs Sharpe-only vs Union ---')
    for name in ['12M+Mom3', '12M+Sh126', '12M+Sh126 only', '12M+Sh252 only', '12M+Sh63 only', '12M+Sh126']:
        ms = by_config.get(name, [])
        if ms:
            sharpes = [m['Sharpe'] for m in ms]
            print(f'  {name:<15}: Avg {np.mean(sharpes):.3f}')


if __name__ == '__main__':
    main()
