#!/usr/bin/env python3
"""Re-test Health filter with canary flip enabled.
Verifies if Health=None conclusion holds when canary flip is ON."""

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
    flip_count = 0
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
            if prev_risk_on is not None and risk_on != prev_risk_on:
                flip_count += 1
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
    df.attrs['flip_count'] = flip_count
    return df


def worker(args):
    h_name, day, p = args
    from test_stock_improve import _g_prices, _g_ind
    df = run_bt_flip(_g_prices, _g_ind, p)
    m = metrics(df) if df is not None else None
    return h_name, day, m


def main():
    print('Loading prices...')
    prices = load_prices(ALL_TICKERS)
    print(f'  {len(prices)} tickers loaded')

    print('Precomputing indicators...')
    ind = precompute(prices)
    print('  Done')

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

    all_tasks = []
    for h_name, h_val in HEALTHS:
        for day in DAYS:
            p = SP(offensive=R8, defensive=DEF,
                   canary_assets=('EEM',), canary_hyst=0.005,
                   health=h_val, defense='top3', mom_style='12m', weight='ew',
                   _anchor=day)
            all_tasks.append((h_name, day, p))

    print(f'Running {len(all_tasks)} configs (canary flip ON)...\n')

    with Pool(24, initializer=_init, initargs=(prices, ind)) as pool:
        results = pool.map(worker, all_tasks)

    by_health = defaultdict(list)
    per_day = {}
    for h_name, day, m in results:
        if m:
            by_health[h_name].append(m)
            per_day[(h_name, day)] = m

    # Table 1: Comparison with flip OFF (previous results)
    print("WITH canary flip immediate rebalancing:")
    print(f"{'Health':<12} {'Avg Sh':>8} {'σ(Sh)':>7} {'Avg CAGR':>10} {'Avg MDD':>9} {'Min Sh':>8} {'Max Sh':>8}")
    print('=' * 72)

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
    for h_name, avg_sh, std_sh, avg_cagr, avg_mdd, min_sh, max_sh in rows:
        mark = ' ★' if h_name == rows[0][0] else ''
        print(f'{h_name:<12} {avg_sh:>8.3f} {std_sh:>7.3f} {avg_cagr:>+9.1%} {avg_mdd:>8.1%} {min_sh:>8.3f} {max_sh:>8.3f}{mark}')

    # Table 2: Per-day
    top_h = [r[0] for r in rows]
    print(f"\n{'Day':>4}", end='')
    for h in top_h: print(f'  {h:>8}', end='')
    print()
    print('-' * (4 + 10 * len(top_h)))

    for day in DAYS:
        print(f'{day:>4}', end='')
        for h in top_h:
            m = per_day.get((h, day))
            if m: print(f'  {m["Sharpe"]:>8.3f}', end='')
            else: print(f'  {"N/A":>8}', end='')
        print()

    # Winner per day
    print(f'\nWinner per anchor day:')
    wins = defaultdict(int)
    for day in DAYS:
        best_h, best_sh = None, -999
        for h_name, _ in HEALTHS:
            m = per_day.get((h_name, day))
            if m and m['Sharpe'] > best_sh:
                best_sh = m['Sharpe']
                best_h = h_name
        none_m = per_day.get(('None', day))
        none_sh = none_m['Sharpe'] if none_m else 0
        delta = best_sh - none_sh
        print(f'  Day {day:>2}: {best_h:<10} ({best_sh:.3f})  vs None: {delta:+.3f}')
        wins[best_h] += 1

    print(f'\nWins: ', end='')
    for h, cnt in sorted(wins.items(), key=lambda x: -x[1]):
        print(f'{h}={cnt}  ', end='')
    print()


if __name__ == '__main__':
    main()
