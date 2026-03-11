#!/usr/bin/env python3
"""Test drift-triggered rebalancing for stock strategy.

Questions:
1. Does drift-triggered intra-month rebalancing help?
2. Does adding a cooldown after drift rebal reduce whipsaw?
3. What drift threshold is optimal?
"""

import sys, os, numpy as np, pandas as pd
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from test_stock_improve import (
    SP, load_prices, precompute, run_bt, metrics, _init, ALL_TICKERS,
    resolve_canary, get_price, get_val, filter_healthy,
    select_offensive, select_defensive, check_crash, check_dd_exit
)


def run_bt_drift(prices_dict, ind, params, drift_thresh=0.0, drift_cool=0):
    """run_bt clone with drift-triggered rebalancing added."""
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
    target_weights = {}
    drift_cool_remaining = 0

    for date in dates:
        cur_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and cur_month != prev_month)
        is_first = (prev_month is None)

        if is_month_change:
            rebalanced_this_month = False

        # Portfolio value
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

        if drift_cool_remaining > 0:
            drift_cool_remaining -= 1

        # Rebalance trigger
        is_rebal = False
        drift_rebal = False
        if is_first:
            is_rebal = True
        elif not rebalanced_this_month and date.day >= anchor:
            is_rebal = True
        if crash_just_ended and not holdings:
            is_rebal = True
        if dd_triggered:
            is_rebal = True

        # *** DRIFT CHECK ***
        if (not is_rebal and drift_thresh > 0 and target_weights
            and pv > 0 and drift_cool_remaining <= 0 and crash_cooldown <= 0):
            cur_weights = {}
            for t, shares in holdings.items():
                p = get_price(ind, t, date)
                if not np.isnan(p):
                    cur_weights[t] = shares * p / pv

            all_tickers_set = set(list(target_weights.keys()) + list(cur_weights.keys()))
            max_dev = 0.0
            for t in all_tickers_set:
                dev = abs(cur_weights.get(t, 0.0) - target_weights.get(t, 0.0))
                if dev > max_dev:
                    max_dev = dev

            if max_dev >= drift_thresh:
                is_rebal = True
                drift_rebal = True

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

            # Determine target portfolio
            if risk_on:
                candidates = filter_healthy(params, ind, sig_date, params.offensive)
                if not candidates:
                    candidates = list(params.offensive)
                weights = select_offensive(params, ind, sig_date, candidates)
            else:
                weights = select_defensive(params, ind, sig_date)

            # Execute: sell all, buy new
            for t in list(holdings.keys()):
                p = get_price(ind, t, date)
                shares = holdings.pop(t, 0)
                if shares > 0 and not np.isnan(p):
                    cash += shares * p * (1 - params.tx_cost)

            holdings = {}
            target_weights = {}
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
                    target_weights[t] = w

            if drift_rebal:
                drift_cool_remaining = drift_cool

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

    df = pd.DataFrame(history).set_index('Date')
    df.attrs['rebal_count'] = rebal_count
    df.attrs['flip_count'] = flip_count
    return df


def worker(args):
    name, p, drift_thresh, drift_cool = args
    from test_stock_improve import _g_prices, _g_ind
    df = run_bt_drift(_g_prices, _g_ind, p, drift_thresh, drift_cool)
    m = metrics(df) if df is not None else None
    return name, m


def main():
    print('Loading prices...')
    prices = load_prices(ALL_TICKERS)
    print(f'  {len(prices)} tickers loaded')

    print('Precomputing indicators...')
    ind = precompute(prices)
    print('  Done')

    R8 = ('SPY','QQQ','VGK','EWJ','EEM','VWO','GLD','PDBC')
    DEF = ('IEF','BIL','BNDX','GLD','PDBC')

    def mkp(**kw):
        d = dict(offensive=R8, defensive=DEF,
                 canary_assets=('EEM',), canary_hyst=0.005,
                 health='none', defense='top3', mom_style='12m', weight='ew')
        d.update(kw)
        return SP(**d)

    DAYS = [1, 2, 3, 5, 8, 10, 15, 20]

    configs = [
        # Baseline
        ('Monthly Only',    0.0,  0),
        # Pure drift thresholds
        ('Drift 5%',        0.05, 0),
        ('Drift 8%',        0.08, 0),
        ('Drift 10%',       0.10, 0),
        ('Drift 15%',       0.15, 0),
        ('Drift 20%',       0.20, 0),
        ('Drift 25%',       0.25, 0),
        # Drift 10% + cooldowns
        ('D10%+Cool5d',     0.10, 5),
        ('D10%+Cool10d',    0.10, 10),
        ('D10%+Cool21d',    0.10, 21),
        # Drift 15% + cooldowns
        ('D15%+Cool5d',     0.15, 5),
        ('D15%+Cool10d',    0.15, 10),
        ('D15%+Cool21d',    0.15, 21),
        # Drift 20% + cooldowns
        ('D20%+Cool5d',     0.20, 5),
        ('D20%+Cool10d',    0.20, 10),
        ('D20%+Cool21d',    0.20, 21),
    ]

    all_tasks = []
    for name, dt, dc in configs:
        for day in DAYS:
            p = mkp(_anchor=day)
            all_tasks.append((f'{name}:D{day:02d}', p, dt, dc))

    print(f'Running {len(all_tasks)} configs with 24 workers...\n')

    with Pool(24, initializer=_init, initargs=(prices, ind)) as pool:
        results = pool.map(worker, all_tasks)

    # Aggregate
    by_config = defaultdict(list)
    for name_day, m in results:
        cfg_name = name_day.rsplit(':D', 1)[0]
        if m:
            by_config[cfg_name].append(m)

    print(f"{'Config':<20} {'Avg Sh':>8} {'σ(Sh)':>7} {'CAGR':>8} {'MDD':>8} {'Rebals':>7}  {'Δ Sharpe':>8}")
    print('=' * 75)

    rows = []
    for name, _, _ in configs:
        ms = by_config[name]
        if not ms: continue
        sharpes = [m['Sharpe'] for m in ms]
        cagrs = [m['CAGR'] for m in ms]
        mdds = [m['MDD'] for m in ms]
        rebals = [m.get('Rebals', 0) for m in ms]
        rows.append((name, np.mean(sharpes), np.std(sharpes),
                      np.mean(cagrs), np.mean(mdds), np.mean(rebals)))

    baseline_sh = rows[0][1] if rows else 0
    for name, avg_sh, std_sh, avg_cagr, avg_mdd, avg_reb in rows:
        delta = avg_sh - baseline_sh
        mark = ' ★' if delta > 0.02 else (' ●' if delta > 0.005 else '')
        print(f'{name:<20} {avg_sh:>8.3f} {std_sh:>7.3f} {avg_cagr:>+7.1%} {avg_mdd:>7.1%} {avg_reb:>7.0f}  {delta:>+8.3f}{mark}')


if __name__ == '__main__':
    main()
