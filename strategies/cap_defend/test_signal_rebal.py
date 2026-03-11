#!/usr/bin/env python3
"""Test signal-change-triggered rebalancing for stock strategy.

Instead of weight drift, check if the SIGNAL (which ETFs to hold) has changed.
If the "ideal portfolio" differs from current holdings, trigger rebalance.

Variants:
1. Monthly only (baseline)
2. Daily signal check → immediate rebal if picks changed
3. Signal change + cooldown (don't rebal again for N days)
4. Signal change + threshold (only rebal if >= K picks changed)
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


def run_bt_signal(prices_dict, ind, params,
                  daily_signal=False, cool_days=0, min_changes=1):
    """run_bt with signal-change-triggered rebalancing.

    daily_signal: if True, compute ideal picks daily
    cool_days: minimum gap between signal-triggered rebals
    min_changes: minimum number of picks that must differ to trigger
    """
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
    current_picks = set()  # what we currently hold
    signal_cool = 0

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

        if signal_cool > 0:
            signal_cool -= 1

        # Rebalance trigger
        is_rebal = False
        if is_first:
            is_rebal = True
        elif not rebalanced_this_month and date.day >= anchor:
            is_rebal = True
        if crash_just_ended and not holdings:
            is_rebal = True
        if dd_triggered:
            is_rebal = True

        # *** DAILY SIGNAL CHECK ***
        if (daily_signal and not is_rebal and not is_first
            and signal_cool <= 0 and crash_cooldown <= 0
            and prev_trading_date is not None):

            sig_date = prev_trading_date
            daily_risk_on = resolve_canary(params, ind, sig_date, prev_risk_on)

            if daily_risk_on:
                candidates = filter_healthy(params, ind, sig_date, params.offensive)
                if not candidates:
                    candidates = list(params.offensive)
                weights = select_offensive(params, ind, sig_date, candidates)
                ideal_picks = set(weights.keys())
            else:
                weights = select_defensive(params, ind, sig_date)
                ideal_picks = set(weights.keys())

            # Check canary flip
            if prev_risk_on is not None and daily_risk_on != prev_risk_on:
                # Canary flipped → definitely rebal
                is_rebal = True
            else:
                # Same regime: check if picks changed
                n_diff = len(ideal_picks.symmetric_difference(current_picks))
                if n_diff >= min_changes:
                    is_rebal = True

            if is_rebal:
                signal_cool = cool_days

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

            # Sell all
            for t in list(holdings.keys()):
                p = get_price(ind, t, date)
                shares = holdings.pop(t, 0)
                if shares > 0 and not np.isnan(p):
                    cash += shares * p * (1 - params.tx_cost)

            # Buy new
            holdings = {}
            current_picks = set()
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
                    current_picks.add(t)

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
    name, p, daily, cool, min_ch = args
    from test_stock_improve import _g_prices, _g_ind
    df = run_bt_signal(_g_prices, _g_ind, p, daily, cool, min_ch)
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

    # (name, daily_signal, cool_days, min_changes)
    configs = [
        ('Monthly Only',       False, 0,  1),
        # Daily signal, no cooldown
        ('Daily ≥1 change',    True,  0,  1),
        ('Daily ≥2 changes',   True,  0,  2),
        ('Daily ≥3 changes',   True,  0,  3),
        # Daily signal + cooldown
        ('≥1ch + Cool5d',      True,  5,  1),
        ('≥1ch + Cool10d',     True,  10, 1),
        ('≥1ch + Cool21d',     True,  21, 1),
        ('≥2ch + Cool5d',      True,  5,  2),
        ('≥2ch + Cool10d',     True,  10, 2),
        ('≥2ch + Cool21d',     True,  21, 2),
        ('≥3ch + Cool5d',      True,  5,  3),
        ('≥3ch + Cool10d',     True,  10, 3),
        ('≥3ch + Cool21d',     True,  21, 3),
    ]

    all_tasks = []
    for name, daily, cool, min_ch in configs:
        for day in DAYS:
            p = mkp(_anchor=day)
            all_tasks.append((f'{name}:D{day:02d}', p, daily, cool, min_ch))

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
    for name, _, _, _ in configs:
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
