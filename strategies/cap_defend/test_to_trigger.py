#!/usr/bin/env python3
"""Test turnover-triggered rebalancing for stock strategy.

Turnover = half-turnover = Σ|target_weight - current_weight| / 2

Rules:
- Monthly rebal: always fires on anchor day (never blocked)
- Canary flip (risk-on ↔ risk-off): always immediate (ignores cooldown)
- Signal turnover trigger: if half-turnover >= threshold → rebal
  - Cooldown only blocks SIGNAL rebals, not monthly or canary flip
"""

import sys, os, numpy as np, pandas as pd
from collections import defaultdict
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(__file__))
from test_stock_improve import (
    SP, load_prices, precompute, metrics, _init, ALL_TICKERS,
    resolve_canary, get_price, get_val, filter_healthy,
    select_offensive, select_defensive, check_crash, check_dd_exit
)


def calc_half_turnover(cur_weights, tgt_weights):
    """Half-turnover = Σ|target - current| / 2."""
    all_t = set(list(cur_weights.keys()) + list(tgt_weights.keys()))
    return sum(abs(tgt_weights.get(t, 0) - cur_weights.get(t, 0)) for t in all_t) / 2


def run_bt_to(prices_dict, ind, params, to_thresh=0.0, cool_days=0):
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
    signal_cool = 0

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

        if signal_cool > 0:
            signal_cool -= 1

        # === REBALANCE TRIGGERS ===
        is_rebal = False
        is_signal_rebal = False

        # 1. First day
        if is_first:
            is_rebal = True

        # 2. Monthly (never blocked)
        if not is_rebal and not rebalanced_this_month and date.day >= anchor:
            is_rebal = True

        # 3. Crash recovery
        if not is_rebal and crash_just_ended and not holdings:
            is_rebal = True

        # 4. DD exit
        if not is_rebal and dd_triggered:
            is_rebal = True

        # 5. Canary flip (ALWAYS, ignores cooldown)
        if (not is_rebal and not is_first and crash_cooldown <= 0
            and prev_trading_date is not None):
            sig_date = prev_trading_date
            daily_risk_on = resolve_canary(params, ind, sig_date, prev_risk_on)
            if prev_risk_on is not None and daily_risk_on != prev_risk_on:
                is_rebal = True  # canary flip = always immediate

        # 6. Turnover signal (respects cooldown)
        if (not is_rebal and to_thresh > 0 and not is_first
            and signal_cool <= 0 and crash_cooldown <= 0
            and prev_trading_date is not None and pv > 0):

            sig_date = prev_trading_date
            daily_risk_on = resolve_canary(params, ind, sig_date, prev_risk_on)

            if daily_risk_on:
                candidates = filter_healthy(params, ind, sig_date, params.offensive)
                if not candidates:
                    candidates = list(params.offensive)
                tgt_weights = select_offensive(params, ind, sig_date, candidates)
            else:
                tgt_weights = select_defensive(params, ind, sig_date)

            if tgt_weights:
                cur_weights = {}
                for t, shares in holdings.items():
                    p = get_price(ind, t, date)
                    if not np.isnan(p) and pv > 0:
                        cur_weights[t] = shares * p / pv

                ht = calc_half_turnover(cur_weights, tgt_weights)
                if ht >= to_thresh:
                    is_rebal = True
                    is_signal_rebal = True

        if crash_cooldown > 0:
            is_rebal = False

        # === EXECUTE ===
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

            if is_signal_rebal:
                signal_cool = cool_days

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
    name, p, to_th, cool = args
    from test_stock_improve import _g_prices, _g_ind
    df = run_bt_to(_g_prices, _g_ind, p, to_th, cool)
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

    # (name, turnover_threshold, cooldown_days)
    configs = [
        ('Monthly Only',     0.0,  0),
        # Pure turnover (no cooldown)
        ('TO≥20%',           0.20, 0),
        ('TO≥30%',           0.30, 0),
        ('TO≥40%',           0.40, 0),
        ('TO≥50%',           0.50, 0),
        ('TO≥60%',           0.60, 0),
        ('TO≥70%',           0.70, 0),
        # TO 30% + cooldowns
        ('TO30+C5d',         0.30, 5),
        ('TO30+C10d',        0.30, 10),
        ('TO30+C21d',        0.30, 21),
        # TO 40% + cooldowns
        ('TO40+C5d',         0.40, 5),
        ('TO40+C10d',        0.40, 10),
        ('TO40+C21d',        0.40, 21),
        # TO 50% + cooldowns
        ('TO50+C5d',         0.50, 5),
        ('TO50+C10d',        0.50, 10),
        ('TO50+C21d',        0.50, 21),
        # TO 60% + cooldowns
        ('TO60+C5d',         0.60, 5),
        ('TO60+C10d',        0.60, 10),
        ('TO60+C21d',        0.60, 21),
    ]

    all_tasks = []
    for name, to_th, cool in configs:
        for day in DAYS:
            p = mkp(_anchor=day)
            all_tasks.append((f'{name}:D{day:02d}', p, to_th, cool))

    print(f'Running {len(all_tasks)} configs with 24 workers...\n')

    with Pool(24, initializer=_init, initargs=(prices, ind)) as pool:
        results = pool.map(worker, all_tasks)

    by_config = defaultdict(list)
    for name_day, m in results:
        cfg_name = name_day.rsplit(':D', 1)[0]
        if m:
            by_config[cfg_name].append(m)

    print(f"{'Config':<18} {'Avg Sh':>8} {'σ(Sh)':>7} {'CAGR':>8} {'MDD':>8} {'Rebals':>7}  {'Δ Sharpe':>8}")
    print('=' * 73)

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
        print(f'{name:<18} {avg_sh:>8.3f} {std_sh:>7.3f} {avg_cagr:>+7.1%} {avg_mdd:>7.1%} {avg_reb:>7.0f}  {delta:>+8.3f}{mark}')


if __name__ == '__main__':
    main()
