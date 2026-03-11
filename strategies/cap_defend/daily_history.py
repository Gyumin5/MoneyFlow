#!/usr/bin/env python3
"""Generate daily portfolio history for V14 strategy (TR3+PFD5+DD+BL).
Output: CSV with date, total value, canary status, holdings detail."""

import os, sys, csv
sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    Params, load_data, get_universe_for_date, resolve_canary,
    get_healthy_coins, select_coins, compute_weights, apply_risk,
    should_rebalance, execute_rebalance, _port_val, get_price,
    check_coin_dd_exit
)


def B(**kw):
    base = dict(
        canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1, canary_band=1.0,
        health='HK', health_sma=0, health_mom_short=21,
        health_mom_long=90, vol_cap=0.05, top_n=40,
        risk='G5',
        dd_exit_lookback=60, dd_exit_threshold=-0.25,
        bl_threshold=-0.15, bl_days=7,
        start_date='2017-01-01',
    )
    base.update(kw)
    return Params(**base)


def run_with_history(prices, universe_map, params):
    """Run backtest and return daily history list."""
    btc = prices.get('BTC-USD')
    if btc is None:
        return []

    all_dates = btc.index[(btc.index >= params.start_date) &
                          (btc.index <= params.end_date)]
    if len(all_dates) == 0:
        return []

    holdings = {}
    cash = params.initial_capital
    state = {
        'prev_canary': False, 'canary_off_days': 0,
        'health_fail_streak': {}, 'prev_picks': [],
        'scaled_months': 2, 'month_start_value': params.initial_capital,
        'high_watermark': params.initial_capital,
        'crash_cooldown': 0, 'coin_cooldowns': {},
        'recent_port_vals': [], 'prev_month': None,
        'catastrophic_triggered': False, 'risk_force_rebal': False,
        'canary_on_date': None, 'post_flip_refreshed': False,
        'blacklist': {}, 'dd_exit_count': 0,
    }

    history = []
    from strategy_engine import _close_to

    for i, date in enumerate(all_dates):
        cur_month = date.strftime('%Y-%m')
        imc = (state['prev_month'] is not None and cur_month != state['prev_month'])

        pv = _port_val(holdings, cash, prices, date)
        state['current_port_val'] = pv
        state['high_watermark'] = max(state['high_watermark'], pv)
        state['recent_port_vals'].append(pv)
        if len(state['recent_port_vals']) > 60:
            state['recent_port_vals'] = state['recent_port_vals'][-60:]

        if imc:
            state['month_start_value'] = pv
            state['catastrophic_triggered'] = False
            if state['prev_canary']:
                state['scaled_months'] = state.get('scaled_months', 2) + 1

        # V14: Blacklist daily update
        if params.bl_threshold < 0:
            bl = state['blacklist']
            for t in list(bl.keys()):
                bl[t] -= 1
                if bl[t] <= 0:
                    del bl[t]
            for t in get_universe_for_date(universe_map, date):
                if t not in bl:
                    c = _close_to(t, prices, date)
                    if len(c) >= 2 and (c.iloc[-1] / c.iloc[-2] - 1) <= params.bl_threshold:
                        bl[t] = params.bl_days

        # V14: DD Exit daily check
        if params.dd_exit_lookback > 0 and holdings:
            dd_exits = [t for t in list(holdings.keys())
                        if check_coin_dd_exit(t, prices, date,
                                              params.dd_exit_lookback,
                                              params.dd_exit_threshold)]
            if dd_exits:
                for t in dd_exits:
                    p = get_price(t, prices, date)
                    units = holdings.pop(t, 0)
                    if units > 0:
                        cash += units * p * (1 - params.tx_cost)
                state['dd_exit_count'] += len(dd_exits)
                pv = _port_val(holdings, cash, prices, date)
                state['current_port_val'] = pv

        canary_on = resolve_canary(prices, date, params, state)
        canary_flipped = (canary_on != state['prev_canary'])

        if canary_on and canary_flipped:
            state['scaled_months'] = 0
            state['canary_on_date'] = date
            state['post_flip_refreshed'] = False
        elif not canary_on and canary_flipped:
            state['canary_on_date'] = None

        state['is_first_day'] = (i == 0)
        state['is_month_change'] = imc
        state['canary_flipped'] = canary_flipped
        state['canary_on'] = canary_on

        universe = get_universe_for_date(universe_map, date)
        if params.bl_threshold < 0:
            universe = [t for t in universe if t not in state['blacklist']]
        state['current_universe'] = universe

        if canary_on:
            healthy = get_healthy_coins(prices, universe, date, params, state)
            state['healthy_count'] = len(healthy)
            state['current_healthy_set'] = set(healthy)
        else:
            healthy = []
            state['healthy_count'] = 0
            state['current_healthy_set'] = set()

        picks = (select_coins(healthy, prices, date, params, state)
                 if canary_on and healthy else [])

        if picks:
            weights = compute_weights(picks, prices, date, params, state)
        else:
            weights = {'CASH': 1.0}

        weights = apply_risk(weights, prices, date, params, state)

        do_rebal = should_rebalance(weights, holdings, cash, prices, date, params, state)

        if not do_rebal and params.post_flip_delay > 0 and canary_on:
            flip_date = state.get('canary_on_date')
            if flip_date and not state.get('post_flip_refreshed', False):
                days_since = (date - flip_date).days
                if days_since >= params.post_flip_delay:
                    state['post_flip_refreshed'] = True
                    do_rebal = True

        if params.rebalancing in ('R2', 'R6', 'R7', 'R8', 'R9') and state.get('catastrophic_triggered'):
            weights = {'CASH': 1.0}
            picks = []

        if do_rebal:
            holdings, cash = execute_rebalance(holdings, cash, weights, prices,
                                               date, params.tx_cost)
            state['prev_picks'] = picks[:]

        # Record daily state
        pv = _port_val(holdings, cash, prices, date)
        coin_values = {}
        for t, units in holdings.items():
            p = get_price(t, prices, date)
            coin_values[t] = units * p

        history.append({
            'date': date,
            'port_value': pv,
            'cash': cash,
            'canary': canary_on,
            'rebalanced': do_rebal,
            'coin_values': coin_values,
            'n_healthy': state.get('healthy_count', 0),
        })

        state['prev_canary'] = canary_on
        state['prev_month'] = cur_month

    return history


def main():
    print("Loading data...")
    prices, universe = load_data(top_n=40)
    print(f"  {len(prices)} tickers loaded")

    # Run 3 tranches
    anchors = [1, 10, 19]
    tranche_histories = []

    for i, d in enumerate(anchors):
        print(f"  Running tranche {chr(65+i)} (anchor day {d})...")
        p = B(rebalancing=f'RX{d}', post_flip_delay=5, initial_capital=10000.0/3)
        h = run_with_history(prices, universe, p)
        tranche_histories.append(h)
        print(f"    {len(h)} days, final: ${h[-1]['port_value']:,.0f}")

    # Merge tranches by date
    print("\n  Merging tranches...")

    # Build date-indexed dicts
    date_map = {}  # date -> {tranche_idx: history_entry}
    all_dates = []
    for ti, hist in enumerate(tranche_histories):
        for entry in hist:
            d = entry['date']
            if d not in date_map:
                date_map[d] = {}
                all_dates.append(d)
            date_map[d][ti] = entry

    all_dates.sort()

    # Generate combined daily records
    output_path = os.path.join(os.path.dirname(__file__), 'daily_portfolio.csv')
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            'Date', 'Total_Value', 'Daily_Return',
            'Canary', 'N_Healthy', 'Rebalanced',
            'Cash_Pct', 'Holdings'
        ])

        prev_val = None
        for date in all_dates:
            entries = date_map[date]

            # Sum across tranches
            total_val = 0
            total_cash = 0
            combined_coins = {}
            canary = False
            n_healthy = 0
            rebalanced = False

            for ti, entry in entries.items():
                total_val += entry['port_value']
                total_cash += entry['cash']
                canary = entry['canary']  # same for all tranches
                n_healthy = entry['n_healthy']
                if entry['rebalanced']:
                    rebalanced = True
                for coin, val in entry['coin_values'].items():
                    combined_coins[coin] = combined_coins.get(coin, 0) + val

            # Calculate daily return
            if prev_val and prev_val > 0:
                daily_ret = total_val / prev_val - 1
            else:
                daily_ret = 0
            prev_val = total_val

            # Format holdings string
            cash_pct = total_cash / total_val * 100 if total_val > 0 else 100
            coin_items = sorted(combined_coins.items(), key=lambda x: x[1], reverse=True)
            holdings_parts = []
            for coin, val in coin_items:
                pct = val / total_val * 100 if total_val > 0 else 0
                if pct > 0.1:
                    holdings_parts.append(f"{coin}:{pct:.1f}%")

            holdings_str = ', '.join(holdings_parts) if holdings_parts else 'CASH:100%'

            writer.writerow([
                date.strftime('%Y-%m-%d'),
                f'{total_val:.0f}',
                f'{daily_ret:.4f}',
                'ON' if canary else 'OFF',
                n_healthy,
                'Y' if rebalanced else '',
                f'{cash_pct:.1f}',
                holdings_str,
            ])

    print(f"\n  Saved: {output_path}")
    print(f"  {len(all_dates)} days ({all_dates[0].date()} ~ {all_dates[-1].date()})")

    # Print summary stats
    print(f"\n{'=' * 80}")
    print(f"  V14 포트폴리오 요약 (TR3+PFD5+DD+BL)")
    print(f"{'=' * 80}")

    # Read back and show yearly summary
    import pandas as pd
    df = pd.read_csv(output_path, parse_dates=['Date'])
    df.set_index('Date', inplace=True)

    initial = df['Total_Value'].iloc[0]
    final = df['Total_Value'].iloc[-1]
    years = (df.index[-1] - df.index[0]).days / 365.25
    cagr = (final / initial) ** (1/years) - 1
    peak = df['Total_Value'].cummax()
    mdd = (df['Total_Value'] / peak - 1).min()
    daily_ret = df['Daily_Return'].dropna()
    sharpe = daily_ret.mean() / daily_ret.std() * (365**0.5) if daily_ret.std() > 0 else 0

    print(f"  기간: {df.index[0].date()} ~ {df.index[-1].date()} ({years:.1f}년)")
    print(f"  초기: ${initial:,.0f} → 최종: ${final:,.0f}")
    print(f"  CAGR: {cagr:+.1%}")
    print(f"  MDD: {mdd:.1%}")
    print(f"  Sharpe: {sharpe:.3f}")

    # Risk-On/Off days
    on_days = (df['Canary'] == 'ON').sum()
    off_days = (df['Canary'] == 'OFF').sum()
    rebal_days = (df['Rebalanced'] == 'Y').sum()
    print(f"\n  Risk-On: {on_days}일 ({on_days/(on_days+off_days)*100:.0f}%)")
    print(f"  Risk-Off: {off_days}일 ({off_days/(on_days+off_days)*100:.0f}%)")
    print(f"  리밸런싱: {rebal_days}회")

    # Yearly
    print(f"\n  연도별:")
    print(f"  {'Year':>6} {'Start':>10} {'End':>10} {'Return':>8} {'MDD':>7}")
    print(f"  {'─' * 50}")
    for y in range(df.index[0].year, df.index[-1].year + 1):
        mask = df.index.year == y
        if mask.sum() < 10:
            continue
        ydf = df[mask]
        ystart = ydf['Total_Value'].iloc[0]
        yend = ydf['Total_Value'].iloc[-1]
        yret = yend / ystart - 1
        ypeak = ydf['Total_Value'].cummax()
        ymdd = (ydf['Total_Value'] / ypeak - 1).min()
        print(f"  {y:>6} ${ystart:>9,.0f} ${yend:>9,.0f} {yret:>+7.1%} {ymdd:>6.1%}")

    # Show last 10 days
    print(f"\n  최근 10일:")
    print(f"  {'Date':>12} {'Value':>10} {'Ret':>7} {'Canary':>7} {'Holdings'}")
    print(f"  {'─' * 75}")
    for _, row in df.tail(10).iterrows():
        print(f"  {_.strftime('%Y-%m-%d'):>12} ${row['Total_Value']:>9,.0f}"
              f" {row['Daily_Return']:>+6.2%} {row['Canary']:>7} {row['Holdings'][:50]}")


if __name__ == '__main__':
    main()
