#!/usr/bin/env python3
"""
Health Check Margin (Hysteresis) Test

Instead of binary health thresholds, use separate entry/exit thresholds:
  - Mom21: entry > +margin, exit < -margin (dead zone around 0)
  - SMA30: entry > SMA30 * (1+margin), exit < SMA30 * (1-margin)
  - Vol90: unchanged (already very stable at 0.28% flip rate)

If a coin is currently held and in the dead zone, keep holding.
If a coin is not held and in the dead zone, don't enter.

Test grid:
  Margins: 0% (current), 1%, 2%, 3%, 5%
  Rebal:   BASE (monthly), TO30, TO50

All use: SMA150 daily canary + Vol5% health + market cap order Top5 + equal weight
"""

import os, sys, warnings
from collections import defaultdict
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from backtest_coin_strategy import (
    load_universe, load_all_prices, calc_metrics,
    calc_ret, get_volatility, STABLECOINS
)

EXCLUDE_SYMBOLS = STABLECOINS | {'PAXG', 'XAUT', 'WBTC', 'USD1', 'USDE'}
TOP_N = 50
START_DATE = '2018-01-01'
END_DATE = '2025-12-31'
SMA_PERIOD = 150
VOL_CAP = 0.05
N_PICKS = 5
TX_COST = 0.004


def filter_universe_topn(universe_map, top_n=TOP_N):
    filtered = {}
    for month_key, tickers in universe_map.items():
        clean = [t for t in tickers if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]
        filtered[month_key] = clean[:top_n]
    return filtered


def get_universe_for_date(universe_map, date):
    month_key = date.strftime('%Y-%m') + '-01'
    for mk in sorted(universe_map.keys(), reverse=True):
        if mk <= month_key:
            return universe_map[mk]
    return []


def check_canary(prices, global_idx):
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < SMA_PERIOD: return False
    return close.iloc[-1] > close.rolling(SMA_PERIOD).mean().iloc[-1]


def check_health_with_margin(ticker, prices, global_idx, margin, is_currently_held):
    """
    Health check with hysteresis margin.

    margin: float (e.g. 0.02 for 2%)
    is_currently_held: whether this coin is in current portfolio

    For currently held coins (exit threshold - harder to exit):
      - Mom21 > -margin (e.g. > -2%)
      - Price > SMA30 * (1 - margin)
      - Vol90 <= VOL_CAP (unchanged)

    For non-held coins (entry threshold - harder to enter):
      - Mom21 > +margin (e.g. > +2%)
      - Price > SMA30 * (1 + margin)
      - Vol90 <= VOL_CAP (unchanged)
    """
    if ticker not in prices:
        return False
    close = prices[ticker]['Close'].iloc[:global_idx+1]
    if len(close) < 90:
        return False

    cur = close.iloc[-1]
    sma30 = close.rolling(30).mean().iloc[-1]
    mom21 = calc_ret(close, 21)
    vol90 = get_volatility(close, 90)

    if vol90 is None or vol90 > VOL_CAP:
        return False
    if mom21 is None:
        return False

    if is_currently_held:
        # Exit threshold: more lenient (harder to kick out)
        mom_ok = mom21 > -margin
        sma_ok = cur > sma30 * (1 - margin)
    else:
        # Entry threshold: more strict (harder to get in)
        mom_ok = mom21 > margin
        sma_ok = cur > sma30 * (1 + margin)

    return mom_ok and sma_ok


def get_healthy_picks_with_margin(prices, universe_map, date, global_idx,
                                   margin, current_holdings):
    """Get top N healthy picks with margin-based hysteresis."""
    uni_tickers = get_universe_for_date(universe_map, date)
    uni_clean = [t.replace('-USD', '') for t in uni_tickers
                if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]

    held_set = set(current_holdings.keys())
    healthy = []
    for sym in uni_clean:
        ticker = f"{sym}-USD"
        is_held = ticker in held_set
        if check_health_with_margin(ticker, prices, global_idx, margin, is_held):
            healthy.append(ticker)
    return healthy[:N_PICKS]


def compute_target(picks):
    if not picks:
        return {'CASH': 1.0}
    return {t: 1.0/len(picks) for t in picks}


def get_price(ticker, prices, date):
    if ticker not in prices: return 0
    idx = prices[ticker].index.get_indexer([date], method='ffill')[0]
    return prices[ticker]['Close'].iloc[idx] if idx >= 0 else 0


def calc_turnover(holdings, cash, target_weights, prices, date):
    port_val = cash
    current_values = {}
    for t, units in holdings.items():
        p = get_price(t, prices, date)
        v = units * p
        current_values[t] = v
        port_val += v

    if port_val <= 0:
        return 1.0

    current_weights = {t: v / port_val for t, v in current_values.items()}
    cash_weight = cash / port_val

    target_cash = target_weights.get('CASH', 0)
    target_coins = {t: w for t, w in target_weights.items() if t != 'CASH'}

    all_tickers = set(list(current_weights.keys()) + list(target_coins.keys()))
    turnover = 0
    for t in all_tickers:
        cw = current_weights.get(t, 0)
        tw = target_coins.get(t, 0)
        turnover += abs(cw - tw)
    turnover += abs(cash_weight - target_cash)
    return turnover / 2


def execute_rebalance(holdings, cash, weights, prices, date):
    current_values = {}
    port_val = cash
    for t, units in holdings.items():
        p = get_price(t, prices, date)
        v = units * p
        current_values[t] = v
        port_val += v

    if 'CASH' in weights and weights['CASH'] == 1.0:
        sell_total = sum(current_values.values())
        return {}, cash + sell_total * (1 - TX_COST)

    target_values = {t: port_val * w for t, w in weights.items() if t != 'CASH'}
    new_holdings = {}
    new_cash = cash
    all_tickers = set(list(current_values.keys()) + list(target_values.keys()))

    for t in all_tickers:
        cur_val = current_values.get(t, 0)
        tgt_val = target_values.get(t, 0)
        p = get_price(t, prices, date)
        if p <= 0: continue
        if tgt_val >= cur_val:
            if cur_val > 0:
                new_holdings[t] = holdings[t]
        else:
            sell_amount = cur_val - tgt_val
            new_cash += sell_amount * (1 - TX_COST)
            if tgt_val > 0:
                new_holdings[t] = tgt_val / p

    buys = {}
    for t in all_tickers:
        cur_val = current_values.get(t, 0)
        tgt_val = target_values.get(t, 0)
        if tgt_val > cur_val:
            buys[t] = tgt_val - cur_val

    total_buy = sum(buys.values())
    if total_buy > 0:
        scale = min(1.0, new_cash / total_buy)
        for t, buy_val in buys.items():
            p = get_price(t, prices, date)
            if p <= 0: continue
            actual_spend = buy_val * scale
            bought_value = actual_spend * (1 - TX_COST)
            new_cash -= actual_spend
            cur_units = new_holdings.get(t, 0)
            new_holdings[t] = cur_units + bought_value / p

    return new_holdings, new_cash


def get_portfolio_value(holdings, cash, prices, date):
    val = cash
    for t, units in holdings.items():
        if t in prices:
            idx = prices[t].index.get_indexer([date], method='ffill')[0]
            if idx >= 0:
                val += units * prices[t]['Close'].iloc[idx]
    return val


def run_backtest(prices, universe_map, margin=0.0, rebal_policy='BASE'):
    """
    margin: health check hysteresis margin (0.0 = current behavior)
    rebal_policy: 'BASE' (monthly), 'TO30', 'TO50'
    """
    btc = prices.get('BTC-USD')
    if btc is None: return None

    all_dates = btc.index[(btc.index >= START_DATE) & (btc.index <= END_DATE)]
    if len(all_dates) == 0: return None

    holdings = {}
    cash = 10000
    portfolio_values = []
    prev_month = None
    rebal_count = 0

    # Turnover threshold parsing
    to_threshold = None
    if rebal_policy.startswith('TO'):
        to_threshold = int(rebal_policy[2:]) / 100.0
    consec_over = 0

    # Track health flip count for stats
    health_change_days = 0
    prev_picks_set = set()

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)
        risk_on = check_canary(prices, global_idx)
        currently_invested = len(holdings) > 0

        current_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and current_month != prev_month)

        # Compute shadow target with margin-aware health check
        if risk_on:
            shadow_picks = get_healthy_picks_with_margin(
                prices, universe_map, date, global_idx,
                margin, holdings
            )
        else:
            shadow_picks = []
        shadow_target = compute_target(shadow_picks)

        # Track pick changes
        if set(shadow_picks) != prev_picks_set and prev_picks_set:
            health_change_days += 1
        if shadow_picks:
            prev_picks_set = set(shadow_picks)

        # Turnover
        turnover = calc_turnover(holdings, cash, shadow_target, prices, date)

        to_fired = False
        if to_threshold is not None:
            if turnover >= to_threshold:
                consec_over += 1
            else:
                consec_over = 0
            if consec_over >= 1:
                to_fired = True

        # Rebalance decision
        do_rebal = False
        if i == 0:
            do_rebal = True
        elif currently_invested and not risk_on:
            do_rebal = True
        elif not currently_invested and risk_on:
            do_rebal = True
        elif rebal_policy == 'BASE':
            if is_month_change:
                do_rebal = True
        elif to_threshold is not None:
            if rebal_policy.startswith('TO'):
                # Pure turnover
                if to_fired:
                    do_rebal = True

        if do_rebal:
            holdings, cash = execute_rebalance(holdings, cash, shadow_target, prices, date)
            rebal_count += 1
            consec_over = 0

        port_val = get_portfolio_value(holdings, cash, prices, date)
        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    result = pd.DataFrame(portfolio_values).set_index('Date')
    result.attrs['rebal_count'] = rebal_count
    result.attrs['health_changes'] = health_change_days
    return result


def run_btc_bh(prices):
    btc = prices.get('BTC-USD')
    if btc is None: return None
    dates = btc.index[(btc.index >= START_DATE) & (btc.index <= END_DATE)]
    if len(dates) == 0: return None
    start_price = btc['Close'].loc[dates[0]]
    values = 10000 * btc['Close'].loc[dates] / start_price
    return pd.DataFrame({'Value': values.values}, index=dates)


def main():
    print("Loading data...")
    universe_map = load_universe()
    filtered_map = filter_universe_topn(universe_map, TOP_N)

    all_tickers = set()
    for mt in filtered_map.values():
        for t in mt:
            all_tickers.add(t)
    all_tickers.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(all_tickers)
    print(f"  {len(prices)} tickers loaded\n")

    margins = [0.00, 0.01, 0.02, 0.03, 0.05]
    policies = ['BASE', 'TO30', 'TO50']

    # ═══ Main Grid ═══
    print("=" * 130)
    print(f"  HEALTH CHECK MARGIN TEST (SMA{SMA_PERIOD} + Vol{int(VOL_CAP*100)}% + Top{N_PICKS})")
    print("=" * 130)

    # BTC B&H reference
    bh = run_btc_bh(prices)
    if bh is not None:
        m_bh = calc_metrics(bh)
        print(f"\n  BTC B&H:  Sharpe {m_bh['Sharpe']:.3f}  MDD {m_bh['MDD']:.1%}  CAGR {m_bh['CAGR']:+.1%}  Final ${m_bh['Final']:,.0f}")

    results = {}

    for policy in policies:
        print(f"\n  {'─'*120}")
        print(f"  리밸런싱: {policy}")
        print(f"  {'Margin':<10} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Final':>12} {'Rebals':>7} {'PickChg':>8}")
        print(f"  {'─'*65}")

        for margin in margins:
            pv = run_backtest(prices, filtered_map, margin=margin, rebal_policy=policy)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                rc = pv.attrs.get('rebal_count', 0)
                hc = pv.attrs.get('health_changes', 0)
                key = (policy, margin)
                results[key] = (m, rc, hc, pv)
                marker = ' ★' if margin == 0 else ''
                print(f"  {margin:>7.0%}   {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7} {hc:>8}{marker}")

    # ═══ Delta vs BASE 0% ═══
    print("\n" + "=" * 130)
    print("  DELTA vs BASE(월간, 마진 0%) — Sharpe 비교")
    print("=" * 130)

    base_key = ('BASE', 0.0)
    if base_key in results:
        base_m = results[base_key][0]
        print(f"\n  BASE(0%): Sharpe {base_m['Sharpe']:.3f}  CAGR {base_m['CAGR']:+.1%}  MDD {base_m['MDD']:.1%}\n")

        print(f"  {'Policy+Margin':<20} {'Sharpe':>8} {'dSharpe':>8} {'dCAGR':>9} {'dMDD':>9} {'Rebals':>7} {'Verdict':>12}")
        print(f"  {'─'*78}")

        for policy in policies:
            for margin in margins:
                key = (policy, margin)
                if key == base_key: continue
                if key not in results: continue
                m, rc, hc, pv = results[key]
                ds = m['Sharpe'] - base_m['Sharpe']
                dc = (m['CAGR'] - base_m['CAGR']) * 100
                dm = (m['MDD'] - base_m['MDD']) * 100

                if ds > 0.03 and dc > 0:
                    verdict = '** BETTER **'
                elif ds < -0.05 or dc < -5:
                    verdict = 'WORSE'
                else:
                    verdict = 'NEUTRAL'

                label = f"{policy} {margin:.0%}"
                print(f"  {label:<20} {m['Sharpe']:>8.3f} {ds:>+7.3f} {dc:>+8.1f}pp {dm:>+8.1f}pp {rc:>7} {verdict:>12}")

    # ═══ Year-by-year for interesting combos ═══
    print("\n" + "=" * 130)
    print("  YEAR-BY-YEAR CAGR (주요 조합)")
    print("=" * 130)

    interesting = [
        ('BASE', 0.00, 'BASE 0%'),
        ('BASE', 0.02, 'BASE 2%'),
        ('BASE', 0.03, 'BASE 3%'),
        ('BASE', 0.05, 'BASE 5%'),
        ('TO30', 0.03, 'TO30 3%'),
        ('TO30', 0.05, 'TO30 5%'),
        ('TO50', 0.03, 'TO50 3%'),
        ('TO50', 0.05, 'TO50 5%'),
    ]

    years = list(range(2018, 2026))
    print(f"\n  {'Strategy':<15}", end='')
    for y in years:
        print(f" {y:>8}", end='')
    print(f" {'Overall':>9}")
    print(f"  {'─'*95}")

    for policy, margin, label in interesting:
        key = (policy, margin)
        if key not in results: continue
        m_full, rc, hc, pv_full = results[key]
        print(f"  {label:<15}", end='')
        for y in years:
            mask = (pv_full.index >= f'{y}-01-01') & (pv_full.index <= f'{y}-12-31')
            pv_slice = pv_full[mask]
            if len(pv_slice) > 10:
                m = calc_metrics(pv_slice)
                print(f" {m['CAGR']:>+7.1%}", end='')
            else:
                print(f" {'N/A':>8}", end='')
        print(f" {m_full['CAGR']:>+8.1%}")

    # BTC B&H year-by-year
    if bh is not None:
        print(f"  {'BTC B&H':<15}", end='')
        for y in years:
            mask = (bh.index >= f'{y}-01-01') & (bh.index <= f'{y}-12-31')
            bh_slice = bh[mask]
            if len(bh_slice) > 10:
                m = calc_metrics(bh_slice)
                print(f" {m['CAGR']:>+7.1%}", end='')
            else:
                print(f" {'N/A':>8}", end='')
        print(f" {m_bh['CAGR']:>+8.1%}")

    # ═══ Health flip reduction ═══
    print("\n" + "=" * 130)
    print("  HEALTH FLIP REDUCTION (마진별 일일 picks 변경 횟수)")
    print("=" * 130)

    print(f"\n  {'Margin':<10}", end='')
    for policy in policies:
        print(f" {policy+' PickChg':>15} {policy+' Rebals':>12}", end='')
    print()
    print(f"  {'─'*75}")

    for margin in margins:
        print(f"  {margin:>7.0%}  ", end='')
        for policy in policies:
            key = (policy, margin)
            if key in results:
                _, rc, hc, _ = results[key]
                print(f" {hc:>15} {rc:>12}", end='')
            else:
                print(f" {'N/A':>15} {'N/A':>12}", end='')
        print()


if __name__ == '__main__':
    main()
