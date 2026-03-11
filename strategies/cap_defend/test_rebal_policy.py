#!/usr/bin/env python3
"""
Rebalancing Policy Comparison:
  A) 현행: 카나리아 ON/OFF만 매일, 코인 선별은 월간만
  B) 개선1: 카나리아 진입 시 + 헬스 통과 코인 수가 변하면 리밸런싱
  C) 개선2: 매일 헬스체크 + 리밸런싱 (코인 구성 변하면)
  D) 매일 풀 리밸런싱 (무조건)

SMA150 + Vol5% + 시총순 Top5 + 균등가중
"""

import os, sys, warnings
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
END_DATE = '2025-12-31'


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
    if universe_map:
        return list(universe_map.values())[0]
    return []


def check_canary_sma(prices, global_idx, sma_period=150):
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < sma_period: return False
    return close.iloc[-1] > close.rolling(sma_period).mean().iloc[-1]


def check_health(ticker, prices, global_idx, vol_cap=0.05):
    if ticker not in prices: return False
    close = prices[ticker]['Close'].iloc[:global_idx+1]
    if len(close) < 90: return False
    cur = close.iloc[-1]
    sma30 = close.rolling(30).mean().iloc[-1]
    mom21 = calc_ret(close, 21)
    vol90 = get_volatility(close, 90)
    return cur > sma30 and mom21 > 0 and (vol90 is not None and vol90 <= vol_cap)


def get_healthy_picks(prices, universe_map, date, global_idx, vol_cap=0.05, n_picks=5):
    """Return sorted list of healthy ticker picks (market cap order)."""
    uni_tickers = get_universe_for_date(universe_map, date)
    uni_clean = [t.replace('-USD', '') for t in uni_tickers
                if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]

    healthy = []
    for sym in uni_clean:
        ticker = f"{sym}-USD"
        if check_health(ticker, prices, global_idx, vol_cap):
            healthy.append(ticker)

    return healthy[:n_picks]


def compute_target(picks):
    """Equal weight from a list of picks."""
    if not picks:
        return {'CASH': 1.0}
    return {t: 1.0/len(picks) for t in picks}


def execute_rebalance(holdings, cash, weights, prices, date, tx_cost):
    def _get_price(ticker):
        if ticker not in prices: return 0
        idx = prices[ticker].index.get_indexer([date], method='ffill')[0]
        return prices[ticker]['Close'].iloc[idx] if idx >= 0 else 0

    current_values = {}
    port_val = cash
    for t, units in holdings.items():
        p = _get_price(t)
        v = units * p
        current_values[t] = v
        port_val += v

    if 'CASH' in weights and weights['CASH'] == 1.0:
        sell_total = sum(current_values.values())
        return {}, cash + sell_total * (1 - tx_cost)

    target_values = {t: port_val * w for t, w in weights.items() if t != 'CASH'}
    new_holdings = {}
    new_cash = cash
    all_tickers = set(list(current_values.keys()) + list(target_values.keys()))

    for t in all_tickers:
        cur_val = current_values.get(t, 0)
        tgt_val = target_values.get(t, 0)
        p = _get_price(t)
        if p <= 0: continue
        if tgt_val >= cur_val:
            if cur_val > 0:
                new_holdings[t] = holdings[t]
        else:
            sell_amount = cur_val - tgt_val
            new_cash += sell_amount * (1 - tx_cost)
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
            p = _get_price(t)
            if p <= 0: continue
            actual_spend = buy_val * scale
            bought_value = actual_spend * (1 - tx_cost)
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


def run_backtest(prices, universe_map, policy='A', sma_period=150, vol_cap=0.05,
                 tx_cost=0.004, start_date='2018-01-01', end_date=END_DATE):
    """
    Policies:
      A: canary daily + coin monthly only (현행)
      B: canary daily + coin rebal when pick set changes
      C: same as B but also rebal when # of healthy coins increases (from 1→3 etc)
      D: full daily rebalancing every day
    """
    btc = prices.get('BTC-USD')
    if btc is None: return None

    all_dates = btc.index[(btc.index >= start_date) & (btc.index <= end_date)]
    if len(all_dates) == 0: return None

    holdings = {}
    cash = 10000
    portfolio_values = []
    prev_month = None
    prev_picks = []
    rebal_count = 0
    rebal_reasons = {'init': 0, 'monthly': 0, 'canary_exit': 0, 'canary_entry': 0, 'pick_change': 0, 'daily': 0}

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)

        current_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and current_month != prev_month)

        risk_on = check_canary_sma(prices, global_idx, sma_period)
        currently_invested = len(holdings) > 0

        # Compute today's picks (needed for policies B/C/D)
        if risk_on:
            today_picks = get_healthy_picks(prices, universe_map, date, global_idx, vol_cap)
        else:
            today_picks = []

        do_rebal = False
        reason = ''

        if policy == 'A':
            # Current: canary daily, coins monthly
            if i == 0:
                do_rebal, reason = True, 'init'
            elif is_month_change:
                do_rebal, reason = True, 'monthly'
            elif currently_invested and not risk_on:
                do_rebal, reason = True, 'canary_exit'
            elif not currently_invested and risk_on:
                do_rebal, reason = True, 'canary_entry'

        elif policy == 'B':
            # Canary daily + rebal when pick composition changes
            if i == 0:
                do_rebal, reason = True, 'init'
            elif is_month_change:
                do_rebal, reason = True, 'monthly'
            elif currently_invested and not risk_on:
                do_rebal, reason = True, 'canary_exit'
            elif not currently_invested and risk_on:
                do_rebal, reason = True, 'canary_entry'
            elif set(today_picks) != set(prev_picks):
                do_rebal, reason = True, 'pick_change'

        elif policy == 'C':
            # Same as B but also when # healthy increases (catch new coins faster)
            if i == 0:
                do_rebal, reason = True, 'init'
            elif is_month_change:
                do_rebal, reason = True, 'monthly'
            elif currently_invested and not risk_on:
                do_rebal, reason = True, 'canary_exit'
            elif not currently_invested and risk_on:
                do_rebal, reason = True, 'canary_entry'
            elif sorted(today_picks) != sorted(prev_picks):
                do_rebal, reason = True, 'pick_change'

        elif policy == 'D':
            # Full daily rebalancing
            do_rebal, reason = True, 'daily'

        if do_rebal:
            target = compute_target(today_picks)
            holdings, cash = execute_rebalance(holdings, cash, target, prices, date, tx_cost)
            rebal_count += 1
            rebal_reasons[reason] = rebal_reasons.get(reason, 0) + 1
            prev_picks = today_picks

        port_val = get_portfolio_value(holdings, cash, prices, date)
        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    result = pd.DataFrame(portfolio_values).set_index('Date')
    result.attrs['rebal_count'] = rebal_count
    result.attrs['rebal_reasons'] = rebal_reasons
    return result


def run_btc_bh(prices, start_date='2018-01-01', end_date=END_DATE):
    btc = prices.get('BTC-USD')
    if btc is None: return None
    dates = btc.index[(btc.index >= start_date) & (btc.index <= end_date)]
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

    # ═══ TEST: Rebalancing Policy Comparison ═══
    print("=" * 115)
    print("  REBALANCING POLICY COMPARISON (SMA150 + Vol5% + 시총순 Top5 + 균등)")
    print("=" * 115)

    policies = [
        ('A', 'A: 카나리아만 매일 (현행)'),
        ('B', 'B: +코인 구성 변경시'),
        ('C', 'C: +코인 수 증가시도'),
        ('D', 'D: 매일 풀 리밸런싱'),
    ]

    print(f"\n  {'Policy':<30} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Final':>12} {'Rebals':>7} {'세부 내역'}")
    print(f"  {'-'*100}")

    # BTC B&H
    bh = run_btc_bh(prices)
    if bh is not None and len(bh) > 10:
        m = calc_metrics(bh)
        print(f"  {'BTC B&H':<30} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f}")

    for pol, label in policies:
        pv = run_backtest(prices, filtered_map, policy=pol)
        if pv is not None and len(pv) > 10:
            m = calc_metrics(pv)
            rc = pv.attrs.get('rebal_count', 0)
            reasons = pv.attrs.get('rebal_reasons', {})
            reason_str = ', '.join(f"{k}={v}" for k, v in sorted(reasons.items()) if v > 0)
            print(f"  {label:<30} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7} {reason_str}")

    # ═══ Year-by-year comparison ═══
    print("\n" + "=" * 115)
    print("  YEAR-BY-YEAR CAGR")
    print("=" * 115)

    years = list(range(2018, 2026))
    print(f"\n  {'Policy':<30}", end='')
    for y in years:
        print(f" {y:>9}", end='')
    print(f" {'Overall':>10}")
    print(f"  {'-'*110}")

    for pol, label in policies:
        pv_full = run_backtest(prices, filtered_map, policy=pol)
        print(f"  {label:<30}", end='')
        for y in years:
            mask = (pv_full.index >= f'{y}-01-01') & (pv_full.index <= f'{y}-12-31')
            pv_slice = pv_full[mask]
            if len(pv_slice) > 10:
                m = calc_metrics(pv_slice)
                print(f" {m['CAGR']:>+8.1%}", end='')
            else:
                print(f" {'N/A':>9}", end='')
        m_full = calc_metrics(pv_full)
        print(f" {m_full['CAGR']:>+9.1%}")

    # BTC B&H year-by-year
    bh = run_btc_bh(prices)
    print(f"  {'BTC B&H':<30}", end='')
    for y in years:
        mask = (bh.index >= f'{y}-01-01') & (bh.index <= f'{y}-12-31')
        bh_slice = bh[mask]
        if len(bh_slice) > 10:
            m = calc_metrics(bh_slice)
            print(f" {m['CAGR']:>+8.1%}", end='')
        else:
            print(f" {'N/A':>9}", end='')
    m_bh = calc_metrics(bh)
    print(f" {m_bh['CAGR']:>+9.1%}")

    # ═══ Concentration analysis ═══
    print("\n" + "=" * 115)
    print("  CONCENTRATION ANALYSIS: Policy A에서 1코인 100% 집중 기간")
    print("=" * 115)

    # Run policy A with detailed logging
    btc = prices.get('BTC-USD')
    all_dates = btc.index[(btc.index >= '2018-01-01') & (btc.index <= END_DATE)]

    holdings = {}
    cash = 10000
    prev_month = None
    single_coin_days = 0
    total_invested_days = 0
    concentration_log = []

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)
        risk_on = check_canary_sma(prices, global_idx, 150)
        currently_invested = len(holdings) > 0

        current_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and current_month != prev_month)

        do_rebal = False
        if i == 0:
            do_rebal = True
        elif is_month_change:
            do_rebal = True
        elif currently_invested and not risk_on:
            do_rebal = True
        elif not currently_invested and risk_on:
            do_rebal = True

        if do_rebal:
            if risk_on:
                picks = get_healthy_picks(prices, filtered_map, date, global_idx, 0.05)
            else:
                picks = []
            target = compute_target(picks)
            holdings, cash = execute_rebalance(holdings, cash, target, prices, date, tx_cost=0.004)

        if len(holdings) > 0:
            total_invested_days += 1
            if len(holdings) == 1:
                single_coin_days += 1
                coin = list(holdings.keys())[0].replace('-USD', '')
                concentration_log.append((date.strftime('%Y-%m-%d'), coin))

        prev_month = current_month

    print(f"\n  총 투자일: {total_invested_days}")
    print(f"  1코인 집중일: {single_coin_days} ({single_coin_days/max(1,total_invested_days)*100:.1f}%)")

    if concentration_log:
        print(f"\n  1코인 집중 기간:")
        # Group consecutive days
        groups = []
        start = concentration_log[0]
        prev = concentration_log[0]
        for item in concentration_log[1:]:
            if item[1] == prev[1]:
                prev = item
            else:
                groups.append((start[0], prev[0], start[1]))
                start = item
                prev = item
        groups.append((start[0], prev[0], start[1]))

        for s, e, coin in groups:
            if s == e:
                print(f"    {s}: {coin} 100%")
            else:
                print(f"    {s} ~ {e}: {coin} 100%")


if __name__ == '__main__':
    main()
