#!/usr/bin/env python3
"""
Vol Cap Period Analysis: Is Vol5% performance driven by 2018-2019 outliers?
Tests Vol5%, Vol7%, Vol10% across different start periods.
Uses: SMA150 daily canary + health filter (no scoring) + top 5 market cap order + equal weight
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


def check_health(ticker, prices, global_idx, vol_cap=0.10):
    if ticker not in prices: return False
    close = prices[ticker]['Close'].iloc[:global_idx+1]
    if len(close) < 90: return False
    cur = close.iloc[-1]
    sma30 = close.rolling(30).mean().iloc[-1]
    mom21 = calc_ret(close, 21)
    vol90 = get_volatility(close, 90)
    return cur > sma30 and mom21 > 0 and (vol90 is not None and vol90 <= vol_cap)


def compute_target_simple(prices, universe_map, date, global_idx, sma_period=150, vol_cap=0.10, n_picks=5):
    """Canary + health filter + market cap order (no scoring) + equal weight."""
    if not check_canary_sma(prices, global_idx, sma_period):
        return {'CASH': 1.0}

    uni_tickers = get_universe_for_date(universe_map, date)
    uni_clean = [t.replace('-USD', '') for t in uni_tickers
                if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]

    healthy = []
    for sym in uni_clean:
        ticker = f"{sym}-USD"
        if check_health(ticker, prices, global_idx, vol_cap):
            healthy.append(ticker)

    if not healthy:
        return {'CASH': 1.0}

    picks = healthy[:n_picks]
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


def run_backtest(prices, universe_map, sma_period=150, vol_cap=0.10,
                 tx_cost=0.004, start_date='2018-01-01', end_date=END_DATE):
    btc = prices.get('BTC-USD')
    if btc is None: return None

    all_dates = btc.index[(btc.index >= start_date) & (btc.index <= end_date)]
    if len(all_dates) == 0: return None

    holdings = {}
    cash = 10000
    portfolio_values = []
    prev_month = None
    rebal_count = 0

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)
        port_val = get_portfolio_value(holdings, cash, prices, date)

        current_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and current_month != prev_month)

        # Daily canary check
        risk_on = check_canary_sma(prices, global_idx, sma_period)
        currently_invested = len(holdings) > 0

        do_rebal = False
        if i == 0:
            do_rebal = True
        elif is_month_change:
            do_rebal = True
        elif currently_invested and not risk_on:
            do_rebal = True  # Daily exit
        elif not currently_invested and risk_on:
            do_rebal = True  # Daily entry

        if do_rebal:
            target = compute_target_simple(prices, universe_map, date, global_idx,
                                          sma_period=sma_period, vol_cap=vol_cap)
            holdings, cash = execute_rebalance(holdings, cash, target, prices, date, tx_cost)
            rebal_count += 1

        port_val = get_portfolio_value(holdings, cash, prices, date)
        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    result = pd.DataFrame(portfolio_values).set_index('Date')
    result.attrs['rebal_count'] = rebal_count
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

    # ═══ TEST 1: Vol Cap × Start Period ═══
    print("=" * 110)
    print("  TEST 1: SMA150 + Health (No Scoring, Market Cap Order, Equal Weight)")
    print("  Vol Cap × Start Period Comparison")
    print("=" * 110)

    vol_caps = [0.05, 0.07, 0.08, 0.10, 0.12, 0.15]
    start_dates = ['2018-01-01', '2019-01-01', '2020-01-01', '2021-01-01']

    for sd in start_dates:
        n_years = (2025 - int(sd[:4]) + 1)
        print(f"\n  ── Period: {sd[:4]}-2025 ({n_years} years) ──")
        print(f"  {'Config':<20} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Final':>12} {'Rebals':>7}")
        print(f"  {'-'*68}")

        # BTC B&H benchmark
        bh = run_btc_bh(prices, start_date=sd)
        if bh is not None and len(bh) > 10:
            m = calc_metrics(bh)
            print(f"  {'BTC B&H':<20} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {'':>7}")

        for vc in vol_caps:
            label = f"Vol{int(vc*100)}%"
            pv = run_backtest(prices, filtered_map, sma_period=150, vol_cap=vc, start_date=sd)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                rc = pv.attrs.get('rebal_count', 0)
                print(f"  {label:<20} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7}")
            else:
                print(f"  {label:<20} {'N/A':>8}")

    # ═══ TEST 2: Year-by-year comparison for Vol5% vs Vol10% ═══
    print("\n" + "=" * 110)
    print("  TEST 2: SMA150 Year-by-Year CAGR — Vol5% vs Vol7% vs Vol10%")
    print("=" * 110)

    years = list(range(2018, 2026))
    print(f"\n  {'Config':<12}", end='')
    for y in years:
        print(f" {y:>9}", end='')
    print(f" {'Overall':>10}")
    print(f"  {'-'*105}")

    for vc in [0.05, 0.07, 0.10]:
        label = f"Vol{int(vc*100)}%"
        print(f"  {label:<12}", end='')
        pv_full = run_backtest(prices, filtered_map, sma_period=150, vol_cap=vc, start_date='2018-01-01')

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
    bh = run_btc_bh(prices, start_date='2018-01-01')
    print(f"  {'BTC B&H':<12}", end='')
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

    # ═══ TEST 3: SMA50 vs SMA150 with different Vol Caps ═══
    print("\n" + "=" * 110)
    print("  TEST 3: SMA50 vs SMA150 × Vol Cap (Full Period 2018-2025)")
    print("=" * 110)

    print(f"\n  {'Config':<25} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Final':>12} {'Rebals':>7}")
    print(f"  {'-'*70}")

    for sma in [50, 150]:
        for vc in [0.05, 0.07, 0.10]:
            label = f"SMA{sma}+Vol{int(vc*100)}%"
            pv = run_backtest(prices, filtered_map, sma_period=sma, vol_cap=vc)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                rc = pv.attrs.get('rebal_count', 0)
                print(f"  {label:<25} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7}")

    bh = run_btc_bh(prices)
    if bh is not None and len(bh) > 10:
        m = calc_metrics(bh)
        print(f"  {'BTC B&H':<25} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f}")


if __name__ == '__main__':
    main()
