#!/usr/bin/env python3
"""
Turnover-Based Rebalancing Test

Instead of fixed calendar (monthly) or individual coin health checks,
rebalance only when the "shadow portfolio" (ideal picks today) differs
from current holdings by more than a threshold.

Turnover = sum(|current_weight - target_weight|) / 2
  - 5 coins, 1 coin change = ~20% turnover
  - 2 coins change = ~40% turnover

Strategies:
  BASE:    월간 리밸런싱 (현행)
  TO10:    턴오버 ≥ 10% → 리밸런싱 (거의 매일)
  TO20:    턴오버 ≥ 20% → 리밸런싱 (1코인 변경)
  TO30:    턴오버 ≥ 30% → 리밸런싱 (1.5코인 변경)
  TO40:    턴오버 ≥ 40% → 리밸런싱 (2코인 변경)
  TO50:    턴오버 ≥ 50% → 리밸런싱 (2.5코인 변경)
  TO30_2D: 턴오버 ≥ 30% 2일 연속 → 리밸런싱 (기존 R2)
  TO40_2D: 턴오버 ≥ 40% 2일 연속
  TO30_3D: 턴오버 ≥ 30% 3일 연속

  Calendar + Turnover hybrid:
  M+TO20:  월간 + 턴오버 ≥ 20% 시 추가 리밸런싱
  M+TO30:  월간 + 턴오버 ≥ 30% 시 추가
  M+TO40:  월간 + 턴오버 ≥ 40% 시 추가
  M+TO30_2D: 월간 + 턴오버 ≥ 30% 2일연속 시 추가

  W+TO30:  주간 + 턴오버 ≥ 30% 시 추가

All use: SMA150 daily canary + Vol5% health + market cap order Top5 + equal weight
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
    if universe_map:
        return list(universe_map.values())[0]
    return []


def check_canary(prices, global_idx):
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < SMA_PERIOD: return False
    return close.iloc[-1] > close.rolling(SMA_PERIOD).mean().iloc[-1]


def check_health(ticker, prices, global_idx):
    if ticker not in prices: return False
    close = prices[ticker]['Close'].iloc[:global_idx+1]
    if len(close) < 90: return False
    cur = close.iloc[-1]
    sma30 = close.rolling(30).mean().iloc[-1]
    mom21 = calc_ret(close, 21)
    vol90 = get_volatility(close, 90)
    return cur > sma30 and mom21 > 0 and (vol90 is not None and vol90 <= VOL_CAP)


def get_healthy_picks(prices, universe_map, date, global_idx):
    uni_tickers = get_universe_for_date(universe_map, date)
    uni_clean = [t.replace('-USD', '') for t in uni_tickers
                if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]
    healthy = []
    for sym in uni_clean:
        ticker = f"{sym}-USD"
        if check_health(ticker, prices, global_idx):
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
    """Calculate turnover between current portfolio and target weights.
    Turnover = sum(|current_weight - target_weight|) / 2
    Range: 0 (identical) to 1 (completely different)
    """
    # Current weights
    port_val = cash
    current_values = {}
    for t, units in holdings.items():
        p = get_price(t, prices, date)
        v = units * p
        current_values[t] = v
        port_val += v

    if port_val <= 0:
        return 1.0  # empty portfolio, full turnover needed

    current_weights = {}
    for t, v in current_values.items():
        current_weights[t] = v / port_val
    cash_weight = cash / port_val

    # Add cash to target if present
    target_cash = target_weights.get('CASH', 0)
    target_coins = {t: w for t, w in target_weights.items() if t != 'CASH'}

    # All tickers
    all_tickers = set(list(current_weights.keys()) + list(target_coins.keys()))

    turnover = 0
    for t in all_tickers:
        cw = current_weights.get(t, 0)
        tw = target_coins.get(t, 0)
        turnover += abs(cw - tw)

    # Cash difference
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


def run_backtest(prices, universe_map, strategy='BASE'):
    btc = prices.get('BTC-USD')
    if btc is None: return None

    all_dates = btc.index[(btc.index >= START_DATE) & (btc.index <= END_DATE)]
    if len(all_dates) == 0: return None

    holdings = {}
    cash = 10000
    portfolio_values = []
    prev_month = None
    rebal_count = 0
    to_trigger_count = 0  # turnover-triggered rebalances

    # Parse strategy
    # Format: [calendar]_TO[threshold]_[consec]D
    # or just BASE, TO30, TO30_2D, M+TO30, W+TO30, etc.

    # Determine calendar schedule
    has_monthly = strategy.startswith('M+') or strategy == 'BASE'
    has_weekly = strategy.startswith('W+')
    turnover_only = not has_monthly and not has_weekly and strategy != 'BASE'

    # Determine turnover threshold and consecutive days
    to_threshold = None
    consec_days = 1
    to_part = strategy.replace('M+', '').replace('W+', '')
    if to_part.startswith('TO'):
        parts = to_part.split('_')
        to_threshold = int(parts[0][2:]) / 100.0  # TO30 -> 0.30
        if len(parts) > 1 and parts[1].endswith('D'):
            consec_days = int(parts[1][:-1])

    # State for consecutive turnover days
    consec_over = 0

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)
        risk_on = check_canary(prices, global_idx)
        currently_invested = len(holdings) > 0

        current_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and current_month != prev_month)
        is_monday = date.weekday() == 0

        # Compute shadow target (every day)
        if risk_on:
            shadow_picks = get_healthy_picks(prices, universe_map, date, global_idx)
        else:
            shadow_picks = []
        shadow_target = compute_target(shadow_picks)

        # Calculate turnover vs current holdings
        turnover = calc_turnover(holdings, cash, shadow_target, prices, date)

        # Check turnover threshold
        to_fired = False
        if to_threshold is not None:
            if turnover >= to_threshold:
                consec_over += 1
            else:
                consec_over = 0

            if consec_over >= consec_days:
                to_fired = True

        # Determine if we rebalance
        do_rebal = False
        reason = ''

        if i == 0:
            do_rebal, reason = True, 'init'
        elif currently_invested and not risk_on:
            # Always exit immediately when canary turns off
            do_rebal, reason = True, 'canary_exit'
        elif not currently_invested and risk_on:
            # Always enter immediately when canary turns on
            do_rebal, reason = True, 'canary_entry'
        elif strategy == 'BASE':
            if is_month_change:
                do_rebal, reason = True, 'monthly'
        elif has_monthly:
            if is_month_change:
                do_rebal, reason = True, 'monthly'
            elif to_fired and currently_invested:
                do_rebal, reason = True, 'turnover'
        elif has_weekly:
            if is_monday:
                do_rebal, reason = True, 'weekly'
            elif to_fired and currently_invested:
                do_rebal, reason = True, 'turnover'
        elif turnover_only:
            # Pure turnover-based: rebalance ONLY on turnover threshold
            if to_fired:
                do_rebal, reason = True, 'turnover'

        if do_rebal:
            holdings, cash = execute_rebalance(holdings, cash, shadow_target, prices, date)
            rebal_count += 1
            if reason == 'turnover':
                to_trigger_count += 1
            # Reset consecutive counter after rebal
            consec_over = 0

        port_val = get_portfolio_value(holdings, cash, prices, date)
        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    result = pd.DataFrame(portfolio_values).set_index('Date')
    result.attrs['rebal_count'] = rebal_count
    result.attrs['to_triggers'] = to_trigger_count
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

    strategies = [
        ('BASE',       'BASE: 월간 (현행)'),
        # Pure turnover-based (no calendar)
        ('TO10',       'TO10: 턴오버≥10%'),
        ('TO20',       'TO20: 턴오버≥20%'),
        ('TO30',       'TO30: 턴오버≥30%'),
        ('TO40',       'TO40: 턴오버≥40%'),
        ('TO50',       'TO50: 턴오버≥50%'),
        # Consecutive days required
        ('TO20_2D',    'TO20_2D: 턴오버≥20% 2일연속'),
        ('TO30_2D',    'TO30_2D: 턴오버≥30% 2일연속'),
        ('TO40_2D',    'TO40_2D: 턴오버≥40% 2일연속'),
        ('TO30_3D',    'TO30_3D: 턴오버≥30% 3일연속'),
        ('TO50_2D',    'TO50_2D: 턴오버≥50% 2일연속'),
        # Monthly + Turnover hybrid
        ('M+TO20',     'M+TO20: 월간 + 턴오버≥20%'),
        ('M+TO30',     'M+TO30: 월간 + 턴오버≥30%'),
        ('M+TO40',     'M+TO40: 월간 + 턴오버≥40%'),
        ('M+TO30_2D',  'M+TO30_2D: 월간 + TO30% 2일'),
        ('M+TO50',     'M+TO50: 월간 + 턴오버≥50%'),
        # Weekly + Turnover
        ('W+TO30',     'W+TO30: 주간 + 턴오버≥30%'),
    ]

    # ═══ Overall Performance ═══
    print("=" * 120)
    print(f"  TURNOVER-BASED REBALANCING (SMA{SMA_PERIOD} + Vol{int(VOL_CAP*100)}% + Top{N_PICKS})")
    print("=" * 120)

    print(f"\n  {'Strategy':<35} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Final':>12} {'Rebals':>7} {'TO트리거':>8}")
    print(f"  {'-'*92}")

    bh = run_btc_bh(prices)
    if bh is not None and len(bh) > 10:
        m = calc_metrics(bh)
        print(f"  {'BTC B&H':<35} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f}")

    results = {}
    for strat_id, label in strategies:
        pv = run_backtest(prices, filtered_map, strategy=strat_id)
        if pv is not None and len(pv) > 10:
            m = calc_metrics(pv)
            rc = pv.attrs.get('rebal_count', 0)
            to_t = pv.attrs.get('to_triggers', 0)
            results[strat_id] = (m, rc, to_t, pv)
            print(f"  {label:<35} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7} {to_t:>8}")
        else:
            print(f"  {label:<35} {'ERROR':>8}")

    # ═══ Year-by-year ═══
    print("\n" + "=" * 120)
    print("  YEAR-BY-YEAR CAGR")
    print("=" * 120)

    years = list(range(2018, 2026))
    print(f"\n  {'Strategy':<35}", end='')
    for y in years:
        print(f" {y:>8}", end='')
    print(f" {'Overall':>9}")
    print(f"  {'-'*112}")

    for strat_id, label in strategies:
        if strat_id not in results: continue
        m_full, rc, to_t, pv_full = results[strat_id]
        print(f"  {label:<35}", end='')
        for y in years:
            mask = (pv_full.index >= f'{y}-01-01') & (pv_full.index <= f'{y}-12-31')
            pv_slice = pv_full[mask]
            if len(pv_slice) > 10:
                m = calc_metrics(pv_slice)
                print(f" {m['CAGR']:>+7.1%}", end='')
            else:
                print(f" {'N/A':>8}", end='')
        print(f" {m_full['CAGR']:>+8.1%}")

    print(f"  {'BTC B&H':<35}", end='')
    for y in years:
        mask = (bh.index >= f'{y}-01-01') & (bh.index <= f'{y}-12-31')
        bh_slice = bh[mask]
        if len(bh_slice) > 10:
            m = calc_metrics(bh_slice)
            print(f" {m['CAGR']:>+7.1%}", end='')
        else:
            print(f" {'N/A':>8}", end='')
    m_bh = calc_metrics(bh)
    print(f" {m_bh['CAGR']:>+8.1%}")

    # ═══ Delta vs Baseline ═══
    print("\n" + "=" * 120)
    print("  DELTA vs BASELINE")
    print("=" * 120)

    if 'BASE' in results:
        base_m = results['BASE'][0]
        print(f"\n  {'Strategy':<35} {'dSharpe':>8} {'dMDD':>8} {'dCAGR':>8} {'TO/Yr':>7} {'Verdict':>12}")
        print(f"  {'-'*82}")
        n_years = 8
        for strat_id, label in strategies:
            if strat_id == 'BASE': continue
            if strat_id not in results: continue
            m, rc, to_t, pv = results[strat_id]
            ds = m['Sharpe'] - base_m['Sharpe']
            dm = (m['MDD'] - base_m['MDD']) * 100
            dc = (m['CAGR'] - base_m['CAGR']) * 100
            tyr = to_t / n_years
            if ds > 0.03 and dc > 0:
                verdict = '** BETTER **'
            elif ds > -0.03 and abs(dc) < 3:
                verdict = 'NEUTRAL'
            elif ds < -0.05 or dc < -5:
                verdict = 'WORSE'
            else:
                verdict = 'NEUTRAL'
            print(f"  {label:<35} {ds:>+7.3f} {dm:>+7.1f}pp {dc:>+7.1f}pp {tyr:>6.1f} {verdict:>12}")


if __name__ == '__main__':
    main()
