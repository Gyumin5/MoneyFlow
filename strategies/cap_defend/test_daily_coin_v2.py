#!/usr/bin/env python3
"""
Daily Coin Check v2: Turnover-Controlled Strategies

Previous test showed ALL daily strategies worse than monthly baseline.
Root cause: too many swaps despite "daily check". Need real turnover control.

New strategies:
  CD7:  Daily check + 7-day cooldown after any swap
  CD14: Daily check + 14-day cooldown after any swap
  MB1:  Daily check + max 1 coin swap per month
  MB2:  Daily check + max 2 coin swaps per month
  WH:   Weekly health check (Monday), 2-day fail → swap
  HY3:  Hysteresis: 3-day fail to exit, 3-day pass to enter
  HY5:  Hysteresis: 5-day fail to exit, 5-day pass to enter
  S3:   매수가 -10% 손절 (from v1, was NEUTRAL — re-verify)
  S20:  트레일링 -20% 스탑 (wider stop)
  S25:  트레일링 -25% 스탑

Base: SMA150 daily canary + Vol5% health + market cap order Top5 + equal weight
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


def get_all_healthy(prices, universe_map, date, global_idx):
    uni_tickers = get_universe_for_date(universe_map, date)
    uni_clean = [t.replace('-USD', '') for t in uni_tickers
                if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]
    healthy = []
    for sym in uni_clean:
        ticker = f"{sym}-USD"
        if check_health(ticker, prices, global_idx):
            healthy.append(ticker)
    return healthy


def compute_target(picks):
    if not picks:
        return {'CASH': 1.0}
    return {t: 1.0/len(picks) for t in picks}


def get_price(ticker, prices, date):
    if ticker not in prices: return 0
    idx = prices[ticker].index.get_indexer([date], method='ffill')[0]
    return prices[ticker]['Close'].iloc[idx] if idx >= 0 else 0


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


def sell_single_coin(holdings, cash, ticker, prices, date):
    if ticker not in holdings:
        return holdings, cash
    p = get_price(ticker, prices, date)
    if p <= 0:
        return holdings, cash
    sell_val = holdings[ticker] * p
    new_cash = cash + sell_val * (1 - TX_COST)
    new_holdings = {t: u for t, u in holdings.items() if t != ticker}
    return new_holdings, new_cash


def buy_single_coin(holdings, cash, ticker, target_value, prices, date):
    p = get_price(ticker, prices, date)
    if p <= 0 or cash <= 0:
        return holdings, cash
    spend = min(target_value, cash)
    bought_value = spend * (1 - TX_COST)
    new_holdings = dict(holdings)
    new_holdings[ticker] = new_holdings.get(ticker, 0) + bought_value / p
    return new_holdings, cash - spend


def swap_one_coin(holdings, cash, sell_ticker, buy_ticker, prices, date):
    """Sell one coin and buy another in its place."""
    holdings, cash = sell_single_coin(holdings, cash, sell_ticker, prices, date)
    if buy_ticker:
        port_val = get_portfolio_value(holdings, cash, prices, date)
        target_val = port_val / N_PICKS
        holdings, cash = buy_single_coin(holdings, cash, buy_ticker, target_val, prices, date)
    return holdings, cash


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
    daily_swap_count = 0  # total individual coin swaps

    # State
    health_fail_days = defaultdict(int)
    health_pass_days = defaultdict(int)
    coin_peaks = {}
    coin_buy_prices = {}
    last_swap_idx = -999  # index of last daily swap (for cooldown)
    month_swap_count = 0  # swaps in current month (for MB strategies)
    current_swap_month = ''

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)
        risk_on = check_canary(prices, global_idx)
        currently_invested = len(holdings) > 0

        current_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and current_month != prev_month)
        is_monday = date.weekday() == 0

        # Reset monthly swap counter
        if current_month != current_swap_month:
            month_swap_count = 0
            current_swap_month = current_month

        # Update health tracking for ALL held coins
        for t in list(holdings.keys()):
            p = get_price(t, prices, date)
            if p > coin_peaks.get(t, 0):
                coin_peaks[t] = p
            if check_health(t, prices, global_idx):
                health_fail_days[t] = 0
                health_pass_days[t] = health_pass_days.get(t, 0) + 1
            else:
                health_fail_days[t] = health_fail_days.get(t, 0) + 1
                health_pass_days[t] = 0

        # Update health tracking for non-held coins (for hysteresis entry)
        if strategy in ('HY3', 'HY5', 'A1'):
            all_h = get_all_healthy(prices, universe_map, date, global_idx)
            for t in all_h:
                if t not in holdings:
                    health_pass_days[t] = health_pass_days.get(t, 0) + 1
            uni_tickers = get_universe_for_date(universe_map, date)
            for ut in uni_tickers:
                sym = ut.replace('-USD', '')
                if sym in EXCLUDE_SYMBOLS: continue
                ticker = f"{sym}-USD"
                if ticker not in all_h and ticker not in holdings:
                    health_pass_days[ticker] = 0

        # ── Common: full rebalance triggers ──
        do_full_rebal = False
        if i == 0:
            do_full_rebal = True
        elif is_month_change:
            do_full_rebal = True
        elif currently_invested and not risk_on:
            do_full_rebal = True
        elif not currently_invested and risk_on:
            do_full_rebal = True

        if do_full_rebal:
            if risk_on:
                picks = get_healthy_picks(prices, universe_map, date, global_idx)
            else:
                picks = []
            target = compute_target(picks)
            holdings, cash = execute_rebalance(holdings, cash, target, prices, date)
            rebal_count += 1
            for t in holdings:
                if t not in coin_buy_prices:
                    coin_buy_prices[t] = get_price(t, prices, date)
                    coin_peaks[t] = coin_buy_prices[t]
            # Reset health counters for new picks
            health_fail_days.clear()
            health_pass_days.clear()

        elif currently_invested and risk_on and strategy != 'BASE':
            # ── Strategy-specific daily coin logic ──

            if strategy in ('CD7', 'CD14'):
                cooldown = 7 if strategy == 'CD7' else 14
                if (i - last_swap_idx) < cooldown:
                    pass  # Still in cooldown
                else:
                    # Find worst held coin (longest health fail, must be >= 2 days)
                    worst_t, worst_fail = None, 0
                    for t in list(holdings.keys()):
                        fd = health_fail_days.get(t, 0)
                        if fd >= 2 and fd > worst_fail:
                            worst_fail = fd
                            worst_t = t

                    if worst_t:
                        # Find best replacement
                        all_healthy = get_all_healthy(prices, universe_map, date, global_idx)
                        held_set = set(holdings.keys())
                        replacement = None
                        for c in all_healthy:
                            if c not in held_set:
                                replacement = c
                                break

                        if replacement:
                            holdings, cash = swap_one_coin(holdings, cash, worst_t, replacement, prices, date)
                            coin_buy_prices[replacement] = get_price(replacement, prices, date)
                            coin_peaks[replacement] = coin_buy_prices[replacement]
                            coin_peaks.pop(worst_t, None)
                            coin_buy_prices.pop(worst_t, None)
                            health_fail_days.pop(worst_t, None)
                            last_swap_idx = i
                            daily_swap_count += 1
                            rebal_count += 1
                        else:
                            # No replacement: just sell to cash
                            holdings, cash = sell_single_coin(holdings, cash, worst_t, prices, date)
                            coin_peaks.pop(worst_t, None)
                            coin_buy_prices.pop(worst_t, None)
                            health_fail_days.pop(worst_t, None)
                            last_swap_idx = i
                            daily_swap_count += 1
                            rebal_count += 1

            elif strategy in ('MB1', 'MB2'):
                max_swaps = 1 if strategy == 'MB1' else 2
                if month_swap_count >= max_swaps:
                    pass  # Budget exhausted
                else:
                    worst_t, worst_fail = None, 0
                    for t in list(holdings.keys()):
                        fd = health_fail_days.get(t, 0)
                        if fd >= 2 and fd > worst_fail:
                            worst_fail = fd
                            worst_t = t

                    if worst_t:
                        all_healthy = get_all_healthy(prices, universe_map, date, global_idx)
                        held_set = set(holdings.keys())
                        replacement = None
                        for c in all_healthy:
                            if c not in held_set:
                                replacement = c
                                break

                        if replacement:
                            holdings, cash = swap_one_coin(holdings, cash, worst_t, replacement, prices, date)
                            coin_buy_prices[replacement] = get_price(replacement, prices, date)
                            coin_peaks[replacement] = coin_buy_prices[replacement]
                        else:
                            holdings, cash = sell_single_coin(holdings, cash, worst_t, prices, date)

                        coin_peaks.pop(worst_t, None)
                        coin_buy_prices.pop(worst_t, None)
                        health_fail_days.pop(worst_t, None)
                        month_swap_count += 1
                        daily_swap_count += 1
                        rebal_count += 1

            elif strategy == 'WH':
                # Weekly health check (Monday only)
                if is_monday:
                    worst_t, worst_fail = None, 0
                    for t in list(holdings.keys()):
                        fd = health_fail_days.get(t, 0)
                        if fd >= 2 and fd > worst_fail:
                            worst_fail = fd
                            worst_t = t

                    if worst_t:
                        all_healthy = get_all_healthy(prices, universe_map, date, global_idx)
                        held_set = set(holdings.keys())
                        replacement = None
                        for c in all_healthy:
                            if c not in held_set:
                                replacement = c
                                break

                        if replacement:
                            holdings, cash = swap_one_coin(holdings, cash, worst_t, replacement, prices, date)
                            coin_buy_prices[replacement] = get_price(replacement, prices, date)
                            coin_peaks[replacement] = coin_buy_prices[replacement]
                        else:
                            holdings, cash = sell_single_coin(holdings, cash, worst_t, prices, date)

                        coin_peaks.pop(worst_t, None)
                        coin_buy_prices.pop(worst_t, None)
                        health_fail_days.pop(worst_t, None)
                        daily_swap_count += 1
                        rebal_count += 1

            elif strategy in ('HY3', 'HY5'):
                exit_days = 3 if strategy == 'HY3' else 5
                entry_days = 3 if strategy == 'HY3' else 5

                # Find coin to exit (N consecutive fail days)
                worst_t, worst_fail = None, 0
                for t in list(holdings.keys()):
                    fd = health_fail_days.get(t, 0)
                    if fd >= exit_days and fd > worst_fail:
                        worst_fail = fd
                        worst_t = t

                if worst_t:
                    # Find replacement with N consecutive pass days
                    all_healthy = get_all_healthy(prices, universe_map, date, global_idx)
                    held_set = set(holdings.keys())
                    replacement = None
                    for c in all_healthy:
                        if c not in held_set and health_pass_days.get(c, 0) >= entry_days:
                            replacement = c
                            break

                    if replacement:
                        holdings, cash = swap_one_coin(holdings, cash, worst_t, replacement, prices, date)
                        coin_buy_prices[replacement] = get_price(replacement, prices, date)
                        coin_peaks[replacement] = coin_buy_prices[replacement]
                        coin_peaks.pop(worst_t, None)
                        coin_buy_prices.pop(worst_t, None)
                        health_fail_days.pop(worst_t, None)
                        daily_swap_count += 1
                        rebal_count += 1
                    # If no qualified replacement, do nothing (wait)

            elif strategy in ('S20', 'S25', 'S3'):
                stop_pct = {'S20': 0.20, 'S25': 0.25, 'S3': 0.10}[strategy]
                use_trailing = strategy in ('S20', 'S25')

                coins_to_sell = []
                for t in list(holdings.keys()):
                    p = get_price(t, prices, date)
                    if p <= 0: continue
                    if use_trailing:
                        peak = coin_peaks.get(t, p)
                        if peak > 0 and (p / peak - 1) <= -stop_pct:
                            coins_to_sell.append(t)
                    else:
                        buy_p = coin_buy_prices.get(t, p)
                        if buy_p > 0 and (p / buy_p - 1) <= -stop_pct:
                            coins_to_sell.append(t)

                if coins_to_sell:
                    for t in coins_to_sell:
                        all_healthy = get_all_healthy(prices, universe_map, date, global_idx)
                        held_set = set(holdings.keys())
                        replacement = None
                        for c in all_healthy:
                            if c not in held_set and c not in coins_to_sell:
                                replacement = c
                                break
                        holdings, cash = swap_one_coin(holdings, cash, t, replacement, prices, date)
                        if replacement:
                            coin_buy_prices[replacement] = get_price(replacement, prices, date)
                            coin_peaks[replacement] = coin_buy_prices[replacement]
                        coin_peaks.pop(t, None)
                        coin_buy_prices.pop(t, None)
                        daily_swap_count += 1
                    rebal_count += 1

        # Clean up
        for t in list(coin_buy_prices.keys()):
            if t not in holdings:
                coin_buy_prices.pop(t, None)
                coin_peaks.pop(t, None)

        port_val = get_portfolio_value(holdings, cash, prices, date)
        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    result = pd.DataFrame(portfolio_values).set_index('Date')
    result.attrs['rebal_count'] = rebal_count
    result.attrs['daily_swaps'] = daily_swap_count
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
        ('BASE',  'BASE: 카나리아 매일 + 코인 월간 (현행)'),
        # Cooldown
        ('CD7',   'CD7:  헬스2일퇴출 + 7일 쿨다운'),
        ('CD14',  'CD14: 헬스2일퇴출 + 14일 쿨다운'),
        # Monthly budget
        ('MB1',   'MB1:  헬스2일퇴출 + 월 1회 교체'),
        ('MB2',   'MB2:  헬스2일퇴출 + 월 2회 교체'),
        # Weekly
        ('WH',    'WH:   주간(월) 헬스체크 + 퇴출교체'),
        # Hysteresis
        ('HY3',   'HY3:  3일실패→퇴출 + 3일통과→진입'),
        ('HY5',   'HY5:  5일실패→퇴출 + 5일통과→진입'),
        # Stop-loss (wider)
        ('S3',    'S3:   매수가 -10% 손절+교체'),
        ('S20',   'S20:  트레일링 -20% 스탑+교체'),
        ('S25',   'S25:  트레일링 -25% 스탑+교체'),
    ]

    # ═══ Overall Performance ═══
    print("=" * 120)
    print(f"  DAILY COIN CHECK v2: TURNOVER-CONTROLLED (SMA{SMA_PERIOD} + Vol{int(VOL_CAP*100)}% + Top{N_PICKS})")
    print("=" * 120)

    print(f"\n  {'Strategy':<42} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Final':>12} {'Rebals':>7} {'Swaps':>6}")
    print(f"  {'-'*95}")

    bh = run_btc_bh(prices)
    if bh is not None and len(bh) > 10:
        m = calc_metrics(bh)
        print(f"  {'BTC B&H':<42} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f}")

    results = {}
    for strat_id, label in strategies:
        pv = run_backtest(prices, filtered_map, strategy=strat_id)
        if pv is not None and len(pv) > 10:
            m = calc_metrics(pv)
            rc = pv.attrs.get('rebal_count', 0)
            swaps = pv.attrs.get('daily_swaps', 0)
            results[strat_id] = (m, rc, swaps, pv)
            print(f"  {label:<42} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7} {swaps:>6}")
        else:
            print(f"  {label:<42} {'ERROR':>8}")

    # ═══ Year-by-year ═══
    print("\n" + "=" * 120)
    print("  YEAR-BY-YEAR CAGR")
    print("=" * 120)

    years = list(range(2018, 2026))
    print(f"\n  {'Strategy':<42}", end='')
    for y in years:
        print(f" {y:>8}", end='')
    print(f" {'Overall':>9}")
    print(f"  {'-'*120}")

    for strat_id, label in strategies:
        if strat_id not in results: continue
        m_full, rc, swaps, pv_full = results[strat_id]
        short_label = label[:40]
        print(f"  {short_label:<42}", end='')
        for y in years:
            mask = (pv_full.index >= f'{y}-01-01') & (pv_full.index <= f'{y}-12-31')
            pv_slice = pv_full[mask]
            if len(pv_slice) > 10:
                m = calc_metrics(pv_slice)
                print(f" {m['CAGR']:>+7.1%}", end='')
            else:
                print(f" {'N/A':>8}", end='')
        print(f" {m_full['CAGR']:>+8.1%}")

    # BTC B&H
    print(f"  {'BTC B&H':<42}", end='')
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
        print(f"\n  {'Strategy':<42} {'dSharpe':>8} {'dMDD':>8} {'dCAGR':>8} {'Swaps/Yr':>9} {'Verdict':>10}")
        print(f"  {'-'*90}")
        n_years = 8
        for strat_id, label in strategies:
            if strat_id == 'BASE': continue
            if strat_id not in results: continue
            m, rc, swaps, pv = results[strat_id]
            ds = m['Sharpe'] - base_m['Sharpe']
            dm = (m['MDD'] - base_m['MDD']) * 100
            dc = (m['CAGR'] - base_m['CAGR']) * 100
            syr = swaps / n_years
            if ds > 0.03 and dc > 0:
                verdict = '** BETTER **'
            elif ds > -0.03 and abs(dc) < 3:
                verdict = 'NEUTRAL'
            elif ds < -0.05 or dc < -5:
                verdict = 'WORSE'
            else:
                verdict = 'NEUTRAL'
            print(f"  {label:<42} {ds:>+7.3f} {dm:>+7.1f}pp {dc:>+7.1f}pp {syr:>8.1f} {verdict:>10}")


if __name__ == '__main__':
    main()
