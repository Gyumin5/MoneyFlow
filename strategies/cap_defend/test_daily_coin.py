#!/usr/bin/env python3
"""
Daily Coin Check Strategies Test
Base: SMA150 daily canary + Vol5% health + market cap order Top5 + equal weight

Tests various daily coin management strategies on top of the base.

Category 1 — Defensive Exit (헬스 실패 시 퇴출):
  H1: 헬스 1일 실패 → 매도 + 시총순 교체
  H2: 헬스 2일 연속 실패 → 매도 + 시총순 교체
  H3: 헬스 2일 실패 → 매도만 (현금, 교체 안함)

Category 2 — Asymmetric (비대칭: 빠른 퇴출 + 느린 진입):
  A1: 2일 퇴출 + 5일 연속 통과해야 신규 진입
  A2: 2일 퇴출만 매일 + 진입은 월간만

Category 3 — Stop-Loss (가격 기반 손절):
  S1: 개별 코인 고점 대비 -10% 트레일링 스탑
  S2: 개별 코인 고점 대비 -15% 트레일링 스탑
  S3: 매수가 대비 -10% 절대 손절

Category 4 — Turnover Control (턴오버 제한):
  T1: 하루 최대 1종목 교체
  T2: 주간 리밸런싱 (매주 월요일)

Category 5 — Hybrid:
  X1: H2 + S2 (헬스 2일 OR -15% 스탑)
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
    """Market cap ordered healthy coins, up to N_PICKS."""
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
    """All healthy coins (not limited to N_PICKS)."""
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

    # Pass 1: Sells
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

    # Pass 2: Buys
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
    """Sell a single coin from holdings."""
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
    """Buy a single coin, spending up to target_value from cash."""
    p = get_price(ticker, prices, date)
    if p <= 0 or cash <= 0:
        return holdings, cash
    spend = min(target_value, cash)
    bought_value = spend * (1 - TX_COST)
    new_holdings = dict(holdings)
    new_holdings[ticker] = new_holdings.get(ticker, 0) + bought_value / p
    return new_holdings, cash - spend


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

    # State tracking
    health_fail_days = defaultdict(int)  # ticker -> consecutive fail days
    health_pass_days = defaultdict(int)  # ticker -> consecutive pass days
    coin_peaks = {}  # ticker -> peak price since purchase
    coin_buy_prices = {}  # ticker -> purchase price

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)
        risk_on = check_canary(prices, global_idx)
        currently_invested = len(holdings) > 0

        current_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and current_month != prev_month)
        is_monday = date.weekday() == 0

        # Update coin peaks and health tracking
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

        # Track health pass days for non-held coins too (for A1)
        if strategy == 'A1':
            all_healthy = get_all_healthy(prices, universe_map, date, global_idx)
            for t in all_healthy:
                if t not in holdings:
                    health_pass_days[t] = health_pass_days.get(t, 0) + 1
            # Reset for non-healthy
            uni_tickers = get_universe_for_date(universe_map, date)
            for ut in uni_tickers:
                sym = ut.replace('-USD', '')
                if sym in EXCLUDE_SYMBOLS: continue
                ticker = f"{sym}-USD"
                if ticker not in all_healthy and ticker not in holdings:
                    health_pass_days[ticker] = 0

        # ── Strategy Logic ──
        if strategy == 'BASE':
            # Baseline: canary daily, coins monthly
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
                    picks = get_healthy_picks(prices, universe_map, date, global_idx)
                else:
                    picks = []
                target = compute_target(picks)
                holdings, cash = execute_rebalance(holdings, cash, target, prices, date)
                rebal_count += 1
                # Track buy prices
                for t in holdings:
                    if t not in coin_buy_prices:
                        coin_buy_prices[t] = get_price(t, prices, date)
                        coin_peaks[t] = coin_buy_prices[t]

        elif strategy in ('H1', 'H2', 'H3'):
            fail_threshold = 1 if strategy == 'H1' else 2
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
            elif currently_invested and risk_on:
                # Daily health check on held coins
                coins_to_sell = []
                for t in list(holdings.keys()):
                    if health_fail_days.get(t, 0) >= fail_threshold:
                        coins_to_sell.append(t)

                if coins_to_sell:
                    for t in coins_to_sell:
                        holdings, cash = sell_single_coin(holdings, cash, t, prices, date)
                        health_fail_days.pop(t, None)
                        health_pass_days.pop(t, None)
                        coin_peaks.pop(t, None)
                        coin_buy_prices.pop(t, None)
                    rebal_count += 1

                    # H1/H2: replace with next healthy coin
                    if strategy in ('H1', 'H2') and len(holdings) < N_PICKS:
                        all_healthy = get_all_healthy(prices, universe_map, date, global_idx)
                        held_set = set(holdings.keys())
                        for candidate in all_healthy:
                            if candidate not in held_set and len(holdings) < N_PICKS:
                                port_val = get_portfolio_value(holdings, cash, prices, date)
                                target_val = port_val / N_PICKS
                                holdings, cash = buy_single_coin(holdings, cash, candidate,
                                                                 target_val, prices, date)
                                coin_buy_prices[candidate] = get_price(candidate, prices, date)
                                coin_peaks[candidate] = coin_buy_prices[candidate]
                    # H3: no replacement, stay in cash for that slot

        elif strategy in ('A1', 'A2'):
            do_full_rebal = False

            if i == 0:
                do_full_rebal = True
            elif is_month_change and strategy == 'A1':
                do_full_rebal = True
            elif is_month_change and strategy == 'A2':
                do_full_rebal = True
            elif currently_invested and not risk_on:
                do_full_rebal = True
            elif not currently_invested and risk_on:
                do_full_rebal = True

            if do_full_rebal:
                if risk_on:
                    if strategy == 'A1':
                        # Only pick coins with 5+ pass days
                        all_healthy = get_all_healthy(prices, universe_map, date, global_idx)
                        picks = [t for t in all_healthy if health_pass_days.get(t, 0) >= 5][:N_PICKS]
                        if not picks:
                            picks = get_healthy_picks(prices, universe_map, date, global_idx)
                    else:
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
            elif currently_invested and risk_on:
                # Daily exit check: 2d health fail
                coins_to_sell = []
                for t in list(holdings.keys()):
                    if health_fail_days.get(t, 0) >= 2:
                        coins_to_sell.append(t)

                if coins_to_sell:
                    for t in coins_to_sell:
                        holdings, cash = sell_single_coin(holdings, cash, t, prices, date)
                        health_fail_days.pop(t, None)
                        health_pass_days.pop(t, None)
                        coin_peaks.pop(t, None)
                        coin_buy_prices.pop(t, None)
                    rebal_count += 1

                    if strategy == 'A1':
                        # Replace only with coins that have 5+ consecutive pass days
                        all_healthy = get_all_healthy(prices, universe_map, date, global_idx)
                        held_set = set(holdings.keys())
                        for candidate in all_healthy:
                            if candidate not in held_set and len(holdings) < N_PICKS:
                                if health_pass_days.get(candidate, 0) >= 5:
                                    port_val = get_portfolio_value(holdings, cash, prices, date)
                                    target_val = port_val / N_PICKS
                                    holdings, cash = buy_single_coin(holdings, cash, candidate,
                                                                     target_val, prices, date)
                                    coin_buy_prices[candidate] = get_price(candidate, prices, date)
                                    coin_peaks[candidate] = coin_buy_prices[candidate]
                    # A2: no replacement until monthly

        elif strategy in ('S1', 'S2', 'S3'):
            stop_pct = {'S1': 0.10, 'S2': 0.15, 'S3': 0.10}[strategy]
            use_trailing = strategy in ('S1', 'S2')

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
            elif currently_invested and risk_on:
                # Daily stop-loss check
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
                        holdings, cash = sell_single_coin(holdings, cash, t, prices, date)
                        coin_peaks.pop(t, None)
                        coin_buy_prices.pop(t, None)
                    rebal_count += 1
                    # Replace with next healthy
                    if len(holdings) < N_PICKS:
                        all_healthy = get_all_healthy(prices, universe_map, date, global_idx)
                        held_set = set(holdings.keys())
                        for candidate in all_healthy:
                            if candidate not in held_set and len(holdings) < N_PICKS:
                                port_val = get_portfolio_value(holdings, cash, prices, date)
                                target_val = port_val / N_PICKS
                                holdings, cash = buy_single_coin(holdings, cash, candidate,
                                                                 target_val, prices, date)
                                coin_buy_prices[candidate] = get_price(candidate, prices, date)
                                coin_peaks[candidate] = coin_buy_prices[candidate]

        elif strategy == 'T1':
            # Daily check but max 1 swap per day
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
            elif currently_invested and risk_on:
                # Check if any held coin failed health for 2d
                ideal_picks = get_healthy_picks(prices, universe_map, date, global_idx)
                held_set = set(holdings.keys())
                ideal_set = set(ideal_picks)

                # Find worst held coin to sell (longest health fail)
                to_sell = None
                max_fail = 0
                for t in held_set:
                    if t not in ideal_set and health_fail_days.get(t, 0) >= 2:
                        if health_fail_days[t] > max_fail:
                            max_fail = health_fail_days[t]
                            to_sell = t

                if to_sell:
                    holdings, cash = sell_single_coin(holdings, cash, to_sell, prices, date)
                    coin_peaks.pop(to_sell, None)
                    coin_buy_prices.pop(to_sell, None)
                    rebal_count += 1

                    # Buy one replacement
                    held_set = set(holdings.keys())
                    for candidate in ideal_picks:
                        if candidate not in held_set:
                            port_val = get_portfolio_value(holdings, cash, prices, date)
                            target_val = port_val / N_PICKS
                            holdings, cash = buy_single_coin(holdings, cash, candidate,
                                                             target_val, prices, date)
                            coin_buy_prices[candidate] = get_price(candidate, prices, date)
                            coin_peaks[candidate] = coin_buy_prices[candidate]
                            break

        elif strategy == 'T2':
            # Weekly rebalancing (every Monday)
            do_rebal = False
            if i == 0:
                do_rebal = True
            elif is_monday:
                do_rebal = True
            elif currently_invested and not risk_on:
                do_rebal = True
            elif not currently_invested and risk_on:
                do_rebal = True

            if do_rebal:
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

        elif strategy == 'X1':
            # H2 + S2: health 2d fail OR -15% trailing stop
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
            elif currently_invested and risk_on:
                coins_to_sell = []
                for t in list(holdings.keys()):
                    # Health 2d fail
                    if health_fail_days.get(t, 0) >= 2:
                        coins_to_sell.append(t)
                        continue
                    # Trailing -15%
                    p = get_price(t, prices, date)
                    peak = coin_peaks.get(t, p)
                    if peak > 0 and p > 0 and (p / peak - 1) <= -0.15:
                        coins_to_sell.append(t)

                if coins_to_sell:
                    for t in coins_to_sell:
                        holdings, cash = sell_single_coin(holdings, cash, t, prices, date)
                        health_fail_days.pop(t, None)
                        health_pass_days.pop(t, None)
                        coin_peaks.pop(t, None)
                        coin_buy_prices.pop(t, None)
                    rebal_count += 1

                    # Replace
                    if len(holdings) < N_PICKS:
                        all_healthy = get_all_healthy(prices, universe_map, date, global_idx)
                        held_set = set(holdings.keys())
                        for candidate in all_healthy:
                            if candidate not in held_set and len(holdings) < N_PICKS:
                                port_val = get_portfolio_value(holdings, cash, prices, date)
                                target_val = port_val / N_PICKS
                                holdings, cash = buy_single_coin(holdings, cash, candidate,
                                                                 target_val, prices, date)
                                coin_buy_prices[candidate] = get_price(candidate, prices, date)
                                coin_peaks[candidate] = coin_buy_prices[candidate]

        # Clean up sold coins
        for t in list(coin_buy_prices.keys()):
            if t not in holdings:
                coin_buy_prices.pop(t, None)
                coin_peaks.pop(t, None)

        port_val = get_portfolio_value(holdings, cash, prices, date)
        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    result = pd.DataFrame(portfolio_values).set_index('Date')
    result.attrs['rebal_count'] = rebal_count
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
        ('H1',    'H1: 헬스 1일 실패 → 퇴출+교체'),
        ('H2',    'H2: 헬스 2일 실패 → 퇴출+교체'),
        ('H3',    'H3: 헬스 2일 실패 → 퇴출만(현금)'),
        ('A1',    'A1: 2일 퇴출 + 5일 통과 진입'),
        ('A2',    'A2: 2일 퇴출만 + 진입 월간'),
        ('S1',    'S1: 트레일링 -10% 스탑'),
        ('S2',    'S2: 트레일링 -15% 스탑'),
        ('S3',    'S3: 매수가 -10% 손절'),
        ('T1',    'T1: 하루 1종목 교체 제한'),
        ('T2',    'T2: 주간 리밸런싱'),
        ('X1',    'X1: 헬스2일 OR -15% 스탑'),
    ]

    # ═══ Overall Performance ═══
    print("=" * 115)
    print(f"  DAILY COIN CHECK STRATEGIES (SMA{SMA_PERIOD} + Vol{int(VOL_CAP*100)}% + Top{N_PICKS} 시총순 + 균등)")
    print(f"  Period: {START_DATE} ~ {END_DATE}")
    print("=" * 115)

    print(f"\n  {'Strategy':<40} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Final':>12} {'Rebals':>7}")
    print(f"  {'-'*88}")

    bh = run_btc_bh(prices)
    if bh is not None and len(bh) > 10:
        m = calc_metrics(bh)
        print(f"  {'BTC B&H':<40} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f}")

    results = {}
    for strat_id, label in strategies:
        pv = run_backtest(prices, filtered_map, strategy=strat_id)
        if pv is not None and len(pv) > 10:
            m = calc_metrics(pv)
            rc = pv.attrs.get('rebal_count', 0)
            results[strat_id] = (m, rc, pv)
            print(f"  {label:<40} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7}")
        else:
            print(f"  {label:<40} {'ERROR':>8}")

    # ═══ Year-by-year ═══
    print("\n" + "=" * 115)
    print("  YEAR-BY-YEAR CAGR")
    print("=" * 115)

    years = list(range(2018, 2026))
    print(f"\n  {'Strategy':<40}", end='')
    for y in years:
        print(f" {y:>8}", end='')
    print(f" {'Overall':>9}")
    print(f"  {'-'*115}")

    for strat_id, label in strategies:
        if strat_id not in results: continue
        m_full, rc, pv_full = results[strat_id]
        short_label = label[:38]
        print(f"  {short_label:<40}", end='')
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
    print(f"  {'BTC B&H':<40}", end='')
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
    print("\n" + "=" * 115)
    print("  DELTA vs BASELINE")
    print("=" * 115)

    if 'BASE' in results:
        base_m = results['BASE'][0]
        print(f"\n  {'Strategy':<40} {'dSharpe':>8} {'dMDD':>8} {'dCAGR':>8} {'Verdict':>10}")
        print(f"  {'-'*78}")
        for strat_id, label in strategies:
            if strat_id == 'BASE': continue
            if strat_id not in results: continue
            m = results[strat_id][0]
            ds = m['Sharpe'] - base_m['Sharpe']
            dm = (m['MDD'] - base_m['MDD']) * 100
            dc = (m['CAGR'] - base_m['CAGR']) * 100
            if ds > 0.05 and dc > 0:
                verdict = 'BETTER'
            elif ds < -0.05 or dc < -5:
                verdict = 'WORSE'
            else:
                verdict = 'NEUTRAL'
            print(f"  {label:<40} {ds:>+7.3f} {dm:>+7.1f}pp {dc:>+7.1f}pp {verdict:>10}")


if __name__ == '__main__':
    main()
