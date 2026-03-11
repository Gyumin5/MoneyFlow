#!/usr/bin/env python3
"""
Health Check Margin Test v2 — Daily rebalancing focus

목표: 마진으로 daily picks를 안정화해서 일간 리밸런싱이 월간을 이길 수 있는지 확인.

마진 방식 3가지:
  BOTH:      진입 엄격 + 퇴출 완화 (v1과 동일)
  EXIT_ONLY: 진입은 동일(0%), 퇴출만 완화 (stickiness)
  ENTRY_ONLY: 진입만 엄격, 퇴출은 동일(0%)

리밸런싱:
  BASE:  월간 (마진 효과 확인용 기준선)
  DAILY: 매일 리밸런싱 (핵심 테스트)
  TO30:  턴오버 30% (기존 R2)

마진 수준: 0%, 1%, 2%, 3%, 5%
"""

import os, sys, warnings
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


def check_health_margin(ticker, prices, global_idx, margin, margin_type, is_held):
    """
    margin_type:
      'BOTH'       - 진입 엄격 + 퇴출 완화
      'EXIT_ONLY'  - 진입 동일, 퇴출만 완화 (stickiness)
      'ENTRY_ONLY' - 진입만 엄격, 퇴출 동일
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

    # Determine thresholds based on margin type and held status
    if is_held:
        # Exit thresholds (more lenient = harder to exit)
        if margin_type in ('BOTH', 'EXIT_ONLY'):
            mom_threshold = -margin
            sma_factor = 1 - margin
        else:  # ENTRY_ONLY
            mom_threshold = 0
            sma_factor = 1.0
    else:
        # Entry thresholds (more strict = harder to enter)
        if margin_type in ('BOTH', 'ENTRY_ONLY'):
            mom_threshold = margin
            sma_factor = 1 + margin
        else:  # EXIT_ONLY
            mom_threshold = 0
            sma_factor = 1.0

    return mom21 > mom_threshold and cur > sma30 * sma_factor


def get_healthy_picks(prices, universe_map, date, global_idx,
                      margin, margin_type, current_holdings):
    uni_tickers = get_universe_for_date(universe_map, date)
    uni_clean = [t.replace('-USD', '') for t in uni_tickers
                if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]

    held_set = set(current_holdings.keys())
    healthy = []
    for sym in uni_clean:
        ticker = f"{sym}-USD"
        is_held = ticker in held_set
        if check_health_margin(ticker, prices, global_idx, margin, margin_type, is_held):
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
    turnover = sum(abs(current_weights.get(t, 0) - target_coins.get(t, 0)) for t in all_tickers)
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

    # Sells first
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

    # Then buys
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


def run_backtest(prices, universe_map, margin=0.0, margin_type='BOTH',
                 rebal_policy='BASE'):
    btc = prices.get('BTC-USD')
    if btc is None: return None

    all_dates = btc.index[(btc.index >= START_DATE) & (btc.index <= END_DATE)]
    if len(all_dates) == 0: return None

    holdings = {}
    cash = 10000
    portfolio_values = []
    prev_month = None
    rebal_count = 0
    pick_change_days = 0
    prev_picks_set = set()

    # Turnover threshold
    to_threshold = None
    if rebal_policy.startswith('TO'):
        to_threshold = int(rebal_policy[2:]) / 100.0

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)
        risk_on = check_canary(prices, global_idx)
        currently_invested = len(holdings) > 0
        current_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and current_month != prev_month)

        # Shadow target with margin
        if risk_on:
            shadow_picks = get_healthy_picks(
                prices, universe_map, date, global_idx,
                margin, margin_type, holdings
            )
        else:
            shadow_picks = []
        shadow_target = compute_target(shadow_picks)

        # Track pick changes
        if set(shadow_picks) != prev_picks_set and prev_picks_set:
            pick_change_days += 1
        if shadow_picks:
            prev_picks_set = set(shadow_picks)

        turnover = calc_turnover(holdings, cash, shadow_target, prices, date)

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
        elif rebal_policy == 'DAILY':
            do_rebal = True  # Always rebalance
        elif to_threshold is not None:
            if turnover >= to_threshold:
                do_rebal = True

        if do_rebal:
            holdings, cash = execute_rebalance(holdings, cash, shadow_target, prices, date)
            rebal_count += 1

        port_val = get_portfolio_value(holdings, cash, prices, date)
        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    result = pd.DataFrame(portfolio_values).set_index('Date')
    result.attrs['rebal_count'] = rebal_count
    result.attrs['pick_changes'] = pick_change_days
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
    margin_types = ['EXIT_ONLY', 'BOTH', 'ENTRY_ONLY']
    policies = ['BASE', 'DAILY', 'TO30']

    # BTC B&H
    bh = run_btc_bh(prices)
    m_bh = calc_metrics(bh) if bh is not None else None

    # ═══ Section 1: 마진 방식별 × 리밸런싱 비교 ═══
    print("=" * 130)
    print("  HEALTH MARGIN v2: 일간 리밸런싱이 월간을 이길 수 있는가?")
    print("=" * 130)

    if m_bh:
        print(f"\n  BTC B&H: Sharpe {m_bh['Sharpe']:.3f}  MDD {m_bh['MDD']:.1%}  CAGR {m_bh['CAGR']:+.1%}")

    results = {}

    for mtype in margin_types:
        print(f"\n  {'━'*120}")
        print(f"  마진 방식: {mtype}")
        if mtype == 'EXIT_ONLY':
            print("  (진입=기본, 퇴출=완화 → 보유 코인 stickiness)")
        elif mtype == 'BOTH':
            print("  (진입=엄격, 퇴출=완화 → 양쪽 마진)")
        else:
            print("  (진입=엄격, 퇴출=기본 → 진입 장벽)")
        print(f"  {'━'*120}")

        for policy in policies:
            print(f"\n  리밸런싱: {policy}")
            print(f"  {'Margin':<10} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Final':>12} {'Rebals':>7} {'PickChg':>8}")
            print(f"  {'─'*65}")

            for margin in margins:
                # margin 0% is same for all types
                key = (mtype, policy, margin)
                if margin == 0 and mtype != margin_types[0]:
                    # Reuse margin=0 result from first margin_type
                    ref_key = (margin_types[0], policy, 0)
                    if ref_key in results:
                        results[key] = results[ref_key]
                        m, rc, pc, pv = results[key]
                        print(f"  {margin:>7.0%}   {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7} {pc:>8} ★")
                        continue

                pv = run_backtest(prices, filtered_map, margin=margin,
                                  margin_type=mtype, rebal_policy=policy)
                if pv is not None and len(pv) > 10:
                    m = calc_metrics(pv)
                    rc = pv.attrs.get('rebal_count', 0)
                    pc = pv.attrs.get('pick_changes', 0)
                    results[key] = (m, rc, pc, pv)
                    marker = ' ★' if margin == 0 else ''
                    print(f"  {margin:>7.0%}   {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7} {pc:>8}{marker}")

    # ═══ Section 2: Delta vs BASE 0% ═══
    print("\n" + "=" * 130)
    print("  DELTA vs BASE(월간, 마진 0%)")
    print("=" * 130)

    base_key = (margin_types[0], 'BASE', 0.0)
    if base_key in results:
        base_m = results[base_key][0]
        print(f"\n  기준: BASE 0% — Sharpe {base_m['Sharpe']:.3f}  CAGR {base_m['CAGR']:+.1%}  MDD {base_m['MDD']:.1%}\n")

        print(f"  {'MType':<12} {'Policy':<8} {'Margin':>6} {'Sharpe':>8} {'dSharpe':>8} {'dCAGR':>9} {'dMDD':>9} {'Rebals':>7} {'PickChg':>8} {'Verdict':>10}")
        print(f"  {'─'*96}")

        for mtype in margin_types:
            for policy in policies:
                for margin in margins:
                    key = (mtype, policy, margin)
                    if key == base_key: continue
                    if key not in results: continue
                    # Skip duplicate 0% entries
                    if margin == 0 and (mtype != margin_types[0]):
                        continue
                    m, rc, pc, pv = results[key]
                    ds = m['Sharpe'] - base_m['Sharpe']
                    dc = (m['CAGR'] - base_m['CAGR']) * 100
                    dm = (m['MDD'] - base_m['MDD']) * 100

                    if ds > 0.03 and dc > 0:
                        verdict = '★ BETTER'
                    elif ds < -0.05 or dc < -5:
                        verdict = 'WORSE'
                    else:
                        verdict = 'NEUTRAL'

                    print(f"  {mtype:<12} {policy:<8} {margin:>5.0%} {m['Sharpe']:>8.3f} {ds:>+7.3f} {dc:>+8.1f}pp {dm:>+8.1f}pp {rc:>7} {pc:>8} {verdict:>10}")

    # ═══ Section 3: Best DAILY vs BASE summary ═══
    print("\n" + "=" * 130)
    print("  BEST DAILY REBALANCING vs BASE (핵심 비교)")
    print("=" * 130)

    if base_key in results:
        base_m = results[base_key][0]
        print(f"\n  {'Label':<30} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Final':>12} {'Rebals':>7} {'PickChg':>8}")
        print(f"  {'─'*85}")
        # BASE 0%
        m, rc, pc, _ = results[base_key]
        print(f"  {'BASE 0% (현행 월간)':<30} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7} {pc:>8}")

        # Find best DAILY across all margin types and margins
        best_daily = None
        best_daily_sharpe = -999
        for mtype in margin_types:
            for margin in margins:
                key = (mtype, 'DAILY', margin)
                if key in results:
                    m_d = results[key][0]
                    if m_d['Sharpe'] > best_daily_sharpe:
                        best_daily_sharpe = m_d['Sharpe']
                        best_daily = (mtype, margin)

        if best_daily:
            mt, mg = best_daily
            key = (mt, 'DAILY', mg)
            m, rc, pc, _ = results[key]
            label = f"DAILY best ({mt} {mg:.0%})"
            print(f"  {label:<30} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7} {pc:>8}")

        # DAILY 0% for reference
        key0 = (margin_types[0], 'DAILY', 0.0)
        if key0 in results:
            m, rc, pc, _ = results[key0]
            print(f"  {'DAILY 0% (마진 없는 일간)':<30} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7} {pc:>8}")


if __name__ == '__main__':
    main()
