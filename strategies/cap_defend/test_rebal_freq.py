#!/usr/bin/env python3
"""
Rebalancing Frequency Test — tx_cost=0 vs tx_cost=0.004

목적: 일간 리밸런싱이 월간보다 못한 이유가 거래비용인지 로직(모멘텀 중단)인지 분리.

카나리아: BTC > SMA150, 매일 체크, ON/OFF 즉시 행동 (모든 전략 동일)
헬스체크: SMA30 + Mom21 + Vol5%, 매일 계산 (모든 전략 동일)
변수: 실제 매매 빈도 — 월간 / 격주(2주) / 주간 / 3일 / 일간

테스트:
  1) tx_cost=0 빈도 스윕 (비용 없이 순수 빈도 효과)
  2) tx_cost=0.004 빈도 스윕 (비용 포함)
  3) 코드 검증 — 특정 날짜 picks 로그
  4) DAILY 상세 — 매일 실제 교체 코인 수, 매매 금액
"""

import os, sys, warnings
from collections import defaultdict
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


def check_health(ticker, prices, global_idx):
    if ticker not in prices: return False
    close = prices[ticker]['Close'].iloc[:global_idx+1]
    if len(close) < 90: return False
    cur = close.iloc[-1]
    sma30 = close.rolling(30).mean().iloc[-1]
    mom21 = calc_ret(close, 21)
    vol90 = get_volatility(close, 90)
    return cur > sma30 and mom21 is not None and mom21 > 0 and vol90 is not None and vol90 <= VOL_CAP


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


def execute_rebalance(holdings, cash, weights, prices, date, tx_cost):
    current_values = {}
    port_val = cash
    for t, units in holdings.items():
        p = get_price(t, prices, date)
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
            new_cash += sell_amount * (1 - tx_cost)
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


def run_backtest(prices, universe_map, freq='monthly', tx_cost=0.004,
                 detail_log=False):
    """
    freq: 'monthly', 'biweekly', 'weekly', '3day', 'daily'
    """
    btc = prices.get('BTC-USD')
    if btc is None: return None, []

    all_dates = btc.index[(btc.index >= START_DATE) & (btc.index <= END_DATE)]
    if len(all_dates) == 0: return None, []

    holdings = {}
    cash = 10000
    portfolio_values = []
    prev_month = None
    rebal_count = 0
    days_since_rebal = 0
    trade_log = []  # (date, dropped, added, trade_value)
    total_trade_value = 0

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)
        risk_on = check_canary(prices, global_idx)
        currently_invested = len(holdings) > 0
        current_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and current_month != prev_month)

        # Shadow picks (always computed daily)
        if risk_on:
            shadow_picks = get_healthy_picks(prices, universe_map, date, global_idx)
        else:
            shadow_picks = []
        target = compute_target(shadow_picks)

        # Rebalance decision
        do_rebal = False
        reason = ''

        if i == 0:
            do_rebal, reason = True, 'init'
        elif currently_invested and not risk_on:
            do_rebal, reason = True, 'canary_off'
        elif not currently_invested and risk_on:
            do_rebal, reason = True, 'canary_on'
        else:
            # Frequency-based rebalancing
            if freq == 'monthly' and is_month_change:
                do_rebal, reason = True, 'monthly'
            elif freq == 'biweekly' and days_since_rebal >= 10:  # ~2 weeks trading days
                do_rebal, reason = True, 'biweekly'
            elif freq == 'weekly' and days_since_rebal >= 5:
                do_rebal, reason = True, 'weekly'
            elif freq == '3day' and days_since_rebal >= 3:
                do_rebal, reason = True, '3day'
            elif freq == 'daily':
                do_rebal, reason = True, 'daily'

        if do_rebal:
            # Log what changed
            old_set = set(holdings.keys())
            new_set = set(t for t in target if t != 'CASH')
            dropped = old_set - new_set
            added = new_set - old_set

            # Calculate trade value before rebalance
            port_val = get_portfolio_value(holdings, cash, prices, date)
            old_holdings = dict(holdings)

            holdings, cash = execute_rebalance(holdings, cash, target, prices, date, tx_cost)
            rebal_count += 1
            days_since_rebal = 0

            # Estimate trade value (sum of |delta| for each position)
            trade_val = 0
            for t in set(list(old_holdings.keys()) + list(holdings.keys())):
                p = get_price(t, prices, date)
                old_units = old_holdings.get(t, 0)
                new_units = holdings.get(t, 0)
                trade_val += abs(new_units - old_units) * p
            total_trade_value += trade_val

            if detail_log and (dropped or added):
                trade_log.append({
                    'date': date,
                    'dropped': [t.replace('-USD','') for t in dropped],
                    'added': [t.replace('-USD','') for t in added],
                    'trade_value': trade_val,
                    'port_value': port_val,
                    'reason': reason,
                })
        else:
            days_since_rebal += 1

        port_val = get_portfolio_value(holdings, cash, prices, date)
        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    result = pd.DataFrame(portfolio_values).set_index('Date')
    result.attrs['rebal_count'] = rebal_count
    result.attrs['total_trade_value'] = total_trade_value
    return result, trade_log


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

    freqs = ['monthly', 'biweekly', 'weekly', '3day', 'daily']
    freq_labels = {
        'monthly': '월간',
        'biweekly': '격주(10일)',
        'weekly': '주간(5일)',
        '3day': '3일',
        'daily': '일간',
    }

    # BTC B&H
    bh = run_btc_bh(prices)
    m_bh = calc_metrics(bh) if bh is not None else None

    # ═══ Test 1: tx_cost=0 빈도 스윕 ═══
    print("=" * 130)
    print("  TEST 1: tx_cost=0 (거래비용 없음) — 순수 리밸런싱 빈도 효과")
    print("=" * 130)
    print(f"\n  {'Freq':<15} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Final':>12} {'Rebals':>7} {'TradeVol':>12}")
    print(f"  {'─'*75}")

    results_free = {}
    for freq in freqs:
        pv, _ = run_backtest(prices, filtered_map, freq=freq, tx_cost=0.0)
        if pv is not None and len(pv) > 10:
            m = calc_metrics(pv)
            rc = pv.attrs.get('rebal_count', 0)
            tv = pv.attrs.get('total_trade_value', 0)
            results_free[freq] = (m, rc, tv, pv)
            label = f"{freq_labels[freq]}"
            print(f"  {label:<15} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7} ${tv:>10,.0f}")

    if m_bh:
        print(f"  {'BTC B&H':<15} {m_bh['Sharpe']:>8.3f} {m_bh['MDD']:>7.1%} {m_bh['CAGR']:>+7.1%} {m_bh['Final']:>11,.0f}")

    # ═══ Test 2: tx_cost=0.004 빈도 스윕 ═══
    print("\n" + "=" * 130)
    print("  TEST 2: tx_cost=0.004 (거래비용 있음) — 비용 포함 빈도 효과")
    print("=" * 130)
    print(f"\n  {'Freq':<15} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Final':>12} {'Rebals':>7} {'TradeVol':>12}")
    print(f"  {'─'*75}")

    results_cost = {}
    for freq in freqs:
        pv, _ = run_backtest(prices, filtered_map, freq=freq, tx_cost=0.004)
        if pv is not None and len(pv) > 10:
            m = calc_metrics(pv)
            rc = pv.attrs.get('rebal_count', 0)
            tv = pv.attrs.get('total_trade_value', 0)
            results_cost[freq] = (m, rc, tv, pv)
            label = f"{freq_labels[freq]}"
            print(f"  {label:<15} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Final']:>11,.0f} {rc:>7} ${tv:>10,.0f}")

    # ═══ Test 1 vs Test 2 비교 ═══
    print("\n" + "=" * 130)
    print("  비용 영향 분석: tx=0 vs tx=0.004")
    print("=" * 130)
    print(f"\n  {'Freq':<15} {'CAGR(tx=0)':>12} {'CAGR(tx=.4%)':>14} {'비용손실':>10} {'Sharpe(0)':>10} {'Sharpe(.4%)':>12}")
    print(f"  {'─'*78}")

    for freq in freqs:
        if freq in results_free and freq in results_cost:
            mf = results_free[freq][0]
            mc = results_cost[freq][0]
            cost_loss = (mf['CAGR'] - mc['CAGR']) * 100
            label = freq_labels[freq]
            print(f"  {label:<15} {mf['CAGR']:>+11.1%} {mc['CAGR']:>+13.1%} {cost_loss:>+9.1f}pp {mf['Sharpe']:>10.3f} {mc['Sharpe']:>12.3f}")

    # ═══ Year-by-year (tx=0) ═══
    print("\n" + "=" * 130)
    print("  YEAR-BY-YEAR CAGR (tx_cost=0)")
    print("=" * 130)

    years = list(range(2018, 2026))
    print(f"\n  {'Freq':<15}", end='')
    for y in years:
        print(f" {y:>8}", end='')
    print(f" {'Overall':>9}")
    print(f"  {'─'*95}")

    for freq in freqs:
        if freq not in results_free: continue
        m_full, _, _, pv_full = results_free[freq]
        label = freq_labels[freq]
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
    if bh is not None and m_bh is not None:
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

    # ═══ Year-by-year (tx=0.004) ═══
    print("\n" + "=" * 130)
    print("  YEAR-BY-YEAR CAGR (tx_cost=0.004)")
    print("=" * 130)

    print(f"\n  {'Freq':<15}", end='')
    for y in years:
        print(f" {y:>8}", end='')
    print(f" {'Overall':>9}")
    print(f"  {'─'*95}")

    for freq in freqs:
        if freq not in results_cost: continue
        m_full, _, _, pv_full = results_cost[freq]
        label = freq_labels[freq]
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

    # ═══ Test 4: DAILY 상세 로그 (처음 30건) ═══
    print("\n" + "=" * 130)
    print("  DAILY 리밸런싱 상세 로그 (tx=0, 코인 교체 발생한 처음 30건)")
    print("=" * 130)

    _, trade_log = run_backtest(prices, filtered_map, freq='daily', tx_cost=0.0,
                                detail_log=True)

    print(f"\n  {'Date':<12} {'Reason':<12} {'Dropped':<25} {'Added':<25} {'TradeVal':>10} {'PortVal':>10}")
    print(f"  {'─'*100}")
    shown = 0
    for entry in trade_log:
        if shown >= 30: break
        dropped = ','.join(entry['dropped']) if entry['dropped'] else '-'
        added = ','.join(entry['added']) if entry['added'] else '-'
        print(f"  {entry['date'].strftime('%Y-%m-%d'):<12} {entry['reason']:<12} {dropped:<25} {added:<25} ${entry['trade_value']:>9,.0f} ${entry['port_value']:>9,.0f}")
        shown += 1

    print(f"\n  총 코인교체 이벤트: {len(trade_log)}건")

    # ═══ Test 5: 코드 검증 — 특정 월초 picks 비교 ═══
    print("\n" + "=" * 130)
    print("  코드 검증: 2021년 월초 picks (매일 계산 vs 실제 리밸런싱)")
    print("=" * 130)

    btc_df = prices['BTC-USD']
    dates_2021 = btc_df.index[(btc_df.index >= '2021-01-01') & (btc_df.index <= '2021-12-31')]

    prev_month_v = None
    print(f"\n  {'Date':<12} {'Canary':<8} {'Healthy Picks':<60}")
    print(f"  {'─'*80}")

    for date in dates_2021:
        cm = date.strftime('%Y-%m')
        if cm == prev_month_v:
            continue
        prev_month_v = cm
        global_idx = btc_df.index.get_loc(date)
        risk_on = check_canary(prices, global_idx)
        picks = get_healthy_picks(prices, filtered_map, date, global_idx) if risk_on else []
        picks_str = ', '.join(t.replace('-USD','') for t in picks) if picks else '(cash)'
        print(f"  {date.strftime('%Y-%m-%d'):<12} {'ON' if risk_on else 'OFF':<8} {picks_str}")


if __name__ == '__main__':
    main()
