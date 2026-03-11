#!/usr/bin/env python3
"""
Parameter sensitivity test for coin strategy improvements.
Tests: Market Breadth threshold, Dual MA periods, and their combinations.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from itertools import product

warnings.filterwarnings('ignore')

# Reuse core functions from main backtest
sys.path.insert(0, os.path.dirname(__file__))
from backtest_coin_strategy import (
    load_universe, load_all_prices, calc_metrics,
    calc_sharpe, calc_rsi, calc_macd_hist, calc_bb_pctb,
    calc_ret, get_volatility, STABLECOINS
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def run_param_backtest(prices, universe_map,
                       breadth_threshold=0.40,
                       short_ma=10, long_ma=30,
                       canary_type='market_breadth',  # 'btc_sma50' or 'market_breadth'
                       health_type='dual_ma',  # 'baseline' or 'dual_ma'
                       start_date='2019-01-01', end_date='2025-12-31',
                       initial_capital=10000, tx_cost=0.002):
    """Run backtest with configurable parameters."""

    btc = prices.get('BTC-USD')
    if btc is None: return None

    all_dates = btc.index[(btc.index >= start_date) & (btc.index <= end_date)]
    if len(all_dates) == 0: return None

    capital = initial_capital
    holdings = {}
    cash = initial_capital
    portfolio_values = []
    prev_month = None

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)

        # Mark to market
        port_val = cash
        for t, units in holdings.items():
            if t in prices:
                idx = prices[t].index.get_indexer([date], method='ffill')[0]
                if idx >= 0:
                    port_val += units * prices[t]['Close'].iloc[idx]

        current_month = date.strftime('%Y-%m')
        should_rebal = (prev_month is not None and current_month != prev_month)

        if should_rebal or (i == 0):
            month_key = date.strftime('%Y-%m') + '-01'
            uni_tickers = []
            for mk in sorted(universe_map.keys(), reverse=True):
                if mk <= month_key:
                    uni_tickers = universe_map[mk]
                    break
            if not uni_tickers and universe_map:
                uni_tickers = list(universe_map.values())[0]

            uni_clean = [t.replace('-USD', '') for t in uni_tickers
                        if t.replace('-USD', '') not in STABLECOINS]

            # === CANARY ===
            risk_on = False
            if canary_type == 'btc_sma50':
                btc_close = btc['Close'].iloc[:global_idx+1]
                if len(btc_close) >= 50:
                    risk_on = btc_close.iloc[-1] > btc_close.rolling(50).mean().iloc[-1]
            elif canary_type == 'market_breadth':
                count_above = 0
                count_total = 0
                for sym in uni_clean:
                    ticker = f"{sym}-USD"
                    if ticker not in prices: continue
                    p = prices[ticker]['Close'].iloc[:global_idx+1]
                    if len(p) < 50: continue
                    count_total += 1
                    if p.iloc[-1] > p.rolling(50).mean().iloc[-1]:
                        count_above += 1
                if count_total > 0:
                    risk_on = (count_above / count_total) > breadth_threshold

            if risk_on:
                # === HEALTH FILTER ===
                healthy = []
                for sym in uni_clean:
                    ticker = f"{sym}-USD"
                    if ticker not in prices: continue
                    df = prices[ticker]
                    close = df['Close'].iloc[:global_idx+1]
                    if len(close) < max(long_ma, 90): continue

                    cur = close.iloc[-1]

                    if health_type == 'baseline':
                        sma30 = close.rolling(30).mean().iloc[-1]
                        mom21 = calc_ret(close, 21)
                        vol90 = get_volatility(close, 90)
                        if cur > sma30 and mom21 > 0 and vol90 <= 0.10:
                            healthy.append(ticker)
                    elif health_type == 'dual_ma':
                        sma_s = close.rolling(short_ma).mean().iloc[-1]
                        sma_l = close.rolling(long_ma).mean().iloc[-1]
                        if cur > sma_s and sma_s > sma_l:
                            healthy.append(ticker)

                if healthy:
                    # Score: Sharpe + bonuses (baseline scoring)
                    scores = []
                    for t in healthy:
                        close = prices[t]['Close'].iloc[:global_idx+1]
                        if len(close) < 252: continue
                        base = calc_sharpe(close, 126) + calc_sharpe(close, 252)
                        rsi = calc_rsi(close)
                        macd_h = calc_macd_hist(close)
                        pctb = calc_bb_pctb(close)
                        if pd.notna(rsi) and 45 <= rsi <= 70: base += 0.2
                        if pd.notna(macd_h) and macd_h > 0: base += 0.2
                        if pd.notna(pctb) and pctb > 0.5: base += 0.2
                        scores.append((t, base))

                    scores.sort(key=lambda x: x[1], reverse=True)
                    picks = [t for t, _ in scores[:5]]

                    if picks:
                        # Inverse vol90 weighting
                        vols = {}
                        for t in picks:
                            close = prices[t]['Close'].iloc[:global_idx+1]
                            v = get_volatility(close, 90)
                            if v > 0: vols[t] = v
                        if vols:
                            inv_vols = {t: 1.0/v for t, v in vols.items()}
                            total = sum(inv_vols.values())
                            weights = {t: w/total for t, w in inv_vols.items()}
                        else:
                            weights = {t: 1.0/len(picks) for t in picks}

                        # Rebalance
                        sell_value = sum(
                            units * prices[t]['Close'].iloc[prices[t].index.get_indexer([date], method='ffill')[0]]
                            for t, units in holdings.items() if t in prices and prices[t].index.get_indexer([date], method='ffill')[0] >= 0
                        )
                        total_value = (cash + sell_value) * (1 - tx_cost)

                        holdings = {}
                        cash = 0
                        for t, w in weights.items():
                            if t in prices:
                                idx2 = prices[t].index.get_indexer([date], method='ffill')[0]
                                if idx2 >= 0:
                                    price = prices[t]['Close'].iloc[idx2]
                                    if price > 0:
                                        alloc = total_value * w * (1 - tx_cost)
                                        holdings[t] = alloc / price
                                        cash += total_value * w - alloc
                    else:
                        sell_value = sum(
                            units * prices[t]['Close'].iloc[prices[t].index.get_indexer([date], method='ffill')[0]]
                            for t, units in holdings.items() if t in prices and prices[t].index.get_indexer([date], method='ffill')[0] >= 0
                        )
                        cash = (cash + sell_value) * (1 - tx_cost)
                        holdings = {}
                else:
                    sell_value = sum(
                        units * prices[t]['Close'].iloc[prices[t].index.get_indexer([date], method='ffill')[0]]
                        for t, units in holdings.items() if t in prices and prices[t].index.get_indexer([date], method='ffill')[0] >= 0
                    )
                    cash = (cash + sell_value) * (1 - tx_cost)
                    holdings = {}
            else:
                sell_value = sum(
                    units * prices[t]['Close'].iloc[prices[t].index.get_indexer([date], method='ffill')[0]]
                    for t, units in holdings.items() if t in prices and prices[t].index.get_indexer([date], method='ffill')[0] >= 0
                )
                if sell_value > 0 or holdings:
                    cash = (cash + sell_value) * (1 - tx_cost)
                    holdings = {}

        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    return pd.DataFrame(portfolio_values).set_index('Date')


def main():
    print("=" * 100)
    print("  PARAMETER SENSITIVITY TEST")
    print("=" * 100)

    # Load data
    print("\nLoading data...")
    universe_map = load_universe()
    all_tickers = set()
    for month_tickers in universe_map.values():
        for t in month_tickers:
            sym = t.replace('-USD', '')
            if sym not in STABLECOINS:
                all_tickers.add(t)
    all_tickers.add('BTC-USD')
    all_tickers.add('ETH-USD')
    prices = load_all_prices(all_tickers)
    print(f"  {len(prices)} tickers loaded\n")

    # ═══ TEST 1: Market Breadth Threshold ═══
    print("=" * 100)
    print("  TEST 1: Market Breadth Threshold (with baseline health)")
    print("=" * 100)

    breadth_thresholds = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]

    print(f"\n  {'Threshold':>10} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Sortino':>8}")
    print(f"  {'-'*50}")

    breadth_results = {}
    for thr in breadth_thresholds:
        pv = run_param_backtest(prices, universe_map,
                                breadth_threshold=thr,
                                canary_type='market_breadth',
                                health_type='baseline')
        if pv is not None and len(pv) > 0:
            m = calc_metrics(pv)
            breadth_results[thr] = m
            marker = ' <<<' if m['Sharpe'] == max(r['Sharpe'] for r in breadth_results.values()) else ''
            print(f"  {thr:>9.0%} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Sortino']:>8.3f}{marker}")

    best_breadth = max(breadth_results.items(), key=lambda x: x[1]['Sharpe'])
    print(f"\n  >> Best threshold: {best_breadth[0]:.0%} (Sharpe {best_breadth[1]['Sharpe']:.3f})")

    # ═══ TEST 2: Dual MA Periods ═══
    print("\n" + "=" * 100)
    print("  TEST 2: Dual MA Periods (with BTC>SMA50 canary)")
    print("=" * 100)

    ma_combos = [
        (5, 20), (5, 30), (5, 50),
        (10, 20), (10, 30), (10, 50),
        (15, 30), (15, 50),
        (20, 50), (20, 60),
        (30, 60), (30, 90),
    ]

    print(f"\n  {'Short/Long':>12} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Sortino':>8}")
    print(f"  {'-'*50}")

    ma_results = {}
    for short, long in ma_combos:
        pv = run_param_backtest(prices, universe_map,
                                short_ma=short, long_ma=long,
                                canary_type='btc_sma50',
                                health_type='dual_ma')
        if pv is not None and len(pv) > 0:
            m = calc_metrics(pv)
            ma_results[(short, long)] = m
            print(f"  {f'SMA{short}/SMA{long}':>12} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Sortino']:>8.3f}")

    best_ma = max(ma_results.items(), key=lambda x: x[1]['Sharpe'])
    print(f"\n  >> Best MA combo: SMA{best_ma[0][0]}/SMA{best_ma[0][1]} (Sharpe {best_ma[1]['Sharpe']:.3f})")

    # ═══ TEST 3: Combined - Best Breadth × Best MA combos ═══
    print("\n" + "=" * 100)
    print("  TEST 3: Combined (Breadth × Dual MA)")
    print("=" * 100)

    # Test top 3 breadth × top 5 MA combos
    top_breadths = sorted(breadth_results.items(), key=lambda x: x[1]['Sharpe'], reverse=True)[:4]
    top_mas = sorted(ma_results.items(), key=lambda x: x[1]['Sharpe'], reverse=True)[:5]

    print(f"\n  {'Breadth':>8} {'MA':>12} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Sortino':>8}")
    print(f"  {'-'*60}")

    combined_results = {}
    for thr, _ in top_breadths:
        for (short, long), _ in top_mas:
            pv = run_param_backtest(prices, universe_map,
                                    breadth_threshold=thr,
                                    short_ma=short, long_ma=long,
                                    canary_type='market_breadth',
                                    health_type='dual_ma')
            if pv is not None and len(pv) > 0:
                m = calc_metrics(pv)
                combined_results[(thr, short, long)] = m
                print(f"  {thr:>7.0%} {f'SMA{short}/{long}':>12} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {m['Sortino']:>8.3f}")

    best_combined = max(combined_results.items(), key=lambda x: x[1]['Sharpe'])
    thr, short, long = best_combined[0]
    m = best_combined[1]

    print(f"\n  >> BEST OVERALL: Breadth>{thr:.0%}, SMA{short}/SMA{long}")
    print(f"     Sharpe={m['Sharpe']:.3f}, MDD={m['MDD']:.1%}, CAGR={m['CAGR']:+.1%}, Sortino={m['Sortino']:.3f}")

    # ═══ TEST 4: Robustness check - vary best params ±10-20% ═══
    print("\n" + "=" * 100)
    print(f"  TEST 4: Robustness (around best: Breadth>{thr:.0%}, SMA{short}/SMA{long})")
    print("=" * 100)

    # Nearby breadth values
    nearby_breadths = sorted(set([max(0.1, thr-0.10), max(0.1, thr-0.05), thr, min(0.7, thr+0.05), min(0.7, thr+0.10)]))
    # Nearby MA values
    nearby_shorts = sorted(set([max(3, short-5), short, short+5]))
    nearby_longs = sorted(set([max(10, long-10), long, long+10]))

    print(f"\n  {'Breadth':>8} {'MA':>12} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8}")
    print(f"  {'-'*50}")

    robust_results = {}
    for b in nearby_breadths:
        for s in nearby_shorts:
            for l in nearby_longs:
                if s >= l: continue
                key = (b, s, l)
                if key in combined_results:
                    robust_results[key] = combined_results[key]
                    m = combined_results[key]
                else:
                    pv = run_param_backtest(prices, universe_map,
                                            breadth_threshold=b,
                                            short_ma=s, long_ma=l,
                                            canary_type='market_breadth',
                                            health_type='dual_ma')
                    if pv is not None and len(pv) > 0:
                        m = calc_metrics(pv)
                        robust_results[key] = m
                    else:
                        continue

                marker = ' <<<' if key == (thr, short, long) else ''
                print(f"  {b:>7.0%} {f'SMA{s}/{l}':>12} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%}{marker}")

    # Check robustness: how many nearby params beat baseline?
    baseline_sharpe = 0.905  # from previous test
    robust_sharpes = [m['Sharpe'] for m in robust_results.values()]
    above_baseline = sum(1 for s in robust_sharpes if s > baseline_sharpe)

    print(f"\n  Robustness: {above_baseline}/{len(robust_sharpes)} nearby params beat Baseline (Sharpe>{baseline_sharpe:.3f})")
    print(f"  Sharpe range: {min(robust_sharpes):.3f} ~ {max(robust_sharpes):.3f}")
    print(f"  Mean Sharpe: {np.mean(robust_sharpes):.3f}")

    # ═══ BASELINE COMPARISON ═══
    print("\n" + "=" * 100)
    print("  FINAL COMPARISON")
    print("=" * 100)

    pv_baseline = run_param_backtest(prices, universe_map,
                                     canary_type='btc_sma50', health_type='baseline')
    m_base = calc_metrics(pv_baseline) if pv_baseline is not None else {'Sharpe': 0, 'MDD': 0, 'CAGR': 0, 'Sortino': 0}

    m_best = best_combined[1]
    thr, short, long = best_combined[0]

    print(f"\n  {'':>25} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'Sortino':>8}")
    print(f"  {'-'*60}")
    print(f"  {'Baseline (현재 V13)':<25} {m_base['Sharpe']:>8.3f} {m_base['MDD']:>7.1%} {m_base['CAGR']:>+7.1%} {m_base['Sortino']:>8.3f}")
    print(f"  {f'Best (B>{thr:.0%},SMA{short}/{long})':<25} {m_best['Sharpe']:>8.3f} {m_best['MDD']:>7.1%} {m_best['CAGR']:>+7.1%} {m_best['Sortino']:>8.3f}")
    print(f"\n  Improvement: Sharpe +{m_best['Sharpe']-m_base['Sharpe']:.3f}, MDD {m_best['MDD']-m_base['MDD']:+.1%}p, CAGR {m_best['CAGR']-m_base['CAGR']:+.1%}p")
    print()


if __name__ == '__main__':
    main()
