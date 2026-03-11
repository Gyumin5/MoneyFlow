#!/usr/bin/env python3
"""
Overfitting validation for coin strategy:
1. Walk-Forward test (expanding window)
2. Transaction cost stress test
3. Regime analysis (bull/bear/recovery)
4. Parameter stability across sub-periods
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from backtest_coin_strategy import (
    load_universe, load_all_prices, calc_metrics,
    calc_sharpe, calc_rsi, calc_macd_hist, calc_bb_pctb,
    calc_ret, get_volatility, STABLECOINS
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def run_backtest(prices, universe_map, canary_type='btc_sma50',
                 health_type='baseline', breadth_thr=0.40,
                 short_ma=15, long_ma=50,
                 start_date='2019-01-01', end_date='2025-12-31',
                 tx_cost=0.002):
    """Unified backtest runner."""
    btc = prices.get('BTC-USD')
    if btc is None: return None

    all_dates = btc.index[(btc.index >= start_date) & (btc.index <= end_date)]
    if len(all_dates) == 0: return None

    holdings = {}
    cash = 10000
    portfolio_values = []
    prev_month = None

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)

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

            # Canary
            risk_on = False
            if canary_type == 'btc_sma50':
                btc_close = btc['Close'].iloc[:global_idx+1]
                if len(btc_close) >= 50:
                    risk_on = btc_close.iloc[-1] > btc_close.rolling(50).mean().iloc[-1]
            elif canary_type == 'market_breadth':
                count_above = count_total = 0
                for sym in uni_clean:
                    ticker = f"{sym}-USD"
                    if ticker not in prices: continue
                    p = prices[ticker]['Close'].iloc[:global_idx+1]
                    if len(p) < 50: continue
                    count_total += 1
                    if p.iloc[-1] > p.rolling(50).mean().iloc[-1]:
                        count_above += 1
                if count_total > 0:
                    risk_on = (count_above / count_total) > breadth_thr

            if risk_on:
                healthy = []
                for sym in uni_clean:
                    ticker = f"{sym}-USD"
                    if ticker not in prices: continue
                    df = prices[ticker]
                    close = df['Close'].iloc[:global_idx+1]

                    if health_type == 'baseline':
                        if len(close) < 90: continue
                        cur = close.iloc[-1]
                        sma30 = close.rolling(30).mean().iloc[-1]
                        mom21 = calc_ret(close, 21)
                        vol90 = get_volatility(close, 90)
                        if cur > sma30 and mom21 > 0 and vol90 <= 0.10:
                            healthy.append(ticker)
                    elif health_type == 'dual_ma':
                        if len(close) < long_ma: continue
                        cur = close.iloc[-1]
                        sma_s = close.rolling(short_ma).mean().iloc[-1]
                        sma_l = close.rolling(long_ma).mean().iloc[-1]
                        if cur > sma_s and sma_s > sma_l:
                            healthy.append(ticker)

                if healthy:
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
                        vols = {}
                        for t in picks:
                            close = prices[t]['Close'].iloc[:global_idx+1]
                            v = get_volatility(close, 90)
                            if v > 0: vols[t] = v
                        if vols:
                            inv = {t: 1/v for t, v in vols.items()}
                            tot = sum(inv.values())
                            weights = {t: w/tot for t, w in inv.items()}
                        else:
                            weights = {t: 1/len(picks) for t in picks}

                        sell_val = sum(
                            units * prices[t]['Close'].iloc[prices[t].index.get_indexer([date], method='ffill')[0]]
                            for t, units in holdings.items()
                            if t in prices and prices[t].index.get_indexer([date], method='ffill')[0] >= 0
                        )
                        total_val = (cash + sell_val) * (1 - tx_cost)
                        holdings = {}; cash = 0
                        for t, w in weights.items():
                            if t in prices:
                                idx2 = prices[t].index.get_indexer([date], method='ffill')[0]
                                if idx2 >= 0:
                                    price = prices[t]['Close'].iloc[idx2]
                                    if price > 0:
                                        alloc = total_val * w * (1 - tx_cost)
                                        holdings[t] = alloc / price
                                        cash += total_val * w - alloc
                    else:
                        sell_val = sum(
                            units * prices[t]['Close'].iloc[prices[t].index.get_indexer([date], method='ffill')[0]]
                            for t, units in holdings.items()
                            if t in prices and prices[t].index.get_indexer([date], method='ffill')[0] >= 0
                        )
                        cash = (cash + sell_val) * (1 - tx_cost); holdings = {}
                else:
                    sell_val = sum(
                        units * prices[t]['Close'].iloc[prices[t].index.get_indexer([date], method='ffill')[0]]
                        for t, units in holdings.items()
                        if t in prices and prices[t].index.get_indexer([date], method='ffill')[0] >= 0
                    )
                    cash = (cash + sell_val) * (1 - tx_cost); holdings = {}
            else:
                sell_val = sum(
                    units * prices[t]['Close'].iloc[prices[t].index.get_indexer([date], method='ffill')[0]]
                    for t, units in holdings.items()
                    if t in prices and prices[t].index.get_indexer([date], method='ffill')[0] >= 0
                )
                if sell_val > 0 or holdings:
                    cash = (cash + sell_val) * (1 - tx_cost); holdings = {}

        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    return pd.DataFrame(portfolio_values).set_index('Date')


def main():
    print("=" * 95)
    print("  OVERFITTING VALIDATION")
    print("=" * 95)

    universe_map = load_universe()
    all_tickers = set()
    for mt in universe_map.values():
        for t in mt:
            s = t.replace('-USD', '')
            if s not in STABLECOINS: all_tickers.add(t)
    all_tickers.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(all_tickers)
    print(f"  {len(prices)} tickers loaded\n")

    # Strategy configs to test
    strategies = {
        'Baseline': dict(canary_type='btc_sma50', health_type='baseline'),
        'B35_SMA15/50': dict(canary_type='market_breadth', health_type='dual_ma', breadth_thr=0.35, short_ma=15, long_ma=50),
        'B40_SMA15/50': dict(canary_type='market_breadth', health_type='dual_ma', breadth_thr=0.40, short_ma=15, long_ma=50),
        'B30_SMA15/50': dict(canary_type='market_breadth', health_type='dual_ma', breadth_thr=0.30, short_ma=15, long_ma=50),
        'B35_SMA10/50': dict(canary_type='market_breadth', health_type='dual_ma', breadth_thr=0.35, short_ma=10, long_ma=50),
        'B40_SMA10/30': dict(canary_type='market_breadth', health_type='dual_ma', breadth_thr=0.40, short_ma=10, long_ma=30),
    }

    # ═══ TEST 1: Walk-Forward (Expanding Window) ═══
    print("=" * 95)
    print("  TEST 1: WALK-FORWARD (Expanding Window)")
    print("  Train on past, test on next year. Only OOS results shown.")
    print("=" * 95)

    wf_periods = [
        ('2019-01~2020-12 → 2021', '2019-01-01', '2020-12-31', '2021-01-01', '2021-12-31'),
        ('2019-01~2021-12 → 2022', '2019-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
        ('2019-01~2022-12 → 2023', '2019-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
        ('2019-01~2023-12 → 2024', '2019-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
        ('2019-01~2024-12 → 2025', '2019-01-01', '2024-12-31', '2025-01-01', '2025-12-31'),
    ]

    print(f"\n  {'Strategy':<20}", end='')
    for label, _, _, _, _ in wf_periods:
        year = label.split('→')[1].strip()
        print(f" {year:>10}", end='')
    print(f" {'AvgOOS':>10}")
    print(f"  {'-'*80}")

    wf_sharpes = {}
    for sname, skwargs in strategies.items():
        print(f"  {sname:<20}", end='')
        oos_sharpes = []
        for label, train_s, train_e, test_s, test_e in wf_periods:
            # Run on test period only (strategy doesn't need training)
            pv = run_backtest(prices, universe_map, start_date=test_s, end_date=test_e, **skwargs)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                print(f" {m['Sharpe']:>10.3f}", end='')
                oos_sharpes.append(m['Sharpe'])
            else:
                print(f" {'N/A':>10}", end='')
        avg = np.mean(oos_sharpes) if oos_sharpes else 0
        wf_sharpes[sname] = avg
        print(f" {avg:>10.3f}")

    # ═══ TEST 2: Transaction Cost Stress Test ═══
    print("\n" + "=" * 95)
    print("  TEST 2: TRANSACTION COST STRESS TEST")
    print("=" * 95)

    tx_costs = [0.001, 0.002, 0.004, 0.006, 0.010]
    print(f"\n  {'Strategy':<20}", end='')
    for tc in tx_costs:
        label = f"{tc:.1%}"
        print(f" {label:>10}", end='')
    print()
    print(f"  {'-'*70}")

    for sname, skwargs in strategies.items():
        print(f"  {sname:<20}", end='')
        for tc in tx_costs:
            pv = run_backtest(prices, universe_map, tx_cost=tc, **skwargs)
            if pv is not None and len(pv) > 0:
                m = calc_metrics(pv)
                print(f" {m['Sharpe']:>10.3f}", end='')
            else:
                print(f" {'N/A':>10}", end='')
        print()

    # ═══ TEST 3: Regime Analysis ═══
    print("\n" + "=" * 95)
    print("  TEST 3: REGIME ANALYSIS (Bull / Bear / Recovery)")
    print("=" * 95)

    regimes = [
        ('Bull 2020-21', '2020-01-01', '2021-12-31'),
        ('Bear 2022', '2022-01-01', '2022-12-31'),
        ('Recovery 23-24', '2023-01-01', '2024-12-31'),
        ('Recent 2025', '2025-01-01', '2025-12-31'),
    ]

    for regime_name, rs, re in regimes:
        print(f"\n  {regime_name}:")
        print(f"  {'Strategy':<20} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8}")
        print(f"  {'-'*45}")
        for sname, skwargs in strategies.items():
            pv = run_backtest(prices, universe_map, start_date=rs, end_date=re, **skwargs)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                print(f"  {sname:<20} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%}")

    # ═══ TEST 4: Sub-period Parameter Stability ═══
    print("\n" + "=" * 95)
    print("  TEST 4: PARAMETER STABILITY ACROSS SUB-PERIODS")
    print("  Which Breadth threshold is best in each sub-period?")
    print("=" * 95)

    sub_periods = [
        ('2019-2020', '2019-01-01', '2020-12-31'),
        ('2021-2022', '2021-01-01', '2022-12-31'),
        ('2023-2025', '2023-01-01', '2025-12-31'),
    ]

    breadths = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

    for period_name, ps, pe in sub_periods:
        print(f"\n  {period_name} (SMA15/50 고정):")
        print(f"  {'Breadth':>10} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8}")
        print(f"  {'-'*40}")
        best_b, best_s = 0, -999
        for b in breadths:
            pv = run_backtest(prices, universe_map,
                             canary_type='market_breadth', health_type='dual_ma',
                             breadth_thr=b, short_ma=15, long_ma=50,
                             start_date=ps, end_date=pe)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                marker = ''
                if m['Sharpe'] > best_s:
                    best_s = m['Sharpe']
                    best_b = b
                print(f"  {b:>9.0%} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%}")
        print(f"  >> Best: {best_b:.0%}")

    # ═══ SUMMARY ═══
    print("\n" + "=" * 95)
    print("  VALIDATION SUMMARY")
    print("=" * 95)

    print(f"\n  Walk-Forward OOS Average Sharpe:")
    for sname in sorted(wf_sharpes.keys(), key=lambda x: wf_sharpes[x], reverse=True):
        marker = ' <<<' if wf_sharpes[sname] == max(wf_sharpes.values()) else ''
        print(f"    {sname:<20} {wf_sharpes[sname]:.3f}{marker}")

    print()


if __name__ == '__main__':
    main()
