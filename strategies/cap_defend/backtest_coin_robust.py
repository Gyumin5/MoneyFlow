#!/usr/bin/env python3
"""
Comprehensive Robustness Validation for Coin Strategy:
1. Codex Hysteresis test (Breadth ON 42%, OFF 38%)
2. Outlier exclusion (remove top N% performers each rebalance)
3. Random coin subset bootstrap (50% random universe)
4. Leave-one-year-out cross-validation
5. Maximum adverse excursion analysis
6. Coin concentration dependency (top 1/3/5/10 picks)
"""

import os, sys, warnings, random
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from backtest_coin_strategy import (
    load_universe, load_all_prices, calc_metrics, calc_yearly_metrics,
    calc_sharpe, calc_rsi, calc_macd_hist, calc_bb_pctb,
    calc_ret, get_volatility, STABLECOINS
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')


def run_backtest(prices, universe_map, canary_type='btc_sma50',
                 health_type='baseline', breadth_thr=0.40,
                 short_ma=15, long_ma=50,
                 start_date='2019-01-01', end_date='2025-12-31',
                 tx_cost=0.002, n_picks=5,
                 # Hysteresis
                 use_hysteresis=False, breadth_on=0.42, breadth_off=0.38,
                 # Outlier exclusion
                 exclude_top_pct=0.0,
                 # Random subset
                 universe_sample_pct=1.0, rng_seed=None,
                 # Tracking
                 track_picks=False):
    """Extended backtest runner with hysteresis, outlier exclusion, subset."""
    btc = prices.get('BTC-USD')
    if btc is None: return None

    all_dates = btc.index[(btc.index >= start_date) & (btc.index <= end_date)]
    if len(all_dates) == 0: return None

    holdings = {}
    cash = 10000
    portfolio_values = []
    prev_month = None
    hysteresis_state = None  # None = first check, True = risk-on, False = risk-off
    pick_history = []  # Track which coins were picked

    rng = random.Random(rng_seed) if rng_seed is not None else None

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

            # Random subset of universe
            if universe_sample_pct < 1.0 and rng is not None:
                n_sample = max(5, int(len(uni_clean) * universe_sample_pct))
                if n_sample < len(uni_clean):
                    uni_clean = rng.sample(uni_clean, n_sample)

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
                    breadth = count_above / count_total
                    if use_hysteresis:
                        if hysteresis_state is None:
                            risk_on = breadth > breadth_on
                        elif hysteresis_state:
                            risk_on = breadth > breadth_off  # Stay on until drops below off
                        else:
                            risk_on = breadth > breadth_on  # Need to cross on threshold
                        hysteresis_state = risk_on
                    else:
                        risk_on = breadth > breadth_thr

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

                # Outlier exclusion: remove top N% by recent return
                if exclude_top_pct > 0 and healthy:
                    returns_list = []
                    for t in healthy:
                        close = prices[t]['Close'].iloc[:global_idx+1]
                        if len(close) >= 90:
                            ret90 = calc_ret(close, 90)
                            returns_list.append((t, ret90))
                        else:
                            returns_list.append((t, 0))
                    returns_list.sort(key=lambda x: x[1], reverse=True)
                    n_exclude = max(1, int(len(returns_list) * exclude_top_pct))
                    excluded = set(t for t, _ in returns_list[:n_exclude])
                    healthy = [t for t in healthy if t not in excluded]

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
                    picks = [t for t, _ in scores[:n_picks]]

                    if track_picks and picks:
                        pick_history.append({'date': date, 'picks': picks[:]})

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

    result = pd.DataFrame(portfolio_values).set_index('Date')
    if track_picks:
        result.attrs['pick_history'] = pick_history
    return result


def main():
    print("=" * 100)
    print("  COMPREHENSIVE ROBUSTNESS VALIDATION")
    print("=" * 100)

    universe_map = load_universe()
    all_tickers = set()
    for mt in universe_map.values():
        for t in mt:
            s = t.replace('-USD', '')
            if s not in STABLECOINS: all_tickers.add(t)
    all_tickers.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(all_tickers)
    print(f"  {len(prices)} tickers loaded\n")

    # Base configs
    baseline_kw = dict(canary_type='btc_sma50', health_type='baseline')
    proposed_kw = dict(canary_type='market_breadth', health_type='dual_ma',
                       breadth_thr=0.40, short_ma=15, long_ma=50)

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 1: CODEX HYSTERESIS SUGGESTION
    # ═══════════════════════════════════════════════════════════════════════
    print("=" * 100)
    print("  TEST 1: CODEX HYSTERESIS (ON 42% / OFF 38%)")
    print("  Compare: Fixed 40% vs Hysteresis 42/38% vs Hysteresis 45/35%")
    print("=" * 100)

    hysteresis_configs = {
        'Baseline': baseline_kw,
        'Fixed 40%': proposed_kw,
        'Hyst 42/38': dict(canary_type='market_breadth', health_type='dual_ma',
                           short_ma=15, long_ma=50, use_hysteresis=True, breadth_on=0.42, breadth_off=0.38),
        'Hyst 45/35': dict(canary_type='market_breadth', health_type='dual_ma',
                           short_ma=15, long_ma=50, use_hysteresis=True, breadth_on=0.45, breadth_off=0.35),
        'Hyst 43/37': dict(canary_type='market_breadth', health_type='dual_ma',
                           short_ma=15, long_ma=50, use_hysteresis=True, breadth_on=0.43, breadth_off=0.37),
    }

    years = list(range(2019, 2026))
    print(f"\n  {'Strategy':<16}", end='')
    for y in years:
        print(f" {y:>8}", end='')
    print(f" {'Overall':>10} {'MDD':>8}")
    print(f"  {'-'*100}")

    for name, kw in hysteresis_configs.items():
        print(f"  {name:<16}", end='')
        pv_full = run_backtest(prices, universe_map, **kw)
        for y in years:
            pv = run_backtest(prices, universe_map, start_date=f'{y}-01-01', end_date=f'{y}-12-31', **kw)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                print(f" {m['Sharpe']:>8.3f}", end='')
            else:
                print(f" {'N/A':>8}", end='')
        if pv_full is not None and len(pv_full) > 10:
            m_full = calc_metrics(pv_full)
            print(f" {m_full['Sharpe']:>10.3f} {m_full['MDD']:>7.1%}")
        else:
            print(f" {'N/A':>10} {'N/A':>8}")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 2: OUTLIER EXCLUSION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST 2: OUTLIER EXCLUSION (Remove Top N% performers each rebalance)")
    print("  If strategy depends on lucky outliers, performance drops sharply")
    print("=" * 100)

    exclude_pcts = [0.0, 0.10, 0.20, 0.30, 0.50]
    print(f"\n  {'Strategy':<16}", end='')
    for ep in exclude_pcts:
        label = f"Ex{ep:.0%}"
        print(f" {label:>10}", end='')
    print(f" {'Degradation':>12}")
    print(f"  {'-'*80}")

    for name, kw in [('Baseline', baseline_kw), ('B40_SMA15/50', proposed_kw)]:
        print(f"  {name:<16}", end='')
        sharpes = []
        for ep in exclude_pcts:
            pv = run_backtest(prices, universe_map, exclude_top_pct=ep, **kw)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                print(f" {m['Sharpe']:>10.3f}", end='')
                sharpes.append(m['Sharpe'])
            else:
                print(f" {'N/A':>10}", end='')
                sharpes.append(0)
        # Degradation: how much % Sharpe drops when excluding 20%
        if len(sharpes) >= 3 and sharpes[0] != 0:
            deg = (sharpes[2] - sharpes[0]) / abs(sharpes[0]) * 100
            print(f" {deg:>+10.1f}%")
        else:
            print()

    # Year-by-year outlier test
    print(f"\n  Year-by-year with 20% exclusion:")
    print(f"  {'Strategy':<16}", end='')
    for y in years:
        print(f" {y:>8}", end='')
    print(f" {'Avg':>8}")
    print(f"  {'-'*90}")

    for name, kw in [('B40 (No Excl)', proposed_kw),
                     ('B40 (Excl 20%)', {**proposed_kw, 'exclude_top_pct': 0.20})]:
        print(f"  {name:<16}", end='')
        yearly_sharpes = []
        for y in years:
            pv = run_backtest(prices, universe_map, start_date=f'{y}-01-01', end_date=f'{y}-12-31',
                             **kw)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                print(f" {m['Sharpe']:>8.3f}", end='')
                yearly_sharpes.append(m['Sharpe'])
            else:
                print(f" {'N/A':>8}", end='')
        avg = np.mean(yearly_sharpes) if yearly_sharpes else 0
        print(f" {avg:>8.3f}")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 3: RANDOM UNIVERSE SUBSET BOOTSTRAP
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST 3: RANDOM UNIVERSE SUBSET BOOTSTRAP (50% of universe, 20 trials)")
    print("  Tests if strategy works with different coin subsets")
    print("=" * 100)

    n_trials = 20
    bootstrap_results = {'Baseline': [], 'B40_SMA15/50': []}

    for trial in range(n_trials):
        seed = 42 + trial
        for name, kw in [('Baseline', baseline_kw), ('B40_SMA15/50', proposed_kw)]:
            pv = run_backtest(prices, universe_map,
                             universe_sample_pct=0.50, rng_seed=seed, **kw)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                bootstrap_results[name].append(m['Sharpe'])
            else:
                bootstrap_results[name].append(0)

    print(f"\n  {'Strategy':<16} {'Mean':>8} {'Median':>8} {'Std':>8} {'Min':>8} {'Max':>8} {'%>0':>6} {'%>Baseline':>10}")
    print(f"  {'-'*80}")

    bl_mean = np.mean(bootstrap_results['Baseline']) if bootstrap_results['Baseline'] else 0
    for name in ['Baseline', 'B40_SMA15/50']:
        vals = bootstrap_results[name]
        if vals:
            mean_v = np.mean(vals)
            med_v = np.median(vals)
            std_v = np.std(vals)
            min_v = np.min(vals)
            max_v = np.max(vals)
            pct_pos = np.mean([v > 0 for v in vals]) * 100
            pct_beat = np.mean([v > bl_mean for v in vals]) * 100
            print(f"  {name:<16} {mean_v:>8.3f} {med_v:>8.3f} {std_v:>8.3f} {min_v:>8.3f} {max_v:>8.3f} {pct_pos:>5.0f}% {pct_beat:>9.0f}%")

    # Head-to-head per trial
    wins = sum(1 for i in range(n_trials) if bootstrap_results['B40_SMA15/50'][i] > bootstrap_results['Baseline'][i])
    print(f"\n  Head-to-head: B40 wins {wins}/{n_trials} trials ({wins/n_trials*100:.0f}%)")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 4: LEAVE-ONE-YEAR-OUT CROSS VALIDATION
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST 4: LEAVE-ONE-YEAR-OUT CROSS VALIDATION")
    print("  Run full backtest with each year excluded, check consistency")
    print("=" * 100)

    loyo_years = list(range(2019, 2026))
    print(f"\n  {'Excluded Year':<16}", end='')
    print(f" {'Baseline':>10} {'B40_SMA15/50':>12} {'Diff':>8}")
    print(f"  {'-'*50}")

    loyo_diffs = []
    for exclude_year in loyo_years:
        # Build date ranges excluding one year
        remaining_ranges = []
        for y in loyo_years:
            if y != exclude_year:
                remaining_ranges.append((f'{y}-01-01', f'{y}-12-31'))

        # Run each range and average
        bl_sharpes = []
        prop_sharpes = []
        for rs, re in remaining_ranges:
            pv_bl = run_backtest(prices, universe_map, start_date=rs, end_date=re, **baseline_kw)
            pv_prop = run_backtest(prices, universe_map, start_date=rs, end_date=re, **proposed_kw)
            if pv_bl is not None and len(pv_bl) > 10:
                bl_sharpes.append(calc_metrics(pv_bl)['Sharpe'])
            if pv_prop is not None and len(pv_prop) > 10:
                prop_sharpes.append(calc_metrics(pv_prop)['Sharpe'])

        bl_avg = np.mean(bl_sharpes) if bl_sharpes else 0
        prop_avg = np.mean(prop_sharpes) if prop_sharpes else 0
        diff = prop_avg - bl_avg
        loyo_diffs.append(diff)
        print(f"  ex-{exclude_year:<12} {bl_avg:>10.3f} {prop_avg:>12.3f} {diff:>+8.3f}")

    all_positive = all(d > 0 for d in loyo_diffs)
    print(f"\n  B40 beats Baseline in ALL leave-one-out folds: {'YES' if all_positive else 'NO'}")
    print(f"  Average improvement: {np.mean(loyo_diffs):+.3f} Sharpe")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 5: COIN CONCENTRATION DEPENDENCY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST 5: COIN CONCENTRATION (Top N picks)")
    print("  Does strategy depend on picking exactly 5 coins?")
    print("=" * 100)

    n_picks_list = [1, 3, 5, 7, 10]
    print(f"\n  {'Strategy':<16}", end='')
    for n in n_picks_list:
        print(f" {'Top'+str(n):>8}", end='')
    print()
    print(f"  {'-'*60}")

    for name, kw in [('Baseline', baseline_kw), ('B40_SMA15/50', proposed_kw)]:
        print(f"  {name:<16}", end='')
        for n in n_picks_list:
            pv = run_backtest(prices, universe_map, n_picks=n, **kw)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                print(f" {m['Sharpe']:>8.3f}", end='')
            else:
                print(f" {'N/A':>8}", end='')
        print()

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 6: PICK FREQUENCY ANALYSIS
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST 6: COIN PICK FREQUENCY ANALYSIS")
    print("  Which coins were picked most often? Is one coin driving returns?")
    print("=" * 100)

    pv_track = run_backtest(prices, universe_map, track_picks=True, **proposed_kw)
    if pv_track is not None:
        pick_hist = pv_track.attrs.get('pick_history', [])
        coin_counts = defaultdict(int)
        for entry in pick_hist:
            for coin in entry['picks']:
                coin_counts[coin] += 1

        total_rebals = len(pick_hist)
        sorted_coins = sorted(coin_counts.items(), key=lambda x: x[1], reverse=True)

        print(f"\n  Total rebalances: {total_rebals}")
        print(f"  Unique coins picked: {len(coin_counts)}")
        print(f"\n  {'Coin':<16} {'Times':>6} {'Pct':>6}")
        print(f"  {'-'*30}")
        for coin, count in sorted_coins[:15]:
            pct = count / total_rebals * 100
            print(f"  {coin:<16} {count:>6} {pct:>5.1f}%")

        # Test: remove top 3 most-picked coins entirely and retest
        top3_coins = set(c for c, _ in sorted_coins[:3])
        print(f"\n  Removing top 3 most-picked coins: {top3_coins}")

        # Run backtest with these coins removed from prices
        prices_filtered = {k: v for k, v in prices.items() if k not in top3_coins}
        pv_no_top3 = run_backtest(prices_filtered, universe_map, **proposed_kw)
        m_full = calc_metrics(pv_track)
        if pv_no_top3 is not None and len(pv_no_top3) > 10:
            m_no3 = calc_metrics(pv_no_top3)
            print(f"  With all coins:  Sharpe={m_full['Sharpe']:.3f} CAGR={m_full['CAGR']:+.1%} MDD={m_full['MDD']:.1%}")
            print(f"  Without top 3:   Sharpe={m_no3['Sharpe']:.3f} CAGR={m_no3['CAGR']:+.1%} MDD={m_no3['MDD']:.1%}")
            deg = (m_no3['Sharpe'] - m_full['Sharpe']) / abs(m_full['Sharpe']) * 100
            print(f"  Degradation:     {deg:+.1f}%")

    # ═══════════════════════════════════════════════════════════════════════
    # TEST 7: MAXIMUM ADVERSE EXCURSION (Worst Drawdown Periods)
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST 7: WORST DRAWDOWN ANALYSIS")
    print("=" * 100)

    for name, kw in [('Baseline', baseline_kw), ('B40_SMA15/50', proposed_kw)]:
        pv = run_backtest(prices, universe_map, **kw)
        if pv is None or len(pv) < 30: continue
        values = pv['Value']
        peak = values.cummax()
        dd = values / peak - 1

        # Find top 3 worst drawdown periods
        print(f"\n  {name} - Top 3 worst drawdowns:")
        in_dd = False
        dd_periods = []
        dd_start = None
        dd_max = 0
        dd_max_date = None

        for idx in range(len(dd)):
            if dd.iloc[idx] < -0.01 and not in_dd:
                in_dd = True
                dd_start = dd.index[idx]
                dd_max = dd.iloc[idx]
                dd_max_date = dd.index[idx]
            elif in_dd:
                if dd.iloc[idx] < dd_max:
                    dd_max = dd.iloc[idx]
                    dd_max_date = dd.index[idx]
                if dd.iloc[idx] >= -0.01:
                    dd_periods.append({
                        'start': dd_start,
                        'trough_date': dd_max_date,
                        'end': dd.index[idx],
                        'depth': dd_max,
                        'duration': (dd.index[idx] - dd_start).days
                    })
                    in_dd = False

        # If still in drawdown at end
        if in_dd:
            dd_periods.append({
                'start': dd_start,
                'trough_date': dd_max_date,
                'end': dd.index[-1],
                'depth': dd_max,
                'duration': (dd.index[-1] - dd_start).days
            })

        dd_periods.sort(key=lambda x: x['depth'])
        for j, dp in enumerate(dd_periods[:3]):
            print(f"    #{j+1}: {dp['depth']:.1%} | {dp['start'].strftime('%Y-%m-%d')} → {dp['trough_date'].strftime('%Y-%m-%d')} ({dp['duration']}d)")

    # ═══════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  ROBUSTNESS SUMMARY")
    print("=" * 100)

    print("""
  1. HYSTERESIS: Check table above for best variant
  2. OUTLIER EXCLUSION: Check degradation % — low = strategy is NOT outlier-dependent
  3. BOOTSTRAP (50% universe): Check win rate — high = consistent across coin subsets
  4. LEAVE-ONE-YEAR-OUT: Beats Baseline in ALL folds = robust across time
  5. CONCENTRATION: Stable across N picks = not fragile to exact pick count
  6. COIN DEPENDENCY: Low degradation when removing top coins = diversified alpha
  7. DRAWDOWN ANALYSIS: Compare drawdown depth/duration between strategies
""")


if __name__ == '__main__':
    main()
