#!/usr/bin/env python3
"""
Comprehensive Robustness & Parameter Tuning for Trigger Strategies.

Tests:
  1. All promising combinations vs Baseline (full period + yearly)
  2. Walk-Forward (expanding window OOS)
  3. Transaction Cost Stress Test
  4. Regime Analysis (Bull / Bear / Recovery)
  5. Parameter Sensitivity (R3 gap, E1 vol multiplier, D1 crash threshold)
  6. Sub-period Stability
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from backtest_coin_triggers import (
    filter_universe_topn, run_backtest,
    EXCLUDE_SYMBOLS, TOP_N, START_DATE, END_DATE,
    compute_target, execute_rebalance,
    get_portfolio_value, compute_scores,
    calc_breadth,
)
from backtest_coin_strategy import (
    load_universe, load_all_prices, calc_metrics, STABLECOINS
)


def run_with_custom_params(prices, universe_map, trigger_names, tx_cost=0.004,
                           start_date=None, end_date=None,
                           r3_gap=0.5, r3_consec=2,
                           e1_vol_mult=1.5, e1_consec=3,
                           d1_crash=-0.07, d1_vol_mult=2.0):
    """Run backtest with customizable trigger parameters."""
    btc = prices.get('BTC-USD')
    if btc is None: return None

    sd = start_date or START_DATE
    ed = end_date or END_DATE
    all_dates = btc.index[(btc.index >= sd) & (btc.index <= ed)]
    if len(all_dates) == 0: return None

    holdings = {}
    cash = 10000
    portfolio_values = []
    prev_month = None
    rebal_count = {'monthly': 0, 'trigger_off': 0, 'trigger_def': 0, 'trigger_rot': 0, 'init': 0}
    state = {}

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)
        port_val = get_portfolio_value(holdings, cash, prices, date)

        peak = state.get('peak_value', port_val)
        if port_val > peak:
            state['peak_value'] = port_val

        current_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and current_month != prev_month)

        do_rebal = False
        reason = 'monthly'

        if i == 0:
            do_rebal = True
            reason = 'init'
        elif is_month_change:
            do_rebal = True
            reason = 'monthly'

        # Custom trigger checks
        trigger_fired = False
        trigger_type = None
        is_in_cash = not holdings

        for tname in trigger_names:
            fired = False
            ttype = None

            if tname == 'E1' and is_in_cash:
                # Custom E1 with configurable params
                close = btc['Close'].iloc[:global_idx+1]
                if len(close) >= max(53, e1_consec + 3):
                    sma50 = close.rolling(50).mean()
                    all_above = all(close.iloc[-(j+1)] > sma50.iloc[-(j+1)] for j in range(e1_consec))
                    if all_above and len(close) >= e1_consec + 1:
                        prev_idx = len(close) - e1_consec - 1
                        is_new = close.iloc[prev_idx] <= sma50.iloc[prev_idx]
                        if is_new:
                            vol_ok = True
                            if 'Volume' in btc.columns:
                                vol = btc['Volume'].iloc[:global_idx+1]
                                if len(vol) >= 21:
                                    bd = len(vol) - e1_consec
                                    avg = vol.iloc[max(0,bd-20):bd].mean()
                                    if avg > 0 and vol.iloc[bd] < e1_vol_mult * avg:
                                        vol_ok = False
                            fired = vol_ok
                ttype = 'offensive'

            elif tname == 'D1' and not is_in_cash:
                # Custom D1 with configurable crash/vol
                close = btc['Close'].iloc[:global_idx+1]
                if len(close) >= 22:
                    daily_ret = close.iloc[-1] / close.iloc[-2] - 1 if close.iloc[-2] > 0 else 0
                    if daily_ret <= d1_crash and 'Volume' in btc.columns:
                        vol = btc['Volume'].iloc[:global_idx+1]
                        if len(vol) >= 21:
                            avg_vol = vol.iloc[-21:-1].mean()
                            fired = avg_vol > 0 and vol.iloc[-1] > d1_vol_mult * avg_vol
                ttype = 'defensive'

            elif tname == 'R3' and not is_in_cash:
                # Custom R3 with configurable gap/consec
                all_scores = compute_scores(prices, universe_map, date, global_idx)
                if all_scores:
                    held_scores = {t: all_scores.get(t, -999) for t in holdings if t in all_scores}
                    non_held = {t: s for t, s in all_scores.items() if t not in holdings}
                    gap = False
                    if held_scores and non_held:
                        if max(non_held.values()) - min(held_scores.values()) >= r3_gap:
                            gap = True
                    c = state.get('r3p_consec', 0)
                    c = c + 1 if gap else 0
                    state['r3p_consec'] = c
                    if c >= r3_consec:
                        state['r3p_consec'] = 0
                        fired = True
                ttype = 'rotation'

            elif tname == 'E2' and is_in_cash:
                from backtest_coin_triggers import trigger_E2
                fired = trigger_E2(prices, global_idx, state, universe_map, date)
                ttype = 'offensive'

            elif tname == 'R1' and not is_in_cash:
                from backtest_coin_triggers import trigger_R1
                fired = trigger_R1(prices, global_idx, state, holdings, universe_map, date)
                ttype = 'rotation'

            elif tname == 'R2' and not is_in_cash:
                from backtest_coin_triggers import trigger_R2
                fired = trigger_R2(prices, global_idx, state, holdings, cash, universe_map, date)
                ttype = 'rotation'

            if fired and not trigger_fired:
                trigger_fired = True
                trigger_type = ttype

        if trigger_fired and not do_rebal:
            do_rebal = True
            if trigger_type == 'offensive':
                reason = 'trigger_off'
            elif trigger_type == 'rotation':
                reason = 'trigger_rot'
            else:
                reason = 'trigger_def'

        if do_rebal:
            target = compute_target(prices, universe_map, date, global_idx)
            holdings, cash = execute_rebalance(holdings, cash, target, prices, date, tx_cost)
            rebal_count[reason] = rebal_count.get(reason, 0) + 1
            # Reset stateful trigger counters after rebalance
            state.pop('r3p_consec', None)
            new_val = get_portfolio_value(holdings, cash, prices, date)
            if new_val > state.get('peak_value', 0):
                state['peak_value'] = new_val
            port_val = new_val

        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    result = pd.DataFrame(portfolio_values).set_index('Date')
    result.attrs['rebal_count'] = rebal_count
    return result


def main():
    print("=" * 110)
    print("  COMPREHENSIVE ROBUSTNESS & PARAMETER TUNING")
    print(f"  Trigger Strategies vs Baseline ({START_DATE[:4]}-{END_DATE[:4]}, Top {TOP_N})")
    print("=" * 110)

    universe_map = load_universe()
    filtered_map = filter_universe_topn(universe_map, TOP_N)

    all_tickers = set()
    for mt in filtered_map.values():
        for t in mt:
            all_tickers.add(t)
    all_tickers.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(all_tickers)
    print(f"  {len(prices)} tickers loaded\n")

    # Strategy configs (R2 included for robustness validation)
    strategies = {
        'Baseline (Monthly)':   [],
        'E1 only':              ['E1'],
        'E2 only':              ['E2'],
        'D1 only':              ['D1'],
        'C1: E1+D1':            ['E1', 'D1'],
        'E2+D1':                ['E2', 'D1'],
        'E1+E2+D1':             ['E1', 'E2', 'D1'],
        'C4: E1+D1+R1':         ['E1', 'D1', 'R1'],
        'C5: E1+D1+R2':         ['E1', 'D1', 'R2'],
    }

    # ═══ TEST 1: Full Period + Yearly Comparison ═══
    print("=" * 110)
    print(f"  TEST 1: ALL COMBINATIONS vs BASELINE ({START_DATE[:4]}-{END_DATE[:4]})")
    print("=" * 110)

    years = list(range(int(START_DATE[:4]), int(END_DATE[:4]) + 1))
    print(f"\n  {'Strategy':<22}", end='')
    for y in years:
        print(f" {y:>8}", end='')
    print(f" {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'#Trig':>6}")
    print(f"  {'-'*100}")

    full_results = {}
    for name, trigs in strategies.items():
        print(f"  {name:<22}", end='')
        pv = run_backtest(prices, filtered_map, trigger_names=trigs)
        for y in years:
            if pv is not None:
                mask = (pv.index >= f'{y}-01-01') & (pv.index <= f'{y}-12-31')
                sl = pv[mask]
                if len(sl) > 10:
                    m = calc_metrics(sl)
                    print(f" {m['Sharpe']:>8.3f}", end='')
                else:
                    print(f" {'N/A':>8}", end='')
            else:
                print(f" {'N/A':>8}", end='')

        if pv is not None and len(pv) > 10:
            m = calc_metrics(pv)
            rc = pv.attrs.get('rebal_count', {})
            nt = rc.get('trigger_off', 0) + rc.get('trigger_def', 0) + rc.get('trigger_rot', 0)
            full_results[name] = m
            print(f" {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {nt:>6}")
        else:
            print()

    # Delta vs baseline
    bl = full_results.get('Baseline (Monthly)')
    if bl:
        print(f"\n  {'Strategy':<22} {'dSharpe':>10} {'dMDD':>10} {'dCAGR':>10}")
        print(f"  {'-'*55}")
        for name in strategies:
            if name == 'Baseline (Monthly)' or name not in full_results: continue
            m = full_results[name]
            print(f"  {name:<22} {m['Sharpe']-bl['Sharpe']:>+10.3f} "
                  f"{(m['MDD']-bl['MDD'])*100:>+9.1f}pp {(m['CAGR']-bl['CAGR'])*100:>+9.1f}pp")

    # ═══ TEST 2: Walk-Forward ═══
    print("\n" + "=" * 110)
    print("  TEST 2: WALK-FORWARD (Expanding Window, OOS Sharpe)")
    print("=" * 110)

    wf_test = [
        ('2018', '2018-01-01', '2018-12-31'),
        ('2019', '2019-01-01', '2019-12-31'),
        ('2020', '2020-01-01', '2020-12-31'),
        ('2021', '2021-01-01', '2021-12-31'),
        ('2022', '2022-01-01', '2022-12-31'),
        ('2023', '2023-01-01', '2023-12-31'),
        ('2024', '2024-01-01', '2024-12-31'),
        ('2025', '2025-01-01', '2025-12-31'),
    ]

    top_strategies = ['Baseline (Monthly)', 'E1 only', 'D1 only',
                      'C1: E1+D1', 'E2+D1', 'C4: E1+D1+R1', 'C5: E1+D1+R2']

    print(f"\n  {'Strategy':<22}", end='')
    for label, _, _ in wf_test:
        print(f" {label:>8}", end='')
    print(f" {'AvgOOS':>8} {'StdOOS':>8}")
    print(f"  {'-'*85}")

    for name in top_strategies:
        if name not in strategies: continue
        trigs = strategies[name]
        print(f"  {name:<22}", end='')
        oos_sharpes = []
        for label, ts, te in wf_test:
            pv = run_backtest(prices, filtered_map, trigger_names=trigs,
                              start_date=ts, end_date=te)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                print(f" {m['Sharpe']:>8.3f}", end='')
                oos_sharpes.append(m['Sharpe'])
            else:
                print(f" {'N/A':>8}", end='')
        avg = np.mean(oos_sharpes) if oos_sharpes else 0
        std = np.std(oos_sharpes) if len(oos_sharpes) > 1 else 0
        print(f" {avg:>8.3f} {std:>8.3f}")

    # ═══ TEST 3: Transaction Cost Stress ═══
    print("\n" + "=" * 110)
    print("  TEST 3: TRANSACTION COST STRESS TEST")
    print("=" * 110)

    tx_costs = [0.001, 0.002, 0.003, 0.005, 0.008]
    print(f"\n  {'Strategy':<22}", end='')
    for tc in tx_costs:
        label = f"{tc:.1%}:Shp"
        print(f" {label:>10}", end='')
    print()
    print(f"  {'-'*72}")

    for name in top_strategies:
        if name not in strategies: continue
        trigs = strategies[name]
        print(f"  {name:<22}", end='')
        for tc in tx_costs:
            pv = run_backtest(prices, filtered_map, trigger_names=trigs, tx_cost=tc)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                print(f" {m['Sharpe']:>10.3f}", end='')
            else:
                print(f" {'N/A':>10}", end='')
        print()

    # ═══ TEST 4: Regime Analysis ═══
    print("\n" + "=" * 110)
    print("  TEST 4: REGIME ANALYSIS")
    print("=" * 110)

    regimes = [
        ('Bull 2017-18',   '2017-01-01', '2018-01-31'),
        ('Bear 2018',      '2018-02-01', '2018-12-31'),
        ('Bull 2019-20',   '2019-01-01', '2020-12-31'),
        ('Bull 2021',      '2021-01-01', '2021-12-31'),
        ('Bear 2022',      '2022-01-01', '2022-12-31'),
        ('Recovery 2023',  '2023-01-01', '2023-12-31'),
        ('Bull 2024',      '2024-01-01', '2024-12-31'),
        ('Mixed 2025',     '2025-01-01', '2025-12-31'),
    ]

    for regime_name, rs, re in regimes:
        print(f"\n  {regime_name}:")
        print(f"  {'Strategy':<22} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8}")
        print(f"  {'-'*48}")
        for name in top_strategies:
            if name not in strategies: continue
            trigs = strategies[name]
            pv = run_backtest(prices, filtered_map, trigger_names=trigs,
                              start_date=rs, end_date=re)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                print(f"  {name:<22} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%}")

    # ═══ TEST 5: Parameter Sensitivity ═══
    print("\n" + "=" * 110)
    print("  TEST 5: PARAMETER SENSITIVITY")
    print("=" * 110)

    # 5a: E1 consecutive days
    print(f"\n  5a: E1 Consecutive Days Above SMA50")
    print(f"  {'Days':>6} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'#Off':>6}")
    print(f"  {'-'*42}")

    for days in [2, 3, 4, 5]:
        pv = run_with_custom_params(prices, filtered_map, ['E1', 'D1'],
                                     e1_consec=days)
        if pv is not None and len(pv) > 10:
            m = calc_metrics(pv)
            rc = pv.attrs.get('rebal_count', {})
            no = rc.get('trigger_off', 0)
            print(f"  {days:>6} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {no:>6}")

    # 5b: E1 volume multiplier
    print(f"\n  5b: E1 Volume Multiplier")
    print(f"  {'VolX':>6} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'#Off':>6}")
    print(f"  {'-'*42}")

    for vm in [1.0, 1.2, 1.5, 2.0, 2.5]:
        pv = run_with_custom_params(prices, filtered_map, ['E1', 'D1'],
                                     e1_vol_mult=vm)
        if pv is not None and len(pv) > 10:
            m = calc_metrics(pv)
            rc = pv.attrs.get('rebal_count', {})
            no = rc.get('trigger_off', 0)
            print(f"  {vm:>6.1f} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {no:>6}")

    # 5c: D1 volume multiplier
    print(f"\n  5c: D1 Volume Multiplier")
    print(f"  {'VolX':>6} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'#Def':>6}")
    print(f"  {'-'*42}")

    for vm in [1.2, 1.5, 2.0, 2.5, 3.0]:
        pv = run_with_custom_params(prices, filtered_map, ['E1', 'D1'],
                                     d1_vol_mult=vm)
        if pv is not None and len(pv) > 10:
            m = calc_metrics(pv)
            rc = pv.attrs.get('rebal_count', {})
            nd = rc.get('trigger_def', 0)
            print(f"  {vm:>6.1f} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {nd:>6}")

    # 5d: D1 crash threshold
    print(f"\n  5d: D1 Crash Threshold")
    print(f"  {'Crash':>6} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8} {'#Def':>6}")
    print(f"  {'-'*42}")

    for crash in [-0.05, -0.06, -0.07, -0.08, -0.10]:
        pv = run_with_custom_params(prices, filtered_map, ['E1', 'D1'],
                                     d1_crash=crash)
        if pv is not None and len(pv) > 10:
            m = calc_metrics(pv)
            rc = pv.attrs.get('rebal_count', {})
            nd = rc.get('trigger_def', 0)
            print(f"  {crash:>5.0%} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%} {nd:>6}")

    # ═══ TEST 6: Sub-Period Stability ═══
    print("\n" + "=" * 110)
    print("  TEST 6: SUB-PERIOD STABILITY (Which strategy wins each period?)")
    print("=" * 110)

    sub_periods = [
        ('2018-2019', '2018-01-01', '2019-12-31'),
        ('2020-2021', '2020-01-01', '2021-12-31'),
        ('2022-2023', '2022-01-01', '2023-12-31'),
        ('2024-2025', '2024-01-01', '2025-12-31'),
        ('2018-2021', '2018-01-01', '2021-12-31'),
        ('2022-2025', '2022-01-01', '2025-12-31'),
    ]

    for period_name, ps, pe in sub_periods:
        print(f"\n  {period_name}:")
        print(f"  {'Strategy':<22} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8}")
        print(f"  {'-'*48}")
        best_name, best_sharpe = '', -999
        for name in top_strategies:
            if name not in strategies: continue
            trigs = strategies[name]
            pv = run_backtest(prices, filtered_map, trigger_names=trigs,
                              start_date=ps, end_date=pe)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                marker = ''
                if m['Sharpe'] > best_sharpe:
                    best_sharpe = m['Sharpe']
                    best_name = name
                print(f"  {name:<22} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%}")
        print(f"  >> Winner: {best_name}")

    # ═══ SUMMARY ═══
    print("\n" + "=" * 110)
    print("  FINAL SUMMARY")
    print("=" * 110)

    if bl:
        print(f"\n  Baseline: Sharpe={bl['Sharpe']:.3f} MDD={bl['MDD']:.1%} CAGR={bl['CAGR']:+.1%}")
        print(f"\n  {'Strategy':<22} {'Sharpe':>8} {'dShp':>8} {'MDD':>8} {'dMDD':>8} {'CAGR':>8} {'dCAGR':>8}")
        print(f"  {'-'*70}")
        for name in sorted(full_results.keys(), key=lambda x: full_results[x]['Sharpe'], reverse=True):
            if name == 'Baseline (Monthly)': continue
            m = full_results[name]
            print(f"  {name:<22} {m['Sharpe']:>8.3f} {m['Sharpe']-bl['Sharpe']:>+7.3f} "
                  f"{m['MDD']:>7.1%} {(m['MDD']-bl['MDD'])*100:>+7.1f}pp "
                  f"{m['CAGR']:>+7.1%} {(m['CAGR']-bl['CAGR'])*100:>+7.1f}pp")

    print()


if __name__ == '__main__':
    main()
