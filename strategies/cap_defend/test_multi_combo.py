#!/usr/bin/env python3
"""Test 3-way, 4-way, 5-way combinations with K5 as base."""

import os, sys, time, itertools
import multiprocessing as mp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    Params, load_data, init_pool, run_single, run_backtest, calc_metrics, calc_yearly_metrics
)

N_WORKERS = min(24, mp.cpu_count())

# K5 is fixed. These are the additive layers from Stage 4.
# Exclude G4/S4 (identical to K5 alone)
ADDONS = {
    'H5': {'health': 'H5'},
    'S5': {'selection': 'S5'},
    'W1': {'weighting': 'W1'},
    'W3': {'weighting': 'W3'},
    'R2': {'rebalancing': 'R2'},
    'G2': {'risk': 'G2'},
    'G5': {'risk': 'G5'},
}

def make_combo_params(combo_keys):
    """Create Params with K5 + multiple addons. Skip if same layer conflict."""
    kwargs = {'canary': 'K5'}
    layers_used = {'canary'}
    for key in combo_keys:
        addon = ADDONS[key]
        for layer, val in addon.items():
            if layer in layers_used:
                return None  # conflict — same layer
            kwargs[layer] = val
            layers_used.add(layer)
    return Params(**kwargs)


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded\n")

    # Generate all valid 3, 4, 5-way combos
    addon_keys = list(ADDONS.keys())
    all_params = []

    for size in [3, 4, 5]:
        for combo in itertools.combinations(addon_keys, size):
            p = make_combo_params(combo)
            if p is not None:
                all_params.append(p)

    # Also add the 2-way combos for reference
    for key in addon_keys:
        p = make_combo_params([key])
        if p:
            all_params.append(p)

    # Add baseline and K5 solo
    all_params.append(Params())           # BASELINE
    all_params.append(Params(canary='K5'))  # K5 solo

    print(f"  Total combinations: {len(all_params)}")
    print(f"  (7 × 2-way + {len([1 for s in [3,4,5] for c in itertools.combinations(addon_keys, s) if make_combo_params(c)])} multi-way + 2 reference)\n")

    # Run parallel
    init_pool(prices, universe)
    t0 = time.time()
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, all_params)
    elapsed = time.time() - t0
    print(f"  Completed in {elapsed:.1f}s\n")

    # Sort by Sharpe
    results.sort(key=lambda x: -x['metrics']['Sharpe'])

    # Full period results
    print("=" * 130)
    print("  전구간 성과 (2018-01-01 ~ 2025-06-30, tx=0.4%)")
    print("=" * 130)
    print(f"\n  {'#':>3} {'전략':<28} {'Layers':>2} {'Sharpe':>8} {'Sortino':>8}"
          f" {'CAGR':>8} {'MDD':>8} {'Final($)':>12} {'Calmar':>8} {'Rebals':>7}")
    print(f"  {'─' * 125}")

    for i, r in enumerate(results):
        m = r['metrics']
        label = r['label']
        n_layers = label.count('+') + 1 if label != 'BASELINE' else 0
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        print(f"  {i+1:>3} {label:<28} {n_layers:>2} {m['Sharpe']:>8.3f} {m['Sortino']:>8.3f}"
              f" {m['CAGR']:>+7.1%} {m['MDD']:>7.1%} {m['Final']:>11,.0f} {calmar:>8.3f}"
              f" {r['rebal_count']:>7}")

    # Year-by-year for top 10
    print()
    print("=" * 130)
    print("  상위 10개 전략 연도별 CAGR")
    print("=" * 130)
    years = range(2018, 2026)
    print(f"  {'전략':<28}", end="")
    for y in years:
        print(f" {y:>8}", end="")
    print(f" {'전체':>9}")
    print(f"  {'─' * 115}")

    for r in results[:10]:
        ym = r['yearly']
        m = r['metrics']
        row = f"  {r['label']:<28}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+7.1%}"
            else:
                row += f" {'─':>8}"
        row += f" {m['CAGR']:>+8.1%}"
        print(row)

    # Year-by-year MDD for top 10
    print()
    print("=" * 130)
    print("  상위 10개 전략 연도별 MDD")
    print("=" * 130)
    print(f"  {'전략':<28}", end="")
    for y in years:
        print(f" {y:>8}", end="")
    print(f" {'전체':>9}")
    print(f"  {'─' * 115}")

    for r in results[:10]:
        ym = r['yearly']
        m = r['metrics']
        row = f"  {r['label']:<28}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['MDD']:>7.1%}"
            else:
                row += f" {'─':>8}"
        row += f" {m['MDD']:>8.1%}"
        print(row)

    # Year-by-year Sharpe for top 10
    print()
    print("=" * 130)
    print("  상위 10개 전략 연도별 Sharpe")
    print("=" * 130)
    print(f"  {'전략':<28}", end="")
    for y in years:
        print(f" {y:>8}", end="")
    print(f" {'전체':>9}")
    print(f"  {'─' * 115}")

    for r in results[:10]:
        ym = r['yearly']
        m = r['metrics']
        row = f"  {r['label']:<28}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['Sharpe']:>8.3f}"
            else:
                row += f" {'─':>8}"
        row += f" {m['Sharpe']:>9.3f}"
        print(row)

    # Lockbox (2025) for top 15
    print()
    print("=" * 130)
    print("  상위 15개 전략 — 2025 Lockbox (OOS)")
    print("=" * 130)

    lockbox_params = []
    lockbox_labels = []
    for r in results[:15]:
        p = r['params']
        lp = Params(**{l: getattr(p, l) for l in
                      ('canary','health','selection','weighting','rebalancing','risk')},
                   start_date='2025-01-01', end_date='2025-06-30')
        lockbox_params.append(lp)
        lockbox_labels.append(r['label'])

    with mp.Pool(N_WORKERS) as pool:
        lb_results = pool.map(run_single, lockbox_params)

    print(f"\n  {'#':>3} {'전략':<28} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8} {'Final($)':>10}"
          f"  vs 전구간Sharpe")
    print(f"  {'─' * 85}")

    for i, (lr, fr) in enumerate(zip(lb_results, results[:15])):
        lm = lr['metrics']
        fm = fr['metrics']
        print(f"  {i+1:>3} {lockbox_labels[i]:<28} {lm['Sharpe']:>8.3f} {lm['CAGR']:>+7.1%}"
              f" {lm['MDD']:>7.1%} ${lm['Final']:>9,.0f}  (전구간: {fm['Sharpe']:.3f})")

    # Summary
    print()
    print("=" * 130)
    print("  레이어 수별 최고 전략")
    print("=" * 130)
    by_layers = {}
    for r in results:
        n = r['label'].count('+') + 1 if r['label'] != 'BASELINE' else 0
        if n not in by_layers or r['metrics']['Sharpe'] > by_layers[n]['metrics']['Sharpe']:
            by_layers[n] = r

    print(f"\n  {'Layers':>6} {'전략':<28} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8} {'Final($)':>12}")
    print(f"  {'─' * 80}")
    for n in sorted(by_layers):
        r = by_layers[n]
        m = r['metrics']
        print(f"  {n:>6} {r['label']:<28} {m['Sharpe']:>8.3f} {m['CAGR']:>+7.1%}"
              f" {m['MDD']:>7.1%} {m['Final']:>11,.0f}")


if __name__ == '__main__':
    main()
