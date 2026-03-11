#!/usr/bin/env python3
"""Extended universe size test: T50~T100 × Blacklist.
Reuses test_matrix infrastructure. Breadth excluded (proven ineffective)."""

import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    Params, load_all_prices, filter_universe, load_universe,
)
from test_matrix import (
    B, run_matrix_backtest, ANCHOR_DAYS,
)

def main():
    print("Loading data...")

    um_raw = load_universe()

    universe_sizes = [50, 60, 70, 80, 100]
    universe_maps = {}

    for top_n in universe_sizes:
        fm = filter_universe(um_raw, top_n)
        universe_maps[top_n] = fm

    # Load prices for all tickers
    all_tickers = set()
    for fm in universe_maps.values():
        for ts in fm.values():
            all_tickers.update(ts)
    all_tickers.update(['BTC-USD', 'ETH-USD'])

    prices = load_all_prices(all_tickers)
    print(f"  {len(prices)} tickers loaded")

    # Check actual universe sizes (some months have fewer coins)
    print("\n  실제 유니버스 크기 (price 데이터 보유 기준):")
    for top_n in universe_sizes:
        fm = universe_maps[top_n]
        actual = []
        for k, tickers in fm.items():
            avail = sum(1 for t in tickers if t in prices)
            actual.append(avail)
        print(f"    T{top_n}: min={min(actual)}, max={max(actual)}, avg={np.mean(actual):.0f}")

    t0 = time.time()

    # No breadth, just blacklist options
    bl_options = [(0, 0), (-0.15, 7), (-0.20, 7)]

    configs = []
    for top_n in universe_sizes:
        for bl_drop, bl_days in bl_options:
            label = f"T{top_n}"
            if bl_drop < 0:
                label += f"+BL{int(abs(bl_drop)*100)}%"
            configs.append((label, top_n, bl_drop, bl_days))

    print(f"  {len(configs)} configs × {len(ANCHOR_DAYS)} anchors = {len(configs)*len(ANCHOR_DAYS)} backtests")

    all_results = {c[0]: [] for c in configs}
    all_rebals = {c[0]: [] for c in configs}

    for base_d in ANCHOR_DAYS:
        print(f"  Anchor {base_d:>2}: ", end="", flush=True)

        for label, top_n, bl_drop, bl_days in configs:
            snap_days = [(base_d - 1 + j * 9) % 28 + 1 for j in range(3)]
            r = run_matrix_backtest(
                prices, universe_maps[top_n], snap_days,
                breadth_threshold=0,
                bl_drop=bl_drop, bl_days=bl_days,
                drift_threshold=0.10,
                post_flip_delay=5, params_base=B()
            )
            all_results[label].append(r['metrics'])
            all_rebals[label].append(r['rebal_count'])

        # Show samples
        samples = []
        for tn in universe_sizes:
            s = all_results[f'T{tn}'][-1]['Sharpe']
            samples.append(f"T{tn}={s:.3f}")
        print("  ".join(samples))

    print(f"\n  Completed in {time.time()-t0:.1f}s")

    def avg_metrics(results):
        avg = {}
        for key in results[0]:
            avg[key] = np.mean([r[key] for r in results])
        std_s = np.std([r['Sharpe'] for r in results])
        return avg, std_s

    base_sharpe = avg_metrics(all_results['T50'])[0]['Sharpe']

    # ── Section 1: Universe size comparison (baseline only) ──
    print(f"\n{'=' * 100}")
    print(f"  1. 유니버스 크기별 비교 (블랙리스트 OFF)")
    print(f"{'=' * 100}")

    print(f"\n  {'전략':>12} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'리밸':>5}")
    print(f"  {'─' * 70}")
    for top_n in universe_sizes:
        label = f"T{top_n}"
        avg, std = avg_metrics(all_results[label])
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        delta = avg['Sharpe'] - base_sharpe
        avg_rebal = np.mean(all_rebals[label])
        marker = ' ★' if delta > 0.03 else ' ↑' if delta > 0 else ''
        print(f"  {'Top'+str(top_n):>12} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f} {avg_rebal:>5.0f}{marker}")

    # ── Section 2: Full results ──
    print(f"\n{'=' * 100}")
    print(f"  2. 전체 결과 (유니버스 × 블랙리스트)")
    print(f"{'=' * 100}")

    print(f"\n  {'전략':>20} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'리밸':>5}")
    print(f"  {'─' * 75}")

    for top_n in universe_sizes:
        for label, tn, bl_drop, bl_days in configs:
            if tn != top_n:
                continue
            avg, std = avg_metrics(all_results[label])
            calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
            delta = avg['Sharpe'] - base_sharpe
            avg_rebal = np.mean(all_rebals[label])
            marker = ' ★' if delta > 0.03 else ' ↑' if delta > 0 else ''
            print(f"  {label:>20} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
                  f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
                  f" {avg_rebal:>5.0f}{marker}")
        print()

    # ── Section 3: Rankings ──
    print(f"{'=' * 100}")
    print(f"  3. Sharpe 순위")
    print(f"{'=' * 100}")

    ranked = sorted(all_results.items(),
                    key=lambda x: avg_metrics(x[1])[0]['Sharpe'], reverse=True)

    print(f"\n  {'순위':>3} {'전략':>20} {'Sharpe':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 65}")
    for i, (label, results) in enumerate(ranked, 1):
        avg, std = avg_metrics(results)
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        print(f"  {i:>3}. {label:>20} {avg['Sharpe']:>7.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}")

    print(f"\n  Total: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
