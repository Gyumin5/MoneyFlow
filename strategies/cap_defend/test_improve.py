#!/usr/bin/env python3
"""Test 3 improvements with date-averaging:
   1. Tranche rebalancing (structural anti-overfit)
   2. Yellow mode (partial defense on breadth collapse)
   3. OFF→ON refresh (post-flip extra rebalance)

All tests use 10-date averaging to avoid anchor-date overfitting."""

import os, sys, time
import multiprocessing as mp
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (Params, load_data, init_pool, run_single,
                              run_backtest, calc_metrics, calc_yearly_metrics)

N_WORKERS = min(24, mp.cpu_count())
ANCHOR_DAYS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]


def B(**kw):
    base = dict(
        canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1,
        health='HK', health_sma=2, health_mom_short=21,
        health_mom_long=90, vol_cap=0.05,
    )
    base.update(kw)
    return Params(**base)


# ═══════════════════════════════════════════════════════════════════
# Generate all params for batch execution
# ═══════════════════════════════════════════════════════════════════

def build_all_params():
    """Build all params and a map for post-processing.
    Returns (params_list, param_map).
    param_map: {(test_name, anchor_day, tranche_idx): index_in_list}
    """
    all_params = []
    param_map = {}

    def add(name, anchor_day, tranche_idx, p):
        idx = len(all_params)
        param_map[(name, anchor_day, tranche_idx)] = idx
        all_params.append(p)

    # ── 0. Baseline (date-averaged) ──
    for d in ANCHOR_DAYS:
        add('BASELINE', d, 0, B(rebalancing=f'RX{d}'))

    # ── 1. Tranche rebalancing ──
    # N tranches with roughly equal spacing within a month
    TRANCHE_CONFIGS = {2: 14, 3: 9, 4: 7}
    for n_tr, sp in TRANCHE_CONFIGS.items():
        for d in ANCHOR_DAYS:
            for ti in range(n_tr):
                anchor = (d - 1 + ti * sp) % 28 + 1
                p = B(rebalancing=f'RX{anchor}')
                p = Params(**{k: getattr(p, k) for k in p.__dataclass_fields__})
                p.initial_capital = 10000.0 / n_tr
                add(f'TR{n_tr}', d, ti, p)

    # ── 2. Yellow mode (breadth collapse → 50% cash) ──
    for yt in [2, 3, 4]:
        for d in ANCHOR_DAYS:
            add(f'YT{yt}', d, 0, B(rebalancing=f'RX{d}', yellow_threshold=yt))

    # ── 3. Post-flip delay (OFF→ON refresh) ──
    for pfd in [3, 5, 7, 10]:
        for d in ANCHOR_DAYS:
            add(f'PFD{pfd}', d, 0, B(rebalancing=f'RX{d}', post_flip_delay=pfd))

    # ── 4. Combinations ──
    # TR3 + YT3
    for d in ANCHOR_DAYS:
        for ti in range(3):
            anchor = (d - 1 + ti * 9) % 28 + 1
            p = B(rebalancing=f'RX{anchor}', yellow_threshold=3)
            p = Params(**{k: getattr(p, k) for k in p.__dataclass_fields__})
            p.initial_capital = 10000.0 / 3
            add('TR3+YT3', d, ti, p)

    # TR3 + PFD5
    for d in ANCHOR_DAYS:
        for ti in range(3):
            anchor = (d - 1 + ti * 9) % 28 + 1
            p = B(rebalancing=f'RX{anchor}', post_flip_delay=5)
            p = Params(**{k: getattr(p, k) for k in p.__dataclass_fields__})
            p.initial_capital = 10000.0 / 3
            add('TR3+PFD5', d, ti, p)

    # YT3 + PFD5
    for d in ANCHOR_DAYS:
        add('YT3+PFD5', d, 0, B(rebalancing=f'RX{d}', yellow_threshold=3, post_flip_delay=5))

    # TR3 + YT3 + PFD5
    for d in ANCHOR_DAYS:
        for ti in range(3):
            anchor = (d - 1 + ti * 9) % 28 + 1
            p = B(rebalancing=f'RX{anchor}', yellow_threshold=3, post_flip_delay=5)
            p = Params(**{k: getattr(p, k) for k in p.__dataclass_fields__})
            p.initial_capital = 10000.0 / 3
            add('TR3+YT3+PFD5', d, ti, p)

    return all_params, param_map


# ═══════════════════════════════════════════════════════════════════
# Post-processing: combine tranche equity curves, date-average
# ═══════════════════════════════════════════════════════════════════

def get_test_names(param_map):
    """Get unique test names in order."""
    seen = set()
    names = []
    for (name, _, _) in param_map:
        if name not in seen:
            seen.add(name)
            names.append(name)
    return names


def is_tranche(name):
    return name.startswith('TR')


def get_n_tranches(name):
    """Extract number of tranches from name like TR3, TR3+YT3."""
    for part in name.split('+'):
        if part.startswith('TR'):
            return int(part[2:])
    return 1


def process_results(results, param_map):
    """Process raw results into date-averaged metrics per test."""
    test_names = get_test_names(param_map)
    output = {}

    for name in test_names:
        n_tr = get_n_tranches(name) if is_tranche(name) else 1
        all_metrics = []
        all_yearly = {}

        for d in ANCHOR_DAYS:
            if n_tr > 1:
                # Combine tranche equity curves
                tranche_pvs = []
                for ti in range(n_tr):
                    key = (name, d, ti)
                    if key not in param_map:
                        continue
                    idx = param_map[key]
                    r = results[idx]
                    if r['pv'] is not None and len(r['pv']) > 0:
                        tranche_pvs.append(r['pv'])

                if not tranche_pvs:
                    continue

                # Sum equity curves (align on dates)
                combined = tranche_pvs[0].copy()
                for tv in tranche_pvs[1:]:
                    combined = combined.add(tv, fill_value=0)

                m = calc_metrics(combined)
                ym = calc_yearly_metrics(combined)
            else:
                key = (name, d, 0)
                if key not in param_map:
                    continue
                idx = param_map[key]
                r = results[idx]
                m = r['metrics']
                ym = r['yearly']

            all_metrics.append(m)
            for y, v in ym.items():
                if y not in all_yearly:
                    all_yearly[y] = []
                all_yearly[y].append(v)

        if not all_metrics:
            continue

        # Average metrics across anchor dates
        avg = {}
        for key in all_metrics[0]:
            vals = [am[key] for am in all_metrics]
            avg[key] = np.mean(vals)

        # Std of Sharpe (for robustness check)
        sharpe_std = np.std([am['Sharpe'] for am in all_metrics])

        # Average yearly
        avg_yearly = {}
        for y, yms in all_yearly.items():
            avg_yearly[y] = {}
            for key in yms[0]:
                avg_yearly[y][key] = np.mean([ym[key] for ym in yms])

        output[name] = {
            'metrics': avg,
            'yearly': avg_yearly,
            'sharpe_std': sharpe_std,
            'n_samples': len(all_metrics),
        }

    return output


# ═══════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════

LABELS = {
    'BASELINE': '베이스라인 (월간EW)',
    'TR2': '트랜치 2분할',
    'TR3': '트랜치 3분할',
    'TR4': '트랜치 4분할',
    'YT2': '옐로우 h<2 50%',
    'YT3': '옐로우 h<3 50%',
    'YT4': '옐로우 h<4 50%',
    'PFD3': 'OFF→ON +3일 리프레시',
    'PFD5': 'OFF→ON +5일 리프레시',
    'PFD7': 'OFF→ON +7일 리프레시',
    'PFD10': 'OFF→ON +10일 리프레시',
    'TR3+YT3': '트랜치3+옐로우3',
    'TR3+PFD5': '트랜치3+리프레시5',
    'YT3+PFD5': '옐로우3+리프레시5',
    'TR3+YT3+PFD5': '트랜치3+옐로우3+리프레시5',
}


def print_results(output):
    base = output.get('BASELINE', {}).get('metrics', {})
    base_s = base.get('Sharpe', 0)

    # ── Main table ──
    print(f"\n{'=' * 115}")
    print(f"  개선안 테스트 — 날짜 평균 (10 앵커)")
    print(f"{'=' * 115}")
    print(f"\n  {'전략':<28} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 80}")

    for name in ['BASELINE', 'TR2', 'TR3', 'TR4',
                  'YT2', 'YT3', 'YT4',
                  'PFD3', 'PFD5', 'PFD7', 'PFD10',
                  'TR3+YT3', 'TR3+PFD5', 'YT3+PFD5', 'TR3+YT3+PFD5']:
        if name not in output:
            continue
        r = output[name]
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        delta = m['Sharpe'] - base_s
        label = LABELS.get(name, name)

        # Section separator
        if name in ('TR2', 'YT2', 'PFD3', 'TR3+YT3'):
            print(f"  {'─' * 80}")

        marker = ' ★' if delta > 0.05 else ' ↑' if delta > 0 else ''
        print(f"  {label:<28} {m['Sharpe']:>7.3f} {delta:>+6.3f} {r['sharpe_std']:>6.3f}"
              f" {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}{marker}")

    # ── Year-by-year for key strategies ──
    key_strats = ['BASELINE', 'TR3', 'YT3', 'PFD5', 'TR3+YT3+PFD5']
    key_strats = [s for s in key_strats if s in output]
    years = range(2018, 2026)

    print(f"\n{'=' * 115}")
    print(f"  주요 전략 연도별 CAGR (날짜 평균)")
    print(f"{'=' * 115}")
    print(f"  {'전략':<28}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print()
    print(f"  {'─' * 95}")
    for name in key_strats:
        r = output[name]
        ym = r['yearly']
        label = LABELS.get(name, name)
        row = f"  {label:<28}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+6.1%}"
            else:
                row += f" {'─':>7}"
        print(row)

    # ── Year-by-year Sharpe ──
    print(f"\n  주요 전략 연도별 Sharpe")
    print(f"  {'전략':<28}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print()
    print(f"  {'─' * 95}")
    for name in key_strats:
        r = output[name]
        ym = r['yearly']
        label = LABELS.get(name, name)
        row = f"  {label:<28}"
        for y in years:
            if y in ym:
                row += f" {ym[y].get('Sharpe', 0):>7.2f}"
            else:
                row += f" {'─':>7}"
        print(row)

    # ── Year-by-year MDD ──
    print(f"\n  주요 전략 연도별 MDD")
    print(f"  {'전략':<28}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print()
    print(f"  {'─' * 95}")
    for name in key_strats:
        r = output[name]
        ym = r['yearly']
        label = LABELS.get(name, name)
        row = f"  {label:<28}"
        for y in years:
            if y in ym:
                row += f" {ym[y].get('MDD', 0):>6.1%}"
            else:
                row += f" {'─':>7}"
        print(row)

    # ── Sharpe std comparison (robustness) ──
    print(f"\n{'=' * 115}")
    print(f"  앵커 날짜 민감도 (σ가 낮을수록 강건)")
    print(f"{'=' * 115}")
    items = [(name, output[name]) for name in output]
    items.sort(key=lambda x: x[1]['sharpe_std'])
    print(f"\n  {'전략':<28} {'σ(Sharpe)':>10} {'Sharpe':>7}")
    print(f"  {'─' * 50}")
    for name, r in items:
        label = LABELS.get(name, name)
        print(f"  {label:<28} {r['sharpe_std']:>10.3f} {r['metrics']['Sharpe']:>7.3f}")


def main():
    print("Loading data...")
    prices, universe = load_data()

    all_params, param_map = build_all_params()
    print(f"  {len(prices)} tickers loaded, {len(all_params)} backtests to run")

    t0 = time.time()
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, all_params)
    print(f"\n  Completed {len(results)} backtests in {time.time()-t0:.1f}s")

    output = process_results(results, param_map)
    print_results(output)

    print(f"\n  Total time: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
