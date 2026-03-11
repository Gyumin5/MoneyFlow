#!/usr/bin/env python3
"""Test K6 graduated canary (5-level SMA entry) with H1 health filter."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

STRATEGIES = [
    # References
    ('K5+H1 sma80 (기존최적)',  Params(canary='K5', health='H1', sma_period=80)),
    ('K5+H1 sma100',          Params(canary='K5', health='H1', sma_period=100)),
    ('K5+H5 sma150 (구기준)',  Params(canary='K5', health='H5', sma_period=150)),
    # K6 graduated
    ('K6+H1 (점진적진입)',      Params(canary='K6', health='H1')),
    ('K6+H5',                  Params(canary='K6', health='H5')),
    ('K6+baseline H',         Params(canary='K6')),
]


def run_set(strategies, prices, universe, tx=0.004):
    params_list = []
    for _, p in strategies:
        params_list.append(Params(
            canary=p.canary, health=p.health, selection=p.selection,
            weighting=p.weighting, rebalancing=p.rebalancing,
            risk=p.risk, tx_cost=tx, sma_period=p.sma_period
        ))
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, params_list)
    return results


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()
    r0 = run_set(STRATEGIES, prices, universe, tx=0.0)
    r4 = run_set(STRATEGIES, prices, universe, tx=0.004)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # ── Main comparison ──────────────────────────────────────────
    print(f"\n{'=' * 130}")
    print(f"  K6 점진적 진입 vs K5 이진 카나리 (tx=0 / tx=0.4%)")
    print(f"{'=' * 130}")
    print(f"\n  {'전략':<25}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
          f" │{'ΔCAGR':>7} {'리밸':>4}")
    print(f"  {'':>25}"
          f" │{'─── tx=0 ───':^33}"
          f" │{'─── tx=0.4% ───':^33}"
          f" │{'비용':>7}")
    print(f"  {'─' * 125}")

    for (name, _), rx0, rx4 in zip(STRATEGIES, r0, r4):
        m0, m4 = rx0['metrics'], rx4['metrics']
        drag = m0['CAGR'] - m4['CAGR']
        print(f"  {name:<25}"
              f" │{m0['Sharpe']:>7.3f} {m0['CAGR']:>+6.1%} {m0['MDD']:>6.1%} {m0['Final']:>10,.0f}"
              f" │{m4['Sharpe']:>7.3f} {m4['CAGR']:>+6.1%} {m4['MDD']:>6.1%} {m4['Final']:>10,.0f}"
              f" │{drag:>+6.1%} {rx4['rebal_count']:>4}")

    # ── Year-by-year ─────────────────────────────────────────────
    years = range(2018, 2026)
    print(f"\n  연도별 CAGR (tx=0.4%)")
    print(f"  {'─' * 110}")
    print(f"  {'전략':<25}", end="")
    for y in years:
        print(f" {y:>8}", end="")
    print(f" {'전체':>9}")
    print(f"  {'─' * 100}")

    for (name, _), r in zip(STRATEGIES, r4):
        ym = r['yearly']
        m = r['metrics']
        row = f"  {name:<25}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+7.1%}"
            else:
                row += f" {'─':>8}"
        row += f" {m['CAGR']:>+8.1%}"
        print(row)

    # Sharpe
    print(f"\n  연도별 Sharpe (tx=0.4%)")
    print(f"  {'─' * 100}")
    for (name, _), r in zip(STRATEGIES, r4):
        ym = r['yearly']
        m = r['metrics']
        row = f"  {name:<25}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['Sharpe']:>8.3f}"
            else:
                row += f" {'─':>8}"
        row += f" {m['Sharpe']:>9.3f}"
        print(row)

    # MDD
    print(f"\n  연도별 MDD (tx=0.4%)")
    print(f"  {'─' * 100}")
    for (name, _), r in zip(STRATEGIES, r4):
        ym = r['yearly']
        m = r['metrics']
        row = f"  {name:<25}"
        for y in years:
            if y in ym:
                row += f" {ym[y].get('MDD', 0):>7.1%}"
            else:
                row += f" {'─':>8}"
        row += f" {m['MDD']:>8.1%}"
        print(row)


if __name__ == '__main__':
    main()
