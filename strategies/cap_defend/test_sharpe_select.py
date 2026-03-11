#!/usr/bin/env python3
"""Test Sharpe-based coin selection (S6/S7/S8) vs existing S variants.
   K5+H5 fixed. Compare tx=0 vs tx=0.4%."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

# All selection strategies to compare
STRATEGIES = [
    ('baseline (시총순)',           Params(canary='K5', health='H5')),
    ('S1 (Top15→mom21 Top5)',      Params(canary='K5', health='H5', selection='S1')),
    ('S2 (시총+mom21 블렌드)',      Params(canary='K5', health='H5', selection='S2')),
    ('S3 (시총3 + mom21 Top2)',    Params(canary='K5', health='H5', selection='S3')),
    ('S4 (BTC+ETH 고정)',          Params(canary='K5', health='H5', selection='S4')),
    ('S5 (보유 우선 +2)',          Params(canary='K5', health='H5', selection='S5')),
    ('S6 (Sharpe126+252 순위)',    Params(canary='K5', health='H5', selection='S6')),
    ('S7 (Top15→Sharpe Top5)',     Params(canary='K5', health='H5', selection='S7')),
    ('S8 (Sharpe + 보유우선)',     Params(canary='K5', health='H5', selection='S8')),
]

# Also test with W1 (best weighting)
STRATEGIES_W1 = [
    ('baseline+W1',           Params(canary='K5', health='H5', weighting='W1')),
    ('S5+W1',                 Params(canary='K5', health='H5', selection='S5', weighting='W1')),
    ('S6+W1',                 Params(canary='K5', health='H5', selection='S6', weighting='W1')),
    ('S7+W1',                 Params(canary='K5', health='H5', selection='S7', weighting='W1')),
    ('S8+W1',                 Params(canary='K5', health='H5', selection='S8', weighting='W1')),
]

def run_set(strategies, prices, universe, tx):
    params = [s[1] for s in strategies]
    # Override tx_cost
    params_tx = [Params(canary=p.canary, health=p.health, selection=p.selection,
                        weighting=p.weighting, rebalancing=p.rebalancing,
                        risk=p.risk, tx_cost=tx) for p in params]
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, params_tx)
    return results


def print_table(title, strategies, results_0, results_4):
    print(f"\n{'=' * 130}")
    print(f"  {title}")
    print(f"{'=' * 130}")
    print(f"\n  {'전략':<30}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
          f" │{'비용손실':>7}")
    print(f"  {'':>30}"
          f" │{'─── tx=0 ───':^33}"
          f" │{'─── tx=0.4% ───':^33}"
          f" │{'ΔCAGR':>7}")
    print(f"  {'─' * 125}")

    for (name, _), r0, r4 in zip(strategies, results_0, results_4):
        m0, m4 = r0['metrics'], r4['metrics']
        drag = m0['CAGR'] - m4['CAGR']
        print(f"  {name:<30}"
              f" │{m0['Sharpe']:>7.3f} {m0['CAGR']:>+6.1%} {m0['MDD']:>6.1%} {m0['Final']:>10,.0f}"
              f" │{m4['Sharpe']:>7.3f} {m4['CAGR']:>+6.1%} {m4['MDD']:>6.1%} {m4['Final']:>10,.0f}"
              f" │{drag:>+6.1%}")


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    # Run all
    t0 = time.time()
    r0_base = run_set(STRATEGIES, prices, universe, tx=0.0)
    r4_base = run_set(STRATEGIES, prices, universe, tx=0.004)
    r0_w1 = run_set(STRATEGIES_W1, prices, universe, tx=0.0)
    r4_w1 = run_set(STRATEGIES_W1, prices, universe, tx=0.004)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    print_table("K5+H5 고정 — 코인 선택 방식 비교 (균등배분)", STRATEGIES, r0_base, r4_base)
    print_table("K5+H5+W1 고정 — 코인 선택 방식 비교 (순위가중)", STRATEGIES_W1, r0_w1, r4_w1)

    # Year-by-year for key strategies
    print(f"\n{'=' * 130}")
    print(f"  연도별 CAGR (tx=0.4%) — 주요 선택 전략")
    print(f"{'=' * 130}")

    years = range(2018, 2026)
    key_indices = [0, 4, 5, 6, 7, 8]  # baseline, S4, S5, S6, S7, S8
    print(f"  {'전략':<30}", end="")
    for y in years:
        print(f" {y:>8}", end="")
    print(f" {'전체':>9}")
    print(f"  {'─' * 105}")

    for idx in key_indices:
        name = STRATEGIES[idx][0]
        ym = r4_base[idx]['yearly']
        m = r4_base[idx]['metrics']
        row = f"  {name:<30}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+7.1%}"
            else:
                row += f" {'─':>8}"
        row += f" {m['CAGR']:>+8.1%}"
        print(row)

    # Year-by-year Sharpe
    print(f"\n  연도별 Sharpe (tx=0.4%)")
    print(f"  {'─' * 105}")
    for idx in key_indices:
        name = STRATEGIES[idx][0]
        ym = r4_base[idx]['yearly']
        m = r4_base[idx]['metrics']
        row = f"  {name:<30}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['Sharpe']:>8.3f}"
            else:
                row += f" {'─':>8}"
        row += f" {m['Sharpe']:>9.3f}"
        print(row)


if __name__ == '__main__':
    main()
