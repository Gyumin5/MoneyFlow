#!/usr/bin/env python3
"""Test inverse volatility weighting: baseline(EW) vs W2(70/30) vs W6(100%) vs W1(rank decay).
   K5+H5 variants included. tx=0 and tx=0.4%."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

# ── Strategy definitions ──────────────────────────────────────────
STRATEGIES = [
    # Baseline (no K/H)
    ('EW (균등배분)',              Params()),
    ('W1 (순위가중)',             Params(weighting='W1')),
    ('W2 (inv-vol 70/30)',       Params(weighting='W2')),
    ('W6 (inv-vol 100%)',        Params(weighting='W6')),
    # K5+H5
    ('K5+H5+EW',                 Params(canary='K5', health='H5')),
    ('K5+H5+W1',                 Params(canary='K5', health='H5', weighting='W1')),
    ('K5+H5+W2',                 Params(canary='K5', health='H5', weighting='W2')),
    ('K5+H5+W6',                 Params(canary='K5', health='H5', weighting='W6')),
    # K5+H5+S5 combos (best selection)
    ('K5+H5+S5+EW',              Params(canary='K5', health='H5', selection='S5')),
    ('K5+H5+S5+W1',              Params(canary='K5', health='H5', selection='S5', weighting='W1')),
    ('K5+H5+S5+W2',              Params(canary='K5', health='H5', selection='S5', weighting='W2')),
    ('K5+H5+S5+W6',              Params(canary='K5', health='H5', selection='S5', weighting='W6')),
    # Best combo with R2
    ('K5+H5+S5+W1+R2',           Params(canary='K5', health='H5', selection='S5', weighting='W1', rebalancing='R2')),
    ('K5+H5+S5+W6+R2',           Params(canary='K5', health='H5', selection='S5', weighting='W6', rebalancing='R2')),
]


def run_set(strategies, prices, universe, tx):
    params_list = []
    for _, p in strategies:
        params_list.append(Params(
            canary=p.canary, health=p.health, selection=p.selection,
            weighting=p.weighting, rebalancing=p.rebalancing,
            risk=p.risk, tx_cost=tx
        ))
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, params_list)
    return results


def print_table(title, strategies, results_0, results_4):
    print(f"\n{'=' * 135}")
    print(f"  {title}")
    print(f"{'=' * 135}")
    print(f"\n  {'전략':<25}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
          f" │{'ΔCAGR':>7}")
    print(f"  {'':>25}"
          f" │{'─── tx=0 ───':^33}"
          f" │{'─── tx=0.4% ───':^33}"
          f" │{'비용손실':>7}")
    print(f"  {'─' * 130}")

    for (name, _), r0, r4 in zip(strategies, results_0, results_4):
        m0, m4 = r0['metrics'], r4['metrics']
        drag = m0['CAGR'] - m4['CAGR']
        print(f"  {name:<25}"
              f" │{m0['Sharpe']:>7.3f} {m0['CAGR']:>+6.1%} {m0['MDD']:>6.1%} {m0['Final']:>10,.0f}"
              f" │{m4['Sharpe']:>7.3f} {m4['CAGR']:>+6.1%} {m4['MDD']:>6.1%} {m4['Final']:>10,.0f}"
              f" │{drag:>+6.1%}")


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()
    r0 = run_set(STRATEGIES, prices, universe, tx=0.0)
    r4 = run_set(STRATEGIES, prices, universe, tx=0.004)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    print_table("역변동성(Inverse Volatility) 가중 비교", STRATEGIES, r0, r4)

    # ── Year-by-year CAGR (tx=0.4%) ──────────────────────────────
    print(f"\n{'=' * 135}")
    print(f"  연도별 CAGR (tx=0.4%)")
    print(f"{'=' * 135}")

    years = range(2018, 2026)
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

    # ── Year-by-year Sharpe (tx=0.4%) ──────────────────────────────
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

    # ── Highlight: EW vs W6 direct comparison ──────────────────────
    print(f"\n{'=' * 135}")
    print(f"  핵심 비교: 균등배분(EW) vs 순수 역변동성(W6) — tx=0.4%")
    print(f"{'=' * 135}")

    pairs = [
        (0, 3, "Baseline"),
        (4, 7, "K5+H5"),
        (8, 11, "K5+H5+S5"),
    ]
    print(f"\n  {'구성':<15} {'가중':<8} │{'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Final':>10}"
          f" │ {'ΔCAGR vs EW':>12}")
    print(f"  {'─' * 80}")

    for ew_idx, w6_idx, label in pairs:
        m_ew = r4[ew_idx]['metrics']
        m_w6 = r4[w6_idx]['metrics']
        delta = m_w6['CAGR'] - m_ew['CAGR']
        print(f"  {label:<15} {'EW':<8}"
              f" │{m_ew['Sharpe']:>7.3f} {m_ew['CAGR']:>+7.1%} {m_ew['MDD']:>6.1%} {m_ew['Final']:>10,.0f}"
              f" │ {'─':>12}")
        print(f"  {'':<15} {'W6':<8}"
              f" │{m_w6['Sharpe']:>7.3f} {m_w6['CAGR']:>+7.1%} {m_w6['MDD']:>6.1%} {m_w6['Final']:>10,.0f}"
              f" │ {delta:>+11.1%}")
        print()


if __name__ == '__main__':
    main()
