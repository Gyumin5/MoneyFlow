#!/usr/bin/env python3
"""Final comparison: R alternatives (R2,R6-R9) + S alternatives (S5,S9,S10) + EW baseline.
   Base: K5+H5+EW. All tx=0.4%."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

# ══════════════════════════════════════════════════════════════════
# Part 1: R variants (K5+H5+EW base)
# ══════════════════════════════════════════════════════════════════
R_STRATEGIES = [
    ('K5+H5 (월간리밸)',          Params(canary='K5', health='H5')),
    ('+ R2 (MTD-15%→현금)',      Params(canary='K5', health='H5', rebalancing='R2')),
    ('+ R7 (MTD-10%→현금)',      Params(canary='K5', health='H5', rebalancing='R7')),
    ('+ R8 (MTD-20%→현금)',      Params(canary='K5', health='H5', rebalancing='R8')),
    ('+ R6 (60d고점-20%)',       Params(canary='K5', health='H5', rebalancing='R6')),
    ('+ R9 (30d고점-15%)',       Params(canary='K5', health='H5', rebalancing='R9')),
]

# ══════════════════════════════════════════════════════════════════
# Part 2: S variants (K5+H5+EW base)
# ══════════════════════════════════════════════════════════════════
S_STRATEGIES = [
    ('시총순 Top5 (baseline)',    Params(canary='K5', health='H5')),
    ('S5 (기존보유 +2순위)',       Params(canary='K5', health='H5', selection='S5')),
    ('S9 (히스테리시스 3순위)',     Params(canary='K5', health='H5', selection='S9')),
    ('S10 (보유코인 우선유지)',     Params(canary='K5', health='H5', selection='S10')),
]

# ══════════════════════════════════════════════════════════════════
# Part 3: Best combos
# ══════════════════════════════════════════════════════════════════
COMBO_STRATEGIES = [
    ('K5+H5 (base)',              Params(canary='K5', health='H5')),
    ('K5+H5+S5',                  Params(canary='K5', health='H5', selection='S5')),
    ('K5+H5+S5+R2',              Params(canary='K5', health='H5', selection='S5', rebalancing='R2')),
    ('K5+H5+S9',                  Params(canary='K5', health='H5', selection='S9')),
    ('K5+H5+S10',                 Params(canary='K5', health='H5', selection='S10')),
    ('K5+H5+S9+R2',              Params(canary='K5', health='H5', selection='S9', rebalancing='R2')),
    ('K5+H5+S10+R2',             Params(canary='K5', health='H5', selection='S10', rebalancing='R2')),
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
    print(f"\n{'=' * 130}")
    print(f"  {title}")
    print(f"{'=' * 130}")
    print(f"\n  {'전략':<28}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
          f" │{'ΔCAGR':>7}")
    print(f"  {'':>28}"
          f" │{'─── tx=0 ───':^33}"
          f" │{'─── tx=0.4% ───':^33}"
          f" │{'비용':>7}")
    print(f"  {'─' * 125}")

    for (name, _), r0, r4 in zip(strategies, results_0, results_4):
        m0, m4 = r0['metrics'], r4['metrics']
        drag = m0['CAGR'] - m4['CAGR']
        print(f"  {name:<28}"
              f" │{m0['Sharpe']:>7.3f} {m0['CAGR']:>+6.1%} {m0['MDD']:>6.1%} {m0['Final']:>10,.0f}"
              f" │{m4['Sharpe']:>7.3f} {m4['CAGR']:>+6.1%} {m4['MDD']:>6.1%} {m4['Final']:>10,.0f}"
              f" │{drag:>+6.1%}")


def print_yearly(title, strategies, results):
    print(f"\n{'=' * 130}")
    print(f"  {title}")
    print(f"{'=' * 130}")

    years = range(2018, 2026)
    print(f"\n  {'전략':<28}", end="")
    for y in years:
        print(f" {y:>8}", end="")
    print(f" {'전체':>9}")
    print(f"  {'─' * 105}")

    for (name, _), r in zip(strategies, results):
        ym = r['yearly']
        m = r['metrics']
        row = f"  {name:<28}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+7.1%}"
            else:
                row += f" {'─':>8}"
        row += f" {m['CAGR']:>+8.1%}"
        print(row)

    # Sharpe
    print(f"\n  연도별 Sharpe")
    print(f"  {'─' * 105}")
    for (name, _), r in zip(strategies, results):
        ym = r['yearly']
        m = r['metrics']
        row = f"  {name:<28}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['Sharpe']:>8.3f}"
            else:
                row += f" {'─':>8}"
        row += f" {m['Sharpe']:>9.3f}"
        print(row)

    # MDD
    print(f"\n  연도별 MDD")
    print(f"  {'─' * 105}")
    for (name, _), r in zip(strategies, results):
        ym = r['yearly']
        m = r['metrics']
        row = f"  {name:<28}"
        for y in years:
            if y in ym:
                row += f" {ym[y].get('MDD', 0):>7.1%}"
            else:
                row += f" {'─':>8}"
        row += f" {m['MDD']:>8.1%}"
        print(row)


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()

    # Part 1: R variants
    r0_R = run_set(R_STRATEGIES, prices, universe, tx=0.0)
    r4_R = run_set(R_STRATEGIES, prices, universe, tx=0.004)

    # Part 2: S variants
    r0_S = run_set(S_STRATEGIES, prices, universe, tx=0.0)
    r4_S = run_set(S_STRATEGIES, prices, universe, tx=0.004)

    # Part 3: Best combos
    r0_C = run_set(COMBO_STRATEGIES, prices, universe, tx=0.0)
    r4_C = run_set(COMBO_STRATEGIES, prices, universe, tx=0.004)

    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # ── Print Results ─────────────────────────────────────────────
    print_table("Part 1: 리스크 관리 (R) 비교 — K5+H5+EW 기준", R_STRATEGIES, r0_R, r4_R)
    print_yearly("Part 1: R 전략 연도별 (tx=0.4%)", R_STRATEGIES, r4_R)

    print_table("Part 2: 코인 선택 (S) 비교 — K5+H5+EW 기준", S_STRATEGIES, r0_S, r4_S)
    print_yearly("Part 2: S 전략 연도별 (tx=0.4%)", S_STRATEGIES, r4_S)

    print_table("Part 3: 최종 조합 비교", COMBO_STRATEGIES, r0_C, r4_C)
    print_yearly("Part 3: 조합 연도별 (tx=0.4%)", COMBO_STRATEGIES, r4_C)

    # ── Rebalancing count comparison ─────────────────────────────
    print(f"\n{'=' * 130}")
    print(f"  리밸런싱 횟수 비교 (tx=0.4%)")
    print(f"{'=' * 130}")
    all_strats = list(R_STRATEGIES) + list(S_STRATEGIES[1:]) + list(COMBO_STRATEGIES[1:])
    all_results = list(r4_R) + list(r4_S[1:]) + list(r4_C[1:])
    for (name, _), r in zip(all_strats, all_results):
        print(f"  {name:<28} {r['rebal_count']:>3}회")


if __name__ == '__main__':
    main()
