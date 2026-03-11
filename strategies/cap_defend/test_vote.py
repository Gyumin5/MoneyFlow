#!/usr/bin/env python3
"""Test various vote canary configurations with K8.
   - Different SMA/mom combinations for 2/3 vote
   - Expanded votes: 2/4, 3/4, 3/5
   All with H1 health filter."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())


def P(smas, moms, thr):
    """Shorthand for K8 Params."""
    return Params(canary='K8', health='H1',
                  vote_smas=tuple(smas), vote_moms=tuple(moms),
                  vote_threshold=thr)


# ══════════════════════════════════════════════════════════════════
# Part 1: 2/3 투표 — SMA 조합 변경
# ══════════════════════════════════════════════════════════════════
VOTE_2of3 = [
    # 기존 K5 sma80 (참고)
    ('K5 sma80 (기존)',       Params(canary='K5', health='H1', sma_period=80)),
    # SMA_long + SMA_short + mom 조합 (2/3)
    ('(150,50)+m21  2/3',    P([150, 50], [21], 2)),
    ('(100,50)+m21  2/3',    P([100, 50], [21], 2)),
    ('(80,50)+m21   2/3',    P([80, 50],  [21], 2)),
    ('(80,30)+m21   2/3',    P([80, 30],  [21], 2)),
    ('(100,30)+m21  2/3',    P([100, 30], [21], 2)),
    ('(80,50)+m14   2/3',    P([80, 50],  [14], 2)),
    ('(80,50)+m30   2/3',    P([80, 50],  [30], 2)),
    ('(80,50)+m60   2/3',    P([80, 50],  [60], 2)),
    ('(100,70)+m21  2/3',    P([100, 70], [21], 2)),
    ('(120,50)+m21  2/3',    P([120, 50], [21], 2)),
    ('(60,30)+m21   2/3',    P([60, 30],  [21], 2)),
]

# ══════════════════════════════════════════════════════════════════
# Part 2: 2/4 투표 — 조건 4개 중 2개
# ══════════════════════════════════════════════════════════════════
VOTE_2of4 = [
    ('K5 sma80 (기존)',              Params(canary='K5', health='H1', sma_period=80)),
    ('(100,80,50)+m21   2/4',       P([100, 80, 50],  [21], 2)),
    ('(100,80,50)+m21   3/4',       P([100, 80, 50],  [21], 3)),
    ('(120,80,50)+m21   2/4',       P([120, 80, 50],  [21], 2)),
    ('(120,80,50)+m21   3/4',       P([120, 80, 50],  [21], 3)),
    ('(100,50)+m21,m60  2/4',       P([100, 50], [21, 60], 2)),
    ('(100,50)+m21,m60  3/4',       P([100, 50], [21, 60], 3)),
    ('(80,50)+m21,m60   2/4',       P([80, 50],  [21, 60], 2)),
    ('(80,50)+m21,m60   3/4',       P([80, 50],  [21, 60], 3)),
]

# ══════════════════════════════════════════════════════════════════
# Part 3: 3/5 투표 — 조건 5개 중 3개
# ══════════════════════════════════════════════════════════════════
VOTE_3of5 = [
    ('K5 sma80 (기존)',                  Params(canary='K5', health='H1', sma_period=80)),
    ('(120,80,50)+m21,m60  3/5',        P([120, 80, 50], [21, 60], 3)),
    ('(120,80,50)+m21,m60  2/5',        P([120, 80, 50], [21, 60], 2)),
    ('(100,80,50,30)+m21   3/5',        P([100, 80, 50, 30], [21], 3)),
    ('(100,80,50,30)+m21   2/5',        P([100, 80, 50, 30], [21], 2)),
    ('(120,80,50)+m14,m30  3/5',        P([120, 80, 50], [14, 30], 3)),
    ('(100,70,50)+m21,m60  3/5',        P([100, 70, 50], [21, 60], 3)),
]


def run_set(strategies, prices, universe, tx=0.004):
    params_list = []
    for _, p in strategies:
        params_list.append(Params(
            canary=p.canary, health=p.health, selection=p.selection,
            weighting=p.weighting, rebalancing=p.rebalancing,
            risk=p.risk, tx_cost=tx, sma_period=p.sma_period,
            vote_smas=p.vote_smas, vote_moms=p.vote_moms,
            vote_threshold=p.vote_threshold,
        ))
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, params_list)
    return results


def print_table(title, strategies, results_4, results_0=None):
    print(f"\n{'=' * 120}")
    print(f"  {title}")
    print(f"{'=' * 120}")

    years = range(2018, 2026)

    if results_0:
        print(f"\n  {'전략':<28}"
              f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7}"
              f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
              f" │{'비용':>6}")
        print(f"  {'':>28}"
              f" │{'── tx=0 ──':^22}"
              f" │{'── tx=0.4% ──':^34}"
              f" │")
        print(f"  {'─' * 105}")
        for (name, _), r0, r4 in zip(strategies, results_0, results_4):
            m0, m4 = r0['metrics'], r4['metrics']
            drag = m0['CAGR'] - m4['CAGR']
            print(f"  {name:<28}"
                  f" │{m0['Sharpe']:>7.3f} {m0['CAGR']:>+6.1%} {m0['MDD']:>6.1%}"
                  f" │{m4['Sharpe']:>7.3f} {m4['CAGR']:>+6.1%} {m4['MDD']:>6.1%} {m4['Final']:>10,.0f}"
                  f" │{drag:>+5.1%}")
    else:
        print(f"\n  {'전략':<28} │{'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Final':>10}")
        print(f"  {'─' * 70}")
        for (name, _), r4 in zip(strategies, results_4):
            m4 = r4['metrics']
            print(f"  {name:<28} │{m4['Sharpe']:>7.3f} {m4['CAGR']:>+7.1%} {m4['MDD']:>6.1%} {m4['Final']:>10,.0f}")

    # Year-by-year CAGR
    print(f"\n  연도별 CAGR (tx=0.4%)")
    print(f"  {'전략':<28}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print(f" {'전체':>8}")
    print(f"  {'─' * 100}")
    for (name, _), r in zip(strategies, results_4):
        ym = r['yearly']
        m = r['metrics']
        row = f"  {name:<28}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+6.1%}"
            else:
                row += f" {'─':>7}"
        row += f" {m['CAGR']:>+7.1%}"
        print(row)


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()

    # Part 1
    r1_0 = run_set(VOTE_2of3, prices, universe, tx=0.0)
    r1_4 = run_set(VOTE_2of3, prices, universe, tx=0.004)

    # Part 2
    r2_4 = run_set(VOTE_2of4, prices, universe, tx=0.004)

    # Part 3
    r3_4 = run_set(VOTE_3of5, prices, universe, tx=0.004)

    print(f"\n  Completed in {time.time()-t0:.1f}s")

    print_table("Part 1: 2/3 투표 — SMA+모멘텀 조합 (H1 고정)", VOTE_2of3, r1_4, r1_0)
    print_table("Part 2: 2/4 및 3/4 투표", VOTE_2of4, r2_4)
    print_table("Part 3: 3/5 및 2/5 투표", VOTE_3of5, r3_4)

    # ── Top 10 across all tests ──────────────────────────────────
    print(f"\n{'=' * 120}")
    print(f"  전체 Top 10 (tx=0.4% Sharpe 기준)")
    print(f"{'=' * 120}")

    all_results = []
    for strats, results in [(VOTE_2of3, r1_4), (VOTE_2of4, r2_4), (VOTE_3of5, r3_4)]:
        for (name, _), r in zip(strats, results):
            all_results.append((name, r['metrics']))

    # Deduplicate by name (K5 sma80 appears multiple times)
    seen = set()
    unique = []
    for name, m in all_results:
        if name not in seen:
            seen.add(name)
            unique.append((name, m))

    unique.sort(key=lambda x: x[1]['Sharpe'], reverse=True)
    print(f"\n  {'순위':>3} {'전략':<30} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7}")
    print(f"  {'─' * 60}")
    for i, (name, m) in enumerate(unique[:10], 1):
        print(f"  {i:>3}. {name:<30} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%}")


if __name__ == '__main__':
    main()
