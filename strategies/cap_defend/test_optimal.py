#!/usr/bin/env python3
"""1. Find optimal K5 SMA period (fine-grained search)
   2. Test H1∪H5 union (HU: filter union, HX: pick union)
   3. Best combos with findings"""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

# ══════════════════════════════════════════════════════════════════
# 1. K5 SMA optimal search (fine grid)
# ══════════════════════════════════════════════════════════════════
K5_GRID = [(f'K5 sma={p}', Params(canary='K5', health='H5', sma_period=p))
           for p in [30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 180, 200]]

# Also test baseline canary with same SMA grid (to compare K5 vs simple)
BASE_GRID = [(f'base sma={p}', Params(health='H5', sma_period=p))
             for p in [50, 70, 100, 120, 150, 200]]

# ══════════════════════════════════════════════════════════════════
# 2. H1∪H5 union tests
# ══════════════════════════════════════════════════════════════════
H_UNION = [
    ('H1 only',       Params(canary='K5', health='H1')),
    ('H5 only',       Params(canary='K5', health='H5')),
    ('HU (필터합집합)', Params(canary='K5', health='HU')),   # pass if H1 OR H5, top 5
    ('HX (선택합집합)', Params(canary='K5', health='HX')),   # H1 top5 ∪ H5 top5 (5~10개)
    ('baseline H',    Params(canary='K5')),                 # SMA30+Mom21+Vol5%
]

# ══════════════════════════════════════════════════════════════════
# 3. Best combos: use findings from above
# ══════════════════════════════════════════════════════════════════
# Will be filled after running parts 1 & 2


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


def print_grid(title, strategies, results):
    """Compact grid view for parameter search."""
    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    print(f"\n  {'전략':<18} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Final':>10}  연도별 Sharpe")
    print(f"  {'─' * 95}")

    years = range(2018, 2026)
    for (name, _), r in zip(strategies, results):
        m = r['metrics']
        ym = r['yearly']
        yr_str = ""
        for y in years:
            if y in ym:
                yr_str += f" {ym[y]['Sharpe']:>6.2f}"
            else:
                yr_str += f" {'─':>6}"
        print(f"  {name:<18} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {m['Final']:>10,.0f} {yr_str}")


def print_section(title, strategies, results):
    print(f"\n{'=' * 120}")
    print(f"  {title}")
    print(f"{'=' * 120}")

    years = range(2018, 2026)
    print(f"\n  {'전략':<20} │{'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Final':>10} │ 연도별 CAGR")
    print(f"  {'─' * 115}")
    for (name, _), r in zip(strategies, results):
        m = r['metrics']
        ym = r['yearly']
        yr_str = ""
        for y in years:
            if y in ym:
                yr_str += f" {ym[y]['CAGR']:>+7.1%}"
            else:
                yr_str += f" {'─':>7}"
        print(f"  {name:<20} │{m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {m['Final']:>10,.0f} │{yr_str}")

    # Sharpe
    print(f"\n  {'전략':<20} │ 연도별 Sharpe")
    print(f"  {'─' * 115}")
    for (name, _), r in zip(strategies, results):
        m = r['metrics']
        ym = r['yearly']
        yr_str = ""
        for y in years:
            if y in ym:
                yr_str += f" {ym[y]['Sharpe']:>7.3f}"
            else:
                yr_str += f" {'─':>7}"
        yr_str += f"  전체:{m['Sharpe']:>6.3f}"
        print(f"  {name:<20} │{yr_str}")


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()
    rK5 = run_set(K5_GRID, prices, universe)
    rBase = run_set(BASE_GRID, prices, universe)
    rH = run_set(H_UNION, prices, universe)

    # Part 3: Final combos based on findings
    # Find best K5 SMA
    best_sma_idx = max(range(len(rK5)), key=lambda i: rK5[i]['metrics']['Sharpe'])
    best_sma = K5_GRID[best_sma_idx][1].sma_period
    print(f"\n  Best K5 SMA = {best_sma}")

    FINAL = [
        ('K5+H5 (sma150)',     Params(canary='K5', health='H5', sma_period=150)),
        (f'K5+H5 (sma{best_sma})',  Params(canary='K5', health='H5', sma_period=best_sma)),
        ('K5+H1 (sma150)',     Params(canary='K5', health='H1', sma_period=150)),
        (f'K5+H1 (sma{best_sma})',  Params(canary='K5', health='H1', sma_period=best_sma)),
        ('K5+HX (sma150)',     Params(canary='K5', health='HX', sma_period=150)),
        (f'K5+HX (sma{best_sma})',  Params(canary='K5', health='HX', sma_period=best_sma)),
        ('K5+HU (sma150)',     Params(canary='K5', health='HU', sma_period=150)),
        (f'K5+HU (sma{best_sma})',  Params(canary='K5', health='HU', sma_period=best_sma)),
    ]
    # Also test tx=0 for final
    rF4 = run_set(FINAL, prices, universe, tx=0.004)
    rF0 = run_set(FINAL, prices, universe, tx=0.0)

    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # ── Print ─────────────────────────────────────────────────────
    print_grid("1a. K5 SMA 최적 기간 탐색 (K5+H5, tx=0.4%)", K5_GRID, rK5)
    print_grid("1b. Baseline SMA 비교 (H5 고정, tx=0.4%)", BASE_GRID, rBase)

    print_section("2. H1∪H5 합집합 테스트 (K5 sma=150, tx=0.4%)", H_UNION, rH)

    # Final comparison
    print(f"\n{'=' * 130}")
    print(f"  3. 최종 조합 비교 (tx=0 vs tx=0.4%)")
    print(f"{'=' * 130}")
    years = range(2018, 2026)
    print(f"\n  {'전략':<22}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7}"
          f" │{'ΔCAGR':>7}")
    print(f"  {'':>22}"
          f" │{'─── tx=0 ───':^22}"
          f" │{'─── tx=0.4% ───':^22}"
          f" │{'비용':>7}")
    print(f"  {'─' * 80}")
    for (name, _), r0, r4 in zip(FINAL, rF0, rF4):
        m0, m4 = r0['metrics'], r4['metrics']
        drag = m0['CAGR'] - m4['CAGR']
        print(f"  {name:<22}"
              f" │{m0['Sharpe']:>7.3f} {m0['CAGR']:>+6.1%} {m0['MDD']:>6.1%}"
              f" │{m4['Sharpe']:>7.3f} {m4['CAGR']:>+6.1%} {m4['MDD']:>6.1%}"
              f" │{drag:>+6.1%}")

    # Year-by-year for final
    print(f"\n  연도별 CAGR (tx=0.4%)")
    print(f"  {'─' * 110}")
    print(f"  {'전략':<22}", end="")
    for y in years:
        print(f" {y:>8}", end="")
    print(f" {'전체':>9}")
    print(f"  {'─' * 95}")
    for (name, _), r in zip(FINAL, rF4):
        ym = r['yearly']
        m = r['metrics']
        row = f"  {name:<22}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+7.1%}"
            else:
                row += f" {'─':>8}"
        row += f" {m['CAGR']:>+8.1%}"
        print(row)


if __name__ == '__main__':
    main()
