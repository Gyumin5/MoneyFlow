#!/usr/bin/env python3
"""Test SMA and Momentum as SINGLE canary conditions across many values.
   Goal: find which individual lookback periods are inherently strong."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())


def P_sma(period):
    """Single SMA condition: BTC > SMA(N)."""
    return Params(canary='K8', health='H1',
                  vote_smas=(period,), vote_moms=(), vote_threshold=1)


def P_mom(period):
    """Single momentum condition: mom(N) > 0."""
    return Params(canary='K8', health='H1',
                  vote_smas=(), vote_moms=(period,), vote_threshold=1)


# ── SMA only: wide range ────────────────────────────────────────────
SMA_VALUES = [10, 15, 20, 25, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 150, 200]
SMA_TESTS = [(f'SMA({v})', P_sma(v)) for v in SMA_VALUES]

# ── Momentum only: wide range ───────────────────────────────────────
MOM_VALUES = [3, 5, 7, 10, 14, 21, 30, 45, 60, 90, 120, 150, 200]
MOM_TESTS = [(f'Mom({v})', P_mom(v)) for v in MOM_VALUES]

# ── Reference: K5 sma80 (baseline) ──────────────────────────────────
REF = [('K5 sma80 (기준)', Params(canary='K5', health='H1', sma_period=80))]


def run_set(strategies, prices, universe, tx=0.004):
    params_list = []
    for _, p in strategies:
        params_list.append(Params(
            canary=p.canary, health=p.health, tx_cost=tx,
            sma_period=p.sma_period,
            vote_smas=p.vote_smas, vote_moms=p.vote_moms,
            vote_threshold=p.vote_threshold,
        ))
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, params_list)
    return results


def print_table(title, strategies, results):
    years = range(2018, 2026)
    print(f"\n{'=' * 130}")
    print(f"  {title}")
    print(f"{'=' * 130}")

    # Main metrics
    print(f"\n  {'조건':<15} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Final':>10}"
          f"  │ {'Calmar':>7} {'Win%':>5} {'리밸':>4}  비고")
    print(f"  {'─' * 80}")

    best_sharpe = max(r['metrics']['Sharpe'] for r in results)
    for (name, _), r in zip(strategies, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        marker = ""
        if m['Sharpe'] == best_sharpe:
            marker += " ★"
        print(f"  {name:<15} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {m['Final']:>10,.0f}"
              f"  │ {calmar:>7.2f} {m.get('Win%', 0):>4.0%} {r.get('rebal_count', 0):>4}{marker}")

    # Year-by-year
    print(f"\n  연도별 CAGR")
    print(f"  {'조건':<15}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print(f" {'전체':>8}")
    print(f"  {'─' * 85}")
    for (name, _), r in zip(strategies, results):
        ym = r['yearly']
        m = r['metrics']
        row = f"  {name:<15}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+6.1%}"
            else:
                row += f" {'─':>7}"
        row += f" {m['CAGR']:>+7.1%}"
        print(row)


def find_plateau(label, values, results):
    """Find the robust plateau region."""
    sharpes = [(v, r['metrics']['Sharpe']) for v, r in zip(values, results)]
    best_val, best_s = max(sharpes, key=lambda x: x[1])

    # Find range where Sharpe is within 0.05 of best
    threshold = best_s - 0.05
    plateau = [v for v, s in sharpes if s >= threshold]

    # Find range where Sharpe is within 0.10 of best
    threshold2 = best_s - 0.10
    wide_plateau = [v for v, s in sharpes if s >= threshold2]

    print(f"\n  {label} 고원 분석:")
    print(f"    최고: {best_val} (Sharpe {best_s:.3f})")
    if plateau:
        print(f"    ±0.05 고원: {min(plateau)}~{max(plateau)} ({len(plateau)}개)")
    if wide_plateau:
        print(f"    ±0.10 고원: {min(wide_plateau)}~{max(wide_plateau)} ({len(wide_plateau)}개)")

    # Check if peak is sharp or broad
    spread = max(s for _, s in sharpes) - min(s for _, s in sharpes)
    if len(plateau) >= 3:
        print(f"    판정: 넓은 고원 ✓ (spread={spread:.3f})")
    else:
        print(f"    판정: 좁은 봉우리 ✗ (spread={spread:.3f})")


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()
    r_ref = run_set(REF, prices, universe)
    r_sma = run_set(SMA_TESTS, prices, universe)
    r_mom = run_set(MOM_TESTS, prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # Reference
    print_table("기준: K5 sma80 + H1", REF, r_ref)

    # SMA results
    print_table("SMA 단일 조건 — BTC > SMA(N)", SMA_TESTS, r_sma)
    find_plateau("SMA", SMA_VALUES, r_sma)

    # Mom results
    print_table("Momentum 단일 조건 — mom(N) > 0", MOM_TESTS, r_mom)
    find_plateau("Momentum", MOM_VALUES, r_mom)

    # ── Cross comparison: SMA vs Mom at equivalent lookbacks ────────
    print(f"\n{'=' * 130}")
    print(f"  SMA vs Momentum 등가 비교 (SMA(N) ≈ Mom(N/2) 유효 룩백)")
    print(f"{'=' * 130}")
    print(f"\n  {'유효 룩백':>10} {'SMA 조건':>12} {'Sharpe':>7} {'CAGR':>8}"
          f"  │ {'Mom 조건':>12} {'Sharpe':>7} {'CAGR':>8}  │ {'차이':>7}")
    print(f"  {'─' * 90}")

    # SMA(N) effective lookback ≈ N/2
    # Pair: SMA(40)≈Mom(21), SMA(60)≈Mom(30), SMA(80)≈Mom(40), SMA(100)≈Mom(50)
    pairs = [
        (20, 10), (30, 14), (40, 21), (50, 25),
        (60, 30), (80, 45), (100, 60), (120, 60),
        (150, 90), (200, 120),
    ]

    sma_map = {v: r for v, r in zip(SMA_VALUES, r_sma)}
    mom_map = {v: r for v, r in zip(MOM_VALUES, r_mom)}

    for sma_n, mom_n in pairs:
        if sma_n not in sma_map or mom_n not in mom_map:
            continue
        eff = sma_n // 2
        ms = sma_map[sma_n]['metrics']
        mm = mom_map[mom_n]['metrics']
        diff = ms['Sharpe'] - mm['Sharpe']
        print(f"  {eff:>7}d   SMA({sma_n:>3})    {ms['Sharpe']:>7.3f} {ms['CAGR']:>+7.1%}"
              f"  │ Mom({mom_n:>3})     {mm['Sharpe']:>7.3f} {mm['CAGR']:>+7.1%}"
              f"  │ {diff:>+6.3f}")

    # ── Summary ─────────────────────────────────────────────────────
    print(f"\n{'=' * 130}")
    print(f"  종합 Top 10 (SMA + Mom 통합)")
    print(f"{'=' * 130}")

    all_results = []
    for (name, _), r in zip(SMA_TESTS, r_sma):
        all_results.append((name, r['metrics']))
    for (name, _), r in zip(MOM_TESTS, r_mom):
        all_results.append((name, r['metrics']))
    all_results.append(('K5 sma80', r_ref[0]['metrics']))

    all_results.sort(key=lambda x: x[1]['Sharpe'], reverse=True)
    print(f"\n  {'순위':>3} {'조건':<15} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7}")
    print(f"  {'─' * 45}")
    for i, (name, m) in enumerate(all_results[:10], 1):
        print(f"  {i:>3}. {name:<15} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%}")


if __name__ == '__main__':
    main()
