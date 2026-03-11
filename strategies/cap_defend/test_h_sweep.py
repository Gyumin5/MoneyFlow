#!/usr/bin/env python3
"""Comprehensive H (Health) filter sweep.
   Like K single-condition test, systematically test:
   1. Single conditions: SMA only, Mom only, Vol only
   2. Two-condition combos
   3. Three-condition combos
   All with K=SMA(10)>SMA(65) canary (top K, broad plateau)."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())


def K(**kw):
    """Top K canary with variable health."""
    return Params(canary='K9', cross_short=10, cross_long=65, **kw)


# ══════════════════════════════════════════════════════════════════
# Part 1: No health filter (baseline)
# ══════════════════════════════════════════════════════════════════
P0 = [
    ('none (필터 없음)', K(health='none')),
]

# ══════════════════════════════════════════════════════════════════
# Part 2: Vol cap만 (단일 조건)
# ══════════════════════════════════════════════════════════════════
P_VOL = [
    (f'vol≤{v}%', K(health='HL', health_mom_long=0, vol_cap=v/100,
                     health_vol_window=90))
    for v in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0]
]
# Note: HL needs mom_long>0, so I need HQ for vol-only.
# Actually let me use HQ without SMA by setting health_sma=1 (always passes)
# Or better, just add a vol-only health
# Let me use 'none' health with manual vol filter... no, that doesn't work.
# Let me just test with HQ (SMA+vol) using very short SMA (SMA1 = always passes)
P_VOL = []
for v in [3.0, 3.5, 4.0, 4.5, 5.0, 5.5, 6.0, 7.0, 8.0, 10.0]:
    P_VOL.append((f'vol90≤{v}%', K(health='HQ', health_sma=1, vol_cap=v/100)))

# ══════════════════════════════════════════════════════════════════
# Part 3: SMA 단일 + vol5% (다양한 SMA 기간)
# ══════════════════════════════════════════════════════════════════
P_SMA = []
for sma in [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]:
    P_SMA.append((f'SMA({sma})+v5%', K(health='HQ', health_sma=sma, vol_cap=0.05)))

# ══════════════════════════════════════════════════════════════════
# Part 4: Mom 단일 + vol5% (다양한 Mom 기간)
# ══════════════════════════════════════════════════════════════════
P_MOM = []
for mom in [7, 14, 21, 30, 45, 60, 90, 120, 150]:
    P_MOM.append((f'Mom({mom})+v5%', K(health='HL', health_mom_long=mom, vol_cap=0.05)))

# ══════════════════════════════════════════════════════════════════
# Part 5: SMA + Mom (2조건, vol5% 고정)
# ══════════════════════════════════════════════════════════════════
P_SMA_MOM = []
for sma in [20, 30, 40, 50]:
    for mom in [30, 60, 90, 120]:
        P_SMA_MOM.append((f's{sma}+m{mom}+v5%',
                          K(health='HM', health_sma=sma, health_mom_long=mom, vol_cap=0.05)))

# ══════════════════════════════════════════════════════════════════
# Part 6: SMA crossover in Health (HN)
# ══════════════════════════════════════════════════════════════════
P_CROSS = []
for short_sma in [5, 10, 15, 20]:
    for long_sma in [30, 40, 50, 60]:
        if short_sma >= long_sma:
            continue
        P_CROSS.append((f'h:{short_sma}>{long_sma}+v5%',
                        K(health='HN', health_mom_short=short_sma, health_sma=long_sma, vol_cap=0.05)))

# ══════════════════════════════════════════════════════════════════
# Part 7: Vol acceleration (HP variants)
# ══════════════════════════════════════════════════════════════════
P_VACCEL = []
for mom in [60, 90, 120]:
    for vol in [0.05, 0.055, 0.06]:
        P_VACCEL.append((f'HP:m{mom}+v{vol*100:.1f}%',
                         K(health='HP', health_mom_long=mom, vol_cap=vol)))

# ══════════════════════════════════════════════════════════════════
# Part 8: Full H1 vs variations
# ══════════════════════════════════════════════════════════════════
P_REF = [
    ('H1 (현재)',       K(health='H1', vol_cap=0.05)),
    ('H1 v4.5%',       K(health='H1', vol_cap=0.045)),
    ('H1 v5.5%',       K(health='H1', vol_cap=0.055)),
    ('H5 (vol accel)',  K(health='H5', vol_cap=0.05)),
    ('baseline',        K(health='baseline', vol_cap=0.05)),
]

ALL = P0 + P_VOL + P_SMA + P_MOM + P_SMA_MOM + P_CROSS + P_VACCEL + P_REF


def run_set(strategies, prices, universe, tx=0.004):
    params_list = []
    for _, p in strategies:
        params_list.append(Params(
            canary=p.canary, health=p.health, tx_cost=tx,
            sma_period=p.sma_period,
            vote_smas=p.vote_smas, vote_moms=p.vote_moms,
            vote_threshold=p.vote_threshold,
            cross_short=p.cross_short, cross_long=p.cross_long,
            vol_cap=p.vol_cap,
            health_sma=p.health_sma, health_mom_short=p.health_mom_short,
            health_mom_long=p.health_mom_long, health_vol_window=p.health_vol_window,
        ))
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, params_list)
    return results


def print_section(title, strategies, results):
    print(f"\n  {title}")
    print(f"  {'─' * 75}")
    print(f"  {'조건':<22} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Final':>10} {'Calmar':>7}")
    print(f"  {'─' * 75}")
    best_s = max(r['metrics']['Sharpe'] for r in results)
    for (name, _), r in zip(strategies, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        marker = " ★" if m['Sharpe'] == best_s else ""
        print(f"  {name:<22} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {m['Final']:>10,.0f} {calmar:>7.2f}{marker}")


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")
    print(f"  {len(ALL)} H configurations to test")

    t0 = time.time()
    results = run_set(ALL, prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # Split results
    idx = 0
    def get_slice(strategies):
        nonlocal idx
        r = results[idx:idx+len(strategies)]
        idx += len(strategies)
        return r

    r0 = get_slice(P0)
    r_vol = get_slice(P_VOL)
    r_sma = get_slice(P_SMA)
    r_mom = get_slice(P_MOM)
    r_sm = get_slice(P_SMA_MOM)
    r_cross = get_slice(P_CROSS)
    r_vaccel = get_slice(P_VACCEL)
    r_ref = get_slice(P_REF)

    print(f"\n{'=' * 100}")
    print(f"  H (Health) 체계적 탐색 — K=SMA(10)>SMA(65) 고정")
    print(f"{'=' * 100}")

    print_section("0. 필터 없음 (기준점)", P0, r0)
    print_section("1. Vol Cap만 (다양한 임계치)", P_VOL, r_vol)
    print_section("2. SMA 단일 + Vol5%", P_SMA, r_sma)
    print_section("3. Momentum 단일 + Vol5%", P_MOM, r_mom)
    print_section("4. SMA + Mom + Vol5%", P_SMA_MOM, r_sm)
    print_section("5. SMA Crossover in Health + Vol5%", P_CROSS, r_cross)
    print_section("6. Vol Acceleration (HP)", P_VACCEL, r_vaccel)
    print_section("7. 기존 H 변형들", P_REF, r_ref)

    # ── SMA plateau ─────────────────────────────────────────────
    print(f"\n  SMA 단일 플래토:")
    sma_vals = [10, 15, 20, 25, 30, 40, 50, 60, 80, 100]
    for (name, _), r in zip(P_SMA, r_sma):
        s = r['metrics']['Sharpe']
        bar = '█' * int(s * 20)
        print(f"    {name:<20}: {s:.3f} {bar}")
    sma_sharpes = [r['metrics']['Sharpe'] for r in r_sma]
    spread = max(sma_sharpes) - min(sma_sharpes)
    print(f"    spread: {spread:.3f} {'넓은 고원 ✓' if spread < 0.15 else '다양한 편차'}")

    # ── Mom plateau ─────────────────────────────────────────────
    print(f"\n  Mom 단일 플래토:")
    for (name, _), r in zip(P_MOM, r_mom):
        s = r['metrics']['Sharpe']
        bar = '█' * int(s * 20)
        print(f"    {name:<20}: {s:.3f} {bar}")
    mom_sharpes = [r['metrics']['Sharpe'] for r in r_mom]
    spread_m = max(mom_sharpes) - min(mom_sharpes)
    print(f"    spread: {spread_m:.3f}")

    # ── Vol plateau ─────────────────────────────────────────────
    print(f"\n  Vol Cap 플래토:")
    for (name, _), r in zip(P_VOL, r_vol):
        s = r['metrics']['Sharpe']
        bar = '█' * int(s * 20)
        print(f"    {name:<20}: {s:.3f} {bar}")

    # ── SMA+Mom grid ────────────────────────────────────────────
    print(f"\n  SMA × Mom Grid (Sharpe, vol=5%):")
    mom_vals = [30, 60, 90, 120]
    sma_vals_g = [20, 30, 40, 50]
    print(f"  {'':>12}", end="")
    for m in mom_vals:
        print(f" {'Mom'+str(m):>10}", end="")
    print()
    print(f"  {'─' * 55}")
    grid_idx = 0
    for s in sma_vals_g:
        print(f"  {'SMA'+str(s):<12}", end="")
        for m in mom_vals:
            sh = r_sm[grid_idx]['metrics']['Sharpe']
            print(f" {sh:>10.3f}", end="")
            grid_idx += 1
        print()

    # ── Overall Top 10 ──────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  전체 H Top 10 (Sharpe 기준)")
    print(f"{'=' * 100}")

    all_scored = []
    for (name, _), r in zip(ALL, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        all_scored.append((name, m, r, calmar))

    all_scored.sort(key=lambda x: x[1]['Sharpe'], reverse=True)
    print(f"\n  {'순위':>3} {'H 조건':<22} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 60}")
    for i, (name, m, r, calmar) in enumerate(all_scored[:10], 1):
        print(f"  {i:>3}. {name:<22} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}")

    # ── Top 10 by Calmar ────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  전체 H Top 10 (Calmar 기준)")
    print(f"{'=' * 100}")
    all_scored_c = sorted(all_scored, key=lambda x: x[3], reverse=True)
    print(f"\n  {'순위':>3} {'H 조건':<22} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 60}")
    for i, (name, m, r, calmar) in enumerate(all_scored_c[:10], 1):
        print(f"  {i:>3}. {name:<22} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}")

    # Year-by-year for top 5
    years = range(2018, 2026)
    print(f"\n  Top 5 연도별 CAGR")
    print(f"  {'H 조건':<22}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print(f" {'전체':>8}")
    print(f"  {'─' * 90}")
    for name, m, r, calmar in all_scored[:5]:
        ym = r['yearly']
        row = f"  {name:<22}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+6.1%}"
            else:
                row += f" {'─':>7}"
        row += f" {m['CAGR']:>+7.1%}"
        print(row)

    print(f"\n  Completed all in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
