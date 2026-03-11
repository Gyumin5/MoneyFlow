#!/usr/bin/env python3
"""Test AI-recommended Health filter strategies.
   Gemini + Codex consensus:
   - mom90 is the key driver, SMA30 and mom21 are redundant
   - Focus on volatility control + medium-term momentum
   - Simplify to 2-3 conditions max
   Also tests K robustness (neighborhood of top K strategies)."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

# ══════════════════════════════════════════════════════════════════
# Part 1: H candidates (K=SMA(60) 단일 고정)
# ══════════════════════════════════════════════════════════════════

def K_sma60(**kw):
    """SMA(60) single canary with variable health."""
    return Params(canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1, **kw)

H_STRATEGIES = [
    # ── Reference ────────────────────────────────────────────────
    ('H1 (현재)',          K_sma60(health='H1')),

    # ── HL: minimal (mom_long + vol only) ────────────────────────
    ('HL mom90+vol5.0',   K_sma60(health='HL', health_mom_long=90, vol_cap=0.05)),
    ('HL mom90+vol5.5',   K_sma60(health='HL', health_mom_long=90, vol_cap=0.055)),
    ('HL mom90+vol6.0',   K_sma60(health='HL', health_mom_long=90, vol_cap=0.06)),
    ('HL mom60+vol5.0',   K_sma60(health='HL', health_mom_long=60, vol_cap=0.05)),
    ('HL mom60+vol5.5',   K_sma60(health='HL', health_mom_long=60, vol_cap=0.055)),
    ('HL mom120+vol5.5',  K_sma60(health='HL', health_mom_long=120, vol_cap=0.055)),

    # ── HM: one-fast (SMA + mom_long + vol) ─────────────────────
    ('HM s30+m90+v5.0',  K_sma60(health='HM', health_sma=30, health_mom_long=90, vol_cap=0.05)),
    ('HM s40+m90+v5.5',  K_sma60(health='HM', health_sma=40, health_mom_long=90, vol_cap=0.055)),
    ('HM s50+m90+v5.5',  K_sma60(health='HM', health_sma=50, health_mom_long=90, vol_cap=0.055)),
    ('HM s30+m60+v5.0',  K_sma60(health='HM', health_sma=30, health_mom_long=60, vol_cap=0.05)),
    ('HM s40+m60+v5.5',  K_sma60(health='HM', health_sma=40, health_mom_long=60, vol_cap=0.055)),
    ('HM s50+m90+v6.0',  K_sma60(health='HM', health_sma=50, health_mom_long=90, vol_cap=0.06)),

    # ── HN: smooth crossover (short SMA > long SMA + vol) ───────
    ('HN s20>s50+v5.0',  K_sma60(health='HN', health_mom_short=20, health_sma=50, vol_cap=0.05)),
    ('HN s10>s40+v5.0',  K_sma60(health='HN', health_mom_short=10, health_sma=40, vol_cap=0.05)),
    ('HN s20>s60+v5.5',  K_sma60(health='HN', health_mom_short=20, health_sma=60, vol_cap=0.055)),

    # ── HO: risk-adjusted (mom/vol ratio > 0.5) ─────────────────
    ('HO m60/v60>0.5',   K_sma60(health='HO', health_mom_long=60, health_vol_window=60, vol_cap=0.06)),
    ('HO m90/v90>0.5',   K_sma60(health='HO', health_mom_long=90, health_vol_window=90, vol_cap=0.06)),

    # ── HP: stable winner (mom + vol + vol accel) ───────────────
    ('HP m90+v5.5+accel', K_sma60(health='HP', health_mom_long=90, vol_cap=0.055)),
    ('HP m90+v6.0+accel', K_sma60(health='HP', health_mom_long=90, vol_cap=0.06)),
    ('HP m60+v5.5+accel', K_sma60(health='HP', health_mom_long=60, vol_cap=0.055)),

    # ── HQ: SMA only + vol (simplest) ───────────────────────────
    ('HQ s50+v5.5',      K_sma60(health='HQ', health_sma=50, vol_cap=0.055)),
    ('HQ s50+v6.0',      K_sma60(health='HQ', health_sma=50, vol_cap=0.06)),
    ('HQ s30+v5.0',      K_sma60(health='HQ', health_sma=30, vol_cap=0.05)),

    # ── H5 (vol acceleration) for reference ─────────────────────
    ('H5 (vol accel)',    K_sma60(health='H5')),

    # ── Vol cap sweep with H1 ───────────────────────────────────
    ('H1 vol4.5%',        K_sma60(health='H1', vol_cap=0.045)),
    ('H1 vol5.5%',        K_sma60(health='H1', vol_cap=0.055)),
    ('H1 vol6.0%',        K_sma60(health='H1', vol_cap=0.06)),
]

# ══════════════════════════════════════════════════════════════════
# Part 2: K robustness — neighborhood of top K strategies
# ══════════════════════════════════════════════════════════════════

K_ROBUST = [
    # SMA single neighborhood (around 60)
    ('K: SMA(50)',     Params(canary='K8', health='H1', vote_smas=(50,), vote_moms=(), vote_threshold=1)),
    ('K: SMA(55)',     Params(canary='K8', health='H1', vote_smas=(55,), vote_moms=(), vote_threshold=1)),
    ('K: SMA(60)',     Params(canary='K8', health='H1', vote_smas=(60,), vote_moms=(), vote_threshold=1)),
    ('K: SMA(65)',     Params(canary='K8', health='H1', vote_smas=(65,), vote_moms=(), vote_threshold=1)),
    ('K: SMA(70)',     Params(canary='K8', health='H1', vote_smas=(70,), vote_moms=(), vote_threshold=1)),
    ('K: SMA(75)',     Params(canary='K8', health='H1', vote_smas=(75,), vote_moms=(), vote_threshold=1)),
    ('K: SMA(80)',     Params(canary='K8', health='H1', vote_smas=(80,), vote_moms=(), vote_threshold=1)),

    # Crossover neighborhood (SMA(10)>SMA(X) and SMA(20)>SMA(X))
    ('K: 10>50',       Params(canary='K9', health='H1', cross_short=10, cross_long=50)),
    ('K: 10>55',       Params(canary='K9', health='H1', cross_short=10, cross_long=55)),
    ('K: 10>60',       Params(canary='K9', health='H1', cross_short=10, cross_long=60)),
    ('K: 10>65',       Params(canary='K9', health='H1', cross_short=10, cross_long=65)),
    ('K: 10>70',       Params(canary='K9', health='H1', cross_short=10, cross_long=70)),
    ('K: 10>80',       Params(canary='K9', health='H1', cross_short=10, cross_long=80)),
    ('K: 20>50',       Params(canary='K9', health='H1', cross_short=20, cross_long=50)),
    ('K: 20>55',       Params(canary='K9', health='H1', cross_short=20, cross_long=55)),
    ('K: 20>60',       Params(canary='K9', health='H1', cross_short=20, cross_long=60)),
    ('K: 20>65',       Params(canary='K9', health='H1', cross_short=20, cross_long=65)),
    ('K: 20>70',       Params(canary='K9', health='H1', cross_short=20, cross_long=70)),
    ('K: 20>80',       Params(canary='K9', health='H1', cross_short=20, cross_long=80)),

    # Hybrid vote neighborhood
    ('K: 2/3 50,100+M120',  Params(canary='K8', health='H1',
                                    vote_smas=(50, 100), vote_moms=(120,), vote_threshold=2)),
    ('K: 2/3 60,100+M120',  Params(canary='K8', health='H1',
                                    vote_smas=(60, 100), vote_moms=(120,), vote_threshold=2)),
    ('K: 2/3 60,110+M120',  Params(canary='K8', health='H1',
                                    vote_smas=(60, 110), vote_moms=(120,), vote_threshold=2)),
    ('K: 2/3 60,120+M120',  Params(canary='K8', health='H1',
                                    vote_smas=(60, 120), vote_moms=(120,), vote_threshold=2)),
    ('K: 2/3 70,110+M120',  Params(canary='K8', health='H1',
                                    vote_smas=(70, 110), vote_moms=(120,), vote_threshold=2)),
    ('K: 2/3 60,110+M90',   Params(canary='K8', health='H1',
                                    vote_smas=(60, 110), vote_moms=(90,), vote_threshold=2)),
    ('K: 2/3 60,110+M150',  Params(canary='K8', health='H1',
                                    vote_smas=(60, 110), vote_moms=(150,), vote_threshold=2)),
]


def run_set(strategies, prices, universe, tx=0.004):
    params_list = []
    for _, p in strategies:
        params_list.append(Params(
            canary=p.canary, health=p.health, tx_cost=tx,
            sma_period=p.sma_period,
            vote_smas=p.vote_smas, vote_moms=p.vote_moms,
            vote_threshold=p.vote_threshold,
            cross_short=p.cross_short, cross_long=p.cross_long,
            persist_period=p.persist_period, persist_months=p.persist_months,
            vol_cap=p.vol_cap,
            health_sma=p.health_sma, health_mom_short=p.health_mom_short,
            health_mom_long=p.health_mom_long, health_vol_window=p.health_vol_window,
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

    print(f"\n  {'전략':<22} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Final':>10}"
          f"  │ {'Calmar':>7} {'리밸':>4}")
    print(f"  {'─' * 75}")

    best_sharpe = max(r['metrics']['Sharpe'] for r in results)
    for (name, _), r in zip(strategies, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        marker = " ★" if m['Sharpe'] == best_sharpe else ""
        print(f"  {name:<22} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {m['Final']:>10,.0f}"
              f"  │ {calmar:>7.2f} {r.get('rebal_count', 0):>4}{marker}")

    # Year-by-year
    print(f"\n  연도별 CAGR")
    print(f"  {'전략':<22}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print(f" {'전체':>8}")
    print(f"  {'─' * 90}")
    for (name, _), r in zip(strategies, results):
        ym = r['yearly']
        m = r['metrics']
        row = f"  {name:<22}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+6.1%}"
            else:
                row += f" {'─':>7}"
        row += f" {m['CAGR']:>+7.1%}"
        print(row)


def find_top5(title, strategies, results):
    scored = []
    for (name, _), r in zip(strategies, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        scored.append((name, m, r, calmar))
    scored.sort(key=lambda x: x[1]['Sharpe'], reverse=True)

    print(f"\n{'=' * 100}")
    print(f"  {title}")
    print(f"{'=' * 100}")
    print(f"\n  {'순위':>3} {'전략':<22} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 60}")
    for i, (name, m, r, calmar) in enumerate(scored[:5], 1):
        print(f"  {i:>3}. {name:<22} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}")
    return scored[:5]


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()
    rH = run_set(H_STRATEGIES, prices, universe)
    rK = run_set(K_ROBUST, prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # ── Part 1: H results ───────────────────────────────────────
    print_table("H 필터 비교 (K=SMA(60) 고정, tx=0.4%)", H_STRATEGIES, rH)
    h_top5 = find_top5("H Top 5 (Sharpe 기준)", H_STRATEGIES, rH)

    # ── Part 2: K robustness ────────────────────────────────────
    print_table("K 강건성 — 파라미터 이웃 테스트 (H1 고정)", K_ROBUST, rK)

    # Analyze SMA single plateau
    print(f"\n  SMA 단일 플래토 분석:")
    sma_vals = [50, 55, 60, 65, 70, 75, 80]
    sma_sharpes = []
    for (name, _), r in zip(K_ROBUST[:7], rK[:7]):
        sma_sharpes.append(r['metrics']['Sharpe'])
    for v, s in zip(sma_vals, sma_sharpes):
        bar = '█' * int(s * 20)
        print(f"    SMA({v:>3}): {s:.3f} {bar}")
    spread = max(sma_sharpes) - min(sma_sharpes)
    print(f"    spread: {spread:.3f} {'넓은 고원 ✓' if spread < 0.15 else '좁은 봉우리 ✗'}")

    # Analyze crossover plateau
    print(f"\n  Crossover SMA(10)>SMA(X) 플래토:")
    cross_sharpes_10 = []
    for (name, _), r in zip(K_ROBUST[7:13], rK[7:13]):
        cross_sharpes_10.append((name, r['metrics']['Sharpe']))
    for name, s in cross_sharpes_10:
        bar = '█' * int(s * 20)
        print(f"    {name:<12}: {s:.3f} {bar}")
    vals_10 = [s for _, s in cross_sharpes_10]
    spread_10 = max(vals_10) - min(vals_10)
    print(f"    spread: {spread_10:.3f} {'넓은 고원 ✓' if spread_10 < 0.15 else '좁은 봉우리 ✗'}")

    print(f"\n  Crossover SMA(20)>SMA(X) 플래토:")
    cross_sharpes_20 = []
    for (name, _), r in zip(K_ROBUST[13:19], rK[13:19]):
        cross_sharpes_20.append((name, r['metrics']['Sharpe']))
    for name, s in cross_sharpes_20:
        bar = '█' * int(s * 20)
        print(f"    {name:<12}: {s:.3f} {bar}")
    vals_20 = [s for _, s in cross_sharpes_20]
    spread_20 = max(vals_20) - min(vals_20)
    print(f"    spread: {spread_20:.3f} {'넓은 고원 ✓' if spread_20 < 0.15 else '좁은 봉우리 ✗'}")

    # Hybrid vote plateau
    print(f"\n  Hybrid 2/3 vote 플래토:")
    for (name, _), r in zip(K_ROBUST[19:], rK[19:]):
        s = r['metrics']['Sharpe']
        bar = '█' * int(s * 20)
        print(f"    {name:<25}: {s:.3f} {bar}")

    # K top 5 (updated with robustness info)
    k_top5 = find_top5("K Top 5 (강건성 검증 후)", K_ROBUST, rK)

    print(f"\n  Completed all in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
