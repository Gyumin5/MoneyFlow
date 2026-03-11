#!/usr/bin/env python3
"""Test AI-recommended canary strategies.
   Gemini + Codex consensus: SMA plateau(50-110) 활용, 짧은 Mom 배제.
   Top 5 K candidates selected from AI recommendations."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

# ══════════════════════════════════════════════════════════════════
# AI-recommended K candidates
# ══════════════════════════════════════════════════════════════════

STRATEGIES = [
    # ── Reference ────────────────────────────────────────────────
    ('K5 sma80 (기존)',     Params(canary='K5', health='H1', sma_period=80)),
    ('SMA(60) 단일',        Params(canary='K8', health='H1',
                                   vote_smas=(60,), vote_moms=(), vote_threshold=1)),

    # ── 1. Plateau midpoint SMA(80) 단일 (Gemini 추천) ──────────
    ('SMA(80) 단일',        Params(canary='K8', health='H1',
                                   vote_smas=(80,), vote_moms=(), vote_threshold=1)),

    # ── 2. Multi-TF SMA Vote 2/3: SMA(50,80,110) (Codex 추천) ──
    ('2/3 SMA(50,80,110)',  Params(canary='K8', health='H1',
                                   vote_smas=(50, 80, 110), vote_moms=(), vote_threshold=2)),

    # ── 3. Multi-TF SMA Vote 2/3: SMA(60,80,100) (Gemini 추천) ─
    ('2/3 SMA(60,80,100)',  Params(canary='K8', health='H1',
                                   vote_smas=(60, 80, 100), vote_moms=(), vote_threshold=2)),

    # ── 4. SMA crossover: SMA(20)>SMA(80) (Gemini 추천) ─────────
    ('SMA(20)>SMA(80)',     Params(canary='K9', health='H1',
                                   cross_short=20, cross_long=80)),

    # ── 5. SMA + Long Mom: SMA(80) AND Mom(120) (양쪽 추천) ─────
    ('SMA(80)+Mom(120)',    Params(canary='K8', health='H1',
                                   vote_smas=(80,), vote_moms=(120,), vote_threshold=2)),

    # ── 6. SMA(90) AND Mom(120) (Codex 보수적 안) ───────────────
    ('SMA(90)+Mom(120)',    Params(canary='K8', health='H1',
                                   vote_smas=(90,), vote_moms=(120,), vote_threshold=2)),

    # ── 7. Hybrid 2/3: SMA(60)+SMA(110)+Mom(120) (Codex 추천) ──
    ('2/3 SMA60,110+M120', Params(canary='K8', health='H1',
                                   vote_smas=(60, 110), vote_moms=(120,), vote_threshold=2)),

    # ── 8. SMA crossover variants ───────────────────────────────
    ('SMA(10)>SMA(60)',     Params(canary='K9', health='H1',
                                   cross_short=10, cross_long=60)),
    ('SMA(20)>SMA(60)',     Params(canary='K9', health='H1',
                                   cross_short=20, cross_long=60)),
    ('SMA(30)>SMA(80)',     Params(canary='K9', health='H1',
                                   cross_short=30, cross_long=80)),
    ('SMA(20)>SMA(100)',    Params(canary='K9', health='H1',
                                   cross_short=20, cross_long=100)),

    # ── 9. Broader SMA vote variants ────────────────────────────
    ('2/3 SMA(50,80,120)',  Params(canary='K8', health='H1',
                                   vote_smas=(50, 80, 120), vote_moms=(), vote_threshold=2)),
    ('3/5 SMA(50,60,80,100,110)', Params(canary='K8', health='H1',
                                   vote_smas=(50, 60, 80, 100, 110), vote_moms=(), vote_threshold=3)),
    ('2/3 SMA(60,90,120)',  Params(canary='K8', health='H1',
                                   vote_smas=(60, 90, 120), vote_moms=(), vote_threshold=2)),

    # ── 10. SMA + very long Mom ─────────────────────────────────
    ('SMA(60)+Mom(90)',     Params(canary='K8', health='H1',
                                   vote_smas=(60,), vote_moms=(90,), vote_threshold=2)),
    ('SMA(80)+Mom(90)',     Params(canary='K8', health='H1',
                                   vote_smas=(80,), vote_moms=(90,), vote_threshold=2)),
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
    results = run_set(STRATEGIES, prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    years = range(2018, 2026)

    # ── Main table ──────────────────────────────────────────────
    print(f"\n{'=' * 130}")
    print(f"  AI 추천 카나리아 전략 비교 (H1 고정, tx=0.4%)")
    print(f"{'=' * 130}")

    print(f"\n  {'전략':<25} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Final':>10}"
          f"  │ {'Calmar':>7} {'리밸':>4}")
    print(f"  {'─' * 80}")

    scored = []
    for (name, _), r in zip(STRATEGIES, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        scored.append((name, m, r, calmar))
        print(f"  {name:<25} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {m['Final']:>10,.0f}"
              f"  │ {calmar:>7.2f} {r.get('rebal_count', 0):>4}")

    # ── Year-by-year ────────────────────────────────────────────
    print(f"\n  연도별 CAGR")
    print(f"  {'전략':<25}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print(f" {'전체':>8}")
    print(f"  {'─' * 95}")
    for (name, _), r in zip(STRATEGIES, results):
        ym = r['yearly']
        m = r['metrics']
        row = f"  {name:<25}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+6.1%}"
            else:
                row += f" {'─':>7}"
        row += f" {m['CAGR']:>+7.1%}"
        print(row)

    # ── Top 5 ───────────────────────────────────────────────────
    scored.sort(key=lambda x: x[1]['Sharpe'], reverse=True)
    print(f"\n{'=' * 130}")
    print(f"  K 카나리아 Top 5 (Sharpe 기준)")
    print(f"{'=' * 130}")
    print(f"\n  {'순위':>3} {'전략':<25} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 65}")
    for i, (name, m, r, calmar) in enumerate(scored[:5], 1):
        print(f"  {i:>3}. {name:<25} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}")

    # ── Consistency check: year-by-year for top 5 ───────────────
    print(f"\n  Top 5 연도별 CAGR")
    print(f"  {'전략':<25}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print()
    print(f"  {'─' * 90}")
    for name, m, r, _ in scored[:5]:
        ym = r['yearly']
        row = f"  {name:<25}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+6.1%}"
            else:
                row += f" {'─':>7}"
        print(row)

    # ── Robustness: MDD by year for top 5 ───────────────────────
    print(f"\n  Top 5 연도별 MDD")
    print(f"  {'전략':<25}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print()
    print(f"  {'─' * 90}")
    for name, m, r, _ in scored[:5]:
        ym = r['yearly']
        row = f"  {name:<25}"
        for y in years:
            if y in ym:
                row += f" {ym[y].get('MDD', 0):>6.1%}"
            else:
                row += f" {'─':>7}"
        print(row)


if __name__ == '__main__':
    main()
