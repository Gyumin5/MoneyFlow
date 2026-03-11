#!/usr/bin/env python3
"""Test H voting strategies + K overfitting analysis.
   1. H: AND vs Mom-lookback voting (Vol always hard gate)
   2. K: SMA(60) overfitting check — walk-forward, leave-one-year-out"""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single, run_backtest

N_WORKERS = min(24, mp.cpu_count())


# ══════════════════════════════════════════════════════════════════
# Part A: H Vote — Vol hard gate + Mom lookback vote
# ══════════════════════════════════════════════════════════════════
K_FIXED = dict(canary='K8', vote_smas=(60,), vote_threshold=1)

# Baselines (AND style)
H_BASE = [
    ('AND: m21+m90+v5%',
     Params(**K_FIXED, health='HK', health_sma=2, health_mom_short=21,
            health_mom_long=90, vol_cap=0.05)),
    ('AND: m14+m90+v5%',
     Params(**K_FIXED, health='HK', health_sma=2, health_mom_short=14,
            health_mom_long=90, vol_cap=0.05)),
    ('AND: m30+m90+v5%',
     Params(**K_FIXED, health='HK', health_sma=2, health_mom_short=30,
            health_mom_long=90, vol_cap=0.05)),
    ('AND: m21+m60+v5%',
     Params(**K_FIXED, health='HK', health_sma=2, health_mom_short=21,
            health_mom_long=60, vol_cap=0.05)),
    ('AND: m90만+v5%',
     Params(**K_FIXED, health='HL', health_mom_long=90, vol_cap=0.05)),
]

# Codex style: spaced lookbacks (21, 60, 90) — 2/3 or 3/3
H_VOTE_SPACED = []
spaced = (21, 60, 90)
for thresh in [2, 3]:
    H_VOTE_SPACED.append(
        (f'vote {thresh}/3 m(21,60,90)+v5%',
         Params(**K_FIXED, health='HV', vote_moms=spaced,
                health_mom_short=thresh, vol_cap=0.05)))

spaced2 = (21, 45, 90)
for thresh in [2, 3]:
    H_VOTE_SPACED.append(
        (f'vote {thresh}/3 m(21,45,90)+v5%',
         Params(**K_FIXED, health='HV', vote_moms=spaced2,
                health_mom_short=thresh, vol_cap=0.05)))

spaced3 = (14, 60, 90)
for thresh in [2, 3]:
    H_VOTE_SPACED.append(
        (f'vote {thresh}/3 m(14,60,90)+v5%',
         Params(**K_FIXED, health='HV', vote_moms=spaced3,
                health_mom_short=thresh, vol_cap=0.05)))

spaced4 = (21, 60, 120)
for thresh in [2, 3]:
    H_VOTE_SPACED.append(
        (f'vote {thresh}/3 m(21,60,120)+v5%',
         Params(**K_FIXED, health='HV', vote_moms=spaced4,
                health_mom_short=thresh, vol_cap=0.05)))

# Gemini style: many lookbacks (14, 21, 30, 60, 90) — 3/5 or 4/5
H_VOTE_MANY = []
many = (14, 21, 30, 60, 90)
for thresh in [2, 3, 4, 5]:
    H_VOTE_MANY.append(
        (f'vote {thresh}/5 m(14~90)+v5%',
         Params(**K_FIXED, health='HV', vote_moms=many,
                health_mom_short=thresh, vol_cap=0.05)))

many2 = (7, 14, 21, 45, 90)
for thresh in [2, 3, 4, 5]:
    H_VOTE_MANY.append(
        (f'vote {thresh}/5 m(7~90)+v5%',
         Params(**K_FIXED, health='HV', vote_moms=many2,
                health_mom_short=thresh, vol_cap=0.05)))

many3 = (14, 30, 60, 90, 120)
for thresh in [2, 3, 4, 5]:
    H_VOTE_MANY.append(
        (f'vote {thresh}/5 m(14~120)+v5%',
         Params(**K_FIXED, health='HV', vote_moms=many3,
                health_mom_short=thresh, vol_cap=0.05)))

# 4-lookback combos
four = (14, 30, 60, 90)
for thresh in [2, 3, 4]:
    H_VOTE_MANY.append(
        (f'vote {thresh}/4 m(14,30,60,90)+v5%',
         Params(**K_FIXED, health='HV', vote_moms=four,
                health_mom_short=thresh, vol_cap=0.05)))

four2 = (21, 45, 90, 120)
for thresh in [2, 3, 4]:
    H_VOTE_MANY.append(
        (f'vote {thresh}/4 m(21,45,90,120)+v5%',
         Params(**K_FIXED, health='HV', vote_moms=four2,
                health_mom_short=thresh, vol_cap=0.05)))

ALL_H = H_BASE + H_VOTE_SPACED + H_VOTE_MANY


# ══════════════════════════════════════════════════════════════════
# Part B: K SMA(60) Overfitting Check — Leave-One-Year-Out
# ══════════════════════════════════════════════════════════════════
K_OVERFIT = []
for sma in [50, 55, 60, 65, 70, 75, 80]:
    K_OVERFIT.append(
        (f'SMA({sma})',
         Params(canary='K8', vote_smas=(sma,), vote_moms=(), vote_threshold=1,
                health='HK', health_sma=2, health_mom_short=21,
                health_mom_long=90, vol_cap=0.05)))


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
    print(f"  {'─' * 80}")
    print(f"  {'조건':<35} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 80}")
    best_s = max(r['metrics']['Sharpe'] for r in results) if results else 0
    for (name, _), r in zip(strategies, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        marker = " ★" if m['Sharpe'] == best_s else ""
        print(f"  {name:<35} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}{marker}")


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")
    print(f"  H vote: {len(ALL_H)} configs, K overfit: {len(K_OVERFIT)} configs")

    t0 = time.time()

    # ── Part A: H vote ─────────────────────────────────────────────
    print("\n  Running H vote tests...")
    r_h = run_set(ALL_H, prices, universe)

    idx = 0
    def get_slice(s):
        nonlocal idx
        r = r_h[idx:idx+len(s)]
        idx += len(s)
        return r

    r_base = get_slice(H_BASE)
    r_spaced = get_slice(H_VOTE_SPACED)
    r_many = get_slice(H_VOTE_MANY)

    print(f"\n{'=' * 100}")
    print(f"  H 투표 전략 비교 — K=SMA(60) 고정")
    print(f"{'=' * 100}")

    print_section("0. 기준 (AND 방식)", H_BASE, r_base)
    print_section("1. 간격 벌린 투표 (Codex안: 3개 lookback)", H_VOTE_SPACED, r_spaced)
    print_section("2. 다수 투표 (Gemini안: 4~5개 lookback)", H_VOTE_MANY, r_many)

    # ── H Overall Top 10 ───────────────────────────────────────────
    all_h_scored = []
    for (name, _), r in zip(ALL_H, r_h):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        all_h_scored.append((name, m, r, calmar))
    all_h_scored.sort(key=lambda x: x[1]['Sharpe'], reverse=True)

    print(f"\n{'=' * 100}")
    print(f"  H 전체 Top 10")
    print(f"{'=' * 100}")
    print(f"\n  {'순위':>3} {'H 조건':<35} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 70}")
    for i, (name, m, r, calmar) in enumerate(all_h_scored[:10], 1):
        print(f"  {i:>3}. {name:<35} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}")

    # Year-by-year for top 5 H
    years = range(2018, 2026)
    print(f"\n  H Top 5 연도별 CAGR")
    print(f"  {'H 조건':<35}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print(f" {'전체':>8}")
    print(f"  {'─' * 105}")
    for name, m, r, calmar in all_h_scored[:5]:
        ym = r['yearly']
        row = f"  {name:<35}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+6.1%}"
            else:
                row += f" {'─':>7}"
        row += f" {m['CAGR']:>+7.1%}"
        print(row)

    # ── H Vote vs AND comparison ───────────────────────────────────
    and_best = max(r_base, key=lambda r: r['metrics']['Sharpe'])
    and_idx = r_base.index(and_best)
    vote_best_entry = all_h_scored[0]

    print(f"\n  H 결론:")
    print(f"    AND 최고: {H_BASE[and_idx][0]} → Sharpe {and_best['metrics']['Sharpe']:.3f}, MDD {and_best['metrics']['MDD']:.1%}")
    print(f"    투표 1위: {vote_best_entry[0]} → Sharpe {vote_best_entry[1]['Sharpe']:.3f}, MDD {vote_best_entry[1]['MDD']:.1%}")
    diff = vote_best_entry[1]['Sharpe'] - and_best['metrics']['Sharpe']
    if diff > 0.05:
        print(f"    → 투표가 AND보다 +{diff:.3f} 개선 ✓")
    elif diff > 0:
        print(f"    → 투표가 소폭 개선 +{diff:.3f} (유의미하지 않을 수 있음)")
    else:
        print(f"    → AND 방식이 투표보다 나음 ({diff:+.3f})")

    # ══════════════════════════════════════════════════════════════
    # Part B: K SMA(60) Overfitting — Leave-One-Year-Out
    # ══════════════════════════════════════════════════════════════
    print(f"\n\n{'=' * 100}")
    print(f"  K SMA 과적합 검증 — Leave-One-Year-Out Cross-Validation")
    print(f"{'=' * 100}")

    sma_vals = [50, 55, 60, 65, 70, 75, 80]
    test_years = [2019, 2020, 2021, 2022, 2023, 2024]

    print(f"\n  각 연도를 제외하고 나머지로 학습 → 제외된 연도에서 검증")
    print(f"  {'제외 연도':>8}", end="")
    for sma in sma_vals:
        print(f" {'SMA'+str(sma):>8}", end="")
    print(f" {'최적':>8}")
    print(f"  {'─' * (10 + 9 * len(sma_vals) + 10)}")

    oos_winners = []  # which SMA wins out-of-sample each year
    for test_year in test_years:
        # Train: all years except test_year
        train_start = '2018-01-01'
        train_end = '2025-06-30'

        # Run full period first, then look at yearly metrics
        print(f"  {test_year:>8}", end="")
        best_train_sma = None
        best_train_sharpe = -999
        year_results = {}

        for sma_i, sma in enumerate(sma_vals):
            p = Params(canary='K8', vote_smas=(sma,), vote_moms=(), vote_threshold=1,
                       health='HK', health_sma=2, health_mom_short=21,
                       health_mom_long=90, vol_cap=0.05)
            r = run_backtest(prices, universe, p)

            # Get test year performance
            ym = r.get('yearly', {})
            if test_year in ym:
                test_sharpe = ym[test_year].get('Sharpe', 0)
            else:
                test_sharpe = 0

            # Get train sharpe (average of non-test years)
            train_sharpes = []
            for y, yd in ym.items():
                if y != test_year and y >= 2018:
                    train_sharpes.append(yd.get('Sharpe', 0))
            train_sharpe = sum(train_sharpes) / len(train_sharpes) if train_sharpes else 0

            year_results[sma] = (train_sharpe, test_sharpe)
            if train_sharpe > best_train_sharpe:
                best_train_sharpe = train_sharpe
                best_train_sma = sma

            print(f" {test_sharpe:>8.3f}", end="")

        # Mark which was best in training
        oos_winners.append(best_train_sma)
        print(f" SMA({best_train_sma})")

    print(f"\n  학습 데이터 최적 SMA 분포: {oos_winners}")
    from collections import Counter
    counts = Counter(oos_winners)
    print(f"  빈도: {dict(counts)}")
    if len(counts) <= 2:
        print(f"  → SMA(60) 근처에 집중 = 과적합 아님 ✓")
    else:
        print(f"  → 분산됨 = 특정 값 과적합 가능성")

    # ── Full period K comparison ───────────────────────────────────
    print(f"\n  전체 기간 K SMA 비교:")
    r_k = run_set(K_OVERFIT, prices, universe)
    for (name, _), r in zip(K_OVERFIT, r_k):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        ym = r['yearly']
        # Count years where this SMA is top 2
        print(f"    {name:<10} Sharpe {m['Sharpe']:.3f}  MDD {m['MDD']:>6.1%}  Cal {calmar:.2f}", end="")
        # Show per-year sharpe
        for y in [2019, 2020, 2021, 2022, 2023, 2024]:
            if y in ym:
                print(f"  {y}:{ym[y].get('Sharpe', 0):>5.2f}", end="")
        print()

    sharpes = [r['metrics']['Sharpe'] for r in r_k]
    s60 = sharpes[sma_vals.index(60)]
    neighbors = [sharpes[sma_vals.index(s)] for s in [55, 65] if s in sma_vals]
    avg_neighbor = sum(neighbors) / len(neighbors)
    print(f"\n  SMA(60) Sharpe: {s60:.3f}")
    print(f"  이웃 평균 (55,65): {avg_neighbor:.3f}")
    print(f"  차이: {s60 - avg_neighbor:+.3f}")
    if abs(s60 - avg_neighbor) < 0.05:
        print(f"  → 이웃과 차이 미미 = SMA(60) 과적합 아님 ✓")
    else:
        print(f"  → 이웃과 차이 있음 = 주의 필요")

    total_spread = max(sharpes) - min(sharpes)
    plateau_spread = max(sharpes[sma_vals.index(s)] for s in [55, 60, 65, 70]) - \
                     min(sharpes[sma_vals.index(s)] for s in [55, 60, 65, 70])
    print(f"  전체 spread (50-80): {total_spread:.3f}")
    print(f"  고원 spread (55-70): {plateau_spread:.3f}")
    if plateau_spread < 0.10:
        print(f"  → 55-70 구간은 매우 안정적인 고원 ✓")

    print(f"\n  Completed all in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
