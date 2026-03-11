#!/usr/bin/env python3
"""Test unified indicator approach:
   All-SMA (K+H both SMA) vs All-Mom (K+H both Mom) vs Mixed."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())


# ══════════════════════════════════════════════════════════════════
# All-Mom: K=Mom, H=Mom+Mom+Vol
# ══════════════════════════════════════════════════════════════════
ALL_MOM = []
for k_mom in [30, 35, 40, 45, 50, 60]:
    for h_short in [14, 21, 30]:
        for h_long in [60, 90, 120]:
            if h_short >= h_long:
                continue
            ALL_MOM.append(
                (f'K:m{k_mom} H:m{h_short}+m{h_long}',
                 Params(canary='K8', vote_smas=(), vote_moms=(k_mom,), vote_threshold=1,
                        health='HK', health_sma=2, health_mom_short=h_short,
                        health_mom_long=h_long, vol_cap=0.05)))

# ══════════════════════════════════════════════════════════════════
# All-SMA: K=SMA, H=SMA+SMA+Vol (crossover in H)
# ══════════════════════════════════════════════════════════════════
ALL_SMA = []
for k_sma in [50, 55, 60, 65, 70, 80]:
    for h_short in [10, 15, 20, 30]:
        for h_long in [60, 80, 100]:
            if h_short >= h_long:
                continue
            ALL_SMA.append(
                (f'K:s{k_sma} H:s{h_short}>s{h_long}',
                 Params(canary='K8', vote_smas=(k_sma,), vote_moms=(), vote_threshold=1,
                        health='HN', health_mom_short=h_short, health_sma=h_long,
                        vol_cap=0.05)))

# ══════════════════════════════════════════════════════════════════
# Mixed: K=SMA, H=Mom+Mom+Vol (현재 방향)
# ══════════════════════════════════════════════════════════════════
MIXED_SK_HM = []
for k_sma in [50, 55, 60, 65, 70, 80]:
    for h_short in [14, 21, 30]:
        for h_long in [60, 90, 120]:
            if h_short >= h_long:
                continue
            MIXED_SK_HM.append(
                (f'K:s{k_sma} H:m{h_short}+m{h_long}',
                 Params(canary='K8', vote_smas=(k_sma,), vote_moms=(), vote_threshold=1,
                        health='HK', health_sma=2, health_mom_short=h_short,
                        health_mom_long=h_long, vol_cap=0.05)))

# ══════════════════════════════════════════════════════════════════
# Mixed reverse: K=Mom, H=SMA+Mom+Vol (HM style)
# ══════════════════════════════════════════════════════════════════
MIXED_MK_SH = []
for k_mom in [30, 40, 45, 50, 60]:
    for h_sma in [20, 30, 40]:
        for h_long in [60, 90, 120]:
            MIXED_MK_SH.append(
                (f'K:m{k_mom} H:s{h_sma}+m{h_long}',
                 Params(canary='K8', vote_smas=(), vote_moms=(k_mom,), vote_threshold=1,
                        health='HM', health_sma=h_sma, health_mom_long=h_long,
                        vol_cap=0.05)))


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


def analyze(title, strategies, results):
    scored = [(name, r['metrics'], r) for (name, _), r in zip(strategies, results)]
    scored.sort(key=lambda x: x[1]['Sharpe'], reverse=True)

    # Stats
    sharpes = [s[1]['Sharpe'] for s in scored]
    avg = sum(sharpes) / len(sharpes)
    best = max(sharpes)
    worst = min(sharpes)
    top5_avg = sum(s[1]['Sharpe'] for s in scored[:5]) / 5

    print(f"\n  {title}")
    print(f"  {'─' * 85}")
    print(f"  조합 수: {len(scored)}")
    print(f"  평균 Sharpe: {avg:.3f}  |  최고: {best:.3f}  |  최저: {worst:.3f}  |  Top5 평균: {top5_avg:.3f}")
    print(f"\n  Top 5:")
    print(f"  {'조합':<30} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 65}")
    for name, m, r in scored[:5]:
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        print(f"  {name:<30} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}")

    return scored


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")
    total = len(ALL_MOM) + len(ALL_SMA) + len(MIXED_SK_HM) + len(MIXED_MK_SH)
    print(f"  총 {total}개 조합 테스트")

    t0 = time.time()
    r1 = run_set(ALL_MOM, prices, universe)
    r2 = run_set(ALL_SMA, prices, universe)
    r3 = run_set(MIXED_SK_HM, prices, universe)
    r4 = run_set(MIXED_MK_SH, prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    print(f"\n{'=' * 100}")
    print(f"  지표 통일 vs 혼합 비교")
    print(f"{'=' * 100}")

    s1 = analyze("A. All-Mom (K=Mom, H=Mom+Mom+Vol)", ALL_MOM, r1)
    s2 = analyze("B. All-SMA (K=SMA, H=SMA>SMA+Vol)", ALL_SMA, r2)
    s3 = analyze("C. Mixed: K=SMA, H=Mom+Mom+Vol", MIXED_SK_HM, r3)
    s4 = analyze("D. Mixed: K=Mom, H=SMA+Mom+Vol", MIXED_MK_SH, r4)

    # ── Summary comparison ──────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  방식별 비교 요약")
    print(f"{'=' * 100}")
    print(f"\n  {'방식':<35} {'평균':>7} {'Top5평균':>9} {'최고':>7} {'최고 MDD':>8} {'최고 Calmar':>11}")
    print(f"  {'─' * 85}")

    for label, scored in [
        ('A. All-Mom', s1), ('B. All-SMA', s2),
        ('C. K=SMA, H=Mom+Mom', s3), ('D. K=Mom, H=SMA+Mom', s4)
    ]:
        sharpes = [s[1]['Sharpe'] for s in scored]
        avg = sum(sharpes) / len(sharpes)
        top5_avg = sum(s[1]['Sharpe'] for s in scored[:5]) / 5
        best_entry = scored[0]
        calmar = best_entry[1]['CAGR'] / abs(best_entry[1]['MDD']) if best_entry[1]['MDD'] != 0 else 0
        print(f"  {label:<35} {avg:>7.3f} {top5_avg:>9.3f} {best_entry[1]['Sharpe']:>7.3f}"
              f" {best_entry[1]['MDD']:>7.1%} {calmar:>11.2f}")

    # ── Overall Top 10 ──────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  전체 Top 10 (모든 방식 통합)")
    print(f"{'=' * 100}")

    all_scored = s1 + s2 + s3 + s4
    all_scored.sort(key=lambda x: x[1]['Sharpe'], reverse=True)
    print(f"\n  {'순위':>3} {'방식':>5} {'조합':<30} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 75}")
    for i, (name, m, r) in enumerate(all_scored[:10], 1):
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        # Determine type
        if name.startswith('K:m') and 'H:m' in name:
            typ = 'A'
        elif name.startswith('K:s') and 'H:s' in name:
            typ = 'B'
        elif name.startswith('K:s') and 'H:m' in name:
            typ = 'C'
        else:
            typ = 'D'
        print(f"  {i:>3}. [{typ}] {name:<30} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}")

    # Year-by-year for top 5
    years = range(2018, 2026)
    print(f"\n  Top 5 연도별 CAGR")
    print(f"  {'조합':<30}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print(f" {'전체':>8}")
    print(f"  {'─' * 100}")
    for name, m, r in all_scored[:5]:
        ym = r['yearly']
        row = f"  {name:<30}"
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
