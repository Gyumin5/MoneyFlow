#!/usr/bin/env python3
"""Stack promising layers on new baseline.
   Test G5, W1, W6, R7, S8 individually and in combinations."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

def B(**kw):
    base = dict(
        canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1,
        health='HK', health_sma=2, health_mom_short=21,
        health_mom_long=90, vol_cap=0.05,
    )
    base.update(kw)
    return Params(**base)

COMBOS = [
    # Baseline
    ('BASELINE', B()),

    # Singles (confirmed improvements)
    ('G5', B(risk='G5')),
    ('W1', B(weighting='W1')),
    ('W6', B(weighting='W6')),
    ('R7', B(rebalancing='R7')),
    ('S8', B(selection='S8')),

    # 2-layer stacks
    ('G5+W1', B(risk='G5', weighting='W1')),
    ('G5+W6', B(risk='G5', weighting='W6')),
    ('G5+R7', B(risk='G5', rebalancing='R7')),
    ('G5+S8', B(risk='G5', selection='S8')),
    ('W1+R7', B(weighting='W1', rebalancing='R7')),
    ('W1+S8', B(weighting='W1', selection='S8')),
    ('W6+R7', B(weighting='W6', rebalancing='R7')),

    # 3-layer stacks
    ('G5+W1+R7', B(risk='G5', weighting='W1', rebalancing='R7')),
    ('G5+W1+S8', B(risk='G5', weighting='W1', selection='S8')),
    ('G5+W6+R7', B(risk='G5', weighting='W6', rebalancing='R7')),
    ('G5+W6+S8', B(risk='G5', weighting='W6', selection='S8')),

    # 4-layer stack
    ('G5+W1+R7+S8', B(risk='G5', weighting='W1', rebalancing='R7', selection='S8')),
    ('G5+W6+R7+S8', B(risk='G5', weighting='W6', rebalancing='R7', selection='S8')),

    # With different n_picks
    ('G5+W1 n=3', B(risk='G5', weighting='W1', n_picks=3)),
    ('G5+W1 n=7', B(risk='G5', weighting='W1', n_picks=7)),
    ('G5+W1 n=10', B(risk='G5', weighting='W1', n_picks=10)),
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
            vol_cap=p.vol_cap,
            health_sma=p.health_sma, health_mom_short=p.health_mom_short,
            health_mom_long=p.health_mom_long, health_vol_window=p.health_vol_window,
            selection=p.selection, weighting=p.weighting,
            rebalancing=p.rebalancing, risk=p.risk,
            n_picks=p.n_picks, top_n=p.top_n,
        ))
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, params_list)
    return results


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded, {len(COMBOS)} combos")

    t0 = time.time()
    results = run_set(COMBOS, prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # Full results
    print(f"\n{'=' * 110}")
    print(f"  레이어 스택 결과 — K=SMA(60) + H=Mom(21)&Mom(90)&Vol5%")
    print(f"{'=' * 110}")
    print(f"\n  {'조합':<22} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'Rebal':>6}")
    print(f"  {'─' * 65}")

    all_scored = []
    for (name, _), r in zip(COMBOS, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        rc = r.get('rebal_count', 0)
        print(f"  {name:<22} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f} {rc:>6}")
        all_scored.append((name, m, r, calmar))

    # Sorted by Sharpe
    all_scored.sort(key=lambda x: x[1]['Sharpe'], reverse=True)
    print(f"\n{'=' * 110}")
    print(f"  Sharpe 순위")
    print(f"{'=' * 110}")
    print(f"\n  {'순위':>3} {'조합':<22} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 60}")
    for i, (name, m, r, calmar) in enumerate(all_scored[:10], 1):
        print(f"  {i:>3}. {name:<22} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}")

    # Sorted by Calmar
    all_scored_c = sorted(all_scored, key=lambda x: x[3], reverse=True)
    print(f"\n{'=' * 110}")
    print(f"  Calmar 순위")
    print(f"{'=' * 110}")
    print(f"\n  {'순위':>3} {'조합':<22} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 60}")
    for i, (name, m, r, calmar) in enumerate(all_scored_c[:10], 1):
        print(f"  {i:>3}. {name:<22} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}")

    # Year-by-year for top 5
    years = range(2018, 2026)
    print(f"\n  Top 5 연도별 CAGR")
    print(f"  {'조합':<22}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print(f" {'전체':>8}")
    print(f"  {'─' * 95}")
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

    # Stack effect analysis
    base_s = results[0]['metrics']['Sharpe']
    g5_s = results[1]['metrics']['Sharpe']
    w1_s = results[2]['metrics']['Sharpe']
    g5w1_s = results[6]['metrics']['Sharpe']
    expected = (g5_s - base_s) + (w1_s - base_s) + base_s
    print(f"\n  스택 효과 분석:")
    print(f"    G5 단독 개선: +{g5_s - base_s:.3f}")
    print(f"    W1 단독 개선: +{w1_s - base_s:.3f}")
    print(f"    기대 합산:     +{expected - base_s:.3f} (→ {expected:.3f})")
    print(f"    실제 G5+W1:    +{g5w1_s - base_s:.3f} (→ {g5w1_s:.3f})")
    if g5w1_s >= expected:
        print(f"    → 시너지 효과 ✓ (실제 > 기대)")
    elif g5w1_s >= base_s + 0.05:
        print(f"    → 부분 합산 (실제 < 기대지만 유의미)")
    else:
        print(f"    → 상호 상쇄 ✗")

    print(f"\n  Completed all in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
