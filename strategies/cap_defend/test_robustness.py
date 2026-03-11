#!/usr/bin/env python3
"""Robustness test: Are K5, H5, R2 real effects or lucky parameter picks?
   Test all variants + parameter neighborhoods."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

# ══════════════════════════════════════════════════════════════════
# 1. Canary concept: ALL K variants (is "canary" itself useful?)
# ══════════════════════════════════════════════════════════════════
K_TEST = [
    ('카나리 없음',              Params(canary='none')),
    ('baseline (BTC>SMA150)',  Params()),
    ('K1 (히스테리시스)',         Params(canary='K1')),
    ('K2 (비대칭 MA)',           Params(canary='K2')),
    ('K3 (7일 유예)',            Params(canary='K3')),
    ('K4 (BTC+ETH 듀얼)',       Params(canary='K4')),
    ('K5 (2/3 투표)',            Params(canary='K5')),
]

# ══════════════════════════════════════════════════════════════════
# 2. K5 parameter neighborhood: vary SMA period
# ══════════════════════════════════════════════════════════════════
K5_NEIGHBORHOOD = [
    ('K5 sma=100',  Params(canary='K5', sma_period=100)),
    ('K5 sma=120',  Params(canary='K5', sma_period=120)),
    ('K5 sma=150',  Params(canary='K5', sma_period=150)),
    ('K5 sma=180',  Params(canary='K5', sma_period=180)),
    ('K5 sma=200',  Params(canary='K5', sma_period=200)),
    # Also vary the baseline canary SMA
    ('base sma=100', Params(sma_period=100)),
    ('base sma=120', Params(sma_period=120)),
    ('base sma=150', Params(sma_period=150)),
    ('base sma=200', Params(sma_period=200)),
]

# ══════════════════════════════════════════════════════════════════
# 3. Health concept: ALL H variants (K5 fixed)
# ══════════════════════════════════════════════════════════════════
H_TEST = [
    ('헬스 없음',                Params(canary='K5', health='none')),
    ('baseline (SMA30+Mom21)', Params(canary='K5')),
    ('H1 (+Mom90)',            Params(canary='K5', health='H1')),
    ('H2 (Dual MA)',           Params(canary='K5', health='H2')),
    ('H4 (BTC 상대강도)',       Params(canary='K5', health='H4')),
    ('H5 (Vol Accel Block)',   Params(canary='K5', health='H5')),
]

# ══════════════════════════════════════════════════════════════════
# 4. R concept: with K5+H5 fixed
# ══════════════════════════════════════════════════════════════════
R_TEST = [
    ('R 없음 (월간만)',          Params(canary='K5', health='H5')),
    ('R2 (MTD-15%)',           Params(canary='K5', health='H5', rebalancing='R2')),
    ('R7 (MTD-10%)',           Params(canary='K5', health='H5', rebalancing='R7')),
    ('R8 (MTD-20%)',           Params(canary='K5', health='H5', rebalancing='R8')),
]

# ══════════════════════════════════════════════════════════════════
# 5. S concept: with K5+H5 fixed
# ══════════════════════════════════════════════════════════════════
S_TEST = [
    ('S 없음 (시총순)',          Params(canary='K5', health='H5')),
    ('S5 (+2순위)',             Params(canary='K5', health='H5', selection='S5')),
    ('S9 (히스테리시스)',         Params(canary='K5', health='H5', selection='S9')),
    ('S10 (보유우선)',           Params(canary='K5', health='H5', selection='S10')),
]


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


def print_section(title, strategies, results):
    print(f"\n{'=' * 120}")
    print(f"  {title}")
    print(f"{'=' * 120}")

    years = range(2018, 2026)

    # Main metrics
    print(f"\n  {'전략':<25} │{'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Final':>10} │ 연도별 CAGR")
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
        print(f"  {name:<25} │{m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {m['Final']:>10,.0f} │{yr_str}")

    # Year-by-year Sharpe
    print(f"\n  {'전략':<25} │ 연도별 Sharpe")
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
        print(f"  {name:<25} │{yr_str}")


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()
    rK = run_set(K_TEST, prices, universe)
    rK5n = run_set(K5_NEIGHBORHOOD, prices, universe)
    rH = run_set(H_TEST, prices, universe)
    rR = run_set(R_TEST, prices, universe)
    rS = run_set(S_TEST, prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    print_section("1. 카나리 개념 자체가 유효한가? (모든 K 변형, tx=0.4%)", K_TEST, rK)
    print_section("2. K5 파라미터 이웃 (SMA 기간 변경)", K5_NEIGHBORHOOD, rK5n)
    print_section("3. 헬스체크 개념 자체가 유효한가? (K5 고정, 모든 H 변형)", H_TEST, rH)
    print_section("4. 리스크관리 R (K5+H5 고정)", R_TEST, rR)
    print_section("5. 코인선택 S (K5+H5 고정)", S_TEST, rS)

    # ══════════════════════════════════════════════════════════════
    # Summary: concept-level analysis
    # ══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 120}")
    print(f"  종합: 각 레이어의 '개념' 효과 (최악 vs 최선)")
    print(f"{'=' * 120}")

    def summarize(label, strategies, results):
        sharpes = [(name, r['metrics']['Sharpe']) for (name,_), r in zip(strategies, results)]
        sharpes.sort(key=lambda x: x[1])
        worst = sharpes[0]
        best = sharpes[-1]
        avg = sum(s for _,s in sharpes) / len(sharpes)
        spread = best[1] - worst[1]
        print(f"\n  [{label}]")
        print(f"    최악: {worst[0]:<25} Sharpe {worst[1]:.3f}")
        print(f"    최선: {best[0]:<25} Sharpe {best[1]:.3f}")
        print(f"    평균: {avg:.3f}  |  편차: {spread:.3f}")
        # Check: does every variant beat "none"?
        none_sharpe = None
        for (name,_), r in zip(strategies, results):
            if '없음' in name or 'none' in name.lower():
                none_sharpe = r['metrics']['Sharpe']
                break
        if none_sharpe is not None:
            beats = sum(1 for _,s in sharpes if s > none_sharpe)
            print(f"    '없음' 대비 개선: {beats}/{len(sharpes)-1}개 변형")

    summarize("카나리 (K)", K_TEST, rK)
    summarize("헬스 (H)", H_TEST, rH)
    summarize("리스크 (R)", R_TEST, rR)
    summarize("선택 (S)", S_TEST, rS)

    # K5 neighborhood check
    print(f"\n  [K5 SMA 이웃]")
    for (name, _), r in zip(K5_NEIGHBORHOOD, rK5n):
        if 'K5' in name:
            m = r['metrics']
            print(f"    {name:<15} Sharpe {m['Sharpe']:.3f}  CAGR {m['CAGR']:>+7.1%}  MDD {m['MDD']:>6.1%}")


if __name__ == '__main__':
    main()
