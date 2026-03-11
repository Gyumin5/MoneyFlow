#!/usr/bin/env python3
"""Detailed analysis:
   1. Dynamic trailing stop variants (vs rigid MTD)
   2. Baseline vs W6 year-by-year deep comparison
   3. Selection method impact (is market cap really best?)
   4. Universe bias check"""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single, run_backtest

N_WORKERS = min(24, mp.cpu_count())

def B(**kw):
    base = dict(
        canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1,
        health='HK', health_sma=2, health_mom_short=21,
        health_mom_long=90, vol_cap=0.05,
    )
    base.update(kw)
    return Params(**base)

# ══════════════════════════════════════════════════════════════════
# Part 1: Dynamic trailing stop variants
# R6=60d high -20%, R9=30d high -15%, R7=MTD -10%
# Need new variants: different window × threshold combos
# ══════════════════════════════════════════════════════════════════
TRAILING = [
    ('월간만 (baseline)', B(risk='G5')),
    ('R7: MTD -10%',     B(risk='G5', rebalancing='R7')),
    ('R2: MTD -15%',     B(risk='G5', rebalancing='R2')),
    ('R8: MTD -20%',     B(risk='G5', rebalancing='R8')),
    ('R6: 60d고점 -20%', B(risk='G5', rebalancing='R6')),
    ('R9: 30d고점 -15%', B(risk='G5', rebalancing='R9')),
]

# ══════════════════════════════════════════════════════════════════
# Part 2: W baseline(EW) vs W6(inv-vol) detailed
# ══════════════════════════════════════════════════════════════════
W_COMPARE = [
    ('EW (균등)', B(risk='G5')),
    ('W1 순위감소', B(risk='G5', weighting='W1')),
    ('W2 inv-vol70%', B(risk='G5', weighting='W2')),
    ('W6 inv-vol100%', B(risk='G5', weighting='W6')),
]

# ══════════════════════════════════════════════════════════════════
# Part 3: Selection deep comparison
# ══════════════════════════════════════════════════════════════════
S_COMPARE = [
    ('시총순 (baseline)', B(risk='G5')),
    ('S1: cap15→mom top5', B(risk='G5', selection='S1')),
    ('S6: Sharpe score', B(risk='G5', selection='S6')),
    ('S7: cap15→Sharpe', B(risk='G5', selection='S7')),
    ('S2: cap+mom blend', B(risk='G5', selection='S2')),
]

ALL = TRAILING + W_COMPARE + S_COMPARE


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


def print_full(title, strategies, results):
    print(f"\n  {title}")
    print(f"  {'─' * 95}")
    print(f"  {'조건':<22} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7} {'Rebal':>6}")
    print(f"  {'─' * 95}")
    for (name, _), r in zip(strategies, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        rc = r.get('rebal_count', 0)
        print(f"  {name:<22} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f} {rc:>6}")

    # Year-by-year
    years = range(2018, 2026)
    print(f"\n  연도별 CAGR:")
    print(f"  {'조건':<22}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print()
    print(f"  {'─' * 85}")
    for (name, _), r in zip(strategies, results):
        ym = r['yearly']
        row = f"  {name:<22}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+6.1%}"
            else:
                row += f" {'─':>7}"
        print(row)

    # Year-by-year Sharpe
    print(f"\n  연도별 Sharpe:")
    print(f"  {'조건':<22}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print()
    print(f"  {'─' * 85}")
    for (name, _), r in zip(strategies, results):
        ym = r['yearly']
        row = f"  {name:<22}"
        for y in years:
            if y in ym:
                row += f" {ym[y].get('Sharpe', 0):>7.2f}"
            else:
                row += f" {'─':>7}"
        print(row)

    # Year-by-year MDD
    print(f"\n  연도별 MDD:")
    print(f"  {'조건':<22}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print()
    print(f"  {'─' * 85}")
    for (name, _), r in zip(strategies, results):
        ym = r['yearly']
        row = f"  {name:<22}"
        for y in years:
            if y in ym:
                row += f" {ym[y].get('MDD', 0):>6.1%}"
            else:
                row += f" {'─':>7}"
        print(row)


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded, {len(ALL)} configs")

    t0 = time.time()
    results = run_set(ALL, prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    idx = 0
    def get_slice(s):
        nonlocal idx
        r = results[idx:idx+len(s)]
        idx += len(s)
        return r

    r_trail = get_slice(TRAILING)
    r_w = get_slice(W_COMPARE)
    r_s = get_slice(S_COMPARE)

    print(f"\n{'=' * 110}")
    print(f"  상세 분석 — G5 고정, 레이어별 디테일")
    print(f"{'=' * 110}")

    print_full("1. 리밸런싱 트리거: MTD vs Trailing Stop (G5 고정)", TRAILING, r_trail)
    print_full("2. 가중 비교: 균등 vs 역변동성 (G5 고정)", W_COMPARE, r_w)
    print_full("3. 선정 비교: 시총 vs 모멘텀 vs Sharpe (G5 고정)", S_COMPARE, r_s)

    # ── W 상세 분석 ────────────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print(f"  W 균등 vs 역변동성 — 연도별 승패")
    print(f"{'=' * 110}")
    ew = r_w[0]  # EW
    w6 = r_w[3]  # W6
    years = range(2018, 2026)
    ew_wins = 0
    w6_wins = 0
    for y in years:
        ew_y = ew['yearly'].get(y, {})
        w6_y = w6['yearly'].get(y, {})
        ew_s = ew_y.get('Sharpe', 0)
        w6_s = w6_y.get('Sharpe', 0)
        ew_c = ew_y.get('CAGR', 0)
        w6_c = w6_y.get('CAGR', 0)
        winner = "EW" if ew_s > w6_s else "W6" if w6_s > ew_s else "TIE"
        if winner == "EW": ew_wins += 1
        elif winner == "W6": w6_wins += 1
        print(f"  {y}: EW Sharpe {ew_s:>6.2f} CAGR {ew_c:>+6.1%} │ W6 Sharpe {w6_s:>6.2f} CAGR {w6_c:>+6.1%} → {winner}")
    print(f"\n  승리: EW {ew_wins}승, W6 {w6_wins}승")

    ew_m = ew['metrics']
    w6_m = w6['metrics']
    print(f"\n  전체: EW Sharpe {ew_m['Sharpe']:.3f} MDD {ew_m['MDD']:.1%} │ W6 Sharpe {w6_m['Sharpe']:.3f} MDD {w6_m['MDD']:.1%}")
    print(f"  차이: Sharpe {w6_m['Sharpe'] - ew_m['Sharpe']:+.3f}")

    # ── Universe bias check ────────────────────────────────────────
    print(f"\n{'=' * 110}")
    print(f"  유니버스 편향 분석")
    print(f"{'=' * 110}")

    # Check if universe is constructed from historical data (month-start)
    # or if there's look-ahead bias
    import json
    uni_path = os.path.join(os.path.dirname(__file__), '..', '..', 'backup_20260125', 'historical_universe.json')
    with open(uni_path) as f:
        raw_uni = json.load(f)

    # Show sample: what coins were in universe at different times
    sample_dates = ['2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01']
    print(f"\n  유니버스 구성 (월초 시총 Top 10):")
    for sd in sample_dates:
        if sd in raw_uni:
            coins = [c for c in raw_uni[sd][:10]
                     if c.replace('-USD','') not in {'USDT','USDC','BUSD','DAI','WBTC','FDUSD','USDE','USD1'}]
            print(f"  {sd}: {', '.join(coins[:7])}")

    # Check: does baseline selection just pick BTC, ETH, ... always?
    # Run a diagnostic backtest to see what gets picked
    print(f"\n  실제 선택 코인 분포 (baseline 시총순 vs S6 Sharpe):")
    for sel_name, sel_val in [('시총순', 'baseline'), ('S6:Sharpe', 'S6')]:
        p = B(risk='G5', selection=sel_val)
        r = run_backtest(prices, universe, p)
        # Count coin appearances
        coin_count = {}
        for entry in r.get('history', []):
            for t in entry.get('holdings', {}):
                coin_count[t] = coin_count.get(t, 0) + 1
        top_coins = sorted(coin_count.items(), key=lambda x: x[1], reverse=True)[:10]
        print(f"\n  {sel_name} — 보유 빈도 Top 10:")
        total_days = len(r.get('history', []))
        for coin, cnt in top_coins:
            pct = cnt / total_days * 100 if total_days > 0 else 0
            bar = '█' * int(pct / 2)
            print(f"    {coin:<12} {cnt:>5}일 ({pct:>4.1f}%) {bar}")

    print(f"\n  Completed all in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
