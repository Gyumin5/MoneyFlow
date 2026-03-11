#!/usr/bin/env python3
"""K6 level distribution analysis + K7 (3-stage) + K5 H1 vs H5 comparison."""

import os, sys, time
import pandas as pd
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    Params, load_data, init_pool, run_single,
    _close_to, get_sma
)

N_WORKERS = min(24, mp.cpu_count())


def analyze_levels(prices):
    """Show how often BTC is at each K6 level (0-5)."""
    btc_close = prices['BTC-USD']['Close']
    dates = btc_close.index[btc_close.index >= '2018-01-01']

    levels_60_100 = []  # K6: SMA 60,70,80,90,100
    levels_3stage = []  # K7: SMA 60,80,100

    for date in dates:
        close = btc_close.loc[:date]
        if len(close) < 100:
            continue
        cur = float(close.iloc[-1])

        # K6: 5 levels
        def above_sma(c, p):
            sma = get_sma(c, p)
            if sma is None or (isinstance(sma, float) and np.isnan(sma)):
                return False
            return cur > float(sma)

        k6_count = sum(1 for p in [60, 70, 80, 90, 100] if above_sma(close, p))
        levels_60_100.append({'date': date, 'level': k6_count})

        # K7: 3 levels
        k7_count = sum(1 for p in [60, 80, 100] if above_sma(close, p))
        levels_3stage.append({'date': date, 'level': k7_count})

    df6 = pd.DataFrame(levels_60_100)
    df7 = pd.DataFrame(levels_3stage)

    print(f"\n{'=' * 80}")
    print(f"  BTC SMA 레벨 분포 분석 (2018~)")
    print(f"{'=' * 80}")

    # K6 distribution
    print(f"\n  K6 (SMA 60/70/80/90/100) — 5단계:")
    total = len(df6)
    for lvl in range(6):
        cnt = (df6['level'] == lvl).sum()
        pct = cnt / total * 100
        bar = '█' * int(pct / 2)
        print(f"    Level {lvl} ({lvl*20:>3}% 투자): {cnt:>5}일 ({pct:>5.1f}%) {bar}")

    # K7 distribution
    print(f"\n  K7 (SMA 60/80/100) — 3단계:")
    for lvl in range(4):
        cnt = (df7['level'] == lvl).sum()
        pct = cnt / total * 100
        alloc = [0, 33, 66, 100][lvl]
        bar = '█' * int(pct / 2)
        print(f"    Level {lvl} ({alloc:>3}% 투자): {cnt:>5}일 ({pct:>5.1f}%) {bar}")

    # Year-by-year level distribution for K6
    print(f"\n  K6 연도별 평균 레벨 (= 평균 투자비율):")
    df6['year'] = pd.to_datetime(df6['date']).dt.year
    for year in range(2018, 2026):
        sub = df6[df6['year'] == year]
        if len(sub) == 0:
            continue
        avg_lvl = sub['level'].mean()
        avg_pct = avg_lvl * 20
        at5 = (sub['level'] == 5).sum() / len(sub) * 100
        at0 = (sub['level'] == 0).sum() / len(sub) * 100
        print(f"    {year}: 평균 {avg_lvl:.1f}/5 ({avg_pct:.0f}% 투자)"
              f"  |  5/5={at5:.0f}%  0/5={at0:.0f}%")

    # K7 year-by-year
    print(f"\n  K7 연도별 평균 레벨:")
    df7['year'] = pd.to_datetime(df7['date']).dt.year
    for year in range(2018, 2026):
        sub = df7[df7['year'] == year]
        if len(sub) == 0:
            continue
        avg_lvl = sub['level'].mean()
        avg_alloc = avg_lvl * 33.3
        at3 = (sub['level'] == 3).sum() / len(sub) * 100
        at0 = (sub['level'] == 0).sum() / len(sub) * 100
        print(f"    {year}: 평균 {avg_lvl:.1f}/3 ({avg_alloc:.0f}% 투자)"
              f"  |  3/3={at3:.0f}%  0/3={at0:.0f}%")

    # Level transitions (K6)
    changes6 = (df6['level'].diff() != 0).sum()
    changes7 = (df7['level'].diff() != 0).sum()
    print(f"\n  레벨 변경 횟수:")
    print(f"    K6 (5단계): {changes6}회 ({changes6/total*365:.0f}회/년)")
    print(f"    K7 (3단계): {changes7}회 ({changes7/total*365:.0f}회/년)")


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    # ── Part 0: Level analysis ────────────────────────────────────
    analyze_levels(prices)

    # ── Part 1: K5 H1 vs H5 (clear comparison) ───────────────────
    K5_COMPARE = [
        ('K5+H1 sma80',   Params(canary='K5', health='H1', sma_period=80)),
        ('K5+H5 sma80',   Params(canary='K5', health='H5', sma_period=80)),
        ('K5+H1 sma100',  Params(canary='K5', health='H1', sma_period=100)),
        ('K5+H5 sma100',  Params(canary='K5', health='H5', sma_period=100)),
        ('K5+H1 sma150',  Params(canary='K5', health='H1', sma_period=150)),
        ('K5+H5 sma150',  Params(canary='K5', health='H5', sma_period=150)),
    ]

    # ── Part 2: K7 (3-stage) ─────────────────────────────────────
    K7_TEST = [
        ('K5+H1 sma80 (기준)',   Params(canary='K5', health='H1', sma_period=80)),
        ('K5+H5 sma150 (구기준)', Params(canary='K5', health='H5', sma_period=150)),
        ('K6+H1 (5단계)',         Params(canary='K6', health='H1')),
        ('K6+H5 (5단계)',         Params(canary='K6', health='H5')),
        ('K7+H1 (3단계)',         Params(canary='K7', health='H1')),
        ('K7+H5 (3단계)',         Params(canary='K7', health='H5')),
    ]

    t0 = time.time()
    rC0 = run_set(K5_COMPARE, prices, universe, tx=0.0)
    rC4 = run_set(K5_COMPARE, prices, universe, tx=0.004)
    r70 = run_set(K7_TEST, prices, universe, tx=0.0)
    r74 = run_set(K7_TEST, prices, universe, tx=0.004)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # Print K5 H1 vs H5
    print(f"\n{'=' * 130}")
    print(f"  K5: H1 vs H5 직접 비교 (tx=0 / tx=0.4%)")
    print(f"{'=' * 130}")
    print(f"\n  {'전략':<22}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
          f" │{'ΔCAGR':>7} {'리밸':>4}")
    print(f"  {'':>22}"
          f" │{'── tx=0 ──':^22}"
          f" │{'── tx=0.4% ──':^34}"
          f" │{'비용':>7}")
    print(f"  {'─' * 100}")
    for (name, _), r0, r4 in zip(K5_COMPARE, rC0, rC4):
        m0, m4 = r0['metrics'], r4['metrics']
        drag = m0['CAGR'] - m4['CAGR']
        print(f"  {name:<22}"
              f" │{m0['Sharpe']:>7.3f} {m0['CAGR']:>+6.1%} {m0['MDD']:>6.1%}"
              f" │{m4['Sharpe']:>7.3f} {m4['CAGR']:>+6.1%} {m4['MDD']:>6.1%} {m4['Final']:>10,.0f}"
              f" │{drag:>+6.1%} {r4['rebal_count']:>4}")

    # Print K6 vs K7
    print(f"\n{'=' * 130}")
    print(f"  K6(5단계) vs K7(3단계) 점진적 진입 비교")
    print(f"{'=' * 130}")
    print(f"\n  {'전략':<25}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
          f" │{'ΔCAGR':>7} {'리밸':>4}")
    print(f"  {'':>25}"
          f" │{'── tx=0 ──':^22}"
          f" │{'── tx=0.4% ──':^34}"
          f" │{'비용':>7}")
    print(f"  {'─' * 105}")
    for (name, _), r0, r4 in zip(K7_TEST, r70, r74):
        m0, m4 = r0['metrics'], r4['metrics']
        drag = m0['CAGR'] - m4['CAGR']
        print(f"  {name:<25}"
              f" │{m0['Sharpe']:>7.3f} {m0['CAGR']:>+6.1%} {m0['MDD']:>6.1%}"
              f" │{m4['Sharpe']:>7.3f} {m4['CAGR']:>+6.1%} {m4['MDD']:>6.1%} {m4['Final']:>10,.0f}"
              f" │{drag:>+6.1%} {r4['rebal_count']:>4}")

    # Year-by-year for K7
    years = range(2018, 2026)
    print(f"\n  연도별 CAGR (tx=0.4%)")
    print(f"  {'─' * 110}")
    print(f"  {'전략':<25}", end="")
    for y in years:
        print(f" {y:>8}", end="")
    print(f" {'전체':>9}")
    print(f"  {'─' * 100}")
    for (name, _), r in zip(K7_TEST, r74):
        ym = r['yearly']
        m = r['metrics']
        row = f"  {name:<25}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+7.1%}"
            else:
                row += f" {'─':>8}"
        row += f" {m['CAGR']:>+8.1%}"
        print(row)

    # MDD
    print(f"\n  연도별 MDD (tx=0.4%)")
    print(f"  {'─' * 100}")
    for (name, _), r in zip(K7_TEST, r74):
        ym = r['yearly']
        m = r['metrics']
        row = f"  {name:<25}"
        for y in years:
            if y in ym:
                row += f" {ym[y].get('MDD', 0):>7.1%}"
            else:
                row += f" {'─':>8}"
        row += f" {m['MDD']:>8.1%}"
        print(row)


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


if __name__ == '__main__':
    main()
