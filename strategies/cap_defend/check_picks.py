#!/usr/bin/env python3
"""Check what coins are actually picked and their market cap ranks."""

import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    Params, load_data, _close_to, calc_ret, get_vol, get_sma,
    get_universe_for_date, get_healthy_coins, select_coins, resolve_canary
)

def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded\n")

    params = Params(canary='K5', health='H5')
    btc = prices['BTC-USD']
    all_dates = btc.index[(btc.index >= '2018-01-01') & (btc.index <= '2025-06-30')]

    state = {
        'prev_canary': False, 'canary_off_days': 0,
        'health_fail_streak': {}, 'prev_picks': [],
        'scaled_months': 2, 'prev_month': None,
    }

    # Track picks per month
    monthly_picks = {}  # {YYYY-MM: picks_list}
    monthly_healthy = {}
    monthly_universe = {}
    pick_rank_dist = []  # list of ranks of picked coins

    prev_month = None
    for date in all_dates:
        cur_month = date.strftime('%Y-%m')
        imc = prev_month is not None and cur_month != prev_month

        canary_on = resolve_canary(prices, date, params, state)
        canary_flipped = canary_on != state['prev_canary']
        if canary_on and canary_flipped:
            state['scaled_months'] = 0

        state['is_month_change'] = imc
        state['canary_flipped'] = canary_flipped
        state['canary_on'] = canary_on

        if imc and canary_on:
            univ = get_universe_for_date(universe, date)
            state['current_universe'] = univ
            healthy = get_healthy_coins(prices, univ, date, params, state)
            state['healthy_count'] = len(healthy)
            picks = select_coins(healthy, prices, date, params, state) if healthy else []

            if picks:
                monthly_picks[cur_month] = picks
                monthly_healthy[cur_month] = healthy
                monthly_universe[cur_month] = univ

                # Track rank of each pick in the universe
                for p in picks:
                    if p in univ:
                        rank = univ.index(p) + 1
                        pick_rank_dist.append(rank)

        state['prev_canary'] = canary_on
        state['prev_picks'] = monthly_picks.get(cur_month, state['prev_picks'])
        prev_month = cur_month

    # ═══════════════════════════════════════════════════════════════
    # 1. Monthly picks log
    # ═══════════════════════════════════════════════════════════════
    print("=" * 100)
    print("  월별 편입 코인 (K5+H5, 시총순 Top5)")
    print("=" * 100)
    print(f"\n  {'월':>8} {'카나리':>4} {'건강':>3} {'편입 코인 (시총 순위)':<70}")
    print(f"  {'─' * 95}")

    for month in sorted(monthly_picks.keys()):
        picks = monthly_picks[month]
        univ = monthly_universe[month]
        healthy = monthly_healthy[month]
        pick_info = []
        for p in picks:
            rank = univ.index(p) + 1 if p in univ else '?'
            ticker = p.replace('-USD', '')
            pick_info.append(f"{ticker}({rank}위)")
        print(f"  {month:>8}   ON {len(healthy):>3}  {', '.join(pick_info)}")

    # ═══════════════════════════════════════════════════════════════
    # 2. Rank distribution
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  편입 코인 시총 순위 분포")
    print(f"{'=' * 100}")

    ranks = pd.Series(pick_rank_dist)
    print(f"\n  총 편입 횟수: {len(ranks)}")
    print(f"  평균 순위: {ranks.mean():.1f}")
    print(f"  중앙값: {ranks.median():.1f}")
    print(f"  최대 순위: {ranks.max()}")

    print(f"\n  {'순위 범위':<15} {'횟수':>6} {'비율':>7}")
    print(f"  {'─' * 30}")
    for lo, hi, label in [(1,3,'1~3위'), (4,5,'4~5위'), (6,10,'6~10위'),
                           (11,20,'11~20위'), (21,50,'21~50위')]:
        cnt = ((ranks >= lo) & (ranks <= hi)).sum()
        pct = cnt / len(ranks) * 100
        print(f"  {label:<15} {cnt:>6} {pct:>6.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # 3. Unique coins ever picked
    # ═══════════════════════════════════════════════════════════════
    all_picked = set()
    for picks in monthly_picks.values():
        all_picked.update(picks)

    print(f"\n  전체 기간 중 한 번이라도 편입된 코인: {len(all_picked)}종")

    # Count frequency
    freq = {}
    for picks in monthly_picks.values():
        for p in picks:
            freq[p] = freq.get(p, 0) + 1

    total_months = len(monthly_picks)
    freq_sorted = sorted(freq.items(), key=lambda x: -x[1])

    print(f"\n  {'코인':<12} {'편입횟수':>8} {'편입률':>7}")
    print(f"  {'─' * 30}")
    for ticker, cnt in freq_sorted:
        pct = cnt / total_months * 100
        print(f"  {ticker.replace('-USD',''):<12} {cnt:>8} {pct:>6.1f}%")

    # ═══════════════════════════════════════════════════════════════
    # 4. How often does rank 6+ get picked?
    # ═══════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  시총 6위 이하 코인이 뽑힌 월 목록")
    print(f"{'=' * 100}")

    deep_months = 0
    for month in sorted(monthly_picks.keys()):
        picks = monthly_picks[month]
        univ = monthly_universe[month]
        deep = [(p, univ.index(p)+1) for p in picks if p in univ and univ.index(p)+1 > 5]
        if deep:
            deep_months += 1
            deep_str = ', '.join(f"{p.replace('-USD','')}({r}위)" for p,r in deep)
            top5_miss = [univ[i].replace('-USD','') for i in range(min(5,len(univ)))
                        if univ[i] not in picks]
            miss_str = ', '.join(top5_miss) if top5_miss else '-'
            print(f"  {month}: {deep_str}  ← 대신 탈락: {miss_str}")

    print(f"\n  6위 이하 편입 발생: {deep_months}/{total_months}개월 ({deep_months/total_months*100:.1f}%)")


if __name__ == '__main__':
    main()
