#!/usr/bin/env python3
"""1. Why 2021 differs: canary signal timing comparison
   2. Quarterly leave-out cross-validation for overfitting check."""

import os, sys, time
import pandas as pd
import numpy as np
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    Params, load_data, init_pool, run_single, run_backtest,
    _close_to, get_sma, calc_ret
)

N_WORKERS = min(24, mp.cpu_count())

# Top candidates to compare
STRATEGIES = [
    ('K5 (80,50)+m21 2/3',         Params(canary='K5', health='H1', sma_period=80)),
    ('(100,50)+m21,m60 2/4',       Params(canary='K8', health='H1',
                                          vote_smas=(100,50), vote_moms=(21,60), vote_threshold=2)),
    ('(80,50)+m21,m60 2/4',        Params(canary='K8', health='H1',
                                          vote_smas=(80,50), vote_moms=(21,60), vote_threshold=2)),
    ('(120,80,50)+m21,m60 2/5',    Params(canary='K8', health='H1',
                                          vote_smas=(120,80,50), vote_moms=(21,60), vote_threshold=2)),
    ('(80,50)+m30 2/3',            Params(canary='K8', health='H1',
                                          vote_smas=(80,50), vote_moms=(30,), vote_threshold=2)),
]


def analyze_2021_signals(prices):
    """Compare canary ON/OFF signals in 2021 for top strategies."""
    print(f"\n{'=' * 100}")
    print(f"  2021년 카나리 신호 비교 — 왜 2/5가 +1171%이고 2/3이 +730%인가?")
    print(f"{'=' * 100}")

    btc_close = prices['BTC-USD']['Close']
    dates_2021 = btc_close.index[(btc_close.index >= '2021-01-01') &
                                  (btc_close.index <= '2021-12-31')]

    configs = [
        ('2/3 (80,50,m21)',  [80, 50], [21], 2),
        ('2/4 (100,50,m21,m60)', [100, 50], [21, 60], 2),
        ('2/5 (120,80,50,m21,m60)', [120, 80, 50], [21, 60], 2),
    ]

    # Track monthly ON% and key transition dates
    print(f"\n  {'월':>8}", end="")
    for name, _, _, _ in configs:
        print(f"  {name:>25}", end="")
    print(f"  {'BTC가격':>10}")
    print(f"  {'─' * 85}")

    prev_month = None
    monthly_on = {name: 0 for name, _, _, _ in configs}
    monthly_days = 0

    for date in dates_2021:
        cur_month = date.strftime('%Y-%m')
        close = btc_close.loc[:date]
        cur = float(close.iloc[-1])

        if prev_month and cur_month != prev_month:
            # Print monthly summary
            print(f"  {prev_month:>8}", end="")
            for name, _, _, _ in configs:
                pct = monthly_on[name] / monthly_days * 100
                status = "ON" if pct > 50 else "OFF"
                print(f"  {status:>4} ({pct:>4.0f}% ON)          ", end="")
            print(f"  {cur:>10,.0f}")
            monthly_on = {name: 0 for name, _, _, _ in configs}
            monthly_days = 0

        monthly_days += 1
        for name, smas, moms, thr in configs:
            votes = []
            for p in smas:
                sma = get_sma(close, p)
                if sma is not None and not np.isnan(sma):
                    votes.append(cur > float(sma))
                else:
                    votes.append(False)
            for p in moms:
                mom = calc_ret(close, p)
                votes.append(mom > 0)
            if sum(votes) >= thr:
                monthly_on[name] += 1

        prev_month = cur_month

    # Last month
    if monthly_days > 0:
        print(f"  {prev_month:>8}", end="")
        for name, _, _, _ in configs:
            pct = monthly_on[name] / monthly_days * 100
            status = "ON" if pct > 50 else "OFF"
            print(f"  {status:>4} ({pct:>4.0f}% ON)          ", end="")
        print(f"  {cur:>10,.0f}")

    # Also show 2021 key events
    print(f"\n  2021 주요 이벤트:")
    print(f"    1월: BTC 29k→33k  |  5월: BTC 58k→37k (대폭락)")
    print(f"    7월: BTC 33k→42k  |  11월: BTC 61k→57k (고점)")


def quarterly_cv(prices, universe):
    """Leave-one-quarter-out cross-validation."""
    print(f"\n{'=' * 130}")
    print(f"  분기 제거 교차검증 — 각 분기를 제외했을 때 Sharpe 변화")
    print(f"{'=' * 130}")

    # Run full backtest for each strategy to get daily portfolio values
    results = {}
    for name, params in STRATEGIES:
        p = Params(
            canary=params.canary, health=params.health,
            selection=params.selection, weighting=params.weighting,
            rebalancing=params.rebalancing, risk=params.risk,
            tx_cost=0.004, sma_period=params.sma_period,
            vote_smas=params.vote_smas, vote_moms=params.vote_moms,
            vote_threshold=params.vote_threshold,
        )
        r = run_backtest(prices, universe, p)
        results[name] = r

    # Define quarters
    quarters = []
    for year in range(2018, 2026):
        for q in range(4):
            m_start = q * 3 + 1
            m_end = m_start + 2
            q_start = f"{year}-{m_start:02d}-01"
            if m_end == 12:
                q_end = f"{year}-12-31"
            else:
                q_end = f"{year}-{m_end+1:02d}-01"
            label = f"{year}Q{q+1}"
            quarters.append((label, q_start, q_end))

    # For each strategy, compute quarterly returns and leave-out Sharpe
    print(f"\n  1) 분기별 수익률 (%)")
    print(f"  {'분기':<8}", end="")
    for name, _ in STRATEGIES:
        short = name[:20]
        print(f" {short:>20}", end="")
    print()
    print(f"  {'─' * 110}")

    quarterly_rets = {name: [] for name, _ in STRATEGIES}

    for q_label, q_start, q_end in quarters:
        row = f"  {q_label:<8}"
        for name, _ in STRATEGIES:
            pv = results[name]['pv']  # Date is index, 'Value' column
            q_data = pv[(pv.index >= q_start) & (pv.index < q_end)]
            if len(q_data) >= 2:
                q_ret = q_data['Value'].iloc[-1] / q_data['Value'].iloc[0] - 1
                quarterly_rets[name].append((q_label, q_ret))
                row += f" {q_ret:>+19.1%}"
            else:
                quarterly_rets[name].append((q_label, 0))
                row += f" {'─':>20}"
        print(row)

    # 2) For each strategy pair, count how many quarters the candidate beats baseline
    print(f"\n  2) 기준(K5 2/3) 대비 승률 (분기별)")
    print(f"  {'─' * 100}")
    base_rets = [r for _, r in quarterly_rets[STRATEGIES[0][0]]]
    for name, _ in STRATEGIES[1:]:
        cand_rets = [r for _, r in quarterly_rets[name]]
        wins = sum(1 for b, c in zip(base_rets, cand_rets) if c > b)
        losses = sum(1 for b, c in zip(base_rets, cand_rets) if c < b)
        ties = len(base_rets) - wins - losses
        total = len(base_rets)
        avg_diff = np.mean([c - b for b, c in zip(base_rets, cand_rets)]) * 100
        print(f"  {name:<35} 승 {wins:>2} / 패 {losses:>2} / 동 {ties:>2}"
              f"  ({wins/total*100:.0f}% 승률)  평균차 {avg_diff:>+.1f}%p/분기")

    # 3) Leave-one-quarter-out: compute Sharpe excluding each quarter
    print(f"\n  3) 분기 제거 시 Sharpe 변화 (민감도)")
    print(f"  {'─' * 130}")

    # Compute daily returns for each strategy
    daily_rets = {}
    for name, _ in STRATEGIES:
        pv = results[name]['pv']  # Date is already index
        daily_rets[name] = pv['Value'].pct_change().dropna()

    # Full Sharpe
    print(f"\n  {'전략':<35} {'Full':>6}", end="")
    # Show min/max quarter impact
    print(f"  {'제거시 최저':>10} {'제거시 최고':>10} {'편차':>6}")
    print(f"  {'─' * 75}")

    for name, _ in STRATEGIES:
        dr = daily_rets[name]
        full_sharpe = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0

        leave_out_sharpes = []
        for q_label, q_start, q_end in quarters:
            mask = ~((dr.index >= q_start) & (dr.index < q_end))
            dr_excl = dr[mask]
            if len(dr_excl) > 20 and dr_excl.std() > 0:
                lo_sharpe = dr_excl.mean() / dr_excl.std() * np.sqrt(252)
                leave_out_sharpes.append((q_label, lo_sharpe))

        if leave_out_sharpes:
            sharpes = [s for _, s in leave_out_sharpes]
            min_s = min(leave_out_sharpes, key=lambda x: x[1])
            max_s = max(leave_out_sharpes, key=lambda x: x[1])
            spread = max_s[1] - min_s[1]
            print(f"  {name:<35} {full_sharpe:>6.3f}"
                  f"  {min_s[1]:>6.3f}({min_s[0]})"
                  f"  {max_s[1]:>6.3f}({max_s[0]})"
                  f"  {spread:>6.3f}")

    # 4) Detailed: which quarters cause biggest Sharpe swing for top strategy
    print(f"\n  4) 1위 전략 분기별 Sharpe 영향도 (제거 시 Sharpe 변화)")
    print(f"  {'─' * 80}")
    top_name = STRATEGIES[3][0]  # 2/5
    base_name = STRATEGIES[0][0]  # K5 2/3
    dr_top = daily_rets[top_name]
    dr_base = daily_rets[base_name]
    full_top = dr_top.mean() / dr_top.std() * np.sqrt(252)
    full_base = dr_base.mean() / dr_base.std() * np.sqrt(252)

    impacts = []
    for q_label, q_start, q_end in quarters:
        mask = ~((dr_top.index >= q_start) & (dr_top.index < q_end))
        dr_t = dr_top[mask]
        dr_b = dr_base[mask]
        if len(dr_t) > 20 and dr_t.std() > 0 and dr_b.std() > 0:
            s_t = dr_t.mean() / dr_t.std() * np.sqrt(252)
            s_b = dr_b.mean() / dr_b.std() * np.sqrt(252)
            # How much does removing this quarter change the GAP between strategies?
            full_gap = full_top - full_base
            new_gap = s_t - s_b
            impacts.append((q_label, s_t, s_b, new_gap, new_gap - full_gap))

    print(f"  {'분기':<8}  {'2/5 Sharpe':>10}  {'2/3 Sharpe':>10}  {'차이':>8}  {'Δ vs Full':>10}")
    print(f"  {'─' * 55}")
    impacts.sort(key=lambda x: x[4])
    for q, st, sb, gap, delta in impacts:
        marker = " ←" if abs(delta) > 0.02 else ""
        print(f"  {q:<8}  {st:>10.3f}  {sb:>10.3f}  {gap:>+7.3f}  {delta:>+9.3f}{marker}")

    full_gap = full_top - full_base
    print(f"\n  Full gap: {full_gap:>+.3f}")
    print(f"  어떤 분기를 빼도 gap이 양수면 = 일관된 우위")
    print(f"  특정 분기에서만 gap이 양수면 = 그 분기에 의존 (과적합)")


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()
    analyze_2021_signals(prices)
    quarterly_cv(prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
