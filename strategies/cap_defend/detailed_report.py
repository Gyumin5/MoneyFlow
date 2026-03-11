#!/usr/bin/env python3
"""Top strategy detailed report — full period + yearly breakdown."""

import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, run_backtest, calc_metrics, calc_yearly_metrics

# Top strategies to analyze
STRATEGIES = [
    ('BASELINE',   Params(), 'BTC>SMA150 카나리아 + SMA30/Mom21/Vol5% 헬스 + 시총Top5 균등배분 + 월간리밸'),
    ('K5',         Params(canary='K5'),
     '2-of-3 투표 카나리아: BTC>SMA150, BTC>SMA50, BTC mom90>0 중 2개↑ 충족 시 Risk-On'),
    ('K4',         Params(canary='K4'),
     'Dual Canary: BTC>SMA150 AND ETH>SMA150 동시 충족 시에만 Risk-On'),
    ('K5+W1',      Params(canary='K5', weighting='W1'),
     'K5 + Rank-Decay 가중: 1위 30%, 2위 25%, 3위 20%, 4위 15%, 5위 10%'),
    ('K5+H5',      Params(canary='K5', health='H5'),
     'K5 + Vol Acceleration Block: vol30 > vol90×1.5이면 탈락'),
    ('K5+S5',      Params(canary='K5', selection='S5'),
     'K5 + Incumbent Carry: 기존 보유 코인에 +2 rank 보너스 (턴오버 감소)'),
    ('K5+W3',      Params(canary='K5', weighting='W3'),
     'K5 + Momentum Tilt: 기본 20% + mom21 상위2개 +5%, 하위2개 -5%'),
    ('K5+G4',      Params(canary='K5', risk='G4'),
     'K5 + Rank Floor: 시총 50위 밖으로 밀린 보유 코인 즉시 매도'),
    ('K5+G5',      Params(canary='K5', risk='G5'),
     'K5 + Crash Breaker: BTC 일간 -10% 시 다음 3일 현금 유지'),
    ('K5+R2',      Params(canary='K5', rebalancing='R2'),
     'K5 + Catastrophic Exit: 포트폴리오 MTD(월중) -15% 시 전량 현금화'),
    ('K5+G2',      Params(canary='K5', risk='G2'),
     'K5 + Vol Target: 포트폴리오 30일 변동성 > 연80% 시 비중 축소'),
    ('K4+W1',      Params(canary='K4', weighting='W1'),
     'Dual Canary(BTC+ETH) + Rank-Decay 가중'),
]

# BTC B&H benchmark
def btc_bh_metrics(prices):
    btc = prices['BTC-USD']
    mask = (btc.index >= '2018-01-01') & (btc.index <= '2025-06-30')
    btc_period = btc.loc[mask, 'Close']
    v0 = btc_period.iloc[0]
    pv = pd.DataFrame({'Value': btc_period / v0 * 10000}, index=btc_period.index)
    return calc_metrics(pv), calc_yearly_metrics(pv)


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded\n")

    # Run all strategies
    results = []
    for label, params, desc in STRATEGIES:
        r = run_backtest(prices, universe, params)
        ym = r['yearly']
        results.append((label, params, desc, r, ym))

    btc_m, btc_ym = btc_bh_metrics(prices)

    # ═══════════════════════════════════════════════════════════════
    # 1. FULL PERIOD SUMMARY
    # ═══════════════════════════════════════════════════════════════
    print("=" * 120)
    print("  전구간 성과 요약 (2018-01-01 ~ 2025-06-30, 초기자본 $10,000, tx=0.4%)")
    print("=" * 120)
    print()
    print(f"  {'전략':<14} {'Sharpe':>8} {'Sortino':>8} {'CAGR':>8} {'MDD':>8}"
          f" {'Final($)':>12} {'Rebals':>7} {'Win%':>6} {'AvgDD':>7} {'설명'}")
    print(f"  {'─' * 115}")

    for label, params, desc, r, ym in results:
        m = r['metrics']
        pv = r['pv']
        dr = pv['Value'].pct_change().dropna()
        win_rate = (dr > 0).sum() / len(dr) * 100
        # Average drawdown
        peak = pv['Value'].cummax()
        dd = pv['Value'] / peak - 1
        avg_dd = dd.mean()

        print(f"  {label:<14} {m['Sharpe']:>8.3f} {m['Sortino']:>8.3f} {m['CAGR']:>+7.1%}"
              f" {m['MDD']:>7.1%} {m['Final']:>11,.0f} {r['rebal_count']:>7}"
              f" {win_rate:>5.1f} {avg_dd:>6.1%}  {desc[:60]}")

    # BTC B&H row
    print(f"  {'BTC B&H':<14} {btc_m['Sharpe']:>8.3f} {btc_m['Sortino']:>8.3f}"
          f" {btc_m['CAGR']:>+7.1%} {btc_m['MDD']:>7.1%} {btc_m['Final']:>11,.0f}"
          f" {'─':>7} {'─':>6} {'─':>7}  비트코인 장기 보유")

    # ═══════════════════════════════════════════════════════════════
    # 2. DETAILED METRICS TOP 5
    # ═══════════════════════════════════════════════════════════════
    top5 = sorted(results, key=lambda x: -x[3]['metrics']['Sharpe'])[:5]

    print()
    print("=" * 120)
    print("  상위 5개 전략 상세 지표")
    print("=" * 120)

    for label, params, desc, r, ym in top5:
        m = r['metrics']
        pv = r['pv']
        dr = pv['Value'].pct_change().dropna()
        peak = pv['Value'].cummax()
        dd = pv['Value'] / peak - 1
        win_rate = (dr > 0).sum() / len(dr) * 100

        # Calmar ratio = CAGR / |MDD|
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0

        # Max consecutive loss days
        losing = (dr < 0).astype(int)
        losing_streaks = losing.groupby((losing != losing.shift()).cumsum())
        max_loss_streak = max((g.sum() for _, g in losing_streaks), default=0)

        # Monthly returns
        monthly = pv['Value'].resample('M').last().pct_change().dropna()
        monthly_win = (monthly > 0).sum() / len(monthly) * 100
        best_month = monthly.max()
        worst_month = monthly.min()

        # Recovery: time from MDD bottom to new high
        mdd_date = dd.idxmin()
        recovery_mask = pv.loc[mdd_date:, 'Value'] >= peak.loc[mdd_date]
        if recovery_mask.any():
            recovery_date = recovery_mask[recovery_mask].index[0]
            recovery_days = (recovery_date - mdd_date).days
        else:
            recovery_days = '미회복'

        # Downside deviation
        down_dev = dr[dr < 0].std() * np.sqrt(365)

        # Profit factor = sum(gains) / abs(sum(losses))
        gains = dr[dr > 0].sum()
        losses = abs(dr[dr < 0].sum())
        profit_factor = gains / losses if losses > 0 else float('inf')

        print(f"\n  ┌── {label} ──────────────────────────────────────────────────")
        print(f"  │ {desc}")
        print(f"  ├─────────────────────────────────────────────────────────────")
        print(f"  │ 수익성                      │ 위험                          │ 효율성")
        print(f"  │  CAGR:      {m['CAGR']:>+8.1%}        │  MDD:       {m['MDD']:>8.1%}        │  Sharpe:     {m['Sharpe']:>6.3f}")
        print(f"  │  Final:    ${m['Final']:>10,.0f}      │  Avg DD:    {dd.mean():>8.1%}        │  Sortino:    {m['Sortino']:>6.3f}")
        print(f"  │  Total Return: {m['Final']/10000-1:>+8.0%}     │  Downside σ:{down_dev:>8.1%}        │  Calmar:     {calmar:>6.3f}")
        print(f"  │  Best Month: {best_month:>+8.1%}       │  Worst Month:{worst_month:>+7.1%}        │  Profit Factor:{profit_factor:>5.2f}")
        print(f"  │  월간 승률:    {monthly_win:>5.1f}%       │  Max Loss Streak:{max_loss_streak:>3.0f}일     │  리밸런싱:   {r['rebal_count']:>5}회")
        print(f"  │  일간 승률:    {win_rate:>5.1f}%       │  MDD 발생일: {mdd_date.strftime('%Y-%m-%d')}   │  회복기간:   {recovery_days}")
        print(f"  └─────────────────────────────────────────────────────────────")

    # ═══════════════════════════════════════════════════════════════
    # 3. YEAR-BY-YEAR BREAKDOWN
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("  연도별 CAGR (전구간)")
    print("=" * 120)

    years = range(2018, 2026)
    header = f"  {'전략':<14}"
    for y in years:
        header += f" {y:>8}"
    header += f" {'전체':>9}"
    print(header)
    print(f"  {'─' * 105}")

    for label, params, desc, r, ym in results:
        m = r['metrics']
        row = f"  {label:<14}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+7.1%}"
            else:
                row += f" {'─':>8}"
        row += f" {m['CAGR']:>+8.1%}"
        print(row)

    # BTC B&H
    row = f"  {'BTC B&H':<14}"
    for y in years:
        if y in btc_ym:
            row += f" {btc_ym[y]['CAGR']:>+7.1%}"
        else:
            row += f" {'─':>8}"
    row += f" {btc_m['CAGR']:>+8.1%}"
    print(row)

    # ═══════════════════════════════════════════════════════════════
    # 4. YEAR-BY-YEAR MDD
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("  연도별 MDD")
    print("=" * 120)

    print(header.replace('CAGR', 'MDD'))
    print(f"  {'─' * 105}")

    for label, params, desc, r, ym in results:
        m = r['metrics']
        row = f"  {label:<14}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['MDD']:>7.1%}"
            else:
                row += f" {'─':>8}"
        row += f" {m['MDD']:>8.1%}"
        print(row)

    # BTC B&H
    row = f"  {'BTC B&H':<14}"
    for y in years:
        if y in btc_ym:
            row += f" {btc_ym[y]['MDD']:>7.1%}"
        else:
            row += f" {'─':>8}"
    row += f" {btc_m['MDD']:>8.1%}"
    print(row)

    # ═══════════════════════════════════════════════════════════════
    # 5. YEAR-BY-YEAR SHARPE
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("  연도별 Sharpe Ratio")
    print("=" * 120)

    print(f"  {'전략':<14}", end="")
    for y in years:
        print(f" {y:>8}", end="")
    print(f" {'전체':>9}")
    print(f"  {'─' * 105}")

    for label, params, desc, r, ym in results:
        m = r['metrics']
        row = f"  {label:<14}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['Sharpe']:>8.3f}"
            else:
                row += f" {'─':>8}"
        row += f" {m['Sharpe']:>9.3f}"
        print(row)

    # BTC B&H
    row = f"  {'BTC B&H':<14}"
    for y in years:
        if y in btc_ym:
            row += f" {btc_ym[y]['Sharpe']:>8.3f}"
        else:
            row += f" {'─':>8}"
    row += f" {btc_m['Sharpe']:>9.3f}"
    print(row)

    # ═══════════════════════════════════════════════════════════════
    # 6. COMPARATIVE ANALYSIS
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 120)
    print("  베이스라인 대비 변화량 (Delta)")
    print("=" * 120)

    base_m = results[0][3]['metrics']

    print(f"\n  {'전략':<14} {'ΔSharpe':>9} {'ΔSortino':>9} {'ΔCAGR':>8} {'ΔMDD':>8}"
          f" {'ΔFinal($)':>12} {'ΔRebals':>8}")
    print(f"  {'─' * 65}")

    for label, params, desc, r, ym in results[1:]:
        m = r['metrics']
        ds = m['Sharpe'] - base_m['Sharpe']
        dso = m['Sortino'] - base_m['Sortino']
        dc = m['CAGR'] - base_m['CAGR']
        dm = m['MDD'] - base_m['MDD']
        df = m['Final'] - base_m['Final']
        dr_cnt = r['rebal_count'] - results[0][3]['rebal_count']

        print(f"  {label:<14} {ds:>+8.3f} {dso:>+8.3f} {dc:>+7.1%} {dm:>+7.1%}"
              f" {df:>+11,.0f} {dr_cnt:>+7}")


if __name__ == '__main__':
    main()
