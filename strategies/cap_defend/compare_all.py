"""
Cap Defend V12 — 종합 파라미터 비교 분석
========================================
모든 전략 변형을 다양한 시작 시점에서 비교

Usage:
    python3 strategies/cap_defend/compare_all.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..'))

import numpy as np
import pandas as pd
from dataclasses import replace
from strategies.cap_defend.backtest import (
    Params, load_historical_universe, collect_all_tickers,
    load_all_data, run_backtest, calc_metrics, calc_benchmark,
)


# ---------------------------------------------------------------------------
# 비교할 전략 변형 정의 (stock/coin 60:40은 고정)
# ---------------------------------------------------------------------------

VARIANTS = {
    # --- Baseline ---
    'Baseline (현재)': {},

    # --- Vol Cap 변형 ---
    'VolCap 4%': {'vol_cap': 0.04},
    'VolCap 6%': {'vol_cap': 0.06},
    'VolCap 8%': {'vol_cap': 0.08},
    'VolCap 15%': {'vol_cap': 0.15},
    'VolCap 20% (거의무제한)': {'vol_cap': 0.20},

    # --- Coin Canary MA ---
    'CoinCanary MA20': {'coin_canary_ma': 20},
    'CoinCanary MA30': {'coin_canary_ma': 30},
    'CoinCanary MA70': {'coin_canary_ma': 70},
    'CoinCanary MA100': {'coin_canary_ma': 100},

    # --- N Coin Picks ---
    'Top3 Coins': {'n_coin_picks': 3},
    'Top7 Coins': {'n_coin_picks': 7},
    'Top10 Coins': {'n_coin_picks': 10},

    # --- Turnover Threshold ---
    'Turnover 15%': {'turnover_threshold': 0.15},
    'Turnover 50%': {'turnover_threshold': 0.50},

    # --- Health Filter ---
    'Health MA15': {'health_ma': 15},
    'Health MA20': {'health_ma': 20},
    'Health MA50': {'health_ma': 50},
    'Health Mom10': {'health_mom': 10},
    'Health Mom42': {'health_mom': 42},

    # --- Stock Canary MA ---
    'StockCanary MA100': {'stock_canary_ma': 100},
    'StockCanary MA150': {'stock_canary_ma': 150},
    'StockCanary MA250': {'stock_canary_ma': 250},

    # --- 복합 변형: 유망 조합 ---
    'Aggressive (VolCap15+Top7+MA30)': {'vol_cap': 0.15, 'n_coin_picks': 7, 'coin_canary_ma': 30},
    'Conservative (VolCap6+Top3+MA70)': {'vol_cap': 0.06, 'n_coin_picks': 3, 'coin_canary_ma': 70},
    'LowTurnover (TO50+VolCap8)': {'turnover_threshold': 0.50, 'vol_cap': 0.08},
}

START_DATES = ['2019-01-01', '2020-01-01', '2021-01-01', '2022-01-01', '2023-01-01']
END_DATE = '2025-12-31'


def main():
    base_params = Params(end_date=END_DATE)

    # Load data once (use earliest start date with buffer)
    hist_universe = load_historical_universe(base_params.universe_file)
    all_tickers = collect_all_tickers(base_params, hist_universe)

    print(f"Loading data for {len(all_tickers)} tickers...")
    # Use earliest start to load data once
    load_params = replace(base_params, start_date='2019-01-01')
    data = load_all_data(all_tickers, load_params)

    total_runs = len(VARIANTS) * len(START_DATES)
    print(f"\nRunning {len(VARIANTS)} variants x {len(START_DATES)} periods = {total_runs} backtests...\n")

    # Collect results
    rows = []
    run_count = 0

    for name, overrides in VARIANTS.items():
        for start in START_DATES:
            run_count += 1
            params = replace(base_params, start_date=start, end_date=END_DATE)
            for k, v in overrides.items():
                setattr(params, k, v)

            try:
                res = run_backtest(data, params, hist_universe, label=name)
                m = calc_metrics(res['Value'])
                rebal = res.attrs.get('rebal_count', 0)

                rows.append({
                    'Variant': name,
                    'Start': start,
                    'Period': f"{start[:4]}-25",
                    'Final': m['final'],
                    'CAGR': m['cagr'],
                    'MDD': m['mdd'],
                    'Sharpe': m['sharpe'],
                    'Sortino': m['sortino'],
                    'WinRate': m['win_rate'],
                    'Rebals': rebal,
                })
            except Exception as e:
                rows.append({
                    'Variant': name, 'Start': start, 'Period': f"{start[:4]}-25",
                    'Final': 0, 'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'WinRate': 0, 'Rebals': 0,
                })

            if run_count % 10 == 0:
                print(f"  Progress: {run_count}/{total_runs}")

    df = pd.DataFrame(rows)

    # =====================================================================
    # 출력 1: 시작시점별 전략 비교 (각 시작시점 테이블)
    # =====================================================================
    for start in START_DATES:
        period = df[df['Start'] == start].copy()
        period = period.sort_values('Sharpe', ascending=False)

        print(f"\n{'=' * 100}")
        print(f"  START: {start} ~ {END_DATE}")
        print(f"{'=' * 100}")
        print(f"{'Rank':>4} {'Variant':<32} {'Final($)':>11} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'Sortino':>8} {'Rebals':>7}")
        print(f"{'-' * 100}")

        for i, (_, r) in enumerate(period.iterrows(), 1):
            marker = ' ***' if i <= 3 else ''
            print(f"{i:>4} {r['Variant']:<32} {r['Final']:>11,.0f} {r['CAGR']:>7.1%} {r['MDD']:>7.1%} "
                  f"{r['Sharpe']:>7.2f} {r['Sortino']:>8.2f} {r['Rebals']:>7}{marker}")

        # Benchmarks
        bench_spy = calc_benchmark(data, 'SPY', start, END_DATE)
        bench_btc = calc_benchmark(data, 'BTC-USD', start, END_DATE)
        print(f"{'-' * 100}")
        print(f"     {'SPY B&H':<32} {'':>11} {bench_spy['cagr']:>7.1%} {bench_spy['mdd']:>7.1%} {bench_spy['sharpe']:>7.2f}")
        print(f"     {'BTC B&H':<32} {'':>11} {bench_btc['cagr']:>7.1%} {bench_btc['mdd']:>7.1%} {bench_btc['sharpe']:>7.2f}")

    # =====================================================================
    # 출력 2: 전략별 안정성 (모든 시작시점에서의 평균)
    # =====================================================================
    print(f"\n\n{'=' * 110}")
    print(f"  STABILITY RANKING — 모든 시작시점 평균 (안정적으로 좋은 전략 = 높은 순위)")
    print(f"{'=' * 110}")

    stability = df.groupby('Variant').agg({
        'CAGR': 'mean',
        'MDD': 'mean',
        'Sharpe': 'mean',
        'Sortino': 'mean',
        'Rebals': 'mean',
    }).reset_index()

    # Composite score: Sharpe * 0.5 + (1+MDD)*2 + CAGR*1
    # Higher is better for all components
    stability['Score'] = stability['Sharpe'] * 0.5 + (1 + stability['MDD']) * 2 + stability['CAGR']
    stability = stability.sort_values('Score', ascending=False)

    print(f"{'Rank':>4} {'Variant':<32} {'Avg CAGR':>9} {'Avg MDD':>9} {'Avg Sharpe':>11} {'Avg Sortino':>12} {'Score':>7}")
    print(f"{'-' * 110}")

    for i, (_, r) in enumerate(stability.iterrows(), 1):
        marker = ' <-- TOP' if i <= 5 else ''
        print(f"{i:>4} {r['Variant']:<32} {r['CAGR']:>8.1%} {r['MDD']:>8.1%} "
              f"{r['Sharpe']:>11.2f} {r['Sortino']:>12.2f} {r['Score']:>7.2f}{marker}")

    # =====================================================================
    # 출력 3: 파라미터 그룹별 인사이트
    # =====================================================================
    print(f"\n\n{'=' * 110}")
    print(f"  PARAMETER INSIGHTS — 파라미터별 영향 분석")
    print(f"{'=' * 110}")

    groups = {
        'Vol Cap': ['VolCap 4%', 'VolCap 6%', 'VolCap 8%', 'Baseline (현재)', 'VolCap 15%', 'VolCap 20% (거의무제한)'],
        'Coin Canary MA': ['CoinCanary MA20', 'CoinCanary MA30', 'Baseline (현재)', 'CoinCanary MA70', 'CoinCanary MA100'],
        'N Coins': ['Top3 Coins', 'Baseline (현재)', 'Top7 Coins', 'Top10 Coins'],
        'Turnover Threshold': ['Turnover 15%', 'Baseline (현재)', 'Turnover 50%'],
        'Health MA': ['Health MA15', 'Health MA20', 'Baseline (현재)', 'Health MA50'],
        'Health Momentum': ['Health Mom10', 'Baseline (현재)', 'Health Mom42'],
        'Stock Canary MA': ['StockCanary MA100', 'StockCanary MA150', 'Baseline (현재)', 'StockCanary MA250'],
        'Combo': ['Conservative (VolCap6+Top3+MA70)', 'Baseline (현재)', 'Aggressive (VolCap15+Top7+MA30)', 'LowTurnover (TO50+VolCap8)'],
    }

    for group_name, variant_names in groups.items():
        print(f"\n  [{group_name}]")
        print(f"  {'Variant':<36} {'Avg CAGR':>9} {'Avg MDD':>9} {'Avg Sharpe':>11}")
        print(f"  {'-' * 70}")
        for vname in variant_names:
            row = stability[stability['Variant'] == vname]
            if row.empty:
                continue
            r = row.iloc[0]
            marker = '  <-- baseline' if 'Baseline' in vname else ''
            print(f"  {r['Variant']:<36} {r['CAGR']:>8.1%} {r['MDD']:>8.1%} {r['Sharpe']:>11.2f}{marker}")

    print(f"\n{'=' * 110}")
    print("Done.")


if __name__ == '__main__':
    main()
