"""V17 snapshot iterative refine — pruned resume (iter3+ 부터).

iter1/2 결과로 확정된 열세 값을 하드 프룬:
  canary_sma: 150 만 유지 (iter2 top 79 에서 150 독점)
  canary_hyst: 0.005 drop (top 79 에서 4건만, 열세)
  def_mom_period: 50/90/130/300 drop (top 79 에서 0건) → 63, 252 유지
  snap_days: 전부 유지 (모두 top 에 등장)
  select/health: 전부 유지 (세 방법 모두 경쟁력 있음)

기존 iter_refine 로직 그대로. 단 초기 그리드를 pruned 로 시작.
출력: iter_3.csv, iter_4.csv, ..., top_peaks.csv
"""
from __future__ import annotations
import os, sys, time, math
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

# v17_snap_iter 에서 함수 재사용
from v17_snap_iter import (
    round3, round10, round_hyst, zoom_numeric,
    run_one, run_grid, OUT, UNIVERSE_B, DEF,
)


def main(max_iter: int = 5, start_iter: int = 3):
    # Iter3 시작 그리드 (iter2 분석 기반 prune)
    grid = {
        # iter_2 결과 반영: top 50 전부 snap_days=126. 126 중심 zoom.
        'snap_days':      [105, 114, 120, 126, 132, 138, 150],
        'canary_sma':     [150],                       # 전체 확정
        'canary_hyst':    [0.010, 0.014, 0.017, 0.020, 0.022, 0.024, 0.028],
        'canary_type':    ['sma'],
        'select':         ['sh3', 'mom126', 'mom3_sh3'],
        'def_mom_period': [63, 252],
        'health':         ['none', 'sma200', 'mom126'],
    }

    # Load prices once
    from stock_engine import load_prices, precompute, _init, ALL_TICKERS
    print('Loading prices...')
    prices = load_prices(sorted(set(ALL_TICKERS) | set(UNIVERSE_B) | set(DEF)),
                         start='2014-01-01')
    ind = precompute(prices)
    _init(prices, ind)

    all_results = []
    converged_axes = set()

    for it in range(start_iter, start_iter + max_iter):
        print(f'\n=== Iteration {it} ===')
        for k, v in grid.items():
            print(f'  {k}: {v}')
        n = 1
        for v in grid.values(): n *= len(v)
        print(f'  configs: {n}')

        t0 = time.time()
        df = run_grid(grid)
        df['iter'] = it
        print(f'  완료 ({time.time()-t0:.0f}s, {len(df)} rows)')

        if df is None or len(df) == 0:
            print('결과 없음, 중단.')
            break

        all_results.append(df)
        combined = pd.concat(all_results, ignore_index=True)
        combined.drop_duplicates(
            subset=['snap_days', 'canary_sma', 'canary_hyst', 'canary_type',
                    'select', 'def_mom_period', 'health'],
            keep='last', inplace=True,
        )

        out_csv = os.path.join(OUT, f'iter_{it}.csv')
        df.to_csv(out_csv, index=False)

        top_cal = df.nlargest(5, 'Cal')
        print('\n  Top 5 by Cal:')
        print(top_cal[['snap_days', 'canary_sma', 'canary_hyst', 'select',
                       'def_mom_period', 'health', 'CAGR', 'MDD', 'Cal']]
              .to_string(index=False))

        # === Zoom (기존 로직) ===
        peaks = {k: [] for k in grid.keys()}
        for _, row in top_cal.head(2).iterrows():
            for k in grid.keys():
                v = row[k]
                if v not in peaks[k]:
                    peaks[k].append(v)

        old_grid = {k: list(v) for k, v in grid.items()}

        if 'snap_days' not in converged_axes:
            new_vals = zoom_numeric(grid['snap_days'], peaks['snap_days'], round3)
            if set(new_vals) == set(grid['snap_days']):
                converged_axes.add('snap_days')
            else:
                grid['snap_days'] = new_vals

        # canary_sma 는 150 만 남김, 더 이상 확장 X
        converged_axes.add('canary_sma')

        if 'canary_hyst' not in converged_axes:
            new_vals = zoom_numeric(grid['canary_hyst'], peaks['canary_hyst'], round_hyst)
            # 0.005 이하는 하드 프룬
            new_vals = [v for v in new_vals if v >= 0.01]
            if set(new_vals) == set(grid['canary_hyst']):
                converged_axes.add('canary_hyst')
            else:
                grid['canary_hyst'] = new_vals

        if 'def_mom_period' not in converged_axes:
            new_vals = zoom_numeric(grid['def_mom_period'], peaks['def_mom_period'], round10)
            # 63/252 주변만 유지 (40~110 또는 200~310)
            new_vals = [v for v in new_vals if (40 <= v <= 110) or (200 <= v <= 310)]
            if set(new_vals) == set(grid['def_mom_period']):
                converged_axes.add('def_mom_period')
            else:
                grid['def_mom_period'] = new_vals

        # 카테고리 축 prune: top 3 에 들어온 것만 유지
        if it >= start_iter + 1:
            top_sel = set(df.nlargest(5, 'Cal')['select'])
            if len(top_sel) < len(grid['select']):
                grid['select'] = list(top_sel)
                print(f'  select 축소: {grid["select"]}')
            top_health = set(df.nlargest(5, 'Cal')['health'])
            if len(top_health) < len(grid['health']):
                grid['health'] = list(top_health)
                print(f'  health 축소: {grid["health"]}')

        all_numeric_converged = all(
            ax in converged_axes
            for ax in ['snap_days', 'canary_sma', 'canary_hyst', 'def_mom_period']
        )
        if all_numeric_converged:
            print('\n모든 수치 축 수렴 — 종료.')
            break

    # === 최종 top_peaks.csv 저장 ===
    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        combined.drop_duplicates(
            subset=['snap_days', 'canary_sma', 'canary_hyst', 'canary_type',
                    'select', 'def_mom_period', 'health'],
            keep='last', inplace=True,
        )
        # iter_1, iter_2 도 포함
        for prev in (1, 2):
            prev_path = os.path.join(OUT, f'iter_{prev}.csv')
            if os.path.exists(prev_path):
                prev_df = pd.read_csv(prev_path)
                prev_df['iter'] = prev
                combined = pd.concat([combined, prev_df], ignore_index=True)
        combined.drop_duplicates(
            subset=['snap_days', 'canary_sma', 'canary_hyst', 'canary_type',
                    'select', 'def_mom_period', 'health'],
            keep='first', inplace=True,
        )
        top_final = combined.nlargest(50, 'Cal')
        top_final.to_csv(os.path.join(OUT, 'top_peaks.csv'), index=False)
        combined.to_csv(os.path.join(OUT, 'all_iters.csv'), index=False)
        print(f'\n저장: top_peaks.csv ({len(top_final)} rows), all_iters.csv ({len(combined)} rows)')


if __name__ == '__main__':
    main()
