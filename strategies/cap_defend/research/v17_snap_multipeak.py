"""V17 snapshot multi-peak grid — iter greedy 보완.

iter_2 에서 snap_days 별 best Cal:
  126: 0.896, 180: 0.641, 90: 0.623, 51: 0.597, 30: 0.592, 24: 0.524

greedy iter (top2 공통 값만 zoom) 는 126 단일 피크만 남김.
본 스크립트는 5개 snap 버킷 각각 독립 zoom 으로 다피크 탐색.

버킷: snap_days ∈ {30, 60, 90, 126, 180}
축: canary_hyst, select, def_mom_period, health (canary_sma=150, canary_type=sma 고정)
"""
from __future__ import annotations
import os, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from v17_snap_iter import run_grid, OUT, UNIVERSE_B, DEF


def main():
    # Round 1: 5 snap 버킷 × 세 축 공동 그리드
    grid = {
        'snap_days':      [30, 60, 90, 126, 180],
        'canary_sma':     [150],
        'canary_hyst':    [0.010, 0.015, 0.020, 0.025],
        'canary_type':    ['sma'],
        'select':         ['sh3', 'mom126', 'mom3_sh3'],
        'def_mom_period': [63, 252],
        'health':         ['none', 'sma200', 'mom126'],
    }

    from stock_engine import load_prices, precompute, _init, ALL_TICKERS
    print('Loading prices...')
    prices = load_prices(sorted(set(ALL_TICKERS) | set(UNIVERSE_B) | set(DEF)),
                         start='2014-01-01')
    ind = precompute(prices)
    _init(prices, ind)

    print('=== Round1: 5 snap 버킷 공동 grid ===')
    n = 1
    for v in grid.values(): n *= len(v)
    print(f'  configs: {n}')
    for k, v in grid.items():
        print(f'  {k}: {v}')

    t0 = time.time()
    df = run_grid(grid)
    print(f'  완료 ({time.time()-t0:.0f}s, {len(df)} rows)')
    out_csv = os.path.join(OUT, 'mp_round1.csv')
    df.to_csv(out_csv, index=False)
    print(f'  저장: {out_csv}')

    # 버킷별 best
    print('\n=== 버킷별 best Cal ===')
    top_per_bucket = df.sort_values('Cal', ascending=False).groupby('snap_days').head(3)
    for snap in sorted(df['snap_days'].unique()):
        sub = df[df['snap_days'] == snap].sort_values('Cal', ascending=False).head(3)
        print(f'\nsnap={snap}:')
        print(sub[['canary_hyst','select','def_mom_period','health','CAGR','MDD','Sharpe','Cal']].to_string(index=False))

    # Round 2: 각 버킷 best 주변 hyst zoom + top select/health 유지
    print('\n=== Round2: 버킷별 zoom (hyst ±0.003, step 0.002) ===')
    all_r2 = []
    for snap in sorted(df['snap_days'].unique()):
        sub = df[df['snap_days'] == snap]
        best = sub.loc[sub['Cal'].idxmax()]
        h = best['canary_hyst']
        zoom_hyst = sorted(set([round(h + d, 4) for d in (-0.005, -0.003, -0.001, 0, 0.001, 0.003, 0.005)]))
        zoom_hyst = [x for x in zoom_hyst if 0.001 <= x <= 0.050]
        # top 2 select × top 2 health
        top_sel = sub.sort_values('Cal', ascending=False).head(6)['select'].value_counts().head(2).index.tolist()
        top_health = sub.sort_values('Cal', ascending=False).head(6)['health'].value_counts().head(2).index.tolist()
        grid_r2 = {
            'snap_days':      [int(snap)],
            'canary_sma':     [150],
            'canary_hyst':    zoom_hyst,
            'canary_type':    ['sma'],
            'select':         top_sel,
            'def_mom_period': [int(best['def_mom_period'])],
            'health':         top_health,
        }
        n2 = 1
        for v in grid_r2.values(): n2 *= len(v)
        print(f'  snap={snap} configs={n2} hyst={zoom_hyst} sel={top_sel} health={top_health}')
        df2 = run_grid(grid_r2)
        df2['bucket_snap'] = snap
        all_r2.append(df2)

    r2 = pd.concat(all_r2, ignore_index=True)
    out_r2 = os.path.join(OUT, 'mp_round2.csv')
    r2.to_csv(out_r2, index=False)
    print(f'\n저장: {out_r2}')

    # 최종 합본
    combined = pd.concat([df, r2], ignore_index=True, sort=False)
    combined.drop_duplicates(
        subset=['snap_days','canary_sma','canary_hyst','canary_type','select','def_mom_period','health'],
        keep='last', inplace=True,
    )
    combined.to_csv(os.path.join(OUT, 'mp_combined.csv'), index=False)

    print('\n=== Final 버킷별 top 3 ===')
    for snap in sorted(combined['snap_days'].unique()):
        sub = combined[combined['snap_days'] == snap].sort_values('Cal', ascending=False).head(3)
        print(f'\nsnap={snap}:')
        print(sub[['canary_hyst','select','def_mom_period','health','CAGR','MDD','Sharpe','Cal']].to_string(index=False))


if __name__ == '__main__':
    main()
