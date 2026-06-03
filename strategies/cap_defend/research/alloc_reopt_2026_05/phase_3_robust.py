#!/usr/bin/env python3
"""Phase 3 — Yearly rank sum + plateau 검증.

phase_2 결과 + yearly metrics 로:
1) 연도별 Cal 순위 합산 → 매년 안정적 우수한 config 찾기
2) Top 후보 각각에 대해 비율 ±5pp, 임계값 ±1step plateau 검사
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))

grid = pd.read_csv(os.path.join(HERE, 'alloc_grid_results.csv'))
yr = pd.read_csv(os.path.join(HERE, 'alloc_yearly.csv'))

# cfg key (정확히 같은 type)
key_cols = ['trigger', 'thr', 'w_stock', 'w_spot', 'w_fut']
grid['cfg'] = grid[key_cols].astype(str).agg('|'.join, axis=1)
yr['cfg'] = yr[key_cols].astype(str).agg('|'.join, axis=1)

print(f"grid rows: {len(grid)}, yearly rows: {len(yr)}")
print(f"unique cfgs in grid: {grid.cfg.nunique()}, in yearly: {yr.cfg.nunique()}")
print(f"years: {sorted(yr.year.unique())}")

# 연도별 cfg 의 Cal rank (낮을수록 좋음 = rank=1)
yr['rank'] = yr.groupby('year')['y_Cal'].rank(ascending=False, method='min')

# cfg 별 ranksum
ranksum = yr.groupby('cfg').agg(
    ranksum=('rank', 'sum'),
    avg_rank=('rank', 'mean'),
    n_years=('rank', 'count'),
    min_year_cal=('y_Cal', 'min'),
    max_year_cal=('y_Cal', 'max'),
).reset_index()

# 전체 metrics merge
rs_full = ranksum.merge(grid[['cfg', 'Cal', 'CAGR', 'MDD', 'Sharpe', 'rebal']], on='cfg')

# 충분히 많은 연도 (전 기간 등장) 만 보존
n_years_max = ranksum.n_years.max()
rs_full = rs_full[rs_full.n_years >= n_years_max - 1]
print(f"\nn_years_max={n_years_max}, 충분 cfg: {len(rs_full)}")

# 다축 종합 ranking
rs_full = rs_full.sort_values('ranksum')

# key cols recover
for i, k in enumerate(key_cols):
    rs_full[k] = rs_full.cfg.str.split('|').str[i]

# 출력 정리
cols_out = key_cols + ['ranksum', 'avg_rank', 'min_year_cal', 'Cal', 'CAGR', 'MDD', 'Sharpe', 'rebal']
out = rs_full[cols_out].head(30)
print("\n=== Top 30 by yearly ranksum ===")
print(out.to_string(index=False))
out.to_csv(os.path.join(HERE, 'alloc_ranksum_top30.csv'), index=False)

# 추가: Cal + Sharpe + ranksum 가중 종합 점수 (각 z-score)
all_full = ranksum.merge(grid[['cfg', 'Cal', 'CAGR', 'MDD', 'Sharpe', 'rebal']], on='cfg')
all_full = all_full[all_full.n_years >= n_years_max - 1].copy()
all_full['z_cal'] = (all_full.Cal - all_full.Cal.mean()) / all_full.Cal.std()
all_full['z_sharpe'] = (all_full.Sharpe - all_full.Sharpe.mean()) / all_full.Sharpe.std()
all_full['z_ranksum'] = -(all_full.ranksum - all_full.ranksum.mean()) / all_full.ranksum.std()  # 낮을수록 좋음 → 부호 반전
all_full['composite'] = all_full.z_cal + all_full.z_sharpe + all_full.z_ranksum
all_full = all_full.sort_values('composite', ascending=False)
for i, k in enumerate(key_cols):
    all_full[k] = all_full.cfg.str.split('|').str[i]
print("\n=== Top 30 by composite (z_cal + z_sharpe + z_ranksum) ===")
comp_out = all_full[key_cols + ['composite', 'Cal', 'CAGR', 'MDD', 'Sharpe', 'ranksum', 'avg_rank']].head(30)
print(comp_out.to_string(index=False))
comp_out.to_csv(os.path.join(HERE, 'alloc_composite_top30.csv'), index=False)

# Plateau 검사: top 5 each (ranksum / composite / Cal 등) 후보에 대해 ±5pp / ±1 thr step robust
print("\n=== Plateau 검사 (top by composite) ===")
top_candidates = all_full.head(5)
plateau_results = []
for _, c in top_candidates.iterrows():
    ws, wc, wf = int(float(c.w_stock)*100), int(float(c.w_spot)*100), int(float(c.w_fut)*100)
    trig = c.trigger
    thr = float(c.thr)

    # 인접 비율 (각 자산 ±5pp, 합 100 유지)
    neighbors = []
    for dws in (-5, 0, 5):
        for dwc in (-5, 0, 5):
            nws = ws + dws
            nwc = wc + dwc
            nwf = 100 - nws - nwc
            if 0 <= nws <= 100 and 0 <= nwc <= 100 and 0 <= nwf <= 100 and (dws, dwc) != (0, 0):
                neighbors.append((nws/100, nwc/100, nwf/100))
    # 인접 임계값
    thr_list = sorted(grid[grid.trigger == trig].thr.unique())
    idx = thr_list.index(thr) if thr in thr_list else -1
    thr_neighbors = []
    if idx > 0: thr_neighbors.append(thr_list[idx-1])
    if 0 <= idx < len(thr_list)-1: thr_neighbors.append(thr_list[idx+1])

    # 본인 Cal
    base_cal = float(c.Cal)
    neigh_cals = []
    for (nws, nwc, nwf) in neighbors:
        sub = grid[(grid.trigger==trig)&(grid.thr==thr)&(grid.w_stock==nws)&(grid.w_spot==nwc)&(grid.w_fut==nwf)]
        if len(sub) > 0:
            neigh_cals.append(float(sub.Cal.iloc[0]))
    for nthr in thr_neighbors:
        sub = grid[(grid.trigger==trig)&(grid.thr==nthr)&(grid.w_stock==ws/100)&(grid.w_spot==wc/100)&(grid.w_fut==wf/100)]
        if len(sub) > 0:
            neigh_cals.append(float(sub.Cal.iloc[0]))
    if neigh_cals:
        min_neigh = min(neigh_cals)
        avg_neigh = sum(neigh_cals)/len(neigh_cals)
        drop = (base_cal - min_neigh) / base_cal
        print(f"  {trig} thr={thr} {ws}/{wc}/{wf}: base Cal={base_cal:.2f}, neigh min={min_neigh:.2f} avg={avg_neigh:.2f} (max drop {drop*100:.0f}%)")
        plateau_results.append(dict(
            trigger=trig, thr=thr, w_stock=ws, w_spot=wc, w_fut=wf,
            base_cal=base_cal, neigh_min=min_neigh, neigh_avg=avg_neigh,
            max_drop_pct=drop*100, n_neighbors=len(neigh_cals),
        ))

pd.DataFrame(plateau_results).to_csv(os.path.join(HERE, 'alloc_plateau.csv'), index=False)

# 현재 70/15/15 baseline 위치
print("\n=== 현재 70/15/15 ranking ===")
cur_rows = all_full[(all_full.w_stock=='0.7')&(all_full.w_spot=='0.15')&(all_full.w_fut=='0.15')]
cur_rows = cur_rows.sort_values('composite', ascending=False)
print(cur_rows.head(5)[key_cols + ['composite', 'Cal', 'CAGR', 'MDD', 'Sharpe', 'ranksum']].to_string(index=False))
print(f"  composite top 5 위치 (전체 중): ranks {all_full.reset_index().index[all_full.cfg.isin(cur_rows.cfg)].tolist()[:5]} / {len(all_full)}")
