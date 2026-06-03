#!/usr/bin/env python3
"""Phase 5 — 코인 보수적 시나리오 (coin returns haircut) + 전체 그리드 요약 dump.

coin bull 편향 우려 → spot/fut 일일 수익률 ×0.5, ×0.7 시나리오 추가 BT.
주식은 unchanged. 같은 alloc grid + 트리거.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from phase_4_extended import simulate, metrics, yearly_cals, load_curves

returns = load_curves()
print(f"공통 dates: {returns.index[0]} ~ {returns.index[-1]} (n={len(returns)})")

# 비율 grid (제약 ws ≥ wc ≥ wf)
ratios = []
for ws in range(0, 101, 5):
    for wc in range(0, 101 - ws, 5):
        wf = 100 - ws - wc
        if wf < 0: continue
        if ws >= wc and wc >= wf:
            ratios.append((ws/100, wc/100, wf/100))

triggers = {
    'T1': [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
    'T2': [0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15],
    'T3': [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
}

scenarios = {
    'base': 1.0,
    'coin_x0.7': 0.7,
    'coin_x0.5': 0.5,
}

all_rows = []
for sname, factor in scenarios.items():
    print(f"\n[{sname}] factor={factor}")
    ret_mod = returns.copy()
    ret_mod['spot'] = ret_mod['spot'] * factor
    ret_mod['fut'] = ret_mod['fut'] * factor
    done = 0
    for trig, thrs in triggers.items():
        for thr in thrs:
            for (ws, wc, wf) in ratios:
                eq, rebal = simulate(ret_mod, ws, wc, wf, trig, thr)
                m = metrics(eq)
                yc = yearly_cals(eq)
                yr_mean = np.mean(list(yc.values())) if yc else 0
                yr_min = min(yc.values()) if yc else 0
                all_rows.append(dict(
                    scenario=sname,
                    trigger=trig, thr=thr,
                    w_stock=ws, w_spot=wc, w_fut=wf,
                    rebal=rebal,
                    yr_min_cal=yr_min,
                    yr_mean_cal=yr_mean,
                    **m,
                ))
                done += 1
        print(f"  done {done}")

df = pd.DataFrame(all_rows)
df.to_csv(os.path.join(HERE, 'ext_scenarios.csv'), index=False)

# 시나리오별 TOP 20 + 현재 비교
print("\n=== 시나리오별 TOP 10 by Cal ===")
for s in scenarios:
    sub = df[df.scenario == s].sort_values('Cal', ascending=False).head(10)
    print(f"\n[{s}]")
    print(sub[['trigger', 'thr', 'w_stock', 'w_spot', 'w_fut', 'rebal',
               'Cal', 'CAGR', 'MDD', 'Sharpe', 'yr_min_cal']].to_string(index=False))

print("\n=== 현재 운영 (70/15/15 T1 15pp) 시나리오별 ===")
cur = df[(df.w_stock == 0.7) & (df.w_spot == 0.15) & (df.w_fut == 0.15) &
         (df.trigger == 'T1') & (df.thr == 0.15)]
print(cur[['scenario', 'rebal', 'Cal', 'CAGR', 'MDD', 'Sharpe', 'yr_min_cal']].to_string(index=False))

# 시나리오 horizon 동일성: base 에서 top 10 의 coin_x0.5 결과 (강건성 체크)
print("\n=== Base TOP 10 의 coin_x0.5 성과 (강건성 체크) ===")
base_top = df[df.scenario == 'base'].sort_values('Cal', ascending=False).head(10)
for _, r in base_top.iterrows():
    cons = df[(df.scenario == 'coin_x0.5') &
              (df.trigger == r.trigger) & (df.thr == r.thr) &
              (df.w_stock == r.w_stock) & (df.w_spot == r.w_spot) & (df.w_fut == r.w_fut)]
    if len(cons) > 0:
        c = cons.iloc[0]
        ratio = c.Cal / r.Cal if r.Cal > 0 else 0
        print(f"  {r.trigger} {r.thr} {r.w_stock}/{r.w_spot}/{r.w_fut}: base Cal={r.Cal:.2f} → coin_x0.5 Cal={c.Cal:.2f} ({ratio*100:.0f}% 유지)")

# 모든 시나리오에서 top 30 평균 rank (robust under coin haircut)
print("\n=== 모든 시나리오 robust TOP 20 (avg rank) ===")
df['rank_in_scenario'] = df.groupby('scenario')['Cal'].rank(ascending=False, method='min')
key = ['trigger', 'thr', 'w_stock', 'w_spot', 'w_fut']
df['cfg'] = df[key].astype(str).agg('|'.join, axis=1)
robust_agg = df.groupby('cfg').agg(
    avg_rank=('rank_in_scenario', 'mean'),
    max_rank=('rank_in_scenario', 'max'),
    base_cal=('Cal', lambda x: x.iloc[0] if len(x) > 0 else None),  # 첫 행 = base
).reset_index().sort_values('avg_rank').head(20)
splits = robust_agg['cfg'].str.split('|', expand=True)
splits.columns = key
robust_out = pd.concat([splits, robust_agg[['avg_rank', 'max_rank']].reset_index(drop=True)], axis=1)
# Merge base + coin_x0.5 Cal
base_cal = df[df.scenario == 'base'].set_index('cfg')['Cal'].rename('base_Cal')
cons_cal = df[df.scenario == 'coin_x0.5'].set_index('cfg')['Cal'].rename('coin_x0.5_Cal')
mid_cal = df[df.scenario == 'coin_x0.7'].set_index('cfg')['Cal'].rename('coin_x0.7_Cal')
robust_out['cfg'] = robust_agg['cfg'].values
robust_out = robust_out.merge(base_cal.reset_index(), on='cfg', how='left')
robust_out = robust_out.merge(mid_cal.reset_index(), on='cfg', how='left')
robust_out = robust_out.merge(cons_cal.reset_index(), on='cfg', how='left')
print(robust_out.to_string(index=False))
robust_out.to_csv(os.path.join(HERE, 'ext_scenario_robust_top20.csv'), index=False)
