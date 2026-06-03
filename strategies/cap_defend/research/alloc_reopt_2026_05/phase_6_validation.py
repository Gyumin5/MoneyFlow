#!/usr/bin/env python3
"""Phase 6 — 종합 검증: 코인 ×0.3 / vol stress / 비용 민감도 / subperiod / walk-forward.

목적: 60/20/20 T1 12pp 권고가 추가 스트레스 하에서도 유지되는지 + 경쟁 후보 (40/30/30 T1 15pp, 50/25/25 T3 50%, 65/17.5/17.5 중간안, 70/15/15 현재) 비교.
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase_4_extended import simulate, metrics, yearly_cals, load_curves

returns = load_curves()
print(f"공통 dates: {returns.index[0]} ~ {returns.index[-1]} (n={len(returns)})")

# 검증 후보 (제약 ws ≥ wc ≥ wf 적용)
CANDIDATES = [
    dict(name='현재 70/15/15 T1 15pp', ws=0.70, wc=0.15, wf=0.15, trig='T1', thr=0.15),
    dict(name='중간 65/17.5/17.5 T1 12pp', ws=0.65, wc=0.175, wf=0.175, trig='T1', thr=0.12),  # 5pp 미스라 못나옴 가능, fallback
    dict(name='권고 60/20/20 T1 12pp', ws=0.60, wc=0.20, wf=0.20, trig='T1', thr=0.12),
    dict(name='대안 50/25/25 T3 50%', ws=0.50, wc=0.25, wf=0.25, trig='T3', thr=0.50),
    dict(name='robust 40/30/30 T1 15pp', ws=0.40, wc=0.30, wf=0.30, trig='T1', thr=0.15),
    dict(name='보수 55/25/20 T1 12pp', ws=0.55, wc=0.25, wf=0.20, trig='T1', thr=0.12),
]


def run(returns, c, tx=0.0005):
    eq, rebal = simulate(returns, c['ws'], c['wc'], c['wf'], c['trig'], c['thr'], tx_cost=tx)
    return eq, rebal


def metrics_full(eq):
    if eq.iloc[-1] <= 0:
        return dict(Cal=-999, CAGR=-1, MDD=-1, Sharpe=-99, yr_min=-99)
    m = metrics(eq)
    yc = yearly_cals(eq)
    m['yr_min'] = min(yc.values()) if yc else -99
    return m


# 1) 시나리오 (수익률 × factor)
print("\n=== 1) Coin 수익률 스케일 시나리오 ===")
factors = {'x1.0': 1.0, 'x0.7': 0.7, 'x0.5': 0.5, 'x0.3': 0.3, 'x0.0(주식만)': 0.0}
res1 = []
for c in CANDIDATES:
    row = dict(name=c['name'])
    for fname, f in factors.items():
        r = returns.copy()
        r['spot'] *= f
        r['fut'] *= f
        eq, rebal = run(r, c)
        m = metrics_full(eq)
        row[f'Cal_{fname}'] = m['Cal']
        row[f'MDD_{fname}'] = m['MDD']
    res1.append(row)
df1 = pd.DataFrame(res1)
print(df1.to_string(index=False))

# 2) Vol stress: daily return 의 mean 유지 + std 증폭 (×1.5)
print("\n=== 2) Vol stress (coin daily return std × 1.5) ===")
res2 = []
for c in CANDIDATES:
    row = dict(name=c['name'])
    r = returns.copy()
    for col in ('spot', 'fut'):
        mu = r[col].mean()
        r[col] = mu + (r[col] - mu) * 1.5
    eq, rebal = run(r, c)
    m = metrics_full(eq)
    row.update({'Cal': m['Cal'], 'CAGR': m['CAGR'], 'MDD': m['MDD'], 'Sharpe': m['Sharpe'], 'yr_min': m['yr_min'], 'rebal': rebal})
    res2.append(row)
df2 = pd.DataFrame(res2)
print(df2.to_string(index=False))

# 3) 거래비용 민감도
print("\n=== 3) 거래비용 민감도 (5bp / 10bp / 20bp / 50bp) ===")
res3 = []
for c in CANDIDATES:
    row = dict(name=c['name'])
    for tx in (0.0005, 0.001, 0.002, 0.005):
        eq, rebal = run(returns, c, tx=tx)
        m = metrics_full(eq)
        row[f'Cal_{int(tx*10000)}bp'] = m['Cal']
        row[f'rebal_{int(tx*10000)}bp'] = rebal
    res3.append(row)
df3 = pd.DataFrame(res3)
print(df3.to_string(index=False))

# 4) Subperiod
print("\n=== 4) Subperiod 분리 ===")
periods = {
    'bull_2020-11_2021-11': ('2020-11-01', '2021-11-30'),
    'bear_2021-12_2023-01': ('2021-12-01', '2023-01-31'),
    'rec_2023-02_2024-12': ('2023-02-01', '2024-12-31'),
    'alt_rally_2024-09_2025-06': ('2024-09-01', '2025-06-30'),
    'recent_1yr': ('2025-05-14', '2026-05-13'),
}
res4 = []
for pname, (s, e) in periods.items():
    sub = returns[(returns.index >= s) & (returns.index <= e)]
    if len(sub) < 20:
        continue
    for c in CANDIDATES:
        eq, rebal = run(sub, c)
        m = metrics_full(eq)
        res4.append(dict(period=pname, name=c['name'], rebal=rebal,
                         Cal=m['Cal'], CAGR=m['CAGR'], MDD=m['MDD'], Sharpe=m['Sharpe']))
df4 = pd.DataFrame(res4)
print(df4.to_string(index=False))

# 5) Walk-forward: 3년 IS, 2.5년 OOS (단순 fixed split, 후보 비교 용도)
print("\n=== 5) Walk-forward (IS 3년 vs OOS 2.5년) ===")
half = returns.index[len(returns) // 2]
is_end = returns.index[0] + pd.Timedelta(days=365*3)
oos_start = is_end
res5 = []
for c in CANDIDATES:
    is_ret = returns[returns.index < is_end]
    oos_ret = returns[returns.index >= oos_start]
    is_eq, _ = run(is_ret, c)
    oos_eq, _ = run(oos_ret, c)
    is_m = metrics_full(is_eq)
    oos_m = metrics_full(oos_eq)
    res5.append(dict(name=c['name'],
                     IS_Cal=is_m['Cal'], IS_MDD=is_m['MDD'], IS_Sharpe=is_m['Sharpe'],
                     OOS_Cal=oos_m['Cal'], OOS_MDD=oos_m['MDD'], OOS_Sharpe=oos_m['Sharpe'],
                     OOS_vs_IS_drop_pct=(is_m['Cal']-oos_m['Cal'])/is_m['Cal']*100 if is_m['Cal']>0 else 0))
df5 = pd.DataFrame(res5)
print(df5.to_string(index=False))

# 저장
df1.to_csv(os.path.join(HERE, 'val_1_scenarios.csv'), index=False)
df2.to_csv(os.path.join(HERE, 'val_2_vol_stress.csv'), index=False)
df3.to_csv(os.path.join(HERE, 'val_3_tx_cost.csv'), index=False)
df4.to_csv(os.path.join(HERE, 'val_4_subperiod.csv'), index=False)
df5.to_csv(os.path.join(HERE, 'val_5_walkforward.csv'), index=False)

print("\n저장 완료. val_1~5_*.csv")
