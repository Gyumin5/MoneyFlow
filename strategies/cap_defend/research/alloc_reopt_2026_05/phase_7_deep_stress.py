#!/usr/bin/env python3
"""Phase 7 — 심층 stress test + hard loss limit + rolling loss + C 주변 grid.

목적:
1) Joint stress (coin x0.3 + vol x1.5)
2) Event replay 2020-03 COVID crash / 2022-05 LUNA / 2022-11 FTX
3) Rolling 3/6/12개월 max loss
4) C 주변 grid (60/22/18, 58/22/20 등) — 5pp 그리드 밖이지만 비교 위해 시뮬
5) Hard loss constraint (rolling 12개월 -40%) 통과 후보만 살림
"""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase_4_extended import simulate, metrics, yearly_cals, load_curves

returns = load_curves()

CANDIDATES = [
    dict(name='A 70/15/15 T1 15pp', ws=0.70, wc=0.15, wf=0.15, trig='T1', thr=0.15),
    dict(name='B 65/17.5/17.5 T1 12pp', ws=0.65, wc=0.175, wf=0.175, trig='T1', thr=0.12),
    dict(name='C 60/20/20 T1 12pp', ws=0.60, wc=0.20, wf=0.20, trig='T1', thr=0.12),
    # C 주변 (5pp 그리드 밖)
    dict(name='C+ 60/22/18 T1 12pp', ws=0.60, wc=0.22, wf=0.18, trig='T1', thr=0.12),
    dict(name='C- 58/22/20 T1 12pp', ws=0.58, wc=0.22, wf=0.20, trig='T1', thr=0.12),
    dict(name='F 55/25/20 T1 12pp', ws=0.55, wc=0.25, wf=0.20, trig='T1', thr=0.12),
    dict(name='F2 55/22.5/22.5 T1 12pp', ws=0.55, wc=0.225, wf=0.225, trig='T1', thr=0.12),
    dict(name='G 50/30/20 T1 12pp', ws=0.50, wc=0.30, wf=0.20, trig='T1', thr=0.12),
]


def sim(returns, c, tx=0.0005):
    eq, _ = simulate(returns, c['ws'], c['wc'], c['wf'], c['trig'], c['thr'], tx_cost=tx)
    return eq


def rolling_max_loss(eq, window_days):
    # window 내 max loss = max(1 - eq[t+w]/eq[t]) over rolling window
    pv = eq.values
    n = len(pv)
    max_loss = 0
    for i in range(n - window_days):
        end_pv = pv[i + window_days]
        loss = end_pv / pv[i] - 1
        max_loss = min(max_loss, loss)
    return float(max_loss)


def metrics_extended(eq):
    if eq.iloc[-1] <= 0:
        return dict(Cal=-999, CAGR=-1, MDD=-1, Sharpe=-99, yr_min=-99,
                    r3m=-1, r6m=-1, r12m=-1)
    m = metrics(eq)
    yc = yearly_cals(eq)
    m['yr_min'] = min(yc.values()) if yc else -99
    m['r3m'] = rolling_max_loss(eq, 63)
    m['r6m'] = rolling_max_loss(eq, 126)
    m['r12m'] = rolling_max_loss(eq, 252)
    return m


# 1) Base
print("=== 1) Base ===")
res1 = []
for c in CANDIDATES:
    eq = sim(returns, c)
    m = metrics_extended(eq)
    res1.append(dict(name=c['name'], **m))
df1 = pd.DataFrame(res1)
print(df1.to_string(index=False))

# 2) Joint stress: coin x0.5 + vol x1.5
print("\n=== 2) Joint stress: coin x0.5 + vol x1.5 ===")
ret_j = returns.copy()
for col in ('spot', 'fut'):
    mu = ret_j[col].mean()
    ret_j[col] = (mu * 0.5) + (ret_j[col] - mu) * 1.5
res2 = []
for c in CANDIDATES:
    eq = sim(ret_j, c)
    m = metrics_extended(eq)
    res2.append(dict(name=c['name'], **m))
df2 = pd.DataFrame(res2)
print(df2.to_string(index=False))

# 3) Joint stress: coin x0.3 + vol x1.5
print("\n=== 3) Joint stress: coin x0.3 + vol x1.5 ===")
ret_j2 = returns.copy()
for col in ('spot', 'fut'):
    mu = ret_j2[col].mean()
    ret_j2[col] = (mu * 0.3) + (ret_j2[col] - mu) * 1.5
res3 = []
for c in CANDIDATES:
    eq = sim(ret_j2, c)
    m = metrics_extended(eq)
    res3.append(dict(name=c['name'], **m))
df3 = pd.DataFrame(res3)
print(df3.to_string(index=False))

# 4) Event replay — 특정 구간만 cutout 후 BT
print("\n=== 4) Event replay (특정 crash 구간) ===")
events = {
    'COVID_2020_03': ('2020-02-15', '2020-04-15'),  # data 시작 후라 부족할 수 있음
    'BTC_crash_2021_05': ('2021-05-01', '2021-08-01'),
    'LUNA_2022_05': ('2022-04-15', '2022-06-30'),
    'FTX_2022_11': ('2022-10-15', '2022-12-31'),
}
res4 = []
for ename, (s, e) in events.items():
    sub = returns[(returns.index >= s) & (returns.index <= e)]
    if len(sub) < 10:
        continue
    for c in CANDIDATES:
        eq = sim(sub, c)
        if len(eq) < 5:
            continue
        loss = eq.iloc[-1] / eq.iloc[0] - 1
        mdd = (eq / eq.cummax() - 1).min()
        res4.append(dict(event=ename, name=c['name'], n=len(sub),
                         ret=loss, mdd=float(mdd)))
df4 = pd.DataFrame(res4)
print(df4.to_string(index=False))

# 5) Hard loss constraint: rolling 12개월 ≥ -40%, MDD ≥ -25%
print("\n=== 5) Hard loss constraint 통과 (base 기준, r12m ≥ -40%, MDD ≥ -25%) ===")
df1_h = df1.copy()
df1_h['pass_r12m_40'] = df1_h['r12m'] >= -0.40
df1_h['pass_r12m_50'] = df1_h['r12m'] >= -0.50
df1_h['pass_mdd_25'] = df1_h['MDD'] >= -0.25
df1_h['pass_mdd_30'] = df1_h['MDD'] >= -0.30
df1_h['hard_pass_40_25'] = df1_h['pass_r12m_40'] & df1_h['pass_mdd_25']
df1_h['hard_pass_50_30'] = df1_h['pass_r12m_50'] & df1_h['pass_mdd_30']
print(df1_h[['name', 'Cal', 'MDD', 'r12m', 'pass_r12m_40', 'pass_mdd_25', 'hard_pass_40_25', 'hard_pass_50_30']].to_string(index=False))

# 6) Joint stress hard constraint
print("\n=== 6) Hard loss constraint (joint stress coin x0.3 + vol x1.5) ===")
df3_h = df3.copy()
df3_h['stress_pass_r12m_40'] = df3_h['r12m'] >= -0.40
df3_h['stress_pass_mdd_30'] = df3_h['MDD'] >= -0.30
df3_h['stress_pass_full'] = df3_h['stress_pass_r12m_40'] & df3_h['stress_pass_mdd_30']
print(df3_h[['name', 'Cal', 'MDD', 'r12m', 'stress_pass_full']].to_string(index=False))

# 저장
df1.to_csv(os.path.join(HERE, 'deep_1_base.csv'), index=False)
df2.to_csv(os.path.join(HERE, 'deep_2_joint_x0.5.csv'), index=False)
df3.to_csv(os.path.join(HERE, 'deep_3_joint_x0.3.csv'), index=False)
df4.to_csv(os.path.join(HERE, 'deep_4_events.csv'), index=False)
df1_h.to_csv(os.path.join(HERE, 'deep_5_hard_constraint.csv'), index=False)
df3_h.to_csv(os.path.join(HERE, 'deep_6_stress_hard.csv'), index=False)

print("\n저장 완료. deep_1~6_*.csv")
