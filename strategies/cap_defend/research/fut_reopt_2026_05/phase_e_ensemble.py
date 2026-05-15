#!/usr/bin/env python3
"""DEPRECATED — DO NOT RUN. 잘못된 시뮬 방식 (equity-curve EW).

Phase E — V23 + C1 EW 앙상블 시뮬레이션 (equity curve EW).

실패 이유: 두 단독 BT 의 equity 곡선을 EW 평균만 함. 단일 capital pool 효과,
target merge 시 netting, universe 합집합 효과 등을 반영 못함. 실제 ensemble 운영
모델과 다름.

정식 경로: phase_e3_target_ew_proper.py
- backtest_futures_full.py 의 external_target_schedule 모드 사용
- 청산/crash/DD/BL forced exit + 펀딩 + tx + slip + drift 모두 정확
- V23/C1 replay parity 0% 검증 통과
"""
import os, sys
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
import pandas as pd
import numpy as np
from backtest_futures_full import load_data, run

OUT_DIR = os.path.dirname(os.path.abspath(__file__))

CANDIDATES = {
    'V23': dict(sma_days=42, mom_short_days=30, mom_long_days=90,
                n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03),
    'C1':  dict(sma_days=38, mom_short_days=20, mom_long_days=122,
                n_snapshots=3, snap_interval_bars=111, drift_threshold=0.020),
}
COMMON = dict(
    interval='D', leverage=3.0,
    universe_size=3, selection='greedy', cap=1/3,
    tx_cost=0.0006, maint_rate=0.004,
    vol_days=90, vol_threshold=0.05,
    canary_hyst=0.015,
    health_mode='mom2vol',
    start_date='2020-10-01', end_date='2026-05-13',
)

def metrics(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return dict(Sharpe=sh, CAGR=cagr, MDD=mdd, Cal=cal)

def yearly_metrics(eq):
    out = []
    for year, g in eq.groupby(eq.index.year):
        if len(g) < 20: continue
        ret = g.iloc[-1] / g.iloc[0] - 1
        dr = g.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
        mdd = (g / g.cummax() - 1).min()
        cal = ret / abs(mdd) if mdd != 0 else 0
        out.append(dict(year=year, ret=ret, MDD=mdd, Cal=cal, Sharpe=sh))
    return pd.DataFrame(out)

bars, funding = load_data('D')
print("="*70)
print("Phase E: V23 + C1 EW 앙상블 (equity curve 평균)")
print("="*70)

eqs = {}
for name, cfg in CANDIDATES.items():
    m = run(bars, funding, **{**COMMON, **cfg})
    eqs[name] = m['_equity']
    m2 = metrics(eqs[name])
    print(f"{name}: Sharpe={m2['Sharpe']:.2f} CAGR={m2['CAGR']:+.1%} MDD={m2['MDD']:+.1%} Cal={m2['Cal']:.2f}")

# 일일 수익률 단위로 EW 앙상블 (capital 합산 방식이 아니라 daily return EW)
ret_v23 = eqs['V23'].pct_change().fillna(0)
ret_c1  = eqs['C1'].pct_change().fillna(0)
# 두 시리즈가 같은 index일 때만 합집합
idx = ret_v23.index.union(ret_c1.index)
ret_v23 = ret_v23.reindex(idx, fill_value=0)
ret_c1  = ret_c1.reindex(idx, fill_value=0)
ret_ew = 0.5 * ret_v23 + 0.5 * ret_c1
eq_ew = (1 + ret_ew).cumprod() * 10000
eq_ew.index = idx

m_ew = metrics(eq_ew)
print(f"\nEW 앙상블 (50/50):")
print(f"  Sharpe={m_ew['Sharpe']:.2f} CAGR={m_ew['CAGR']:+.1%} MDD={m_ew['MDD']:+.1%} Cal={m_ew['Cal']:.2f}")

# 상관 (daily return)
corr = ret_v23.corr(ret_c1)
print(f"\nV23 vs C1 daily return 상관: {corr:.3f}")

# yearly
yr_ew = yearly_metrics(eq_ew)
print(f"\nEW 연도별:")
print(yr_ew.round(3).to_string(index=False))

# yearly diff vs V23/C1
yr_v23 = yearly_metrics(eqs['V23'])
yr_c1  = yearly_metrics(eqs['C1'])

print("\n연도별 Sharpe 비교:")
print(f"{'year':<6} {'V23':>8} {'C1':>8} {'EW':>8}")
for y in yr_ew.year:
    v = yr_v23[yr_v23.year==y].Sharpe.iloc[0] if y in yr_v23.year.values else None
    c = yr_c1[yr_c1.year==y].Sharpe.iloc[0] if y in yr_c1.year.values else None
    e = yr_ew[yr_ew.year==y].Sharpe.iloc[0]
    print(f"{y:<6} {v:>8.2f} {c:>8.2f} {e:>8.2f}")

print("\n연도별 Cal 비교:")
print(f"{'year':<6} {'V23':>8} {'C1':>8} {'EW':>8}")
for y in yr_ew.year:
    v = yr_v23[yr_v23.year==y].Cal.iloc[0] if y in yr_v23.year.values else None
    c = yr_c1[yr_c1.year==y].Cal.iloc[0] if y in yr_c1.year.values else None
    e = yr_ew[yr_ew.year==y].Cal.iloc[0]
    print(f"{y:<6} {v:>8.2f} {c:>8.2f} {e:>8.2f}")
