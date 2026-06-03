#!/usr/bin/env python3
"""Phase 9 — 자산별 단독 vs 70/15/15 통합 비교. 배분 효과 검증."""
import os, sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from phase_4_extended import simulate, metrics, load_curves

returns = load_curves()

def eq_from_returns(r):
    return (1 + r).cumprod()

# 자산별 단독 (100% 배분)
stock_eq = eq_from_returns(returns['stock'])
spot_eq = eq_from_returns(returns['spot'])
fut_eq = eq_from_returns(returns['fut'])
combined_eq, _ = simulate(returns, 0.70, 0.15, 0.15, 'T1', 0.15)

def analyze(eq, name):
    if eq.iloc[-1] <= 0:
        return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    peak = eq.cummax()
    dd = eq / peak - 1
    n5 = (dd < -0.05).sum() / len(eq) * 100
    n10 = (dd < -0.10).sum() / len(eq) * 100
    n20 = (dd < -0.20).sum() / len(eq) * 100
    return dict(name=name,
                CAGR=cagr*100, MDD=mdd*100, Cal=cal, Sharpe=sh,
                vol_ann=dr.std()*np.sqrt(252)*100,
                dd_avg=dd.mean()*100, dd_med=dd.median()*100,
                pct_at_dd5=n5, pct_at_dd10=n10, pct_at_dd20=n20)

rows = [analyze(stock_eq, '주식 단독 100%'),
        analyze(spot_eq, '코인 spot 단독 100%'),
        analyze(fut_eq, '선물 단독 100%'),
        analyze(combined_eq, '70/15/15 통합')]

print('=== 자산별 성과 비교 ===')
df = pd.DataFrame(rows)
print(df.to_string(index=False))

# 상관
print('\\n=== 일일 수익률 상관 ===')
corr = returns[['stock','spot','fut']].corr()
print(corr.to_string())

# 분산 효과: 가중 평균 vs 실제 통합 — Sharpe / Cal
ws, wc, wf = 0.70, 0.15, 0.15
wavg_cagr = ws*rows[0]['CAGR'] + wc*rows[1]['CAGR'] + wf*rows[2]['CAGR']
wavg_vol = ws*rows[0]['vol_ann'] + wc*rows[1]['vol_ann'] + wf*rows[2]['vol_ann']
actual_cagr = rows[3]['CAGR']
actual_vol = rows[3]['vol_ann']
print('\\n=== 분산 효과 (70/15/15 기준) ===')
print(f'  가중평균 CAGR: {wavg_cagr:.1f}% (단순 합산)')
print(f'  실제 통합 CAGR: {actual_cagr:.1f}%')
print(f'  가중평균 vol: {wavg_vol:.1f}%')
print(f'  실제 통합 vol: {actual_vol:.1f}% (분산 효과로 {wavg_vol-actual_vol:.1f}%pt 감소)')
print(f'  가중평균 MDD: {ws*rows[0]["MDD"] + wc*rows[1]["MDD"] + wf*rows[2]["MDD"]:.1f}%')
print(f'  실제 통합 MDD: {rows[3]["MDD"]:.1f}% (분산 효과)')

# 그래프
fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
fig.suptitle('Per-Asset vs 70/15/15 Combined — 5.5y BT', fontsize=13)

ax = axes[0]
for eq, label, color in [(stock_eq,'Stock 100%','#1f77b4'),
                          (spot_eq,'Coin Spot 100%','#ff7f0e'),
                          (fut_eq,'Futures 100%','#d62728'),
                          (combined_eq,'70/15/15','#2ca02c')]:
    ax.plot(eq.index, eq.values, label=f'{label} (end={eq.iloc[-1]:.1f})', linewidth=1.1, color=color)
ax.set_yscale('log')
ax.set_ylabel('Equity (log)')
ax.legend(loc='upper left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.set_title('Equity curves')

ax = axes[1]
for eq, label, color in [(stock_eq,'Stock','#1f77b4'),
                          (spot_eq,'Coin Spot','#ff7f0e'),
                          (fut_eq,'Futures','#d62728'),
                          (combined_eq,'70/15/15','#2ca02c')]:
    dd = (eq / eq.cummax() - 1) * 100
    ax.plot(dd.index, dd.values, label=f'{label} (MDD {dd.min():.1f}%)', linewidth=1.0, color=color)
ax.set_ylabel('Drawdown (%)')
ax.set_xlabel('Date')
ax.legend(loc='lower left', fontsize=9)
ax.grid(True, alpha=0.3)
ax.axhline(0, color='black', linewidth=0.4)
ax.set_title('Drawdown comparison')

plt.tight_layout()
out_path = os.path.join(HERE, 'v23_per_asset_compare.png')
plt.savefig(out_path, dpi=110, bbox_inches='tight')
print(f'\\n그래프 저장: {out_path}')
