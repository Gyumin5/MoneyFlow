#!/usr/bin/env python3
"""Phase 8 — 현재 V23 70/15/15 DD 분석 + 그래프."""
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
eq, rebal = simulate(returns, 0.70, 0.15, 0.15, 'T1', 0.15)
print(f'기간: {eq.index[0].date()} ~ {eq.index[-1].date()}')

# DD 시계열
peak = eq.cummax()
dd = eq / peak - 1

# MDD + 발생 시점
mdd = dd.min()
mdd_date = dd.idxmin()
# Peak 시점 (직전 peak)
peak_date = eq[:mdd_date].idxmax()
peak_val = eq.loc[peak_date]
trough_val = eq.loc[mdd_date]
# Recovery date (eq 가 peak_val 회복한 첫 날)
post = eq[mdd_date:]
rec_idx = post[post >= peak_val]
rec_date = rec_idx.index[0] if len(rec_idx) > 0 else None

print(f'\\n=== MDD 상세 ===')
print(f'  MDD: {mdd*100:.2f}%')
print(f'  Peak: {peak_date.date()} (equity {peak_val:.4f})')
print(f'  Trough: {mdd_date.date()} (equity {trough_val:.4f})')
print(f'  하락 기간: {(mdd_date - peak_date).days} 일')
print(f'  회복 시점: {rec_date.date() if rec_date is not None else "미회복"}')
if rec_date is not None:
    print(f'  회복 기간 (trough → recovery): {(rec_date - mdd_date).days} 일')
    print(f'  전체 DD 기간 (peak → recovery): {(rec_date - peak_date).days} 일')

# Top 5 DD episodes (separated)
print('\\n=== Top 5 DD episodes ===')
# Identify each DD episode: peak → trough → recovery
episodes = []
in_dd = False
ep_peak_date = None
ep_peak_val = None
ep_trough_date = None
ep_trough_val = None
for d, v in eq.items():
    if not in_dd:
        if v < peak.loc[d]:
            in_dd = True
            ep_peak_date = peak.loc[:d].idxmax()
            ep_peak_val = peak.loc[d]
            ep_trough_date = d
            ep_trough_val = v
    else:
        if v < ep_trough_val:
            ep_trough_val = v
            ep_trough_date = d
        if v >= ep_peak_val:
            episodes.append(dict(
                peak_date=ep_peak_date, peak_val=ep_peak_val,
                trough_date=ep_trough_date, trough_val=ep_trough_val,
                recovery_date=d,
                dd_pct=ep_trough_val/ep_peak_val - 1,
                fall_days=(ep_trough_date - ep_peak_date).days,
                recovery_days=(d - ep_trough_date).days,
            ))
            in_dd = False
# ongoing DD if 미회복
if in_dd:
    episodes.append(dict(
        peak_date=ep_peak_date, peak_val=ep_peak_val,
        trough_date=ep_trough_date, trough_val=ep_trough_val,
        recovery_date=None,
        dd_pct=ep_trough_val/ep_peak_val - 1,
        fall_days=(ep_trough_date - ep_peak_date).days,
        recovery_days=None,
    ))

ep_df = pd.DataFrame(episodes).sort_values('dd_pct')
for i, r in ep_df.head(5).iterrows():
    rec = str(r['recovery_date'].date()) if r['recovery_date'] is not None else '미회복'
    rd = f'{r["recovery_days"]}일' if r['recovery_days'] is not None else '진행 중'
    print(f'  {r["peak_date"].date()} → {r["trough_date"].date()} → {rec}: {r["dd_pct"]*100:+.2f}% (하락 {r["fall_days"]}일 / 회복 {rd})')

# Rolling MDD
print('\\n=== Rolling DD (다양한 window) ===')
for window_days in [21, 63, 126, 252, 504]:
    rmin = eq.rolling(window_days).apply(lambda x: x.iloc[-1]/x.max() - 1, raw=False).min()
    print(f'  rolling {window_days}d (~{window_days//21}월) min DD: {rmin*100:+.2f}%')

# DD 분포
print('\\n=== DD 분포 ===')
n_at_dd = (dd < -0.05).sum()
n_at_dd10 = (dd < -0.10).sum()
n_at_dd15 = (dd < -0.15).sum()
print(f'  전체 일수: {len(eq)}')
print(f'  DD > 5% 일수: {n_at_dd} ({n_at_dd/len(eq)*100:.1f}%)')
print(f'  DD > 10% 일수: {n_at_dd10} ({n_at_dd10/len(eq)*100:.1f}%)')
print(f'  DD > 15% 일수: {n_at_dd15} ({n_at_dd15/len(eq)*100:.1f}%)')

# Underwater curve avg
print(f'\\n  평균 DD: {dd.mean()*100:.2f}%')
print(f'  중위 DD: {dd.median()*100:.2f}%')

# 그래프
fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)
fig.suptitle('V23 70/15/15 T1 15pp (current ops) — 5.5y BT (2020-11 ~ 2026-05)', fontsize=13)

# 1) Equity log scale
ax = axes[0]
ax.plot(eq.index, eq.values, color='#1f77b4', linewidth=1.2)
ax.set_yscale('log')
ax.set_ylabel('Equity (log scale, start=1)')
ax.grid(True, alpha=0.3)
ax.set_title(f'Equity curve: 1.00 → {eq.iloc[-1]:.2f} ({(eq.iloc[-1]-1)*100:+.0f}%)')

# 2) DD underwater
ax = axes[1]
ax.fill_between(dd.index, dd.values * 100, 0, color='#d62728', alpha=0.4)
ax.plot(dd.index, dd.values * 100, color='#8b0000', linewidth=0.8)
ax.set_ylabel('Drawdown (%)')
ax.grid(True, alpha=0.3)
ax.axhline(-10, color='gray', linestyle=':', alpha=0.5, label='-10%')
ax.axhline(-15, color='orange', linestyle=':', alpha=0.5, label='-15%')
ax.axhline(mdd * 100, color='red', linestyle='--', alpha=0.6, label=f'MDD {mdd*100:.1f}%')
ax.legend(loc='lower left', fontsize=8)
ax.set_title(f'Drawdown — MDD {mdd*100:.2f}% @ {mdd_date.date()}')

# 3) Rolling 252d return
ax = axes[2]
ret_252 = eq.pct_change(252).dropna() * 100
ax.plot(ret_252.index, ret_252.values, color='#2ca02c', linewidth=1.0)
ax.axhline(0, color='black', linewidth=0.5)
ax.set_ylabel('Rolling 1Y Return (%)')
ax.set_xlabel('Date')
ax.grid(True, alpha=0.3)
ax.set_title(f'Rolling 1Y Return (min {ret_252.min():.0f}% / max {ret_252.max():.0f}%)')

plt.tight_layout()
out_path = os.path.join(HERE, 'v23_70_15_15_dd_analysis.png')
plt.savefig(out_path, dpi=110, bbox_inches='tight')
print(f'\\n그래프 저장: {out_path}')
