"""V22 spot 두 멤버 (D_SMA42, H4_SMA240) 일별 수익 상관 진단.

목적: V22 = 1D + 4h 2멤버 EW 가 진짜 다양성을 가지는지 측정.
방법:
- 각 멤버 단독 BT → equity curve → 일별 수익
- D 멤버 (1D 봉) 와 H4 멤버 (4h 봉, 일별 리샘플) 정렬
- pearson / spearman corr + 상관 시계열 (rolling 90d)
- canary on/off 일치율
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub

START = '2020-10-01'
END   = '2026-04-27'

D_CFG = dict(
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, vol_threshold=0.05,
    canary_hyst=0.015, n_snapshots=3,
    universe_size=3, cap=1/3, tx_cost=0.004,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=60,  # 60 일
    start_date=START, end_date=END,
)
H4_CFG = dict(
    interval='4h', asset_type='spot', leverage=1.0,
    sma_bars=240, mom_short_bars=12*6, mom_long_bars=180*6,
    vol_days=90, vol_threshold=0.05,
    canary_hyst=0.015, n_snapshots=3,
    universe_size=3, cap=1/3, tx_cost=0.004,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=360,  # 60 일 = 360 × 4h
    start_date=START, end_date=END,
)

print("loading 1h bars (will resample)...")
bars_1h, funding = ub.load_data('1h')
print("loading D bars...")
bars_D, _ = ub.load_data('D')
print("loading 4h bars...")
bars_4h, _ = ub.load_data('4h')

print(f"\n=== running D_SMA42 ===")
res_d = ub.run(bars_D, funding, **D_CFG)
print(f"=== running H4_SMA240 ===")
res_h = ub.run(bars_4h, funding, **H4_CFG)

eq_d = res_d['_equity']
eq_h = res_h['_equity']
print(f"\nD equity len: {len(eq_d)}, H4 equity len: {len(eq_h)}")

# 일별 returns
ret_d = eq_d.pct_change().dropna()
ret_h_daily = eq_h.resample('1D').last().pct_change().dropna()

# 정렬
df = pd.DataFrame({'D': ret_d, 'H4': ret_h_daily}).dropna()
print(f"\naligned daily returns: {len(df)} rows ({df.index.min()} ~ {df.index.max()})")

pearson = df.corr(method='pearson').iloc[0, 1]
spearman = df.corr(method='spearman').iloc[0, 1]
print(f"\n전체기간 daily return corr")
print(f"  pearson : {pearson:.4f}")
print(f"  spearman: {spearman:.4f}")

# 연도별
print("\n연도별 pearson corr")
for yr, sub in df.groupby(df.index.year):
    if len(sub) < 30: continue
    print(f"  {yr}: {sub.corr().iloc[0,1]:.4f}  (n={len(sub)})")

# rolling 90d
roll = df['D'].rolling(90).corr(df['H4']).dropna()
print(f"\nrolling 90d corr: mean={roll.mean():.4f}, min={roll.min():.4f}, max={roll.max():.4f}")
print(f"  rolling >0.7 인 비율: {(roll>0.7).mean():.1%}")
print(f"  rolling <0.3 인 비율: {(roll<0.3).mean():.1%}")

# 단독 sleeve 메트릭 비교
def _stats(eq):
    rs = eq.resample('1D').last().dropna()
    rt = rs.pct_change().dropna()
    cagr = (rs.iloc[-1]/rs.iloc[0])**(252/len(rt)) - 1
    eq_max = rs.cummax()
    mdd = ((rs - eq_max) / eq_max).min()
    sh = rt.mean() / rt.std() * np.sqrt(252) if rt.std()>0 else 0
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return cagr, mdd, sh, cal

print("\n단독 sleeve 성과 (재계산)")
for name, eq in [("D_SMA42", eq_d), ("H4_SMA240", eq_h)]:
    cagr, mdd, sh, cal = _stats(eq)
    print(f"  {name}: CAGR={cagr:.1%} MDD={mdd:.1%} Sh={sh:.2f} Cal={cal:.2f}")

# combined EW
combined_eq = (eq_d.resample('1D').last().reindex(df.index, method='ffill') +
               eq_h.resample('1D').last().reindex(df.index, method='ffill')) / 2
cagr, mdd, sh, cal = _stats(combined_eq)
print(f"  EW combined: CAGR={cagr:.1%} MDD={mdd:.1%} Sh={sh:.2f} Cal={cal:.2f}")

# 저장
out = {
    'period': f'{START} ~ {END}',
    'pearson': float(pearson), 'spearman': float(spearman),
    'rolling90_mean': float(roll.mean()), 'rolling90_min': float(roll.min()), 'rolling90_max': float(roll.max()),
    'pct_above_0.7': float((roll>0.7).mean()), 'pct_below_0.3': float((roll<0.3).mean()),
}
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_member_corr.json')
with open(out_path, 'w') as f: json.dump(out, f, indent=2)
print(f"\n결과 저장: {out_path}")
