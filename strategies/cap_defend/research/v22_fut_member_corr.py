"""V22 선물 두 멤버 (D_SMA42 L3 + H4_SMA240 L3) 일별 수익 상관 진단.

목적: 선물 V22 ensemble 도 spot 처럼 corr 높고 dilution 발생하는지 확인.
"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub

START = '2020-10-01'
END   = '2026-04-27'

D_CFG = dict(
    interval='D', asset_type='fut', leverage=3.0,
    sma_days=42, mom_short_days=18, mom_long_days=127,
    vol_days=90, vol_threshold=0.05,
    canary_hyst=0.015, n_snapshots=3,
    universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=90,
    start_date=START, end_date=END,
)
H4_CFG = dict(
    interval='4h', asset_type='fut', leverage=3.0,
    sma_bars=240, mom_short_bars=12*6, mom_long_bars=180*6,
    vol_days=90, vol_threshold=0.05,
    canary_hyst=0.015, n_snapshots=3,
    universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=540,  # 90일 = 540 × 4h
    start_date=START, end_date=END,
)

bars_D, funding = ub.load_data('D')
bars_4h, _ = ub.load_data('4h')

print("=== running fut D_SMA42 L3 ===")
res_d = ub.run(bars_D, funding, **D_CFG)
print("=== running fut H4_SMA240 L3 ===")
res_h = ub.run(bars_4h, funding, **H4_CFG)

eq_d = res_d['_equity']
eq_h = res_h['_equity']

ret_d = eq_d.pct_change().dropna()
ret_h_daily = eq_h.resample('1D').last().pct_change().dropna()
df = pd.DataFrame({'D': ret_d, 'H4': ret_h_daily}).dropna()

pearson = df.corr(method='pearson').iloc[0, 1]
spearman = df.corr(method='spearman').iloc[0, 1]
print(f"\n[FUT] daily return corr: pearson={pearson:.4f} spearman={spearman:.4f}")

print("\n[FUT] 연도별 pearson corr")
for yr, sub in df.groupby(df.index.year):
    if len(sub) < 30: continue
    print(f"  {yr}: {sub.corr().iloc[0,1]:.4f}  (n={len(sub)})")

roll = df['D'].rolling(90).corr(df['H4']).dropna()
print(f"\n[FUT] rolling 90d: mean={roll.mean():.4f} min={roll.min():.4f} max={roll.max():.4f}")
print(f"  >0.7 비율: {(roll>0.7).mean():.1%} / <0.3 비율: {(roll<0.3).mean():.1%}")

def _stats(eq):
    rs = eq.resample('1D').last().dropna()
    rt = rs.pct_change().dropna()
    cagr = (rs.iloc[-1]/rs.iloc[0])**(252/len(rt)) - 1
    eq_max = rs.cummax()
    mdd = ((rs - eq_max) / eq_max).min()
    sh = rt.mean() / rt.std() * np.sqrt(252) if rt.std()>0 else 0
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return cagr, mdd, sh, cal

print("\n[FUT] 단독 sleeve")
for name, eq in [("D_SMA42 L3", eq_d), ("H4_SMA240 L3", eq_h)]:
    c, m, s, ca = _stats(eq)
    print(f"  {name}: CAGR={c:.1%} MDD={m:.1%} Sh={s:.2f} Cal={ca:.2f}")

combined = (eq_d.resample('1D').last().reindex(df.index, method='ffill') +
            eq_h.resample('1D').last().reindex(df.index, method='ffill')) / 2
c, m, s, ca = _stats(combined)
print(f"  EW combined : CAGR={c:.1%} MDD={m:.1%} Sh={s:.2f} Cal={ca:.2f}")
