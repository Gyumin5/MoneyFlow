"""각 자산별 (snap_period × n_snap × drift) 3D 그리드 BT — 자산 단독 최적 조합 결정.

목적: drift–snap 상호작용을 보면서 각 자산의 최적 (snap, n_snap, drift) 결정.
- spot: snap [60,120,180,240,300] × n [3,5,7] × drift [0, 0.05, 0.10, 0.20] = 60 BT
- fut:  snap [60,90,120,180,240]  × n [3,5,7] × drift [0, 0.05, 0.10, 0.20] = 60 BT
- stock: sd [63,125,180,252] × n [3,5] = 8 BT (drift 미지원)
"""
import os, sys, itertools, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START='2020-10-01'; END='2026-04-27'
bars_D, funding = ub.load_data('D')

def run_spot(snap, n, drift):
    return ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
        sma_days=42, mom_short_days=20, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=0.004,
        health_mode='mom2vol', vol_mode='daily', drift_threshold=drift,
        snap_interval_bars=snap, start_date=START, end_date=END)['_equity']

def run_fut(snap, n, drift):
    return ub.run(bars_D, funding, interval='D', asset_type='fut', leverage=3.0,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
        health_mode='mom2vol', vol_mode='daily', drift_threshold=drift,
        snap_interval_bars=snap, start_date=START, end_date=END)['_equity']

OFF=('SPY','QQQ','VEA','EEM','EWJ','GLD','PDBC')
DEF=('IEF','BIL','BNDX','GLD','PDBC'); CAN=('EEM',)
ts._g_prices = ts.load_prices(list(set(OFF+DEF+CAN)), start='2014-01-01')
ts._g_ind = ts.precompute(ts._g_prices)
sp = ts.SP(offensive=OFF, defensive=DEF, canary_assets=CAN,
    canary_sma=300, canary_hyst=0.020, canary_type='sma',
    health='none', defense='top2', defense_sma=100, def_mom_period=126,
    select='zscore3', n_mom=3, n_sh=3, sharpe_lookback=126,
    weight='ew', crash='none',
    tx_cost=0.001, start=START, end=END, capital=10000.0)
def run_stock(sn, n): return tss.run_snapshot(sp, snap_days=sn, n_snap=n)['Value']

SPOT_SN = [60, 120, 180, 240, 300]
SPOT_N  = [3, 5, 7]
SPOT_D  = [0.0, 0.05, 0.10, 0.20]

FUT_SN = [60, 90, 120, 180, 240]
FUT_N  = [3, 5, 7]
FUT_D  = [0.0, 0.05, 0.10, 0.20]

STOCK_SN = [63, 125, 180, 252]
STOCK_N  = [3, 5]

def daily_norm(eq):
    s = eq.resample('1D').last().dropna()
    return s/s.iloc[0]
def stats(eq):
    rt = eq.pct_change().dropna()
    n = len(rt)
    cagr = (eq.iloc[-1]/eq.iloc[0])**(252/n)-1
    mdd = ((eq-eq.cummax())/eq.cummax()).min()
    sh = rt.mean()/rt.std()*np.sqrt(252) if rt.std()>0 else 0
    cal = cagr/abs(mdd) if mdd!=0 else 0
    yc=[]
    for _,sub in rt.groupby(rt.index.year):
        if len(sub)<30: continue
        eyr=(1+sub).cumprod(); myr=((eyr-eyr.cummax())/eyr.cummax()).min()
        yc.append((eyr.iloc[-1]-1)/abs(myr) if myr else 0)
    return cagr, mdd, sh, cal, (min(yc) if yc else 0)

print(f"=== spot {len(SPOT_SN)*len(SPOT_N)*len(SPOT_D)} BTs ===")
spot_rows = []
t0 = time.time()
for sn, n, d in itertools.product(SPOT_SN, SPOT_N, SPOT_D):
    eq = daily_norm(run_spot(sn, n, d))
    cagr, mdd, sh, cal, ymin = stats(eq)
    spot_rows.append(dict(snap=sn, n_snap=n, drift=d, cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin))
print(f"  spot {time.time()-t0:.0f}s")

print(f"=== fut {len(FUT_SN)*len(FUT_N)*len(FUT_D)} BTs ===")
fut_rows = []
t0 = time.time()
for sn, n, d in itertools.product(FUT_SN, FUT_N, FUT_D):
    eq = daily_norm(run_fut(sn, n, d))
    cagr, mdd, sh, cal, ymin = stats(eq)
    fut_rows.append(dict(snap=sn, n_snap=n, drift=d, cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin))
print(f"  fut {time.time()-t0:.0f}s")

print(f"=== stock {len(STOCK_SN)*len(STOCK_N)} BTs ===")
stock_rows = []
for sn, n in itertools.product(STOCK_SN, STOCK_N):
    eq = daily_norm(run_stock(sn, n))
    cagr, mdd, sh, cal, ymin = stats(eq)
    stock_rows.append(dict(snap=sn, n_snap=n, cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin))

sp_df = pd.DataFrame(spot_rows)
fu_df = pd.DataFrame(fut_rows)
st_df = pd.DataFrame(stock_rows)

base = os.path.dirname(os.path.abspath(__file__))
sp_df.to_csv(os.path.join(base,'v22_per_asset_3d_spot.csv'), index=False)
fu_df.to_csv(os.path.join(base,'v22_per_asset_3d_fut.csv'), index=False)
st_df.to_csv(os.path.join(base,'v22_per_asset_3d_stock.csv'), index=False)

# 자산별 top10 (Cal) 와 top10 (ymin)
print(f"\n=== SPOT top10 by Cal ===")
print(f"{'snap':>5} {'n':>3} {'drift':>6}  Cal   CAGR    MDD    Sh   ymin")
for _, r in sp_df.sort_values('cal', ascending=False).head(10).iterrows():
    print(f"  {r['snap']:>3.0f}  {r['n_snap']:>3.0f}  {r['drift']:.2f}  {r['cal']:.2f}  {r['cagr']:+.0%}  {r['mdd']:+.0%}  {r['sh']:.2f}  {r['ymin']:+.2f}")

print(f"\n=== SPOT top10 by ymin ===")
for _, r in sp_df.sort_values('ymin', ascending=False).head(10).iterrows():
    print(f"  {r['snap']:>3.0f}  {r['n_snap']:>3.0f}  {r['drift']:.2f}  {r['cal']:.2f}  {r['cagr']:+.0%}  {r['mdd']:+.0%}  {r['sh']:.2f}  {r['ymin']:+.2f}")

print(f"\n=== FUT top10 by Cal ===")
for _, r in fu_df.sort_values('cal', ascending=False).head(10).iterrows():
    print(f"  {r['snap']:>3.0f}  {r['n_snap']:>3.0f}  {r['drift']:.2f}  {r['cal']:.2f}  {r['cagr']:+.0%}  {r['mdd']:+.0%}  {r['sh']:.2f}  {r['ymin']:+.2f}")

print(f"\n=== FUT top10 by ymin ===")
for _, r in fu_df.sort_values('ymin', ascending=False).head(10).iterrows():
    print(f"  {r['snap']:>3.0f}  {r['n_snap']:>3.0f}  {r['drift']:.2f}  {r['cal']:.2f}  {r['cagr']:+.0%}  {r['mdd']:+.0%}  {r['sh']:.2f}  {r['ymin']:+.2f}")

print(f"\n=== STOCK all ===")
for _, r in st_df.sort_values('cal', ascending=False).iterrows():
    print(f"  sd={r['snap']:>3.0f}  n={r['n_snap']:>3.0f}  Cal={r['cal']:.2f}  CAGR={r['cagr']:+.0%}  MDD={r['mdd']:+.0%}  Sh={r['sh']:.2f}  ymin={r['ymin']:+.2f}")

# drift 별 (snap×n) heatmap-like 요약
def drift_marginal(df, label):
    print(f"\n=== {label} drift marginal (Cal mean across snap×n) ===")
    print(df.groupby('drift')[['cal','ymin','cagr','mdd']].mean().round(3).to_string())
    print(f"\n=== {label} drift marginal (ymin mean) ===")
    print(df.groupby('drift')['ymin'].mean().round(3).to_string())
    print(f"\n=== {label} (snap, n_snap) marginal — best drift per cell, Cal ===")
    best = df.loc[df.groupby(['snap','n_snap'])['cal'].idxmax()].copy()
    print(best[['snap','n_snap','drift','cal','ymin','cagr','mdd']].to_string(index=False))
    print(f"\n=== {label} (snap, n_snap) marginal — best drift per cell, ymin ===")
    best_y = df.loc[df.groupby(['snap','n_snap'])['ymin'].idxmax()].copy()
    print(best_y[['snap','n_snap','drift','cal','ymin','cagr','mdd']].to_string(index=False))

drift_marginal(sp_df, 'SPOT')
drift_marginal(fu_df, 'FUT')
