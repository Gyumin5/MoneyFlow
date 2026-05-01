"""V22 권고 후보 plateau check.

균형형: st_mix_63_180 + sp_sn240 + fu_sn180  Cal 4.76 ymin 1.20
공격형: st_sd63 + sp_sn240 + fu_sn180        Cal 4.83 ymin 0.95

축별 인접값 ±perturbation 으로 단조성/plateau 확인.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START='2020-10-01'; END='2026-04-27'; ALLOC=(0.60,0.30,0.10)
bars_D, funding = ub.load_data('D')

def spot_run(snap):
    return ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
        sma_days=42, mom_short_days=20, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=3,
        universe_size=3, cap=1/3, tx_cost=0.004,
        health_mode='mom2vol', vol_mode='daily',
        snap_interval_bars=snap, start_date=START, end_date=END)['_equity']

def fut_run(snap):
    return ub.run(bars_D, funding, interval='D', asset_type='fut', leverage=3.0,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=3,
        universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
        health_mode='mom2vol', vol_mode='daily',
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
def stock_run(snap, n=3):
    return tss.run_snapshot(sp, snap_days=snap, n_snap=n)['Value']

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
    for _, sub in rt.groupby(rt.index.year):
        if len(sub)<30: continue
        eyr=(1+sub).cumprod()
        myr=((eyr-eyr.cummax())/eyr.cummax()).min()
        yc.append((eyr.iloc[-1]-1)/abs(myr) if myr else 0)
    return cagr, mdd, sh, cal, (min(yc) if yc else 0), tuple(round(c,2) for c in yc)
def portfolio(s,c,f,w=ALLOC):
    idx=s.index.intersection(c.index).intersection(f.index)
    return (1+w[0]*s.loc[idx].pct_change().fillna(0)+w[1]*c.loc[idx].pct_change().fillna(0)+w[2]*f.loc[idx].pct_change().fillna(0)).cumprod()

def mix_ew(*series):
    idx=series[0].index
    for s in series[1:]: idx=idx.intersection(s.index)
    return sum(s.loc[idx] for s in series)/len(series)

# === 인접값 정의 ===
SPOT_GRID = [180, 200, 220, 240, 270, 300]
FUT_GRID  = [120, 150, 180, 200, 240]
STOCK_SHORT_GRID = [42, 50, 63, 75, 90]
STOCK_LONG_GRID  = [125, 150, 180, 220, 252]

# 각 자산 후보 BT
print("=== running spot variants ===")
spot_eq = {s: daily_norm(spot_run(s)) for s in SPOT_GRID}
for k in spot_eq: print(f"  sp_sn{k}")
print("=== running fut variants ===")
fut_eq = {s: daily_norm(fut_run(s)) for s in FUT_GRID}
for k in fut_eq: print(f"  fu_sn{k}")
print("=== running stock_short variants ===")
stock_short_eq = {s: daily_norm(stock_run(s)) for s in STOCK_SHORT_GRID}
print("=== running stock_long variants ===")
stock_long_eq = {s: daily_norm(stock_run(s)) for s in STOCK_LONG_GRID}

# === 균형형 plateau: 한 축씩 변화시켜 ===
print(f"\n=== 균형형 baseline = st_mix(sd63+sd180) + sp_sn240 + fu_sn180 ===")

# spot 축
print(f"\n[spot snap 변화] (stock=mix_63_180, fut=sn180 고정)")
base_st = mix_ew(stock_short_eq[63], stock_long_eq[180])
for s in SPOT_GRID:
    p = portfolio(base_st, spot_eq[s], fut_eq[180])
    cagr, mdd, sh, cal, ymin, _ = stats(p)
    flag = '←' if s==240 else ' '
    print(f"  sp_sn{s:3d}  Cal={cal:.2f} CAGR={cagr:+.1%} MDD={mdd:+.1%} ymin={ymin:+.2f}  {flag}")

# fut 축
print(f"\n[fut snap 변화] (stock=mix_63_180, spot=sn240 고정)")
for f in FUT_GRID:
    p = portfolio(base_st, spot_eq[240], fut_eq[f])
    cagr, mdd, sh, cal, ymin, _ = stats(p)
    flag = '←' if f==180 else ' '
    print(f"  fu_sn{f:3d}  Cal={cal:.2f} CAGR={cagr:+.1%} MDD={mdd:+.1%} ymin={ymin:+.2f}  {flag}")

# stock short 축
print(f"\n[stock short snap 변화] (long=180, spot=sn240, fut=sn180)")
for ss in STOCK_SHORT_GRID:
    base = mix_ew(stock_short_eq[ss], stock_long_eq[180])
    p = portfolio(base, spot_eq[240], fut_eq[180])
    cagr, mdd, sh, cal, ymin, _ = stats(p)
    flag = '←' if ss==63 else ' '
    print(f"  st_short={ss}  Cal={cal:.2f} CAGR={cagr:+.1%} MDD={mdd:+.1%} ymin={ymin:+.2f}  {flag}")

# stock long 축
print(f"\n[stock long snap 변화] (short=63, spot=sn240, fut=sn180)")
for sl in STOCK_LONG_GRID:
    base = mix_ew(stock_short_eq[63], stock_long_eq[sl])
    p = portfolio(base, spot_eq[240], fut_eq[180])
    cagr, mdd, sh, cal, ymin, _ = stats(p)
    flag = '←' if sl==180 else ' '
    print(f"  st_long={sl}  Cal={cal:.2f} CAGR={cagr:+.1%} MDD={mdd:+.1%} ymin={ymin:+.2f}  {flag}")
