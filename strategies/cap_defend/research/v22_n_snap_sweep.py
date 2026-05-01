"""snap × n_snapshots sweep — long snap + many tracks 으로 stale 완화 가능한지 확인.

snap=240 n=3 (stagger 80d) → snap=240 n=5/7 (stagger 48/34d) 등.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START='2020-10-01'; END='2026-04-27'; ALLOC=(0.60,0.30,0.10)
bars_D, funding = ub.load_data('D')

def run_spot(snap, n=3):
    return ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
        sma_days=42, mom_short_days=20, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=0.004,
        health_mode='mom2vol', vol_mode='daily',
        snap_interval_bars=snap, start_date=START, end_date=END)['_equity']

def run_fut(snap, n=3):
    return ub.run(bars_D, funding, interval='D', asset_type='fut', leverage=3.0,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
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
def run_stock(snap, n=3):
    return tss.run_snapshot(sp, snap_days=snap, n_snap=n)['Value']

# === 코인 spot 변형 ===
SPOT_GRID = [(180,3),(180,5),(240,3),(240,5),(240,7),(360,5),(360,7),(360,9)]
print(f"=== spot {len(SPOT_GRID)} variants ===")
spot_eq = {}
for snap, n in SPOT_GRID:
    spot_eq[(snap,n)] = run_spot(snap, n)
    print(f"  sp_sn{snap}_n{n}")

# === 선물 변형 ===
FUT_GRID = [(180,3),(180,5),(240,3),(240,5),(240,7),(360,5)]
print(f"=== fut {len(FUT_GRID)} variants ===")
fut_eq = {}
for snap, n in FUT_GRID:
    fut_eq[(snap,n)] = run_fut(snap, n)
    print(f"  fu_sn{snap}_n{n}")

# === 주식 변형 ===
STOCK_GRID = [(180,3),(180,5),(180,7),(252,3),(252,5),(252,7),(360,5)]
print(f"=== stock {len(STOCK_GRID)} variants ===")
stock_eq = {}
for snap, n in STOCK_GRID:
    stock_eq[(snap,n)] = run_stock(snap, n)
    print(f"  st_sd{snap}_n{n}")

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
    return cagr, mdd, sh, cal, (min(yc) if yc else 0), tuple(round(c,2) for c in yc)
def portfolio(s,c,f,w=ALLOC):
    idx=s.index.intersection(c.index).intersection(f.index)
    return (1+w[0]*s.loc[idx].pct_change().fillna(0)+w[1]*c.loc[idx].pct_change().fillna(0)+w[2]*f.loc[idx].pct_change().fillna(0)).cumprod()

st_n = {k: daily_norm(v) for k,v in stock_eq.items()}
sp_n = {k: daily_norm(v) for k,v in spot_eq.items()}
fu_n = {k: daily_norm(v) for k,v in fut_eq.items()}

# 단독 sleeve summary
print(f"\n=== 단독 sleeve ===")
for label, d in [('spot', sp_n), ('fut', fu_n), ('stock', st_n)]:
    for k, eq in d.items():
        cagr, mdd, sh, cal, ymin, _ = stats(eq)
        print(f"  {label}_sn{k[0]}_n{k[1]:<2}  Cal={cal:.2f} CAGR={cagr:+.0%} MDD={mdd:+.0%} ymin={ymin:+.2f}")

# 모든 조합 60/30/10
rows = []
for sk, s in st_n.items():
    for ck, c in sp_n.items():
        for fk, f in fu_n.items():
            p = portfolio(s, c, f)
            cagr, mdd, sh, cal, ymin, yc = stats(p)
            rows.append(dict(
                stock=f'sd{sk[0]}_n{sk[1]}', spot=f'sn{ck[0]}_n{ck[1]}', fut=f'sn{fk[0]}_n{fk[1]}',
                cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin,
                yearly='|'.join(f'{c:.2f}' for c in yc),
            ))
df = pd.DataFrame(rows).sort_values('cal', ascending=False)

print(f"\n=== top 15 portfolio combinations (60/30/10) ===")
print(f"{'stock':12s} {'spot':12s} {'fut':12s}  Cal    CAGR    MDD     ymin    yearly")
for _, r in df.head(15).iterrows():
    print(f"  {r['stock']:11s} {r['spot']:11s} {r['fut']:11s}  {r['cal']:.2f}  {r['cagr']:+.1%}  {r['mdd']:+.1%}  {r['ymin']:+.2f}  {r['yearly']}")

print(f"\n=== top 5 by ymin ===")
for _, r in df.sort_values('ymin', ascending=False).head(5).iterrows():
    print(f"  {r['stock']:11s} {r['spot']:11s} {r['fut']:11s}  Cal={r['cal']:.2f} ymin={r['ymin']:+.2f}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_n_snap_sweep.csv')
df.to_csv(out, index=False)
print(f"\n저장: {out}, 총 {len(df)} 조합")
