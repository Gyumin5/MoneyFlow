"""소수 stagger 강제 + spot vs fut 의 n_snap 차별화.

규칙: snap_length = n_snap × prime → stagger = prime (소수)
- spot:  n=5, snap = 5×{37,41,47} = 185/205/235  → stagger 37/41/47
- fut:   n=7, snap = 7×{23,29,31} = 161/203/217  → stagger 23/29/31
- stock: n=3, snap = 3×{53,61,71} = 159/183/213  → stagger 53/61/71

추가 비교용으로 직전 안 (sd180_n3 + sn240_n3 + sn180_n3) 도 포함.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START='2020-10-01'; END='2026-04-27'; ALLOC=(0.60,0.30,0.10)
bars_D, funding = ub.load_data('D')

def run_spot(snap, n):
    return ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
        sma_days=42, mom_short_days=20, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=0.004,
        health_mode='mom2vol', vol_mode='daily',
        snap_interval_bars=snap, start_date=START, end_date=END)['_equity']

def run_fut(snap, n):
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
def run_stock(snap, n):
    return tss.run_snapshot(sp, snap_days=snap, n_snap=n)['Value']

# === 후보 (소수 stagger) ===
SPOT  = [(185,5),(205,5),(235,5),(247,7),(217,7), (180,3),(240,3)]   # spot n=5 우선
FUT   = [(161,7),(203,7),(217,7),(185,5),(205,5), (180,3),(240,3)]   # fut n=7 우선
STOCK = [(159,3),(183,3),(213,3),(159,5),(183,5),(180,3)]            # stock n=3 우선 + n=5 비교

print(f"=== spot {len(SPOT)} ==="); spot_eq = {(s,n): run_spot(s,n) for s,n in SPOT}
print(f"=== fut {len(FUT)} ===");   fut_eq  = {(s,n): run_fut(s,n)  for s,n in FUT}
print(f"=== stock {len(STOCK)} ==="); stock_eq = {(s,n): run_stock(s,n) for s,n in STOCK}

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

stk = {k: daily_norm(v) for k,v in stock_eq.items()}
sp_n = {k: daily_norm(v) for k,v in spot_eq.items()}
fu_n = {k: daily_norm(v) for k,v in fut_eq.items()}

# 단독 sleeve
print(f"\n=== 단독 sleeve ===")
for label, d in [('spot ', sp_n), ('fut  ', fu_n), ('stock', stk)]:
    for k, eq in d.items():
        cagr, mdd, sh, cal, ymin, _ = stats(eq)
        stagger = k[0]/k[1]
        print(f"  {label} sn{k[0]:>3d}_n{k[1]} (stagger {stagger:.0f}d) Cal={cal:.2f} CAGR={cagr:+.0%} MDD={mdd:+.0%} ymin={ymin:+.2f}")

rows = []
for sk, s in stk.items():
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

print(f"\n=== top 15 by Cal (총 {len(df)} 조합) ===")
print(f"{'stock':12s} {'spot':12s} {'fut':12s}  Cal    CAGR    MDD     ymin  yearly")
for _, r in df.head(15).iterrows():
    print(f"  {r['stock']:11s} {r['spot']:11s} {r['fut']:11s}  {r['cal']:.2f}  {r['cagr']:+.1%}  {r['mdd']:+.1%}  {r['ymin']:+.2f}  {r['yearly']}")

print(f"\n=== top 5 by ymin ===")
for _, r in df.sort_values('ymin', ascending=False).head(5).iterrows():
    print(f"  {r['stock']:11s} {r['spot']:11s} {r['fut']:11s}  Cal={r['cal']:.2f} ymin={r['ymin']:+.2f}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_prime_stagger.csv')
df.to_csv(out, index=False)
print(f"\n저장: {out}")
