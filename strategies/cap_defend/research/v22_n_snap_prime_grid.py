"""n_snap × prime stagger 그리드 — spot vs fut 차별화.

권고 baseline: spot sn=240 d=0.20, fut sn=180 d=0.05, stock sd=180 n=3.
변경: 각 자산 sn = n_snap × prime (가까운 값), spot vs fut 다른 n_snap.

stock 은 고정 (n=3, sd=183=3*61) 으로 단순화.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START='2020-10-01'; END='2026-04-27'; ALLOC=(0.60,0.30,0.10)
bars_D, funding = ub.load_data('D')

def run_spot(snap, drift, n):
    return ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
        sma_days=42, mom_short_days=20, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=0.004,
        health_mode='mom2vol', vol_mode='daily', drift_threshold=drift,
        snap_interval_bars=snap, start_date=START, end_date=END)['_equity']

def run_fut(snap, drift, n):
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

# Spot 후보 (drift=0.20, sn 약 240 근방, n × prime)
SPOT = [
    (3, 237),  # 3*79 stagger 79
    (5, 235),  # 5*47 stagger 47
    (5, 265),  # 5*53 stagger 53
    (7, 217),  # 7*31 stagger 31
    (7, 259),  # 7*37 stagger 37
    (11, 253), # 11*23 stagger 23
    (13, 247), # 13*19 stagger 19
]

# Fut 후보 (drift=0.05, sn 약 180 근방)
FUT = [
    (3, 183),  # 3*61 stagger 61
    (5, 185),  # 5*37 stagger 37
    (7, 161),  # 7*23 stagger 23
    (7, 203),  # 7*29 stagger 29
    (11, 187), # 11*17 stagger 17
    (13, 169), # 13*13 (사실상 정사각수, 비추) -> use 13*17=221 instead
]
FUT[-1] = (13, 221)  # 13*17 stagger 17

# Stock 후보 (n=3 고정, sn 약 180 근방)
STOCK = [
    (3, 183),  # 3*61
]

print(f"=== spot {len(SPOT)} candidates (drift=0.20) ===")
spot_eq = {}
for n, sn in SPOT:
    spot_eq[(n, sn)] = run_spot(sn, 0.20, n)
    print(f"  spot n={n} sn={sn} stagger={sn//n}")

print(f"\n=== fut {len(FUT)} candidates (drift=0.05) ===")
fut_eq = {}
for n, sn in FUT:
    fut_eq[(n, sn)] = run_fut(sn, 0.05, n)
    print(f"  fut  n={n} sn={sn} stagger={sn//n}")

print(f"\n=== stock {len(STOCK)} candidates ===")
stock_eq = {(n, sn): tss.run_snapshot(sp, snap_days=sn, n_snap=n)['Value'] for n, sn in STOCK}

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
def portfolio(s,c,f,w=ALLOC):
    idx=s.index.intersection(c.index).intersection(f.index)
    return (1+w[0]*s.loc[idx].pct_change().fillna(0)+w[1]*c.loc[idx].pct_change().fillna(0)+w[2]*f.loc[idx].pct_change().fillna(0)).cumprod()

# 단독 sleeve
print(f"\n=== 단독 sleeve ===")
for label, d in [('spot', spot_eq), ('fut', fut_eq), ('stock', stock_eq)]:
    for k, eq in d.items():
        eq_n = daily_norm(eq)
        cagr, mdd, sh, cal, ymin = stats(eq_n)
        print(f"  {label}_n{k[0]}_sn{k[1]} (stagger {k[1]//k[0]:>2d}d) Cal={cal:.2f} CAGR={cagr:+.0%} MDD={mdd:+.0%} ymin={ymin:+.2f}")

# 모든 조합
sp_n = {k: daily_norm(v) for k,v in spot_eq.items()}
fu_n = {k: daily_norm(v) for k,v in fut_eq.items()}
st_n = {k: daily_norm(v) for k,v in stock_eq.items()}

rows = []
for sk, s in st_n.items():
    for ck, c in sp_n.items():
        for fk, f in fu_n.items():
            p = portfolio(s, c, f)
            cagr, mdd, sh, cal, ymin = stats(p)
            rows.append(dict(
                stock=f'sd{sk[1]}_n{sk[0]}',
                spot=f'sn{ck[1]}_n{ck[0]}',
                fut=f'sn{fk[1]}_n{fk[0]}',
                cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin,
            ))
df = pd.DataFrame(rows).sort_values('cal', ascending=False)

print(f"\n=== top 10 portfolio (60/30/10), 총 {len(df)} 조합 ===")
print(f"{'stock':14s} {'spot':14s} {'fut':14s}  Cal    CAGR    MDD     Sh    ymin")
for _, r in df.head(10).iterrows():
    print(f"  {r['stock']:13s}  {r['spot']:13s}  {r['fut']:13s}  {r['cal']:.2f}  {r['cagr']:+.1%}  {r['mdd']:+.1%}  {r['sh']:.2f}  {r['ymin']:+.2f}")

print(f"\n=== top 5 by ymin ===")
for _, r in df.sort_values('ymin', ascending=False).head(5).iterrows():
    print(f"  {r['stock']:13s}  {r['spot']:13s}  {r['fut']:13s}  Cal={r['cal']:.2f} ymin={r['ymin']:+.2f}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_n_snap_prime_grid.csv')
df.to_csv(out, index=False)
print(f"\n저장: {out}")
