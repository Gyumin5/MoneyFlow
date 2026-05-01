"""엄격한 n*prime 제약 + 차별화·서로소 강제 최종 검증.

후보 (모두 stagger = snap/n 가 소수)
- stock n=3: sd ∈ {51(3*17), 57(3*19), 69(3*23), 87(3*29), 93(3*31)}
        n=5: {55(5*11), 65(5*13), 85(5*17), 95(5*19)}
        n=7: {77(7*11), 91(7*13)}
- spot drift ∈ {0.10, 0.15, 0.20}
        n=3: {177(3*59), 183(3*61), 201(3*67), 213(3*71), 237(3*79)}
        n=5: {185(5*37), 205(5*41), 215(5*43), 235(5*47)}
        n=7: {161(7*23), 203(7*29), 217(7*31), 259(7*37)}
- fut drift ∈ {0.00, 0.05, 0.10}
        n=3: {57(3*19), 69(3*23), 87(3*29), 93(3*31)}
        n=5: {55(5*11), 65(5*13), 85(5*17), 95(5*19)}
        n=7: {77(7*11), 91(7*13)}

constraint: spot_n ≠ fut_n (서로소)
"""
import os, sys, time
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

START='2020-10-01'; END='2026-04-27'
_g = {}
def _init():
    import unified_backtest as ub, stock_engine as ts, stock_engine_snap as tss
    bars_D, funding = ub.load_data('D')
    OFF=('SPY','QQQ','VEA','EEM','EWJ','GLD','PDBC')
    DEF=('IEF','BIL','BNDX','GLD','PDBC'); CAN=('EEM',)
    ts._g_prices = ts.load_prices(list(set(OFF+DEF+CAN)), start='2014-01-01')
    ts._g_ind = ts.precompute(ts._g_prices)
    sp = ts.SP(offensive=OFF, defensive=DEF, canary_assets=CAN,
        canary_sma=300, canary_hyst=0.020, canary_type='sma',
        health='none', defense='top2', defense_sma=100, def_mom_period=126,
        select='zscore3', n_mom=3, n_sh=3, sharpe_lookback=126,
        weight='ew', crash='none', tx_cost=0.001, start=START, end=END, capital=10000.0)
    _g.update(ub=ub, tss=tss, bars_D=bars_D, funding=funding, sp=sp)

def _spot(args):
    sn, n, d = args
    eq = _g['ub'].run(_g['bars_D'], _g['funding'], interval='D', asset_type='spot', leverage=1.0,
        sma_days=42, mom_short_days=20, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=0.004,
        health_mode='mom2vol', vol_mode='daily', drift_threshold=d,
        snap_interval_bars=sn, start_date=START, end_date=END)['_equity']
    s = eq.resample('1D').last().dropna()
    return ('spot', sn, n, d), s/s.iloc[0]

def _fut(args):
    sn, n, d = args
    eq = _g['ub'].run(_g['bars_D'], _g['funding'], interval='D', asset_type='fut', leverage=3.0,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
        health_mode='mom2vol', vol_mode='daily', drift_threshold=d,
        snap_interval_bars=sn, start_date=START, end_date=END)['_equity']
    s = eq.resample('1D').last().dropna()
    return ('fut', sn, n, d), s/s.iloc[0]

def _stock(args):
    sn, n = args
    eq = _g['tss'].run_snapshot(_g['sp'], snap_days=sn, n_snap=n)['Value']
    s = eq.resample('1D').last().dropna()
    return ('stock', sn, n, None), s/s.iloc[0]

STOCK = (
    [(sd, 3) for sd in [51, 57, 69, 87]] +
    [(sd, 5) for sd in [65, 85, 95]] +
    [(sd, 7) for sd in [77, 91]]
)

SPOT_BASE = (
    [(sn, 3) for sn in [177, 183, 213, 237]] +
    [(sn, 5) for sn in [185, 235]] +
    [(sn, 7) for sn in [203, 217, 259]]
)
SPOT_DRIFT = [0.10, 0.15, 0.20]
SPOT = [(sn, n, d) for (sn, n) in SPOT_BASE for d in SPOT_DRIFT]

FUT_BASE = (
    [(sn, 3) for sn in [57, 69, 87, 93]] +
    [(sn, 5) for sn in [65, 85, 95]] +
    [(sn, 7) for sn in [77, 91]]
)
FUT_DRIFT = [0.00, 0.05, 0.10]
FUT = [(sn, n, d) for (sn, n) in FUT_BASE for d in FUT_DRIFT]

ALLOCS = [
    (80,20,0),(70,30,0),(60,40,0),(50,50,0),
    (60,35,5),(60,30,10),(60,25,15),(60,20,20),
    (50,40,10),(50,30,20),(50,25,25),
    (70,20,10),(80,10,10),(40,40,20),(33,33,34),
]

def stats(eq):
    rt = eq.pct_change().dropna()
    days = (eq.index[-1]-eq.index[0]).days
    cagr = (eq.iloc[-1]/eq.iloc[0])**(365.25/days)-1 if days>0 else 0
    mdd = ((eq-eq.cummax())/eq.cummax()).min()
    sh = rt.mean()/rt.std()*np.sqrt(252) if rt.std()>0 else 0
    cal = cagr/abs(mdd) if mdd!=0 else 0
    yc=[]
    for _,sub in rt.groupby(rt.index.year):
        if len(sub)<30: continue
        eyr=(1+sub).cumprod(); myr=((eyr-eyr.cummax())/eyr.cummax()).min()
        yc.append((eyr.iloc[-1]-1)/abs(myr) if myr else 0)
    return cagr, mdd, sh, cal, (min(yc) if yc else 0)
def portfolio(s,c,f,ws,wc,wf):
    idx = s.index.intersection(c.index).intersection(f.index)
    rs = s.loc[idx].pct_change().fillna(0)
    rc = c.loc[idx].pct_change().fillna(0)
    rf = f.loc[idx].pct_change().fillna(0)
    return (1 + ws*rs + wc*rc + wf*rf).cumprod()

def main():
    n_workers = min(8, os.cpu_count() or 4)
    print(f"=== sleeve BTs: stock {len(STOCK)} + spot {len(SPOT)} + fut {len(FUT)} (workers={n_workers}) ===")
    stock_eq = {}; spot_eq = {}; fut_eq = {}
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init) as ex:
        futs = []
        for a in STOCK: futs.append(ex.submit(_stock, a))
        for a in SPOT:  futs.append(ex.submit(_spot, a))
        for a in FUT:   futs.append(ex.submit(_fut, a))
        for f in futs:
            (asset, sn, n, d), eq = f.result()
            if asset=='stock':
                stock_eq[(sn,n)] = eq
            elif asset=='spot':
                spot_eq[(sn,n,d)] = eq
            else:
                fut_eq[(sn,n,d)] = eq
    print(f"  {time.time()-t0:.0f}s")

    # 단독 sleeve top
    rows_s = []
    for k, eq in stock_eq.items():
        cagr,mdd,sh,cal,ymin = stats(eq)
        rows_s.append(dict(asset='stock', sn=k[0], n=k[1], drift=None, cal=cal, ymin=ymin, cagr=cagr, mdd=mdd))
    for k, eq in spot_eq.items():
        cagr,mdd,sh,cal,ymin = stats(eq)
        rows_s.append(dict(asset='spot', sn=k[0], n=k[1], drift=k[2], cal=cal, ymin=ymin, cagr=cagr, mdd=mdd))
    for k, eq in fut_eq.items():
        cagr,mdd,sh,cal,ymin = stats(eq)
        rows_s.append(dict(asset='fut', sn=k[0], n=k[1], drift=k[2], cal=cal, ymin=ymin, cagr=cagr, mdd=mdd))
    sleeve = pd.DataFrame(rows_s)
    base = os.path.dirname(os.path.abspath(__file__))
    sleeve.to_csv(os.path.join(base,'v22_strict_sleeve.csv'), index=False)

    print(f"\n=== STOCK sleeve (n*prime, n=3/5/7) ===")
    for _, r in sleeve[sleeve.asset=='stock'].sort_values('cal',ascending=False).iterrows():
        sd, n = r['sn'], r['n']
        prime = sd//n
        print(f"  sd={sd:>3} n={n}  stagger={prime:>2}  Cal={r['cal']:.2f}  CAGR={r['cagr']:+.0%}  MDD={r['mdd']:+.0%}  ymin={r['ymin']:+.2f}")

    print(f"\n=== SPOT top10 by Cal ===")
    for _, r in sleeve[sleeve.asset=='spot'].sort_values('cal',ascending=False).head(10).iterrows():
        sn, n, d = r['sn'], r['n'], r['drift']
        prime = sn//n
        print(f"  sn={sn:>3} n={n} d={d:.2f}  stagger={prime:>2}  Cal={r['cal']:.2f}  ymin={r['ymin']:+.2f}")
    print(f"\n=== SPOT top5 by ymin ===")
    for _, r in sleeve[sleeve.asset=='spot'].sort_values('ymin',ascending=False).head(5).iterrows():
        sn, n, d = r['sn'], r['n'], r['drift']
        prime = sn//n
        print(f"  sn={sn:>3} n={n} d={d:.2f}  stagger={prime:>2}  Cal={r['cal']:.2f}  ymin={r['ymin']:+.2f}")

    print(f"\n=== FUT top10 by Cal ===")
    for _, r in sleeve[sleeve.asset=='fut'].sort_values('cal',ascending=False).head(10).iterrows():
        sn, n, d = r['sn'], r['n'], r['drift']
        prime = sn//n
        print(f"  sn={sn:>3} n={n} d={d:.2f}  stagger={prime:>2}  Cal={r['cal']:.2f}  ymin={r['ymin']:+.2f}")
    print(f"\n=== FUT top5 by ymin ===")
    for _, r in sleeve[sleeve.asset=='fut'].sort_values('ymin',ascending=False).head(5).iterrows():
        sn, n, d = r['sn'], r['n'], r['drift']
        prime = sn//n
        print(f"  sn={sn:>3} n={n} d={d:.2f}  stagger={prime:>2}  Cal={r['cal']:.2f}  ymin={r['ymin']:+.2f}")

    # portfolio (constraint: spot_n ≠ fut_n)
    rows = []
    for sk, s in stock_eq.items():
        for spk, sp_e in spot_eq.items():
            for fuk, fu_e in fut_eq.items():
                if spk[1] == fuk[1]: continue  # 서로소·차별화
                for alloc in ALLOCS:
                    ws, wc, wf = alloc[0]/100, alloc[1]/100, alloc[2]/100
                    p = portfolio(s, sp_e, fu_e, ws, wc, wf)
                    cagr,mdd,sh,cal,ymin = stats(p)
                    rows.append(dict(
                        alloc=f"{alloc[0]}/{alloc[1]}/{alloc[2]}",
                        stock=f"sd{sk[0]}_n{sk[1]}",
                        spot=f"sn{spk[0]}_n{spk[1]}_d{spk[2]:.2f}",
                        fut=f"sn{fuk[0]}_n{fuk[1]}_d{fuk[2]:.2f}",
                        spot_n=spk[1], fut_n=fuk[1],
                        cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin,
                    ))
    df = pd.DataFrame(rows)
    print(f"\n=== portfolio combos (filter spot_n!=fut_n): {len(df)} ===")
    df.to_csv(os.path.join(base,'v22_strict_portfolio.csv'), index=False)

    df['rank_cal']  = df.groupby('alloc')['cal'].rank(ascending=False, method='min')
    df['rank_ymin'] = df.groupby('alloc')['ymin'].rank(ascending=False, method='min')
    df['rank_sum']  = df['rank_cal'] + df['rank_ymin']

    def asset_summary(col):
        g = df.groupby(col).agg(
            avg_rank_cal=('rank_cal','mean'),
            avg_rank_ymin=('rank_ymin','mean'),
            avg_rank_sum=('rank_sum','mean'),
            avg_cal=('cal','mean'),
            avg_ymin=('ymin','mean'),
        ).round(3)
        return g.sort_values('avg_rank_sum')

    print(f"\n=== 자산별 rank summary (top8) ===")
    for col in ['stock','spot','fut']:
        print(f"\n-- {col} --")
        print(asset_summary(col).head(8).to_string())

    print(f"\n=== (spot_n, fut_n) 페어 평균 ===")
    pair = df.groupby(['spot_n','fut_n']).agg(
        avg_cal=('cal','mean'),
        avg_ymin=('ymin','mean'),
        avg_rank_sum=('rank_sum','mean'),
    ).round(3).sort_values('avg_rank_sum')
    print(pair.to_string())

    print(f"\n=== alloc 별 best (Cal 1위) ===")
    for alloc in ALLOCS:
        al = f"{alloc[0]}/{alloc[1]}/{alloc[2]}"
        sub = df[df['alloc']==al].sort_values('cal',ascending=False).head(1)
        if not len(sub): continue
        r = sub.iloc[0]
        print(f"  {al:10s}  {r['stock']:12s}  {r['spot']:22s}  {r['fut']:22s}  Cal={r['cal']:.2f} ymin={r['ymin']:+.2f}")

    print(f"\n=== alloc 별 best (ymin 1위) ===")
    for alloc in ALLOCS:
        al = f"{alloc[0]}/{alloc[1]}/{alloc[2]}"
        sub = df[df['alloc']==al].sort_values('ymin',ascending=False).head(1)
        if not len(sub): continue
        r = sub.iloc[0]
        print(f"  {al:10s}  {r['stock']:12s}  {r['spot']:22s}  {r['fut']:22s}  Cal={r['cal']:.2f} ymin={r['ymin']:+.2f}")

    print(f"\n=== 최종 단일 추천 (자산별 rank-sum 1위) ===")
    for col in ['stock','spot','fut']:
        cand = asset_summary(col).index[0]
        s = asset_summary(col).iloc[0]
        print(f"  {col}: {cand:30s}  rank_sum={s['avg_rank_sum']:.1f}  avg_Cal={s['avg_cal']:.2f}  avg_ymin={s['avg_ymin']:+.2f}")

if __name__ == '__main__':
    main()
