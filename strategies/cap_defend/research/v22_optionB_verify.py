"""Option B verify — 차별화/서로소 제약 (spot_n ≠ fut_n) + portfolio rank-sum.

stock (n=3): sd ∈ {54, 66, 69, 96}
spot 후보 (drift, n, sn):
  n=3 d=0.20: 183, 200, 220
  n=5 d=0.15: 150, 210
  n=7 d=0.15: 210, 220
fut 후보:
  n=3 d=0.05: 78, 87, 90
  n=5 d=0.05: 96, 99
  n=7 d=0.05: 66, 72, 78

constraint: spot_n_snap ≠ fut_n_snap
"""
import os, sys, time, itertools
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

STOCK = [(sd, 3) for sd in [54, 66, 69, 96]]
SPOT = [
    (183,3,0.20),(200,3,0.20),(220,3,0.20),
    (150,5,0.15),(210,5,0.15),
    (210,7,0.15),(220,7,0.15),
]
FUT = [
    (78,3,0.05),(87,3,0.05),(90,3,0.05),
    (96,5,0.05),(99,5,0.05),
    (66,7,0.05),(72,7,0.05),(78,7,0.05),
]

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
                stock_eq[f"st_sd{sn}_n{n}"] = eq
            elif asset=='spot':
                spot_eq[(sn,n,d)] = eq
            else:
                fut_eq[(sn,n,d)] = eq
    print(f"  {time.time()-t0:.0f}s")

    # 단독 sleeve
    print("\n=== 단독 sleeve ===")
    for k, eq in stock_eq.items():
        cagr,mdd,sh,cal,ymin = stats(eq)
        print(f"  {k:20s}  Cal={cal:.2f} CAGR={cagr:+.0%} MDD={mdd:+.0%} ymin={ymin:+.2f}")
    for (sn,n,d), eq in spot_eq.items():
        cagr,mdd,sh,cal,ymin = stats(eq)
        print(f"  spot sn={sn} n={n} d={d:.2f}  Cal={cal:.2f} CAGR={cagr:+.0%} MDD={mdd:+.0%} ymin={ymin:+.2f}")
    for (sn,n,d), eq in fut_eq.items():
        cagr,mdd,sh,cal,ymin = stats(eq)
        print(f"  fut  sn={sn} n={n} d={d:.2f}  Cal={cal:.2f} CAGR={cagr:+.0%} MDD={mdd:+.0%} ymin={ymin:+.2f}")

    # portfolio (constraint: spot_n ≠ fut_n)
    rows = []
    for (sk, s) in stock_eq.items():
        for (spk, sp_eq) in spot_eq.items():
            for (fuk, fu_eq) in fut_eq.items():
                spn, fun = spk[1], fuk[1]
                if spn == fun: continue  # 차별화 강제
                for alloc in ALLOCS:
                    ws, wc, wf = alloc[0]/100, alloc[1]/100, alloc[2]/100
                    p = portfolio(s, sp_eq, fu_eq, ws, wc, wf)
                    cagr,mdd,sh,cal,ymin = stats(p)
                    rows.append(dict(
                        alloc=f"{alloc[0]}/{alloc[1]}/{alloc[2]}",
                        stock=sk,
                        spot=f"sn{spk[0]}_n{spk[1]}_d{spk[2]:.2f}",
                        fut=f"sn{fuk[0]}_n{fuk[1]}_d{fuk[2]:.2f}",
                        spot_n=spn, fut_n=fun,
                        cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin,
                    ))
    df = pd.DataFrame(rows)
    print(f"\n=== portfolio combos (after spot_n!=fut_n filter): {len(df)} ===")
    base = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(base,'v22_optionB_verify.csv'), index=False)

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
            avg_cagr=('cagr','mean'),
            avg_mdd=('mdd','mean'),
        ).round(3)
        return g.sort_values('avg_rank_sum')

    print(f"\n=== 자산별 후보 rank summary ===")
    for col in ['stock','spot','fut']:
        print(f"\n-- {col} --")
        print(asset_summary(col).to_string())

    # n_snap 페어별 평균
    print(f"\n=== (spot_n, fut_n) 페어 평균 ===")
    pair = df.groupby(['spot_n','fut_n']).agg(
        avg_cal=('cal','mean'),
        avg_ymin=('ymin','mean'),
        avg_rank_sum=('rank_sum','mean'),
    ).round(3).sort_values('avg_rank_sum')
    print(pair.to_string())

    # alloc 별 best
    print(f"\n=== alloc 별 best (Cal 1위) ===")
    for alloc in ALLOCS:
        al = f"{alloc[0]}/{alloc[1]}/{alloc[2]}"
        sub = df[df['alloc']==al].sort_values('cal',ascending=False).head(1)
        if not len(sub): continue
        r = sub.iloc[0]
        print(f"  {al:10s}  {r['stock']:14s}  {r['spot']:22s}  {r['fut']:22s}  Cal={r['cal']:.2f}  ymin={r['ymin']:+.2f}")

    print(f"\n=== alloc 별 best (ymin 1위) ===")
    for alloc in ALLOCS:
        al = f"{alloc[0]}/{alloc[1]}/{alloc[2]}"
        sub = df[df['alloc']==al].sort_values('ymin',ascending=False).head(1)
        if not len(sub): continue
        r = sub.iloc[0]
        print(f"  {al:10s}  {r['stock']:14s}  {r['spot']:22s}  {r['fut']:22s}  Cal={r['cal']:.2f} ymin={r['ymin']:+.2f}")

    # 최종 단일 추천 (자산별 rank-sum 1위)
    print(f"\n=== 최종 단일 추천 (자산별 rank-sum 최저) ===")
    for col in ['stock','spot','fut']:
        s = asset_summary(col).iloc[0]
        print(f"  {col}: {asset_summary(col).index[0]:30s}  rank_sum={s['avg_rank_sum']:.1f}  Cal={s['avg_cal']:.2f} ymin={s['avg_ymin']:+.2f}")

if __name__ == '__main__':
    main()
