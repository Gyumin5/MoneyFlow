"""범용 best 후보 — alloc × 자산조합 rank-sum. 병렬 실행.

제약: snap_days = n_snap × prime
변수: snap, n_snap, drift (spot/fut). stock 은 (snap, n) only.

drift grid (확장)
- spot: {0.05, 0.10, 0.15, 0.20}
- fut:  {0.00, 0.05, 0.10, 0.15, 0.20}
"""
import os, sys, itertools, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd

START='2020-10-01'; END='2026-04-27'

# ---- worker init/funcs ----
_g = {}
def _init():
    import unified_backtest as ub
    import stock_engine as ts
    import stock_engine_snap as tss
    bars_D, funding = ub.load_data('D')
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
    _g['ub'] = ub; _g['tss'] = tss; _g['bars_D'] = bars_D; _g['funding'] = funding; _g['sp'] = sp

def _spot(args):
    sn, n, d = args
    ub = _g['ub']
    eq = ub.run(_g['bars_D'], _g['funding'], interval='D', asset_type='spot', leverage=1.0,
        sma_days=42, mom_short_days=20, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=0.004,
        health_mode='mom2vol', vol_mode='daily', drift_threshold=d,
        snap_interval_bars=sn, start_date=START, end_date=END)['_equity']
    s = eq.resample('1D').last().dropna()
    return (sn, n, d), (s/s.iloc[0])

def _fut(args):
    sn, n, d = args
    ub = _g['ub']
    eq = ub.run(_g['bars_D'], _g['funding'], interval='D', asset_type='fut', leverage=3.0,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
        health_mode='mom2vol', vol_mode='daily', drift_threshold=d,
        snap_interval_bars=sn, start_date=START, end_date=END)['_equity']
    s = eq.resample('1D').last().dropna()
    return (sn, n, d), (s/s.iloc[0])

def _stock(args):
    sn, n = args
    eq = _g['tss'].run_snapshot(_g['sp'], snap_days=sn, n_snap=n)['Value']
    s = eq.resample('1D').last().dropna()
    return (sn, n), (s/s.iloc[0])

# ---- candidates (snap = n × prime) ----
STOCK = [(69,3),(129,3),(183,3),(249,3)]
_SPOT_SN_N = [(183,3),(237,3),(185,5),(235,5),(203,7),(259,7)]
_FUT_SN_N  = [(87,3),(93,3),(183,3),(85,5),(95,5),(185,5),(91,7),(161,7)]
SPOT_DRIFT = [0.05, 0.10, 0.15, 0.20]
FUT_DRIFT  = [0.00, 0.05, 0.10, 0.15, 0.20]
SPOT = [(sn,n,d) for (sn,n) in _SPOT_SN_N for d in SPOT_DRIFT]
FUT  = [(sn,n,d) for (sn,n) in _FUT_SN_N  for d in FUT_DRIFT]

ALLOCS = [
    (80,20,0),(70,30,0),(60,40,0),(50,50,0),
    (60,35,5),(60,30,10),(60,25,15),(60,20,20),
    (50,40,10),(50,30,20),(50,25,25),
    (70,20,10),(80,10,10),(40,40,20),(33,33,34),
]

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
def portfolio(s,c,f,ws,wc,wf):
    idx = s.index.intersection(c.index).intersection(f.index)
    rs = s.loc[idx].pct_change().fillna(0)
    rc = c.loc[idx].pct_change().fillna(0)
    rf = f.loc[idx].pct_change().fillna(0)
    return (1 + ws*rs + wc*rc + wf*rf).cumprod()

def main():
    print(f"=== sleeve BTs: stock {len(STOCK)}, spot {len(SPOT)}, fut {len(FUT)} ===")
    nworkers = min(8, os.cpu_count() or 4)
    print(f"  workers={nworkers}")

    stock_eq = {}
    spot_eq  = {}
    fut_eq   = {}

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=nworkers, initializer=_init) as ex:
        futs = []
        for args in STOCK:
            futs.append(('st', args, ex.submit(_stock, args)))
        for args in SPOT:
            futs.append(('sp', args, ex.submit(_spot, args)))
        for args in FUT:
            futs.append(('fu', args, ex.submit(_fut, args)))
        for kind, args, f in futs:
            key, ser = f.result()
            if kind=='st':
                stock_eq[f"st_sd{key[0]}_n{key[1]}"] = ser
            elif kind=='sp':
                stock_label = f"sp_sn{key[0]}_n{key[1]}_d{key[2]:.2f}"
                spot_eq[stock_label] = ser
            else:
                fut_eq[f"fu_sn{key[0]}_n{key[1]}_d{key[2]:.2f}"] = ser
    print(f"  total sleeve BT {time.time()-t0:.0f}s")

    # 단독 sleeve metrics
    sleeve_rows = []
    for label, d in [('stock',stock_eq),('spot',spot_eq),('fut',fut_eq)]:
        for k, eq in d.items():
            cagr, mdd, sh, cal, ymin = stats(eq)
            sleeve_rows.append(dict(asset=label, name=k, cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin))
    sleeve_df = pd.DataFrame(sleeve_rows)
    print(f"\n=== 단독 sleeve top per asset ===")
    for asset in ['stock','spot','fut']:
        sub = sleeve_df[sleeve_df['asset']==asset].sort_values('cal',ascending=False).head(5)
        print(f"  -- {asset} top5 by Cal --")
        for _, r in sub.iterrows():
            print(f"    {r['name']:30s} Cal={r['cal']:.2f} CAGR={r['cagr']:+.0%} MDD={r['mdd']:+.0%} ymin={r['ymin']:+.2f}")
        sub2 = sleeve_df[sleeve_df['asset']==asset].sort_values('ymin',ascending=False).head(3)
        print(f"  -- {asset} top3 by ymin --")
        for _, r in sub2.iterrows():
            print(f"    {r['name']:30s} Cal={r['cal']:.2f} ymin={r['ymin']:+.2f}")

    # 전체 (st × sp × fu × alloc) 조합 portfolio
    n_combos = len(STOCK)*len(SPOT)*len(FUT)*len(ALLOCS)
    print(f"\n=== portfolio combos: {n_combos} ===")
    t0 = time.time()
    rows = []
    for (sk, s), (ck, c), (fk, f) in itertools.product(stock_eq.items(), spot_eq.items(), fut_eq.items()):
        for alloc in ALLOCS:
            ws, wc, wf = alloc[0]/100, alloc[1]/100, alloc[2]/100
            p = portfolio(s, c, f, ws, wc, wf)
            cagr, mdd, sh, cal, ymin = stats(p)
            rows.append(dict(
                alloc=f"{alloc[0]}/{alloc[1]}/{alloc[2]}",
                stock=sk, spot=ck, fut=fk,
                cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin,
            ))
    df = pd.DataFrame(rows)
    print(f"  computed {len(df)} combos in {time.time()-t0:.0f}s")

    base = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(base,'v22_rank_sum_full.csv'), index=False)

    df['rank_cal']  = df.groupby('alloc')['cal'].rank(ascending=False, method='min')
    df['rank_ymin'] = df.groupby('alloc')['ymin'].rank(ascending=False, method='min')
    df['rank_sum']  = df['rank_cal'] + df['rank_ymin']

    def asset_summary(asset_col, sort='avg_rank_sum'):
        g = df.groupby(asset_col).agg(
            avg_rank_cal=('rank_cal','mean'),
            avg_rank_ymin=('rank_ymin','mean'),
            avg_rank_sum=('rank_sum','mean'),
            avg_cal=('cal','mean'),
            avg_ymin=('ymin','mean'),
            avg_cagr=('cagr','mean'),
            avg_mdd=('mdd','mean'),
        ).round(3)
        return g.sort_values(sort)

    print(f"\n=== 자산별 후보 rank summary (낮을수록 좋음, top5) ===")
    for col in ['stock','spot','fut']:
        print(f"\n-- {col} (sort by avg_rank_sum) --")
        print(asset_summary(col).head(8).to_string())

    print(f"\n=== alloc 별 best 조합 (Cal 1위) ===")
    print(f"{'alloc':12s} {'stock':18s} {'spot':28s} {'fut':28s}  Cal   CAGR    MDD    ymin")
    for alloc in ALLOCS:
        al = f"{alloc[0]}/{alloc[1]}/{alloc[2]}"
        sub = df[df['alloc']==al].sort_values('cal',ascending=False).head(1)
        if not len(sub): continue
        r = sub.iloc[0]
        print(f"  {al:10s}  {r['stock']:16s}  {r['spot']:26s}  {r['fut']:26s}  {r['cal']:.2f}  {r['cagr']:+.0%}  {r['mdd']:+.0%}  {r['ymin']:+.2f}")

    print(f"\n=== alloc 별 best (ymin 1위) ===")
    for alloc in ALLOCS:
        al = f"{alloc[0]}/{alloc[1]}/{alloc[2]}"
        sub = df[df['alloc']==al].sort_values('ymin',ascending=False).head(1)
        if not len(sub): continue
        r = sub.iloc[0]
        print(f"  {al:10s}  {r['stock']:16s}  {r['spot']:26s}  {r['fut']:26s}  Cal={r['cal']:.2f} ymin={r['ymin']:+.2f}")

    # 자산별 단일 추천 (rank-sum 최저)
    print(f"\n=== 각 자산 범용 best (avg_rank_sum 최저, Cal+ymin 합산) ===")
    for col in ['stock','spot','fut']:
        s = asset_summary(col).iloc[0]
        cand = asset_summary(col).index[0]
        print(f"  {col}: {cand:30s}  rank_sum={s['avg_rank_sum']:.1f}  rank_cal={s['avg_rank_cal']:.1f}  rank_ymin={s['avg_rank_ymin']:.1f}  avg_cal={s['avg_cal']:.2f}  avg_ymin={s['avg_ymin']:+.2f}")

    print(f"\n=== 각 자산 범용 best — Cal/ymin 분리 ===")
    for col in ['stock','spot','fut']:
        cal_b = asset_summary(col,'avg_rank_cal').index[0]
        ymin_b = asset_summary(col,'avg_rank_ymin').index[0]
        print(f"  {col}: Cal-best = {cal_b}   |   ymin-best = {ymin_b}")

if __name__ == '__main__':
    main()
