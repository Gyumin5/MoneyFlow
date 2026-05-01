"""평탄화 검증 — picked best 주변 (n*prime 제약 없음). 짧은 spot snap + n_snap 변형 포함.

picks 주변 + 짧은 snap + n_snap ∈ {3,5,7}
- stock: sd ∈ {54..120} × n ∈ {3, 5}
- spot:  sn ∈ {60..240} × n ∈ {3,5,7} × d ∈ {0.10,0.15,0.20}
- fut:   sn ∈ {60..150} × n ∈ {3,5,7} × d ∈ {0.0,0.05,0.10}
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

STOCK_SN = [54, 57, 60, 63, 66, 69, 72, 75, 78, 81, 84, 90, 96, 105, 120]
STOCK_N  = [3, 5]
SPOT_SN  = [60, 75, 90, 105, 120, 135, 150, 165, 174, 183, 192, 200, 210, 220, 240]
SPOT_N   = [3, 5, 7]
SPOT_D   = [0.10, 0.15, 0.20]
FUT_SN   = [60, 66, 72, 78, 84, 87, 90, 93, 96, 99, 105, 111, 120, 135, 150]
FUT_N    = [3, 5, 7]
FUT_D    = [0.00, 0.05, 0.10]

STOCK = [(sd, n) for sd in STOCK_SN for n in STOCK_N]
SPOT  = [(sn, n, d) for sn in SPOT_SN for n in SPOT_N for d in SPOT_D]
FUT   = [(sn, n, d) for sn in FUT_SN  for n in FUT_N  for d in FUT_D]

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

def main():
    n_workers = min(8, os.cpu_count() or 4)
    print(f"=== plateau check: stock {len(STOCK)} + spot {len(SPOT)} + fut {len(FUT)} = {len(STOCK)+len(SPOT)+len(FUT)} BTs (workers={n_workers}) ===")
    rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers, initializer=_init) as ex:
        futs = []
        for a in STOCK: futs.append(ex.submit(_stock, a))
        for a in SPOT:  futs.append(ex.submit(_spot,  a))
        for a in FUT:   futs.append(ex.submit(_fut,   a))
        for f in futs:
            (asset, sn, n, d), eq = f.result()
            cagr, mdd, sh, cal, ymin = stats(eq)
            rows.append(dict(asset=asset, snap=sn, n_snap=n, drift=d,
                             cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin))
    print(f"  done {time.time()-t0:.0f}s")

    df = pd.DataFrame(rows)
    base = os.path.dirname(os.path.abspath(__file__))
    df.to_csv(os.path.join(base,'v22_plateau_around_pick.csv'), index=False)

    print(f"\n=== STOCK best per (sd, n) ===")
    print(f"{'sd':>5} {'n':>3}  Cal   CAGR    MDD    ymin")
    for _, r in df[df.asset=='stock'].sort_values(['n_snap','snap']).iterrows():
        print(f"  {r['snap']:>3.0f}  {r['n_snap']:>1.0f}   {r['cal']:.2f}  {r['cagr']:+.0%}  {r['mdd']:+.0%}  {r['ymin']:+.2f}")
    print("\n  STOCK top 5 by Cal:")
    for _, r in df[df.asset=='stock'].sort_values('cal',ascending=False).head(5).iterrows():
        print(f"    sd={r['snap']:>3.0f} n={r['n_snap']:>1.0f}  Cal={r['cal']:.2f} ymin={r['ymin']:+.2f}")

    print(f"\n=== SPOT best per drift, n ===")
    for d in SPOT_D:
        for n in SPOT_N:
            sub = df[(df.asset=='spot')&(df.drift==d)&(df.n_snap==n)]
            top = sub.sort_values('cal',ascending=False).head(3)
            for _, r in top.iterrows():
                print(f"  d={d:.2f} n={n}  sn={r['snap']:>3.0f}  Cal={r['cal']:.2f}  CAGR={r['cagr']:+.0%}  MDD={r['mdd']:+.0%}  ymin={r['ymin']:+.2f}")
    print("\n  SPOT top 10 overall by Cal:")
    for _, r in df[df.asset=='spot'].sort_values('cal',ascending=False).head(10).iterrows():
        print(f"    sn={r['snap']:>3.0f} n={r['n_snap']:>1.0f} d={r['drift']:.2f}  Cal={r['cal']:.2f}  ymin={r['ymin']:+.2f}")
    print("\n  SPOT top 5 by ymin:")
    for _, r in df[df.asset=='spot'].sort_values('ymin',ascending=False).head(5).iterrows():
        print(f"    sn={r['snap']:>3.0f} n={r['n_snap']:>1.0f} d={r['drift']:.2f}  Cal={r['cal']:.2f}  ymin={r['ymin']:+.2f}")

    print(f"\n=== FUT best per drift, n ===")
    for d in FUT_D:
        for n in FUT_N:
            sub = df[(df.asset=='fut')&(df.drift==d)&(df.n_snap==n)]
            top = sub.sort_values('cal',ascending=False).head(3)
            for _, r in top.iterrows():
                print(f"  d={d:.2f} n={n}  sn={r['snap']:>3.0f}  Cal={r['cal']:.2f}  CAGR={r['cagr']:+.0%}  MDD={r['mdd']:+.0%}  ymin={r['ymin']:+.2f}")
    print("\n  FUT top 10 overall by Cal:")
    for _, r in df[df.asset=='fut'].sort_values('cal',ascending=False).head(10).iterrows():
        print(f"    sn={r['snap']:>3.0f} n={r['n_snap']:>1.0f} d={r['drift']:.2f}  Cal={r['cal']:.2f}  ymin={r['ymin']:+.2f}")
    print("\n  FUT top 5 by ymin:")
    for _, r in df[df.asset=='fut'].sort_values('ymin',ascending=False).head(5).iterrows():
        print(f"    sn={r['snap']:>3.0f} n={r['n_snap']:>1.0f} d={r['drift']:.2f}  Cal={r['cal']:.2f}  ymin={r['ymin']:+.2f}")

    # n_snap 마진얼: 같은 (snap, drift) 에서 n_snap 별 평균
    print(f"\n=== n_snap 마진얼 (자산별 평균 Cal/ymin) ===")
    for asset in ['stock','spot','fut']:
        sub = df[df.asset==asset]
        print(f"  {asset}:")
        print(sub.groupby('n_snap')[['cal','ymin','cagr','mdd']].mean().round(3).to_string())

if __name__ == '__main__':
    main()
