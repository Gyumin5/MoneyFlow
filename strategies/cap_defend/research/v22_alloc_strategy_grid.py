"""자산배분 × 전략 그리드 — 60/40/0 포함 다양한 alloc 에서 각 자산 최적 전략 결정.

후보 sleeve
- stock: sd125_n3 (LIVE), sd180_n3 (D1)
- spot:  sn60_n3 d=0 (LIVE), sn240_n3 d=0.20 (D1), sn235_n5 d=0.20 (prime Cal), sn265_n5 d=0.20 (prime ymin)
- fut:   sn90_n3 d=0 (LIVE), sn180_n3 d=0.05 (D1), sn203_n7 d=0.05 (prime Cal), sn183_n3 d=0.05 (prime ymin)

자산배분 그리드 (stock, coin, fut)
- (100,0,0), (80,20,0), (70,30,0), (60,40,0), (50,50,0)
- (60,35,5), (60,30,10), (60,25,15), (60,20,20)
- (50,40,10), (50,30,20), (50,25,25)
- (70,20,10), (80,10,10), (40,40,20), (33,33,34)
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START='2020-10-01'; END='2026-04-27'
bars_D, funding = ub.load_data('D')

def run_spot(snap, drift, n=3):
    return ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
        sma_days=42, mom_short_days=20, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=0.004,
        health_mode='mom2vol', vol_mode='daily', drift_threshold=drift,
        snap_interval_bars=snap, start_date=START, end_date=END)['_equity']

def run_fut(snap, drift, n=3):
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
def run_stock(sn, n=3): return tss.run_snapshot(sp, snap_days=sn, n_snap=n)['Value']

STOCK_CANDS = {
    'st_LIVE_sd125': (125, 3),
    'st_D1_sd180':   (180, 3),
}
SPOT_CANDS = {
    'sp_LIVE_sn60':       (60,  0.00, 3),
    'sp_D1_sn240_d20':    (240, 0.20, 3),
    'sp_PR_sn235_n5_d20': (235, 0.20, 5),
    'sp_PR_sn265_n5_d20': (265, 0.20, 5),
}
FUT_CANDS = {
    'fu_LIVE_sn90':       (90,  0.00, 3),
    'fu_D1_sn180_d05':    (180, 0.05, 3),
    'fu_PR_sn203_n7_d05': (203, 0.05, 7),
    'fu_PR_sn183_n3_d05': (183, 0.05, 3),
}

ALLOCS = [
    (100, 0, 0), (80, 20, 0), (70, 30, 0), (60, 40, 0), (50, 50, 0),
    (60, 35, 5), (60, 30, 10), (60, 25, 15), (60, 20, 20),
    (50, 40, 10), (50, 30, 20), (50, 25, 25),
    (70, 20, 10), (80, 10, 10), (40, 40, 20), (33, 33, 34),
]

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
def portfolio(s, c, f, ws, wc, wf):
    if wc == 0 and wf == 0:
        return s
    if ws == 0 and wf == 0:
        return c
    idx = s.index.intersection(c.index)
    if wf > 0: idx = idx.intersection(f.index)
    rs = s.loc[idx].pct_change().fillna(0)
    rc = c.loc[idx].pct_change().fillna(0)
    rf = f.loc[idx].pct_change().fillna(0) if wf > 0 else 0
    return (1 + ws*rs + wc*rc + wf*rf).cumprod()

print("=== sleeve BT (cache) ===")
stock_cache = {k: daily_norm(run_stock(*v)) for k,v in STOCK_CANDS.items()}
print("stock done")
spot_cache = {k: daily_norm(run_spot(v[0], v[1], v[2])) for k,v in SPOT_CANDS.items()}
print("spot done")
fut_cache = {k: daily_norm(run_fut(v[0], v[1], v[2])) for k,v in FUT_CANDS.items()}
print("fut done")

# 단독 sleeve stats
print("\n=== 단독 sleeve ===")
for label, d in [('stock', stock_cache), ('spot', spot_cache), ('fut', fut_cache)]:
    for k, eq in d.items():
        cagr, mdd, sh, cal, ymin = stats(eq)
        print(f"  {k:25s}  Cal={cal:.2f} CAGR={cagr:+.0%} MDD={mdd:+.0%} Sh={sh:.2f} ymin={ymin:+.2f}")

# 모든 (alloc, stock_cand, spot_cand, fut_cand) 조합
rows = []
for alloc in ALLOCS:
    ws, wc, wf = alloc[0]/100, alloc[1]/100, alloc[2]/100
    for sk, s in stock_cache.items():
        # stock 0% 면 stock 후보 1개만
        if ws == 0 and sk != list(STOCK_CANDS)[0]: continue
        for ck, c in spot_cache.items():
            if wc == 0 and ck != list(SPOT_CANDS)[0]: continue
            for fk, f in fut_cache.items():
                if wf == 0 and fk != list(FUT_CANDS)[0]: continue
                p = portfolio(s, c, f, ws, wc, wf)
                cagr, mdd, sh, cal, ymin = stats(p)
                rows.append(dict(
                    alloc=f"{alloc[0]}/{alloc[1]}/{alloc[2]}",
                    stock=sk, spot=ck, fut=fk,
                    cagr=cagr, mdd=mdd, sh=sh, cal=cal, ymin=ymin,
                ))

df = pd.DataFrame(rows)
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_alloc_strategy_grid.csv')
df.to_csv(out, index=False)
print(f"\n총 {len(df)} 조합, 저장: {out}")

# 각 alloc 별 best (Cal 기준)
print(f"\n=== 각 alloc 별 Cal best ===")
print(f"{'alloc':12s} {'stock':22s} {'spot':22s} {'fut':22s}  Cal    CAGR    MDD     Sh    ymin")
for alloc in ALLOCS:
    al = f"{alloc[0]}/{alloc[1]}/{alloc[2]}"
    sub = df[df['alloc']==al].sort_values('cal', ascending=False)
    if len(sub)==0: continue
    r = sub.iloc[0]
    print(f"  {al:10s}  {r['stock']:20s}  {r['spot']:20s}  {r['fut']:20s}  {r['cal']:.2f}  {r['cagr']:+.1%}  {r['mdd']:+.1%}  {r['sh']:.2f}  {r['ymin']:+.2f}")

# 각 alloc 별 best (ymin 기준)
print(f"\n=== 각 alloc 별 ymin best ===")
for alloc in ALLOCS:
    al = f"{alloc[0]}/{alloc[1]}/{alloc[2]}"
    sub = df[df['alloc']==al].sort_values('ymin', ascending=False)
    if len(sub)==0: continue
    r = sub.iloc[0]
    print(f"  {al:10s}  {r['stock']:20s}  {r['spot']:20s}  {r['fut']:20s}  Cal={r['cal']:.2f} ymin={r['ymin']:+.2f}")

# 자산별 전략 우세도 - 각 자산에서 어떤 전략이 가장 자주 best 인가
print(f"\n=== 자산별 best 전략 분포 (Cal 기준 alloc 별 top1) ===")
best_per_alloc = df.sort_values('cal', ascending=False).groupby('alloc').first().reset_index()
print("stock 분포:"); print(best_per_alloc['stock'].value_counts().to_string())
print("\nspot 분포:"); print(best_per_alloc['spot'].value_counts().to_string())
print("\nfut 분포:"); print(best_per_alloc['fut'].value_counts().to_string())

# 자산 고정 + alloc 변화 robustness
print(f"\n=== 자산별 전략 robustness (alloc 평균 Cal) ===")
print("\nstock candidate × alloc 평균:")
print(df.groupby('stock')[['cal','ymin','cagr','mdd']].mean().to_string())
print("\nspot candidate × alloc 평균:")
print(df.groupby('spot')[['cal','ymin','cagr','mdd']].mean().to_string())
print("\nfut candidate × alloc 평균:")
print(df.groupby('fut')[['cal','ymin','cagr','mdd']].mean().to_string())
