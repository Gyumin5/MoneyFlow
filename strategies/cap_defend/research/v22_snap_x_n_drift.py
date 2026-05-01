"""drift=0.10 활성 상태에서 snap_length × n_snap 동시 sweep + turnover/final_weight.

drift 가 alpha 임이 확인되어 default 값 유지. 다양한 snap_length × n_snap 비교.
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START='2020-10-01'; END='2026-04-27'; ALLOC=(0.60,0.30,0.10)
bars_D, funding = ub.load_data('D')

def run_spot(snap, n=3, drift=0.10, _trace=None):
    return ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
        sma_days=42, mom_short_days=20, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=0.004,
        health_mode='mom2vol', vol_mode='daily', drift_threshold=drift,
        snap_interval_bars=snap, start_date=START, end_date=END, _trace=_trace)

def run_fut(snap, n=3, drift=0.10, _trace=None):
    return ub.run(bars_D, funding, interval='D', asset_type='fut', leverage=3.0,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
        health_mode='mom2vol', vol_mode='daily', drift_threshold=drift,
        snap_interval_bars=snap, start_date=START, end_date=END, _trace=_trace)

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

# 그리드 — snap × n_snap (소수 stagger 일부 포함)
SPOT_GRID = [
    (60,3),(120,3),(180,3),(240,3),(300,3),(360,3),
    (180,5),(240,5),(300,5),(360,5),
    (180,7),(240,7),(300,7),(360,7),
    (185,5),(205,5),(235,5),  # 5*prime
    (217,7),(259,7),           # 7*prime
]
FUT_GRID = [
    (90,3),(120,3),(180,3),(240,3),(300,3),
    (180,5),(240,5),(300,5),
    (180,7),(240,7),
    (161,7),(203,7),(217,7),
    (185,5),(205,5),
]
STOCK_GRID = [(125,3),(180,3),(252,3),(180,5),(252,5)]

def stock_eq(snap, n): return tss.run_snapshot(sp, snap_days=snap, n_snap=n)['Value']

# 각 BT — equity 와 trace (turnover 계산용)
def bt_with_trace(runner, snap, n):
    trace = []
    res = runner(snap, n, _trace=trace)
    eq = res['_equity']
    rebal_count = sum(1 for t in trace if t.get('rebal'))
    # turnover: 각 rebal 의 half_turnover 합산 × 2
    # trace 에 직접 turnover 없으므로 인접 target 차이로 계산
    cum_turnover = 0
    prev_target = None
    for t in trace:
        if t.get('rebal') and prev_target is not None:
            tg = t['target']
            ks = set(tg.keys()) | set(prev_target.keys())
            cum_turnover += sum(abs(tg.get(k,0) - prev_target.get(k,0)) for k in ks)
        if t.get('rebal'):
            prev_target = t['target']
    return eq, rebal_count, cum_turnover, (trace[-1]['target'] if trace else {})

print(f"\n=== spot {len(SPOT_GRID)} variants ===")
spot = {}
for snap, n in SPOT_GRID:
    eq, rc, tov, last_w = bt_with_trace(run_spot, snap, n)
    spot[(snap, n)] = dict(eq=eq, rebal=rc, turnover=tov, last_w=last_w)

print(f"=== fut {len(FUT_GRID)} variants ===")
fut = {}
for snap, n in FUT_GRID:
    eq, rc, tov, last_w = bt_with_trace(run_fut, snap, n)
    fut[(snap, n)] = dict(eq=eq, rebal=rc, turnover=tov, last_w=last_w)

print(f"=== stock {len(STOCK_GRID)} variants ===")
stock = {(snap,n): {'eq': stock_eq(snap, n)} for snap, n in STOCK_GRID}

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

# 단독 sleeve summary
print(f"\n{'asset/snap_n':18s}  Cal   CAGR    MDD     ymin   rebals  turnov  최종비중")
for label, d in [('spot', spot), ('fut', fut)]:
    for k, v in d.items():
        eq_n = daily_norm(v['eq'])
        cagr, mdd, sh, cal, ymin = stats(eq_n)
        last_w = v['last_w']
        wstr = ' '.join(f"{c}={w:.2f}" for c,w in sorted(last_w.items(), key=lambda x: -x[1])[:4])
        print(f"  {label}_sn{k[0]}_n{k[1]:<2}  {cal:.2f}  {cagr:+.0%}  {mdd:+.0%}  {ymin:+.2f}  {v['rebal']:4d}    {v['turnover']:.1f}    {wstr}")

# 모든 조합 portfolio
rows = []
st_n = {k: daily_norm(v['eq']) for k,v in stock.items()}
sp_n = {k: daily_norm(v['eq']) for k,v in spot.items()}
fu_n = {k: daily_norm(v['eq']) for k,v in fut.items()}

for sk, s in st_n.items():
    for ck, c in sp_n.items():
        for fk, f in fu_n.items():
            p = portfolio(s, c, f)
            cagr, mdd, sh, cal, ymin = stats(p)
            rows.append(dict(
                stock=f'sd{sk[0]}_n{sk[1]}', spot=f'sn{ck[0]}_n{ck[1]}', fut=f'sn{fk[0]}_n{fk[1]}',
                cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin,
                spot_tov=spot[ck]['turnover'], fut_tov=fut[fk]['turnover'],
                spot_rebal=spot[ck]['rebal'], fut_rebal=fut[fk]['rebal'],
            ))
df = pd.DataFrame(rows).sort_values('cal', ascending=False)
print(f"\n=== top 15 portfolio (총 {len(df)} 조합) ===")
print(f"{'stock':12s} {'spot':12s} {'fut':12s}  Cal    CAGR    MDD     ymin   sp_re  fu_re  sp_tov  fu_tov")
for _, r in df.head(15).iterrows():
    print(f"  {r['stock']:11s} {r['spot']:11s} {r['fut']:11s}  {r['cal']:.2f}  {r['cagr']:+.1%}  {r['mdd']:+.1%}  {r['ymin']:+.2f}  {r['spot_rebal']:4d}  {r['fut_rebal']:4d}  {r['spot_tov']:.1f}    {r['fut_tov']:.1f}")

print(f"\n=== top 5 by ymin ===")
for _, r in df.sort_values('ymin', ascending=False).head(5).iterrows():
    print(f"  {r['stock']:11s} {r['spot']:11s} {r['fut']:11s}  Cal={r['cal']:.2f} ymin={r['ymin']:+.2f}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_snap_x_n_drift.csv')
df.to_csv(out, index=False)
print(f"\n저장: {out}")
