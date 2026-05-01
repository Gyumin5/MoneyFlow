"""V22 최종 비교 — snap × drift 의 모든 의미 있는 조합.

평가 시나리오 (60/30/10 portfolio, BT 5.4yr):
A. 현재 V22 LIVE      = sn60 + sn90 + sd125 + drift=0 (실제 라이브 정합)
B. 현재 V22 BT inflate = sn60 + sn90 + sd125 + drift=0.10 (이전 baseline, BT alpha)
C. 권고 snap 길게 LIVE = sn240 + sn180 + sd180 + drift=0 (snap 변경만, 라이브 정합)
D. 권고 snap 길게 + drift 라이브 추가 = sn240 + sn180 + sd180 + drift=0.10
E. 짧은 snap + drift 없음 = sn30 + sn30 + sd63 + drift=0 (drift 없이 빠른 cadence)
F. 짧은 snap + drift 있음 = sn30 + sn30 + sd63 + drift=0.10
G. 중간 snap + drift 0 = sn120 + sn120 + sd125 + drift=0
H. 중간 snap + drift 0.10 = sn120 + sn120 + sd125 + drift=0.10
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START='2020-10-01'; END='2026-04-27'; ALLOC=(0.60,0.30,0.10)
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
def stock_eq(snap, n=3): return tss.run_snapshot(sp, snap_days=snap, n_snap=n)['Value']

# 시나리오 정의
# 드리프트 최적값: spot 0.20, fut 0.05 (앞선 sensitivity 결과)
SCENARIOS = [
    ('A_현재V22_LIVE',          dict(spot=(60, 0.00),  fut=(90, 0.00),  stock=125)),
    ('C_긴snap_LIVE',           dict(spot=(240, 0.00), fut=(180, 0.00), stock=180)),
    ('D1_긴snap_drift_optimal', dict(spot=(240, 0.20), fut=(180, 0.05), stock=180)),
    ('D2_긴snap_drift_default', dict(spot=(240, 0.10), fut=(180, 0.10), stock=180)),
    ('E_짧은snap_LIVE',         dict(spot=(30, 0.00),  fut=(30, 0.00),  stock=63)),
    ('F_짧은snap_drift_optimal',dict(spot=(30, 0.20),  fut=(30, 0.05),  stock=63)),
    ('G_중간snap_LIVE',         dict(spot=(120, 0.00), fut=(120, 0.00), stock=125)),
    ('H_중간snap_drift_optimal',dict(spot=(120, 0.20),fut=(120, 0.05), stock=125)),
    ('I_매우긴snap_LIVE',       dict(spot=(360, 0.00), fut=(360, 0.00), stock=252)),
    ('J_매우긴snap_drift_opt',  dict(spot=(360, 0.20), fut=(360, 0.05), stock=252)),
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
    return cagr, mdd, sh, cal, (min(yc) if yc else 0), tuple(round(c,2) for c in yc)
def portfolio(s,c,f,w=ALLOC):
    idx=s.index.intersection(c.index).intersection(f.index)
    return (1+w[0]*s.loc[idx].pct_change().fillna(0)+w[1]*c.loc[idx].pct_change().fillna(0)+w[2]*f.loc[idx].pct_change().fillna(0)).cumprod()

print("=== running scenarios ===")
results = []
cache_spot = {}
cache_fut = {}
cache_stock = {}
for name, cfg in SCENARIOS:
    spk = cfg['spot']; fuk = cfg['fut']; stk = cfg['stock']
    if spk not in cache_spot:
        cache_spot[spk] = daily_norm(run_spot(spk[0], spk[1]))
    if fuk not in cache_fut:
        cache_fut[fuk] = daily_norm(run_fut(fuk[0], fuk[1]))
    if stk not in cache_stock:
        cache_stock[stk] = daily_norm(stock_eq(stk))
    s = cache_stock[stk]; c = cache_spot[spk]; f = cache_fut[fuk]
    p = portfolio(s, c, f)
    cagr, mdd, sh, cal, ymin, yc = stats(p)
    results.append(dict(name=name, cagr=cagr, mdd=mdd, sh=sh, cal=cal, ymin=ymin, yearly=yc,
                        spot=f'sn{spk[0]}_d{spk[1]}', fut=f'sn{fuk[0]}_d{fuk[1]}', stock=f'sd{stk}'))
    print(f"  {name}")

df = pd.DataFrame(results).sort_values('cal', ascending=False)
print(f"\n=== 최종 시나리오 비교 (60/30/10, BT 5.4yr) ===")
print(f"{'시나리오':30s} {'spot':14s} {'fut':14s} {'stock':6s}  Cal    CAGR    MDD     Sh    ymin  yearly")
for _, r in df.iterrows():
    yc_s = '|'.join(f'{c:.2f}' for c in r['yearly'])
    print(f"  {r['name']:28s}  {r['spot']:13s}  {r['fut']:13s}  {r['stock']:5s}  {r['cal']:.2f}  {r['cagr']:+.1%}  {r['mdd']:+.1%}  {r['sh']:.2f}  {r['ymin']:+.2f}  {yc_s}")

print(f"\n=== top 3 by Cal ===")
for _, r in df.head(3).iterrows():
    print(f"  {r['name']}  Cal={r['cal']:.2f} CAGR={r['cagr']:+.1%} MDD={r['mdd']:+.1%} ymin={r['ymin']:+.2f}")

# LIVE-aligned (drift=0) 만 따로
print(f"\n=== LIVE 정합 (drift=0) 만 ===")
live = df[df['name'].str.contains('LIVE')].sort_values('cal', ascending=False)
for _, r in live.iterrows():
    print(f"  {r['name']:28s}  Cal={r['cal']:.2f} CAGR={r['cagr']:+.1%} ymin={r['ymin']:+.2f}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_final_compare.csv')
df.drop(columns=['yearly']).to_csv(out, index=False)
print(f"\n저장: {out}")
