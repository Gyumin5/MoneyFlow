"""drift_threshold sensitivity 테스트 + 라이브 정합 (drift=0) 재검증.

목적
1. BT 의 drift_threshold 효과 측정 — drift=0 (라이브 정합) 부터 0.05~0.20 까지
2. 권고 후보 (st_sd180 + sp_sn240 + fu_sn180) 의 drift 민감도
3. drift 가 진짜 alpha 인지 확인 — 좋으면 라이브 엔진에 추가 검토
"""
import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START='2020-10-01'; END='2026-04-27'; ALLOC=(0.60,0.30,0.10)
bars_D, funding = ub.load_data('D')

def run_spot(snap, n=3, drift=0.10):
    return ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
        sma_days=42, mom_short_days=20, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=0.004,
        health_mode='mom2vol', vol_mode='daily', drift_threshold=drift,
        snap_interval_bars=snap, start_date=START, end_date=END)['_equity']

def run_fut(snap, n=3, drift=0.10):
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
def run_stock(snap, n=3): return tss.run_snapshot(sp, snap_days=snap, n_snap=n)['Value']

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

# stock 1번만 (drift 영향 안 받음)
print("=== stock sd180_n3 ===")
stock_n = daily_norm(run_stock(180, 3))

# spot/fut: drift 변화 × 권고 snap
DRIFTS = [0.0, 0.05, 0.08, 0.10, 0.15, 0.20]

print("=== spot variants ===")
spot_eq = {d: daily_norm(run_spot(240, 3, d)) for d in DRIFTS}
for d in DRIFTS:
    cagr, mdd, sh, cal, ymin = stats(spot_eq[d])
    print(f"  spot drift={d:.2f}  단독 Cal={cal:.2f} CAGR={cagr:+.0%} MDD={mdd:+.0%} ymin={ymin:+.2f}")

print("\n=== fut variants ===")
fut_eq = {d: daily_norm(run_fut(180, 3, d)) for d in DRIFTS}
for d in DRIFTS:
    cagr, mdd, sh, cal, ymin = stats(fut_eq[d])
    print(f"  fut  drift={d:.2f}  단독 Cal={cal:.2f} CAGR={cagr:+.0%} MDD={mdd:+.0%} ymin={ymin:+.2f}")

# 모든 (spot drift, fut drift) 조합 portfolio
print(f"\n=== portfolio (st sd180_n3 + sp sn240 + fu sn180, drift sweep) ===")
print(f"{'spot_d':10s} {'fut_d':10s}  Cal    CAGR    MDD     Sh    ymin")
rows = []
for sd in DRIFTS:
    for fd in DRIFTS:
        p = portfolio(stock_n, spot_eq[sd], fut_eq[fd])
        cagr, mdd, sh, cal, ymin = stats(p)
        rows.append(dict(spot_drift=sd, fut_drift=fd, cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin))
        flag = '←' if sd==0 and fd==0 else ('★' if sd==0.10 and fd==0.10 else ' ')
        print(f"  {sd:.2f}      {fd:.2f}      {cal:.2f}  {cagr:+.1%}  {mdd:+.1%}  {sh:.2f}  {ymin:+.2f}  {flag}")

df = pd.DataFrame(rows).sort_values('cal', ascending=False)
print(f"\n=== top 5 by Cal ===")
for _, r in df.head(5).iterrows():
    print(f"  spot_d={r['spot_drift']:.2f}  fut_d={r['fut_drift']:.2f}  Cal={r['cal']:.2f} ymin={r['ymin']:+.2f}")

print(f"\n=== drift=0 (라이브 정합) ===")
zero = df[(df['spot_drift']==0) & (df['fut_drift']==0)].iloc[0]
print(f"  Cal={zero['cal']:.2f} CAGR={zero['cagr']:+.1%} MDD={zero['mdd']:+.1%} ymin={zero['ymin']:+.2f}")

print(f"\n=== drift=0.10 (BT 기본, 이전 권고에 사용된 값) ===")
ten = df[(df['spot_drift']==0.10) & (df['fut_drift']==0.10)].iloc[0]
print(f"  Cal={ten['cal']:.2f} CAGR={ten['cagr']:+.1%} MDD={ten['mdd']:+.1%} ymin={ten['ymin']:+.2f}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_drift_sensitivity.csv')
df.to_csv(out, index=False)
print(f"\n저장: {out}")
