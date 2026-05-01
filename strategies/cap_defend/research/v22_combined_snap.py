"""모든 자산 snap 동시 최적화 → 60/30/10 최종 비교.

Stock 변형: sd125_n3 (현 V22) / sd63_n3 / sd180_n3 / mix_sd63+sd180
Spot 변형:  sn60 (현 V22) / sn180 / sn240
Fut 변형:   sn90 (현 V22) / sn180 / sn240
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START='2020-10-01'; END='2026-04-27'; ALLOC=(0.60,0.30,0.10)

bars_D, funding = ub.load_data('D')

# === spot 변형 ===
print("=== spot variants ===")
spot_configs = {
    'sp_sn60':  60,
    'sp_sn180': 180,
    'sp_sn240': 240,
}
spot_eq = {}
for n, snap in spot_configs.items():
    spot_eq[n] = ub.run(bars_D, funding,
        interval='D', asset_type='spot', leverage=1.0,
        sma_days=42, mom_short_days=20, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=3,
        universe_size=3, cap=1/3, tx_cost=0.004,
        health_mode='mom2vol', vol_mode='daily',
        snap_interval_bars=snap, start_date=START, end_date=END)['_equity']
    print(f"  {n} done")

# === fut 변형 ===
print("=== fut variants ===")
fut_configs = {
    'fu_sn90':  90,
    'fu_sn180': 180,
    'fu_sn240': 240,
}
fut_eq = {}
for n, snap in fut_configs.items():
    fut_eq[n] = ub.run(bars_D, funding,
        interval='D', asset_type='fut', leverage=3.0,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=3,
        universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
        health_mode='mom2vol', vol_mode='daily',
        snap_interval_bars=snap, start_date=START, end_date=END)['_equity']
    print(f"  {n} done")

# === stock 변형 ===
print("=== stock variants ===")
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

stock_eq = {
    'st_sd125': tss.run_snapshot(sp, snap_days=125, n_snap=3)['Value'],
    'st_sd63':  tss.run_snapshot(sp, snap_days=63,  n_snap=3)['Value'],
    'st_sd180': tss.run_snapshot(sp, snap_days=180, n_snap=3)['Value'],
}
print("  base done")

def daily_norm(eq):
    s = eq.resample('1D').last().dropna()
    return s / s.iloc[0]

stock_n = {k: daily_norm(v) for k,v in stock_eq.items()}
spot_n = {k: daily_norm(v) for k,v in spot_eq.items()}
fut_n = {k: daily_norm(v) for k,v in fut_eq.items()}

# stock mix sd63+sd180
def mix_ew(*series):
    idx = series[0].index
    for s in series[1:]: idx = idx.intersection(s.index)
    arrs = [s.loc[idx] for s in series]
    return sum(arrs) / len(arrs)

stock_n['st_mix_63_180'] = mix_ew(stock_n['st_sd63'], stock_n['st_sd180'])

def stats(eq):
    rt = eq.pct_change().dropna()
    n = len(rt)
    cagr = (eq.iloc[-1]/eq.iloc[0])**(252/n)-1
    mdd = ((eq-eq.cummax())/eq.cummax()).min()
    sh = rt.mean()/rt.std()*np.sqrt(252) if rt.std()>0 else 0
    cal = cagr/abs(mdd) if mdd!=0 else 0
    yc = []
    for y, sub in rt.groupby(rt.index.year):
        if len(sub)<30: continue
        eyr = (1+sub).cumprod()
        cyr = eyr.iloc[-1]-1
        myr = ((eyr-eyr.cummax())/eyr.cummax()).min()
        yc.append(cyr/abs(myr) if myr else 0)
    return cagr, mdd, sh, cal, (min(yc) if yc else 0), yc

def portfolio(s, c, f, w=ALLOC):
    idx = s.index.intersection(c.index).intersection(f.index)
    sr = s.loc[idx].pct_change().fillna(0)
    cr = c.loc[idx].pct_change().fillna(0)
    fr = f.loc[idx].pct_change().fillna(0)
    pr = w[0]*sr+w[1]*cr+w[2]*fr
    return (1+pr).cumprod()

# === 모든 조합 ===
rows = []
for sn, s in stock_n.items():
    for cn, c in spot_n.items():
        for fn, f in fut_n.items():
            port = portfolio(s, c, f)
            cagr, mdd, sh, cal, ymin, yc = stats(port)
            rows.append(dict(
                combo=f'{sn}+{cn}+{fn}',
                cal=cal, cagr=cagr, mdd=mdd, sh=sh, ymin=ymin,
                yc_str='|'.join(f'{c:.2f}' for c in yc),
            ))

df = pd.DataFrame(rows).sort_values('cal', ascending=False)
print(f"\n=== 36 조합 60/30/10 portfolio ===")
print(f"{'combo':45s}  Cal    CAGR    MDD     Sh    ymin")
for _, r in df.iterrows():
    print(f"  {r['combo']:43s}  {r['cal']:.2f}  {r['cagr']:+.1%}  {r['mdd']:+.1%}  {r['sh']:.2f}  {r['ymin']:+.2f}")

print(f"\n=== top 5 by Cal ===")
for _, r in df.head(5).iterrows():
    print(f"  {r['combo']}  Cal={r['cal']:.2f}  CAGR={r['cagr']:+.1%}  yearly={r['yc_str']}")

# 현재 V22 (sd125+sn60+sn90)
cur = df[df['combo']=='st_sd125+sp_sn60+fu_sn90'].iloc[0]
print(f"\n=== 현재 V22 baseline ===\n  {cur['combo']} Cal={cur['cal']:.2f} CAGR={cur['cagr']:+.1%} ymin={cur['ymin']:+.2f}")

print(f"\n=== top 5 by yearly worst (ymin) ===")
for _, r in df.sort_values('ymin', ascending=False).head(5).iterrows():
    print(f"  {r['combo']}  ymin={r['ymin']:+.2f}  Cal={r['cal']:.2f}")

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_combined_snap.csv')
df.to_csv(out, index=False)
print(f"\n저장: {out}")
