"""주식 short snap + long snap 앙상블 → 60/30/10 비교.
sd63 (단독 Cal 1.00) + sd180/sd252 (단독 0.89/0.95) 50:50 EW.
코인/선물은 V22 원래 (sn60/sn90).
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START='2020-10-01'; END='2026-04-27'; ALLOC=(0.60,0.30,0.10)

bars_D, funding = ub.load_data('D')

spot_eq = ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=3,
    universe_size=3, cap=1/3, tx_cost=0.004,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=60, start_date=START, end_date=END)['_equity']

fut_eq = ub.run(bars_D, funding, interval='D', asset_type='fut', leverage=3.0,
    sma_days=42, mom_short_days=18, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=3,
    universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=90, start_date=START, end_date=END)['_equity']

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

# 후보 stock 시리즈
def run(snap, n=3): return tss.run_snapshot(sp, snap_days=snap, n_snap=n)['Value']
print("=== running stock variants ===")
st = {
    'sd63':  run(63),
    'sd125': run(125),
    'sd180': run(180),
    'sd252': run(252),
    'sd365': run(365),
}

def daily_norm(eq):
    s = eq.resample('1D').last().dropna()
    return s / s.iloc[0]

def stats(eq):
    rt = eq.pct_change().dropna()
    n = len(rt)
    cagr = (eq.iloc[-1]/eq.iloc[0])**(252/n)-1
    mdd = ((eq-eq.cummax())/eq.cummax()).min()
    sh = rt.mean()/rt.std()*np.sqrt(252) if rt.std()>0 else 0
    cal = cagr/abs(mdd) if mdd!=0 else 0
    yc = []
    for yr, sub in rt.groupby(rt.index.year):
        if len(sub)<30: continue
        eyr = (1+sub).cumprod()
        cyr = eyr.iloc[-1]-1
        myr = ((eyr-eyr.cummax())/eyr.cummax()).min()
        yc.append(cyr/abs(myr) if myr else 0)
    return cagr, mdd, sh, cal, (min(yc) if yc else 0)

def portfolio(s, c, f, w=ALLOC):
    idx = s.index.intersection(c.index).intersection(f.index)
    sr = s.loc[idx].pct_change().fillna(0)
    cr = c.loc[idx].pct_change().fillna(0)
    fr = f.loc[idx].pct_change().fillna(0)
    pr = w[0]*sr+w[1]*cr+w[2]*fr
    return (1+pr).cumprod()

st_n = {k: daily_norm(v) for k,v in st.items()}
spot_n = daily_norm(spot_eq); fut_n = daily_norm(fut_eq)

# === mix 후보들 ===
def mix_ew(*series):
    """EW combine of normalized equity series."""
    idx = series[0].index
    for s in series[1:]: idx = idx.intersection(s.index)
    arrs = [s.loc[idx] for s in series]
    return sum(arrs) / len(arrs)

mixes = {
    'pure_sd63': st_n['sd63'],
    'pure_sd125': st_n['sd125'],
    'pure_sd180': st_n['sd180'],
    'pure_sd252': st_n['sd252'],
    'mix_sd63+sd180': mix_ew(st_n['sd63'], st_n['sd180']),
    'mix_sd63+sd252': mix_ew(st_n['sd63'], st_n['sd252']),
    'mix_sd63+sd365': mix_ew(st_n['sd63'], st_n['sd365']),
    'mix_sd63+sd125+sd252': mix_ew(st_n['sd63'], st_n['sd125'], st_n['sd252']),
    'mix_sd63+sd180+sd365': mix_ew(st_n['sd63'], st_n['sd180'], st_n['sd365']),
    'mix_sd63+sd125+sd180+sd252': mix_ew(st_n['sd63'], st_n['sd125'], st_n['sd180'], st_n['sd252']),
}

rows = []
for name, ts_eq in mixes.items():
    sc, smdd, ssh, scal, symin = stats(ts_eq)
    port = portfolio(ts_eq, spot_n, fut_n)
    pc, pmdd, psh, pcal, pymin = stats(port)
    rows.append(dict(combo=name, s_cal=scal, s_cagr=sc, s_mdd=smdd, s_ymin=symin,
                     p_cal=pcal, p_cagr=pc, p_mdd=pmdd, p_sh=psh, p_ymin=pymin))

df = pd.DataFrame(rows).sort_values('p_cal', ascending=False)
print(f"\n=== 60/30/10 portfolio Cal 순 ===")
print(f"{'stock combo':35s}  단독(Cal/CAGR/MDD/ymin)  포트(Cal/CAGR/MDD/Sh/ymin)")
for _, r in df.iterrows():
    print(f"  {r['combo']:33s}  {r['s_cal']:.2f}/{r['s_cagr']:+.0%}/{r['s_mdd']:+.0%}/{r['s_ymin']:+.2f}    "
          f"{r['p_cal']:.2f}/{r['p_cagr']:+.0%}/{r['p_mdd']:+.0%}/{r['p_sh']:.2f}/{r['p_ymin']:+.2f}")

# corr matrix
print("\n=== stock 단독 returns corr ===")
ret_m = pd.DataFrame({k: v.pct_change() for k,v in st_n.items()}).dropna()
print(ret_m.corr().round(3).to_string())

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_stock_short_long_mix.csv')
df.to_csv(out, index=False)
print(f"\n저장: {out}")
