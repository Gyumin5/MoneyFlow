"""주식 그리드 → 60/30/10 포트폴리오 비교.
고정: 코인 spot V22 (D S42 M20/127 sn60), 선물 V22 (D S42 M18/127 sn90 L3) — 원래 V22.
탐색: 주식 snap_days × n_snap 그리드.
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START = '2020-10-01'
END   = '2026-04-27'
ALLOC = (0.60, 0.30, 0.10)

bars_D, funding = ub.load_data('D')

# 코인 V22 원래 (sn60)
spot_eq = ub.run(bars_D, funding,
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=3,
    universe_size=3, cap=1/3, tx_cost=0.004,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=60, start_date=START, end_date=END,
)['_equity']
print("spot V22 done")

# 선물 V22 원래 (sn90)
fut_eq = ub.run(bars_D, funding,
    interval='D', asset_type='fut', leverage=3.0,
    sma_days=42, mom_short_days=18, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=3,
    universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=90, start_date=START, end_date=END,
)['_equity']
print("fut V22 done")

# 주식 V22 prep
OFF = ('SPY','QQQ','VEA','EEM','EWJ','GLD','PDBC')
DEF = ('IEF','BIL','BNDX','GLD','PDBC')
CAN = ('EEM',)
ts._g_prices = ts.load_prices(list(set(OFF + DEF + CAN)), start='2014-01-01')
ts._g_ind = ts.precompute(ts._g_prices)
sp_v22 = ts.SP(
    offensive=OFF, defensive=DEF, canary_assets=CAN,
    canary_sma=300, canary_hyst=0.020, canary_type='sma',
    health='none', defense='top2', defense_sma=100, def_mom_period=126,
    select='zscore3', n_mom=3, n_sh=3, sharpe_lookback=126,
    weight='ew', crash='none',
    tx_cost=0.001, start=START, end=END, capital=10000.0,
)

# 주식 grid: snap_days × n_snap
GRID = [
    (63, 3), (90, 3), (125, 3), (180, 3), (252, 3), (365, 3),    # snap_days 변형 (n_snap=3)
    (125, 1), (125, 5), (125, 7),                                # n_snap 변형 (snap_days=125)
    (252, 5), (180, 5),                                          # 둘 다 변형
    (90, 1), (252, 1),                                           # 1-snap 비교
]

print(f"\n=== running {len(GRID)} stock variants ===")
stock_results = {}
for i, (snap, n) in enumerate(GRID, 1):
    t0 = time.time()
    res = tss.run_snapshot(sp_v22, snap_days=snap, n_snap=n)
    name = f'sd{snap}_n{n}'
    stock_results[name] = res['Value']
    print(f"  [{i}/{len(GRID)}] {name:15s} ({time.time()-t0:.1f}s)")

def daily_norm(eq):
    s = eq.resample('1D').last().dropna()
    return s / s.iloc[0]

def stats(eq):
    rt = eq.pct_change().dropna()
    n = len(rt)
    cagr = (eq.iloc[-1]/eq.iloc[0])**(252/n) - 1
    mdd = ((eq - eq.cummax())/eq.cummax()).min()
    sh = rt.mean()/rt.std() * np.sqrt(252) if rt.std()>0 else 0
    cal = cagr/abs(mdd) if mdd != 0 else 0
    yc = []
    for yr, sub in rt.groupby(rt.index.year):
        if len(sub) < 30: continue
        eyr = (1+sub).cumprod()
        cyr = eyr.iloc[-1] - 1
        myr = ((eyr - eyr.cummax())/eyr.cummax()).min()
        yc.append(cyr/abs(myr) if myr else 0)
    return cagr, mdd, sh, cal, (min(yc) if yc else 0)

def portfolio(s, c, f, w=ALLOC):
    idx = s.index.intersection(c.index).intersection(f.index)
    sr = s.loc[idx].pct_change().fillna(0)
    cr = c.loc[idx].pct_change().fillna(0)
    fr = f.loc[idx].pct_change().fillna(0)
    pr = w[0]*sr + w[1]*cr + w[2]*fr
    return (1+pr).cumprod()

spot_n = daily_norm(spot_eq)
fut_n = daily_norm(fut_eq)

rows = []
for name, eq in stock_results.items():
    stock_n = daily_norm(eq)
    sc, smdd, ssh, scal, symin = stats(stock_n)
    port = portfolio(stock_n, spot_n, fut_n)
    pc, pmdd, psh, pcal, pymin = stats(port)
    rows.append(dict(
        stock=name,
        s_cal=scal, s_cagr=sc, s_mdd=smdd, s_sh=ssh, s_ymin=symin,
        p_cal=pcal, p_cagr=pc, p_mdd=pmdd, p_sh=psh, p_ymin=pymin,
    ))

df = pd.DataFrame(rows).sort_values('p_cal', ascending=False)

print(f"\n=== 60/30/10 portfolio Cal 순 ===")
print(f"{'stock':18s}  단독(Cal/CAGR/MDD/ymin)        포트(Cal/CAGR/MDD/Sh/ymin)")
for _, r in df.iterrows():
    print(f"  {r['stock']:15s}   {r['s_cal']:.2f}/{r['s_cagr']:+.0%}/{r['s_mdd']:+.0%}/{r['s_ymin']:+.2f}    "
          f"{r['p_cal']:.2f}/{r['p_cagr']:+.0%}/{r['p_mdd']:+.0%}/{r['p_sh']:.2f}/{r['p_ymin']:+.2f}")

print(f"\n=== top 5 by portfolio Cal ===")
for _, r in df.head(5).iterrows():
    print(f"  {r['stock']}  포트Cal={r['p_cal']:.2f}  단독Cal={r['s_cal']:.2f}")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_stock_grid_60_30_10.csv')
df.to_csv(out_path, index=False)
print(f"\n저장: {out_path}")
