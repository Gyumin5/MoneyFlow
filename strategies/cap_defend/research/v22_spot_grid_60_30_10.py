"""코인 spot 그리드 → 60/30/10 포트폴리오 비교.
고정: 주식 V22 (3-stagger snap125), 선물 V22 (D S42 M18/127 sn90 L3, 원래 설정).
탐색: 코인 spot D 멤버 그리드. snap 주기 외 SMA/Mom/universe 변형.
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

print("loading bars...")
bars_D, funding = ub.load_data('D')

# 선물 V22 원래 (D S42 M18/127 sn90 L3)
fut_eq = ub.run(bars_D, funding,
    interval='D', asset_type='fut', leverage=3.0,
    sma_days=42, mom_short_days=18, mom_long_days=127,
    vol_days=90, vol_threshold=0.05,
    canary_hyst=0.015, n_snapshots=3,
    universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
    health_mode='mom2vol', vol_mode='daily',
    snap_interval_bars=90, start_date=START, end_date=END,
)['_equity']
print("fut V22 original done")

# 주식 V22
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
stock_eq = tss.run_snapshot(sp_v22, snap_days=125, n_snap=3)['Value']
print("stock V22 done")

# === spot 후보 grid (선물과 같은 축) ===
SPOT_CANDIDATES = [
    # 기준 (현재 V22 spot D 멤버)
    dict(name='S01_S42_M20_127_us3_sn60', sma=42, ms=20, ml=127, us=3, snap=60),
    # snap 변형 (핵심 의문)
    dict(name='S02_S42_M20_127_us3_sn30', sma=42, ms=20, ml=127, us=3, snap=30),
    dict(name='S03_S42_M20_127_us3_sn90', sma=42, ms=20, ml=127, us=3, snap=90),
    dict(name='S04_S42_M20_127_us3_sn120',sma=42, ms=20, ml=127, us=3, snap=120),
    dict(name='S05_S42_M20_127_us3_sn180',sma=42, ms=20, ml=127, us=3, snap=180),
    dict(name='S06_S42_M20_127_us3_sn240',sma=42, ms=20, ml=127, us=3, snap=240),
    # SMA 변형
    dict(name='S07_S21_M10_60_us3_sn60',  sma=21, ms=10, ml=60,  us=3, snap=60),
    dict(name='S08_S60_M30_180_us3_sn90', sma=60, ms=30, ml=180, us=3, snap=90),
    dict(name='S09_S90_M40_252_us3_sn120',sma=90, ms=40, ml=252, us=3, snap=120),
    # mom 변형
    dict(name='S10_S42_M5_30_us3_sn60',   sma=42, ms=5,  ml=30,  us=3, snap=60),
    dict(name='S11_S42_M40_252_us3_sn120',sma=42, ms=40, ml=252, us=3, snap=120),
    # universe 변형
    dict(name='S12_S42_M20_127_us5_sn60', sma=42, ms=20, ml=127, us=5, snap=60),
    dict(name='S13_S42_M20_127_us2_sn60', sma=42, ms=20, ml=127, us=2, snap=60),
    # SMA50
    dict(name='S14_S50_M20_90_us3_sn60',  sma=50, ms=20, ml=90,  us=3, snap=60),
    dict(name='S15_S50_M20_90_us3_sn180', sma=50, ms=20, ml=90,  us=3, snap=180),
]

def spot_cfg(c):
    cap = 1.0 / c['us']
    return dict(
        interval='D', asset_type='spot', leverage=1.0,
        sma_days=c['sma'], mom_short_days=c['ms'], mom_long_days=c['ml'],
        vol_days=90, vol_threshold=0.05,
        canary_hyst=0.015, n_snapshots=3,
        universe_size=c['us'], cap=cap, tx_cost=0.004,
        health_mode='mom2vol', vol_mode='daily',
        snap_interval_bars=c['snap'],
        start_date=START, end_date=END,
    )

print(f"\n=== running {len(SPOT_CANDIDATES)} spot candidates ===")
spot_results = {}
for i, c in enumerate(SPOT_CANDIDATES, 1):
    t0 = time.time()
    res = ub.run(bars_D, funding, **spot_cfg(c))
    spot_results[c['name']] = res['_equity']
    print(f"  [{i}/{len(SPOT_CANDIDATES)}] {c['name']:35s} ({time.time()-t0:.0f}s)")

def daily_norm(eq):
    s = eq.resample('1D').last().dropna()
    return s / s.iloc[0]

def stats(eq):
    rt = eq.pct_change().dropna()
    n = len(rt)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252/n) - 1
    mdd = ((eq - eq.cummax()) / eq.cummax()).min()
    sh = rt.mean() / rt.std() * np.sqrt(252) if rt.std()>0 else 0
    cal = cagr / abs(mdd) if mdd != 0 else 0
    yc = []
    for yr, sub in rt.groupby(rt.index.year):
        if len(sub) < 30: continue
        eyr = (1 + sub).cumprod()
        cyr = eyr.iloc[-1] - 1
        myr = ((eyr - eyr.cummax()) / eyr.cummax()).min()
        yc.append(cyr / abs(myr) if myr else 0)
    ymin = min(yc) if yc else 0
    return cagr, mdd, sh, cal, ymin

def portfolio(s, c, f, w=ALLOC):
    idx = s.index.intersection(c.index).intersection(f.index)
    sr = s.loc[idx].pct_change().fillna(0)
    cr = c.loc[idx].pct_change().fillna(0)
    fr = f.loc[idx].pct_change().fillna(0)
    pr = w[0]*sr + w[1]*cr + w[2]*fr
    return (1+pr).cumprod()

stock_n = daily_norm(stock_eq)
fut_n = daily_norm(fut_eq)

rows = []
for name, eq in spot_results.items():
    spot_n = daily_norm(eq)
    sc, smdd, ssh, scal, symin = stats(spot_n)
    port = portfolio(stock_n, spot_n, fut_n)
    pc, pmdd, psh, pcal, pymin = stats(port)
    rows.append(dict(
        spot=name,
        s_cal=scal, s_cagr=sc, s_mdd=smdd, s_sh=ssh, s_ymin=symin,
        p_cal=pcal, p_cagr=pc, p_mdd=pmdd, p_sh=psh, p_ymin=pymin,
    ))

df = pd.DataFrame(rows).sort_values('p_cal', ascending=False)

print(f"\n=== 60/30/10 portfolio Cal 순 ===")
print(f"{'spot 후보':38s}  단독(Cal/CAGR/MDD/ymin)        포트폴리오(Cal/CAGR/MDD/Sh/ymin)")
for _, r in df.iterrows():
    print(f"  {r['spot']:35s}  {r['s_cal']:.2f}/{r['s_cagr']:+.0%}/{r['s_mdd']:+.0%}/{r['s_ymin']:+.2f}    "
          f"{r['p_cal']:.2f}/{r['p_cagr']:+.0%}/{r['p_mdd']:+.0%}/{r['p_sh']:.2f}/{r['p_ymin']:+.2f}")

print(f"\n=== top 5 by portfolio Cal ===")
for _, r in df.head(5).iterrows():
    print(f"  {r['spot']}  포트Cal={r['p_cal']:.2f}  단독Cal={r['s_cal']:.2f}  ymin={r['p_ymin']:+.2f}")

out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_spot_grid_60_30_10.csv')
df.to_csv(out_path, index=False)
print(f"\n저장: {out_path}")
