"""V22 자산배분 60/30/10 포트폴리오 비교: ensemble vs single sleeve.

주식은 항상 3-stagger 유지 (스냅 베이스가 검증된 우월).
A: 주식 3-stagger + 코인 D+H4 EW    + 선물 D+H4 EW L3
B: 주식 3-stagger + 코인 D_SMA42 단독 + 선물 D_SMA42 L3 단독
C: 주식 3-stagger + 코인 D 단독       + 선물 D+H4 EW L3
D: 주식 3-stagger + 코인 D+H4 EW    + 선물 D 단독 L3

비교: 일별 60/30/10 리밸 포트폴리오 Cal/Sh/CAGR/MDD/yearly Cal
"""
import os, sys, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START_BT = '2020-10-01'
END_BT   = '2026-04-27'
ALLOC = (0.60, 0.30, 0.10)  # stock, spot, fut

# === 코인/선물 BT ===
def fut_or_spot_eq(asset_type, interval, sma_d, sma_b, ms_d, ml_d, ms_b, ml_b, snap_b, lev=1.0, tx=0.0004):
    bars, funding = ub.load_data(interval)
    cfg = dict(
        interval=interval, asset_type=asset_type, leverage=lev,
        vol_days=90, vol_threshold=0.05,
        canary_hyst=0.015, n_snapshots=3,
        universe_size=3, cap=1/3, tx_cost=tx, maint_rate=0.004,
        health_mode='mom2vol', vol_mode='daily',
        snap_interval_bars=snap_b,
        start_date=START_BT, end_date=END_BT,
    )
    if interval == 'D':
        cfg.update(sma_days=sma_d, mom_short_days=ms_d, mom_long_days=ml_d)
    else:
        cfg.update(sma_bars=sma_b, mom_short_bars=ms_b, mom_long_bars=ml_b)
    res = ub.run(bars, funding, **cfg)
    return res['_equity']

print("=== coin spot ===")
spot_d  = fut_or_spot_eq('spot', 'D',  42, 0,   20, 127, 0, 0,    60,  1.0, 0.004)
spot_h4 = fut_or_spot_eq('spot', '4h', 0,  240, 0,  0,   72, 1080, 360, 1.0, 0.004)
print("spot D/H4 done")

print("=== fut ===")
fut_d  = fut_or_spot_eq('fut', 'D',  42, 0,   18, 127, 0, 0,    90,  3.0, 0.0004)
fut_h4 = fut_or_spot_eq('fut', '4h', 0,  240, 0,  0,   72, 1080, 540, 3.0, 0.0004)
print("fut D/H4 done")

# === 주식 ===
# V22 R7B universe + canary EEM SMA300 hyst 2.0% + zscore3 + def_top2 mom126 + 가드 없음
# stock_engine_snap.run_snapshot_ensemble 사용
print("=== stock prep ===")
OFF = ('SPY', 'QQQ', 'VEA', 'EEM', 'EWJ', 'GLD', 'PDBC')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')
CAN = ('EEM',)
# 가격 로드
ts._g_prices = ts.load_prices(list(set(OFF + DEF + CAN)), start='2014-01-01')
ts._g_ind = ts.precompute(ts._g_prices)

stock_params = ts.SP(
    offensive=OFF, defensive=DEF, canary_assets=CAN,
    canary_sma=300, canary_hyst=0.020, canary_type='sma',
    health='none',
    defense='top2', defense_sma=100, def_mom_period=126,
    select='zscore3', n_mom=3, n_sh=3, sharpe_lookback=126,
    weight='ew',
    crash='none',
    tx_cost=0.001, start=START_BT, end=END_BT, capital=10000.0,
)

print("=== stock variants 1/3/5 snap ===")
# 5-snap: snap_days=125 (125/5=25 정수). 1/3 도 125 일관 사용 (3 → stagger 41~42)
stock_eq = {}
for n_snap in (1, 3, 5):
    res = tss.run_snapshot(stock_params, snap_days=125, n_snap=n_snap)
    stock_eq[n_snap] = res['Value']
    print(f"  n_snap={n_snap}  len={len(stock_eq[n_snap])}")
stock_ens_eq = stock_eq[3]  # baseline ref

def daily_norm(eq):
    s = eq.resample('1D').last().dropna()
    return s / s.iloc[0]

stock_n = {n: daily_norm(eq) for n, eq in stock_eq.items()}
spot_d_n  = daily_norm(spot_d)
spot_h4_n = daily_norm(spot_h4)
fut_d_n   = daily_norm(fut_d)
fut_h4_n  = daily_norm(fut_h4)

# spot ensemble = EW(D, H4)
def ew(a, b):
    idx = a.index.intersection(b.index)
    return ((a.loc[idx] + b.loc[idx]) / 2)

spot_ens_n = ew(spot_d_n, spot_h4_n)
fut_ens_n  = ew(fut_d_n,  fut_h4_n)

# 60/30/10 일별 리밸 포트폴리오 (각 자산 정규화 equity 곱하기 weight)
def portfolio(stock_n, spot_n, fut_n, weights=ALLOC):
    idx = stock_n.index.intersection(spot_n.index).intersection(fut_n.index)
    stock_r = stock_n.loc[idx].pct_change().fillna(0)
    spot_r  = spot_n.loc[idx].pct_change().fillna(0)
    fut_r   = fut_n.loc[idx].pct_change().fillna(0)
    w_s, w_c, w_f = weights
    port_r = w_s * stock_r + w_c * spot_r + w_f * fut_r
    eq = (1 + port_r).cumprod()
    return eq

def stats(eq):
    rt = eq.pct_change().dropna()
    n = len(rt)
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (252/n) - 1
    mdd = ((eq - eq.cummax()) / eq.cummax()).min()
    sh = rt.mean() / rt.std() * np.sqrt(252) if rt.std()>0 else 0
    cal = cagr / abs(mdd) if mdd != 0 else 0
    yearly_cal = []
    for yr, sub in rt.groupby(rt.index.year):
        if len(sub) < 30: continue
        eyr = (1 + sub).cumprod()
        cyr = eyr.iloc[-1] - 1
        myr = ((eyr - eyr.cummax()) / eyr.cummax()).min()
        yearly_cal.append((yr, cyr / abs(myr) if myr else 0))
    return cagr, mdd, sh, cal, yearly_cal

# === 시나리오 비교 ===
print("\n" + "="*70)
print(f"포트폴리오 비교 (60/30/10, 일별 리밸, BT {START_BT}~{END_BT})")
print("="*70)

stock_opts = {f'st_n{n}': stock_n[n] for n in (1, 3, 5)}
spot_opts  = {'sp_D': spot_d_n, 'sp_DH4': spot_ens_n}
fut_opts   = {'fu_D': fut_d_n, 'fu_DH4': fut_ens_n}

# === 시나리오 단독 비교 (참고) ===
print("\n=== sleeve 단독 (참고) ===")
for name, eq in [('st_n1', stock_n[1]), ('st_n3', stock_n[3]), ('st_n5', stock_n[5]),
                  ('sp_D', spot_d_n), ('sp_H4', spot_h4_n), ('sp_DH4', spot_ens_n),
                  ('fu_D', fut_d_n), ('fu_H4', fut_h4_n), ('fu_DH4', fut_ens_n)]:
    cagr, mdd, sh, cal, _ = stats(eq)
    print(f"  {name:10s} CAGR={cagr:+.1%}  MDD={mdd:+.1%}  Sh={sh:.2f}  Cal={cal:.2f}")

# === 모든 조합 60/30/10 ===
print(f"\n=== portfolio 60/30/10 (12 조합) ===")
rows = []
for s_name, s in stock_opts.items():
    for c_name, c in spot_opts.items():
        for f_name, f in fut_opts.items():
            eq = portfolio(s, c, f)
            cagr, mdd, sh, cal, yc = stats(eq)
            ymin = min(c for _, c in yc) if yc else 0
            yavg = np.mean([c for _, c in yc]) if yc else 0
            rows.append(dict(
                combo=f'{s_name}+{c_name}+{f_name}',
                cagr=cagr, mdd=mdd, sh=sh, cal=cal, ymin=ymin, yavg=yavg,
                yearly=[(yr, round(c, 2)) for yr, c in yc],
            ))

df = pd.DataFrame(rows).sort_values('cal', ascending=False)
print(f"\n{'조합':35s}  CAGR    MDD    Sh    Cal   ymin  yavg")
for _, r in df.iterrows():
    print(f"  {r['combo']:35s}  {r['cagr']:+.1%}  {r['mdd']:+.1%}  {r['sh']:.2f}  {r['cal']:.2f}  {r['ymin']:+.2f}  {r['yavg']:+.2f}")

print(f"\n=== top 3 by Cal ===")
for _, r in df.head(3).iterrows():
    print(f"  {r['combo']}  Cal={r['cal']:.2f}  yearly={r['yearly']}")

# 저장
out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_portfolio_compare.csv')
df.drop(columns=['yearly']).to_csv(out_path, index=False)
print(f"\n저장: {out_path}")
