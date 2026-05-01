"""V22 1D+1D 페어 후보 sweep — 진짜 다양성 갖는 ensemble 탐색.

배경: V22 (D_SMA42 + H4_SMA240) corr 0.82 — interval 다양성 가설 약함.
목표: 1D 단독 후보 N개 → 모든 pair 대해 (corr, EW Cal, EW MDD) 계산.
필터: corr<0.5, 단독 Cal>2.0, EW Cal>=D_SMA42 단독 (2.37)

단독 멤버 후보: SMA × Mom × universe_size 조합으로 시그널 다양화.
"""
import os, sys, json, itertools, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub

START = '2020-10-01'
END   = '2026-04-27'

# 후보 1D 멤버 — 시그널 다양성 위해 SMA/Mom/health/universe 축 분산
CANDIDATES = [
    # 기준
    dict(name='M01_S42_M20_127',  sma=42,  ms=20, ml=127, health='mom2vol', us=3),
    # 빠른 추세
    dict(name='M02_S21_M10_63',   sma=21,  ms=10, ml=63,  health='mom2vol', us=3),
    # 느린 추세
    dict(name='M03_S90_M40_252',  sma=90,  ms=40, ml=252, health='mom2vol', us=3),
    dict(name='M04_S150_M60_252', sma=150, ms=60, ml=252, health='mom2vol', us=3),
    # 짧은 mom 강조 (다른 시그널)
    dict(name='M05_S42_M5_30',    sma=42,  ms=5,  ml=30,  health='mom2vol', us=3),
    # 긴 mom 강조
    dict(name='M06_S42_M60_252',  sma=42,  ms=60, ml=252, health='mom2vol', us=3),
    # health 변형
    dict(name='M07_S42_M20_127_h_none', sma=42, ms=20, ml=127, health='none', us=3),
    dict(name='M08_S42_M20_127_h_vol',  sma=42, ms=20, ml=127, health='vol',  us=3),
    # 넓은 universe
    dict(name='M09_S42_M20_127_us5', sma=42, ms=20, ml=127, health='mom2vol', us=5),
    # SMA50 (V21 시절)
    dict(name='M10_S50_M20_90',   sma=50,  ms=20, ml=90,  health='mom2vol', us=3),
    # SMA100
    dict(name='M11_S100_M20_120', sma=100, ms=20, ml=120, health='mom2vol', us=3),
    # 매우 빠른 + 좁은 mom
    dict(name='M12_S15_M5_20',    sma=15,  ms=5,  ml=20,  health='mom2vol', us=3),
]

def base_cfg(c):
    cap = 1.0 / c['us']
    return dict(
        interval='D', asset_type='spot', leverage=1.0,
        sma_days=c['sma'], mom_short_days=c['ms'], mom_long_days=c['ml'],
        vol_days=90, vol_threshold=0.05,
        canary_hyst=0.015, n_snapshots=3,
        universe_size=c['us'], cap=cap, tx_cost=0.004,
        health_mode=c['health'], vol_mode='daily',
        snap_interval_bars=60,
        start_date=START, end_date=END,
    )

def compute_stats(eq):
    rs = eq.resample('1D').last().dropna()
    rt = rs.pct_change().dropna()
    if len(rt) < 30:
        return dict(cagr=0, mdd=0, sh=0, cal=0)
    cagr = (rs.iloc[-1]/rs.iloc[0])**(252/len(rt)) - 1
    eq_max = rs.cummax()
    mdd = ((rs - eq_max) / eq_max).min()
    sh = rt.mean() / rt.std() * np.sqrt(252) if rt.std()>0 else 0
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return dict(cagr=float(cagr), mdd=float(mdd), sh=float(sh), cal=float(cal))

print("loading D bars...")
bars_D, funding = ub.load_data('D')
print(f"loaded {len(bars_D)} coins")

# 단독 BT 모두 실행
single = {}
print(f"\n=== running {len(CANDIDATES)} single members ===")
for i, c in enumerate(CANDIDATES, 1):
    t0 = time.time()
    res = ub.run(bars_D, funding, **base_cfg(c))
    eq = res['_equity']
    rt = eq.resample('1D').last().pct_change().dropna()
    stats = compute_stats(eq)
    single[c['name']] = dict(
        cfg=c, eq=eq, rt=rt, stats=stats
    )
    elapsed = time.time() - t0
    print(f"  [{i}/{len(CANDIDATES)}] {c['name']:30s} Cal={stats['cal']:.2f} CAGR={stats['cagr']:.1%} MDD={stats['mdd']:.1%} Sh={stats['sh']:.2f} ({elapsed:.0f}s)")

# pair 분석
print(f"\n=== computing pairs ===")
rows = []
names = list(single.keys())
baseline_d_cal = single['M01_S42_M20_127']['stats']['cal']
print(f"baseline D_SMA42 Cal = {baseline_d_cal:.2f}\n")

for i, a in enumerate(names):
    for j, b in enumerate(names):
        if j <= i: continue
        ra = single[a]['rt']
        rb = single[b]['rt']
        df = pd.DataFrame({'a': ra, 'b': rb}).dropna()
        if len(df) < 100: continue
        corr = df.corr().iloc[0, 1]
        # EW combined: 두 equity 평균
        ea = single[a]['eq'].resample('1D').last()
        eb = single[b]['eq'].resample('1D').last()
        idx = ea.index.intersection(eb.index)
        # 정규화 후 EW
        ea_n = ea.loc[idx] / ea.loc[idx].iloc[0]
        eb_n = eb.loc[idx] / eb.loc[idx].iloc[0]
        ew = (ea_n + eb_n) / 2
        ew_stats = compute_stats(ew)
        # yearly Cal sigma (consistency)
        ew_rt = ew.pct_change().dropna()
        yearly_cal = []
        for yr, sub in ew_rt.groupby(ew_rt.index.year):
            if len(sub) < 30: continue
            eq_yr = (1 + sub).cumprod()
            cagr_yr = eq_yr.iloc[-1] - 1
            mdd_yr = ((eq_yr - eq_yr.cummax()) / eq_yr.cummax()).min()
            cal_yr = cagr_yr / abs(mdd_yr) if mdd_yr != 0 else 0
            yearly_cal.append(cal_yr)
        ymin = min(yearly_cal) if yearly_cal else 0
        yavg = np.mean(yearly_cal) if yearly_cal else 0
        rows.append(dict(
            a=a, b=b, corr=corr,
            a_cal=single[a]['stats']['cal'], b_cal=single[b]['stats']['cal'],
            ew_cal=ew_stats['cal'], ew_cagr=ew_stats['cagr'], ew_mdd=ew_stats['mdd'], ew_sh=ew_stats['sh'],
            ymin=float(ymin), yavg=float(yavg),
        ))

df_pairs = pd.DataFrame(rows)
df_pairs.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'v22_1d_pair_sweep.csv'), index=False)

# 필터 결과
print(f"\n=== top pairs by EW Cal ===")
top = df_pairs.sort_values('ew_cal', ascending=False).head(10)
for _, r in top.iterrows():
    print(f"  {r['a'][:20]:20s} × {r['b'][:20]:20s} corr={r['corr']:+.2f} sCal={r['a_cal']:.2f}/{r['b_cal']:.2f} EW Cal={r['ew_cal']:.2f} CAGR={r['ew_cagr']:.1%} MDD={r['ew_mdd']:.1%} ymin={r['ymin']:+.2f}")

print(f"\n=== filtered: corr<0.5 AND a_cal>=2.0 AND b_cal>=2.0 AND ew_cal>{baseline_d_cal:.2f} ===")
filtered = df_pairs[
    (df_pairs['corr'] < 0.5) &
    (df_pairs['a_cal'] >= 2.0) & (df_pairs['b_cal'] >= 2.0) &
    (df_pairs['ew_cal'] > baseline_d_cal)
].sort_values('ew_cal', ascending=False)
print(f"  matches: {len(filtered)}")
for _, r in filtered.head(10).iterrows():
    print(f"  {r['a'][:20]:20s} × {r['b'][:20]:20s} corr={r['corr']:+.2f} EW Cal={r['ew_cal']:.2f} CAGR={r['ew_cagr']:.1%} MDD={r['ew_mdd']:.1%} ymin={r['ymin']:+.2f}")

# 가장 다양성 있는 (corr 가장 낮은) 페어
print(f"\n=== lowest corr pairs (top 5) ===")
low_corr = df_pairs.sort_values('corr').head(5)
for _, r in low_corr.iterrows():
    print(f"  {r['a'][:20]:20s} × {r['b'][:20]:20s} corr={r['corr']:+.2f} sCal={r['a_cal']:.2f}/{r['b_cal']:.2f} EW Cal={r['ew_cal']:.2f}")

print("\n저장: v22_1d_pair_sweep.csv")
