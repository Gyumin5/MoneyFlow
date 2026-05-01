"""후보 C 직도입 전 BT 최대 검증 — 5종 항목 일괄.

후보 C
- stock: sd=69 n=3
- spot:  sn=217 n=7 d=0.10
- fut:   sn=57 n=3 d=0.05

검증
1. drift on/off ablation (sleeve 단독, drift 효과 분리)
2. drift 발화 빈도, turnover, rebal 횟수 (whipsaw 정량)
3. tx_cost 민감도: 0.5x, 1x, 2x, 3x
4. yearly Cal 분포 (per sleeve, per year)
5. portfolio 자산 간 동시 drawdown (60/40/0 + 60/30/10)
"""
import os, sys, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START='2020-10-01'; END='2026-04-27'
bars_D, funding = ub.load_data('D')

OFF=('SPY','QQQ','VEA','EEM','EWJ','GLD','PDBC')
DEF=('IEF','BIL','BNDX','GLD','PDBC'); CAN=('EEM',)
ts._g_prices = ts.load_prices(list(set(OFF+DEF+CAN)), start='2014-01-01')
ts._g_ind = ts.precompute(ts._g_prices)

def stock_eq(sd, n=3, tx=0.001):
    sp = ts.SP(offensive=OFF, defensive=DEF, canary_assets=CAN,
        canary_sma=300, canary_hyst=0.020, canary_type='sma',
        health='none', defense='top2', defense_sma=100, def_mom_period=126,
        select='zscore3', n_mom=3, n_sh=3, sharpe_lookback=126,
        weight='ew', crash='none', tx_cost=tx, start=START, end=END, capital=10000.0)
    return tss.run_snapshot(sp, snap_days=sd, n_snap=n)['Value']

def spot_run(sn, n, d, tx=0.004, _trace=None):
    return ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
        sma_days=42, mom_short_days=20, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=tx,
        health_mode='mom2vol', vol_mode='daily', drift_threshold=d,
        snap_interval_bars=sn, start_date=START, end_date=END, _trace=_trace)

def fut_run(sn, n, d, tx=0.0004, _trace=None):
    return ub.run(bars_D, funding, interval='D', asset_type='fut', leverage=3.0,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=n,
        universe_size=3, cap=1/3, tx_cost=tx, maint_rate=0.004,
        health_mode='mom2vol', vol_mode='daily', drift_threshold=d,
        snap_interval_bars=sn, start_date=START, end_date=END, _trace=_trace)

def daily_norm(eq):
    s = eq.resample('1D').last().dropna()
    return s/s.iloc[0]

def stats(eq):
    rt = eq.pct_change().dropna()
    days = (eq.index[-1]-eq.index[0]).days
    cagr = (eq.iloc[-1]/eq.iloc[0])**(365.25/days)-1 if days>0 else 0
    mdd = ((eq-eq.cummax())/eq.cummax()).min()
    sh = rt.mean()/rt.std()*np.sqrt(252) if rt.std()>0 else 0
    cal = cagr/abs(mdd) if mdd!=0 else 0
    return cagr, mdd, sh, cal

def yearly_cal(eq):
    rt = eq.pct_change().dropna()
    out = {}
    for y, sub in rt.groupby(rt.index.year):
        if len(sub) < 30: continue
        eyr = (1+sub).cumprod()
        myr = ((eyr-eyr.cummax())/eyr.cummax()).min()
        cagr_y = eyr.iloc[-1]-1
        cal_y = cagr_y/abs(myr) if myr<0 else 0
        out[y] = (cagr_y, myr, cal_y)
    return out

def trace_metrics(trace):
    rebal_count = sum(1 for t in trace if t.get('rebal'))
    cum_turnover = 0.0
    prev_target = None
    for t in trace:
        if t.get('rebal'):
            tg = t['target'] or {}
            if prev_target is not None:
                ks = set(tg.keys()) | set(prev_target.keys())
                cum_turnover += sum(abs(tg.get(k,0) - prev_target.get(k,0)) for k in ks)
            prev_target = tg
    n_days = len(trace)
    return rebal_count, cum_turnover, cum_turnover/max(rebal_count,1), n_days

print("="*60)
print("후보 C 검증 — 1. drift on/off ablation + turnover")
print("="*60)

# spot — drift 0/0.10
trace_spot_off = []
res_spot_off = spot_run(217, 7, 0.0, _trace=trace_spot_off)
trace_spot_on  = []
res_spot_on  = spot_run(217, 7, 0.10, _trace=trace_spot_on)
trace_fut_off = []
res_fut_off = fut_run(57, 3, 0.0, _trace=trace_fut_off)
trace_fut_on  = []
res_fut_on  = fut_run(57, 3, 0.05, _trace=trace_fut_on)
res_stock = stock_eq(69, 3)

eq_spot_off = daily_norm(res_spot_off['_equity'])
eq_spot_on  = daily_norm(res_spot_on['_equity'])
eq_fut_off  = daily_norm(res_fut_off['_equity'])
eq_fut_on   = daily_norm(res_fut_on['_equity'])
eq_stock    = daily_norm(res_stock)

for label, eq in [('stock_sd69', eq_stock), ('spot_sn217_n7_d0', eq_spot_off), ('spot_sn217_n7_d0.10', eq_spot_on),
                   ('fut_sn57_n3_d0', eq_fut_off), ('fut_sn57_n3_d0.05', eq_fut_on)]:
    cagr,mdd,sh,cal = stats(eq)
    yc = yearly_cal(eq)
    yc_str = ' '.join(f"{y}={v[2]:+.2f}" for y, v in yc.items())
    print(f"  {label:28s}  CAGR={cagr:+.0%}  MDD={mdd:+.0%}  Cal={cal:.2f}  Sh={sh:.2f}")
    print(f"    yearly Cal: {yc_str}")

print("\n=== drift on/off 효과 (Cal 변화) ===")
cagr_off, mdd_off, _, cal_off = stats(eq_spot_off)
cagr_on,  mdd_on,  _, cal_on  = stats(eq_spot_on)
print(f"  spot drift OFF: Cal {cal_off:.2f} CAGR {cagr_off:+.0%} MDD {mdd_off:+.0%}")
print(f"  spot drift ON : Cal {cal_on:.2f} CAGR {cagr_on:+.0%} MDD {mdd_on:+.0%}")
print(f"  spot drift effect: ΔCal {cal_on-cal_off:+.2f}  ΔCAGR {cagr_on-cagr_off:+.0%}")
cagr_off_f, mdd_off_f, _, cal_off_f = stats(eq_fut_off)
cagr_on_f,  mdd_on_f,  _, cal_on_f  = stats(eq_fut_on)
print(f"  fut  drift OFF: Cal {cal_off_f:.2f} CAGR {cagr_off_f:+.0%} MDD {mdd_off_f:+.0%}")
print(f"  fut  drift ON : Cal {cal_on_f:.2f} CAGR {cagr_on_f:+.0%} MDD {mdd_on_f:+.0%}")
print(f"  fut  drift effect: ΔCal {cal_on_f-cal_off_f:+.2f}  ΔCAGR {cagr_on_f-cagr_off_f:+.0%}")

print("\n=== 2. 발화 빈도 / turnover (whipsaw 정량) ===")
sp_off_rb, sp_off_to, sp_off_to_per, sp_off_n = trace_metrics(trace_spot_off)
sp_on_rb,  sp_on_to,  sp_on_to_per,  sp_on_n  = trace_metrics(trace_spot_on)
fu_off_rb, fu_off_to, fu_off_to_per, fu_off_n = trace_metrics(trace_fut_off)
fu_on_rb,  fu_on_to,  fu_on_to_per,  fu_on_n  = trace_metrics(trace_fut_on)
print(f"  spot drift OFF: rebal={sp_off_rb}  total_TO={sp_off_to:.2f}  TO/rebal={sp_off_to_per:.3f}  freq={sp_off_rb/sp_off_n*365:.1f}/yr")
print(f"  spot drift ON : rebal={sp_on_rb }  total_TO={sp_on_to:.2f}  TO/rebal={sp_on_to_per:.3f}  freq={sp_on_rb/sp_on_n*365:.1f}/yr")
print(f"  spot drift 추가 발화: {sp_on_rb-sp_off_rb} 회 ({(sp_on_rb-sp_off_rb)/sp_off_n*365:.1f}/yr 증가)")
print(f"  fut drift OFF: rebal={fu_off_rb}  total_TO={fu_off_to:.2f}  TO/rebal={fu_off_to_per:.3f}  freq={fu_off_rb/fu_off_n*365:.1f}/yr")
print(f"  fut drift ON : rebal={fu_on_rb }  total_TO={fu_on_to:.2f}  TO/rebal={fu_on_to_per:.3f}  freq={fu_on_rb/fu_on_n*365:.1f}/yr")
print(f"  fut drift 추가 발화: {fu_on_rb-fu_off_rb} 회 ({(fu_on_rb-fu_off_rb)/fu_off_n*365:.1f}/yr 증가)")

print("\n=== 3. tx_cost 민감도 (Cal 보존 여부) ===")
print(f"{'tx_mult':10s} {'asset':10s}  Cal   CAGR    MDD    ymin")
for mult, label in [(0.5,'0.5x'),(1.0,'1x'),(2.0,'2x'),(3.0,'3x')]:
    eq_st = daily_norm(stock_eq(69, 3, tx=0.001*mult))
    eq_sp = daily_norm(spot_run(217, 7, 0.10, tx=0.004*mult)['_equity'])
    eq_fu = daily_norm(fut_run(57, 3, 0.05, tx=0.0004*mult)['_equity'])
    for k, eq in [('stock',eq_st),('spot',eq_sp),('fut',eq_fu)]:
        cagr,mdd,sh,cal = stats(eq)
        rt = eq.pct_change().dropna()
        yc = yearly_cal(eq)
        ymin = min(v[2] for v in yc.values()) if yc else 0
        print(f"  tx_{label:5s} {k:6s}    {cal:.2f}  {cagr:+.0%}  {mdd:+.0%}  {ymin:+.2f}")

print("\n=== 4. yearly Cal 분포 (per sleeve, drift ON) ===")
for label, eq in [('stock_sd69',eq_stock),('spot_sn217_d0.10',eq_spot_on),('fut_sn57_d0.05',eq_fut_on)]:
    yc = yearly_cal(eq)
    print(f"\n  {label}:")
    for y, (cagr_y, mdd_y, cal_y) in sorted(yc.items()):
        print(f"    {y}: CAGR {cagr_y:+.1%}  MDD {mdd_y:+.1%}  Cal {cal_y:+.2f}")

print("\n=== 5. portfolio 자산 간 동시 drawdown ===")
def portfolio(s,c,f,ws,wc,wf):
    idx = s.index.intersection(c.index).intersection(f.index)
    rs = s.loc[idx].pct_change().fillna(0)
    rc = c.loc[idx].pct_change().fillna(0)
    rf = f.loc[idx].pct_change().fillna(0)
    return (1 + ws*rs + wc*rc + wf*rf).cumprod(), idx

# 자산별 drawdown 시리즈
def dd_series(eq):
    return (eq-eq.cummax())/eq.cummax()

for label, alloc in [('60/40/0', (0.6,0.4,0.0)), ('60/30/10', (0.6,0.3,0.1))]:
    p, idx = portfolio(eq_stock, eq_spot_on, eq_fut_on, *alloc)
    cagr,mdd,sh,cal = stats(p)
    yc = yearly_cal(p)
    yc_str = ' '.join(f"{y}={v[2]:+.2f}" for y, v in sorted(yc.items()))
    print(f"\n  {label}  Cal={cal:.2f} CAGR={cagr:+.0%} MDD={mdd:+.0%}")
    print(f"    yearly Cal: {yc_str}")
    # 동시 drawdown: 모든 자산이 drawdown <= -5% 인 일수
    s_dd = dd_series(eq_stock.loc[idx])
    c_dd = dd_series(eq_spot_on.loc[idx])
    f_dd = dd_series(eq_fut_on.loc[idx])
    n = len(idx)
    if alloc[2] > 0:
        sim = (s_dd <= -0.05) & (c_dd <= -0.05) & (f_dd <= -0.05)
        print(f"    3자산 동시 -5% DD 일수: {sim.sum()}/{n} ({sim.mean()*100:.1f}%)")
    sim2 = (s_dd <= -0.05) & (c_dd <= -0.05)
    print(f"    stock+spot 동시 -5% DD: {sim2.sum()}/{n} ({sim2.mean()*100:.1f}%)")
    # 상관
    rt_s = eq_stock.loc[idx].pct_change().dropna()
    rt_c = eq_spot_on.loc[idx].pct_change().dropna()
    rt_f = eq_fut_on.loc[idx].pct_change().dropna()
    common = rt_s.index.intersection(rt_c.index).intersection(rt_f.index)
    print(f"    daily return corr: stock-spot={rt_s.loc[common].corr(rt_c.loc[common]):.2f}  stock-fut={rt_s.loc[common].corr(rt_f.loc[common]):.2f}  spot-fut={rt_c.loc[common].corr(rt_f.loc[common]):.2f}")

print("\n=== 종합 ===")
print("- drift 알파 (Cal 변화)")
print("- 발화 빈도/whipsaw")
print("- 비용 stress (tx 3x)")
print("- 연도별 Cal 변동 (ymin 우연성)")
print("- 자산 간 동시 drawdown 빈도")
