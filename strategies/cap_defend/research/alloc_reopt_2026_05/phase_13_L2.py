#!/usr/bin/env python3
"""Phase 13 — L2 (2x) 시나리오 검증."""
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP_DEFEND = os.path.abspath(os.path.join(HERE, '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CAP_DEFEND, '..', '..'))
sys.path.insert(0, CAP_DEFEND)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'trade'))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, HERE)

from backtest_futures_full import load_data, run

# ============ 1) L2 sleeve 단독 ============
print('=== 1) L2 단독 BT (V23 sleeve params, leverage 2) ===')
bars, funding = load_data('D')
cfg_base = dict(
    interval='D', leverage=2.0,
    sma_days=42, mom_short_days=18, mom_long_days=127,
    n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03,
    universe_size=3, selection='greedy', cap=1/3,
    tx_cost=0.0006, maint_rate=0.004,
    vol_days=90, vol_threshold=0.05,
    canary_hyst=0.015, health_mode='mom2vol',
)

# OOS split
for name, s, e in [('Full', '2020-10-01', '2026-05-13'),
                    ('IS 2020-10~2023-12', '2020-10-01', '2023-12-31'),
                    ('OOS 2024-01~2026-05', '2024-01-01', '2026-05-13')]:
    m = run(bars, funding, start_date=s, end_date=e, **cfg_base)
    print(f'  {name}: Cal {m["Cal"]:.2f} / CAGR {m["CAGR"]*100:.0f}% / MDD {m["MDD"]*100:.0f}% / Sharpe {m["Sharpe"]:.2f} / Liq {m["Liq"]}')

# Full BT for equity
m_full = run(bars, funding, start_date='2020-10-01', end_date='2026-05-13', **cfg_base)
fut_L2 = m_full['_equity'].copy()
fut_L2.to_csv(os.path.join(HERE, 'fut_equity_L2.csv'), header=['Value'])
print(f"  L2 청산 이력: {len(m_full['_liq_log'])} 회")
for e in m_full['_liq_log']:
    print(f'    {str(e["date"])[:10]} {e["coin"]} 손실 {e["loss_pct"]*100:.0f}%')

# ============ 2) Alloc grid with L2 ============
print('\n=== 2) L2 alloc grid (제약 ws≥wc≥wf) ===')
stock = pd.read_csv(os.path.join(HERE, 'stock_equity_ext.csv'), index_col=0, parse_dates=True)
spot = pd.read_csv(os.path.join(HERE, 'spot_equity.csv'), index_col=0, parse_dates=True)
fut = pd.read_csv(os.path.join(HERE, 'fut_equity_L2.csv'), index_col=0, parse_dates=True)

stock.index = pd.to_datetime(stock.index).normalize()
spot.index = pd.to_datetime(spot.index, utc=True).tz_convert(None).normalize()
fut.index = pd.to_datetime(fut.index).normalize()
stock = stock.groupby(stock.index).last()
spot = spot.groupby(spot.index).last()
fut = fut.groupby(fut.index).last()

idx = stock.index.intersection(spot.index).intersection(fut.index)
s_r = stock.loc[idx, 'Value'].pct_change().fillna(0)
c_r = spot.loc[idx, 'Value'].pct_change().fillna(0)
f_r = fut.loc[idx, 'Value'].pct_change().fillna(0)
returns = pd.DataFrame({'stock': s_r, 'spot': c_r, 'fut': f_r})


def simulate(returns, w_s, w_c, w_f, trigger, threshold, tx_cost=0.0005):
    tgt = np.array([w_s, w_c, w_f])
    cur = tgt.copy()
    eq = []
    rebal = 0
    R = returns[['stock','spot','fut']].values
    for t in range(len(R)):
        r = R[t]
        cur = cur * (1.0 + r)
        pv = cur.sum()
        if pv <= 0:
            eq.append(pv); continue
        cur_w = cur / pv
        diff = cur_w - tgt
        if trigger == 'T1': metric = np.sum(np.abs(diff))/2
        elif trigger == 'T2': metric = np.max(np.abs(diff))
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.where(tgt > 0, np.abs(diff)/tgt, 0)
            metric = np.max(rel)
        if metric >= threshold:
            turnover = np.sum(np.abs(cur_w-tgt))/2
            pv_new = pv * (1 - tx_cost*2*turnover)
            cur = tgt * pv_new
            pv = pv_new
            rebal += 1
        eq.append(pv)
    return pd.Series(eq, index=returns.index), rebal


def metrics(eq):
    if eq.iloc[-1] <= 0: return dict(Cal=-999, CAGR=-1, MDD=-1, Sharpe=-99)
    yrs = (eq.index[-1]-eq.index[0]).days/365.25
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs)-1
    dr = eq.pct_change().dropna()
    sh = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    mdd = (eq/eq.cummax()-1).min()
    cal = cagr/abs(mdd) if mdd!=0 else 0
    return dict(Cal=float(cal), CAGR=float(cagr), MDD=float(mdd), Sharpe=float(sh))


ratios = []
for ws in range(0,101,5):
    for wc in range(0,101-ws,5):
        wf = 100-ws-wc
        if wf<0: continue
        if ws>=wc and wc>=wf:
            ratios.append((ws/100, wc/100, wf/100))

triggers = {'T1':[0.03,0.05,0.08,0.10,0.12,0.15,0.18,0.20,0.25],
            'T2':[0.02,0.03,0.05,0.07,0.10,0.12,0.15],
            'T3':[0.05,0.10,0.15,0.20,0.30,0.40,0.50]}

rows = []
yr_rows = []
for trig, thrs in triggers.items():
    for thr in thrs:
        for (ws,wc,wf) in ratios:
            eq, rebal = simulate(returns, ws, wc, wf, trig, thr)
            m = metrics(eq)
            rows.append(dict(trigger=trig, thr=thr, w_stock=ws, w_spot=wc, w_fut=wf,
                             rebal=rebal, **m))

df = pd.DataFrame(rows)

# 현재 70/15/15
cur = df[(df.w_stock==0.7)&(df.w_spot==0.15)&(df.w_fut==0.15)].sort_values('Cal', ascending=False).head(5)
print('\n=== L2 + 70/15/15 (현재 비중 유지) ===')
print(cur[['trigger','thr','rebal','Cal','CAGR','MDD','Sharpe']].to_string(index=False))

print('\n=== L2 TOP 15 (full grid) ===')
print(df.nlargest(15, 'Cal')[['trigger','thr','w_stock','w_spot','w_fut','rebal','Cal','CAGR','MDD','Sharpe']].to_string(index=False))

# 비교: L3 baseline (현재) — Cal 2.92, CAGR 54%, MDD -18%
df.to_csv(os.path.join(HERE, 'L2_alloc_grid.csv'), index=False)
print('\n저장: L2_alloc_grid.csv')
