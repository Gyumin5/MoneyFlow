"""V25 LMAX 확장 — 옵션 B (K2 step 축소) × C (긴 SMA) 매트릭스.

grid:
  LMAX ∈ {4, 6, 8, 10}
  k2_period ∈ {7, 14, 21}
  k2_step ∈ {0.025, 0.05}

BTC cap: period=42, step=0.035 고정.
각 케이스 단독 sleeve + alloc 60/25/15 + L 분포.
"""
from __future__ import annotations
import sys, os, importlib.util
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
spec = importlib.util.spec_from_file_location("bt_cross", "/tmp/bt_fut_cross.py")
assert spec and spec.loader
bt_cross = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt_cross)
os.environ['DRIFT_HEALTH_MODE'] = 'refill'


def metrics(s):
    s = s.dropna()
    if len(s) < 30: return None
    yrs = (s.index[-1] - s.index[0]).days / 365.25
    if yrs <= 0: return None
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1 / yrs) - 1
    peak = s.cummax(); mdd = float((s / peak - 1).min())
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return dict(CAGR=cagr, MDD=mdd, Cal=cal)


bars0, funding0 = bt_cross.load_data('D')
btc_close = pd.Series(bars0['BTC']['Close'].values, index=bars0['BTC'].index)
btc_sma42 = btc_close.rolling(42).mean()
btc_ratio = btc_close / btc_sma42


def build_signal(lmin, lmax, k2_period, k2_step,
                 btc_base=1.015, btc_step=0.035, k2_base=1.025):
    """L = min(BTC_cap, K2_per_coin). 오름차순 평가 (큰 thr 가 덮어쓰기)."""
    # BTC cap (period 42, step 0.035 고정)
    btc_thr = [(btc_base + (k - lmin - 1) * btc_step, float(k)) for k in range(lmin + 1, lmax + 1)]
    btc_thr.sort()
    btc_cap = pd.Series(float(lmin), index=btc_ratio.index)
    for thr, lev in btc_thr:
        btc_cap = btc_cap.where(~(btc_ratio > thr), lev)
    btc_cap = btc_cap.shift(1).ffill().fillna(float(lmin))

    k2_thr = [(k2_base + (k - lmin - 1) * k2_step, float(k)) for k in range(lmin + 1, lmax + 1)]
    k2_thr.sort()
    out = {}
    for c in bars0:
        close = bars0[c]['Close']
        sma = close.rolling(k2_period).mean()
        ratio = close / sma
        sig = pd.Series(float(lmin), index=close.index)
        for thr, lev in k2_thr:
            sig = sig.where(~(ratio > thr), lev)
        sig = sig.shift(1).ffill().fillna(float(lmin))
        idx = sig.index.intersection(btc_cap.index)
        out[c] = pd.Series(np.minimum(sig.loc[idx].values, btc_cap.loc[idx].values), index=idx)
    return out


def run_fut_bt(lev_signal):
    return bt_cross.run(bars0, funding0, interval='D', leverage=lev_signal,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03,
        universe_size=3, selection='greedy', cap=1/3,
        tx_cost=0.0006, maint_rate=0.004, vol_days=90, vol_threshold=0.05,
        canary_hyst=0.015, health_mode='mom2vol',
        start_date='2020-10-01', end_date='2026-05-13')['_equity']


def L_dist(sig):
    arr = np.concatenate([s.dropna().astype(int).values for s in sig.values()])
    if len(arr) == 0: return {}
    u, c = np.unique(arr, return_counts=True)
    return {int(k): float(v / c.sum() * 100) for k, v in zip(u, c)}


BUF_STOCK, BUF_SPOT, BUF_FUT = 0.07, 0.01, 0.01
RES_DIR = '/home/gmoh/mon/251229/strategies/cap_defend/research/alloc_reopt_2026_05'
stock_eq = pd.read_csv(f'{RES_DIR}/stock_equity.csv', index_col='Date', parse_dates=True)['Value']
spot_eq = pd.read_csv(f'{RES_DIR}/spot_equity.csv', index_col='Date', parse_dates=True)['Value']


def alloc_bt(stock, spot, fut, t1=0.20, t3u=0.20):
    target = np.array([0.60, 0.25, 0.15])
    val = target.copy()
    r_st = stock.pct_change().fillna(0) * (1 - BUF_STOCK)
    r_sp = spot.pct_change().fillna(0) * (1 - BUF_SPOT)
    r_ft = fut.pct_change().fillna(0) * (1 - BUF_FUT)
    btc_sub = btc_close.reindex(stock.index).ffill()
    btc_sma_sub = btc_sub.rolling(42).mean()
    btc_canary = (btc_sub > btc_sma_sub * 1.015).shift(1).fillna(False)
    eq = [val.sum()]
    for i in range(1, len(stock.index)):
        val[0] *= 1 + r_st.iloc[i]; val[1] *= 1 + r_sp.iloc[i]; val[2] *= 1 + r_ft.iloc[i]
        tot = val.sum(); cur = val / tot if tot > 0 else target.copy()
        ht = abs(cur - target).sum() / 2
        canary = [True, bool(btc_canary.iloc[i]), bool(btc_canary.iloc[i])]
        rel = max(((target[j] - cur[j]) / target[j]) if (canary[j] and target[j] > 0) else 0 for j in range(3))
        if ht >= t1 or rel >= t3u: val = tot * target
        eq.append(val.sum())
    return pd.Series(eq, index=stock.index)


def align(s, start, end):
    return s[(s.index >= start) & (s.index <= end)].reindex(pd.date_range(start, end, freq='D')).ffill().dropna()


LMAX_GRID = [4, 6, 8, 10]
P_GRID = [3, 5, 7, 14, 21]
STEP_GRID = [0.025, 0.05]

print("=" * 120)
print("V25 LMAX×SMA×step 매트릭스 (B+C 옵션)")
print("=" * 120)
print(f"{'LMAX':>5}{'SMA':>5}{'step':>7} | {'solo CAGR':>10}{'solo MDD':>10}{'solo Cal':>10} | {'alloc CAGR':>11}{'alloc MDD':>10}{'alloc Cal':>10} | L 분포 (top 5)")
print("-" * 120)

results = []
for lmax in LMAX_GRID:
    for period in P_GRID:
        for step in STEP_GRID:
            sig = build_signal(2, lmax, period, step)
            dist = L_dist(sig)
            top = sorted(dist.items(), key=lambda x: -x[1])[:5]
            try:
                fut_eq = run_fut_bt(sig)
                m_solo = metrics(fut_eq)
                start = max(stock_eq.index[0], spot_eq.index[0], fut_eq.index[0])
                end = min(stock_eq.index[-1], spot_eq.index[-1], fut_eq.index[-1])
                st_a = align(stock_eq, start, end); sp_a = align(spot_eq, start, end); ft_a = align(fut_eq, start, end)
                alloc_eq = alloc_bt(st_a, sp_a, ft_a)
                m_alloc = metrics(alloc_eq)
                dist_str = " ".join([f"L{k}:{v:.1f}%" for k, v in top])
                print(f"{lmax:>5}{period:>5}{step:>7.3f} | {m_solo['CAGR']*100:>9.1f}%{m_solo['MDD']*100:>9.1f}%{m_solo['Cal']:>10.2f} | {m_alloc['CAGR']*100:>10.1f}%{m_alloc['MDD']*100:>9.1f}%{m_alloc['Cal']:>10.2f} | {dist_str}", flush=True)
                results.append(dict(lmax=lmax, period=period, step=step,
                                   solo_cal=m_solo['Cal'], solo_cagr=m_solo['CAGR'], solo_mdd=m_solo['MDD'],
                                   alloc_cal=m_alloc['Cal'], alloc_cagr=m_alloc['CAGR'], alloc_mdd=m_alloc['MDD'],
                                   dist=dist))
            except Exception as e:
                print(f"{lmax:>5}{period:>5}{step:>7.3f} | ERROR: {e}")

print("\n" + "=" * 60)
print("Top 10 by alloc Cal")
print("=" * 60)
for r in sorted(results, key=lambda r: -r['alloc_cal'])[:10]:
    print(f"  LMAX={r['lmax']:>2} SMA={r['period']:>2} step={r['step']:.3f} | "
          f"alloc Cal={r['alloc_cal']:.2f} CAGR={r['alloc_cagr']*100:.1f}% MDD={r['alloc_mdd']*100:.1f}% | "
          f"solo Cal={r['solo_cal']:.2f}")

print("\nTop 10 by solo Cal")
print("=" * 60)
for r in sorted(results, key=lambda r: -r['solo_cal'])[:10]:
    print(f"  LMAX={r['lmax']:>2} SMA={r['period']:>2} step={r['step']:.3f} | "
          f"solo Cal={r['solo_cal']:.2f} CAGR={r['solo_cagr']*100:.1f}% MDD={r['solo_mdd']*100:.1f}% | "
          f"alloc Cal={r['alloc_cal']:.2f}")
