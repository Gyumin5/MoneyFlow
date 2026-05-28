"""V25 L 상한 5~20x 확장 BT — BTC cap / K2 per-coin 임계값 자동 조정.

룰:
  - LMIN=2 고정 (방어 하한)
  - LMAX = 4, 5, 6, 8, 10, 15, 20 그리드
  - BTC cap thresholds: 1.0 위로 균등 분할 + step=0.035 기준
      L=LMIN(2): ratio<=1.015 즉 default
      L=k (k=3..LMAX): ratio > 1.015 + (k-3)*step_btc
  - K2 per-coin thresholds: step=0.05 기준
      L=k (k=3..LMAX): ratio > 1.025 + (k-3)*step_k2
  - per-coin final L = min(BTC_cap_L, K2_L)

기존 LMAX=4 케이스 일치성:
  BTC: L3 thr=1.015, L4 thr=1.050 (=1.015+0.035)
  K2:  L3 thr=1.025, L4 thr=1.075 (=1.025+0.050)

청산 모델 (bt_cross.run): maint_rate=0.004 고정. L 높아질수록 liquidation 확률 급증.
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
    sh = (s.pct_change().mean() / s.pct_change().std() * np.sqrt(365)) if s.pct_change().std() > 0 else 0
    return dict(CAGR=cagr, MDD=mdd, Cal=cal, Sharpe=sh)


bars0, funding0 = bt_cross.load_data('D')
btc_close = pd.Series(bars0['BTC']['Close'].values, index=bars0['BTC'].index)
btc_sma42 = btc_close.rolling(42).mean()
btc_ratio = btc_close / btc_sma42


def build_K2_extended(lmin: int, lmax: int, step_btc: float = 0.035, step_k2: float = 0.050,
                      btc_base: float = 1.015, k2_base: float = 1.025,
                      k2_period: int = 7):
    """LMIN..LMAX 사이 정수 L 시그널 생성. min(BTC_cap, per_coin_K2)."""
    # BTC cap: L=LMIN default. k=LMIN+1..LMAX 각각 threshold = btc_base + (k-LMIN-1)*step_btc
    btc_thresholds = [(btc_base + (k - lmin - 1) * step_btc, float(k)) for k in range(lmin + 1, lmax + 1)]
    # 오름차순: 낮은 thr 먼저 적용 후 높은 thr 가 덮어쓰기 → 최종 = 가장 큰 만족 L
    btc_thresholds_asc = sorted(btc_thresholds, key=lambda x: x[0])

    btc_cap = pd.Series(float(lmin), index=btc_ratio.index)
    for thr, lev in btc_thresholds_asc:
        btc_cap = btc_cap.where(~(btc_ratio > thr), lev)
    btc_cap = btc_cap.shift(1).ffill().fillna(float(lmin))

    k2_thresholds = [(k2_base + (k - lmin - 1) * step_k2, float(k)) for k in range(lmin + 1, lmax + 1)]
    k2_thresholds_asc = sorted(k2_thresholds, key=lambda x: x[0])

    out = {}
    for c in bars0:
        close = bars0[c]['Close']
        sma = close.rolling(k2_period).mean()
        ratio = close / sma
        sig = pd.Series(float(lmin), index=close.index)
        for thr, lev in k2_thresholds_asc:
            sig = sig.where(~(ratio > thr), lev)
        sig = sig.shift(1).ffill().fillna(float(lmin))
        idx = sig.index.intersection(btc_cap.index)
        out[c] = pd.Series(np.minimum(sig.loc[idx].values, btc_cap.loc[idx].values), index=idx)
    return out, btc_cap


def run_fut_bt(leverage, tx=0.0006, maint=0.004):
    return bt_cross.run(bars0, funding0, interval='D', leverage=leverage,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03,
        universe_size=3, selection='greedy', cap=1/3,
        tx_cost=tx, maint_rate=maint, vol_days=90, vol_threshold=0.05,
        canary_hyst=0.015, health_mode='mom2vol',
        start_date='2020-10-01', end_date='2026-05-13')


# ─── L 분포 통계 ───
def L_distribution(sig_dict, btc_cap):
    """각 coin 의 per-date L 분포 (counts per integer L)."""
    all_L = []
    for c, s in sig_dict.items():
        all_L.extend(s.dropna().astype(int).tolist())
    arr = np.array(all_L)
    if len(arr) == 0: return {}
    unique, counts = np.unique(arr, return_counts=True)
    pct = counts / counts.sum() * 100
    return {int(u): float(p) for u, p in zip(unique, pct)}


# ─── alloc 60/25/15 + buffer ───
BUF_STOCK, BUF_SPOT, BUF_FUT = 0.07, 0.01, 0.01
RES_DIR = '/home/gmoh/mon/251229/strategies/cap_defend/research/alloc_reopt_2026_05'
stock_eq = pd.read_csv(f'{RES_DIR}/stock_equity.csv', index_col='Date', parse_dates=True)['Value']
spot_eq = pd.read_csv(f'{RES_DIR}/spot_equity.csv', index_col='Date', parse_dates=True)['Value']


def alloc_bt(stock, spot, fut, w_st=0.60, w_sp=0.25, w_ft=0.15, t1=0.20, t3u=0.20):
    r_st = stock.pct_change().fillna(0) * (1 - BUF_STOCK)
    r_sp = spot.pct_change().fillna(0) * (1 - BUF_SPOT)
    r_ft = fut.pct_change().fillna(0) * (1 - BUF_FUT)
    target = np.array([w_st, w_sp, w_ft])
    val = target.copy()
    btc_sub = btc_close.reindex(stock.index).ffill()
    btc_sma_sub = btc_sub.rolling(42).mean()
    btc_canary = (btc_sub > btc_sma_sub * 1.015).shift(1).fillna(False)
    eq_list = [val.sum()]
    for i in range(1, len(stock.index)):
        val[0] *= (1 + r_st.iloc[i]); val[1] *= (1 + r_sp.iloc[i]); val[2] *= (1 + r_ft.iloc[i])
        total = val.sum()
        cur_w = val / total if total > 0 else target.copy()
        ht = abs(cur_w - target).sum() / 2
        canary = [True, bool(btc_canary.iloc[i]), bool(btc_canary.iloc[i])]
        rel_under = max(((target[j] - cur_w[j]) / target[j]) if (canary[j] and target[j] > 0) else 0 for j in range(3))
        if (ht >= t1) or (rel_under >= t3u):
            val = total * target
        eq_list.append(val.sum())
    return pd.Series(eq_list, index=stock.index)


def align(s, start, end):
    return s[(s.index >= start) & (s.index <= end)].reindex(pd.date_range(start, end, freq='D')).ffill().dropna()


# ─── 메인 ───
print("=" * 100)
print("V25 LMAX 확장 BT — LMIN=2 고정, LMAX 그리드")
print("=" * 100)

LMAX_GRID = [4, 5, 6, 8, 10, 15, 20]
results = {}

for lmax in LMAX_GRID:
    print(f"\n[LMAX={lmax}] BT 실행 ...", flush=True)
    sig, btc_cap = build_K2_extended(lmin=2, lmax=lmax)
    L_dist = L_distribution(sig, btc_cap)
    print(f"  L 분포: {sorted(L_dist.items())}")
    try:
        res = run_fut_bt(sig)
        fut_eq = res['_equity']
        m_solo = metrics(fut_eq)
        n_liq = res.get('_liquidations', 0) if isinstance(res, dict) else 0
        # alloc
        start = max(stock_eq.index[0], spot_eq.index[0], fut_eq.index[0])
        end = min(stock_eq.index[-1], spot_eq.index[-1], fut_eq.index[-1])
        st_a = align(stock_eq, start, end); sp_a = align(spot_eq, start, end); ft_a = align(fut_eq, start, end)
        alloc_eq = alloc_bt(st_a, sp_a, ft_a)
        m_alloc = metrics(alloc_eq)
        results[lmax] = dict(solo=m_solo, alloc=m_alloc, L_dist=L_dist, n_liq=n_liq)
        print(f"  단독 sleeve: CAGR={m_solo['CAGR']*100:>7.1f}% MDD={m_solo['MDD']*100:>6.1f}% Cal={m_solo['Cal']:>5.2f} liq={n_liq}")
        print(f"  alloc 60/25/15: CAGR={m_alloc['CAGR']*100:>6.1f}% MDD={m_alloc['MDD']*100:>5.1f}% Cal={m_alloc['Cal']:>5.2f}")
    except Exception as e:
        print(f"  ERROR: {e}")
        results[lmax] = {'error': str(e)}

# 요약 표
print("\n" + "=" * 100)
print("요약")
print("=" * 100)
print(f"{'LMAX':<6}{'단독 CAGR':>12}{'단독 MDD':>11}{'단독 Cal':>10}{'청산':>6}"
      f"{'alloc CAGR':>12}{'alloc MDD':>11}{'alloc Cal':>10}")
for lmax, r in results.items():
    if 'error' in r:
        print(f"{lmax:<6} ERROR: {r['error']}")
        continue
    s = r['solo']; a = r['alloc']
    print(f"{lmax:<6}{s['CAGR']*100:>11.1f}%{s['MDD']*100:>10.1f}%{s['Cal']:>10.2f}{r['n_liq']:>6}"
          f"{a['CAGR']*100:>11.1f}%{a['MDD']*100:>10.1f}%{a['Cal']:>10.2f}")

# L 분포 표
print("\nL 분포 (전체 코인-날짜 비율 %):")
all_levels = sorted(set().union(*[r.get('L_dist', {}).keys() for r in results.values() if 'L_dist' in r]))
header = "LMAX  " + "".join([f"L{l:<3} " for l in all_levels])
print(header)
for lmax, r in results.items():
    if 'L_dist' not in r: continue
    row = f"{lmax:<6}" + "".join([f"{r['L_dist'].get(l, 0):>4.1f}%" for l in all_levels])
    print(row)
