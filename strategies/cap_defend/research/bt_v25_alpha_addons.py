"""V25 추가 alpha 시그널 테스트 (펀딩/regime/vol).

baseline (V25 현재): LMAX=4 SMA=7 step=0.05 → alloc Cal 4.19

테스트 challenger:
  A) BTC trend regime — BTC>SMA200 → LMAX=6, else LMAX=4
  B) 펀딩 cap — 7d avg funding > thr → LMAX=3
  C) regime+펀딩 결합
  D) 펀딩 ratio scale — funding 음수 시 contrarian L↑

각 결과를 baseline 과 absolute Δ 로 비교.
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
    return dict(CAGR=cagr, MDD=mdd, Cal=cagr / abs(mdd) if mdd < 0 else 0)


bars0, funding0 = bt_cross.load_data('D')
btc_close = pd.Series(bars0['BTC']['Close'].values, index=bars0['BTC'].index)
btc_sma42 = btc_close.rolling(42).mean()
btc_sma200 = btc_close.rolling(200).mean()
btc_ratio = btc_close / btc_sma42
btc_regime = (btc_close > btc_sma200)  # daily bool


def _funding_daily_avg(coin, window_days=7):
    """funding (8h) → daily mean × window_days rolling avg."""
    f = funding0.get(coin)
    if f is None: return None
    daily = f.resample('1D').mean()
    return daily.rolling(window_days).mean()


def build_signal(lmin=2, lmax=4, k2_period=7, k2_step=0.05,
                 btc_base=1.015, btc_step=0.035, k2_base=1.025,
                 regime_aware=False, regime_lmax_bull=None,
                 funding_cap=None, funding_avg_days=7, funding_cap_lmax=3,
                 funding_negative_bonus=False):
    """challenger 시그널 빌더."""
    eff_lmax = lmax
    # BTC cap thresholds
    def btc_cap_for(lmax_eff):
        thr = [(btc_base + (k - lmin - 1) * btc_step, float(k)) for k in range(lmin + 1, lmax_eff + 1)]
        thr.sort()
        cap = pd.Series(float(lmin), index=btc_ratio.index)
        for t, lv in thr:
            cap = cap.where(~(btc_ratio > t), lv)
        return cap.shift(1).ffill().fillna(float(lmin))

    def k2_for(close, lmax_eff):
        thr = [(k2_base + (k - lmin - 1) * k2_step, float(k)) for k in range(lmin + 1, lmax_eff + 1)]
        thr.sort()
        sma = close.rolling(k2_period).mean()
        ratio = close / sma
        sig = pd.Series(float(lmin), index=close.index)
        for t, lv in thr:
            sig = sig.where(~(ratio > t), lv)
        return sig.shift(1).ffill().fillna(float(lmin))

    base_btc_cap = btc_cap_for(eff_lmax)
    if regime_aware and regime_lmax_bull is not None:
        # bull 시 lmax_bull thr cap, bear 시 lmax (=원래) cap
        bull_cap = btc_cap_for(regime_lmax_bull)
        bull_mask = btc_regime.shift(1).fillna(False).reindex(base_btc_cap.index, method='ffill').fillna(False).astype(bool)
        base_btc_cap = pd.Series(np.where(bull_mask.values, bull_cap.values, base_btc_cap.values), index=base_btc_cap.index)

    out = {}
    for coin in bars0:
        close = bars0[coin]['Close']
        lmax_eff_coin = regime_lmax_bull if (regime_aware and regime_lmax_bull) else eff_lmax
        sig = k2_for(close, lmax_eff_coin)

        # funding cap
        if funding_cap is not None:
            fa = _funding_daily_avg(coin, funding_avg_days)
            if fa is not None:
                fa_aligned = fa.reindex(sig.index, method='ffill').shift(1)
                cap_mask = (fa_aligned > funding_cap).fillna(False)
                sig = sig.where(~cap_mask, float(funding_cap_lmax))

        # funding negative bonus (contrarian: funding 음수 → +1 L)
        if funding_negative_bonus:
            fa = _funding_daily_avg(coin, funding_avg_days)
            if fa is not None:
                fa_aligned = fa.reindex(sig.index, method='ffill').shift(1)
                neg_mask = (fa_aligned < 0).fillna(False)
                sig = sig.where(~neg_mask, np.minimum(sig + 1, float(eff_lmax + 1)))

        idx = sig.index.intersection(base_btc_cap.index)
        out[coin] = pd.Series(np.minimum(sig.loc[idx].values, base_btc_cap.loc[idx].values), index=idx)
    return out


def run_fut(lev):
    return bt_cross.run(bars0, funding0, interval='D', leverage=lev,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03,
        universe_size=3, selection='greedy', cap=1/3,
        tx_cost=0.0006, maint_rate=0.004, vol_days=90, vol_threshold=0.05,
        canary_hyst=0.015, health_mode='mom2vol',
        start_date='2020-10-01', end_date='2026-05-13')['_equity']


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


def evaluate(name, sig_args):
    sig = build_signal(**sig_args)
    fut_eq = run_fut(sig)
    m_solo = metrics(fut_eq)
    start = max(stock_eq.index[0], spot_eq.index[0], fut_eq.index[0])
    end = min(stock_eq.index[-1], spot_eq.index[-1], fut_eq.index[-1])
    st_a = align(stock_eq, start, end); sp_a = align(spot_eq, start, end); ft_a = align(fut_eq, start, end)
    alloc_eq = alloc_bt(st_a, sp_a, ft_a)
    m_alloc = metrics(alloc_eq)
    return name, m_solo, m_alloc


cfgs = [
    ("baseline V25 (LMAX=4 SMA=7 step=0.05)", dict()),
    ("B-best (LMAX=4 SMA=7 step=0.025)", dict(k2_step=0.025)),
    ("BC-best (LMAX=6 SMA=7 step=0.025)", dict(lmax=6, k2_step=0.025)),
    ("A1 regime BTC>SMA200: bull LMAX=6 / bear LMAX=4 (step=0.05)",
     dict(regime_aware=True, regime_lmax_bull=6)),
    ("A2 regime bull LMAX=8 / bear LMAX=4 (step=0.05)",
     dict(regime_aware=True, regime_lmax_bull=8)),
    ("A3 regime bull LMAX=6 + step=0.025",
     dict(regime_aware=True, regime_lmax_bull=6, k2_step=0.025)),
    ("B1 funding cap (7d avg > 0.01%/8h → LMAX=3)",
     dict(funding_cap=0.0001, funding_avg_days=7, funding_cap_lmax=3)),
    ("B2 funding cap (7d avg > 0.025%/8h → LMAX=3)",
     dict(funding_cap=0.00025, funding_avg_days=7, funding_cap_lmax=3)),
    ("B3 funding cap (3d avg > 0.05%/8h → LMAX=2)",
     dict(funding_cap=0.0005, funding_avg_days=3, funding_cap_lmax=2)),
    ("C combined: regime LMAX=6 + funding cap 0.025%",
     dict(regime_aware=True, regime_lmax_bull=6, funding_cap=0.00025, funding_avg_days=7, funding_cap_lmax=3)),
    ("D funding negative bonus (음수 funding → +1 L)",
     dict(funding_negative_bonus=True)),
]

print("=" * 110)
print(f"{'cfg':<60} | {'solo CAGR':>10}{'solo MDD':>10}{'solo Cal':>9} | {'alloc CAGR':>11}{'alloc MDD':>10}{'alloc Cal':>9}{'Δ vs base':>11}")
print("-" * 110)

baseline_cal = None
results = []
for name, kw in cfgs:
    try:
        n, ms, ma = evaluate(name, kw)
        if baseline_cal is None:
            baseline_cal = ma['Cal']
        delta = ma['Cal'] - baseline_cal
        results.append((n, ms, ma, delta))
        print(f"{n[:58]:<60} | {ms['CAGR']*100:>9.1f}%{ms['MDD']*100:>9.1f}%{ms['Cal']:>9.2f} | "
              f"{ma['CAGR']*100:>10.1f}%{ma['MDD']*100:>9.1f}%{ma['Cal']:>9.2f}{delta:>+11.2f}", flush=True)
    except Exception as e:
        print(f"{name[:58]:<60} | ERROR: {e}")

print("\n=== alloc Cal Δ 정렬 (baseline=0.00) ===")
for n, _, ma, delta in sorted(results, key=lambda x: -x[3]):
    flag = "★" if delta >= 0.20 else ("·" if abs(delta) < 0.05 else "")
    print(f"  {flag} Δ={delta:>+5.2f} | Cal={ma['Cal']:.2f} CAGR={ma['CAGR']*100:.1f}% MDD={ma['MDD']*100:.1f}% | {n}")
