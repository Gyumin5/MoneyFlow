"""K2 하방 L1 티어 추가 검증.

현행 per-coin: close/SMA7 > 1.075 → L4, > 1.025 → L3, else L2 (하한 L2)
제안: L2 아래에 L1 추가 — ratio <= thr_low → L1 (하방 디레버리지)
  thr_low 후보: 1.000 / 0.975 / 0.950

가설: 붕괴장에서 SMA 아래로 빠질 때 L2(2x) 대신 L1(1x) → MDD/청산↓
반론: 모멘텀은 pullback whipsaw 비용 → trade-off. BT 로 측정.

비교: baseline(하한 L2) vs L1 변형 3종. fut sleeve + alloc 60/25/15.
BNB+SOL 제외, baseline spec (ms=18 ml=127 sn=95 n=5).
look-ahead 차단: build 에서 shift(1) 유지.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from unified_backtest import load_data

START = "2020-10-01"; END = "2026-05-29"
EXCLUDE = ['BNB', 'SOL']
K2_HYST = 0.025


def build_K2_with_L1(bars, thr_low=None, l_floor=1.0,
                     btc_cap_sma_period=42, btc_cap_thr_mid=1.015, btc_cap_thr_max=1.05,
                     k2_sma_period=7, k2_hyst=K2_HYST,
                     l_min=2.0, l_mid=3.0, l_max=4.0):
    """thr_low=None 이면 현행(하한 L2). thr_low 지정 시 ratio<=thr_low → l_floor(L1)."""
    btc_df = bars.get('BTC')
    if btc_df is None: return {}
    btc_close = pd.Series(btc_df['Close'].values, index=btc_df.index)
    btc_sma = btc_close.rolling(btc_cap_sma_period).mean()
    btc_ratio = btc_close / btc_sma
    btc_cap = pd.Series(np.where(btc_ratio > btc_cap_thr_max, l_max,
                        np.where(btc_ratio > btc_cap_thr_mid, l_mid, l_min)),
                        index=btc_ratio.index).shift(1).ffill().fillna(l_min)
    thr_max = 1.0 + k2_hyst * 3
    thr_mid = 1.0 + k2_hyst
    out = {}
    for coin in bars:
        close = bars[coin]['Close']
        sma = close.rolling(k2_sma_period).mean()
        ratio = close / sma
        base = np.where(ratio > thr_max, l_max, np.where(ratio > thr_mid, l_mid, l_min))
        if thr_low is not None:
            base = np.where(ratio <= thr_low, l_floor, base)
        pc = pd.Series(base, index=close.index).shift(1).ffill().fillna(l_min)
        idx = pc.index.intersection(btc_cap.index)
        # 주의: btc_cap 도 하한이므로 min 적용. l1 변형은 per-coin 만 낮춤 → min 으로 자동 반영
        out[coin] = pd.Series(np.minimum(pc.loc[idx].values, btc_cap.loc[idx].values), index=idx)
    return out


def run_fut(k2):
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    os.environ['ANCHOR_TRADE_MODE'] = 'on'
    from backtest_futures_v25 import run as fbt_run
    bars_full, funding = load_data('D')
    bars = {c: df for c, df in bars_full.items() if c not in EXCLUDE}
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
        tx_cost=0.0006, maint_rate=0.004,
        sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
        canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=5, snap_interval_bars=95,
        start_date=START, end_date=END)
    return m.get('_equity') if m else None


def run_spot():
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    from unified_backtest import run as bt_run
    bars, _ = load_data('D')
    m = bt_run(bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=20, mom_long_bars=127,
        vol_threshold=0.05, vol_mode='daily',
        n_snapshots=7, snap_interval_bars=217,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0, bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol', exclude_assets=frozenset(EXCLUDE))
    return m.get('_equity') if m else None


def run_stock():
    from bt_stock_mom3 import run_multi_3mom
    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    ranked, mom_off, mom_def, canary = precompute(pdf, [30, 72, 230], [42, 63, 126])
    return run_multi_3mom(pdf, ranked, mom_off, mom_def, canary,
                          pd.Timestamp(START), pd.Timestamp(END), anchor=0,
                          drift_thr=0.05, cash_buf=0.07, ms=30, mid=72, ml=230,
                          snap_int=69, n_snaps=3)


def build_alloc(eq_st, eq_sp, eq_fu, w=(0.60, 0.25, 0.15)):
    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)
    s = w[0]*eq_st.loc[common].pct_change().fillna(0) + \
        w[1]*eq_sp.loc[common].pct_change().fillna(0) + \
        w[2]*eq_fu.loc[common].pct_change().fillna(0)
    return (1 + s).cumprod()


def metrics(eq):
    if eq is None or len(eq.dropna()) < 30: return None
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1]/eq.iloc[0]) ** (1/yrs) - 1
    peak = eq.cummax(); mdd = (eq/peak - 1).min()
    rets = eq.pct_change().dropna()
    sh = rets.mean()/rets.std()*np.sqrt(252) if rets.std() > 0 else 0
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return cagr*100, mdd*100, sh, cal


def main():
    t0 = time.time()
    bars_full, _ = load_data('D')
    eq_st = run_stock()
    eq_sp = run_spot()
    print(f"[stock] CAGR {metrics(eq_st)[0]:.1f}% | [spot] CAGR {metrics(eq_sp)[0]:.1f}%\n")

    variants = [
        ('baseline (하한 L2)', None, None),
        ('L1 @ ratio<=1.000', 1.000, 1.0),
        ('L1 @ ratio<=0.975', 0.975, 1.0),
        ('L1 @ ratio<=0.950', 0.950, 1.0),
    ]
    print(f"{'variant':<22} {'fut_CAGR':>9} {'fut_MDD':>8} {'fut_Cal':>8} | {'al_CAGR':>8} {'al_MDD':>8} {'al_Sh':>6} {'al_Cal':>7}")
    for tag, thr_low, lf in variants:
        k2 = build_K2_with_L1(bars_full, thr_low=thr_low, l_floor=lf if lf else 1.0)
        # exclude 적용은 run_fut 내부 bars filter 에서. k2 는 전체 코인 가지지만 held 만 사용됨.
        eq_fu = run_fut(k2)
        if eq_fu is None:
            print(f"{tag:<22} FAILED"); continue
        m_fu = metrics(eq_fu)
        alloc = build_alloc(eq_st, eq_sp, eq_fu)
        m_al = metrics(alloc)
        print(f"{tag:<22} {m_fu[0]:>8.1f}% {m_fu[1]:>+7.1f}% {m_fu[3]:>8.2f} | "
              f"{m_al[0]:>7.1f}% {m_al[1]:>+7.1f}% {m_al[2]:>6.2f} {m_al[3]:>7.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
