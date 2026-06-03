"""코인 spot V24 + fut V25 — multi-snap vs single-snap 비교 BT.

spot: leverage=1, n_snapshots ∈ {1, 3, 5, 7}, snap_interval ∈ {21, 42, 69, 95, 217}
fut:  leverage=K2 signal, n_snapshots ∈ {1, 3, 5}, snap_interval ∈ {21, 42, 69, 95}

기간: 2020-10-01 ~ 2026-05-13 (전체).
"""
from __future__ import annotations
import sys, os, time, importlib.util
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
    cagr = (s.iloc[-1] / s.iloc[0]) ** (1/yrs) - 1
    peak = s.cummax(); mdd = float((s/peak - 1).min())
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return dict(CAGR=cagr, MDD=mdd, Cal=cal)


bars0, funding0 = bt_cross.load_data('D')
btc_close = pd.Series(bars0['BTC']['Close'].values, index=bars0['BTC'].index)
btc_sma42 = btc_close.rolling(42).mean()
btc_ratio = btc_close / btc_sma42
btc_cap_signal = pd.Series(np.where(btc_ratio > 1.05, 4.0,
                          np.where(btc_ratio > 1.015, 3.0, 2.0)),
                          index=btc_ratio.index).shift(1).ffill().fillna(2.0)


def signal_K2(period=7, h=0.025):
    out = {}
    for c in bars0:
        close = bars0[c]['Close']
        sma = close.rolling(period).mean()
        ratio = close / sma
        sig = pd.Series(np.where(ratio > (1 + h*3), 4.0,
                        np.where(ratio > (1 + h), 3.0, 2.0)), index=close.index)
        sig = sig.shift(1).ffill().fillna(2.0)
        idx = sig.index.intersection(btc_cap_signal.index)
        out[c] = pd.Series(np.minimum(sig.loc[idx].values, btc_cap_signal.loc[idx].values), index=idx)
    return out


def run_bt(leverage, n_snap, snap_int, drift_thr):
    return bt_cross.run(bars0, funding0, interval='D', leverage=leverage,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        n_snapshots=n_snap, snap_interval_bars=snap_int, drift_threshold=drift_thr,
        universe_size=3, selection='greedy', cap=1/3,
        tx_cost=0.0006, maint_rate=0.004, vol_days=90, vol_threshold=0.05,
        canary_hyst=0.015, health_mode='mom2vol',
        start_date='2020-10-01', end_date='2026-05-13')


def main():
    t0 = time.time()
    # snap_interval = n * stagger 만 사용 (균등 stagger)
    SPOT_GRID = {
        1: [21, 42, 69, 95, 126, 168, 217],
        3: [21, 42, 63, 84, 105, 126, 168, 210],
        5: [20, 40, 60, 80, 100, 125, 160, 200],
        7: [21, 42, 63, 84, 105, 147, 175, 210],
    }
    print("=" * 90)
    print("코인 SPOT V24 (leverage=1, D_SMA42) — snap_int 가 n 의 배수만")
    print("=" * 90)
    print(f"  {'n_snap':>6} {'snap_int':>8} {'stagger':>7} {'thr':>5} {'CAGR':>8} {'MDD':>8} {'Cal':>6}")
    for ns, si_list in SPOT_GRID.items():
        for si in si_list:
            for thr in (0.05, 0.10):
                try:
                    res = run_bt(leverage=1.0, n_snap=ns, snap_int=si, drift_thr=thr)
                    eq = res['_equity']; m = metrics(eq)
                    if m:
                        print(f"  {ns:>6} {si:>8} {si//ns:>7} {thr:>5.2f} {m['CAGR']*100:>7.1f}% {m['MDD']*100:>7.1f}% {m['Cal']:>6.2f}")
                except Exception as e:
                    print(f"  ERR ns={ns} si={si} thr={thr}: {e}")

    print()
    print("=" * 90)
    print("코인 FUT V25 (K2 dynamic L, D_SMA42)")
    print("=" * 90)
    FUT_GRID = {
        1: [21, 42, 63, 84, 95, 126],
        3: [21, 42, 63, 84, 96, 126],
        5: [20, 40, 60, 80, 95, 120],
    }
    print(f"  {'n_snap':>6} {'snap_int':>8} {'stagger':>7} {'thr':>5} {'CAGR':>8} {'MDD':>8} {'Cal':>6}")
    sig = signal_K2(7, 0.025)
    for ns, si_list in FUT_GRID.items():
        for si in si_list:
            for thr in (0.03, 0.05, 0.10):
                try:
                    res = run_bt(leverage=sig, n_snap=ns, snap_int=si, drift_thr=thr)
                    eq = res['_equity']; m = metrics(eq)
                    if m:
                        print(f"  {ns:>6} {si:>8} {si//ns:>7} {thr:>5.2f} {m['CAGR']*100:>7.1f}% {m['MDD']*100:>7.1f}% {m['Cal']:>6.2f}")
                except Exception as e:
                    print(f"  ERR ns={ns} si={si} thr={thr}: {e}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
