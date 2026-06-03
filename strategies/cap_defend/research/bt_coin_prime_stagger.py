"""코인 spot V24 + fut V25 — PRIME stagger only.

stagger 는 소수(prime)여야 한다 (사용자 제약 — 충돌방지·예외성).
타 자산 stagger 와 충돌 회피: stock=23, spot=31 (현 V24), fut=19 (현 V25).
SPOT 후보: exclude 19, 23 (fut/stock 충돌) → {7, 11, 13, 17, 29, 31, 37, 41, 43, 47, 53}
FUT  후보: exclude 23, 31 (stock/spot 충돌) → {7, 11, 13, 17, 19, 29, 37, 41, 43, 47, 53}
snap_interval = n_snap × stagger.
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


SPOT_PRIMES = [7, 11, 13, 17, 29, 31, 37, 41, 43, 47, 53]   # exclude 19 (fut), 23 (stock)
FUT_PRIMES  = [7, 11, 13, 17, 19, 29, 37, 41, 43, 47, 53]   # exclude 23 (stock), 31 (spot)


def main():
    t0 = time.time()
    print("=" * 100)
    print("코인 SPOT V24 — PRIME stagger only (snap_int = n × prime). exclude 19,23.")
    print("=" * 100)
    print(f"  {'n_snap':>6} {'stagger':>7} {'snap_int':>8} {'thr':>5} {'CAGR':>8} {'MDD':>8} {'Cal':>6}")
    spot_results = []
    for ns in (1, 3, 5, 7):
        for p in SPOT_PRIMES:
            si = ns * p
            if si > 350: continue
            for thr in (0.05, 0.10):
                try:
                    res = run_bt(leverage=1.0, n_snap=ns, snap_int=si, drift_thr=thr)
                    m = metrics(res['_equity'])
                    if m:
                        print(f"  {ns:>6} {p:>7} {si:>8} {thr:>5.2f} {m['CAGR']*100:>7.1f}% {m['MDD']*100:>7.1f}% {m['Cal']:>6.2f}")
                        spot_results.append((ns, p, si, thr, m['CAGR'], m['MDD'], m['Cal']))
                except Exception as e:
                    print(f"  ERR ns={ns} p={p} si={si} thr={thr}: {e}")

    print()
    print("=" * 100)
    print("코인 FUT V25 — PRIME stagger only. exclude 23,31.")
    print("=" * 100)
    print(f"  {'n_snap':>6} {'stagger':>7} {'snap_int':>8} {'thr':>5} {'CAGR':>8} {'MDD':>8} {'Cal':>6}")
    sig = signal_K2(7, 0.025)
    fut_results = []
    for ns in (1, 3, 5, 7):
        for p in FUT_PRIMES:
            si = ns * p
            if si > 250: continue
            for thr in (0.03, 0.05, 0.10):
                try:
                    res = run_bt(leverage=sig, n_snap=ns, snap_int=si, drift_thr=thr)
                    m = metrics(res['_equity'])
                    if m:
                        print(f"  {ns:>6} {p:>7} {si:>8} {thr:>5.2f} {m['CAGR']*100:>7.1f}% {m['MDD']*100:>7.1f}% {m['Cal']:>6.2f}")
                        fut_results.append((ns, p, si, thr, m['CAGR'], m['MDD'], m['Cal']))
                except Exception as e:
                    print(f"  ERR ns={ns} p={p} si={si} thr={thr}: {e}")

    print()
    print("=" * 100)
    print("TOP 10 (by Cal)")
    print("=" * 100)
    spot_results.sort(key=lambda r: -r[6])
    fut_results.sort(key=lambda r: -r[6])
    print("SPOT TOP:")
    for r in spot_results[:10]:
        print(f"  n={r[0]} stag={r[1]} int={r[2]} thr={r[3]:.2f}  CAGR={r[4]*100:.1f}% MDD={r[5]*100:.1f}% Cal={r[6]:.2f}")
    print("FUT TOP:")
    for r in fut_results[:10]:
        print(f"  n={r[0]} stag={r[1]} int={r[2]} thr={r[3]:.2f}  CAGR={r[4]*100:.1f}% MDD={r[5]*100:.1f}% Cal={r[6]:.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
