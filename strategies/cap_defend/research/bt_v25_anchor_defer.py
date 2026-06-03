"""V25 — 앵커 거래 정책 비교.

baseline (ANCHOR_TRADE_MODE=on, 현행): snap fire → 강제 거래
defer    (ANCHOR_TRADE_MODE=defer): snap 은 target 갱신만, drift fire 시만 거래

설계 직관: 앵커일에 새 target 과 cur 의 ht 가 threshold 위면 어차피 거래되고,
threshold 아래면 며칠 후 가격 변동으로 drift 자연 발화 → "지연" 효과만.
거래 수 감소 vs entry/exit lag 의 trade-off 정량 확인.

BNB+SOL 제외 + baseline spec (spot ms=20 ml=127 sn=217 n=7, fut ms=18 ml=127 sn=95 n=5).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

START = "2020-10-01"; END = "2026-05-29"
EXCLUDE = ['BNB', 'SOL']


def run_spot(anchor_mode):
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    os.environ['ANCHOR_TRADE_MODE'] = anchor_mode
    from unified_backtest import run as bt_run, load_data
    bars, _ = load_data('D')
    m = bt_run(bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=20, mom_long_bars=127,
        vol_threshold=0.05, vol_mode='daily',
        n_snapshots=7, snap_interval_bars=217,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol',
        exclude_assets=frozenset(EXCLUDE))
    return m if m else None


def run_fut(anchor_mode):
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    os.environ['ANCHOR_TRADE_MODE'] = anchor_mode
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    bars_full, funding = load_data('D')
    k2 = build_K2_signal(bars_full, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    bars = {c: df for c, df in bars_full.items() if c not in EXCLUDE}
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
        tx_cost=0.0006, maint_rate=0.004,
        sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
        canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=5, snap_interval_bars=95,
        start_date=START, end_date=END)
    return m if m else None


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
    print(f"=== Anchor trade policy 비교 (BNB+SOL 제외, baseline spec) ===\n")

    print("[SPOT D_SMA42 sn=217 n=7]")
    for mode in ['on', 'defer']:
        m = run_spot(mode)
        eq = m['_equity']; rc = m.get('rebal_count', '-'); tc = m.get('trade_count', '-')
        ms = metrics(eq)
        print(f"  ANCHOR_TRADE={mode:<6}: CAGR {ms[0]:5.1f}%  MDD {ms[1]:+6.1f}%  Sharpe {ms[2]:.2f}  Cal {ms[3]:.2f}  rebal={rc}  trades={tc}")

    print("\n[FUT D_SMA42 sn=95 n=5 + dynamic K2]")
    for mode in ['on', 'defer']:
        m = run_fut(mode)
        eq = m['_equity']; rc = m.get('rebal_count', '-'); tc = m.get('trade_count', '-')
        ms = metrics(eq)
        print(f"  ANCHOR_TRADE={mode:<6}: CAGR {ms[0]:5.1f}%  MDD {ms[1]:+6.1f}%  Sharpe {ms[2]:.2f}  Cal {ms[3]:.2f}  rebal={rc}  trades={tc}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
