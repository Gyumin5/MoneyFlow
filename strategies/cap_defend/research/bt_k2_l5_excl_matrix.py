"""L5 상방 티어 × 제외조합 매트릭스 — robustness 검증.

L5 개선(alloc Cal 2.98→3.14)이 BNB+SOL 제외 시나리오 특화인지,
아니면 제외조합 무관하게 일관된지 확인.

variants: baseline(ceiling L4) vs L5(pc1.10/btc1.08)
exclusions: none(full) / BNB only / SOL only / BNB+SOL

각 셀: fut Cal + alloc(60/25/15) CAGR/MDD/Sharpe/Cal
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from unified_backtest import load_data
from bt_k2_upside_l5 import build_K2_L5
from bt_k2_l1_downside import run_stock, build_alloc, metrics

START = "2020-10-01"; END = "2026-05-29"


def run_fut_excl(k2, exclude):
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'; os.environ['ANCHOR_TRADE_MODE'] = 'on'
    from backtest_futures_v25 import run as fbt_run
    bars_full, funding = load_data('D')
    bars = {c: df for c, df in bars_full.items() if c not in exclude}
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
        tx_cost=0.0006, maint_rate=0.004,
        sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
        canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=5, snap_interval_bars=95, start_date=START, end_date=END)
    return m.get('_equity') if m else None


def run_spot_excl(exclude):
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    from unified_backtest import run as bt_run
    bars, _ = load_data('D')
    m = bt_run(bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=20, mom_long_bars=127,
        vol_threshold=0.05, vol_mode='daily', n_snapshots=7, snap_interval_bars=217,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy',
        stop_kind='none', stop_pct=0.0, dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0, health_mode='mom2vol',
        exclude_assets=frozenset(exclude) if exclude else None)
    return m.get('_equity') if m else None


def main():
    t0 = time.time()
    bars_full, _ = load_data('D')
    eq_st = run_stock()

    exclusions = [
        ('none(full)', []),
        ('BNB only',   ['BNB']),
        ('SOL only',   ['SOL']),
        ('BNB+SOL',    ['BNB', 'SOL']),
    ]
    variants = [
        ('baseline L4', None, None),
        ('L5 1.10/1.08', 1.100, 1.08),
    ]

    print(f"{'variant':<14} {'exclude':<12} {'fut_Cal':>8} | {'al_CAGR':>8} {'al_MDD':>8} {'al_Sh':>6} {'al_Cal':>7}")
    rows = {}
    for vtag, pc5, btc5 in variants:
        for etag, excl in exclusions:
            k2 = build_K2_L5(bars_full, pc_thr_l5=pc5, btc_thr_l5=btc5)
            eq_fu = run_fut_excl(k2, set(excl))
            eq_sp = run_spot_excl(excl)
            if eq_fu is None or eq_sp is None:
                print(f"{vtag:<14} {etag:<12} FAILED"); continue
            m_fu = metrics(eq_fu)
            alloc = build_alloc(eq_st, eq_sp, eq_fu)
            m_al = metrics(alloc)
            rows[(vtag, etag)] = m_al[3]
            print(f"{vtag:<14} {etag:<12} {m_fu[3]:>8.2f} | {m_al[0]:>7.1f}% {m_al[1]:>+7.1f}% {m_al[2]:>6.2f} {m_al[3]:>7.2f}")
        print()

    print("[L5 - baseline alloc Cal delta per exclusion]")
    for _, etag, *_ in [(0, e, 0) for e in [x[0] for x in exclusions]]:
        b = rows.get(('baseline L4', etag)); l = rows.get(('L5 1.10/1.08', etag))
        if b is not None and l is not None:
            print(f"  {etag:<12} baseline {b:.2f} → L5 {l:.2f}  delta {l-b:+.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
