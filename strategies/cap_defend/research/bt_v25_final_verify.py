"""V25 최종 검증 — fut ms=18 vs ms=20 + ms=12_ml118 분포 확인 + alloc 비교.

1. fut ms 변경 시 합성 alloc 영향
2. fut ms=12 ml=118 의 기간별 win 분포 (anchor 변동, yearly Cal)
3. spot/fut ms × snap 결합 최종 그리드
"""
from __future__ import annotations
import os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from bt_v25_excl_fine import (run_spot as run_spot_snap, run_fut as run_fut_snap,
                              run_stock, build_alloc, metrics, window_rs)

START = "2020-10-01"
END = "2026-05-29"
EXCLUDE = ['BNB', 'SOL']


def run_spot_ms_ml(ms, ml, sn=217, n=7):
    from unified_backtest import run as bt_run, load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    bars, _ = load_data('D')
    m = bt_run(bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=ms, mom_long_bars=ml,
        vol_threshold=0.05, vol_mode='daily',
        n_snapshots=n, snap_interval_bars=sn,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol',
        exclude_assets=frozenset(EXCLUDE))
    return m.get('_equity') if m else None


def run_fut_ms_ml(ms, ml, sn=95, n=5):
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    bars_full, funding = load_data('D')
    k2 = build_K2_signal(bars_full, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    bars = {c: df for c, df in bars_full.items() if c not in EXCLUDE}
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
        tx_cost=0.0006, maint_rate=0.004,
        sma_days=42, mom_short_days=ms, mom_long_days=ml, vol_days=90,
        canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=n, snap_interval_bars=sn,
        start_date=START, end_date=END)
    return m.get('_equity') if m else None


def yearly_metrics(eq, tag):
    eq = eq.dropna()
    yearly = eq.resample('A').last()
    pct = yearly.pct_change() * 100
    return [(y.strftime('%Y'), pct.get(y)) for y in yearly.index]


def main():
    t0 = time.time()
    eq_st = run_stock()
    print("[stock V25] CAGR", metrics(eq_st)[0])

    # 1. fut ms=18 vs 20 alloc
    print("\n========== FUT ms=18 vs 20 alloc ==========")
    eq_sp = run_spot_ms_ml(20, 127)
    fut_cfgs = [
        ('ms18_ml127 (현행)', (18, 127)),
        ('ms20_ml127',        (20, 127)),
        ('ms12_ml118',        (12, 118)),
    ]
    for tag, (ms, ml) in fut_cfgs:
        eq_fu = run_fut_ms_ml(ms, ml)
        alloc = build_alloc(eq_st, eq_sp, eq_fu)
        m_fu = metrics(eq_fu); m_al = metrics(alloc)
        print(f"  fut {tag}: CAGR {m_fu[0]:5.1f}% MDD {m_fu[1]:+6.1f}% Cal {m_fu[3]:.2f}")
        print(f"    alloc 60/25/15: CAGR {m_al[0]:5.1f}% MDD {m_al[1]:+6.1f}% Sharpe {m_al[2]:.2f} Cal {m_al[3]:.2f}")
        print(f"    yearly: ", end='')
        yr = yearly_metrics(eq_fu, tag)
        for y, p in yr:
            if not pd.isna(p):
                print(f" {y}:{p:+.0f}%", end='')
        print()

    # 2. ms × snap 결합 그리드 (winner 주변)
    print("\n========== ms × snap 결합 그리드 ==========")
    # spot: ms=20 (고정), ml=127, sn ∈ {161, 217, 287, 319}, n ∈ {7, 11}
    print("\n[spot — ms=20 ml=127 fixed]")
    spot_combos = [(217,7),(287,7),(319,11),(391,17),(481,13),(583,11),(781,11)]
    eq_sp_all = {}
    for sn, n in spot_combos:
        tag = f"sn{sn}_n{n}"
        eq = run_spot_ms_ml(20, 127, sn, n)
        if eq is None: continue
        eq_sp_all[tag] = eq
        m = metrics(eq)
        print(f"  {tag:<12} CAGR {m[0]:5.1f}% MDD {m[1]:+6.1f}% Cal {m[3]:.2f}")
    rs = window_rs(eq_sp_all)
    if rs:
        sums, _, n_w = rs
        print(f"\n  spot top by rank (n_w={n_w}):")
        for tag, v in sorted(sums.items(), key=lambda x: x[1]):
            print(f"    {tag:<12} avg_rank={v/n_w:.2f}")

    # fut: ms ∈ {18, 20}, ml=127, sn ∈ {95, 133, 209}, n
    print("\n[fut — ml=127 fixed, ms × snap]")
    fut_combos = []
    for ms in [18, 20]:
        for sn, n in [(95,5),(133,7),(209,11),(247,13),(371,7)]:
            fut_combos.append((ms, sn, n))
    eq_fu_all = {}
    for ms, sn, n in fut_combos:
        tag = f"ms{ms}_sn{sn}_n{n}"
        eq = run_fut_ms_ml(ms, 127, sn, n)
        if eq is None: continue
        eq_fu_all[tag] = eq
        m = metrics(eq)
        print(f"  {tag:<16} CAGR {m[0]:5.1f}% MDD {m[1]:+6.1f}% Cal {m[3]:.2f}")
    rs = window_rs(eq_fu_all)
    if rs:
        sums, _, n_w = rs
        print(f"\n  fut top by rank (n_w={n_w}):")
        for tag, v in sorted(sums.items(), key=lambda x: x[1]):
            print(f"    {tag:<16} avg_rank={v/n_w:.2f}")

    # 3. 최종 후보 alloc 비교
    print("\n========== 최종 후보 alloc 비교 ==========")
    # baseline + 가능한 winner 조합들
    finalists = [
        ('Z baseline', (20, 127, 217, 7), (18, 127, 95, 5)),
        ('A fut ms→20', (20, 127, 217, 7), (20, 127, 95, 5)),
        ('B fut snap', (20, 127, 217, 7), (18, 127, 133, 7)),
        ('C combined', (20, 127, 217, 7), (20, 127, 133, 7)),
        ('D risky',    (20, 127, 217, 7), (12, 118, 95, 5)),
    ]
    for tag, (s_ms, s_ml, s_sn, s_n), (f_ms, f_ml, f_sn, f_n) in finalists:
        eq_sp = run_spot_ms_ml(s_ms, s_ml, s_sn, s_n)
        eq_fu = run_fut_ms_ml(f_ms, f_ml, f_sn, f_n)
        alloc = build_alloc(eq_st, eq_sp, eq_fu)
        m_al = metrics(alloc)
        m_sp = metrics(eq_sp); m_fu = metrics(eq_fu)
        print(f"\n{tag}: spot(ms{s_ms},ml{s_ml},sn{s_sn},n{s_n}) fut(ms{f_ms},ml{f_ml},sn{f_sn},n{f_n})")
        print(f"  spot Cal {m_sp[3]:.2f} | fut Cal {m_fu[3]:.2f} | alloc CAGR {m_al[0]:.1f}% MDD {m_al[1]:+.1f}% Sharpe {m_al[2]:.2f} Cal {m_al[3]:.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
