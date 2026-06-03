"""V25 추가 검증 (Round 2) — bootstrap outlier + cost stress + event count.

1. Bootstrap outlier — 랜덤 1/2/3 종목 제거 N회. baseline vs winner alloc Cal 분포
2. Cost stress — tx 1x/2x/3x/5x. CAGR/MDD/Cal 변화
3. T1/T3U event count — winner 후보 T1/T3U setting 별 송금 이벤트 분포
4. Sub-period window Cal — window rank-sum 의 cfg 별 Cal 분포 분석
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from bt_v25_excl_matrix import run_spot as _run_spot, run_fut as _run_fut, run_stock, build_alloc, metrics

START = "2020-10-01"; END = "2026-05-29"


def run_spot_tx(ms, ml, sn, n, exclude, tx):
    from unified_backtest import run as bt_run, load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    bars, _ = load_data('D')
    m = bt_run(bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=tx,
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
        exclude_assets=frozenset(exclude) if exclude else None)
    return m.get('_equity') if m else None


def run_fut_tx(ms, ml, sn, n, exclude, tx):
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    bars_full, funding = load_data('D')
    k2 = build_K2_signal(bars_full, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    bars = {c: df for c, df in bars_full.items() if c not in exclude} if exclude else bars_full
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
        tx_cost=tx, maint_rate=0.004,
        sma_days=42, mom_short_days=ms, mom_long_days=ml, vol_days=90,
        canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=n, snap_interval_bars=sn,
        start_date=START, end_date=END)
    return m.get('_equity') if m else None


def part1_bootstrap(eq_st):
    """랜덤 N 종목 제거 (BTC 항상 유지). baseline vs winner alloc Cal."""
    print("\n=== PART 1: Bootstrap outlier stress ===")
    from unified_backtest import load_data
    bars, _ = load_data('D')
    candidates = sorted([c for c in bars.keys() if c not in ('BTC', 'CASH')])
    print(f"  bootstrap candidate pool: {len(candidates)} coins")

    # 고정 seed (재현성)
    rng = np.random.RandomState(42)
    samples = []
    for n_drop in [1, 2, 3]:
        for _ in range(10):  # 10 trials per n_drop
            picks = list(rng.choice(candidates, size=n_drop, replace=False))
            samples.append((n_drop, picks))

    print(f"  total samples: {len(samples)}")
    print(f"\n  {'n_drop':<6} {'dropped':<30} {'base_Cal':>9} {'win_Cal':>8} {'delta':>7}")
    for n_drop, picks in samples:
        excl = picks
        eq_sp = _run_spot(20, 127, 217, 7, exclude=excl)
        eq_fu_base = _run_fut(18, 127, 95, 5, exclude=excl)
        eq_fu_win = _run_fut(12, 118, 95, 5, exclude=excl)
        if eq_sp is None or eq_fu_base is None or eq_fu_win is None: continue
        a_base = build_alloc(eq_st, eq_sp, eq_fu_base)
        a_win = build_alloc(eq_st, eq_sp, eq_fu_win)
        m_b = metrics(a_base); m_w = metrics(a_win)
        if m_b is None or m_w is None: continue
        diff = m_w[3] - m_b[3]
        print(f"  {n_drop:<6} {','.join(picks):<30} {m_b[3]:>8.2f} {m_w[3]:>7.2f} {diff:>+6.2f}")


def part2_cost_stress(eq_st):
    """tx 1x/2x/3x/5x stress."""
    print("\n=== PART 2: Cost stress ===")
    base_tx_sp = 0.004; base_tx_fu = 0.0006
    EXCL = ['BNB', 'SOL']
    print(f"  {'tx_mult':<8} {'cfg':<12} {'fut_Cal':>8} {'alloc_Cal':>10} {'alloc_CAGR':>10}")
    for mult in [1, 2, 3, 5]:
        for tag, fut_cfg in [('base ms18', (18, 127, 95, 5)), ('win ms12', (12, 118, 95, 5))]:
            eq_sp = run_spot_tx(20, 127, 217, 7, EXCL, base_tx_sp * mult)
            eq_fu = run_fut_tx(*fut_cfg, EXCL, base_tx_fu * mult)
            if eq_sp is None or eq_fu is None: continue
            alloc = build_alloc(eq_st, eq_sp, eq_fu)
            m_fu = metrics(eq_fu); m_al = metrics(alloc)
            print(f"  {mult}x       {tag:<12} {m_fu[3]:>8.2f} {m_al[3]:>9.2f} {m_al[0]:>9.1f}%")


def part3_event_count(eq_st):
    """T1/T3U event count — 후보 spec 별 송금 이벤트 분포."""
    print("\n=== PART 3: T1/T3U event count (BNB,SOL 제외, baseline sleeve) ===")
    EXCL = ['BNB', 'SOL']
    eq_sp = _run_spot(20, 127, 217, 7, EXCL)
    eq_fu = _run_fut(18, 127, 95, 5, EXCL)

    from bt_v25_t1_t3u import simulate_alloc
    print(f"  {'T1':<5} {'T3U':<5} {'n_T1/yr':>9} {'n_T3U/yr':>9} {'CAGR':>7} {'Cal':>5}")
    for T1, T3U in [(0.20, 0.20), (0.17, 0.15), (0.15, 0.15), (0.25, 0.25)]:
        res = simulate_alloc(eq_st, eq_sp, eq_fu, T1=T1, T3U=T3U)
        if res is None: continue
        eq, n_t1, n_t3u = res
        m = metrics(eq)
        n_years = (eq.index[-1] - eq.index[0]).days / 365.25
        print(f"  {T1*100:>4.0f}% {T3U*100:>4.0f}% {n_t1/n_years:>8.1f} {n_t3u/n_years:>8.1f} {m[0]:>6.1f}% {m[3]:>4.2f}")


def part4_window_subperiods(eq_st):
    """Window-binned Cal — winner vs baseline subperiod 일관성."""
    print("\n=== PART 4: Window sub-period Cal (winner vs baseline, BNB+SOL 제외) ===")
    EXCL = ['BNB', 'SOL']
    eq_sp = _run_spot(20, 127, 217, 7, EXCL)
    eq_fu_b = _run_fut(18, 127, 95, 5, EXCL)
    eq_fu_w = _run_fut(12, 118, 95, 5, EXCL)
    a_b = build_alloc(eq_st, eq_sp, eq_fu_b)
    a_w = build_alloc(eq_st, eq_sp, eq_fu_w)
    common = a_b.index.intersection(a_w.index)
    common = sorted(common)
    WIN = [504, 756, 1008]; STR = [63, 126, 252]
    diffs = []
    for size in WIN:
        for stride in STR:
            for i in range(0, len(common) - size, stride):
                d0 = common[i]; d1 = common[i + size - 1]
                def cal(s):
                    seg = s.loc[d0:d1].dropna()
                    if len(seg) < 30: return None
                    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                    if yrs <= 0: return None
                    cagr = (seg.iloc[-1]/seg.iloc[0]) ** (1/yrs) - 1
                    peak = seg.cummax(); mdd = float((seg/peak - 1).min())
                    return cagr/abs(mdd) if mdd < 0 else 0
                cb = cal(a_b); cw = cal(a_w)
                if cb is None or cw is None: continue
                diffs.append(cw - cb)
    print(f"  windows: {len(diffs)}")
    print(f"  winner > base: {sum(1 for d in diffs if d > 0)}/{len(diffs)} ({sum(1 for d in diffs if d > 0)/len(diffs)*100:.1f}%)")
    arr = np.array(diffs)
    print(f"  delta Cal — mean {arr.mean():+.3f}, median {np.median(arr):+.3f}, std {arr.std():.3f}")
    print(f"  p5 {np.percentile(arr, 5):+.3f}, p25 {np.percentile(arr, 25):+.3f}, p75 {np.percentile(arr, 75):+.3f}, p95 {np.percentile(arr, 95):+.3f}")


def main():
    t0 = time.time()
    eq_st = run_stock()
    print(f"\n[stock V25] CAGR {metrics(eq_st)[0]:.1f}%")

    part1_bootstrap(eq_st)
    part2_cost_stress(eq_st)
    part3_event_count(eq_st)
    part4_window_subperiods(eq_st)

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
