"""V25 BNB/SOL 제외 후 코인 최적 그리드 1차 — ms × ml × vol.

spot + fut 각각. 동일 spec (snap/drift/cap) 유지. 2-mom mom2vol.
window rank-sum 으로 plateau 후보 식별.
"""
from __future__ import annotations
import os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

START = "2020-10-01"
END = "2026-05-29"
EXCLUDE = ['BNB', 'SOL']

WIN_SIZES = [504, 756, 1008]
STRIDES = [63, 126, 252]


def run_spot(ms, ml, vol):
    from unified_backtest import run as bt_run, load_data
    bars, _ = load_data('D')
    m = bt_run(bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=ms, mom_long_bars=ml,
        vol_threshold=vol, vol_mode='daily',
        n_snapshots=7, snap_interval_bars=217,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol',
        exclude_assets=frozenset(EXCLUDE))
    return m.get('_equity') if m else None


def run_fut(ms, ml, vol):
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    bars_full, funding = load_data('D')
    k2 = build_K2_signal(bars_full, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    bars = {c: df for c, df in bars_full.items() if c not in EXCLUDE}
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
        tx_cost=0.0006, maint_rate=0.004,
        sma_days=42, mom_short_days=ms, mom_long_days=ml, vol_days=90,
        canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=vol,
        n_snapshots=5, snap_interval_bars=95,
        start_date=START, end_date=END)
    return m.get('_equity') if m else None


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


def window_rs(eq_dict):
    common = None
    for s in eq_dict.values():
        if s is None: continue
        if common is None: common = s.index
        else: common = common.intersection(s.index)
    if common is None: return None
    common = sorted(common)
    if len(common) < max(WIN_SIZES) + max(STRIDES): return None
    sums = defaultdict(float); wins = defaultdict(int); n = 0
    for size in WIN_SIZES:
        for stride in STRIDES:
            for i in range(0, len(common) - size, stride):
                d0 = common[i]; d1 = common[i + size - 1]
                cals = {}
                for k, s in eq_dict.items():
                    if s is None: cals[k] = np.nan; continue
                    seg = s.loc[d0:d1].dropna()
                    if len(seg) < 30: cals[k] = np.nan; continue
                    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                    if yrs <= 0: cals[k] = np.nan; continue
                    cagr = (seg.iloc[-1]/seg.iloc[0]) ** (1/yrs) - 1
                    peak = seg.cummax(); mdd = float((seg/peak - 1).min())
                    cals[k] = cagr/abs(mdd) if mdd < 0 else 0
                if any(np.isnan(v) for v in cals.values()): continue
                ranked = sorted(cals.items(), key=lambda x: -x[1])
                for r, (mk, _) in enumerate(ranked, 1): sums[mk] += r
                wins[ranked[0][0]] += 1; n += 1
    return sums, wins, n


def main():
    t0 = time.time()
    SPOT_MS = [10, 14, 20, 30, 42, 60]
    SPOT_ML = [84, 100, 127, 150, 200, 250]
    FUT_MS = [10, 14, 18, 25, 35]
    FUT_ML = [84, 100, 127, 150, 200]
    VOLS = [0.04, 0.05, 0.07]

    for asset, runner, MS, ML, baseline_ms, baseline_ml in [
        ('SPOT', run_spot, SPOT_MS, SPOT_ML, 20, 127),
        ('FUT',  run_fut,  FUT_MS,  FUT_ML,  18, 127),
    ]:
        print(f"\n========== {asset} ==========")
        eq_dict = {}
        results = []
        for ms in MS:
            for ml in ML:
                if ml <= ms * 2: continue  # 의미 있는 ml/ms 비율 (mom_short < mom_long)
                for vol in VOLS:
                    tag = f"ms{ms}_ml{ml}_v{int(vol*100)}"
                    eq = runner(ms, ml, vol)
                    m = metrics(eq)
                    if m is None: continue
                    eq_dict[tag] = eq
                    results.append((tag, ms, ml, vol, m))
        print(f"  total cfgs: {len(eq_dict)}")
        # 기본값 확인
        base_tag = f"ms{baseline_ms}_ml{baseline_ml}_v5"
        if base_tag in eq_dict:
            for tag, ms, ml, vol, m in results:
                if tag == base_tag:
                    print(f"  baseline {base_tag}: CAGR {m[0]:.1f}% MDD {m[1]:+.1f}% Cal {m[3]:.2f}")
                    break
        rs = window_rs(eq_dict)
        if rs is None: continue
        sums, wins, n = rs
        ranked = sorted(sums.items(), key=lambda x: x[1])
        print(f"\n  Top 10 (by avg_rank, n_windows={n}):")
        print(f"  {'cfg':<22} {'avg_rank':>9} {'win%':>6} {'CAGR':>7} {'MDD':>7} {'Cal':>6}")
        for tag, v in ranked[:10]:
            m = next((m for t, _, _, _, m in results if t == tag), None)
            if m:
                print(f"  {tag:<22} {v/n:>9.3f} {wins[tag]/n*100:>5.1f}% {m[0]:>6.0f}% {m[1]:>+6.0f}% {m[3]:>6.2f}")
        # baseline rank
        for i, (tag, v) in enumerate(ranked, 1):
            if tag == base_tag:
                print(f"\n  baseline 위치: rank {i}/{len(ranked)} (avg_rank={v/n:.3f})")
                break

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
