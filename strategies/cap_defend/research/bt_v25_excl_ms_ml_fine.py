"""V25 BNB/SOL 제외 — ms × ml fine grid (baseline snap 고정).

spot: ms ∈ {14..30}, ml ∈ {110..150}
fut:  ms ∈ {12..26}, ml ∈ {110..150}
vol=0.05 고정. baseline snap (sn=217 n=7 spot, sn=95 n=5 fut) 유지.
"""
from __future__ import annotations
import os, sys, time
from collections import defaultdict
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

START = "2020-10-01"
END = "2026-05-29"
EXCLUDE = ['BNB', 'SOL']

WIN_SIZES = [504, 756, 1008]
STRIDES = [63, 126, 252]


def run_spot(ms, ml):
    from unified_backtest import run as bt_run, load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    bars, _ = load_data('D')
    m = bt_run(bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=ms, mom_long_bars=ml,
        vol_threshold=0.05, vol_mode='daily',
        n_snapshots=7, snap_interval_bars=217,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol',
        exclude_assets=frozenset(EXCLUDE))
    return m.get('_equity') if m else None


def run_fut(ms, ml):
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
        n_snapshots=5, snap_interval_bars=95,
        start_date=START, end_date=END)
    return m.get('_equity') if m else None


def metrics(eq):
    if eq is None or len(eq.dropna()) < 30: return None
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1]/eq.iloc[0]) ** (1/yrs) - 1
    peak = eq.cummax(); mdd = (eq/peak - 1).min()
    return cagr*100, mdd*100, cagr/abs(mdd) if mdd < 0 else 0


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
    SPOT_MS = [14, 16, 18, 20, 22, 24, 26, 28, 30]
    FUT_MS = [12, 14, 16, 18, 20, 22, 24, 26]
    MLS = [110, 118, 127, 135, 143, 150]

    for asset, runner, MS, base_ms, base_ml in [
        ('SPOT', run_spot, SPOT_MS, 20, 127),
        ('FUT',  run_fut,  FUT_MS,  18, 127),
    ]:
        print(f"\n========== {asset} ==========")
        eq_dict = {}
        results = []
        for ms in MS:
            for ml in MLS:
                if ml <= ms * 2: continue
                tag = f"ms{ms:02d}_ml{ml}"
                eq = runner(ms, ml)
                m = metrics(eq)
                if m is None: continue
                eq_dict[tag] = eq
                results.append((tag, ms, ml, m))
        print(f"  total cfgs: {len(eq_dict)}")
        rs = window_rs(eq_dict)
        if rs is None: continue
        sums, wins, n_w = rs
        print(f"\n  Top 15 (n_w={n_w}):")
        print(f"  {'cfg':<14} {'rank':>5} {'win%':>6} {'CAGR':>6} {'MDD':>7} {'Cal':>5}")
        for tag, v in sorted(sums.items(), key=lambda x: x[1])[:15]:
            m = next((m for t, _, _, m in results if t == tag), None)
            if m:
                print(f"  {tag:<14} {v/n_w:>5.2f} {wins[tag]/n_w*100:>5.1f}% "
                      f"{m[0]:>5.0f}% {m[1]:>+6.0f}% {m[2]:>5.2f}")
        base_tag = f"ms{base_ms:02d}_ml{base_ml}"
        for i, (tag, v) in enumerate(sorted(sums.items(), key=lambda x: x[1]), 1):
            if tag == base_tag:
                print(f"\n  baseline {base_tag}: rank {i}/{len(eq_dict)} avg_rank={v/n_w:.2f}")
                break

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
