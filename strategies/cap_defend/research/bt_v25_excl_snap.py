"""V25 BNB/SOL 제외 — snap × n_snap × refill 2단계 그리드.

baseline 고정: spot ms20/ml127/vol5, fut ms18/ml127/vol5.
탐색: snap_int × n_snap → prime stagger 만 채택.
refill ON / OFF 비교.
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


def is_prime(n):
    if n < 2: return False
    if n in (2, 3): return True
    if n % 2 == 0: return False
    for d in range(3, int(n**0.5) + 1, 2):
        if n % d == 0: return False
    return True


def run_spot(snap_int, n_snap, refill, drift_mode):
    from unified_backtest import run as bt_run, load_data
    os.environ['DRIFT_HEALTH_MODE'] = drift_mode
    bars, _ = load_data('D')
    m = bt_run(bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=20, mom_long_bars=127,
        vol_threshold=0.05, vol_mode='daily',
        n_snapshots=n_snap, snap_interval_bars=snap_int,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol',
        exclude_assets=frozenset(EXCLUDE))
    return m.get('_equity') if m else None


def run_fut(snap_int, n_snap, refill, drift_mode):
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    os.environ['DRIFT_HEALTH_MODE'] = drift_mode
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
        n_snapshots=n_snap, snap_interval_bars=snap_int,
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


def gen_pairs(snap_candidates, n_snap_candidates):
    """반환: [(snap_int, n_snap, stagger)] — stagger prime 만."""
    pairs = []
    for sn in snap_candidates:
        for n in n_snap_candidates:
            if sn % n != 0: continue
            stagger = sn // n
            if not is_prime(stagger): continue
            pairs.append((sn, n, stagger))
    return pairs


def main():
    t0 = time.time()
    # spot: 현행 stagger=31. 다른 prime stagger 후보
    # snap 후보 와 n_snap 후보 의 곱 = stagger prime
    spot_snaps = [69, 91, 95, 119, 126, 161, 217, 287, 319, 391]
    spot_ns = [3, 5, 7, 11, 13]
    # fut: 현행 stagger=19. snap=95 n=5
    fut_snaps = [39, 57, 65, 95, 115, 133, 161, 209, 247]
    fut_ns = [3, 5, 7, 11]

    for asset, runner, snaps, ns, baseline_sn, baseline_n in [
        ('SPOT', run_spot, spot_snaps, spot_ns, 217, 7),
        ('FUT',  run_fut,  fut_snaps,  fut_ns,  95,  5),
    ]:
        print(f"\n========== {asset} ==========")
        pairs = gen_pairs(snaps, ns)
        # baseline 포함 보장
        if (baseline_sn, baseline_n, baseline_sn // baseline_n) not in pairs:
            pairs.append((baseline_sn, baseline_n, baseline_sn // baseline_n))
        print(f"  pairs (prime stagger): {len(pairs)}")
        for sn, n, st in pairs:
            print(f"    sn={sn} n={n} stagger={st}")

        # refill ON / OFF 2회씩
        eq_dict = {}
        results = []
        for sn, n, st in pairs:
            for refill_mode, label in [('refill', 'rfON'), ('off', 'rfOFF')]:
                tag = f"sn{sn}_n{n}_st{st}_{label}"
                eq = runner(sn, n, refill_mode == 'refill', refill_mode)
                m = metrics(eq)
                if m is None: continue
                eq_dict[tag] = eq
                results.append((tag, sn, n, st, refill_mode, m))
        rs = window_rs(eq_dict)
        if rs is None: continue
        sums, wins, n_w = rs
        print(f"\n  Top 15 (avg_rank, n_windows={n_w}):")
        print(f"  {'cfg':<28} {'rank':>6} {'win%':>6} {'CAGR':>7} {'MDD':>7} {'Cal':>6}")
        ranked = sorted(sums.items(), key=lambda x: x[1])
        for tag, v in ranked[:15]:
            m = next((m for t, _, _, _, _, m in results if t == tag), None)
            if m:
                print(f"  {tag:<28} {v/n_w:>6.2f} {wins[tag]/n_w*100:>5.1f}% "
                      f"{m[0]:>6.0f}% {m[1]:>+6.0f}% {m[2]:>6.2f}")
        # baseline 위치 찾기
        base_tag_on = f"sn{baseline_sn}_n{baseline_n}_st{baseline_sn//baseline_n}_rfON"
        base_tag_off = f"sn{baseline_sn}_n{baseline_n}_st{baseline_sn//baseline_n}_rfOFF"
        for bt in [base_tag_on, base_tag_off]:
            for i, (tag, v) in enumerate(ranked, 1):
                if tag == bt:
                    print(f"  baseline {bt}: rank {i}/{len(ranked)} avg_rank={v/n_w:.2f}")
                    break

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
