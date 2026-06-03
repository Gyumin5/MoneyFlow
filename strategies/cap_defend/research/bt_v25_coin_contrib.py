"""V25 전구간 코인별 기여도 — spot/fut 각각.

산출:
- 코인별 보유 일수, 평균 weight
- 코인별 누적 기여 수익 (= sum daily weight × daily price return)
- 보유 일수 동안 코인 자체 수익률
- 상위 기여 / 손실 코인

기간: 2020-10-01 ~ 2026-05-29
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

START = "2020-10-01"
END = "2026-05-29"


def run_spot_trace():
    from unified_backtest import run as bt_run, load_data
    os.environ['DRIFT_CASH_REFILL'] = 'off'
    bars, _ = load_data('D')
    trace = []
    m = bt_run(
        bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=20, mom_long_bars=127,
        vol_threshold=0.05, vol_mode='daily',
        n_snapshots=7, snap_interval_bars=217,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol', _trace=trace,
    )
    return trace, bars


def run_fut_trace():
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    os.environ['DRIFT_CASH_REFILL'] = 'off'
    bars, funding = load_data('D')
    k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    trace = []
    m = fbt_run(
        bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
        tx_cost=0.0006, maint_rate=0.004,
        sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
        canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=5, snap_interval_bars=95,
        start_date=START, end_date=END, _trace=trace,
    )
    return trace, bars, k2


def analyze(trace, bars, asset, k2=None):
    """trace -> 코인별 기여."""
    # target per date
    tgt = {pd.Timestamp(t['date']).normalize(): t['target'] for t in trace}
    dates = sorted(tgt.keys())
    if not dates:
        return []
    # Build daily price return Series per coin
    coin_set = set()
    for d in dates:
        for c, w in tgt[d].items():
            if c not in ('CASH', 'Cash') and w > 0.001:
                coin_set.add(c)
    coin_ret = {}
    coin_close = {}
    for c in coin_set:
        df = bars.get(c)
        if df is None: continue
        s = pd.Series(df['Close'].values, index=df.index).sort_index()
        coin_close[c] = s
        coin_ret[c] = s.pct_change()

    # For each day, compute coin's contribution = weight(t-1) × ret(t) [optionally × L(t-1) for fut]
    rows = []
    sum_contrib = {c: 0.0 for c in coin_set}
    sum_w_days = {c: 0.0 for c in coin_set}  # weight × day count
    days_held = {c: 0 for c in coin_set}
    for i, d in enumerate(dates):
        if i == 0: continue
        d_prev = dates[i-1]
        w_prev = tgt[d_prev]
        for c, w in w_prev.items():
            if c in ('CASH', 'Cash') or w <= 0.001: continue
            r = coin_ret.get(c)
            if r is None: continue
            if d not in r.index: continue
            ret_d = r.loc[d]
            if pd.isna(ret_d): continue
            lev = 1.0
            if asset == 'fut' and k2 is not None and c in k2:
                try: lev = float(k2[c].asof(d_prev))
                except: lev = 1.0
            contrib = w * ret_d * lev
            sum_contrib[c] += contrib
            sum_w_days[c] += w
            days_held[c] += 1

    # Per-coin own return over held period (simple: from first held day to last held day close)
    coin_own_ret = {}
    for c in coin_set:
        first_held = None; last_held = None
        for d in dates:
            w = tgt[d].get(c, 0)
            if w > 0.001:
                if first_held is None: first_held = d
                last_held = d
        if first_held and last_held and c in coin_close:
            s = coin_close[c]
            try:
                p0 = s.asof(first_held); p1 = s.asof(last_held)
                if p0 > 0:
                    coin_own_ret[c] = (p1/p0 - 1) * 100
            except: pass

    results = []
    for c in coin_set:
        results.append({
            'coin': c,
            'days_held': days_held[c],
            'avg_weight': sum_w_days[c] / max(1, days_held[c]),
            'contrib_pct': sum_contrib[c] * 100,  # 누적 기여 (단순 합산, 복리 미반영)
            'own_ret_held': coin_own_ret.get(c, np.nan),
        })
    return sorted(results, key=lambda r: -r['contrib_pct'])


def print_table(results, title, top_n=15):
    print(f"\n=== {title} ===")
    print(f"  {'coin':<7} {'days':>5} {'avg_w':>7} {'contrib%':>10} {'own_ret%':>10}")
    for r in results[:top_n]:
        own = f"{r['own_ret_held']:+.1f}" if not pd.isna(r['own_ret_held']) else '-'
        print(f"  {r['coin']:<7} {r['days_held']:>5} {r['avg_weight']*100:>6.1f}% "
              f"{r['contrib_pct']:>+9.1f}% {own:>9}%")
    if len(results) > top_n:
        print(f"  ... ({len(results) - top_n} more)")
        print(f"\n  하위 5개:")
        for r in results[-5:]:
            own = f"{r['own_ret_held']:+.1f}" if not pd.isna(r['own_ret_held']) else '-'
            print(f"  {r['coin']:<7} {r['days_held']:>5} {r['avg_weight']*100:>6.1f}% "
                  f"{r['contrib_pct']:>+9.1f}% {own:>9}%")


def main():
    print("[spot trace]")
    tr_sp, bars_sp = run_spot_trace()
    res_sp = analyze(tr_sp, bars_sp, 'spot')
    print_table(res_sp, "spot 전구간 코인별 기여 (정렬: 누적 기여%)")

    print("\n[fut trace]")
    tr_fu, bars_fu, k2 = run_fut_trace()
    res_fu = analyze(tr_fu, bars_fu, 'fut', k2)
    print_table(res_fu, "fut 전구간 코인별 기여 (L 반영, 정렬: 누적 기여%)")


if __name__ == "__main__":
    main()
