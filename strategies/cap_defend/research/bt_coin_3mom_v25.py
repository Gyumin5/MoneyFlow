"""Coin spot/fut 3-mom 탐색 BT — V25 health rule 후보.

두 접근:
- A (mid 탐색): 현 ms+ml 의 중간 mid 값 grid → 3mom 적용
- B (ml 확장): 현 ms+mid + 더 큰 ml grid

대상:
- spot: ms=20, ml=127 base (live V24)
- fut:  ms=18, ml=127 base (live V25), L=3 고정 (동적 L 별도)

비교:
- baseline: mom2vol (live)
- 3momvol:  3 mom 모두 양수 AND vol≤5%
- 3mom:     3 mom 모두 양수 (vol cap 없음)

평가:
- 5.4yr 단일 anchor (phase_offset=0) per cfg
- window rank-sum (WIN={504,756,1008} × STRIDE={63,126,252})

CLI:
  python bt_coin_3mom_v25.py spot A   # spot 접근A
  python bt_coin_3mom_v25.py spot B
  python bt_coin_3mom_v25.py fut A
  python bt_coin_3mom_v25.py fut B
"""
from __future__ import annotations
import os, sys, time
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)

from unified_backtest import run as bt_run, load_data

WIN_SIZES = [504, 756, 1008]
STRIDES = [63, 126, 252]

START = "2020-10-01"
END = "2026-04-13"


def live_params(asset: str) -> dict:
    """V24 spot / V25 fut live spec."""
    if asset == 'spot':
        return dict(
            sma_bars=42, vol_threshold=0.05, vol_mode='daily',
            n_snapshots=7, snap_interval_bars=217,
            canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
            universe_size=3, cap=1/3, selection='greedy',
            stop_kind='none', stop_pct=0.0,
            dd_lookback=60, dd_threshold=-99.0,  # 가드 off
            bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
            ms_base=20, ml_base=127,
        )
    else:
        return dict(
            sma_bars=42, vol_threshold=0.05, vol_mode='daily',
            n_snapshots=5, snap_interval_bars=95,
            canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
            universe_size=3, cap=1/3, selection='greedy',
            stop_kind='none', stop_pct=0.0,
            dd_lookback=60, dd_threshold=-99.0,
            bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
            ms_base=18, ml_base=127,
        )


def grids(approach: str) -> list[tuple[int, int]]:
    """Return list of (mom_mid_bars, mom_long_bars) for 3-mom."""
    if approach == 'A':
        mids = [40, 50, 60, 70, 80, 90, 100]
        ml_base = 127
        return [(m, ml_base) for m in mids]
    elif approach == 'B':
        mid_base = 127
        mls = [160, 200, 250, 300, 360]
        return [(mid_base, m) for m in mls]
    else:
        raise ValueError(approach)


def run_one(bars, funding, asset, lev, tx, lp, ms, mid, ml, health_mode):
    common = dict(
        sma_bars=lp['sma_bars'],
        mom_short_bars=ms, mom_long_bars=ml,
        mom_mid_bars=mid,
        vol_threshold=lp['vol_threshold'], vol_mode=lp['vol_mode'],
        n_snapshots=lp['n_snapshots'], snap_interval_bars=lp['snap_interval_bars'],
        canary_hyst=lp['canary_hyst'], drift_threshold=lp['drift_threshold'],
        post_flip_delay=lp['post_flip_delay'],
        universe_size=lp['universe_size'], cap=lp['cap'], selection=lp['selection'],
        stop_kind=lp['stop_kind'], stop_pct=lp['stop_pct'],
        dd_lookback=lp['dd_lookback'], dd_threshold=lp['dd_threshold'],
        bl_drop=lp['bl_drop'], bl_days=lp['bl_days'],
        crash_threshold=lp['crash_threshold'],
        health_mode=health_mode,
    )
    m = bt_run(
        bars, funding, interval='D',
        asset_type=asset, leverage=lev, tx_cost=tx,
        start_date=START, end_date=END,
        **common,
    )
    if not m or '_equity' not in m:
        return None, None
    eq = m['_equity']
    summary = {k: m.get(k) for k in ('Sharpe', 'CAGR', 'MDD', 'Cal', 'Rebal')}
    return eq, summary


def window_rank_sum(eq_dict):
    common = None
    for s in eq_dict.values():
        if common is None:
            common = s.index
        else:
            common = common.intersection(s.index)
    common = sorted(common)
    if len(common) < max(WIN_SIZES) + max(STRIDES):
        return None
    sums = defaultdict(float); wins = defaultdict(int); n = 0
    for size in WIN_SIZES:
        for stride in STRIDES:
            for s_idx in range(0, len(common) - size, stride):
                d0 = common[s_idx]; d1 = common[s_idx + size - 1]
                cals = {}
                for k, s in eq_dict.items():
                    seg = s.loc[d0:d1].dropna()
                    if len(seg) < 30:
                        cals[k] = np.nan; continue
                    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                    if yrs <= 0:
                        cals[k] = np.nan; continue
                    cagr = (seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
                    peak = seg.cummax(); mdd = float((seg / peak - 1).min())
                    cals[k] = cagr / abs(mdd) if mdd < 0 else 0
                if any(np.isnan(v) for v in cals.values()):
                    continue
                ranked = sorted(cals.items(), key=lambda x: -x[1])
                for r, (mk, _) in enumerate(ranked, 1):
                    sums[mk] += r
                wins[ranked[0][0]] += 1
                n += 1
    return sums, wins, n


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    asset = sys.argv[1]  # spot | fut
    approach = sys.argv[2]  # A | B
    if asset not in ('spot', 'fut') or approach not in ('A', 'B'):
        print("usage: spot|fut  A|B"); sys.exit(1)

    t0 = time.time()
    print(f"[load_data D]"); bars, funding = load_data('D')
    lp = live_params(asset)
    tx = 0.004 if asset == 'spot' else 0.0004
    lev = 1.0 if asset == 'spot' else 3.0

    print(f"[BT] asset={asset} approach={approach} tx={tx} lev={lev}")
    print(f"     ms_base={lp['ms_base']} ml_base={lp['ml_base']}")

    cfgs = grids(approach)
    eqs = {}

    # baseline (live mom2vol)
    print(f"\n[baseline] mom2vol ms={lp['ms_base']} ml={lp['ml_base']}")
    eq_base, sm_base = run_one(bars, funding, asset, lev, tx, lp,
                                lp['ms_base'], 0, lp['ml_base'], 'mom2vol')
    if eq_base is None:
        print("ERROR: baseline empty"); sys.exit(1)
    eqs['BASE_mom2vol'] = eq_base
    print(f"  Sh={sm_base['Sharpe']:.2f} CAGR={sm_base['CAGR']:.1%} MDD={sm_base['MDD']:.1%} Cal={sm_base['Cal']:.2f} Rebal={sm_base['Rebal']}")

    for mid, ml in cfgs:
        for hm in ('3momvol', '3mom'):
            tag = f"{hm}_ms{lp['ms_base']}_mid{mid}_ml{ml}"
            print(f"\n[{tag}]")
            eq, sm = run_one(bars, funding, asset, lev, tx, lp,
                              lp['ms_base'], mid, ml, hm)
            if eq is None:
                print("  EMPTY"); continue
            eqs[tag] = eq
            print(f"  Sh={sm['Sharpe']:.2f} CAGR={sm['CAGR']:.1%} MDD={sm['MDD']:.1%} Cal={sm['Cal']:.2f} Rebal={sm['Rebal']}")

    print(f"\n--- window rank-sum ({len(eqs)} cfgs) ---")
    rs = window_rank_sum(eqs)
    if rs is None:
        print("rank-sum 데이터 부족"); sys.exit(1)
    sums, wins, n = rs
    base_avg = sums.get('BASE_mom2vol', 0) / n
    items = sorted(sums.items(), key=lambda x: x[1])
    print(f"n_windows={n}, base avg_rank={base_avg:.3f}")
    print(f"  {'tag':<28} {'avg_rank':>9} {'win%':>6} {'vs_base':>8}")
    for k, v in items:
        marker = ' ← live' if k == 'BASE_mom2vol' else ''
        diff = v / n - base_avg
        print(f"  {k:<28} {v/n:>9.3f} {wins[k]/n*100:>5.1f}% {diff:>+8.3f}{marker}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
