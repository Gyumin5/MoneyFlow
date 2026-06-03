"""Coin spot/fut selection 비교: cap (현행) vs Z-score (다양한 룩백 조합).

Z-score = standardize(Mom_M) + standardize(Sharpe_S). M, S 룩백 조합:
- M: 60, 90, 126, 180, 252, 360, 504
- S: 30, 60, 90, 126, 180
- M, S 독립 그리드 (제약 없음)

같은 health (baseline mom2vol) + 같은 weight cap=1/3+Cash.
spot/fut 각각 BT. window rank-sum 으로 비교.

CLI:
  python bt_coin_selection_v25.py spot
  python bt_coin_selection_v25.py fut
"""
from __future__ import annotations
import os, sys, time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

from unified_backtest import run as bt_run, load_data
from bt_coin_3mom_v25 import live_params, window_rank_sum

START = "2020-10-01"
END = "2026-04-13"

MOM_LBS = [60, 90, 126, 180, 252, 360, 504]
SH_LBS = [30, 60, 90, 126, 180]


def run_cfg(bars, funding, asset, lev, tx, lp, selection, mom_lb=0, sh_lb=0):
    m = bt_run(
        bars, funding, interval='D',
        asset_type=asset, leverage=lev, tx_cost=tx,
        start_date=START, end_date=END,
        sma_bars=lp['sma_bars'],
        mom_short_bars=lp['ms_base'], mom_long_bars=lp['ml_base'],
        vol_threshold=lp['vol_threshold'], vol_mode=lp['vol_mode'],
        n_snapshots=lp['n_snapshots'], snap_interval_bars=lp['snap_interval_bars'],
        canary_hyst=lp['canary_hyst'], drift_threshold=lp['drift_threshold'],
        post_flip_delay=lp['post_flip_delay'],
        universe_size=lp['universe_size'], cap=lp['cap'], selection=selection,
        zscore_mom_bars=mom_lb, zscore_sharpe_bars=sh_lb,
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol',
    )
    if not m or '_equity' not in m:
        return None, None
    return m['_equity'], {k: m.get(k) for k in ('Sharpe', 'CAGR', 'MDD', 'Cal', 'Rebal')}


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in ('spot', 'fut'):
        print(__doc__); sys.exit(1)
    asset = sys.argv[1]
    t0 = time.time()
    print(f"[load_data D]"); bars, funding = load_data('D')
    lp = live_params(asset)
    tx = 0.004 if asset == 'spot' else 0.0004
    lev = 1.0 if asset == 'spot' else 3.0
    print(f"[selection BT] asset={asset} tx={tx} lev={lev}")

    eqs = {}
    # baseline: cap-sort (greedy)
    eq, sm = run_cfg(bars, funding, asset, lev, tx, lp, 'greedy')
    if eq is None:
        print("baseline empty"); sys.exit(1)
    eqs['BASE_cap'] = eq
    print(f"[BASE_cap] Sh={sm['Sharpe']:.2f} CAGR={sm['CAGR']:.1%} MDD={sm['MDD']:.1%} Cal={sm['Cal']:.2f} Rebal={sm['Rebal']}")

    # Z-score grids
    for m_lb in MOM_LBS:
        for s_lb in SH_LBS:
            tag = f"Z_m{m_lb}_s{s_lb}"
            eq, sm = run_cfg(bars, funding, asset, lev, tx, lp, 'zscore', m_lb, s_lb)
            if eq is None:
                print(f"[{tag}] empty"); continue
            eqs[tag] = eq
            print(f"[{tag}] Sh={sm['Sharpe']:.2f} CAGR={sm['CAGR']:.1%} MDD={sm['MDD']:.1%} Cal={sm['Cal']:.2f} Rebal={sm['Rebal']}")

    rs = window_rank_sum(eqs)
    if rs is None:
        print("rank-sum 부족"); sys.exit(1)
    sums, wins, n = rs
    base_avg = sums.get('BASE_cap', 0) / n
    items = sorted(sums.items(), key=lambda x: x[1])
    print(f"\nn_windows={n} (cfgs={len(eqs)})  base_avg_rank={base_avg:.3f}")
    print(f"  {'tag':<22} {'avg_rank':>9} {'win%':>6} {'vs_base':>8}")
    for k, v in items:
        marker = ' ← cap' if k == 'BASE_cap' else ''
        diff = v / n - base_avg
        print(f"  {k:<22} {v/n:>9.3f} {wins[k]/n*100:>5.1f}% {diff:>+8.3f}{marker}")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
