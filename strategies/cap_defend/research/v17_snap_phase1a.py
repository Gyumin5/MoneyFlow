"""V17 snapshot ensemble Phase-1a coarse grid.

Grid (7 축):
  snap_days: 30 / 90 / 180 (3)
  canary_sma: 50 / 150 / 300 (3)
  canary_hyst: 0.005 / 0.020 (2)
  canary_type: sma / ema (2)
  select: mom3_sh3 / mom126 / sh3 (3)
  def_mom_period: 63 / 252 (2)
  health: none / sma200 / mom126 (3)

FIX:
  universe=B (US-light), canary_asset=EEM,
  sharpe_lookback=252, crash='none',
  defense=top2, offensive (n_mom=3), tx=0.25%

= 3×3×2×2×3×2×3 = 648 configs × 3-snap ensemble
"""
from __future__ import annotations
import os, sys, time
from itertools import product
from joblib import Parallel, delayed

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # cap_defend

from stock_engine import SP, load_prices, precompute, _init, ALL_TICKERS
import stock_engine as tsi
from stock_engine_snap import run_snapshot_ensemble

OUT = os.path.join(HERE, "v17_snap_out")
os.makedirs(OUT, exist_ok=True)

UNIVERSE_B = ('SPY', 'VEA', 'EEM', 'EWJ', 'INDA', 'GLD', 'PDBC')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')

# Grid values
G_SNAP = [30, 90, 180]
G_CSMA = [50, 150, 300]
G_HYST = [0.005, 0.020]
G_CTYPE = ['sma', 'ema']
G_SELECT = ['mom3_sh3', 'mom126', 'sh3']
G_DEFM = [63, 252]
G_HEALTH = ['none', 'sma200', 'mom126']


def _metrics(df):
    if df is None or len(df) < 30:
        return None
    v = df['Value']
    y = (v.index[-1] - v.index[0]).days / 365.25
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1/y) - 1
    mdd = (v / v.cummax() - 1).min()
    dr = v.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {
        'CAGR': round(cagr * 100, 2),
        'MDD': round(mdd * 100, 2),
        'Sharpe': round(sh, 3),
        'Cal': round(cal, 3),
        'Final': round(v.iloc[-1], 2),
        'Rebals': df.attrs.get('rebal_count', 0),
    }


def run_one(snap_days, csma, hyst, ctype, select, defm, health):
    """Single config run."""
    p = SP(
        offensive=UNIVERSE_B,
        defensive=DEF,
        canary_assets=('EEM',),
        canary_sma=csma,
        canary_hyst=hyst,
        canary_type=ctype,
        select=select,
        weight='ew',
        defense='top2',
        def_mom_period=defm,
        health=health,
        tx_cost=0.0025,
        crash='none',
        sharpe_lookback=252,
        start='2017-04-01',  # SMA300 warmup OK (2016-01-04 데이터 시작, 300 거래일 ≈ 14개월)
        end='2025-12-31',
    )
    try:
        df = run_snapshot_ensemble(tsi._g_prices, tsi._g_ind, p,
                                     snap_days=snap_days, n_snap=3,
                                     monthly_anchor_mode=False)  # calendar mode
        m = _metrics(df)
        if m is None:
            return None
        return {
            'snap_days': snap_days, 'canary_sma': csma, 'canary_hyst': hyst,
            'canary_type': ctype, 'select': select,
            'def_mom_period': defm, 'health': health,
            **m,
        }
    except Exception as e:
        return {'snap_days': snap_days, 'canary_sma': csma, 'canary_hyst': hyst,
                'canary_type': ctype, 'select': select,
                'def_mom_period': defm, 'health': health,
                'ERR': str(e)[:80]}


def main():
    print("Loading prices...")
    t0 = time.time()
    prices = load_prices(ALL_TICKERS, start='2014-01-01')
    ind = precompute(prices)
    _init(prices, ind)
    print(f"  ({time.time()-t0:.0f}s)")

    configs = list(product(G_SNAP, G_CSMA, G_HYST, G_CTYPE,
                            G_SELECT, G_DEFM, G_HEALTH))
    print(f"Grid: {len(configs)} configs")

    t0 = time.time()
    rows = Parallel(n_jobs=24, prefer='threads')(
        delayed(run_one)(*c) for c in configs)
    rows = [r for r in rows if r is not None]
    print(f"Completed {len(rows)} / {len(configs)} ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, 'phase1a.csv')
    df.to_csv(path, index=False)

    # Summary
    if 'Cal' in df.columns:
        print("\n=== Top 10 by Cal ===")
        top = df.sort_values('Cal', ascending=False).head(10)
        print(top[['snap_days','canary_sma','canary_hyst','canary_type','select',
                   'def_mom_period','health','CAGR','MDD','Sharpe','Cal']].to_string(index=False))
        print("\n=== Top 10 by Sharpe ===")
        top_sh = df.sort_values('Sharpe', ascending=False).head(10)
        print(top_sh[['snap_days','canary_sma','canary_hyst','canary_type','select',
                      'def_mom_period','health','CAGR','MDD','Sharpe','Cal']].to_string(index=False))
    print(f"\n저장: {path}")


if __name__ == '__main__':
    main()
