"""V17 snapshot phase1a_v2 — 전축 동시 최적화.

모든 축 동시 탐색 (축단위 순차 최적화 X). 로컬 옵티마 탈출 목적.
병렬처리 28 cores.

축:
  snap_days: [30, 60, 90, 126, 180, 252]      (6)  min 30
  canary_sma: [50, 100, 150, 200, 300]        (5)
  canary_hyst: [0.005, 0.010, 0.015, 0.020, 0.030]  (5)
  canary_type: [sma, ema]                      (2)
  select: [mom3_sh3, mom3, sh3, comp3, zscore3]     (5)
  def_mom_period: [63, 126, 252]               (3)
  health: [none, mom63, mom126, mom63_vol]     (4)
  weight: [ew, inv_vol]                        (2)

= 6×5×5×2×5×3×4×2 = 36,000 configs
"""
from __future__ import annotations
import os, sys, time
from itertools import product
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from stock_engine import (
    load_prices, precompute, _init, ALL_TICKERS, SP,
)
from stock_engine_snap import run_snapshot_ensemble
import stock_engine as tsi

OUT = os.path.join(HERE, 'v17_snap_out')
os.makedirs(OUT, exist_ok=True)

UNIVERSE_B = ('SPY', 'VEA', 'EEM', 'EWJ', 'INDA', 'GLD', 'PDBC')
DEF = ('TLT', 'IEF', 'BIL')


def _metrics(df):
    if df is None or len(df) < 30:
        return None
    v = df['Value']
    y = (v.index[-1] - v.index[0]).days / 365.25
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / y) - 1
    mdd = (v / v.cummax() - 1).min()
    dr = v.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {
        'CAGR': round(cagr * 100, 2), 'MDD': round(mdd * 100, 2),
        'Sharpe': round(sh, 3), 'Cal': round(cal, 3),
        'Final': round(v.iloc[-1], 2),
        'Rebals': df.attrs.get('rebal_count', 0),
    }


def run_one(snap_days, csma, hyst, ctype, sel, defm, health, weight):
    p = SP(offensive=UNIVERSE_B, defensive=DEF, canary_assets=('EEM',),
           canary_sma=csma, canary_hyst=hyst, canary_type=ctype,
           select=sel, weight=weight, defense='top2',
           def_mom_period=defm, health=health, tx_cost=0.0025,
           crash='none', sharpe_lookback=252,
           start='2017-04-01', end='2025-12-31')
    try:
        df = run_snapshot_ensemble(tsi._g_prices, tsi._g_ind, p,
                                   snap_days=snap_days, n_snap=3,
                                   monthly_anchor_mode=False)
        m = _metrics(df)
        if m is None:
            return None
        return {
            'snap_days': snap_days, 'canary_sma': csma,
            'canary_hyst': hyst, 'canary_type': ctype,
            'select': sel, 'def_mom_period': defm,
            'health': health, 'weight': weight,
            **m,
        }
    except Exception:
        return None


def main():
    grid = {
        'snap_days':      [30, 60, 90, 126, 180, 252],
        'canary_sma':     [50, 100, 150, 200, 300],
        'canary_hyst':    [0.005, 0.010, 0.015, 0.020, 0.030],
        'canary_type':    ['sma', 'ema'],
        'select':         ['mom3_sh3', 'mom3', 'sh3', 'comp3', 'zscore3'],
        'def_mom_period': [63, 126, 252],
        'health':         ['none', 'mom63', 'mom126', 'mom63_vol'],
        'weight':         ['ew', 'inv_vol'],
    }
    n = 1
    for v in grid.values():
        n *= len(v)
    print(f'Total configs: {n}')
    for k, v in grid.items():
        print(f'  {k}: {v}')

    print('Loading prices...')
    t0 = time.time()
    prices = load_prices(
        sorted(set(ALL_TICKERS) | set(UNIVERSE_B) | set(DEF)),
        start='2014-01-01',
    )
    ind = precompute(prices)
    _init(prices, ind)
    print(f'Load done ({time.time()-t0:.0f}s)')

    configs = list(product(*grid.values()))
    print(f'Running {len(configs)} configs on 28 threads...')
    t1 = time.time()
    # checkpoint every chunk (save partial)
    CHUNK = 2000
    all_rows = []
    for i in range(0, len(configs), CHUNK):
        chunk = configs[i:i+CHUNK]
        rows = Parallel(n_jobs=28, prefer='threads')(
            delayed(run_one)(*c) for c in chunk)
        rows = [r for r in rows if r]
        all_rows.extend(rows)
        elapsed = time.time() - t1
        rate = (i + len(chunk)) / max(elapsed, 1)
        eta = (len(configs) - i - len(chunk)) / max(rate, 0.01)
        print(f'  [{i+len(chunk)}/{len(configs)}] elapsed={elapsed:.0f}s rate={rate:.1f}/s eta={eta:.0f}s')
        # partial dump
        pd.DataFrame(all_rows).to_csv(
            os.path.join(OUT, 'phase1a_v2_partial.csv'), index=False)

    print(f'Done ({time.time()-t1:.0f}s, {len(all_rows)} valid rows)')
    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, 'phase1a_v2.csv'), index=False)

    print('\n=== Top 30 by Cal ===')
    top = df.sort_values('Cal', ascending=False).head(30)
    print(top[['snap_days', 'canary_sma', 'canary_hyst', 'canary_type',
               'select', 'def_mom_period', 'health', 'weight',
               'CAGR', 'MDD', 'Sharpe', 'Cal']].to_string(index=False))

    print('\n=== snap_days 별 best ===')
    bucket = df.sort_values('Cal', ascending=False).groupby('snap_days').head(1)
    print(bucket[['snap_days', 'canary_sma', 'canary_hyst', 'canary_type',
                  'select', 'def_mom_period', 'health', 'weight',
                  'CAGR', 'MDD', 'Sharpe', 'Cal']].to_string(index=False))


if __name__ == '__main__':
    main()
