#!/usr/bin/env python3
"""V17 개선 그리드 — tx 0.25% (KIS 실수수료) 기준.

V17 baseline (tx 0.1%): Sharpe 0.89, CAGR 10.1%, MDD -13.3%, Cal 0.77
V17 baseline (tx 0.25%): Sharpe 0.60, CAGR 6.5%, MDD -16.1%, Cal 0.41

개선 방향 (턴오버 축소 + flip 안정):
- canary_hyst: 0.005 → 0.01 / 0.015 / 0.02 (flip 빈도 감소)
- def_mom_period: 126 → 189 / 252 (Mom 안정성)
- crash_thresh: 0.03 → 0.04 / 0.05 (덜 민감)
- crash_cool: 3 → 5 / 10 (복귀 지연)
- defense: top3 유지 / top5 (다변화)
"""
from __future__ import annotations
import os, sys, time
from dataclasses import replace
from itertools import product

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # cap_defend

import numpy as np
import pandas as pd

from stock_engine import SP, load_prices, precompute, _init, _run_one, ALL_TICKERS
import stock_engine as tsi

OFF_R7 = ('SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC', 'VNQ')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')


def check_crash_vt(params, ind, date):
    if params.crash == 'vt':
        try:
            d = ind.get('VT', {})
            ret = d.get('ret', {}).get(date, np.nan) if hasattr(d, 'get') else np.nan
            return not np.isnan(ret) and ret <= -params.crash_thresh
        except Exception:
            return False
    return False


def bench(params_base, grid_configs):
    rows = []
    n = len(grid_configs)
    t0 = time.time()
    for i, cfg in enumerate(grid_configs, 1):
        sp = replace(params_base, **cfg)
        try:
            rs = [_run_one(replace(sp, _anchor=a)) for a in range(1, 12)]
            rs = [r for r in rs if r]
            if not rs: continue
            sharpe = np.mean([r['Sharpe'] for r in rs])
            cagr = np.mean([r['CAGR'] for r in rs])
            mdd = np.mean([r['MDD'] for r in rs])
            cal = np.mean([r.get('Calmar', 0) for r in rs])
            rows.append({**cfg,
                         'Sharpe': round(sharpe, 3),
                         'CAGR': round(cagr*100, 2),
                         'MDD': round(mdd*100, 2),
                         'Cal': round(cal, 2)})
        except Exception as e:
            print(f"  [{i}/{n}] cfg={cfg} err={e}")
            continue
        if i % 5 == 0 or i == n:
            el = time.time() - t0
            eta = el / i * (n - i)
            print(f"  [{i}/{n}] elapsed={el:.0f}s eta={eta:.0f}s")
    return pd.DataFrame(rows)


def main():
    print("Loading data...")
    t0 = time.time()
    prices = load_prices(ALL_TICKERS, start='2005-01-01')
    ind = precompute(prices)
    _init(prices, ind)
    tsi.check_crash = check_crash_vt
    print(f"  ({time.time()-t0:.0f}s)")

    TX = 0.0025  # KIS 실수수료
    BASE = SP(offensive=OFF_R7, defensive=DEF, canary_assets=('EEM',),
              canary_sma=200, canary_hyst=0.005, select='zscore3', weight='ew',
              defense='top3', def_mom_period=126, health='none', tx_cost=TX,
              crash='vt', crash_thresh=0.03, crash_cool=3, sharpe_lookback=252,
              start='2017-01-01', end='2025-12-31')

    # 1. Baseline (params = current V17)
    print("\n=== V17 현재 파라미터 @ tx 0.25% ===")
    base_df = bench(BASE, [{}])
    base_row = base_df.iloc[0].to_dict()
    print(f"  Sharpe {base_row['Sharpe']}  CAGR {base_row['CAGR']}%  MDD {base_row['MDD']}%  Cal {base_row['Cal']}")

    # 2. Grid: 턴오버 축소 중심
    print("\n=== Grid search (tx 0.25% 고정) ===")
    grid_configs = []
    for h in [0.005, 0.010, 0.015, 0.020]:
        for m in [126, 189, 252]:
            for ct in [0.03, 0.04, 0.05]:
                for cc in [3, 5, 10]:
                    grid_configs.append({
                        'canary_hyst': h, 'def_mom_period': m,
                        'crash_thresh': ct, 'crash_cool': cc,
                    })
    print(f"  {len(grid_configs)} configs × 11 anchor = {len(grid_configs)*11} runs")

    df = bench(BASE, grid_configs)

    print("\n=== Top 10 by Cal ===")
    top_cal = df.sort_values('Cal', ascending=False).head(10)
    print(top_cal.to_string(index=False))

    print("\n=== Top 10 by Sharpe ===")
    top_sh = df.sort_values('Sharpe', ascending=False).head(10)
    print(top_sh.to_string(index=False))

    # 3. Baseline vs best Cal
    best = df.sort_values('Cal', ascending=False).iloc[0]
    print(f"\n=== 비교 (Baseline vs Best Cal) ===")
    print(f"  Baseline: Sharpe {base_row['Sharpe']:.2f}  CAGR {base_row['CAGR']:+.2f}%  MDD {base_row['MDD']:+.2f}%  Cal {base_row['Cal']:.2f}")
    print(f"  Best    : Sharpe {best['Sharpe']:.2f}  CAGR {best['CAGR']:+.2f}%  MDD {best['MDD']:+.2f}%  Cal {best['Cal']:.2f}")
    print(f"  Δ Cal:  +{best['Cal'] - base_row['Cal']:.2f}  ({(best['Cal']/base_row['Cal']-1)*100:+.0f}%)")

    out = os.path.join(HERE, 'v17_improve_sweep.csv')
    df.to_csv(out, index=False)
    print(f"\n저장: {out}")


if __name__ == '__main__':
    main()
