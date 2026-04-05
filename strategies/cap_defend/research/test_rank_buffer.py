#!/usr/bin/env python3
"""Rank Buffer 실험: V17 주식 baseline vs rank_buffer=1,2,3."""

import os, sys, time, warnings
import numpy as np
from dataclasses import replace

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_engine import (
    SP, load_prices, precompute, metrics,
    _init as stock_init, _run_one, get_val,
    ALL_TICKERS,
)
import stock_engine as tsi
import numpy as _np

def check_crash_vt(params, ind, date):
    if params.crash == 'vt':
        r = get_val(ind, 'VT', date, 'ret')
        return not _np.isnan(r) and r <= -params.crash_thresh
    return False

OFF_R7 = ('SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC', 'VNQ')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')

# V17 baseline
V17_BASE = SP(
    offensive=OFF_R7, defensive=DEF, canary_assets=('EEM',),
    canary_sma=200, canary_hyst=0.005, select='zscore3', weight='ew',
    defense='top3', def_mom_period=126, health='none', tx_cost=0.001,
    crash='vt', crash_thresh=0.03, crash_cool=3, sharpe_lookback=252,
)

PERIODS = [
    ('2017-01-01', '2025-12-31'),
    ('2018-01-01', '2025-12-31'),
    ('2021-01-01', '2025-12-31'),
]
ANCHORS = range(1, 12)  # 11-anchor average


def run_experiment(label, sp_template):
    """Run 11-anchor average for all periods."""
    results = []
    for start, end in PERIODS:
        sharpes, cagrs, mdds, calmars, rebals = [], [], [], [], []
        for a in ANCHORS:
            sp = replace(sp_template, start=start, end=end, _anchor=a)
            r = _run_one(sp)
            if r:
                sharpes.append(r['Sharpe'])
                cagrs.append(r['CAGR'])
                mdds.append(r['MDD'])
                calmars.append(r.get('Calmar', 0))
                rebals.append(r.get('Rebal', 0))
        if sharpes:
            results.append({
                'period': f'{start[:4]}~{end[:4]}',
                'sharpe': np.mean(sharpes),
                'sigma_sh': np.std(sharpes),
                'cagr': np.mean(cagrs),
                'mdd': np.mean(mdds),
                'calmar': np.mean(calmars),
                'rebal': np.mean(rebals),
            })
    return results


def main():
    t0 = time.time()
    print("데이터 로딩...")
    stock_prices = load_prices(ALL_TICKERS, start='2005-01-01')
    stock_ind = precompute(stock_prices)
    stock_init(stock_prices, stock_ind)
    tsi.check_crash = check_crash_vt
    print(f"  완료 ({time.time()-t0:.1f}s)\n")

    configs = [
        ('V17 baseline (buf=0)', replace(V17_BASE, rank_buffer=0)),
        ('V17 + buf=1 (Top4유지)', replace(V17_BASE, rank_buffer=1)),
        ('V17 + buf=2 (Top5유지)', replace(V17_BASE, rank_buffer=2)),
        ('V17 + buf=3 (Top6유지)', replace(V17_BASE, rank_buffer=3)),
    ]

    print(f"{'설정':<25s} {'기간':>10s} {'Sharpe':>7s} {'σ(Sh)':>6s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s} {'Rebal':>6s}")
    print("-" * 80)

    for label, sp in configs:
        results = run_experiment(label, sp)
        for r in results:
            print(f"{label:<25s} {r['period']:>10s} {r['sharpe']:>7.3f} {r['sigma_sh']:>6.3f} {r['cagr']:>+8.1%} {r['mdd']:>+8.1%} {r['calmar']:>7.2f} {r['rebal']:>6.1f}")
        print()

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
