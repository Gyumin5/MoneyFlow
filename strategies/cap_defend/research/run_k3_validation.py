#!/usr/bin/env python3
"""k=3 (V19 + 4h_L720/Sn120 + 4h_L120/Sn60) 로버스트 검증.
run_k12_validation.py와 동일 구조, k3 단독."""
import csv
import os
import sys
import time
import numpy as np
from datetime import datetime, timedelta

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from backtest_futures_full import load_data, run
from run_ensemble import SingleAccountEngine, combine_targets

TX = 0.004
OUT_DIR = os.path.join(_here, 'grid_results')

BASE = dict(canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
            bl_drop=0, bl_days=0, health_mode='mom2vol', n_snapshots=3,
            vol_mode='daily', vol_threshold=0.05)

MEMBERS = {
    'V19':           dict(interval='D',  sma_bars=50,  mom_short_bars=30, mom_long_bars=90,  snap_interval_bars=30,  **BASE),
    '4h_L120_Sn60':  dict(interval='4h', sma_bars=240, mom_short_bars=30, mom_long_bars=120, snap_interval_bars=60,  **BASE),
    '4h_L720_Sn120': dict(interval='4h', sma_bars=240, mom_short_bars=30, mom_long_bars=720, snap_interval_bars=120, **BASE),
}

STRATS = {'k3_V19+4hL120+4hL720': ['V19', '4h_L120_Sn60', '4h_L720_Sn120']}

PERIODS = {
    'Full': ('2020-10-01', '2026-03-28'),
    'H1':   ('2020-10-01', '2022-12-31'),
    'H2':   ('2023-01-01', '2024-06-30'),
    'H3':   ('2024-07-01', '2026-03-28'),
}


def gen_trace(data, mem_name, start, end):
    cfg = dict(MEMBERS[mem_name])
    interval = cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0, tx_cost=TX,
        start_date=start, end_date=end, _trace=trace, **cfg)
    return trace


def eval_strat(bars_1h, funding_1h, traces, members, start, end):
    w = {m: 1.0/len(members) for m in members}
    btc = bars_1h['BTC']
    all_dates = btc.index[(btc.index >= start) & (btc.index <= end)]
    sub = {m: traces[m] for m in members}
    combined = combine_targets(sub, w, all_dates)
    eng = SingleAccountEngine(
        bars_1h, funding_1h, leverage=1.0, tx_cost=TX, stop_kind='none',
        leverage_mode='fixed', per_coin_leverage_mode='none',
    )
    return eng.run(combined)


def main():
    t0 = time.time()
    print('Loading data: D, 4h, 1h ...', flush=True)
    data = {iv: load_data(iv) for iv in ['D', '4h', '1h']}
    bars_1h, funding_1h = data['1h']

    print('\n=== Part 1: 10-anchor dispersion (Full) ===', flush=True)
    rows1 = []
    for sname, members in STRATS.items():
        shs, cagrs, mdds = [], [], []
        for shift in range(10):
            start = (datetime(2020, 10, 1) + timedelta(days=shift)).strftime('%Y-%m-%d')
            end = '2026-03-28'
            traces = {m: gen_trace(data, m, start, end) for m in members}
            res = eval_strat(bars_1h, funding_1h, traces, members, start, end)
            shs.append(res.get('Sharpe', 0))
            cagrs.append(res.get('CAGR', 0))
            mdds.append(res.get('MDD', 0))
            rows1.append(dict(strategy=sname, shift=shift, start=start,
                              Sharpe=res.get('Sharpe',0), CAGR=res.get('CAGR',0),
                              MDD=res.get('MDD',0), Cal=res.get('Cal',0)))
            print(f'    shift{shift} Sh{res.get("Sharpe",0):.2f} CAGR{res.get("CAGR",0):+.1%} MDD{res.get("MDD",0):+.1%}', flush=True)
        print(f'  {sname} Sh {np.mean(shs):.3f}±{np.std(shs):.3f}  '
              f'CAGR {np.mean(cagrs):+.1%}±{np.std(cagrs):.1%}  '
              f'MDD {np.mean(mdds):+.1%}±{np.std(mdds):.1%}', flush=True)

    print('\n=== Part 2: Sub-period ===', flush=True)
    rows2 = []
    for sname, members in STRATS.items():
        for pname, (start, end) in PERIODS.items():
            traces = {m: gen_trace(data, m, start, end) for m in members}
            res = eval_strat(bars_1h, funding_1h, traces, members, start, end)
            rows2.append(dict(strategy=sname, period=pname, start=start, end=end,
                              Sharpe=res.get('Sharpe',0), CAGR=res.get('CAGR',0),
                              MDD=res.get('MDD',0), Cal=res.get('Cal',0)))
            print(f'  {sname} {pname:<5} Sh{res.get("Sharpe",0):.2f} '
                  f'CAGR{res.get("CAGR",0):+.1%} MDD{res.get("MDD",0):+.1%} '
                  f'Cal{res.get("Cal",0):.2f}', flush=True)

    with open(os.path.join(OUT_DIR, 'k3_validation_anchors.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows1[0].keys()))
        w.writeheader(); w.writerows(rows1)
    with open(os.path.join(OUT_DIR, 'k3_validation_periods.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows2[0].keys()))
        w.writeheader(); w.writerows(rows2)

    print(f'\nDone. {time.time()-t0:.0f}s.', flush=True)


if __name__ == '__main__':
    main()
