#!/usr/bin/env python3
"""V19 단일전략 변형 1x 비교 (SingleAccountEngine).

변형:
  V19_guard_33cap  — DD/BL/Crash 가드 ON, cap=1/3 (프로덕션)
  V19_noguard_33cap — DD/BL/Crash 가드 OFF, cap=1/3
  V19_noguard_20cap — DD/BL/Crash 가드 OFF, cap=0.20

모드: SPOT_1x (tx 0.4%), FUT_1x (tx 0.04%)
기간: 2020.10 ~ 2026.03
"""
import csv
import os
import sys
import time

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from backtest_futures_full import load_data, run
from run_ensemble import SingleAccountEngine, combine_targets

START = '2020-10-01'
END = '2026-03-28'
OUT_CSV = os.path.join(_here, 'grid_results', 'v19_variants_1x.csv')

V19_BASE = dict(
    interval='D',
    sma_bars=50, mom_short_bars=30, mom_long_bars=90,
    snap_interval_bars=30,
    vol_mode='daily', vol_threshold=0.05,
    canary_hyst=0.015,
    health_mode='mom2vol',
    n_snapshots=3,
)

VARIANTS = {
    'V19_guard_33cap': dict(V19_BASE,
        drift_threshold=0.10,
        dd_threshold=-0.25, dd_lookback=60,
        bl_drop=-0.15, bl_days=7,
        cap=1.0/3.0,
    ),
    'V19_noguard_33cap': dict(V19_BASE,
        drift_threshold=0,
        dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0,
        crash_threshold=-1.0,
        cap=1.0/3.0,
    ),
    'V19_noguard_20cap': dict(V19_BASE,
        drift_threshold=0,
        dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0,
        crash_threshold=-1.0,
        cap=0.20,
    ),
}

MODES = {
    'SPOT_1x': dict(tx_cost=0.004),
    'FUT_1x':  dict(tx_cost=0.0004),
}


def gen_trace(data, cfg):
    cfg = dict(cfg)
    interval = cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=START, end_date=END, _trace=trace, **cfg)
    return trace


def eval_single(bars_1h, funding_1h, trace, name, tx_cost):
    btc = bars_1h['BTC']
    all_dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    combined = combine_targets({name: trace}, {name: 1.0}, all_dates)
    eng = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=1.0, tx_cost=tx_cost, stop_kind='none',
        leverage_mode='fixed', per_coin_leverage_mode='none',
    )
    return eng.run(combined)


def main():
    t0 = time.time()
    print('Loading data: D, 1h ...', flush=True)
    data = {iv: load_data(iv) for iv in ['D', '1h']}
    bars_1h, funding_1h = data['1h']

    # Generate traces (3 variants)
    traces = {}
    for name, cfg in VARIANTS.items():
        ts = time.time()
        traces[name] = gen_trace(data, cfg)
        print(f'  trace {name}: {time.time()-ts:.1f}s', flush=True)

    rows = []
    for mode_name, mode_cfg in MODES.items():
        print(f'\n=== {mode_name} ===', flush=True)
        for vname, vcfg in VARIANTS.items():
            ts = time.time()
            m = eval_single(bars_1h, funding_1h, traces[vname], vname, mode_cfg['tx_cost'])
            dt = time.time() - ts
            rows.append(dict(
                mode=mode_name, strategy=vname,
                Sharpe=m.get('Sharpe', 0), CAGR=m.get('CAGR', 0),
                MDD=m.get('MDD', 0), Cal=m.get('Cal', 0),
                Liq=m.get('Liq', 0), Stops=m.get('Stops', 0),
            ))
            print(f'  {vname:<22} Sh{m.get("Sharpe",0):.2f} Cal{m.get("Cal",0):.2f} '
                  f'CAGR{m.get("CAGR",0):+.1%} MDD{m.get("MDD",0):+.1%} ({dt:.0f}s)', flush=True)

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved: {OUT_CSV}  ({time.time()-t0:.0f}s total)', flush=True)

    # Pivot
    print('\n=== Summary ===')
    print(f'{"variant":<22}', end='')
    for m in MODES:
        print(f' | {m:<32}', end='')
    print()
    for v in VARIANTS:
        print(f'{v:<22}', end='')
        for mn in MODES:
            r = next((x for x in rows if x['strategy']==v and x['mode']==mn), None)
            if r:
                cell = f'Sh{r["Sharpe"]:.2f} C{r["Cal"]:.2f} G{r["CAGR"]:+.0%} M{r["MDD"]:+.0%}'
                print(f' | {cell:<32}', end='')
        print()


if __name__ == '__main__':
    main()
