#!/usr/bin/env python3
"""d005 decomposition Top 8 단일전략 자유조합 sweep (k=1~4).

Top 8 (G4 Cal 기준, ens4bar 제외):
  1. M1_d005__4h_orig     4h SMA240/Ms20/Ml720/daily5%/Sn60 (= d005 멤버 1)
  2. M2_2hS240__2h_orig   2h SMA240/Ms20/Ml720/bar60%/Sn120 (= 멤버 2)
  3. M2_2hS240__4h_A      4h SMA240/Ms20/Ml720/bar60%/Sn120
  4. M1_d005__2h_B        2h SMA480/Ms40/Ml1440/daily5%/Sn120
  5. M3_2hS120__4h_B      4h SMA60/Ms10/Ml360/bar60%/Sn60
  6. M3_2hS120__2h_orig   2h SMA120/Ms20/Ml720/bar60%/Sn120 (= 멤버 3)
  7. M4_4hM20__4h_orig    4h SMA240/Ms20/Ml120/bar60%/Sn21 (= 멤버 4)
  8. M1_d005__1h_B        1h SMA960/Ms80/Ml2880/daily5%/Sn240

자유조합 k=1~4: 8+28+56+70 = 162 combos × G0/G4 = 324 cases.
"""
import csv
import os
import sys
import time
from itertools import combinations

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from backtest_futures_full import load_data, run
from run_ensemble import SingleAccountEngine, combine_targets

START = '2020-10-01'
END = '2026-03-28'
OUT_CSV = os.path.join(_here, 'decomp_top8_freecombo.csv')

BASE = dict(canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
            bl_drop=0, bl_days=0, health_mode='mom2vol', n_snapshots=3)

TOP8 = {
    'd005_4h':        dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                           snap_interval_bars=60,  vol_mode='daily', vol_threshold=0.05, **BASE),
    '2hS240_2h':      dict(interval='2h', sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                           snap_interval_bars=120, vol_mode='bar',   vol_threshold=0.60, **BASE),
    '2hS240_4hA':     dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                           snap_interval_bars=120, vol_mode='bar',   vol_threshold=0.60, **BASE),
    'd005_2hB':       dict(interval='2h', sma_bars=480, mom_short_bars=40, mom_long_bars=1440,
                           snap_interval_bars=120, vol_mode='daily', vol_threshold=0.05, **BASE),
    '2hS120_4hB':     dict(interval='4h', sma_bars=60,  mom_short_bars=10, mom_long_bars=360,
                           snap_interval_bars=60,  vol_mode='bar',   vol_threshold=0.60, **BASE),
    '2hS120_2h':      dict(interval='2h', sma_bars=120, mom_short_bars=20, mom_long_bars=720,
                           snap_interval_bars=120, vol_mode='bar',   vol_threshold=0.60, **BASE),
    '4hM20_4h':       dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=120,
                           snap_interval_bars=21,  vol_mode='bar',   vol_threshold=0.60, **BASE),
    'd005_1hB':       dict(interval='1h', sma_bars=960, mom_short_bars=80, mom_long_bars=2880,
                           snap_interval_bars=240, vol_mode='daily', vol_threshold=0.05, **BASE),
}

GUARDS = {
    'G0': dict(leverage=3.0, leverage_mode='fixed', per_coin_leverage_mode='none', stop_kind='none'),
    'G4': dict(leverage=4.0, leverage_mode='fixed', per_coin_leverage_mode='cap_mom_blend_543_cash',
               stop_kind='prev_close_pct', stop_pct=0.15,
               stop_gate='cash_guard', stop_gate_cash_threshold=0.34),
}


def gen_trace(data, cfg):
    cfg = dict(cfg)
    interval = cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=START, end_date=END, _trace=trace, **cfg)
    return trace


def run_case(bars_1h, funding_1h, traces, members, guard):
    weights = {k: 1.0/len(members) for k in members}
    btc = bars_1h['BTC']
    all_dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    sub = {k: traces[k] for k in members}
    combined = combine_targets(sub, weights, all_dates)
    g = dict(guard)
    eng = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=g.get('leverage', 4.0),
        stop_kind=g.get('stop_kind', 'none'),
        stop_pct=g.get('stop_pct', 0.0),
        stop_gate=g.get('stop_gate', 'always'),
        stop_gate_cash_threshold=g.get('stop_gate_cash_threshold', 0.0),
        leverage_mode=g.get('leverage_mode', 'fixed'),
        per_coin_leverage_mode=g.get('per_coin_leverage_mode', 'none'),
        leverage_floor=3.0, leverage_mid=4.0, leverage_ceiling=5.0,
        leverage_cash_threshold=0.34,
        leverage_partial_cash_threshold=0.0,
        leverage_count_floor_max=2, leverage_count_mid_max=4,
        leverage_canary_floor_gap=0.015,
        leverage_canary_mid_gap=0.04,
        leverage_canary_high_gap=0.08,
        leverage_canary_sma_bars=1200,
        leverage_mom_lookback_bars=24*30,
        leverage_vol_lookback_bars=24*90,
    )
    return eng.run(combined)


def main():
    t0 = time.time()
    print('Loading data: D, 4h, 2h, 1h ...', flush=True)
    data = {iv: load_data(iv) for iv in ['D', '4h', '2h', '1h']}
    bars_1h, funding_1h = data['1h']

    names = list(TOP8.keys())
    print(f'Generating {len(names)} traces...', flush=True)
    traces = {}
    for n in names:
        ts = time.time()
        traces[n] = gen_trace(data, TOP8[n])
        print(f'  {n} ({time.time()-ts:.0f}s)', flush=True)

    combos = []
    for k in [1, 2, 3, 4]:
        for c in combinations(names, k):
            combos.append(list(c))
    total = len(combos) * 2
    print(f'Combos: {len(combos)} × 2 guards = {total} cases', flush=True)

    rows = []
    i = 0
    for members in combos:
        for gd in ['G0', 'G4']:
            i += 1
            ts = time.time()
            m = run_case(bars_1h, funding_1h, traces, members, GUARDS[gd])
            dt = time.time() - ts
            rows.append(dict(
                case=f'{"+".join(members)}|{gd}',
                k=len(members), guard=gd, members='+'.join(members),
                Sharpe=m.get('Sharpe', 0), CAGR=m.get('CAGR', 0),
                MDD=m.get('MDD', 0), Cal=m.get('Cal', 0),
                Liq=m.get('Liq', 0), Stops=m.get('Stops', 0), elapsed=round(dt, 1),
            ))
            if i % 30 == 0 or i == total:
                r = rows[-1]
                print(f'[{i:03d}/{total}] k{r["k"]} {r["case"][:55]:<55} '
                      f'Sh{r["Sharpe"]:.2f} Cal{r["Cal"]:.2f} ({dt:.1f}s)', flush=True)

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved: {OUT_CSV}  ({time.time()-t0:.0f}s total)', flush=True)

    # Top by Cal G4
    g4 = sorted([r for r in rows if r['guard'] == 'G4'], key=lambda r: -r['Cal'])
    print('\nTop 25 by Cal (G4):')
    print(f'{"members":<60} k {"Sh":>5} {"Cal":>5} {"CAGR":>7} {"MDD":>7} Liq Stops')
    for r in g4[:25]:
        print(f'{r["members"][:60]:<60} {r["k"]} {r["Sharpe"]:>5.2f} '
              f'{r["Cal"]:>5.2f} {r["CAGR"]:>+7.1%} {r["MDD"]:>+7.1%} '
              f'{r["Liq"]:>3} {r["Stops"]:>3}')


if __name__ == '__main__':
    main()
