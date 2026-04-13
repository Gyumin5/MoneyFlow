#!/usr/bin/env python3
"""각 봉주기에서 단일전략 Top 2를 뽑아 자유조합 앙상블 sweep.

후보 풀(8개):
  D_M20      D, SMA50/Ms20/Ml90/Sn120 (코인엔진 그리드 1위)
  D_M80      D, SMA50/Ms80/Ml240/Sn120
  4h_d005    4h, SMA240/Ms20/Ml720/daily5/Sn60 (d005 멤버)
  4hA_S180   4h, SMA180/Ms20/Ml720/daily5/Sn60 (67-case Cal 4.49)
  4h_M20     4h, SMA240/Ms20/Ml120/bar60/Sn21 (d005 멤버)
  4hB_S240n60 4h, SMA240/Ms20/Ml120/bar60/Sn60 (67-case Cal 3.28)
  2h_S240    2h, SMA240/Ms20/Ml720/bar60/Sn120 (d005 멤버)
  2h_S120    2h, SMA120/Ms20/Ml720/bar60/Sn120 (d005 멤버)

자유조합: 2~5 멤버 (모든 부분집합), G0(fixed 3x) + G4(production guards) 평가.
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
OUT_CSV = os.path.join(_here, 'top2_per_bar_ensemble.csv')

STRATEGIES = {
    'D_M20': dict(interval='D', sma_bars=50, mom_short_bars=20, mom_long_bars=90,
                  canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                  bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='daily',
                  vol_threshold=0.05, n_snapshots=3, snap_interval_bars=120),
    'D_M80': dict(interval='D', sma_bars=50, mom_short_bars=80, mom_long_bars=240,
                  canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                  bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='daily',
                  vol_threshold=0.05, n_snapshots=3, snap_interval_bars=120),
    '4h_d005': dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                    canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                    bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='daily',
                    vol_threshold=0.05, n_snapshots=3, snap_interval_bars=60),
    '4hA_S180': dict(interval='4h', sma_bars=180, mom_short_bars=20, mom_long_bars=720,
                     canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                     bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='daily',
                     vol_threshold=0.05, n_snapshots=3, snap_interval_bars=60),
    '4h_M20': dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=120,
                   canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                   bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
                   vol_threshold=0.60, n_snapshots=3, snap_interval_bars=21),
    '4hB_S240n60': dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=120,
                        canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                        bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
                        vol_threshold=0.60, n_snapshots=3, snap_interval_bars=60),
    '2h_S240': dict(interval='2h', sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                    canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                    bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
                    vol_threshold=0.60, n_snapshots=3, snap_interval_bars=120),
    '2h_S120': dict(interval='2h', sma_bars=120, mom_short_bars=20, mom_long_bars=720,
                    canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                    bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
                    vol_threshold=0.60, n_snapshots=3, snap_interval_bars=120),
}

GUARDS = {
    'G0': dict(leverage=3.0, leverage_mode='fixed', per_coin_leverage_mode='none', stop_kind='none'),
    'G4': dict(leverage=4.0, leverage_mode='fixed', per_coin_leverage_mode='cap_mom_blend_543_cash',
               stop_kind='prev_close_pct', stop_pct=0.15,
               stop_gate='cash_guard', stop_gate_cash_threshold=0.34),
}


def gen_trace(data, name, cfg):
    cfg = dict(cfg)
    interval = cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=START, end_date=END, _trace=trace, **cfg)
    return trace


def run_case(bars_1h, funding_1h, traces, members, guard):
    weights = {k: 1.0 / len(members) for k in members}
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

    names = list(STRATEGIES.keys())
    print(f'Generating {len(names)} single-strategy traces...', flush=True)
    traces = {}
    for n in names:
        ts = time.time()
        traces[n] = gen_trace(data, n, STRATEGIES[n])
        print(f'  {n} ({time.time()-ts:.0f}s)', flush=True)

    # Free combinations: 2~5 members
    combos = []
    for k in [2, 3, 4, 5]:
        for c in combinations(names, k):
            combos.append(list(c))
    print(f'Total combos: {len(combos)} × 2 guards = {len(combos)*2} cases', flush=True)

    rows = []
    total = len(combos) * 2
    i = 0
    for members in combos:
        for gd in ['G0', 'G4']:
            i += 1
            ts = time.time()
            m = run_case(bars_1h, funding_1h, traces, members, GUARDS[gd])
            dt = time.time() - ts
            row = dict(
                case=f'{"+".join(members)}|{gd}',
                k=len(members), guard=gd,
                members='+'.join(members),
                Sharpe=m.get('Sharpe', 0), CAGR=m.get('CAGR', 0),
                MDD=m.get('MDD', 0), Cal=m.get('Cal', 0),
                Liq=m.get('Liq', 0), Stops=m.get('Stops', 0),
                elapsed=round(dt, 1),
            )
            rows.append(row)
            if i % 20 == 0 or i == total:
                print(f'[{i:03d}/{total}] {row["case"][:50]:<50} '
                      f'Sh{row["Sharpe"]:.2f} Cal{row["Cal"]:.2f} ({dt:.1f}s)',
                      flush=True)

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved: {OUT_CSV}  ({time.time()-t0:.0f}s total)', flush=True)

    # Top by Cal (G4 only — production guards)
    g4 = [r for r in rows if r['guard'] == 'G4']
    g4.sort(key=lambda r: -r['Cal'])
    print('\nTop 20 by Cal (G4 production guards):')
    print(f'{"members":<60} k {"Sh":>5} {"Cal":>5} {"CAGR":>7} {"MDD":>7} Liq Stops')
    for r in g4[:20]:
        print(f'{r["members"][:60]:<60} {r["k"]} {r["Sharpe"]:>5.2f} '
              f'{r["Cal"]:>5.2f} {r["CAGR"]:>+7.1%} {r["MDD"]:>+7.1%} '
              f'{r["Liq"]:>3} {r["Stops"]:>3}')


if __name__ == '__main__':
    main()
