#!/usr/bin/env python3
"""엔진 B(SingleAccountEngine) 기반 d005 vs 신규 앙상블 검증.

Stage 0: B0 (d005) × G0/G1/G2/G4 = 4 cases (베이스라인 고정)
Stage 1: B1/B2/N0/N1/N2 × G0/G4 = 10 cases (후보 스크리닝)
총 14 cases. Stage 2/3는 결과 보고 별도 실행.

후보 정의:
  B0 = 4h_d005 + 2h_S240 + 2h_S120 + 4h_M20  (d005 production)
  B1 = B0 - 2h_S240                            (2h_S240 기여도)
  B2 = B0 - 2h_S120                            (2h_S120 기여도)
  N0 = D_M20 + D_M80 + 4h_S240new + 2h_S480    (신규 챔피언)
  N1 = N0 - 2h_S480                            (신규 2h 기여도)
  N2 = N0 - 2h_S480 + 4h_M20                   (2h를 4h로 교체)

가드:
  G0: fixed 3x, no stop
  G1: fixed 5x, no stop
  G2: cap_mom_blend_543_cash, no stop
  G4: cap_mom_blend_543_cash + prev_close15 + cash_guard34
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
OUT_CSV = os.path.join(_here, 'engine_b_validation_results.csv')

STRATEGIES = {
    # d005 production set
    '4h_d005': dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                    canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                    bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='daily',
                    vol_threshold=0.05, n_snapshots=3, snap_interval_bars=60),
    '2h_S240': dict(interval='2h', sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                    canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                    bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
                    vol_threshold=0.60, n_snapshots=3, snap_interval_bars=120),
    '2h_S120': dict(interval='2h', sma_bars=120, mom_short_bars=20, mom_long_bars=720,
                    canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                    bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
                    vol_threshold=0.60, n_snapshots=3, snap_interval_bars=120),
    '4h_M20': dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=120,
                   canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                   bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='bar',
                   vol_threshold=0.60, n_snapshots=3, snap_interval_bars=21),
    # new ensemble candidates
    'D_M20': dict(interval='D', sma_bars=50, mom_short_bars=20, mom_long_bars=90,
                  canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                  bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='daily',
                  vol_threshold=0.05, n_snapshots=3, snap_interval_bars=120),
    'D_M80': dict(interval='D', sma_bars=50, mom_short_bars=80, mom_long_bars=240,
                  canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                  bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='daily',
                  vol_threshold=0.05, n_snapshots=3, snap_interval_bars=120),
    '4h_S240new': dict(interval='4h', sma_bars=240, mom_short_bars=30, mom_long_bars=720,
                       canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                       bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='daily',
                       vol_threshold=0.05, n_snapshots=3, snap_interval_bars=120),
    '2h_S480': dict(interval='2h', sma_bars=480, mom_short_bars=10, mom_long_bars=120,
                    canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
                    bl_drop=0, bl_days=0, health_mode='mom2vol', vol_mode='daily',
                    vol_threshold=0.05, n_snapshots=3, snap_interval_bars=120),
}

ENSEMBLES = {
    'B0': ['4h_d005', '2h_S240', '2h_S120', '4h_M20'],
    'B1': ['4h_d005', '2h_S120', '4h_M20'],
    'B2': ['4h_d005', '2h_S240', '4h_M20'],
    'N0': ['D_M20', 'D_M80', '4h_S240new', '2h_S480'],
    'N1': ['D_M20', 'D_M80', '4h_S240new'],
    'N2': ['D_M20', 'D_M80', '4h_S240new', '4h_M20'],
}

GUARDS = {
    'G0': dict(leverage=3.0, leverage_mode='fixed', per_coin_leverage_mode='none', stop_kind='none'),
    'G1': dict(leverage=5.0, leverage_mode='fixed', per_coin_leverage_mode='none', stop_kind='none'),
    'G2': dict(leverage=4.0, leverage_mode='fixed', per_coin_leverage_mode='cap_mom_blend_543_cash',
               stop_kind='none'),
    'G4': dict(leverage=4.0, leverage_mode='fixed', per_coin_leverage_mode='cap_mom_blend_543_cash',
               stop_kind='prev_close_pct', stop_pct=0.15,
               stop_gate='cash_guard', stop_gate_cash_threshold=0.34),
}

CASES = [
    # Stage 0: B0 × all 4 guards
    ('B0', 'G0'), ('B0', 'G1'), ('B0', 'G2'), ('B0', 'G4'),
    # Stage 1: 5 candidates × G0/G4
    ('B1', 'G0'), ('B1', 'G4'),
    ('B2', 'G0'), ('B2', 'G4'),
    ('N0', 'G0'), ('N0', 'G4'),
    ('N1', 'G0'), ('N1', 'G4'),
    ('N2', 'G0'), ('N2', 'G4'),
]


def gen_trace(data, name, cfg):
    cfg = dict(cfg)
    interval = cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    print(f'  trace: {name} ({interval})', flush=True)
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

    # 필요한 단일 전략 trace 생성
    needed = set()
    for members in ENSEMBLES.values():
        needed.update(members)
    print(f'Generating {len(needed)} single-strategy traces...', flush=True)
    traces = {}
    for name in sorted(needed):
        cfg = STRATEGIES[name]
        traces[name] = gen_trace(data, name, cfg)

    # 케이스 실행
    rows = []
    for i, (ens, gd) in enumerate(CASES, 1):
        members = ENSEMBLES[ens]
        guard = GUARDS[gd]
        elapsed_start = time.time()
        m = run_case(bars_1h, funding_1h, traces, members, guard)
        dt = time.time() - elapsed_start
        row = dict(case=f'{ens}_{gd}', ens=ens, guard=gd,
                   members='+'.join(members),
                   Sharpe=m.get('Sharpe', 0), CAGR=m.get('CAGR', 0),
                   MDD=m.get('MDD', 0), Cal=m.get('Cal', 0),
                   Liq=m.get('Liq', 0), Stops=m.get('Stops', 0),
                   elapsed=round(dt, 1))
        rows.append(row)
        print(f'[{i:02d}/{len(CASES)}] {row["case"]:<10} Sh{row["Sharpe"]:.2f} '
              f'CAGR{row["CAGR"]:+.1%} MDD{row["MDD"]:+.1%} Cal{row["Cal"]:.2f} '
              f'Liq{row["Liq"]} Stops{row["Stops"]} ({dt:.0f}s)', flush=True)

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved: {OUT_CSV}  ({time.time()-t0:.0f}s total)', flush=True)


if __name__ == '__main__':
    main()
