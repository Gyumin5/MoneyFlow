#!/usr/bin/env python3
"""Top 5 후보(1x 레버리지) vs V19-equivalent 동일 조건 비교.

공정 비교 조건:
- 엔진: SingleAccountEngine (leverage=1.0, stop=none, per_coin_leverage=none)
- 기간: 2020-10-01 ~ 2026-03-28
- 데이터: 선물 bars/funding (basis+funding 포함)
- 거래비용: 0.04% (기본 선물 수수료)
- 가드: 없음 (pure alpha)
- V19 앵커: 30-day 롤링 (snap_interval_bars=30)

대상:
  Top1  4hA_S180 + 4h_M20 + 2h_S120              (k=3, Cal 7.18)
  Top2  4hA_S180 + 4hB_S240n60 + 2h_S240 + 2h_S120 (k=4, Cal 6.93)
  Top3  4hA_S180 + 4h_M20 + 2h_S240 + 2h_S120    (k=4, Cal 7.00)
  Top4  d005B0 변형 (d005_4h + 2hS120_4hB + 2hS120_2h + 4hM20)
  Top5  d005 B0 (= 현 선물 프로덕션)
  V19   D bar, SMA50/Ms30/Ml90/daily5%, snap30, DD60/25, BL15/7, Crash-10%

Robustness: Full + H1(2020.10~2023.06) + H2(2023.07~2026.03) 서브피리어드.
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

PERIODS = {
    'FULL': ('2020-10-01', '2026-03-28'),
    'H1':   ('2020-10-01', '2023-06-30'),
    'H2':   ('2023-07-01', '2026-03-28'),
}

OUT_CSV = os.path.join(_here, '1x_vs_v19_results.csv')
TX_COST = 0.004  # 0.4% 편도 (현물 V19 동일)

BASE = dict(canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
            bl_drop=0, bl_days=0, health_mode='mom2vol', n_snapshots=3)

# Top 후보 멤버 단일전략 trace 정의
STRATS = {
    '4h_d005':    dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                       snap_interval_bars=60,  vol_mode='daily', vol_threshold=0.05, **BASE),
    '4hA_S180':   dict(interval='4h', sma_bars=180, mom_short_bars=20, mom_long_bars=720,
                       snap_interval_bars=60,  vol_mode='daily', vol_threshold=0.05, **BASE),
    '4h_M20':     dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=120,
                       snap_interval_bars=21,  vol_mode='bar',   vol_threshold=0.60, **BASE),
    '4hB_S240n60':dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=120,
                       snap_interval_bars=60,  vol_mode='bar',   vol_threshold=0.60, **BASE),
    '2h_S240':    dict(interval='2h', sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                       snap_interval_bars=120, vol_mode='bar',   vol_threshold=0.60, **BASE),
    '2h_S120':    dict(interval='2h', sma_bars=120, mom_short_bars=20, mom_long_bars=720,
                       snap_interval_bars=120, vol_mode='bar',   vol_threshold=0.60, **BASE),
    '2hS120_4hB': dict(interval='4h', sma_bars=60,  mom_short_bars=10, mom_long_bars=360,
                       snap_interval_bars=60,  vol_mode='bar',   vol_threshold=0.60, **BASE),
}

# V19 equivalent (D bar, 30-day rolling snap, with DD/BL/Crash)
V19_EQUIV_GUARD = dict(
    interval='D',
    sma_bars=50, mom_short_bars=30, mom_long_bars=90,
    snap_interval_bars=30,
    vol_mode='daily', vol_threshold=0.05,
    canary_hyst=0.015,
    drift_threshold=0.10,
    dd_threshold=-0.25, dd_lookback=60,
    bl_drop=-0.15, bl_days=7,
    health_mode='mom2vol',
    n_snapshots=3,
)
# V19는 crash_threshold=-0.10 기본값 사용

# V19 no-guard (same alpha, no DD/BL/Crash defense)
V19_EQUIV_NOGUARD = dict(
    interval='D',
    sma_bars=50, mom_short_bars=30, mom_long_bars=90,
    snap_interval_bars=30,
    vol_mode='daily', vol_threshold=0.05,
    canary_hyst=0.015,
    drift_threshold=0,
    dd_threshold=0, dd_lookback=0,
    bl_drop=0, bl_days=0,
    crash_threshold=-1.0,  # -100% = disabled
    health_mode='mom2vol',
    n_snapshots=3,
)

V19_VARIANTS = {
    'V19_guard':   V19_EQUIV_GUARD,
    'V19_noguard': V19_EQUIV_NOGUARD,
}

ENSEMBLES = {
    'Top1_k3':  ['4hA_S180', '4h_M20', '2h_S120'],
    'Top2_k4':  ['4hA_S180', '4hB_S240n60', '2h_S240', '2h_S120'],
    'Top3_k4':  ['4hA_S180', '4h_M20', '2h_S240', '2h_S120'],
    'Top4_k4':  ['4h_d005', '2hS120_4hB', '2h_S120', '4h_M20'],  # decomp winner 변형
    'd005_B0':  ['4h_d005', '2h_S240', '2h_S120', '4h_M20'],     # 현 프로덕션
}


def gen_trace(data, cfg, start, end):
    cfg = dict(cfg)
    interval = cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0, tx_cost=TX_COST,
        start_date=start, end_date=end, _trace=trace, **cfg)
    return trace


def run_ensemble_1x(bars_1h, funding_1h, traces_dict, members, start, end):
    weights = {k: 1.0/len(members) for k in members}
    btc = bars_1h['BTC']
    all_dates = btc.index[(btc.index >= start) & (btc.index <= end)]
    sub = {k: traces_dict[k] for k in members}
    combined = combine_targets(sub, weights, all_dates)
    eng = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=1.0, tx_cost=TX_COST, stop_kind='none',
        leverage_mode='fixed', per_coin_leverage_mode='none',
    )
    return eng.run(combined)


def run_v19_1x(bars_D, funding_D, cfg, start, end):
    """V19는 단일 D-bar 전략. `run`을 1x로 직접 호출 후 동일 엔진 경로 재현 위해
    trace → SingleAccountEngine(leverage=1.0)로 평가."""
    # 방법: 1x trace 생성 후 SingleAccountEngine으로 단일 멤버 평가
    cfg = dict(cfg)
    interval = cfg.pop('interval')
    assert interval == 'D'
    trace = []
    run(bars_D, funding_D, interval='D', leverage=1.0, tx_cost=TX_COST,
        start_date=start, end_date=end, _trace=trace, **cfg)
    return trace


def main():
    t0 = time.time()
    print('Loading data: D, 4h, 2h, 1h ...', flush=True)
    data = {iv: load_data(iv) for iv in ['D', '4h', '2h', '1h']}
    bars_1h, funding_1h = data['1h']
    bars_D, funding_D = data['D']

    # For each period, regenerate traces (since run() uses start/end)
    rows = []
    for pname, (start, end) in PERIODS.items():
        print(f'\n=== Period {pname} ({start} ~ {end}) ===', flush=True)
        # Build all member traces
        needed = set()
        for mems in ENSEMBLES.values():
            needed.update(mems)
        traces = {}
        for n in sorted(needed):
            ts = time.time()
            traces[n] = gen_trace(data, STRATS[n], start, end)
            print(f'  trace {n}: {time.time()-ts:.0f}s', flush=True)

        # Ensembles at 1x, no guard
        for ens_name, members in ENSEMBLES.items():
            ts = time.time()
            m = run_ensemble_1x(bars_1h, funding_1h, traces, members, start, end)
            dt = time.time() - ts
            rows.append(dict(
                period=pname, strategy=ens_name, members='+'.join(members),
                Sharpe=m.get('Sharpe', 0), CAGR=m.get('CAGR', 0),
                MDD=m.get('MDD', 0), Cal=m.get('Cal', 0),
                Liq=m.get('Liq', 0), Stops=m.get('Stops', 0),
                elapsed=round(dt, 1),
            ))
            print(f'  {ens_name:<10} Sh{m.get("Sharpe",0):.2f} Cal{m.get("Cal",0):.2f} '
                  f'CAGR{m.get("CAGR",0):+.1%} MDD{m.get("MDD",0):+.1%} ({dt:.0f}s)', flush=True)

        # V19 equivalent at 1x (single strategy) — guard and no-guard variants
        for v19_name, v19_cfg in V19_VARIANTS.items():
            ts = time.time()
            v19_trace = run_v19_1x(bars_D, funding_D, v19_cfg, start, end)
            btc = bars_1h['BTC']
            all_dates = btc.index[(btc.index >= start) & (btc.index <= end)]
            combined = combine_targets({v19_name: v19_trace}, {v19_name: 1.0}, all_dates)
            eng = SingleAccountEngine(
                bars_1h, funding_1h,
                leverage=1.0, tx_cost=TX_COST, stop_kind='none',
                leverage_mode='fixed', per_coin_leverage_mode='none',
            )
            m = eng.run(combined)
            dt = time.time() - ts
            desc = 'D_SMA50_Ms30_Ml90_snap30' + ('_DD60-25_BL15-7_Crash10' if v19_name == 'V19_guard' else '_pure_alpha')
            rows.append(dict(
                period=pname, strategy=v19_name, members=desc,
                Sharpe=m.get('Sharpe', 0), CAGR=m.get('CAGR', 0),
                MDD=m.get('MDD', 0), Cal=m.get('Cal', 0),
                Liq=m.get('Liq', 0), Stops=m.get('Stops', 0),
                elapsed=round(dt, 1),
            ))
            print(f'  {v19_name:<11} Sh{m.get("Sharpe",0):.2f} Cal{m.get("Cal",0):.2f} '
                  f'CAGR{m.get("CAGR",0):+.1%} MDD{m.get("MDD",0):+.1%} ({dt:.0f}s)', flush=True)

    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved: {OUT_CSV}  ({time.time()-t0:.0f}s total)', flush=True)

    # Pivot table: strategy × period
    print('\n=== Summary (Sharpe / CAGR / MDD / Cal) ===')
    strats = ['Top1_k3', 'Top2_k4', 'Top3_k4', 'Top4_k4', 'd005_B0', 'V19_guard', 'V19_noguard']
    periods = list(PERIODS.keys())
    print(f'{"strategy":<12}', end='')
    for p in periods:
        print(f' | {p:<30}', end='')
    print()
    for s in strats:
        print(f'{s:<12}', end='')
        for p in periods:
            r = next((x for x in rows if x['strategy']==s and x['period']==p), None)
            if r:
                cell = f'Sh{r["Sharpe"]:.2f} C{r["Cal"]:.2f} G{r["CAGR"]:+.0%} M{r["MDD"]:+.0%}'
                print(f' | {cell:<30}', end='')
            else:
                print(f' | {"-":<30}', end='')
        print()


if __name__ == '__main__':
    main()
