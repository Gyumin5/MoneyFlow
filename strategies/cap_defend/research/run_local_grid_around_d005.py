#!/usr/bin/env python3
"""d005 인근 국소 그리드 — 엔진 B(SingleAccountEngine) 기준 단일 전략 평가.

목적: 원래 그리드(coin_engine 기반)가 놓친 d005 영역(vol_mode=bar, Ms20, Sn21, N_picks=5)을
     실제 운영 엔진(futures_ensemble_engine)으로 직접 탐색.

후보 구성 (각 d005 sleeve 인근 변형):
  Family A — 4h_d005 인근 (daily vol)
    SMA × Ml × Sn × vol_thresh
  Family B — 4h_M20 인근 (bar vol)
    SMA × vol_thresh × Sn
  Family C — 2h_S* 인근 (bar vol)
    SMA × vol_thresh × Sn

평가: 단일 전략을 fixed_3x로 백테스트 → Sharpe/Cal로 ranking.
"""
import csv
import os
import sys
import time
from itertools import product

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from backtest_futures_full import load_data, run
from run_ensemble import SingleAccountEngine, combine_targets

START = '2020-10-01'
END = '2026-03-28'
OUT_CSV = os.path.join(_here, 'local_grid_around_d005.csv')


def make_cfg(interval, sma, ms, ml, vol_mode, vol_threshold, snap):
    return dict(
        interval=interval, sma_bars=sma, mom_short_bars=ms, mom_long_bars=ml,
        canary_hyst=0.015, drift_threshold=0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode='mom2vol', vol_mode=vol_mode, vol_threshold=vol_threshold,
        n_snapshots=3, snap_interval_bars=snap,
    )


def build_candidates():
    cands = {}
    # Family A: 4h daily-vol (4h_d005 = SMA240/M20/L720/daily5%/Sn60)
    for sma in [180, 240, 360]:
        for ml in [480, 720]:
            for vt in [0.05, 0.07]:
                for sn in [60, 90]:
                    name = f'4hA_S{sma}_L{ml}_v{int(vt*100)}_n{sn}'
                    cands[name] = make_cfg('4h', sma, 20, ml, 'daily', vt, sn)

    # Family B: 4h bar-vol (4h_M20 = SMA240/M20/L120/bar60%/Sn21)
    for sma in [180, 240, 360]:
        for vt in [0.50, 0.60, 0.80]:
            for sn in [21, 42, 60]:
                name = f'4hB_S{sma}_v{int(vt*100)}_n{sn}'
                cands[name] = make_cfg('4h', sma, 20, 120, 'bar', vt, sn)

    # Family C: 2h bar-vol (2h_S240 = SMA240/M20/L720/bar60%/Sn120, 2h_S120 = SMA120/...)
    for sma in [60, 120, 240, 360]:
        for vt in [0.60, 0.80]:
            for sn in [90, 120]:
                name = f'2hC_S{sma}_v{int(vt*100)}_n{sn}'
                cands[name] = make_cfg('2h', sma, 20, 720, 'bar', vt, sn)

    return cands


def gen_trace(data, name, cfg):
    cfg = dict(cfg)
    interval = cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=START, end_date=END, _trace=trace, **cfg)
    return trace


def run_single(bars_1h, funding_1h, trace, leverage=3.0):
    btc = bars_1h['BTC']
    all_dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    combined = combine_targets({'x': trace}, {'x': 1.0}, all_dates)
    eng = SingleAccountEngine(
        bars_1h, funding_1h, leverage=leverage, stop_kind='none',
        leverage_mode='fixed', per_coin_leverage_mode='none',
    )
    return eng.run(combined)


def main():
    t0 = time.time()
    cands = build_candidates()
    print(f'후보 수: {len(cands)}', flush=True)

    print('데이터 로드: D, 4h, 2h, 1h ...', flush=True)
    data = {iv: load_data(iv) for iv in ['D', '4h', '2h', '1h']}
    bars_1h, funding_1h = data['1h']

    rows = []
    for i, (name, cfg) in enumerate(cands.items(), 1):
        ts = time.time()
        trace = gen_trace(data, name, cfg)
        m = run_single(bars_1h, funding_1h, trace, leverage=3.0)
        dt = time.time() - ts
        row = dict(
            name=name, interval=cfg['interval'], sma=cfg['sma_bars'],
            ms=cfg['mom_short_bars'], ml=cfg['mom_long_bars'],
            vol_mode=cfg['vol_mode'], vol_thresh=cfg['vol_threshold'],
            snap=cfg['snap_interval_bars'],
            Sharpe=m.get('Sharpe', 0), CAGR=m.get('CAGR', 0),
            MDD=m.get('MDD', 0), Cal=m.get('Cal', 0),
            Liq=m.get('Liq', 0), elapsed=round(dt, 1),
        )
        rows.append(row)
        if i % 5 == 0 or i == len(cands):
            print(f'[{i:03d}/{len(cands)}] {name:<28} Sh{row["Sharpe"]:.2f} '
                  f'Cal{row["Cal"]:.2f} ({dt:.0f}s)', flush=True)

    # 저장
    with open(OUT_CSV, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f'\nSaved: {OUT_CSV}  ({time.time()-t0:.0f}s)', flush=True)

    # Top 결과
    rows_cal = sorted(rows, key=lambda r: -r['Cal'])
    rows_sh = sorted(rows, key=lambda r: -r['Sharpe'])
    print('\nTop 15 by Cal (3x fixed, 단일 전략):')
    print(f'{"name":<28} {"iv":<3} {"SMA":>4} {"Ml":>4} {"vol":>8} {"Sn":>4} '
          f'{"Sh":>5} {"Cal":>5} {"CAGR":>7} {"MDD":>7}')
    for r in rows_cal[:15]:
        vol_s = f'{r["vol_mode"]}{r["vol_thresh"]}'
        print(f'{r["name"]:<28} {r["interval"]:<3} {r["sma"]:>4} {r["ml"]:>4} '
              f'{vol_s:>8} {r["snap"]:>4} {r["Sharpe"]:>5.2f} {r["Cal"]:>5.2f} '
              f'{r["CAGR"]:>+7.1%} {r["MDD"]:>+7.1%}')


if __name__ == '__main__':
    main()
