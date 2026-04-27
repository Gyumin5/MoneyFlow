"""Phase 0c — V21 baseline [120,30,120] phase (0,0,0) 가 과적합인지 검증.

3개 축 독립 테스트:
  A. 공통 phase sweep: (k,k,k), k ∈ 9 값. 3 멤버 동시 시프트.
     → 분산 크면 alpha 가 phase 의존 = overfit
  B. 시작일 sweep: 동일 파라미터, start_date 만 6개월 이동 7개
     → 기간 시작점 민감도
  C. Rolling sub-period: 2년 윈도우 4개
     → 알파 시간 안정성 (bull/bear/sideways 구간별)

SN = [120, 30, 120] 고정, L3 3x, 가드 없음, EW 1/3.
"""
from __future__ import annotations
import os, sys, time
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from backtest_futures_full import load_data, run as bf_run
from futures_ensemble_engine import SingleAccountEngine, combine_targets

OUT = HERE

MEMBER_BASE = {
    'M1_S240_SN120': dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                          canary_hyst=0.015, drift_threshold=0.0,
                          dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
                          health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                          n_snapshots=3, snap_interval_bars=120),
    'M2_S240_SN30':  dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=480,
                          canary_hyst=0.015, drift_threshold=0.0,
                          dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
                          health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                          n_snapshots=3, snap_interval_bars=30),
    'M3_S120_SN120': dict(interval='4h', sma_bars=120, mom_short_bars=20, mom_long_bars=720,
                          canary_hyst=0.015, drift_threshold=0.0,
                          dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
                          health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                          n_snapshots=3, snap_interval_bars=120),
}


def gen_trace(data, cfg, phase_off, start, end):
    bars, funding = data[cfg['interval']]
    trace = []
    rc = dict(cfg)
    interval = rc.pop('interval')
    bf_run(bars, funding, interval=interval, leverage=1.0,
           start_date=start, end_date=end,
           phase_offset_bars=phase_off,
           _trace=trace, **rc)
    return trace


def run_config(data, phase_tuple, start, end, label):
    traces = {name: gen_trace(data, base, po, start, end)
              for (name, base), po in zip(MEMBER_BASE.items(), phase_tuple)}
    bars_1h, funding_1h = data['1h']
    all_dates = bars_1h['BTC'].index[(bars_1h['BTC'].index >= start) &
                                     (bars_1h['BTC'].index <= end)]
    weights = {k: 1/3 for k in MEMBER_BASE}
    combined = combine_targets(traces, weights, all_dates)
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=3.0,
        stop_kind='none', stop_pct=0.0,
        stop_gate='none', stop_gate_cash_threshold=0.0,
        per_coin_leverage_mode='fixed',
        leverage_floor=3.0, leverage_mid=3.0, leverage_ceiling=3.0,
    )
    m = engine.run(combined)
    m['label'] = label
    m['phase'] = str(phase_tuple)
    m['start'] = start
    m['end'] = end
    return m


def main():
    t0 = time.time()
    print('[P0c] Loading data...')
    data = {iv: load_data(iv) for iv in ['4h', '1h']}
    print(f'  done ({time.time()-t0:.0f}s)')

    results = []

    # A. 공통 phase sweep (k, k, k)
    print('\n=== A. 공통 phase sweep ===')
    FULL_START = '2020-10-01'
    FULL_END = '2026-03-28'
    for k in [0, 7, 13, 19, 31, 47, 59, 83, 113]:
        t1 = time.time()
        label = f'A_common_k{k:03d}'
        print(f'  {label} ... ', end='', flush=True)
        m = run_config(data, (k, k, k), FULL_START, FULL_END, label)
        m['axis'] = 'A_common_phase'
        m['axis_value'] = k
        m['elapsed_s'] = round(time.time() - t1, 1)
        results.append(m)
        print(f"Cal={m.get('Cal',0):.2f} MDD={m.get('MDD',0):+.1%} ({m['elapsed_s']}s)")

    # B. 시작일 sweep (phase=0)
    print('\n=== B. 시작일 sweep ===')
    START_CANDIDATES = [
        ('2020-10-01', '2026-03-28'),
        ('2020-11-01', '2026-03-28'),
        ('2020-12-01', '2026-03-28'),
        ('2021-01-01', '2026-03-28'),
        ('2021-02-01', '2026-03-28'),
        ('2021-03-01', '2026-03-28'),
        ('2021-06-01', '2026-03-28'),
    ]
    for s, e in START_CANDIDATES:
        t1 = time.time()
        label = f'B_start_{s[:7]}'
        print(f'  {label} ... ', end='', flush=True)
        m = run_config(data, (0, 0, 0), s, e, label)
        m['axis'] = 'B_start_date'
        m['axis_value'] = s
        m['elapsed_s'] = round(time.time() - t1, 1)
        results.append(m)
        print(f"Cal={m.get('Cal',0):.2f} MDD={m.get('MDD',0):+.1%} ({m['elapsed_s']}s)")

    # C. Rolling 2-year windows (phase=0)
    print('\n=== C. Rolling 2-year sub-period ===')
    WINDOWS = [
        ('2020-10-01', '2022-10-01'),
        ('2021-10-01', '2023-10-01'),
        ('2022-10-01', '2024-10-01'),
        ('2023-10-01', '2025-10-01'),
        ('2024-10-01', '2026-03-28'),
    ]
    for s, e in WINDOWS:
        t1 = time.time()
        label = f'C_win_{s[:7]}_{e[:7]}'
        print(f'  {label} ... ', end='', flush=True)
        m = run_config(data, (0, 0, 0), s, e, label)
        m['axis'] = 'C_rolling'
        m['axis_value'] = f'{s[:7]}~{e[:7]}'
        m['elapsed_s'] = round(time.time() - t1, 1)
        results.append(m)
        print(f"Cal={m.get('Cal',0):.2f} CAGR={m.get('CAGR',0):+.1%} ({m['elapsed_s']}s)")

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, 'phase0c_results.csv'), index=False)

    cols_show = ['label', 'axis', 'axis_value', 'Cal', 'CAGR', 'MDD',
                 'Sharpe', 'Rebal', 'Liq', 'elapsed_s']
    cols_present = [c for c in cols_show if c in df.columns]
    print('\n=== 전체 요약 ===')
    print(df[cols_present].to_string(index=False))

    # 축별 분산 분석
    print('\n=== 축별 분산 ===')
    for axis in ['A_common_phase', 'B_start_date', 'C_rolling']:
        sub = df[df['axis'] == axis]
        if len(sub) == 0:
            continue
        vals = sub['Cal'].values
        print(f'\n[{axis}] n={len(sub)}')
        print(f'  Cal: median={np.median(vals):.3f} mean={np.mean(vals):.3f} '
              f'std={np.std(vals):.3f} min={vals.min():.3f} max={vals.max():.3f} '
              f'spread={vals.max()-vals.min():.3f}')
        cv = np.std(vals) / max(abs(np.mean(vals)), 1e-9)
        print(f'  CV (std/mean) = {cv:.3f}  ← 0.1 이하면 plateau, 0.2 이상이면 overfit 의심')

    print(f'\n[P0c] Total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
