"""Phase 0c 현물 버전 — V21 현물 Cal 과적합 검증.

선물 phase0c 와 동일한 3축 (공통 phase / 시작일 / rolling) 을
V21 현물 (D봉 3멤버 SMA50/100/150 EW, snap 90, TX 0.4%) 에 적용.

Entry: run_current_coin_v20_backtest.run_backtest (initial_phases 인자 추가됨)
멤버: MEMBERS (V21 정의, coin_live_engine 에서 import)
"""
from __future__ import annotations
import os, sys, time
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
ROOT = os.path.dirname(PARENT)
sys.path.insert(0, PARENT)
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, 'trade'))

from run_current_coin_v20_backtest import run_backtest

OUT = HERE

FULL_START = '2020-10-01'
FULL_END = '2026-03-31'

MEMBER_NAMES = ['D_SMA50', 'D_SMA150', 'D_SMA100']


def run_one(phase_tuple, start, end, label):
    phases = {name: k for name, k in zip(MEMBER_NAMES, phase_tuple)}
    t0 = time.time()
    res = run_backtest(start=start, end=end, initial_phases=phases)
    m = res['metrics']
    return {
        'label': label,
        'phase_M1': phase_tuple[0],
        'phase_M2': phase_tuple[1],
        'phase_M3': phase_tuple[2],
        'start': start,
        'end': end,
        'CAGR': m.get('CAGR', 0),
        'MDD': m.get('MDD', 0),
        'Sharpe': m.get('Sharpe', 0),
        'Cal': m.get('Cal', 0),
        'Rebal': res.get('rebal_count', 0),
        'Final': m.get('Final', 0),
        'elapsed_s': round(time.time() - t0, 1),
    }


def main():
    t0 = time.time()
    results = []

    # A. 공통 phase sweep
    print('=== A. 공통 phase (k,k,k) ===', flush=True)
    for k in [0, 7, 13, 19, 31, 47, 59, 83]:
        label = f'A_common_k{k:03d}'
        print(f'  {label} ...', flush=True)
        r = run_one((k, k, k), FULL_START, FULL_END, label)
        r['axis'] = 'A_common_phase'
        r['axis_value'] = k
        results.append(r)
        print(f"  Cal={r['Cal']:.3f} MDD={r['MDD']:+.1%} ({r['elapsed_s']}s)", flush=True)

    # B. 시작일 sweep
    print('\n=== B. 시작일 sweep ===', flush=True)
    STARTS = [
        ('2020-10-01', FULL_END),
        ('2020-11-01', FULL_END),
        ('2020-12-01', FULL_END),
        ('2021-01-01', FULL_END),
        ('2021-03-01', FULL_END),
        ('2021-06-01', FULL_END),
    ]
    for s, e in STARTS:
        label = f'B_start_{s[:7]}'
        print(f'  {label} ...', flush=True)
        r = run_one((0, 0, 0), s, e, label)
        r['axis'] = 'B_start_date'
        r['axis_value'] = s
        results.append(r)
        print(f"  Cal={r['Cal']:.3f} MDD={r['MDD']:+.1%} ({r['elapsed_s']}s)", flush=True)

    # C. Rolling 2년
    print('\n=== C. Rolling 2-year ===', flush=True)
    WINDOWS = [
        ('2020-10-01', '2022-10-01'),
        ('2021-10-01', '2023-10-01'),
        ('2022-10-01', '2024-10-01'),
        ('2023-10-01', '2025-10-01'),
        ('2024-10-01', FULL_END),
    ]
    for s, e in WINDOWS:
        label = f'C_win_{s[:7]}_{e[:7]}'
        print(f'  {label} ...', flush=True)
        r = run_one((0, 0, 0), s, e, label)
        r['axis'] = 'C_rolling'
        r['axis_value'] = f'{s[:7]}~{e[:7]}'
        results.append(r)
        print(f"  Cal={r['Cal']:.3f} CAGR={r['CAGR']:+.1%} ({r['elapsed_s']}s)", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, 'spot_overfit_results.csv'), index=False)

    print('\n=== 전체 ===')
    print(df[['label', 'axis_value', 'Cal', 'CAGR', 'MDD', 'Sharpe', 'Rebal']].to_string(index=False))

    print('\n=== 축별 분산 ===')
    for axis in ['A_common_phase', 'B_start_date', 'C_rolling']:
        sub = df[df['axis'] == axis]
        if len(sub) == 0:
            continue
        vals = sub['Cal'].to_numpy(dtype=float)
        cv = float(np.std(vals) / max(abs(np.mean(vals)), 1e-9))
        print(f'[{axis}] n={len(sub)} '
              f'median={np.median(vals):.3f} mean={np.mean(vals):.3f} '
              f'std={np.std(vals):.3f} '
              f'min={vals.min():.3f} max={vals.max():.3f} '
              f'CV={cv:.3f}')

    print(f'\nTotal {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
