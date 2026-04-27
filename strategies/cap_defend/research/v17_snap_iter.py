"""V17 snapshot iterative refine — 자동 zoom 반복.

각 iteration:
  1. 현재 grid 로 전 configs 실행
  2. Top 2 peaks 식별 (Cal 기준 + Sharpe 기준)
  3. 각 peak 주변 축별 zoom 추가:
     - 외곽 1.2배
     - 사이 기하평균
     - 3의 배수 (snap_days) or 10의 배수 (canary_sma, def_mom) or 0.001 단위 (hyst)
     - 1.2배 차이 미만이면 수렴 (추가 X)
  4. 중복 제거 후 다음 iteration 실행
  5. 모든 축이 수렴 또는 max_iter 도달 시 종료

출력: v17_snap_out/iter_N.csv + top_peaks.csv
"""
from __future__ import annotations
import os, sys, time
import math
from itertools import product
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from stock_engine import SP, load_prices, precompute, _init, ALL_TICKERS
import stock_engine as tsi
from stock_engine_snap import run_snapshot_ensemble

OUT = os.path.join(HERE, "v17_snap_out")
os.makedirs(OUT, exist_ok=True)

UNIVERSE_B = ('SPY', 'VEA', 'EEM', 'EWJ', 'INDA', 'GLD', 'PDBC')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')

# 축별 rounding rule
def round3(x): return max(3, int(round(x / 3) * 3))
def round10(x): return max(10, int(round(x / 10) * 10))
def round_hyst(x): return round(max(0.001, x / 0.001) * 0.001, 4)


def zoom_numeric(vals, peaks, round_fn):
    """수치 축 zoom: vals=현재 grid, peaks=top 2 peak 값.
    각 peak 주변 외곽 1.2배 + 사이 기하평균 추가. 1.2배 차이 미만이면 skip.
    """
    vals_sorted = sorted(set(vals))
    new = set(vals_sorted)
    for peak in peaks:
        idx = vals_sorted.index(peak) if peak in vals_sorted else None
        if idx is None: continue
        # Lower neighbor
        if idx > 0:
            lower = vals_sorted[idx - 1]
            if peak / lower >= 1.2:
                mid = round_fn(math.sqrt(peak * lower))
                if mid != peak and mid != lower: new.add(mid)
        else:
            outer = round_fn(peak / 1.2)
            if outer != peak and outer > 0: new.add(outer)
        # Upper neighbor
        if idx + 1 < len(vals_sorted):
            upper = vals_sorted[idx + 1]
            if upper / peak >= 1.2:
                mid = round_fn(math.sqrt(peak * upper))
                if mid != peak and mid != upper: new.add(mid)
        else:
            outer = round_fn(peak * 1.2)
            if outer != peak: new.add(outer)
    return sorted(new)


def _metrics(df):
    if df is None or len(df) < 30: return None
    v = df['Value']
    y = (v.index[-1] - v.index[0]).days / 365.25
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1/y) - 1
    mdd = (v / v.cummax() - 1).min()
    dr = v.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {'CAGR': round(cagr*100, 2), 'MDD': round(mdd*100, 2),
            'Sharpe': round(sh, 3), 'Cal': round(cal, 3),
            'Final': round(v.iloc[-1], 2),
            'Rebals': df.attrs.get('rebal_count', 0)}


def run_one(snap_days, csma, hyst, ctype, select, defm, health):
    p = SP(offensive=UNIVERSE_B, defensive=DEF, canary_assets=('EEM',),
           canary_sma=csma, canary_hyst=hyst, canary_type=ctype,
           select=select, weight='ew', defense='top2',
           def_mom_period=defm, health=health, tx_cost=0.0025,
           crash='none', sharpe_lookback=252,
           start='2017-04-01', end='2025-12-31')
    try:
        df = run_snapshot_ensemble(tsi._g_prices, tsi._g_ind, p,
                                     snap_days=snap_days, n_snap=3,
                                     monthly_anchor_mode=False)
        m = _metrics(df)
        if m is None: return None
        return {'snap_days':snap_days, 'canary_sma':csma, 'canary_hyst':hyst,
                'canary_type':ctype, 'select':select, 'def_mom_period':defm,
                'health':health, **m}
    except Exception:
        return None


def run_grid(grid):
    configs = list(product(*grid.values()))
    rows = Parallel(n_jobs=24, prefer='threads')(
        delayed(run_one)(*c) for c in configs)
    return pd.DataFrame([r for r in rows if r])


def main(max_iter=5):
    print("Loading prices...")
    prices = load_prices(ALL_TICKERS, start='2014-01-01')
    ind = precompute(prices)
    _init(prices, ind)

    # 초기 grid (사용자 확정)
    grid = {
        'snap_days':      [30, 90, 180],
        'canary_sma':     [50, 150, 300],
        'canary_hyst':    [0.005, 0.020],
        'canary_type':    ['sma'],          # ema는 Phase-1a에서 dominant 아니었음
        'select':         ['sh3', 'mom126', 'mom3_sh3'],
        'def_mom_period': [63, 252],
        'health':         ['none', 'sma200', 'mom126'],
    }

    all_results = []
    converged_axes = set()

    for it in range(1, max_iter + 1):
        print(f"\n=== Iteration {it} ===")
        for k, v in grid.items():
            print(f"  {k}: {v}")
        n = 1
        for v in grid.values(): n *= len(v)
        print(f"  configs: {n}")
        t0 = time.time()
        df = run_grid(grid)
        print(f"  완료 ({time.time()-t0:.0f}s, {len(df)} rows)")
        df['iter'] = it
        all_results.append(df)
        df.to_csv(os.path.join(OUT, f'iter_{it}.csv'), index=False)

        # Top peaks
        top_cal = df.sort_values('Cal', ascending=False).head(5)
        top_sh = df.sort_values('Sharpe', ascending=False).head(5)
        print("\n  Top 5 by Cal:")
        print(top_cal[['snap_days','canary_sma','canary_hyst','select','def_mom_period','health','CAGR','MDD','Cal']].to_string(index=False))

        # Peak 값 추출 (top 2 configs)
        peaks = {k: [] for k in grid.keys()}
        for _, row in top_cal.head(2).iterrows():
            for k in grid.keys():
                v = row[k]
                if v not in peaks[k]:
                    peaks[k].append(v)

        # Zoom: 수치 축
        old_grid = {k: list(v) for k, v in grid.items()}
        # snap_days (3배수)
        if 'snap_days' not in converged_axes:
            new_vals = zoom_numeric(grid['snap_days'], peaks['snap_days'], round3)
            if set(new_vals) == set(grid['snap_days']):
                converged_axes.add('snap_days')
                print('  snap_days 수렴')
            else:
                grid['snap_days'] = new_vals
        # canary_sma (10배수)
        if 'canary_sma' not in converged_axes:
            new_vals = zoom_numeric(grid['canary_sma'], peaks['canary_sma'], round10)
            if set(new_vals) == set(grid['canary_sma']):
                converged_axes.add('canary_sma')
                print('  canary_sma 수렴')
            else:
                grid['canary_sma'] = new_vals
        # canary_hyst (0.001 단위)
        if 'canary_hyst' not in converged_axes:
            new_vals = zoom_numeric(grid['canary_hyst'], peaks['canary_hyst'], round_hyst)
            if set(new_vals) == set(grid['canary_hyst']):
                converged_axes.add('canary_hyst')
                print('  canary_hyst 수렴')
            else:
                grid['canary_hyst'] = new_vals
        # def_mom_period (10배수)
        if 'def_mom_period' not in converged_axes:
            new_vals = zoom_numeric(grid['def_mom_period'], peaks['def_mom_period'], round10)
            if set(new_vals) == set(grid['def_mom_period']):
                converged_axes.add('def_mom_period')
                print('  def_mom_period 수렴')
            else:
                grid['def_mom_period'] = new_vals

        # 카테고리 축 (select, canary_type, health) 은 그대로 유지 — top 만 남기기
        # 단 iter 3 이상이면 카테고리 축도 top 2 만 남김 (grid 축소)
        if it >= 3:
            top_cats = set()
            for _, r in top_cal.head(3).iterrows():
                top_cats.add(r['select'])
            if len(top_cats) < len(grid['select']):
                grid['select'] = list(top_cats)
                print(f'  select 축소: {grid["select"]}')
            top_health = set()
            for _, r in top_cal.head(3).iterrows():
                top_health.add(r['health'])
            if len(top_health) < len(grid['health']):
                grid['health'] = list(top_health)
                print(f'  health 축소: {grid["health"]}')

        # 수렴 체크 전체
        all_numeric_converged = all(ax in converged_axes for ax in
                                      ['snap_days','canary_sma','canary_hyst','def_mom_period'])
        if all_numeric_converged:
            print("\n모든 수치 축 수렴. 종료.")
            break

    # 최종 결과
    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(os.path.join(OUT, 'all_iters.csv'), index=False)
    top_final = combined.drop_duplicates(subset=['snap_days','canary_sma','canary_hyst',
                                                    'canary_type','select','def_mom_period','health']
                                            ).sort_values('Cal', ascending=False).head(20)
    top_final.to_csv(os.path.join(OUT, 'top_peaks.csv'), index=False)
    print("\n=== Final Top 20 ===")
    print(top_final[['iter','snap_days','canary_sma','canary_hyst','select','def_mom_period',
                     'health','CAGR','MDD','Sharpe','Cal']].to_string(index=False))
    print(f"\n저장: {OUT}/all_iters.csv, top_peaks.csv")


if __name__ == '__main__':
    main(max_iter=6)
