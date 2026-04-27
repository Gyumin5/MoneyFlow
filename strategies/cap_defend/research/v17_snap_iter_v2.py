"""V17 주식 iter_refine v2 — V17 실운영 universe 기준 (SPY/QQQ/VEA/EEM/GLD/PDBC/VNQ).

v17_snap_iter.py 의 UNIVERSE_B (EWJ/INDA 포함) 대체. 플랜 v3 기준.
출력: v17_snap_v2_out/iter_N.csv + all_iters.csv

변경점 v1 → v2
- offensive: SPY/QQQ/VEA/EEM/GLD/PDBC/VNQ (recommend.py 일치)
- defensive: IEF/BIL/BNDX/GLD/PDBC (변동 없음)
- canary: EEM (변동 없음)
- n_snap=3 (V21 스타일)
- tx_cost=0.0025 (KIS 실수수료)
- 기간 2017-04-01 ~ 2025-12-31
"""
from __future__ import annotations
import math
import os
import sys
import time
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

OUT = os.path.join(HERE, "v17_snap_v2_out")
os.makedirs(OUT, exist_ok=True)

# V17 실운영 universe (recommend.py 기준 2026-04-24)
UNIVERSE_V17 = ('SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC', 'VNQ')
DEFENSE_V17 = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')
CANARY_V17 = ('EEM',)


def round3(x): return max(3, int(round(x / 3) * 3))
def round10(x): return max(10, int(round(x / 10) * 10))
def round_hyst(x): return round(max(0.001, x / 0.001) * 0.001, 4)


MAX_AXIS_LEN = 7  # 축당 최대 값 개수 (grid 폭증 방지)


def zoom_numeric(vals, peaks, round_fn):
    """interior gap 만 fill. boundary 확장 금지 (min/max 고정).
    축 길이 MAX_AXIS_LEN 초과 시 추가 안함.
    """
    vals_sorted = sorted(set(vals))
    new = set(vals_sorted)
    for peak in peaks:
        if len(new) >= MAX_AXIS_LEN:
            break
        idx = vals_sorted.index(peak) if peak in vals_sorted else None
        if idx is None:
            continue
        # lower 보간 (interior 만)
        if idx > 0:
            lower = vals_sorted[idx - 1]
            if peak / lower >= 1.2 and len(new) < MAX_AXIS_LEN:
                mid = round_fn(math.sqrt(peak * lower))
                if mid != peak and mid != lower:
                    new.add(mid)
        # upper 보간 (interior 만)
        if idx + 1 < len(vals_sorted):
            upper = vals_sorted[idx + 1]
            if upper / peak >= 1.2 and len(new) < MAX_AXIS_LEN:
                mid = round_fn(math.sqrt(peak * upper))
                if mid != peak and mid != upper:
                    new.add(mid)
        # boundary 확장 (idx==0 or idx==last) 제거 — min/max 고정
    return sorted(new)


def _metrics(df):
    if df is None or len(df) < 30:
        return None
    v = df['Value']
    y = (v.index[-1] - v.index[0]).days / 365.25
    if y <= 0 or v.iloc[-1] <= 0:
        return None
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / y) - 1
    mdd = (v / v.cummax() - 1).min()
    dr = v.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {'CAGR': round(cagr * 100, 2), 'MDD': round(mdd * 100, 2),
            'Sharpe': round(sh, 3), 'Cal': round(cal, 3),
            'Final': round(v.iloc[-1], 2),
            'Rebals': df.attrs.get('rebal_count', 0)}


NPICK = 3  # 종목 수 고정 (사용자 확정)


def select_from_family(family):
    """family ∈ {mom_sh, zscore, comp, comp_sort, sh, mom} → select 문자열."""
    if family == 'mom_sh':
        return f'mom{NPICK}_sh{NPICK}'
    return f'{family}{NPICK}'


_WORKER_LOADED = False


def _ensure_worker_loaded():
    global _WORKER_LOADED
    if _WORKER_LOADED:
        return
    if tsi._g_prices is None or tsi._g_ind is None:
        prices = load_prices(ALL_TICKERS, start='2014-01-01')
        ind = precompute(prices)
        _init(prices, ind)
    _WORKER_LOADED = True


def run_one(snap_days, csma, hyst, ctype, select_family, defm, health,
            sharpe_lb, mom_style):
    _ensure_worker_loaded()
    sel = select_from_family(select_family)
    p = SP(offensive=UNIVERSE_V17, defensive=DEFENSE_V17, canary_assets=CANARY_V17,
           canary_sma=csma, canary_hyst=hyst, canary_type=ctype,
           select=sel, weight='ew', defense='top2',
           def_mom_period=defm, health=health, tx_cost=0.0025,
           crash='none', sharpe_lookback=sharpe_lb, mom_style=mom_style,
           n_mom=NPICK, n_sh=NPICK,
           start='2017-04-01', end='2025-12-31')
    try:
        df = run_snapshot_ensemble(tsi._g_prices, tsi._g_ind, p,
                                    snap_days=snap_days, n_snap=3,
                                    monthly_anchor_mode=False)
        m = _metrics(df)
        if m is None:
            return None
        tag = (f"stk_sn{snap_days}_sma{csma}_h{hyst:.3f}_{ctype}_{select_family}"
               f"_dm{defm}_{health}_sh{sharpe_lb}_mst{mom_style}")
        return {'tag': tag,
                'asset': 'stock', 'iv': 'D',
                'snap': snap_days, 'snap_days': snap_days,
                'canary_sma': csma, 'canary_hyst': hyst,
                'canary_type': ctype, 'select': sel,
                'select_family': select_family,
                'def_mom_period': defm, 'health': health,
                'sharpe_lookback': sharpe_lb, 'mom_style': mom_style,
                'n_pick': NPICK, **m}
    except Exception as e:
        return {'tag': 'err', 'error': str(e)[:100]}


def cfg_to_tag(cfg_tuple, grid_keys):
    """grid key 순서대로 tag 생성 (run_one 과 일치)."""
    d = dict(zip(grid_keys, cfg_tuple))
    return (f"stk_sn{d['snap_days']}_sma{d['canary_sma']}_h{d['canary_hyst']:.3f}"
            f"_{d['canary_type']}_{d['select_family']}_dm{d['def_mom_period']}"
            f"_{d['health']}_sh{d['sharpe_lookback']}_mst{d['mom_style']}")


def run_grid(grid, skip_tags=None):
    """skip_tags 에 있는 tag 는 재계산 스킵 (cross-iter dedup)."""
    skip = skip_tags or set()
    keys = list(grid.keys())
    all_configs = list(product(*grid.values()))
    configs = [c for c in all_configs if cfg_to_tag(c, keys) not in skip]
    n_skip = len(all_configs) - len(configs)
    if n_skip:
        print(f"  [dedup] {n_skip} configs already computed, skipping", flush=True)
    if not configs:
        return pd.DataFrame()
    rows = Parallel(n_jobs=24, backend='loky')(
        delayed(run_one)(*c) for c in configs)
    return pd.DataFrame([r for r in rows if r and 'error' not in r])


def main(max_iter=6):
    print("Loading prices (V17 실운영 universe)...")
    prices = load_prices(ALL_TICKERS, start='2014-01-01')
    ind = precompute(prices)
    _init(prices, ind)

    grid = {
        'snap_days':       [30, 90, 180],
        'canary_sma':      [50, 150, 300],
        'canary_hyst':     [0.005, 0.020],
        'canary_type':     ['sma'],
        'select_family':   ['mom_sh', 'zscore', 'comp', 'comp_sort', 'sh', 'mom'],
        'def_mom_period':  [63, 252],
        'health':          ['none', 'sma200', 'mom126'],
        'sharpe_lookback': [63, 126, 252],
        'mom_style':       ['default', 'eq', 'dual'],
    }

    # Resume: 기존 iter_*.csv 있으면 skip + grid 재구성
    all_results = []
    start_it = 1
    for it in range(1, max_iter + 1):
        p = os.path.join(OUT, f'iter_{it}.csv')
        if not os.path.exists(p):
            break
        df = pd.read_csv(p)
        if df.empty:
            break
        all_results.append(df)
        start_it = it + 1
        # zoom 재실행해서 다음 iter 의 grid 복원
        top_cal = df.sort_values('Cal', ascending=False).head(5)
        peaks = {k: [] for k in grid.keys()}
        for _, row in top_cal.head(2).iterrows():
            for k in grid.keys():
                v = row[k]
                if v not in peaks[k]:
                    peaks[k].append(v)
        grid['snap_days'] = zoom_numeric(grid['snap_days'], peaks['snap_days'], round3)
        grid['canary_sma'] = zoom_numeric(grid['canary_sma'], peaks['canary_sma'], round10)
        grid['def_mom_period'] = zoom_numeric(grid['def_mom_period'], peaks['def_mom_period'], round10)
        grid['canary_hyst'] = zoom_numeric(grid['canary_hyst'], peaks['canary_hyst'], round_hyst)
        grid['sharpe_lookback'] = zoom_numeric(grid['sharpe_lookback'], peaks['sharpe_lookback'], round10)
        # categorical iter 4+ narrow (top 2 peaks 유지, 3 이상 보존)
        if it >= 4:
            for cat_k in ['canary_type', 'select_family', 'health', 'mom_style']:
                if peaks[cat_k]:
                    # top 3 까지는 남김 (codex 지적: 조기 수렴 방지)
                    grid[cat_k] = peaks[cat_k][:3] if len(peaks[cat_k]) > 3 else peaks[cat_k]
        print(f"  [resume] iter {it} loaded ({len(df)} rows)")

    for it in range(start_it, max_iter + 1):
        print(f"\n=== Iteration {it} ===")
        for k, v in grid.items():
            print(f"  {k}: {v}")
        n = 1
        for v in grid.values():
            n *= len(v)
        print(f"  configs: {n}")
        t0 = time.time()
        # cross-iter dedup: 이전 iter 에서 계산된 tag 는 skip
        seen_tags = set()
        for prev in all_results:
            if 'tag' in prev.columns:
                seen_tags.update(prev['tag'].astype(str).tolist())
        df = run_grid(grid, skip_tags=seen_tags)
        print(f"  완료 ({time.time()-t0:.0f}s, {len(df)} new rows)")
        df['iter'] = it
        all_results.append(df)
        df.to_csv(os.path.join(OUT, f'iter_{it}.csv'), index=False)

        if df.empty:
            print("  빈 결과 (모두 기존 iter 에서 계산됨). 중단.")
            break

        # Top 5 peak 기반 zoom (codex: top 2 는 조기수렴 위험)
        # 누적 결과 기준으로 peak 찾기
        combined = pd.concat(all_results, ignore_index=True).drop_duplicates(subset=['tag'])
        top_cal = combined.sort_values('Cal', ascending=False).head(5)
        print("\n  Top 5 by Cal (누적):")
        disp_cols = ['snap_days', 'canary_sma', 'canary_hyst', 'select_family',
                     'def_mom_period', 'health', 'sharpe_lookback', 'mom_style',
                     'CAGR', 'MDD', 'Cal']
        print(top_cal[[c for c in disp_cols if c in top_cal.columns]].to_string(index=False))

        # Zoom peaks (top 5 수집)
        peaks = {k: [] for k in grid.keys()}
        for _, row in top_cal.iterrows():
            for k in grid.keys():
                v = row[k]
                if v not in peaks[k]:
                    peaks[k].append(v)

        old_grid = {k: list(v) for k, v in grid.items()}
        grid['snap_days'] = zoom_numeric(grid['snap_days'], peaks['snap_days'], round3)
        grid['canary_sma'] = zoom_numeric(grid['canary_sma'], peaks['canary_sma'], round10)
        grid['def_mom_period'] = zoom_numeric(grid['def_mom_period'], peaks['def_mom_period'], round10)
        grid['canary_hyst'] = zoom_numeric(grid['canary_hyst'], peaks['canary_hyst'], round_hyst)
        grid['sharpe_lookback'] = zoom_numeric(grid['sharpe_lookback'], peaks['sharpe_lookback'], round10)

        # categorical narrow iter 4+ (codex: iter 3 은 너무 이름)
        if it >= 4:
            for cat_k in ['canary_type', 'select_family', 'health', 'mom_style']:
                if peaks[cat_k]:
                    grid[cat_k] = peaks[cat_k][:3] if len(peaks[cat_k]) > 3 else peaks[cat_k]

        # 수렴 check
        changed = any(old_grid[k] != grid[k] for k in grid.keys())
        if not changed:
            print(f"\n  수렴. Iter {it} 에서 종료.")
            break

    all_df = pd.concat(all_results, ignore_index=True)
    all_df.to_csv(os.path.join(OUT, 'all_iters.csv'), index=False)
    # raw_combined.csv 로도 저장 (extract_top500 연동용)
    all_df.to_csv(os.path.join(OUT, 'raw_combined.csv'), index=False)


if __name__ == '__main__':
    main(max_iter=6)
