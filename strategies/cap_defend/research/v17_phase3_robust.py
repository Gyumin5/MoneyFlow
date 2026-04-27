"""V17 Phase-3 robustness screen on iter_v2 top finalists.

Codex 로드맵 기준 robustness budget per finalist:
- plateau ±1 step : snap±15%, hyst±20%, canary_sma±1(supported), def_mom±1(supported)
                    → 3×3×3×3 = 81 runs
- jitter          : phase_offset 9종 (0/3/6/9/12/15/18/21/24)            → 9 runs
- jackknife LOYO  : 2017~2023 중 1년 빼고 평가 (7 years)                  → 7 runs
- expanding OOS   : 2019/20/21/22/23 를 각 1년 walk-forward holdout       → 5 runs
  (각 year 구간만 평가)

= 102 runs × 6 finalist = 612 runs.

Holdout (2024-01 ~ 2025-12) 은 여기서 보지 않음. Phase-5 에서 별도 평가.
"""
from __future__ import annotations
import os, sys, time, glob, json
from itertools import product
from joblib import Parallel, delayed
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from stock_engine import SP, load_prices, precompute, _init, ALL_TICKERS
import stock_engine as tsi
from stock_engine_snap import run_snapshot_ensemble

ITER_OUT = os.path.join(HERE, "v17_iter_v2_out")
OUT      = os.path.join(HERE, "v17_phase3_robust_out")
os.makedirs(OUT, exist_ok=True)

UNIVERSE_B = ('SPY', 'VEA', 'EEM', 'EWJ', 'INDA', 'GLD', 'PDBC')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')

SUPPORTED_CANARY_SMA = [50, 100, 150, 200, 250, 300]
SUPPORTED_DEF_MOM    = [21, 42, 63, 126, 252]

TRAIN_START = '2017-04-01'
TRAIN_END   = '2023-12-31'

KEY_COLS = ['snap_days', 'canary_hyst', 'canary_sma', 'def_mom',
            'canary_type', 'canary_asset', 'canary_extra', 'select', 'health']


def _metrics(df, years=None):
    """LOYO 처럼 중간 구간이 빠진 경우 years 를 실제 누적 거래 기간으로 전달."""
    if df is None or len(df) < 30: return None
    v = df['Value']
    y = (v.index[-1] - v.index[0]).days / 365.25 if years is None else years
    if y <= 0: return None
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1/y) - 1
    mdd  = (v / v.cummax() - 1).min()
    dr   = v.pct_change().dropna()
    sh   = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    cal  = cagr / abs(mdd) if mdd < 0 else 0
    return {'CAGR': cagr*100, 'MDD': mdd*100, 'Sharpe': sh, 'Cal': cal}


def _make_params(cfg, start, end):
    return SP(
        offensive=UNIVERSE_B, defensive=DEF,
        canary_assets=(cfg['canary_asset'],),
        canary_sma=int(cfg['canary_sma']),
        canary_hyst=float(cfg['canary_hyst']),
        canary_type=cfg['canary_type'],
        canary_extra=cfg['canary_extra'],
        canary_band=0.0,
        select=cfg['select'], weight='ew',
        n_mom=3, n_sh=3, mom_style='default',
        defense='top2',
        def_mom_period=int(cfg['def_mom']),
        health=cfg['health'],
        tx_cost=0.0025, crash='none', sharpe_lookback=252,
        start=start, end=end,
    )


def _run_one(cfg, start, end, phase_offset=0):
    p = _make_params(cfg, start, end)
    df = run_snapshot_ensemble(tsi._g_prices, tsi._g_ind, p,
                                snap_days=int(cfg['snap_days']), n_snap=3,
                                monthly_anchor_mode=False,
                                phase_offset=int(phase_offset))
    return _metrics(df)


# ─── 1) Plateau ±1 step ────────────────────────────────────────────
def _plateau_configs(cfg):
    snap  = int(cfg['snap_days'])
    hyst  = float(cfg['canary_hyst'])
    sma   = int(cfg['canary_sma'])
    dm    = int(cfg['def_mom'])
    snap_grid = sorted(set(max(21, int(round(snap * r / 3) * 3)) for r in [0.85, 1.0, 1.15]))
    hyst_grid = sorted(set(round(max(0.005, min(0.05, hyst * r)), 3) for r in [0.8, 1.0, 1.2]))
    if sma in SUPPORTED_CANARY_SMA:
        i = SUPPORTED_CANARY_SMA.index(sma)
        sma_grid = sorted(set([SUPPORTED_CANARY_SMA[max(0, i-1)], sma,
                               SUPPORTED_CANARY_SMA[min(len(SUPPORTED_CANARY_SMA)-1, i+1)]]))
    else:
        sma_grid = [sma]
    if dm in SUPPORTED_DEF_MOM:
        i = SUPPORTED_DEF_MOM.index(dm)
        dm_grid = sorted(set([SUPPORTED_DEF_MOM[max(0, i-1)], dm,
                              SUPPORTED_DEF_MOM[min(len(SUPPORTED_DEF_MOM)-1, i+1)]]))
    else:
        dm_grid = [dm]

    variants = []
    for s, h, c_sma, c_dm in product(snap_grid, hyst_grid, sma_grid, dm_grid):
        v = dict(cfg)
        v.update({'snap_days': s, 'canary_hyst': h, 'canary_sma': c_sma, 'def_mom': c_dm})
        variants.append(v)
    return variants


def run_plateau(cfg):
    variants = _plateau_configs(cfg)
    results = []
    for v in variants:
        m = _run_one(v, TRAIN_START, TRAIN_END, phase_offset=0)
        results.append({'test': 'plateau',
                        **{k: v[k] for k in KEY_COLS}, **(m or {})})
    return results


def run_jitter(cfg):
    results = []
    for ph in [0, 3, 6, 9, 12, 15, 18, 21, 24]:
        m = _run_one(cfg, TRAIN_START, TRAIN_END, phase_offset=ph)
        results.append({'test': f'jitter_ph{ph}', 'phase_offset': ph,
                        **{k: cfg[k] for k in KEY_COLS}, **(m or {})})
    return results


def run_loyo(cfg):
    """7 years (2017~2023) 중 한 해씩 제거 후 2개 sub-interval 각각 평가,
    equity 를 이어붙여 combined metric."""
    years = list(range(2017, 2024))
    results = []
    for skip_y in years:
        parts = []
        if skip_y > 2017:
            parts.append((TRAIN_START, f'{skip_y-1}-12-31'))
        if skip_y < 2023:
            parts.append((f'{skip_y+1}-01-01', TRAIN_END))
        equities = []
        for start, end in parts:
            p = _make_params(cfg, start, end)
            df = run_snapshot_ensemble(tsi._g_prices, tsi._g_ind, p,
                                        snap_days=int(cfg['snap_days']), n_snap=3,
                                        monthly_anchor_mode=False, phase_offset=0)
            if df is not None:
                equities.append(df['Value'])
        if equities:
            combined = []
            base = 1.0
            total_days = 0
            for idx_eq, eq in enumerate(equities):
                rel = eq / eq.iloc[0]
                new_part = rel * base
                if combined:  # 2번째 이후 파트는 seam 첫날 제외 (0% 인위 일 방지)
                    new_part = new_part.iloc[1:]
                combined.append(new_part)
                if len(new_part):
                    base = new_part.iloc[-1]
                total_days += (eq.index[-1] - eq.index[0]).days
            eq_all = pd.concat(combined)
            eq_all = eq_all[~eq_all.index.duplicated(keep='first')]
            # 실제 거래 기간 (skip 연도 제외) 으로 CAGR 계산
            m = _metrics(pd.DataFrame({'Value': eq_all}),
                         years=total_days / 365.25 if total_days > 0 else None)
        else:
            m = None
        results.append({'test': f'loyo_skip{skip_y}', 'skip_year': skip_y,
                        **{k: cfg[k] for k in KEY_COLS}, **(m or {})})
    return results


def run_expanding_oos(cfg):
    """2019~2023 각 해를 1년 holdout 으로 평가."""
    years = [2019, 2020, 2021, 2022, 2023]
    results = []
    for y in years:
        m = _run_one(cfg, f'{y}-01-01', f'{y}-12-31', phase_offset=0)
        results.append({'test': f'oos_{y}', 'year': y,
                        **{k: cfg[k] for k in KEY_COLS}, **(m or {})})
    return results


def load_finalists(top_k: int = 6) -> list:
    """iter_v2_out 최근 all_rounds.csv 에서 중복 제거 top-K."""
    cands = sorted(glob.glob(os.path.join(ITER_OUT, 'run_*_all_rounds.csv')))
    if not cands:
        print(f"ERROR: no all_rounds.csv in {ITER_OUT}")
        sys.exit(2)
    df = pd.read_csv(cands[-1])
    ok = df[df['Cal_p25'].notna()].copy()
    ok['sig'] = ok[KEY_COLS].astype(str).agg('|'.join, axis=1)
    ok = ok.sort_values('Cal_p25', ascending=False).drop_duplicates('sig')
    top = ok.head(top_k)
    return top[KEY_COLS].to_dict('records')


def run_finalist(finalist_idx: int, cfg: dict) -> list:
    t0 = time.time()
    all_rows = []
    for fn, name in [(run_plateau,'plateau'), (run_jitter,'jitter'),
                     (run_loyo,'loyo'), (run_expanding_oos,'oos')]:
        t1 = time.time()
        rows = fn(cfg)
        for r in rows:
            r['finalist_idx'] = finalist_idx
        all_rows.extend(rows)
        print(f"  [F{finalist_idx}] {name}: {len(rows)} runs ({time.time()-t1:.0f}s)")
    print(f"  [F{finalist_idx}] total {time.time()-t0:.0f}s")
    return all_rows


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--top_k', type=int, default=6)
    args = ap.parse_args()

    print("Loading prices...")
    t0 = time.time()
    prices = load_prices(ALL_TICKERS, start='2014-01-01')
    ind = precompute(prices)
    _init(prices, ind)
    print(f"  ({time.time()-t0:.0f}s)")

    finalists = load_finalists(top_k=args.top_k)
    print(f"\nFinalists (top {args.top_k} by Cal_p25):")
    for i, cfg in enumerate(finalists):
        print(f"  F{i}: {cfg}")

    # Parallel over finalists (6 * 102 runs sequentially per finalist inside).
    # 각 finalist 는 LOYO 의 equity stitching 때문에 serial 이 편함. 병렬은 finalist 간.
    all_results = Parallel(n_jobs=min(6, len(finalists)), backend='multiprocessing', verbose=1)(
        delayed(run_finalist)(i, cfg) for i, cfg in enumerate(finalists))
    all_rows = [r for rows in all_results for r in rows]

    df = pd.DataFrame(all_rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(OUT, f'phase3_{ts}.csv')
    df.to_csv(path, index=False)
    print(f"\nSaved: {path} ({len(df)} rows)")

    # Summary
    print("\n=== Finalist Robustness Summary ===")
    header = "  Fidx test     n  med     p25     min     max     score"
    print(header)
    summary_rows = []
    for i, cfg in enumerate(finalists):
        sub = df[df['finalist_idx'] == i]
        fidx_sum = {'finalist_idx': i, **{k: cfg[k] for k in KEY_COLS}}
        for test_prefix in ['plateau', 'jitter', 'loyo', 'oos']:
            s = sub[sub['test'].str.startswith(test_prefix)]
            if 'Cal' not in s.columns: continue
            cals = s['Cal'].dropna()
            if len(cals) == 0: continue
            print(f"  F{i}   {test_prefix:8} {len(cals):2}  {cals.median():.3f}  "
                  f"{np.percentile(cals,25):.3f}  {cals.min():.3f}  {cals.max():.3f}")
            fidx_sum[f'{test_prefix}_med'] = round(float(cals.median()), 3)
            fidx_sum[f'{test_prefix}_p25'] = round(float(np.percentile(cals, 25)), 3)
            fidx_sum[f'{test_prefix}_min'] = round(float(cals.min()), 3)
        # Composite robustness score: plateau p25 + loyo p25 + oos min (weighted)
        ps = fidx_sum.get('plateau_p25', 0); ls = fidx_sum.get('loyo_p25', 0)
        om = fidx_sum.get('oos_min', 0)
        fidx_sum['robust_score'] = round(0.4*ps + 0.3*ls + 0.3*om, 3)
        summary_rows.append(fidx_sum)

    summary = pd.DataFrame(summary_rows).sort_values('robust_score', ascending=False)
    summary_path = os.path.join(OUT, f'phase3_summary_{ts}.csv')
    summary.to_csv(summary_path, index=False)
    print(f"\nSummary: {summary_path}")
    print("\n=== Top finalists by robust_score (0.4 plateau_p25 + 0.3 loyo_p25 + 0.3 oos_min) ===")
    disp = ['finalist_idx'] + KEY_COLS + ['plateau_p25','loyo_p25','oos_min','robust_score']
    print(summary[disp].to_string(index=False))


if __name__ == '__main__':
    main()
