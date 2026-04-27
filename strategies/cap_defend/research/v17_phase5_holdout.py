"""V17 Phase-5 — untouched holdout 평가 (2024-01 ~ 2025-12).

Phase-3 top finalists 를 train (2017-04 ~ 2023-12) 에서 한 번도 안 본
holdout 구간에서 평가. 추가로 full window (2017~2025) 성과도 같이 보여줌.

Phase-3 robust_score 상위 5개 → 각각 6 phase jitter (jitter 평균 metric).
Holdout 은 2년 bull 구간 (bear 부재) — tie-break 용, 단독 채택 기준 아님.
"""
from __future__ import annotations
import os, sys, time, glob
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from stock_engine import SP, load_prices, precompute, _init, ALL_TICKERS
import stock_engine as tsi
from stock_engine_snap import run_snapshot_ensemble

PHASE3_OUT = os.path.join(HERE, "v17_phase3_robust_out")
OUT        = os.path.join(HERE, "v17_phase5_holdout_out")
os.makedirs(OUT, exist_ok=True)

UNIVERSE_B = ('SPY', 'VEA', 'EEM', 'EWJ', 'INDA', 'GLD', 'PDBC')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')

KEY_COLS = ['snap_days','canary_hyst','canary_sma','def_mom',
            'canary_type','canary_asset','canary_extra','select','health']

HOLDOUT_START = '2024-01-01'
HOLDOUT_END   = '2025-12-31'
TRAIN_START   = '2017-04-01'
TRAIN_END     = '2023-12-31'
FULL_START    = '2017-04-01'
FULL_END      = '2025-12-31'

PHASE_OFFSETS = [0, 5, 10, 15, 20, 25]


def _metrics(df):
    if df is None or len(df) < 15: return None
    v = df['Value']
    y = (v.index[-1] - v.index[0]).days / 365.25
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


def _run_jitter_avg(cfg, start, end):
    """6 phase jitter 평균 metrics."""
    cals, cagrs, mdds, shs = [], [], [], []
    for ph in PHASE_OFFSETS:
        p = _make_params(cfg, start, end)
        df = run_snapshot_ensemble(tsi._g_prices, tsi._g_ind, p,
                                    snap_days=int(cfg['snap_days']), n_snap=3,
                                    monthly_anchor_mode=False, phase_offset=ph)
        m = _metrics(df)
        if m is None: continue
        cals.append(m['Cal']); cagrs.append(m['CAGR'])
        mdds.append(m['MDD']); shs.append(m['Sharpe'])
    if not cals: return None
    return {
        'Cal_med': round(float(np.median(cals)), 3),
        'Cal_p25': round(float(np.percentile(cals,25)), 3),
        'Cal_min': round(float(np.min(cals)), 3),
        'CAGR_med': round(float(np.median(cagrs)), 2),
        'MDD_med':  round(float(np.median(mdds)), 2),
        'Sh_med':   round(float(np.median(shs)), 3),
        'n_phase':  len(cals),
    }


def load_phase3_top(top_k: int = 5) -> list:
    cands = sorted(glob.glob(os.path.join(PHASE3_OUT, 'phase3_summary_*.csv')))
    if not cands:
        print(f"ERROR: no phase3_summary_*.csv in {PHASE3_OUT}"); sys.exit(2)
    df = pd.read_csv(cands[-1])
    df = df.sort_values('robust_score', ascending=False).head(top_k)
    return df[KEY_COLS + ['robust_score','plateau_p25','loyo_p25','oos_min']].to_dict('records')


def main():
    print("Loading prices...")
    t0 = time.time()
    prices = load_prices(ALL_TICKERS, start='2014-01-01')
    ind = precompute(prices)
    _init(prices, ind)
    print(f"  ({time.time()-t0:.0f}s)")

    finalists = load_phase3_top(top_k=5)
    print(f"\nPhase-3 Top 5 → Holdout evaluation")
    for i, cfg in enumerate(finalists):
        print(f"  F{i}: robust={cfg['robust_score']:.3f} | {cfg['snap_days']}/{cfg['canary_hyst']}"
              f"/{cfg['canary_sma']}/{cfg['def_mom']}/{cfg['canary_type']}/{cfg['canary_asset']}"
              f"/{cfg['canary_extra']}/{cfg['select']}/{cfg['health']}")

    rows = []
    for i, cfg in enumerate(finalists):
        print(f"\n=== F{i} ===")
        cfg_params = {k: cfg[k] for k in KEY_COLS}

        train = _run_jitter_avg(cfg_params, TRAIN_START, TRAIN_END)
        holdout = _run_jitter_avg(cfg_params, HOLDOUT_START, HOLDOUT_END)
        full = _run_jitter_avg(cfg_params, FULL_START, FULL_END)
        print(f"  Train   (2017~2023): {train}")
        print(f"  Holdout (2024~2025): {holdout}")
        print(f"  Full    (2017~2025): {full}")
        rows.append({
            'finalist_idx': i, **cfg_params,
            'robust_score': cfg['robust_score'],
            'plateau_p25':  cfg['plateau_p25'],
            'loyo_p25':     cfg['loyo_p25'],
            'oos_min':      cfg['oos_min'],
            **{f'train_{k}': v  for k, v in (train or {}).items()},
            **{f'holdout_{k}': v for k, v in (holdout or {}).items()},
            **{f'full_{k}': v   for k, v in (full or {}).items()},
        })

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(OUT, f'phase5_{ts}.csv')
    df.to_csv(path, index=False)
    print(f"\nSaved: {path}")

    # Display
    print("\n=== Finalist Holdout Comparison ===")
    cols_disp = ['finalist_idx','snap_days','canary_hyst','canary_type','canary_asset','select','health',
                 'robust_score','train_Cal_med','holdout_Cal_med','full_Cal_med',
                 'train_CAGR_med','holdout_CAGR_med','train_MDD_med','holdout_MDD_med']
    print(df[cols_disp].to_string(index=False))

    # Recommend winner: check holdout non-regression (> 0 Cal, > -20% MDD) AND highest robust_score
    df['holdout_ok'] = (df['holdout_Cal_med'] > 0) & (df['holdout_MDD_med'] > -25)
    qualified = df[df['holdout_ok']]
    if len(qualified):
        winner = qualified.sort_values('robust_score', ascending=False).iloc[0]
        print("\n=== RECOMMENDED WINNER ===")
        for k in KEY_COLS:
            print(f"  {k:14}: {winner[k]}")
        print(f"  robust_score   : {winner['robust_score']:.3f}")
        print(f"  train_Cal_med  : {winner['train_Cal_med']:.3f}")
        print(f"  holdout_Cal_med: {winner['holdout_Cal_med']:.3f}")
        print(f"  full_Cal_med   : {winner['full_Cal_med']:.3f}")
        print(f"  full CAGR/MDD  : {winner['full_CAGR_med']:.2f}% / {winner['full_MDD_med']:.2f}%")
    else:
        print("\nNo finalist passed holdout non-regression. Manual review needed.")


if __name__ == '__main__':
    main()
