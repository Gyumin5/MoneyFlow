"""V17 Phase-1A (v2) — canary/structure screening with proper indicator cache.

Codex AI 리뷰 반영:
  - stock_engine.precompute 에 sma{50,100,150,200,250,300} + ema{50,100,150,200,250,300} 확장 완료.
  - select='rmom126_3' 신설 (raw mom126 기준 top-3). 기존 'mom126' 은 "top 126 by wmom"
    semantic (universe 7개면 사실상 all-healthy EW) 이라 다른 family 로 분리.

스크리닝 목적: canary 구조 + 기본 family 축 확인. plateau 판단용으로
  각 config 을 6 anchor offset 에서 돌려 median Cal / p25 Cal 기록.

Grid (축/값):
  snap_days     : 30 / 60 / 90                                  (3)
  canary_sma    : 100 / 150 / 200                               (3)
  canary_hyst   : 0.010 / 0.020                                 (2)
  canary_type   : sma                                           (1)
  canary_asset  : EEM / VEA                                     (2)
  canary_extra  : none / 3asset_50                              (2)
  select        : sh3 / mom3_sh3 / rmom126_3 / comp3            (4)
  health        : none / sma200 / mom63                         (3)
  def_mom_period: 63 / 252                                      (2)

= 3×3×2×1×2×2×4×3×2 = 1728 configs × 6 anchors ≈ 10k runs.

FIX: universe=B, defense=top2, n_mom=3, tx=0.25%, sharpe_lookback=252, crash='none'.
"""
from __future__ import annotations
import os, sys, time
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

OUT = os.path.join(HERE, "v17_phase1A_v2_out")
os.makedirs(OUT, exist_ok=True)

UNIVERSE_B = ('SPY', 'VEA', 'EEM', 'EWJ', 'INDA', 'GLD', 'PDBC')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')

G_SNAP     = [30, 60, 90]
G_CSMA     = [100, 150, 200]
G_HYST     = [0.010, 0.020]
G_CTYPE    = ['sma']
G_CASSET   = [('EEM',), ('VEA',)]
G_CEXTRA   = ['none', '3asset_50']
G_SELECT   = ['sh3', 'mom3_sh3', 'rmom126_3', 'comp3']
G_HEALTH   = ['none', 'sma200', 'mom63']
G_DEFM     = [63, 252]

# 6 phase offsets — snapshot timing jitter without shifting eval window.
# snap_days={30,60,90} 에서 {0,5,10,15,20,25} 는 snap 30 에서 6 unique residues,
# snap 60 은 6 unique, snap 90 도 6 unique.
PHASE_OFFSETS = [0, 5, 10, 15, 20, 25]
START = '2017-04-01'
END = '2025-12-31'
N_ANCHOR_REQUIRED = len(PHASE_OFFSETS)  # 부분 실패 탈락


def _metrics(df):
    if df is None or len(df) < 30: return None
    v = df['Value']
    y = (v.index[-1] - v.index[0]).days / 365.25
    if y <= 0: return None
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1/y) - 1
    mdd  = (v / v.cummax() - 1).min()
    dr   = v.pct_change().dropna()
    sh   = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    cal  = cagr / abs(mdd) if mdd < 0 else 0
    return {'CAGR': cagr*100, 'MDD': mdd*100, 'Sharpe': sh, 'Cal': cal,
            'Final': v.iloc[-1], 'Rebals': df.attrs.get('rebal_count', 0)}


def _make_params(cfg):
    return SP(
        offensive=UNIVERSE_B, defensive=DEF,
        canary_assets=cfg['canary_asset'],
        canary_sma=cfg['canary_sma'],
        canary_hyst=cfg['canary_hyst'],
        canary_type=cfg['canary_type'],
        canary_extra=cfg['canary_extra'],
        select=cfg['select'], weight='ew',
        defense='top2',
        def_mom_period=cfg['def_mom'],
        health=cfg['health'],
        tx_cost=0.0025, crash='none', sharpe_lookback=252,
        start=START, end=END,
    )


def run_one(cfg):
    cals = []
    cagrs = []; mdds = []; shs = []
    err = None
    p = _make_params(cfg)
    for phase in PHASE_OFFSETS:
        try:
            df = run_snapshot_ensemble(tsi._g_prices, tsi._g_ind, p,
                                        snap_days=cfg['snap_days'], n_snap=3,
                                        monthly_anchor_mode=False,
                                        phase_offset=phase)
            m = _metrics(df)
            if m is None:
                err = f'metric-none@phase={phase}'
                break
            cals.append(m['Cal']); cagrs.append(m['CAGR'])
            mdds.append(m['MDD']); shs.append(m['Sharpe'])
        except Exception as e:
            err = f'{type(e).__name__}@phase={phase}: {str(e)[:60]}'
            break
    out = {**cfg, 'canary_asset': cfg['canary_asset'][0], 'n_anchor': len(cals)}
    if len(cals) != N_ANCHOR_REQUIRED:
        out['ERR'] = err or f'partial({len(cals)}/{N_ANCHOR_REQUIRED})'
        return out
    arr = np.array(cals)
    out.update({
        'Cal_med': round(float(np.median(arr)), 3),
        'Cal_p25': round(float(np.percentile(arr, 25)), 3),
        'Cal_min': round(float(arr.min()), 3),
        'Cal_max': round(float(arr.max()), 3),
        'Cal_std': round(float(arr.std()), 3),
        'CAGR_med': round(float(np.median(cagrs)), 2),
        'MDD_med':  round(float(np.median(mdds)), 2),
        'Sh_med':   round(float(np.median(shs)), 3),
    })
    return out


def main():
    print("Loading prices...")
    t0 = time.time()
    prices = load_prices(ALL_TICKERS, start='2014-01-01')
    ind = precompute(prices)
    _init(prices, ind)
    print(f"  ({time.time()-t0:.0f}s, {len(prices)} tickers)")

    axes = [G_SNAP, G_CSMA, G_HYST, G_CTYPE, G_CASSET, G_CEXTRA,
            G_SELECT, G_HEALTH, G_DEFM]
    keys = ['snap_days','canary_sma','canary_hyst','canary_type',
            'canary_asset','canary_extra','select','health','def_mom']
    configs = [dict(zip(keys, c)) for c in product(*axes)]
    print(f"Grid: {len(configs)} configs × {len(PHASE_OFFSETS)} phase offsets "
          f"= {len(configs)*len(PHASE_OFFSETS)} runs")

    t0 = time.time()
    # Multiprocessing (fork COW) — globals _g_prices/_g_ind 상속. loky(spawn)는 globals 손실.
    rows = Parallel(n_jobs=24, backend='multiprocessing')(
        delayed(run_one)(c) for c in configs)
    print(f"Done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    path = os.path.join(OUT, f'phase1A_v2_{ts}.csv')
    df.to_csv(path, index=False)

    err_ct = df['ERR'].notna().sum() if 'ERR' in df.columns else 0
    ok = df[df['Cal_med'].notna()] if 'Cal_med' in df.columns else df.iloc[:0]
    print(f"\nERR: {err_ct}, OK: {len(ok)}")

    if len(ok):
        cols = ['snap_days','canary_sma','canary_hyst','canary_type',
                'canary_asset','canary_extra','select','health','def_mom',
                'Cal_med','Cal_p25','Cal_std','CAGR_med','MDD_med','Sh_med']
        print("\n=== Top 10 by Cal_med ===")
        print(ok.sort_values('Cal_med', ascending=False).head(10)[cols].to_string(index=False))
        print("\n=== Top 10 by Cal_p25 (robust) ===")
        print(ok.sort_values('Cal_p25', ascending=False).head(10)[cols].to_string(index=False))
    print(f"\n저장: {path}")


if __name__ == '__main__':
    main()
