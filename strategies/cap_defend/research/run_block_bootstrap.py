#!/usr/bin/env python3
"""Block bootstrap stress test for top candidates.

- Block size: 60 days (2 months, 단위: 월간 블록)
- N resamples: 500
- For each resample: compute Cal/CAGR/MDD on resampled daily returns
- Output: per-candidate distribution (mean/p5/p50/p95) + worst-5% MDD

Use: python3 run_block_bootstrap.py
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from phase_common import equity_metrics, atomic_write_csv
from phase4_3asset import mix_eq, _load_stock_v17, build_ensemble_full_equity

# Candidates (same spec as leverage_comparison)
CANDIDATES = [
    # label, (st, sp, fu), lev, band_mode, band_raw, spot, fut
    ('60/35/5 L3 sleeve', (0.60, 0.35, 0.05), 3.0, 'sleeve',
     'st0.180_sp0.105_fu0.015',
     'ENS_spot_k3_4b270476', 'ENS_fut_L3_k3_12652d57'),
    ('60/30/10 L3 sleeve (B)', (0.60, 0.30, 0.10), 3.0, 'sleeve',
     'st0.180_sp0.090_fu0.030',
     'ENS_spot_k3_4b270476', 'ENS_fut_L3_k3_12652d57'),
    ('60/25/15 L3 sleeve (C)', (0.60, 0.25, 0.15), 3.0, 'sleeve',
     'st0.180_sp0.075_fu0.045',
     'ENS_spot_k3_4b270476', 'ENS_fut_L3_k3_12652d57'),
    ('60/20/20 L3 sleeve (D)', (0.60, 0.20, 0.20), 3.0, 'sleeve',
     'st0.180_sp0.060_fu0.060',
     'ENS_spot_k3_4b270476', 'ENS_fut_L3_k3_fd2dfed2'),
    ('60/35/5 L4 abs15 (CURRENT)', (0.60, 0.35, 0.05), 4.0, 'abs',
     '0.15',
     'ENS_spot_k3_4b270476', None),  # fut will be looked up
]

N_BOOT = 500
BLOCK_DAYS = 60
SEED = 42

OUT_DIR = os.path.join(HERE, 'phase4_10x_robustness')


def strip_tz(s):
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s = s.copy(); s.index = s.index.tz_localize(None)
    return s


def parse_band(band_raw):
    try:
        return float(band_raw)
    except:
        parts = str(band_raw).split('_')
        band = {}
        for p in parts:
            if p.startswith('st'): band['st'] = float(p[2:])
            elif p.startswith('sp'): band['sp'] = float(p[2:])
            elif p.startswith('fu'): band['fut'] = float(p[2:])
        return band


def build_equity(label, alloc, lev, _band_mode, band_raw, spot_tag, fut_tag,
                 stock_eq, spot_top, fut_top, raw):
    st, sp, fu = alloc
    # Resolve fut_tag for CURRENT case (pick best abs15 for L4)
    if fut_tag is None:
        sub = raw[(raw['band_mode']=='abs') & (raw['fut_lev']==lev) &
                  (abs(raw['st_w']-st)<0.001) & (abs(raw['sp_w']-sp)<0.001) &
                  (raw['spot']==spot_tag)]
        sub = sub[sub['band'].astype(str).str.startswith('0.1500') | (sub['band'].astype(float)==0.15)]
        if len(sub):
            fut_tag = sub.sort_values('Cal', ascending=False).iloc[0]['fut']
        else:
            print(f'[warn] {label}: no abs15 row found')
            return None

    spot_ens = spot_top[spot_top['ensemble_tag'] == spot_tag].iloc[0]
    spot_eq = build_ensemble_full_equity(spot_ens)
    weights = {'st': st, 'sp': sp}
    if fu > 0.001:
        fut_ens = fut_top[fut_top['ensemble_tag'] == fut_tag].iloc[0]
        fut_eq = build_ensemble_full_equity(fut_ens)
        series_dict = {'st': stock_eq, 'sp': spot_eq, 'fut': fut_eq}
        weights['fut'] = fu
    else:
        series_dict = {'st': stock_eq, 'sp': spot_eq}
    band = parse_band(band_raw)
    return strip_tz(mix_eq(series_dict, weights, band))


def bootstrap_metrics(eq, n_boot, block_days, seed):
    rng = np.random.default_rng(seed)
    rets = eq.pct_change().dropna().values
    n = len(rets)
    n_blocks = (n + block_days - 1) // block_days
    cal_list, cagr_list, mdd_list = [], [], []
    for _ in range(n_boot):
        starts = rng.integers(0, n - block_days, size=n_blocks)
        blocks = np.concatenate([rets[s:s+block_days] for s in starts])[:n]
        eq_b = (1 + blocks).cumprod()
        if eq_b[-1] <= 0:
            continue
        years = len(blocks) / 252
        cagr = eq_b[-1] ** (1/years) - 1
        run_max = np.maximum.accumulate(eq_b)
        mdd = float(((eq_b - run_max) / run_max).min())
        cal = cagr / abs(mdd) if mdd < 0 else 0.0
        cal_list.append(cal)
        cagr_list.append(cagr)
        mdd_list.append(mdd)
    return {
        'cal_mean': np.mean(cal_list),
        'cal_p5': np.percentile(cal_list, 5),
        'cal_p50': np.percentile(cal_list, 50),
        'cal_p95': np.percentile(cal_list, 95),
        'cagr_mean': np.mean(cagr_list),
        'cagr_p5': np.percentile(cagr_list, 5),
        'cagr_p50': np.percentile(cagr_list, 50),
        'cagr_p95': np.percentile(cagr_list, 95),
        'mdd_mean': np.mean(mdd_list),
        'mdd_p5': np.percentile(mdd_list, 5),   # worst 5%
        'mdd_p50': np.percentile(mdd_list, 50),
        'mdd_p95': np.percentile(mdd_list, 95),
    }


def main():
    print(f'[info] Block bootstrap: N={N_BOOT}, block={BLOCK_DAYS}d')
    raw = pd.read_csv(os.path.join(HERE, 'phase4_10x', 'raw.csv'))
    spot_top = pd.read_csv(os.path.join(HERE, 'phase3_10x', 'spot_top.csv'))
    fut_top = pd.read_csv(os.path.join(HERE, 'phase3_10x', 'fut_top.csv'))
    stock_eq = _load_stock_v17()
    print('[info] stock loaded')

    rows = []
    for (label, alloc, lev, bm, band_raw, spot_tag, fut_tag) in CANDIDATES:
        print(f'  building: {label}', flush=True)
        eq = build_equity(label, alloc, lev, bm, band_raw, spot_tag, fut_tag,
                          stock_eq, spot_top, fut_top, raw)
        if eq is None:
            continue
        fp = equity_metrics(eq)
        fp_cal = fp['Cal']; fp_cagr = fp['CAGR']; fp_mdd = fp['MDD']
        print(f'    full-period: Cal={fp_cal:.2f} CAGR={fp_cagr:.1%} MDD={fp_mdd:.1%}')
        print(f'    bootstrapping {N_BOOT}x block={BLOCK_DAYS}d ...', flush=True)
        bm_stats = bootstrap_metrics(eq, N_BOOT, BLOCK_DAYS, SEED)
        row = {
            'label': label,
            'fp_Cal': round(fp_cal, 3),
            'fp_CAGR': round(fp_cagr, 4),
            'fp_MDD': round(fp_mdd, 4),
            **{k: round(v, 4) for k, v in bm_stats.items()},
        }
        rows.append(row)

    df = pd.DataFrame(rows)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, 'block_bootstrap.csv')
    atomic_write_csv(df, out_path)

    print('\n=== BLOCK BOOTSTRAP RESULTS ===')
    cols = ['label', 'fp_Cal', 'fp_CAGR', 'fp_MDD',
            'cal_p5', 'cal_p50', 'cal_p95',
            'cagr_p5', 'cagr_p50', 'cagr_p95',
            'mdd_p5', 'mdd_p50', 'mdd_p95']
    print(df[cols].to_string(index=False))
    print(f'\nSaved: {out_path}')


if __name__ == '__main__':
    main()
