"""V17 Phase-2 — Plateau 검증.

Phase-1b top 5 후보에 대해 수치 축 ±10%(실질 ±20% 3점) 섭동 3^4=81 per 후보.
축: snap_days(3배수), canary_sma(10배수), canary_hyst(0.001), def_mom_period(10배수).
Plateau 판정: min_Cal >= 0.7 * base_Cal AND sigma_Cal / mu_Cal < 0.20.
출력: phase2_plateau.csv (전체) + phase2_winners.csv.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from stock_engine import SP, load_prices, precompute, _init, ALL_TICKERS
import stock_engine as tsi
from stock_engine_snap import run_snapshot_ensemble

from v17_phase1b import (
    OFFENSIVE_9, DEFENSIVE_5, EXTRA_CANARY, CANARY_COMBOS, _metrics,
)

OUT = os.path.join(HERE, 'v17_snap_out')
os.makedirs(OUT, exist_ok=True)


def round3(x):
    return max(3, int(round(x / 3) * 3))


def round10(x):
    return max(10, int(round(x / 10) * 10))


def round_hyst(x):
    return round(max(0.001, round(x / 0.001) * 0.001), 4)


def perturb(base_val, round_fn, factors=(0.8, 1.0, 1.2)):
    return sorted({round_fn(base_val * f) for f in factors})


def parse_canary(s):
    return tuple(s.split('+'))


def run_one(cfg):
    canary = parse_canary(cfg['canary_assets'])
    p = SP(
        offensive=OFFENSIVE_9,
        defensive=DEFENSIVE_5,
        canary_assets=canary,
        canary_sma=int(cfg['canary_sma']),
        canary_hyst=float(cfg['canary_hyst']),
        canary_type=str(cfg['canary_type']),
        select=str(cfg['select']),
        weight='ew',
        defense='top2',
        def_mom_period=int(cfg['def_mom_period']),
        health=str(cfg['health']),
        tx_cost=0.0025,
        crash='none',
        sharpe_lookback=252,
        start='2017-04-01',
        end='2025-12-31',
    )
    try:
        df = run_snapshot_ensemble(
            tsi._g_prices, tsi._g_ind, p,
            snap_days=int(cfg['snap_days']), n_snap=3,
            monthly_anchor_mode=False,
        )
        m = _metrics(df)
        if m is None:
            return None
        out = dict(cfg)
        out.update(m)
        return out
    except Exception:
        return None


def main():
    in_path = os.path.join(OUT, 'phase1b_top.csv')
    if not os.path.exists(in_path):
        raise FileNotFoundError(f'{in_path} not found. Phase-1b 먼저.')
    top = pd.read_csv(in_path).sort_values('Cal', ascending=False).head(5).reset_index(drop=True)
    print('Phase-1b top 5 (base):')
    print(top[['snap_days', 'canary_sma', 'canary_hyst', 'canary_type', 'select',
               'def_mom_period', 'health', 'canary_assets', 'Cal']].to_string(index=False))

    # 티커 로딩
    tickers = set(ALL_TICKERS)
    tickers.update(OFFENSIVE_9)
    tickers.update(DEFENSIVE_5)
    for c in CANARY_COMBOS:
        tickers.update(c)
    tickers.update(EXTRA_CANARY)
    print(f'Loading {len(tickers)} tickers...')
    prices = load_prices(sorted(tickers), start='2014-01-01')
    ind = precompute(prices)
    _init(prices, ind)

    all_rows = []
    for cand_i, base in top.iterrows():
        base_cfg = base.to_dict()
        base_cal = float(base_cfg['Cal'])
        snap_vals = perturb(int(base_cfg['snap_days']), round3)
        sma_vals = perturb(int(base_cfg['canary_sma']), round10)
        hyst_vals = perturb(float(base_cfg['canary_hyst']), round_hyst)
        defm_vals = perturb(int(base_cfg['def_mom_period']), round10)

        configs = []
        for sd, sm, hy, dm in product(snap_vals, sma_vals, hyst_vals, defm_vals):
            cfg = dict(base_cfg)
            cfg['snap_days'] = sd
            cfg['canary_sma'] = sm
            cfg['canary_hyst'] = hy
            cfg['def_mom_period'] = dm
            cfg['base_id'] = int(cand_i)
            cfg['base_Cal'] = base_cal
            configs.append(cfg)

        print(f'\nCandidate #{cand_i}: base_Cal={base_cal}, '
              f'{len(configs)} perturbations '
              f'(snap={snap_vals}, sma={sma_vals}, hyst={hyst_vals}, defm={defm_vals})')
        t0 = time.time()
        rows = Parallel(n_jobs=-1, prefer='threads')(
            delayed(run_one)(c) for c in configs
        )
        rows = [r for r in rows if r]
        print(f'  done {time.time() - t0:.0f}s, {len(rows)} ok')
        all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, 'phase2_plateau.csv'), index=False)

    # Robustness 판정
    summary_rows = []
    for bid, grp in df.groupby('base_id'):
        base_cal = float(grp['base_Cal'].iloc[0])
        cals = grp['Cal'].astype(float).values
        mu = float(np.mean(cals))
        sigma = float(np.std(cals, ddof=0))
        mn = float(np.min(cals))
        cv = sigma / mu if mu > 0 else float('inf')
        passed = (mn >= 0.7 * base_cal) and (cv < 0.20)
        base_row = grp.iloc[0].to_dict()
        summary_rows.append({
            'base_id': int(bid),
            'snap_days': int(base_row['snap_days']),  # NOTE: 섭동 첫 샘플. base 파라미터는 아래 재설정
            'canary_sma': int(base_row['canary_sma']),
            'canary_hyst': float(base_row['canary_hyst']),
            'canary_type': base_row['canary_type'],
            'select': base_row['select'],
            'def_mom_period': int(base_row['def_mom_period']),
            'health': base_row['health'],
            'canary_assets': base_row['canary_assets'],
            'base_Cal': base_cal,
            'mu_Cal': round(mu, 3),
            'sigma_Cal': round(sigma, 3),
            'min_Cal': round(mn, 3),
            'cv_Cal': round(cv, 3),
            'n_runs': int(len(cals)),
            'plateau_pass': bool(passed),
        })

    # base_id 를 phase1b_top.csv 원본 파라미터로 덮어씀 (섭동 중 첫 row 가 아닌 base)
    for i, row in top.iterrows():
        for s in summary_rows:
            if s['base_id'] == i:
                s['snap_days'] = int(row['snap_days'])
                s['canary_sma'] = int(row['canary_sma'])
                s['canary_hyst'] = float(row['canary_hyst'])
                s['canary_type'] = row['canary_type']
                s['select'] = row['select']
                s['def_mom_period'] = int(row['def_mom_period'])
                s['health'] = row['health']
                s['canary_assets'] = row['canary_assets']

    sdf = pd.DataFrame(summary_rows)
    winners = sdf[sdf['plateau_pass']].sort_values('mu_Cal', ascending=False)
    winners.to_csv(os.path.join(OUT, 'phase2_winners.csv'), index=False)
    sdf.to_csv(os.path.join(OUT, 'phase2_summary.csv'), index=False)

    print('\n=== Phase-2 Plateau summary ===')
    print(sdf.to_string(index=False))
    print(f'\nWinners ({len(winners)}): saved to phase2_winners.csv')
    print(f'저장: {OUT}/phase2_plateau.csv, phase2_summary.csv, phase2_winners.csv')


if __name__ == '__main__':
    main()
