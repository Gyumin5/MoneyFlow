"""V17 Phase-1b — Canary asset 조합 sweep.

Phase-1a (v17_snap_iter.py) 의 top_peaks.csv 에서 Cal top-5 수치 config 를 가져와,
canary_assets 조합 9종 (단일 4 + AND2 4 + AND3 1) 을 sweep.
유니버스는 공격 9종 (SPY,QQQ,VEA,EEM,EWJ,INDA,GLD,PDBC,VNQ), 방어 5종 유지.
출력: v17_snap_out/phase1b.csv + phase1b_top.csv.
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

OUT = os.path.join(HERE, 'v17_snap_out')
os.makedirs(OUT, exist_ok=True)

# Phase-1b 전용 유니버스 (공격 9종 / 방어 5종 유지)
OFFENSIVE_9 = ('SPY', 'QQQ', 'VEA', 'EEM', 'EWJ', 'INDA', 'GLD', 'PDBC', 'VNQ')
DEFENSIVE_5 = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')

# Canary 조합 9종
CANARY_COMBOS = [
    ('EEM',),
    ('VEA',),
    ('VT',),
    ('ACWX',),
    ('VT', 'EEM'),
    ('EEM', 'VEA'),
    ('VT', 'VEA'),
    ('EEM', 'ACWX'),
    ('VT', 'EEM', 'VEA'),
]

# canary 에 필요한 추가 티커 (ALL_TICKERS 에 없을 가능성 대비)
EXTRA_CANARY = ('VT', 'ACWX')


def _metrics(df):
    if df is None or len(df) < 30:
        return None
    v = df['Value']
    y = (v.index[-1] - v.index[0]).days / 365.25
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / y) - 1
    mdd = (v / v.cummax() - 1).min()
    dr = v.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {'CAGR': round(cagr * 100, 2), 'MDD': round(mdd * 100, 2),
            'Sharpe': round(sh, 3), 'Cal': round(cal, 3),
            'Final': round(v.iloc[-1], 2),
            'Rebals': df.attrs.get('rebal_count', 0)}


def run_one(cfg, canary_tuple):
    p = SP(
        offensive=OFFENSIVE_9,
        defensive=DEFENSIVE_5,
        canary_assets=tuple(canary_tuple),
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
        return {
            'snap_days': int(cfg['snap_days']),
            'canary_sma': int(cfg['canary_sma']),
            'canary_hyst': float(cfg['canary_hyst']),
            'canary_type': str(cfg['canary_type']),
            'select': str(cfg['select']),
            'def_mom_period': int(cfg['def_mom_period']),
            'health': str(cfg['health']),
            'canary_assets': '+'.join(canary_tuple),
            **m,
        }
    except Exception:
        return None


def main():
    top_peaks_path = os.path.join(OUT, 'top_peaks.csv')
    if not os.path.exists(top_peaks_path):
        raise FileNotFoundError(f'{top_peaks_path} not found. Phase-1a 먼저 실행.')

    top = pd.read_csv(top_peaks_path)
    # 4-way union: Cal top20, CAGR top20, Cal*CAGR top20, ranksum(Cal+CAGR) top20
    t = top.copy()
    t['CalCAGR'] = t['Cal'] * t['CAGR']
    t['rank_Cal'] = t['Cal'].rank(ascending=False, method='min')
    t['rank_CAGR'] = t['CAGR'].rank(ascending=False, method='min')
    t['ranksum'] = t['rank_Cal'] + t['rank_CAGR']

    idx_cal = set(t.nlargest(20, 'Cal').index)
    idx_cagr = set(t.nlargest(20, 'CAGR').index)
    idx_calcagr = set(t.nlargest(20, 'CalCAGR').index)
    idx_ranksum = set(t.nsmallest(20, 'ranksum').index)
    union_idx = sorted(idx_cal | idx_cagr | idx_calcagr | idx_ranksum)

    top5 = t.loc[union_idx].sort_values('Cal', ascending=False).reset_index(drop=True)
    print(f'Phase-1a union pool: Cal20 + CAGR20 + Cal*CAGR20 + RankSum20 = {len(top5)} uniq')
    print(top5[['snap_days', 'canary_sma', 'canary_hyst', 'canary_type',
                'select', 'def_mom_period', 'health', 'CAGR', 'MDD', 'Cal']]
          .to_string(index=False))

    # 필요한 티커 로딩 (ALL_TICKERS + 공격 9 + 방어 5 + canary 조합)
    tickers = set(ALL_TICKERS)
    tickers.update(OFFENSIVE_9)
    tickers.update(DEFENSIVE_5)
    for c in CANARY_COMBOS:
        tickers.update(c)
    tickers.update(EXTRA_CANARY)

    print(f'\nLoading prices for {len(tickers)} tickers...')
    prices = load_prices(sorted(tickers), start='2014-01-01')
    ind = precompute(prices)
    _init(prices, ind)

    # 45 configs
    configs = []
    for _, row in top5.iterrows():
        cfg = row.to_dict()
        for canary in CANARY_COMBOS:
            configs.append((cfg, canary))
    print(f'Total configs: {len(configs)}')

    t0 = time.time()
    rows = Parallel(n_jobs=-1, prefer='threads')(
        delayed(run_one)(cfg, canary) for cfg, canary in configs
    )
    df = pd.DataFrame([r for r in rows if r])
    print(f'Done in {time.time() - t0:.0f}s, {len(df)} rows')

    df.to_csv(os.path.join(OUT, 'phase1b.csv'), index=False)
    top_df = df.sort_values('Cal', ascending=False).head(10)
    top_df.to_csv(os.path.join(OUT, 'phase1b_top.csv'), index=False)

    print('\n=== Phase-1b Top 10 by Cal ===')
    print(top_df[['snap_days', 'canary_sma', 'canary_hyst', 'canary_type', 'select',
                  'def_mom_period', 'health', 'canary_assets',
                  'CAGR', 'MDD', 'Sharpe', 'Cal']].to_string(index=False))
    print(f'\n저장: {OUT}/phase1b.csv, phase1b_top.csv')


if __name__ == '__main__':
    main()
