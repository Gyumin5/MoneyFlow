"""V17 Phase-4 — EW 앙상블 (k=1, 2, 3).

입력: phase3_final.csv
후보 선정 (4-way union):
  - final_score top 5
  - jitter_mu_Cal top 5
  - CAGR top 5 (base 성능)
  - ranksum(final_score + jitter_mu_Cal + CAGR) top 5
합집합 = 최대 20개 (중복 제거 후 보통 8~15).

각 멤버의 일일 equity 를 다시 계산 → EW 앙상블 (k=1, 2, 3 전 조합).
출력: phase4_ensemble.csv (모든 앙상블 + metrics + 멤버 상관).
"""
from __future__ import annotations
import os, sys, time
from itertools import combinations
import numpy as np
import pandas as pd
from joblib import Parallel, delayed

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from stock_engine import SP, load_prices, precompute, _init, ALL_TICKERS
import stock_engine as tsi
from stock_engine_snap import run_snapshot_ensemble

from v17_phase1b import OFFENSIVE_9, DEFENSIVE_5, EXTRA_CANARY, CANARY_COMBOS

OUT = os.path.join(HERE, 'v17_snap_out')


def _metrics_from_equity(v: pd.Series) -> dict:
    if v is None or len(v) < 30:
        return {}
    y = (v.index[-1] - v.index[0]).days / 365.25
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1 / y) - 1
    mdd = (v / v.cummax() - 1).min()
    dr = v.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {
        'CAGR': round(cagr * 100, 2),
        'MDD': round(mdd * 100, 2),
        'Sharpe': round(sh, 3),
        'Cal': round(cal, 3),
    }


def run_member(cfg):
    canary_tuple = tuple(cfg['canary_assets'].split('+'))
    p = SP(
        offensive=OFFENSIVE_9,
        defensive=DEFENSIVE_5,
        canary_assets=canary_tuple,
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
        if df is None or 'Value' not in df.columns:
            return None
        return df['Value'].copy()
    except Exception:
        return None


def select_pool(df: pd.DataFrame, n_each: int = 5) -> pd.DataFrame:
    t = df.copy()
    if 'CAGR' not in t.columns or 'final_score' not in t.columns or 'jitter_mu_Cal' not in t.columns:
        raise ValueError('phase3_final.csv missing CAGR/final_score/jitter_mu_Cal')
    t['rank_fs'] = t['final_score'].rank(ascending=False, method='min')
    t['rank_mu'] = t['jitter_mu_Cal'].rank(ascending=False, method='min')
    t['rank_cagr'] = t['CAGR'].rank(ascending=False, method='min')
    t['ranksum'] = t['rank_fs'] + t['rank_mu'] + t['rank_cagr']
    idx = (set(t.nlargest(n_each, 'final_score').index)
           | set(t.nlargest(n_each, 'jitter_mu_Cal').index)
           | set(t.nlargest(n_each, 'CAGR').index)
           | set(t.nsmallest(n_each, 'ranksum').index))
    return t.loc[sorted(idx)].reset_index(drop=True)


def main():
    path = os.path.join(OUT, 'phase3_final.csv')
    if not os.path.exists(path):
        raise FileNotFoundError(f'{path} not found. Phase-3 먼저 실행.')
    df_in = pd.read_csv(path)
    pool = select_pool(df_in, n_each=5)
    print(f'Phase-4 pool: {len(pool)} members (4-way union)')

    tickers = set(ALL_TICKERS)
    tickers.update(OFFENSIVE_9); tickers.update(DEFENSIVE_5)
    for c in CANARY_COMBOS:
        tickers.update(c)
    tickers.update(EXTRA_CANARY)
    print(f'Loading prices for {len(tickers)} tickers...')
    prices = load_prices(sorted(tickers), start='2014-01-01')
    ind = precompute(prices)
    _init(prices, ind)

    t0 = time.time()
    equities = Parallel(n_jobs=-1, prefer='threads')(
        delayed(run_member)(row.to_dict()) for _, row in pool.iterrows()
    )
    print(f'Members evaluated in {time.time() - t0:.0f}s')

    valid_idx = [i for i, e in enumerate(equities) if e is not None]
    print(f'Valid members: {len(valid_idx)} / {len(pool)}')

    member_returns = {}
    for i in valid_idx:
        eq = equities[i]
        assert eq is not None
        label = f"m{i}"
        member_returns[label] = eq.pct_change().dropna()

    if len(member_returns) < 2:
        print('멤버 부족 — 앙상블 불가')
        pool.to_csv(os.path.join(OUT, 'phase4_pool.csv'), index=False)
        return

    rets_df = pd.DataFrame(member_returns).dropna(how='any')
    print(f'Aligned daily returns: {len(rets_df)} days, {len(rets_df.columns)} members')

    corr = rets_df.corr()
    corr.to_csv(os.path.join(OUT, 'phase4_corr.csv'))

    results = []
    labels = list(rets_df.columns)
    for k in (1, 2, 3):
        for combo in combinations(labels, k):
            combo_rets = rets_df[list(combo)].mean(axis=1)
            eq = (1 + combo_rets).cumprod()
            m = _metrics_from_equity(eq)
            if not m:
                continue
            pair_corrs = []
            if k >= 2:
                for a, b in combinations(combo, 2):
                    pair_corrs.append(corr.loc[a, b])
            results.append({
                'k': k,
                'members': '+'.join(combo),
                'avg_pair_corr': round(np.mean(pair_corrs), 3) if pair_corrs else 1.0,
                **m,
            })
    out = pd.DataFrame(results)
    out.to_csv(os.path.join(OUT, 'phase4_ensemble.csv'), index=False)
    pool.to_csv(os.path.join(OUT, 'phase4_pool.csv'), index=False)

    print('\n=== Cal top 10 (all k) ===')
    print(out.nlargest(10, 'Cal').to_string(index=False))
    print('\n=== CAGR top 10 (all k) ===')
    print(out.nlargest(10, 'CAGR').to_string(index=False))
    for k in (1, 2, 3):
        sub = out[out['k'] == k]
        if len(sub):
            best = sub.nlargest(1, 'Cal').iloc[0]
            print(f'\n[k={k}] best Cal: {best["members"]} → CAGR {best["CAGR"]}, MDD {best["MDD"]}, Cal {best["Cal"]}')


if __name__ == '__main__':
    main()
