#!/usr/bin/env python3
"""Final exact sweep — m3_engine_final simulate() 사용. 병렬."""
from __future__ import annotations
import os, sys, time
from itertools import product
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from m3_engine_final import (load_v21, load_universe_hist, list_available_futures,
                              load_coin_daily, simulate, metrics, extract_events)
from c_engine_v5 import run_c_v5, load_coin

STRAT_DIR = os.path.join(HERE, 'strat_C_v3')
N_JOBS = 24


def sim_job(args):
    events_pkl, coin_daily_pkl, v21_pkl, hist_pkl, params = args
    events = pd.read_pickle(events_pkl)
    coin_daily = pd.read_pickle(coin_daily_pkl)
    v21 = pd.read_pickle(v21_pkl)
    hist = pd.read_pickle(hist_pkl)
    port_eq, stats = simulate(events, coin_daily, v21, hist, **params)
    return {'params': params, **stats}


def main():
    v21 = load_v21()
    hist = load_universe_hist()
    avail = list_available_futures()
    coin_daily = load_coin_daily(avail)

    # Pickles for worker sharing
    v21_pkl = '/tmp/v21_final.pkl'
    cd_pkl = '/tmp/cd_final.pkl'
    hist_pkl = '/tmp/hist_final.pkl'
    v21.to_pickle(v21_pkl)
    pd.to_pickle(coin_daily, cd_pkl)
    pd.to_pickle(hist, hist_pkl)

    print(f'V21 단독: {metrics(v21["equity"], bpy=252)}')

    # ─── Phase 1: Param sweep (n=1, uni=15 고정) ───
    print('\n===== Phase 1: Param sweep =====')
    param_configs = []
    for db in [12, 24, 48, 72]:
        for dt in [-0.10, -0.12, -0.15, -0.18, -0.20]:
            for tp in [0.04, 0.06, 0.08, 0.12]:
                for ts in [12, 24, 48]:
                    param_configs.append({'dip_bars':db,'dip_thr':dt,'tp':tp,'tstop':ts})

    print(f'  {len(param_configs)} param configs')
    t0 = time.time()
    phase1 = []
    for i, P in enumerate(param_configs):
        events = extract_events(avail, P)
        ev_pkl = '/tmp/ev_final.pkl'
        events.to_pickle(ev_pkl)
        port_eq, stats = simulate(events, coin_daily, v21, hist,
                                   n_pick=1, cap_per_slot=0.333,
                                   universe_size=15, tx_cost=0.003,
                                   swap_edge_threshold=1)
        row = {**P, **{k: stats[k] for k in ['Sharpe','CAGR','MDD','Cal']},
               'n_events': len(events), 'n_entries': stats['n_entries'],
               'n_swaps': stats['n_swaps'], 'n_shrinks': stats['n_shrinks']}
        phase1.append(row)
        if (i+1) % 30 == 0:
            print(f'  {i+1}/{len(param_configs)} ({time.time()-t0:.0f}s)')
    print(f'  완료 ({time.time()-t0:.0f}s)')
    df1 = pd.DataFrame(phase1).sort_values('Cal', ascending=False)
    df1.to_csv(os.path.join(STRAT_DIR, 'exact_phase1.csv'), index=False)
    print(f'\nTop 10 (Cal 기준):')
    print(df1.head(10).to_string(index=False))

    # ─── Phase 2: Best param × universe/n_pick/cap ───
    best = df1.iloc[0]
    best_P = {'dip_bars':int(best['dip_bars']),'dip_thr':float(best['dip_thr']),
              'tp':float(best['tp']),'tstop':int(best['tstop'])}
    print(f'\n===== Phase 2: best_P={best_P} × universe/n_pick/cap =====')
    events_best = extract_events(avail, best_P)
    ev_best_pkl = '/tmp/ev_best_final.pkl'
    events_best.to_pickle(ev_best_pkl)

    configs = []
    for uni in [10, 15, 20, 30]:
        for n_pick in [1, 2, 3]:
            for cap in [0.10, 0.20, 0.30, 0.333, 0.50]:
                configs.append({'n_pick':n_pick,'cap_per_slot':cap,
                               'universe_size':uni,'tx_cost':0.003,
                               'swap_edge_threshold':1})

    rows = []
    for c in configs:
        port_eq, stats = simulate(events_best, coin_daily, v21, hist, **c)
        rows.append({**c, **{k: stats[k] for k in ['Sharpe','CAGR','MDD','Cal']},
                     'n_entries':stats['n_entries'],'n_swaps':stats['n_swaps'],
                     'n_shrinks':stats['n_shrinks']})

    df2 = pd.DataFrame(rows).sort_values('Cal', ascending=False)
    df2.to_csv(os.path.join(STRAT_DIR, 'exact_phase2.csv'), index=False)
    print(f'\nTop 15:')
    cols = ['universe_size','n_pick','cap_per_slot','Sharpe','CAGR','MDD','Cal','n_entries','n_swaps','n_shrinks']
    print(df2.head(15)[cols].to_string(index=False))

    # ─── Phase E: Robustness ───
    best2 = df2.iloc[0]
    final_P = {'n_pick':int(best2['n_pick']),'cap_per_slot':float(best2['cap_per_slot']),
               'universe_size':int(best2['universe_size']),'tx_cost':0.003,
               'swap_edge_threshold':int(best2['swap_edge_threshold']) if 'swap_edge_threshold' in best2 else 1}
    print(f'\n===== Phase E: Robustness (best={final_P}) =====')
    print(f'  dip: {best_P}')

    # E1: 2021 제거
    print('\n--- E1: 2021 제거 ablation ---')
    v21_no = v21[v21.index >= pd.Timestamp('2022-01-01')].copy()
    v21_no['equity'] = v21_no['equity'] / v21_no['equity'].iloc[0]
    v21_no['v21_ret'] = v21_no['equity'].pct_change().fillna(0)
    v21_no['prev_cash'] = v21_no['cash_ratio'].shift(1).fillna(v21_no['cash_ratio'].iloc[0])
    ev_no = events_best[events_best['entry_ts'] >= pd.Timestamp('2022-01-01')]
    m_v = metrics(v21_no['equity'], bpy=252)
    port_no, stats_no = simulate(ev_no, coin_daily, v21_no, hist, **final_P)
    m_p = {k: stats_no[k] for k in ['Sharpe','CAGR','MDD','Cal']}
    print(f'  V21 단독 2022+: {m_v}')
    print(f'  V21+C 2022+: {m_p}')

    # E2: Slippage
    print('\n--- E2: Slippage (TX 30/80/130/200bps) ---')
    for tx in [0.003, 0.008, 0.013, 0.020]:
        events_tx = extract_events(avail, best_P, tx=tx)
        port_eq, stats = simulate(events_tx, coin_daily, v21, hist,
                                   **{**final_P, 'tx_cost':tx})
        m = {k: stats[k] for k in ['Sharpe','CAGR','MDD','Cal']}
        print(f'  TX={tx*100:.2f}%: {m}')

    # E3: Trade delay
    print('\n--- E3: Trade delay ---')
    for fd in [0, 1, 4]:
        events_fd = extract_events(avail, best_P, fill_delay=fd)
        port_eq, stats = simulate(events_fd, coin_daily, v21, hist, **final_P)
        m = {k: stats[k] for k in ['Sharpe','CAGR','MDD','Cal']}
        print(f'  fill_delay={fd}: {m}')

    # E4: 연도별
    print('\n--- E4: 연도별 분해 ---')
    for yr in [2021, 2022, 2023, 2024, 2025]:
        v_y = v21[v21.index.year == yr].copy()
        if len(v_y) < 50: continue
        v_y['equity'] = v_y['equity'] / v_y['equity'].iloc[0]
        v_y['v21_ret'] = v_y['equity'].pct_change().fillna(0)
        v_y['prev_cash'] = v_y['cash_ratio'].shift(1).fillna(v_y['cash_ratio'].iloc[0])
        ev_y = events_best[events_best['entry_ts'].dt.year == yr]
        m_v = metrics(v_y['equity'], bpy=252)
        port_y, stats_y = simulate(ev_y, coin_daily, v_y, hist, **final_P)
        m_p = {k: stats_y[k] for k in ['Sharpe','CAGR','MDD','Cal']}
        print(f'  {yr}: V21 Cal={m_v["Cal"]:.2f} CAGR={m_v["CAGR"]:.2%} / V21+C Cal={m_p["Cal"]:.2f} CAGR={m_p["CAGR"]:.2%}')

    # E5: Top N event 제거
    print('\n--- E5: Top N 수익 event 제거 ---')
    for n_drop in [0, 1, 3, 5, 10, 20]:
        ev_keep = events_best.sort_values('pnl_pct', ascending=False).iloc[n_drop:]
        port_eq, stats = simulate(ev_keep, coin_daily, v21, hist, **final_P)
        m = {k: stats[k] for k in ['Sharpe','CAGR','MDD','Cal']}
        print(f'  Top {n_drop} 제거: {m}')

    # E6: Block bootstrap
    print('\n--- E6: Block bootstrap (N=200, 60일) ---')
    port_eq_best, _ = simulate(events_best, coin_daily, v21, hist, **final_P)
    rets = port_eq_best.pct_change().dropna().values
    n_rets = len(rets)
    rng = np.random.default_rng(42)
    cagr_list, mdd_list = [], []
    for _ in range(200):
        starts = rng.integers(0, n_rets-60, size=(n_rets//60)+1)
        blocks = np.concatenate([rets[s:s+60] for s in starts])[:n_rets]
        eq_b = (1 + blocks).cumprod()
        if eq_b[-1] <= 0: continue
        years = len(blocks) / 252
        cagr = eq_b[-1] ** (1/years) - 1
        run_max = np.maximum.accumulate(eq_b)
        mdd = float(((eq_b - run_max) / run_max).min())
        cagr_list.append(cagr); mdd_list.append(mdd)
    print(f'  CAGR: p5={np.percentile(cagr_list, 5):.2%} p50={np.percentile(cagr_list, 50):.2%} p95={np.percentile(cagr_list, 95):.2%}')
    print(f'  MDD:  p5={np.percentile(mdd_list, 5):.2%} p50={np.percentile(mdd_list, 50):.2%} worst={min(mdd_list):.2%}')

    print('\n===== 전체 완료 =====')


if __name__ == '__main__':
    main()
