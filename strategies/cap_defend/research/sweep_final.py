#!/usr/bin/env python3
"""최종 통합 sweep — 병렬. m3_lot_engine 기반.

Phase C1: Param sweep (dip 파라미터 최적 찾기, n=1, cap=0.3 고정)
Phase C2: Best param × universe × n_pick × cap 확장
Phase E: 로버스트 테스트 (2021 제거, Top N event 제거, slippage, delay)
"""
from __future__ import annotations
import os, sys, json
from itertools import product
from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import time

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
STRAT_DIR = os.path.join(HERE, 'strat_C_v3')
sys.path.insert(0, HERE)

from c_engine_v5 import run_c_v5, load_coin, metrics
from m3_lot_engine import (load_universe_hist, get_cap_rank, list_available_futures,
                            load_v21, simulate_lot)

N_JOBS = 24


def extract_events_worker(args):
    sym, p, tx, fill_delay = args
    df = load_coin(sym + 'USDT' if not sym.endswith('USDT') else sym)
    if df is None: return []
    _, evs = run_c_v5(df, tx=tx, fill_delay=fill_delay, **p)
    for e in evs:
        e['coin'] = sym.replace('USDT', '')
    return evs


def extract_events_parallel(coins, p, tx=0.003, fill_delay=0):
    results = Parallel(n_jobs=N_JOBS)(
        delayed(extract_events_worker)((c, p, tx, fill_delay)) for c in coins)
    rows = [e for batch in results for e in batch]
    return pd.DataFrame(rows)


def sim_worker(args):
    v21_pkl, events_pkl, hist, n_pick, cap_per_slot, uni_size, \
      swap_thr, swap_cd, uni_filter, tag = args
    v21 = pd.read_pickle(v21_pkl)
    events = pd.read_pickle(events_pkl)
    port_eq, stats = simulate_lot(v21, events, hist,
                                    n_pick=n_pick, cap_per_slot=cap_per_slot,
                                    universe_size=uni_size,
                                    swap_edge_threshold=swap_thr,
                                    swap_cooldown_days=swap_cd,
                                    universe_filter=uni_filter)
    m = metrics(port_eq, bpy=252)
    return {'tag': tag, 'n_pick': n_pick, 'cap_per_slot': cap_per_slot,
            'uni_size': uni_size, 'swap_thr': swap_thr, 'swap_cd': swap_cd,
            'uni_filter': uni_filter, **m, **stats}


def main():
    v21 = load_v21()
    hist = load_universe_hist()
    avail = sorted(list_available_futures())
    print(f'V21 단독: {metrics(v21["equity"], bpy=252)}')
    print(f'Coin universe: {len(avail)}')

    v21_pkl = '/tmp/v21_sweep.pkl'
    v21.to_pickle(v21_pkl)

    # ─── Phase C1: Param sweep ───
    print('\n##### Phase C1: Param sweep (n=1, cap=0.3 고정) #####')
    params_list = []
    for db in [12, 24, 48, 72]:
        for dt in [-0.10, -0.12, -0.15, -0.18, -0.20]:
            for tp in [0.04, 0.06, 0.08, 0.12]:
                for ts in [12, 24, 48]:
                    params_list.append({'dip_bars':db,'dip_thr':dt,'tp':tp,'tstop':ts})

    print(f'  {len(params_list)} param configs')
    t0 = time.time()

    phase1_results = []
    for p in params_list:
        events = extract_events_parallel(avail, p)
        ev_pkl = '/tmp/events_sweep.pkl'
        events.to_pickle(ev_pkl)
        port_eq, stats = simulate_lot(v21, events, hist,
                                        n_pick=1, cap_per_slot=0.30,
                                        universe_size=15, swap_edge_threshold=0,
                                        swap_cooldown_days=0)
        m = metrics(port_eq, bpy=252)
        phase1_results.append({**p, **m, **stats, 'n_events': len(events)})
    print(f'  Phase C1 완료 ({time.time()-t0:.1f}s)')

    df1 = pd.DataFrame(phase1_results).sort_values('Cal', ascending=False)
    df1.to_csv(os.path.join(STRAT_DIR, 'sweep_final_phase_c1.csv'), index=False)
    print(f'\nTop 10 by Cal (Phase C1):')
    print(df1.head(10)[['dip_bars','dip_thr','tp','tstop','Sharpe','CAGR','MDD','Cal','n_events','n_entries','n_swaps']].to_string(index=False))

    # ─── Phase C2: Best param × universe/n_pick/cap ───
    best = df1.iloc[0]
    best_p = {'dip_bars': int(best['dip_bars']), 'dip_thr': float(best['dip_thr']),
              'tp': float(best['tp']), 'tstop': int(best['tstop'])}
    print(f'\n##### Phase C2: best param {best_p} × universe/n_pick/cap #####')

    events_best = extract_events_parallel(avail, best_p)
    ev_pkl = '/tmp/events_best.pkl'
    events_best.to_pickle(ev_pkl)
    print(f'  Best events: {len(events_best)}')

    configs = []
    for uni in [5, 10, 15, 20, 30]:
        for n_pick in [1, 2, 3, 5]:
            for cap in [0.10, 0.15, 0.20, 0.30, 0.50]:
                for swap_thr in [0, 1]:
                    configs.append((v21_pkl, ev_pkl, hist, n_pick, cap, uni, swap_thr, 0,
                                    'entry_only', f'u{uni}_n{n_pick}_c{cap}_sw{swap_thr}'))

    print(f'  {len(configs)} configs × {N_JOBS} workers')
    t0 = time.time()
    c2_results = Parallel(n_jobs=N_JOBS, verbose=0)(
        delayed(sim_worker)(c) for c in configs)
    print(f'  Phase C2 완료 ({time.time()-t0:.1f}s)')
    df2 = pd.DataFrame(c2_results).sort_values('Cal', ascending=False)
    df2.to_csv(os.path.join(STRAT_DIR, 'sweep_final_phase_c2.csv'), index=False)
    print(f'\nTop 15 by Cal (Phase C2):')
    cols = ['uni_size','n_pick','cap_per_slot','swap_thr','Sharpe','CAGR','MDD','Cal','n_entries','n_swaps','n_shrinks']
    print(df2.head(15)[cols].to_string(index=False))

    # ─── Phase E: 로버스트 테스트 ───
    print('\n##### Phase E: 로버스트 테스트 #####')
    best2 = df2.iloc[0]
    final_params = {'n_pick': int(best2['n_pick']),
                    'cap_per_slot': float(best2['cap_per_slot']),
                    'universe_size': int(best2['uni_size']),
                    'swap_edge_threshold': int(best2['swap_thr']),
                    'swap_cooldown_days': 0,
                    'universe_filter': 'entry_only'}
    print(f'\nBest config: {final_params}')
    print(f'  dip param: {best_p}')

    # E1: 2021 제거
    print('\n--- E1: 2021 제거 ablation ---')
    v21_no = v21[v21.index >= pd.Timestamp('2022-01-01')].copy()
    v21_no['equity'] = v21_no['equity'] / v21_no['equity'].iloc[0]
    v21_no['v21_ret'] = v21_no['equity'].pct_change().fillna(0)
    v21_no['prev_cash'] = v21_no['cash_ratio'].shift(1).fillna(v21_no['cash_ratio'].iloc[0])
    ev_no = events_best[events_best['entry_ts'] >= pd.Timestamp('2022-01-01')]
    m_v21_no = metrics(v21_no['equity'], bpy=252)
    port_no, stats_no = simulate_lot(v21_no, ev_no, hist, **final_params)
    m_port_no = metrics(port_no, bpy=252)
    print(f'  V21 단독 2022+: {m_v21_no}')
    print(f'  V21+C 2022+: {m_port_no} stats={stats_no}')

    # E2: 슬리피지 stress (TX 재계산)
    print('\n--- E2: Slippage stress (TX 30/80/130/200bps) ---')
    for tx in [0.003, 0.008, 0.013, 0.020]:
        events_tx = extract_events_parallel(avail, best_p, tx=tx)
        port_eq, stats = simulate_lot(v21, events_tx, hist, tx_cost=tx, **final_params)
        m = metrics(port_eq, bpy=252)
        print(f'  TX={tx*100:.2f}% bps: {m}')

    # E3: Trade delay (0/1/4 bars)
    print('\n--- E3: Trade delay ---')
    for fd in [0, 1, 4]:
        events_fd = extract_events_parallel(avail, best_p, fill_delay=fd)
        port_eq, _ = simulate_lot(v21, events_fd, hist, **final_params)
        m = metrics(port_eq, bpy=252)
        print(f'  fill_delay={fd} bars: {m}')

    # E4: Top N event 제거
    print('\n--- E4: Top N 수익 event 제거 ---')
    for remove_n in [0, 1, 3, 5, 10, 20]:
        ev_keep = events_best.sort_values('pnl_pct', ascending=False).iloc[remove_n:]
        port_eq, _ = simulate_lot(v21, ev_keep, hist, **final_params)
        m = metrics(port_eq, bpy=252)
        print(f'  Top {remove_n} 제거: {m}')

    # E5: 연도별 분해
    print('\n--- E5: 연도별 (V21 vs V21+C) ---')
    for yr in [2021, 2022, 2023, 2024, 2025]:
        v_y = v21[v21.index.year == yr].copy()
        if len(v_y) < 50: continue
        v_y['equity'] = v_y['equity'] / v_y['equity'].iloc[0]
        v_y['v21_ret'] = v_y['equity'].pct_change().fillna(0)
        v_y['prev_cash'] = v_y['cash_ratio'].shift(1).fillna(v_y['cash_ratio'].iloc[0])
        ev_y = events_best[events_best['entry_ts'].dt.year == yr]
        m_v = metrics(v_y['equity'], bpy=252)
        port_y, _ = simulate_lot(v_y, ev_y, hist, **final_params)
        m_p = metrics(port_y, bpy=252)
        print(f'  {yr}: V21 Cal={m_v["Cal"]:.2f} CAGR={m_v["CAGR"]:.2%} / V21+C Cal={m_p["Cal"]:.2f} CAGR={m_p["CAGR"]:.2%}')

    # E6: Block bootstrap
    print('\n--- E6: Block bootstrap (200회, 60일 블록) ---')
    port_eq_best, _ = simulate_lot(v21, events_best, hist, **final_params)
    rets = port_eq_best.pct_change().dropna().values
    n = len(rets)
    rng = np.random.default_rng(42)
    cagr_list, mdd_list = [], []
    for _ in range(200):
        starts = rng.integers(0, n-60, size=(n//60)+1)
        blocks = np.concatenate([rets[s:s+60] for s in starts])[:n]
        eq_b = (1 + blocks).cumprod()
        if eq_b[-1] <= 0: continue
        years = len(blocks) / 252
        cagr = eq_b[-1] ** (1/years) - 1
        run_max = np.maximum.accumulate(eq_b)
        mdd = float(((eq_b - run_max) / run_max).min())
        cagr_list.append(cagr); mdd_list.append(mdd)
    print(f'  CAGR: p5={np.percentile(cagr_list, 5):.2%} p50={np.percentile(cagr_list, 50):.2%} p95={np.percentile(cagr_list, 95):.2%}')
    print(f'  MDD:  p5={np.percentile(mdd_list, 5):.2%} p50={np.percentile(mdd_list, 50):.2%} worst={min(mdd_list):.2%}')

    print('\n##### 전체 sweep 완료 #####')


if __name__ == '__main__':
    main()
