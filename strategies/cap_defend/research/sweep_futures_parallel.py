#!/usr/bin/env python3
"""Parallel sweep for futures V21+C with lev=3."""
from __future__ import annotations
import os, sys, time
from joblib import Parallel, delayed
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from m3_engine_futures import (load_v21_futures, load_universe_hist, list_available_futures,
                                load_coin_daily, simulate_fut, metrics, extract_events)
from c_engine_v5 import run_c_v5, load_coin

STRAT_DIR = os.path.join(HERE, 'strat_C_v3')
N_JOBS = 24


def _phase1_worker(args):
    P, coins, v21_pkl, cd_pkl, hist_pkl = args
    v21 = pd.read_pickle(v21_pkl)
    coin_daily = pd.read_pickle(cd_pkl)
    hist = pd.read_pickle(hist_pkl)

    rows = []
    for sym in coins:
        df = load_coin(sym + 'USDT')
        if df is None: continue
        _, evs = run_c_v5(df, **P)
        for e in evs:
            e['coin'] = sym
            rows.append(e)
    events = pd.DataFrame(rows)
    if len(events) == 0:
        return {**P, 'Sharpe':0,'CAGR':0,'MDD':0,'Cal':0,'n_events':0}

    port_eq, stats = simulate_fut(events, coin_daily, v21, hist,
                                    n_pick=1, cap_per_slot=0.333,
                                    universe_size=15, tx_cost=0.003,
                                    swap_edge_threshold=1, leverage=3.0)
    return {**P, **{k: stats[k] for k in ['Sharpe','CAGR','MDD','Cal']},
            'n_events': len(events), 'n_entries':stats['n_entries'],
            'n_swaps':stats['n_swaps'], 'n_liq':stats['n_liquidations']}


def main():
    v21_fut = load_v21_futures()
    hist = load_universe_hist()
    avail = sorted(list_available_futures())
    coin_daily = load_coin_daily(avail)
    print(f'V21 선물 단독: {metrics(v21_fut["equity"], bpy=252)}')
    print(f'Coins: {len(avail)}')

    v21_pkl = '/tmp/v21fut_par.pkl'
    cd_pkl = '/tmp/cdfut_par.pkl'
    hist_pkl = '/tmp/histfut_par.pkl'
    v21_fut.to_pickle(v21_pkl)
    pd.to_pickle(coin_daily, cd_pkl)
    pd.to_pickle(hist, hist_pkl)

    # Phase 1: param sweep
    configs = []
    for db in [12, 24, 48, 72]:
        for dt in [-0.10, -0.12, -0.15, -0.18, -0.20]:
            for tp in [0.04, 0.06, 0.08, 0.12]:
                for ts in [12, 24, 48]:
                    configs.append({'dip_bars':db,'dip_thr':dt,'tp':tp,'tstop':ts})

    print(f'\n===== Phase 1: {len(configs)} configs × {N_JOBS} =====')
    t0 = time.time()
    args_list = [(P, avail, v21_pkl, cd_pkl, hist_pkl) for P in configs]
    results = Parallel(n_jobs=N_JOBS, verbose=5)(
        delayed(_phase1_worker)(a) for a in args_list)
    print(f'  완료 ({time.time()-t0:.0f}s)')

    df1 = pd.DataFrame(results).sort_values('Cal', ascending=False)
    df1.to_csv(os.path.join(STRAT_DIR, 'fut_phase1.csv'), index=False)
    print('\nTop 15:')
    print(df1.head(15).to_string(index=False))

    # Phase 2
    best = df1.iloc[0]
    best_P = {'dip_bars':int(best['dip_bars']),'dip_thr':float(best['dip_thr']),
              'tp':float(best['tp']),'tstop':int(best['tstop'])}
    print(f'\n===== Phase 2: best_P={best_P} =====')

    rows = []
    for sym in avail:
        df = load_coin(sym + 'USDT')
        if df is None: continue
        _, evs = run_c_v5(df, **best_P)
        for e in evs:
            e['coin'] = sym
            rows.append(e)
    events_best = pd.DataFrame(rows)
    print(f'  events: {len(events_best)}')

    p2 = []
    for uni in [10, 15, 20, 30]:
        for n_pick in [1, 2, 3]:
            for cap in [0.05, 0.10, 0.15, 0.20, 0.30, 0.333]:
                port_eq, stats = simulate_fut(events_best, coin_daily, v21_fut, hist,
                                                n_pick=n_pick, cap_per_slot=cap,
                                                universe_size=uni, leverage=3.0)
                p2.append({'uni':uni,'n_pick':n_pick,'cap':cap,
                          **{k: stats[k] for k in ['Sharpe','CAGR','MDD','Cal']},
                          'n_entries':stats['n_entries'],'n_swaps':stats['n_swaps'],
                          'n_shrinks':stats['n_shrinks'],'n_liq':stats['n_liquidations']})

    df2 = pd.DataFrame(p2).sort_values('Cal', ascending=False)
    df2.to_csv(os.path.join(STRAT_DIR, 'fut_phase2.csv'), index=False)
    print('\nTop 15:')
    print(df2.head(15).to_string(index=False))

    # Robustness
    best2 = df2.iloc[0]
    FP = {'n_pick':int(best2['n_pick']),'cap_per_slot':float(best2['cap']),
          'universe_size':int(best2['uni']),'tx_cost':0.003,
          'swap_edge_threshold':1,'leverage':3.0}
    print(f'\n===== Robustness (FP={FP}, dip={best_P}) =====')

    # 2021 제거
    v21_no = v21_fut[v21_fut.index >= pd.Timestamp('2022-01-01')].copy()
    v21_no['equity'] = v21_no['equity'] / v21_no['equity'].iloc[0]
    v21_no['v21_ret'] = v21_no['equity'].pct_change().fillna(0)
    v21_no['prev_cash'] = v21_no['cash_ratio'].shift(1).fillna(v21_no['cash_ratio'].iloc[0])
    ev_no = events_best[events_best['entry_ts'] >= pd.Timestamp('2022-01-01')]
    m_v = metrics(v21_no['equity'], bpy=252)
    port_no, st_no = simulate_fut(ev_no, coin_daily, v21_no, hist, **FP)
    m_p = {k: st_no[k] for k in ['Sharpe','CAGR','MDD','Cal']}
    print(f'  V21 선물 단독 2022+: {m_v}')
    print(f'  V21+C 선물 2022+: {m_p}')

    # Slippage
    print('\n--- Slippage ---')
    for tx in [0.003, 0.008, 0.013, 0.020]:
        evs = []
        for sym in avail:
            df = load_coin(sym + 'USDT')
            if df is None: continue
            _, e = run_c_v5(df, tx=tx, **best_P)
            for x in e:
                x['coin'] = sym; evs.append(x)
        edf = pd.DataFrame(evs)
        port_eq, st = simulate_fut(edf, coin_daily, v21_fut, hist, **{**FP, 'tx_cost':tx})
        m = {k: st[k] for k in ['Sharpe','CAGR','MDD','Cal']}
        print(f'  TX={tx*100:.2f}%: {m}')

    # Delay
    print('\n--- Delay ---')
    for fd in [0, 1, 4]:
        evs = []
        for sym in avail:
            df = load_coin(sym + 'USDT')
            if df is None: continue
            _, e = run_c_v5(df, fill_delay=fd, **best_P)
            for x in e:
                x['coin'] = sym; evs.append(x)
        edf = pd.DataFrame(evs)
        port_eq, st = simulate_fut(edf, coin_daily, v21_fut, hist, **FP)
        m = {k: st[k] for k in ['Sharpe','CAGR','MDD','Cal']}
        print(f'  fd={fd}: {m}')

    # 연도별
    print('\n--- 연도별 ---')
    for yr in [2021, 2022, 2023, 2024, 2025]:
        v_y = v21_fut[v21_fut.index.year == yr].copy()
        if len(v_y) < 50: continue
        v_y['equity'] = v_y['equity'] / v_y['equity'].iloc[0]
        v_y['v21_ret'] = v_y['equity'].pct_change().fillna(0)
        v_y['prev_cash'] = v_y['cash_ratio'].shift(1).fillna(v_y['cash_ratio'].iloc[0])
        ev_y = events_best[events_best['entry_ts'].dt.year == yr]
        m_v = metrics(v_y['equity'], bpy=252)
        port_y, st_y = simulate_fut(ev_y, coin_daily, v_y, hist, **FP)
        m_p = {k: st_y[k] for k in ['Sharpe','CAGR','MDD','Cal']}
        print(f'  {yr}: V21 Cal={m_v["Cal"]:.2f} CAGR={m_v["CAGR"]:.2%} / V21+C Cal={m_p["Cal"]:.2f} CAGR={m_p["CAGR"]:.2%}')

    # Top N 제거
    print('\n--- Top N 제거 ---')
    for n in [0, 1, 3, 5, 10, 20]:
        ev_keep = events_best.sort_values('pnl_pct', ascending=False).iloc[n:]
        port_eq, st = simulate_fut(ev_keep, coin_daily, v21_fut, hist, **FP)
        m = {k: st[k] for k in ['Sharpe','CAGR','MDD','Cal']}
        print(f'  Top {n}: {m}')

    # Bootstrap
    print('\n--- Bootstrap ---')
    port_best, _ = simulate_fut(events_best, coin_daily, v21_fut, hist, **FP)
    rets = port_best.pct_change().dropna().values
    n_r = len(rets)
    rng = np.random.default_rng(42)
    cagr_l, mdd_l = [], []
    for _ in range(200):
        starts = rng.integers(0, n_r-60, size=(n_r//60)+1)
        blocks = np.concatenate([rets[s:s+60] for s in starts])[:n_r]
        eq_b = (1 + blocks).cumprod()
        if eq_b[-1] <= 0: continue
        years = len(blocks) / 252
        cagr_l.append(eq_b[-1]**(1/years) - 1)
        mx = np.maximum.accumulate(eq_b)
        mdd_l.append(float(((eq_b-mx)/mx).min()))
    print(f'  CAGR p5={np.percentile(cagr_l, 5):.2%} p50={np.percentile(cagr_l, 50):.2%} p95={np.percentile(cagr_l, 95):.2%}')
    print(f'  MDD p5={np.percentile(mdd_l, 5):.2%} p50={np.percentile(mdd_l, 50):.2%} worst={min(mdd_l):.2%}')

    print('\n===== 완료 =====')


if __name__ == '__main__':
    main()
