#!/usr/bin/env python3
"""v4 병렬 sweep — joblib으로 24 worker.

각 worker가 한 param config 전체 처리 (10 코인 event 추출 + EW + M3 시뮬).
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from itertools import product
from joblib import Parallel, delayed

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
from c_engine_v4 import run_c_v4, load_coin, UNIVERSE_TOP10, metrics

STRAT_DIR = os.path.join(HERE, 'strat_C_v3')
N_JOBS = 24


def load_v21():
    v21 = pd.read_csv(os.path.join(STRAT_DIR, 'v21_daily.csv'), index_col=0, parse_dates=True)
    v21['equity'] = v21['equity'] / v21['equity'].iloc[0]
    return v21


# 전역 캐시 (각 worker가 개별 로드)
_DATA = None
def get_data():
    global _DATA
    if _DATA is None:
        _DATA = {}
        for c in UNIVERSE_TOP10:
            df = load_coin(c)
            if df is not None:
                _DATA[c] = df
    return _DATA


def build_c_daily(coins, dip_bars, dip_thr, tp, tstop):
    data = get_data()
    eq_list = []
    for c in coins:
        if c not in data: continue
        eq, _ = run_c_v4(data[c], dip_bars=dip_bars, dip_thr=dip_thr, tp=tp, tstop=tstop)
        eq = eq / eq.iloc[0]
        d = eq.resample('D').last().ffill()
        eq_list.append(d.rename(c))
    if not eq_list: return None
    df = pd.concat(eq_list, axis=1).ffill()
    avg = df.pct_change().fillna(0).mean(axis=1)
    return (1 + avg).cumprod()


def simulate_m3(v21, c_eq, hard_cap):
    """정확한 M3: C 포지션 있는 날만 c_w 적용.
    C 포지션 있는 날 = c_ret != 0 (즉 그날 C 수익 변동 있는 날).
    """
    idx = v21.index.intersection(c_eq.index)
    v21_sub = v21.loc[idx]
    c_sub = c_eq.loc[idx]
    v21_ret = v21_sub['equity'].pct_change().fillna(0)
    c_ret = c_sub.pct_change().fillna(0)
    # C active 여부: c_ret 변화 있는 날
    c_active = (c_ret.abs() > 1e-10).astype(float)
    c_w_potential = np.minimum(hard_cap, v21_sub['cash_ratio'])
    c_w = c_w_potential * c_active  # 포지션 없는 날 0
    port_r = (1 - c_w) * v21_ret + c_w * c_ret
    return (1 + port_r).cumprod()


def process_config(args):
    db, dt, tp, ts, v21_pickle = args
    v21 = pd.read_pickle(v21_pickle)
    c_eq = build_c_daily(UNIVERSE_TOP10, db, dt, tp, ts)
    if c_eq is None: return None
    m_c = metrics(c_eq, bpy=252)
    port_eq = simulate_m3(v21, c_eq, 0.30)
    m_p = metrics(port_eq, bpy=252)
    return {
        'dip_bars': db, 'dip_thr': dt, 'tp': tp, 'tstop': ts,
        'C_Sharpe': m_c['Sharpe'], 'C_CAGR': m_c['CAGR'], 'C_Cal': m_c['Cal'], 'C_MDD': m_c['MDD'],
        'Port_Sharpe': m_p['Sharpe'], 'Port_CAGR': m_p['CAGR'],
        'Port_MDD': m_p['MDD'], 'Port_Cal': m_p['Cal'],
    }


def main():
    v21 = load_v21()
    v21_pickle = '/tmp/v21_sweep.pkl'
    v21.to_pickle(v21_pickle)
    print(f'V21 단독: {metrics(v21["equity"], bpy=252)}')

    # Phase 1 configs
    dip_bars_list = [12, 24, 48, 72]
    dip_thr_list = [-0.10, -0.12, -0.15, -0.18, -0.20]
    tp_list = [0.04, 0.06, 0.08, 0.12]
    tstop_list = [12, 24, 48]
    configs = [(db, dt, tp, ts, v21_pickle) for db, dt, tp, ts in product(
        dip_bars_list, dip_thr_list, tp_list, tstop_list)]
    print(f'\n=== Phase 1: {len(configs)} configs × {N_JOBS} worker 병렬 ===')

    import time
    t0 = time.time()
    results = Parallel(n_jobs=N_JOBS, verbose=5)(delayed(process_config)(c) for c in configs)
    results = [r for r in results if r is not None]
    print(f'\nPhase 1 완료 ({time.time()-t0:.1f}s)')

    df = pd.DataFrame(results).sort_values('Port_Cal', ascending=False)
    df.to_csv(os.path.join(STRAT_DIR, 'sweep_v4p_phase1.csv'), index=False)
    print(f'\nTop 15 by Port_Cal:')
    print(df.head(15).to_string(index=False))

    # Phase 2: best param × universe/cap 확장
    best = df.iloc[0]
    best_p = {'dip_bars': int(best['dip_bars']), 'dip_thr': float(best['dip_thr']),
              'tp': float(best['tp']), 'tstop': int(best['tstop'])}
    print(f'\n=== Phase 2: best param {best_p} × universe/cap ===')

    UNI = {'Top3': UNIVERSE_TOP10[:3], 'Top5': UNIVERSE_TOP10[:5],
           'Top7': UNIVERSE_TOP10[:7], 'Top10': UNIVERSE_TOP10}
    rows = []
    for uname, coins in UNI.items():
        c_eq = build_c_daily(coins, **best_p)
        if c_eq is None: continue
        m_c = metrics(c_eq, bpy=252)
        for cap in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]:
            port = simulate_m3(v21, c_eq, cap)
            m_p = metrics(port, bpy=252)
            rows.append({'uni': uname, 'cap': cap,
                         'C_Sharpe': m_c['Sharpe'], 'C_CAGR': m_c['CAGR'], 'C_Cal': m_c['Cal'],
                         'Port_Sharpe': m_p['Sharpe'], 'Port_CAGR': m_p['CAGR'],
                         'Port_MDD': m_p['MDD'], 'Port_Cal': m_p['Cal']})
    df2 = pd.DataFrame(rows).sort_values('Port_Cal', ascending=False)
    df2.to_csv(os.path.join(STRAT_DIR, 'sweep_v4p_phase2.csv'), index=False)
    print(df2.to_string(index=False))

    # Phase 3: 2021 제거 (phase2 top 5)
    print('\n=== Phase 3: 2021 제거 ===')
    v21_no = v21[v21.index >= pd.Timestamp('2022-01-01')].copy()
    v21_no['equity'] = v21_no['equity'] / v21_no['equity'].iloc[0]
    m_v_no = metrics(v21_no['equity'], bpy=252)
    print(f'V21 단독 2022+: {m_v_no}')

    for _, r in df2.head(5).iterrows():
        coins = UNI[r['uni']]
        c_eq = build_c_daily(coins, **best_p)
        c_no = c_eq[c_eq.index >= pd.Timestamp('2022-01-01')].copy()
        c_no = c_no / c_no.iloc[0]
        port_no = simulate_m3(v21_no, c_no, r['cap'])
        m_no = metrics(port_no, bpy=252)
        print(f"  {r['uni']} cap={r['cap']:.0%}: all_Cal={r['Port_Cal']:.2f} / 2022+_Cal={m_no['Cal']:.2f}")


if __name__ == '__main__':
    main()
