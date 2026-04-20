#!/usr/bin/env python3
"""v4 넓은 그리드 sweep: C 단독 + V21+C 앙상블 둘 다."""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd
from itertools import product

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
from c_engine_v4 import run_c_v4, load_coin, UNIVERSE_TOP10, metrics

STRAT_DIR = os.path.join(HERE, 'strat_C_v3')


def load_v21():
    v21 = pd.read_csv(os.path.join(STRAT_DIR, 'v21_daily.csv'), index_col=0, parse_dates=True)
    v21['equity'] = v21['equity'] / v21['equity'].iloc[0]
    return v21


def cached_load_coins():
    """Top 10 1h 데이터 미리 로드."""
    data = {}
    for c in UNIVERSE_TOP10:
        df = load_coin(c)
        if df is not None:
            data[c] = df
    return data


def build_c_daily_from_data(data, coins, **params):
    """각 코인 v4 equity → EW portfolio daily return."""
    eq_list = []
    for c in coins:
        if c not in data: continue
        eq, _ = run_c_v4(data[c], **params)
        eq = eq / eq.iloc[0]
        d = eq.resample('D').last().ffill()
        eq_list.append(d.rename(c))
    if not eq_list: return None
    df = pd.concat(eq_list, axis=1).ffill()
    avg = df.pct_change().fillna(0).mean(axis=1)
    return (1 + avg).cumprod()


def simulate_m3(v21, c_eq, hard_cap):
    idx = v21.index.intersection(c_eq.index)
    v21_sub = v21.loc[idx]
    c_sub = c_eq.loc[idx]
    v21_ret = v21_sub['equity'].pct_change().fillna(0)
    c_ret = c_sub.pct_change().fillna(0)
    c_w = np.minimum(hard_cap, v21_sub['cash_ratio'])
    port_r = (1 - c_w) * v21_ret + c_w * c_ret
    return (1 + port_r).cumprod()


def main():
    v21 = load_v21()
    m_v21 = metrics(v21['equity'], bpy=252)
    print(f'V21 단독: {m_v21}')

    data = cached_load_coins()
    print(f'코인 로드: {list(data.keys())}')

    # === Phase 1: param sweep (Top 10 dynamic) ===
    results = []
    dip_bars_list = [12, 24, 48, 72]
    dip_thr_list = [-0.10, -0.12, -0.15, -0.18, -0.20]
    tp_list = [0.04, 0.06, 0.08, 0.12]
    tstop_list = [12, 24, 48]

    total = len(dip_bars_list) * len(dip_thr_list) * len(tp_list) * len(tstop_list)
    print(f'\n=== Phase 1: param sweep Top 10 dyn + M3 cap30% ({total} configs) ===')
    done = 0
    for db, dt, tp, ts in product(dip_bars_list, dip_thr_list, tp_list, tstop_list):
        p = {'dip_bars': db, 'dip_thr': dt, 'tp': tp, 'tstop': ts}
        c_eq = build_c_daily_from_data(data, UNIVERSE_TOP10, **p)
        if c_eq is None: continue
        m_c = metrics(c_eq, bpy=252)
        port_eq = simulate_m3(v21, c_eq, 0.30)
        m_p = metrics(port_eq, bpy=252)
        results.append({
            **p, 'uni': 'Top10', 'cap': 0.30,
            'C_Sharpe': m_c['Sharpe'], 'C_CAGR': m_c['CAGR'], 'C_Cal': m_c['Cal'],
            'Port_Sharpe': m_p['Sharpe'], 'Port_CAGR': m_p['CAGR'],
            'Port_MDD': m_p['MDD'], 'Port_Cal': m_p['Cal'],
        })
        done += 1
        if done % 50 == 0:
            print(f'  진행 {done}/{total}')

    df = pd.DataFrame(results)
    df = df.sort_values('Port_Cal', ascending=False)
    df.to_csv(os.path.join(STRAT_DIR, 'sweep_v4_phase1.csv'), index=False)

    print(f'\nTop 10 (Port_Cal 기준):')
    print(df.head(10).to_string(index=False))

    # === Phase 2: best param × universe/n_pick/cap 확장 ===
    best_p = df.iloc[0]
    best_params = {'dip_bars': int(best_p['dip_bars']),
                   'dip_thr': float(best_p['dip_thr']),
                   'tp': float(best_p['tp']),
                   'tstop': int(best_p['tstop'])}
    print(f'\n=== Phase 2: best param {best_params} × universe/cap 확장 ===')

    UNI = {
        'Top3': UNIVERSE_TOP10[:3],
        'Top5': UNIVERSE_TOP10[:5],
        'Top7': UNIVERSE_TOP10[:7],
        'Top10': UNIVERSE_TOP10,
    }
    rows2 = []
    for uname, coins in UNI.items():
        c_eq = build_c_daily_from_data(data, coins, **best_params)
        if c_eq is None: continue
        m_c = metrics(c_eq, bpy=252)
        for cap in [0.05, 0.10, 0.20, 0.30, 0.50, 0.70, 1.00]:
            port_eq = simulate_m3(v21, c_eq, cap)
            m_p = metrics(port_eq, bpy=252)
            rows2.append({
                'uni': uname, 'cap': cap,
                'C_Sharpe': m_c['Sharpe'], 'C_Cal': m_c['Cal'], 'C_CAGR': m_c['CAGR'],
                'Port_Sharpe': m_p['Sharpe'], 'Port_CAGR': m_p['CAGR'],
                'Port_MDD': m_p['MDD'], 'Port_Cal': m_p['Cal'],
            })
    df2 = pd.DataFrame(rows2).sort_values('Port_Cal', ascending=False)
    df2.to_csv(os.path.join(STRAT_DIR, 'sweep_v4_phase2.csv'), index=False)
    print(df2.to_string(index=False))

    # === Phase 3: 2021 제거 (top 5 best configs) ===
    print('\n=== Phase 3: 2021 제거 ablation (phase2 top5) ===')
    v21_no = v21[v21.index >= pd.Timestamp('2022-01-01')].copy()
    v21_no['equity'] = v21_no['equity'] / v21_no['equity'].iloc[0]
    m_v_no = metrics(v21_no['equity'], bpy=252)
    print(f'V21 단독 (2022+): {m_v_no}')

    for _, row in df2.head(5).iterrows():
        coins = UNI[row['uni']]
        c_eq = build_c_daily_from_data(data, coins, **best_params)
        c_no = c_eq[c_eq.index >= pd.Timestamp('2022-01-01')].copy()
        c_no = c_no / c_no.iloc[0]
        port_no = simulate_m3(v21_no, c_no, row['cap'])
        m_no = metrics(port_no, bpy=252)
        print(f"  {row['uni']} cap={row['cap']:.0%}: all_period Cal={row['Port_Cal']:.2f} → 2022+ Cal={m_no['Cal']:.2f}")


if __name__ == '__main__':
    main()
