#!/usr/bin/env python3
"""M3 시뮬 v4 — 버그 수정된 c_engine_v4 사용. 최종 베스트 전략 찾기."""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from c_engine_v4 import (run_c_v4, load_coin, extract_events_all,
                         UNIVERSE_TOP10, CAP_RANK, metrics)

STRAT_DIR = os.path.join(HERE, 'strat_C_v3')


def load_v21():
    v21 = pd.read_csv(os.path.join(STRAT_DIR, 'v21_daily.csv'), index_col=0, parse_dates=True)
    v21['equity'] = v21['equity'] / v21['equity'].iloc[0]
    v21['v21_ret'] = v21['equity'].pct_change().fillna(0)
    return v21


def build_c_daily(coins, tx=0.003, fill_delay=0, **kwargs):
    """각 코인 v4 equity → EW portfolio daily return."""
    eq_list = []
    for c in coins:
        df = load_coin(c)
        if df is None: continue
        eq, _ = run_c_v4(df, tx=tx, fill_delay=fill_delay, **kwargs)
        eq = eq / eq.iloc[0]
        d = eq.resample('D').last().ffill()
        eq_list.append(d.rename(c))
    if not eq_list: return None
    df = pd.concat(eq_list, axis=1).ffill()
    avg_ret = df.pct_change().fillna(0).mean(axis=1)
    return (1 + avg_ret).cumprod()


def simulate_m3(v21, c_eq, hard_cap=0.30, cap_mode='absolute'):
    """M3: c_weight = min(hard_cap, cash_ratio) × c_active_indicator (c_ret !=0)
    간단한 근사: port_ret = (1 - c_w) × v21_ret + c_w × c_ret
    """
    idx = v21.index.intersection(c_eq.index)
    v21 = v21.loc[idx]
    c_eq = c_eq.loc[idx]
    v21_ret = v21['equity'].pct_change().fillna(0)
    c_ret = c_eq.pct_change().fillna(0)
    cash_r = v21['cash_ratio']

    # c_weight series
    if cap_mode == 'absolute':
        c_w = np.minimum(hard_cap, cash_r)
    else:  # cash_pct
        c_w = hard_cap * cash_r

    port_r = (1 - c_w) * v21_ret + c_w * c_ret
    port_eq = (1 + port_r).cumprod()
    return port_eq, c_w


def main():
    v21 = load_v21()
    print(f'V21 단독: {metrics(v21["equity"], bpy=252)}')

    # C 단독 (100% 투입) - Top 10 EW
    print('\n=== C 단독 (v4 버그 수정, Top 10 EW) ===')
    for coins_name, coins in [('Top10 EW', UNIVERSE_TOP10),
                               ('BTC+ADA+ETH', ['BTCUSDT','ADAUSDT','ETHUSDT']),
                               ('BTC+ADA', ['BTCUSDT','ADAUSDT']),
                               ('BTC 단독', ['BTCUSDT'])]:
        c_eq = build_c_daily(coins)
        if c_eq is None: continue
        m = metrics(c_eq, bpy=252)
        print(f'  {coins_name:>14}: {m}')

    # M3 시뮬: Universe × Hard Cap sweep
    print('\n=== M3 시뮬 (v4, absolute) ===')
    print(f"{'universe':>14} {'hc':>5} {'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Cal':>6}")
    for uname, coins in [('Top10', UNIVERSE_TOP10),
                          ('BTC+ADA+ETH', ['BTCUSDT','ADAUSDT','ETHUSDT']),
                          ('BTC+ADA', ['BTCUSDT','ADAUSDT']),
                          ('BTC', ['BTCUSDT'])]:
        c_eq = build_c_daily(coins)
        if c_eq is None: continue
        for hc in [0.10, 0.20, 0.30, 0.50, 0.70, 1.00]:
            port_eq, c_w = simulate_m3(v21, c_eq, hc, 'absolute')
            m = metrics(port_eq, bpy=252)
            print(f"{uname:>14} {hc:>5.2f} {m['Sharpe']:>7.3f} {m['CAGR']:>7.1%} {m['MDD']:>7.1%} {m['Cal']:>6.2f}")

    # 2021 제거
    print('\n=== 2021 제거 ablation (Top10 cap 30%) ===')
    c_eq = build_c_daily(UNIVERSE_TOP10)
    v21_2 = v21[v21.index >= pd.Timestamp('2022-01-01')].copy()
    v21_2['equity'] = v21_2['equity'] / v21_2['equity'].iloc[0]
    c_2 = c_eq[c_eq.index >= pd.Timestamp('2022-01-01')].copy()
    c_2 = c_2 / c_2.iloc[0]
    port, _ = simulate_m3(v21_2, c_2, 0.30)
    m_v = metrics(v21_2['equity'], bpy=252)
    m_p = metrics(port, bpy=252)
    print(f'  V21 단독: {m_v}')
    print(f'  V21+C 30%: {m_p}')


if __name__ == '__main__':
    main()
