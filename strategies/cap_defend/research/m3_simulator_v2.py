#!/usr/bin/env python3
"""M3 simulator v2 — 단순하고 검증 가능한 구현.

핵심 아이디어:
- V21 daily return(v21_ret), V21 daily cash_ratio(이미 주어짐)
- 각 코인 C daily return은 이벤트 기반으로 합성 (같은 기간 내 V21과 동일 resolution)
- 매일:
  port_ret = (1 - c_weight_t) × v21_ret + c_weight_t × c_ret_t
  c_weight_t = min(hard_cap, v21_cash_ratio_t) AND has active C event

구현 단순성:
- C daily return 시계열 생성 (각 코인별 single-coin v3 equity의 daily pct_change)
- C events list도 병행 (active 여부만 확인용)

다코인 n_pick 분할은 단순화: c_weight 고정, 여러 코인 동시 active면 EW 평균
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, ROOT)

STRAT_DIR = os.path.join(HERE, 'strat_C_v3')
DATA_DIR = os.path.join(ROOT, 'data', 'futures')

START = '2020-10-01'
END = '2026-03-30'

PARAMS = {
    'dip_bars': 24, 'dip_threshold': -0.15,
    'take_profit': 0.08, 'time_stop_bars': 24, 'lev': 1.0,
}
TX_COST = 0.003


def load_coin_1h(sym):
    path = os.path.join(DATA_DIR, f'{sym}_1h.csv')
    if not os.path.isfile(path): return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df.loc[START:END].copy()


def c_equity_single(df, **p):
    """검증된 v3 single-coin (보수 체결, High 매수 + Open 매도)"""
    df = df.copy()
    df['dip_pct'] = df['Close'] / df['Close'].shift(p['dip_bars']) - 1.0
    df['dip_sig'] = df['dip_pct'].shift(1) <= p['dip_threshold']
    eq = 10000.0
    equity = []
    position = 0; entry = 0; bars = 0
    for i, (ts, row) in enumerate(df.iterrows()):
        if position > 0:
            pnl = row['Open']/entry - 1.0
            if pnl >= p['take_profit'] or bars >= p['time_stop_bars']:
                eq *= (1 + p['lev']*pnl - TX_COST)
                position = 0; bars = 0
        if position == 0 and row['dip_sig']:
            entry = row['High']; position = 1; bars = 0
            eq *= (1 - TX_COST)
            eq *= (1 + p['lev'] * (row['Close']/entry - 1))
            bars += 1
        elif position > 0:
            prev_close = df.iloc[i-1]['Close'] if i > 0 else row['Open']
            eq *= (1 + p['lev'] * (row['Close']/prev_close - 1))
            bars += 1
        equity.append(eq)
    return pd.Series(equity, index=df.index)


def build_c_daily(coins):
    """각 코인 C equity daily로 resample + EW 합산 daily return"""
    eq_daily_list = []
    for c in coins:
        df = load_coin_1h(c)
        if df is None: continue
        eq = c_equity_single(df, **PARAMS)
        eq = eq / eq.iloc[0]
        # daily
        d = eq.resample('D').last().ffill()
        eq_daily_list.append(d.rename(c))
    if not eq_daily_list:
        return None
    df_all = pd.concat(eq_daily_list, axis=1).ffill()
    # EW portfolio daily return (각 코인 등가중)
    daily_rets = df_all.pct_change().fillna(0)
    avg_ret = daily_rets.mean(axis=1)
    # Equity from avg return
    c_eq = (1 + avg_ret).cumprod()
    return c_eq


def simulate_m3(v21, c_eq, hard_cap=0.10):
    """매일 port_ret = (1 - c_w) × v21_ret + c_w × c_ret, c_w = min(hard_cap, v21_cash_ratio)"""
    # align index
    idx = v21.index.intersection(c_eq.index)
    v21 = v21.loc[idx].copy()
    c_eq = c_eq.loc[idx].copy()
    v21['v21_ret'] = v21['equity'].pct_change().fillna(0)
    c_ret = c_eq.pct_change().fillna(0)

    port_rets = []
    c_weights = []
    for date in idx:
        c_w = min(hard_cap, v21.loc[date, 'cash_ratio'])
        port_r = (1 - c_w) * v21.loc[date, 'v21_ret'] + c_w * c_ret.loc[date]
        port_rets.append(port_r)
        c_weights.append(c_w)
    port_series = pd.Series(port_rets, index=idx, name='port_ret')
    port_eq = (1 + port_series).cumprod()
    return port_eq, pd.Series(c_weights, index=idx, name='c_weight')


def metrics(eq):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'Sharpe':0,'CAGR':0,'MDD':0,'Cal':0,'Final':0}
    bpy = 252
    sh = (rets.mean()*bpy) / (rets.std()*np.sqrt(bpy)) if rets.std() > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days/365.25
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return {'Sharpe':round(float(sh),3),'CAGR':round(float(cagr),4),
            'MDD':round(float(mdd),4),'Cal':round(float(cal),3),
            'Final':round(float(eq.iloc[-1]/eq.iloc[0]),3)}


def main():
    # V21 daily
    v21 = pd.read_csv(os.path.join(STRAT_DIR, 'v21_daily.csv'), index_col=0, parse_dates=True)
    v21['equity'] = v21['equity'] / v21['equity'].iloc[0]
    print(f'V21 단독: {metrics(v21["equity"])}')

    # C equity per universe
    UNIVERSES = {
        'BTC_only': ['BTCUSDT'],
        'BTC+ADA': ['BTCUSDT','ADAUSDT'],
        'BTC+ADA+ETH': ['BTCUSDT','ADAUSDT','ETHUSDT'],
        'Top5_all': ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT'],
        'Top10_all': ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT',
                      'DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','DOTUSDT'],
    }

    print('\n=== C 단독 (EW portfolio) ===')
    c_cache = {}
    for name, coins in UNIVERSES.items():
        c_eq = build_c_daily(coins)
        if c_eq is None: continue
        c_cache[name] = c_eq
        m = metrics(c_eq)
        print(f'  {name:>14}: Sharpe={m["Sharpe"]} CAGR={m["CAGR"]:.2%} MDD={m["MDD"]:.2%} Cal={m["Cal"]}')

    print('\n=== M3 시뮬 (Hard Cap 기반 동적 할당) ===')
    rows = []
    for name, coins in UNIVERSES.items():
        if name not in c_cache: continue
        c_eq = c_cache[name]
        for cap in [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]:
            port_eq, c_w = simulate_m3(v21, c_eq, hard_cap=cap)
            m = metrics(port_eq)
            row = {'universe': name, 'hard_cap': cap, **m, 'c_w_mean': round(c_w.mean(), 3)}
            rows.append(row)

    rdf = pd.DataFrame(rows)
    rdf.to_csv(os.path.join(STRAT_DIR, 'm3_v2_sweep.csv'), index=False)
    print(f'\nTop 10 by Cal:')
    print(rdf.sort_values('Cal', ascending=False).head(10).to_string(index=False))

    # 2021 제거 ablation (best)
    best = rdf.sort_values('Cal', ascending=False).iloc[0]
    print(f'\n=== 2021 제거 ablation (best: {best["universe"]} cap={best["hard_cap"]}) ===')
    best_c = c_cache[best['universe']]
    v21_2 = v21[v21.index >= pd.Timestamp('2022-01-01')].copy()
    v21_2['equity'] = v21_2['equity'] / v21_2['equity'].iloc[0]
    c_2 = best_c[best_c.index >= pd.Timestamp('2022-01-01')].copy()
    c_2 = c_2 / c_2.iloc[0]
    port_2, _ = simulate_m3(v21_2, c_2, hard_cap=best['hard_cap'])
    m_v21_2 = metrics(v21_2['equity'])
    m_port_2 = metrics(port_2)
    print(f'  V21 단독 (2022+): {m_v21_2}')
    print(f'  V21+C M3 (2022+): {m_port_2}')

    print(f'\n저장: {STRAT_DIR}/m3_v2_sweep.csv')


if __name__ == '__main__':
    main()
