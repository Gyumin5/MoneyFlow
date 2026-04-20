#!/usr/bin/env python3
"""최종 M3 시뮬 — 정밀 force_close 조건.

변경:
- 기존: cash 5%p 감소 시 강제 청산 (단순 heuristic)
- 신규: C 사용량 > V21 target_cash 인 경우에만 강제 청산 (정밀)
- swap 로직 유지
- n_pick 1, cap-first 기본
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
STRAT_DIR = os.path.join(HERE, 'strat_C_v3')
DATA_DIR = os.path.join(ROOT, 'data', 'futures')

START = '2020-10-01'
END = '2026-03-30'
UNIVERSE_TOP10 = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT',
                  'DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','DOTUSDT']
CAP_RANK = {c: i for i, c in enumerate(UNIVERSE_TOP10)}


def load_coin_1h(sym):
    path = os.path.join(DATA_DIR, f'{sym}_1h.csv')
    if not os.path.isfile(path): return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df.loc[START:END].copy()


def run_v3_track(df, dip_bars=24, dip_thr=-0.15, tp=0.08, tstop=24, tx=0.003):
    df = df.copy()
    df['dip_pct'] = df['Close'] / df['Close'].shift(dip_bars) - 1.0
    df['dip_sig'] = df['dip_pct'].shift(1) <= dip_thr

    eq = 10000.0
    equity, events = [], []
    position = 0; entry = 0; bars = 0; entry_ts = None
    for i, (ts, row) in enumerate(df.iterrows()):
        if position > 0:
            pnl = row['Open']/entry - 1.0
            if pnl >= tp or bars >= tstop:
                eq *= (1 + pnl - tx)
                events.append({'entry_ts':entry_ts,'exit_ts':ts,'entry_px':entry,
                              'exit_px':row['Open'],'pnl_pct':pnl*100,'bars_held':bars,
                              'reason':'TP' if pnl>=tp else 'timeout'})
                position = 0; bars = 0
        if position == 0 and row['dip_sig']:
            entry = row['High']; position = 1; bars = 0; entry_ts = ts
            eq *= (1 - tx)
            eq *= (1 + (row['Close']/entry - 1))
            bars += 1
        elif position > 0:
            prev_close = df.iloc[i-1]['Close'] if i > 0 else row['Open']
            eq *= (1 + (row['Close']/prev_close - 1))
            bars += 1
        equity.append(eq)
    return pd.Series(equity, index=df.index), events


def extract_events_all(tx=0.003):
    all_e = []
    for c in UNIVERSE_TOP10:
        df = load_coin_1h(c)
        if df is None: continue
        _, evs = run_v3_track(df, tx=tx)
        for e in evs:
            e['coin'] = c
            all_e.append(e)
    return pd.DataFrame(all_e)


def load_v21():
    v21 = pd.read_csv(os.path.join(STRAT_DIR, 'v21_daily.csv'), index_col=0, parse_dates=True)
    v21['equity'] = v21['equity'] / v21['equity'].iloc[0]
    v21['v21_ret'] = v21['equity'].pct_change().fillna(0)
    return v21


def simulate(v21, events, hard_cap, n_pick=1, select_method='cap',
             tx_cost=0.003, force_close_mode='precise', allow_swap=True):
    """
    force_close_mode:
      'none': 강제 청산 없음
      'heuristic': cash_ratio 5%p 감소 (기존)
      'precise': C 사용량 > cash_ratio 시 청산 (신규, 사용자 제안)
    """
    events = events.copy()
    events['entry_date'] = events['entry_ts'].dt.normalize()
    events['exit_date'] = events['exit_ts'].dt.normalize()
    events['cap_r'] = events['coin'].map(CAP_RANK)
    events['sort_key'] = events['cap_r'] if select_method == 'cap' else events['entry_ts'].astype('int64')
    events = events.sort_values('entry_ts').reset_index(drop=True)

    idx = v21.index
    positions = []
    port_rets = []
    prev_cash = None
    forced = 0
    swaps = 0

    for date in idx:
        cash_ratio = v21.loc[date, 'cash_ratio']
        v21_ret = v21.loc[date, 'v21_ret']
        day_exit_pnl = 0.0

        # exit (TP/timeout)
        still_open = []
        for p in positions:
            if p['exit_date'] <= date:
                day_exit_pnl += p['slot_ratio'] * (p['pnl_pct'] / 100.0 - tx_cost)
            else:
                still_open.append(p)
        positions = still_open

        # force close 조건
        c_used = sum(p['slot_ratio'] for p in positions)
        if force_close_mode == 'heuristic':
            if prev_cash is not None and cash_ratio < prev_cash - 0.05:
                for p in positions:
                    day_exit_pnl -= p['slot_ratio'] * tx_cost
                positions = []
                forced += 1
        elif force_close_mode == 'precise':
            if c_used > cash_ratio + 1e-9:
                for p in positions:
                    day_exit_pnl -= p['slot_ratio'] * tx_cost
                positions = []
                forced += 1

        # entry
        today = events[events['entry_date'] == date]
        open_slots = n_pick - len(positions)

        # swap
        if allow_swap and len(today) > 0 and len(positions) > 0:
            cur_worst_rank = max(p['cap_r'] for p in positions)
            best_new = today.sort_values('cap_r').iloc[0]
            if best_new['cap_r'] < cur_worst_rank:
                worst_p = None
                for p in positions:
                    if p['cap_r'] == cur_worst_rank:
                        worst_p = p; break
                if worst_p:
                    day_exit_pnl -= worst_p['slot_ratio'] * tx_cost
                    positions.remove(worst_p)
                    open_slots += 1
                    swaps += 1

        if open_slots > 0 and len(today) > 0:
            open_coins = {p['coin'] for p in positions}
            picks = today[~today['coin'].isin(open_coins)].sort_values('sort_key').head(open_slots)
            max_c = min(hard_cap, cash_ratio)
            slot_r = max_c / n_pick
            for _, ev in picks.iterrows():
                positions.append({
                    'coin': ev['coin'],
                    'cap_r': ev['cap_r'],
                    'entry_date': date,
                    'exit_date': ev['exit_date'],
                    'pnl_pct': ev['pnl_pct'],
                    'slot_ratio': slot_r,
                })

        c_w = sum(p['slot_ratio'] for p in positions)
        c_w = min(c_w, cash_ratio)
        port_ret = (1 - c_w) * v21_ret + day_exit_pnl
        port_rets.append(port_ret)
        prev_cash = cash_ratio

    port_eq = (1 + pd.Series(port_rets, index=idx)).cumprod()
    return port_eq, forced, swaps


def metrics(eq):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'Sharpe':0,'CAGR':0,'MDD':0,'Cal':0}
    bpy = 252
    sh = (rets.mean()*bpy) / (rets.std()*np.sqrt(bpy)) if rets.std() > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days/365.25 if days > 0 else 0.001
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return {'Sharpe':round(float(sh),3),'CAGR':round(float(cagr),4),
            'MDD':round(float(mdd),4),'Cal':round(float(cal),3)}


def main():
    v21 = load_v21()
    events = extract_events_all()
    print(f'V21 단독: {metrics(v21["equity"])}')

    print('\n=== Force Close mode 비교 (absolute Hard Cap, n_pick=1, swap ON) ===')
    print(f"{'mode':>12} {'hc':>6} {'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Cal':>6} {'forced':>7} {'swaps':>6}")
    for mode in ['none', 'heuristic', 'precise']:
        for hc in [0.20, 0.30, 0.50, 0.70]:
            eq, forced, swaps = simulate(v21, events, hc, n_pick=1,
                                          force_close_mode=mode, allow_swap=True)
            m = metrics(eq)
            print(f'{mode:>12} {hc:>6.2f} {m["Sharpe"]:>7.3f} {m["CAGR"]:>7.1%} {m["MDD"]:>7.1%} {m["Cal"]:>6.3f} {forced:>7} {swaps:>6}')

    # Best config 찾기 + 2021 제거
    print('\n=== 2021 제거 ablation (best precise config) ===')
    best_hc = 0.30
    ev_no = events[events['entry_ts'] >= pd.Timestamp('2022-01-01')]
    v21_no = v21[v21.index >= pd.Timestamp('2022-01-01')].copy()
    v21_no['equity'] = v21_no['equity'] / v21_no['equity'].iloc[0]
    v21_no['v21_ret'] = v21_no['equity'].pct_change().fillna(0)
    eq_no, forced_no, swaps_no = simulate(v21_no, ev_no, best_hc, n_pick=1,
                                           force_close_mode='precise', allow_swap=True)
    m_v21 = metrics(v21_no['equity'])
    m_port = metrics(eq_no)
    print(f'  V21 단독 (2022+): {m_v21}')
    print(f'  V21+C precise (2022+): {m_port} forced={forced_no} swaps={swaps_no}')


if __name__ == '__main__':
    main()
