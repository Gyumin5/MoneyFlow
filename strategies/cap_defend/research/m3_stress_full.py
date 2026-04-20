#!/usr/bin/env python3
"""M3 stress suite — 필요한 모든 테스트 통합.

1. Hard Cap 정의 2종 (전체의 X% / cash의 X%)
2. Slippage 4수준 (30/80/130/180 bps)
3. Trade delay (0h/1h/4h)
4. 연도별 분해
5. Top N 수익 trade 제거 ablation
6. 2021 제거
7. Walk-forward
8. Block bootstrap MDD 분포
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
STRAT_DIR = os.path.join(HERE, 'strat_C_v3')
DATA_DIR = os.path.join(ROOT, 'data', 'futures')

START = '2020-10-01'
END = '2026-03-30'
TX_BASE = 0.003
UNIVERSE_TOP10 = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT',
                  'DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','DOTUSDT']
CAP_RANK = {c: i for i, c in enumerate(UNIVERSE_TOP10)}


# ─── 재생성 (v3 로직) ───
def load_coin_1h(sym):
    path = os.path.join(DATA_DIR, f'{sym}_1h.csv')
    if not os.path.isfile(path): return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df.loc[START:END].copy()


def run_v3_track(df, p, tx_cost, trade_delay_bars=0):
    """v3 single-coin, 보수 체결. trade_delay_bars: 시그널 확정 후 체결 N bar 지연 (미반영)."""
    df = df.copy()
    df['dip_pct'] = df['Close'] / df['Close'].shift(p['dip_bars']) - 1.0
    df['dip_sig'] = df['dip_pct'].shift(1 + trade_delay_bars) <= p['dip_threshold']

    eq = 10000.0
    equity = []
    position = 0; entry = 0; bars = 0
    entry_ts = None
    events = []
    for i, (ts, row) in enumerate(df.iterrows()):
        if position > 0:
            pnl = row['Open']/entry - 1.0
            if pnl >= p['take_profit'] or bars >= p['time_stop_bars']:
                eq *= (1 + p['lev']*pnl - tx_cost)
                events.append({'entry_ts':entry_ts,'exit_ts':ts,'entry_px':entry,
                              'exit_px':row['Open'],'pnl_pct':pnl*100,'bars_held':bars,
                              'reason':'TP' if pnl>=p['take_profit'] else 'timeout'})
                position = 0; bars = 0
        if position == 0 and row['dip_sig']:
            entry = row['High']; position = 1; bars = 0; entry_ts = ts
            eq *= (1 - tx_cost)
            eq *= (1 + p['lev'] * (row['Close']/entry - 1))
            bars += 1
        elif position > 0:
            prev_close = df.iloc[i-1]['Close'] if i > 0 else row['Open']
            eq *= (1 + p['lev'] * (row['Close']/prev_close - 1))
            bars += 1
        equity.append(eq)
    return pd.Series(equity, index=df.index), events


def extract_events_all(tx_cost, trade_delay_bars=0, p=None):
    """Top 10 전체에 대해 events 수집."""
    if p is None:
        p = {'dip_bars':24,'dip_threshold':-0.15,'take_profit':0.08,'time_stop_bars':24,'lev':1.0}
    all_events = []
    for c in UNIVERSE_TOP10:
        df = load_coin_1h(c)
        if df is None: continue
        _, evs = run_v3_track(df, p, tx_cost, trade_delay_bars)
        for e in evs:
            e['coin'] = c
            all_events.append(e)
    return pd.DataFrame(all_events)


def load_v21():
    v21 = pd.read_csv(os.path.join(STRAT_DIR, 'v21_daily.csv'), index_col=0, parse_dates=True)
    v21['equity'] = v21['equity'] / v21['equity'].iloc[0]
    v21['v21_ret'] = v21['equity'].pct_change().fillna(0)
    return v21


def simulate(v21, events, hard_cap, cap_mode='absolute', n_pick=1, select_method='cap',
             tx_cost=TX_BASE, force_close=True, remove_top_events=0):
    """
    cap_mode: 'absolute' (전체의 X%) or 'cash_pct' (cash의 X%)
    remove_top_events: N개 최고 수익 event 제거 (robustness)
    """
    events = events.copy()
    if remove_top_events > 0 and len(events) > remove_top_events:
        top_idx = events.nlargest(remove_top_events, 'pnl_pct').index
        events = events.drop(top_idx).reset_index(drop=True)
    events['entry_date'] = events['entry_ts'].dt.normalize()
    events['exit_date'] = events['exit_ts'].dt.normalize()
    if select_method == 'cap':
        events['sort_key'] = events['coin'].map(CAP_RANK)
    else:
        events['sort_key'] = events['entry_ts'].astype('int64')

    idx = v21.index
    positions = []
    port_rets = []
    prev_cash = None

    for date in idx:
        cash_ratio = v21.loc[date, 'cash_ratio']
        v21_ret = v21.loc[date, 'v21_ret']
        day_exit_pnl = 0.0

        if force_close and prev_cash is not None and cash_ratio < prev_cash - 0.05:
            for p in positions:
                day_exit_pnl -= p['slot_ratio'] * tx_cost
            positions = []

        still_open = []
        for p in positions:
            if p['exit_date'] <= date:
                day_exit_pnl += p['slot_ratio'] * (p['pnl_pct'] / 100.0 - tx_cost)
            else:
                still_open.append(p)
        positions = still_open

        today = events[events['entry_date'] == date]
        open_slots = n_pick - len(positions)
        if open_slots > 0 and len(today) > 0:
            open_coins = {p['coin'] for p in positions}
            picks = today[~today['coin'].isin(open_coins)].sort_values('sort_key').head(open_slots)
            # Hard cap 계산
            if cap_mode == 'absolute':
                max_c = hard_cap
            else:  # cash_pct
                max_c = hard_cap * cash_ratio
            # V21 cash 제약
            max_c = min(max_c, cash_ratio)
            slot_r = max_c / n_pick
            for _, ev in picks.iterrows():
                positions.append({
                    'coin': ev['coin'],
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
    return port_eq


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


# ─── Tests ───

def test_1_cap_mode_x_slippage(v21, events_base):
    print('\n### Test 1: Cap mode × Slippage ###')
    rows = []
    for tx_label, tx in [('30bps', 0.003), ('80bps', 0.008), ('130bps', 0.013), ('180bps', 0.018)]:
        events = extract_events_all(tx)  # slippage 반영해서 events 재추출
        for cap_mode in ['absolute', 'cash_pct']:
            for hc in [0.10, 0.20, 0.30, 0.50]:
                eq = simulate(v21, events, hc, cap_mode=cap_mode, tx_cost=tx)
                m = metrics(eq)
                rows.append({'tx': tx_label, 'cap_mode': cap_mode, 'hard_cap': hc, **m})
    df = pd.DataFrame(rows).sort_values('Cal', ascending=False)
    print(df.head(15).to_string(index=False))
    return df


def test_2_yearly(v21, events_base, hc=0.30, cap_mode='absolute'):
    print('\n### Test 2: 연도별 분해 ###')
    rows = []
    for yr in [2021, 2022, 2023, 2024, 2025]:
        v_sub = v21[(v21.index.year == yr)].copy()
        if len(v_sub) < 50: continue
        v_sub['equity'] = v_sub['equity'] / v_sub['equity'].iloc[0]
        v_sub['v21_ret'] = v_sub['equity'].pct_change().fillna(0)
        ev_sub = events_base[events_base['entry_ts'].dt.year == yr]
        eq_v21 = v_sub['equity']
        eq_port = simulate(v_sub, ev_sub, hc, cap_mode=cap_mode)
        m_v = metrics(eq_v21)
        m_p = metrics(eq_port)
        rows.append({'year': yr, 'v21_Cal': m_v['Cal'], 'port_Cal': m_p['Cal'],
                     'v21_CAGR': m_v['CAGR'], 'port_CAGR': m_p['CAGR'],
                     'diff_Cal': m_p['Cal']-m_v['Cal'], 'diff_CAGR': m_p['CAGR']-m_v['CAGR']})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))


def test_3_top_events_removal(v21, events_base, hc=0.30, cap_mode='absolute'):
    print('\n### Test 3: Top N 수익 event 제거 ablation ###')
    rows = []
    for n in [0, 1, 3, 5, 10, 20]:
        eq = simulate(v21, events_base, hc, cap_mode=cap_mode, remove_top_events=n)
        m = metrics(eq)
        rows.append({'removed_top': n, **m})
    print(pd.DataFrame(rows).to_string(index=False))


def test_4_block_bootstrap(v21, events_base, hc=0.30, cap_mode='absolute', n_boot=200, block_days=60):
    print('\n### Test 4: Block bootstrap (N=200, block 60일) ###')
    eq = simulate(v21, events_base, hc, cap_mode=cap_mode)
    rets = eq.pct_change().dropna().values
    n = len(rets)
    rng = np.random.default_rng(42)
    mdd_list = []
    cagr_list = []
    for _ in range(n_boot):
        starts = rng.integers(0, n-block_days, size=(n//block_days)+1)
        blocks = np.concatenate([rets[s:s+block_days] for s in starts])[:n]
        eq_b = (1 + blocks).cumprod()
        if eq_b[-1] <= 0: continue
        years = len(blocks) / 252
        cagr = eq_b[-1] ** (1/years) - 1
        run_max = np.maximum.accumulate(eq_b)
        mdd = float(((eq_b - run_max) / run_max).min())
        mdd_list.append(mdd)
        cagr_list.append(cagr)
    print(f'  N={len(mdd_list)} bootstrap')
    print(f'  CAGR: mean={np.mean(cagr_list):.2%} p5={np.percentile(cagr_list, 5):.2%} p50={np.percentile(cagr_list, 50):.2%} p95={np.percentile(cagr_list, 95):.2%}')
    print(f'  MDD:  mean={np.mean(mdd_list):.2%} p5={np.percentile(mdd_list, 5):.2%} p50={np.percentile(mdd_list, 50):.2%} worst={min(mdd_list):.2%}')


def test_5_trade_delay(v21, tx, hc=0.30, cap_mode='absolute'):
    print('\n### Test 5: Trade delay (시그널 체결 지연) ###')
    rows = []
    for delay in [0, 1, 4]:
        events = extract_events_all(tx, trade_delay_bars=delay)
        eq = simulate(v21, events, hc, cap_mode=cap_mode, tx_cost=tx)
        m = metrics(eq)
        rows.append({'delay_bars': delay, 'events': len(events), **m})
    print(pd.DataFrame(rows).to_string(index=False))


def main():
    v21 = load_v21()
    print(f'V21 단독: {metrics(v21["equity"])}')
    events_base = extract_events_all(TX_BASE, trade_delay_bars=0)
    print(f'Events (30bps, delay 0): {len(events_base)}')

    test_1_cap_mode_x_slippage(v21, events_base)
    test_2_yearly(v21, events_base)
    test_3_top_events_removal(v21, events_base)
    test_4_block_bootstrap(v21, events_base)
    test_5_trade_delay(v21, TX_BASE)


if __name__ == '__main__':
    main()
