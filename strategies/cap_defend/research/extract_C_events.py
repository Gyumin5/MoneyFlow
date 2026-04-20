#!/usr/bin/env python3
"""Step 1: 검증된 v3 single-coin engine으로 각 코인의 C events 추출.

출력: CSV, (coin, entry_ts, exit_ts, entry_px, exit_px, pnl_pct, bars_held, reason)

기존 validate_strategy_C.py의 로직(Open 체결 버전, Sharpe 1.88 검증됨)과
validate_strategy_C_v3.py의 보수 체결 (High 매수, Open 매도, Sharpe 1.28 검증됨) 두 가지.
"""
from __future__ import annotations
import os, sys
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')

START = '2020-10-01'
END = '2026-03-30'

PARAMS = {
    'dip_bars': 24,
    'dip_threshold': -0.15,
    'take_profit': 0.08,
    'time_stop_bars': 24,
    'lev': 1.0,
}
TX_COST = 0.003

UNIVERSE_TOP10 = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT',
                  'DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','DOTUSDT']


def load_coin(sym, interval='1h'):
    path = os.path.join(DATA_DIR, f'{sym}_{interval}.csv')
    if not os.path.isfile(path): return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df.loc[START:END].copy()


def run_v3_track(df, dip_bars, dip_threshold, take_profit, time_stop_bars,
                 lev=1.0, tx_cost=TX_COST, buy_at='high', sell_at='open'):
    """검증된 v3 single-coin. track_events=True."""
    df = df.copy()
    df['dip_pct'] = df['Close'] / df['Close'].shift(dip_bars) - 1.0
    df['dip_sig'] = df['dip_pct'].shift(1) <= dip_threshold

    eq = 10000.0
    equity = []
    position = 0.0
    entry_price = 0.0
    entry_ts = None
    bars_held = 0
    events = []

    for i, (ts, row) in enumerate(df.iterrows()):
        if position > 0:
            sell_px = row[sell_at.capitalize()]
            pnl = sell_px / entry_price - 1.0
            if pnl >= take_profit or bars_held >= time_stop_bars:
                eq *= (1 + lev * pnl - tx_cost)
                events.append({
                    'entry_ts': entry_ts, 'exit_ts': ts,
                    'entry_px': entry_price, 'exit_px': sell_px,
                    'pnl_pct': round(pnl * 100, 3),
                    'bars_held': bars_held,
                    'reason': 'TP' if pnl >= take_profit else 'timeout',
                })
                position = 0.0; entry_price = 0.0; bars_held = 0; entry_ts = None

        if position == 0 and row['dip_sig']:
            buy_px = row[buy_at.capitalize()]
            entry_price = buy_px; entry_ts = ts; position = 1.0; bars_held = 0
            eq *= (1 - tx_cost)
            bar_ret = row['Close'] / entry_price - 1.0
            eq *= (1 + lev * bar_ret)
            bars_held += 1
        elif position > 0:
            prev_close = df.iloc[i-1]['Close'] if i > 0 else row['Open']
            bar_ret = row['Close'] / prev_close - 1.0
            eq *= (1 + lev * bar_ret)
            bars_held += 1

        if eq < 0: eq = 0
        equity.append(eq)

    return pd.Series(equity, index=df.index), events


def main():
    all_events = []
    out_dir = os.path.join(HERE, 'strat_C_v3')
    os.makedirs(out_dir, exist_ok=True)

    print('=== Step 1: C events 추출 (v3 검증 engine, 보수 체결) ===')
    for coin in UNIVERSE_TOP10:
        df = load_coin(coin, '1h')
        if df is None:
            print(f'  {coin}: 데이터 없음')
            continue
        eq, events = run_v3_track(df, **PARAMS, buy_at='high', sell_at='open')
        rows = [{'coin': coin, **e} for e in events]
        all_events.extend(rows)

        # 지표도 계산
        import numpy as np
        rets = eq.pct_change().dropna()
        bpy = 24*365
        if len(rets) > 0 and rets.std() > 0:
            sh = (rets.mean()*bpy) / (rets.std()*np.sqrt(bpy))
        else:
            sh = 0
        days = (eq.index[-1] - eq.index[0]).days
        cagr = (eq.iloc[-1]/eq.iloc[0])**(365.25/days) - 1 if days > 0 else 0
        mdd = (eq / eq.cummax() - 1).min()
        cal = cagr/abs(mdd) if mdd < 0 else 0
        print(f'  {coin}: events={len(events):3d} Sharpe={sh:.3f} CAGR={cagr:.2%} MDD={mdd:.2%} Cal={cal:.3f}')

    edf = pd.DataFrame(all_events)
    if len(edf):
        edf['entry_ts'] = pd.to_datetime(edf['entry_ts'])
        edf['exit_ts'] = pd.to_datetime(edf['exit_ts'])
        edf = edf.sort_values('entry_ts').reset_index(drop=True)
        edf.to_csv(os.path.join(out_dir, 'events_top10.csv'), index=False)
        print(f'\n총 events: {len(edf)}')
        print(f'코인별 발동 횟수:')
        print(edf['coin'].value_counts().to_string())
        print(f'\nSaved: {out_dir}/events_top10.csv')


if __name__ == '__main__':
    main()
