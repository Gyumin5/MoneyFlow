#!/usr/bin/env python3
"""Strategy C engine v5 — 최종 확정 엔진.

v4 대비 변경:
- Lot 단위 event 생성 (slot 축소 대비)
- Fill delay 옵션 확장
- 회계 정확: entry 봉 (close-entry)/x=1 + (1-tx), hold 봉 close/prev_close, exit 봉 sell/prev_close + (1-tx)

Usage:
    from c_engine_v5 import run_c_v5, load_coin

    eq, events = run_c_v5(df, dip_bars=24, dip_thr=-0.15, tp=0.08, tstop=24,
                         tx=0.003, buy_at='high', sell_at='open', fill_delay=0)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')


def load_coin(sym, interval='1h', start='2020-10-01', end='2026-03-30'):
    path = os.path.join(DATA_DIR, f'{sym}_{interval}.csv')
    if not os.path.isfile(path): return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df.loc[start:end].copy()


def run_c_v5(df, dip_bars=24, dip_thr=-0.15, tp=0.08, tstop=24,
             tx=0.003, buy_at='high', sell_at='open', fill_delay=0):
    """
    signal 확정 (t봉 close) → t+1+fill_delay 봉 체결.
    Exit: t 봉 판정 후 다음 봉 체결 (또는 같은 봉 sell_at 가격).

    return: (equity Series, events list of dict)
    events 각각:
        entry_ts, exit_ts, entry_px, exit_px, pnl_pct, bars_held, reason
    """
    df = df.copy()
    df['dip_pct'] = df['Close'] / df['Close'].shift(dip_bars) - 1.0
    # signal shift(1+fill_delay): t-1-delay 봉 close 확정 시그널이 t 봉에 체결
    df['dip_sig'] = df['dip_pct'].shift(1 + fill_delay) <= dip_thr

    eq = 10000.0
    equity = []
    position = 0
    entry_price = 0.0
    entry_ts = None
    bars_held = 0
    events = []

    for i, (ts, row) in enumerate(df.iterrows()):
        # Exit 판정 (total_pnl 기준) + exit 체결
        if position > 0:
            sell_px = row[sell_at.capitalize()]
            prev_close = df.iloc[i-1]['Close']
            total_pnl = sell_px / entry_price - 1.0
            if total_pnl >= tp or bars_held >= tstop:
                # Exit bar 수익만 반영 (entry → prev_close 이미 반영됨)
                exit_bar_ret = sell_px / prev_close - 1.0
                eq *= (1 + exit_bar_ret - tx)
                events.append({
                    'entry_ts': entry_ts, 'exit_ts': ts,
                    'entry_px': float(entry_price), 'exit_px': float(sell_px),
                    'pnl_pct': round(float(total_pnl) * 100, 3),
                    'bars_held': int(bars_held),
                    'reason': 'TP' if total_pnl >= tp else 'timeout',
                })
                position = 0; bars_held = 0; entry_ts = None

        # Entry
        if position == 0 and row['dip_sig']:
            buy_px = row[buy_at.capitalize()]
            entry_price = float(buy_px); entry_ts = ts; position = 1; bars_held = 0
            # Entry bar: entry_price → Close 수익 + tx
            bar_ret = row['Close'] / entry_price - 1.0
            eq *= (1 + bar_ret - tx)
            bars_held += 1
        elif position > 0:
            # Hold bar: prev_close → close
            prev_close = df.iloc[i-1]['Close']
            bar_ret = row['Close'] / prev_close - 1.0
            eq *= (1 + bar_ret)
            bars_held += 1

        equity.append(eq)

    return pd.Series(equity, index=df.index), events


def extract_all_events(coin_list, tx=0.003, fill_delay=0, **kwargs):
    """Multi-coin events 추출. 각 event에 coin 컬럼 추가."""
    all_rows = []
    for sym in coin_list:
        df = load_coin(sym + 'USDT' if not sym.endswith('USDT') else sym)
        if df is None: continue
        _, evs = run_c_v5(df, tx=tx, fill_delay=fill_delay, **kwargs)
        for e in evs:
            e['coin'] = sym.replace('USDT', '')
            all_rows.append(e)
    return pd.DataFrame(all_rows)


def metrics(eq, bpy=24*365):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'Sharpe': 0, 'CAGR': 0, 'MDD': 0, 'Cal': 0, 'Final': 0}
    std = rets.std()
    sh = (rets.mean()*bpy) / (std * np.sqrt(bpy)) if std > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days/365.25 if days > 0 else 0.001
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return {'Sharpe': round(float(sh), 3), 'CAGR': round(float(cagr), 4),
            'MDD': round(float(mdd), 4), 'Cal': round(float(cal), 3),
            'Final': round(float(eq.iloc[-1]/eq.iloc[0]), 3)}
