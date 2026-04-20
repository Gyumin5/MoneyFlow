#!/usr/bin/env python3
"""Strategy C 정확한 회계 engine v4.

v3 버그 수정:
1. Entry 봉에 tx + (close/entry) 반영, exit 봉에 prev_close → sell_px만 반영 (double counting 제거)
2. Hold 봉 close/prev_close 반영 (정상)
3. Exit 시 total_pnl로 판정하되 eq에는 exit bar ret만 곱

Signal-fill 분리 옵션:
- fill_delay_bars: 시그널 확정 후 N봉 뒤 체결
  (0=same bar High, 1=next bar Open, ...)

미체결 모델:
- miss_high: High 매수 시도에서 호가 부재로 fail 가능 확률 (단순 확률 파라미터)
  (실제 검증은 별도 — 여기선 옵션으로)
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')
STRAT_DIR = os.path.join(HERE, 'strat_C_v3')

START = '2020-10-01'
END = '2026-03-30'
UNIVERSE_TOP10 = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT',
                  'DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','DOTUSDT']
CAP_RANK = {c: i for i, c in enumerate(UNIVERSE_TOP10)}


def load_coin(sym, interval='1h'):
    path = os.path.join(DATA_DIR, f'{sym}_{interval}.csv')
    if not os.path.isfile(path): return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df.loc[START:END].copy()


def run_c_v4(df, dip_bars=24, dip_thr=-0.15, tp=0.08, tstop=24,
             tx=0.003, buy_at='high', sell_at='open', fill_delay=0):
    """정확한 회계. fill_delay=0이면 시그널 확정 후 다음 봉에 체결 (기본).
    fill_delay=N이면 시그널 확정 후 N+1봉 뒤 체결.
    """
    df = df.copy()
    df['dip_pct'] = df['Close'] / df['Close'].shift(dip_bars) - 1.0
    # 시그널 t봉 close 시점 확정 → t+1+fill_delay 봉 체결
    df['dip_sig'] = df['dip_pct'].shift(1 + fill_delay) <= dip_thr

    eq = 10000.0
    equity = []
    position = 0
    entry_price = 0.0
    entry_ts = None
    bars = 0
    events = []

    for i, (ts, row) in enumerate(df.iterrows()):
        # Exit 판정 (현재 bar open에)
        if position > 0:
            prev_close = df.iloc[i-1]['Close']
            sell_px_candidate = row[sell_at.capitalize()]  # next bar open
            total_pnl = sell_px_candidate / entry_price - 1.0
            if total_pnl >= tp or bars >= tstop:
                # exit bar 수익: prev_close → sell_px (올바른 방식)
                exit_bar_ret = sell_px_candidate / prev_close - 1.0
                eq *= (1 + exit_bar_ret - tx)
                events.append({
                    'entry_ts': entry_ts, 'exit_ts': ts,
                    'entry_px': entry_price, 'exit_px': sell_px_candidate,
                    'pnl_pct': total_pnl * 100,
                    'bars_held': bars,
                    'reason': 'TP' if total_pnl >= tp else 'timeout',
                })
                position = 0; bars = 0; entry_ts = None

        # Entry 판정
        if position == 0 and row['dip_sig']:
            buy_px = row[buy_at.capitalize()]
            entry_price = buy_px; entry_ts = ts; position = 1; bars = 0
            # entry 봉 수익: entry → close (진입가 기준)
            bar_ret = row['Close'] / entry_price - 1.0
            eq *= (1 + bar_ret - tx)  # tx 1회 (진입)
            bars += 1
        elif position > 0:
            prev_close = df.iloc[i-1]['Close']
            bar_ret = row['Close'] / prev_close - 1.0
            eq *= (1 + bar_ret)
            bars += 1
        # else: flat, eq 유지

        equity.append(eq)

    return pd.Series(equity, index=df.index), events


def extract_events_all(tx=0.003, fill_delay=0, **kwargs):
    all_e = []
    for c in UNIVERSE_TOP10:
        df = load_coin(c)
        if df is None: continue
        _, evs = run_c_v4(df, tx=tx, fill_delay=fill_delay, **kwargs)
        for e in evs:
            e['coin'] = c
            all_e.append(e)
    return pd.DataFrame(all_e)


def metrics(eq, bpy=24*365):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'Sharpe':0,'CAGR':0,'MDD':0,'Cal':0,'Final':0}
    sh = (rets.mean()*bpy) / (rets.std()*np.sqrt(bpy)) if rets.std() > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days/365.25 if days > 0 else 0.001
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return {'Sharpe':round(float(sh),3),'CAGR':round(float(cagr),4),
            'MDD':round(float(mdd),4),'Cal':round(float(cal),3),
            'Final':round(float(eq.iloc[-1]/eq.iloc[0]),3)}
