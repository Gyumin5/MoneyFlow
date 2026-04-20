#!/usr/bin/env python3
"""Strategy C: Short-term Mean Reversion (dip scalp).

극단 급락 후 bounce 단기 long.
- 1h 봉 기준
- 조건: N시간 누적 수익률 <= -X% (strong selloff)
- 청산: +Y% 또는 M봉 후
- 레버리지 1x 또는 2x
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')

START = '2020-10-01'
END = '2026-03-30'
TX_COST = 0.0004
INITIAL_CAPITAL = 10000.0


def load_btc(interval='1h'):
    df = pd.read_csv(os.path.join(DATA_DIR, f'BTCUSDT_{interval}.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df.loc[START:END].copy()


def run_strategy(df, dip_bars, dip_threshold, take_profit, time_stop_bars, lev=1.0):
    """Simple dip-buy:
    - signal: close 대비 dip_bars 전 close 하락율 <= dip_threshold (예: -5%)
    - 진입: 다음 봉 open
    - 청산: take_profit 도달 OR time_stop_bars 경과 (open 체결)
    - no overlap: 기존 포지션 있으면 신규 스킵
    """
    df = df.copy()
    df['dip_pct'] = df['Close'] / df['Close'].shift(dip_bars) - 1.0
    # t-1 close 기준 시그널
    df['dip_sig'] = df['dip_pct'].shift(1) <= dip_threshold

    eq = INITIAL_CAPITAL
    equity = []
    position = 0.0
    entry_price = 0.0
    bars_held = 0

    for i, (ts, row) in enumerate(df.iterrows()):
        prev_eq = eq

        # 청산 체크 (bar open에)
        if position > 0:
            open_px = row['Open']
            pnl_ratio = (open_px / entry_price - 1.0)
            if pnl_ratio >= take_profit or bars_held >= time_stop_bars:
                # 청산 at open
                eq *= (1 + lev * pnl_ratio - TX_COST)  # 진입 비용은 이미 뺐으니 청산만
                position = 0.0
                entry_price = 0.0
                bars_held = 0

        # 진입 체크 (bar open에, 기존 포지션 없으면)
        if position == 0 and row['dip_sig']:
            entry_price = row['Open']
            position = 1.0
            bars_held = 0
            eq *= (1 - TX_COST)  # 진입 비용
            # bar_ret close까지 (현재 봉)
            bar_ret = row['Close'] / row['Open'] - 1.0
            eq *= (1 + lev * bar_ret)
            bars_held += 1
        elif position > 0:
            # 포지션 유지, bar close까지 수익
            bar_ret = row['Close'] / (df.iloc[i-1]['Close'] if i > 0 else row['Open']) - 1.0
            eq *= (1 + lev * bar_ret)
            bars_held += 1
        # else: flat, eq 유지

        if eq < 0: eq = 0
        equity.append(eq)

    return pd.Series(equity, index=df.index)


def metrics(eq, interval='1h'):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'Sharpe': 0, 'CAGR': 0, 'MDD': 0, 'Cal': 0, 'Final': 0}
    bars_per_year = {'1h': 24*365, '4h': 6*365, '15m': 96*365, '30m': 48*365}[interval]
    sharpe = (rets.mean() * bars_per_year) / (rets.std() * np.sqrt(bars_per_year)) if rets.std() > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else 0.001
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {'Sharpe': round(sharpe, 3), 'CAGR': round(cagr, 4),
            'MDD': round(mdd, 4), 'Cal': round(cal, 3), 'Final': round(eq.iloc[-1]/eq.iloc[0], 3)}


def main():
    print('[C: Mean Reversion Dip Buy]')
    df_1h = load_btc('1h')
    print(f'  BTC 1h bars: {len(df_1h)}')
    df_4h = load_btc('4h')
    print(f'  BTC 4h bars: {len(df_4h)}')

    # Grid sweep: 1h first
    results = []
    for interval, df in [('1h', df_1h), ('4h', df_4h)]:
        for dip_bars in [6, 12, 24, 48, 72] if interval == '1h' else [3, 6, 12, 24]:
            for dip_thr in [-0.05, -0.08, -0.10, -0.15, -0.20]:
                for tp in [0.03, 0.05, 0.08, 0.12]:
                    for tstop in [24, 48, 120] if interval == '1h' else [6, 12, 30]:
                        for lev in [1.0, 2.0]:
                            eq = run_strategy(df, dip_bars, dip_thr, tp, tstop, lev)
                            m = metrics(eq, interval)
                            m['label'] = f'{interval}_dip{dip_bars}_thr{dip_thr}_tp{tp}_ts{tstop}_lev{lev}'
                            m['interval'] = interval
                            results.append(m)

    rdf = pd.DataFrame(results)
    rdf = rdf.sort_values('Cal', ascending=False)

    print(f'\nTotal configs: {len(rdf)}')
    print(f'\nTop 10 by Cal:')
    top = rdf.head(10)
    cols = ['label', 'Sharpe', 'CAGR', 'MDD', 'Cal', 'Final']
    print(top[cols].to_string(index=False))
    print(f'\nTop 5 by Sharpe:')
    print(rdf.sort_values('Sharpe', ascending=False).head(5)[cols].to_string(index=False))

    # Best results saved
    out_dir = os.path.join(HERE, 'strat_C_mr')
    os.makedirs(out_dir, exist_ok=True)
    rdf.to_csv(os.path.join(out_dir, 'sweep.csv'), index=False)
    print(f'\nSaved: {out_dir}/sweep.csv')


if __name__ == '__main__':
    main()
