#!/usr/bin/env python3
"""Strategy A v2: 더 엄격한 short 필터 + 레버리지 축소.

v1 실패 원인:
- 조건 단순 (OFF + mom<0)으로 휩쏘
- 청산 너무 빠름
- 3x 증폭

v2 개선:
- 2-stage 확인: SMA 하회 N봉 지속 + Mom60 < -5%
- 청산: SMA 회복 + M봉 지속
- 레버리지 sweep: 1x, 2x, 3x
- 비중 sweep (포지션 크기)
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


def load_btc_4h():
    df = pd.read_csv(os.path.join(DATA_DIR, 'BTCUSDT_4h.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df.loc[START:END].copy()


def compute_signals(df, sma_bars, mom_bars, mom_thr, confirm_bars):
    df = df.copy()
    df['SMA'] = df['Close'].rolling(sma_bars).mean()
    df['Mom'] = df['Close'] / df['Close'].shift(mom_bars) - 1.0
    df['below_sma'] = df['Close'] < df['SMA']
    df['above_sma'] = df['Close'] > df['SMA']
    # 연속 N봉 이상 하회 확인
    df['below_consec'] = df['below_sma'].rolling(confirm_bars).sum() == confirm_bars
    df['above_consec'] = df['above_sma'].rolling(confirm_bars).sum() == confirm_bars
    # t-1 시그널
    df['below_sig'] = df['below_consec'].shift(1)
    df['above_sig'] = df['above_consec'].shift(1)
    df['mom_sig'] = df['Mom'].shift(1)
    df['mom_strong_down'] = df['mom_sig'] < mom_thr   # 예: -0.05
    return df


def run_backtest(df, leverage, weight):
    """단순화: 포지션 = 0 or -weight. 포지션 보유 중에도 매일 체결가는 close, 수익률 계산은 bar 기준."""
    df = df.dropna(subset=['SMA', 'Mom', 'below_sig', 'above_sig']).copy()
    eq = INITIAL_CAPITAL
    position = 0.0
    equity = []
    for i, (ts, row) in enumerate(df.iterrows()):
        # 시그널 기반 포지션 결정
        prev_position = position
        if position == 0:
            if row['below_sig'] and row['mom_strong_down']:
                position = -weight
        elif position < 0:
            if row['above_sig']:
                position = 0

        # bar 수익
        if i > 0:
            ret = (row['Close'] / df.iloc[i-1]['Close']) - 1.0
        else:
            ret = 0.0

        tx = 0.0
        if position != prev_position:
            tx = TX_COST * abs(position - prev_position)

        eq *= (1 + prev_position * leverage * ret - tx)
        if eq < 0: eq = 0
        equity.append(eq)
    return pd.Series(equity, index=df.index)


def metrics(eq, label=''):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'label': label, 'Sharpe': None, 'CAGR': None, 'MDD': None, 'Cal': None, 'Final': 0}
    bars_per_year = 6 * 365
    sharpe = (rets.mean() * bars_per_year) / (rets.std() * np.sqrt(bars_per_year)) if rets.std() > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else 0.001
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {'label': label, 'Sharpe': round(sharpe, 3), 'CAGR': round(cagr, 4),
            'MDD': round(mdd, 4), 'Cal': round(cal, 3), 'Final': round(eq.iloc[-1]/eq.iloc[0], 3)}


def main():
    df = load_btc_4h()
    print(f'[A v2] BTC 4h bars: {len(df)}')
    print(f'  range: {df.index[0]} ~ {df.index[-1]}')

    # Grid sweep
    results = []
    for sma in [240, 360]:              # 40d, 60d
        for mom_bars in [180, 360, 540]:  # 30d, 60d, 90d
            for mom_thr in [-0.05, -0.10, -0.15]:
                for confirm in [6, 12, 24]:  # 1d, 2d, 4d 지속
                    for lev in [1.0, 2.0]:
                        for weight in [0.5, 1.0]:
                            sig = compute_signals(df, sma, mom_bars, mom_thr, confirm)
                            eq = run_backtest(sig, lev, weight)
                            m = metrics(eq, f'sma{sma}_m{mom_bars}_thr{mom_thr}_conf{confirm}_lev{lev}_w{weight}')
                            results.append(m)

    rdf = pd.DataFrame(results).sort_values('Cal', ascending=False)
    print(f'\nTop 10 by Cal (양수만):')
    top = rdf[rdf['Cal'] > 0].head(10)
    if len(top):
        print(top.to_string(index=False))
    else:
        print('양수 Cal 없음')
    print(f'\nTop 10 by Sharpe:')
    top_sh = rdf[rdf['Sharpe'].notna()].sort_values('Sharpe', ascending=False).head(10)
    print(top_sh.to_string(index=False))

    out = os.path.join(HERE, 'strat_A_short', 'v2_sweep.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    rdf.to_csv(out, index=False)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
