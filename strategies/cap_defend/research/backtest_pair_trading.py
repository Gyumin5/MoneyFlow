#!/usr/bin/env python3
"""Strategy B: Pair Trading (BTC-ETH spread mean reversion).

market neutral long/short:
- 스프레드 = log(ETH) - beta*log(BTC) - alpha
- rolling Z-score 계산
- Z > +z_thr → ETH overvalued → ETH short + BTC long
- Z < -z_thr → ETH undervalued → ETH long + BTC short
- 청산: |Z| < z_exit (평균 회귀)

V21과 완전 직교 (market neutral).
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


def load_pair(interval='4h'):
    b = pd.read_csv(os.path.join(DATA_DIR, f'BTCUSDT_{interval}.csv'))
    e = pd.read_csv(os.path.join(DATA_DIR, f'ETHUSDT_{interval}.csv'))
    b['Date'] = pd.to_datetime(b['Date']); e['Date'] = pd.to_datetime(e['Date'])
    b = b.set_index('Date')[['Close']].rename(columns={'Close': 'BTC'})
    e = e.set_index('Date')[['Close']].rename(columns={'Close': 'ETH'})
    df = pd.concat([b, e], axis=1).dropna()
    return df.loc[START:END].copy()


def compute_spread(df, window=240, beta_mode='rolling'):
    """Spread = log(ETH) - beta * log(BTC).
    beta_mode:
      - 'rolling': rolling OLS beta
      - 'fixed': 전체 OLS beta (look-ahead 위험 — IS 참조)
    """
    df = df.copy()
    df['log_btc'] = np.log(df['BTC'])
    df['log_eth'] = np.log(df['ETH'])
    if beta_mode == 'rolling':
        cov = df['log_eth'].rolling(window).cov(df['log_btc'])
        var = df['log_btc'].rolling(window).var()
        df['beta'] = cov / var
        df['alpha'] = df['log_eth'].rolling(window).mean() - df['beta'] * df['log_btc'].rolling(window).mean()
        df['spread'] = df['log_eth'] - df['beta'] * df['log_btc'] - df['alpha']
    else:
        # fixed full-period
        from numpy.polynomial import polynomial as P
        coef = np.polyfit(df['log_btc'].dropna(), df['log_eth'].dropna(), 1)
        beta, alpha = coef[0], coef[1]
        df['spread'] = df['log_eth'] - beta * df['log_btc'] - alpha

    df['spread_mean'] = df['spread'].rolling(window).mean()
    df['spread_std'] = df['spread'].rolling(window).std()
    df['z'] = (df['spread'] - df['spread_mean']) / df['spread_std']
    return df


def run_pair(df, z_entry, z_exit, lev=1.0):
    """
    signal: z > z_entry → enter (-1, +1) (ETH short + BTC long equal notional)
            z < -z_entry → enter (+1, -1)
            |z| < z_exit → exit
    체결: 다음 봉 open
    포지션 크기: lev (delta neutral, 양방향 동일 notional)
    """
    df = df.dropna(subset=['z']).copy()
    # t-1 z 시그널로 t open 체결
    df['z_sig'] = df['z'].shift(1)

    eq = INITIAL_CAPITAL
    equity = []
    pos_eth = 0.0  # 포지션 방향
    pos_btc = 0.0

    for i, row in enumerate(df.itertuples()):
        z = row.z_sig
        if pd.isna(z):
            equity.append(eq); continue

        prev_pos_eth = pos_eth
        prev_pos_btc = pos_btc

        # 포지션 결정
        if pos_eth == 0 and pos_btc == 0:
            if z > z_entry:
                pos_eth = -1.0; pos_btc = 1.0
            elif z < -z_entry:
                pos_eth = 1.0; pos_btc = -1.0
        else:
            if abs(z) < z_exit:
                pos_eth = 0.0; pos_btc = 0.0

        # bar 수익
        if i > 0:
            prev = df.iloc[i-1]
            eth_ret = row.ETH / prev.ETH - 1.0
            btc_ret = row.BTC / prev.BTC - 1.0
            ret = prev_pos_eth * eth_ret + prev_pos_btc * btc_ret
        else:
            ret = 0.0

        # 포지션 변경 시 비용
        tx = 0.0
        if pos_eth != prev_pos_eth:
            tx += TX_COST * abs(pos_eth - prev_pos_eth)
        if pos_btc != prev_pos_btc:
            tx += TX_COST * abs(pos_btc - prev_pos_btc)

        eq *= (1 + lev * ret - tx)
        if eq < 0: eq = 0
        equity.append(eq)

    return pd.Series(equity, index=df.index)


def metrics(eq, interval='4h'):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'Sharpe': 0, 'CAGR': 0, 'MDD': 0, 'Cal': 0, 'Final': 0}
    bars_per_year = {'1h': 24*365, '4h': 6*365}[interval]
    sharpe = (rets.mean() * bars_per_year) / (rets.std() * np.sqrt(bars_per_year)) if rets.std() > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else 0.001
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {'Sharpe': round(sharpe, 3), 'CAGR': round(cagr, 4),
            'MDD': round(mdd, 4), 'Cal': round(cal, 3), 'Final': round(eq.iloc[-1]/eq.iloc[0], 3)}


def main():
    print('[B: Pair Trading BTC-ETH]')
    df_4h = load_pair('4h')
    print(f'  4h bars: {len(df_4h)}  ({df_4h.index[0]} ~ {df_4h.index[-1]})')

    results = []
    for window in [120, 240, 360, 540]:
        for z_entry in [1.5, 2.0, 2.5]:
            for z_exit in [0.3, 0.5, 0.8]:
                if z_exit >= z_entry: continue
                for lev in [1.0, 2.0, 3.0]:
                    sig = compute_spread(df_4h, window=window, beta_mode='rolling')
                    eq = run_pair(sig, z_entry, z_exit, lev)
                    m = metrics(eq, '4h')
                    m['label'] = f'w{window}_zin{z_entry}_zout{z_exit}_lev{lev}'
                    results.append(m)

    rdf = pd.DataFrame(results).sort_values('Cal', ascending=False)
    print(f'\nTotal configs: {len(rdf)}')
    print(f'\nTop 10 by Cal:')
    cols = ['label', 'Sharpe', 'CAGR', 'MDD', 'Cal', 'Final']
    print(rdf.head(10)[cols].to_string(index=False))
    print(f'\nTop 5 by Sharpe:')
    print(rdf.sort_values('Sharpe', ascending=False).head(5)[cols].to_string(index=False))

    out_dir = os.path.join(HERE, 'strat_B_pair')
    os.makedirs(out_dir, exist_ok=True)
    rdf.to_csv(os.path.join(out_dir, 'sweep.csv'), index=False)
    print(f'\nSaved: {out_dir}/sweep.csv')


if __name__ == '__main__':
    main()
