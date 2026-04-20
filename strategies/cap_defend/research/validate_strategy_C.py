#!/usr/bin/env python3
"""Strategy C (Mean Reversion dip-buy) 정교 검증.

1. 발동 이벤트 열거
2. Sub-period rank-sum (10 semi-annual windows)
3. Parameter sensitivity
4. Slippage stress test
5. 여러 코인 cross-validation
6. Walk-forward
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')

START = '2020-10-01'
END = '2026-03-30'

# 기본 최적 파라미터
BEST_PARAMS = {
    'dip_bars': 24,
    'dip_threshold': -0.15,
    'take_profit': 0.08,
    'time_stop_bars': 24,
    'lev': 1.0,
}
INITIAL_CAPITAL = 10000.0


def load_coin(sym, interval='1h'):
    path = os.path.join(DATA_DIR, f'{sym}_{interval}.csv')
    if not os.path.isfile(path): return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df.loc[START:END].copy()


def run_strategy(df, dip_bars, dip_threshold, take_profit, time_stop_bars, lev=1.0, tx_cost=0.0004, track_events=False):
    df = df.copy()
    df['dip_pct'] = df['Close'] / df['Close'].shift(dip_bars) - 1.0
    df['dip_sig'] = df['dip_pct'].shift(1) <= dip_threshold

    eq = INITIAL_CAPITAL
    equity = []
    position = 0.0
    entry_price = 0.0
    entry_ts = None
    bars_held = 0
    events = []

    for i, (ts, row) in enumerate(df.iterrows()):
        if position > 0:
            open_px = row['Open']
            pnl_ratio = (open_px / entry_price - 1.0)
            if pnl_ratio >= take_profit or bars_held >= time_stop_bars:
                eq *= (1 + lev * pnl_ratio - tx_cost)
                if track_events:
                    events.append({
                        'entry_ts': entry_ts, 'exit_ts': ts,
                        'entry_px': entry_price, 'exit_px': open_px,
                        'pnl_pct': round(pnl_ratio * 100, 2),
                        'bars_held': bars_held,
                        'reason': 'TP' if pnl_ratio >= take_profit else 'timeout',
                    })
                position = 0.0; entry_price = 0.0; bars_held = 0; entry_ts = None

        if position == 0 and row['dip_sig']:
            entry_price = row['Open']; entry_ts = ts; position = 1.0; bars_held = 0
            eq *= (1 - tx_cost)
            bar_ret = row['Close'] / row['Open'] - 1.0
            eq *= (1 + lev * bar_ret)
            bars_held += 1
        elif position > 0:
            bar_ret = row['Close'] / (df.iloc[i-1]['Close'] if i > 0 else row['Open']) - 1.0
            eq *= (1 + lev * bar_ret)
            bars_held += 1

        if eq < 0: eq = 0
        equity.append(eq)

    eq_series = pd.Series(equity, index=df.index)
    return eq_series, events


def metrics(eq, bars_per_year=24*365):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'Sharpe': 0, 'CAGR': 0, 'MDD': 0, 'Cal': 0, 'Final': 0}
    sharpe = (rets.mean() * bars_per_year) / (rets.std() * np.sqrt(bars_per_year)) if rets.std() > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else 0.001
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {'Sharpe': round(sharpe, 3), 'CAGR': round(cagr, 4),
            'MDD': round(mdd, 4), 'Cal': round(cal, 3), 'Final': round(eq.iloc[-1]/eq.iloc[0], 3)}


def analysis_1_events():
    print('\n=== 1. 발동 이벤트 열거 (BTC, 최적 param) ===')
    df = load_coin('BTCUSDT', '1h')
    eq, events = run_strategy(df, **BEST_PARAMS, track_events=True)
    print(f'총 발동 횟수: {len(events)}')
    if events:
        print(f'{"entry":<20} {"exit":<20} {"bars":>5} {"pnl%":>7} {"reason"}')
        for e in events:
            print(f'{str(e["entry_ts"]):<20} {str(e["exit_ts"]):<20} {e["bars_held"]:>5} {e["pnl_pct"]:>7} {e["reason"]}')
        avg_pnl = np.mean([e['pnl_pct'] for e in events])
        win_rate = sum(1 for e in events if e['pnl_pct'] > 0) / len(events) * 100
        print(f'평균 pnl: {avg_pnl:.2f}% / 승률: {win_rate:.0f}%')
    m = metrics(eq)
    print(f'BTC C metrics: {m}')
    return events, eq


def analysis_2_subperiod(eq):
    print('\n=== 2. Sub-period Rank-sum (10 windows) ===')
    WINDOWS = [
        ('H1_2021','2021-01-01','2021-06-30'),('H2_2021','2021-07-01','2021-12-31'),
        ('H1_2022','2022-01-01','2022-06-30'),('H2_2022','2022-07-01','2022-12-31'),
        ('H1_2023','2023-01-01','2023-06-30'),('H2_2023','2023-07-01','2023-12-31'),
        ('H1_2024','2024-01-01','2024-06-30'),('H2_2024','2024-07-01','2024-12-31'),
        ('H1_2025','2025-01-01','2025-06-30'),('H2_2025','2025-07-01','2025-12-31'),
    ]
    rows = []
    for name, s, e in WINDOWS:
        sub = eq.loc[s:e]
        if len(sub) < 100: continue
        # normalize
        sub = sub / sub.iloc[0]
        m = metrics(sub)
        rows.append({'window': name, **m})
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    neg = (df['CAGR'] < 0).sum()
    print(f'음수 수익률 구간: {neg}/{len(df)}')


def analysis_3_param_sensitivity():
    print('\n=== 3. Parameter Sensitivity ===')
    df = load_coin('BTCUSDT', '1h')
    base = BEST_PARAMS
    variations = {
        'dip_bars': [12, 18, 24, 30, 36, 48],
        'dip_threshold': [-0.10, -0.12, -0.15, -0.18, -0.20],
        'take_profit': [0.04, 0.06, 0.08, 0.10, 0.15],
        'time_stop_bars': [12, 24, 48, 96],
    }
    rows = []
    for axis, values in variations.items():
        for v in values:
            p = dict(base); p[axis] = v
            eq, _ = run_strategy(df, **p)
            m = metrics(eq)
            rows.append({'axis': axis, 'value': v, **m})
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


def analysis_4_slippage():
    print('\n=== 4. Slippage Stress Test ===')
    df = load_coin('BTCUSDT', '1h')
    for tx in [0.0004, 0.0010, 0.0015, 0.0020, 0.0030, 0.0050, 0.0080, 0.0100]:
        eq, _ = run_strategy(df, **BEST_PARAMS, tx_cost=tx)
        m = metrics(eq)
        print(f'  TX={tx*100:.2f}% ({int(tx*10000)}bps): Sharpe={m["Sharpe"]:>6} CAGR={m["CAGR"]:>7.2%} MDD={m["MDD"]:>7.2%} Cal={m["Cal"]:>6}')


def analysis_5_cross_coins():
    print('\n=== 5. 여러 코인 Cross-validation ===')
    coins = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT','DOGEUSDT',
             'ADAUSDT','AVAXUSDT','LINKUSDT','MATICUSDT','DOTUSDT','LTCUSDT','TRXUSDT','NEARUSDT']
    rows = []
    for c in coins:
        df = load_coin(c, '1h')
        if df is None or len(df) < 1000: continue
        eq, _ = run_strategy(df, **BEST_PARAMS)
        m = metrics(eq)
        m['coin'] = c
        m['bars'] = len(df)
        rows.append(m)
    df = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
    print(df[['coin','bars','Sharpe','CAGR','MDD','Cal','Final']].to_string(index=False))
    pos_ct = (df['Sharpe'] > 0).sum()
    print(f'양수 Sharpe 코인: {pos_ct}/{len(df)}')


def analysis_6_walkforward():
    print('\n=== 6. Walk-Forward Validation ===')
    df = load_coin('BTCUSDT', '1h')
    # 매년 OOS
    periods = [
        ('2021', '2020-10-01', '2020-12-31', '2021-01-01', '2021-12-31'),
        ('2022', '2021-01-01', '2021-12-31', '2022-01-01', '2022-12-31'),
        ('2023', '2022-01-01', '2022-12-31', '2023-01-01', '2023-12-31'),
        ('2024', '2023-01-01', '2023-12-31', '2024-01-01', '2024-12-31'),
        ('2025', '2024-01-01', '2024-12-31', '2025-01-01', '2025-12-31'),
    ]
    rows = []
    for yr, _, _, test_s, test_e in periods:
        sub = df.loc[test_s:test_e]
        if len(sub) < 100: continue
        eq, _ = run_strategy(sub, **BEST_PARAMS)
        m = metrics(eq)
        m['year'] = yr
        rows.append(m)
    out = pd.DataFrame(rows)
    print(out.to_string(index=False))


def main():
    events, eq = analysis_1_events()
    analysis_2_subperiod(eq)
    analysis_3_param_sensitivity()
    analysis_4_slippage()
    analysis_5_cross_coins()
    analysis_6_walkforward()


if __name__ == '__main__':
    main()
