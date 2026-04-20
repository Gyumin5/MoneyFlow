#!/usr/bin/env python3
"""Strategy A: 카나리 OFF + trend down 확정 시 BTC short.

V21 long-only 전략의 약점(카나리 OFF 구간 현금 방치)을 보완.
같은 바이낸스 선물 엔진 재활용, 방향만 반대.

로직:
- 매 4h 봉
- BTC < SMA * (1 - hyst)  →  canary OFF
- AND Mom20 < 0           →  trend down 확정
- →  BTC short 진입 (3x)
- 청산: BTC > SMA * (1 + hyst) OR Mom20 > 0
- 비용: 0.04%

기간: 2020-10-01 ~ 2026-03 (V21 백테스트 기간과 동일)
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')

# 파라미터
START = '2020-10-01'
END = '2026-03-30'
SMA_BARS = 240          # 4h × 240 = 40일 SMA (기존 V21 선물 SMA 동일)
HYST = 0.015            # 1.5% hysteresis
MOM_BARS = 180          # 4h × 180 = 30일 모멘텀
LEVERAGE = 3.0
TX_COST = 0.0004        # 0.04% 편도

INITIAL_CAPITAL = 10000.0


def load_btc_4h():
    df = pd.read_csv(os.path.join(DATA_DIR, 'BTCUSDT_4h.csv'))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.loc[START:END].copy()
    return df


def compute_signals(df):
    df = df.copy()
    df['SMA'] = df['Close'].rolling(SMA_BARS).mean()
    df['Mom'] = df['Close'] / df['Close'].shift(MOM_BARS) - 1.0
    # t-1 종가 기준 시그널, t 봉 시가 체결 (look-ahead 방지)
    df['SMA_sig'] = df['SMA'].shift(1)
    df['Mom_sig'] = df['Mom'].shift(1)
    df['Close_sig'] = df['Close'].shift(1)
    df['canary_off'] = df['Close_sig'] < df['SMA_sig'] * (1 - HYST)
    df['canary_on'] = df['Close_sig'] > df['SMA_sig'] * (1 + HYST)
    df['mom_down'] = df['Mom_sig'] < 0
    df['mom_up'] = df['Mom_sig'] > 0
    return df


def run_backtest(df):
    """체결: 당일 봉 Open (prev_close 시그널)"""
    df = df.dropna(subset=['SMA_sig', 'Mom_sig', 'Close_sig']).copy()
    equity = [INITIAL_CAPITAL]
    position = 0.0   # -1 = short, 0 = flat. (1 = long은 V21이므로 여기선 제외)
    entry_price = None
    records = []

    for i in range(1, len(df)):
        row = df.iloc[i]
        open_px = row['Open']
        close_px = row['Close']
        prev_eq = equity[-1]

        # 진입/청산 판단 (prev bar 종가 시그널 기반)
        want_short = row['canary_off'] and row['mom_down']

        new_position = position
        if position == 0 and want_short:
            new_position = -1
            entry_price = open_px
        elif position == -1:
            # 청산 조건: 카나리 ON (SMA 회복) OR 모멘텀 양전환
            if row['canary_on'] or row['mom_up']:
                new_position = 0

        # bar 내 수익 계산 (position은 이번 봉 open에 적용된 상태로 close까지 holding)
        if new_position != position:
            # 전환 봉: open에 기존 포지션 청산 + 신규 진입
            # 단순화: open 시점 청산 → close까지 신규 포지션 holding
            bar_ret = 0.0
            if position != 0 and entry_price is not None:
                # 기존 포지션 close (open에)
                bar_ret_from_prev = position * (open_px / df.iloc[i-1]['Close'] - 1.0)
                bar_ret_from_prev -= TX_COST  # 청산 비용
                prev_eq *= (1 + LEVERAGE * bar_ret_from_prev)
            if new_position != 0:
                # 신규 진입 (open)
                entry_price = open_px
                bar_ret = new_position * (close_px / open_px - 1.0)
                bar_ret -= TX_COST  # 진입 비용
                eq = prev_eq * (1 + LEVERAGE * bar_ret)
            else:
                entry_price = None
                eq = prev_eq
            position = new_position
        else:
            # 포지션 유지
            if position != 0:
                bar_ret = position * (close_px / df.iloc[i-1]['Close'] - 1.0)
                eq = prev_eq * (1 + LEVERAGE * bar_ret)
            else:
                eq = prev_eq

        equity.append(eq)
        records.append({'Date': row.name, 'Close': close_px, 'Position': position, 'Equity': eq,
                       'canary_off': row['canary_off'], 'mom_down': row['mom_down']})

    eq_series = pd.Series(equity[1:], index=df.index[1:])
    return eq_series, pd.DataFrame(records)


def metrics(eq: pd.Series) -> dict:
    # 4h 봉 기준 annualize
    bars_per_year = 6 * 365
    rets = eq.pct_change().dropna()
    if len(rets) == 0:
        return {}
    mean = rets.mean() * bars_per_year
    std = rets.std() * np.sqrt(bars_per_year)
    sharpe = mean / std if std > 0 else 0
    years = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    peak = eq.cummax()
    dd = (eq / peak - 1)
    mdd = dd.min()
    cal = cagr / abs(mdd) if mdd < 0 else 0
    # Exposure: 포지션 있을 때 비율
    return {
        'Sharpe': round(sharpe, 3),
        'CAGR': round(cagr, 4),
        'MDD': round(mdd, 4),
        'Cal': round(cal, 3),
        'Final': round(eq.iloc[-1] / eq.iloc[0], 3),
    }


def main():
    print('[A: Short Strategy Backtest]')
    df = load_btc_4h()
    print(f'  BTC 4h bars: {len(df)}  ({df.index[0]} ~ {df.index[-1]})')
    sig = compute_signals(df)
    eq, rec = run_backtest(sig)
    m = metrics(eq)
    print(f'\n  결과:')
    for k, v in m.items():
        print(f'    {k}: {v}')

    # 포지션 통계
    pos_bars = (rec['Position'] != 0).sum()
    pos_pct = pos_bars / len(rec) * 100
    n_trades = (rec['Position'].diff().abs() > 0).sum()
    print(f'\n  Position bars: {pos_bars}/{len(rec)} ({pos_pct:.1f}%)')
    print(f'  # 거래 (state change): {n_trades}')

    # Save equity
    out_dir = os.path.join(HERE, 'strat_A_short')
    os.makedirs(out_dir, exist_ok=True)
    eq.to_csv(os.path.join(out_dir, 'equity.csv'), header=['equity'])
    rec.to_csv(os.path.join(out_dir, 'records.csv'), index=False)
    print(f'\n  Saved: {out_dir}/equity.csv, records.csv')

    return eq, m


if __name__ == '__main__':
    main()
