#!/usr/bin/env python3
"""Step 1b: V21 현물 엔진 실행 + daily (equity, CASH_ratio) 시계열 저장.

run_current_coin_v20_backtest.run_backtest()가 detail DataFrame 반환하므로
거기에서 combined CASH 비중 추출.
"""
from __future__ import annotations
import os, sys
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "trade"))
sys.path.insert(0, os.path.join(ROOT, 'strategies', 'cap_defend'))

START = '2020-10-01'
END = '2026-03-30'


def main():
    print('=== V21 현물 엔진 실행 + daily (equity, CASH) 추출 ===')
    import run_current_coin_v20_backtest as spot_bt
    res = spot_bt.run_backtest(start=START, end=END)
    detail = res['detail']
    print(f'detail columns: {list(detail.columns)}')
    print(f'rows: {len(detail)}')
    # detail 에는 CASH 컬럼 있을 것 (combined target의 CASH)
    eq = res['equity']
    if isinstance(eq.index, pd.DatetimeIndex) and eq.index.tz is not None:
        eq = eq.copy(); eq.index = eq.index.tz_localize(None)

    # detail index 정규화 (tz 제거)
    d2 = detail.copy()
    if isinstance(d2.index, pd.DatetimeIndex) and d2.index.tz is not None:
        d2.index = d2.index.tz_localize(None)
    daily = pd.DataFrame({
        'equity': d2['Value'],
        'cash_ratio': d2['CASH'],
    })
    daily_d = daily.resample('D').last().ffill()

    print(f'\nCash ratio 분포:')
    print(daily_d['cash_ratio'].describe())

    # 카나리 OFF 추정: cash > 0.8
    off_days = (daily_d['cash_ratio'] > 0.8).sum()
    print(f'\n카나리 OFF 추정 (cash > 80%): {off_days}/{len(daily_d)} days ({off_days/len(daily_d)*100:.1f}%)')

    out = os.path.join(HERE, 'strat_C_v3', 'v21_daily.csv')
    os.makedirs(os.path.dirname(out), exist_ok=True)
    daily_d.to_csv(out)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
