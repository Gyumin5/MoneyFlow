#!/usr/bin/env python3
"""전략별 카나리 OFF 구간 유니버스 기반 숏 바스켓 테스트.

목적:
- BTC 단일 숏이 아니라, 롱과 동일한 시총 유니버스에서 약한 코인을 골라 숏
- 4h1 OFF / 4h2 OFF / 1h1 OFF 를 각각 독립적으로 평가
- 각 전략은 자기 시간축에서 평가
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data, SLIPPAGE_MAP, get_mcap
from run_stoploss_test import START, END
from run_short_per_canary_off_test import STRATEGIES, ShortOnlyEngine


UNIVERSE_SIZE = 5

CASE_GRID = [
    dict(name='cap_top_1_none', desc='시총 상위 1개 / health 없음', select='cap_top', picks=1, health='none'),
    dict(name='cap_bottom_1_none', desc='시총 하위 1개 / health 없음', select='cap_bottom', picks=1, health='none'),
    dict(name='mom_short_1_none', desc='MomShort weakest 1개 / health 없음', select='mom_short', picks=1, health='none'),
    dict(name='mom_long_1_none', desc='MomLong weakest 1개 / health 없음', select='mom_long', picks=1, health='none'),
    dict(name='mom_blend_1_none', desc='MomShort+MomLong weakest 1개 / health 없음', select='mom_blend', picks=1, health='none'),

    dict(name='cap_top_3_m1', desc='시총 상위 3개 / MomShort<0', select='cap_top', picks=3, health='m1'),
    dict(name='cap_bottom_3_m1', desc='시총 하위 3개 / MomShort<0', select='cap_bottom', picks=3, health='m1'),
    dict(name='mom_short_3_m1', desc='MomShort weakest 3개 / MomShort<0', select='mom_short', picks=3, health='m1'),
    dict(name='mom_long_3_m1', desc='MomLong weakest 3개 / MomShort<0', select='mom_long', picks=3, health='m1'),
    dict(name='mom_blend_3_m1', desc='MomShort+MomLong weakest 3개 / MomShort<0', select='mom_blend', picks=3, health='m1'),

    dict(name='cap_top_5_m2', desc='시총 상위 5개 / MomShort<0 & MomLong<0', select='cap_top', picks=5, health='m2'),
    dict(name='cap_bottom_5_m2', desc='시총 하위 5개 / MomShort<0 & MomLong<0', select='cap_bottom', picks=5, health='m2'),
    dict(name='mom_short_5_m2', desc='MomShort weakest 5개 / MomShort<0 & MomLong<0', select='mom_short', picks=5, health='m2'),
    dict(name='mom_long_5_m2', desc='MomLong weakest 5개 / MomShort<0 & MomLong<0', select='mom_long', picks=5, health='m2'),
    dict(name='mom_blend_5_m2', desc='MomShort+MomLong weakest 5개 / MomShort<0 & MomLong<0', select='mom_blend', picks=5, health='m2'),
]


def build_target_series(data, strat_name, case):
    cfg = STRATEGIES[strat_name]
    bars, _ = data[cfg['interval']]
    btc = bars['BTC']
    dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    close = btc['Close'].values
    prev_canary = False
    out = []

    for date in dates:
        ci_btc = btc.index.get_loc(date)
        if ci_btc < cfg['sma_bars']:
            canary = False
        else:
            sma = float(np.mean(close[ci_btc - cfg['sma_bars'] + 1:ci_btc + 1]))
            ratio = float(close[ci_btc] / sma) if sma > 0 else np.nan
            if prev_canary:
                canary = ratio >= (1.0 - cfg['canary_hyst'])
            else:
                canary = ratio > (1.0 + cfg['canary_hyst'])

        if canary:
            out.append((date, {'CASH': 1.0}))
            prev_canary = canary
            continue

        candidates = []
        for rank, coin in enumerate(get_mcap(date)[:UNIVERSE_SIZE]):
            df = bars.get(coin)
            if df is None:
                continue
            ci = df.index.get_indexer([date], method='ffill')[0]
            if ci < 0:
                continue
            c = df['Close'].values
            if ci < max(cfg['mom_short_bars'], cfg['mom_long_bars']):
                continue
            mom_short = float(c[ci] / c[ci - cfg['mom_short_bars']] - 1.0)
            mom_long = float(c[ci] / c[ci - cfg['mom_long_bars']] - 1.0)
            if case['health'] == 'm1' and not (mom_short < 0):
                continue
            if case['health'] == 'm2' and not (mom_short < 0 and mom_long < 0):
                continue
            if case['select'] == 'cap_top':
                score = rank
                order_key = (score, coin)
            elif case['select'] == 'cap_bottom':
                score = -rank
                order_key = (score, coin)
            elif case['select'] == 'mom_short':
                score = mom_short
                order_key = (score, coin)
            elif case['select'] == 'mom_long':
                score = mom_long
                order_key = (score, coin)
            else:
                score = mom_short + mom_long
                order_key = (score, coin)
            candidates.append((coin, order_key))

        candidates.sort(key=lambda x: x[1])
        picks = [coin for coin, _ in candidates[:case['picks']]]
        if not picks:
            out.append((date, {'CASH': 1.0}))
            prev_canary = canary
            continue

        w = 1.0 / len(picks)
        target = {coin: w for coin in picks}
        out.append((date, target))
        prev_canary = canary

    return out


def main():
    t0 = time.time()
    print('Loading data...')
    data = {iv: load_data(iv) for iv in ['4h', '1h']}

    print('\nResults')
    rows = []
    for strat_name, cfg in STRATEGIES.items():
        bars, funding = data[cfg['interval']]
        engine = ShortOnlyEngine(bars, funding, leverage=1.0, tx_cost=0.0004, maint_rate=0.004, initial_capital=10000.0)
        print(f'\n== {strat_name} ({cfg["interval"]}) ==')
        for case in CASE_GRID:
            target_series = build_target_series(data, strat_name, case)
            m = engine.run(target_series)
            row = {
                'strategy': strat_name,
                'interval': cfg['interval'],
                'case': case['name'],
                'desc': case['desc'],
                'Cal': m.get('Cal', 0),
                'CAGR': m.get('CAGR', 0),
                'MDD': m.get('MDD', 0),
                'Sharpe': m.get('Sharpe', 0),
                'Liq': m.get('Liq', 0),
                'Rebal': m.get('Rebal', 0),
            }
            rows.append(row)
            print(
                f"{row['case']:<24} Cal={row['Cal']:.2f} "
                f"CAGR={row['CAGR']:+.1%} MDD={row['MDD']:+.1%} "
                f"Liq={row['Liq']} Rebal={row['Rebal']}"
            )

    rows.sort(key=lambda r: (-r['Cal'], -r['Sharpe'], r['strategy'], r['case']))
    print('\nTop candidates')
    for row in rows[:10]:
        print(
            f"- {row['strategy']} / {row['case']}: "
            f"Cal={row['Cal']:.2f}, CAGR={row['CAGR']:+.1%}, "
            f"MDD={row['MDD']:+.1%}, {row['desc']}"
        )
    print(f'\nElapsed: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
