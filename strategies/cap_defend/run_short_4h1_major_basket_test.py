#!/usr/bin/env python3
"""4h1 OFF 구간 메이저 바스켓 숏 테스트.

방향:
- 지금까지 결과상 숏 sleeve로 그나마 의미가 있던 것은 4h1 OFF
- 약한 알트/하위 시총보다 유동성 큰 메이저 바스켓을 단순하게 숏
- health는 최소한만 본다: 없음 / BTC<SMA
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data
from run_short_per_canary_off_test import ShortOnlyEngine
from run_stoploss_test import START, END


CFG = dict(interval='4h', sma_bars=240, canary_hyst=0.015)

BASKETS = [
    dict(name='btc', desc='BTC 단일', coins=['BTC']),
    dict(name='btc_eth', desc='BTC/ETH', coins=['BTC', 'ETH']),
    dict(name='btc_eth_sol', desc='BTC/ETH/SOL', coins=['BTC', 'ETH', 'SOL']),
    dict(name='btc_eth_xrp', desc='BTC/ETH/XRP', coins=['BTC', 'ETH', 'XRP']),
]

FILTERS = [
    dict(name='none', desc='추가 필터 없음', require_below_sma=False),
    dict(name='btc_below_sma', desc='BTC < SMA240', require_below_sma=True),
]


def build_target_series(data, basket, filt):
    bars, funding = data['4h']
    btc = bars['BTC']
    dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    close = btc['Close'].values
    prev_canary = False
    out = []

    for date in dates:
        ci = btc.index.get_loc(date)
        if ci < CFG['sma_bars']:
            canary = False
            below_sma = False
        else:
            sma = float(np.mean(close[ci - CFG['sma_bars'] + 1:ci + 1]))
            ratio = float(close[ci] / sma) if sma > 0 else np.nan
            if prev_canary:
                canary = ratio >= (1.0 - CFG['canary_hyst'])
            else:
                canary = ratio > (1.0 + CFG['canary_hyst'])
            below_sma = close[ci] < sma if sma > 0 else False

        active = not canary
        if filt['require_below_sma']:
            active = active and below_sma

        if not active:
            out.append((date, {'CASH': 1.0}))
        else:
            w = 1.0 / len(basket['coins'])
            target = {coin: w for coin in basket['coins']}
            out.append((date, target))
        prev_canary = canary

    return out, funding


def main():
    t0 = time.time()
    print('Loading data...')
    data = {'4h': load_data('4h')}
    bars_4h, funding_4h = data['4h']
    engine = ShortOnlyEngine(
        bars_4h,
        funding_4h,
        leverage=1.0,
        tx_cost=0.0004,
        maint_rate=0.004,
        initial_capital=10000.0,
    )

    print('\nResults')
    rows = []
    for basket in BASKETS:
        for filt in FILTERS:
            target_series, _ = build_target_series(data, basket, filt)
            m = engine.run(target_series)
            row = {
                'basket': basket['name'],
                'filter': filt['name'],
                'desc': f"{basket['desc']} / {filt['desc']}",
                'Cal': m.get('Cal', 0),
                'CAGR': m.get('CAGR', 0),
                'MDD': m.get('MDD', 0),
                'Sharpe': m.get('Sharpe', 0),
                'Liq': m.get('Liq', 0),
                'Rebal': m.get('Rebal', 0),
            }
            rows.append(row)
            print(
                f"{row['basket']:<12} {row['filter']:<14} "
                f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
                f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Rebal={row['Rebal']}"
            )

    rows.sort(key=lambda r: (-r['Cal'], -r['Sharpe'], r['basket'], r['filter']))
    print('\nTop candidates')
    for row in rows:
        print(
            f"- {row['basket']} / {row['filter']}: "
            f"Cal={row['Cal']:.2f}, CAGR={row['CAGR']:+.1%}, "
            f"MDD={row['MDD']:+.1%}, {row['desc']}"
        )
    print(f'\nElapsed: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
