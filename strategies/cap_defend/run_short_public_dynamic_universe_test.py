#!/usr/bin/env python3
"""동적 시총 유니버스 + 공개 추세추종 숏 benchmark 테스트.

원칙:
- 4h1 / 4h2 / 1h1 OFF 를 각각 독립적으로 사용
- 각 전략은 자기 시간축에서 평가
- 공개 전략의 day 개념은 bar-based로 환산
- 유니버스는 고정 코인이 아니라 매 시점 시총 상위 N개
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data, get_mcap
from run_short_per_canary_off_test import STRATEGIES, ShortOnlyEngine
from run_stoploss_test import START, END


UNIVERSE_SIZES = [3, 5, 8]
RULES = [
    dict(name='sma200', desc='Close < SMA200d'),
    dict(name='tsmom252', desc='252d TSMOM < 0'),
    dict(name='donchian50', desc='50d 저점 이탈'),
    dict(name='sma200_and_tsmom252', desc='Close < SMA200d and 252d TSMOM < 0'),
]


def _bars_per_day(interval):
    return {'4h': 6, '1h': 24}[interval]


def _build_gate_series(data, strat_name):
    cfg = STRATEGIES[strat_name]
    bars, _ = data[cfg['interval']]
    btc = bars['BTC']
    dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    close = btc['Close'].values
    prev_canary = False
    out = {}
    for date in dates:
        ci = btc.index.get_loc(date)
        if ci < cfg['sma_bars']:
            canary = False
        else:
            sma = float(np.mean(close[ci - cfg['sma_bars'] + 1:ci + 1]))
            ratio = float(close[ci] / sma) if sma > 0 else np.nan
            if prev_canary:
                canary = ratio >= (1.0 - cfg['canary_hyst'])
            else:
                canary = ratio > (1.0 + cfg['canary_hyst'])
        out[date] = not canary
        prev_canary = canary
    return out


def _signal_for_coin(df, ci, rule_name, bpd):
    close = df['Close'].values
    price = float(close[ci])

    sma200 = None
    if ci >= 200 * bpd:
        sma200 = float(np.mean(close[ci - 200 * bpd + 1:ci + 1]))

    tsmom252 = None
    if ci >= 252 * bpd:
        tsmom252 = float(close[ci] / close[ci - 252 * bpd] - 1.0)

    donchian50 = None
    if ci >= 50 * bpd:
        donchian50 = float(np.min(close[ci - 50 * bpd:ci]))

    if rule_name == 'sma200':
        return sma200 is not None and price < sma200
    if rule_name == 'tsmom252':
        return tsmom252 is not None and tsmom252 < 0
    if rule_name == 'donchian50':
        return donchian50 is not None and price < donchian50
    if rule_name == 'sma200_and_tsmom252':
        return sma200 is not None and tsmom252 is not None and price < sma200 and tsmom252 < 0
    return False


def build_target_series(data, strat_name, universe_size, rule):
    cfg = STRATEGIES[strat_name]
    bars, _ = data[cfg['interval']]
    dates = bars['BTC'].index[(bars['BTC'].index >= START) & (bars['BTC'].index <= END)]
    bpd = _bars_per_day(cfg['interval'])
    gate = _build_gate_series(data, strat_name)
    out = []

    for date in dates:
        if not gate.get(date, False):
            out.append((date, {'CASH': 1.0}))
            continue

        picks = []
        for coin in get_mcap(date)[:universe_size]:
            df = bars.get(coin)
            if df is None:
                continue
            ci = df.index.get_indexer([date], method='ffill')[0]
            if ci < 0:
                continue
            if _signal_for_coin(df, ci, rule['name'], bpd):
                picks.append(coin)

        if not picks:
            out.append((date, {'CASH': 1.0}))
            continue

        w = 1.0 / len(picks)
        out.append((date, {coin: w for coin in picks}))

    return out


def main():
    t0 = time.time()
    print('Loading data...')
    data = {iv: load_data(iv) for iv in ['4h', '1h']}

    print('\nResults')
    rows = []
    for strat_name, cfg in STRATEGIES.items():
        bars, funding = data[cfg['interval']]
        engine = ShortOnlyEngine(
            bars,
            funding,
            leverage=1.0,
            tx_cost=0.0004,
            maint_rate=0.004,
            initial_capital=10000.0,
        )
        print(f'\n== {strat_name} ({cfg["interval"]}) ==')
        for universe_size in UNIVERSE_SIZES:
            for rule in RULES:
                target_series = build_target_series(data, strat_name, universe_size, rule)
                m = engine.run(target_series)
                row = {
                    'strategy': strat_name,
                    'interval': cfg['interval'],
                    'universe_size': universe_size,
                    'rule': rule['name'],
                    'desc': f"top {universe_size} dynamic / {rule['desc']}",
                    'Cal': m.get('Cal', 0),
                    'CAGR': m.get('CAGR', 0),
                    'MDD': m.get('MDD', 0),
                    'Sharpe': m.get('Sharpe', 0),
                    'Liq': m.get('Liq', 0),
                    'Rebal': m.get('Rebal', 0),
                }
                rows.append(row)
                print(
                    f"top{row['universe_size']:<2} {row['rule']:<20} "
                    f"Cal={row['Cal']:.2f} CAGR={row['CAGR']:+.1%} "
                    f"MDD={row['MDD']:+.1%} Liq={row['Liq']} Rebal={row['Rebal']}"
                )

    rows.sort(key=lambda r: (-r['Cal'], -r['Sharpe'], r['strategy'], r['universe_size'], r['rule']))
    print('\nTop candidates')
    for row in rows[:12]:
        print(
            f"- {row['strategy']} / top{row['universe_size']} / {row['rule']}: "
            f"Cal={row['Cal']:.2f}, CAGR={row['CAGR']:+.1%}, "
            f"MDD={row['MDD']:+.1%}, {row['desc']}"
        )
    print(f'\nElapsed: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
