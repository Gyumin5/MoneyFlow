#!/usr/bin/env python3
"""코인 양방향 유연 선정 변형 비교.

목표:
- 기존 `시총순 Top 5 고정` 대비
- 큰 코인이 강하면 유지하고
- 작은 코인이 충분히 강하면 교체 허용하는
  선택 규칙 5종을 같은 프레임에서 비교한다.
"""

import os
import sys
import time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coin_engine import load_universe, load_all_prices, filter_universe
from coin_helpers import B, ANCHOR_DAYS
from backtest_official import run_coin_backtest


BASE_CFG = dict(
    params=dict(
        sma_period=60,
        canary_band=1.0,
        health_sma=0,
        health_mom_short=30,
        health_mom_long=90,
        selection='baseline',
        weighting='WC',
        n_picks=5,
        top_n=40,
        risk='G5',
    ),
    dd_lookback=60,
    dd_threshold=-0.25,
    bl_drop=-0.15,
    bl_days=7,
    drift_threshold=0.10,
    post_flip_delay=5,
)


VARIANTS = {
    'BASE': dict(
        selection='baseline',
        weighting='WC',
        note='기존 시총순 Top5 고정',
    ),
    'F1': dict(
        selection='S11',
        weighting='WC',
        note='Mom30 x 시총 보너스',
    ),
    'F2': dict(
        selection='S12',
        weighting='WC',
        note='Mom30 rank + 시총 rank 가중합',
    ),
    'F3': dict(
        selection='S13',
        weighting='WC',
        note='Sharpe-like(Mom30/Vol90) rank + 시총 rank',
    ),
    'F4': dict(
        selection='S14',
        weighting='W8',
        note='전체 헬스 통과를 Sharpe-like로 정렬 후 시총 비중 가중',
    ),
    'F5': dict(
        selection='S15',
        weighting='WC',
        note='하위 시총은 incumbent 대비 확실히 강할 때만 교체',
    ),
}


PERIODS = [
    ('2018-01-01', '2025-06-30'),
    ('2019-01-01', '2025-06-30'),
    ('2021-01-01', '2025-06-30'),
]


def run_multi_anchor(prices, universe_map, cfg, start, end):
    params = B(**cfg['params'])
    params.start_date = start
    params.end_date = end
    results = []
    for base_d in ANCHOR_DAYS:
        snap_days = tuple((base_d - 1 + j * 10) % 28 + 1 for j in range(3))
        result = run_coin_backtest(
            prices,
            universe_map,
            snapshot_days=snap_days,
            dd_lookback=cfg['dd_lookback'],
            dd_threshold=cfg['dd_threshold'],
            bl_drop=cfg['bl_drop'],
            bl_days=cfg['bl_days'],
            drift_threshold=cfg['drift_threshold'],
            post_flip_delay=cfg['post_flip_delay'],
            params_base=params,
        )
        results.append(result)
    return results


def summarize(results):
    sharpes = [r['metrics']['Sharpe'] for r in results]
    cagrs = [r['metrics']['CAGR'] for r in results]
    mdds = [r['metrics']['MDD'] for r in results]
    rebals = [r['rebal_count'] for r in results]
    dd_exits = [r['dd_exit_count'] for r in results]
    cagr = np.mean(cagrs)
    mdd = np.mean(mdds)
    return {
        'Sharpe': np.mean(sharpes),
        'SharpeStd': np.std(sharpes),
        'CAGR': cagr,
        'MDD': mdd,
        'Calmar': cagr / abs(mdd) if mdd != 0 else 0,
        'Rebal': np.mean(rebals),
        'DDExit': np.mean(dd_exits),
    }


def main():
    t0 = time.time()
    print("데이터 로딩...")
    um_raw = load_universe()
    um40 = filter_universe(um_raw, 40)
    all_tickers = set()
    for tickers in um40.values():
        all_tickers.update(tickers)
    all_tickers.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(sorted(all_tickers))
    print(f"  완료 ({time.time() - t0:.1f}s)\n")

    print("변형 정의")
    for code, cfg in VARIANTS.items():
        print(f"  {code:<4s} {cfg['note']}")
    print()

    for start, end in PERIODS:
        print(f"{'=' * 104}")
        print(f"기간: {start} ~ {end}")
        print(f"{'=' * 104}")
        print(f"{'코드':<5s} {'Sharpe':>7s} {'σ(Sh)':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s} {'Rebal':>6s} {'DD_Exit':>7s}  설명")
        print(f"{'-' * 104}")

        for code, variant in VARIANTS.items():
            cfg = dict(BASE_CFG)
            cfg['params'] = dict(BASE_CFG['params'])
            cfg['params']['selection'] = variant['selection']
            cfg['params']['weighting'] = variant['weighting']
            results = run_multi_anchor(prices, um40, cfg, start, end)
            s = summarize(results)
            print(
                f"{code:<5s} {s['Sharpe']:>7.3f} {s['SharpeStd']:>7.3f} "
                f"{s['CAGR']:>+8.1%} {s['MDD']:>+8.1%} {s['Calmar']:>7.2f} "
                f"{s['Rebal']:>6.0f} {s['DDExit']:>7.1f}  {variant['note']}"
            )
        print()

    print(f"총 소요: {time.time() - t0:.1f}s")


if __name__ == '__main__':
    main()
