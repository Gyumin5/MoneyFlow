#!/usr/bin/env python3
"""가드 ablation 테스트 — 공식 엔진(backtest_official)에서 동일 조건 비교.

비교:
1. V19 현행 (가드 전부 ON)
2. V19 가드 OFF (신호만)
3. 그리드 상위 후보를 공식 엔진 파라미터로 매핑, 가드 ON/OFF

그리드 D 후보 → 공식 엔진 매핑:
  sma_bars=50 → sma_period=50
  mom_short_bars → health_mom_short
  mom_long_bars → health_mom_long  (B() 기본값 90)
  vol_threshold=0.05 → vol_cap=0.05
  canary_hyst=0.015 → canary_band=1.5
  나머지(selection, weighting, risk)는 V19 설정 사용
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from coin_engine import load_universe, load_all_prices, filter_universe, DEFENSE_TICKERS
from coin_helpers import B
from backtest_official import run_coin_backtest

# ─── 전략 정의 ──────────────────────────────────────────────

# V19 현행 (공식 엔진 파라미터)
V19_PARAMS = dict(
    sma_period=50, canary_band=1.5, vote_smas=(50,),
    health_sma=0, health_mom_short=30,
    selection='SG', n_picks=5, weighting='WG', top_n=40,
    risk='G5',
)
V19_GUARDS = dict(
    dd_lookback=60, dd_threshold=-0.25,
    bl_drop=-0.15, bl_days=7,
    drift_threshold=0.10, post_flip_delay=0,
)
NO_GUARDS = dict(
    dd_lookback=0, dd_threshold=0,
    bl_drop=0, bl_days=0,
    drift_threshold=0, post_flip_delay=0,
)

# 그리드 상위 후보 (D 봉 → 공식 엔진은 일간이므로 직접 매핑)
GRID_CANDIDATES = {
    # 그리드 D 1위: SMA50/Ms20/Ml90
    'Ms20_Ml90': dict(
        sma_period=50, canary_band=1.5, vote_smas=(50,),
        health_sma=0, health_mom_short=20, health_mom_long=90,
        selection='SG', n_picks=5, weighting='WG', top_n=40,
        risk='G5',
    ),
    # Mom short 변형들
    'Ms10_Ml90': dict(
        sma_period=50, canary_band=1.5, vote_smas=(50,),
        health_sma=0, health_mom_short=10, health_mom_long=90,
        selection='SG', n_picks=5, weighting='WG', top_n=40,
        risk='G5',
    ),
    'Ms40_Ml90': dict(
        sma_period=50, canary_band=1.5, vote_smas=(50,),
        health_sma=0, health_mom_short=40, health_mom_long=90,
        selection='SG', n_picks=5, weighting='WG', top_n=40,
        risk='G5',
    ),
    # Mom long 변형
    'Ms20_Ml60': dict(
        sma_period=50, canary_band=1.5, vote_smas=(50,),
        health_sma=0, health_mom_short=20, health_mom_long=60,
        selection='SG', n_picks=5, weighting='WG', top_n=40,
        risk='G5',
    ),
    'Ms20_Ml120': dict(
        sma_period=50, canary_band=1.5, vote_smas=(50,),
        health_sma=0, health_mom_short=20, health_mom_long=120,
        selection='SG', n_picks=5, weighting='WG', top_n=40,
        risk='G5',
    ),
    # V19 현행 (가드 OFF 비교용)
    'V19_Ms30_Ml90': dict(
        sma_period=50, canary_band=1.5, vote_smas=(50,),
        health_sma=0, health_mom_short=30, health_mom_long=90,
        selection='SG', n_picks=5, weighting='WG', top_n=40,
        risk='G5',
    ),
}

SNAP = (1, 10, 19)
PERIODS = [
    ('2018-01-01', '2026-03-31'),
    ('2019-01-01', '2026-03-31'),
    ('2021-01-01', '2026-03-31'),
]


def run_10anchor(prices, um, params_dict, guards, period):
    """10-anchor 평균 실행."""
    start, end = period
    anchors = [(d, d+9, d+18) for d in range(1, 11)]
    results = []
    for snap in anchors:
        p = B(**params_dict)
        p.start_date = start
        p.end_date = end
        r = run_coin_backtest(prices, um, snap,
                              dd_lookback=guards['dd_lookback'],
                              dd_threshold=guards['dd_threshold'],
                              bl_drop=guards['bl_drop'],
                              bl_days=guards['bl_days'],
                              drift_threshold=guards['drift_threshold'],
                              post_flip_delay=guards['post_flip_delay'],
                              params_base=p, defense=False)
        m = r['metrics']
        cal = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        results.append(dict(
            Sharpe=m['Sharpe'], CAGR=m['CAGR'], MDD=m['MDD'],
            Cal=cal, DD=r['dd_exit_count'], Rebal=r['rebal_count'],
        ))
    return {
        'Sharpe': np.mean([r['Sharpe'] for r in results]),
        'CAGR': np.mean([r['CAGR'] for r in results]),
        'MDD': np.mean([r['MDD'] for r in results]),
        'Cal': np.mean([r['Cal'] for r in results]),
        'DD': np.mean([r['DD'] for r in results]),
        'Rebal': np.mean([r['Rebal'] for r in results]),
        'sigma_Sh': np.std([r['Sharpe'] for r in results]),
    }


def fmt(m):
    return (f"Sh {m['Sharpe']:.3f}(σ{m['sigma_Sh']:.3f}) "
            f"Cal {m['Cal']:.2f} CAGR {m['CAGR']:+.1%} "
            f"MDD {m['MDD']:+.1%} DD {m['DD']:.0f} Rebal {m['Rebal']:.0f}")


def main():
    t0 = time.time()
    print("데이터 로딩...")
    um_raw = load_universe()
    um = filter_universe(um_raw, 40)
    all_t = set()
    for ts in um.values():
        all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    all_t.update(DEFENSE_TICKERS)
    prices = load_all_prices(list(all_t))
    print(f"  완료 ({time.time()-t0:.1f}s)")

    for start, end in PERIODS:
        period = (start, end)
        print(f"\n{'='*90}")
        print(f"  [{start} ~ {end}]  10-anchor 평균")
        print(f"{'='*90}")
        print(f"  {'Label':<25s} {'Guard':>5s} | {' '*60}")
        print(f"  {'-'*90}")

        # 1. V19 현행 (가드 ON)
        m = run_10anchor(prices, um, V19_PARAMS, V19_GUARDS, period)
        print(f"  {'V19 (현행)':<25s} {'ON':>5s} | {fmt(m)}")

        # 2. V19 가드 OFF
        m = run_10anchor(prices, um, V19_PARAMS, NO_GUARDS, period)
        print(f"  {'V19 (가드OFF)':<25s} {'OFF':>5s} | {fmt(m)}")

        print()

        # 3. 그리드 후보: 가드 ON/OFF 둘 다
        for name, params in GRID_CANDIDATES.items():
            m_on = run_10anchor(prices, um, params, V19_GUARDS, period)
            m_off = run_10anchor(prices, um, params, NO_GUARDS, period)
            print(f"  {name:<25s} {'ON':>5s} | {fmt(m_on)}")
            print(f"  {'':<25s} {'OFF':>5s} | {fmt(m_off)}")
            print()

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
