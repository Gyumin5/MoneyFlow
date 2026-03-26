#!/usr/bin/env python3
"""코인 3트랜치 vs 4트랜치 비교 백테스트.

현재 V17: 3트랜치 (Day 1/11/21, 간격 10일)
후보:     4트랜치 (Day 1/8/15/22, 간격 7일)

10-anchor 평균으로 robust 비교.
"""

import sys, os, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coin_engine import load_universe, load_all_prices, filter_universe
from coin_helpers import B, ANCHOR_DAYS
from backtest_official import run_coin_backtest

# ═══ V17 파라미터 ═══
V17_CFG = dict(
    params=dict(sma_period=60, canary_band=1.0, health_sma=0,
                health_mom_short=30,
                selection='baseline', n_picks=5, weighting='WC', top_n=40,
                risk='G5'),
    dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
    drift_threshold=0.10, post_flip_delay=5,
)

# ═══ 트랜치 설정 ═══
TRANCHE_CONFIGS = {
    '3T (9일간격)': {'n': 3, 'spacing': 9},   # 현재: Day 1/10/19
    '3T (10일간격)': {'n': 3, 'spacing': 10},  # 현재 실매매: Day 1/11/21
    '4T (7일간격)': {'n': 4, 'spacing': 7},    # 주식과 동일: Day 1/8/15/22
    '4T (6일간격)': {'n': 4, 'spacing': 6},    # 더 촘촘한 4T
    '2T (14일간격)': {'n': 2, 'spacing': 14},  # 참고용: 2트랜치
}

PERIODS = [
    ('2018-01-01', '2025-06-30'),
    ('2019-01-01', '2025-06-30'),
    ('2021-01-01', '2025-06-30'),
]


def make_snap_days(base_d, n_tranches, spacing):
    """base_d 기준으로 n_tranches개 snap day 생성."""
    return [(base_d - 1 + j * spacing) % 28 + 1 for j in range(n_tranches)]


def run_multi_anchor(prices, um, n_tranches, spacing, cfg, start, end):
    """10-anchor 평균 실행."""
    results = []
    p = B(**cfg['params'])
    p.start_date = start
    p.end_date = end

    for base_d in ANCHOR_DAYS:
        snap_days = make_snap_days(base_d, n_tranches, spacing)
        r = run_coin_backtest(
            prices, um, tuple(snap_days),
            dd_lookback=cfg['dd_lookback'], dd_threshold=cfg['dd_threshold'],
            bl_drop=cfg['bl_drop'], bl_days=cfg['bl_days'],
            drift_threshold=cfg['drift_threshold'],
            post_flip_delay=cfg['post_flip_delay'],
            params_base=p,
        )
        results.append(r)
    return results


def main():
    t0 = time.time()
    print("데이터 로딩...")
    um_raw = load_universe()
    um40 = filter_universe(um_raw, 40)
    all_t = set()
    for ts in um40.values():
        all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(list(all_t))
    print(f"  완료 ({time.time()-t0:.1f}s)\n")

    # ═══ 전체 기간별 비교 ═══
    for start, end in PERIODS:
        print(f"{'='*90}")
        print(f"  기간: {start} ~ {end}")
        print(f"{'='*90}")
        print(f"  {'트랜치':<16s} {'Sharpe':>7s} {'σ(Sh)':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s} {'Rebal':>6s} {'DD_Exit':>7s}")
        print(f"  {'-'*72}")

        for label, tcfg in TRANCHE_CONFIGS.items():
            results = run_multi_anchor(
                prices, um40, tcfg['n'], tcfg['spacing'],
                V17_CFG, start, end,
            )
            sharpes = [r['metrics']['Sharpe'] for r in results]
            cagrs = [r['metrics']['CAGR'] for r in results]
            mdds = [r['metrics']['MDD'] for r in results]
            rebals = [r['rebal_count'] for r in results]
            dd_exits = [r['dd_exit_count'] for r in results]

            s_mean = np.mean(sharpes)
            s_std = np.std(sharpes)
            c_mean = np.mean(cagrs)
            mdd_mean = np.mean(mdds)
            cal = c_mean / abs(mdd_mean) if mdd_mean != 0 else 0
            rb_mean = np.mean(rebals)
            dd_mean = np.mean(dd_exits)

            marker = " <-- 현재" if label == '3T (10일간격)' else ""
            print(f"  {label:<16s} {s_mean:>7.3f} {s_std:>7.3f} {c_mean:>+8.1%} {mdd_mean:>+8.1%} {cal:>7.2f} {rb_mean:>6.0f} {dd_mean:>7.1f}{marker}")

        print()

    # ═══ 앵커별 상세 (전체 기간) ═══
    start, end = PERIODS[0]
    print(f"\n{'='*90}")
    print(f"  앵커별 Sharpe 상세 ({start}~{end})")
    print(f"{'='*90}")

    for label, tcfg in TRANCHE_CONFIGS.items():
        results = run_multi_anchor(
            prices, um40, tcfg['n'], tcfg['spacing'],
            V17_CFG, start, end,
        )
        sharpes = [r['metrics']['Sharpe'] for r in results]
        anchors_str = " ".join(f"{s:.3f}" for s in sharpes)
        print(f"  {label:<16s}: [{anchors_str}]  mean={np.mean(sharpes):.3f} σ={np.std(sharpes):.3f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
