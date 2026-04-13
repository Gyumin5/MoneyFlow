#!/usr/bin/env python3
"""가드 파라미터 그리드 테스트 — 신호 상위 후보 x 가드 조합.

신호 후보: V19(Ms30), Ms20, Ms10, Ms40  (Ml90 고정)
가드 축:
  - DD exit: (lookback, threshold) = {없음, (30,-0.20), (60,-0.25), (60,-0.20), (60,-0.30), (90,-0.25)}
  - Blacklist: (drop, days) = {없음, (-0.10,7), (-0.15,7), (-0.20,7), (-0.15,14)}
  - Drift: {0, 0.05, 0.10, 0.15}
  - Crash(G5): ON/OFF → risk='G5' vs risk='baseline'

전체 조합은 너무 많으므로, 의미있는 패키지로 묶어서 테스트.
"""
import os, sys, time
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from coin_engine import load_universe, load_all_prices, filter_universe, DEFENSE_TICKERS
from coin_helpers import B
from backtest_official import run_coin_backtest

SNAP = (1, 10, 19)
PERIOD = ('2018-01-01', '2026-03-31')  # 메인 기간


# ─── 신호 후보 ──────────────────────────────────────────────
SIGNAL_BASE = dict(
    sma_period=50, canary_band=1.5, vote_smas=(50,),
    health_sma=0, health_mom_long=90,
    selection='SG', n_picks=5, weighting='WG', top_n=40,
)

SIGNALS = {
    'Ms10': dict(**SIGNAL_BASE, health_mom_short=10),
    'Ms20': dict(**SIGNAL_BASE, health_mom_short=20),
    'Ms30': dict(**SIGNAL_BASE, health_mom_short=30),  # V19 현행
    'Ms40': dict(**SIGNAL_BASE, health_mom_short=40),
}


# ─── 가드 패키지 ──────────────────────────────────────────
# (dd_lookback, dd_threshold, bl_drop, bl_days, drift_threshold, crash)
GUARD_PACKAGES = {
    'NoGuard':          dict(dd_lookback=0, dd_threshold=0, bl_drop=0, bl_days=0, drift_threshold=0, risk='baseline'),
    'CrashOnly':        dict(dd_lookback=0, dd_threshold=0, bl_drop=0, bl_days=0, drift_threshold=0, risk='G5'),
    'DD60_25':          dict(dd_lookback=60, dd_threshold=-0.25, bl_drop=0, bl_days=0, drift_threshold=0, risk='baseline'),
    'DD60_20':          dict(dd_lookback=60, dd_threshold=-0.20, bl_drop=0, bl_days=0, drift_threshold=0, risk='baseline'),
    'DD60_30':          dict(dd_lookback=60, dd_threshold=-0.30, bl_drop=0, bl_days=0, drift_threshold=0, risk='baseline'),
    'DD30_25':          dict(dd_lookback=30, dd_threshold=-0.25, bl_drop=0, bl_days=0, drift_threshold=0, risk='baseline'),
    'DD90_25':          dict(dd_lookback=90, dd_threshold=-0.25, bl_drop=0, bl_days=0, drift_threshold=0, risk='baseline'),
    'BL15':             dict(dd_lookback=0, dd_threshold=0, bl_drop=-0.15, bl_days=7, drift_threshold=0, risk='baseline'),
    'BL10':             dict(dd_lookback=0, dd_threshold=0, bl_drop=-0.10, bl_days=7, drift_threshold=0, risk='baseline'),
    'BL20':             dict(dd_lookback=0, dd_threshold=0, bl_drop=-0.20, bl_days=7, drift_threshold=0, risk='baseline'),
    'BL15_14d':         dict(dd_lookback=0, dd_threshold=0, bl_drop=-0.15, bl_days=14, drift_threshold=0, risk='baseline'),
    'Drift05':          dict(dd_lookback=0, dd_threshold=0, bl_drop=0, bl_days=0, drift_threshold=0.05, risk='baseline'),
    'Drift10':          dict(dd_lookback=0, dd_threshold=0, bl_drop=0, bl_days=0, drift_threshold=0.10, risk='baseline'),
    'Drift15':          dict(dd_lookback=0, dd_threshold=0, bl_drop=0, bl_days=0, drift_threshold=0.15, risk='baseline'),
    # 복합 패키지
    'V19_Full':         dict(dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7, drift_threshold=0.10, risk='G5'),  # 현행
    'Crash+DD60_25':    dict(dd_lookback=60, dd_threshold=-0.25, bl_drop=0, bl_days=0, drift_threshold=0, risk='G5'),
    'Crash+BL15':       dict(dd_lookback=0, dd_threshold=0, bl_drop=-0.15, bl_days=7, drift_threshold=0, risk='G5'),
    'Crash+DD+BL':      dict(dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7, drift_threshold=0, risk='G5'),
    'DD60_20+BL10':     dict(dd_lookback=60, dd_threshold=-0.20, bl_drop=-0.10, bl_days=7, drift_threshold=0.10, risk='G5'),
    'DD60_25+BL15+D05': dict(dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7, drift_threshold=0.05, risk='G5'),
    'DD30_20+BL10':     dict(dd_lookback=30, dd_threshold=-0.20, bl_drop=-0.10, bl_days=7, drift_threshold=0.10, risk='G5'),
    'DD90_30+BL20':     dict(dd_lookback=90, dd_threshold=-0.30, bl_drop=-0.20, bl_days=7, drift_threshold=0.10, risk='G5'),
    'Tight':            dict(dd_lookback=30, dd_threshold=-0.15, bl_drop=-0.10, bl_days=7, drift_threshold=0.05, risk='G5'),
    'Loose':            dict(dd_lookback=90, dd_threshold=-0.30, bl_drop=-0.20, bl_days=14, drift_threshold=0.15, risk='G5'),
}


def run_10anchor(prices, um, signal_params, guard_pkg, period):
    """10-anchor 평균."""
    start, end = period
    anchors = [(d, d+9, d+18) for d in range(1, 11)]
    results = []

    # Merge signal params + guard risk setting
    params = dict(signal_params)
    risk = guard_pkg.get('risk', 'baseline')
    params['risk'] = risk

    for snap in anchors:
        p = B(**params)
        p.start_date = start
        p.end_date = end

        r = run_coin_backtest(prices, um, snap,
                              dd_lookback=guard_pkg['dd_lookback'],
                              dd_threshold=guard_pkg['dd_threshold'],
                              bl_drop=guard_pkg['bl_drop'],
                              bl_days=guard_pkg['bl_days'],
                              drift_threshold=guard_pkg['drift_threshold'],
                              post_flip_delay=0,
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
    print(f"  완료 ({time.time()-t0:.1f}s)\n")

    total = len(SIGNALS) * len(GUARD_PACKAGES)
    print(f"테스트: {len(SIGNALS)} 신호 x {len(GUARD_PACKAGES)} 가드 = {total} 조합")
    print(f"기간: {PERIOD[0]} ~ {PERIOD[1]}, 10-anchor 평균\n")

    # 헤더
    print(f"{'Signal':<8s} {'Guard':<20s} | {'Sh':>6s} {'σSh':>5s} {'Cal':>5s} {'CAGR':>7s} {'MDD':>7s} {'DD':>4s} {'Rebal':>5s}")
    print("-" * 85)

    all_results = []
    count = 0
    for sig_name, sig_params in SIGNALS.items():
        for grd_name, grd_pkg in GUARD_PACKAGES.items():
            count += 1
            t1 = time.time()
            m = run_10anchor(prices, um, sig_params, grd_pkg, PERIOD)
            elapsed = time.time() - t1
            print(f"{sig_name:<8s} {grd_name:<20s} | "
                  f"{m['Sharpe']:>6.3f} {m['sigma_Sh']:>5.3f} "
                  f"{m['Cal']:>5.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} "
                  f"{m['DD']:>4.0f} {m['Rebal']:>5.0f}  ({elapsed:.0f}s) [{count}/{total}]")
            all_results.append((sig_name, grd_name, m))

    # ─── 순위합 요약 ──────────────────────────────────────
    print(f"\n{'='*85}")
    print("순위합 Top 20 (Sharpe + Calmar + CAGR 순위합, 낮을수록 좋음)")
    print(f"{'='*85}")

    n = len(all_results)
    sh_rank = sorted(range(n), key=lambda i: -all_results[i][2]['Sharpe'])
    cal_rank = sorted(range(n), key=lambda i: -all_results[i][2]['Cal'])
    cagr_rank = sorted(range(n), key=lambda i: -all_results[i][2]['CAGR'])

    ranks = {}
    for rank, idx in enumerate(sh_rank):
        ranks.setdefault(idx, []).append(rank + 1)
    for rank, idx in enumerate(cal_rank):
        ranks[idx].append(rank + 1)
    for rank, idx in enumerate(cagr_rank):
        ranks[idx].append(rank + 1)

    rank_sums = [(idx, sum(ranks[idx])) for idx in range(n)]
    rank_sums.sort(key=lambda x: x[1])

    print(f"{'Rank':>4s} {'Signal':<8s} {'Guard':<20s} | {'Sh':>6s} {'σSh':>5s} {'Cal':>5s} {'CAGR':>7s} {'MDD':>7s} {'RkSum':>5s}")
    print("-" * 80)
    for rank, (idx, rsum) in enumerate(rank_sums[:20], 1):
        sig, grd, m = all_results[idx]
        print(f"{rank:>4d} {sig:<8s} {grd:<20s} | "
              f"{m['Sharpe']:>6.3f} {m['sigma_Sh']:>5.3f} "
              f"{m['Cal']:>5.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} {rsum:>5d}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
