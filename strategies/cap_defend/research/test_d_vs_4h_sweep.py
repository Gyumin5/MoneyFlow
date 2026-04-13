#!/usr/bin/env python3
"""D vs 4h 현물 전략 공정 비교 — 봉 기반 엔진(barfreq).

1단계: 신호 coarse grid (가드 OFF) — D와 4h 동일 일수 기준 파라미터
2단계: 상위 후보에 가드 ablation
3단계: D+4h 앙상블

일 기준 master grid → 봉 단위 자동 환산.
Vol_cap: daily_eq / sqrt(bars_per_day)
"""
import os, sys, time, math, itertools
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest_spot_barfreq import load_binance_prices, run_backtest
from coin_engine import Params

# ─── 고정 조건 ──────────────────────────────────────────────
N_PICKS = 3
TOP_N = 40
TX_COST = 0.004
N_SNAP = 3
SELECTION = 'baseline'
WEIGHTING = 'baseline'  # n=3 EW = 각 33.3%

# ─── 일 기준 Master Grid ──────────────────────────────────
SMA_DAYS = [30, 50, 80]
MOM_SHORT_DAYS = [10, 20, 30, 40]
MOM_LONG_DAYS = [60, 90, 120]
VOL_CAP_DAILY = [0.05, 0.06, 0.07]
SNAP_INTERVAL_DAYS = [14, 21, 28, 42]

# 가드 패키지 (일 기준)
GUARD_PACKAGES = {
    'NoGuard': dict(dd_lookback=0, dd_threshold=0, bl_drop=0, bl_bars=0,
                    crash_enabled=False, crash_threshold=-0.10, crash_cooldown_bars=0,
                    drift_threshold=0),
    'DD60_25': dict(dd_lookback=60, dd_threshold=-0.25, bl_drop=0, bl_bars=0,
                    crash_enabled=False, crash_threshold=-0.10, crash_cooldown_bars=0,
                    drift_threshold=0),
    'BL_15_7': dict(dd_lookback=0, dd_threshold=0, bl_drop=-0.15, bl_bars=7,
                    crash_enabled=False, crash_threshold=-0.10, crash_cooldown_bars=0,
                    drift_threshold=0),
    'Crash_10_3': dict(dd_lookback=0, dd_threshold=0, bl_drop=0, bl_bars=0,
                       crash_enabled=True, crash_threshold=-0.10, crash_cooldown_bars=3,
                       drift_threshold=0),
    'Drift_10': dict(dd_lookback=0, dd_threshold=0, bl_drop=0, bl_bars=0,
                     crash_enabled=False, crash_threshold=-0.10, crash_cooldown_bars=0,
                     drift_threshold=0.10),
    'Base': dict(dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_bars=7,
                 crash_enabled=True, crash_threshold=-0.10, crash_cooldown_bars=3,
                 drift_threshold=0.10),
    'Tight': dict(dd_lookback=30, dd_threshold=-0.15, bl_drop=-0.10, bl_bars=7,
                  crash_enabled=True, crash_threshold=-0.08, crash_cooldown_bars=3,
                  drift_threshold=0.05),
    'Loose': dict(dd_lookback=90, dd_threshold=-0.30, bl_drop=-0.20, bl_bars=14,
                  crash_enabled=True, crash_threshold=-0.12, crash_cooldown_bars=5,
                  drift_threshold=0.15),
}


def convert_guards_to_bars(guard_pkg, bpd):
    """일 기준 가드 파라미터를 봉 단위로 변환."""
    g = dict(guard_pkg)
    if g['dd_lookback'] > 0:
        g['dd_lookback'] = round(g['dd_lookback'] * bpd)
    if g['bl_bars'] > 0:
        g['bl_bars'] = round(g['bl_bars'] * bpd)
    if g['crash_cooldown_bars'] > 0:
        g['crash_cooldown_bars'] = round(g['crash_cooldown_bars'] * bpd)
    return g


def run_single(prices, um, interval, bpd, sma_d, ms_d, ml_d, vc_d, snap_d,
               guard_pkg=None, start='2018-01-01', end='2026-03-31'):
    """단일 조합 실행."""
    if ml_d <= ms_d:
        return None  # mom_long > mom_short 필수

    sma_bars = round(sma_d * bpd)
    ms_bars = round(ms_d * bpd)
    ml_bars = round(ml_d * bpd)
    vol_window = round(90 * bpd)
    vc_bar = vc_d / math.sqrt(bpd)
    snap_bars = max(round(snap_d * bpd), N_SNAP)

    p = Params(
        sma_period=sma_bars, canary_band=1.5, vote_smas=(sma_bars,),
        health='HK', health_sma=0,
        health_mom_short=ms_bars, health_mom_long=ml_bars,
        health_vol_window=vol_window, vol_cap=vc_bar,
        n_picks=N_PICKS, top_n=TOP_N,
        selection=SELECTION, weighting=WEIGHTING,
        start_date=start, end_date=end,
    )

    gkw = {'dd_lookback': 0, 'dd_threshold': 0, 'bl_drop': 0, 'bl_bars': 0,
            'crash_enabled': False, 'crash_threshold': -0.10, 'crash_cooldown_bars': 0,
            'drift_threshold': 0}
    if guard_pkg:
        gkw = convert_guards_to_bars(guard_pkg, bpd)

    r = run_backtest(prices, um, p,
                     bars_per_day=bpd, snap_interval_bars=snap_bars,
                     mode='spot', leverage=1.0, tx_cost=TX_COST,
                     n_snap=N_SNAP, **gkw)
    m = r['metrics']
    cal = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
    return dict(
        Sharpe=m['Sharpe'], CAGR=m['CAGR'], MDD=m['MDD'], Cal=cal,
        Rebal=r['rebal_count'],
        DD=r.get('dd_exit_count', 0),
    )


def run_10anchor(prices, um, interval, bpd, sma_d, ms_d, ml_d, vc_d, snap_d,
                 guard_pkg=None, start='2018-01-01', end='2026-03-31'):
    """10-anchor 평균."""
    results = []
    for offset in range(10):
        p = Params(
            sma_period=round(sma_d * bpd), canary_band=1.5,
            vote_smas=(round(sma_d * bpd),),
            health='HK', health_sma=0,
            health_mom_short=round(ms_d * bpd),
            health_mom_long=round(ml_d * bpd),
            health_vol_window=round(90 * bpd),
            vol_cap=vc_d / math.sqrt(bpd),
            n_picks=N_PICKS, top_n=TOP_N,
            selection=SELECTION, weighting=WEIGHTING,
            start_date=start, end_date=end,
        )
        snap_bars = max(round(snap_d * bpd), N_SNAP)
        gkw = {'dd_lookback': 0, 'dd_threshold': 0, 'bl_drop': 0, 'bl_bars': 0,
               'crash_enabled': False, 'crash_threshold': -0.10, 'crash_cooldown_bars': 0,
               'drift_threshold': 0}
        if guard_pkg:
            gkw = convert_guards_to_bars(guard_pkg, bpd)

        r = run_backtest(prices, um, p,
                         bars_per_day=bpd, snap_interval_bars=snap_bars,
                         mode='spot', leverage=1.0, tx_cost=TX_COST,
                         n_snap=N_SNAP, anchor_offset=offset, **gkw)
        m = r['metrics']
        cal = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        results.append(dict(Sharpe=m['Sharpe'], CAGR=m['CAGR'], MDD=m['MDD'],
                            Cal=cal, Rebal=r['rebal_count'],
                            DD=r.get('dd_exit_count', 0)))
    return {
        'Sharpe': np.mean([r['Sharpe'] for r in results]),
        'CAGR': np.mean([r['CAGR'] for r in results]),
        'MDD': np.mean([r['MDD'] for r in results]),
        'Cal': np.mean([r['Cal'] for r in results]),
        'Rebal': np.mean([r['Rebal'] for r in results]),
        'DD': np.mean([r['DD'] for r in results]),
        'sigma_Sh': np.std([r['Sharpe'] for r in results]),
    }


def fmt(m):
    return (f"Sh {m['Sharpe']:.3f}(σ{m['sigma_Sh']:.3f}) "
            f"Cal {m['Cal']:.2f} CAGR {m['CAGR']:+.1%} MDD {m['MDD']:+.1%}")


def main():
    t0 = time.time()

    # ─── 데이터 로드 ──────────────────────
    print("데이터 로딩...")
    data = {}
    for iv, bpd in [('D', 1), ('4h', 6)]:
        prices, um, _ = load_binance_prices(iv, top_n=TOP_N)
        data[iv] = (prices, um, bpd)
        print(f"  {iv}: {len(prices)} coins")
    print(f"  완료 ({time.time()-t0:.1f}s)\n")

    # ═══ 1단계: 신호 coarse grid (가드 OFF) ═══
    print("=" * 90)
    print("1단계: 신호 Coarse Grid (NoGuard, 단일 앵커)")
    print("=" * 90)

    combos = [(sma, ms, ml, vc, snap)
              for sma in SMA_DAYS for ms in MOM_SHORT_DAYS
              for ml in MOM_LONG_DAYS for vc in VOL_CAP_DAILY
              for snap in SNAP_INTERVAL_DAYS
              if ml > ms]
    print(f"  총 {len(combos)} 조합 x 2봉(D, 4h)\n")

    print(f"{'IV':>3s} {'SMA':>4s} {'Ms':>3s} {'Ml':>4s} {'VC':>5s} {'Snap':>4s} | "
          f"{'Sh':>6s} {'Cal':>5s} {'CAGR':>7s} {'MDD':>7s}")
    print("-" * 70)

    all_results = []
    count = 0
    for iv in ['D', '4h']:
        prices, um, bpd = data[iv]
        for sma, ms, ml, vc, snap in combos:
            count += 1
            r = run_single(prices, um, iv, bpd, sma, ms, ml, vc, snap)
            if r is None:
                continue
            label = f"{iv} S{sma}/Ms{ms}/Ml{ml}/V{vc}/Sn{snap}"
            print(f"{iv:>3s} {sma:>4d} {ms:>3d} {ml:>4d} {vc:>5.2f} {snap:>4d} | "
                  f"{r['Sharpe']:>6.3f} {r['Cal']:>5.2f} {r['CAGR']:>+7.1%} {r['MDD']:>+7.1%}"
                  f"  [{count}]")
            all_results.append((iv, sma, ms, ml, vc, snap, r))

    # ─── 순위합 Top 10 (D, 4h 각각) ──────────────────
    for iv in ['D', '4h']:
        subset = [(i, r) for i, (v, *_, r) in enumerate(all_results) if v == iv]
        if not subset:
            continue
        n = len(subset)
        sh_order = sorted(range(n), key=lambda j: -subset[j][1]['Sharpe'])
        cal_order = sorted(range(n), key=lambda j: -subset[j][1]['Cal'])
        cagr_order = sorted(range(n), key=lambda j: -subset[j][1]['CAGR'])

        ranks = {}
        for rank, j in enumerate(sh_order): ranks.setdefault(j, []).append(rank + 1)
        for rank, j in enumerate(cal_order): ranks[j].append(rank + 1)
        for rank, j in enumerate(cagr_order): ranks[j].append(rank + 1)

        ranked = sorted(range(n), key=lambda j: sum(ranks[j]))

        print(f"\n{'='*70}")
        print(f"  {iv} Top 10 (순위합)")
        print(f"{'='*70}")
        print(f"{'Rk':>3s} {'SMA':>4s} {'Ms':>3s} {'Ml':>4s} {'VC':>5s} {'Sn':>3s} | "
              f"{'Sh':>6s} {'Cal':>5s} {'CAGR':>7s} {'MDD':>7s} {'RkS':>4s}")
        for rank_i, j in enumerate(ranked[:10], 1):
            orig_i = subset[j][0]
            _, sma, ms, ml, vc, snap, r = all_results[orig_i]
            rsum = sum(ranks[j])
            print(f"{rank_i:>3d} {sma:>4d} {ms:>3d} {ml:>4d} {vc:>5.2f} {snap:>3d} | "
                  f"{r['Sharpe']:>6.3f} {r['Cal']:>5.2f} {r['CAGR']:>+7.1%} {r['MDD']:>+7.1%} {rsum:>4d}")

    # ═══ 2단계: 상위 5개에 10-anchor + 가드 ablation ═══
    print(f"\n\n{'='*90}")
    print("2단계: 상위 후보 x 가드 ablation (10-anchor)")
    print(f"{'='*90}\n")

    # 각 봉에서 상위 5개 추출
    top_candidates = {}
    for iv in ['D', '4h']:
        subset = [(i, r) for i, (v, *_, r) in enumerate(all_results) if v == iv]
        n = len(subset)
        sh_order = sorted(range(n), key=lambda j: -subset[j][1]['Sharpe'])
        cal_order = sorted(range(n), key=lambda j: -subset[j][1]['Cal'])
        cagr_order = sorted(range(n), key=lambda j: -subset[j][1]['CAGR'])
        ranks = {}
        for rank, j in enumerate(sh_order): ranks.setdefault(j, []).append(rank + 1)
        for rank, j in enumerate(cal_order): ranks[j].append(rank + 1)
        for rank, j in enumerate(cagr_order): ranks[j].append(rank + 1)
        ranked = sorted(range(n), key=lambda j: sum(ranks[j]))
        top_candidates[iv] = []
        for j in ranked[:5]:
            orig_i = subset[j][0]
            _, sma, ms, ml, vc, snap, _ = all_results[orig_i]
            top_candidates[iv].append((sma, ms, ml, vc, snap))

    # 가드 ablation
    for iv in ['D', '4h']:
        prices, um, bpd = data[iv]
        print(f"\n--- {iv} 상위 5개 x 가드 패키지 (10-anchor) ---")
        print(f"{'SMA':>4s} {'Ms':>3s} {'Ml':>4s} {'VC':>5s} {'Sn':>3s} {'Guard':<12s} | "
              f"{'Sh':>6s} {'σ':>5s} {'Cal':>5s} {'CAGR':>7s} {'MDD':>7s}")
        print("-" * 75)

        for sma, ms, ml, vc, snap in top_candidates[iv]:
            for gname, gpkg in GUARD_PACKAGES.items():
                t1 = time.time()
                m = run_10anchor(prices, um, iv, bpd, sma, ms, ml, vc, snap,
                                 guard_pkg=gpkg)
                elapsed = time.time() - t1
                print(f"{sma:>4d} {ms:>3d} {ml:>4d} {vc:>5.2f} {snap:>3d} {gname:<12s} | "
                      f"{fmt(m)}  ({elapsed:.0f}s)")
            print()

    # ═══ 3단계: D+4h 앙상블 ═══
    print(f"\n\n{'='*90}")
    print("3단계: D+4h 앙상블 (TODO — 2단계 결과로 최적 선택 후)")
    print(f"{'='*90}\n")
    print("앙상블은 2단계 결과 확인 후 수동으로 구성 예정.")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
