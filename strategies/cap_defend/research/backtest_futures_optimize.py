#!/usr/bin/env python3
"""선물 백테스트 파라미터 최적화 — 4h / 1h 각각 최적 전략 탐색.

Grid search:
  - SMA 기간, 모멘텀 윈도우, 카나리 히스테리시스
  - Drift 임계값, DD/BL 파라미터
  - daily_gate (일간 체크 빈도 제한)
  - 레버리지

과적합 방지: 10-anchor 평균 대신 단일 앵커이므로 plateau 확인 필수.
"""

import time, sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_futures_full import load_data, run

START = '2020-10-01'
END = '2026-03-28'


def sweep(bars, funding, interval, configs, leverage=1.0):
    """configs: list of dict (run kwargs). 각 config 실행 후 결과 반환."""
    results = []
    for cfg in configs:
        m = run(bars, funding, interval=interval, leverage=leverage,
                start_date=START, end_date=END, **cfg)
        if not m or m.get('CAGR', 0) == 0:
            continue
        results.append({**cfg, **m})
    return results


def build_grid():
    """파라미터 그리드 생성."""
    grid = []
    for sma in [20, 30, 40, 50, 60]:
        for mom_s in [15, 21, 30, 45]:
            for mom_l in [60, 90, 120]:
                if mom_l <= mom_s:
                    continue
                for hyst in [0.0, 0.01, 0.015, 0.02, 0.03]:
                    for drift in [0.0, 0.05, 0.10, 0.15, 0.20, 0.30]:
                        for dd_th in [0, -0.15, -0.20, -0.25, -0.30]:
                            for dg in [False, True]:
                                grid.append(dict(
                                    sma_days=sma,
                                    mom_short_days=mom_s,
                                    mom_long_days=mom_l,
                                    canary_hyst=hyst,
                                    drift_threshold=drift,
                                    dd_threshold=dd_th,
                                    dd_lookback=60 if dd_th != 0 else 0,
                                    daily_gate=dg,
                                ))
    return grid


def build_grid_reduced():
    """핵심 축만 탐색하는 축소 그리드. Drift 제외, BL on/off 추가."""
    grid = []
    for sma in [30, 40, 50, 60]:
        for mom_s in [15, 21, 30]:
            for mom_l in [60, 90]:
                if mom_l <= mom_s:
                    continue
                for hyst in [0.0, 0.015, 0.03]:
                    for dd_th in [0, -0.25]:
                        for bl in [0, -0.15]:
                            for dg in [False, True]:
                                grid.append(dict(
                                    sma_days=sma,
                                    mom_short_days=mom_s,
                                    mom_long_days=mom_l,
                                    canary_hyst=hyst,
                                    drift_threshold=0.0,  # drift 제외
                                    dd_threshold=dd_th,
                                    dd_lookback=60 if dd_th != 0 else 0,
                                    bl_drop=bl,
                                    daily_gate=dg,
                                ))
    return grid


def print_top(results, n=20, sort_key='Cal'):
    """상위 N개 결과 출력."""
    results.sort(key=lambda x: x.get(sort_key, 0), reverse=True)
    header_params = "SMA  Mom   Hyst   DD    BL   Gate"
    header_metrics = "  Sharpe    CAGR     MDD  Calmar  Liq  Rebal"
    print(f"  {header_params} | {header_metrics}")
    print(f"  {'-' * 85}")
    for r in results[:n]:
        sma = r['sma_days']
        mom = f"{r['mom_short_days']}/{r['mom_long_days']}"
        hyst = f"{r['canary_hyst']:.1%}" if r['canary_hyst'] > 0 else "off"
        dd = f"{r['dd_threshold']:.0%}" if r['dd_threshold'] != 0 else "off"
        bl = f"{r.get('bl_drop', -0.15):.0%}" if r.get('bl_drop', -0.15) != 0 else "off"
        gate = "Y" if r['daily_gate'] else "N"
        liq = f"💀{r['Liq']}" if r['Liq'] > 0 else ""
        print(f"  {sma:>3d}  {mom:<5s} {hyst:>5s} {dd:>5s} {bl:>5s}  {gate}"
              f"  | {r['Sharpe']:>6.2f} {r['CAGR']:>+8.1%} {r['MDD']:>+8.1%} {r['Cal']:>7.2f} {liq:>4s} {r['Rebal']:>6d}")


if __name__ == '__main__':
    t0 = time.time()

    grid = build_grid_reduced()
    print(f"축소 그리드: {len(grid)} 조합\n")

    for interval in ['4h', '1h']:
        bars, funding = load_data(interval)
        if 'BTC' not in bars:
            print(f"  {interval}: BTC 데이터 없음")
            continue

        print(f"\n{'='*90}")
        print(f"  {interval} | 1x leverage | 파라미터 탐색")
        print(f"{'='*90}")

        results_1x = sweep(bars, funding, interval, grid, leverage=1.0)
        print(f"\n  총 {len(results_1x)}개 유효 결과 (1x)")
        print(f"\n  ── Top 20 by Calmar (1x) ──")
        print_top(results_1x, 20, 'Cal')

        # 상위 5개 config로 레버리지 테스트
        top5 = sorted(results_1x, key=lambda x: x.get('Cal', 0), reverse=True)[:5]
        print(f"\n  ── Top 5 config × 레버리지 ──")
        for i, cfg in enumerate(top5):
            cfg_params = {k: cfg[k] for k in ['sma_days', 'mom_short_days', 'mom_long_days',
                                                'canary_hyst', 'drift_threshold', 'dd_threshold',
                                                'dd_lookback', 'daily_gate']}
            print(f"\n  Config #{i+1}: SMA{cfg['sma_days']} Mom{cfg['mom_short_days']}/{cfg['mom_long_days']}"
                  f" Hyst{cfg['canary_hyst']:.1%} Drift{cfg['drift_threshold']:.0%}"
                  f" DD{cfg['dd_threshold']:.0%} Gate={'Y' if cfg['daily_gate'] else 'N'}")
            print(f"  {'Lev':<5s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s} {'Liq':>4s} {'Rb':>5s}")
            for lev in [1.0, 1.5, 2.0, 3.0]:
                m = run(bars, funding, interval=interval, leverage=lev,
                        start_date=START, end_date=END, **cfg_params)
                if not m:
                    continue
                liq = f"💀{m['Liq']}" if m['Liq'] > 0 else ""
                print(f"  {lev:<5.1f} {m['Sharpe']:>7.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%}"
                      f" {m['Cal']:>7.2f} {liq:>4s} {m['Rebal']:>5d}")

        sys.stdout.flush()

    print(f"\n총 소요: {time.time() - t0:.0f}s")
