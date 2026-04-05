#!/usr/bin/env python3
"""과적합 검증: 최종 후보 앙상블의 robustness 테스트.

Test 1: 단일 전략 인접 파라미터 (vol_threshold, sma, mom_short, snap ±1단계)
Test 2: 서브기간 분리 (전반 2020.10~2023.06, 후반 2023.07~2026.03)
Test 3: 앙상블 파라미터 섭동 (각 전략 핵심 파라미터를 인접값으로 교체)
"""

import os, sys, time, itertools, json, copy
import numpy as np
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data, run
from futures_live_config import START, END
from futures_ensemble_engine import SingleAccountEngine, combine_targets

ENGINE_KWARGS = dict(
    leverage=5.0,
    stop_kind="prev_close_pct",
    stop_pct=0.15,
    stop_gate="cash_guard",
    stop_gate_cash_threshold=0.34,
    per_coin_leverage_mode="cap_mom_blend_543_cash",
    leverage_floor=3.0,
    leverage_mid=4.0,
    leverage_ceiling=5.0,
    leverage_cash_threshold=0.34,
    leverage_partial_cash_threshold=0.0,
    leverage_count_floor_max=2,
    leverage_count_mid_max=4,
    leverage_canary_floor_gap=0.015,
    leverage_canary_mid_gap=0.04,
    leverage_canary_high_gap=0.08,
    leverage_canary_sma_bars=1200,
    leverage_mom_lookback_bars=24 * 30,
    leverage_vol_lookback_bars=24 * 90,
)

FIXED_BASE = dict(
    canary_hyst=0.015,
    drift_threshold=0.0,
    dd_threshold=0,
    dd_lookback=0,
    bl_drop=0,
    bl_days=0,
    n_snapshots=3,
)

# ═══ 최종 후보 전략 정의 ═══

# 후보 1위: 4전략 (Cal=6.86)
COMBO_4 = {
    "2h_d003": dict(interval="2h", sma_bars=60, mom_short_bars=20, mom_long_bars=720,
                    health_mode="mom2vol", vol_mode="daily", vol_threshold=0.03,
                    snap_interval_bars=120, **FIXED_BASE),
    "4h_b60_sn60": dict(interval="4h", sma_bars=240, mom_short_bars=10, mom_long_bars=720,
                        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
                        snap_interval_bars=60, **FIXED_BASE),
    "4h_b60_M20": dict(interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=30,
                       health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
                       snap_interval_bars=60, **FIXED_BASE),
    "2h_b60_S120": dict(interval="2h", sma_bars=120, mom_short_bars=20, mom_long_bars=720,
                        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
                        snap_interval_bars=120, **FIXED_BASE),
}

# 후보 2위: 3전략 (Cal=6.49) — 4전략에서 마지막 제거
COMBO_3 = {k: v for k, v in list(COMBO_4.items())[:3]}

# 후보 3위: 2전략 (Cal=6.44)
COMBO_2 = {k: v for k, v in list(COMBO_4.items())[:2]}

# 현행 baseline
COMBO_BASELINE = {
    "live_1h1": dict(interval="1h", sma_bars=168, mom_short_bars=36, mom_long_bars=720,
                     health_mode="mom2vol", vol_mode="bar", vol_threshold=0.80,
                     snap_interval_bars=27, **FIXED_BASE),
    "live_4h1": dict(interval="4h", sma_bars=240, mom_short_bars=10, mom_long_bars=30,
                     health_mode="mom1vol", vol_mode="daily", vol_threshold=0.05,
                     snap_interval_bars=120, **FIXED_BASE),
    "live_4h2": dict(interval="4h", sma_bars=120, mom_short_bars=20, mom_long_bars=120,
                     health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
                     snap_interval_bars=21, **FIXED_BASE),
}


# ═══ 인접 파라미터 그리드 ═══

# 각 전략의 핵심 파라미터 인접값
PERTURBATIONS = {
    "2h_d003": {
        "vol_threshold": [0.02, 0.025, 0.03, 0.035, 0.04, 0.05],
        "sma_bars": [40, 50, 60, 70, 80],
        "mom_short_bars": [10, 15, 20, 25, 30],
        "snap_interval_bars": [60, 90, 120, 150, 180],
    },
    "4h_b60_sn60": {
        "vol_threshold": [0.50, 0.55, 0.60, 0.65, 0.70, 0.80],
        "sma_bars": [168, 200, 240, 280, 336],
        "mom_short_bars": [5, 8, 10, 15, 20],
        "snap_interval_bars": [30, 45, 60, 90, 120],
    },
    "4h_b60_M20": {
        "vol_threshold": [0.50, 0.55, 0.60, 0.65, 0.70, 0.80],
        "sma_bars": [168, 200, 240, 280, 336],
        "mom_short_bars": [10, 15, 20, 25, 30],
        "snap_interval_bars": [30, 45, 60, 90, 120],
    },
    "2h_b60_S120": {
        "vol_threshold": [0.50, 0.55, 0.60, 0.65, 0.70, 0.80],
        "sma_bars": [80, 100, 120, 150, 180],
        "mom_short_bars": [10, 15, 20, 25, 30],
        "snap_interval_bars": [60, 90, 120, 150, 180],
    },
}


# ═══ Global data ═══
_DATA = {}
_BARS_1H = None
_FUNDING_1H = None


def run_ensemble(combo, start_date=START, end_date=END):
    """앙상블 조합을 5x로 실행. combo: {name: cfg_with_interval}"""
    traces = {}
    for name, cfg in combo.items():
        c = dict(cfg)
        iv = c.pop('interval')
        bars, funding = _DATA[iv]
        trace = []
        run(bars, funding, interval=iv, leverage=1.0,
            start_date=start_date, end_date=end_date, _trace=trace, **c)
        traces[name] = trace

    bars_1h, funding_1h = _DATA['1h']
    all_dates = bars_1h["BTC"].index[
        (bars_1h["BTC"].index >= start_date) & (bars_1h["BTC"].index <= end_date)
    ]
    weights = {k: 1.0 / len(combo) for k in combo}
    combined = combine_targets(traces, weights, all_dates)
    engine = SingleAccountEngine(bars_1h, funding_1h, **ENGINE_KWARGS)
    m = engine.run(combined)
    return m


def run_single_1x(cfg_with_interval, start_date=START, end_date=END):
    """단일 전략 1x 실행."""
    c = dict(cfg_with_interval)
    iv = c.pop('interval')
    bars, funding = _DATA[iv]
    m = run(bars, funding, interval=iv, leverage=1.0,
            start_date=start_date, end_date=end_date, **c)
    return m


def main():
    global _DATA, _BARS_1H, _FUNDING_1H
    t0 = time.time()
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robustness_results.txt")
    outf = open(outpath, 'w')

    def log(msg=""):
        print(msg, flush=True)
        outf.write(msg + "\n")
        outf.flush()

    log("과적합 검증 테스트")
    log(f"기간: {START} ~ {END}")
    log()

    # Load data
    log("데이터 로딩...")
    for iv in ['4h', '2h', '1h']:
        _DATA[iv] = load_data(iv)
    _BARS_1H, _FUNDING_1H = _DATA['1h']
    log(f"  완료 ({time.time()-t0:.0f}s)")

    # ═══ Test 1: 단일 전략 인접 파라미터 (1x) ═══
    log(f"\n{'='*110}")
    log("Test 1: 단일 전략 인접 파라미터 안정성 (1x)")
    log('='*110)

    for strat_name, base_cfg in COMBO_4.items():
        log(f"\n  --- {strat_name} ---")
        base_m = run_single_1x(base_cfg)
        log(f"  기준: Cal={base_m['Cal']:.2f} CAGR={base_m['CAGR']:+.1%} MDD={base_m['MDD']:+.1%} Sharpe={base_m['Sharpe']:.2f}")

        if strat_name not in PERTURBATIONS:
            continue

        for param, values in PERTURBATIONS[strat_name].items():
            log(f"\n  {param}:")
            results = []
            for val in values:
                cfg = dict(base_cfg)
                cfg[param] = val
                m = run_single_1x(cfg)
                marker = " ◀" if val == base_cfg[param] else ""
                log(f"    {val:>8}: Cal={m['Cal']:>5.2f} CAGR={m['CAGR']:>+7.1%} MDD={m['MDD']:>+7.1%} Sharpe={m['Sharpe']:>5.2f}{marker}")
                results.append(m['Cal'])

            # Plateau 분석
            base_idx = values.index(base_cfg[param]) if base_cfg[param] in values else -1
            if base_idx >= 0 and len(results) >= 3:
                cal_range = max(results) - min(results)
                # 인접값과의 차이
                adjacent_cals = []
                if base_idx > 0:
                    adjacent_cals.append(results[base_idx - 1])
                if base_idx < len(results) - 1:
                    adjacent_cals.append(results[base_idx + 1])
                if adjacent_cals:
                    max_drop = results[base_idx] - min(adjacent_cals)
                    pct_drop = max_drop / results[base_idx] * 100 if results[base_idx] > 0 else 0
                    log(f"    → 전체 range: {cal_range:.2f}, 인접 최대 하락: {max_drop:.2f} ({pct_drop:.1f}%)")

    log(f"\n  Test 1 완료: {time.time()-t0:.0f}s")

    # ═══ Test 2: 서브기간 분리 ═══
    log(f"\n{'='*110}")
    log("Test 2: 서브기간 분리 (전반/후반)")
    log('='*110)

    periods = [
        ("전체", START, END),
        ("전반(2020.10~2023.06)", START, "2023-06-30"),
        ("후반(2023.07~2026.03)", "2023-07-01", END),
    ]

    combos = [
        ("후보1: 4전략", COMBO_4),
        ("후보2: 3전략", COMBO_3),
        ("후보3: 2전략", COMBO_2),
        ("baseline: 현행", COMBO_BASELINE),
    ]

    log(f"\n  {'조합':<25s} {'기간':<25s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'Liq':>4s} {'Stop':>5s} {'Sharpe':>7s}")
    log(f"  {'-'*100}")

    for combo_name, combo in combos:
        for period_name, start, end in periods:
            try:
                m = run_ensemble(combo, start_date=start, end_date=end)
                log(f"  {combo_name:<25s} {period_name:<25s} {m['Cal']:>6.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Liq']:>4d} {m.get('Stops',0):>5d} {m['Sharpe']:>7.2f}")
            except Exception as e:
                log(f"  {combo_name:<25s} {period_name:<25s} ERROR: {e}")
        log()

    log(f"\n  Test 2 완료: {time.time()-t0:.0f}s")

    # ═══ Test 3: 앙상블 파라미터 섭동 (5x) ═══
    log(f"\n{'='*110}")
    log("Test 3: 앙상블 파라미터 섭동 안정성 (5x)")
    log("  각 전략의 vol_threshold를 인접값으로 교체한 앙상블 성능")
    log('='*110)

    # 4전략 앙상블의 vol_threshold 섭동
    log(f"\n  --- 후보1 (4전략) vol_threshold 섭동 ---")
    base_m = run_ensemble(COMBO_4)
    log(f"  기준: Cal={base_m['Cal']:.2f} CAGR={base_m['CAGR']:+.1%} MDD={base_m['MDD']:+.1%} Liq={base_m['Liq']} Sharpe={base_m['Sharpe']:.2f}")

    # 각 전략의 vol_threshold를 하나씩 변경
    for strat_name in COMBO_4:
        if strat_name not in PERTURBATIONS:
            continue
        vt_values = PERTURBATIONS[strat_name]["vol_threshold"]
        base_vt = COMBO_4[strat_name]["vol_threshold"]
        log(f"\n  {strat_name} vol_threshold 변경:")
        for vt in vt_values:
            combo_copy = {}
            for k, v in COMBO_4.items():
                combo_copy[k] = dict(v)
            combo_copy[strat_name]["vol_threshold"] = vt
            m = run_ensemble(combo_copy)
            marker = " ◀" if vt == base_vt else ""
            log(f"    {vt:>6}: Cal={m['Cal']:>5.2f} CAGR={m['CAGR']:>+7.1%} MDD={m['MDD']:>+7.1%} Liq={m['Liq']} Sharpe={m['Sharpe']:>5.2f}{marker}")

    # 3전략 앙상블 섭동
    log(f"\n  --- 후보2 (3전략) vol_threshold 섭동 ---")
    base_m = run_ensemble(COMBO_3)
    log(f"  기준: Cal={base_m['Cal']:.2f} CAGR={base_m['CAGR']:+.1%} MDD={base_m['MDD']:+.1%} Liq={base_m['Liq']} Sharpe={base_m['Sharpe']:.2f}")

    for strat_name in COMBO_3:
        if strat_name not in PERTURBATIONS:
            continue
        vt_values = PERTURBATIONS[strat_name]["vol_threshold"]
        base_vt = COMBO_3[strat_name]["vol_threshold"]
        log(f"\n  {strat_name} vol_threshold 변경:")
        for vt in vt_values:
            combo_copy = {}
            for k, v in COMBO_3.items():
                combo_copy[k] = dict(v)
            combo_copy[strat_name]["vol_threshold"] = vt
            m = run_ensemble(combo_copy)
            marker = " ◀" if vt == base_vt else ""
            log(f"    {vt:>6}: Cal={m['Cal']:>5.2f} CAGR={m['CAGR']:>+7.1%} MDD={m['MDD']:>+7.1%} Liq={m['Liq']} Sharpe={m['Sharpe']:>5.2f}{marker}")

    # 2전략 앙상블 섭동
    log(f"\n  --- 후보3 (2전략) vol_threshold 섭동 ---")
    base_m = run_ensemble(COMBO_2)
    log(f"  기준: Cal={base_m['Cal']:.2f} CAGR={base_m['CAGR']:+.1%} MDD={base_m['MDD']:+.1%} Liq={base_m['Liq']} Sharpe={base_m['Sharpe']:.2f}")

    for strat_name in COMBO_2:
        if strat_name not in PERTURBATIONS:
            continue
        vt_values = PERTURBATIONS[strat_name]["vol_threshold"]
        base_vt = COMBO_2[strat_name]["vol_threshold"]
        log(f"\n  {strat_name} vol_threshold 변경:")
        for vt in vt_values:
            combo_copy = {}
            for k, v in COMBO_2.items():
                combo_copy[k] = dict(v)
            combo_copy[strat_name]["vol_threshold"] = vt
            m = run_ensemble(combo_copy)
            marker = " ◀" if vt == base_vt else ""
            log(f"    {vt:>6}: Cal={m['Cal']:>5.2f} CAGR={m['CAGR']:>+7.1%} MDD={m['MDD']:>+7.1%} Liq={m['Liq']} Sharpe={m['Sharpe']:>5.2f}{marker}")

    log(f"\n  Test 3 완료: {time.time()-t0:.0f}s")

    # ═══ Test 4: SMA 섭동 앙상블 (5x) ═══
    log(f"\n{'='*110}")
    log("Test 4: 앙상블 SMA/mom_short 섭동 (5x, 4전략)")
    log('='*110)

    for strat_name in COMBO_4:
        if strat_name not in PERTURBATIONS:
            continue
        for param in ["sma_bars", "mom_short_bars"]:
            values = PERTURBATIONS[strat_name][param]
            base_val = COMBO_4[strat_name][param]
            log(f"\n  {strat_name} {param}:")
            for val in values:
                combo_copy = {}
                for k, v in COMBO_4.items():
                    combo_copy[k] = dict(v)
                combo_copy[strat_name][param] = val
                m = run_ensemble(combo_copy)
                marker = " ◀" if val == base_val else ""
                log(f"    {val:>6}: Cal={m['Cal']:>5.2f} CAGR={m['CAGR']:>+7.1%} MDD={m['MDD']:>+7.1%} Liq={m['Liq']} Sharpe={m['Sharpe']:>5.2f}{marker}")

    log(f"\n  Test 4 완료: {time.time()-t0:.0f}s")

    log(f"\n\n총 소요: {time.time()-t0:.1f}s")
    log(f"결과: {outpath}")
    outf.close()


if __name__ == "__main__":
    main()
