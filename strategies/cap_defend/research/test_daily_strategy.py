#!/usr/bin/env python3
"""선물 D(일봉) 전략 테스트: 단일 + 앙상블.

1) D 전략 단독 성능 (다양한 파라미터)
2) D+4h2+1h1 앙상블 vs 4h1+4h2+1h1 baseline
"""

import os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data, run
from futures_live_config import CURRENT_LIVE_COMBO, CURRENT_STRATEGIES, START, END
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


def generate_trace(data, cfg):
    run_cfg = dict(cfg)
    interval = run_cfg.pop("interval")
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=START, end_date=END, _trace=trace, **run_cfg)
    return trace


def run_single(data, cfg, label=""):
    """단일 전략 백테스트 (1x leverage, no engine)."""
    run_cfg = dict(cfg)
    interval = run_cfg.pop("interval")
    bars, funding = data[interval]
    result = run(bars, funding, interval=interval, leverage=1.0,
                 start_date=START, end_date=END, **run_cfg)
    return result


def run_ensemble(data, traces_dict, combo_weights, bars_1h, funding_1h, all_dates):
    """앙상블 백테스트."""
    combined = combine_targets(traces_dict, combo_weights, all_dates)
    engine = SingleAccountEngine(bars_1h, funding_1h, **ENGINE_KWARGS)
    return engine.run(combined)


# D 전략 후보들
D_CONFIGS = {
    # 현물 V18과 동일
    "D_spot(SMA50,M30/90,V5%)": dict(
        interval="D",
        sma_days=50, mom_short_days=30, mom_long_days=90,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=10,
    ),
    # 이전 선물D 최적
    "D_prev(SMA40,M15/90)": dict(
        interval="D",
        sma_days=40, mom_short_days=15, mom_long_days=90,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=10,
    ),
    # SMA 인접값
    "D(SMA45,M30/90)": dict(
        interval="D",
        sma_days=45, mom_short_days=30, mom_long_days=90,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=10,
    ),
    "D(SMA55,M30/90)": dict(
        interval="D",
        sma_days=55, mom_short_days=30, mom_long_days=90,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=10,
    ),
    # Mom 인접값
    "D(SMA50,M21/90)": dict(
        interval="D",
        sma_days=50, mom_short_days=21, mom_long_days=90,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=10,
    ),
    "D(SMA50,M30/60)": dict(
        interval="D",
        sma_days=50, mom_short_days=30, mom_long_days=60,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=10,
    ),
    # Health mode 변형
    "D(SMA50,M30/90,mom1vol)": dict(
        interval="D",
        sma_days=50, mom_short_days=30, mom_long_days=90,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode="mom1vol", vol_mode="daily", vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=10,
    ),
    # Snap interval 변형
    "D(SMA50,M30/90,snap7)": dict(
        interval="D",
        sma_days=50, mom_short_days=30, mom_long_days=90,
        canary_hyst=0.015, drift_threshold=0.0,
        dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
        health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
        n_snapshots=3, snap_interval_bars=7,
    ),
}


def main():
    t0 = time.time()
    print("Loading data...")
    data = {iv: load_data(iv) for iv in ["D", "4h", "1h"]}
    bars_1h, funding_1h = data["1h"]
    all_dates = bars_1h["BTC"].index[
        (bars_1h["BTC"].index >= START) & (bars_1h["BTC"].index <= END)
    ]
    print(f"  완료 ({time.time()-t0:.1f}s)\n")

    # ═══ Part 1: 단일 전략 비교 ═══
    print("=" * 80)
    print("Part 1: 단일 전략 성능 (1x leverage, 엔진 없음)")
    print("=" * 80)
    print(f"{'전략':<30s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Sharpe':>7s}")
    print("-" * 65)

    # Baseline: 현행 4h1
    m = run_single(data, CURRENT_STRATEGIES["live_4h1"], "4h1")
    print(f"{'4h1(현행)':<30s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m.get('Sharpe',0):>7.3f}")

    # D 후보들
    d_results = {}
    for name, cfg in D_CONFIGS.items():
        m = run_single(data, cfg, name)
        d_results[name] = m
        print(f"{name:<30s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m.get('Sharpe',0):>7.3f}")

    # ═══ Part 2: 앙상블 비교 ═══
    print(f"\n\n{'=' * 80}")
    print("Part 2: 3전략 앙상블 (5x leverage, SingleAccountEngine)")
    print("=" * 80)

    # Generate base traces
    print("Generating traces...")
    traces = {}
    for name, cfg in CURRENT_STRATEGIES.items():
        traces[name] = generate_trace(data, cfg)

    # Baseline ensemble
    baseline_combo = {k: traces[k] for k in CURRENT_LIVE_COMBO}
    m_base = run_ensemble(data, baseline_combo, CURRENT_LIVE_COMBO, bars_1h, funding_1h, all_dates)

    print(f"\n{'앙상블':<35s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stops':>5s}")
    print("-" * 70)
    print(f"{'baseline(4h1+4h2+1h1)':<35s} {m_base['CAGR']:>+8.1%} {m_base['MDD']:>+8.1%} {m_base['Cal']:>6.2f} {m_base['Liq']:>4d} {m_base.get('Stops',0):>5d}")

    # Top D candidates in ensemble
    # Sort by Calmar, pick top 4
    ranked = sorted(d_results.items(), key=lambda x: x[1]['Cal'], reverse=True)
    top_d = ranked[:min(5, len(ranked))]

    for d_name, _ in top_d:
        d_cfg = D_CONFIGS[d_name]
        d_trace = generate_trace(data, d_cfg)

        ensemble_traces = {
            "D": d_trace,
            "live_4h2": traces["live_4h2"],
            "live_1h1": traces["live_1h1"],
        }
        ensemble_weights = {"D": 1/3, "live_4h2": 1/3, "live_1h1": 1/3}
        m = run_ensemble(data, ensemble_traces, ensemble_weights, bars_1h, funding_1h, all_dates)
        print(f"{'D('+d_name.split('(')[1] if '(' in d_name else d_name:35s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}")

    # 4h2를 D로 바꾼 경우도 테스트
    print(f"\n--- 4h2를 D로 교체 ---")
    for d_name, _ in top_d[:3]:
        d_cfg = D_CONFIGS[d_name]
        d_trace = generate_trace(data, d_cfg)

        ensemble_traces = {
            "live_4h1": traces["live_4h1"],
            "D": d_trace,
            "live_1h1": traces["live_1h1"],
        }
        ensemble_weights = {"live_4h1": 1/3, "D": 1/3, "live_1h1": 1/3}
        m = run_ensemble(data, ensemble_traces, ensemble_weights, bars_1h, funding_1h, all_dates)
        print(f"{'4h1+D('+d_name.split('(')[1]+'+1h1' if '(' in d_name else d_name:35s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
