#!/usr/bin/env python3
"""15m/30m/2h 타임프레임 스윕: 단일 + 앙상블.

1) 각 타임프레임에서 파라미터 스윕 → best 선정
2) best를 4h1 대신 넣어 앙상블 비교
"""

import os, sys, time, itertools

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


def run_single(data, cfg):
    run_cfg = dict(cfg)
    interval = run_cfg.pop("interval")
    bars, funding = data[interval]
    return run(bars, funding, interval=interval, leverage=1.0,
               start_date=START, end_date=END, **run_cfg)


def run_ensemble(data, traces_dict, combo_weights, bars_1h, funding_1h, all_dates):
    combined = combine_targets(traces_dict, combo_weights, all_dates)
    engine = SingleAccountEngine(bars_1h, funding_1h, **ENGINE_KWARGS)
    return engine.run(combined)


# ─── 파라미터 그리드 ───
# 각 타임프레임에 맞게 bars 단위로 변환
# 기준: 현물 V18 = SMA50d, Mom30d/90d
# 현행 4h1 = SMA240bars(40d), Mom10/30bars

# Bar-based grids — 현행 전략 bar 수 기준으로 스윕
# 현행: 1h1(S168,M36/720), 4h1(S240,M10/30), 4h2(S120,M20/120)
# 풀 서치: 현행 3전략 전체 범위 포괄
# 1h1: S168,M36/720,mom2vol,bar0.80,snap27
# 4h1: S240,M10/30,mom1vol,daily0.05,snap120
# 4h2: S120,M20/120,mom2vol,bar0.60,snap21
TIMEFRAME_CONFIGS = {
    # 30m: 2h best 파라미터로만 검증 (조합당 2분이라 풀 서치 비현실적)
    # → Part 1 후 동적으로 2h top5 파라미터를 30m으로 테스트
    '2h': {
        'grids': [
            {'sma_bars': [120, 168, 240, 336],
             'mom_short_bars': [10, 20, 36],
             'mom_long_bars': [30, 120, 720],
             'health_mode': ['mom2vol', 'mom1vol'],
             'vol_mode': ['bar'],
             'vol_threshold': [0.60, 0.80],
             'snap_interval_bars': [21, 54],
            },
        ],
    },
}


def build_configs(interval, grid):
    """Grid에서 모든 조합 생성."""
    keys = list(grid.keys())
    values = [grid[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        cfg['interval'] = interval
        cfg['canary_hyst'] = 0.015
        cfg['drift_threshold'] = 0.0
        cfg['dd_threshold'] = 0
        cfg['dd_lookback'] = 0
        cfg['bl_drop'] = 0
        cfg['bl_days'] = 0
        cfg['n_snapshots'] = 3
        configs.append(cfg)
    return configs


def cfg_label(cfg):
    iv = cfg['interval']
    sma = cfg['sma_bars']
    ms = cfg['mom_short_bars']
    ml = cfg['mom_long_bars']
    hm = cfg['health_mode']
    si = cfg['snap_interval_bars']
    return f"{iv}(S{sma},M{ms}/{ml},{hm},si{si})"


def main():
    t0 = time.time()
    print("Loading data...")
    intervals_needed = ['1h', '4h', '2h', '30m']
    # 15m은 22만바×38코인으로 조합당 2.5분 → 스킵 (초기 결과도 CAGR -20%로 나쁨)
    print("  30m + 2h 테스트 (15m 스킵: 너무 느리고 성능 나쁨)")

    data = {}
    for iv in intervals_needed:
        print(f"  Loading {iv}...", end='', flush=True)
        data[iv] = load_data(iv)
        n = len(data[iv][0])
        print(f" {n} coins")

    bars_1h, funding_1h = data["1h"]
    all_dates = bars_1h["BTC"].index[
        (bars_1h["BTC"].index >= START) & (bars_1h["BTC"].index <= END)
    ]
    print(f"  완료 ({time.time()-t0:.1f}s)\n")

    # ═══ Part 1: 타임프레임별 스윕 ═══
    print("=" * 90)
    print("Part 1: 타임프레임별 파라미터 스윕 (단일 전략, 1x)")
    print("=" * 90)

    # Baseline: 4h1
    m_4h1 = run_single(data, CURRENT_STRATEGIES["live_4h1"])
    print(f"\n  baseline 4h1: CAGR={m_4h1['CAGR']:+.1%} MDD={m_4h1['MDD']:+.1%} Cal={m_4h1['Cal']:.2f}")

    all_tf_results = {}  # tf → [(label, cfg, metrics)]

    for tf, tf_cfg in TIMEFRAME_CONFIGS.items():
        if tf == '15m':
            continue  # 15m 스킵

        print(f"\n--- {tf} 스윕 ---")
        print(f"  {'설정':<45s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s}")
        print(f"  {'-'*70}")

        results = []
        for grid in tf_cfg['grids']:
            configs = build_configs(tf, grid)
            for cfg in configs:
                try:
                    m = run_single(data, cfg)
                    label = cfg_label(cfg)
                    results.append((label, cfg, m))
                    print(f"  {label:<45s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f}")
                except Exception as e:
                    print(f"  {cfg_label(cfg):<45s} ERROR: {e}")

        if results:
            results.sort(key=lambda x: x[2]['Cal'], reverse=True)
            all_tf_results[tf] = results
            best = results[0]
            print(f"\n  ★ Best {tf}: {best[0]} Cal={best[2]['Cal']:.2f}")
            print(f"    Top5:")
            for i, (l, c, m) in enumerate(results[:5], 1):
                print(f"    {i}. {l} Cal={m['Cal']:.2f} CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%}")

    # ═══ Part 1b: 2h top5 파라미터를 30m으로 검증 ═══
    if '2h' in all_tf_results and '30m' in data:
        print(f"\n--- 30m 검증 (2h top5 파라미터 적용) ---")
        print(f"  {'설정':<45s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s}")
        print(f"  {'-'*70}")
        results_30m = []
        for label_2h, cfg_2h, _ in all_tf_results['2h'][:5]:
            cfg_30m = dict(cfg_2h)
            cfg_30m['interval'] = '30m'
            try:
                m = run_single(data, cfg_30m)
                label = cfg_label(cfg_30m)
                results_30m.append((label, cfg_30m, m))
                print(f"  {label:<45s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f}")
            except Exception as e:
                print(f"  {cfg_label(cfg_30m):<45s} ERROR: {e}")
        if results_30m:
            results_30m.sort(key=lambda x: x[2]['Cal'], reverse=True)
            all_tf_results['30m'] = results_30m
            best = results_30m[0]
            print(f"\n  ★ Best 30m: {best[0]} Cal={best[2]['Cal']:.2f}")

    # ═══ Part 2: 앙상블 비교 ═══
    print(f"\n\n{'=' * 90}")
    print("Part 2: 앙상블 (4h1 → top3 교체, 5x)")
    print("=" * 90)

    print("\nGenerating base traces...")
    traces = {}
    for name, cfg in CURRENT_STRATEGIES.items():
        traces[name] = generate_trace(data, cfg)

    baseline_combo = {k: traces[k] for k in CURRENT_LIVE_COMBO}
    m_base = run_ensemble(data, baseline_combo, CURRENT_LIVE_COMBO, bars_1h, funding_1h, all_dates)

    print(f"\n  {'앙상블':<50s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stops':>5s}")
    print(f"  {'-'*80}")
    print(f"  {'baseline(4h1+4h2+1h1)':<50s} {m_base['CAGR']:>+8.1%} {m_base['MDD']:>+8.1%} {m_base['Cal']:>6.2f} {m_base['Liq']:>4d} {m_base.get('Stops',0):>5d}")

    for tf, results in all_tf_results.items():
        print(f"\n  --- {tf} top3 → 4h1 교체 ---")
        for label, cfg, single_m in results[:3]:
            try:
                t_trace = generate_trace(data, cfg)
                ens_traces = {"NEW": t_trace, "live_4h2": traces["live_4h2"], "live_1h1": traces["live_1h1"]}
                ens_weights = {"NEW": 1/3, "live_4h2": 1/3, "live_1h1": 1/3}
                m = run_ensemble(data, ens_traces, ens_weights, bars_1h, funding_1h, all_dates)
                print(f"  {label:<50s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}")
            except Exception as e:
                print(f"  {label:<50s} ERROR: {e}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
