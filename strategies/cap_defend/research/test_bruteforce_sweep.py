#!/usr/bin/env python3
"""4h/2h/1h 브루트 포스 스윕 — 병렬 처리.

모든 타임프레임에서 동일한 파라미터 그리드를 돌려
best 전략을 찾고, 앙상블 교체 테스트까지 수행.
"""

import os, sys, time, itertools
from multiprocessing import Pool, cpu_count

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

# ─── 변수 그리드 ───
GRID = {
    'sma_bars': [40, 60, 80, 120, 168, 240],
    'mom_short_bars': [5, 10, 20, 36, 48],
    'mom_long_bars': [15, 30, 60, 120, 240, 720],
    'health_mode': ['mom2vol', 'mom1vol'],
    'vol_threshold': [0.60, 0.80],
    'snap_interval_bars': [30],
}

FIXED = dict(
    vol_mode='bar',
    canary_hyst=0.015,
    drift_threshold=0.0,
    dd_threshold=0,
    dd_lookback=0,
    bl_drop=0,
    bl_days=0,
    n_snapshots=3,
)

TIMEFRAMES = ['4h', '2h', '1h']


def build_all_configs():
    keys = list(GRID.keys())
    values = [GRID[k] for k in keys]
    configs = []
    for combo in itertools.product(*values):
        cfg = dict(zip(keys, combo))
        cfg.update(FIXED)
        configs.append(cfg)
    return configs


def cfg_label(cfg):
    iv = cfg.get('_interval', '?')
    sma = cfg['sma_bars']
    ms = cfg['mom_short_bars']
    ml = cfg['mom_long_bars']
    hm = cfg['health_mode']
    vt = cfg['vol_threshold']
    return f"{iv}(S{sma},M{ms}/{ml},{hm},v{vt:.0%})"


# ─── Worker ───
# 데이터는 global (fork로 공유)
_DATA = {}


def _init_worker(intervals):
    global _DATA
    for iv in intervals:
        _DATA[iv] = load_data(iv)


def _run_one(args):
    interval, cfg = args
    try:
        bars, funding = _DATA[interval]
        m = run(bars, funding, interval=interval, leverage=1.0,
                start_date=START, end_date=END, **cfg)
        cfg_with_iv = dict(cfg, _interval=interval)
        return (interval, cfg_with_iv, m)
    except Exception as e:
        cfg_with_iv = dict(cfg, _interval=interval)
        return (interval, cfg_with_iv, {'CAGR': -999, 'MDD': -999, 'Cal': -999, 'error': str(e)})


def generate_trace_from_data(cfg):
    c = dict(cfg)
    interval = c.pop("_interval", c.pop("interval", "4h"))
    # Remove non-run keys
    for k in list(c.keys()):
        if k.startswith('_'):
            del c[k]
    bars, funding = _DATA[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=START, end_date=END, _trace=trace, **c)
    return trace


def main():
    t0 = time.time()
    configs = build_all_configs()
    n_configs = len(configs)
    n_workers = max(1, min(cpu_count() - 1, 24))

    print(f"브루트 포스 스윕: {n_configs}개 × {len(TIMEFRAMES)} TF = {n_configs * len(TIMEFRAMES)}개 조합")
    print(f"Workers: {n_workers}, 그리드:")
    for k, v in GRID.items():
        print(f"  {k}: {v}")
    print()

    # Build work items
    work = []
    for iv in TIMEFRAMES:
        for cfg in configs:
            work.append((iv, cfg))

    print(f"Loading data & starting pool...", flush=True)

    # Run parallel
    with Pool(n_workers, initializer=_init_worker, initargs=(TIMEFRAMES,)) as pool:
        results_by_tf = {iv: [] for iv in TIMEFRAMES}
        done = 0
        total = len(work)

        for result in pool.imap_unordered(_run_one, work, chunksize=4):
            iv, cfg, m = result
            if 'error' not in m:
                results_by_tf[iv].append((cfg_label(cfg), cfg, m))
            done += 1
            if done % 100 == 0:
                print(f"  {done}/{total} ({done/total:.0%}, {time.time()-t0:.0f}s)", flush=True)

    print(f"\n스윕 완료: {time.time()-t0:.0f}s\n")

    # ═══ Part 1: 타임프레임별 Top 10 ═══
    print("=" * 100)
    print("Part 1: 타임프레임별 Top 10 (단일 전략, 1x)")
    print("=" * 100)

    # 현행 전략 baseline
    print(f"\n  현행 전략:")
    # Load data in main process too for ensemble
    global _DATA
    for iv in TIMEFRAMES + ['1h']:
        if iv not in _DATA:
            _DATA[iv] = load_data(iv)

    for name, cfg in CURRENT_STRATEGIES.items():
        c = dict(cfg)
        interval = c.pop('interval')
        bars, funding = _DATA[interval]
        m = run(bars, funding, interval=interval, leverage=1.0,
                start_date=START, end_date=END, **c)
        print(f"    {name}({interval}): CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%} Cal={m['Cal']:.2f}")

    all_tf_best = {}
    for iv in TIMEFRAMES:
        results = results_by_tf[iv]
        results.sort(key=lambda x: x[2]['Cal'], reverse=True)
        all_tf_best[iv] = results

        print(f"\n  --- {iv} Top 10 ---")
        print(f"  {'#':>3s} {'설정':<45s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s}")
        print(f"  {'-'*75}")
        for i, (label, cfg, m) in enumerate(results[:10], 1):
            print(f"  {i:>3d} {label:<45s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f}")

    # ═══ Part 2: 앙상블 교체 테스트 ═══
    print(f"\n\n{'=' * 100}")
    print("Part 2: 앙상블 교체 (각 TF top3, 5x)")
    print("=" * 100)

    # Load data for ensemble
    bars_1h, funding_1h = _DATA["1h"]
    all_dates = bars_1h["BTC"].index[
        (bars_1h["BTC"].index >= START) & (bars_1h["BTC"].index <= END)
    ]

    # Generate base traces
    print("\n  Generating base traces...")
    traces = {}
    for name, cfg in CURRENT_STRATEGIES.items():
        c = dict(cfg)
        interval = c.pop("interval")
        bars, funding = _DATA[interval]
        trace = []
        run(bars, funding, interval=interval, leverage=1.0,
            start_date=START, end_date=END, _trace=trace, **c)
        traces[name] = trace

    # Baseline
    baseline_combo = {k: traces[k] for k in CURRENT_LIVE_COMBO}
    combined = combine_targets(baseline_combo, CURRENT_LIVE_COMBO, all_dates)
    engine = SingleAccountEngine(bars_1h, funding_1h, **ENGINE_KWARGS)
    m_base = engine.run(combined)

    print(f"\n  {'앙상블':<55s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stops':>5s}")
    print(f"  {'-'*85}")
    print(f"  {'baseline(4h1+4h2+1h1)':<55s} {m_base['CAGR']:>+8.1%} {m_base['MDD']:>+8.1%} {m_base['Cal']:>6.2f} {m_base['Liq']:>4d} {m_base.get('Stops',0):>5d}")

    # Test top3 of each TF replacing each position
    replace_targets = [
        ("4h1 교체", "live_4h1", ["live_4h2", "live_1h1"]),
        ("4h2 교체", "live_4h2", ["live_4h1", "live_1h1"]),
        ("1h1 교체", "live_1h1", ["live_4h1", "live_4h2"]),
    ]

    for tf in TIMEFRAMES:
        if not all_tf_best.get(tf):
            continue
        print(f"\n  === {tf} top3 ===")

        for replace_label, replace_key, keep_keys in replace_targets:
            print(f"\n  --- {tf} → {replace_label} ---")
            for label, cfg, single_m in all_tf_best[tf][:3]:
                try:
                    c = dict(cfg)
                    interval = c.pop('_interval')
                    for k in list(c.keys()):
                        if k.startswith('_'):
                            del c[k]
                    bars, funding = _DATA[interval]
                    trace = []
                    run(bars, funding, interval=interval, leverage=1.0,
                        start_date=START, end_date=END, _trace=trace, **c)

                    ens_traces = {"NEW": trace}
                    ens_weights = {"NEW": 1/3}
                    for kk in keep_keys:
                        ens_traces[kk] = traces[kk]
                        ens_weights[kk] = 1/3

                    combined = combine_targets(ens_traces, ens_weights, all_dates)
                    engine = SingleAccountEngine(bars_1h, funding_1h, **ENGINE_KWARGS)
                    m = engine.run(combined)
                    print(f"  {label:<55s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}")
                except Exception as e:
                    print(f"  {label:<55s} ERROR: {e}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
