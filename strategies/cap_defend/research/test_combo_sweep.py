#!/usr/bin/env python3
"""Phase 4: 앙상블 조합 전수 테스트 (병렬).

Phase 3 결과의 바별 Top5 = 15개 전략에서 2~5개 조합을 병렬 테스트.
4h mom1vol 중복 제거 → 실제 고유 전략만 사용.
"""

import os, sys, time, itertools, json
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

# Phase 3에서 선정된 바별 Top 5 (중복 제거)
# 4h: mom1vol은 mom_long이 dead code이므로 하나만 유지
SELECTED = {
    "4h_S240_M10_mom1v_v60": dict(interval="4h", sma_bars=240, mom_short_bars=10, mom_long_bars=60, health_mode="mom1vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    "4h_S240_M10_mom2v_v60": dict(interval="4h", sma_bars=240, mom_short_bars=10, mom_long_bars=60, health_mode="mom2vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    "4h_S240_M20_mom2v_v60": dict(interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=60, health_mode="mom2vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    "4h_S120_M20_mom2v_v60": dict(interval="4h", sma_bars=120, mom_short_bars=20, mom_long_bars=120, health_mode="mom2vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    "4h_S240_M20_240_mom2v": dict(interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=240, health_mode="mom2vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    "2h_S240_M5_720_v60":   dict(interval="2h", sma_bars=240, mom_short_bars=5, mom_long_bars=720, health_mode="mom2vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    "2h_S240_M5_240_v60":   dict(interval="2h", sma_bars=240, mom_short_bars=5, mom_long_bars=240, health_mode="mom2vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    "2h_S60_M36_720_v60":   dict(interval="2h", sma_bars=60, mom_short_bars=36, mom_long_bars=720, health_mode="mom2vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    "2h_S240_M48_720_v60":  dict(interval="2h", sma_bars=240, mom_short_bars=48, mom_long_bars=720, health_mode="mom2vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    "2h_S168_M36_720_v60":  dict(interval="2h", sma_bars=168, mom_short_bars=36, mom_long_bars=720, health_mode="mom2vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    "1h_S168_M48_720_v80":  dict(interval="1h", sma_bars=168, mom_short_bars=48, mom_long_bars=720, health_mode="mom2vol", vol_threshold=0.80, snap_interval_bars=30, **FIXED),
    "1h_S168_M36_720_v80":  dict(interval="1h", sma_bars=168, mom_short_bars=36, mom_long_bars=720, health_mode="mom2vol", vol_threshold=0.80, snap_interval_bars=30, **FIXED),
    "1h_S240_M48_720_v80":  dict(interval="1h", sma_bars=240, mom_short_bars=48, mom_long_bars=720, health_mode="mom2vol", vol_threshold=0.80, snap_interval_bars=30, **FIXED),
    "1h_S120_M36_720_v80":  dict(interval="1h", sma_bars=120, mom_short_bars=36, mom_long_bars=720, health_mode="mom2vol", vol_threshold=0.80, snap_interval_bars=30, **FIXED),
    "1h_S240_M36_720_v80":  dict(interval="1h", sma_bars=240, mom_short_bars=36, mom_long_bars=720, health_mode="mom2vol", vol_threshold=0.80, snap_interval_bars=30, **FIXED),
}

# 현행 전략도 추가 (비교용)
SELECTED["live_4h1"] = dict(CURRENT_STRATEGIES["live_4h1"])
SELECTED["live_4h2"] = dict(CURRENT_STRATEGIES["live_4h2"])
SELECTED["live_1h1"] = dict(CURRENT_STRATEGIES["live_1h1"])

# Global data for fork sharing
_DATA = {}
_TRACES = {}
_ALL_DATES = None
_BARS_1H = None
_FUNDING_1H = None


def _init_combo_worker():
    """Worker initializer — globals already set by fork."""
    pass


def _run_combo(args):
    """Run one ensemble combo through SingleAccountEngine."""
    combo_keys, n_strats = args
    try:
        weight = 1.0 / n_strats
        ens_traces = {k: _TRACES[k] for k in combo_keys}
        ens_weights = {k: weight for k in combo_keys}
        combined = combine_targets(ens_traces, ens_weights, _ALL_DATES)
        engine = SingleAccountEngine(_BARS_1H, _FUNDING_1H, **ENGINE_KWARGS)
        m = engine.run(combined)
        return (combo_keys, n_strats, m)
    except Exception as e:
        return (combo_keys, n_strats, {'CAGR': -999, 'MDD': -999, 'Cal': -999, 'Liq': 0, 'error': str(e)})


def main():
    t0 = time.time()
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combo_sweep_results.txt")
    outf = open(outpath, 'w')

    def log(msg=""):
        print(msg, flush=True)
        outf.write(msg + "\n")
        outf.flush()

    global _DATA, _TRACES, _ALL_DATES, _BARS_1H, _FUNDING_1H

    # Load data
    log("Loading data...")
    for iv in ['4h', '2h', '1h']:
        _DATA[iv] = load_data(iv)

    _BARS_1H, _FUNDING_1H = _DATA["1h"]
    _ALL_DATES = _BARS_1H["BTC"].index[
        (_BARS_1H["BTC"].index >= START) & (_BARS_1H["BTC"].index <= END)
    ]

    # Generate traces for all selected strategies
    log(f"\nGenerating traces for {len(SELECTED)} strategies...")
    for name, cfg in SELECTED.items():
        c = dict(cfg)
        interval = c.pop("interval")
        bars, funding = _DATA[interval]
        trace = []
        run(bars, funding, interval=interval, leverage=1.0,
            start_date=START, end_date=END, _trace=trace, **c)
        _TRACES[name] = trace
        log(f"  {name}: {len(trace)} trace entries")

    # ═══ Part 0: 단일 전략 1x/3x/4x/5x ═══
    log(f"\n{'=' * 100}")
    log("Part 0: 단일 전략 1x/3x/4x/5x (SingleAccountEngine)")
    log("=" * 100)

    single_results = {}
    log(f"\n  {'전략':<35s} {'1x Cal':>7s} {'3x CAGR':>8s} {'3x Cal':>7s} {'4x CAGR':>8s} {'4x Cal':>7s} {'5x CAGR':>8s} {'5x Cal':>7s} {'5x Liq':>6s}")
    log(f"  {'-'*100}")

    for name, cfg in SELECTED.items():
        c = dict(cfg)
        interval = c.pop("interval")
        bars, funding = _DATA[interval]
        m_1x = run(bars, funding, interval=interval, leverage=1.0,
                   start_date=START, end_date=END, **c)

        single_combo = {name: _TRACES[name]}
        single_weights = {name: 1.0}
        combined = combine_targets(single_combo, single_weights, _ALL_DATES)

        results = {'1x': m_1x}
        for lev in [3.0, 4.0, 5.0]:
            ek = dict(ENGINE_KWARGS)
            ek['leverage'] = lev
            ek['leverage_floor'] = min(lev, 3.0)
            ek['leverage_mid'] = min(lev, 4.0)
            ek['leverage_ceiling'] = lev
            engine = SingleAccountEngine(_BARS_1H, _FUNDING_1H, **ek)
            m = engine.run(combined)
            results[f'{int(lev)}x'] = m

        single_results[name] = results
        m1 = results['1x']
        m3 = results['3x']
        m4 = results['4x']
        m5 = results['5x']
        log(f"  {name:<35s} {m1['Cal']:>7.2f} {m3['CAGR']:>+8.1%} {m3['Cal']:>7.2f} {m4['CAGR']:>+8.1%} {m4['Cal']:>7.2f} {m5['CAGR']:>+8.1%} {m5['Cal']:>7.2f} {m5['Liq']:>6d}")

    # ═══ Baseline ═══
    log(f"\n  --- Baseline ---")
    baseline_combo = {k: _TRACES[k] for k in CURRENT_LIVE_COMBO}
    combined_bl = combine_targets(baseline_combo, CURRENT_LIVE_COMBO, _ALL_DATES)
    for lev in [3.0, 4.0, 5.0]:
        ek = dict(ENGINE_KWARGS)
        ek['leverage'] = lev
        ek['leverage_floor'] = min(lev, 3.0)
        ek['leverage_mid'] = min(lev, 4.0)
        ek['leverage_ceiling'] = lev
        engine = SingleAccountEngine(_BARS_1H, _FUNDING_1H, **ek)
        m = engine.run(combined_bl)
        log(f"  {'baseline(4h1+4h2+1h1)':<35s} {lev:.0f}x: CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%} Cal={m['Cal']:.2f} Liq={m['Liq']} Stops={m.get('Stops',0)}")

    # ═══ Part 1: 전수 조합 테스트 ═══
    log(f"\n\n{'=' * 100}")
    log("Part 1: 2~5개 앙상블 조합 전수 테스트 (5x)")
    log("=" * 100)

    # Use only the 15 new strategies (not live ones for combos to avoid redundancy)
    new_keys = [k for k in SELECTED.keys() if not k.startswith("live_")]
    n_new = len(new_keys)

    from math import comb
    total_combos = sum(comb(n_new, k) for k in range(2, min(6, n_new + 1)))
    log(f"\n  전략 수: {n_new}, 조합 수: {total_combos}")

    # Build all work items
    all_work = []
    for n_strats in range(2, min(6, n_new + 1)):
        for combo_keys in itertools.combinations(new_keys, n_strats):
            all_work.append((combo_keys, n_strats))

    n_workers = max(1, min(cpu_count() - 1, 24))
    log(f"  Workers: {n_workers}")
    log(f"  Starting parallel ensemble test...\n")

    combo_results = []
    done = 0
    total = len(all_work)

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(_run_combo, all_work, chunksize=8):
            combo_keys, n_strats, m = result
            if 'error' not in m:
                label = " + ".join(combo_keys)
                combo_results.append((n_strats, label, combo_keys, m))
            done += 1
            if done % 200 == 0:
                log(f"  {done}/{total} ({done/total:.0%}, {time.time()-t0:.0f}s)")

    log(f"\n  조합 테스트 완료: {len(combo_results)}개, {time.time()-t0:.0f}s")

    # ═══ Results ═══
    log(f"\n\n{'=' * 100}")
    log("결과 요약")
    log("=" * 100)
    log(f"\n  baseline(4h1+4h2+1h1) 5x: CAGR=+221.1% MDD=-44.4% Cal=4.98 Liq=4")

    for n_strats in range(2, min(6, n_new + 1)):
        subset = [r for r in combo_results if r[0] == n_strats]
        subset.sort(key=lambda x: x[3]['Cal'], reverse=True)

        log(f"\n  === {n_strats}개 조합 Top 15 ===")
        log(f"  {'#':>3s} {'조합':<90s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stops':>5s}")
        log(f"  {'-'*120}")
        for i, (ns, label, keys, m) in enumerate(subset[:15], 1):
            log(f"  {i:>3d} {label:<90s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}")

    # Overall top 20
    combo_results.sort(key=lambda x: x[3]['Cal'], reverse=True)
    log(f"\n  === 전체 Top 20 ===")
    log(f"  {'#':>3s} {'N':>2s} {'조합':<90s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stops':>5s}")
    log(f"  {'-'*125}")
    for i, (ns, label, keys, m) in enumerate(combo_results[:20], 1):
        log(f"  {i:>3d} {ns:>2d} {label:<90s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}")

    # Also show top by CAGR
    combo_results_cagr = sorted(combo_results, key=lambda x: x[3]['CAGR'], reverse=True)
    log(f"\n  === 전체 Top 20 (CAGR 순) ===")
    log(f"  {'#':>3s} {'N':>2s} {'조합':<90s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stops':>5s}")
    log(f"  {'-'*125}")
    for i, (ns, label, keys, m) in enumerate(combo_results_cagr[:20], 1):
        log(f"  {i:>3d} {ns:>2d} {label:<90s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}")

    # Save JSON
    json_results = {
        'single': {},
        'combos_by_cal': [],
        'combos_by_cagr': [],
    }
    for name, res in single_results.items():
        json_results['single'][name] = {}
        for lev_key in ['1x', '3x', '4x', '5x']:
            if lev_key in res:
                m = res[lev_key]
                json_results['single'][name][lev_key] = {
                    'CAGR': round(m['CAGR'], 4),
                    'MDD': round(m['MDD'], 4),
                    'Cal': round(m['Cal'], 2),
                    'Liq': m.get('Liq', 0),
                }

    for ns, label, keys, m in combo_results[:100]:
        json_results['combos_by_cal'].append({
            'n': ns, 'keys': list(keys),
            'CAGR': round(m['CAGR'], 4), 'MDD': round(m['MDD'], 4),
            'Cal': round(m['Cal'], 2), 'Liq': m['Liq'], 'Stops': m.get('Stops', 0),
        })

    for ns, label, keys, m in combo_results_cagr[:100]:
        json_results['combos_by_cagr'].append({
            'n': ns, 'keys': list(keys),
            'CAGR': round(m['CAGR'], 4), 'MDD': round(m['MDD'], 4),
            'Cal': round(m['Cal'], 2), 'Liq': m['Liq'], 'Stops': m.get('Stops', 0),
        })

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "combo_sweep_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    log(f"\n총 소요: {time.time()-t0:.1f}s")
    log(f"텍스트: {outpath}")
    log(f"JSON: {json_path}")
    outf.close()


if __name__ == "__main__":
    main()
