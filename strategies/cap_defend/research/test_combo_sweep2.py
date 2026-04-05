#!/usr/bin/env python3
"""Phase 4 최적화: Top 2/TF + live 3 = 9개 전략, 2~5개 조합 병렬 테스트.

15개는 4,928개 조합 → 16시간.
9개로 줄이면 372개 조합 → ~30분.
"""

import os, sys, time, itertools, json
from multiprocessing import Pool, cpu_count
from math import comb

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

# 5x Cal 기준 바별 Top 2 (Part 0 결과) + live 3개
SELECTED = {
    # 4h top 2 (5x Cal)
    "4h_S240_M10_m1v_v60": dict(interval="4h", sma_bars=240, mom_short_bars=10, mom_long_bars=60,
                                 health_mode="mom1vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    "4h_S240_M10_m2v_v60": dict(interval="4h", sma_bars=240, mom_short_bars=10, mom_long_bars=60,
                                 health_mode="mom2vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    # 2h top 2 (5x Cal)
    "2h_S240_M5_720_v60":  dict(interval="2h", sma_bars=240, mom_short_bars=5, mom_long_bars=720,
                                 health_mode="mom2vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    "2h_S60_M36_720_v60":  dict(interval="2h", sma_bars=60, mom_short_bars=36, mom_long_bars=720,
                                 health_mode="mom2vol", vol_threshold=0.60, snap_interval_bars=30, **FIXED),
    # 1h top 2 (5x Cal)
    "1h_S168_M48_720_v80": dict(interval="1h", sma_bars=168, mom_short_bars=48, mom_long_bars=720,
                                 health_mode="mom2vol", vol_threshold=0.80, snap_interval_bars=30, **FIXED),
    "1h_S168_M36_720_v80": dict(interval="1h", sma_bars=168, mom_short_bars=36, mom_long_bars=720,
                                 health_mode="mom2vol", vol_threshold=0.80, snap_interval_bars=30, **FIXED),
    # Live strategies
    "live_4h1": dict(CURRENT_STRATEGIES["live_4h1"]),
    "live_4h2": dict(CURRENT_STRATEGIES["live_4h2"]),
    "live_1h1": dict(CURRENT_STRATEGIES["live_1h1"]),
}

# Global data for fork
_TRACES = {}
_ALL_DATES = None
_BARS_1H = None
_FUNDING_1H = None


def _run_combo(args):
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

    global _TRACES, _ALL_DATES, _BARS_1H, _FUNDING_1H

    # Load data
    log("Loading data...")
    data = {}
    for iv in ['4h', '2h', '1h']:
        data[iv] = load_data(iv)

    _BARS_1H, _FUNDING_1H = data["1h"]
    _ALL_DATES = _BARS_1H["BTC"].index[
        (_BARS_1H["BTC"].index >= START) & (_BARS_1H["BTC"].index <= END)
    ]

    # Generate traces
    log(f"Generating traces for {len(SELECTED)} strategies...")
    for name, cfg in SELECTED.items():
        c = dict(cfg)
        interval = c.pop("interval")
        bars, funding = data[interval]
        trace = []
        run(bars, funding, interval=interval, leverage=1.0,
            start_date=START, end_date=END, _trace=trace, **c)
        _TRACES[name] = trace
        log(f"  {name}: {len(trace)} entries")

    # Baseline
    log(f"\n{'=' * 110}")
    log("Baseline 앙상블")
    log("=" * 110)
    baseline_combo = {k: _TRACES[k] for k in CURRENT_LIVE_COMBO}
    combined_bl = combine_targets(baseline_combo, CURRENT_LIVE_COMBO, _ALL_DATES)
    engine = SingleAccountEngine(_BARS_1H, _FUNDING_1H, **ENGINE_KWARGS)
    m_bl = engine.run(combined_bl)
    log(f"  baseline(live_4h1+live_4h2+live_1h1) 5x: CAGR={m_bl['CAGR']:+.1%} MDD={m_bl['MDD']:+.1%} Cal={m_bl['Cal']:.2f} Liq={m_bl['Liq']} Stops={m_bl.get('Stops',0)}")

    # Combo test
    log(f"\n{'=' * 110}")
    log("조합 전수 테스트 (5x)")
    log("=" * 110)

    all_keys = list(SELECTED.keys())
    n_keys = len(all_keys)
    total_combos = sum(comb(n_keys, k) for k in range(2, min(6, n_keys + 1)))

    log(f"\n  전략 수: {n_keys}")
    for k in all_keys:
        log(f"    - {k}")
    for k in range(2, min(6, n_keys + 1)):
        log(f"  {k}개 조합: C({n_keys},{k}) = {comb(n_keys, k)}")
    log(f"  총 조합: {total_combos}")

    all_work = []
    for n_strats in range(2, min(6, n_keys + 1)):
        for combo_keys in itertools.combinations(all_keys, n_strats):
            all_work.append((combo_keys, n_strats))

    n_workers = max(1, min(cpu_count() - 1, 24))
    log(f"  Workers: {n_workers}")
    log(f"  Starting...\n")

    combo_results = []
    done = 0
    total = len(all_work)

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(_run_combo, all_work, chunksize=4):
            combo_keys, n_strats, m = result
            if 'error' not in m:
                label = " + ".join(combo_keys)
                combo_results.append((n_strats, label, combo_keys, m))
            done += 1
            if done % 50 == 0:
                log(f"  {done}/{total} ({done/total:.0%}, {time.time()-t0:.0f}s)")

    log(f"\n  완료: {len(combo_results)}개, {time.time()-t0:.0f}s")

    # Results
    log(f"\n\n{'=' * 110}")
    log("결과")
    log("=" * 110)
    log(f"\n  baseline: CAGR={m_bl['CAGR']:+.1%} MDD={m_bl['MDD']:+.1%} Cal={m_bl['Cal']:.2f} Liq={m_bl['Liq']}")

    for n_strats in range(2, min(6, n_keys + 1)):
        subset = [r for r in combo_results if r[0] == n_strats]
        subset.sort(key=lambda x: x[3]['Cal'], reverse=True)

        log(f"\n  === {n_strats}개 조합 Top 15 (Cal 순) ===")
        log(f"  {'#':>3s} {'조합':<80s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stop':>5s}")
        log(f"  {'-'*110}")
        for i, (ns, label, keys, m) in enumerate(subset[:15], 1):
            marker = " ***" if m['Cal'] > m_bl['Cal'] else ""
            log(f"  {i:>3d} {label:<80s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}{marker}")

    # Overall top 20 by Cal
    combo_results.sort(key=lambda x: x[3]['Cal'], reverse=True)
    log(f"\n  === 전체 Top 20 (Cal 순) ===")
    log(f"  {'#':>3s} {'N':>2s} {'조합':<80s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stop':>5s}")
    log(f"  {'-'*115}")
    for i, (ns, label, keys, m) in enumerate(combo_results[:20], 1):
        marker = " ***" if m['Cal'] > m_bl['Cal'] else ""
        log(f"  {i:>3d} {ns:>2d} {label:<80s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}{marker}")

    # Top 20 by CAGR
    combo_results_cagr = sorted(combo_results, key=lambda x: x[3]['CAGR'], reverse=True)
    log(f"\n  === 전체 Top 20 (CAGR 순) ===")
    log(f"  {'#':>3s} {'N':>2s} {'조합':<80s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stop':>5s}")
    log(f"  {'-'*115}")
    for i, (ns, label, keys, m) in enumerate(combo_results_cagr[:20], 1):
        marker = " ***" if m['Cal'] > m_bl['Cal'] else ""
        log(f"  {i:>3d} {ns:>2d} {label:<80s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}{marker}")

    # Baseline beat count
    beat_cal = sum(1 for r in combo_results if r[3]['Cal'] > m_bl['Cal'])
    beat_cagr = sum(1 for r in combo_results if r[3]['CAGR'] > m_bl['CAGR'])
    log(f"\n  Baseline Cal {m_bl['Cal']:.2f} 초과: {beat_cal}/{len(combo_results)}")
    log(f"  Baseline CAGR {m_bl['CAGR']:+.1%} 초과: {beat_cagr}/{len(combo_results)}")

    # Save JSON
    json_results = {
        'baseline': {
            'CAGR': round(m_bl['CAGR'], 4), 'MDD': round(m_bl['MDD'], 4),
            'Cal': round(m_bl['Cal'], 2), 'Liq': m_bl['Liq'], 'Stops': m_bl.get('Stops', 0),
        },
        'strategies': list(SELECTED.keys()),
        'combos_by_cal': [],
        'combos_by_cagr': [],
        'beat_baseline_cal': beat_cal,
        'beat_baseline_cagr': beat_cagr,
        'total_combos': len(combo_results),
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
