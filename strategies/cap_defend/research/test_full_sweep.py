#!/usr/bin/env python3
"""전체 브루트 포스 스윕 — snap 포함 + 상관성 기반 Top5 + 조합 테스트.

Phase 1: 4h/2h/1h × 4320 = 12,960개 1x 스윕 (병렬)
Phase 2: 바별 Top5 선정 (상관성 최소화 그리디)
Phase 3: Top5×3TF + live3 = 18개 → 2~4개 조합 앙상블 테스트 (병렬)
"""

import os, sys, time, itertools, json
import numpy as np
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

GRID = {
    'sma_bars': [40, 60, 80, 120, 168, 240],
    'mom_short_bars': [5, 10, 20, 36, 48],
    'mom_long_bars': [15, 30, 60, 120, 240, 720],
    'health_mode': ['mom2vol', 'mom1vol'],
    'vol_threshold': [0.60, 0.80],
    'snap_interval_bars': [15, 21, 27, 30, 60, 120],
}

TIMEFRAMES = ['4h', '2h', '1h']
TOP_N_CAL = 30       # Cal 상위 후보 풀
TOP_N_SELECT = 5     # 상관성 기반 최종 선정
MAX_COMBO_SIZE = 4   # 최대 조합 크기 (5개는 너무 느림)


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
    iv = cfg.get('_interval', cfg.get('interval', '?'))
    sma = cfg['sma_bars']
    ms = cfg['mom_short_bars']
    ml = cfg['mom_long_bars']
    hm = cfg['health_mode']
    vt = cfg['vol_threshold']
    sn = cfg['snap_interval_bars']
    return f"{iv}(S{sma},M{ms}/{ml},{hm},v{vt:.0%},sn{sn})"


def cfg_key(cfg):
    iv = cfg.get('_interval', cfg.get('interval', '?'))
    sma = cfg['sma_bars']
    ms = cfg['mom_short_bars']
    ml = cfg['mom_long_bars']
    hm = cfg['health_mode'][:4]
    vt = int(cfg['vol_threshold'] * 100)
    sn = cfg['snap_interval_bars']
    return f"{iv}_S{sma}_M{ms}_{ml}_{hm}_v{vt}_sn{sn}"


# ─── Workers ───
_DATA = {}
_TRACES = {}
_ALL_DATES = None
_BARS_1H = None
_FUNDING_1H = None


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


def trace_to_daily_returns(trace, all_dates):
    """trace의 CASH 비중 시계열 → 일간 변화율 (상관성 계산용)."""
    sorted_entries = sorted(trace, key=lambda e: e['date'])
    idx = 0
    cash_series = []
    last_cash = 1.0
    for date in all_dates:
        while idx < len(sorted_entries) and sorted_entries[idx]['date'] <= date:
            last_cash = sorted_entries[idx]['target'].get('CASH', 0.0)
            idx += 1
        cash_series.append(1.0 - last_cash)  # exposure = 1 - cash
    arr = np.array(cash_series)
    # daily changes in exposure → proxy for return correlation
    returns = np.diff(arr)
    return returns


def greedy_select_by_correlation(candidates, all_dates, n_select=5):
    """Cal 상위 후보에서 상관성 최소화 그리디 선정.

    candidates: list of (label, cfg, metrics, trace)
    """
    if len(candidates) <= n_select:
        return candidates

    # Compute return series for all candidates
    return_series = []
    for label, cfg, m, trace in candidates:
        rs = trace_to_daily_returns(trace, all_dates)
        return_series.append(rs)

    # Min length alignment
    min_len = min(len(rs) for rs in return_series)
    return_series = [rs[:min_len] for rs in return_series]

    selected = []
    remaining = list(range(len(candidates)))

    # Pick #1: highest Cal
    selected.append(remaining[0])  # already sorted by Cal
    remaining.remove(selected[0])

    for _ in range(n_select - 1):
        best_idx = None
        best_score = -999

        for idx in remaining:
            # Average correlation with all selected
            corrs = []
            for sel_idx in selected:
                corr = np.corrcoef(return_series[idx], return_series[sel_idx])[0, 1]
                if np.isnan(corr):
                    corr = 1.0
                corrs.append(abs(corr))
            avg_corr = np.mean(corrs)

            # Score: Cal - correlation penalty
            cal = candidates[idx][2]['Cal']
            # score = cal * (1 - avg_corr)  # Cal weighted by decorrelation
            score = cal - avg_corr * 2  # Cal with correlation penalty

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def main():
    t0 = time.time()
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_sweep_results.txt")
    outf = open(outpath, 'w')

    def log(msg=""):
        print(msg, flush=True)
        outf.write(msg + "\n")
        outf.flush()

    configs = build_all_configs()
    n_configs = len(configs)
    n_workers = max(1, min(cpu_count() - 1, 24))

    log(f"전체 브루트 포스 스윕: {n_configs}개 × {len(TIMEFRAMES)} TF = {n_configs * len(TIMEFRAMES)}개")
    log(f"Workers: {n_workers}")
    for k, v in GRID.items():
        log(f"  {k}: {v}")
    log()

    # ═══ Phase 1: 브루트 포스 1x ═══
    log("=" * 110)
    log("Phase 1: 브루트 포스 1x 스윕")
    log("=" * 110)

    work = []
    for iv in TIMEFRAMES:
        for cfg in configs:
            work.append((iv, cfg))

    log(f"Loading data & starting pool...")

    with Pool(n_workers, initializer=_init_worker, initargs=(TIMEFRAMES,)) as pool:
        results_by_tf = {iv: [] for iv in TIMEFRAMES}
        done = 0
        total = len(work)

        for result in pool.imap_unordered(_run_one, work, chunksize=8):
            iv, cfg, m = result
            if 'error' not in m:
                results_by_tf[iv].append((cfg_label(cfg), cfg, m))
            done += 1
            if done % 500 == 0:
                log(f"  {done}/{total} ({done/total:.0%}, {time.time()-t0:.0f}s)")

    log(f"\n1x 스윕 완료: {time.time()-t0:.0f}s")

    # Sort by Cal
    for iv in TIMEFRAMES:
        results_by_tf[iv].sort(key=lambda x: x[2]['Cal'], reverse=True)

    # Print top 15 per TF
    for iv in TIMEFRAMES:
        log(f"\n  --- {iv} Top 15 (1x, Cal 순) ---")
        log(f"  {'#':>3s} {'설정':<60s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s}")
        log(f"  {'-'*90}")
        for i, (label, cfg, m) in enumerate(results_by_tf[iv][:15], 1):
            log(f"  {i:>3d} {label:<60s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f}")

    # ═══ Phase 2: 상관성 기반 Top 5 선정 ═══
    log(f"\n\n{'=' * 110}")
    log("Phase 2: 상관성 최소화 그리디 선정 (바별 Top 5)")
    log("=" * 110)

    # Load data in main process
    global _DATA, _TRACES, _ALL_DATES, _BARS_1H, _FUNDING_1H
    for iv in TIMEFRAMES + ['1h']:
        if iv not in _DATA:
            _DATA[iv] = load_data(iv)

    _BARS_1H, _FUNDING_1H = _DATA["1h"]
    _ALL_DATES = _BARS_1H["BTC"].index[
        (_BARS_1H["BTC"].index >= START) & (_BARS_1H["BTC"].index <= END)
    ]

    # Generate traces for Cal top candidates
    log(f"\n  Generating traces for top {TOP_N_CAL} per TF...")
    selected_by_tf = {}

    for iv in TIMEFRAMES:
        top_candidates = results_by_tf[iv][:TOP_N_CAL]
        candidates_with_trace = []

        for label, cfg, m in top_candidates:
            c = dict(cfg)
            interval = c.pop('_interval')
            for k in list(c.keys()):
                if k.startswith('_'):
                    del c[k]
            bars, funding = _DATA[interval]
            trace = []
            run(bars, funding, interval=interval, leverage=1.0,
                start_date=START, end_date=END, _trace=trace, **c)
            candidates_with_trace.append((label, cfg, m, trace))

        # Deduplicate by checking if traces are identical (mom1vol dead code issue)
        unique = []
        seen_traces = set()
        for label, cfg, m, trace in candidates_with_trace:
            # Simple hash: first 100 target dicts
            hash_key = str([(e['date'].isoformat(), sorted(e['target'].items())) for e in trace[:100]])
            if hash_key not in seen_traces:
                seen_traces.add(hash_key)
                unique.append((label, cfg, m, trace))

        log(f"  {iv}: {len(top_candidates)} → {len(unique)} unique traces")

        # Greedy select by correlation
        selected = greedy_select_by_correlation(unique, _ALL_DATES, TOP_N_SELECT)
        selected_by_tf[iv] = selected

        log(f"\n  --- {iv} 선정 (상관성 최소화) ---")
        log(f"  {'#':>3s} {'설정':<60s} {'Cal':>6s}")
        log(f"  {'-'*75}")

        # Show correlation matrix
        for i, (label, cfg, m, trace) in enumerate(selected, 1):
            log(f"  {i:>3d} {label:<60s} {m['Cal']:>6.2f}")
            key = cfg_key(cfg)
            _TRACES[key] = trace

        # Show pairwise correlations
        if len(selected) >= 2:
            rs_list = [trace_to_daily_returns(t[3], _ALL_DATES) for t in selected]
            min_len = min(len(rs) for rs in rs_list)
            rs_list = [rs[:min_len] for rs in rs_list]
            log(f"\n  상관계수 매트릭스:")
            log(f"  {'':>5s} " + " ".join(f"#{j+1:>5d}" for j in range(len(selected))))
            for i in range(len(selected)):
                row = f"  #{i+1:>3d} "
                for j in range(len(selected)):
                    corr = np.corrcoef(rs_list[i], rs_list[j])[0, 1]
                    row += f" {corr:>5.2f}"
                log(row)

    # Add live strategies
    log(f"\n  Live strategies:")
    traces_live = {}
    for name, cfg in CURRENT_STRATEGIES.items():
        c = dict(cfg)
        interval = c.pop("interval")
        bars, funding = _DATA[interval]
        trace = []
        run(bars, funding, interval=interval, leverage=1.0,
            start_date=START, end_date=END, _trace=trace, **c)
        _TRACES[name] = trace
        traces_live[name] = trace
        log(f"  {name}")

    # ═══ Phase 3: 조합 테스트 ═══
    log(f"\n\n{'=' * 110}")
    log(f"Phase 3: 2~{MAX_COMBO_SIZE}개 앙상블 조합 전수 테스트 (5x)")
    log("=" * 110)

    # All strategy keys
    all_keys = list(_TRACES.keys())
    n_keys = len(all_keys)

    from math import comb
    total_combos = sum(comb(n_keys, k) for k in range(2, MAX_COMBO_SIZE + 1))

    log(f"\n  전략 수: {n_keys}")
    for k in all_keys:
        log(f"    - {k}")
    for k in range(2, MAX_COMBO_SIZE + 1):
        log(f"  {k}개 조합: C({n_keys},{k}) = {comb(n_keys, k)}")
    log(f"  총 조합: {total_combos}")

    # Baseline
    baseline_combo = {k: traces_live[k] for k in CURRENT_LIVE_COMBO}
    combined_bl = combine_targets(baseline_combo, CURRENT_LIVE_COMBO, _ALL_DATES)
    engine = SingleAccountEngine(_BARS_1H, _FUNDING_1H, **ENGINE_KWARGS)
    m_bl = engine.run(combined_bl)
    log(f"\n  baseline(live_4h1+live_4h2+live_1h1) 5x:")
    log(f"    CAGR={m_bl['CAGR']:+.1%} MDD={m_bl['MDD']:+.1%} Cal={m_bl['Cal']:.2f} Liq={m_bl['Liq']} Stops={m_bl.get('Stops',0)}")

    # Build work
    all_work = []
    for n_strats in range(2, MAX_COMBO_SIZE + 1):
        for combo_keys in itertools.combinations(all_keys, n_strats):
            all_work.append((combo_keys, n_strats))

    log(f"\n  Workers: {n_workers}")
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
            if done % 100 == 0:
                log(f"  {done}/{total} ({done/total:.0%}, {time.time()-t0:.0f}s)")

    log(f"\n  완료: {len(combo_results)}개, {time.time()-t0:.0f}s")

    # ═══ Results ═══
    log(f"\n\n{'=' * 110}")
    log("최종 결과")
    log("=" * 110)
    log(f"\n  baseline: CAGR={m_bl['CAGR']:+.1%} MDD={m_bl['MDD']:+.1%} Cal={m_bl['Cal']:.2f} Liq={m_bl['Liq']}")

    for n_strats in range(2, MAX_COMBO_SIZE + 1):
        subset = [r for r in combo_results if r[0] == n_strats]
        subset.sort(key=lambda x: x[3]['Cal'], reverse=True)

        log(f"\n  === {n_strats}개 조합 Top 15 (Cal 순) ===")
        log(f"  {'#':>3s} {'조합':<90s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stop':>5s}")
        log(f"  {'-'*120}")
        for i, (ns, label, keys, m) in enumerate(subset[:15], 1):
            marker = " ***" if m['Cal'] > m_bl['Cal'] else ""
            log(f"  {i:>3d} {label:<90s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}{marker}")

    # Overall top 20
    combo_results.sort(key=lambda x: x[3]['Cal'], reverse=True)
    log(f"\n  === 전체 Top 20 (Cal 순) ===")
    log(f"  {'#':>3s} {'N':>2s} {'조합':<90s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stop':>5s}")
    log(f"  {'-'*125}")
    for i, (ns, label, keys, m) in enumerate(combo_results[:20], 1):
        marker = " ***" if m['Cal'] > m_bl['Cal'] else ""
        log(f"  {i:>3d} {ns:>2d} {label:<90s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}{marker}")

    # Stats
    beat_cal = sum(1 for r in combo_results if r[3]['Cal'] > m_bl['Cal'])
    beat_cagr = sum(1 for r in combo_results if r[3]['CAGR'] > m_bl['CAGR'])
    log(f"\n  Baseline Cal {m_bl['Cal']:.2f} 초과: {beat_cal}/{len(combo_results)}")
    log(f"  Baseline CAGR {m_bl['CAGR']:+.1%} 초과: {beat_cagr}/{len(combo_results)}")

    # Save JSON
    json_results = {
        'grid': {k: [str(v) for v in vs] for k, vs in GRID.items()},
        'baseline': {
            'CAGR': round(m_bl['CAGR'], 4), 'MDD': round(m_bl['MDD'], 4),
            'Cal': round(m_bl['Cal'], 2), 'Liq': m_bl['Liq'],
        },
        'selected_strategies': {},
        'combos_by_cal': [],
    }
    for iv in TIMEFRAMES:
        json_results['selected_strategies'][iv] = []
        for label, cfg, m, trace in selected_by_tf[iv]:
            json_results['selected_strategies'][iv].append({
                'label': label, 'key': cfg_key(cfg),
                'Cal': round(m['Cal'], 2), 'CAGR': round(m['CAGR'], 4), 'MDD': round(m['MDD'], 4),
            })
    for ns, label, keys, m in combo_results[:100]:
        json_results['combos_by_cal'].append({
            'n': ns, 'keys': list(keys),
            'CAGR': round(m['CAGR'], 4), 'MDD': round(m['MDD'], 4),
            'Cal': round(m['Cal'], 2), 'Liq': m['Liq'], 'Stops': m.get('Stops', 0),
        })

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "full_sweep_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    log(f"\n총 소요: {time.time()-t0:.1f}s")
    log(f"텍스트: {outpath}")
    log(f"JSON: {json_path}")
    outf.close()


if __name__ == "__main__":
    main()
