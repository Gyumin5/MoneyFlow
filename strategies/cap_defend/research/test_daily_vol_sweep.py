#!/usr/bin/env python3
"""Daily vol 추가 스윕 — mom1vol + daily vol 변형 브루트 포스.

기존 스윕(bar only)에서 빠진 vol_mode='daily' 조합을 탐색.
mom1vol에서는 mom_long이 dead code이므로 제거하여 효율화.

Phase 1: 1x 스윕 (mom1vol+daily, mom2vol+daily 모두)
Phase 2: 상관성 기반 선정 (기존 bar 결과와 합산)
Phase 3: 조합 테스트
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
    canary_hyst=0.015,
    drift_threshold=0.0,
    dd_threshold=0,
    dd_lookback=0,
    bl_drop=0,
    bl_days=0,
    n_snapshots=3,
)

TIMEFRAMES = ['4h', '2h', '1h']
TOP_N_CAL = 20
TOP_N_SELECT = 5
MAX_COMBO_SIZE = 4


def build_configs():
    """mom1vol(daily) + mom2vol(daily) + mom1vol(bar) 조합 생성.

    mom1vol: mom_long 제거 (dead code)
    daily vol: [0.03, 0.05, 0.08]
    bar vol: [0.60, 0.80] (기존 스윕에 없는 mom1vol+bar 추가 확인용)
    """
    configs = []

    sma_list = [60, 80, 120, 168, 240]
    mom_short_list = [5, 10, 20, 36, 48]
    snap_list = [21, 27, 30, 60, 120]

    # === mom1vol family ===
    # daily vol
    for sma in sma_list:
        for ms in mom_short_list:
            for vt in [0.03, 0.05, 0.08]:
                for snap in snap_list:
                    cfg = dict(
                        sma_bars=sma,
                        mom_short_bars=ms,
                        mom_long_bars=30,  # dead code for mom1vol
                        health_mode='mom1vol',
                        vol_mode='daily',
                        vol_threshold=vt,
                        snap_interval_bars=snap,
                        **FIXED,
                    )
                    configs.append(cfg)

    # bar vol (mom1vol + bar — 기존에서 일부 포함되었지만 snap 변형 추가)
    for sma in sma_list:
        for ms in mom_short_list:
            for vt in [0.60, 0.80]:
                for snap in snap_list:
                    cfg = dict(
                        sma_bars=sma,
                        mom_short_bars=ms,
                        mom_long_bars=30,
                        health_mode='mom1vol',
                        vol_mode='bar',
                        vol_threshold=vt,
                        snap_interval_bars=snap,
                        **FIXED,
                    )
                    configs.append(cfg)

    # === mom2vol + daily vol ===
    mom_long_list = [30, 60, 120, 240, 720]
    for sma in sma_list:
        for ms in mom_short_list:
            for ml in mom_long_list:
                for vt in [0.03, 0.05, 0.08]:
                    for snap in snap_list:
                        cfg = dict(
                            sma_bars=sma,
                            mom_short_bars=ms,
                            mom_long_bars=ml,
                            health_mode='mom2vol',
                            vol_mode='daily',
                            vol_threshold=vt,
                            snap_interval_bars=snap,
                            **FIXED,
                        )
                        configs.append(cfg)

    return configs


def cfg_label(cfg):
    iv = cfg.get('_interval', cfg.get('interval', '?'))
    sma = cfg['sma_bars']
    ms = cfg['mom_short_bars']
    ml = cfg['mom_long_bars']
    hm = cfg['health_mode']
    vm = cfg['vol_mode']
    vt = cfg['vol_threshold']
    sn = cfg['snap_interval_bars']
    vstr = f"d{vt}" if vm == 'daily' else f"b{vt:.0%}"
    return f"{iv}(S{sma},M{ms}/{ml},{hm},{vstr},sn{sn})"


def cfg_key(cfg):
    iv = cfg.get('_interval', cfg.get('interval', '?'))
    sma = cfg['sma_bars']
    ms = cfg['mom_short_bars']
    ml = cfg['mom_long_bars']
    hm = cfg['health_mode'][:4]
    vm = cfg['vol_mode'][0]
    vt = cfg['vol_threshold']
    sn = cfg['snap_interval_bars']
    return f"{iv}_S{sma}_M{ms}_{ml}_{hm}_{vm}{vt}_sn{sn}"


# Workers
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


def trace_to_returns(trace, all_dates):
    sorted_entries = sorted(trace, key=lambda e: e['date'])
    idx = 0
    cash_series = []
    last_cash = 1.0
    for date in all_dates:
        while idx < len(sorted_entries) and sorted_entries[idx]['date'] <= date:
            last_cash = sorted_entries[idx]['target'].get('CASH', 0.0)
            idx += 1
        cash_series.append(1.0 - last_cash)
    return np.diff(np.array(cash_series))


def greedy_select(candidates, all_dates, n_select=5):
    if len(candidates) <= n_select:
        return candidates

    return_series = []
    for label, cfg, m, trace in candidates:
        rs = trace_to_returns(trace, all_dates)
        return_series.append(rs)

    min_len = min(len(rs) for rs in return_series)
    return_series = [rs[:min_len] for rs in return_series]

    selected = [0]
    remaining = list(range(1, len(candidates)))

    for _ in range(n_select - 1):
        best_idx = None
        best_score = -999

        for idx in remaining:
            corrs = []
            for sel_idx in selected:
                corr = np.corrcoef(return_series[idx], return_series[sel_idx])[0, 1]
                if np.isnan(corr):
                    corr = 1.0
                corrs.append(abs(corr))
            avg_corr = np.mean(corrs)
            cal = candidates[idx][2]['Cal']
            score = cal - avg_corr * 2

            if score > best_score:
                best_score = score
                best_idx = idx

        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates[i] for i in selected]


def main():
    t0 = time.time()
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_vol_sweep_results.txt")
    outf = open(outpath, 'w')

    def log(msg=""):
        print(msg, flush=True)
        outf.write(msg + "\n")
        outf.flush()

    configs = build_configs()
    n_configs = len(configs)
    n_workers = max(1, min(cpu_count() - 1, 24))

    # Count breakdown
    n_m1v_daily = sum(1 for c in configs if c['health_mode'] == 'mom1vol' and c['vol_mode'] == 'daily')
    n_m1v_bar = sum(1 for c in configs if c['health_mode'] == 'mom1vol' and c['vol_mode'] == 'bar')
    n_m2v_daily = sum(1 for c in configs if c['health_mode'] == 'mom2vol' and c['vol_mode'] == 'daily')

    log(f"Daily Vol 스윕: {n_configs}/TF × {len(TIMEFRAMES)} TF = {n_configs * len(TIMEFRAMES)}개")
    log(f"  mom1vol+daily: {n_m1v_daily}")
    log(f"  mom1vol+bar: {n_m1v_bar}")
    log(f"  mom2vol+daily: {n_m2v_daily}")
    log(f"Workers: {n_workers}")
    log()

    # Phase 1
    log("=" * 110)
    log("Phase 1: 1x 스윕")
    log("=" * 110)

    work = []
    for iv in TIMEFRAMES:
        for cfg in configs:
            work.append((iv, cfg))

    log(f"Loading data...")

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

    for iv in TIMEFRAMES:
        results_by_tf[iv].sort(key=lambda x: x[2]['Cal'], reverse=True)

    for iv in TIMEFRAMES:
        log(f"\n  --- {iv} Top 15 ---")
        log(f"  {'#':>3s} {'설정':<65s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s}")
        log(f"  {'-'*95}")
        for i, (label, cfg, m) in enumerate(results_by_tf[iv][:15], 1):
            log(f"  {i:>3d} {label:<65s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f}")

    # Phase 2: Selection
    log(f"\n\n{'=' * 110}")
    log("Phase 2: 상관성 기반 Top 5 선정")
    log("=" * 110)

    global _DATA, _TRACES, _ALL_DATES, _BARS_1H, _FUNDING_1H
    for iv in TIMEFRAMES + ['1h']:
        if iv not in _DATA:
            _DATA[iv] = load_data(iv)

    _BARS_1H, _FUNDING_1H = _DATA["1h"]
    _ALL_DATES = _BARS_1H["BTC"].index[
        (_BARS_1H["BTC"].index >= START) & (_BARS_1H["BTC"].index <= END)
    ]

    selected_by_tf = {}
    for iv in TIMEFRAMES:
        top_cands = results_by_tf[iv][:TOP_N_CAL]
        cands_with_trace = []

        for label, cfg, m in top_cands:
            c = dict(cfg)
            interval = c.pop('_interval')
            for k in list(c.keys()):
                if k.startswith('_'):
                    del c[k]
            bars, funding = _DATA[interval]
            trace = []
            run(bars, funding, interval=interval, leverage=1.0,
                start_date=START, end_date=END, _trace=trace, **c)
            cands_with_trace.append((label, cfg, m, trace))

        # Dedup
        unique = []
        seen = set()
        for label, cfg, m, trace in cands_with_trace:
            hk = str([(e['date'].isoformat(), sorted(e['target'].items())) for e in trace[:100]])
            if hk not in seen:
                seen.add(hk)
                unique.append((label, cfg, m, trace))

        log(f"  {iv}: {len(top_cands)} → {len(unique)} unique")

        selected = greedy_select(unique, _ALL_DATES, TOP_N_SELECT)
        selected_by_tf[iv] = selected

        log(f"\n  --- {iv} 선정 ---")
        for i, (label, cfg, m, trace) in enumerate(selected, 1):
            log(f"  {i}. {label} Cal={m['Cal']:.2f}")
            key = cfg_key(cfg)
            _TRACES[key] = trace

    # Add live strategies
    for name, cfg in CURRENT_STRATEGIES.items():
        c = dict(cfg)
        interval = c.pop("interval")
        bars, funding = _DATA[interval]
        trace = []
        run(bars, funding, interval=interval, leverage=1.0,
            start_date=START, end_date=END, _trace=trace, **c)
        _TRACES[name] = trace

    # Phase 3: Combos
    log(f"\n\n{'=' * 110}")
    log(f"Phase 3: 2~{MAX_COMBO_SIZE}개 조합 테스트 (5x)")
    log("=" * 110)

    all_keys = list(_TRACES.keys())
    n_keys = len(all_keys)

    from math import comb
    total_combos = sum(comb(n_keys, k) for k in range(2, MAX_COMBO_SIZE + 1))
    log(f"\n  전략 수: {n_keys}, 조합 수: {total_combos}")
    for k in all_keys:
        log(f"    - {k}")

    # Baseline
    baseline_combo = {k: _TRACES[k] for k in CURRENT_LIVE_COMBO}
    combined_bl = combine_targets(baseline_combo, CURRENT_LIVE_COMBO, _ALL_DATES)
    engine = SingleAccountEngine(_BARS_1H, _FUNDING_1H, **ENGINE_KWARGS)
    m_bl = engine.run(combined_bl)
    log(f"\n  baseline 5x: CAGR={m_bl['CAGR']:+.1%} MDD={m_bl['MDD']:+.1%} Cal={m_bl['Cal']:.2f} Liq={m_bl['Liq']}")

    all_work = []
    for n_strats in range(2, MAX_COMBO_SIZE + 1):
        for combo_keys in itertools.combinations(all_keys, n_strats):
            all_work.append((combo_keys, n_strats))

    log(f"  Workers: {n_workers}, Starting...\n")

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

    # Results
    log(f"\n\n{'=' * 110}")
    log("결과")
    log("=" * 110)
    log(f"\n  baseline: Cal={m_bl['Cal']:.2f}")

    for n_strats in range(2, MAX_COMBO_SIZE + 1):
        subset = [r for r in combo_results if r[0] == n_strats]
        subset.sort(key=lambda x: x[3]['Cal'], reverse=True)
        log(f"\n  === {n_strats}개 Top 10 ===")
        log(f"  {'#':>3s} {'조합':<90s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s}")
        log(f"  {'-'*115}")
        for i, (ns, label, keys, m) in enumerate(subset[:10], 1):
            marker = " ***" if m['Cal'] > m_bl['Cal'] else ""
            log(f"  {i:>3d} {label:<90s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d}{marker}")

    combo_results.sort(key=lambda x: x[3]['Cal'], reverse=True)
    log(f"\n  === 전체 Top 20 ===")
    log(f"  {'#':>3s} {'N':>2s} {'조합':<90s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s}")
    log(f"  {'-'*120}")
    for i, (ns, label, keys, m) in enumerate(combo_results[:20], 1):
        marker = " ***" if m['Cal'] > m_bl['Cal'] else ""
        log(f"  {i:>3d} {ns:>2d} {label:<90s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d}{marker}")

    beat_cal = sum(1 for r in combo_results if r[3]['Cal'] > m_bl['Cal'])
    log(f"\n  Baseline 초과: {beat_cal}/{len(combo_results)}")

    # Save JSON
    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "daily_vol_sweep_results.json")
    json_results = {
        'baseline': {'Cal': round(m_bl['Cal'], 2)},
        'combos': [{'n': ns, 'keys': list(keys), 'Cal': round(m['Cal'], 2), 'CAGR': round(m['CAGR'], 4), 'MDD': round(m['MDD'], 4), 'Liq': m['Liq']}
                   for ns, label, keys, m in combo_results[:100]],
    }
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    log(f"\n총 소요: {time.time()-t0:.1f}s")
    outf.close()


if __name__ == "__main__":
    main()
