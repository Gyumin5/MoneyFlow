#!/usr/bin/env python3
"""확장 스윕 — Part 1: 바별 Top10 × 3x/4x/5x, Part 2: 전수 앙상블 조합.

브루트 포스 스윕(test_bruteforce_sweep.py) 결과를 기반으로:
1) 바별 Top10 단일 전략을 SingleAccountEngine 3x/4x/5x로 테스트
2) 배수별 순위합으로 바별 Top5 선정
3) Top5 × 3 TF = 15개 전략에서 2~5개 앙상블 조합 전수 테스트
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

GRID = {
    'sma_bars': [40, 60, 80, 120, 168, 240],
    'mom_short_bars': [5, 10, 20, 36, 48],
    'mom_long_bars': [15, 30, 60, 120, 240, 720],
    'health_mode': ['mom2vol', 'mom1vol'],
    'vol_threshold': [0.60, 0.80],
    'snap_interval_bars': [30],
}

TIMEFRAMES = ['4h', '2h', '1h']
TOP_N = 10  # 바별 top N for leverage test
TOP_N_COMBO = 5  # 바별 top N for combo test


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
    return f"{iv}(S{sma},M{ms}/{ml},{hm},v{vt:.0%})"


def cfg_short(cfg):
    iv = cfg.get('_interval', cfg.get('interval', '?'))
    sma = cfg['sma_bars']
    ms = cfg['mom_short_bars']
    ml = cfg['mom_long_bars']
    hm = cfg['health_mode'][:4]
    vt = int(cfg['vol_threshold'] * 100)
    return f"{iv}_S{sma}_M{ms}_{ml}_{hm}_v{vt}"


# ─── Worker (fork 공유) ───
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


def main():
    t0 = time.time()
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extended_sweep_results.txt")
    outf = open(outpath, 'w')

    def log(msg=""):
        print(msg, flush=True)
        outf.write(msg + "\n")
        outf.flush()

    configs = build_all_configs()
    n_configs = len(configs)
    n_workers = max(1, min(cpu_count() - 1, 24))

    log(f"확장 스윕: {n_configs}개 × {len(TIMEFRAMES)} TF = {n_configs * len(TIMEFRAMES)}개 조합")
    log(f"Workers: {n_workers}")
    log()

    # ═══ Phase 1: 브루트 포스 1x 스윕 ═══
    log("=" * 100)
    log("Phase 1: 브루트 포스 1x 스윕 (Cal 순위)")
    log("=" * 100)

    work = []
    for iv in TIMEFRAMES:
        for cfg in configs:
            work.append((iv, cfg))

    log(f"Loading data & starting pool...")

    with Pool(n_workers, initializer=_init_worker, initargs=(TIMEFRAMES,)) as pool:
        results_by_tf = {iv: [] for iv in TIMEFRAMES}
        done = 0
        total = len(work)

        for result in pool.imap_unordered(_run_one, work, chunksize=4):
            iv, cfg, m = result
            if 'error' not in m:
                results_by_tf[iv].append((cfg_label(cfg), cfg, m))
            done += 1
            if done % 200 == 0:
                log(f"  {done}/{total} ({done/total:.0%}, {time.time()-t0:.0f}s)")

    log(f"\n1x 스윕 완료: {time.time()-t0:.0f}s")

    # Sort by Cal
    for iv in TIMEFRAMES:
        results_by_tf[iv].sort(key=lambda x: x[2]['Cal'], reverse=True)

    # Print top 10 per TF
    for iv in TIMEFRAMES:
        log(f"\n  --- {iv} Top {TOP_N} (1x) ---")
        log(f"  {'#':>3s} {'설정':<50s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s}")
        log(f"  {'-'*80}")
        for i, (label, cfg, m) in enumerate(results_by_tf[iv][:TOP_N], 1):
            log(f"  {i:>3d} {label:<50s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f}")

    # ═══ Phase 2: Top10 × 3x/4x/5x (SingleAccountEngine) ═══
    log(f"\n\n{'=' * 100}")
    log("Phase 2: 바별 Top10 단일 전략 3x/4x/5x (SingleAccountEngine)")
    log("=" * 100)

    # Load data in main process
    global _DATA
    for iv in TIMEFRAMES + ['1h']:
        if iv not in _DATA:
            _DATA[iv] = load_data(iv)

    bars_1h, funding_1h = _DATA["1h"]
    all_dates = bars_1h["BTC"].index[
        (bars_1h["BTC"].index >= START) & (bars_1h["BTC"].index <= END)
    ]

    # 현행 전략 baseline
    log(f"\n  현행 전략 baseline:")
    traces_live = {}
    for name, cfg in CURRENT_STRATEGIES.items():
        c = dict(cfg)
        interval = c.pop("interval")
        bars, funding = _DATA[interval]
        trace = []
        run(bars, funding, interval=interval, leverage=1.0,
            start_date=START, end_date=END, _trace=trace, **c)
        traces_live[name] = trace

    baseline_combo = {k: traces_live[k] for k in CURRENT_LIVE_COMBO}
    combined = combine_targets(baseline_combo, CURRENT_LIVE_COMBO, all_dates)

    leverages_to_test = [3.0, 4.0, 5.0]
    log(f"\n  {'전략':<55s} {'Lev':>4s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stops':>5s}")
    log(f"  {'-'*90}")

    for lev in leverages_to_test:
        ek = dict(ENGINE_KWARGS)
        ek['leverage'] = lev
        ek['leverage_floor'] = min(lev, 3.0)
        ek['leverage_mid'] = min(lev, 4.0)
        ek['leverage_ceiling'] = lev
        engine = SingleAccountEngine(bars_1h, funding_1h, **ek)
        m = engine.run(combined)
        log(f"  {'baseline(4h1+4h2+1h1)':<55s} {lev:>4.0f}x {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}")

    # Test top N from each TF at 3x/4x/5x
    all_single_results = {}  # key: cfg_short -> {lev: metrics}
    all_traces = {}  # key: cfg_short -> trace

    for iv in TIMEFRAMES:
        log(f"\n  === {iv} Top {TOP_N} ===")
        log(f"  {'전략':<55s} {'Lev':>4s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stops':>5s}")
        log(f"  {'-'*90}")

        for i, (label, cfg, m_1x) in enumerate(results_by_tf[iv][:TOP_N]):
            c = dict(cfg)
            interval = c.pop('_interval')
            for k in list(c.keys()):
                if k.startswith('_'):
                    del c[k]

            # Generate trace
            bars, funding = _DATA[interval]
            trace = []
            run(bars, funding, interval=interval, leverage=1.0,
                start_date=START, end_date=END, _trace=trace, **c)

            skey = cfg_short(dict(cfg, interval=interval))
            all_traces[skey] = trace
            all_single_results[skey] = {'1x': m_1x, 'cfg': cfg, 'label': label, 'interval': interval}

            # Run through engine at different leverages
            single_combo = {"SINGLE": trace}
            single_weights = {"SINGLE": 1.0}
            combined_single = combine_targets(single_combo, single_weights, all_dates)

            for lev in leverages_to_test:
                ek = dict(ENGINE_KWARGS)
                ek['leverage'] = lev
                ek['leverage_floor'] = min(lev, 3.0)
                ek['leverage_mid'] = min(lev, 4.0)
                ek['leverage_ceiling'] = lev
                engine = SingleAccountEngine(bars_1h, funding_1h, **ek)
                m = engine.run(combined_single)
                all_single_results[skey][f'{int(lev)}x'] = m
                log(f"  {label:<55s} {lev:>4.0f}x {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}")

    # ═══ Phase 3: 배수별 순위합 → 바별 Top5 ═══
    log(f"\n\n{'=' * 100}")
    log("Phase 3: 배수별 순위합 → 바별 Top 5 선정")
    log("=" * 100)

    ranked_by_tf = {}
    for iv in TIMEFRAMES:
        candidates = []
        for label, cfg, m_1x in results_by_tf[iv][:TOP_N]:
            skey = cfg_short(dict(cfg, interval=iv))
            if skey not in all_single_results:
                continue
            sr = all_single_results[skey]
            candidates.append((skey, sr))

        # Rank by Cal at each leverage
        for lev_key in ['3x', '4x', '5x']:
            candidates.sort(key=lambda x: x[1].get(lev_key, {}).get('Cal', -999), reverse=True)
            for rank, (skey, sr) in enumerate(candidates):
                sr.setdefault('ranks', {})[lev_key] = rank + 1

        # Sum ranks
        for skey, sr in candidates:
            sr['rank_sum'] = sum(sr.get('ranks', {}).values())

        candidates.sort(key=lambda x: x[1]['rank_sum'])
        ranked_by_tf[iv] = candidates

        log(f"\n  --- {iv} 순위 ---")
        log(f"  {'#':>3s} {'전략':<50s} {'R3x':>4s} {'R4x':>4s} {'R5x':>4s} {'합':>4s} {'Cal3x':>7s} {'Cal4x':>7s} {'Cal5x':>7s}")
        log(f"  {'-'*90}")
        for i, (skey, sr) in enumerate(candidates, 1):
            r = sr.get('ranks', {})
            m3 = sr.get('3x', {})
            m4 = sr.get('4x', {})
            m5 = sr.get('5x', {})
            log(f"  {i:>3d} {sr['label']:<50s} {r.get('3x','-'):>4} {r.get('4x','-'):>4} {r.get('5x','-'):>4} {sr['rank_sum']:>4d} {m3.get('Cal',0):>7.2f} {m4.get('Cal',0):>7.2f} {m5.get('Cal',0):>7.2f}")

    # Select top 5 per TF
    selected = {}
    for iv in TIMEFRAMES:
        selected[iv] = ranked_by_tf[iv][:TOP_N_COMBO]

    all_selected_keys = []
    for iv in TIMEFRAMES:
        for skey, sr in selected[iv]:
            all_selected_keys.append(skey)
    n_selected = len(all_selected_keys)

    log(f"\n  선정 전략: {n_selected}개 ({TOP_N_COMBO}/TF × {len(TIMEFRAMES)} TF)")

    # ═══ Phase 4: 2~5개 앙상블 조합 전수 테스트 ═══
    log(f"\n\n{'=' * 100}")
    log("Phase 4: 2~5개 앙상블 조합 전수 테스트 (5x)")
    log("=" * 100)

    # Count combos
    from math import comb
    total_combos = sum(comb(n_selected, k) for k in range(2, min(6, n_selected + 1)))
    log(f"\n  전략 수: {n_selected}, 조합 수: {total_combos}")

    # Add live strategies for comparison
    for name in CURRENT_LIVE_COMBO:
        if name not in all_traces:
            all_traces[name] = traces_live[name]

    # Build all combo results
    combo_results = []
    done = 0

    for n_strats in range(2, min(6, n_selected + 1)):
        log(f"\n  --- {n_strats}개 조합 (C({n_selected},{n_strats}) = {comb(n_selected, n_strats)}) ---")
        combos = list(itertools.combinations(all_selected_keys, n_strats))

        for combo_keys in combos:
            try:
                weight = 1.0 / n_strats
                ens_traces = {k: all_traces[k] for k in combo_keys}
                ens_weights = {k: weight for k in combo_keys}
                combined = combine_targets(ens_traces, ens_weights, all_dates)

                engine = SingleAccountEngine(bars_1h, funding_1h, **ENGINE_KWARGS)
                m = engine.run(combined)
                combo_label = " + ".join(combo_keys)
                combo_results.append((n_strats, combo_label, combo_keys, m))
                done += 1

                if done % 100 == 0:
                    log(f"    {done}/{total_combos} ({done/total_combos:.0%}, {time.time()-t0:.0f}s)")
            except Exception as e:
                done += 1

    log(f"\n  조합 테스트 완료: {len(combo_results)}개, {time.time()-t0:.0f}s")

    # Sort by Cal and print top results per combo size
    log(f"\n\n{'=' * 100}")
    log("결과: 조합 크기별 Top 10")
    log("=" * 100)

    log(f"\n  baseline(4h1+4h2+1h1) 5x: CAGR=+221.1% MDD=-44.4% Cal=4.98 Liq=4")

    for n_strats in range(2, min(6, n_selected + 1)):
        subset = [r for r in combo_results if r[0] == n_strats]
        subset.sort(key=lambda x: x[3]['Cal'], reverse=True)

        log(f"\n  === {n_strats}개 조합 Top 10 ===")
        log(f"  {'#':>3s} {'조합':<80s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stops':>5s}")
        log(f"  {'-'*110}")
        for i, (ns, label, keys, m) in enumerate(subset[:10], 1):
            log(f"  {i:>3d} {label:<80s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}")

    # Overall top 10
    combo_results.sort(key=lambda x: x[3]['Cal'], reverse=True)
    log(f"\n  === 전체 Top 10 ===")
    log(f"  {'#':>3s} {'N':>2s} {'조합':<80s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stops':>5s}")
    log(f"  {'-'*120}")
    for i, (ns, label, keys, m) in enumerate(combo_results[:10], 1):
        log(f"  {i:>3d} {ns:>2d} {label:<80s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}")

    # Save detailed JSON
    json_results = {
        'single_results': {},
        'combo_results': [],
        'baseline': {'CAGR': 221.1, 'MDD': -44.4, 'Cal': 4.98, 'Liq': 4, 'Stops': 16},
    }
    for skey, sr in all_single_results.items():
        json_results['single_results'][skey] = {
            'label': sr['label'],
            'interval': sr.get('interval', '?'),
            '1x': {'CAGR': sr['1x']['CAGR'], 'MDD': sr['1x']['MDD'], 'Cal': sr['1x']['Cal']},
        }
        for lev_key in ['3x', '4x', '5x']:
            if lev_key in sr:
                json_results['single_results'][skey][lev_key] = {
                    'CAGR': sr[lev_key]['CAGR'], 'MDD': sr[lev_key]['MDD'],
                    'Cal': sr[lev_key]['Cal'], 'Liq': sr[lev_key]['Liq'],
                }

    for ns, label, keys, m in combo_results[:50]:
        json_results['combo_results'].append({
            'n_strats': ns, 'label': label, 'keys': list(keys),
            'CAGR': m['CAGR'], 'MDD': m['MDD'], 'Cal': m['Cal'],
            'Liq': m['Liq'], 'Stops': m.get('Stops', 0),
        })

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "extended_sweep_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2, default=str)

    log(f"\n총 소요: {time.time()-t0:.1f}s")
    log(f"결과 저장: {outpath}")
    log(f"JSON 저장: {json_path}")
    outf.close()


if __name__ == "__main__":
    main()
