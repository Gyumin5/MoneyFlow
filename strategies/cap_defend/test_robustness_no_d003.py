#!/usr/bin/env python3
"""d0.03 제외 앙상블 테스트 + robustness 검증.

Phase 1: d0.03 제외한 상위 전략으로 앙상블 조합 (5x)
Phase 2: 최상위 후보들의 서브기간 + 파라미터 섭동
"""

import os, sys, time, itertools, json
import numpy as np
from multiprocessing import Pool, cpu_count
from math import comb

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

# ═══ d0.03 제외 전략 풀 (unified sweep Phase 4 순위합 상위, d0.03 제거) ═══
# 원래 18개 중 d0.03 3개 제거 → 15개
STRATEGIES = {
    "4h_S240_M10_720_b60_sn60": dict(interval="4h", sma_bars=240, mom_short_bars=10, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60, snap_interval_bars=60, **FIXED_BASE),
    "2h_S240_M5_720_b60_sn120": dict(interval="2h", sma_bars=240, mom_short_bars=5, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60, snap_interval_bars=120, **FIXED_BASE),
    "4h_S240_M20_30_b60_sn60": dict(interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=30,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60, snap_interval_bars=60, **FIXED_BASE),
    "4h_S120_M20_720_b80_sn15": dict(interval="4h", sma_bars=120, mom_short_bars=20, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.80, snap_interval_bars=15, **FIXED_BASE),
    "2h_S240_M20_720_b60_sn120": dict(interval="2h", sma_bars=240, mom_short_bars=20, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60, snap_interval_bars=120, **FIXED_BASE),
    "4h_S240_M20_120_b60_sn21": dict(interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=120,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60, snap_interval_bars=21, **FIXED_BASE),
    "2h_S240_M5_720_b60_sn30": dict(interval="2h", sma_bars=240, mom_short_bars=5, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60, snap_interval_bars=30, **FIXED_BASE),
    "2h_S120_M20_720_b60_sn120": dict(interval="2h", sma_bars=120, mom_short_bars=20, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60, snap_interval_bars=120, **FIXED_BASE),
    "4h_S168_M20_120_b60_sn21": dict(interval="4h", sma_bars=168, mom_short_bars=20, mom_long_bars=120,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60, snap_interval_bars=21, **FIXED_BASE),
    "2h_S240_M5_720_b60_sn60": dict(interval="2h", sma_bars=240, mom_short_bars=5, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60, snap_interval_bars=60, **FIXED_BASE),
    "4h_S120_M20_120_b60_sn21": dict(interval="4h", sma_bars=120, mom_short_bars=20, mom_long_bars=120,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60, snap_interval_bars=21, **FIXED_BASE),
    "live_4h2": dict(interval="4h", sma_bars=120, mom_short_bars=20, mom_long_bars=120,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60, snap_interval_bars=21, **FIXED_BASE),
    "1h_S240_M48_720_b80_sn120": dict(interval="1h", sma_bars=240, mom_short_bars=48, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.80, snap_interval_bars=120, **FIXED_BASE),
    "4h_S120_M10_720_b60_sn60": dict(interval="4h", sma_bars=120, mom_short_bars=10, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60, snap_interval_bars=60, **FIXED_BASE),
    "1h_S120_M36_720_d003_sn120": dict(interval="1h", sma_bars=120, mom_short_bars=36, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="daily", vol_threshold=0.03, snap_interval_bars=120, **FIXED_BASE),
}

# 잠깐 — 1h_S120_M36_720_d003도 d0.03이다. 제거.
# d0.05도 포함하자 (원래 풀에서 d0.05 전략 추가)
# live_4h2와 4h_S120_M20_120_b60_sn21은 동일 — 중복 제거

# 재정의: d0.03 완전 제거 + d0.05 전략 추가
STRATEGIES_CLEAN = {}
for k, v in STRATEGIES.items():
    if v.get('vol_threshold') == 0.03 and v.get('vol_mode') == 'daily':
        continue  # d0.03 제거
    STRATEGIES_CLEAN[k] = v

# d0.05 전략 추가 (unified sweep에서 확인된 것들)
STRATEGIES_CLEAN["4h_S240_M20_720_d005_sn60"] = dict(
    interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=720,
    health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
    snap_interval_bars=60, **FIXED_BASE)
STRATEGIES_CLEAN["4h_S120_M20_720_d005_sn21"] = dict(
    interval="4h", sma_bars=120, mom_short_bars=20, mom_long_bars=720,
    health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
    snap_interval_bars=21, **FIXED_BASE)
STRATEGIES_CLEAN["2h_S60_M20_720_d005_sn120"] = dict(
    interval="2h", sma_bars=60, mom_short_bars=20, mom_long_bars=720,
    health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
    snap_interval_bars=120, **FIXED_BASE)

# baseline
BASELINE = {
    "live_1h1": dict(interval="1h", sma_bars=168, mom_short_bars=36, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.80, snap_interval_bars=27, **FIXED_BASE),
    "live_4h1": dict(interval="4h", sma_bars=240, mom_short_bars=10, mom_long_bars=30,
        health_mode="mom1vol", vol_mode="daily", vol_threshold=0.05, snap_interval_bars=120, **FIXED_BASE),
    "live_4h2_bl": dict(interval="4h", sma_bars=120, mom_short_bars=20, mom_long_bars=120,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60, snap_interval_bars=21, **FIXED_BASE),
}

# ═══ Global data ═══
_DATA = {}
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


def run_ensemble_from_cfgs(combo_cfgs, start_date=START, end_date=END):
    """cfg dict → 앙상블 5x 실행."""
    traces = {}
    for name, cfg in combo_cfgs.items():
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
    weights = {k: 1.0 / len(combo_cfgs) for k in combo_cfgs}
    combined = combine_targets(traces, weights, all_dates)
    engine = SingleAccountEngine(bars_1h, funding_1h, **ENGINE_KWARGS)
    return engine.run(combined)


def main():
    global _DATA, _TRACES, _ALL_DATES, _BARS_1H, _FUNDING_1H
    t0 = time.time()
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "robustness_no_d003_results.txt")
    outf = open(outpath, 'w')

    def log(msg=""):
        print(msg, flush=True)
        outf.write(msg + "\n")
        outf.flush()

    strategies = STRATEGIES_CLEAN
    log(f"d0.03 제외 앙상블 테스트")
    log(f"전략 수: {len(strategies)}개 (d0.03 제거 + d0.05 추가)")
    log(f"기간: {START} ~ {END}")
    for k in strategies:
        cfg = strategies[k]
        vm = cfg.get('vol_mode', 'bar')
        vt = cfg['vol_threshold']
        vstr = f"d{vt}" if vm == 'daily' else f"b{vt:.0%}"
        log(f"  {k}: {cfg['interval']}(S{cfg['sma_bars']},M{cfg['mom_short_bars']}/{cfg['mom_long_bars']},{vstr},sn{cfg['snap_interval_bars']})")
    log()

    # Load data
    log("데이터 로딩...")
    for iv in ['4h', '2h', '1h']:
        _DATA[iv] = load_data(iv)
    _BARS_1H, _FUNDING_1H = _DATA['1h']
    _ALL_DATES = _BARS_1H["BTC"].index[
        (_BARS_1H["BTC"].index >= START) & (_BARS_1H["BTC"].index <= END)
    ]
    log(f"  완료 ({time.time()-t0:.0f}s)")

    # ═══ Phase 1: 1x trace + 단일 5x 성능 ═══
    log(f"\n{'='*110}")
    log("Phase 1: 1x trace 생성 + 단일 5x 성능")
    log('='*110)

    single_results = {}
    for key, cfg in strategies.items():
        c = dict(cfg)
        iv = c.pop('interval')
        bars, funding = _DATA[iv]
        trace = []
        m_1x = run(bars, funding, interval=iv, leverage=1.0,
                    start_date=START, end_date=END, _trace=trace, **c)
        _TRACES[key] = trace

        # 5x single
        ens_traces = {key: trace}
        ens_weights = {key: 1.0}
        combined = combine_targets(ens_traces, ens_weights, _ALL_DATES)
        engine = SingleAccountEngine(_BARS_1H, _FUNDING_1H, **ENGINE_KWARGS)
        m_5x = engine.run(combined)
        single_results[key] = {'1x': m_1x, '5x': m_5x}

        log(f"  {key:<45s} 1x: Cal={m_1x['Cal']:>5.2f} CAGR={m_1x['CAGR']:>+7.1%} Sh={m_1x['Sharpe']:>5.2f}  |  5x: Cal={m_5x['Cal']:>5.2f} CAGR={m_5x['CAGR']:>+7.1%} MDD={m_5x['MDD']:>+7.1%} Liq={m_5x['Liq']}")

    log(f"\n  Phase 1 완료: {time.time()-t0:.0f}s")

    # ═══ Phase 2: 앙상블 조합 (5x) ═══
    log(f"\n{'='*110}")
    log("Phase 2: 앙상블 조합 전수 테스트 (5x, 2~5개)")
    log('='*110)

    # Baseline
    bl_traces = {}
    for name, cfg in BASELINE.items():
        c = dict(cfg)
        iv = c.pop('interval')
        bars, funding = _DATA[iv]
        trace = []
        run(bars, funding, interval=iv, leverage=1.0,
            start_date=START, end_date=END, _trace=trace, **c)
        bl_traces[name] = trace
    bl_weights = {k: 1.0/3 for k in BASELINE}
    bl_combined = combine_targets(bl_traces, bl_weights, _ALL_DATES)
    engine = SingleAccountEngine(_BARS_1H, _FUNDING_1H, **ENGINE_KWARGS)
    m_bl = engine.run(bl_combined)
    log(f"\n  baseline(live): Cal={m_bl['Cal']:.2f} CAGR={m_bl['CAGR']:+.1%} MDD={m_bl['MDD']:+.1%} Liq={m_bl['Liq']} Sharpe={m_bl['Sharpe']:.2f}")

    # 5x Cal 기준 상위 15개로 조합
    ranked = sorted(strategies.keys(), key=lambda k: single_results[k]['5x']['Cal'], reverse=True)
    combo_pool = ranked[:15]
    n_keys = len(combo_pool)

    log(f"\n  조합 풀: {n_keys}개 (5x Cal 상위)")
    for k in combo_pool:
        m5 = single_results[k]['5x']
        log(f"    {k}: 5x Cal={m5['Cal']:.2f}")

    total_combos = sum(comb(n_keys, k) for k in range(2, 6))
    log(f"  총 조합: {total_combos}")

    all_work = []
    for n_strats in range(2, 6):
        for combo in itertools.combinations(combo_pool, n_strats):
            all_work.append((combo, n_strats))

    n_workers = max(1, min(cpu_count() - 1, 24))
    log(f"  Workers: {n_workers}")

    combo_results = []
    done = 0
    total = len(all_work)

    with Pool(n_workers) as pool:
        for result in pool.imap_unordered(_run_combo, all_work, chunksize=4):
            combo_keys, n_strats, m = result
            if 'error' not in m:
                combo_results.append((n_strats, combo_keys, m))
            done += 1
            if done % 200 == 0 or done == total:
                log(f"  {done}/{total} ({done/total:.0%}, {time.time()-t0:.0f}s)")

    # Top 30 by Cal
    combo_results.sort(key=lambda x: x[2]['Cal'], reverse=True)
    log(f"\n  === Top 30 (Cal 순, d0.03 제외) ===")
    log(f"  {'#':>3s} {'N':>2s} {'조합':<90s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stop':>5s} {'Sharpe':>7s}")
    log(f"  {'-'*130}")
    for i, (ns, keys, m) in enumerate(combo_results[:30], 1):
        label = " + ".join(keys)
        marker = " ***" if m['Cal'] > m_bl['Cal'] else ""
        log(f"  {i:>3d} {ns:>2d} {label:<90s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d} {m['Sharpe']:>7.2f}{marker}")

    # Top 30 by CAGR
    combo_cagr = sorted(combo_results, key=lambda x: x[2]['CAGR'], reverse=True)
    log(f"\n  === Top 30 (CAGR 순, d0.03 제외) ===")
    log(f"  {'#':>3s} {'N':>2s} {'조합':<90s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stop':>5s} {'Sharpe':>7s}")
    log(f"  {'-'*130}")
    for i, (ns, keys, m) in enumerate(combo_cagr[:30], 1):
        label = " + ".join(keys)
        marker = " ***" if m['Cal'] > m_bl['Cal'] else ""
        log(f"  {i:>3d} {ns:>2d} {label:<90s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d} {m['Sharpe']:>7.2f}{marker}")

    beat_cal = sum(1 for r in combo_results if r[2]['Cal'] > m_bl['Cal'])
    log(f"\n  Baseline Cal {m_bl['Cal']:.2f} 초과: {beat_cal}/{len(combo_results)}")

    log(f"\n  Phase 2 완료: {time.time()-t0:.0f}s")

    # ═══ Phase 3: Top 3 후보의 robustness ═══
    log(f"\n{'='*110}")
    log("Phase 3: Top 3 후보 robustness (서브기간 + vol_threshold 섭동)")
    log('='*110)

    top3_combos = []
    for i, (ns, keys, m) in enumerate(combo_results[:3]):
        combo_cfgs = {k: strategies[k] for k in keys}
        top3_combos.append((f"후보{i+1}({ns}전략)", combo_cfgs, keys))

    # Add baseline
    top3_combos.append(("baseline", BASELINE, list(BASELINE.keys())))

    # 서브기간
    periods = [
        ("전체", START, END),
        ("전반(2020.10~2023.06)", START, "2023-06-30"),
        ("후반(2023.07~2026.03)", "2023-07-01", END),
    ]

    log(f"\n  --- 서브기간 ---")
    log(f"  {'조합':<25s} {'기간':<25s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'Liq':>4s} {'Stop':>5s} {'Sharpe':>7s}")
    log(f"  {'-'*100}")

    for combo_name, combo_cfgs, _ in top3_combos:
        for period_name, start, end in periods:
            try:
                m = run_ensemble_from_cfgs(combo_cfgs, start_date=start, end_date=end)
                log(f"  {combo_name:<25s} {period_name:<25s} {m['Cal']:>6.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Liq']:>4d} {m.get('Stops',0):>5d} {m['Sharpe']:>7.2f}")
            except Exception as e:
                log(f"  {combo_name:<25s} {period_name:<25s} ERROR: {e}")
        log()

    # vol_threshold 섭동 (top 1 후보만)
    log(f"\n  --- Top 1 후보 vol_threshold 섭동 ---")
    top1_name, top1_cfgs, top1_keys = top3_combos[0]
    m_base = run_ensemble_from_cfgs(top1_cfgs)
    log(f"  기준: Cal={m_base['Cal']:.2f} CAGR={m_base['CAGR']:+.1%} MDD={m_base['MDD']:+.1%} Sharpe={m_base['Sharpe']:.2f}")

    for strat_key in top1_keys:
        cfg = top1_cfgs[strat_key]
        base_vt = cfg['vol_threshold']
        vm = cfg.get('vol_mode', 'bar')

        if vm == 'bar':
            vt_values = [0.50, 0.55, 0.60, 0.65, 0.70, 0.80]
        else:
            vt_values = [0.03, 0.04, 0.05, 0.06, 0.08]

        log(f"\n  {strat_key} vol_threshold ({vm}):")
        for vt in vt_values:
            combo_copy = {k: dict(v) for k, v in top1_cfgs.items()}
            combo_copy[strat_key]['vol_threshold'] = vt
            m = run_ensemble_from_cfgs(combo_copy)
            marker = " ◀" if vt == base_vt else ""
            log(f"    {vt:>6}: Cal={m['Cal']:>5.2f} CAGR={m['CAGR']:>+7.1%} MDD={m['MDD']:>+7.1%} Liq={m['Liq']} Sharpe={m['Sharpe']:>5.2f}{marker}")

    log(f"\n  Phase 3 완료: {time.time()-t0:.0f}s")

    log(f"\n\n총 소요: {time.time()-t0:.1f}s")
    log(f"결과: {outpath}")
    outf.close()


if __name__ == "__main__":
    main()
