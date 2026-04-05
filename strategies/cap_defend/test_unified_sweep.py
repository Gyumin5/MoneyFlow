#!/usr/bin/env python3
"""통합 스윕: full_sweep + daily_vol_sweep Top 후보 → 3x/4x/5x 순위합 → 앙상블 조합.

Phase 1: 두 sweep의 Top 15/TF에서 추출한 ~80개 고유 전략 1x trace 생성
Phase 2: TF별 상관성 기반 greedy로 10개 선정 (총 ~30개)
Phase 3: 30개 × 3x/4x/5x SingleAccountEngine 테스트
Phase 4: 배수별 Cal 순위합 → 최종 후보 선정
Phase 5: 2~5개 앙상블 조합 전수 테스트 (5x)
"""

import os, sys, time, itertools, json
import numpy as np
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

FIXED_BASE = dict(
    canary_hyst=0.015,
    drift_threshold=0.0,
    dd_threshold=0,
    dd_lookback=0,
    bl_drop=0,
    bl_days=0,
    n_snapshots=3,
)

TOP_N_SELECT = 10     # TF별 상관성 기반 선정 수 (Phase 2)
MAX_COMBO_SIZE = 5    # 최대 앙상블 조합 크기
COMBO_TOP_N = 18      # Phase 5에서 순위합 상위 N개만 조합 테스트


def build_candidates():
    """두 sweep의 Top 15/TF에서 추출한 고유 전략 + live 3개."""
    candidates = {}  # key → (interval, cfg_dict)

    # === full_sweep Top 15 (vol_mode='bar') ===
    bar_fixed = dict(vol_mode='bar', **FIXED_BASE)

    fs_4h = [
        dict(sma_bars=240, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=60),
        dict(sma_bars=240, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=120),
        dict(sma_bars=120, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=60),
        dict(sma_bars=120, mom_short_bars=20, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=15),
        dict(sma_bars=240, mom_short_bars=20, mom_long_bars=30, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=60),
        dict(sma_bars=240, mom_short_bars=20, mom_long_bars=120, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=21),
        dict(sma_bars=120, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=120),
        dict(sma_bars=120, mom_short_bars=20, mom_long_bars=240, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=21),
        dict(sma_bars=168, mom_short_bars=20, mom_long_bars=120, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=21),
        dict(sma_bars=120, mom_short_bars=20, mom_long_bars=120, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=21),
        dict(sma_bars=168, mom_short_bars=20, mom_long_bars=240, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=21),
        dict(sma_bars=240, mom_short_bars=10, mom_long_bars=60, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=60),
        dict(sma_bars=240, mom_short_bars=20, mom_long_bars=240, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=21),
        dict(sma_bars=240, mom_short_bars=36, mom_long_bars=15, health_mode='mom1vol', vol_threshold=0.60, snap_interval_bars=120),
        dict(sma_bars=240, mom_short_bars=36, mom_long_bars=30, health_mode='mom1vol', vol_threshold=0.60, snap_interval_bars=120),
    ]
    for cfg in fs_4h:
        cfg.update(bar_fixed)
        k = make_key('4h', cfg)
        candidates[k] = ('4h', cfg)

    fs_2h = [
        dict(sma_bars=240, mom_short_bars=20, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=120),
        dict(sma_bars=240, mom_short_bars=5, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=21),
        dict(sma_bars=240, mom_short_bars=5, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=120),
        dict(sma_bars=240, mom_short_bars=5, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=30),
        dict(sma_bars=120, mom_short_bars=20, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=120),
        dict(sma_bars=60, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=60),
        dict(sma_bars=240, mom_short_bars=5, mom_long_bars=60, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=21),
        dict(sma_bars=240, mom_short_bars=5, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=60),
        dict(sma_bars=60, mom_short_bars=48, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=30),
        dict(sma_bars=240, mom_short_bars=20, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=60),
        dict(sma_bars=40, mom_short_bars=20, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=120),
        dict(sma_bars=240, mom_short_bars=48, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=30),
        dict(sma_bars=60, mom_short_bars=48, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=27),
        dict(sma_bars=240, mom_short_bars=48, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=15),
        dict(sma_bars=60, mom_short_bars=20, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.60, snap_interval_bars=120),
    ]
    for cfg in fs_2h:
        cfg.update(bar_fixed)
        k = make_key('2h', cfg)
        candidates[k] = ('2h', cfg)

    fs_1h = [
        dict(sma_bars=120, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=120),
        dict(sma_bars=168, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=30),
        dict(sma_bars=168, mom_short_bars=48, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=30),
        dict(sma_bars=168, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=120),
        dict(sma_bars=240, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=120),
        dict(sma_bars=168, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=60),
        dict(sma_bars=168, mom_short_bars=48, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=27),
        dict(sma_bars=240, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=30),
        dict(sma_bars=168, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=60),
        dict(sma_bars=168, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=120),
        dict(sma_bars=120, mom_short_bars=48, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=120),
        dict(sma_bars=168, mom_short_bars=48, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=120),
        dict(sma_bars=240, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=60),
        dict(sma_bars=240, mom_short_bars=48, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=120),
        dict(sma_bars=168, mom_short_bars=48, mom_long_bars=720, health_mode='mom2vol', vol_threshold=0.80, snap_interval_bars=60),
    ]
    for cfg in fs_1h:
        cfg.update(bar_fixed)
        k = make_key('1h', cfg)
        candidates[k] = ('1h', cfg)

    # === daily_vol_sweep Top 15 (vol_mode='daily' + some bar) ===
    dv_4h = [
        dict(sma_bars=240, mom_short_bars=20, mom_long_bars=30, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=60),
        dict(sma_bars=240, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=240, mom_short_bars=36, mom_long_bars=30, health_mode='mom1vol', vol_mode='bar', vol_threshold=0.60, snap_interval_bars=120),
        dict(sma_bars=240, mom_short_bars=20, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05, snap_interval_bars=60),
        dict(sma_bars=120, mom_short_bars=20, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05, snap_interval_bars=21),
        dict(sma_bars=240, mom_short_bars=20, mom_long_bars=30, health_mode='mom1vol', vol_mode='bar', vol_threshold=0.60, snap_interval_bars=60),
        dict(sma_bars=240, mom_short_bars=10, mom_long_bars=30, health_mode='mom1vol', vol_mode='bar', vol_threshold=0.60, snap_interval_bars=60),
        dict(sma_bars=240, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=60),
        dict(sma_bars=120, mom_short_bars=20, mom_long_bars=240, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=21),
        dict(sma_bars=240, mom_short_bars=10, mom_long_bars=30, health_mode='mom1vol', vol_mode='bar', vol_threshold=0.60, snap_interval_bars=30),
        dict(sma_bars=120, mom_short_bars=20, mom_long_bars=120, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=21),
        dict(sma_bars=240, mom_short_bars=20, mom_long_bars=30, health_mode='mom1vol', vol_mode='bar', vol_threshold=0.60, snap_interval_bars=21),
        dict(sma_bars=120, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=120, mom_short_bars=20, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05, snap_interval_bars=30),
        dict(sma_bars=168, mom_short_bars=20, mom_long_bars=240, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=21),
    ]
    for cfg in dv_4h:
        c = dict(cfg)
        if 'vol_mode' not in c:
            c['vol_mode'] = 'daily'
        c.update(FIXED_BASE)
        k = make_key('4h', c)
        candidates[k] = ('4h', c)

    dv_2h = [
        dict(sma_bars=60, mom_short_bars=20, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=240, mom_short_bars=20, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=60, mom_short_bars=20, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=60),
        dict(sma_bars=60, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=60, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=60),
        dict(sma_bars=240, mom_short_bars=5, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=21),
        dict(sma_bars=80, mom_short_bars=20, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=240, mom_short_bars=5, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=60),
        dict(sma_bars=240, mom_short_bars=5, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=80, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=60, mom_short_bars=5, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=60, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=30),
        dict(sma_bars=240, mom_short_bars=5, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=30),
        dict(sma_bars=240, mom_short_bars=5, mom_long_bars=240, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=60),
        dict(sma_bars=240, mom_short_bars=20, mom_long_bars=240, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
    ]
    for cfg in dv_2h:
        c = dict(cfg)
        c.update(FIXED_BASE)
        k = make_key('2h', c)
        candidates[k] = ('2h', c)

    dv_1h = [
        dict(sma_bars=120, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=240, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=80, mom_short_bars=36, mom_long_bars=60, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=240, mom_short_bars=48, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=120, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=168, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=120, mom_short_bars=48, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=120, mom_short_bars=10, mom_long_bars=30, health_mode='mom1vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=60),
        dict(sma_bars=168, mom_short_bars=36, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=120, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=60),
        dict(sma_bars=168, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=60),
        dict(sma_bars=168, mom_short_bars=48, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=240, mom_short_bars=10, mom_long_bars=720, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=120, mom_short_bars=36, mom_long_bars=60, health_mode='mom2vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
        dict(sma_bars=120, mom_short_bars=36, mom_long_bars=30, health_mode='mom1vol', vol_mode='daily', vol_threshold=0.03, snap_interval_bars=120),
    ]
    for cfg in dv_1h:
        c = dict(cfg)
        c.update(FIXED_BASE)
        k = make_key('1h', c)
        candidates[k] = ('1h', c)

    # === Live strategies ===
    for name, cfg in CURRENT_STRATEGIES.items():
        c = dict(cfg)
        iv = c.pop('interval')
        k = f"live_{name}"
        candidates[k] = (iv, c)

    return candidates


def make_key(iv, cfg):
    sma = cfg['sma_bars']
    ms = cfg['mom_short_bars']
    ml = cfg['mom_long_bars']
    hm = cfg['health_mode'][:4]
    vm = cfg.get('vol_mode', 'bar')
    vt = cfg['vol_threshold']
    sn = cfg['snap_interval_bars']
    vstr = f"d{vt}" if vm == 'daily' else f"b{vt:.0%}"
    return f"{iv}_S{sma}_M{ms}_{ml}_{hm}_{vstr}_sn{sn}"


def cfg_label(iv, cfg):
    sma = cfg['sma_bars']
    ms = cfg['mom_short_bars']
    ml = cfg['mom_long_bars']
    hm = cfg['health_mode']
    vm = cfg.get('vol_mode', 'bar')
    vt = cfg['vol_threshold']
    sn = cfg['snap_interval_bars']
    vstr = f"d{vt}" if vm == 'daily' else f"b{vt:.0%}"
    return f"{iv}(S{sma},M{ms}/{ml},{hm},{vstr},sn{sn})"


# ─── Global data for fork ───
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


def trace_to_exposure(trace, all_dates):
    """trace → exposure 시계열 (상관성 계산용)."""
    sorted_entries = sorted(trace, key=lambda e: e['date'])
    idx = 0
    exposure = []
    last_cash = 1.0
    for date in all_dates:
        while idx < len(sorted_entries) and sorted_entries[idx]['date'] <= date:
            last_cash = sorted_entries[idx]['target'].get('CASH', 0.0)
            idx += 1
        exposure.append(1.0 - last_cash)
    arr = np.array(exposure)
    return np.diff(arr)


def greedy_select(candidates_list, all_dates, n_select=10):
    """Cal 상위 후보에서 상관성 최소화 그리디 선정.
    candidates_list: [(key, iv, cfg, metrics, trace), ...]
    """
    if len(candidates_list) <= n_select:
        return candidates_list

    returns = []
    for item in candidates_list:
        rs = trace_to_exposure(item[4], all_dates)
        returns.append(rs)

    min_len = min(len(rs) for rs in returns)
    returns = [rs[:min_len] for rs in returns]

    selected = [0]  # #1: highest Cal
    remaining = list(range(1, len(candidates_list)))

    for _ in range(n_select - 1):
        best_idx = None
        best_score = -999
        for idx in remaining:
            corrs = []
            for sel_idx in selected:
                corr = np.corrcoef(returns[idx], returns[sel_idx])[0, 1]
                if np.isnan(corr):
                    corr = 1.0
                corrs.append(abs(corr))
            avg_corr = np.mean(corrs)
            cal = candidates_list[idx][3]['Cal']
            score = cal - avg_corr * 2
            if score > best_score:
                best_score = score
                best_idx = idx
        if best_idx is not None:
            selected.append(best_idx)
            remaining.remove(best_idx)

    return [candidates_list[i] for i in selected]


def main():
    t0 = time.time()
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unified_sweep_results.txt")
    outf = open(outpath, 'w')

    def log(msg=""):
        print(msg, flush=True)
        outf.write(msg + "\n")
        outf.flush()

    global _TRACES, _ALL_DATES, _BARS_1H, _FUNDING_1H, _DATA

    # Build candidates
    candidates = build_candidates()
    by_tf = {'4h': [], '2h': [], '1h': []}
    for k, (iv, cfg) in candidates.items():
        by_tf[iv].append((k, cfg))

    log(f"통합 스윕: {len(candidates)}개 고유 전략")
    for iv in ['4h', '2h', '1h']:
        log(f"  {iv}: {len(by_tf[iv])}개")
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

    # ═══ Phase 1: 1x trace 생성 ═══
    log(f"\n{'='*110}")
    log("Phase 1: 1x trace 생성")
    log('='*110)

    all_results = {}  # key → (iv, cfg, metrics, trace)
    for iv in ['4h', '2h', '1h']:
        bars, funding = _DATA[iv]
        for key, cfg in by_tf[iv]:
            trace = []
            m = run(bars, funding, interval=iv, leverage=1.0,
                    start_date=START, end_date=END, _trace=trace, **cfg)
            all_results[key] = (iv, cfg, m, trace)
            if 'error' not in m:
                label = cfg_label(iv, cfg) if not key.startswith('live_') else key
                log(f"  {key}: Cal={m['Cal']:.2f} CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%}")

    log(f"\n  Phase 1 완료: {len(all_results)}개, {time.time()-t0:.0f}s")

    # ═══ Phase 2: TF별 상관성 기반 선정 ═══
    log(f"\n{'='*110}")
    log(f"Phase 2: TF별 상관성 기반 Top {TOP_N_SELECT} 선정")
    log('='*110)

    # Deduplicate by trace hash
    def trace_hash(trace):
        if len(trace) < 100:
            return hash(str(trace))
        sample = [str(trace[i]['target']) for i in range(0, min(100, len(trace)))]
        return hash(tuple(sample))

    selected_all = {}  # key → (iv, cfg, metrics, trace)

    for iv in ['4h', '2h', '1h']:
        tf_results = [(k, r) for k, r in all_results.items()
                      if r[0] == iv and 'error' not in r[2] and not k.startswith('live_')]
        tf_results.sort(key=lambda x: x[1][2]['Cal'], reverse=True)

        # Deduplicate
        seen_hashes = set()
        unique = []
        for k, (_, cfg, m, trace) in tf_results:
            h = trace_hash(trace)
            if h not in seen_hashes:
                seen_hashes.add(h)
                unique.append((k, iv, cfg, m, trace))

        log(f"\n  {iv}: {len(tf_results)} → {len(unique)} unique")

        # Greedy select
        selected = greedy_select(unique, _ALL_DATES, TOP_N_SELECT)

        log(f"\n  --- {iv} 선정 (상관성 최소화) ---")
        for i, (k, _, cfg, m, trace) in enumerate(selected, 1):
            label = cfg_label(iv, cfg)
            log(f"    {i:>2d}. {label:<65s} Cal={m['Cal']:.2f} CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%}")
            selected_all[k] = (iv, cfg, m, trace)

        # Correlation matrix
        if len(selected) > 1:
            returns = [trace_to_exposure(s[4], _ALL_DATES) for s in selected]
            min_len = min(len(rs) for rs in returns)
            returns = [rs[:min_len] for rs in returns]
            n = len(selected)
            log(f"\n  상관계수:")
            header = "      " + " ".join(f"#{i+1:>4d}" for i in range(n))
            log(header)
            for i in range(n):
                row = f"  #{i+1:>2d}  "
                for j in range(n):
                    corr = np.corrcoef(returns[i], returns[j])[0, 1]
                    row += f" {corr:5.2f}"
                log(row)

    # Add live strategies
    for k, (iv, cfg, m, trace) in all_results.items():
        if k.startswith('live_'):
            selected_all[k] = (iv, cfg, m, trace)
            log(f"\n  + {k}: Cal={m['Cal']:.2f} CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%}")

    log(f"\n  Phase 2 완료: {len(selected_all)}개 선정, {time.time()-t0:.0f}s")

    # ═══ Phase 3: 3x/4x/5x 엔진 테스트 ═══
    log(f"\n{'='*110}")
    log("Phase 3: 3x/4x/5x SingleAccountEngine 테스트")
    log('='*110)

    engine_results = {}  # key → {3: metrics, 4: metrics, 5: metrics}
    for key, (iv, cfg, m_1x, trace) in selected_all.items():
        engine_results[key] = {}
        for lev in [3, 4, 5]:
            ens_traces = {key: trace}
            ens_weights = {key: 1.0}
            combined = combine_targets(ens_traces, ens_weights, _ALL_DATES)
            eng_kwargs = dict(ENGINE_KWARGS)
            # 레버리지와 floor/mid/ceiling을 비례 스케일링
            scale = lev / 5.0
            eng_kwargs['leverage'] = float(lev)
            eng_kwargs['leverage_floor'] = 3.0 * scale
            eng_kwargs['leverage_mid'] = 4.0 * scale
            eng_kwargs['leverage_ceiling'] = float(lev)
            engine = SingleAccountEngine(_BARS_1H, _FUNDING_1H, **eng_kwargs)
            m = engine.run(combined)
            engine_results[key][lev] = m

        m3, m4, m5 = engine_results[key][3], engine_results[key][4], engine_results[key][5]
        label = cfg_label(iv, cfg) if not key.startswith('live_') else key
        log(f"  {label:<65s}")
        log(f"    3x: Cal={m3['Cal']:>5.2f} CAGR={m3['CAGR']:>+7.1%} MDD={m3['MDD']:>+7.1%} Liq={m3['Liq']}")
        log(f"    4x: Cal={m4['Cal']:>5.2f} CAGR={m4['CAGR']:>+7.1%} MDD={m4['MDD']:>+7.1%} Liq={m4['Liq']}")
        log(f"    5x: Cal={m5['Cal']:>5.2f} CAGR={m5['CAGR']:>+7.1%} MDD={m5['MDD']:>+7.1%} Liq={m5['Liq']}")

    log(f"\n  Phase 3 완료: {time.time()-t0:.0f}s")

    # ═══ Phase 4: 배수별 Cal 순위합 ═══
    log(f"\n{'='*110}")
    log("Phase 4: 배수별 Cal 순위합 → 최종 후보")
    log('='*110)

    # Rank by Cal for each leverage
    keys = list(engine_results.keys())
    ranks = {k: 0 for k in keys}
    for lev in [3, 4, 5]:
        sorted_keys = sorted(keys, key=lambda k: engine_results[k][lev]['Cal'], reverse=True)
        for rank, k in enumerate(sorted_keys, 1):
            ranks[k] += rank

    # Sort by rank sum (lower is better)
    ranked = sorted(keys, key=lambda k: ranks[k])

    log(f"\n  {'#':>3s} {'전략':<65s} {'순위합':>6s} {'3xCal':>6s} {'4xCal':>6s} {'5xCal':>6s} {'5xCAGR':>8s} {'5xMDD':>8s} {'5xLiq':>5s}")
    log(f"  {'-'*115}")
    for i, k in enumerate(ranked, 1):
        iv, cfg, m_1x, _ = selected_all[k]
        label = cfg_label(iv, cfg) if not k.startswith('live_') else k
        m3, m4, m5 = engine_results[k][3], engine_results[k][4], engine_results[k][5]
        log(f"  {i:>3d} {label:<65s} {ranks[k]:>6d} {m3['Cal']:>6.2f} {m4['Cal']:>6.2f} {m5['Cal']:>6.2f} {m5['CAGR']:>+8.1%} {m5['MDD']:>+8.1%} {m5['Liq']:>5d}")

    log(f"\n  Phase 4 완료: {time.time()-t0:.0f}s")

    # ═══ Phase 5: 앙상블 조합 테스트 ═══
    log(f"\n{'='*110}")
    log("Phase 5: 앙상블 조합 전수 테스트 (5x)")
    log('='*110)

    # Store traces in global for multiprocessing
    for k in selected_all:
        _TRACES[k] = selected_all[k][3]

    # Baseline — live keys use "live_live_xxx" format in selected_all
    live_key_map = {}
    for k in selected_all:
        if k.startswith('live_live_'):
            original = k.replace('live_live_', 'live_')
            live_key_map[original] = k
    baseline_combo_traces = {}
    for orig_k, weight in CURRENT_LIVE_COMBO.items():
        mapped = live_key_map.get(orig_k)
        if mapped and mapped in _TRACES:
            baseline_combo_traces[mapped] = _TRACES[mapped]
    if len(baseline_combo_traces) == len(CURRENT_LIVE_COMBO):
        baseline_weights = {k: 1.0/len(CURRENT_LIVE_COMBO) for k in baseline_combo_traces}
        combined_bl = combine_targets(baseline_combo_traces, baseline_weights, _ALL_DATES)
        engine = SingleAccountEngine(_BARS_1H, _FUNDING_1H, **ENGINE_KWARGS)
        m_bl = engine.run(combined_bl)
        log(f"\n  baseline(live_4h1+live_4h2+live_1h1) 5x: CAGR={m_bl['CAGR']:+.1%} MDD={m_bl['MDD']:+.1%} Cal={m_bl['Cal']:.2f} Liq={m_bl['Liq']} Stops={m_bl.get('Stops',0)}")
    else:
        m_bl = {'CAGR': 0, 'MDD': 0, 'Cal': 0, 'Liq': 0}
        log(f"\n  baseline: N/A (live traces missing: have {list(baseline_combo_traces.keys())})")

    # Phase 5에서는 순위합 상위 COMBO_TOP_N개만 사용
    combo_keys_pool = ranked[:COMBO_TOP_N]
    n_keys = len(combo_keys_pool)
    total_combos = sum(comb(n_keys, k) for k in range(2, min(MAX_COMBO_SIZE + 1, n_keys + 1)))

    log(f"\n  전략 수: {n_keys} (순위합 상위 {COMBO_TOP_N}개)")
    for k in combo_keys_pool:
        log(f"    - {k} (순위합 {ranks[k]})")
    for k in range(2, min(MAX_COMBO_SIZE + 1, n_keys + 1)):
        log(f"  {k}개 조합: C({n_keys},{k}) = {comb(n_keys, k)}")
    log(f"  총 조합: {total_combos}")

    all_work = []
    for n_strats in range(2, min(MAX_COMBO_SIZE + 1, n_keys + 1)):
        for combo in itertools.combinations(combo_keys_pool, n_strats):
            all_work.append((combo, n_strats))

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
            if done % 100 == 0 or done == total:
                log(f"  {done}/{total} ({done/total:.0%}, {time.time()-t0:.0f}s)")

    log(f"\n  완료: {len(combo_results)}개, {time.time()-t0:.0f}s")

    # Results
    log(f"\n\n{'='*110}")
    log("결과")
    log('='*110)
    log(f"\n  baseline: CAGR={m_bl['CAGR']:+.1%} MDD={m_bl['MDD']:+.1%} Cal={m_bl['Cal']:.2f} Liq={m_bl['Liq']}")

    for n_strats in range(2, min(MAX_COMBO_SIZE + 1, n_keys + 1)):
        subset = [r for r in combo_results if r[0] == n_strats]
        subset.sort(key=lambda x: x[3]['Cal'], reverse=True)

        log(f"\n  === {n_strats}개 조합 Top 20 (Cal 순) ===")
        log(f"  {'#':>3s} {'조합':<90s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stop':>5s}")
        log(f"  {'-'*120}")
        for i, (ns, label, keys, m) in enumerate(subset[:20], 1):
            marker = " ***" if m['Cal'] > m_bl['Cal'] else ""
            log(f"  {i:>3d} {label:<90s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}{marker}")

    # Overall top 30
    combo_results.sort(key=lambda x: x[3]['Cal'], reverse=True)
    log(f"\n  === 전체 Top 30 (Cal 순) ===")
    log(f"  {'#':>3s} {'N':>2s} {'조합':<90s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stop':>5s}")
    log(f"  {'-'*125}")
    for i, (ns, label, keys, m) in enumerate(combo_results[:30], 1):
        marker = " ***" if m['Cal'] > m_bl['Cal'] else ""
        log(f"  {i:>3d} {ns:>2d} {label:<90s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}{marker}")

    # Top 30 by CAGR
    combo_results_cagr = sorted(combo_results, key=lambda x: x[3]['CAGR'], reverse=True)
    log(f"\n  === 전체 Top 30 (CAGR 순) ===")
    log(f"  {'#':>3s} {'N':>2s} {'조합':<90s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Liq':>4s} {'Stop':>5s}")
    log(f"  {'-'*125}")
    for i, (ns, label, keys, m) in enumerate(combo_results_cagr[:30], 1):
        marker = " ***" if m['Cal'] > m_bl['Cal'] else ""
        log(f"  {i:>3d} {ns:>2d} {label:<90s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} {m['Liq']:>4d} {m.get('Stops',0):>5d}{marker}")

    # Baseline beat
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
        'selected_strategies': {k: {
            'interval': selected_all[k][0],
            'Cal_1x': round(selected_all[k][2]['Cal'], 2),
            'CAGR_1x': round(selected_all[k][2]['CAGR'], 4),
            'rank_sum': ranks.get(k, 999),
            'Cal_3x': round(engine_results[k][3]['Cal'], 2) if k in engine_results else None,
            'Cal_4x': round(engine_results[k][4]['Cal'], 2) if k in engine_results else None,
            'Cal_5x': round(engine_results[k][5]['Cal'], 2) if k in engine_results else None,
        } for k in selected_all},
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

    json_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unified_sweep_results.json")
    with open(json_path, 'w') as f:
        json.dump(json_results, f, indent=2)

    log(f"\n총 소요: {time.time()-t0:.1f}s")
    log(f"텍스트: {outpath}")
    log(f"JSON: {json_path}")
    outf.close()


if __name__ == "__main__":
    main()
