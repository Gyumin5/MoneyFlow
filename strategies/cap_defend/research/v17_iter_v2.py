"""V17 Iterative grid refinement (v2 — 수정 엔진 기반).

엔진 수정 완료 (Codex APPROVE):
- stock_engine.precompute: sma/ema {50,100,150,200,250,300}
- stock_engine.select_offensive: rmom{P}_{N} family
- stock_engine.resolve_canary: 3asset_{n} LQD SMA n-map
- stock_engine.get_val: O(log n) searchsorted (17x 가속)
- stock_engine.load_prices: cache/data newer 자동 선택
- stock_engine_snap: phase_offset (timing jitter), _should_rebal_scheduled (fixed-schedule)

반복 refinement 로 단일 최선 전략 탐색.

동작 방식:
  Round 1: 모든 축 coarse grid, 6-phase jitter.
  Round N+1:
    - numeric 축 (snap_days, canary_sma, canary_hyst, def_mom): top-K 값 주변 확장
      (min*0.8 / max*1.2 outside + 인접쌍 기하평균 inside). unit snap (snap/sma integer,
      hyst 0.001).
    - categorical 축 (canary_asset, canary_type, canary_extra, select, health):
      top-K 내 등장 빈도 top N 만 유지 (Round 3+ 에서 축소).
  수렴 조건: Round peak Cal_p25 변화 ratio < CONVERGE_RATIO (1.10) AND
            top-K overlap >= CONVERGE_OVERLAP (0.70).

과적합 방어:
  - 모든 round 에서 Cal_p25 (6 phase jitter 하위 25%) 로 ranking
  - holdout (2024~2025) 은 이 루프 완전 밖. 최종 finalist 만 한 번 평가.
  - 각 round 저장 → 외부에서 plateau / robustness 검증 가능

Usage:
  python3 v17_iter_v2.py              # 전체 round 실행 (max 5 rounds)
  python3 v17_iter_v2.py --round 1    # round 1 만
"""
from __future__ import annotations
import os, sys, time, json, math
from itertools import product
from joblib import Parallel, delayed
from datetime import datetime

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

from stock_engine import SP, load_prices, precompute, _init, ALL_TICKERS
import stock_engine as tsi
from stock_engine_snap import run_snapshot_ensemble

OUT = os.path.join(HERE, "v17_iter_v2_out")
os.makedirs(OUT, exist_ok=True)

UNIVERSE_B = ('SPY', 'VEA', 'EEM', 'EWJ', 'INDA', 'GLD', 'PDBC')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')

PHASE_OFFSETS = [0, 5, 10, 15, 20, 25]
# Train/holdout split: 2024~2025 는 iter 루프 밖 (선택 편향 방지).
# 최종 Phase-5 에서 finalist 한 번만 holdout 평가.
START         = '2017-04-01'
TRAIN_END     = '2023-12-31'
HOLDOUT_START = '2024-01-01'
HOLDOUT_END   = '2025-12-31'
END           = TRAIN_END  # iter 루프는 train 구간만 본다
N_ANCHOR_REQUIRED = len(PHASE_OFFSETS)

# 엔진이 precompute 한 값 (stock_engine.precompute) — 이 외 값은 NaN 으로 탈락.
#   sma/ema : {50,100,150,200,250,300}
#   mom     : {21,42,63,126,252}
# canary_sma 는 이 discrete set 에서만 선택. def_mom 도 동일 제한.
# snap_days 는 임의 integer, hyst 는 임의 float (continuous).
SUPPORTED_CANARY_SMA = [50, 100, 150, 200, 250, 300]
SUPPORTED_DEF_MOM    = [21, 42, 63, 126, 252]

# Round 1 coarse grid (iter 씨드) — 모두 supported set 안.
INIT_GRID = {
    'snap_days':    [30, 60, 90],                       # integer, 임의
    'canary_sma':   [100, 150, 200, 250],               # supported 중 4개
    'canary_hyst':  [0.010, 0.020, 0.030],              # float, 임의
    'canary_type':  ['sma', 'ema'],
    'canary_asset': ['EEM', 'VEA', 'VT', 'SPY', 'ACWX'],
    'canary_extra': ['none', 'hyg_ief_50', '3asset_100'],
    'select':       ['sh3', 'mom3_sh3', 'rmom63_3', 'rmom126_3', 'rmom252_3',
                     'comp3', 'comp4', 'zscore3', 'comp_sort3'],
    'health':       ['none', 'sma100', 'sma200', 'mom63', 'mom21_63', 'mom63_vol'],
    'def_mom':      [63, 126, 252],                     # supported 중 3개
}

NUMERIC_AXES_CONT    = ['snap_days', 'canary_hyst']      # refine 자유
NUMERIC_AXES_DISCRETE = {                                # refine 은 supported 이웃만
    'canary_sma': SUPPORTED_CANARY_SMA,
    'def_mom':    SUPPORTED_DEF_MOM,
}
CATEG_AXES   = ['canary_type', 'canary_asset', 'canary_extra', 'select', 'health']

MAX_CONFIGS_PER_ROUND = 4000    # round 당 config 상한
TOP_K_EXPAND = 20               # round N+1 수확 top-K
MAX_ROUNDS = 5
CONVERGE_RATIO  = 1.10          # peak Cal_p25 <=prev*1.10 (상승 10% 미만) 이면 수렴
CONVERGE_OVERLAP = 0.70         # top-K 동일 70%+ 이면 수렴
MIN_CAT_SIZE = 2
MIN_NUM_VALUES = 3
REFINE_OUTSIDE_LO = 0.80        # 최소 peak 에 대한 outside 확장 계수
REFINE_OUTSIDE_HI = 1.20        # 최대 peak 에 대한 outside 확장 계수


# ─── Metrics ──────────────────────────────────────────────────────
def _metrics(df):
    if df is None or len(df) < 30: return None
    v = df['Value']
    y = (v.index[-1] - v.index[0]).days / 365.25
    if y <= 0: return None
    cagr = (v.iloc[-1] / v.iloc[0]) ** (1/y) - 1
    mdd  = (v / v.cummax() - 1).min()
    dr   = v.pct_change().dropna()
    sh   = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    cal  = cagr / abs(mdd) if mdd < 0 else 0
    return {'CAGR': cagr*100, 'MDD': mdd*100, 'Sharpe': sh, 'Cal': cal,
            'Final': v.iloc[-1]}


def _make_params(cfg):
    # 암묵 기본값에 의존하지 않도록 모든 축을 명시적으로 고정.
    return SP(
        offensive=UNIVERSE_B, defensive=DEF,
        canary_assets=(cfg['canary_asset'],),
        canary_sma=int(cfg['canary_sma']),
        canary_hyst=float(cfg['canary_hyst']),
        canary_type=cfg['canary_type'],
        canary_extra=cfg['canary_extra'],
        canary_band=0.0,               # snap engine 미지원 — 명시적 비활성
        select=cfg['select'], weight='ew',
        n_mom=3, n_sh=3, mom_style='default',   # mom3_sh3 family 의 기본 pin
        defense='top2',
        def_mom_period=int(cfg['def_mom']),
        health=cfg['health'],
        tx_cost=0.0025, crash='none', sharpe_lookback=252,
        start=START, end=END,
    )


def run_one(cfg):
    cals = []; cagrs = []; mdds = []; shs = []
    err = None
    try:
        p = _make_params(cfg)
    except Exception as e:
        return {**cfg, 'n_anchor': 0, 'ERR': f'params: {e}'}
    for phase in PHASE_OFFSETS:
        try:
            df = run_snapshot_ensemble(tsi._g_prices, tsi._g_ind, p,
                                        snap_days=int(cfg['snap_days']), n_snap=3,
                                        monthly_anchor_mode=False,
                                        phase_offset=phase)
            m = _metrics(df)
            if m is None:
                err = f'metric-none@phase={phase}'; break
            cals.append(m['Cal']); cagrs.append(m['CAGR'])
            mdds.append(m['MDD']); shs.append(m['Sharpe'])
        except Exception as e:
            err = f'{type(e).__name__}@phase={phase}: {str(e)[:60]}'
            break
    out = {**cfg, 'n_anchor': len(cals)}
    if len(cals) != N_ANCHOR_REQUIRED:
        out['ERR'] = err or f'partial({len(cals)}/{N_ANCHOR_REQUIRED})'
        return out
    arr = np.array(cals)
    out.update({
        'Cal_med': round(float(np.median(arr)), 3),
        'Cal_p25': round(float(np.percentile(arr, 25)), 3),
        'Cal_std': round(float(arr.std()), 3),
        'Cal_min': round(float(arr.min()), 3),
        'Cal_max': round(float(arr.max()), 3),
        'CAGR_med': round(float(np.median(cagrs)), 2),
        'MDD_med':  round(float(np.median(mdds)), 2),
        'Sh_med':   round(float(np.median(shs)), 3),
    })
    return out


# ─── Grid expansion logic ─────────────────────────────────────────
def _snap_value_cont(axis: str, v: float):
    """Continuous axes 단위 snap (제한 없음, step 만 정렬)."""
    if axis == 'snap_days':
        return max(21, int(round(v / 3) * 3))
    if axis == 'canary_hyst':
        return round(max(0.005, min(0.05, v)), 3)
    return v


def _nearest_supported(val, supported):
    """val 에 가장 가까운 supported 값."""
    return min(supported, key=lambda s: abs(s - val))


def _refine_numeric_cont(peak_values, prev_values) -> list:
    """Continuous numeric 축 local refinement.
    peak_values 주변 기하평균(inside) + 가장자리 작은 외곽 확장(±20%).
    prev grid 전체를 유지하지 않음 (monotonic 확산 방지)."""
    peaks = sorted(set(peak_values))
    if not peaks:
        return prev_values
    out = set(peaks)
    # inside: 인접 peak 쌍 사이 기하평균
    for a, b in zip(peaks[:-1], peaks[1:]):
        if a > 0 and b > 0 and b > a * 1.1:
            out.add(math.sqrt(a * b))
    # outside: 최소/최대 peak 에 대해서만 소폭 확장 (탐색공간 새나감 방지)
    if peaks:
        out.add(peaks[0] * REFINE_OUTSIDE_LO)
        out.add(peaks[-1] * REFINE_OUTSIDE_HI)
    return sorted(out)


def _refine_numeric_discrete(peak_values, supported, prev_values) -> list:
    """Discrete numeric 축 (canary_sma, def_mom) — supported set 중
    peak 이웃값만 유지. peak index 의 ±1 neighbor 만."""
    peaks = sorted(set(peak_values))
    if not peaks:
        return prev_values
    keep = set()
    for p in peaks:
        # p 자체 + supported 에서 ±1 neighbor
        nearest = _nearest_supported(p, supported)
        idx = supported.index(nearest)
        keep.add(nearest)
        if idx > 0: keep.add(supported[idx - 1])
        if idx < len(supported) - 1: keep.add(supported[idx + 1])
    return sorted(keep)


def _expand_grid(prev_grid: dict, top_rows: list[dict], round_idx: int) -> dict:
    """top-K rows 로 각 축 local refinement.
    Round 1→2: categorical 축 축소 보수적 (상위 2/3 유지)
    Round 3+: 10% threshold 로 더 공격적 축소.
    Numeric: 확산 방지 — fallback 은 peak ±1 step 정도만."""
    new_grid = {}
    # Continuous numeric — 확산 방지, 부족시에도 prev 전체 복원 안함
    for axis in NUMERIC_AXES_CONT:
        peaks = [row[axis] for row in top_rows if axis in row]
        vals = _refine_numeric_cont(peaks, prev_grid[axis])
        snapped = sorted(set(_snap_value_cont(axis, v) for v in vals))
        # fallback: peak 이 1개면 prev 에서 peak 인근 값 2개만 추가
        if len(snapped) < MIN_NUM_VALUES:
            ps = sorted(set(peaks))
            if ps:
                p0 = ps[0]
                neighbors = sorted(prev_grid[axis], key=lambda v: abs(v - p0))[:2]
                snapped = sorted(set(snapped + [_snap_value_cont(axis, v) for v in neighbors]))
        # 마지막 방어: 여전히 0개면 prev 전체
        if not snapped:
            snapped = sorted(prev_grid[axis])
        new_grid[axis] = snapped
    # Discrete numeric — supported set 이웃값
    for axis, supported in NUMERIC_AXES_DISCRETE.items():
        peaks = [row[axis] for row in top_rows if axis in row]
        new_grid[axis] = _refine_numeric_discrete(peaks, supported, prev_grid[axis])
    # Categorical: round 에 따라 threshold 조정
    # round 1→2: 지지율 >= 5% + 최소 3 유지 (보수적)
    # round 3+ : 지지율 >= 10% + 최소 2 유지 (공격적)
    if round_idx <= 1:
        threshold_pct = 0.05; min_keep = 3
    else:
        threshold_pct = 0.10; min_keep = MIN_CAT_SIZE
    for axis in CATEG_AXES:
        freq: dict = {}
        for row in top_rows:
            v = row.get(axis)
            if v is None: continue
            freq[v] = freq.get(v, 0) + 1
        if not freq:
            new_grid[axis] = prev_grid[axis]
            continue
        sorted_cats = sorted(freq, key=lambda k: freq[k], reverse=True)
        threshold = max(1, int(len(top_rows) * threshold_pct))
        keep = [c for c in sorted_cats if freq[c] >= threshold]
        # min_keep 이지만 prev 보다 크게 잡진 않는다.
        target_min = min(min_keep, len(prev_grid[axis]))
        if len(keep) < target_min:
            keep = sorted_cats[:target_min]
        new_grid[axis] = keep
    return new_grid


def _sample_configs(grid: dict, max_n: int, seed: int = 42) -> list[dict]:
    """full grid ≤ max_n → 전체 열거. 아니면 stratified sampling
    (각 categorical 값이 최소 1회 등장하도록 보장)."""
    keys = list(grid.keys())
    axes = [grid[k] for k in keys]
    total = 1
    for a in axes: total *= len(a)
    if total <= max_n:
        return [dict(zip(keys, c)) for c in product(*axes)]

    # Stratified: categorical 축별 전 값 × iid numeric 샘플링
    rng = np.random.default_rng(seed)
    cat_idx = [i for i, k in enumerate(keys) if k in CATEG_AXES]
    cat_values = [axes[i] for i in cat_idx]
    # 모든 categorical 조합 (이 조합 수가 max_n 넘으면 iid 로)
    cat_full = list(product(*cat_values))
    samples = set()
    # Step 1: 각 categorical 조합마다 최소 2회 반복
    base_n_per_cat = max(2, max_n // max(1, len(cat_full) * 2))
    for cat_combo in cat_full:
        for _ in range(base_n_per_cat):
            cfg = list(None for _ in keys)
            for ci, val in zip(cat_idx, cat_combo):
                cfg[ci] = val
            for j, ax in enumerate(axes):
                if cfg[j] is None:
                    cfg[j] = ax[rng.integers(len(ax))]
            samples.add(tuple(cfg))
            if len(samples) >= max_n: break
        if len(samples) >= max_n: break
    # Step 2: 남은 budget iid 로
    attempts = 0
    while len(samples) < max_n and attempts < max_n * 10:
        cfg = tuple(ax[rng.integers(len(ax))] for ax in axes)
        samples.add(cfg)
        attempts += 1
    return [dict(zip(keys, c)) for c in samples]


# ─── Round runner ─────────────────────────────────────────────────
def run_round(round_idx: int, grid: dict) -> pd.DataFrame:
    configs = _sample_configs(grid, MAX_CONFIGS_PER_ROUND)
    print(f"\n=== Round {round_idx} ===")
    for k, vs in grid.items():
        print(f"  {k:14} ({len(vs)}): {vs if len(vs) <= 12 else str(vs[:12])+'...'}")
    print(f"  configs: {len(configs)} × {len(PHASE_OFFSETS)} phase "
          f"= {len(configs)*len(PHASE_OFFSETS)} runs")

    t0 = time.time()
    rows = Parallel(n_jobs=24, backend='multiprocessing', verbose=1)(
        delayed(run_one)(c) for c in configs)
    print(f"  Done ({time.time()-t0:.0f}s)")

    df = pd.DataFrame(rows)
    df['round'] = round_idx
    return df


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--round', type=int, default=None,
                    help='Run only 1 round (must pair with --resume_tag if round>1)')
    ap.add_argument('--resume_tag', type=str, default=None,
                    help='Explicit run_tag to resume (no glob guessing). e.g. run_20260422_120000')
    ap.add_argument('--max_rounds', type=int, default=MAX_ROUNDS)
    args = ap.parse_args()

    print("Loading prices...")
    t0 = time.time()
    prices = load_prices(ALL_TICKERS, start='2014-01-01')
    ind = precompute(prices)
    _init(prices, ind)
    print(f"  ({time.time()-t0:.0f}s, {len(prices)} tickers)")

    # 계약: --round N (N>=2) → --resume_tag 필수. 새 lineage 는 --round 없이 시작.
    if args.round is not None and args.round >= 2 and not args.resume_tag:
        print("ERROR: --round >= 2 requires --resume_tag (prevent lineage drift)"); sys.exit(2)
    if args.resume_tag and (args.round is None or args.round <= 1):
        print("ERROR: --resume_tag requires --round >= 2"); sys.exit(2)
    if args.resume_tag:
        run_tag = args.resume_tag
    else:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        run_tag = f"run_{ts}"

    all_rounds: list = []
    # Resume: 1..(round-1) CSV 가 contiguous 하게 존재해야 하고,
    # round 이상의 prior CSV 가 있으면 error (stale/overwrite 방지).
    if args.resume_tag:
        import glob, re
        target_round = args.round  # guaranteed >=2 by validation
        # 1) 모든 round*.csv 수집 — exact filename 만 허용 (copy/backup 중복 방지).
        all_prior = sorted(glob.glob(os.path.join(OUT, f'{run_tag}_round*.csv')))
        prior_rounds = []
        exact_re = re.compile(re.escape(run_tag) + r'_round(\d+)\.csv$')
        for pth in all_prior:
            name = os.path.basename(pth)
            m = exact_re.match(name)
            if not m: continue
            prior_rounds.append((int(m.group(1)), pth))
        # 2) round >= target_round 가 있으면 error
        conflict = [p for n, p in prior_rounds if n >= target_round]
        if conflict:
            print(f"ERROR: stale/conflict CSV exists for round >= {target_round}:")
            for p in conflict: print(f"  {p}")
            print("  delete or use a fresh resume_tag.")
            sys.exit(2)
        # 3) 1..target_round-1 contiguous 검증
        have = sorted({n for n, _ in prior_rounds})
        expected = list(range(1, target_round))
        if have != expected:
            print(f"ERROR: non-contiguous prior CSVs: have={have}, expected={expected}")
            sys.exit(2)
        # 4) Load
        for n, pth in sorted(prior_rounds):
            all_rounds.append(pd.read_csv(pth))
        print(f"Loaded {len(all_rounds)} prior round CSV(s) for {run_tag} (rounds 1..{target_round-1})")
    prev_peak_p25 = 0.0
    prev_top_set: set = set()

    # Resume: --resume_tag + --round N (N>=2) 필수. 명시적 lineage 만 복원.
    if args.resume_tag and args.round and args.round > 1:
        state_path = os.path.join(OUT, f'{run_tag}_state_r{args.round}.json')
        if not os.path.exists(state_path):
            print(f"ERROR: missing state file {state_path}"); sys.exit(2)
        with open(state_path) as f:
            state = json.load(f)
        grid = state['grid']
        for k in list(NUMERIC_AXES_DISCRETE.keys()) + NUMERIC_AXES_CONT:
            if k in grid:
                grid[k] = [type(INIT_GRID[k][0])(v) for v in grid[k]]
        # state['prev_round'] 이 (args.round - 1) 이어야 정합 (belt-and-suspenders).
        saved_prev = state.get('prev_round')
        if saved_prev is not None and int(saved_prev) != args.round - 1:
            print(f"ERROR: state prev_round={saved_prev} != expected {args.round-1}. "
                  "Use fresh resume_tag or correct --round."); sys.exit(2)
        prev_peak_p25 = float(state.get('prev_peak_p25', 0.0))
        prev_top_set = set(tuple(t) for t in state.get('prev_top_set', []))
        print(f"Resume from {state_path}: prev_peak_p25={prev_peak_p25:.3f} "
              f"prev_top_set size={len(prev_top_set)}")
    else:
        grid = dict(INIT_GRID)

    ALL_NUMERIC = NUMERIC_AXES_CONT + list(NUMERIC_AXES_DISCRETE.keys())
    start_round = args.round or 1
    end_round   = args.round or args.max_rounds
    for r in range(start_round, end_round + 1):
        df = run_round(r, grid)
        path = os.path.join(OUT, f'{run_tag}_round{r}.csv')
        df.to_csv(path, index=False)
        all_rounds.append(df)

        ok = df[df['Cal_p25'].notna()] if 'Cal_p25' in df.columns else df.iloc[:0]
        if not len(ok):
            print(f"  Round {r}: no valid configs — abort.")
            break
        top = ok.sort_values('Cal_p25', ascending=False).head(TOP_K_EXPAND)
        peak_p25 = float(top['Cal_p25'].iloc[0])
        # Top-K config hash (tuple of all axes)
        axis_keys = ALL_NUMERIC + CATEG_AXES
        top_set = set(tuple(row[k] for k in axis_keys) for row in top.to_dict('records'))
        overlap = len(prev_top_set & top_set) / max(1, len(prev_top_set))
        print(f"\n  Round {r} peak Cal_p25 = {peak_p25:.3f} (prev {prev_peak_p25:.3f})"
              f" / top-{TOP_K_EXPAND} overlap vs prev = {overlap:.0%}")
        cols = ALL_NUMERIC + CATEG_AXES + ['Cal_med','Cal_p25','CAGR_med','MDD_med']
        print(top[cols].head(10).to_string(index=False))

        # 수렴 조건: peak 변화 < ratio AND top-K 70% 이상 overlap
        converged = (r > 1
                     and peak_p25 <= prev_peak_p25 * CONVERGE_RATIO
                     and overlap >= CONVERGE_OVERLAP)
        if converged:
            print(f"  Converged (ratio {peak_p25/max(prev_peak_p25, 1e-6):.3f}"
                  f", overlap {overlap:.0%}).")
            prev_peak_p25 = peak_p25; prev_top_set = top_set
            break
        prev_peak_p25 = peak_p25
        prev_top_set = top_set

        # --round N 의 경우에도 다음 round grid + state 를 항상 저장해 resume 가능.
        grid = _expand_grid(grid, top.to_dict('records'), round_idx=r)
        state = {
            'grid': {k: list(map(float, v)) if k in ALL_NUMERIC else list(v)
                     for k, v in grid.items()},
            'prev_peak_p25': prev_peak_p25,
            'prev_top_set':  [list(t) for t in prev_top_set],  # tuple → list for JSON
            'prev_round':    r,
        }
        state_path = os.path.join(OUT, f'{run_tag}_state_r{r+1}.json')
        with open(state_path, 'w') as f:
            json.dump(state, f, indent=2, default=str)

    # Final summary
    if all_rounds:
        combined = pd.concat(all_rounds, ignore_index=True)
        final_path = os.path.join(OUT, f'{run_tag}_all_rounds.csv')
        combined.to_csv(final_path, index=False)
        ok_all = combined[combined['Cal_p25'].notna()] if 'Cal_p25' in combined.columns else combined.iloc[:0]
        if len(ok_all):
            print("\n=== Overall Top 20 by Cal_p25 ===")
            axis_keys = NUMERIC_AXES_CONT + list(NUMERIC_AXES_DISCRETE.keys()) + CATEG_AXES
            cols = axis_keys + ['round','Cal_med','Cal_p25','Cal_std','CAGR_med','MDD_med','Sh_med']
            print(ok_all.sort_values('Cal_p25', ascending=False).head(20)[cols].to_string(index=False))
        print(f"\n저장: {final_path}")
        print(f"Train 구간: {START} ~ {TRAIN_END}  (holdout {HOLDOUT_START} ~ {HOLDOUT_END} 는 Phase-5 에서 평가)")


if __name__ == '__main__':
    main()
