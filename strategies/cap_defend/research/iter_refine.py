#!/usr/bin/env python3
"""Per-axis dynamic peak iterative refinement for cap_defend Phase-1.

흐름:
  Stage 1  — coarse global sweep (축당 5)
  Stage 2+ — per-axis peak detection (prominence>=0.15, PEAK_CAP=3)
             + opt-out drop (marg<max*0.3, peak/이웃 보호)
             + geom-mid 삽입 (b/a>=MIN_GAP_RATIO 인접쌍)
             + AXIS_CAP=7 하드캡 (peaks+geom-mid 우선, 나머지 marg 순)
             + 경계 1.5x push
  수렴     — grid change<5% AND top Cal Δ<1% 2회 연속 (AND)
  참고: 기존 soft/hard band 로직은 AXIS_CAP 하드캡으로 대체됨.

엔진은 unified_backtest.run() 직접 호출. multiprocessing Pool 24 worker.
출력: research/iter_refine/{stage_N}/raw.csv

CLI:
  python iter_refine.py --workers 24 --max-iters 5
"""
from __future__ import annotations
import argparse
import gc
import itertools
import json
import math
import os
import sys
import time
from collections import defaultdict
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
RES = HERE
CAP = os.path.dirname(RES)
REPO = os.path.dirname(CAP)
sys.path.insert(0, REPO)
sys.path.insert(0, CAP)

OUT_ROOT = os.path.join(HERE, "iter_refine")
FULL_END = "2026-04-13"
ANCHOR = "2020-10-01"

# ─────────────────────────────────────────────────────────────────────────
# 1. Coarse Stage-1 grid (≈10x 듬성)
# ─────────────────────────────────────────────────────────────────────────
COARSE = {
    "D": {
        "sma":  [20, 30, 50, 90],
        "ms":   [10, 20, 40, 60],
        "ml":   [60, 90, 180, 360],
        "vol":  [("daily", 0.05), ("daily", 0.07)],
        "snap": [24, 30, 60, 90],
    },
    "4h": {
        "sma":  [60, 120, 180, 240, 360, 480, 720],
        "ms":   [30, 60, 120, 240, 360],
        "ml":   [180, 360, 540, 720, 1080, 1440],
        "vol":  [("daily", 0.05), ("daily", 0.07), ("bar", 0.70)],
        "snap": [144, 180, 360, 540],
    },
}

ASSET_LEV = [
    ("spot", "D",  1.0),
    ("spot", "4h", 1.0),
    ("fut",  "D",  2.0), ("fut",  "D",  3.0), ("fut",  "D",  4.0),
    ("fut",  "4h", 2.0), ("fut",  "4h", 3.0), ("fut",  "4h", 4.0),
]

NUMERIC_AXES = ["sma", "ms", "ml", "snap"]
CAT_AXES = ["vol"]
ALL_AXES = NUMERIC_AXES + CAT_AXES

RAW_COLUMNS = [
    "tag", "asset", "iv", "lev",
    "sma", "ms", "ml", "vmode", "vthr", "snap",
    "Sh", "Cal", "CAGR", "MDD", "rebal", "liq", "error",
]

EXPAND_RATIO = 1.5
MIN_GAP_RATIO = 1.2   # 인접 비율 1.2x 이상이면 geom-mid 삽입
TOP_PCT = 0.20        # per-axis peak 검출용 상위 20%
TOP_Q_RANK = 0.25     # marginal 집계: 해당 축값 rows의 상위 25% 평균 (top-q mean)
OPT_OUT_RATIO = 0.30  # marginal < max*0.3 인 축끝값 opt-out drop (0.5→0.3 완화)
INTERIOR_DROP_RATIO = 0.30  # 내부값도 marginal < max*0.3 이면 drop (단, peak 및 peak 인접은 보호). 2-stage persistence는 stage_to_bucket_grids에서 적용
PEAK_PROMINENCE = 0.15      # peak 판정: (marg[p]-max(이웃))/max_marg ≥ 0.15 여야 진짜 peak (noise 흔들림 무시)
PEAK_CAP = 3                # 축당 peak 최대 3개 (marg 상위순)
AXIS_CAP = 7                # 축당 최종 값 개수 hard cap (peaks+new_vals 우선, 나머지는 marg 상위순)
VOL_DROP_RATIO = 0.30 # vol 카테고리 drop 조건: worst top-Q mean < max*0.3 일 때만
MIN_VOL = 2           # vol 최소 유지 카테고리 수

# ─────────────────────────────────────────────────────────────────────────
# 2. Engine wrapper (unified_backtest 호출)
# ─────────────────────────────────────────────────────────────────────────
_DATA = None  # iv → (bars, funding)


def preload():
    global _DATA
    from unified_backtest import load_data
    _DATA = {iv: load_data(iv) for iv in ["D", "4h"]}
    print(f"[preload] iv keys: {list(_DATA.keys())}", flush=True)


def cfg_id(cfg, asset, lev):
    iv = cfg["iv"].replace("D", "1D")
    vtag = f"{cfg['vmode'][0]}{cfg['vthr']:.2f}"
    return (f"{asset}_{iv}_S{cfg['sma']}_M{cfg['ms']}_{cfg['ml']}"
            f"_{vtag}_SN{cfg['snap']}_L{int(lev)}")


def _params(cfg):
    return {
        "sma_bars": cfg["sma"],
        "mom_short_bars": cfg["ms"],
        "mom_long_bars": cfg["ml"],
        "vol_mode": cfg["vmode"],
        "vol_threshold": cfg["vthr"],
        "snap_interval_bars": cfg["snap"],
        "n_snapshots": 3,
        "canary_hyst": 0.015,
        "universe_size": 5, "cap": 1 / 3,
        "health_mode": "mom2vol",
        "stop_kind": "none", "stop_pct": 0.0,
        "drift_threshold": 0.10, "post_flip_delay": 5,
        "dd_lookback": 60, "dd_threshold": -0.25,
        "bl_drop": -0.15, "bl_days": 7,
        "crash_threshold": -0.10,
    }


def _run_task(task):
    from unified_backtest import run as bt_run
    cfg, asset, lev = task["cfg"], task["asset"], task["lev"]
    iv = cfg["iv"]
    bars, funding = _DATA[iv]
    tx = 0.004 if asset == "spot" else 0.0004
    lev_use = 1.0 if asset == "spot" else lev
    try:
        m = bt_run(
            bars, funding, interval=iv,
            asset_type=asset, leverage=lev_use, tx_cost=tx,
            start_date=ANCHOR, end_date=FULL_END,
            **_params(cfg),
        )
        if not m:
            return {"tag": task["tag"], "asset": asset, "iv": iv, "lev": lev,
                    "sma": cfg["sma"], "ms": cfg["ms"], "ml": cfg["ml"],
                    "vmode": cfg["vmode"], "vthr": cfg["vthr"], "snap": cfg["snap"],
                    "error": "empty"}
        return {
            "tag": task["tag"], "asset": asset, "iv": iv, "lev": lev,
            "sma": cfg["sma"], "ms": cfg["ms"], "ml": cfg["ml"],
            "vmode": cfg["vmode"], "vthr": cfg["vthr"], "snap": cfg["snap"],
            "Sh": float(m.get("Sharpe", 0)), "Cal": float(m.get("Cal") or 0),
            "CAGR": float(m.get("CAGR", 0)), "MDD": float(m.get("MDD", 0)),
            "rebal": int(m.get("Rebal", 0)), "liq": int(m.get("Liq", 0) or 0),
        }
    except Exception as e:
        return {"tag": task["tag"], "asset": asset, "iv": iv, "lev": lev,
                "sma": cfg["sma"], "ms": cfg["ms"], "ml": cfg["ml"],
                "vmode": cfg["vmode"], "vthr": cfg["vthr"], "snap": cfg["snap"],
                "error": str(e)[:200]}


# ─────────────────────────────────────────────────────────────────────────
# 3. Grid utils
# ─────────────────────────────────────────────────────────────────────────
def enum_cfg(grid_iv, iv):
    for sma, ms, ml, (vm, vt), snap in itertools.product(
        grid_iv["sma"], grid_iv["ms"], grid_iv["ml"],
        grid_iv["vol"], grid_iv["snap"],
    ):
        if ml < 2 * ms:
            continue
        yield {"iv": iv, "sma": sma, "ms": ms, "ml": ml,
               "vmode": vm, "vthr": vt, "snap": snap}


def build_tasks(grids, bucket_filter: list | None = None) -> list:
    """grids: 두 형태 모두 지원
       - {iv: axes_grid}                    (Stage1 — iv 단위 공통)
       - {(asset, iv, lev): axes_grid}      (Stage2+ — 버킷별 multi-peak 유지)
    bucket_filter: ["asset_iv_lev", "asset_iv", "iv", "asset"] 부분키 매칭.
    """
    tasks = []
    def match(asset, iv, lev):
        if not bucket_filter:
            return True
        keys = [f"{asset}_{iv}_{int(lev)}",
                f"{asset}_{iv}", iv, asset]
        return any(k in bucket_filter for k in keys)

    # 형태 자동 판별
    is_per_bucket = grids and isinstance(next(iter(grids.keys())), tuple)
    seen_tags = set()

    for asset, iv, lev in ASSET_LEV:
        if not match(asset, iv, lev):
            continue
        if is_per_bucket:
            g_or_list = grids.get((asset, iv, lev))
            if not g_or_list:
                continue
            # bucket 값이 grid 단일이면 [grid], list면 그대로
            grid_list = g_or_list if isinstance(g_or_list, list) else [g_or_list]
        else:
            g = grids.get(iv)
            if not g:
                continue
            grid_list = [g]
        for g in grid_list:
            for cfg in enum_cfg(g, iv):
                tag = cfg_id(cfg, asset, lev)
                if tag in seen_tags:
                    continue
                seen_tags.add(tag)
                tasks.append({"tag": tag,
                              "cfg": cfg, "asset": asset, "lev": lev})
    return tasks


# ─────────────────────────────────────────────────────────────────────────
# 4. Per-axis dynamic peak detection
# ─────────────────────────────────────────────────────────────────────────
def _clean_rows(rows: list[dict]) -> list[dict]:
    def _is_no_error(e):
        if e is None or e == "":
            return True
        return isinstance(e, float) and e != e
    rows = [r for r in rows if _is_no_error(r.get("error"))]
    rows = [r for r in rows if isinstance(r.get("Cal"), (int, float))
            and not (r.get("Cal") != r.get("Cal"))]
    return rows


def _top_q_mean(vals: list[float], q: float = TOP_Q_RANK) -> float:
    if not vals:
        return 0.0
    vs = sorted(vals, reverse=True)
    k = max(1, int(len(vs) * q))
    return float(sum(vs[:k]) / k)


def _axis_marginals(top_rows: list[dict], axis: str) -> dict:
    """축 v 별 top-q mean Cal 집계."""
    by_v = defaultdict(list)
    for r in top_rows:
        key = r[axis] if axis != "vol" else (r["vmode"], float(r["vthr"]))
        by_v[key].append(float(r.get("Cal", 0) or 0))
    return {v: _top_q_mean(vals) for v, vals in by_v.items()}


def _detect_peaks(marg: dict) -> list:
    """정렬된 값 리스트에서 local peak (이웃 비교) 검출.
    v가 peak iff marg[v] >= 양쪽 이웃. 경계값은 한쪽만 비교."""
    if not marg:
        return []
    vs = sorted(marg.keys())
    if len(vs) == 1:
        return vs
    peaks = []
    for i, v in enumerate(vs):
        left_ok = (i == 0) or (marg[v] >= marg[vs[i - 1]])
        right_ok = (i == len(vs) - 1) or (marg[v] >= marg[vs[i + 1]])
        if left_ok and right_ok:
            peaks.append(v)
    return peaks


def _opt_out_drop(marg: dict) -> dict:
    """marginal < max * OPT_OUT_RATIO 인 축끝값만 drop (양끝에서부터 진행)."""
    if len(marg) <= 2:
        return dict(marg)
    vs = sorted(marg.keys())
    mx = max(marg.values())
    thr = mx * OPT_OUT_RATIO
    # 양쪽 끝에서 안쪽으로, 끝값이 threshold 미달이면 drop (단 중간 값은 유지)
    lo, hi = 0, len(vs) - 1
    while lo < hi and marg[vs[lo]] < thr:
        lo += 1
    while hi > lo and marg[vs[hi]] < thr:
        hi -= 1
    kept = vs[lo:hi + 1]
    return {v: marg[v] for v in kept}


# ─────────────────────────────────────────────────────────────────────────
# 5. Refine grid generator (per cluster)
# ─────────────────────────────────────────────────────────────────────────
def _geom_mid(a, b):
    m = int(round(math.sqrt(a * b)))
    return m if a < m < b else None


def _refine_axis_numeric(values: list[int], expand_left: bool,
                         expand_right: bool, expand: float = EXPAND_RATIO,
                         vs_orig: list[int] | None = None):
    """kept 값들에만 geom-mid 삽입. vs_orig 제공 시 원본에서 연속인 페어만 허용.
    → peak ±1 밖으로 geom-mid 확장 방지 (먼 peak 간 사이값 생성 차단)."""
    vs = sorted(set(int(v) for v in values))
    if not vs:
        return []
    out = list(vs)
    kept_set = set(vs)
    # interior fill: 원본 sorted 에서 연속한 페어이면서 둘 다 kept 인 경우만 geom-mid
    if vs_orig is not None:
        for i in range(len(vs_orig) - 1):
            a, b = vs_orig[i], vs_orig[i + 1]
            if a not in kept_set or b not in kept_set:
                continue
            if a <= 0 or b / a < MIN_GAP_RATIO:
                continue
            m = _geom_mid(a, b)
            if m is not None:
                out.append(m)
    else:
        for i in range(len(vs) - 1):
            a, b = vs[i], vs[i + 1]
            if a <= 0 or b / a < MIN_GAP_RATIO:
                continue
            m = _geom_mid(a, b)
            if m is not None:
                out.append(m)
    # boundary push
    if expand_left:
        new_lo = max(1, int(round(vs[0] / expand)))
        if new_lo < vs[0]:
            out.append(new_lo)
    if expand_right:
        new_hi = int(round(vs[-1] * expand))
        if new_hi > vs[-1]:
            out.append(new_hi)
    return sorted(set(out))


def _keep_band(vs_sorted: list, peaks: list, band: int) -> list:
    """peak 인덱스 주변 ±band 이웃 포함, 중복 제거."""
    idx_of = {v: i for i, v in enumerate(vs_sorted)}
    keep_idx = set()
    for p in peaks:
        if p not in idx_of:
            continue
        i = idx_of[p]
        for j in range(max(0, i - band), min(len(vs_sorted), i + band + 1)):
            keep_idx.add(j)
    return [vs_sorted[i] for i in sorted(keep_idx)]


def bucket_to_grid(top_rows: list[dict], iv: str, stage_idx: int,
                   prev_grid: dict | None = None) -> dict:
    """버킷 top rows → per-axis peak 검출 + opt-out drop + geom-mid + AXIS_CAP 하드캡.
    stage_idx, prev_grid 는 signature 호환용으로 유지 (현재 refinement 로직에서 직접 사용 안 함.
    boundary expand 는 이후 _refine_axis_numeric 에서 처리)."""
    if not top_rows:
        return {}
    grid = {}
    for axis in NUMERIC_AXES:
        marg = _axis_marginals(top_rows, axis)
        if not marg:
            return {}
        # opt-out: marginal < max*0.3 인 축끝 drop
        marg = _opt_out_drop(marg)
        vs_sorted = sorted(int(v) for v in marg.keys())
        peaks = _detect_peaks(marg)
        mx_marg = max(marg.values())
        # prominence filter: 이웃 대비 얼마나 두드러지는지. 약한 peak(noise)는 제거.
        if peaks and len(vs_sorted) > 1:
            filtered = []
            for p in peaks:
                p_int = int(p)
                if p_int not in marg:
                    continue
                idx = vs_sorted.index(p_int)
                neigh = []
                if idx > 0: neigh.append(marg[vs_sorted[idx - 1]])
                if idx + 1 < len(vs_sorted): neigh.append(marg[vs_sorted[idx + 1]])
                nb = max(neigh) if neigh else -1e18
                prom = (marg[p_int] - nb) / mx_marg if mx_marg > 0 else 0
                if prom >= PEAK_PROMINENCE or (idx == 0 or idx == len(vs_sorted) - 1):
                    filtered.append(p)
            peaks = filtered
        if not peaks:
            peaks = [max(marg.keys(), key=lambda k: marg[k])]
        # top-K cap: marg 상위순 PEAK_CAP 개만
        peaks = sorted(peaks, key=lambda k: marg[k], reverse=True)[:PEAK_CAP]
        drop_thr = mx_marg * INTERIOR_DROP_RATIO
        # 보호 set: peak + peak 좌/우 인접
        protected: set[int] = set()
        for p in peaks:
            p_int = int(p)
            if p_int not in marg:
                continue
            protected.add(p_int)
            idx = vs_sorted.index(p_int)
            if idx > 0:
                protected.add(vs_sorted[idx - 1])
            if idx + 1 < len(vs_sorted):
                protected.add(vs_sorted[idx + 1])
        # 내부 drop: marg < drop_thr 이고 protected 아니면 제외
        kept = [v for v in vs_sorted if v in protected or marg[v] >= drop_thr]
        kept_set = set(kept)
        # geom-mid: 각 peak 좌/우 양쪽 모두 (gap 조건 만족 시). peak 갯수 cap 없음.
        new_vals: set[int] = set()
        for p in peaks:
            p_int = int(p)
            if p_int not in kept_set:
                continue
            # 좌측
            idx = kept.index(p_int) if p_int in kept else -1
            if idx > 0:
                left = kept[idx - 1]
                if left > 0 and p_int / left >= MIN_GAP_RATIO:
                    m = _geom_mid(left, p_int)
                    if m is not None and m not in kept_set:
                        new_vals.add(m)
            # 우측
            if 0 <= idx < len(kept) - 1:
                right = kept[idx + 1]
                if p_int > 0 and right / p_int >= MIN_GAP_RATIO:
                    m = _geom_mid(p_int, right)
                    if m is not None and m not in kept_set:
                        new_vals.add(m)
        final_set = kept_set | new_vals
        # axis hard cap: peaks + geom-mid(new_vals) 우선 보존, 나머지는 marg 상위순으로 AXIS_CAP까지.
        if len(final_set) > AXIS_CAP:
            peaks_set = set(int(p) for p in peaks)
            new_set = set(new_vals)
            must_keep = peaks_set | new_set
            if len(must_keep) > AXIS_CAP:
                # must_keep도 초과 → peaks + top-(AXIS_CAP-len(peaks)) new_vals
                slots = max(0, AXIS_CAP - len(peaks_set))
                new_list = sorted(new_set)[:slots]
                final_set = peaks_set | set(new_list)
            else:
                rest = [v for v in final_set if v not in must_keep]
                rest_sorted = sorted(rest, key=lambda v: marg.get(v, -1e18), reverse=True)
                slots = AXIS_CAP - len(must_keep)
                final_set = must_keep | set(rest_sorted[:slots])
        grid[axis] = sorted(final_set)
    # vol (categorical): 조건부 drop - worst top-Q mean < max*0.3 일 때만, min 2 유지
    vol_marg = _axis_marginals(top_rows, "vol")
    if vol_marg:
        pairs = sorted(vol_marg.keys(), key=lambda k: vol_marg[k], reverse=True)
        if len(pairs) > MIN_VOL:
            worst_key = pairs[-1]
            mx = max(vol_marg.values())
            if vol_marg[worst_key] < mx * VOL_DROP_RATIO:
                pairs = pairs[:len(pairs) - 1]
        grid["vol"] = sorted([(m, float(t)) for (m, t) in pairs])
    return grid


# ─────────────────────────────────────────────────────────────────────────
# 6. Stage runner
# ─────────────────────────────────────────────────────────────────────────
def run_stage(stage_idx: int, grids: dict, workers: int = 24,
              pilot: int | None = None,
              bucket_filter: list | None = None) -> pd.DataFrame:
    out_dir = os.path.join(OUT_ROOT, f"stage_{stage_idx}")
    os.makedirs(out_dir, exist_ok=True)
    raw_path = os.path.join(out_dir, "raw.csv")
    grid_path = os.path.join(out_dir, "grid.json")
    def _serialize_one(ax):
        return {a: [list(v) if isinstance(v, tuple) else v for v in vals]
                for a, vals in ax.items()}
    def _serialize_grids(gs):
        out = {}
        for k, val in gs.items():
            sk = "_".join(map(str, k)) if isinstance(k, tuple) else str(k)
            if isinstance(val, list):
                out[sk] = [_serialize_one(g) for g in val]
            else:
                out[sk] = _serialize_one(val)
        return out
    with open(grid_path, "w") as f:
        json.dump(_serialize_grids(grids), f, indent=2, default=str)
    all_tasks = build_tasks(grids, bucket_filter=bucket_filter)
    if pilot:
        all_tasks = all_tasks[:pilot]

    # 재시작 안전 + cross-stage dedup: 현재 stage + 모든 이전 stage raw.csv
    done_tags = set()
    for s in range(1, stage_idx + 1):
        p = os.path.join(OUT_ROOT, f"stage_{s}", "raw.csv")
        if os.path.exists(p):
            try:
                prev = pd.read_csv(p, usecols=["tag"])
                n_before = len(done_tags)
                done_tags.update(prev["tag"].astype(str).tolist())
                added = len(done_tags) - n_before
                print(f"[stage{stage_idx}] dedup from stage{s}: +{added} "
                      f"(total {len(done_tags)})", flush=True)
            except Exception:
                pass
    tasks = [t for t in all_tasks if t["tag"] not in done_tags]
    n_tasks = len(tasks)
    print(f"[stage{stage_idx}] tasks: {n_tasks} (skip {len(all_tasks)-n_tasks}) "
          f"workers: {workers}", flush=True)
    if n_tasks == 0:
        if os.path.exists(raw_path):
            return pd.read_csv(raw_path).reindex(columns=RAW_COLUMNS)
        return pd.DataFrame(columns=RAW_COLUMNS)

    # 스트리밍 append: 매 N rows마다 디스크 flush
    write_header = not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0
    fout = open(raw_path, "a", buffering=1)
    try:
        if write_header:
            fout.write(",".join(RAW_COLUMNS) + "\n")
        t0 = time.time()
        buf = []
        with Pool(workers, initializer=preload) as pool:
            for i, r in enumerate(pool.imap_unordered(_run_task, tasks,
                                                       chunksize=4), 1):
                buf.append(r)
                if len(buf) >= 50:
                    _flush_rows(fout, buf)
                    buf.clear()
                if i % 200 == 0 or i == n_tasks:
                    el = time.time() - t0
                    rate = i / max(el, 1e-6)
                    eta = (n_tasks - i) / max(rate, 1e-6)
                    print(f"  [{i}/{n_tasks}] {rate:.1f}/s eta={eta/60:.1f}m",
                          flush=True)
        if buf:
            _flush_rows(fout, buf)
    finally:
        fout.close()
    return pd.read_csv(raw_path).reindex(columns=RAW_COLUMNS)


def _flush_rows(fout, rows):
    for r in rows:
        vals = []
        for c in RAW_COLUMNS:
            v = r.get(c, "")
            if v is None:
                vals.append("")
            else:
                vals.append(str(v).replace(",", ";"))
        fout.write(",".join(vals) + "\n")


def _add_rank_sum(rows: list[dict]) -> list[dict]:
    """rank_sum = rank(Cal desc) + rank(Sh desc) + rank(CAGR desc) + rank(MDD asc).
    낮을수록 좋음. 동순위는 평균 rank."""
    def _rk(vals, reverse=True):
        # reverse=True: 큰 값이 rank 1
        indexed = list(enumerate(vals))
        indexed.sort(key=lambda t: t[1], reverse=reverse)
        ranks = [0.0] * len(vals)
        i = 0
        while i < len(indexed):
            j = i
            while j + 1 < len(indexed) and indexed[j + 1][1] == indexed[i][1]:
                j += 1
            avg = (i + j) / 2 + 1
            for k in range(i, j + 1):
                ranks[indexed[k][0]] = avg
            i = j + 1
        return ranks
    cal = [float(r.get("Cal") or 0) for r in rows]
    sh  = [float(r.get("Sh") or 0) for r in rows]
    cagr = [float(r.get("CAGR") or 0) for r in rows]
    mdd  = [float(r.get("MDD") or 0) for r in rows]  # MDD는 음수 — 덜 음수일수록 좋음 → reverse=True
    rk_cal = _rk(cal, True)
    rk_sh  = _rk(sh, True)
    rk_cagr = _rk(cagr, True)
    rk_mdd  = _rk(mdd, True)
    for i, r in enumerate(rows):
        r["rank_sum"] = rk_cal[i] + rk_sh[i] + rk_cagr[i] + rk_mdd[i]
    return rows


def stage_to_bucket_grids(df: pd.DataFrame, stage_idx: int,
                          prev_grids: dict | None = None,
                          pending_drops: dict | None = None) -> tuple[dict, dict]:
    """stage raw → {(asset,iv,lev): grid}. per-axis peak 기반.
    rank_sum 상위 20% 선정. 2-stage persistence: 이번에 drop 후보인 값이
    지난번 pending_drops에도 있었을 때만 실제 drop. 아니면 이전 grid에 재포함.
    반환: (new_grids, new_pending_drops) — pending_drops는 이번 stage의 drop candidates.
    """
    out = {}
    new_pending = {}
    pending_drops = pending_drops or {}
    prev_grids = prev_grids or {}
    for (asset, iv, lev), g in df.groupby(["asset", "iv", "lev"]):
        bucket_key = (asset, iv, lev)
        rows = _clean_rows(g.to_dict("records"))
        # orphan axis value 제거: 현재 stage 에서 의도한 prev_grid 값만 유지.
        # 이전 partial run 이 append 해둔 과거 grid 결과가 marg 를 오염시키는 것을 차단.
        prev_bucket = prev_grids.get(bucket_key) if prev_grids else None
        if prev_bucket:
            allowed = {}
            for ax in NUMERIC_AXES:
                if ax in prev_bucket:
                    try:
                        allowed[ax] = {int(v) for v in prev_bucket[ax]}
                    except (ValueError, TypeError):
                        pass
            def _ok(r):
                for ax, s in allowed.items():
                    v = r.get(ax)
                    if v is None:
                        continue
                    try:
                        if int(v) not in s:
                            return False
                    except (ValueError, TypeError):
                        return False
                return True
            rows = [r for r in rows if _ok(r)]
        if not rows:
            continue
        rows = _add_rank_sum(rows)
        # rank_sum 낮을수록 좋음
        rows_sorted = sorted(rows, key=lambda r: r.get("rank_sum", 1e9))
        n_top = max(8, int(len(rows_sorted) * TOP_PCT))
        n_top = min(n_top, len(rows_sorted))
        top = rows_sorted[:n_top]
        candidate = bucket_to_grid(top, iv, stage_idx)
        if not candidate or not all(candidate.get(a) for a in NUMERIC_AXES + ["vol"]):
            continue
        prev = prev_grids.get(bucket_key, {})
        last_drops = pending_drops.get(bucket_key, {})
        final = {}
        cur_drops = {}
        for axis in NUMERIC_AXES + ["vol"]:
            cur = candidate.get(axis, [])
            prev_vals = prev.get(axis, cur)
            def _to_set(vals):
                return {tuple(v) if isinstance(v, (list, tuple)) else v for v in vals}
            cur_set = _to_set(cur)
            prev_set = _to_set(prev_vals)
            dropped_now = prev_set - cur_set
            prev_drop_set = last_drops.get(axis, set())
            real_drops = dropped_now & prev_drop_set
            # 2-stage persistence 제거: 현재 stage 에서 drop 후보는 즉시 drop.
            # reinstated union 이 축 폭을 보존시켜 폭발의 원인이 됨. axis hard cap 과 prominence/peak-cap 이 noise 방어를 대체.
            merged = cur_set
            # back to list form
            merged_list = []
            for item in merged:
                if isinstance(item, tuple) and axis == "vol":
                    merged_list.append((item[0], float(item[1])))
                else:
                    merged_list.append(item)
            if axis == "vol":
                final[axis] = sorted(merged_list)
            else:
                final[axis] = sorted(int(v) for v in merged_list)
            cur_drops[axis] = dropped_now
            if real_drops:
                print(f"  [{asset}_{iv}_{int(lev)} {axis}] real drop (2-stage): {real_drops}", flush=True)
        out[bucket_key] = final
        new_pending[bucket_key] = cur_drops
    return out, new_pending


def grid_change_pct(g_old, g_new) -> float:
    """두 grid의 unique-value-set 변화율. iv-키와 bucket-키 둘 다 처리.
    bucket-키일 때는 (asset,iv,lev,axis,value) 단위로 비교."""
    def flat(g):
        s = set()
        if not g:
            return s
        for k, val in g.items():
            tag = (k,) if isinstance(k, str) else tuple(map(str, k))
            grid_list = val if isinstance(val, list) else [val]
            for axes in grid_list:
                for axis, vals in axes.items():
                    for v in vals:
                        s.add(tag + (axis, str(v)))
        return s
    a, b = flat(g_old), flat(g_new)
    if not a:
        return 1.0
    diff = (a ^ b)
    return len(diff) / max(len(a), 1)


# ─────────────────────────────────────────────────────────────────────────
# 7. Main loop
# ─────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--max-iters", type=int, default=3)
    ap.add_argument("--pilot", type=int, default=None,
                    help="첫 stage에서 N tasks만 (스모크)")
    ap.add_argument("--coarse-only", action="store_true")
    ap.add_argument("--bucket", nargs="+", default=None,
                    help="버킷 필터: 'spot_D_1' / 'spot_D' / 'D' / 'spot' 가능")
    ap.add_argument("--resume-from-stage1", type=str, default=None,
                    help="기존 stage1 raw.csv 경로 — 그걸 읽어 stage2부터 시작")
    args = ap.parse_args()

    os.makedirs(OUT_ROOT, exist_ok=True)
    grids = COARSE
    all_dfs = []
    history = []
    start_iter = 1

    pending_drops = {}
    if args.resume_from_stage1:
        df_resume = pd.read_csv(args.resume_from_stage1)
        all_dfs.append(df_resume)
        history.append({"stage": 1, "resumed_from": args.resume_from_stage1,
                        "rows": len(df_resume)})
        # stage_idx=2 설정 → iter1 soft band(±2)
        grids, pending_drops = stage_to_bucket_grids(
            df_resume, stage_idx=2, prev_grids=None, pending_drops=None)
        change = grid_change_pct(COARSE, grids)
        print(f"[resume stage1] rows={len(df_resume)} → grid change "
              f"{change*100:.1f}% → start at stage 2", flush=True)
        start_iter = 2

    prev_top_cal = None
    no_improve_count = 0
    for it in range(start_iter, args.max_iters + 1):
        df = run_stage(it, grids, workers=args.workers,
                       pilot=args.pilot if it == 1 else None,
                       bucket_filter=args.bucket)
        all_dfs.append(df)
        history.append({"stage": it, "rows": len(df)})
        if args.coarse_only:
            break
        # 다음 스테이지 grid = 현재 df 기반 per-axis peak + 2-stage persistence
        new_grids, pending_drops = stage_to_bucket_grids(
            df, stage_idx=it + 1,
            prev_grids=grids if isinstance(next(iter(grids.keys()), None), tuple) else None,
            pending_drops=pending_drops)
        change = grid_change_pct(grids, new_grids)
        # top Cal 개선도
        try:
            cur_top = float(df["Cal"].astype(float).max())
        except Exception:
            cur_top = 0.0
        improve = (cur_top - prev_top_cal) / max(abs(prev_top_cal), 1e-6) \
                  if prev_top_cal is not None else float("inf")
        print(f"[stage{it}→{it+1}] grid change: {change*100:.1f}% "
              f"top_Cal={cur_top:.3f} Δ={improve*100:.2f}%", flush=True)
        history.append({"stage_change": it, "change_pct": change,
                        "top_cal": cur_top, "improve_pct": improve})
        # 수렴: grid change<5% AND top Cal 개선<1% 2회 연속 (둘 다 만족해야 종료)
        if it > 1:
            grid_stable = change < 0.05
            if improve < 0.01:
                no_improve_count += 1
            else:
                no_improve_count = 0
            cal_stable = no_improve_count >= 2
            if grid_stable and cal_stable:
                print(f"[converged (grid+Cal) at stage {it}]", flush=True)
                break
        prev_top_cal = cur_top
        grids = new_grids

    final = pd.concat(all_dfs, ignore_index=True)
    final = final.drop_duplicates(subset=["tag"], keep="last")
    final.to_csv(os.path.join(OUT_ROOT, "raw_combined.csv"), index=False)
    with open(os.path.join(OUT_ROOT, "history.json"), "w") as f:
        json.dump(history, f, indent=2)
    print(f"[done] combined rows: {len(final)} → {OUT_ROOT}/raw_combined.csv")


if __name__ == "__main__":
    main()
