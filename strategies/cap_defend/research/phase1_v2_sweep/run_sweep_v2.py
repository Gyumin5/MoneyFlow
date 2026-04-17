#!/usr/bin/env python3
"""Phase-1 브루트포스 sweep.

스위프 대상:
- 현물 (L=1): 1D + 4h, daily/bar vol (top-2 per interval)
- 선물 L2/L3/L4: 1D + 4h + 2h
- 3 start-date anchors (3주 간격 stagger) → FULL_END 2026-04-13
- 제약: Mom_l >= 2 × Mom_s
- 스탑 없음, 갭/excl 탈출 없음

Output: phase1_sweep/{raw.csv, summary.csv, run.log}
"""
from __future__ import annotations
import argparse
import gc
import itertools
import json
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
RES = os.path.dirname(HERE)
CAP = os.path.dirname(RES)
REPO = os.path.dirname(CAP)
sys.path.insert(0, REPO)
sys.path.insert(0, CAP)
sys.path.insert(0, os.path.join(REPO, "trade"))

FULL_END = "2026-04-13"
# Phase-1: 3개 시작일을 3주 간격으로 두어 snap-phase 다양성 확보. 모두 FULL_END까지 풀 기간.
ANCHORS = ["2020-10-01"]

# ─── Grid definitions (v2 extended: 경계 해소용) ───
# 기존 16,416 configs는 phase1_sweep/raw.csv에서 복사해 resume으로 스킵.
# 추가 축:
#   1D  ml{360,480} / vol{d0.07,d0.10} / snap{90,120}
#   4H  sma{1080,1440} / ms{10,15} / ml{1080,1440,2160}
#       / vol{d0.05,b0.60,b0.70} / snap{240,336}
#   2H  sma{720} / ml{720} / vol{d0.05,b0.60,b0.70} / snap{6,8,240}
GRID_1D = {
    "sma": [20, 30, 40, 50, 60, 90, 150],
    "ms":  [10, 20, 30, 40, 60, 90],
    "ml":  [60, 90, 120, 240, 360, 480],
    "vol": [("daily", 0.03), ("daily", 0.05),
            ("daily", 0.07), ("daily", 0.10)],
    "snap": [12, 21, 30, 45, 60, 90, 120],
}
GRID_4H = {
    "sma": [120, 180, 240, 360, 480, 720, 1080, 1440],
    "ms":  [10, 15, 20, 30, 40, 60, 90, 120],
    "ml":  [120, 240, 480, 720, 1080, 1440, 2160],
    "vol": [("daily", 0.03), ("daily", 0.05),
            ("bar", 0.50), ("bar", 0.60), ("bar", 0.70)],
    "snap": [21, 30, 60, 84, 90, 168, 240, 336],
}
GRID_2H = {
    "sma": [60, 120, 180, 240, 360, 480, 720],
    "ms":  [10, 20, 30, 40, 60, 90],
    "ml":  [60, 120, 240, 480, 720],
    "vol": [("bar", 0.50), ("daily", 0.03), ("daily", 0.05),
            ("bar", 0.60), ("bar", 0.70)],
    "snap": [6, 8, 12, 21, 30, 60, 84, 120, 168, 240],
}
GRIDS = {"D": GRID_1D, "4h": GRID_4H, "2h": GRID_2H}

FUTURES_LEVS = [2.0, 3.0, 4.0]
CHECKPOINT_EVERY = 200
SUMMARY_EVERY = 2000  # summary.csv는 더 드물게 갱신 (디스크 전체 재로드 비용)
RAW_COLUMNS = [
    "tag", "anchor", "asset", "lev",
    "Sh", "Cal", "CAGR", "MDD", "CVaR5", "Ulcer", "TUW",
    "rebal", "liq", "error",
]
CORE_METRIC_COLUMNS = ["Sh", "Cal", "CAGR", "MDD"]


def enum_configs(interval: str):
    """Mom_l >= 2*Mom_s 제약 포함."""
    g = GRIDS[interval]
    for sma, ms, ml, (vm, vt), snap in itertools.product(
        g["sma"], g["ms"], g["ml"], g["vol"], g["snap"]
    ):
        if ml < 2 * ms:
            continue
        yield {"interval": interval, "sma": sma, "ms": ms, "ml": ml,
               "vol_mode": vm, "vol_thr": vt, "snap": snap}


def cfg_id(cfg: dict, asset: str, lev: float) -> str:
    iv = cfg["interval"].replace("D", "1D")
    vtag = f"{cfg['vol_mode'][0]}{cfg['vol_thr']:.2f}"
    return (f"{asset}_{iv}_S{cfg['sma']}_M{cfg['ms']}_{cfg['ml']}"
            f"_{vtag}_SN{cfg['snap']}_L{int(lev)}")


# ─── Data loaded in parent process, inherited via fork ───
# V2: spot/fut 공통으로 Binance 1h perp 데이터만 사용 (unified_backtest.py).
# Upbit 의존 제거 — 현물도 Binance USD perp을 proxy로 백테스트.
_FUT_DATA = None


def preload_all():
    """부모 프로세스에서 1회 로드 → Pool fork 시 copy-on-write로 워커에 상속."""
    global _FUT_DATA
    from unified_backtest import load_data
    print("[preload] binance 1h perp (spot+fut 공통)...", flush=True)
    _FUT_DATA = {iv: load_data(iv) for iv in ["1h", "2h", "4h", "D"]}
    print("[preload] done.", flush=True)


def _equity_metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2:
        return {"Sh": 0, "Cal": 0, "CAGR": 0, "MDD": 0,
                "CVaR5": 0, "Ulcer": 0, "TUW": 0}
    ed = eq.resample("D").last().dropna()
    if len(ed) < 2 or ed.iloc[0] <= 0:
        return {"Sh": 0, "Cal": 0, "CAGR": 0, "MDD": 0,
                "CVaR5": 0, "Ulcer": 0, "TUW": 0}
    yrs = (ed.index[-1] - ed.index[0]).days / 365.25
    cagr = (ed.iloc[-1] / ed.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0
    dr = ed.pct_change().dropna()
    sh = float(dr.mean() / dr.std() * np.sqrt(365)) if dr.std() > 0 else 0
    dd = ed / ed.cummax() - 1
    mdd = float(dd.min())
    cal = cagr / abs(mdd) if mdd else 0
    cvar5 = float(np.percentile(dr, 5)) if len(dr) > 10 else 0
    ulcer = float(np.sqrt((dd ** 2).mean()))
    tuw = float((dd < -0.05).mean())  # fraction of days under -5%
    return {"Sh": sh, "Cal": cal, "CAGR": cagr, "MDD": mdd,
            "CVaR5": cvar5, "Ulcer": ulcer, "TUW": tuw}


def _build_params(cfg: dict) -> dict:
    """공통 전략 파라미터 구성. 갭/excl/스탑 모두 off (Phase-1 정책)."""
    return {
        "sma_bars": cfg["sma"],
        "mom_short_bars": cfg["ms"],
        "mom_long_bars": cfg["ml"],
        "vol_mode": cfg["vol_mode"],
        "vol_threshold": cfg["vol_thr"],
        "snap_interval_bars": cfg["snap"],
        "n_snapshots": 3,
        "canary_hyst": 0.015,
        "universe_size": 5,
        "cap": 1 / 3,
        "health_mode": "mom2vol",
        # Phase-1: 스탑/갭/excl 전부 off
        "stop_kind": "none",
        "stop_pct": 0.0,
        # drift/PFD 기본
        "drift_threshold": 0.10,
        "post_flip_delay": 5,
        # DD/BL/Crash (선물 d005 기준값 유지)
        "dd_lookback": 60, "dd_threshold": -0.25,
        "bl_drop": -0.15, "bl_days": 7,
        "crash_threshold": -0.10,
    }


def _run_asset(cfg: dict, asset_type: str, lev: float, anchor: str) -> dict:
    """Unified backtest 호출. asset_type='spot'|'fut'."""
    global _FUT_DATA
    from unified_backtest import run as bt_run

    iv = cfg["interval"]
    bars, funding = _FUT_DATA[iv]
    params = _build_params(cfg)
    if asset_type == "spot":
        tx = 0.004   # 업비트 수수료 기준
        lev_use = 1.0
    else:
        tx = 0.0004  # 바이낸스 maker
        lev_use = lev
    m = bt_run(
        bars, funding, interval=iv,
        asset_type=asset_type,
        leverage=lev_use, tx_cost=tx,
        start_date=anchor, end_date=FULL_END,
        **params,
    )
    if not m:
        return {"Sh": 0, "Cal": 0, "CAGR": 0, "MDD": 0,
                "CVaR5": 0, "Ulcer": 0, "TUW": 0, "rebal": 0, "liq": 0}
    eq = m.get("_equity")
    if eq is not None:
        if not isinstance(eq, pd.Series):
            eq = pd.Series(eq)
        ext = _equity_metrics(eq)
        cvar5, ulcer, tuw = ext["CVaR5"], ext["Ulcer"], ext["TUW"]
    else:
        cvar5 = ulcer = tuw = 0.0
    return {"Sh": float(m.get("Sharpe", 0)), "Cal": float(m.get("Cal", 0)),
            "CAGR": float(m.get("CAGR", 0)), "MDD": float(m.get("MDD", 0)),
            "CVaR5": float(cvar5), "Ulcer": float(ulcer), "TUW": float(tuw),
            "rebal": int(m.get("Rebal", 0)),
            "liq": int(m.get("Liq", 0))}


def _run_spot(cfg: dict, anchor: str) -> dict:
    return _run_asset(cfg, "spot", 1.0, anchor)


def _run_futures(cfg: dict, lev: float, anchor: str) -> dict:
    return _run_asset(cfg, "fut", lev, anchor)


def _run_task(task: dict) -> dict:
    tag = task["tag"]
    cfg = task["cfg"]
    asset = task["asset"]
    lev = task["lev"]
    anchor = task["anchor"]
    try:
        if asset == "spot":
            m = _run_spot(cfg, anchor)
        else:
            m = _run_futures(cfg, lev, anchor)
        return {"tag": tag, "anchor": anchor, "asset": asset, "lev": lev, **m}
    except Exception as e:
        return {"tag": tag, "anchor": anchor, "asset": asset, "lev": lev,
                "error": str(e)[:200]}


def build_tasks(pilot: bool = False, parts: list = None,
                intervals: list | None = None) -> list:
    tasks = []
    parts = parts or ["spot", "futures"]
    spot_ivs = ["D", "4h"] if intervals is None else [i for i in ["D", "4h"] if i in intervals]
    fut_ivs = ["D", "4h", "2h"] if intervals is None else [i for i in ["D", "4h", "2h"] if i in intervals]

    def add(asset, cfg, lev):
        tag = cfg_id(cfg, asset, lev)
        for a in ANCHORS:
            tasks.append({"tag": tag, "cfg": cfg, "asset": asset,
                          "lev": lev, "anchor": a})

    if "spot" in parts:
        for iv in spot_ivs:
            configs = list(enum_configs(iv))
            if pilot:
                configs = configs[:3]
            for cfg in configs:
                add("spot", cfg, 1.0)

    if "futures" in parts:
        for iv in fut_ivs:
            configs = list(enum_configs(iv))
            if pilot:
                configs = configs[:3]
            for cfg in configs:
                for lev in FUTURES_LEVS:
                    add("fut", cfg, lev)

    return tasks


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    df = df[df.get("error", pd.NA).isna() if "error" in df.columns else slice(None)]
    rows = []
    metric_cols = ["Sh", "Cal", "CAGR", "MDD", "CVaR5", "Ulcer", "TUW"]
    for (tag, asset, lev), g in df.groupby(["tag", "asset", "lev"]):
        d = {"tag": tag, "asset": asset, "lev": lev, "n": len(g)}
        for c in metric_cols:
            if c in g:
                d[f"m{c}"] = float(g[c].mean())
                if c in ("Sh", "Cal", "CAGR"):
                    d[f"s{c}"] = float(g[c].std())
        if "MDD" in g:
            d["wMDD"] = float(g["MDD"].min())
        # anchor win-rate: Cal > 0
        if "Cal" in g:
            d["win_rate"] = float((g["Cal"] > 0).mean())
        # extra
        if "rebal" in g:
            d["rebal_mean"] = float(g["rebal"].mean())
        if "liq" in g:
            d["liq_sum"] = int(g["liq"].sum()) if g["liq"].notna().any() else 0
        rows.append(d)
    return pd.DataFrame(rows)


def load_existing_results(raw_path: str) -> pd.DataFrame:
    if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        return pd.DataFrame(columns=RAW_COLUMNS)
    try:
        df = pd.read_csv(raw_path, on_bad_lines="skip")
    except Exception:
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = df.reindex(columns=RAW_COLUMNS)
    df = df.dropna(subset=["tag", "anchor", "asset", "lev"]).copy()
    success_mask = df["error"].isna()
    valid_core = df[CORE_METRIC_COLUMNS].notna().all(axis=1)
    df = df[(~success_mask) | valid_core].copy()
    df["_row_id"] = np.arange(len(df))
    df["_ok"] = df["error"].isna()
    df = (df.sort_values(["tag", "anchor", "_ok", "_row_id"])
            .drop_duplicates(subset=["tag", "anchor"], keep="last")
            .drop(columns=["_row_id", "_ok"]))
    return df


def load_completed_keys(raw_path: str) -> set[tuple[str, str]]:
    df = load_existing_results(raw_path)
    if df.empty or "tag" not in df.columns or "anchor" not in df.columns:
        return set()
    # error row와 핵심 지표 누락(torn) row는 재시도 대상으로 남긴다
    if "error" in df.columns:
        df = df[df["error"].isna()]
    df = df[df[CORE_METRIC_COLUMNS].notna().all(axis=1)]
    keys = df[["tag", "anchor"]].dropna().drop_duplicates()
    return set(map(tuple, keys.itertuples(index=False, name=None)))


def append_raw_rows(raw_path: str, rows: list[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=RAW_COLUMNS)
    df = pd.DataFrame(rows).reindex(columns=RAW_COLUMNS)
    need_header = not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0
    with open(raw_path, "a", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False, header=need_header)
        f.flush()
        os.fsync(f.fileno())
    return df


def atomic_write_csv(df: pd.DataFrame, path: str) -> None:
    tmp_path = f"{path}.tmp"
    with open(tmp_path, "w", encoding="utf-8", newline="") as f:
        df.to_csv(f, index=False)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp_path, path)


def write_manifest(path: str, data: dict) -> None:
    data = {**data, "updated_at": time.strftime("%Y-%m-%d %H:%M:%S")}
    tmp = f"{path}.tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parts", nargs="+", default=["spot", "futures"],
                    choices=["spot", "futures"])
    ap.add_argument("--processes", type=int, default=24)
    ap.add_argument("--pilot", action="store_true",
                    help="3 configs per interval, fast smoke test")
    ap.add_argument("--out-dir", default=HERE)
    ap.add_argument("--intervals", nargs="+", default=None,
                    choices=["D", "4h", "2h"],
                    help="특정 봉만 실행 (분할 작업용)")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    log_path = os.path.join(args.out_dir, "run.log")
    raw_path = os.path.join(args.out_dir, "raw.csv")
    summary_path = os.path.join(args.out_dir, "summary.csv")
    log = open(log_path, "a", buffering=1)

    def P(msg):
        print(msg, flush=True)
        log.write(msg + "\n")

    P(f"\n{'='*60}")
    P(f"[RUN START] {time.strftime('%Y-%m-%d %H:%M:%S')} pid={os.getpid()} argv={sys.argv[1:]}")
    P(f"{'='*60}")

    all_tasks = build_tasks(pilot=args.pilot, parts=args.parts, intervals=args.intervals)
    completed = load_completed_keys(raw_path)
    existing_df = load_existing_results(raw_path)
    tasks = [t for t in all_tasks if (t["tag"], t["anchor"]) not in completed]

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    write_manifest(manifest_path, {
        "status": "running", "stage": "phase1_sweep",
        "total_tasks": len(all_tasks),
        "done_tasks": len(completed),
        "parts": args.parts, "pilot": args.pilot,
    })

    P(f"Tasks: {len(all_tasks)} ({len(all_tasks)//len(ANCHORS)} configs × {len(ANCHORS)} anchors)")
    P(f"Parts: {args.parts}  Processes: {args.processes}  Pilot: {args.pilot}")
    P(f"Resume: {len(completed)} done, {len(tasks)} remaining")

    if not tasks:
        sdf = summarize(existing_df)
        atomic_write_csv(sdf, summary_path)
        write_manifest(manifest_path, {
            "status": "done", "stage": "phase1_sweep",
            "total_tasks": len(all_tasks),
            "done_tasks": len(all_tasks),
        })
        P("Nothing to run. summary.csv refreshed from existing raw.csv")
        P(f"\nDone in 0s")
        log.close()
        return

    preload_all()

    # existing_df no longer kept in memory — summary is recomputed from disk each checkpoint
    del existing_df
    gc.collect()

    t0 = time.time()
    pending_rows = []
    with Pool(processes=args.processes, maxtasksperchild=150) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_task, tasks, chunksize=1), 1):
            pending_rows.append(r)
            if i % CHECKPOINT_EVERY == 0 or i == len(tasks):
                append_raw_rows(raw_path, pending_rows)
                pending_rows.clear()
                write_manifest(manifest_path, {
                    "status": "running", "stage": "phase1_sweep",
                    "total_tasks": len(all_tasks),
                    "done_tasks": len(completed) + i,
                })
                if i % SUMMARY_EVERY == 0 or i == len(tasks):
                    disk_df = load_existing_results(raw_path)
                    sdf = summarize(disk_df)
                    atomic_write_csv(sdf, summary_path)
                    del disk_df, sdf
                gc.collect()
                elapsed = int(time.time() - t0)
                eta = int(elapsed / i * (len(tasks) - i))
                done_total = len(completed) + i
                P(
                    f"  [{done_total}/{len(all_tasks)}] pending={len(tasks)-i} "
                    f"elapsed={elapsed}s eta={eta}s last={r.get('tag','')[:40]}"
                )

    df = load_existing_results(raw_path)
    sdf = summarize(df)
    atomic_write_csv(sdf, summary_path)
    write_manifest(manifest_path, {
        "status": "done", "stage": "phase1_sweep",
        "total_tasks": len(all_tasks),
        "done_tasks": len(all_tasks),
    })

    P(f"\n=== TOP 10 by mCal per (asset, lev) ===")
    for (a, l), g in sdf.groupby(["asset", "lev"]):
        top = g.sort_values("mCal", ascending=False).head(10)
        P(f"\n-- {a} L{l} ({len(g)} configs) --")
        P(top[["tag", "mCal", "mSh", "mCAGR", "wMDD", "win_rate"]].to_string(index=False))

    P(f"\nDone in {int(time.time()-t0)}s")
    log.close()


if __name__ == "__main__":
    main()
