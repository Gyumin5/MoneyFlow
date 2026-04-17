#!/usr/bin/env python3
"""Phase-1 10배수 그리드 sweep (과적합 방지용 재설계).

변경점 vs phase1_sweep:
- 10배수 그리드만 사용 (iter_refine 생성 값 배제)
- D봉 SMA plateau를 넓은 간격으로 커버 (20/30/50/100/150)
- ML은 2×MS 제약 유지
- 2h 인터벌 제외 (실운영 리스크)
- 3 futures leverage (2/3/4), spot L=1
- 1 anchor (2020-10-01) 풀 기간

Output: phase1_10x/{raw.csv, summary.csv, run.log}
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

FULL_END = "2023-12-31"
ANCHORS = ["2020-10-01"]

# 10배수 그리드 — 넓은 spacing, plateau 탐지용
GRID_1D = {
    "sma":  [20, 30, 50, 100, 150],
    "ms":   [10, 20, 30, 60],
    "ml":   [60, 90, 120, 240],
    "vol":  [("daily", 0.03), ("daily", 0.05), ("daily", 0.07)],
    "snap": [20, 30, 60, 90],
}
GRID_4H = {
    "sma":  [120, 240, 480, 720],
    "ms":   [20, 60, 120],
    "ml":   [120, 240, 480, 720],
    "vol":  [("daily", 0.05), ("bar", 0.50), ("bar", 0.70)],
    "snap": [30, 60, 120, 180],
}
GRIDS = {"D": GRID_1D, "4h": GRID_4H}

FUTURES_LEVS = [2.0, 3.0, 4.0]
CHECKPOINT_EVERY = 100
SUMMARY_EVERY = 500
RAW_COLUMNS = [
    "tag", "anchor", "asset", "lev",
    "Sh", "Cal", "CAGR", "MDD", "CVaR5", "Ulcer", "TUW",
    "rebal", "liq", "error",
]
CORE_METRIC_COLUMNS = ["Sh", "Cal", "CAGR", "MDD"]


def enum_configs(interval: str):
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


_FUT_DATA = None
_SPOT_BARS = None
_SPOT_UNIV = None


def preload_all():
    global _FUT_DATA, _SPOT_BARS, _SPOT_UNIV
    from backtest_futures_full import load_data
    import run_current_coin_v20_backtest as spot_bt
    print("[preload] futures data...", flush=True)
    _FUT_DATA = {iv: load_data(iv) for iv in ["1h", "4h", "D"]}
    print("[preload] spot universe + bars...", flush=True)
    _SPOT_UNIV = spot_bt.load_universe(top_n=40)
    _SPOT_BARS = spot_bt.load_price_bars(_SPOT_UNIV)
    spot_bt.load_price_bars = lambda um: _SPOT_BARS
    spot_bt.load_universe = lambda top_n=40: _SPOT_UNIV
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
    tuw = float((dd < -0.05).mean())
    return {"Sh": sh, "Cal": cal, "CAGR": cagr, "MDD": mdd,
            "CVaR5": cvar5, "Ulcer": ulcer, "TUW": tuw}


def _run_spot(cfg: dict, anchor: str) -> dict:
    import run_current_coin_v20_backtest as spot_bt
    from coin_live_engine import MEMBER_D_SMA50, MEMBER_4H_SMA240

    template = MEMBER_D_SMA50 if cfg["interval"] == "D" else MEMBER_4H_SMA240
    member = dict(template)
    member.update({
        "interval": cfg["interval"],
        "sma_bars": cfg["sma"],
        "mom_short_bars": cfg["ms"],
        "mom_long_bars": cfg["ml"],
        "snap_interval_bars": cfg["snap"],
        "vol_mode": cfg["vol_mode"],
        "vol_threshold": cfg["vol_thr"],
        "gap_threshold": -1.0,
        "exclusion_days": 0,
    })
    spot_bt.MEMBERS = {"single": member}
    spot_bt.ENSEMBLE_WEIGHTS = {"single": 1.0}

    res = spot_bt.run_backtest(start=anchor, end=FULL_END)
    eq = res["equity"]
    m = _equity_metrics(eq)
    m["rebal"] = int(res.get("rebal_count", 0))
    return m


def _run_futures(cfg: dict, lev: float, anchor: str) -> dict:
    global _FUT_DATA
    from backtest_futures_full import run as bt_run
    from futures_ensemble_engine import SingleAccountEngine, combine_targets
    from run_futures_fixedlev_search import FIXED_CFG

    params = dict(FIXED_CFG)
    params.update({
        "vol_mode": cfg["vol_mode"],
        "sma_bars": cfg["sma"],
        "mom_short_bars": cfg["ms"],
        "mom_long_bars": cfg["ml"],
        "vol_threshold": cfg["vol_thr"],
        "snap_interval_bars": cfg["snap"],
    })
    iv = cfg["interval"]
    bars, funding = _FUT_DATA[iv]
    bars_1h, funding_1h = _FUT_DATA["1h"]
    trace: list = []
    bt_run(bars, funding, interval=iv, leverage=1.0,
           start_date=anchor, end_date=FULL_END, _trace=trace, **params)
    dates = bars_1h["BTC"].index
    ddates = dates[(dates >= anchor) & (dates <= FULL_END)]
    combined = combine_targets({"x": trace}, {"x": 1.0}, ddates)
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=lev, leverage_mode="fixed", per_coin_leverage_mode="none",
        stop_kind="none", stop_pct=0.0, stop_lookback_bars=0, stop_gate="always",
    )
    m = engine.run(combined)
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


def build_tasks(pilot: bool = False, parts: list = None) -> list:
    tasks = []
    parts = parts or ["spot", "futures"]

    def add(asset, cfg, lev):
        tag = cfg_id(cfg, asset, lev)
        for a in ANCHORS:
            tasks.append({"tag": tag, "cfg": cfg, "asset": asset,
                          "lev": lev, "anchor": a})

    if "spot" in parts:
        for iv in ["D", "4h"]:
            configs = list(enum_configs(iv))
            if pilot:
                configs = configs[:3]
            for cfg in configs:
                add("spot", cfg, 1.0)

    if "futures" in parts:
        for iv in ["D", "4h"]:
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
        if "Cal" in g:
            d["win_rate"] = float((g["Cal"] > 0).mean())
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
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--out-dir", default=HERE)
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

    all_tasks = build_tasks(pilot=args.pilot, parts=args.parts)
    completed = load_completed_keys(raw_path)
    existing_df = load_existing_results(raw_path)
    tasks = [t for t in all_tasks if (t["tag"], t["anchor"]) not in completed]

    manifest_path = os.path.join(args.out_dir, "manifest.json")
    write_manifest(manifest_path, {
        "status": "running", "stage": "phase1_10x",
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
            "status": "done", "stage": "phase1_10x",
            "total_tasks": len(all_tasks),
            "done_tasks": len(all_tasks),
        })
        P("Nothing to run.")
        log.close()
        return

    preload_all()
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
                    "status": "running", "stage": "phase1_10x",
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
        "status": "done", "stage": "phase1_10x",
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
