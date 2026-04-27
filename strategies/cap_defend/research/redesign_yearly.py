"""Step 4 — Yearly consistency 백테스트.

각 config 에 대해 연도별 (단일 년 단위) 백테스트 수행.
- 주식: 2017~2025 (9년)
- 코인: 2020~2025 (6년)

출력: redesign_yearly_{asset}.csv  (tag, year, Cal, CAGR, MDD, Sh)

지표 집계는 redesign_analyze.py 에서 수행 (본 스크립트는 raw 만).

Soft gate: 2022 Cal 이 자산별 p40 (코인) 또는 0.3 (주식) 이하면 Hard reject 플래그.

사용: python redesign_yearly.py --asset fut --workers 24
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
from redesign_common import parse_cfg
CAP = os.path.dirname(HERE)
REPO = os.path.dirname(CAP)
sys.path.insert(0, REPO)
sys.path.insert(0, CAP)

YEARS_BY_ASSET = {
    "fut":   list(range(2020, 2026)),
    "spot":  list(range(2020, 2026)),
    "stock": list(range(2017, 2026)),
}
TX_BY_ASSET = {"fut": 0.0004, "spot": 0.004, "stock": 0.0025}

_DATA = None


def preload():
    global _DATA
    from unified_backtest import load_data
    _DATA = {iv: load_data(iv) for iv in ("D", "4h")}


def year_range(year):
    return f"{year}-01-01", f"{year}-12-31"


def run_one(task):
    cfg = task["cfg"]
    asset = task["asset"]
    year = task["year"]
    tx = TX_BY_ASSET[asset]
    start, end = year_range(year)
    try:
        if asset == "stock":
            from redesign_stock_adapter import run_stock_from_cfg
            m = run_stock_from_cfg(cfg, phase_offset=0, tx_cost=tx,
                                    year_only=year)
            if m.get("status") != "ok":
                return {"tag": task["tag"], "asset": asset, "year": year,
                        "status": "error", "error": m.get("error", "")[:200]}
            return {"tag": task["tag"], "asset": asset, "year": year,
                    "status": "ok",
                    "Cal": m["Cal"], "Sh": m["Sh"],
                    "CAGR": m["CAGR"], "MDD": m["MDD"]}
        from unified_backtest import run as bt_run
        iv = cfg["iv"]
        bars, funding = _DATA[iv]
        lev = 1.0 if asset == "spot" else cfg["lev"]
        m = bt_run(
            bars, funding, interval=iv, asset_type=asset,
            leverage=lev, universe_size=3, cap=1 / 3, tx_cost=tx,
            sma_bars=cfg["sma"], mom_short_bars=cfg["ms"], mom_long_bars=cfg["ml"],
            vol_mode=cfg["vmode"], vol_threshold=cfg["vthr"],
            snap_interval_bars=cfg["snap"], n_snapshots=3,
            phase_offset_bars=0,
            canary_hyst=0.015, health_mode="mom2vol",
            stop_kind="none", stop_pct=0.0,
            drift_threshold=0.10, post_flip_delay=5,
            dd_lookback=60, dd_threshold=-0.25,
            bl_drop=-0.15, bl_days=7, crash_threshold=-0.10,
            start_date=start, end_date=end,
        )
        return {
            "tag": task["tag"], "asset": asset, "year": year,
            "status": "ok",
            "Cal": float(m.get("Cal") or 0),
            "Sh": float(m.get("Sharpe", 0) or 0),
            "CAGR": float(m.get("CAGR", 0) or 0),
            "MDD": float(m.get("MDD", 0) or 0),
        }
    except Exception as e:
        return {"tag": task["tag"], "asset": asset, "year": year,
                "status": "error", "error": str(e)[:200]}


def build_tasks(asset, phase_csv, out_csv, resume):
    """Step 3 명시 survivor CSV 기반 yearly tasks 생성.
    survivor 없으면 raw phase median>0 fallback.
    """
    surv_csv = os.path.join(HERE, f"redesign_phase_survivors_{asset}.csv")
    if os.path.exists(surv_csv):
        surv = pd.read_csv(surv_csv)
        if "pass_all" in surv.columns:
            survivors = surv[surv["pass_all"] == True]["tag"].astype(str).tolist()
        else:
            survivors = surv["tag"].astype(str).tolist()
    else:
        df = pd.read_csv(phase_csv)
        if "status" in df.columns:
            df = df[df["status"] == "ok"]
        df = df.dropna(subset=["Cal"])
        stats = df.groupby("tag")["Cal"].median()
        survivors = stats[stats > 0].index.astype(str).tolist()

    top_csv = os.path.join(HERE, f"redesign_top500_{asset}_k1.csv")
    base = pd.read_csv(top_csv)
    base = base[base["tag"].astype(str).isin(survivors)].copy()

    done = set()
    if os.path.exists(out_csv) and resume:
        try:
            prev = pd.read_csv(out_csv)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            prev = pd.DataFrame()
        if not prev.empty:
            if "status" in prev.columns:
                prev = prev[prev["status"] == "ok"]
            elif "error" in prev.columns:
                prev = prev[prev["error"].isna() | (prev["error"].astype(str).str.strip() == "")]
            if "Cal" in prev.columns:
                prev = prev.dropna(subset=["Cal"])

            if "MDD" in prev.columns:
                prev = prev.dropna(subset=["MDD"])
            if {"tag", "year"}.issubset(prev.columns):
                done = set(zip(prev["tag"].astype(str), prev["year"].astype(int)))

    tasks = []
    for _, row in base.iterrows():
        tag = str(row["tag"])
        cfg = parse_cfg(asset, row)
        if cfg is None:
            continue
        for year in YEARS_BY_ASSET[asset]:
            if (tag, int(year)) in done:
                continue
            tasks.append({"tag": tag, "asset": asset, "cfg": cfg, "year": year})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=["fut", "spot", "stock"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--flush", type=int, default=200)
    args = ap.parse_args()

    phase_csv = os.path.join(HERE, f"redesign_rerank_phase_{args.asset}.csv")
    out_csv = os.path.join(HERE, f"redesign_yearly_{args.asset}.csv")

    tasks = build_tasks(args.asset, phase_csv, out_csv, args.resume)
    print(f"[{args.asset}] {len(tasks)} yearly tasks", flush=True)
    if not tasks:
        return

    t0 = time.time()
    header_written = os.path.exists(out_csv)
    buf = []

    def flush_buf():
        nonlocal header_written, buf
        if not buf:
            return
        pd.DataFrame(buf).to_csv(out_csv, mode="a", header=not header_written, index=False)
        header_written = True
        buf = []

    initializer = preload if args.asset != "stock" else None
    with Pool(args.workers, initializer=initializer) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, tasks, chunksize=2), 1):
            buf.append(res)
            if len(buf) >= args.flush:
                flush_buf()
            if i % 300 == 0:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(tasks) - i) / max(rate, 1e-6)
                print(f"[{args.asset} {i}/{len(tasks)}] {rate:.2f}/s ETA {eta/60:.1f}m",
                      flush=True)
    flush_buf()
    print(f"[{args.asset} done] {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
