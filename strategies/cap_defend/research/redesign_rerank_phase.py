"""Step 3 — Phase robustness sweep (통합: fut/spot/stock).

Phase ratio: {0.00, 0.07, 0.13, 0.31, 0.49}
  → phase_offset_bars = round(ratio × snap_interval_bars), 중복 제거

병렬 24~28 worker. Resume: 출력 CSV 기완료 (tag, phase) skip.

사용:
  python redesign_rerank_phase.py --asset fut --workers 24
  python redesign_rerank_phase.py --asset spot --workers 24
  python redesign_rerank_phase.py --asset stock --workers 24
"""
from __future__ import annotations
import argparse
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
CAP = os.path.dirname(HERE)
REPO = os.path.dirname(CAP)
sys.path.insert(0, REPO)
sys.path.insert(0, CAP)

PHASE_RATIOS = [0.00, 0.07, 0.13, 0.31, 0.49]

ANCHOR = "2020-10-01"
END = "2026-04-13"
ANCHOR_STOCK = "2017-04-01"
END_STOCK = "2025-12-31"

TX_BY_ASSET = {"fut": 0.0004, "spot": 0.004, "stock": 0.0025}

_DATA = None


def preload():
    global _DATA
    from unified_backtest import load_data
    _DATA = {iv: load_data(iv) for iv in ("D", "4h")}


def phase_values(snap_interval_bars):
    vals = []
    for r in PHASE_RATIOS:
        v = int(round(r * snap_interval_bars))
        if v not in vals:
            vals.append(v)
    return vals


def parse_tag_row(row, asset):
    """공통 redesign_common.parse_cfg 위임 (canary_type/sharpe_lookback/mom_style/n_pick 누락 방지)."""
    from redesign_common import parse_cfg
    return parse_cfg(asset, row)


def run_one(task):
    cfg = task["cfg"]
    phase = task["phase"]
    asset = task["asset"]
    tx = TX_BY_ASSET[asset]
    try:
        if asset == "stock":
            # stock adapter 사용 (phase_offset 연결됨)
            from redesign_stock_adapter import run_stock_from_cfg
            m = run_stock_from_cfg(
                cfg, phase_offset=phase, tx_cost=tx,
                start=ANCHOR_STOCK, end=END_STOCK,
            )
            if m.get("status") != "ok":
                return {"tag": task["tag"], "asset": asset, "phase": phase, "k": 1,
                        "status": "error", "error": m.get("error", "")[:200]}
            return {"tag": task["tag"], "asset": asset, "phase": phase, "k": 1,
                    "status": "ok",
                    "Sh": m["Sh"], "Cal": m["Cal"],
                    "CAGR": m["CAGR"], "MDD": m["MDD"],
                    "rebal": m.get("rebal", 0)}
        else:
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
                phase_offset_bars=phase,
                canary_hyst=0.015, health_mode="mom2vol",
                stop_kind="none", stop_pct=0.0,
                drift_threshold=0.10, post_flip_delay=5,
                dd_lookback=60, dd_threshold=-0.25,
                bl_drop=-0.15, bl_days=7, crash_threshold=-0.10,
                start_date=ANCHOR, end_date=END,
            )
        return {
            "tag": task["tag"], "asset": asset, "phase": phase, "k": 1,
            "status": "ok",
            "Sh": float(m.get("Sharpe", 0) or 0),
            "Cal": float(m.get("Cal") or 0),
            "CAGR": float(m.get("CAGR", 0) or 0),
            "MDD": float(m.get("MDD", 0) or 0),
            "rebal": int(m.get("Rebal", 0) or 0),
        }
    except Exception as e:
        return {"tag": task["tag"], "asset": asset, "phase": phase, "k": 1,
                "status": "error", "error": str(e)[:200]}


def build_tasks(asset, in_csv, out_csv, resume):
    df = pd.read_csv(in_csv)
    df = df[df["k"] == 1]
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
            if {"tag", "phase"}.issubset(prev.columns):
                done = set(zip(prev["tag"].astype(str), prev["phase"].astype(int)))
    tasks = []
    for _, row in df.iterrows():
        cfg = parse_tag_row(row, asset)
        if asset != "stock" and cfg["iv"] is None:
            continue
        for phase in phase_values(cfg["snap"]):
            key = (str(row["tag"]), int(phase))
            if key in done:
                continue
            tasks.append({"tag": str(row["tag"]), "cfg": cfg,
                          "asset": asset, "phase": int(phase)})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=["fut", "spot", "stock"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--top", type=int, default=500)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--flush", type=int, default=200)
    args = ap.parse_args()

    in_csv = os.path.join(HERE, f"redesign_top{args.top}_{args.asset}_k1.csv")
    out_csv = os.path.join(HERE, f"redesign_rerank_phase_{args.asset}.csv")

    tasks = build_tasks(args.asset, in_csv, out_csv, args.resume)
    print(f"[{args.asset}] {len(tasks)} tasks", flush=True)
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
            if i % 200 == 0:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(tasks) - i) / max(rate, 1e-6)
                print(f"[{args.asset} {i}/{len(tasks)}] {rate:.2f}/s  "
                      f"ETA {eta/60:.1f}m", flush=True)
    flush_buf()
    print(f"[{args.asset} done] {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
