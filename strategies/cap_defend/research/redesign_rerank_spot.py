"""Tier 1 재랭크 — 현물 (k=1 single strategy, phase sweep).

입력: redesign_top200_spot.csv (k=1 only in this skeleton)
출력: redesign_rerank_spot_tier1.csv

unified_backtest.run(asset_type='spot', phase_offset_bars=phase) 사용.
phase_offset_bars 는 이 스크립트 작성 시 unified_backtest.py 에 추가 (봉 기반 앵커).

Tier 2 (start/yearly) 는 CV 필터 후 별도 실행.
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

IN_CSV = os.path.join(HERE, "redesign_top200_spot.csv")
OUT_CSV = os.path.join(HERE, "redesign_rerank_spot_tier1.csv")

ANCHOR = "2020-10-01"
END = "2026-04-13"

_DATA = None


def preload():
    global _DATA
    from unified_backtest import load_data
    _DATA = {iv: load_data(iv) for iv in ("D", "4h")}


def parse_tag_row(row):
    tag = str(row["tag"])
    iv = "D" if "_1D_" in tag else ("4h" if "_4h_" in tag else None)
    vm = "daily" if "_d" in tag else "bar"
    return {
        "iv": iv,
        "sma": int(row["sma"]),
        "ms": int(row["ms"]),
        "ml": int(row["ml"]),
        "vmode": vm,
        "vthr": float(row["vthr"]),
        "snap": int(row["snap"]),
    }


def run_one(task):
    from unified_backtest import run as bt_run
    cfg = task["cfg"]
    phase = task["phase"]
    iv = cfg["iv"]
    bars, funding = _DATA[iv]
    try:
        m = bt_run(
            bars, funding,
            interval=iv,
            asset_type="spot",
            leverage=1.0,
            universe_size=5, cap=1 / 3,  # iter_refine 일치 (V21 univ=3 과 다름)
            tx_cost=0.004,
            sma_bars=cfg["sma"],
            mom_short_bars=cfg["ms"],
            mom_long_bars=cfg["ml"],
            vol_mode=cfg["vmode"],
            vol_threshold=cfg["vthr"],
            snap_interval_bars=cfg["snap"],
            phase_offset_bars=phase,
            n_snapshots=3,
            canary_hyst=0.015,
            health_mode="mom2vol",
            start_date=ANCHOR, end_date=END,
        )
        return {
            "tag": task["tag"], "phase": phase, "k": task["k"],
            "status": "ok",
            "Sh": float(m.get("Sharpe", 0)),
            "Cal": float(m.get("Cal") or 0),
            "CAGR": float(m.get("CAGR", 0)),
            "MDD": float(m.get("MDD", 0)),
            "rebal": int(m.get("Rebal", 0)),
        }
    except Exception as e:
        return {"tag": task["tag"], "phase": phase, "k": task["k"],
                "status": "error", "error": str(e)[:200]}


def build_tasks(args):
    df = pd.read_csv(IN_CSV)
    df = df[df["k"] == 1]
    if args.top:
        df = df.head(args.top)
    done = set()
    if os.path.exists(OUT_CSV) and args.resume:
        try:
            prev = pd.read_csv(OUT_CSV)
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
        cfg = parse_tag_row(row)
        if cfg["iv"] is None:
            continue
        for phase in args.phases:
            key = (str(row["tag"]), int(phase))
            if key in done:
                continue
            tasks.append({"tag": str(row["tag"]), "cfg": cfg,
                           "phase": int(phase), "k": 1})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--phases", type=str, default="0,13,31,59")
    ap.add_argument("--top", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    args = ap.parse_args()
    args.phases = [int(x) for x in args.phases.split(",")]

    tasks = build_tasks(args)
    print(f"[build] {len(tasks)} tasks (phases={args.phases})", flush=True)
    if not tasks:
        return
    t0 = time.time()
    rows = []
    with Pool(args.workers, initializer=preload) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, tasks, chunksize=1), 1):
            rows.append(res)
            if i % 20 == 0:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(tasks) - i) / max(rate, 1e-6)
                print(f"[{i}/{len(tasks)}] {rate:.2f} /s ETA {eta/60:.1f}m",
                      flush=True)
    df_new = pd.DataFrame(rows)
    if os.path.exists(OUT_CSV) and args.resume:
        df_old = pd.read_csv(OUT_CSV)
        df_new = pd.concat([df_old, df_new], ignore_index=True)
    df_new.to_csv(OUT_CSV, index=False)
    print(f"[done] wrote {OUT_CSV} ({len(df_new)} rows) in "
          f"{(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
