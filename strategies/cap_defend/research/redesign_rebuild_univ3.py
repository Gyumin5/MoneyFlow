"""iter_refine raw_combined 를 universe_size=3 으로 재평가.

출력: redesign_univ3_raw.csv  (asset,iv,lev,... + Sh/Cal/CAGR/MDD/rebal)
이후 redesign_univ3_top200.py 가 asset 별 top 200 추출.

병렬 24 worker. Resume 지원 (tag 단위).
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

IN_CSV = os.path.join(HERE, "iter_refine/raw_combined.csv")
OUT_CSV = os.path.join(HERE, "redesign_univ3_raw.csv")

ANCHOR = "2020-10-01"
END = "2026-04-13"

_DATA = None


def preload():
    global _DATA
    from unified_backtest import load_data
    _DATA = {iv: load_data(iv) for iv in ("D", "4h")}


def run_one(task):
    from unified_backtest import run as bt_run
    cfg = task
    iv = cfg["iv"]
    bars, funding = _DATA[iv]
    tx = 0.004 if cfg["asset"] == "spot" else 0.0004
    lev = 1.0 if cfg["asset"] == "spot" else float(cfg["lev"])
    try:
        m = bt_run(
            bars, funding,
            interval=iv, asset_type=cfg["asset"],
            leverage=lev, universe_size=3, cap=1 / 3,
            tx_cost=tx,
            sma_bars=int(cfg["sma"]),
            mom_short_bars=int(cfg["ms"]),
            mom_long_bars=int(cfg["ml"]),
            vol_mode=str(cfg["vmode"]),
            vol_threshold=float(cfg["vthr"]),
            snap_interval_bars=int(cfg["snap"]),
            n_snapshots=3,
            phase_offset_bars=0,
            canary_hyst=0.015,
            health_mode="mom2vol",
            stop_kind="none", stop_pct=0.0,
            drift_threshold=0.10, post_flip_delay=5,
            dd_lookback=60, dd_threshold=-0.25,
            bl_drop=-0.15, bl_days=7, crash_threshold=-0.10,
            start_date=ANCHOR, end_date=END,
        )
        return {
            "tag": cfg["tag"], "asset": cfg["asset"], "iv": iv, "lev": cfg["lev"],
            "sma": cfg["sma"], "ms": cfg["ms"], "ml": cfg["ml"],
            "vmode": cfg["vmode"], "vthr": cfg["vthr"], "snap": cfg["snap"],
            "status": "ok",
            "Sh": float(m.get("Sharpe", 0) or 0),
            "Cal": float(m.get("Cal") or 0),
            "CAGR": float(m.get("CAGR", 0) or 0),
            "MDD": float(m.get("MDD", 0) or 0),
            "rebal": int(m.get("Rebal", 0) or 0),
            "liq": int(m.get("Liq", 0) or 0),
        }
    except Exception as e:
        return {"tag": cfg["tag"], "asset": cfg["asset"], "iv": iv, "lev": cfg["lev"],
                "sma": cfg["sma"], "ms": cfg["ms"], "ml": cfg["ml"],
                "vmode": cfg["vmode"], "vthr": cfg["vthr"], "snap": cfg["snap"],
                "status": "error", "error": str(e)[:200]}


def build_tasks(args):
    df = pd.read_csv(IN_CSV)
    df = df.dropna(subset=["Cal"]).copy()
    # Unique configs per tag
    df = df.drop_duplicates(subset=["tag"])
    if args.limit:
        df = df.head(args.limit)
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
            if "tag" in prev.columns:
                done = set(prev["tag"].astype(str))
    tasks = []
    for _, row in df.iterrows():
        tag = str(row["tag"])
        if tag in done:
            continue
        tasks.append({
            "tag": tag, "asset": row["asset"], "iv": row["iv"], "lev": row["lev"],
            "sma": row["sma"], "ms": row["ms"], "ml": row["ml"],
            "vmode": row["vmode"], "vthr": row["vthr"], "snap": row["snap"],
        })
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--flush", type=int, default=500,
                    help="이 건 단위로 CSV append")
    args = ap.parse_args()

    tasks = build_tasks(args)
    print(f"[build] {len(tasks)} tasks", flush=True)
    if not tasks:
        return
    t0 = time.time()

    # 기존 csv 가 있으면 header 스킵, 없으면 write header
    header_written = os.path.exists(OUT_CSV)
    buf = []

    def flush_buf():
        nonlocal header_written, buf
        if not buf:
            return
        df_ = pd.DataFrame(buf)
        df_.to_csv(OUT_CSV, mode="a", header=not header_written, index=False)
        header_written = True
        buf = []

    with Pool(args.workers, initializer=preload) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, tasks, chunksize=4), 1):
            buf.append(res)
            if len(buf) >= args.flush:
                flush_buf()
            if i % 500 == 0:
                elapsed = time.time() - t0
                rate = i / max(elapsed, 1e-6)
                eta = (len(tasks) - i) / max(rate, 1e-6)
                print(f"[{i}/{len(tasks)}] {rate:.2f}/s  ETA {eta/3600:.2f}h  "
                      f"elapsed {elapsed/60:.1f}m", flush=True)
    flush_buf()
    print(f"[done] {(time.time()-t0)/60:.1f}m total")


if __name__ == "__main__":
    main()
