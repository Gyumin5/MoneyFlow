"""Step 7 — Final stress test (Top 20~30 후보만).

시나리오:
  - tx_1.5x: tx_cost × 1.5
  - tx_2.0x: tx_cost × 2.0
  - delay_1bar: 체결 시점 +1 bar (TODO: unified_backtest 지원 추가)
  - drop_top_contributor: 기여도 1위 자산 제외 (TODO: engine wrapper)

출력: redesign_stress_{asset}.csv
  tag_or_members, scenario, Cal, CAGR, MDD, Cal_decay, MDD_decay, pass, status, error
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

from redesign_common import parse_cfg, run_bt

# 3자산 동일 5 시나리오 (stock_engine_snap 업그레이드로 exec_delay + exclude_assets 지원)
SCENARIOS = [
    {"name": "baseline", "tx_mult": 1.0, "exec_delay_bars": 0, "drop_top": False},
    {"name": "tx_1.5x", "tx_mult": 1.5, "exec_delay_bars": 0, "drop_top": False},
    {"name": "tx_2.0x", "tx_mult": 2.0, "exec_delay_bars": 0, "drop_top": False},
    {"name": "delay_1bar", "tx_mult": 1.0, "exec_delay_bars": 1, "drop_top": False},
    {"name": "drop_top", "tx_mult": 1.0, "exec_delay_bars": 0, "drop_top": True},
]


def scenarios_for(asset):
    return SCENARIOS

_DATA = None


def preload():
    global _DATA
    from unified_backtest import load_data
    _DATA = {iv: load_data(iv) for iv in ("D", "4h")}


def run_one(task):
    scn = task["scenario"]
    cfg = task["cfg"]
    asset = task["asset"]
    r = run_bt(
        asset, cfg,
        bars_funding=_DATA,
        phase_offset=0,
        tx_mult=scn["tx_mult"],
        exec_delay_bars=scn["exec_delay_bars"],
        drop_top_contributor=scn["drop_top"],
    )
    return {
        "tag": task["tag"], "asset": asset, "scenario": scn["name"],
        **{k: r.get(k) for k in ("status", "error", "Cal", "CAGR", "MDD", "Sh")},
    }


def build_tasks(asset, rank_top_n=25):
    rank_csv = os.path.join(HERE, f"redesign_rank_{asset}.csv")
    if not os.path.exists(rank_csv):
        print(f"[{asset}] rank csv missing: {rank_csv}")
        return []
    rank = pd.read_csv(rank_csv).head(rank_top_n)
    out_csv = os.path.join(HERE, f"redesign_stress_{asset}.csv")
    done = set()
    if os.path.exists(out_csv):
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
            if {"tag", "scenario"}.issubset(prev.columns):
                done = set(zip(prev["tag"].astype(str), prev["scenario"].astype(str)))
    tasks = []
    for _, row in rank.iterrows():
        cfg = parse_cfg(asset, row)
        if cfg is None:
            continue
        for scn in scenarios_for(asset):
            if (str(row["tag"]), scn["name"]) in done:
                continue
            tasks.append({"tag": str(row["tag"]), "asset": asset,
                          "cfg": cfg, "scenario": scn})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=["fut", "spot", "stock"])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    tasks = build_tasks(args.asset, args.top)
    print(f"[{args.asset}] {len(tasks)} stress tasks", flush=True)
    if not tasks:
        return
    out_csv = os.path.join(HERE, f"redesign_stress_{args.asset}.csv")
    t0 = time.time()
    buf = []
    header_written = os.path.exists(out_csv)

    def flush():
        nonlocal header_written, buf
        if not buf:
            return
        pd.DataFrame(buf).to_csv(out_csv, mode="a", header=not header_written, index=False)
        header_written = True
        buf = []

    with Pool(args.workers, initializer=preload) as pool:
        for i, res in enumerate(pool.imap_unordered(run_one, tasks, chunksize=2), 1):
            buf.append(res)
            if len(buf) >= 50:
                flush()
            if i % 50 == 0:
                print(f"[{args.asset} {i}/{len(tasks)}]", flush=True)
    flush()
    print(f"[{args.asset} stress done] {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
