"""Step 3.6 — Parameter neighborhood plateau check.

Top N 후보의 numeric axis (sma/ms/ml/snap 등) 를 ±10% / ±20% 로 perturb 해
plateau 위에 있는지 peak 인지 판단.

지표: neighborhood_Cal_med, neighborhood_worst_Cal, neighborhood_p10_Cal,
      neighborhood_Cal_CV

출력: redesign_plateau_{asset}.csv
  tag, n_neighbors, neighborhood_Cal_med, neighborhood_Cal_CV,
  neighborhood_worst_Cal, neighborhood_p10_Cal, base_Cal, status
"""
from __future__ import annotations
import argparse
import math
import os
import sys
import time
from multiprocessing import Pool

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from redesign_common import parse_cfg, run_bt

COIN_AXES = ("sma", "ms", "ml", "snap")
STOCK_AXES = ("snap", "canary_sma", "def_mom")
# multi-scale percent: ±5/10/15/20 (8 points)
PERTURB = (-0.20, -0.15, -0.10, -0.05, 0.05, 0.10, 0.15, 0.20)
# absolute delta per axis (양방향 자동 -d, +d): snap 은 3 단위 (snap%3==0 유지)
ABS_DELTAS = {
    "snap": (3, 6, 9, 12),
    "sma":  (5, 10, 20, 40),
    "ms":   (2, 5, 10, 20),
    "ml":   (10, 20, 50, 100),
    "canary_sma": (10, 20, 50, 100),
    "def_mom":    (10, 20, 50, 100),
}

_DATA = None


def preload():
    global _DATA
    from unified_backtest import load_data
    _DATA = {iv: load_data(iv) for iv in ("D", "4h")}


def axes_for(asset):
    return STOCK_AXES if asset == "stock" else COIN_AXES


def perturb_cfg(cfg, axis, kind, val):
    """kind: 'pct' → val 은 multiplier (e.g. -0.10), 'abs' → val 은 절대 delta (정수, +/-).
    snap axis 는 perturbed 도 % 3 == 0 보정 (3-tranche 정합성)."""
    new = dict(cfg)
    base = cfg.get(axis)
    if base is None or not isinstance(base, (int, float)):
        return None
    if kind == "pct":
        perturbed = int(round(base * (1 + val)))
    else:
        perturbed = int(base + val)
    if axis == "snap":
        # snap %3 == 0 보장: 가장 가까운 3 배수로 round
        perturbed = int(round(perturbed / 3) * 3)
    if perturbed <= 0 or perturbed == base:
        return None
    new[axis] = perturbed
    return new


def run_one(task):
    cfg = task["cfg"]
    try:
        r = run_bt(task["asset"], cfg, bars_funding=_DATA, phase_offset=0)
        return {"tag": task["tag"], "asset": task["asset"],
                "axis": task["axis"], "kind": task["kind"], "val": task["val"],
                "mult": task["val"] if task["kind"] == "pct" else 0.0,
                "snap_p": cfg.get("snap"),
                "Cal": r.get("Cal"), "status": r.get("status"),
                "error": r.get("error", "")}
    except Exception as e:
        return {"tag": task["tag"], "asset": task["asset"],
                "axis": task["axis"], "kind": task["kind"], "val": task["val"],
                "mult": task["val"] if task["kind"] == "pct" else 0.0,
                "status": "error", "error": str(e)[:200]}


def build_tasks(asset, top_n):
    rank_csv = os.path.join(HERE, f"redesign_rank_{asset}.csv")
    if not os.path.exists(rank_csv):
        return []
    rank = pd.read_csv(rank_csv)
    # snap%3==0 hard filter (3-tranche 정합성)
    if "snap" in rank.columns:
        before = len(rank)
        rank = rank[rank["snap"].astype(int) % 3 == 0].reset_index(drop=True)
        if len(rank) < before:
            print(f"[{asset}] snap%3 filter: {before} → {len(rank)}", flush=True)
    rank = rank.head(top_n)
    out_csv = os.path.join(HERE, f"redesign_plateau_{asset}.csv")
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
            if {"tag", "axis", "kind", "val"}.issubset(prev.columns):
                done = set(zip(prev["tag"].astype(str),
                               prev["axis"].astype(str),
                               prev["kind"].astype(str),
                               prev["val"].astype(float).round(3)))
    tasks = []
    axes = axes_for(asset)
    for _, row in rank.iterrows():
        cfg = parse_cfg(asset, row)
        if cfg is None:
            continue
        tag = str(row["tag"])
        for axis in axes:
            # percent perturbations
            for mult in PERTURB:
                key = (tag, axis, "pct", round(float(mult), 3))
                if key in done:
                    continue
                new_cfg = perturb_cfg(cfg, axis, "pct", mult)
                if new_cfg is None:
                    continue
                tasks.append({"tag": tag, "asset": asset, "axis": axis,
                              "kind": "pct", "val": float(mult), "cfg": new_cfg})
            # absolute deltas (양방향)
            for d in ABS_DELTAS.get(axis, ()):
                for sign in (-1, 1):
                    val = sign * d
                    key = (tag, axis, "abs", round(float(val), 3))
                    if key in done:
                        continue
                    new_cfg = perturb_cfg(cfg, axis, "abs", val)
                    if new_cfg is None:
                        continue
                    tasks.append({"tag": tag, "asset": asset, "axis": axis,
                                  "kind": "abs", "val": float(val), "cfg": new_cfg})
    return tasks


def aggregate(asset):
    raw_csv = os.path.join(HERE, f"redesign_plateau_{asset}.csv")
    if not os.path.exists(raw_csv):
        return pd.DataFrame()
    df = pd.read_csv(raw_csv)
    if df.empty or "status" not in df.columns:
        return pd.DataFrame()
    df = df[df["status"] == "ok"].dropna(subset=["Cal"]).copy()
    # (tag, axis, kind, val) 중복 방지 (concurrent write 대비)
    if {"axis", "kind", "val"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["tag", "axis", "kind", "val"], keep="first")
    elif {"axis", "mult"}.issubset(df.columns):
        df = df.drop_duplicates(subset=["tag", "axis", "mult"], keep="first")
    if df.empty:
        return pd.DataFrame()
    rank_csv = os.path.join(HERE, f"redesign_rank_{asset}.csv")
    base_map = {}
    if os.path.exists(rank_csv):
        rk = pd.read_csv(rank_csv)
        if "tag" in rk.columns and "Cal" in rk.columns:
            base_map = dict(zip(rk["tag"].astype(str), rk["Cal"].astype(float)))
    rows = []
    for tag, g in df.groupby("tag"):
        cal = g["Cal"]
        rows.append({
            "tag": tag,
            "n_neighbors": int(len(cal)),
            "neighborhood_Cal_med": float(cal.median()),
            "neighborhood_worst_Cal": float(cal.min()),
            "neighborhood_p10_Cal": float(np.percentile(cal, 10)),
            "neighborhood_Cal_CV": float(cal.std() / abs(cal.mean())) if abs(cal.mean()) > 1e-9 else 99.0,
            "base_Cal": base_map.get(tag),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=["fut", "spot", "stock"])
    ap.add_argument("--top", type=int, default=25)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    tasks = build_tasks(args.asset, args.top)
    print(f"[{args.asset}] {len(tasks)} plateau tasks", flush=True)
    out_csv = os.path.join(HERE, f"redesign_plateau_{args.asset}.csv")
    if tasks:
        t0 = time.time()
        buf = []
        header_written = os.path.exists(out_csv)
        def flush():
            nonlocal buf, header_written
            if not buf: return
            pd.DataFrame(buf).to_csv(out_csv, mode="a",
                                      header=not header_written, index=False)
            header_written = True
            buf = []
        with Pool(args.workers, initializer=preload) as pool:
            for i, res in enumerate(pool.imap_unordered(run_one, tasks, chunksize=2), 1):
                buf.append(res)
                if len(buf) >= 50:
                    flush()
                if i % 100 == 0:
                    print(f"[{args.asset} {i}/{len(tasks)}]", flush=True)
        flush()
        print(f"[{args.asset} plateau raw done] {(time.time()-t0)/60:.1f}m")

    # aggregate
    agg = aggregate(args.asset)
    agg_csv = os.path.join(HERE, f"redesign_plateau_agg_{args.asset}.csv")
    agg.to_csv(agg_csv, index=False)
    print(f"[{args.asset}] wrote {agg_csv} ({len(agg)} rows)")


if __name__ == "__main__":
    main()
