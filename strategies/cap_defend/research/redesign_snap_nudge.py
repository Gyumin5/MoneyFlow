"""Step 3.5 — Snap cadence nudge (soft gate).

각 config 의 base snap 근방에서 prime×3 값 3~5개로 백테스트.
범위: base_snap ± max(round(0.10 × snap), 3) bars

출력: redesign_snap_nudge_{asset}.csv
  tag, base_snap, nudge_snap, phase=0, Cal, ..., nudge_CV (post-hoc)

Soft gate: 최종 rank 단계에서 CV 가 자산별 p70 이상이면 rank penalty +10.
Hard reject 아님.

사용: python redesign_snap_nudge.py --asset fut --workers 24
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

# 단일 소스: redesign_common.is_prime_x3_snap() 로 생성
from redesign_common import generate_prime_x3_list, parse_cfg
PRIME_X3 = generate_prime_x3_list(max_snap=600)

ANCHOR = "2020-10-01"
END = "2026-04-13"
TX_BY_ASSET = {"fut": 0.0004, "spot": 0.004, "stock": 0.0025}

_DATA = None


def preload():
    global _DATA
    from unified_backtest import load_data
    _DATA = {iv: load_data(iv) for iv in ("D", "4h")}


def nudge_candidates(base_snap):
    """base_snap 주변 prime×3 3~5개 선택.
    기본 범위: max(round(0.10*snap), 3) bars
    3개 미만이면 거리 순으로 확장해서 보장.
    """
    band = max(round(0.10 * base_snap), 3)
    # 가까운 거리 순으로 정렬된 prime×3 후보
    by_dist = sorted(
        (s for s in PRIME_X3 if s != base_snap),
        key=lambda s: abs(s - base_snap)
    )
    # 1차: 기본 band 내
    primary = [s for s in by_dist if abs(s - base_snap) <= band]
    if len(primary) >= 3:
        return sorted(primary[:5])
    # 2차: 거리 순 확장해 최소 3개 보장
    cands = list(primary)
    for s in by_dist:
        if s in cands:
            continue
        cands.append(s)
        if len(cands) >= 5:
            break
    return sorted(cands[:5])


def run_one(task):
    cfg = task["cfg"]
    asset = task["asset"]
    snap = task["nudge_snap"]
    tx = TX_BY_ASSET[asset]
    try:
        if asset == "stock":
            from redesign_stock_adapter import run_stock_from_cfg
            cfg_nudge = dict(cfg, snap=snap)
            m = run_stock_from_cfg(cfg_nudge, phase_offset=0, tx_cost=tx)
            if m.get("status") != "ok":
                return {"tag": task["tag"], "asset": asset,
                        "base_snap": task["base_snap"], "nudge_snap": snap,
                        "status": "error", "error": m.get("error", "")[:200]}
            return {"tag": task["tag"], "asset": asset,
                    "base_snap": task["base_snap"], "nudge_snap": snap,
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
            snap_interval_bars=snap, n_snapshots=3,
            phase_offset_bars=0,
            canary_hyst=0.015, health_mode="mom2vol",
            stop_kind="none", stop_pct=0.0,
            drift_threshold=0.10, post_flip_delay=5,
            dd_lookback=60, dd_threshold=-0.25,
            bl_drop=-0.15, bl_days=7, crash_threshold=-0.10,
            start_date=ANCHOR, end_date=END,
        )
        return {
            "tag": task["tag"], "asset": asset,
            "base_snap": task["base_snap"], "nudge_snap": snap,
            "status": "ok",
            "Cal": float(m.get("Cal") or 0),
            "Sh": float(m.get("Sharpe", 0) or 0),
            "CAGR": float(m.get("CAGR", 0) or 0),
            "MDD": float(m.get("MDD", 0) or 0),
        }
    except Exception as e:
        return {"tag": task["tag"], "asset": asset,
                "base_snap": task["base_snap"], "nudge_snap": snap,
                "status": "error", "error": str(e)[:200]}


def build_tasks(asset, phase_csv, out_csv, resume):
    """Step 3 phase sweep survivor 기반 nudge tasks 생성.
    명시 survivor CSV (redesign_filter_phase.py 출력) 우선 사용.
    없을 때만 raw phase csv 로 median>0 fallback.
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
            if {"tag", "nudge_snap"}.issubset(prev.columns):
                done = set(zip(prev["tag"].astype(str), prev["nudge_snap"].astype(int)))

    tasks = []
    for _, row in base.iterrows():
        tag = str(row["tag"])
        cfg = parse_cfg(asset, row)
        if cfg is None:
            continue
        base_snap = int(cfg["snap"])
        if base_snap <= 0:
            continue
        for ns in nudge_candidates(base_snap):
            if (tag, int(ns)) in done:
                continue
            tasks.append({"tag": tag, "asset": asset, "cfg": cfg,
                          "base_snap": base_snap, "nudge_snap": int(ns)})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=["fut", "spot", "stock"])
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--flush", type=int, default=200)
    args = ap.parse_args()

    phase_csv = os.path.join(HERE, f"redesign_rerank_phase_{args.asset}.csv")
    out_csv = os.path.join(HERE, f"redesign_snap_nudge_{args.asset}.csv")

    tasks = build_tasks(args.asset, phase_csv, out_csv, args.resume)
    print(f"[{args.asset}] {len(tasks)} nudge tasks", flush=True)
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
                print(f"[{args.asset} {i}/{len(tasks)}] {rate:.2f}/s ETA {eta/60:.1f}m",
                      flush=True)
    flush_buf()
    print(f"[{args.asset} done] {(time.time()-t0)/60:.1f}m")


if __name__ == "__main__":
    main()
