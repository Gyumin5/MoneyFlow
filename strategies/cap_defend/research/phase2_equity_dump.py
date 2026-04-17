#!/usr/bin/env python3
"""Phase-2 equity dump (plateau 전용, OOS 없음).

각 plateau survivor를 단일 앵커 2020-10-01에서 full-end까지 실행하여
daily equity csv.gz 저장. phase3 correlation 게이트 입력으로 사용.

Input : phase2_extract_v2/survivors.csv (plateau_ok=True)
Output: phase2_equity_dump/
        ├─ raw.csv
        ├─ summary.csv (oos_pass 없음, plateau_ok=True 전부 포함)
        └─ equity/{tag}__00.csv.gz (2020-10-01 ~ FULL_END, 252D 초과 저장)
"""
from __future__ import annotations

import argparse
import gc
import os
import sys
import time
from multiprocessing import Pool

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from phase_common import (
    FULL_END, RAW_COLUMNS, atomic_append_csv, atomic_write_csv,
    emit_daily_equity, run_single_target, summarize_phase_raw, write_manifest,
)

OUT_DIR = os.path.join(HERE, "phase2_equity_dump")
ANCHORS = ["2020-10-01"]
CHECKPOINT_EVERY = 200
SUMMARY_EVERY = 1000
HORIZON_DAYS = 2500  # 전체 기간 보존


def _run_task(task: dict) -> dict:
    tag = task["tag"]
    cfg = task["cfg"]
    asset = task["asset"]
    lev = float(task["lev"])
    anchor = task["anchor"]
    anchor_idx = task["anchor_idx"]
    end = task["end"]
    equity_dir = task["equity_dir"]
    try:
        m = run_single_target(asset, cfg, lev, anchor, end=end, want_equity=True)
        eq = m.pop("_equity", None)
        if eq is not None:
            out_path = os.path.join(equity_dir, f"{tag}__{anchor_idx:02d}.csv.gz")
            emit_daily_equity(eq, out_path, horizon_days=HORIZON_DAYS)
        return {"tag": tag, "anchor": anchor, "asset": asset, "lev": lev,
                **{k: m.get(k) for k in ("Sh", "Cal", "CAGR", "MDD",
                                          "CVaR5", "Ulcer", "TUW", "rebal", "liq")},
                "error": ""}
    except Exception as e:
        return {"tag": tag, "anchor": anchor, "asset": asset, "lev": lev,
                "error": str(e)[:200]}


def _row_to_cfg(row: dict) -> tuple[str, dict, float]:
    cfg = {
        "interval": row["interval"],
        "sma": int(row["sma"]),
        "ms": int(row["ms"]),
        "ml": int(row["ml"]),
        "vol_mode": row["vol_mode"],
        "vol_thr": float(row["vol_thr"]),
        "snap": int(row["snap"]),
    }
    return row["asset"], cfg, float(row["lev"])


def build_tasks(surv: pd.DataFrame, equity_dir: str) -> list[dict]:
    tasks = []
    for _, r in surv.iterrows():
        asset, cfg, lev = _row_to_cfg(r.to_dict())
        tag = r["tag"]
        for idx, anchor in enumerate(ANCHORS):
            tasks.append({
                "tag": tag, "cfg": cfg, "asset": asset, "lev": lev,
                "anchor": anchor, "anchor_idx": idx, "end": FULL_END,
                "equity_dir": equity_dir,
            })
    return tasks


def load_completed_keys(raw_path: str) -> set[tuple[str, str]]:
    if not os.path.exists(raw_path) or os.path.getsize(raw_path) == 0:
        return set()
    try:
        df = pd.read_csv(raw_path, on_bad_lines="skip")
    except Exception:
        return set()
    if df.empty:
        return set()
    if "error" in df.columns:
        df = df[df["error"].fillna("") == ""]
    keys = df[["tag", "anchor"]].dropna().drop_duplicates()
    return set(map(tuple, keys.itertuples(index=False, name=None)))


def attach_plateau_ok(summary_df: pd.DataFrame, surv: pd.DataFrame) -> pd.DataFrame:
    """phase3가 plateau_ok=True 만 뽑도록 True 표시. phase1_mCal도 끼워 맞춤."""
    phase1_cal = surv.set_index("tag")["mCal"].to_dict()
    out = summary_df.copy()
    out["phase1_mCal"] = out["tag"].map(phase1_cal)
    out["plateau_ok"] = True
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--survivors", required=True)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--processes", type=int, default=24)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    equity_dir = os.path.join(args.out_dir, "equity")
    os.makedirs(equity_dir, exist_ok=True)
    raw_path = os.path.join(args.out_dir, "raw.csv")
    summary_path = os.path.join(args.out_dir, "summary.csv")
    manifest_path = os.path.join(args.out_dir, "manifest.json")

    surv = pd.read_csv(args.survivors)
    all_tasks = build_tasks(surv, equity_dir)
    completed = load_completed_keys(raw_path)
    tasks = [t for t in all_tasks if (t["tag"], t["anchor"]) not in completed]

    write_manifest(manifest_path, {
        "status": "running", "stage": "phase2_equity_dump",
        "total_tasks": len(all_tasks),
        "done_tasks": len(completed),
    })
    print(f"Phase-2 equity dump: tasks={len(all_tasks)}, resume done={len(completed)}, "
          f"remaining={len(tasks)}")

    if not tasks:
        df = pd.read_csv(raw_path, on_bad_lines="skip") if os.path.exists(raw_path) else pd.DataFrame()
        sdf = summarize_phase_raw(df) if not df.empty else pd.DataFrame()
        if not sdf.empty:
            sdf = attach_plateau_ok(sdf, surv)
            atomic_write_csv(sdf, summary_path)
        write_manifest(manifest_path, {
            "status": "done", "stage": "phase2_equity_dump",
            "total_tasks": len(all_tasks),
            "done_tasks": len(all_tasks),
        })
        return

    t0 = time.time()
    pending: list[dict] = []
    with Pool(processes=args.processes) as pool:
        for i, res in enumerate(pool.imap_unordered(_run_task, tasks, chunksize=1), 1):
            pending.append(res)
            if i % CHECKPOINT_EVERY == 0 or i == len(tasks):
                atomic_append_csv(raw_path, pending, RAW_COLUMNS)
                pending.clear()
                write_manifest(manifest_path, {
                    "status": "running", "stage": "phase2_equity_dump",
                    "total_tasks": len(all_tasks),
                    "done_tasks": len(completed) + i,
                })
                gc.collect()
                if i % SUMMARY_EVERY == 0 or i == len(tasks):
                    df = pd.read_csv(raw_path, on_bad_lines="skip")
                    sdf = summarize_phase_raw(df)
                    if not sdf.empty:
                        sdf = attach_plateau_ok(sdf, surv)
                        atomic_write_csv(sdf, summary_path)
                el = int(time.time() - t0)
                eta = int(el / i * (len(tasks) - i)) if i else 0
                print(f"  [{len(completed)+i}/{len(all_tasks)}] elapsed={el}s eta={eta}s")

    df = pd.read_csv(raw_path, on_bad_lines="skip")
    sdf = summarize_phase_raw(df)
    sdf = attach_plateau_ok(sdf, surv)
    atomic_write_csv(sdf, summary_path)
    write_manifest(manifest_path, {
        "status": "done", "stage": "phase2_equity_dump",
        "total_tasks": len(all_tasks),
        "done_tasks": len(all_tasks),
        "n_survivors": int(len(sdf)),
    })
    print(f"Done. survivors={len(sdf)}")


if __name__ == "__main__":
    main()
