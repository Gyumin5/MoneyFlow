#!/usr/bin/env python3
"""iter_refine/raw_combined.csv → phase1_sweep/{summary.csv, raw.csv, manifest.json}.

iter_refine 단일 앵커 결과를 기존 phase2_extract.py가 읽는 스키마로 변환.
"""
from __future__ import annotations
import argparse
import os
import sys

import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from phase_common import atomic_write_csv, write_manifest

ANCHOR = "2020-10-01"


def convert(raw_combined: str, out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)
    df = pd.read_csv(raw_combined)
    # error 걸러서 summary 구성
    if "error" in df.columns:
        err = df["error"].fillna("").astype(str)
        mask = err == ""
    else:
        mask = pd.Series([True] * len(df), index=df.index)
    good = df[mask].copy()
    good["interval"] = good["iv"].astype(str)
    out = pd.DataFrame({
        "tag": good["tag"],
        "asset": good["asset"],
        "lev": good["lev"].astype(float),
        "interval": good["interval"],
        "sma": good["sma"].astype(int),
        "ms": good["ms"].astype(int),
        "ml": good["ml"].astype(int),
        "snap": good["snap"].astype(int),
        "vol_mode": good["vmode"],
        "vol_thr": good["vthr"].astype(float),
        "mCal": good["Cal"].astype(float),
        "mSh": good["Sh"].astype(float),
        "mCAGR": good["CAGR"].astype(float),
        "wMDD": good["MDD"].astype(float),
        "n": 1,
        "rebal_mean": good["rebal"].astype(float) if "rebal" in good.columns else 0.0,
        "liq_sum": good["liq"].fillna(0).astype(int) if "liq" in good.columns else 0,
    })
    atomic_write_csv(out, os.path.join(out_dir, "summary.csv"))

    anchor_col = df["anchor"] if "anchor" in df.columns else ANCHOR
    raw_out = pd.DataFrame({
        "tag": df["tag"],
        "anchor": anchor_col,
        "asset": df["asset"],
        "lev": df["lev"].astype(float),
        "Sh": df.get("Sh", 0),
        "Cal": df.get("Cal", 0),
        "CAGR": df.get("CAGR", 0),
        "MDD": df.get("MDD", 0),
        "rebal": df.get("rebal", 0),
        "liq": df.get("liq", 0),
        "error": df.get("error", ""),
    })
    atomic_write_csv(raw_out, os.path.join(out_dir, "raw.csv"))

    write_manifest(os.path.join(out_dir, "manifest.json"), {
        "status": "done",
        "source": raw_combined,
        "total_tasks": int(len(df)),
        "done_tasks": int(len(df)),
        "summary_rows": int(len(out)),
    })
    print(f"bridge: summary={len(out)} raw={len(df)} → {out_dir}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw-combined", default=os.path.join(HERE, "iter_refine", "raw_combined.csv"))
    ap.add_argument("--out-dir", default=os.path.join(HERE, "phase1_sweep"))
    args = ap.parse_args()
    convert(args.raw_combined, args.out_dir)


if __name__ == "__main__":
    main()
