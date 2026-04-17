#!/usr/bin/env python3
"""Plateau robustness check — post iter_refine convergence.

각 (asset, iv, lev) 버킷별 상위 N config 에 대해 sma/ms/ml/snap 4축 ±5/±10% perturbation 실행.
통과 기준: min(perturbed_Cal) / center_Cal >= RATIO.

입력: iter_refine/raw_combined.csv (모든 stage 합본)
출력: plateau_check/{raw.csv, survivors.csv, summary.csv}
"""
from __future__ import annotations
import argparse
import math
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

from iter_refine import _params, _run_task, preload, ANCHOR, FULL_END, cfg_id  # noqa: E402

PERTURB_PCTS = (-0.10, -0.05, 0.05, 0.10)
PERTURB_AXES = ("sma", "ms", "ml", "snap")


def _add_rank_sum(df: pd.DataFrame) -> pd.DataFrame:
    def _rk(col, asc=False):
        return col.rank(ascending=asc, method="min")
    df = df.copy()
    df["rank_sum"] = (
        _rk(df["Cal"].astype(float)) + _rk(df["Sh"].astype(float))
        + _rk(df["CAGR"].astype(float)) + _rk(df["MDD"].astype(float))
    )
    return df


def pick_top_per_bucket(df: pd.DataFrame, top_n: int) -> pd.DataFrame:
    if "error" in df.columns:
        df = df[df["error"].fillna("").astype(str) == ""].copy()
    df = df.dropna(subset=["Cal", "Sh", "CAGR", "MDD"])
    df = _add_rank_sum(df)
    out = []
    for (asset, iv, lev), g in df.groupby(["asset", "iv", "lev"]):
        g2 = g.sort_values("rank_sum").head(top_n)
        out.append(g2)
    return pd.concat(out, ignore_index=True) if out else df.iloc[0:0]


def build_perturbations(row: dict) -> list[dict]:
    """row → list of perturbation tasks (center + 4 축 × 4 perturb)."""
    base_cfg = {
        "iv": row["iv"],
        "sma": int(row["sma"]),
        "ms": int(row["ms"]),
        "ml": int(row["ml"]),
        "snap": int(row["snap"]),
        "vmode": row["vmode"],
        "vthr": float(row["vthr"]),
    }
    asset = row["asset"]
    lev = float(row["lev"])
    center_tag = f"PCHK_CENTER_{cfg_id(base_cfg, asset, lev)}"
    tasks = [{"tag": center_tag, "cfg": base_cfg, "asset": asset, "lev": lev,
              "parent": center_tag, "axis": "center", "pct": 0.0}]
    for axis in PERTURB_AXES:
        for pct in PERTURB_PCTS:
            v = base_cfg[axis]
            new_v = int(round(v * (1 + pct)))
            if new_v <= 0:
                continue
            if new_v == v:
                continue
            cfg2 = dict(base_cfg)
            cfg2[axis] = new_v
            # 구조 제약 검증: mom_short < mom_long, sma > 0
            if cfg2["ms"] >= cfg2["ml"]:
                continue
            if cfg2["sma"] <= 0 or cfg2["snap"] <= 0:
                continue
            tag = f"PCHK_{axis}{pct:+.2f}_{cfg_id(cfg2, asset, lev)}"
            tasks.append({"tag": tag, "cfg": cfg2, "asset": asset, "lev": lev,
                          "parent": center_tag, "axis": axis, "pct": pct})
    return tasks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True,
                    help="iter_refine/raw_combined.csv path")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-n", type=int, default=100,
                    help="버킷별 rank_sum 상위 N (default 100)")
    ap.add_argument("--workers", type=int, default=24)
    ap.add_argument("--pass-ratio", type=float, default=0.85,
                    help="min(perturbed_Cal)/center_Cal 통과 기준")
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    df = pd.read_csv(args.raw)
    top = pick_top_per_bucket(df, args.top_n)
    print(f"[plateau] top per bucket: {len(top)} configs "
          f"(buckets={top.groupby(['asset','iv','lev']).ngroups})", flush=True)

    all_tasks = []
    center_map = {}
    for _, row in top.iterrows():
        tasks = build_perturbations(row.to_dict())
        center_tag = tasks[0]["tag"]
        center_map[center_tag] = row.to_dict()
        all_tasks.extend(tasks)
    print(f"[plateau] total tasks: {len(all_tasks)} "
          f"(center + 16 perturbations per config)", flush=True)

    results = []
    t0 = time.time()
    with Pool(args.workers, initializer=preload) as pool:
        for i, r in enumerate(pool.imap_unordered(_run_task, all_tasks,
                                                    chunksize=4), 1):
            results.append(r)
            if i % 500 == 0 or i == len(all_tasks):
                el = time.time() - t0
                rate = i / max(el, 1e-6)
                eta = (len(all_tasks) - i) / max(rate, 1e-6)
                print(f"  [{i}/{len(all_tasks)}] {rate:.1f}/s eta={eta/60:.1f}m",
                      flush=True)

    # task tag → parent 역매핑
    tag_to_meta = {t["tag"]: t for t in all_tasks}
    raw_rows = []
    for r in results:
        meta = tag_to_meta.get(r.get("tag"), {})
        r["parent"] = meta.get("parent")
        r["axis"] = meta.get("axis")
        r["pct"] = meta.get("pct")
        raw_rows.append(r)
    raw_df = pd.DataFrame(raw_rows)
    raw_df.to_csv(os.path.join(args.out_dir, "raw.csv"), index=False)

    # parent 단위로 center/perturb Cal 모아 통과 판정
    survivors = []
    summary = []
    for parent, g in raw_df.groupby("parent"):
        center = g[g["axis"] == "center"]
        perturb = g[g["axis"] != "center"]
        if center.empty or perturb.empty:
            continue
        center_cal = float(center["Cal"].iloc[0]) if "Cal" in center.columns else 0.0
        perturb_cal = perturb["Cal"].astype(float)
        if center_cal <= 0 or perturb_cal.isna().all():
            continue
        min_ratio = float(perturb_cal.min() / center_cal)
        mean_ratio = float(perturb_cal.mean() / center_cal)
        meta = center_map.get(parent, {})
        row = {
            "parent_tag": parent,
            "tag": meta.get("tag"),
            "asset": meta.get("asset"), "iv": meta.get("iv"), "lev": meta.get("lev"),
            "sma": meta.get("sma"), "ms": meta.get("ms"), "ml": meta.get("ml"),
            "snap": meta.get("snap"), "vmode": meta.get("vmode"), "vthr": meta.get("vthr"),
            # bridge 호환: iter_refine raw_combined 스키마와 동일 컬럼명 유지
            "Cal": center_cal, "Sh": meta.get("Sh"),
            "CAGR": meta.get("CAGR"), "MDD": meta.get("MDD"),
            "rebal": meta.get("rebal"), "liq": meta.get("liq"),
            "min_perturbed_Cal": float(perturb_cal.min()),
            "mean_perturbed_Cal": float(perturb_cal.mean()),
            "plateau_min_ratio": min_ratio,
            "plateau_mean_ratio": mean_ratio,
            "plateau_ok": min_ratio >= args.pass_ratio,
        }
        summary.append(row)
        if row["plateau_ok"]:
            survivors.append(row)
    pd.DataFrame(summary).to_csv(os.path.join(args.out_dir, "summary.csv"), index=False)
    pd.DataFrame(survivors).to_csv(os.path.join(args.out_dir, "survivors.csv"), index=False)
    print(f"[plateau] survivors: {len(survivors)}/{len(summary)} "
          f"(ratio>={args.pass_ratio})", flush=True)


if __name__ == "__main__":
    main()
