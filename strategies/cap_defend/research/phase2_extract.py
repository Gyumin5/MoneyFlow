#!/usr/bin/env python3
"""Phase-2 axis-neighbor plateau 추출.

- 입력: phase1_sweep/{summary.csv, raw.csv}
- 평면: (asset, lev, interval, vol_mode, vol_thr) 공통
- 축: sma, ms, ml, snap 각각 ±1 step (grid 내 인접값)
- 통과 기준:
    valid_neighbor_n >= 5
    median(neighbor mCal) / self mCal >= 0.85
    mean(1[neighbor mCal >= 0.80 * self mCal]) >= 0.70
    std([self + neighbors mCal]) <= 0.25 * |self mCal|
- 상위 top_k_per_group 후보(rank_sum) 대상으로 적용
- 출력: phase2_extract/{survivors.csv, audit.csv, manifest.json}
"""
from __future__ import annotations

import argparse
import os
import sys
from dataclasses import dataclass

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from phase_common import parse_tag, write_manifest, atomic_write_csv

PHASE1_DIR = os.path.join(HERE, "phase1_sweep")
OUT_DIR = os.path.join(HERE, "phase2_extract")

# GRIDS는 summary에서 동적으로 추론한다 (iter_refine 동적 확장 대응).
# 각 (interval, axis) 당 실제 데이터에 존재하는 유니크값 set을 이웃 계산에 사용.
GRIDS: dict = {}


def infer_grids(sdf: pd.DataFrame) -> dict:
    g: dict = {}
    for iv, sub in sdf.groupby("interval"):
        g[str(iv)] = {
            "sma": sorted(set(int(v) for v in sub["sma"].dropna().unique())),
            "ms":  sorted(set(int(v) for v in sub["ms"].dropna().unique())),
            "ml":  sorted(set(int(v) for v in sub["ml"].dropna().unique())),
            "snap": sorted(set(int(v) for v in sub["snap"].dropna().unique())),
        }
    return g


@dataclass(frozen=True)
class PlateauConfig:
    top_k_per_group: int = 75
    min_valid_neighbors: int = 5
    min_median_ratio: float = 0.75
    min_good_neighbor_ratio: float = 0.60
    good_neighbor_threshold: float = 0.75
    max_std_ratio: float = 0.35
    require_full_anchor_count: int = 1


def load_phase1(summary_csv: str, raw_csv: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    sdf = pd.read_csv(summary_csv)
    rdf = pd.read_csv(raw_csv, on_bad_lines="skip")
    params = pd.DataFrame([parse_tag(tag) for tag in sdf["tag"]])
    # summary에 이미 있으면 중복 방지
    existing = set(sdf.columns)
    params = params.drop(columns=[c for c in params.columns if c in existing],
                         errors="ignore")
    if not params.empty and params.shape[1] > 0:
        sdf = pd.concat([sdf.reset_index(drop=True),
                         params.reset_index(drop=True)], axis=1)
    global GRIDS
    GRIDS = infer_grids(sdf)
    return sdf, rdf


def attach_quality_flags(sdf: pd.DataFrame, rdf: pd.DataFrame,
                         required_anchors: int) -> pd.DataFrame:
    err_df = rdf.groupby("tag", dropna=False).agg(
        raw_n=("anchor", "count"),
        err_n=("error", lambda s: int(s.fillna("").ne("").sum())),
    ).reset_index()
    out = sdf.merge(err_df, on="tag", how="left")
    out["raw_n"] = out["raw_n"].fillna(0).astype(int)
    out["err_n"] = out["err_n"].fillna(0).astype(int)
    out["has_full_anchors"] = out["n"].fillna(0).astype(int) >= required_anchors
    out["eligible"] = (
        (out["err_n"] <= 0)
        & out["has_full_anchors"]
    )
    return out


def add_rank_sum(sdf: pd.DataFrame) -> pd.DataFrame:
    out = sdf.copy()
    out["rank_mCal"] = out.groupby(["asset", "lev"])["mCal"].rank(ascending=False, method="min")
    out["rank_mSh"] = out.groupby(["asset", "lev"])["mSh"].rank(ascending=False, method="min")
    out["rank_mCAGR"] = out.groupby(["asset", "lev"])["mCAGR"].rank(ascending=False, method="min")
    out["rank_sum"] = out["rank_mCal"] + out["rank_mSh"] + out["rank_mCAGR"]
    return out


def axis_neighbor_values(interval: str, axis: str, value: int) -> list[int]:
    grid = GRIDS.get(interval)
    if not grid or axis not in grid:
        return []
    vs = sorted(grid[axis])
    if value not in vs:
        return []
    i = vs.index(value)
    out = []
    if i > 0:
        out.append(vs[i - 1])
    if i + 1 < len(vs):
        out.append(vs[i + 1])
    return out


def find_axis_neighbors(plane_df: pd.DataFrame, row: pd.Series) -> pd.DataFrame:
    iv = row["interval"]
    nb_keys = []
    for axis in ("sma", "ms", "ml", "snap"):
        for nv in axis_neighbor_values(iv, axis, int(row[axis])):
            key = {"sma": int(row["sma"]), "ms": int(row["ms"]),
                   "ml": int(row["ml"]), "snap": int(row["snap"])}
            key[axis] = nv
            nb_keys.append(key)
    if not nb_keys:
        return plane_df.iloc[0:0]
    mask = pd.Series(False, index=plane_df.index)
    for k in nb_keys:
        m = ((plane_df["sma"] == k["sma"]) & (plane_df["ms"] == k["ms"])
             & (plane_df["ml"] == k["ml"]) & (plane_df["snap"] == k["snap"]))
        mask = mask | m
    return plane_df[mask]


def score_plateau(plane_df: pd.DataFrame, row: pd.Series,
                  cfg: PlateauConfig) -> dict:
    nbrs = find_axis_neighbors(plane_df, row)
    nbrs = nbrs[nbrs["eligible"]].copy()
    self_cal = float(row["mCal"])
    n = int(len(nbrs))
    if n == 0 or self_cal <= 0:
        return {"plateau_ok": False, "neighbor_n": n,
                "neighbor_cal_median_ratio": 0.0,
                "good_neighbor_ratio": 0.0,
                "neighbor_std_ratio": float("inf")}
    cal_med = float(nbrs["mCal"].median())
    cal_med_ratio = cal_med / self_cal if self_cal else 0.0
    good_mask = nbrs["mCal"] >= self_cal * cfg.good_neighbor_threshold
    good_ratio = float(good_mask.mean())
    arr = np.concatenate([[self_cal], nbrs["mCal"].to_numpy()])
    std_ratio = float(arr.std()) / abs(self_cal) if self_cal else float("inf")
    ok = (
        n >= cfg.min_valid_neighbors
        and cal_med_ratio >= cfg.min_median_ratio
        and good_ratio >= cfg.min_good_neighbor_ratio
        and std_ratio <= cfg.max_std_ratio
    )
    return {"plateau_ok": bool(ok), "neighbor_n": n,
            "neighbor_cal_median_ratio": cal_med_ratio,
            "good_neighbor_ratio": good_ratio,
            "neighbor_std_ratio": std_ratio}


def extract_candidates(sdf: pd.DataFrame, cfg: PlateauConfig
                       ) -> tuple[pd.DataFrame, pd.DataFrame]:
    work = add_rank_sum(sdf)
    audits: list[dict] = []
    survivors: list[dict] = []
    for (asset, lev, interval), g in work.groupby(["asset", "lev", "interval"], sort=False):
        elig = g[g["eligible"]]
        top_cal = elig.sort_values("mCal", ascending=False).head(cfg.top_k_per_group)
        top_sh = elig.sort_values("mSh", ascending=False).head(cfg.top_k_per_group)
        top_cagr = elig.sort_values("mCAGR", ascending=False).head(cfg.top_k_per_group)
        top_rs = elig.sort_values("rank_sum", ascending=True).head(cfg.top_k_per_group)
        seed_pool = (
            pd.concat([top_cal, top_sh, top_cagr, top_rs])
            .drop_duplicates(subset=["tag"])
            .sort_values(["rank_sum", "mCal", "mSh"], ascending=[True, False, False])
        )
        for _, row in seed_pool.iterrows():
            plane_mask = (
                (work["asset"] == asset) & (work["lev"] == lev)
                & (work["interval"] == row["interval"])
                & (work["vol_mode"] == row["vol_mode"])
                & np.isclose(work["vol_thr"], row["vol_thr"])
            )
            plane_df = work[plane_mask]
            plateau = score_plateau(plane_df, row, cfg)
            rec = {**row.to_dict(), **plateau}
            audits.append(rec)
            if plateau["plateau_ok"]:
                survivors.append(rec)
    return pd.DataFrame(audits), pd.DataFrame(survivors)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase1-dir", default=PHASE1_DIR)
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--top-k", type=int, default=75)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    manifest_path = os.path.join(args.out_dir, "manifest.json")
    write_manifest(manifest_path, {"status": "running", "stage": "phase2_extract"})

    summary_csv = os.path.join(args.phase1_dir, "summary.csv")
    raw_csv = os.path.join(args.phase1_dir, "raw.csv")
    cfg = PlateauConfig(top_k_per_group=args.top_k)

    sdf, rdf = load_phase1(summary_csv, raw_csv)
    sdf = attach_quality_flags(sdf, rdf, required_anchors=cfg.require_full_anchor_count)
    audit_df, surv_df = extract_candidates(sdf, cfg)

    atomic_write_csv(audit_df, os.path.join(args.out_dir, "audit.csv"))
    atomic_write_csv(surv_df, os.path.join(args.out_dir, "survivors.csv"))

    write_manifest(manifest_path, {
        "status": "done", "stage": "phase2_extract",
        "n_audit": int(len(audit_df)),
        "n_survivors": int(len(surv_df)),
        "top_k_per_group": cfg.top_k_per_group,
    })
    print(f"audit={len(audit_df)} survivors={len(surv_df)} -> {args.out_dir}")


if __name__ == "__main__":
    main()
