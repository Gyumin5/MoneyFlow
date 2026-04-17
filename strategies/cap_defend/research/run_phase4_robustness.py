#!/usr/bin/env python3
"""Phase-4 allocation-level robustness (3-anchor OOS + per-year LOYO).

Input: phase4_robustness_candidates.csv (top allocation 후보)
Output: phase4_robustness/results.csv
         phase4_robustness/yearly.csv

For each candidate:
  - anchor={2020-10-01, 2020-10-12, 2020-10-23} build stock/spot/fut full-period
    equity, mix with band → Cal/Sh/CAGR/MDD per anchor
  - From 2020-10-01 mix equity: per-year breakdown (2021..2025)
  - trimmed_cal (drop worst year), year_cal_cv, max_yearly_drawdown
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from phase_common import (
    FULL_END, atomic_write_csv, build_trace, equity_metrics,
    parse_tag, preload_futures, run_spot_ensemble,
)
from phase4_3asset import mix_eq, _load_stock_v17

ANCHORS = ["2020-10-01", "2020-10-12", "2020-10-23"]
YEARS = [2021, 2022, 2023, 2024, 2025]
OUT_DIR = os.path.join(HERE, "phase4_robustness")


def _members_for(ens_id: str, top_df: pd.DataFrame) -> list[str]:
    row = top_df[top_df["ensemble_tag"] == ens_id]
    if len(row) == 0:
        raise RuntimeError(f"ensemble not found: {ens_id}")
    return str(row.iloc[0]["members"]).split(";")


def build_spot_eq(ens_id: str, spot_top: pd.DataFrame, anchor: str) -> pd.Series:
    members = _members_for(ens_id, spot_top)
    k = len(members)
    w_each = min(1.0 / k, 0.5)
    total = w_each * k
    weights = {m: w_each / total for m in members}
    member_cfgs = {}
    for m_tag in members:
        meta = parse_tag(m_tag)
        member_cfgs[m_tag] = {
            "interval": meta["interval"], "sma": meta["sma"],
            "ms": meta["ms"], "ml": meta["ml"],
            "vol_mode": meta["vol_mode"], "vol_thr": meta["vol_thr"],
            "snap": meta["snap"],
        }
    r = run_spot_ensemble(member_cfgs, weights, anchor,
                           end=FULL_END, want_equity=True)
    eq = r.get("_equity")
    if not isinstance(eq, pd.Series):
        eq = pd.Series(eq)
    return eq


def build_fut_eq(ens_id: str, lev: float, fut_top: pd.DataFrame,
                 anchor: str) -> pd.Series:
    from futures_ensemble_engine import SingleAccountEngine, combine_targets
    members = _members_for(ens_id, fut_top)
    k = len(members)
    w_each = min(1.0 / k, 0.5)
    total = w_each * k
    weights = {m: w_each / total for m in members}
    data = preload_futures()
    bars_1h, funding_1h = data["1h"]
    all_dates_1h = bars_1h["BTC"].index
    traces = {}
    for m_tag in members:
        meta = parse_tag(m_tag)
        cfg = {"interval": meta["interval"], "sma": meta["sma"],
               "ms": meta["ms"], "ml": meta["ml"],
               "vol_mode": meta["vol_mode"], "vol_thr": meta["vol_thr"],
               "snap": meta["snap"]}
        tr = build_trace("fut", cfg, lev, anchor, end=FULL_END)["trace"]
        traces[m_tag] = tr
    dates = all_dates_1h[(all_dates_1h >= anchor) & (all_dates_1h <= FULL_END)]
    combined = combine_targets(traces, weights, dates)
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=lev, leverage_mode="fixed", per_coin_leverage_mode="none",
        stop_kind="none", stop_pct=0.0, stop_lookback_bars=0, stop_gate="always",
    )
    res = engine.run(combined)
    eq = res.get("_equity")
    if not isinstance(eq, pd.Series):
        eq = pd.Series(eq)
    return eq


def _strip_tz_index(eq: pd.Series) -> pd.Series:
    if isinstance(eq.index, pd.DatetimeIndex) and eq.index.tz is not None:
        eq = eq.copy()
        eq.index = eq.index.tz_localize(None)
    return eq


def yearly_metrics(mix_eq_series: pd.Series) -> dict:
    s = _strip_tz_index(mix_eq_series).sort_index()
    result = {}
    per_year = {}
    for y in YEARS:
        start = f"{y}-01-01"
        end = f"{y}-12-31"
        sub = s.loc[start:end]
        if len(sub) < 30:
            per_year[y] = {"Cal": None, "CAGR": None, "MDD": None}
            continue
        m = equity_metrics(sub)
        per_year[y] = {"Cal": m["Cal"], "CAGR": m["CAGR"], "MDD": m["MDD"]}
    result["per_year"] = per_year
    cals = [v["Cal"] for v in per_year.values() if v["Cal"] is not None and np.isfinite(v["Cal"])]
    cagrs = [v["CAGR"] for v in per_year.values() if v["CAGR"] is not None and np.isfinite(v["CAGR"])]
    mdds = [v["MDD"] for v in per_year.values() if v["MDD"] is not None and np.isfinite(v["MDD"])]
    if len(cals) >= 2:
        result["trimmed_cal"] = float(np.mean(sorted(cals)[1:]))  # drop worst
        result["cal_cv"] = float(np.std(cals) / (abs(np.mean(cals)) + 1e-9))
        result["min_cal"] = float(min(cals))
        result["max_cal"] = float(max(cals))
    else:
        result.update(trimmed_cal=None, cal_cv=None, min_cal=None, max_cal=None)
    if cagrs:
        result["worst_year_cagr"] = float(min(cagrs))
        result["best_year_cagr"] = float(max(cagrs))
    else:
        result.update(worst_year_cagr=None, best_year_cagr=None)
    if mdds:
        result["worst_year_mdd"] = float(min(mdds))
    else:
        result["worst_year_mdd"] = None
    return result


def run_candidate_anchor(cand: dict, anchor: str,
                          spot_top_path: str, fut_top_path: str) -> dict:
    """단일 (candidate, anchor) 실행."""
    spot_top = pd.read_csv(spot_top_path)
    fut_top = pd.read_csv(fut_top_path)
    stock_eq = _load_stock_v17()
    spot_eq = build_spot_eq(cand["spot"], spot_top, anchor)
    lev = float(cand["fut_lev"])
    if lev <= 0:
        # 60/40/0 special — mix only stock+spot
        series_dict = {"st": stock_eq, "sp": spot_eq}
        weights = {"st": float(cand["st_w"]), "sp": float(cand["sp_w"])}
    else:
        fut_eq = build_fut_eq(cand["fut"], lev, fut_top, anchor)
        series_dict = {"st": stock_eq, "sp": spot_eq, "fut": fut_eq}
        weights = {"st": float(cand["st_w"]),
                   "sp": float(cand["sp_w"]),
                   "fut": float(cand["fu_w"])}
    band_raw = cand["band"]
    try:
        band_val = float(band_raw)
        band = band_val
    except (TypeError, ValueError):
        # sleeve-relative band string
        parts = str(band_raw).split("_")
        band = {}
        for p in parts:
            if p.startswith("st"):
                band["st"] = float(p[2:])
            elif p.startswith("sp"):
                band["sp"] = float(p[2:])
            elif p.startswith("fu"):
                band["fut"] = float(p[2:])
    mix = mix_eq(series_dict, weights, band)
    m = equity_metrics(mix)
    out = {
        "label": cand["label"],
        "anchor": anchor,
        "Cal": m["Cal"], "Sh": m["Sh"],
        "CAGR": m["CAGR"], "MDD": m["MDD"],
    }
    if anchor == ANCHORS[0]:
        # 가장 초기 anchor에서만 per-year 분해
        yr = yearly_metrics(mix)
        out["yearly"] = yr
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--candidates", default="phase4_robustness_candidates.csv")
    ap.add_argument("--spot-top", default="phase3_ensembles/spot_top.csv")
    ap.add_argument("--fut-top", default="phase3_ensembles/fut_top.csv")
    ap.add_argument("--out-dir", default=OUT_DIR)
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    cand_df = pd.read_csv(args.candidates)
    cand_df["band"] = cand_df["band"].astype(str)
    print(f"[info] {len(cand_df)} candidates × {len(ANCHORS)} anchors = "
          f"{len(cand_df)*len(ANCHORS)} jobs", flush=True)

    jobs = []
    for _, row in cand_df.iterrows():
        cand = row.to_dict()
        for a in ANCHORS:
            jobs.append((cand, a))

    results = []
    yearly_rows = []
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = [ex.submit(run_candidate_anchor, j[0], j[1],
                          args.spot_top, args.fut_top) for j in jobs]
        for i, f in enumerate(as_completed(futs)):
            try:
                r = f.result()
            except Exception as e:
                print(f"[error] job {i}: {e}", flush=True)
                continue
            results.append({k: v for k, v in r.items() if k != "yearly"})
            if "yearly" in r:
                yr = r["yearly"]
                row = {"label": r["label"]}
                for y, v in yr["per_year"].items():
                    row[f"Cal_{y}"] = v.get("Cal")
                    row[f"CAGR_{y}"] = v.get("CAGR")
                    row[f"MDD_{y}"] = v.get("MDD")
                row.update({
                    "trimmed_cal": yr.get("trimmed_cal"),
                    "cal_cv": yr.get("cal_cv"),
                    "min_cal": yr.get("min_cal"),
                    "worst_year_cagr": yr.get("worst_year_cagr"),
                    "worst_year_mdd": yr.get("worst_year_mdd"),
                })
                yearly_rows.append(row)
            if (i + 1) % 5 == 0 or (i + 1) == len(jobs):
                print(f"  {i+1}/{len(jobs)} done", flush=True)

    res_df = pd.DataFrame(results).sort_values(["label", "anchor"])
    yr_df = pd.DataFrame(yearly_rows).sort_values("label")

    # 3앵커 집계
    agg = res_df.groupby("label").agg(
        Cal_mean=("Cal", "mean"), Cal_min=("Cal", "min"),
        Cal_sigma=("Cal", "std"),
        Sh_mean=("Sh", "mean"), CAGR_mean=("CAGR", "mean"),
        MDD_mean=("MDD", "mean"),
    ).reset_index()

    atomic_write_csv(res_df, os.path.join(out_dir, "per_anchor.csv"))
    atomic_write_csv(agg, os.path.join(out_dir, "anchor_agg.csv"))
    atomic_write_csv(yr_df, os.path.join(out_dir, "yearly.csv"))

    print("\n=== 3-ANCHOR AGG (by label) ===")
    print(agg.to_string(index=False))
    print("\n=== YEARLY ===")
    print(yr_df.to_string(index=False))


if __name__ == "__main__":
    main()
