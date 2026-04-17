#!/usr/bin/env python3
"""Sub-period rank-sum (Borda count) robustness analysis.

8 full-period candidates × 10 semi-annual windows (H1 2021 ~ H2 2025).
Each window: rank by Cal, CAGR, CalxCAGR -> sum ranks across all windows.
Most consistent candidate = lowest rank_sum.

Input: phase4_10x/raw.csv + phase3_10x/{spot_top,fut_top}.csv
Output: phase4_10x_robustness/subperiod_ranksum.csv
        phase4_10x_robustness/subperiod_detail.csv
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from phase_common import (
    FULL_END, equity_metrics, atomic_write_csv,
)
from phase4_3asset import mix_eq, _load_stock_v17, build_ensemble_full_equity

WINDOWS = [
    ("H1_2021", "2021-01-01", "2021-06-30"),
    ("H2_2021", "2021-07-01", "2021-12-31"),
    ("H1_2022", "2022-01-01", "2022-06-30"),
    ("H2_2022", "2022-07-01", "2022-12-31"),
    ("H1_2023", "2023-01-01", "2023-06-30"),
    ("H2_2023", "2023-07-01", "2023-12-31"),
    ("H1_2024", "2024-01-01", "2024-06-30"),
    ("H2_2024", "2024-07-01", "2024-12-31"),
    ("H1_2025", "2025-01-01", "2025-06-30"),
    ("H2_2025", "2025-07-01", "2025-12-31"),
]

OUT_DIR = os.path.join(HERE, "phase4_10x_robustness")


def _strip_tz(s: pd.Series) -> pd.Series:
    if isinstance(s.index, pd.DatetimeIndex) and s.index.tz is not None:
        s = s.copy()
        s.index = s.index.tz_localize(None)
    return s


def pick_candidates(raw_path: str) -> pd.DataFrame:
    df = pd.read_csv(raw_path)
    df["CalxCAGR"] = df["Cal"] * df["CAGR"]
    allocs = [(0.60, 0.25, 0.15), (0.60, 0.30, 0.10),
              (0.60, 0.35, 0.05), (0.60, 0.40, 0.00)]
    rows = []
    for st, sp, fu in allocs:
        sub = df[(abs(df["st_w"] - st) < 0.001) & (abs(df["sp_w"] - sp) < 0.001)]
        for bm in ["abs", "sleeve"]:
            bsub = sub[sub["band_mode"] == bm]
            if len(bsub) == 0:
                continue
            top = bsub.sort_values("CalxCAGR", ascending=False).iloc[0].to_dict()
            top["label"] = f"{int(st*100)}_{int(sp*100)}_{int(fu*100)}_{bm}"
            rows.append(top)
    return pd.DataFrame(rows)


def build_mix_equity(cand: dict, stock_eq: pd.Series,
                     spot_top: pd.DataFrame, fut_top: pd.DataFrame) -> pd.Series:
    spot_ens = spot_top[spot_top["ensemble_tag"] == cand["spot"]].iloc[0]
    spot_eq = build_ensemble_full_equity(spot_ens)

    fu_w = float(cand["fu_w"])
    weights = {"st": float(cand["st_w"]), "sp": float(cand["sp_w"])}

    if fu_w > 0.001:
        fut_ens = fut_top[fut_top["ensemble_tag"] == cand["fut"]].iloc[0]
        fut_eq = build_ensemble_full_equity(fut_ens)
        series_dict = {"st": stock_eq, "sp": spot_eq, "fut": fut_eq}
        weights["fut"] = fu_w
    else:
        series_dict = {"st": stock_eq, "sp": spot_eq}

    band_raw = cand["band"]
    try:
        band = float(band_raw)
    except (TypeError, ValueError):
        parts = str(band_raw).split("_")
        band = {}
        for p in parts:
            if p.startswith("st"):
                band["st"] = float(p[2:])
            elif p.startswith("sp"):
                band["sp"] = float(p[2:])
            elif p.startswith("fu"):
                band["fut"] = float(p[2:])

    return mix_eq(series_dict, weights, band)


def main():
    raw_path = os.path.join(HERE, "phase4_10x", "raw.csv")
    spot_top = pd.read_csv(os.path.join(HERE, "phase3_10x", "spot_top.csv"))
    fut_top = pd.read_csv(os.path.join(HERE, "phase3_10x", "fut_top.csv"))

    cands = pick_candidates(raw_path)
    print(f"[info] {len(cands)} candidates × {len(WINDOWS)} windows", flush=True)

    stock_eq = _load_stock_v17()
    print("[info] stock V17 loaded", flush=True)

    equities = {}
    for _, row in cands.iterrows():
        label = row["label"]
        print(f"  building equity: {label} ...", flush=True)
        eq = build_mix_equity(row.to_dict(), stock_eq, spot_top, fut_top)
        equities[label] = _strip_tz(eq)
    print(f"[info] all {len(equities)} equities built", flush=True)

    detail_rows = []
    for wname, wstart, wend in WINDOWS:
        for label, eq in equities.items():
            sub = eq.loc[wstart:wend]
            if len(sub) < 20:
                detail_rows.append({
                    "label": label, "window": wname,
                    "Sh": None, "Cal": None, "CAGR": None, "MDD": None, "CalxCAGR": None,
                })
                continue
            m = equity_metrics(sub)
            cal = m["Cal"]
            cagr = m["CAGR"]
            calxcagr = cal * cagr if cal is not None and cagr is not None else None
            detail_rows.append({
                "label": label, "window": wname,
                "Sh": round(m["Sh"], 4),
                "Cal": round(cal, 4),
                "CAGR": round(cagr, 4),
                "MDD": round(m["MDD"], 4),
                "CalxCAGR": round(calxcagr, 4) if calxcagr is not None else None,
            })

    detail_df = pd.DataFrame(detail_rows)

    for wname in [w[0] for w in WINDOWS]:
        sub = detail_df[detail_df["window"] == wname]
        for metric in ["Sh", "Cal", "CAGR", "CalxCAGR"]:
            vals = sub[metric]
            if vals.isna().all():
                continue
            ranks = vals.rank(ascending=False, method="min")
            detail_df.loc[detail_df["window"] == wname, f"rank_{metric}"] = ranks.values

    labels = detail_df["label"].unique()
    agg_rows = []
    for label in labels:
        sub = detail_df[detail_df["label"] == label]
        row = {"label": label}
        for metric in ["Sh", "Cal", "CAGR", "CalxCAGR"]:
            rc = f"rank_{metric}"
            if rc in sub.columns:
                ranks = sub[rc].dropna()
                row[f"ranksum_{metric}"] = float(ranks.sum())
                row[f"mean_rank_{metric}"] = round(float(ranks.mean()), 2)
                row[f"worst_rank_{metric}"] = float(ranks.max())
                row[f"n_wins_{metric}"] = int((ranks == 1.0).sum())
            else:
                row[f"ranksum_{metric}"] = None
                row[f"mean_rank_{metric}"] = None
                row[f"worst_rank_{metric}"] = None
                row[f"n_wins_{metric}"] = 0
        rsum_vals = [v for k, v in row.items()
                     if k.startswith("ranksum_") and v is not None]
        row["ranksum_total"] = sum(rsum_vals) if rsum_vals else None
        agg_rows.append(row)

    agg_df = pd.DataFrame(agg_rows).sort_values("ranksum_total")

    os.makedirs(OUT_DIR, exist_ok=True)
    atomic_write_csv(detail_df, os.path.join(OUT_DIR, "subperiod_detail.csv"))
    atomic_write_csv(agg_df, os.path.join(OUT_DIR, "subperiod_ranksum.csv"))

    print("\n=== SUB-PERIOD RANK-SUM (Borda Count) ===")
    cols = ["label", "ranksum_total",
            "ranksum_Cal", "ranksum_CAGR", "ranksum_CalxCAGR",
            "n_wins_CalxCAGR", "worst_rank_CalxCAGR"]
    print(agg_df[cols].to_string(index=False))

    print("\n=== PER-WINDOW CalxCAGR ===")
    pivot = detail_df.pivot(index="label", columns="window", values="CalxCAGR")
    window_order = [w[0] for w in WINDOWS]
    pivot = pivot[[c for c in window_order if c in pivot.columns]]
    print(pivot.to_string())

    print("\n=== PER-WINDOW Cal ===")
    pivot_cal = detail_df.pivot(index="label", columns="window", values="Cal")
    pivot_cal = pivot_cal[[c for c in window_order if c in pivot_cal.columns]]
    print(pivot_cal.to_string())


if __name__ == "__main__":
    main()
