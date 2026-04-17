#!/usr/bin/env python3
"""Phase A → Phase B seed 추출.

phase A single_results.csv에서 lev × tf별 Top 후보를 뽑아 phase_b_seeds.json으로 저장.

선정 규칙:
- error 없는 행만
- Liq <= 1 (청산 1회 이하만 허용; lev별로 다른 청산 환경 고려해 완화)
- per (lev, tf): Cal Top + Sharpe Top 합집합으로 후보 풀
- Pareto frontier (Cal, Sharpe, -MDD) 기반 비지배 해 선별
- per (lev, tf): 2x/3x은 N=10, 4x/5x는 N=5
"""
from __future__ import annotations
import argparse
import json
import os
import sys

import pandas as pd

DEFAULT_RUN_ROOT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "fixedlev_search_runs",
)


def is_pareto(points: pd.DataFrame, cols: list) -> pd.Series:
    """columns에서 클수록 좋은 해의 비지배 마스크."""
    arr = points[cols].to_numpy()
    n = len(arr)
    keep = [True] * n
    for i in range(n):
        if not keep[i]:
            continue
        for j in range(n):
            if i == j or not keep[j]:
                continue
            if all(arr[j] >= arr[i]) and any(arr[j] > arr[i]):
                keep[i] = False
                break
    return pd.Series(keep, index=points.index)


def select_seeds(df: pd.DataFrame, lev: float, tf: str, top_n: int, max_liq: int = 1) -> pd.DataFrame:
    sub = df[(df["leverage"] == lev) & (df["interval"] == tf)].copy()
    sub = sub[sub["error"].isna() | (sub["error"] == "")]
    sub = sub[sub["Liq"] <= max_liq]
    if len(sub) == 0:
        return sub
    # 후보 풀: Cal Top 2N + Sharpe Top 2N 합집합
    pool_n = max(top_n * 2, 30)
    pool = pd.concat([
        sub.nlargest(pool_n, "Cal"),
        sub.nlargest(pool_n, "Sharpe"),
    ]).drop_duplicates(subset=["case_id"])

    # Pareto frontier (Cal, Sharpe, -MDD)
    pool = pool.assign(_negMDD=-pool["MDD"])
    mask = is_pareto(pool, ["Cal", "Sharpe", "_negMDD"])
    pareto = pool[mask].copy()

    # Pareto가 너무 많으면 Cal 기준 Top N
    if len(pareto) > top_n:
        pareto = pareto.nlargest(top_n, "Cal")
    elif len(pareto) < top_n:
        # 부족하면 Pareto 외에서 Cal로 채움
        extra = pool[~mask].nlargest(top_n - len(pareto), "Cal")
        pareto = pd.concat([pareto, extra])

    return pareto.sort_values("Cal", ascending=False).head(top_n)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default=DEFAULT_RUN_ROOT)
    parser.add_argument("--phase-a-name", default="fixedlev_v1")
    parser.add_argument("--out-name", default="phase_b_seeds.json")
    parser.add_argument("--top-2x", type=int, default=10)
    parser.add_argument("--top-3x", type=int, default=10)
    parser.add_argument("--top-4x", type=int, default=5)
    parser.add_argument("--top-5x", type=int, default=5)
    parser.add_argument("--max-liq", type=int, default=1, help="seed 후보의 최대 청산 횟수")
    args = parser.parse_args()

    src_csv = os.path.join(args.run_root, args.phase_a_name, "single_results.csv")
    if not os.path.isfile(src_csv):
        print(f"ERROR: not found {src_csv}", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(src_csv)
    print(f"Loaded {len(df)} rows from {src_csv}")

    top_per_lev = {2.0: args.top_2x, 3.0: args.top_3x, 4.0: args.top_4x, 5.0: args.top_5x}

    seeds: list = []
    summary: dict = {}
    for lev in [2.0, 3.0, 4.0, 5.0]:
        n_target = top_per_lev[lev]
        for tf in ["D", "4h", "2h"]:
            picked = select_seeds(df, lev, tf, n_target, max_liq=args.max_liq)
            for _, r in picked.iterrows():
                seeds.append({
                    "lev": float(r["leverage"]),
                    "interval": str(r["interval"]),
                    "label": str(r["label"]),
                    "sma_bars": int(r["sma_bars"]),
                    "mom_short_bars": int(r["mom_short_bars"]),
                    "mom_long_bars": int(r["mom_long_bars"]),
                    "vol_threshold": float(r["vol_threshold"]),
                    "snap_interval_bars": int(r["snap_interval_bars"]),
                    "phase_a_metrics": {
                        "Sharpe": float(r["Sharpe"]),
                        "CAGR": float(r["CAGR"]),
                        "MDD": float(r["MDD"]),
                        "Cal": float(r["Cal"]),
                        "Liq": int(r["Liq"]),
                        "Stops": int(r["Stops"]),
                    },
                    "phase_a_guard": str(r["guard_mode"]),
                })
            summary[f"L{int(lev)}_{tf}"] = {
                "picked": len(picked),
                "best_cal": float(picked["Cal"].max()) if len(picked) else None,
                "median_cal": float(picked["Cal"].median()) if len(picked) else None,
            }

    out_path = os.path.join(args.run_root, args.phase_a_name, args.out_name)
    with open(out_path, "w") as f:
        json.dump({"seeds": seeds, "summary": summary, "total": len(seeds)}, f, indent=2)
    print(f"Wrote {len(seeds)} seeds to {out_path}")
    print("Summary:")
    for k, v in summary.items():
        print(f"  {k}: picked={v['picked']} best_Cal={v['best_cal']} med_Cal={v['median_cal']}")


if __name__ == "__main__":
    main()
