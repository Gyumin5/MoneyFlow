#!/usr/bin/env python3
"""Phase-4 v2 사후 분석.

phase4_3asset_v2/raw.csv top-N (rank_sum) 후보에 대해:
  1) 연도별 Cal/Sh/MDD
  2) 위기 구간 수익률 (2020-03/04, 2022, 2024-08, 2025-04)
  3) 멤버 의존도 (앙상블 멤버 분포)
  4) 비중 ±5pp sensitivity (raw.csv 인접 grid)
  5) 밴드 sensitivity (같은 비중 다른 밴드)
출력: post_analysis_v4/{report.csv, report.md, summary.txt}
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

from phase_common import equity_metrics
from phase4_3asset import (build_ensemble_full_equity, mix_eq, _load_stock_v17)

CRISIS = [
    ("2020-03", "2020-03-01", "2020-04-30"),
    ("2022_bear", "2022-01-01", "2022-12-31"),
    ("2024-08", "2024-08-01", "2024-08-31"),
    ("2025-04", "2025-04-01", "2025-04-30"),
]


def yearly_metrics(eq: pd.Series) -> dict:
    out = {}
    ed = eq.resample("D").last().dropna()
    for yr, sub in ed.groupby(ed.index.year):
        if len(sub) < 30:
            continue
        m = equity_metrics(sub)
        out[int(yr)] = {"Cal": round(m["Cal"], 3),
                        "Sh": round(m["Sh"], 3),
                        "MDD": round(m["MDD"], 3),
                        "CAGR": round(m["CAGR"], 3)}
    return out


def crisis_returns(eq: pd.Series) -> dict:
    out = {}
    ed = eq.resample("D").last().dropna()
    for name, s, e in CRISIS:
        sub = ed[(ed.index >= s) & (ed.index <= e)]
        if len(sub) < 2:
            out[name] = None
            continue
        ret = sub.iloc[-1] / sub.iloc[0] - 1.0
        dd = (sub / sub.cummax() - 1).min()
        out[name] = {"ret": round(float(ret), 4),
                     "MDD": round(float(dd), 4)}
    return out


def weight_sensitivity(raw: pd.DataFrame, row: pd.Series, pp: int = 5) -> dict:
    same = raw[(raw["spot"] == row["spot"]) & (raw["fut"] == row["fut"])
               & (raw["band"] == row["band"])]
    near = []
    for _, r in same.iterrows():
        ds = abs(r["sp_w"] - row["sp_w"])
        df_ = abs(r["fu_w"] - row["fu_w"])
        if max(ds, df_) <= pp / 100 + 1e-9:
            near.append({"sp_w": r["sp_w"], "fu_w": r["fu_w"],
                         "Cal": round(r["Cal"], 3), "Sh": round(r["Sh"], 3),
                         "MDD": round(r["MDD"], 3)})
    if not near:
        return {"n": 0}
    cals = [n["Cal"] for n in near]
    return {"n": len(near),
            "Cal_mean": round(float(np.mean(cals)), 3),
            "Cal_std": round(float(np.std(cals)), 3),
            "Cal_min": round(min(cals), 3),
            "Cal_max": round(max(cals), 3),
            "neighbors": near[:10]}


def band_sensitivity(raw: pd.DataFrame, row: pd.Series) -> list[dict]:
    same = raw[(raw["spot"] == row["spot"]) & (raw["fut"] == row["fut"])
               & (raw["sp_w"] == row["sp_w"]) & (raw["fu_w"] == row["fu_w"])]
    out = []
    for _, r in same.sort_values("band").iterrows():
        out.append({"band": r["band"], "Cal": round(r["Cal"], 3),
                    "Sh": round(r["Sh"], 3), "MDD": round(r["MDD"], 3)})
    return out


def member_breakdown(spot_df: pd.DataFrame, fut_df: pd.DataFrame,
                     spot_id: str, fut_id: str) -> dict:
    s = spot_df[spot_df["ensemble_tag"] == spot_id]
    f = fut_df[fut_df["ensemble_tag"] == fut_id]
    return {
        "spot_members": s.iloc[0]["members"].split(";") if not s.empty else [],
        "fut_members": f.iloc[0]["members"].split(";") if not f.empty else [],
        "fut_lev": float(f.iloc[0]["lev"]) if not f.empty else None,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase4-dir",
                    default=os.path.join(HERE, "phase4_3asset_v2"))
    ap.add_argument("--phase3-dir",
                    default=os.path.join(HERE, "phase3_ensembles_v2"))
    ap.add_argument("--out-dir",
                    default=os.path.join(HERE, "post_analysis_v4"))
    ap.add_argument("--top-n", type=int, default=10)
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    raw_path = os.path.join(args.phase4_dir, "raw.csv")
    raw = pd.read_csv(raw_path)
    spot_top = pd.read_csv(os.path.join(args.phase3_dir, "spot_top.csv"))
    fut_top = pd.read_csv(os.path.join(args.phase3_dir, "fut_top.csv"))

    # 4-rank average for top selection (rank_sum이 이미 있음)
    if "rank_sum" not in raw.columns:
        raise RuntimeError("rank_sum missing in raw.csv")
    # 자산배분 운영용으로 CAGR이 1순위 → CAGR 가중 ranking 추가
    raw["rank_CAGR2"] = (raw["rank_CAGR"] * 2 + raw["rank_Cal"] + raw["rank_Sh"])
    top_default = raw.sort_values("rank_sum").head(args.top_n).reset_index(drop=True)
    top_cagr = raw.sort_values("CAGR", ascending=False).head(args.top_n).reset_index(drop=True)
    top_cagr2 = raw.sort_values("rank_CAGR2").head(args.top_n).reset_index(drop=True)
    # union 후 중복 제거 (key=spot/fut/weights/band)
    keycols = ["spot", "fut", "st_w", "sp_w", "fu_w", "band"]
    top = (pd.concat([top_default, top_cagr, top_cagr2], ignore_index=True)
           .drop_duplicates(subset=keycols).reset_index(drop=True))
    print(f"Top union: rank_sum={len(top_default)} + CAGR={len(top_cagr)} + CAGR2={len(top_cagr2)} → unique {len(top)}")
    # 추가: ranking별 top 리스트 별도 저장
    rank_lists = {
        "rank_sum_top": top_default[keycols + ["Cal", "Sh", "CAGR", "MDD"]].to_dict("records"),
        "CAGR_top": top_cagr[keycols + ["Cal", "Sh", "CAGR", "MDD"]].to_dict("records"),
        "CAGR2_top": top_cagr2[keycols + ["Cal", "Sh", "CAGR", "MDD"]].to_dict("records"),
    }

    print(f"Top-{args.top_n} 후보 분석 시작")
    print("V17 stock + spot/fut 앙상블 equity 재구성...")
    stock_eq = _load_stock_v17()

    # 캐싱: 같은 spot/fut id는 한 번만 계산
    spot_eq_cache: dict[str, pd.Series] = {}
    fut_eq_cache: dict[str, pd.Series] = {}
    needed_spot = set(top["spot"]) | set(spot_top["ensemble_tag"][:0])
    needed_fut = set(top["fut"])
    for sp_id in needed_spot:
        row = spot_top[spot_top["ensemble_tag"] == sp_id]
        if row.empty:
            continue
        print(f"  build spot {sp_id}")
        spot_eq_cache[sp_id] = build_ensemble_full_equity(row.iloc[0])
    for fu_id in needed_fut:
        row = fut_top[fut_top["ensemble_tag"] == fu_id]
        if row.empty:
            continue
        print(f"  build fut {fu_id}")
        fut_eq_cache[fu_id] = build_ensemble_full_equity(row.iloc[0])

    report = []
    member_counter: dict[str, int] = {}
    for i, r in top.iterrows():
        sp_eq = spot_eq_cache.get(r["spot"])
        fu_eq = fut_eq_cache.get(r["fut"])
        if sp_eq is None or fu_eq is None:
            print(f"skip rank {i+1}: missing equity")
            continue
        mix = mix_eq({"st": stock_eq, "sp": sp_eq, "fut": fu_eq},
                     {"st": float(r["st_w"]), "sp": float(r["sp_w"]),
                      "fut": float(r["fu_w"])},
                     band=float(r["band"]))
        yearly = yearly_metrics(mix)
        crisis = crisis_returns(mix)
        wsens = weight_sensitivity(raw, r, pp=5)
        bsens = band_sensitivity(raw, r)
        members = member_breakdown(spot_top, fut_top, r["spot"], r["fut"])
        for m in members["spot_members"] + members["fut_members"]:
            member_counter[m] = member_counter.get(m, 0) + 1
        entry = {
            "rank": i + 1,
            "spot": r["spot"], "fut": r["fut"], "fut_lev": r["fut_lev"],
            "weights": f"{int(r['st_w']*100)}/{int(r['sp_w']*100)}/{int(r['fu_w']*100)}",
            "band": int(r["band"] * 100),
            "Cal": round(r["Cal"], 3), "Sh": round(r["Sh"], 3),
            "CAGR": round(r["CAGR"], 3), "MDD": round(r["MDD"], 3),
            "yearly": yearly,
            "crisis": crisis,
            "weight_sens_5pp": wsens,
            "band_sens": bsens,
            "members": members,
        }
        report.append(entry)
        print(f"  rank {i+1} done ({r['spot']}/{r['fut']} {entry['weights']} b{entry['band']})")

    # 멤버 쏠림
    member_top = sorted(member_counter.items(), key=lambda x: -x[1])[:15]

    out_json = os.path.join(args.out_dir, "report.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"top": report, "member_usage_top15": member_top,
                   "rank_lists": rank_lists},
                  f, ensure_ascii=False, indent=2, default=str)

    # 텔레그램용 요약
    lines = [f"[Phase-4 사후분석] top-{len(report)} 후보 결과", ""]
    for e in report:
        lines.append(f"#{e['rank']} {e['weights']} b{e['band']} "
                     f"({e['spot']}/{e['fut']} L{int(e['fut_lev'])})")
        lines.append(f"  Cal={e['Cal']} Sh={e['Sh']} CAGR={e['CAGR']} MDD={e['MDD']}")
        yrs = e["yearly"]
        worst_yr = min(yrs.items(), key=lambda x: x[1]["Cal"]) if yrs else (None, None)
        best_yr = max(yrs.items(), key=lambda x: x[1]["Cal"]) if yrs else (None, None)
        if yrs:
            lines.append(f"  연도 Cal min={worst_yr[0]}({worst_yr[1]['Cal']}) "
                         f"max={best_yr[0]}({best_yr[1]['Cal']})")
        ws = e["weight_sens_5pp"]
        if ws.get("n", 0):
            lines.append(f"  ±5pp Cal mean={ws['Cal_mean']} std={ws['Cal_std']} "
                         f"min={ws['Cal_min']}")
        c = e["crisis"]
        crisis_str = " ".join(
            f"{n}={c[n]['ret']*100:.1f}%" if c.get(n) else f"{n}=NA"
            for n in ("2022_bear", "2024-08"))
        lines.append(f"  위기: {crisis_str}")
        lines.append("")
    lines.append("=== 멤버 사용 빈도 top15 (top-N 합산) ===")
    for m, c in member_top:
        lines.append(f"  {m}: {c}")

    lines.append("")
    lines.append("=== CAGR 우선 ranking top10 ===")
    for r in rank_lists["CAGR_top"][:10]:
        lines.append(f"  {r['spot']}/{r['fut']} "
                     f"{int(r['st_w']*100)}/{int(r['sp_w']*100)}/{int(r['fu_w']*100)} b{int(r['band']*100)} "
                     f"CAGR={r['CAGR']*100:.1f}% Cal={r['Cal']:.2f} MDD={r['MDD']*100:.1f}%")
    lines.append("")
    lines.append("=== CAGR×2 + Cal + Sh 가중 ranking top10 ===")
    for r in rank_lists["CAGR2_top"][:10]:
        lines.append(f"  {r['spot']}/{r['fut']} "
                     f"{int(r['st_w']*100)}/{int(r['sp_w']*100)}/{int(r['fu_w']*100)} b{int(r['band']*100)} "
                     f"CAGR={r['CAGR']*100:.1f}% Cal={r['Cal']:.2f} Sh={r['Sh']:.2f} MDD={r['MDD']*100:.1f}%")

    summary_path = os.path.join(args.out_dir, "summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Done. report: {out_json}")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
