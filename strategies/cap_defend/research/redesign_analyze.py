"""Step 5 + 6 — Rank-sum 랭킹 + Ensemble 후보 메타 생성.

입력:
  redesign_phase_survivors_{asset}.csv   (Step 3 명시 survivor, filter_phase.py 출력)
  redesign_rerank_phase_{asset}.csv      (phase 축 raw — 지표용)
  redesign_snap_nudge_{asset}.csv        (Step 3.5 cadence raw)
  redesign_yearly_{asset}.csv            (Step 4 년도별 raw)
  redesign_stress_{asset}.csv            (Step 7 — tx_stress 축 용)
  redesign_top500_{asset}_k1.csv         (cfg/snap/interval 참조)

출력:
  redesign_rank_{asset}.csv              (k=1 rank-sum 순위)
  redesign_ensemble_candidates_{asset}.csv (Step 6 bucket combo, BT 별도)

Rank 축:
  R_phase      : phase_med 높을수록 좋음
  R_yearly     : yearly_med 높을수록 좋음
  R_adverse    : worst_year (또는 2022 Cal) 높을수록 좋음
  R_phase_CV   : phase_CV 낮을수록 좋음
  R_snap_CV    : snap_nudge_CV 낮을수록 좋음 (옵션)
  R_turnover   : rebal 수 낮을수록 좋음 (phase 축 median)
  R_tx_stress  : tx_2x_Cal / tx_1x_Cal 비율 높을수록 좋음 (옵션 — stress csv 있을때만)
  R_positive   : positive_years_ratio 높을수록 좋음

Penalty:
  + 5 if base snap 이 nice number (Codex 합의 D1)
  + 10 if snap_nudge_CV > p70
  + Hard reject: 2022 Cal < 자산별 gate (코인 p40, 주식 0.3)
"""
from __future__ import annotations
import argparse
import itertools
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
from redesign_common import is_prime_x3_snap, NICE_SNAPS


def _cv(std, mean):
    if abs(mean) < 1e-9:
        return 99.0
    return float(std / abs(mean))


def load_phase_raw(asset):
    path = os.path.join(HERE, f"redesign_rerank_phase_{asset}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df.dropna(subset=["Cal"])
    if "MDD" in df.columns:
        df = df.dropna(subset=["MDD"])
    return df


def load_snap_raw(asset):
    path = os.path.join(HERE, f"redesign_snap_nudge_{asset}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df.dropna(subset=["Cal"])
    if "MDD" in df.columns:
        df = df.dropna(subset=["MDD"])
    return df


def load_yearly_raw(asset):
    path = os.path.join(HERE, f"redesign_yearly_{asset}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df.dropna(subset=["Cal"])
    if "MDD" in df.columns:
        df = df.dropna(subset=["MDD"])
    return df


def load_stress_raw(asset):
    path = os.path.join(HERE, f"redesign_stress_{asset}.csv")
    if not os.path.exists(path):
        return pd.DataFrame()
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    df = df.dropna(subset=["Cal"])
    if "MDD" in df.columns:
        df = df.dropna(subset=["MDD"])
    return df


def aggregate_phase(df):
    if df is None or df.empty or "tag" not in df.columns:
        return pd.DataFrame(columns=["tag", "phase_med", "phase_p20", "phase_CV", "phase_rebal_med"])
    # 동시 write 등에서 생긴 (tag, phase) 중복 방지
    if "phase" in df.columns:
        df = df.drop_duplicates(subset=["tag", "phase"], keep="first")
    rows = []
    for tag, g in df.groupby("tag"):
        cal = g["Cal"]
        reb = g.get("rebal", pd.Series(dtype=float))
        rows.append({
            "tag": tag,
            "phase_med": cal.median(),
            "phase_p20": np.percentile(cal, 20),
            "phase_CV": _cv(cal.std(), cal.mean()),
            "phase_rebal_med": reb.median() if len(reb) else None,
        })
    return pd.DataFrame(rows)


def aggregate_snap(df):
    if df is None or df.empty or "tag" not in df.columns:
        return pd.DataFrame(columns=["tag", "snap_nudge_med", "snap_nudge_CV"])
    if "snap" in df.columns:
        # snap_nudge 출력은 "nudge_snap" 컬럼. legacy "snap" fallback.
        snap_subset = "nudge_snap" if "nudge_snap" in df.columns else "snap"
        df = df.drop_duplicates(subset=["tag", snap_subset], keep="first")
    rows = []
    for tag, g in df.groupby("tag"):
        cal = g["Cal"]
        rows.append({
            "tag": tag,
            "snap_nudge_med": cal.median(),
            "snap_nudge_CV": _cv(cal.std(), cal.mean()),
        })
    return pd.DataFrame(rows)


def aggregate_yearly(df):
    if df is None or df.empty or "tag" not in df.columns:
        return pd.DataFrame(columns=["tag", "yearly_med", "worst_year", "pos_ratio",
                                     "negative_years", "n_years", "yearly_2022",
                                     "yearly_rank_mean", "yearly_rank_worst"])
    if "year" in df.columns:
        df = df.drop_duplicates(subset=["tag", "year"], keep="first")
    # yearly_rank: 각 해 config 간 Cal rank → tag 의 평균/최악 rank (Borda 스타일)
    # 각 year 내 tag 를 Cal 높은 순 (ascending=False) 으로 rank
    tmp = df.copy()
    tmp["year_rank"] = tmp.groupby("year")["Cal"].rank(ascending=False, na_option="bottom")
    rank_agg = tmp.groupby("tag")["year_rank"].agg(["mean", "max"]).reset_index()
    rank_agg.columns = ["tag", "yearly_rank_mean", "yearly_rank_worst"]
    rows = []
    for tag, g in df.groupby("tag"):
        cal = g["Cal"]
        cal_2022 = float(g[g["year"] == 2022]["Cal"].iloc[0]) if (g["year"] == 2022).any() else None
        rows.append({
            "tag": tag,
            "yearly_med": cal.median(),
            "worst_year": cal.min(),
            "pos_ratio": float((cal > 0).sum()) / max(len(cal), 1),
            "negative_years": int((cal < 0).sum()),
            "n_years": int(len(cal)),
            "yearly_2022": cal_2022,
        })
    base = pd.DataFrame(rows)
    return base.merge(rank_agg, on="tag", how="left")


def aggregate_stress(df):
    """tx_2x_Cal / baseline_Cal 비율. 없으면 None."""
    if df is None or df.empty or "tag" not in df.columns:
        return pd.DataFrame(columns=["tag", "tx_stress_ratio"])
    if "scenario" in df.columns:
        df = df.drop_duplicates(subset=["tag", "scenario"], keep="first")
    rows = []
    for tag, g in df.groupby("tag"):
        base = g[g["scenario"] == "baseline"]
        tx2 = g[g["scenario"] == "tx_2.0x"]
        if len(base) and len(tx2):
            b = float(base["Cal"].iloc[0])
            t = float(tx2["Cal"].iloc[0])
            ratio = t / b if abs(b) > 1e-9 else None
        else:
            ratio = None
        rows.append({"tag": tag, "tx_2x_ratio": ratio})
    return pd.DataFrame(rows)


def apply_hard_gates(df, asset):
    """Hard Gates: 2022 Cal + snap%3==0 (3-tranche 정합성).
    yearly_2022 컬럼이 없으면 (yearly 미실행) 전부 FAIL — codex 지적.
    """
    snap_col = "snap" if "snap" in df.columns else ("snap_days" if "snap_days" in df.columns else None)
    if snap_col is not None:
        before = len(df)
        df = df[df[snap_col].astype(int) % 3 == 0].reset_index(drop=True)
        if len(df) < before:
            print(f"[{asset}] snap%3 filter: {before} → {len(df)}", flush=True)
    if "yearly_2022" not in df.columns:
        print(f"[{asset}] WARN: yearly_2022 missing → all FAIL 2022 gate")
        return df.assign(hard_gate_pass=False)
    # stock 도 코인과 동일하게 percentile 기반 (p40) — hardcoded 0.3 은 너무 strict.
    v = df["yearly_2022"].dropna()
    gate = float(np.percentile(v, 40)) if len(v) else -99
    # 2022 Cal 결측인 configs 는 FAIL (sentinel -99 → gate 보다 낮음)
    df = df.assign(hard_gate_pass=df["yearly_2022"].fillna(-99) >= gate)
    return df


def add_nice_penalty(df):
    """D1: base snap 이 nice number 면 +5 rank penalty, prime×3 면 0."""
    df = df.copy()
    # stock 은 snap_days, fut/spot 은 snap 컬럼
    snap_col = "snap" if "snap" in df.columns else ("snap_days" if "snap_days" in df.columns else None)
    if snap_col is None:
        df["nice_penalty"] = 0
    else:
        df["nice_penalty"] = df[snap_col].apply(lambda s: 5 if int(s) in NICE_SNAPS else 0)
    return df


def rank_sum(df):
    """자산 내 rank-sum. 낮을수록 좋음. NaN 은 큰 수로 취급."""
    ranks = pd.DataFrame(index=df.index)
    def _rank(col, asc):
        return df[col].rank(ascending=asc, na_option="bottom") if col in df.columns else pd.Series(0, index=df.index)
    ranks["R_phase"] = _rank("phase_med", False)
    ranks["R_yearly"] = _rank("yearly_med", False)
    ranks["R_adverse"] = _rank("worst_year", False)
    ranks["R_positive"] = _rank("pos_ratio", False)
    ranks["R_phase_CV"] = _rank("phase_CV", True)
    if "snap_nudge_CV" in df.columns:
        ranks["R_snap_CV"] = df["snap_nudge_CV"].rank(ascending=True, na_option="bottom")
    if "phase_rebal_med" in df.columns:
        ranks["R_turnover"] = df["phase_rebal_med"].rank(ascending=True, na_option="bottom")
    if "tx_2x_ratio" in df.columns and df["tx_2x_ratio"].notna().any():
        ranks["R_tx_stress"] = df["tx_2x_ratio"].rank(ascending=False, na_option="bottom")
    # 신규 축 (라운드9): plateau / bootstrap / yearly_rank
    if "neighborhood_Cal_CV" in df.columns and df["neighborhood_Cal_CV"].notna().any():
        ranks["R_plateau_CV"] = df["neighborhood_Cal_CV"].rank(ascending=True, na_option="bottom")
    if "neighborhood_p10_Cal" in df.columns and df["neighborhood_p10_Cal"].notna().any():
        ranks["R_plateau_p10"] = df["neighborhood_p10_Cal"].rank(ascending=False, na_option="bottom")
    # R_bootstrap 제거 (Codex r10): CAGR permutation 은 total CAGR 에 영향 없음 → 진단용으로만 report 에 표시
    if "yearly_rank_mean" in df.columns and df["yearly_rank_mean"].notna().any():
        ranks["R_yearly_rank"] = df["yearly_rank_mean"].rank(ascending=True, na_option="bottom")

    penalty = df["nice_penalty"].fillna(0).astype(float)
    # snap_nudge_CV p70 penalty
    if "snap_nudge_CV" in df.columns and df["snap_nudge_CV"].notna().any():
        p70 = float(np.percentile(df["snap_nudge_CV"].dropna(), 70))
        penalty = penalty + (df["snap_nudge_CV"].fillna(99) > p70).astype(int) * 10
    df = df.copy()
    df["rank_sum"] = ranks.sum(axis=1).astype(float) + penalty
    return df.sort_values("rank_sum").reset_index(drop=True)


def run_step5(asset):
    top500_path = os.path.join(HERE, f"redesign_top500_{asset}_k1.csv")
    if not os.path.exists(top500_path):
        print(f"[{asset}] top500 csv missing: {top500_path}")
        return pd.DataFrame()
    top500 = pd.read_csv(top500_path)

    # Survivor CSV 우선 (Step 3 명시). 없으면 raw phase 로 통 계산
    surv_path = os.path.join(HERE, f"redesign_phase_survivors_{asset}.csv")
    if os.path.exists(surv_path):
        surv = pd.read_csv(surv_path)
        surv = surv[surv.get("pass_all", True) == True]
        phase_agg = surv.rename(columns={"phase_med": "phase_med"})
    else:
        phase_raw = load_phase_raw(asset)
        if phase_raw.empty:
            print(f"[{asset}] no phase data")
            return pd.DataFrame()
        phase_agg = aggregate_phase(phase_raw)

    phase_raw = load_phase_raw(asset)
    if not phase_raw.empty:
        turnover = phase_raw.groupby("tag")["rebal"].median().rename("phase_rebal_med").reset_index()
        phase_agg = phase_agg.merge(turnover, on="tag", how="left")

    snap_agg = aggregate_snap(load_snap_raw(asset))
    yearly_agg = aggregate_yearly(load_yearly_raw(asset))
    stress_agg = aggregate_stress(load_stress_raw(asset))
    # 신규 (라운드9)
    def _safe_read(p):
        if not os.path.exists(p) or os.path.getsize(p) <= 1:
            return pd.DataFrame()
        try:
            return pd.read_csv(p)
        except pd.errors.EmptyDataError:
            return pd.DataFrame()
    plateau_agg = _safe_read(os.path.join(HERE, f"redesign_plateau_agg_{asset}.csv"))
    bootstrap_agg = _safe_read(os.path.join(HERE, f"redesign_bootstrap_{asset}.csv"))

    df = top500.merge(phase_agg, on="tag", how="inner")
    for other in (snap_agg, yearly_agg, stress_agg, plateau_agg, bootstrap_agg):
        if len(other):
            df = df.merge(other, on="tag", how="left")

    df = apply_hard_gates(df, asset)
    df = df[df["hard_gate_pass"]]
    df = add_nice_penalty(df)
    df = rank_sum(df)

    out = os.path.join(HERE, f"redesign_rank_{asset}.csv")
    df.to_csv(out, index=False)
    print(f"[{asset}] rank: {len(df)} configs → {out}")
    return df


def run_step6(asset, rank_df, top_n=30):
    if len(rank_df) == 0:
        return pd.DataFrame()
    pool = rank_df.head(top_n).copy()
    # snap_days_eq = 실제 시간 기간 (일 단위). iv 가 달라도 시간 일치하면 ensemble 가능.
    # D: 1 bar/day → snap_days = snap. 4h: 6 bars/day → snap_days = snap / 6.
    snap_col = "snap" if "snap" in pool.columns else ("snap_days" if "snap_days" in pool.columns else None)
    if snap_col is None:
        return pd.DataFrame()
    def _snap_days(row):
        snap = int(row[snap_col])
        iv = str(row.get("iv", "D"))
        if iv == "D":
            return snap
        if iv == "4h":
            return snap / 6.0
        return float(snap)
    pool["snap_days_eq"] = pool.apply(_snap_days, axis=1)
    # 부동소수 비교 안전화 (round 0.5 단위)
    pool["snap_days_eq"] = pool["snap_days_eq"].round(2)
    # snap_days_eq + lev 동일 그룹만 묶음 (mixed leverage 시 SAE 가 cfgs[0].lev 만 쓰니 위험)
    if "lev" in pool.columns:
        buckets = pool.groupby(["snap_days_eq", "lev"])
    else:
        buckets = pool.groupby("snap_days_eq")
    pairs = []
    for _, g in buckets:
        if len(g) < 2:
            continue
        tags = g["tag"].tolist()
        for k in (2, 3):
            if len(tags) < k:
                continue
            for combo in itertools.combinations(tags, k):
                pairs.append({
                    "asset": asset, "k": k,
                    "snap": int(g[snap_col].iloc[0]),
                    "snap_days_eq": float(g["snap_days_eq"].iloc[0]) if "snap_days_eq" in g.columns else float(g[snap_col].iloc[0]),
                    "iv": "|".join(sorted(set(g["iv"].astype(str)))) if "iv" in g.columns else "",
                    "members": "|".join(combo),
                    "rank_sum_mean": float(rank_df[rank_df["tag"].isin(combo)]["rank_sum"].mean()),
                })
    return pd.DataFrame(pairs)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asset", required=True, choices=["fut", "spot", "stock"])
    ap.add_argument("--top", type=int, default=30)
    args = ap.parse_args()
    rank_df = run_step5(args.asset)
    ens = run_step6(args.asset, rank_df, args.top)
    out = os.path.join(HERE, f"redesign_ensemble_candidates_{args.asset}.csv")
    ens.to_csv(out, index=False)
    print(f"[{args.asset}] ensemble candidates: {len(ens)} combos → {out}")


if __name__ == "__main__":
    main()
