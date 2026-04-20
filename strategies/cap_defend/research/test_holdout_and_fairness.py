#!/usr/bin/env python3
"""Phase-3 top k=1 후보 재검증 테스트.

목적 (AI 리뷰 대응):
1. k=1 (phase2 mCal 복사) vs k≥2 (full-period 재시뮬) 공정 비교 (Codex 지적)
2. Train(2020-10-01~2023-12-31) / Holdout(2024-01-01~) 분리로 selection leakage 추정
3. top 후보간 realized daily-return 상관 (near-duplicate 필터 한계 검증)

출력: test_holdout_fairness/
  - kequal1_fair.csv: tag, mCal, fairCal_full, Cal_train, Cal_holdout, rank_shift
  - pairwise_corr.csv: tag1, tag2, corr
"""
from __future__ import annotations
import os, sys, json
import pandas as pd
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from phase_common import parse_tag, run_single_target, equity_metrics

OUT_DIR = os.path.join(HERE, "test_holdout_fairness")
os.makedirs(OUT_DIR, exist_ok=True)

ANCHOR = "2020-10-01"
FULL_END = "2026-04-13"
TRAIN_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"

TOP_N_SPOT = 5
TOP_N_FUT_PER_LEV = 3


def pick_candidates():
    spot = pd.read_csv(os.path.join(HERE, "phase3_ensembles", "spot_top.csv"))
    fut = pd.read_csv(os.path.join(HERE, "phase3_ensembles", "fut_top.csv"))
    spot_k1 = spot[spot["k"] == 1].head(TOP_N_SPOT).copy()
    fut_k1 = fut[fut["k"] == 1].groupby("lev").head(TOP_N_FUT_PER_LEV).copy()
    picks = []
    for _, r in spot_k1.iterrows():
        picks.append({"tag": r["members"], "asset": "spot", "lev": 1.0,
                      "original_Cal": float(r["Cal"])})
    for _, r in fut_k1.iterrows():
        picks.append({"tag": r["members"], "asset": "fut", "lev": float(r["lev"]),
                      "original_Cal": float(r["Cal"])})
    return picks


def slice_metrics(eq: pd.Series, start: str | None, end: str | None) -> dict:
    eq = eq.dropna()
    if not isinstance(eq.index, pd.DatetimeIndex):
        eq.index = pd.to_datetime(eq.index)
    if start:
        eq = eq[eq.index >= start]
    if end:
        eq = eq[eq.index <= end]
    if len(eq) < 30:
        return {"Cal": 0.0, "CAGR": 0.0, "MDD": 0.0, "Sh": 0.0, "n": len(eq)}
    eq_norm = eq / eq.iloc[0]
    m = equity_metrics(eq_norm)
    m["n"] = len(eq)
    return m


def main():
    picks = pick_candidates()
    print(f"Candidates: {len(picks)} ({sum(1 for p in picks if p['asset']=='spot')} spot, "
          f"{sum(1 for p in picks if p['asset']=='fut')} fut)")

    rows = []
    equities = {}
    for i, p in enumerate(picks, 1):
        tag = p["tag"]
        print(f"[{i}/{len(picks)}] {tag} (lev={p['lev']})")
        cfg_meta = parse_tag(tag)
        cfg = {k: cfg_meta[k] for k in
               ("interval", "sma", "ms", "ml", "vol_mode", "vol_thr", "snap")}
        try:
            res = run_single_target(p["asset"], cfg, lev=p["lev"], anchor=ANCHOR,
                                    end=FULL_END, want_equity=True)
        except Exception as e:
            print(f"  error: {e}")
            rows.append({"tag": tag, "asset": p["asset"], "lev": p["lev"],
                         "original_Cal": p["original_Cal"], "error": str(e)[:120]})
            continue
        eq = res.get("_equity")
        if eq is None:
            continue
        if not isinstance(eq, pd.Series):
            eq = pd.Series(eq)
        if not isinstance(eq.index, pd.DatetimeIndex):
            eq.index = pd.to_datetime(eq.index)
        equities[tag] = eq

        m_full = slice_metrics(eq, ANCHOR, FULL_END)
        m_train = slice_metrics(eq, ANCHOR, TRAIN_END)
        m_holdout = slice_metrics(eq, HOLDOUT_START, FULL_END)

        row = {
            "tag": tag, "asset": p["asset"], "lev": p["lev"],
            "original_Cal": p["original_Cal"],
            "fairCal_full": round(m_full.get("Cal", 0), 4),
            "Cal_train": round(m_train.get("Cal", 0), 4),
            "Cal_holdout": round(m_holdout.get("Cal", 0), 4),
            "CAGR_full": round(m_full.get("CAGR", 0), 4),
            "CAGR_holdout": round(m_holdout.get("CAGR", 0), 4),
            "MDD_full": round(m_full.get("MDD", 0), 4),
            "MDD_holdout": round(m_holdout.get("MDD", 0), 4),
            "n_train": m_train.get("n", 0),
            "n_holdout": m_holdout.get("n", 0),
        }
        rows.append(row)
        print(f"  orig(mCal)={row['original_Cal']:.2f} fairCal={row['fairCal_full']:.2f} "
              f"train={row['Cal_train']:.2f} holdout={row['Cal_holdout']:.2f}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT_DIR, "kequal1_fair.csv"), index=False)
    print(f"\n=== kequal1_fair.csv 저장 ({len(df)}행) ===")
    print(df.to_string(index=False))

    # Rank shift analysis (within each asset/lev group)
    print("\n=== 랭킹 시프트 (train vs holdout) ===")
    for (asset, lev), sub in df.groupby(["asset", "lev"]):
        if sub.empty or "Cal_train" not in sub.columns:
            continue
        s = sub.copy()
        s["train_rank"] = s["Cal_train"].rank(ascending=False).astype(int)
        s["holdout_rank"] = s["Cal_holdout"].rank(ascending=False).astype(int)
        s["shift"] = (s["train_rank"] - s["holdout_rank"]).abs()
        print(f"\n-- {asset} lev={lev} --")
        print(s[["tag", "Cal_train", "Cal_holdout",
                 "train_rank", "holdout_rank", "shift"]].to_string(index=False))

    # Pairwise daily-return correlation
    print("\n=== Pairwise daily-return corr ===")
    if len(equities) >= 2:
        ret_df = pd.DataFrame({t: eq.resample("D").last().pct_change().fillna(0)
                               for t, eq in equities.items()})
        ret_df = ret_df.dropna(how="all")
        corr = ret_df.corr()
        corr.to_csv(os.path.join(OUT_DIR, "pairwise_corr.csv"))
        # 각 asset 내 high-corr 쌍
        for asset in ("spot", "fut"):
            tags = [p["tag"] for p in picks if p["asset"] == asset
                    and p["tag"] in equities]
            if len(tags) < 2:
                continue
            sub = corr.loc[tags, tags]
            pairs = []
            for i in range(len(tags)):
                for j in range(i + 1, len(tags)):
                    pairs.append((tags[i], tags[j], float(sub.iloc[i, j])))
            pairs.sort(key=lambda x: -x[2])
            print(f"\n-- {asset} ({len(pairs)}쌍) --")
            for t1, t2, c in pairs[:10]:
                flag = " ⚠️" if c >= 0.85 else ""
                print(f"  {c:.3f}{flag}  {t1[:50]}  vs  {t2[:50]}")
            n_high = sum(1 for _, _, c in pairs if c >= 0.85)
            print(f"  (>=0.85: {n_high}/{len(pairs)}, >=0.95: "
                  f"{sum(1 for _,_,c in pairs if c>=0.95)}/{len(pairs)})")

    print(f"\n결과 저장: {OUT_DIR}/")


if __name__ == "__main__":
    main()
