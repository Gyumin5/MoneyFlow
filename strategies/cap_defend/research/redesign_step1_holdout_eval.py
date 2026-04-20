#!/usr/bin/env python3
"""Redesign Step 1: 43개 phase3 top 후보 전부 train/holdout 분리 재평가.

- 입력: phase3_ensembles/spot_top.csv + fut_top.csv (union)
- 각 후보 공정 재시뮬 (run_single_target for k=1, run_combo logic for k>=2)
- Train(~2023-12-31) / Holdout(2024-01-01~) Cal 각각 계산
- Realized daily-return corr matrix 저장
- 출력: redesign_step1/holdout_eval.csv, pairwise_corr.csv
"""
from __future__ import annotations
import os, sys, time
import pandas as pd
import numpy as np

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)
from phase_common import (parse_tag, run_single_target, run_spot_ensemble,
                          equity_metrics, preload_futures, build_trace, FULL_END)

OUT = os.path.join(HERE, "redesign_step1")
os.makedirs(OUT, exist_ok=True)

ANCHOR = "2020-10-01"
TRAIN_END = "2023-12-31"
HOLDOUT_START = "2024-01-01"


def slice_cal(eq: pd.Series, start=None, end=None) -> dict:
    eq = eq.dropna()
    if not isinstance(eq.index, pd.DatetimeIndex):
        eq.index = pd.to_datetime(eq.index)
    if getattr(eq.index, 'tz', None) is not None:
        eq.index = eq.index.tz_localize(None)
    if start:
        eq = eq[eq.index >= start]
    if end:
        eq = eq[eq.index <= end]
    if len(eq) < 30 or eq.iloc[0] <= 0:
        return {"Cal": 0.0, "CAGR": 0.0, "MDD": 0.0, "n": len(eq)}
    eq_n = eq / eq.iloc[0]
    m = equity_metrics(eq_n)
    m["n"] = len(eq)
    return m


def run_candidate(row: pd.Series) -> dict:
    asset = row["asset"]
    lev = float(row["lev"])
    k = int(row["k"])
    members = str(row["members"]).split(";")
    ens_tag = row["ensemble_tag"]

    if k == 1:
        meta = parse_tag(members[0])
        cfg = {kk: meta[kk] for kk in ("interval","sma","ms","ml","vol_mode","vol_thr","snap")}
        res = run_single_target(asset, cfg, lev=lev, anchor=ANCHOR, end=FULL_END, want_equity=True)
    else:
        weights = {m: 1.0/k for m in members}
        if asset == "fut":
            from futures_ensemble_engine import SingleAccountEngine, combine_targets
            data = preload_futures()
            bars_1h, funding_1h = data["1h"]
            all_dates = bars_1h["BTC"].index
            traces = {}
            for m_tag in members:
                mt = parse_tag(m_tag)
                cfg = {kk: mt[kk] for kk in ("interval","sma","ms","ml","vol_mode","vol_thr","snap")}
                tr = build_trace("fut", cfg, lev, ANCHOR, end=FULL_END)["trace"]
                traces[m_tag] = tr
            dates = all_dates[(all_dates >= ANCHOR) & (all_dates <= FULL_END)]
            combined = combine_targets(traces, weights, dates)
            engine = SingleAccountEngine(
                bars_1h, funding_1h, leverage=lev, leverage_mode="fixed",
                per_coin_leverage_mode="none", stop_kind="none", stop_pct=0.0,
                stop_lookback_bars=0, stop_gate="always")
            res = engine.run(combined)
        else:
            member_cfgs = {}
            for m_tag in members:
                mt = parse_tag(m_tag)
                member_cfgs[m_tag] = {kk: mt[kk] for kk in
                                      ("interval","sma","ms","ml","vol_mode","vol_thr","snap")}
            res = run_spot_ensemble(member_cfgs, weights, ANCHOR, end=FULL_END, want_equity=True)

    eq = res.get("_equity")
    if eq is None:
        return {"ensemble_tag": ens_tag, "error": "no_equity"}
    if not isinstance(eq, pd.Series):
        eq = pd.Series(eq)

    m_full = slice_cal(eq, ANCHOR, FULL_END)
    m_train = slice_cal(eq, ANCHOR, TRAIN_END)
    m_holdout = slice_cal(eq, HOLDOUT_START, FULL_END)

    return {
        "ensemble_tag": ens_tag, "asset": asset, "lev": lev, "k": k,
        "members": row["members"],
        "original_Cal": float(row.get("Cal", 0)),
        "Cal_full": round(m_full.get("Cal", 0), 4),
        "CAGR_full": round(m_full.get("CAGR", 0), 4),
        "MDD_full": round(m_full.get("MDD", 0), 4),
        "Cal_train": round(m_train.get("Cal", 0), 4),
        "CAGR_train": round(m_train.get("CAGR", 0), 4),
        "Cal_holdout": round(m_holdout.get("Cal", 0), 4),
        "CAGR_holdout": round(m_holdout.get("CAGR", 0), 4),
        "MDD_holdout": round(m_holdout.get("MDD", 0), 4),
        "n_train": m_train.get("n", 0),
        "n_holdout": m_holdout.get("n", 0),
        "_eq": eq,  # keep for corr
    }


def main():
    spot = pd.read_csv(os.path.join(HERE, "phase3_ensembles", "spot_top.csv"))
    fut = pd.read_csv(os.path.join(HERE, "phase3_ensembles", "fut_top.csv"))
    cands = pd.concat([spot, fut], ignore_index=True)
    cands = cands[cands["status"] == "ok"].copy()
    print(f"Candidates: {len(cands)} (spot={len(spot)}, fut={len(fut)})")

    rows = []
    equities = {}
    t0 = time.time()
    for i, (_, row) in enumerate(cands.iterrows(), 1):
        print(f"[{i}/{len(cands)}] {row['ensemble_tag']} (k={row['k']}, lev={row['lev']}) "
              f"elapsed={time.time()-t0:.0f}s")
        try:
            res = run_candidate(row)
        except Exception as e:
            print(f"  ERROR: {e}")
            rows.append({"ensemble_tag": row["ensemble_tag"], "error": str(e)[:120]})
            continue
        if "_eq" in res:
            eq = res.pop("_eq")
            idx = pd.to_datetime(eq.index)
            if getattr(idx, 'tz', None) is not None:
                idx = idx.tz_localize(None)
            eq.index = idx
            equities[res["ensemble_tag"]] = eq.resample("D").last().pct_change().fillna(0)
        rows.append(res)
        # incremental save
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "holdout_eval.csv"), index=False)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "holdout_eval.csv"), index=False)
    print(f"\nSaved holdout_eval.csv ({len(df)} rows)")

    # Pairwise corr
    if len(equities) >= 2:
        ret_df = pd.DataFrame(equities).dropna(how="all")
        corr = ret_df.corr()
        corr.to_csv(os.path.join(OUT, "pairwise_corr.csv"))
        print(f"Saved pairwise_corr.csv ({corr.shape})")


if __name__ == "__main__":
    main()
