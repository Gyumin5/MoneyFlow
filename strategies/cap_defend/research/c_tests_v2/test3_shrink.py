#!/usr/bin/env python3
"""Test 3: Shrink frequency — 실전 cap 달성률 측정 (재설계).

Codex 지적 반영: "days_cash_below_cap_pct" 단독 지표는 약함.
개선:
- entry일 기준 actual_alloc / target_alloc 추정 비율
- 활성 포지션일 기준 forced_zero/shrink 발생률
- V21 prev_cash 분포 (평균/p5/p50/p95)
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CAP_SPOT, CAP_FUT_OPTS, HOLDOUT_START, FULL_END,
                     load_all, load_cached_events, slice_v21,
                     run_spot_combo, run_fut_combo)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def estimate_fill_ratio(v21_slice: pd.DataFrame, events: pd.DataFrame,
                         cap: float) -> dict:
    """Entry일의 V21 prev_cash vs target cap으로 평균 달성률 추정.

    actual_alloc_est = min(cap, prev_cash) × port
    target_alloc = cap × port
    fill_ratio = actual / target
    """
    if len(events) == 0:
        return {"n_entry_days": 0}
    ev = events.copy()
    ev["entry_date"] = pd.to_datetime(ev["entry_ts"]).dt.normalize()
    entry_dates = ev["entry_date"].unique()

    # 키 normalize 방어 (Gemini 지적): datetime → midnight Timestamp
    pc_map = {pd.to_datetime(k).normalize(): float(v)
              for k, v in v21_slice["prev_cash"].to_dict().items()}
    fill_ratios = []
    full_fill_count = 0
    partial_count = 0
    zero_count = 0
    prev_cash_samples = []
    for d in entry_dates:
        if pd.Timestamp(d) not in pc_map:
            continue
        pc = float(pc_map[pd.Timestamp(d)])
        prev_cash_samples.append(pc)
        actual = min(cap, pc)
        target = cap
        ratio = actual / target if target > 0 else 0
        fill_ratios.append(ratio)
        if ratio >= 0.99: full_fill_count += 1
        elif ratio > 0.01: partial_count += 1
        else: zero_count += 1

    if not fill_ratios:
        return {"n_entry_days": 0}

    return {
        "n_entry_days": len(fill_ratios),
        "avg_fill_ratio": round(float(np.mean(fill_ratios)), 3),
        "p5_fill_ratio": round(float(np.percentile(fill_ratios, 5)), 3),
        "p50_fill_ratio": round(float(np.percentile(fill_ratios, 50)), 3),
        "full_fill_rate": round(full_fill_count / len(fill_ratios), 3),
        "partial_rate": round(partial_count / len(fill_ratios), 3),
        "zero_rate": round(zero_count / len(fill_ratios), 3),
        "avg_prev_cash": round(float(np.mean(prev_cash_samples)), 3),
        "min_prev_cash": round(float(min(prev_cash_samples)), 3),
    }


def main():
    v21_s, v21_f, hist, avail, cd = load_all()
    ev_s = load_cached_events("spot")
    ev_f = load_cached_events("fut")

    rows = []
    for label, v21, ev, sim, cap in [
        ("spot_full", v21_s, ev_s, run_spot_combo, CAP_SPOT),
        ("spot_holdout", v21_s, ev_s, run_spot_combo, CAP_SPOT),
        ("fut_cap0.12_full", v21_f, ev_f, run_fut_combo, 0.12),
        ("fut_cap0.12_holdout", v21_f, ev_f, run_fut_combo, 0.12),
        ("fut_cap0.25_full", v21_f, ev_f, run_fut_combo, 0.25),
        ("fut_cap0.25_holdout", v21_f, ev_f, run_fut_combo, 0.25),
        ("fut_cap0.30_full", v21_f, ev_f, run_fut_combo, 0.30),
        ("fut_cap0.30_holdout", v21_f, ev_f, run_fut_combo, 0.30),
    ]:
        if "holdout" in label:
            v21_slice = slice_v21(v21, HOLDOUT_START, FULL_END)
        else:
            v21_slice = slice_v21(v21, v21.index[0], FULL_END)
        ev_sub = ev[(ev["entry_ts"] >= v21_slice.index[0])
                    & (ev["entry_ts"] <= v21_slice.index[-1])].copy()

        # 엔진 실행 → 실제 stats
        _, stats = sim(ev_sub, cd, v21_slice, hist, cap)
        # fill ratio 추정
        fr = estimate_fill_ratio(v21_slice, ev_sub, cap)

        rows.append({
            "label": label,
            "cap": cap,
            "n_events": len(ev_sub),
            "n_entries": stats["n_entries"],
            "entry_rate": round(stats["n_entries"] / max(len(ev_sub), 1), 3),
            "n_shrinks": stats["n_shrinks"],
            "n_expands": stats.get("n_expands", 0),
            "n_forced_zero": stats.get("n_forced_zero", 0),
            "n_liq": stats.get("n_liquidations", 0),
            "shrink_per_entry": round(stats["n_shrinks"] / max(stats["n_entries"], 1), 3),
            **fr,
            "Cal": round(stats["Cal"], 3),
            "MDD": round(stats["MDD"], 4),
        })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test3_shrink.csv"), index=False)
    print(df.to_string(index=False))
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
