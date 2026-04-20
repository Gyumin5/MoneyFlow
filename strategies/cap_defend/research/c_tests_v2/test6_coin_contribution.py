#!/usr/bin/env python3
"""Test 6: Coin-level contribution — 종목별 realized PnL 기여도 (재설계).

Codex 지적 반영: event.pnl_pct 단순 합은 실제 포트 기여도 아님.
개선:
- 각 event에 대해 "실제 진입 가능했을" 것만 필터 (V21 prev_cash >= cap)
- event별 realized contribution 근사 = pnl_pct × cap × (1.0 if V21 여유 있었으면)
- 그 기반 종목별 sum/mean/win_rate 집계
- top N 집중도 계산

c_engine_v5 pnl_pct는 % 단위 (e.g. 3.5 = 3.5%).
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CAP_SPOT, CAP_FUT_OPTS, HOLDOUT_START, FULL_END,
                     load_all, load_cached_events, slice_v21)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def filter_enterable(ev: pd.DataFrame, v21_slice: pd.DataFrame, cap: float) -> pd.DataFrame:
    """V21 prev_cash가 cap 이상인 날에 발생한 event만 entry 가능 근사."""
    if len(ev) == 0:
        return ev
    ev = ev.copy()
    ev["entry_date"] = pd.to_datetime(ev["entry_ts"]).dt.normalize()
    # 키 normalize (Gemini 지적)
    pc_map = {pd.to_datetime(k).normalize(): float(v)
              for k, v in v21_slice["prev_cash"].to_dict().items()}
    # 엔트리일 prev_cash 기준 진입 가능 여부 + 실제 allocated cap
    def fill_for(d):
        pc = pc_map.get(pd.Timestamp(d), 0.0)
        return min(cap, float(pc))
    ev["alloc_cap"] = ev["entry_date"].apply(fill_for)
    ev = ev[ev["alloc_cap"] > 0.005].copy()  # 0.5% 이상만 진입 가정
    return ev


def summarize(ev: pd.DataFrame, cap: float, label: str, lev: float = 1.0) -> pd.DataFrame:
    """종목별 realized 기여도 집계.

    realized_contribution = (pnl_pct/100) × alloc_cap × lev
    pnl_pct는 % 단위이므로 /100.
    """
    if len(ev) == 0:
        return pd.DataFrame()
    ev = ev.copy()
    ev["contribution"] = (ev["pnl_pct"] / 100.0) * ev["alloc_cap"] * lev
    g = ev.groupby("coin").agg(
        n=("pnl_pct", "count"),
        win_rate=("pnl_pct", lambda s: float((s > 0).mean())),
        avg_pnl_pct=("pnl_pct", "mean"),
        sum_contribution=("contribution", "sum"),
        avg_contribution=("contribution", "mean"),
        max_pnl=("pnl_pct", "max"),
        min_pnl=("pnl_pct", "min"),
    ).round(4)
    g = g.sort_values("sum_contribution", ascending=False)
    g["label"] = label
    return g.reset_index()


def main():
    v21_s, v21_f, hist, avail, cd = load_all()
    ev_s = load_cached_events("spot")
    ev_f = load_cached_events("fut")

    frames = []

    # SPOT
    for period_label, ps, pe in [
        ("전구간", v21_s.index[0], FULL_END),
        ("holdout", HOLDOUT_START, FULL_END),
    ]:
        v21_slice = slice_v21(v21_s, ps, pe)
        ev_sub = ev_s[(ev_s["entry_ts"] >= v21_slice.index[0])
                      & (ev_s["entry_ts"] <= v21_slice.index[-1])]
        ev_ok = filter_enterable(ev_sub, v21_slice, CAP_SPOT)
        frames.append(summarize(ev_ok, CAP_SPOT, f"spot_{period_label}", lev=1.0))

    # FUT — cap 3종
    for cap in CAP_FUT_OPTS:
        for period_label, ps, pe in [
            ("전구간", v21_f.index[0], FULL_END),
            ("holdout", HOLDOUT_START, FULL_END),
        ]:
            v21_slice = slice_v21(v21_f, ps, pe)
            ev_sub = ev_f[(ev_f["entry_ts"] >= v21_slice.index[0])
                          & (ev_f["entry_ts"] <= v21_slice.index[-1])]
            ev_ok = filter_enterable(ev_sub, v21_slice, cap)
            frames.append(summarize(ev_ok, cap, f"fut_cap{cap}_{period_label}", lev=3.0))

    out = pd.concat(frames, ignore_index=True)
    out.to_csv(os.path.join(OUT, "test6_coin_contribution.csv"), index=False)

    # 집중도 요약
    print("=== 종목 집중도 (realized contribution 기준) ===")
    summary_rows = []
    for label in out["label"].unique():
        sub = out[out["label"] == label].sort_values("sum_contribution", ascending=False)
        if sub.empty: continue
        total = sub["sum_contribution"].sum()
        if total == 0: continue
        top5 = sub.head(5)["sum_contribution"].sum() / total
        top10 = sub.head(10)["sum_contribution"].sum() / total
        summary_rows.append({
            "label": label, "n_coins": len(sub), "total_contribution": round(float(total), 4),
            "top5_share": round(float(top5), 3), "top10_share": round(float(top10), 3),
        })
        print(f"\n{label}: coins={len(sub)}, total={total:.4f}, top5={top5:.1%}, top10={top10:.1%}")
        print(sub.head(10).to_string(index=False))

    pd.DataFrame(summary_rows).to_csv(os.path.join(OUT, "test6_summary.csv"), index=False)
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
