#!/usr/bin/env python3
"""Test 13: Event signature — top N winners/losers 분석.

Train / Holdout 각 구간에서:
- top 10 best event (pnl_pct 기준)
- top 10 worst event
- 월별 event count 분포
- coin별 event count 분포
- 승률, 평균 승/패 크기
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import TRAIN_END, HOLDOUT_START, FULL_END, load_cached_events

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def analyze(ev: pd.DataFrame, label: str) -> dict:
    if len(ev) == 0:
        return {"label": label, "n": 0}
    ev = ev.copy()
    winners = ev[ev["pnl_pct"] > 0]
    losers = ev[ev["pnl_pct"] <= 0]
    return {
        "label": label,
        "n": len(ev),
        "win_rate": round(len(winners) / len(ev), 3),
        "avg_win_pct": round(float(winners["pnl_pct"].mean()), 3) if len(winners) else 0.0,
        "avg_loss_pct": round(float(losers["pnl_pct"].mean()), 3) if len(losers) else 0.0,
        "sum_pnl": round(float(ev["pnl_pct"].sum()), 3),
        "max_pnl": round(float(ev["pnl_pct"].max()), 3),
        "min_pnl": round(float(ev["pnl_pct"].min()), 3),
        "top5_share": round(float(ev.nlargest(5, "pnl_pct")["pnl_pct"].sum()
                                    / ev["pnl_pct"].sum()), 3)
                      if ev["pnl_pct"].sum() != 0 else 0.0,
        "n_coins": ev["coin"].nunique(),
    }


def main():
    ev_s = load_cached_events("spot")
    ev_f = load_cached_events("fut")

    rows = []
    for name, ev in [("spot_전구간", ev_s), ("fut_전구간", ev_f)]:
        rows.append(analyze(ev, name))
        rows.append(analyze(ev[ev["entry_ts"] <= TRAIN_END], name.replace("_전구간", "_Train")))
        rows.append(analyze(ev[ev["entry_ts"] >= HOLDOUT_START], name.replace("_전구간", "_Holdout")))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test13_event_signature.csv"), index=False)

    # Top/Bottom events in holdout
    print("\n=== SPOT Holdout Top 10 best ===")
    ev_sh = ev_s[ev_s["entry_ts"] >= HOLDOUT_START]
    print(ev_sh.nlargest(10, "pnl_pct")[["coin","entry_ts","exit_ts","pnl_pct","bars_held","reason"]].to_string(index=False))
    print("\n=== SPOT Holdout Top 10 worst ===")
    print(ev_sh.nsmallest(10, "pnl_pct")[["coin","entry_ts","exit_ts","pnl_pct","bars_held","reason"]].to_string(index=False))

    print("\n=== FUT Holdout Top 10 best ===")
    ev_fh = ev_f[ev_f["entry_ts"] >= HOLDOUT_START]
    print(ev_fh.nlargest(10, "pnl_pct")[["coin","entry_ts","exit_ts","pnl_pct","bars_held","reason"]].to_string(index=False))
    print("\n=== FUT Holdout Top 10 worst ===")
    print(ev_fh.nsmallest(10, "pnl_pct")[["coin","entry_ts","exit_ts","pnl_pct","bars_held","reason"]].to_string(index=False))

    ev_sh.to_csv(os.path.join(OUT, "test13_spot_holdout_events.csv"), index=False)
    ev_fh.to_csv(os.path.join(OUT, "test13_fut_holdout_events.csv"), index=False)

    print("\n=== Summary ===")
    print(df.to_string(index=False))
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
