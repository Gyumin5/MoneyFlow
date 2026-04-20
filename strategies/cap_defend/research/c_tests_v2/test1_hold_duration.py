#!/usr/bin/env python3
"""Test 1: C trade 보유시간 분포 (tp 도달 vs tstop 강제청산).

목적: stop-loss 없는 특성 검증. 반등 실패 시 얼마나 오래 물리는가?
출력: c_tests_v2/out/test1_hold_duration.csv + summary.txt
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (P_SPOT, P_FUT, TRAIN_END, HOLDOUT_START, FULL_END,
                     load_all, extract_events)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def analyze(ev: pd.DataFrame, label: str) -> dict:
    """각 event의 reason/hold_bars/pnl_pct 분포."""
    # c_engine_v5 returns: entry_ts, exit_ts, entry_px, exit_px, pnl_pct, bars_held, reason
    if len(ev) == 0:
        return {"label": label, "n": 0}
    # c_engine_v5 reason: "TP" / "timeout". 요약에서는 tp / tstop으로 정규화.
    ev = ev.copy()
    ev["reason_norm"] = ev["reason"].map({"TP": "tp", "timeout": "tstop"}).fillna(ev["reason"].astype(str).str.lower())
    reason_counts = ev["reason_norm"].value_counts().to_dict()
    tp = ev[ev["reason_norm"] == "tp"]
    tstop = ev[ev["reason_norm"] == "tstop"]

    out = {
        "label": label,
        "n_total": len(ev),
        "n_tp": len(tp),
        "n_tstop": len(tstop),
        "tp_rate": len(tp) / len(ev) if len(ev) else 0,
        "avg_hold_bars": float(ev["bars_held"].mean()) if "bars_held" in ev else None,
        "avg_hold_bars_tp": float(tp["bars_held"].mean()) if len(tp) else None,
        "avg_hold_bars_tstop": float(tstop["bars_held"].mean()) if len(tstop) else None,
        "avg_pnl_tp": float(tp["pnl_pct"].mean()) if len(tp) else None,
        "avg_pnl_tstop": float(tstop["pnl_pct"].mean()) if len(tstop) else None,
        "worst_pnl": float(ev["pnl_pct"].min()),
        "p5_pnl": float(np.percentile(ev["pnl_pct"], 5)),
        "p95_pnl": float(np.percentile(ev["pnl_pct"], 95)),
        "reasons": str(reason_counts),
    }
    return out


def main():
    v21_s, v21_f, hist, avail, cd = load_all()

    ev_s = extract_events(avail, P_SPOT)
    ev_f = extract_events(avail, P_FUT)

    # split by period
    rows = []
    for label, ev, ps, pe in [
        ("SPOT_전구간", ev_s, "2020-10-01", FULL_END),
        ("SPOT_Train",  ev_s, "2020-10-01", TRAIN_END),
        ("SPOT_Holdout", ev_s, HOLDOUT_START, FULL_END),
        ("FUT_전구간", ev_f, "2020-10-01", FULL_END),
        ("FUT_Train",  ev_f, "2020-10-01", TRAIN_END),
        ("FUT_Holdout", ev_f, HOLDOUT_START, FULL_END),
    ]:
        sub = ev[(ev["entry_ts"] >= pd.Timestamp(ps)) & (ev["entry_ts"] <= pd.Timestamp(pe))]
        rows.append(analyze(sub, label))

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test1_hold_duration.csv"), index=False)

    # Raw events for further analysis
    ev_s.to_csv(os.path.join(OUT, "test1_events_spot.csv"), index=False)
    ev_f.to_csv(os.path.join(OUT, "test1_events_fut.csv"), index=False)

    with open(os.path.join(OUT, "test1_summary.txt"), "w") as f:
        f.write("=== Test 1: Hold Duration Distribution ===\n\n")
        f.write(df.to_string(index=False))
        f.write("\n\n--- 해석 포인트 ---\n")
        f.write("- tp_rate가 낮으면 tstop 강제청산 비율 높음 → 반등 실패 많음\n")
        f.write("- avg_hold_bars_tstop: tstop 도달까지 평균 시간 (선물은 48, 현물은 24)\n")
        f.write("- avg_pnl_tstop: 실패 trade 평균 손실 (stop-loss 없으므로 실제 손실)\n")
        f.write("- worst_pnl / p5_pnl: 최악/5% 분위 손실\n")
    print(df.to_string(index=False))
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
