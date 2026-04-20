#!/usr/bin/env python3
"""Test 10: Walk-forward rolling — 여러 split point에서 Cal 안정성.

train_end 여러 지점으로 옮겨가며 V21+C 성과 비교.
파라미터 freeze 상태에서 경로 의존성 확인.
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CAP_SPOT, CAP_FUT_OPTS, FULL_END,
                     load_all, load_cached_events, slice_v21,
                     run_spot_combo, run_fut_combo)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def main():
    v21_s, v21_f, hist, avail, cd = load_all()
    ev_s = load_cached_events("spot")
    ev_f = load_cached_events("fut")

    splits = [
        ("2023-01", "2022-12-31", "2023-01-01"),
        ("2023-07", "2023-06-30", "2023-07-01"),
        ("2024-01", "2023-12-31", "2024-01-01"),
        ("2024-07", "2024-06-30", "2024-07-01"),
        ("2025-01", "2024-12-31", "2025-01-01"),
    ]

    rows = []
    for sname, te, hs in splits:
        for label, v21, ev, sim, cap in [
            ("spot", v21_s, ev_s, run_spot_combo, CAP_SPOT),
            ("fut_cap0.12", v21_f, ev_f, run_fut_combo, 0.12),
            ("fut_cap0.25", v21_f, ev_f, run_fut_combo, 0.25),
            ("fut_cap0.30", v21_f, ev_f, run_fut_combo, 0.30),
        ]:
            # train
            v21_t = slice_v21(v21, v21.index[0], pd.Timestamp(te))
            ev_t = ev[(ev["entry_ts"] >= v21_t.index[0]) & (ev["entry_ts"] <= v21_t.index[-1])].copy()
            _, st_t = sim(ev_t, cd, v21_t, hist, cap)
            # holdout
            v21_h = slice_v21(v21, pd.Timestamp(hs), FULL_END)
            ev_h = ev[(ev["entry_ts"] >= v21_h.index[0]) & (ev["entry_ts"] <= v21_h.index[-1])].copy()
            _, st_h = sim(ev_h, cd, v21_h, hist, cap)
            rows.append({
                "split": sname, "asset": label, "cap": cap,
                "train_Cal": round(st_t["Cal"], 3),
                "train_CAGR": round(st_t["CAGR"], 4),
                "train_MDD": round(st_t["MDD"], 4),
                "hout_Cal": round(st_h["Cal"], 3),
                "hout_CAGR": round(st_h["CAGR"], 4),
                "hout_MDD": round(st_h["MDD"], 4),
                "hout_entries": st_h["n_entries"],
            })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test10_walkforward.csv"), index=False)
    print(df.to_string(index=False))
    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
