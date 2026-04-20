#!/usr/bin/env python3
"""추가 검증 B: dip_bars=30 / dip_thr=-0.20 파라미터 풀기간 재시뮬.

Test 9 plateau top: spot/fut 모두 dip_bars=30 dip_thr=-0.20 우위.
Full/Train/Holdout 3구간 모두에서 현재 설정 대비 검증.
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (P_SPOT, P_FUT, CAP_SPOT, CAP_FUT_OPTS,
                     HOLDOUT_START, TRAIN_END, FULL_END,
                     load_all, extract_events, slice_v21,
                     run_spot_combo, run_fut_combo)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def run_split(name, v21, ev, sim, cap):
    rows = []
    for span, s, e in [("full", v21.index[0], FULL_END),
                        ("train", v21.index[0], TRAIN_END),
                        ("holdout", HOLDOUT_START, FULL_END)]:
        v21s = slice_v21(v21, s, e)
        evs = ev[(ev["entry_ts"] >= v21s.index[0]) & (ev["entry_ts"] <= v21s.index[-1])].copy()
        _, st = sim(evs, kwargs["cd"], v21s, kwargs["hist"], cap)
        rows.append({"label": name, "span": span,
                     "Cal": round(st["Cal"], 3),
                     "CAGR": round(st["CAGR"], 4),
                     "MDD": round(st["MDD"], 4),
                     "n_entries": st["n_entries"]})
    return rows


def main():
    global kwargs
    v21_s, v21_f, hist, avail, cd = load_all()
    kwargs = {"cd": cd, "hist": hist}

    # 기존 파라미터 이벤트
    ev_s_cur = extract_events(avail, P_SPOT)
    ev_f_cur = extract_events(avail, P_FUT)

    # 신규 파라미터 (Test 9 plateau top)
    P_SPOT_NEW = {"dip_bars":30, "dip_thr":-0.20, "tp":0.05, "tstop":24}
    P_FUT_NEW  = {"dip_bars":30, "dip_thr":-0.20, "tp":0.10, "tstop":48}
    print(f"현재 spot P: {P_SPOT}")
    print(f"신규 spot P: {P_SPOT_NEW}")
    print(f"현재 fut  P: {P_FUT}")
    print(f"신규 fut  P: {P_FUT_NEW}")

    print("[1/2] 신규 spot events 추출...")
    ev_s_new = extract_events(avail, P_SPOT_NEW)
    print(f"  신규 spot: {len(ev_s_new)} events")
    print("[2/2] 신규 fut events 추출...")
    ev_f_new = extract_events(avail, P_FUT_NEW)
    print(f"  신규 fut: {len(ev_f_new)} events")

    rows = []
    # Spot
    rows += run_split("spot_cur", v21_s, ev_s_cur, run_spot_combo, CAP_SPOT)
    rows += run_split("spot_new", v21_s, ev_s_new, run_spot_combo, CAP_SPOT)
    # Fut 3 caps
    for cap in CAP_FUT_OPTS:
        rows += run_split(f"fut_cur_cap{cap}", v21_f, ev_f_cur, run_fut_combo, cap)
        rows += run_split(f"fut_new_cap{cap}", v21_f, ev_f_new, run_fut_combo, cap)

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "addl_v2_dipbars30.csv"), index=False)

    print("\n=== SPOT 비교 (Cal) ===")
    print(df[df["label"].str.startswith("spot_")]
            .pivot_table(index="label", columns="span", values="Cal").to_string())
    print("\n=== FUT 비교 (Cal) ===")
    print(df[df["label"].str.startswith("fut_")]
            .pivot_table(index="label", columns="span", values="Cal").to_string())

    print(f"\n저장: {OUT}/addl_v2_dipbars30.csv")


if __name__ == "__main__":
    main()
