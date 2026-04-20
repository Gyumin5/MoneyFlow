#!/usr/bin/env python3
"""추가 검증 E: Portfolio Circuit Breaker 시뮬.

로직:
  V21+C port equity 일별 관찰 → 최근 고점 대비 drawdown 계산
  DD ≥ threshold (-15/-20/-25%) 도달 시:
    → 이후 N일(cooldown) 동안 신규 entries 차단
    → DD 회복 (고점 50% 복귀) 시 재진입 허용
  기존 포지션은 유지 (강제 close 아님)

구현: simulate_fut 직접 수정 대신 events 사전 필터링 근사
  1. V21+C port_equity 일별 series 획득 (circuit 없는 모드)
  2. 각 날짜 DD 계산
  3. DD 악화 구간의 entries 제거 (차단 시뮬)
  4. 재시뮬
"""
from __future__ import annotations
import os, sys
import pandas as pd
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CAP_FUT_OPTS, TRAIN_END, HOLDOUT_START, FULL_END,
                     load_all, load_cached_events, slice_v21,
                     run_fut_combo)

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def apply_circuit(ev: pd.DataFrame, port_eq: pd.Series,
                  threshold: float, recovery: float = 0.5) -> pd.DataFrame:
    """port_eq DD가 threshold 이하로 떨어지면 blocked,
    고점 대비 recovery 수준까지 회복할 때까지 entries 차단.
    """
    if len(ev) == 0:
        return ev
    idx = pd.to_datetime(port_eq.index)
    if getattr(idx, "tz", None) is not None:
        idx = idx.tz_localize(None)
    eq = pd.Series(port_eq.values, index=idx)
    peak = eq.cummax()
    dd = eq / peak - 1.0

    # 차단 구간 식별
    blocked = []
    in_block = False
    block_peak_ref = 0.0
    for ts, d in dd.items():
        p = peak.loc[ts]
        if not in_block and d <= threshold:
            in_block = True
            block_peak_ref = p * (1 + threshold * recovery)  # 회복 기준
        elif in_block and eq.loc[ts] >= block_peak_ref:
            in_block = False
        blocked.append(in_block)
    block_s = pd.Series(blocked, index=idx)

    ev = ev.copy()
    ev["entry_date"] = pd.to_datetime(ev["entry_ts"]).dt.normalize()
    ev_ok = ev[~ev["entry_date"].map(lambda d: block_s.get(d, False))].copy()
    n_blocked = len(ev) - len(ev_ok)
    return ev_ok, n_blocked


def main():
    v21_s, v21_f, hist, avail, cd = load_all()
    ev_f = load_cached_events("fut")
    print(f"Base fut events: {len(ev_f)}")

    rows = []
    for cap in [0.12, 0.20, 0.25, 0.30]:
        # 1. baseline port equity (circuit 없음)
        v21_full = slice_v21(v21_f, v21_f.index[0], FULL_END)
        ev_full = ev_f[(ev_f["entry_ts"] >= v21_full.index[0])
                       & (ev_f["entry_ts"] <= v21_full.index[-1])].copy()
        port_eq, st_base = run_fut_combo(ev_full, cd, v21_full.copy(), hist, cap)

        for thresh in [None, -0.15, -0.20, -0.25]:
            if thresh is None:
                ev_filt = ev_full
                n_blocked = 0
            else:
                ev_filt, n_blocked = apply_circuit(ev_full, port_eq, thresh, recovery=0.5)

            # split metrics
            for span, s, e in [("full", v21_f.index[0], FULL_END),
                                ("train", v21_f.index[0], TRAIN_END),
                                ("holdout", HOLDOUT_START, FULL_END)]:
                v21s = slice_v21(v21_f, s, e)
                evs = ev_filt[(ev_filt["entry_ts"] >= v21s.index[0])
                              & (ev_filt["entry_ts"] <= v21s.index[-1])].copy()
                _, st = run_fut_combo(evs, cd, v21s.copy(), hist, cap)
                thr_lbl = "none" if thresh is None else f"cb{int(thresh*100)}"
                rows.append({
                    "cap": cap, "thresh": thr_lbl, "span": span,
                    "Cal": round(st["Cal"], 3),
                    "CAGR": round(st["CAGR"], 4),
                    "MDD": round(st["MDD"], 4),
                    "n_entries": st["n_entries"],
                    "n_blocked": n_blocked,
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "addl_v5_circuit.csv"), index=False)

    # pivot
    print("\n=== Cal by (cap, thresh, span) ===")
    pv = df.pivot_table(index=["cap", "thresh"], columns="span",
                         values="Cal", aggfunc="first")
    print(pv.to_string())
    print("\n=== MDD ===")
    pv2 = df.pivot_table(index=["cap", "thresh"], columns="span",
                          values="MDD", aggfunc="first")
    print(pv2.to_string())
    print("\n=== n_blocked (총 차단된 events) ===")
    sub_blk = df[df["span"] == "full"][["cap", "thresh", "n_blocked"]].drop_duplicates()
    print(sub_blk.to_string(index=False))

    print(f"\n저장: {OUT}/addl_v5_circuit.csv")


if __name__ == "__main__":
    main()
