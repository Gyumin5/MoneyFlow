#!/usr/bin/env python3
"""Group F — robustness.

F2. tp 변경 (2/6/10/12%): 기존 events에서 tp만 바꿀 수 없으니 event-level 시뮬로 근사
    — tp 달성 여부는 bars_held 구간에서 High 탐색 필요 → 재계산
F3. universe Top10/20/30 — simulate(_fut) 의 universe_size 파라미터로 변경
F4. 연도별 분리 성과 — v21/events를 각 연도로 슬라이스하여 별도 시뮬
"""
from __future__ import annotations
import os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from _common3 import (
    load_all, load_cached_events,
    CAP_SPOT, CAP_FUT_OPTS, OUT, slice_v21,
    _get_bars_span,
)
sys.path.insert(0, os.path.abspath(os.path.join(HERE, "..")))
from m3_engine_final import simulate
from m3_engine_futures import simulate_fut

import pandas as pd


def apply_custom_tp(ev: pd.DataFrame, tp_new: float, tstop: int) -> pd.DataFrame:
    """events의 bars_held 구간에서 tp_new 달성 여부 재계산.
    달성 안 하면 원 exit 유지 (tstop 기준 유지).
    """
    if len(ev) == 0:
        return ev
    ev = ev.copy()
    ev["entry_ts"] = pd.to_datetime(ev["entry_ts"])
    ev["exit_ts"]  = pd.to_datetime(ev["exit_ts"])
    new_pnls, new_exits, new_reasons = [], [], []
    for _, r in ev.iterrows():
        sub = _get_bars_span(r["coin"], r["entry_ts"], r["exit_ts"])
        if sub is None or len(sub) == 0:
            new_pnls.append(r["pnl_pct"]); new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", "")); continue
        tp_price = float(r["entry_px"]) * (1 + tp_new)
        hit = sub[sub["High"] >= tp_price]
        if len(hit) > 0:
            new_pnls.append(round(tp_new * 100, 3))
            new_exits.append(hit.index[0])
            new_reasons.append("TP_custom")
        else:
            new_pnls.append(r["pnl_pct"])
            new_exits.append(r["exit_ts"])
            new_reasons.append(r.get("reason", ""))
    ev["pnl_pct"] = new_pnls
    ev["exit_ts"] = new_exits
    ev["reason"]  = new_reasons
    return ev


def run_universe(kind, ev, v21, cd, hist, cap, uni_size, lev=3.0):
    from _common3 import TRAIN_END, HOLDOUT_START, FULL_END
    runner = simulate_fut if kind == "fut" else simulate
    rows = []
    for span, start, end in [
        ("full",    v21.index[0], FULL_END),
        ("train",   v21.index[0], TRAIN_END),
        ("holdout", HOLDOUT_START, FULL_END),
    ]:
        v21s = slice_v21(v21, start, end)
        if v21s is None: continue
        ev_s = ev[(pd.to_datetime(ev["entry_ts"]) >= v21s.index[0]) &
                   (pd.to_datetime(ev["entry_ts"]) <= v21s.index[-1])]
        kwargs = dict(n_pick=1, cap_per_slot=cap, universe_size=uni_size,
                      tx_cost=0.003, swap_edge_threshold=1)
        if kind == "fut":
            kwargs["leverage"] = lev
        _, st = runner(ev_s, cd, v21s.copy(), hist, **kwargs)
        rows.append({"label": f"F3_uni{uni_size}", "kind": kind, "span": span,
                     "cap": cap, "Cal": round(st.get("Cal", 0), 3),
                     "CAGR": round(st.get("CAGR", 0), 4),
                     "MDD": round(st.get("MDD", 0), 4),
                     "Sharpe": round(st.get("Sharpe", 0), 3),
                     "n_entries": int(st.get("n_entries", 0))})
    return rows


def run_yearly(kind, ev, v21, cd, hist, caps):
    from _common3 import FULL_END
    rows = []
    runner = simulate_fut if kind == "fut" else simulate
    ev2 = ev.copy()
    ev2["entry_ts"] = pd.to_datetime(ev2["entry_ts"])
    years = sorted(set(ev2["entry_ts"].dt.year))
    for y in years:
        start = pd.Timestamp(f"{y}-01-01")
        end = min(pd.Timestamp(f"{y}-12-31"), FULL_END)
        v21s = slice_v21(v21, start, end)
        if v21s is None or len(v21s) < 30: continue
        ev_s = ev2[(ev2["entry_ts"] >= v21s.index[0]) &
                    (ev2["entry_ts"] <= v21s.index[-1])]
        for cap in caps:
            kwargs = dict(n_pick=1, cap_per_slot=cap, universe_size=15,
                          tx_cost=0.003, swap_edge_threshold=1)
            if kind == "fut":
                kwargs["leverage"] = 3.0
            _, st = runner(ev_s, cd, v21s.copy(), hist, **kwargs)
            rows.append({"label": f"F4_y{y}", "kind": kind, "span": f"year{y}",
                         "cap": cap, "Cal": round(st.get("Cal", 0), 3),
                         "CAGR": round(st.get("CAGR", 0), 4),
                         "MDD": round(st.get("MDD", 0), 4),
                         "Sharpe": round(st.get("Sharpe", 0), 3),
                         "n_entries": int(st.get("n_entries", 0))})
    return rows


def main():
    from _common3 import run_splits, P_SPOT, P_FUT
    v21_spot, v21_fut, hist, _, cd = load_all()
    ev_spot = load_cached_events("spot")
    ev_fut  = load_cached_events("fut")

    rows = []
    rows += run_splits("baseline", "spot", ev_spot, v21_spot, cd, hist, [CAP_SPOT])
    rows += run_splits("baseline", "fut",  ev_fut,  v21_fut,  cd, hist, CAP_FUT_OPTS)

    # F2 tp 변경
    for tp in [0.02, 0.06, 0.10, 0.12]:
        for kind, ev, v21, caps, tstop in [
            ("spot", ev_spot, v21_spot, [CAP_SPOT], P_SPOT["tstop"]),
            ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS, P_FUT["tstop"]),
        ]:
            ev2 = apply_custom_tp(ev, tp, tstop)
            lab = f"F2_tp_{tp:.0%}"
            rows += run_splits(lab, kind, ev2, v21, cd, hist, caps)

    # F3 universe 크기
    for u in [10, 20, 30]:
        for kind, ev, v21, caps in [
            ("spot", ev_spot, v21_spot, [CAP_SPOT]),
            ("fut",  ev_fut,  v21_fut,  CAP_FUT_OPTS),
        ]:
            for cap in caps:
                rows += run_universe(kind, ev, v21, cd, hist, cap, u)

    # F4 연도별
    rows += run_yearly("spot", ev_spot, v21_spot, cd, hist, [CAP_SPOT])
    rows += run_yearly("fut",  ev_fut,  v21_fut,  cd, hist, CAP_FUT_OPTS)

    df = pd.DataFrame(rows)
    path = os.path.join(OUT, "test_robustness.csv")
    df.to_csv(path, index=False)
    print(f"\n저장: {path} ({len(df)} rows)")


if __name__ == "__main__":
    main()
