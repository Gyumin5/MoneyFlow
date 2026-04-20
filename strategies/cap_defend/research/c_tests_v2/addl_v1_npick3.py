#!/usr/bin/env python3
"""추가 검증 A: n_pick=3 + uni=10 조합 풀기간/train/holdout 재시뮬.

Test 11에서 Holdout Cal 기준 top5 모두 n_pick=3 uni=10. 이제 전구간·train·holdout 3구간 재검증.
비교: 현재 확정 (n_pick=1 uni=15) vs 새 후보 (n_pick=3 uni=10).
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (CAP_SPOT, CAP_FUT_OPTS, TRAIN_END, HOLDOUT_START, FULL_END,
                     load_all, load_cached_events, slice_v21)
from m3_engine_final import simulate
from m3_engine_futures import simulate_fut

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def run_cfg(name, v21, ev, sim, kwargs, extra):
    all_rows = []
    for span, s, e in [("full", v21.index[0], FULL_END),
                        ("train", v21.index[0], TRAIN_END),
                        ("holdout", HOLDOUT_START, FULL_END)]:
        v21s = slice_v21(v21, s, e)
        evs = ev[(ev["entry_ts"] >= v21s.index[0]) & (ev["entry_ts"] <= v21s.index[-1])].copy()
        _, st = sim(evs, kwargs["cd"], v21s.copy(), kwargs["hist"], **extra)
        all_rows.append({"label": f"{name}_{span}", "span": span,
                         "Cal": round(st["Cal"], 3),
                         "CAGR": round(st["CAGR"], 4),
                         "MDD": round(st["MDD"], 4),
                         "n_entries": st["n_entries"]})
    return all_rows


def main():
    v21_s, v21_f, hist, avail, cd = load_all()
    ev_s = load_cached_events("spot")
    ev_f = load_cached_events("fut")

    # 현재 vs 신규 설정
    configs_spot = [
        ("spot_cur_n1_u15_cap0.333",  dict(n_pick=1, cap_per_slot=CAP_SPOT, universe_size=15,
                                            tx_cost=0.003, swap_edge_threshold=1)),
        ("spot_new_n3_u10_cap0.333",  dict(n_pick=3, cap_per_slot=CAP_SPOT, universe_size=10,
                                            tx_cost=0.003, swap_edge_threshold=1)),
        ("spot_new_n3_u15_cap0.333",  dict(n_pick=3, cap_per_slot=CAP_SPOT, universe_size=15,
                                            tx_cost=0.003, swap_edge_threshold=1)),
        ("spot_new_n2_u10_cap0.333",  dict(n_pick=2, cap_per_slot=CAP_SPOT, universe_size=10,
                                            tx_cost=0.003, swap_edge_threshold=1)),
    ]
    configs_fut = []
    for cap in CAP_FUT_OPTS:
        configs_fut += [
            (f"fut_cur_n1_u15_cap{cap}", dict(n_pick=1, cap_per_slot=cap, universe_size=15,
                                                tx_cost=0.003, swap_edge_threshold=1, leverage=3.0)),
            (f"fut_new_n3_u10_cap{cap}", dict(n_pick=3, cap_per_slot=cap, universe_size=10,
                                                tx_cost=0.003, swap_edge_threshold=1, leverage=3.0)),
            (f"fut_new_n3_u15_cap{cap}", dict(n_pick=3, cap_per_slot=cap, universe_size=15,
                                                tx_cost=0.003, swap_edge_threshold=1, leverage=3.0)),
            (f"fut_new_n2_u10_cap{cap}", dict(n_pick=2, cap_per_slot=cap, universe_size=10,
                                                tx_cost=0.003, swap_edge_threshold=1, leverage=3.0)),
        ]

    kwargs = {"cd": cd, "hist": hist}
    all_rows = []
    for name, extra in configs_spot:
        all_rows += run_cfg(name, v21_s, ev_s, simulate, kwargs, extra)
    for name, extra in configs_fut:
        all_rows += run_cfg(name, v21_f, ev_f, simulate_fut, kwargs, extra)

    df = pd.DataFrame(all_rows)
    df.to_csv(os.path.join(OUT, "addl_v1_npick3.csv"), index=False)

    # 비교 출력
    print("\n=== SPOT 비교 (현재 n=1 vs 신규 n=3) ===")
    spot_df = df[df["label"].str.startswith("spot_")]
    print(spot_df.pivot_table(index="label", columns="span",
                               values="Cal", aggfunc="first").to_string())

    print("\n=== FUT 비교 (3 cap × 4 config) ===")
    fut_df = df[df["label"].str.startswith("fut_")]
    print(fut_df.pivot_table(index="label", columns="span",
                              values="Cal", aggfunc="first").to_string())

    print(f"\n저장: {OUT}/addl_v1_npick3.csv")


if __name__ == "__main__":
    main()
