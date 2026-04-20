#!/usr/bin/env python3
"""Test 5: Fill delay + TX 극단 스트레스 (Gemini 권고 반영).

실전 체결 지연 (네트워크/cron 간격) + 급락 호가창 얇아짐에 따른 슬리피지.
체결 지연 fd=0/1/4/12h × TX=0.3/0.5/0.8/1.3% 비교.
"""
from __future__ import annotations
import os, sys
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (P_SPOT, P_FUT, CAP_SPOT, CAP_FUT_OPTS,
                     HOLDOUT_START, FULL_END,
                     load_all, extract_events, load_cached_events, slice_v21,
                     run_spot_combo, run_fut_combo)
from m3_engine_final import simulate
from m3_engine_futures import simulate_fut

OUT = os.path.join(os.path.dirname(__file__), "out")
os.makedirs(OUT, exist_ok=True)


def run_spot_with_tx(events, coin_daily, v21_slice, hist, cap, tx):
    return simulate(events, coin_daily, v21_slice.copy(), hist,
                    n_pick=1, cap_per_slot=cap, universe_size=15,
                    tx_cost=tx, swap_edge_threshold=1)


def run_fut_with_tx(events, coin_daily, v21_slice, hist, cap, tx):
    return simulate_fut(events, coin_daily, v21_slice.copy(), hist,
                        n_pick=1, cap_per_slot=cap, universe_size=15,
                        tx_cost=tx, swap_edge_threshold=1, leverage=3.0)


def main():
    v21_s, v21_f, hist, avail, cd = load_all()

    v21_sH = slice_v21(v21_s, HOLDOUT_START, FULL_END)
    v21_fH = slice_v21(v21_f, HOLDOUT_START, FULL_END)

    rows = []

    # fill_delay × TX grid
    # tx는 c_engine_v5 시그널/체결 판정 자체를 바꾸지 않음 → fd별로만 이벤트 추출,
    # tx는 simulate cost로만 반영 (Codex 권고, 성능 개선).
    for fd in [0, 1, 4, 12]:
        # fd=0이고 기본 tx일 때는 공통 캐시 활용
        if fd == 0:
            ev_s = load_cached_events("spot")
            ev_f = load_cached_events("fut")
        else:
            ev_s = extract_events(avail, P_SPOT, fd=fd)
            ev_f = extract_events(avail, P_FUT,  fd=fd)
        for tx in [0.003, 0.005, 0.008, 0.013]:

            # spot
            ev_sH = ev_s[(ev_s["entry_ts"] >= HOLDOUT_START) & (ev_s["entry_ts"] <= FULL_END)].copy()
            _, st_s = run_spot_with_tx(ev_sH, cd, v21_sH, hist, CAP_SPOT, tx)
            rows.append({
                "asset": "spot", "cap": CAP_SPOT, "fill_delay_h": fd, "tx": tx,
                "Cal": round(st_s["Cal"], 3),
                "CAGR": round(st_s["CAGR"], 4),
                "MDD": round(st_s["MDD"], 4),
                "n_entries": st_s["n_entries"],
            })

            # fut 3 caps
            ev_fH = ev_f[(ev_f["entry_ts"] >= HOLDOUT_START) & (ev_f["entry_ts"] <= FULL_END)].copy()
            for cap in CAP_FUT_OPTS:
                _, st_f = run_fut_with_tx(ev_fH, cd, v21_fH, hist, cap, tx)
                rows.append({
                    "asset": f"fut_cap{cap}", "cap": cap, "fill_delay_h": fd, "tx": tx,
                    "Cal": round(st_f["Cal"], 3),
                    "CAGR": round(st_f["CAGR"], 4),
                    "MDD": round(st_f["MDD"], 4),
                    "n_entries": st_f["n_entries"],
                })

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(OUT, "test5_fill_delay.csv"), index=False)

    # Pivot by (asset, cap): tx vs fill_delay
    for asset_val in df["asset"].unique():
        sub = df[df["asset"] == asset_val]
        pv = sub.pivot_table(index="tx", columns="fill_delay_h",
                              values="Cal", aggfunc="first")
        print(f"\n=== {asset_val} Cal (rows: tx, cols: fill_delay_h) ===")
        print(pv.to_string())

    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
