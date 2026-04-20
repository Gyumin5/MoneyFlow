#!/usr/bin/env python3
"""C 최종 스트레스 테스트 (AI 권고 사항).

1. Top N trade 제거 (희소성 의존도 체크)
2. 선물 cap 미세 비교 (0.20/0.25/0.28/0.30)
3. TX 스트레스 (0.3 / 0.4 / 0.6 / 1.0 %)
"""
from __future__ import annotations
import os, sys
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from m3_engine_futures import (load_v21_futures, simulate_fut, metrics,
                                load_universe_hist, list_available_futures,
                                load_coin_daily)
from m3_engine_final import load_v21, simulate
from c_engine_v5 import run_c_v5, load_coin

OUT = os.path.join(HERE, "c_stress_final")
os.makedirs(OUT, exist_ok=True)


def extract(avail, P, tx=0.003, fd=0):
    rows = []
    for c in avail:
        df = load_coin(c + "USDT")
        if df is None: continue
        _, evs = run_c_v5(df, tx=tx, fill_delay=fd, **P)
        for e in evs:
            e["coin"] = c
            rows.append(e)
    return pd.DataFrame(rows)


def slc(v21, s, e):
    sub = v21[(v21.index >= s) & (v21.index <= e)].copy()
    sub["equity"] = sub["equity"] / sub["equity"].iloc[0]
    sub["v21_ret"] = sub["equity"].pct_change().fillna(0)
    sub["prev_cash"] = sub["cash_ratio"].shift(1).fillna(sub["cash_ratio"].iloc[0])
    return sub


def main():
    print("Loading...")
    v21_f = load_v21_futures()
    v21_s = load_v21()
    hist = load_universe_hist()
    avail = sorted(list_available_futures())
    cd = load_coin_daily(avail)

    P_spot = {"dip_bars":24, "dip_thr":-0.20, "tp":0.04, "tstop":24}
    P_fut  = {"dip_bars":24, "dip_thr":-0.18, "tp":0.08, "tstop":48}

    ev_s = extract(avail, P_spot)
    ev_f = extract(avail, P_fut)

    HOLDOUT_S = pd.Timestamp("2024-01-01")
    v21_sH = slc(v21_s, HOLDOUT_S, v21_s.index[-1])
    v21_fH = slc(v21_f, HOLDOUT_S, v21_f.index[-1])

    # ── Test 1: Top N trade 제거 ──
    print("\n=== Test 1: Top N 수익 이벤트 제거 (Holdout) ===")
    for name, ev, v21H, sim, cap, extra in [
        ("spot", ev_s, v21_sH, simulate, 0.333, {}),
        ("fut",  ev_f, v21_fH, simulate_fut, 0.30, {"leverage": 3.0}),
    ]:
        print(f"\n-- {name} --")
        FP = dict(n_pick=1, cap_per_slot=cap, universe_size=15, tx_cost=0.003,
                  swap_edge_threshold=1, **extra)
        ev_h = ev[(ev["entry_ts"] >= HOLDOUT_S) & (ev["entry_ts"] <= v21H.index[-1])]
        for n in [0, 1, 3, 5, 10]:
            ev_keep = ev_h.sort_values("pnl_pct", ascending=False).iloc[n:] if n else ev_h
            _, st = sim(ev_keep, cd, v21H, hist, **FP)
            print(f"  Top {n} 제거: Cal={st['Cal']:.2f} CAGR={st['CAGR']:.2%} MDD={st['MDD']:.2%} entries={st['n_entries']}")

    # ── Test 2: 선물 cap 미세 비교 (Full + Train + Holdout) ──
    print("\n=== Test 2: 선물 cap 미세 비교 (0.20/0.25/0.28/0.30) ===")
    for cap in [0.20, 0.22, 0.25, 0.28, 0.30, 0.333]:
        FP = dict(n_pick=1, cap_per_slot=cap, universe_size=15, tx_cost=0.003,
                  swap_edge_threshold=1, leverage=3.0)
        print(f"\n  cap={cap}")
        for span_name, s, e in [("전구간","2020-10-01","2026-04-04"),
                                 ("Train","2020-10-01","2023-12-31"),
                                 ("Holdout","2024-01-01","2026-04-04")]:
            v21 = slc(v21_f, pd.Timestamp(s), pd.Timestamp(e))
            ev_sub = ev_f[(ev_f["entry_ts"] >= s) & (ev_f["entry_ts"] <= e)]
            _, st = simulate_fut(ev_sub, cd, v21, hist, **FP)
            print(f"    {span_name}: Cal={st['Cal']:.2f} CAGR={st['CAGR']:.2%} MDD={st['MDD']:.2%}")

    # ── Test 3: TX 스트레스 ──
    print("\n=== Test 3: TX 스트레스 (Holdout 기준) ===")
    for name, P, v21H, sim, cap, extra in [
        ("spot", P_spot, v21_sH, simulate, 0.333, {}),
        ("fut",  P_fut,  v21_fH, simulate_fut, 0.30, {"leverage": 3.0}),
    ]:
        print(f"\n-- {name} --")
        for tx in [0.003, 0.005, 0.008, 0.013]:
            ev_tx = extract(avail, P, tx=tx)
            ev_h = ev_tx[(ev_tx["entry_ts"] >= HOLDOUT_S) & (ev_tx["entry_ts"] <= v21H.index[-1])]
            FP = dict(n_pick=1, cap_per_slot=cap, universe_size=15, tx_cost=tx,
                      swap_edge_threshold=1, **extra)
            _, st = sim(ev_h, cd, v21H, hist, **FP)
            print(f"  TX={tx*100:.2f}%: Cal={st['Cal']:.2f} CAGR={st['CAGR']:.2%} MDD={st['MDD']:.2%}")

    print(f"\n저장: {OUT}/")


if __name__ == "__main__":
    main()
