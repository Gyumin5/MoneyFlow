#!/usr/bin/env python3
"""선물 cap 더 낮은 값도 (0.03~0.15) 비교."""
import os, sys
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
sys.path.insert(0, HERE)

from m3_engine_futures import (load_v21_futures, simulate_fut, load_universe_hist,
                                list_available_futures, load_coin_daily)
from c_engine_v5 import run_c_v5, load_coin


def slc(v21, s, e):
    sub = v21[(v21.index >= s) & (v21.index <= e)].copy()
    sub["equity"] = sub["equity"] / sub["equity"].iloc[0]
    sub["v21_ret"] = sub["equity"].pct_change().fillna(0)
    sub["prev_cash"] = sub["cash_ratio"].shift(1).fillna(sub["cash_ratio"].iloc[0])
    return sub


def main():
    v21_f = load_v21_futures()
    hist = load_universe_hist()
    avail = sorted(list_available_futures())
    cd = load_coin_daily(avail)

    P_fut = {"dip_bars": 24, "dip_thr": -0.18, "tp": 0.08, "tstop": 48}
    rows = []
    for c in avail:
        df = load_coin(c + "USDT")
        if df is None: continue
        _, evs = run_c_v5(df, **P_fut)
        for e in evs:
            e["coin"] = c
            rows.append(e)
    ev_f = pd.DataFrame(rows)

    print("=== 선물 cap 전체 (낮은 범위 추가) ===")
    print(f"{'cap':>6} {'Full_Cal':>9} {'Full_CAGR':>10} {'Full_MDD':>9} "
          f"{'Train_Cal':>10} {'Train_MDD':>10} "
          f"{'Hout_Cal':>9} {'Hout_CAGR':>10} {'Hout_MDD':>9}")
    for cap in [0.03, 0.05, 0.07, 0.10, 0.12, 0.15, 0.18, 0.20, 0.22, 0.25, 0.28, 0.30, 0.333]:
        FP = dict(n_pick=1, cap_per_slot=cap, universe_size=15,
                  tx_cost=0.003, swap_edge_threshold=1, leverage=3.0)
        res = {}
        for name, s, e in [("Full", "2020-10-01", "2026-04-04"),
                           ("Train", "2020-10-01", "2023-12-31"),
                           ("Hout", "2024-01-01", "2026-04-04")]:
            v21 = slc(v21_f, pd.Timestamp(s), pd.Timestamp(e))
            ev_sub = ev_f[(ev_f["entry_ts"] >= s) & (ev_f["entry_ts"] <= e)]
            _, st = simulate_fut(ev_sub, cd, v21, hist, **FP)
            res[name] = st
        print(f"{cap:>6.3f} {res['Full']['Cal']:>9.2f} {res['Full']['CAGR']:>10.2%} {res['Full']['MDD']:>9.2%} "
              f"{res['Train']['Cal']:>10.2f} {res['Train']['MDD']:>10.2%} "
              f"{res['Hout']['Cal']:>9.2f} {res['Hout']['CAGR']:>10.2%} {res['Hout']['MDD']:>9.2%}")

    # V21 단독 baseline
    print("\n=== V21 단독 baseline ===")
    for name, s, e in [("Full", "2020-10-01", "2026-04-04"),
                       ("Train", "2020-10-01", "2023-12-31"),
                       ("Hout", "2024-01-01", "2026-04-04")]:
        v21 = slc(v21_f, pd.Timestamp(s), pd.Timestamp(e))
        from m3_engine_futures import metrics
        m = metrics(v21["equity"])
        print(f"  {name}: Cal={m['Cal']:.2f} CAGR={m['CAGR']:.2%} MDD={m['MDD']:.2%}")


if __name__ == "__main__":
    main()
