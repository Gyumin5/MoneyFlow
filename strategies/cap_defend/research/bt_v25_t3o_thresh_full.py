"""T3O 임계 grid — 라이브 full 유니버스(BNB·SOL 포함). 2차 debate 핵심 미측정 축.
질문: 임계를 20→25/30/35%로 올리면 CAGR 희생이 줄면서 MDD 개선은 유지되나?
base 대비 window 승률·비용 민감도 포함.
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)
from bt_v25_t1_t3u_t3o import run_spot, run_fut, run_stock, load_canaries, simulate_alloc, metrics
from bt_v25_t3o_robust import window_ranksum


def main():
    t0 = time.time()
    print("[full universe = BNB·SOL 포함]")
    eq_st = run_stock()
    eq_sp = run_spot(20, 127, 217, 7, [])    # full
    eq_fu = run_fut(18, 127, 95, 5, [])       # full
    common = sorted(eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index))
    can = load_canaries(common)

    eq_base = simulate_alloc(eq_st, eq_sp, eq_fu, can, T1=0.20, T3U=0.20, T3O=None)[0]
    mb = metrics(eq_base)

    print(f"\n  base(T3O off): CAGR {mb[0]:.1f}% MDD {mb[1]:+.1f}% Sh {mb[2]:.2f} Cal {mb[3]:.2f}")
    print(f"\n  {'T3O':<5} {'cost':<5} {'CAGR':>7} {'ΔCAGR':>7} {'MDD':>7} {'Sh':>6} {'Cal':>5} {'nT3O':>5} {'win%':>6}")
    for T3O in [0.20, 0.25, 0.30, 0.35, 0.40]:
        for cost in [0.0, 0.003]:
            eq, n1, n3u, n3o = simulate_alloc(eq_st, eq_sp, eq_fu, can,
                                              T1=0.20, T3U=0.20, T3O=T3O, t3o_canary='none',
                                              transfer_cost=cost)
            m = metrics(eq)
            _, t1c, nwin = window_ranksum([eq_base, eq], ['b', 't'], by='cal')
            win = t1c['t'] / nwin * 100
            print(f"  {T3O*100:>4.0f}% {int(cost*1e4):>3}bp {m[0]:>6.1f}% {m[0]-mb[0]:>+6.1f}pp {m[1]:>+6.1f}% {m[2]:>6.2f} {m[3]:>5.2f} {n3o:>5} {win:>5.0f}%")
        print()

    print(f"총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
