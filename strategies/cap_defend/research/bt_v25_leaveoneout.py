"""Leave-one-out 코인 기여도 — 각 코인 단독 제외 시 전략(base) 성능 변화.
"BNB 단독 의존인가, 여러 코인 분산 기여인가" 확정.
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)
from bt_v25_t1_t3u_t3o import run_spot, run_fut, run_stock, load_canaries, simulate_alloc, metrics

# BTC 제외 불가(카나리 앵커). ETH 이하만 leave-one-out.
COINS = ['ETH', 'BNB', 'SOL', 'XRP', 'ADA', 'DOGE', 'LTC', 'LINK', 'AVAX']
BASE = dict(T1=0.20, T3U=0.20, T3O=None)


def main():
    t0 = time.time()
    eq_st = run_stock()
    eq_sp0 = run_spot(20, 127, 217, 7, [])
    eq_fu0 = run_fut(18, 127, 95, 5, [])
    common0 = sorted(eq_st.index.intersection(eq_sp0.index).intersection(eq_fu0.index))
    can = load_canaries(common0)
    mfull = metrics(simulate_alloc(eq_st, eq_sp0, eq_fu0, can, **BASE)[0])
    print(f"\n=== Leave-one-out (base, full=전체 기준) ===")
    print(f"  full: CAGR {mfull[0]:.1f}% MDD {mfull[1]:+.1f}% Sh {mfull[2]:.2f} Cal {mfull[3]:.2f}")
    print(f"\n  {'제외코인':<7} {'CAGR':>7} {'ΔCAGR':>8} {'MDD':>7} {'Cal':>5} {'ΔCal':>6}")
    rows = []
    for c in COINS:
        eq_sp = run_spot(20, 127, 217, 7, [c])
        eq_fu = run_fut(18, 127, 95, 5, [c])
        if eq_sp is None or eq_fu is None:
            print(f"  -{c:<6} (데이터 없음/스킵)"); continue
        m = metrics(simulate_alloc(eq_st, eq_sp, eq_fu, can, **BASE)[0])
        rows.append((c, m))
        print(f"  -{c:<6} {m[0]:>6.1f}% {m[0]-mfull[0]:>+7.1f}pp {m[1]:>+6.1f}% {m[3]:>5.2f} {m[3]-mfull[3]:>+6.2f}")
    print("\n  기여도 순위(ΔCAGR 큰 손실 = 의존도 높음):")
    for c, m in sorted(rows, key=lambda r: r[1][0]):
        print(f"    {c}: CAGR 기여 {mfull[0]-m[0]:+.1f}pp, Cal 기여 {mfull[3]-m[3]:+.2f}")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
