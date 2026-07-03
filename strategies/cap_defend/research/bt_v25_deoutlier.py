"""아웃라이어 제외 사다리 — 전략 성능 일반화(survivorship 보정).

freak 점프로 BTC 를 이긴 코인(BNB kurt43, DOGE kurt787, XRP 28, ADA 21)을 단계적으로 제외하고
현행 base 와 T3O 성능이 얼마나 아웃라이어 의존이었나 정량화. SOL(kurt 8.6)은 점프아웃라이어 아님.
"""
from __future__ import annotations
import os, sys, time
import numpy as np, pandas as pd
HERE = os.path.dirname(os.path.abspath(__file__)); CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)
from bt_v25_t1_t3u_t3o import run_spot, run_fut, run_stock, load_canaries, simulate_alloc, metrics
from bt_v25_t3o_robust import window_ranksum

LADDER = [
    ('U0 full(전체)',          []),
    ('U1 -BNB',                ['BNB']),
    ('U2 -BNB,DOGE',           ['BNB', 'DOGE']),
    ('U3 -BNB,DOGE,XRP,ADA',   ['BNB', 'DOGE', 'XRP', 'ADA']),
    ('U4 -BNB,SOL,DOGE',       ['BNB', 'SOL', 'DOGE']),
    ('U5 -BNB,SOL,DOGE,XRP,ADA',['BNB', 'SOL', 'DOGE', 'XRP', 'ADA']),
]
BASE = dict(T1=0.20, T3U=0.20, T3O=None)
T3O20 = dict(T1=0.20, T3U=0.20, T3O=0.20, t3o_canary='none')
T3O35 = dict(T1=0.20, T3U=0.20, T3O=0.35, t3o_canary='none')


def main():
    t0 = time.time()
    print("[load stock + canaries]")
    eq_st = run_stock()
    eq_sp0 = run_spot(20, 127, 217, 7, [])
    eq_fu0 = run_fut(18, 127, 95, 5, [])
    common0 = sorted(eq_st.index.intersection(eq_sp0.index).intersection(eq_fu0.index))
    can = load_canaries(common0)

    print(f"\n=== de-outlier 사다리: 현행 base 성능 일반화 + T3O 효과 ===")
    print(f"  {'유니버스':<24} {'CAGR':>7} {'MDD':>7} {'Sh':>6} {'Cal':>5} | {'spotCal':>7} {'futCal':>6} | T3O20ΔCal T3O35ΔCal  win20% win35%")
    base_full_cagr = None
    for uname, excl in LADDER:
        if excl == []:
            eq_sp, eq_fu = eq_sp0, eq_fu0
        else:
            eq_sp = run_spot(20, 127, 217, 7, excl)
            eq_fu = run_fut(18, 127, 95, 5, excl)
        m_sp = metrics(eq_sp); m_fu = metrics(eq_fu)
        eq_b = simulate_alloc(eq_st, eq_sp, eq_fu, can, **BASE)[0]
        mb = metrics(eq_b)
        if base_full_cagr is None:
            base_full_cagr = mb[0]
        eq_t20 = simulate_alloc(eq_st, eq_sp, eq_fu, can, **T3O20)[0]; mt20 = metrics(eq_t20)
        eq_t35 = simulate_alloc(eq_st, eq_sp, eq_fu, can, **T3O35)[0]; mt35 = metrics(eq_t35)
        _, w20, nw = window_ranksum([eq_b, eq_t20], ['b', 't'], by='cal'); win20 = w20['t']/nw*100
        _, w35, nw = window_ranksum([eq_b, eq_t35], ['b', 't'], by='cal'); win35 = w35['t']/nw*100
        print(f"  {uname:<24} {mb[0]:>6.1f}% {mb[1]:>+6.1f}% {mb[2]:>6.2f} {mb[3]:>5.2f} | {m_sp[3]:>7.2f} {m_fu[3]:>6.2f} | "
              f"{mt20[3]-mb[3]:>+8.2f} {mt35[3]-mb[3]:>+8.2f}  {win20:>5.0f}% {win35:>4.0f}%")

    print(f"\n  ※ base CAGR 가 아웃라이어 제외로 얼마나 빠지나 = 전략 성능의 아웃라이어 의존도")
    print(f"총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
