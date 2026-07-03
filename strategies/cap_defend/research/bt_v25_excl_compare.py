"""신전략(+T3O20) vs 현행(base) × 코인 유니버스 4조합 비교.

유니버스 4조합 (BNB·SOL 포함 여부):
  - 둘다 포함 (full universe)
  - BNB만 포함 (= SOL 제외)
  - SOL만 포함 (= BNB 제외)
  - 둘다 제외

각 조합에서 현행 base(T1 20/T3U 20) vs 신 +T3O20(무게이트) 비교.
sleeve 곡선은 spot/fut 만 유니버스 영향(주식·카나리는 불변).
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from bt_v25_t1_t3u_t3o import run_spot, run_fut, run_stock, load_canaries, simulate_alloc, metrics
from bt_v25_t3o_robust import window_ranksum

UNIVERSES = [
    ('둘다포함(full)', []),
    ('BNB만(SOL제외)', ['SOL']),
    ('SOL만(BNB제외)', ['BNB']),
    ('둘다제외',       ['BNB', 'SOL']),
]
BASE = dict(T1=0.20, T3U=0.20, T3O=None)
T3O = dict(T1=0.20, T3U=0.20, T3O=0.20, t3o_canary='none')


def main():
    t0 = time.time()
    print("[load stock + canaries]")
    eq_st = run_stock()
    # 카나리는 BTC/EEM 기반(코인 유니버스 무관) → full common 으로 1회 생성, 재사용
    eq_sp0 = run_spot(20, 127, 217, 7, [])
    eq_fu0 = run_fut(18, 127, 95, 5, [])
    common0 = sorted(eq_st.index.intersection(eq_sp0.index).intersection(eq_fu0.index))
    can = load_canaries(common0)

    print(f"\n=== 신(+T3O20 무게이트) vs 현행(base) × 유니버스 4조합 (전기간, 비용 0) ===")
    print(f"  {'유니버스':<16} {'전략':<10} {'CAGR':>7} {'MDD':>7} {'Sh':>6} {'Cal':>5}  {'spotCal':>7} {'futCal':>6}")
    rows = {}
    for uname, excl in UNIVERSES:
        if excl == []:
            eq_sp, eq_fu = eq_sp0, eq_fu0
        else:
            eq_sp = run_spot(20, 127, 217, 7, excl)
            eq_fu = run_fut(18, 127, 95, 5, excl)
        m_sp = metrics(eq_sp); m_fu = metrics(eq_fu)
        eq_b = simulate_alloc(eq_st, eq_sp, eq_fu, can, **BASE)[0]
        eq_t = simulate_alloc(eq_st, eq_sp, eq_fu, can, **T3O)[0]
        mb = metrics(eq_b); mt = metrics(eq_t)
        rows[uname] = (mb, mt, eq_b, eq_t)
        print(f"  {uname:<16} {'현행 base':<10} {mb[0]:>6.1f}% {mb[1]:>+6.1f}% {mb[2]:>6.2f} {mb[3]:>5.2f}  {m_sp[3]:>7.2f} {m_fu[3]:>6.2f}")
        print(f"  {'':<16} {'신 +T3O':<10} {mt[0]:>6.1f}% {mt[1]:>+6.1f}% {mt[2]:>6.2f} {mt[3]:>5.2f}  {'ΔCal':>7} {mt[3]-mb[3]:>+6.2f}")
        # window rank-sum: base vs T3O (이 유니버스 안에서)
        rs, t1c, nwin = window_ranksum([eq_b, eq_t], ['base', 'T3O'], by='cal')
        win = t1c['T3O'] / nwin * 100
        print(f"  {'':<16} {'win%(window Cal)':<10} → T3O가 base 이긴 창 {win:.0f}% (n={nwin}), ΔSharpe {mt[2]-mb[2]:+.2f}")
        print()

    # 요약: T3O 우위가 4조합 모두에서 성립하나
    print("=== 요약: ΔCal (신 − 현행) ===")
    for uname in rows:
        mb, mt, _, _ = rows[uname]
        verdict = "T3O 우위" if mt[3] > mb[3] and mt[2] >= mb[2] else ("혼조" if mt[3] > mb[3] else "base 우위")
        print(f"  {uname:<16} base Cal {mb[3]:.2f} → T3O {mt[3]:.2f}  (ΔCal {mt[3]-mb[3]:+.2f}, ΔMDD {mt[1]-mb[1]:+.1f}pp, ΔSh {mt[2]-mb[2]:+.2f})  {verdict}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
