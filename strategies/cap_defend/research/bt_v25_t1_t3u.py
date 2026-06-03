"""V25 — alloc 트리거 T1 × T3U_can 그리드.

설계:
- 일별 sleeve PV 시계열 (이미 BT'd)
- alloc 시뮬레이션: 시작 60/25/15, 매일 트리거 체크
- T1: half_turnover (sum|cur - tgt|/2) >= T1pp → 전체 60/25/15 reset
- T3U_can: 특정 sleeve cur/tgt < (1 - T3U/100) AND canary ON → 그 sleeve 만 reset
  · 단순화: canary ON 신호 = 그 sleeve 의 eq pct_change 최근 20D 평균 > 0
- 트리거 시 모두 100% 가상 송금 (수수료 무시 — manual 송금 가정)

비교: T1 × T3U_can 그리드 + baseline (T1=20, T3U=20)
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

from bt_v25_excl_matrix import run_spot, run_fut, run_stock, metrics

START = "2020-10-01"; END = "2026-05-29"
EXCLUDE = ['BNB', 'SOL']
WEIGHTS = {'stock': 0.60, 'spot': 0.25, 'fut': 0.15}


def simulate_alloc(eq_st, eq_sp, eq_fu, T1=0.20, T3U=0.20):
    """일별 alloc 시뮬레이션. T1/T3U 트리거 발화 시 reset.
    T1, T3U: 비율 (예: 0.20 = 20pp / 20%).
    """
    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)
    common = sorted(common)
    if len(common) < 30: return None

    # daily return per sleeve
    r_st = eq_st.loc[common].pct_change().fillna(0)
    r_sp = eq_sp.loc[common].pct_change().fillna(0)
    r_fu = eq_fu.loc[common].pct_change().fillna(0)

    # canary proxy: 최근 20D 평균 수익률 > 0
    can_st = r_st.rolling(20).mean() > 0
    can_sp = r_sp.rolling(20).mean() > 0
    can_fu = r_fu.rolling(20).mean() > 0

    # initial weights = target
    pv = 1.0
    w = dict(WEIGHTS)  # current weights
    pvs_st = pv * w['stock']
    pvs_sp = pv * w['spot']
    pvs_fu = pv * w['fut']

    eq_alloc = []
    n_t1 = 0; n_t3u = 0
    for i, d in enumerate(common):
        # apply daily return
        pvs_st *= (1 + r_st.iloc[i])
        pvs_sp *= (1 + r_sp.iloc[i])
        pvs_fu *= (1 + r_fu.iloc[i])
        total = pvs_st + pvs_sp + pvs_fu
        if total <= 0:
            eq_alloc.append(0); continue
        cur_w = {'stock': pvs_st/total, 'spot': pvs_sp/total, 'fut': pvs_fu/total}
        # T1
        ht = sum(abs(cur_w[k] - WEIGHTS[k]) for k in WEIGHTS) / 2
        fire_t1 = ht >= T1
        # T3U_can
        fire_t3u = {}
        for k in WEIGHTS:
            tgt = WEIGHTS[k]
            ratio = cur_w[k] / tgt if tgt > 0 else 1
            under = (1 - ratio) >= T3U  # cur/tgt < (1 - T3U)
            can_arr = {'stock': can_st, 'spot': can_sp, 'fut': can_fu}[k]
            can_on = bool(can_arr.iloc[i]) if i < len(can_arr) else False
            if under and can_on:
                fire_t3u[k] = True
        if fire_t1:
            # full reset
            pvs_st = total * WEIGHTS['stock']
            pvs_sp = total * WEIGHTS['spot']
            pvs_fu = total * WEIGHTS['fut']
            n_t1 += 1
        elif fire_t3u:
            # T3U: 해당 under sleeve 만 target 까지 reset (다른 sleeve 들 비율적으로 줄임)
            for k_fire in fire_t3u:
                # 단순화: 해당 sleeve 를 target weight 까지 끌어올림
                target_pv = total * WEIGHTS[k_fire]
                if k_fire == 'stock' and pvs_st < target_pv:
                    delta = target_pv - pvs_st
                    pvs_st = target_pv
                    pvs_sp -= delta * (WEIGHTS['spot'] / (WEIGHTS['spot'] + WEIGHTS['fut']))
                    pvs_fu -= delta * (WEIGHTS['fut'] / (WEIGHTS['spot'] + WEIGHTS['fut']))
                elif k_fire == 'spot' and pvs_sp < target_pv:
                    delta = target_pv - pvs_sp
                    pvs_sp = target_pv
                    pvs_st -= delta * (WEIGHTS['stock'] / (WEIGHTS['stock'] + WEIGHTS['fut']))
                    pvs_fu -= delta * (WEIGHTS['fut'] / (WEIGHTS['stock'] + WEIGHTS['fut']))
                elif k_fire == 'fut' and pvs_fu < target_pv:
                    delta = target_pv - pvs_fu
                    pvs_fu = target_pv
                    pvs_st -= delta * (WEIGHTS['stock'] / (WEIGHTS['stock'] + WEIGHTS['spot']))
                    pvs_sp -= delta * (WEIGHTS['spot'] / (WEIGHTS['stock'] + WEIGHTS['spot']))
            n_t3u += 1
        eq_alloc.append(pvs_st + pvs_sp + pvs_fu)
    eq = pd.Series(eq_alloc, index=common)
    return eq, n_t1, n_t3u


def main():
    t0 = time.time()
    eq_st = run_stock()
    # winner config
    eq_sp = run_spot(20, 127, 217, 7, EXCLUDE)
    eq_fu = run_fut(18, 127, 95, 5, EXCLUDE)  # baseline fut for robust check
    print("\n[BASELINE fut ms=18 ml=127 sleeves loaded]")

    T1_grid = [0.15, 0.17, 0.20, 0.23, 0.25, 0.30]
    T3U_grid = [0.15, 0.18, 0.20, 0.22, 0.25, 0.30]

    print(f"\n  {'T1':<5} {'T3U':<5} {'CAGR':>7} {'MDD':>7} {'Sharpe':>7} {'Cal':>5} {'n_T1':>5} {'n_T3U':>6}")
    results = []
    for T1 in T1_grid:
        for T3U in T3U_grid:
            res = simulate_alloc(eq_st, eq_sp, eq_fu, T1=T1, T3U=T3U)
            if res is None: continue
            eq, n_t1, n_t3u = res
            m = metrics(eq)
            if m is None: continue
            results.append((T1, T3U, m, n_t1, n_t3u))
            tag = "← base" if T1 == 0.20 and T3U == 0.20 else ""
            print(f"  {T1*100:>4.0f}% {T3U*100:>4.0f}% {m[0]:>6.1f}% {m[1]:>+6.1f}% {m[2]:>7.2f} {m[3]:>5.2f} {n_t1:>5} {n_t3u:>6} {tag}")

    # ranking by alloc Cal
    print("\n  Top 10 by Cal:")
    for T1, T3U, m, n1, n3 in sorted(results, key=lambda r: -r[2][3])[:10]:
        print(f"    T1={T1*100:.0f}% T3U={T3U*100:.0f}%: CAGR {m[0]:.1f}% MDD {m[1]:+.1f}% Cal {m[3]:.2f} (T1:{n1} T3U:{n3})")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
