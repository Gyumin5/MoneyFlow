#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T1·T3U 문턱을 움직이면 무엇이 얼마나 달라지나 — 기술(記述)만 한다.

왜 따로 만드나. 오늘 끝난 본 채점(alloc_grid_run.py)은 T1=0.20·T3U=0.20 을 처음부터
끝까지 고정하고 "전이 사건을 더하는 축" 만 폈다. 밴드 축(0/5/10pp)은 새 전이 발화에만
걸린다(alloc_ledger_sim.py 314행). 그래서 그 연구는 "현행 문턱이 최적인가" 에 답하지 않는다.

이 스크립트가 하는 것과 안 하는 것.
  한다  — 문턱 쌍마다 현행 조합(C = T1 OR T3U)을 돌려 발화·요청·완료 횟수, 목표배분
          이탈(95분위·최대), 그리고 전구간 성과를 적는다. 앵커 전부 × 비용 두 팔.
  안 한다 — 채택 판정. 여기 숫자를 보고 문턱을 고르지 않는다. 고르려면 사전등록한
          기준으로 따로 돌려야 하고, 그 설계는 ai-debate 결과를 받아 쓴다.
          지금 이 표는 "문턱이 얼마나 둔감한가" 를 재는 기술 통계다.

쓰기:
  python3 alloc_t1t3u_sensitivity.py                 # 전 앵커 × 두 팔
  python3 alloc_t1t3u_sensitivity.py --anchor 0      # 한 앵커만
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

import alloc_grid_run as RUN
import alloc_ledger_sim as L
import alloc_trigger_space as S

OUT = RUN.OUT
T1_GRID = (0.10, 0.125, 0.15, 0.175, 0.20, 0.25, 0.30)
T3U_GRID = (0.10, 0.15, 0.20, 0.25, 0.30, 0.35)
LIVE = (0.20, 0.20)


def scan_cell(anchor: int, stress: bool):
    """한 칸(앵커 × 비용 팔)에서 문턱 격자를 훑는다."""
    eq_st, eq_sp, eq_fu = RUN.sleeve_curves(anchor, stress)
    axis, Rm, is_bd = L.build_ledger(eq_st, eq_sp, eq_fu)
    tab = L.bd_tables(axis, is_bd)
    g = L.gate_series_ledger(axis)
    gate = np.column_stack([np.asarray(g[s].values, dtype=bool) for s in S.SLEEVES])
    EV, _, _ = L.event_matrix(S.build_E(g), axis)
    month_id = np.asarray(pd.DatetimeIndex(axis).year * 12
                          + pd.DatetimeIndex(axis).month, dtype=np.int64)
    yrs = (axis[-1] - axis[0]).days / 365.25
    spec = RUN.control_spec("현행")

    t1_keep, t3u_keep = L.T1, L.T3U
    rows = []
    try:
        for t1 in T1_GRID:
            for t3u in T3U_GRID:
                L.T1, L.T3U = t1, t3u
                pv, inf = L.simulate_ledger(Rm, is_bd, gate, EV, spec, tab,
                                            month_id=month_id)
                fs = RUN.full_stats(pv, axis)
                rows.append(dict(
                    anchor=anchor, stress=bool(stress), T1=t1, T3U=t3u,
                    fires=int(inf['fires'][0]), dones=int(inf['dones'][0]),
                    per_yr=float(inf['dones'][0]) / yrs,
                    multi=int(inf['multi_months'][0]),
                    ht95=float(inf['ht_p95'][0]), maxht=float(inf['max_ht'][0]),
                    Cal=float(fs['Cal'][0]), MDD=float(fs['MDD'][0]),
                    Sharpe=float(fs['Sharpe'][0]), CAGR=float(fs['CAGR'][0])))
    finally:
        L.T1, L.T3U = t1_keep, t3u_keep
    return rows, yrs


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=int, default=None)
    a = ap.parse_args()
    anchors = [a.anchor] if a.anchor is not None else list(range(7))

    rows = []
    for an in anchors:
        for st in (False, True):
            r, yrs = scan_cell(an, st)
            rows += r
            print(f"  앵커 {an} {'5배' if st else '실비용'} 끝 ({yrs:.3f}년)")
    df = pd.DataFrame(rows)
    p = os.path.join(OUT, "t1t3u_sensitivity.csv")
    df.to_csv(p, index=False)

    print(f"\n문턱 쌍 {len(T1_GRID)}×{len(T3U_GRID)} = {len(T1_GRID)*len(T3U_GRID)}, "
          f"칸 {len(anchors)}×2, 줄 {len(df):,}")
    print(f"저장: {p}\n")

    base = df[(df.T1 == LIVE[0]) & (df.T3U == LIVE[1])]
    print(f"라이브 문턱 T1={LIVE[0]} T3U={LIVE[1]} — 칸 평균")
    print(f"  연 송금 {base.per_yr.mean():.3f}  이탈95 {base.ht95.mean()*100:.2f}pp  "
          f"칼마 {base.Cal.mean():.3f}  낙폭 {base.MDD.mean()*100:.2f}%")

    print("\n격자 전체에서 값이 얼마나 벌어지나 (칸 평균 기준)")
    agg = df.groupby(['T1', 'T3U']).mean(numeric_only=True)
    for f, unit, mul in (('per_yr', '회/년', 1), ('ht95', 'pp', 100),
                         ('Cal', '', 1), ('MDD', '%', 100)):
        v = agg[f] * mul
        print(f"  {f:7s} 최소 {v.min():8.3f}{unit}  최대 {v.max():8.3f}{unit}  "
              f"라이브 {float(agg.loc[LIVE, f])*mul:8.3f}{unit}")

    print("\n라이브 문턱과 완료 횟수가 같은 쌍 (= 그 범위에서 문턱이 둔감하다)")
    bd = float(base.dones.mean())
    same = agg[np.isclose(agg['dones'], bd, atol=0.5)]
    print(f"  {len(same)} / {len(agg)} 쌍, 완료 {bd:.1f}회")
    if len(same):
        t1s = sorted({i[0] for i in same.index}); t3s = sorted({i[1] for i in same.index})
        print(f"  T1 {t1s}")
        print(f"  T3U {t3s}")

    print("\n이 표로 문턱을 고르지 않는다 — 채택은 사전등록한 기준으로 따로 돌린다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
