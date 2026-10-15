#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""달력 리밸런싱 대조군 — 월 고정일·분기 고정일을 실제로 돌린다.

왜 필요한가. 본 채점의 대조 목록(alloc_grid_space.CONTROLS)에 '월 고정일' 과
'분기 고정일' 이 있는데 구현이 없었다. control_spec 이 만드는 딕셔너리에 fixed 키가
없어서 fixed='M'/'Q' 가 조용히 떨어지고, 시뮬레이터에도 그 개념이 없다. 그래서 두
대조가 '송금 없음' 과 똑같은 값을 냈고, 보고서가 그걸 "그 날짜 규칙이 이 기간에 송금을
한 번도 못 냈다" 로 잘못 읽었다(2026-10-15 토론이 지적, 보고서·history 정정 완료).

여기서 고치는 방식. 돌고 있는 위약이 alloc_ledger_sim 을 매번 새로 import 하므로 그
모듈은 건드리지 않는다. 대신 공식 팔과 같은 것을 돈다고 이미 확인된 단일 팔 클론
(alloc_transit_convention.run_one, 커밋 ab4b8dd·474d1b1)에 달력 발화일만 주입한다.

규칙. 달력 리밸런싱은 편차를 안 본다 — 정해진 날이 오면 그날의 목표로 되돌린다.
문턱도 카나리 게이트도 안 본다. 월 고정일은 매달 첫 영업일, 분기 고정일은 1·4·7·10월의
첫 영업일이다. 그 밖의 처리(마찰 0.3%, 수락에서 입금까지 5영업일, 뜬 돈 처리)는
현행 팔과 같게 둔다 — 안 그러면 대조가 규칙 차이가 아니라 모형 차이를 재게 된다.

쓰기: python3 alloc_calendar_controls.py [--anchor N]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import alloc_grid_run as RUN
import alloc_ledger_sim as L
import alloc_trigger_space as S
from alloc_transit_convention import run_one

OUT = RUN.OUT


def first_bd_mask(axis, is_bd, months):
    """months 에 든 달의 첫 영업일에만 참인 가면."""
    idx = pd.DatetimeIndex(axis)
    out = np.zeros(len(axis), dtype=bool)
    seen = set()
    for i in range(len(axis)):
        if not is_bd[i]:
            continue
        m = (idx[i].year, idx[i].month)
        if m in seen or idx[i].month not in months:
            continue
        seen.add(m)
        out[i] = True
    return out


def scan_cell(anchor: int, stress: bool):
    eq_st, eq_sp, eq_fu = RUN.sleeve_curves(anchor, stress)
    axis, Rm, is_bd = L.build_ledger(eq_st, eq_sp, eq_fu)
    tab = L.bd_tables(axis, is_bd)
    g = L.gate_series_ledger(axis)
    gate = np.column_stack([np.asarray(g[s].values, dtype=bool) for s in S.SLEEVES])
    mid = np.asarray(pd.DatetimeIndex(axis).year * 12
                     + pd.DatetimeIndex(axis).month, dtype=np.int64)
    yrs = (axis[-1] - axis[0]).days / 365.25

    arms = {
        "현행": dict(),
        "월 고정일": dict(cal_mask=first_bd_mask(axis, is_bd, set(range(1, 13)))),
        "분기 고정일": dict(cal_mask=first_bd_mask(axis, is_bd, {1, 4, 7, 10})),
        "송금 없음": dict(t1=9.9, t3u=9.9),
    }
    rows = []
    for name, kw in arms.items():
        pv, inf = run_one(axis, Rm, gate, tab, False, month_id=mid, **kw)
        fs = RUN.full_stats(pv[:, None], axis)
        rows.append(dict(anchor=anchor, stress=bool(stress), name=name,
                         fires=inf['fires'], dones=inf['dones'],
                         per_yr=inf['dones'] / yrs, multi=inf['multi'],
                         ht95=inf['ht95'], maxht=inf['maxht'],
                         Cal=float(fs['Cal'][0]), MDD=float(fs['MDD'][0]),
                         Sharpe=float(fs['Sharpe'][0]), CAGR=float(fs['CAGR'][0])))
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
    p = os.path.join(OUT, "calendar_controls.csv")
    df.to_csv(p, index=False)
    print(f"\n줄 {len(df)}  저장: {p}\n")

    g = df.groupby('name').mean(numeric_only=True)
    print(f"칸 평균 ({len(anchors)}앵커 × 비용 2)")
    print(f"  {'':10s} {'발화':>6s} {'완료':>6s} {'연':>7s} {'이탈95':>8s} "
          f"{'최대이탈':>9s} {'몰린달':>6s} {'칼마':>6s} {'낙폭':>8s}")
    for n in ("현행", "월 고정일", "분기 고정일", "송금 없음"):
        x = g.loc[n]
        print(f"  {n:10s} {x.fires:6.2f} {x.dones:6.2f} {x.per_yr:6.3f}회 "
              f"{x.ht95*100:7.3f}pp {x.maxht*100:8.3f}pp {x.multi:6.2f} "
              f"{x.Cal:6.3f} {x.MDD*100:7.2f}%")

    print("\n대조가 '송금 없음' 과 같은 값인지 확인 (같으면 구현이 또 빠진 것)")
    for n in ("월 고정일", "분기 고정일"):
        same = bool(np.isclose(g.loc[n, 'dones'], g.loc['송금 없음', 'dones'])
                    and np.isclose(g.loc[n, 'Cal'], g.loc['송금 없음', 'Cal']))
        print(f"  {n}: {'같다 — 구현 확인 필요' if same else '다르다 — 구현됐다'}")

    print("\n이 표로 라이브를 바꾸지 않는다 — 대조군은 측정기가 무엇을 재는지 보이려고 둔다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
