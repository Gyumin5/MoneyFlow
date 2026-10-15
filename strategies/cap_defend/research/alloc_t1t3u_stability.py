#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""T1·T3U 문턱의 국소 안정성 — 사전등록(alloc_t1t3u_prereg.md) 집행기.

묻는 것은 "어느 쌍이 제일 좋은가" 가 아니라 "라이브 쌍 (20pp, 20%) 이 평평한 자리에
있는가" 다. 최고 성적 쌍을 고르는 판정은 사전등록 문서에서 금지로 못박았다.

단계가 둘이고 순서가 규약이다.
  1단계  운영 지표만 적어 파일로 닫는다 — 발화일, 발화·완료 횟수, 연 환산, 이탈 95분위와
         최대, 한 달에 두 번 이상 입금된 달. 이 파일에는 성과가 한 칸도 안 들어간다.
  2단계  1단계 파일을 닫은 뒤에만 성과를 계산해 따로 쓴다. 보조 진단으로만 읽는다.

쓰기:
  python3 alloc_t1t3u_stability.py --pilot     # 한 칸만, 시간 재기 (먼저 이걸 돌린다)
  python3 alloc_t1t3u_stability.py             # 14칸 한 벌
  python3 alloc_t1t3u_stability.py --judge     # 이미 쓴 1단계 파일로 판정만 다시
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import time

import numpy as np
import pandas as pd

import alloc_grid_run as RUN
import alloc_ledger_sim as L
import alloc_trigger_space as S
from alloc_transit_convention import run_one

OUT = RUN.OUT
OPS = os.path.join(OUT, "t1t3u_stability_ops.csv")        # 1단계 — 운영 지표만
PERF = os.path.join(OUT, "t1t3u_stability_perf.csv")      # 2단계 — 성과
FIRES = os.path.join(OUT, "t1t3u_stability_fires.json")   # 발화일

GRID = (0.15, 0.175, 0.20, 0.225, 0.25, 0.30)
LIVE = (0.20, 0.20)
STEP = 0.025
EPS = 1e-9
# 사전등록한 통과 폭 — 결과를 보고 안 움직인다
MAX_HT95_WORSE = 0.02
MAX_MAXHT_WORSE = 0.02
MAX_PERYR_DIFF = 0.5
NEAR_BD = 5                      # 발화일 대응을 이 영업일 안에서 본다

CONTROLS = ("송금 없음", "T1만", "T3U만")


def _cell_inputs(anchor: int, stress: bool):
    eq_st, eq_sp, eq_fu = RUN.sleeve_curves(anchor, stress)
    axis, Rm, is_bd = L.build_ledger(eq_st, eq_sp, eq_fu)
    tab = L.bd_tables(axis, is_bd)
    g = L.gate_series_ledger(axis)
    gate = np.column_stack([np.asarray(g[s].values, dtype=bool) for s in S.SLEEVES])
    mid = np.asarray(pd.DatetimeIndex(axis).year * 12
                     + pd.DatetimeIndex(axis).month, dtype=np.int64)
    return axis, Rm, is_bd, tab, gate, mid


def fidelity_check(axis, Rm, is_bd, tab, gate, mid):
    """클론이 공식 팔과 같은 것을 도는지 (20, 20) 에서 다시 확인한다."""
    EV, _, _ = L.event_matrix(S.build_E(L.gate_series_ledger(axis)), axis)
    pv_o, inf_o = L.simulate_ledger(Rm, is_bd, gate, EV, RUN.control_spec("현행"),
                                    tab, month_id=mid)
    _, inf_c = run_one(axis, Rm, gate, tab, False, *LIVE, month_id=mid)
    out = dict(
        official=dict(fires=int(inf_o['fires'][0]), dones=int(inf_o['dones'][0]),
                      ht95=float(inf_o['ht_p95'][0]), maxht=float(inf_o['max_ht'][0])),
        clone=dict(fires=inf_c['fires'], dones=inf_c['dones'],
                   ht95=inf_c['ht95'], maxht=inf_c['maxht']))
    ok = (out['official']['fires'] == out['clone']['fires']
          and out['official']['dones'] == out['clone']['dones']
          and abs(out['official']['ht95'] - out['clone']['ht95']) < 1e-6
          and abs(out['official']['maxht'] - out['clone']['maxht']) < 1e-6)
    out['ok'] = bool(ok)
    return out


def scan_cell(anchor: int, stress: bool):
    """한 칸에서 36쌍 + 대조 3 의 운영 지표와 발화일을 뽑는다. 성과는 여기서 안 본다."""
    axis, Rm, is_bd, tab, gate, mid = _cell_inputs(anchor, stress)
    yrs = (axis[-1] - axis[0]).days / 365.25
    rows, fires, curves = [], {}, {}

    def add(name, t1, t3u):
        pv, inf = run_one(axis, Rm, gate, tab, False, t1, t3u, month_id=mid)
        rows.append(dict(anchor=anchor, stress=bool(stress), name=name,
                         T1=t1, T3U=t3u,
                         fires=inf['fires'], dones=inf['dones'],
                         per_yr=inf['dones'] / yrs, multi=inf['multi'],
                         ht95=inf['ht95'], maxht=inf['maxht']))
        fires[name] = [str(axis[i].date()) for i in inf['fire_idx']]
        curves[name] = pv

    for t1, t3u in itertools.product(GRID, GRID):
        add(f"{t1:.3f}/{t3u:.3f}", t1, t3u)
    # 대조 — 격자 밖. 큰 값으로 그 성분을 꺼서 만든다(밴드·전이 축은 안 쓴다).
    add(CONTROLS[0], 9.9, 9.9)
    add(CONTROLS[1], LIVE[0], 9.9)
    add(CONTROLS[2], 9.9, LIVE[1])
    return rows, fires, curves, axis, is_bd


def match_rates(live_dates, nb_dates, axis, is_bd):
    """발화일 비교 둘.

    정확 = 두 집합의 교집합 / 합집합. 대응 = 라이브 발화일 중 이웃이 NEAR_BD 영업일
    안에 대응하는 날의 비율. 뒤는 라이브 기준이라 비대칭이다 — "라이브가 쏜 날을
    이웃도 비슷한 때 쏘나" 를 묻는 것이고, 이웃이 더 자주 쏘는 것은 연 완료 횟수
    기준이 따로 본다.
    """
    sa, sb = set(live_dates), set(nb_dates)
    if not sa and not sb:
        return 1.0, 1.0
    exact = len(sa & sb) / max(len(sa | sb), 1)
    bpos = {str(d.date()): k for k, d in enumerate(axis)}
    bd_rank = np.cumsum(is_bd) - 1          # 각 날의 영업일 번호
    rb = [bd_rank[bpos[e]] for e in sb]
    near = sum(1 for d in sa
               if any(abs(r - bd_rank[bpos[d]]) <= NEAR_BD for r in rb))
    return float(exact), float(near / max(len(sa), 1))


def neighbors():
    out = []
    for dt1, dt3 in itertools.product((-STEP, 0.0, STEP), repeat=2):
        if dt1 == 0.0 and dt3 == 0.0:
            continue
        out.append((round(LIVE[0] + dt1, 4), round(LIVE[1] + dt3, 4)))
    return out


def judge(df, fires_all, axes):
    """주 판정 — 이웃 여덟이 14칸 전부에서 등록한 폭 안인가."""
    live_key = f"{LIVE[0]:.3f}/{LIVE[1]:.3f}"
    breaches, rates = [], []
    for (t1, t3u) in neighbors():
        key = f"{t1:.3f}/{t3u:.3f}"
        for (an, st), sub in df.groupby(['anchor', 'stress']):
            b = sub[sub.name == live_key]
            c = sub[sub.name == key]
            if b.empty or c.empty:
                breaches.append(dict(nb=key, anchor=an, stress=st, what="칸 없음"))
                continue
            b, c = b.iloc[0], c.iloc[0]
            for what, d, lim in (
                    ("이탈95", c.ht95 - b.ht95, MAX_HT95_WORSE),
                    ("최대이탈", c.maxht - b.maxht, MAX_MAXHT_WORSE),
                    ("연완료", abs(c.per_yr - b.per_yr), MAX_PERYR_DIFF)):
                if d > lim + EPS:
                    breaches.append(dict(nb=key, anchor=int(an), stress=bool(st),
                                         what=what, diff=float(d), lim=lim))
            ck = (int(an), bool(st))
            axis, is_bd = axes[ck]
            ex, nr = match_rates(fires_all[ck][live_key], fires_all[ck][key],
                                 axis, is_bd)
            rates.append(dict(nb=key, anchor=int(an), stress=bool(st),
                              exact=ex, near=nr))
    return breaches, pd.DataFrame(rates)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pilot", action="store_true")
    ap.add_argument("--judge", action="store_true")
    a = ap.parse_args()

    cells = [(0, False)] if a.pilot else [(an, st) for an in range(7)
                                          for st in (False, True)]

    rows, fires_all, curves_all, axes = [], {}, {}, {}
    t0 = time.time()
    for an, st in cells:
        tc = time.time()
        r, f, cv, axis, is_bd = scan_cell(an, st)
        rows += r
        fires_all[(an, bool(st))] = f
        curves_all[(an, bool(st))] = cv
        axes[(an, bool(st))] = (axis, is_bd)
        print(f"  앵커 {an} {'5배' if st else '실비용'} 끝 ({time.time()-tc:.1f}초)")
    el = time.time() - t0

    df = pd.DataFrame(rows)
    df.to_csv(OPS, index=False)                       # ── 1단계 닫음
    with open(FIRES, 'w') as fh:
        json.dump({f"{k[0]}_{int(k[1])}": v for k, v in fires_all.items()}, fh)
    print(f"\n1단계 운영 지표 {len(df)}줄 → {OPS}")
    print(f"발화일 → {FIRES}")

    if a.pilot:
        print(f"\n한 칸 {el:.1f}초 → 14칸 추정 {el*14/60:.1f}분")
        fid = fidelity_check(*_cell_inputs(0, False))
        print(f"클론 대조 (20,20) 앵커0 실비용: {'일치' if fid['ok'] else '어긋남'}")
        print(f"  공식 {fid['official']}")
        print(f"  클론 {fid['clone']}")
        if el * 14 > 7200:
            print("\n등록한 비용 상한(두 시간) 초과 — 돌리기 전에 멈춘다.")
        return 0

    fid = fidelity_check(*_cell_inputs(0, False))
    print(f"\n클론 대조 (20,20) 앵커0 실비용: {'일치' if fid['ok'] else '어긋남'}")
    print(f"  공식 {fid['official']}")
    print(f"  클론 {fid['clone']}")
    if not fid['ok']:
        print("\n클론이 공식 팔과 어긋난다 — 사전등록한 멈춤 조건. 판정 미확정.")
        return 1

    br, rates = judge(df, fires_all, axes)
    live_key = f"{LIVE[0]:.3f}/{LIVE[1]:.3f}"
    print(f"\n── 주 판정 — 이웃 여덟 × {len(cells)}칸")
    if not br:
        print("  어긴 자리 0 — 라이브 쌍은 평평한 자리에 있다. 라이브 변경 권고 없음.")
    else:
        print(f"  어긴 자리 {len(br)}건 — 라이브 쌍은 가장자리에 있다.")
        for x in br[:20]:
            print(f"    {x}")
    print("\n  발화일 대응 (라이브 대비, 이웃별 14칸 평균)")
    for nb, sub in rates.groupby('nb'):
        print(f"    {nb}  정확 {sub.exact.mean()*100:5.1f}%  "
              f"{NEAR_BD}영업일내 {sub.near.mean()*100:5.1f}%")

    # ── 2단계 — 여기서야 성과를 본다
    prows = []
    for (an, st), cv in curves_all.items():
        axis = axes[(an, st)][0]
        for name, pv in cv.items():
            fs = RUN.full_stats(pv[:, None], axis)
            prows.append(dict(anchor=an, stress=st, name=name,
                              Cal=float(fs['Cal'][0]), MDD=float(fs['MDD'][0]),
                              Sharpe=float(fs['Sharpe'][0]),
                              CAGR=float(fs['CAGR'][0])))
    pf = pd.DataFrame(prows)
    pf.to_csv(PERF, index=False)
    print(f"\n2단계 성과 {len(pf)}줄 → {PERF}  (보조 진단 — 주 판정에 안 들어간다)")
    g = pf.groupby('name')[['Cal', 'MDD']].mean()
    lv = g.loc[live_key]
    print(f"  라이브 {live_key}: 칼마 {lv.Cal:.3f} 낙폭 {lv.MDD*100:.2f}%")
    print(f"  격자 칼마 최소 {g.Cal.min():.3f} 최대 {g.Cal.max():.3f}")
    print("\n최고 성적 쌍을 고르지 않는다 — 사전등록 금지 항목이다.")
    print(f"총 {el/60:.1f}분")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
