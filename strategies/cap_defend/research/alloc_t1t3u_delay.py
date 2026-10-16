#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""송금 지연 축 — 같은 잣대(흔들림 잡음 폭)로 다시 잰다.

묻는 것: "발화 즉시 옮긴다" 의 이득이 그 날짜들이 짧은 역사에서 우연히 좋은 자리였기
때문인가. 재는 법은 간단하다 — 라이브 쌍이 실제로 쏜 날 전부를 공통으로 d 영업일
밀어 보고(d = -5 … +10), 지표가 d 를 따라 어떻게 움직이는지 본다.

읽는 법은 alloc_t1t3u_power.py 가 낸 잡음 폭과 대조한다. d=0 과 d>0 의 차이가 그
폭 안이면 "즉시성" 은 이 자료로 주장할 수 없는 크기다. 폭을 넘고 d 를 따라 단조로
무너지면 날짜 며칠이 성과를 가른다는 뜻이고, 송금 규칙에서 그건 신호가 아니라
과적합 징후로 읽는다.

쓰기: python3 alloc_t1t3u_delay.py
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

import alloc_grid_run as RUN
from alloc_t1t3u_power import LIVE, CELLS, cell_inputs, metrics
from alloc_transit_convention import run_one

OUT = RUN.OUT
RES = os.path.join(OUT, "t1t3u_delay.json")
DELAYS = list(range(-5, 11))          # 영업일. 음수 = 앞당김
KEYS = ['ht95', 'maxht', 'per_yr', 'Cal', 'MDD']


def shifted_mask(fire_idx, bd_rank, bd_pos, n, d):
    m = np.zeros(n, dtype=bool)
    last = len(bd_pos) - 1
    for i in fire_idx:
        m[bd_pos[min(max(bd_rank[i] + d, 0), last)]] = True
    return m


def main() -> int:
    t0 = time.time()
    rows = {d: [] for d in DELAYS}
    for an, st in CELLS:
        axis, Rm, is_bd, tab, gate, mid = cell_inputs(an, st)
        yrs = (axis[-1] - axis[0]).days / 365.25
        n = len(axis)
        bd_pos = np.flatnonzero(is_bd)
        bd_rank = np.maximum(np.cumsum(is_bd) - 1, 0)

        pv_b, inf_b = run_one(axis, Rm, gate, tab, False, *LIVE, month_id=mid)
        mb = metrics(pv_b, inf_b, axis, yrs)
        for d in DELAYS:
            m = shifted_mask(inf_b['fire_idx'], bd_rank, bd_pos, n, d)
            pv_d, inf_d = run_one(axis, Rm, gate, tab, False, month_id=mid, cal_mask=m)
            md = metrics(pv_d, inf_d, axis, yrs)
            rows[d].append([md[k] - mb[k] for k in KEYS] + [md['Cal'], md['MDD']])
        print(f"  앵커 {an} {'5배' if st else '실비용'} 끝 ({time.time()-t0:.0f}초)")

    A = {d: np.asarray(rows[d]) for d in DELAYS}     # (칸, 지표+2)
    # d=0 은 문턱 팔과 완전히 같지 않다 — 발화일 중 영업일이 아닌 날(칸당 0~2건)이
    # 영업일 번호 왕복에서 직전 영업일로 당겨지기 때문이다. 이 '스냅' 효과를 따로 적고,
    # 지연의 크기는 d=0 을 기준으로 읽는다.
    snap = A[0]
    print("\n── 스냅 효과 (d=0 빼기 문턱 팔, 열네 칸 평균) — 지연의 기준선")
    print(f"   Δ이탈95 {snap[:,0].mean()*100:+.3f}pp  Δ연완료 {snap[:,2].mean():+.3f}  "
          f"Δ칼마 {snap[:,3].mean():+.3f}  Δ낙폭 {snap[:,4].mean()*100:+.3f}pp")

    out = {}
    print("\n── 공통 지연 d 영업일 (열네 칸 평균, 라이브 d=0 대비 차이)")
    print("   d    Δ이탈95     Δ최대이탈    Δ연완료     Δ칼마    Δ낙폭pp   칼마수준")
    for d in DELAYS:
        a = A[d] - np.pad(snap[:, :5], ((0, 0), (0, 2)))   # d=0 기준
        mean = a.mean(axis=0)
        out[str(d)] = dict(zip(KEYS, [float(x) for x in mean[:5]]),
                           Cal_level=float(mean[5]), MDD_level=float(mean[6]),
                           Cal_worse_cells=int((a[:, 3] < 0).sum()))
        print(f"  {d:+3d}  {mean[0]*100:+8.3f}pp {mean[1]*100:+8.3f}pp "
              f"{mean[2]:+8.3f}  {mean[3]:+8.3f}  {mean[4]*100:+8.3f}  {mean[5]:7.3f}")

    # 잡음 폭과 대조
    pj = os.path.join(OUT, "t1t3u_power.json")
    if os.path.exists(pj):
        band = json.load(open(pj))['band']
        print("\n── 흔들림 잡음 폭 대비 (양수면 폭을 넘음)")
        for d in (1, 2, 3, 5, 10):
            a = A[d] - np.pad(snap[:, :5], ((0, 0), (0, 2)))
            cal = float(np.abs(a[:, 3]).max())
            ht = float(a[:, 0].max())
            print(f"  d={d:+3d}  칼마 최대차 {cal:6.3f} (폭 {band['Cal']:.3f}) "
                  f"{'넘음' if cal > band['Cal'] else '안'}   "
                  f"이탈95 최대차 {ht*100:6.3f}pp (폭 {band['ht95']*100:.3f}pp) "
                  f"{'넘음' if ht > band['ht95'] else '안'}")
        out['band'] = band

    json.dump(out, open(RES, 'w'), ensure_ascii=False, indent=1)
    print(f"\n저장: {RES}  ({time.time()-t0:.0f}초)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
