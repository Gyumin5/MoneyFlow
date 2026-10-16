#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""흔들림 벌점의 분해 — 운인가 규칙인가.

afb94fa 는 "발화일을 흔들면 86% 의 복제가 실제보다 나쁘다" 를 냈고, 5df8b89 는
"앞당기면 나빠지고 늦추면 거의 공짜" 를 냈다. 둘을 잇는 물음 하나가 남는다 —
그 벌점이 흔든 방향(앞/뒤)으로 얼마나 설명되나. 방향으로 설명되면 그건 규칙의
성질(드리프트가 쌓인 뒤에 쏜다)이고, 설명이 안 남는 부분이 시점 운이다.

복제마다 평균 부호 이동(영업일)과 사라진 발화 수를 같이 적고, Δ칼마를 그 둘로
설명해 본다. 설명 못 한 흩어짐이 이 자료가 말하는 "운" 의 크기다.

쓰기: python3 alloc_t1t3u_decomp.py [--reps 600]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import alloc_grid_run as RUN
from alloc_t1t3u_power import LIVE, CELLS, JIT_BD, cell_inputs, metrics
from alloc_transit_convention import run_one

OUT = RUN.OUT
RES = os.path.join(OUT, "t1t3u_decomp.json")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=600)
    a = ap.parse_args()
    rng = np.random.default_rng(20261016)

    rows = []
    for an, st in CELLS:
        axis, Rm, is_bd, tab, gate, mid = cell_inputs(an, st)
        yrs = (axis[-1] - axis[0]).days / 365.25
        n = len(axis)
        bd_pos = np.flatnonzero(is_bd)
        bd_rank = np.maximum(np.cumsum(is_bd) - 1, 0)
        last = len(bd_pos) - 1

        pv_b, inf_b = run_one(axis, Rm, gate, tab, False, *LIVE, month_id=mid)
        mb = metrics(pv_b, inf_b, axis, yrs)
        fi = np.asarray(inf_b['fire_idx'])
        # 기준선은 스냅(영업일 왕복)된 d=0 — 5df8b89 와 같은 기준
        m0 = np.zeros(n, dtype=bool)
        m0[bd_pos[bd_rank[fi]]] = True
        pv0, inf0 = run_one(axis, Rm, gate, tab, False, month_id=mid, cal_mask=m0)
        m0m = metrics(pv0, inf0, axis, yrs)

        for _ in range(a.reps):
            off = rng.integers(-JIT_BD, JIT_BD + 1, size=len(fi))
            r = np.clip(bd_rank[fi] + off, 0, last)
            m = np.zeros(n, dtype=bool)
            m[bd_pos[r]] = True
            pv_j, inf_j = run_one(axis, Rm, gate, tab, False, month_id=mid, cal_mask=m)
            mj = metrics(pv_j, inf_j, axis, yrs)
            rows.append([an, float(st), float(off.mean()),
                         float(inf0['dones'] - inf_j['dones']),
                         mj['Cal'] - m0m['Cal'], mj['ht95'] - m0m['ht95']])
        print(f"  앵커 {an} {'5배' if st else '실비용'} 끝")

    A = np.asarray(rows)
    mean_off, lost, dcal = A[:, 2], A[:, 3], A[:, 4]

    print(f"\n복제 {len(A)}개 (칸 14 × {a.reps})  기준선 = 스냅된 d=0")
    print(f"  Δ칼마 < 0 비율 {np.mean(dcal < 0)*100:.1f}%   중앙 {np.median(dcal):+.3f}")
    keep = lost == 0
    print(f"  사건 수 보존 복제 {keep.mean()*100:.1f}%   그 안에서 Δ칼마<0 "
          f"{np.mean(dcal[keep] < 0)*100:.1f}%  중앙 {np.median(dcal[keep]):+.3f}")

    print("\n평균 부호 이동 구간별 Δ칼마 (보존 복제만)")
    edges = [-5, -1.5, -0.75, -0.25, 0.25, 0.75, 1.5, 5]
    for lo, hi in zip(edges[:-1], edges[1:]):
        s = keep & (mean_off >= lo) & (mean_off < hi)
        if s.sum() < 10:
            continue
        print(f"  [{lo:+.2f},{hi:+.2f})  n={s.sum():5d}  중앙 {np.median(dcal[s]):+.3f}  "
              f"평균 {dcal[s].mean():+.3f}")

    # 방향·손실로 얼마나 설명되나 (보존 복제만: 손실 항은 0 이라 방향만)
    X = np.column_stack([np.ones(keep.sum()), mean_off[keep]])
    y = dcal[keep]
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    pred = X @ beta
    r2 = 1 - np.var(y - pred) / np.var(y)
    print(f"\n보존 복제에서 Δ칼마 를 평균 이동으로만 설명: 기울기 {beta[1]:+.4f}/영업일, "
          f"절편 {beta[0]:+.4f}, 설명력 {r2*100:.1f}%")
    print(f"  설명 뒤 남는 흩어짐(잔차 표준편차) {np.std(y - pred):.3f} "
          f"— 이 자료가 말하는 시점 운의 크기")

    X2 = np.column_stack([np.ones(len(A)), mean_off, lost])
    beta2, *_ = np.linalg.lstsq(X2, dcal, rcond=None)
    pred2 = X2 @ beta2
    r22 = 1 - np.var(dcal - pred2) / np.var(dcal)
    print(f"\n전체 복제에서 평균 이동 + 사라진 발화 수로 설명: 이동 {beta2[1]:+.4f}, "
          f"손실 {beta2[2]:+.4f}/건, 설명력 {r22*100:.1f}%")

    json.dump(dict(reps=int(a.reps), n=len(A),
                   worse_share=float(np.mean(dcal < 0)),
                   keep_share=float(keep.mean()),
                   worse_share_keep=float(np.mean(dcal[keep] < 0)),
                   median_keep=float(np.median(dcal[keep])),
                   slope_per_bd=float(beta[1]), r2_keep=float(r2),
                   resid_sd=float(np.std(y - pred)),
                   r2_all=float(r22), slope_lost=float(beta2[2])),
              open(RES, 'w'), ensure_ascii=False, indent=1)
    print(f"\n저장: {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
