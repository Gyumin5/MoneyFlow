#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""드리프트를 맞춘 대조 귀무 — 사전등록(alloc_t1t3u_matched_prereg.md) 집행기.

흔들림 귀무(afb94fa·ccef199)는 "드리프트가 큰 날에 쏜다" 는 성질과 "바로 그 날" 을
한꺼번에 깬다. 그래서 규칙의 정보와 짧은 역사의 운을 못 가른다.

여기서는 성질은 두고 날짜만 바꾼다. 발화일마다 그 날의 이탈(half-turnover)과 비슷한
날을 전 구간에서 모아 하나를 뽑아 대체한다. 현행 칼마가 그 분포의 어느 분위에
앉는지가 판정이다.

쓰기: python3 alloc_t1t3u_matched.py [--reps 2000]
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import alloc_grid_run as RUN
from alloc_t1t3u_power import LIVE, CELLS, cell_inputs, metrics
from alloc_transit_convention import run_one

OUT = RUN.OUT
RES = os.path.join(OUT, "t1t3u_matched.json")
TOLS = (0.01, 0.02, 0.03)        # 사전등록한 허용오차 확장 사다리
MIN_CAND = 20


def candidates(ht_hist, is_bd, i):
    """발화일 i 의 이탈값과 비슷한 영업일 목록. 사전등록한 사다리대로 넓힌다."""
    star = ht_hist[i]
    for tol in TOLS:
        c = np.flatnonzero(is_bd & (np.abs(ht_hist - star) <= tol))
        if len(c) >= MIN_CAND:
            return c, tol
    return c, TOLS[-1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    a = ap.parse_args()
    rng = np.random.default_rng(20261016)

    out = {}
    pct_cal, pct_ht = [], []
    widened = thin = 0
    for an, st in CELLS:
        axis, Rm, is_bd, tab, gate, mid = cell_inputs(an, st)
        yrs = (axis[-1] - axis[0]).days / 365.25
        n = len(axis)

        pv_b, inf_b = run_one(axis, Rm, gate, tab, False, *LIVE, month_id=mid)
        mb = metrics(pv_b, inf_b, axis, yrs)
        ht_hist = inf_b['ht_hist']
        fi = list(inf_b['fire_idx'])

        cands = []
        for i in fi:
            c, tol = candidates(ht_hist, is_bd, i)
            widened += int(tol > TOLS[0])
            thin += int(len(c) < MIN_CAND)
            cands.append(c)

        rows = []
        for _ in range(a.reps):
            m = np.zeros(n, dtype=bool)
            for c in cands:
                m[c[rng.integers(len(c))]] = True
            pv_j, inf_j = run_one(axis, Rm, gate, tab, False, month_id=mid, cal_mask=m)
            mj = metrics(pv_j, inf_j, axis, yrs)
            rows.append([mj['Cal'], mj['ht95'], float(inf_j['dones'])])
        A = np.asarray(rows)
        p_cal = float((A[:, 0] < mb['Cal']).mean())      # 현행이 이 분위만큼 위에 있다
        p_ht = float((A[:, 1] > mb['ht95']).mean())      # 이탈은 작을수록 좋음
        pct_cal.append(p_cal)
        pct_ht.append(p_ht)
        out[f"a{an}_{'x5' if st else 'base'}"] = dict(
            live_Cal=mb['Cal'], null_Cal_med=float(np.median(A[:, 0])),
            null_Cal_p05=float(np.percentile(A[:, 0], 5)),
            null_Cal_p95=float(np.percentile(A[:, 0], 95)),
            pct_Cal=p_cal, pct_ht95=p_ht,
            live_dones=float(inf_b['dones']), null_dones_med=float(np.median(A[:, 2])))
        print(f"  앵커 {an} {'5배' if st else '실비용'}  현행 칼마 {mb['Cal']:.3f}  "
              f"귀무 중앙 {np.median(A[:,0]):.3f}  분위 {p_cal:.3f}")

    med = float(np.median(pct_cal))
    verdict = ("⑴ 드리프트 크기로 설명 안 됨 — 과대평가 쪽" if med >= 0.90 else
               "⑶ 드리프트 크기로 설명됨 — 규칙의 정보 쪽" if med <= 0.60 else
               "⑵ 부분 설명 — 둘 다 있다")
    print(f"\n칼마 분위 열네 칸 중앙 {med:.3f}  →  {verdict}")
    print(f"이탈95 분위 중앙 {np.median(pct_ht):.3f}")
    print(f"허용오차를 넓힌 발화 {widened}건, 넓히고도 후보 20 미만 {thin}건")
    out['summary'] = dict(pct_Cal_median=med, pct_ht95_median=float(np.median(pct_ht)),
                          widened=int(widened), thin=int(thin), verdict=verdict,
                          reps=int(a.reps))
    json.dump(out, open(RES, 'w'), ensure_ascii=False, indent=1)
    print(f"저장: {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
