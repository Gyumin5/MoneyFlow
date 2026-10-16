#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""보험의 개선이 상한 규칙 때문인가 현금 보유 자체 때문인가 — 현금 위약 대조.

alloc_insurance.py 의 보험은 두 가지를 동시에 한다. ⑴ 과한 슬리브를 목표까지 깎는다
⑵ 그만큼을 현금으로 들고 있다. 성과가 좋아진 것이 ⑴ 때문인지 ⑵ 때문인지 못 가른다.

그래서 위약을 만든다. 같은 날 켜지고 같은 조건으로 꺼지되, 깎는 대상을 과한 슬리브가
아니라 세 슬리브 전부로 하고(비중 그대로 유지) 묶는 금액은 보험 팔의 평균과 맞춘다.
개선이 위약에서도 그대로 나오면 값은 현금에서 나온 것이고, 안 나오면 상한 규칙에서
나온 것이다.

이건 사후 분석이다 — 사전등록(alloc_insurance_prereg.md)이 보호하지 않는다.

쓰기: python3 alloc_insurance_placebo.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import alloc_grid_run as RUN
import alloc_ledger_sim as L
from alloc_insurance import CLEAR_HT, DELAYS, RES, name, run_ins
from alloc_t1t3u_power import CELLS, cell_inputs

TGT = L.TGT
T1, T3U, FRIC = L.T1, L.T3U, L.FRIC
OUTF = os.path.join(RUN.OUT, "insurance_placebo.json")


def run_pl(axis, R, gate, tab, delay, frac):
    """보험과 같은 날 켜지지만 비중은 안 건드리고 현금만 frac 만큼 묶는다."""
    n = len(axis)
    v = TGT.astype(float).copy()
    c = np.zeros(3)
    take = np.zeros(3)
    share = np.zeros(3)
    arrive = -1
    capped = False
    low_since = -1
    lag = None if delay is None else tab[delay]

    pv = np.empty(n)
    ht_h = np.empty(n)
    ht_e = np.empty(n)
    dones = 0
    cap_days = 0
    park_sum = 0.0

    for i in range(n):
        v *= (1.0 + R[i])

        if arrive == i:
            v += share
            take = np.zeros(3)
            share = np.zeros(3)
            arrive = -1
            dones += 1
            if capped:
                v = v + c
                c = np.zeros(3)
                capped = False
                low_since = -1

        tot = v.sum() + c.sum() + take.sum()
        if capped:
            own = v + c
            want = min(frac * tot, own.sum())
            c = own * (want / max(own.sum(), 1e-12))
            v = own - c
            cap_days += 1
            park_sum += c.sum() / tot

        w = (v + c) / tot
        ht = float(np.abs(w - TGT).sum() / 2)
        he = float(np.abs(v / tot - TGT).sum() / 2)
        pv[i] = tot
        ht_h[i] = ht
        ht_e[i] = he

        if capped:
            if ht <= CLEAR_HT:
                if low_since < 0:
                    low_since = i
                elif i - low_since >= 1:
                    v = v + c
                    c = np.zeros(3)
                    capped = False
                    low_since = -1
            else:
                low_since = -1

        if arrive >= 0:
            continue

        under = (TGT - w) / TGT
        u = bool(((under >= T3U) & gate[i]).any())
        if not (bool(ht >= T1) or u):
            continue

        if not capped:
            capped = True
            low_since = -1
        if lag is None:
            continue

        tgt_v = tot * TGT
        own = v + c
        over = np.maximum(own - tgt_v, 0.0)
        short = np.maximum(tgt_v - own, 0.0)
        amt = min(short.sum(), over.sum())
        if amt <= 1e-12:
            continue
        take = over * (amt / max(over.sum(), 1e-12))
        share = short * (amt / max(short.sum(), 1e-12)) * (1.0 - FRIC)
        from_v = np.minimum(v, take)
        v = v - from_v
        c = np.maximum(c - (take - from_v), 0.0)
        arrive = int(lag[i])

    return pv, dict(dones=dones, ht95=float(np.percentile(ht_h, 95)),
                    maxht=float(ht_h.max()), ht95e=float(np.percentile(ht_e, 95)),
                    maxhte=float(ht_e.max()), cap_frac=cap_days / n, park=park_sum / n)


def main() -> int:
    prev = json.load(open(RES))['levels']
    # 보험이 켜져 있는 동안의 평균 현금 비중 = 묶인현금 / 상한일
    frac = {}
    for d in DELAYS:
        lv = prev[name(d, True)]
        frac[d] = lv['park'] / max(lv['cap_frac'], 1e-9)
    print("  맞출 현금 비중 (보험 켜진 동안):",
          "  ".join(f"{'없음' if d is None else f'{d}bd'} {frac[d]*100:.2f}%"
                    for d in DELAYS))

    ins, pla = {d: [] for d in DELAYS}, {d: [] for d in DELAYS}
    for an, st in CELLS:
        axis, Rm, is_bd, _t, gate, mid = cell_inputs(an, st)
        tab = L.bd_tables(axis, is_bd, spans=(5, 20, 60))
        for d in DELAYS:
            for store, fn in ((ins, None), (pla, frac[d])):
                if fn is None:
                    pv, inf = run_ins(axis, Rm, gate, tab, d, True)
                else:
                    pv, inf = run_pl(axis, Rm, gate, tab, d, fn)
                fs = RUN.full_stats(pv[:, None], axis)
                store[d].append([float(fs['Cal'][0]), float(fs['Sharpe'][0]),
                                 float(fs['CAGR'][0]), float(fs['MDD'][0]),
                                 inf['ht95e'], inf['maxhte'], inf['park']])
        print(f"  앵커 {an} {'5배' if st else '실비용'} 끝")

    out = {}
    print("\n── 보험 대 현금 위약 (열네 칸 평균)")
    print("  지연      팔      칼마    샤프   연복리  최대낙폭  노출이탈95  묶인현금")
    for d in DELAYS:
        tag = '없음' if d is None else f'{d}영업일'
        A, B = np.asarray(ins[d]), np.asarray(pla[d])
        for lbl, M in (('보험', A), ('위약', B)):
            m = M.mean(axis=0)
            print(f"  {tag:8s} {lbl}  {m[0]:6.3f} {m[1]:6.3f} {m[2]*100:6.2f}% "
                  f"{m[3]*100:8.2f}% {m[4]*100:9.2f}pp {m[6]*100:7.2f}%")
        dd = A - B
        m = dd.mean(axis=0)
        print(f"           차이  {m[0]:+6.3f} {m[1]:+6.3f} {m[2]*100:+6.2f}pp "
              f"{m[3]*100:+7.2f}pp {m[4]*100:+8.2f}pp   "
              f"칼마 나은 칸 {int((dd[:,0]>0).sum())}/14")
        out[tag] = dict(ins=[float(x) for x in A.mean(axis=0)],
                        pla=[float(x) for x in B.mean(axis=0)],
                        dCal=float(m[0]), dMDD=float(m[3]),
                        better=int((dd[:, 0] > 0).sum()))
    json.dump(out, open(OUTF, 'w'), ensure_ascii=False, indent=1)
    print(f"\n저장: {OUTF}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
