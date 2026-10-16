#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""송금이 늦거나 못 할 때의 보험 — 사전등록(alloc_insurance_prereg.md) 집행기.

발화하면 과한 슬리브가 스스로 상한을 걸어 초과분을 그 계좌 현금으로 묶는다. 그 현금은
수익을 못 낸다. 송금이 도착하거나 계좌 이탈이 0.05 이하로 하루 유지되면 풀고 되돌린다.

송금 도착까지 5·20·60영업일과 '영영 없음' 넷 × 보험 끔·켬 = 여덟 팔을
앵커 7 × 비용 2 = 열네 칸에서 돈다.

정지 조건: 지연 5영업일 · 보험 끔 은 기존 문턱 팔(alloc_transit_convention.run_one)과
바이트 단위로 같아야 한다. 어긋나면 그 자리에서 멈춘다.

쓰기: python3 alloc_insurance.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import alloc_grid_run as RUN
import alloc_ledger_sim as L
from alloc_t1t3u_power import CELLS, cell_inputs
from alloc_transit_convention import run_one

OUT = RUN.OUT
RES = os.path.join(OUT, "insurance.json")

TGT = L.TGT
T1, T3U, FRIC = L.T1, L.T3U, L.FRIC
DELAYS = (5, 20, 60, None)            # None = 송금 없음
CLEAR_HT = 0.05                       # 2026-05-23 채택안의 해제 조건 그대로
BAND = dict(Cal=1.307, MDD=0.07280, ht95=0.08497, maxht=0.23577, per_yr=0.684)


def run_ins(axis, R, gate, tab, delay, insure):
    """보험을 켜거나 끄고 한 줄 돌린다. delay=None 이면 송금이 영영 없다."""
    n = len(axis)
    v = TGT.astype(float).copy()
    c = np.zeros(3)                   # 보험으로 묶인 현금 — 슬리브별, 수익 0
    take = np.zeros(3)                # 공중에 뜬 돈
    share = np.zeros(3)
    arrive = -1
    capped = False
    low_since = -1
    lag = None if delay is None else tab[delay]

    pv = np.empty(n)
    ht_h = np.empty(n)
    ht_e = np.empty(n)
    fires = dones = 0
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
            if capped:                # 송금 도착 = 해제
                v = v + c
                c = np.zeros(3)
                capped = False
                low_since = -1

        tot = v.sum() + c.sum() + take.sum()
        if capped:                    # 목표까지 상한 — 넘으면 현금으로, 모자라면 되돌린다
            own = v + c
            nv = np.minimum(own, tot * TGT)
            c = own - nv
            v = nv
            cap_days += 1
            park_sum += c.sum() / tot

        w = (v + c) / tot             # 계좌 기준 (뜬 돈은 분모에만)
        ht = float(np.abs(w - TGT).sum() / 2)
        he = float(np.abs(v / tot - TGT).sum() / 2)   # 노출 기준
        pv[i] = tot
        ht_h[i] = ht
        ht_e[i] = he

        if capped:                    # 해제 조건 ⑵ — 이탈 0.05 이하 하루 유지
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

        if arrive >= 0:               # 이미 떠 있는 동안은 새로 안 쏜다
            continue

        under = (TGT - w) / TGT
        u = bool(((under >= T3U) & gate[i]).any())
        if not (bool(ht >= T1) or u):
            continue

        fires += 1
        if insure and not capped:
            capped = True
            low_since = -1
        if lag is None:               # 송금 없음 — 돈은 계좌 사이를 안 넘는다
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

    return pv, dict(fires=fires, dones=dones,
                    ht95=float(np.percentile(ht_h, 95)), maxht=float(ht_h.max()),
                    ht95e=float(np.percentile(ht_e, 95)), maxhte=float(ht_e.max()),
                    cap_frac=cap_days / n, park=park_sum / n)


ARMS = [(d, ins) for d in DELAYS for ins in (False, True)]
KEYS = ['Cal', 'Sharpe', 'CAGR', 'MDD', 'per_yr', 'ht95', 'maxht', 'ht95e',
        'maxhte', 'cap_frac', 'park']


def name(d, ins):
    return f"{'없음' if d is None else f'{d}영업일'}/{'보험' if ins else '맨몸'}"


def main() -> int:
    acc = {a: [] for a in ARMS}
    for an, st in CELLS:
        axis, Rm, is_bd, _tab, gate, mid = cell_inputs(an, st)
        tab = L.bd_tables(axis, is_bd, spans=(5, 20, 60))
        yrs = (axis[-1] - axis[0]).days / 365.25

        pv_ref, _ = run_one(axis, Rm, gate, tab, False, month_id=mid)
        pv_b, _ = run_ins(axis, Rm, gate, tab, 5, False)
        if not np.array_equal(pv_ref, pv_b):
            raise AssertionError(f"앵커 {an} 비용 {st}: 기준 팔이 문턱 팔과 어긋난다")

        for d, ins in ARMS:
            pv, inf = run_ins(axis, Rm, gate, tab, d, ins)
            fs = RUN.full_stats(pv[:, None], axis)
            acc[(d, ins)].append([
                float(fs['Cal'][0]), float(fs['Sharpe'][0]), float(fs['CAGR'][0]),
                float(fs['MDD'][0]), inf['dones'] / yrs, inf['ht95'], inf['maxht'],
                inf['ht95e'], inf['maxhte'], inf['cap_frac'], inf['park']])
        print(f"  앵커 {an} {'5배' if st else '실비용'} 끝")

    A = {a: np.asarray(v) for a, v in acc.items()}
    print("\n── 열네 칸 평균")
    print("  팔              칼마    샤프   연복리  최대낙폭  연완료 계좌이탈95 노출이탈95 상한일 묶인현금")
    for a in ARMS:
        m = A[a].mean(axis=0)
        print(f"  {name(*a):14s} {m[0]:6.3f} {m[1]:6.3f} {m[2]*100:6.2f}% "
              f"{m[3]*100:8.2f}% {m[4]:6.3f} {m[5]*100:8.2f}pp {m[7]*100:8.2f}pp "
              f"{m[9]*100:6.1f}% {m[10]*100:6.2f}%")

    print("\n── 보험 효과 (같은 지연 안에서 보험 - 맨몸, 괄호는 나아진 칸 수)")
    verdict = []
    for d in DELAYS:
        off, on = A[(d, False)], A[(d, True)]
        dd = on - off
        m = dd.mean(axis=0)
        bc = int((dd[:, 0] > 0).sum())
        bm = int((dd[:, 3] > 0).sum())            # MDD 는 음수 폭이라 +가 개선
        be = int((dd[:, 7] < 0).sum())
        tag = '없음' if d is None else f'{d}영업일'
        print(f"  {tag:8s} 칼마 {m[0]:+.3f}({bc}/14)  샤프 {m[1]:+.3f}  "
              f"연복리 {m[2]*100:+.2f}pp  최대낙폭 {m[3]*100:+.2f}pp({bm}/14)")
        print(f"            노출이탈95 {m[7]*100:+.2f}pp({be}/14)  "
              f"노출최대이탈 {m[8]*100:+.2f}pp  상한일 {on[:,9].mean()*100:.1f}%  "
              f"묶인현금 {on[:,10].mean()*100:.2f}%")
        helps = (m[3] > BAND['MDD']) and (m[0] > -BAND['Cal']) and bm >= 11
        harms = m[0] < -BAND['Cal']
        verdict.append((tag, bool(helps), bool(harms), float(m[0]), float(m[3])))

    any_help = any(h for _, h, _, _, _ in verdict)
    any_harm = any(x for _, _, x, _, _ in verdict)
    v = ("⑴ 값을 한다" if any_help else
         "⑶ 해롭다" if any_harm else "⑵ 값을 안 한다 — 개선이 전부 잡음 폭 안")
    print(f"\n판정: {v}")
    print(f"  (기준 최대낙폭 개선 > {BAND['MDD']*100:.3f}pp 이고 칼마 악화 < "
          f"{BAND['Cal']:.3f} 이며 낙폭 방향이 11/14 이상)")

    out = dict(levels={f"{name(*a)}": dict(zip(KEYS, [float(x) for x in A[a].mean(axis=0)]))
                       for a in ARMS},
               verdict=v, band=BAND,
               per_delay=[dict(delay=t, helps=h, harms=x, dCal=cc, dMDD=mm)
                          for t, h, x, cc, mm in verdict])
    json.dump(out, open(RES, 'w'), ensure_ascii=False, indent=1)
    print(f"저장: {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
