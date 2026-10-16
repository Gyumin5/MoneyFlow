#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""착수 지연을 가른 재측정 — 사전등록(alloc_insurance_start_prereg.md) 집행기.

1차(alloc_insurance.py)가 잰 '지연' 은 도착 지연이었다. 발화하는 날 과한 슬리브가 곧바로
팔리고 돈만 늦게 닿는 세계라, 맨몸 팔에도 위험 축소가 이미 들어가 있었다.

여기서는 지연을 둘로 가른다.
  S = 착수 지연 — 발화일부터 사람이 팔고 출금을 시작하기까지의 영업일.
  A = 도착 지연 — 착수부터 입금까지, 5영업일 고정.
맨몸은 S 가 지나기 전엔 아무것도 안 한다(초과 노출 유지). 보험은 발화 당일 자동으로 켜져
과한 쪽을 목표까지 깎아 그 계좌 현금으로 묶는다.

정지 조건: S=0 · 보험 끔 이 1차의 '5영업일 · 보험 끔'(= alloc_transit_convention.run_one)과
자산곡선 바이트 단위로 같아야 한다.

쓰기: python3 alloc_insurance_start.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import alloc_grid_run as RUN
import alloc_ledger_sim as L
from alloc_insurance import BAND, CLEAR_HT
from alloc_t1t3u_power import CELLS, cell_inputs
from alloc_transit_convention import run_one

TGT = L.TGT
T1, T3U, FRIC = L.T1, L.T3U, L.FRIC
ARRIVE_BD = 5                       # 착수 → 입금, 고정
STARTS = (0, 5, 20, 60, None)       # None = 영영 착수 안 함
RES = os.path.join(RUN.OUT, "insurance_start.json")


def run_start(axis, R, gate, tab, start, insure):
    """착수 지연 start 영업일. start=None 이면 사람이 영영 안 움직인다."""
    n = len(axis)
    v = TGT.astype(float).copy()
    c = np.zeros(3)
    take = np.zeros(3)
    share = np.zeros(3)
    pend = -1                       # 착수 예정일 (사람이 움직일 날)
    arrive = -1
    capped = False
    low_since = -1
    tab_s = None if start is None else tab[start]
    tab_a = tab[ARRIVE_BD]

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
            if capped:
                v = v + c
                c = np.zeros(3)
                capped = False
                low_since = -1

        tot = v.sum() + c.sum() + take.sum()
        if capped:
            own = v + c
            nv = np.minimum(own, tot * TGT)
            c = own - nv
            v = nv
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

        def initiate():             # 사람이 움직이는 날 — 팔고 출금한다
            nonlocal v, c, take, share, arrive
            tgt_v = tot * TGT
            own = v + c
            over = np.maximum(own - tgt_v, 0.0)
            short = np.maximum(tgt_v - own, 0.0)
            amt = min(short.sum(), over.sum())
            if amt <= 1e-12:
                return
            take = over * (amt / max(over.sum(), 1e-12))
            share = short * (amt / max(short.sum(), 1e-12)) * (1.0 - FRIC)
            from_c = np.minimum(c, take)            # 묶인 현금에서 먼저 나간다
            c = c - from_c
            v = v - (take - from_c)
            arrive = int(tab_a[i])                  # 축 밖이면 n — 영영 안 온다

        if pend == i:
            pend = -1
            initiate()

        if arrive >= 0 or pend >= 0:
            continue

        under = (TGT - w) / TGT
        u = bool(((under >= T3U) & gate[i]).any())
        if not (bool(ht >= T1) or u):
            continue

        fires += 1
        if insure and not capped:
            capped = True
            low_since = -1
        if tab_s is not None:
            p = int(tab_s[i])
            if p == i:              # 착수 지연 0 — 그 자리에서 판다(1차 기준 팔과 같은 세계)
                initiate()
            elif p < n:
                pend = p

    return pv, dict(fires=fires, dones=dones,
                    ht95=float(np.percentile(ht_h, 95)), maxht=float(ht_h.max()),
                    ht95e=float(np.percentile(ht_e, 95)), maxhte=float(ht_e.max()),
                    cap_frac=cap_days / n, park=park_sum / n)


ARMS = [(s, ins) for s in STARTS for ins in (False, True)]
KEYS = ['Cal', 'Sharpe', 'CAGR', 'MDD', 'per_yr', 'ht95', 'maxht', 'ht95e',
        'maxhte', 'cap_frac', 'park']


def name(s, ins):
    return f"{'착수없음' if s is None else f'착수{s}bd'}/{'보험' if ins else '맨몸'}"


def main() -> int:
    acc = {a: [] for a in ARMS}
    for an, st in CELLS:
        axis, Rm, is_bd, _t, gate, mid = cell_inputs(an, st)
        tab = L.bd_tables(axis, is_bd, spans=(5, 20, 60))
        yrs = (axis[-1] - axis[0]).days / 365.25

        pv_ref, _ = run_one(axis, Rm, gate, tab, False, month_id=mid)
        pv_b, _ = run_start(axis, Rm, gate, tab, 0, False)
        if not np.array_equal(pv_ref, pv_b):
            raise AssertionError(f"앵커 {an} 비용 {st}: S=0 맨몸이 1차 기준 팔과 어긋난다")

        for a in ARMS:
            pv, inf = run_start(axis, Rm, gate, tab, *a)
            fs = RUN.full_stats(pv[:, None], axis)
            acc[a].append([float(fs['Cal'][0]), float(fs['Sharpe'][0]),
                           float(fs['CAGR'][0]), float(fs['MDD'][0]),
                           inf['dones'] / yrs, inf['ht95'], inf['maxht'],
                           inf['ht95e'], inf['maxhte'], inf['cap_frac'], inf['park']])
        print(f"  앵커 {an} {'5배' if st else '실비용'} 끝")

    A = {a: np.asarray(v) for a, v in acc.items()}
    print("\n── 열네 칸 평균")
    print("  팔                칼마    샤프   연복리  최대낙폭  연완료 노출이탈95 상한일 묶인현금")
    for a in ARMS:
        m = A[a].mean(axis=0)
        print(f"  {name(*a):16s} {m[0]:6.3f} {m[1]:6.3f} {m[2]*100:6.2f}% "
              f"{m[3]*100:8.2f}% {m[4]:6.3f} {m[7]*100:8.2f}pp {m[9]*100:6.1f}% "
              f"{m[10]*100:6.2f}%")

    print("\n── 보험 효과 (같은 착수 지연 안에서 보험 - 맨몸)")
    out, verd = {}, []
    for s in STARTS:
        off, on = A[(s, False)], A[(s, True)]
        d = on - off
        m = d.mean(axis=0)
        bc = int((d[:, 0] > 0).sum())
        bm = int((d[:, 3] > 0).sum())
        tag = '착수없음' if s is None else f'착수{s}bd'
        print(f"  {tag:9s} 칼마 {m[0]:+.3f}({bc}/14)  샤프 {m[1]:+.3f}  "
              f"연복리 {m[2]*100:+.2f}pp  최대낙폭 {m[3]*100:+.2f}pp({bm}/14)")
        print(f"            노출이탈95 {m[7]*100:+.2f}pp  노출최대이탈 {m[8]*100:+.2f}pp  "
              f"연완료 {m[4]:+.3f}  상한일 {on[:,9].mean()*100:.1f}%  "
              f"묶인현금 {on[:,10].mean()*100:.2f}%")
        helps = (s is not None and s > 0 or s is None) and \
                (m[3] > BAND['MDD']) and (m[0] > -BAND['Cal']) and bm >= 11
        harms = (m[0] < -BAND['Cal'])
        verd.append((tag, bool(helps), bool(harms)))
        out[tag] = dict(zip(KEYS, [float(x) for x in m]),
                        better_Cal=bc, better_MDD=bm, helps=bool(helps),
                        harms=bool(harms))

    judge = ("⑴ 값을 한다" if any(h for t, h, _ in verd if t != '착수0bd') else
             "⑶ 해롭다" if any(x for _, _, x in verd) else
             "⑵ 값을 안 한다 — 개선이 전부 잡음 폭 안")
    print(f"\n판정: {judge}")
    d0 = (A[(0, True)] - A[(0, False)]).mean(axis=0)
    print(f"  평시 비용(착수 0bd): 칼마 {d0[0]:+.3f}  연복리 {d0[2]*100:+.2f}pp  "
          f"상한일 {A[(0,True)][:,9].mean()*100:.1f}%")
    out['_judge'] = judge
    out['_levels'] = {name(*a): dict(zip(KEYS, [float(x) for x in A[a].mean(axis=0)]))
                      for a in ARMS}
    json.dump(out, open(RES, 'w'), ensure_ascii=False, indent=1)
    print(f"저장: {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
