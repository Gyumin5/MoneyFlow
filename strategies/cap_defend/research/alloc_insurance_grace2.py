#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""유예 무장 2차 — 사전등록(alloc_insurance_grace2_prereg.md) 집행기.

1차는 발화일 + D영업일째 하루만 봤다. 그날 이탈이 잠깐 문턱 아래면 그 사건은 끝까지
무장을 안 한다. 여기서는 그 하루를 '그 뒤로 매일' 로 바꾼다 — 기한이 지난 뒤로는 매일
다시 보고, 이탈이 문턱 위인 첫날 무장한다.

1차 팔(맨몸·즉시·유예5·유예10)은 1차 실행기 함수를 그대로 불러 같이 돌린다. 1차 결과와
어긋나면 중단한다 — 비교 대상이 흔들린 것이다.

쓰기: python3 alloc_insurance_grace2.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import alloc_grid_run as RUN
import alloc_ledger_sim as L
from alloc_insurance import BAND, CLEAR_HT
from alloc_t1t3u_power import CELLS, cell_inputs
from alloc_insurance_grace import (ARRIVES, KEYS, SPANS, STARTS, row, run_ins)

TGT = L.TGT
T1, T3U, FRIC = L.T1, L.T3U, L.FRIC
DAILY = (5, 10)                      # 상시 무장의 유예 영업일
RES = os.path.join(RUN.OUT, "insurance_grace2.json")
PREV = os.path.join(RUN.OUT, "insurance_grace.json")


def arm_name(a):
    return {'none': '맨몸', 'now': '즉시'}.get(a, f"{'상시' if isinstance(a, tuple) else '유예'}"
                                              f"{a[1] if isinstance(a, tuple) else a}bd")


def run_daily(axis, R, gate, tab, start, arrive_bd, grace_bd):
    """상시 무장 — 발화 + grace_bd 영업일이 지난 뒤로는 매일 다시 본다.

    run_ins 와 같은 장부·같은 상한·같은 해제 조건이고, 무장 판정만 매일이다.
    """
    n = len(axis)
    v = TGT.astype(float).copy()
    c = np.zeros(3)
    take = np.zeros(3)
    share = np.zeros(3)
    pend = -1
    arrive = -1
    watch = -1                       # 이 날부터 매일 본다 (-1 이면 감시 안 함)
    capped = False
    low_since = -1
    tab_s = None if start is None else tab[start]
    tab_a = tab[arrive_bd]
    tab_g = tab[grace_bd]

    pv = np.empty(n)
    ht_h = np.empty(n)
    ht_e = np.empty(n)
    fires = dones = arms = 0
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
            watch = -1               # 돈이 왔다 — 그 사건의 감시는 끝
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

        def fired(ww, hh):
            under = (TGT - ww) / TGT
            return bool(hh >= T1) or bool(((under >= T3U) & gate[i]).any())

        # 상시 무장 — 기한이 지난 뒤로는 매일 본다
        if watch >= 0 and i >= watch and not capped and fired(w, ht):
            capped = True
            low_since = -1
            arms += 1
            watch = -1               # 무장했다 — 그 사건의 감시는 끝

        def initiate():
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
            from_c = np.minimum(c, take)
            c = c - from_c
            v = v - (take - from_c)
            arrive = int(tab_a[i])

        if pend == i:
            pend = -1
            initiate()

        if arrive >= 0 or pend >= 0:
            continue

        if not fired(w, ht):
            continue

        fires += 1
        if watch < 0 and not capped:
            watch = int(tab_g[i])
        if tab_s is not None:
            p = int(tab_s[i])
            if p == i:
                initiate()
            elif p < n:
                pend = p

    return pv, dict(fires=fires, dones=dones, arms=arms,
                    ht95=float(np.percentile(ht_h, 95)), maxht=float(ht_h.max()),
                    ht95e=float(np.percentile(ht_e, 95)), maxhte=float(ht_e.max()),
                    cap_frac=cap_days / n, park=park_sum / n)


OLD = ['none', 'now', 5, 10]
NEW = [(('d'), d) for d in DAILY]
ALL = OLD + NEW
CASES_A = [(s, 5, a) for s in STARTS for a in ALL]
CASES_B = [(5, v, a) for v in ARRIVES for a in ALL]
CASES = CASES_A + [k for k in CASES_B if k not in CASES_A]


def case_name(s, v, a):
    tag = '착수없음' if s is None else f'착수{s}bd'
    return f"{tag}/도착{v}bd/{arm_name(a)}"


def main() -> int:
    acc = {k: [] for k in CASES}
    for an, st in CELLS:
        axis, Rm, is_bd, _t, gate, mid = cell_inputs(an, st)
        tab = L.bd_tables(axis, is_bd, spans=SPANS)
        yrs = (axis[-1] - axis[0]).days / 365.25
        for k in CASES:
            s, av, a = k
            if isinstance(a, tuple):
                pv, inf = run_daily(axis, Rm, gate, tab, s, av, a[1])
            else:
                pv, inf = run_ins(axis, Rm, gate, tab, s, av, a)
            acc[k].append(row(RUN.full_stats(pv[:, None], axis), inf, yrs))
        print(f"  앵커 {an} {'5배' if st else '실비용'} 끝")

    A = {k: np.asarray(v) for k, v in acc.items()}

    # ── 1차와 어긋나면 중단
    prev = json.load(open(PREV))['_levels']
    for k in CASES:
        if isinstance(k[2], tuple):
            continue
        nm = case_name(*k)
        if nm not in prev:
            continue
        m = A[k].mean(axis=0)
        for j, key in enumerate(KEYS):
            a_, b_ = float(m[j]), float(prev[nm][key])
            if abs(a_ - b_) > 1e-12:
                raise AssertionError(f"1차와 어긋난다: {nm} {key} {a_} vs {b_}")
    print("1차 팔 재현 확인 — 열네 칸 평균이 소수점 끝까지 같다")

    out = {'_levels': {case_name(*k): dict(zip(KEYS, [float(x) for x in A[k].mean(axis=0)]))
                       for k in CASES}}

    def block(title, cases):
        print(f"\n── {title}")
        print("  팔                      칼마    샤프   연복리  최대낙폭  상한일 묶인현금 연무장")
        for k in cases:
            m = A[k].mean(axis=0)
            print(f"  {case_name(*k):22s} {m[0]:6.3f} {m[1]:6.3f} {m[2]*100:6.2f}% "
                  f"{m[3]*100:8.2f}% {m[9]*100:6.1f}% {m[10]*100:7.2f}% {m[11]:6.2f}")

    block("블록 A — 착수가 늦는 세계 (도착 5영업일 고정)", CASES_A)
    block("블록 B — 제때 착수, 도착이 늦는 세계 (착수 5영업일 고정)", CASES_B)

    print("\n── 블록 A 효과 (맨몸 대비)")
    eff = {}
    for s in STARTS:
        off = A[(s, 5, 'none')]
        tag = '착수없음' if s is None else f'착수{s}bd'
        for a in ALL[1:]:
            d = A[(s, 5, a)] - off
            m = d.mean(axis=0)
            bc, bm = int((d[:, 0] > 0).sum()), int((d[:, 3] > 0).sum())
            print(f"  {tag:9s} {arm_name(a):8s} 칼마 {m[0]:+.3f}({bc}/14)  "
                  f"연복리 {m[2]*100:+.2f}pp  최대낙폭 {m[3]*100:+.2f}pp({bm}/14)  "
                  f"상한일 {A[(s,5,a)][:,9].mean()*100:.1f}%")
            eff[f"{tag}/{arm_name(a)}"] = dict(zip(KEYS, [float(x) for x in m]),
                                               better_Cal=bc, better_MDD=bm)
    out['_effect_A'] = eff

    print("\n── 판정 (사전등록 alloc_insurance_grace2_prereg.md — 1차와 같은 선)")
    verdict = {}
    for a in NEW:
        nm = arm_name(a)
        peace_ok, why = True, []
        for s in (0, 5):
            dg = (A[(s, 5, a)] - A[(s, 5, 'none')]).mean(axis=0)
            dn = (A[(s, 5, 'now')] - A[(s, 5, 'none')]).mean(axis=0)
            capf = A[(s, 5, a)][:, 9].mean()
            peace_ok &= bool((dg[2] > dn[2]) and (capf < 0.01))
            why.append(f"착수{s}bd 연복리 {dg[2]*100:+.2f}pp vs 즉시 {dn[2]*100:+.2f}pp, "
                       f"상한일 {capf*100:.2f}%")
        keep_ok = True
        for s in (20, 60, None):
            tag = '착수없음' if s is None else f'착수{s}bd'
            dg = (A[(s, 5, a)] - A[(s, 5, 'none')]).mean(axis=0)
            dn = (A[(s, 5, 'now')] - A[(s, 5, 'none')]).mean(axis=0)
            ratio = dg[3] / dn[3] if abs(dn[3]) > 1e-12 else float('nan')
            keep_ok &= bool(ratio >= 0.80) if dn[3] > 0 else keep_ok
            why.append(f"{tag} 낙폭개선 {dg[3]*100:+.2f}pp / 즉시 {dn[3]*100:+.2f}pp = {ratio:.2f}")
        harm_ok, worst = True, 0.0
        for k in CASES:
            if k[2] != a:
                continue
            d = (A[k] - A[(k[0], k[1], 'none')]).mean(axis=0)[0]
            worst = min(worst, d)
            if d < -BAND['Cal']:
                harm_ok = False
        allp = bool(peace_ok and keep_ok and harm_ok)
        print(f"  {nm}: ⑴평시 {'통과' if peace_ok else '미달'} · "
              f"⑵방어 {'통과' if keep_ok else '미달'} · "
              f"⑶무해 {'통과' if harm_ok else '미달'} (최악 칼마차 {worst:+.3f}) "
              f"→ {'최종 규칙 후보' if allp else '탈락'}")
        for t in why:
            print(f"      {t}")
        verdict[nm] = dict(peace=bool(peace_ok), defend=bool(keep_ok),
                           harmless=bool(harm_ok), pass_all=allp,
                           worst_Cal=float(worst), why=why)
    out['_verdict'] = verdict

    cands = [n for n, v in verdict.items() if v['pass_all']]
    if len(cands) == 2:
        c5 = (A[(5, 5, NEW[0])] - A[(5, 5, 'none')]).mean(axis=0)[2]
        c10 = (A[(5, 5, NEW[1])] - A[(5, 5, 'none')]).mean(axis=0)[2]
        pick = arm_name(NEW[0]) if c5 >= c10 else arm_name(NEW[1])
        note = f"둘 다 통과 — 평시 연복리 {c5*100:+.2f}pp 대 {c10*100:+.2f}pp → {pick}"
    elif len(cands) == 1:
        pick, note = cands[0], f"{cands[0]} 하나만 통과"
    else:
        pick, note = None, ("유예 계열을 닫는다 — 즉시 무장을 쓰거나 보험을 안 올린다"
                            "(운영자 판단)")
    print(f"\n결론: {note}")
    out['_pick'] = pick
    out['_note'] = note
    json.dump(out, open(RES, 'w'), ensure_ascii=False, indent=1)
    print(f"저장: {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
