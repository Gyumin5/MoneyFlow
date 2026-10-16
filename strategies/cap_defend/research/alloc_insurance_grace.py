#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""유예 무장 비교 — 사전등록(alloc_insurance_grace_prereg.md) 집행기.

앞 측정(alloc_insurance_start.py)은 보험을 발화하는 날 바로 켰다. 토론이 막은 지점이
그것이다 — 제때 송금하는 평시에도 상한이 켜져 연복리를 깎는다.

여기서 무장 시점을 축으로 세운다.
  없음   — 맨몸.
  즉시   — 발화일에 켠다(앞 측정의 규칙).
  유예 D — 발화일로부터 D영업일 뒤 그날, 계좌 이탈이 아직 발화 문턱 위면 켠다.
           돈이 도착해 문턱 아래로 내려가 있으면 안 켠다. 한 번만 본다.

라이브에서 "사람이 팔기 시작했나" 는 알 수 없지만 "세 계좌 이탈이 아직 문턱 위인가" 는
잔고만으로 안다. 그래서 유예 조건을 착수 여부가 아니라 이탈로 잡았다.

두 블록.
  A. 착수 지연 S ∈ {0,5,20,60,없음}, 도착 지연 A = 5영업일 고정.
  B. S = 5영업일 고정, 도착 지연 A ∈ {5,20,60} — 제때 착수했는데 도착이 늦는 세계.

정지 조건: 무장 '없음' · S=0 · A=5 가 1차 기준 팔(alloc_transit_convention.run_one)과
자산곡선 바이트 단위로 같아야 한다.

쓰기: python3 alloc_insurance_grace.py
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
SPANS = (5, 10, 20, 60)
STARTS = (0, 5, 20, 60, None)        # None = 영영 착수 안 함
ARMS = ('none', 'now', 5, 10)        # 무장 정책 — 유예는 영업일 수로 적는다
ARRIVES = (5, 20, 60)                # 블록 B 의 도착 지연
RES = os.path.join(RUN.OUT, "insurance_grace.json")


def arm_name(a):
    return {'none': '맨몸', 'now': '즉시'}.get(a, f'유예{a}bd')


def run_ins(axis, R, gate, tab, start, arrive_bd, arm):
    """착수 지연 start · 도착 지연 arrive_bd · 무장 정책 arm 으로 장부를 돈다."""
    n = len(axis)
    v = TGT.astype(float).copy()
    c = np.zeros(3)
    take = np.zeros(3)
    share = np.zeros(3)
    pend = -1                        # 착수 예정일 (사람이 움직일 날)
    arrive = -1
    arm_at = -1                      # 유예 무장을 판정할 날
    capped = False
    low_since = -1
    tab_s = None if start is None else tab[start]
    tab_a = tab[arrive_bd]
    tab_g = tab[arm] if isinstance(arm, int) else None

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

        # 유예 무장 판정 — 발화일로부터 D영업일 뒤 그날 한 번만 본다
        if arm_at >= 0 and i >= arm_at:
            if not capped and fired(w, ht):
                capped = True
                low_since = -1
                arms += 1
            arm_at = -1

        def initiate():              # 사람이 움직이는 날 — 팔고 출금한다
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
            from_c = np.minimum(c, take)             # 묶인 현금에서 먼저 나간다
            c = c - from_c
            v = v - (take - from_c)
            arrive = int(tab_a[i])                   # 축 밖이면 n — 영영 안 온다

        if pend == i:
            pend = -1
            initiate()

        if arrive >= 0 or pend >= 0:
            continue

        if not fired(w, ht):
            continue

        fires += 1
        if arm == 'now' and not capped:
            capped = True
            low_since = -1
            arms += 1
        elif tab_g is not None and arm_at < 0 and not capped:
            arm_at = int(tab_g[i])
        if tab_s is not None:
            p = int(tab_s[i])
            if p == i:               # 착수 지연 0 — 그 자리에서 판다
                initiate()
            elif p < n:
                pend = p

    return pv, dict(fires=fires, dones=dones, arms=arms,
                    ht95=float(np.percentile(ht_h, 95)), maxht=float(ht_h.max()),
                    ht95e=float(np.percentile(ht_e, 95)), maxhte=float(ht_e.max()),
                    cap_frac=cap_days / n, park=park_sum / n)


KEYS = ['Cal', 'Sharpe', 'CAGR', 'MDD', 'per_yr', 'ht95', 'maxht', 'ht95e',
        'maxhte', 'cap_frac', 'park', 'arms']

CASES_A = [(s, 5, a) for s in STARTS for a in ARMS]
CASES_B = [(5, v, a) for v in ARRIVES for a in ARMS]
CASES = CASES_A + [k for k in CASES_B if k not in CASES_A]


def case_name(s, v, a):
    tag = '착수없음' if s is None else f'착수{s}bd'
    return f"{tag}/도착{v}bd/{arm_name(a)}"


def row(fs, inf, yrs):
    return [float(fs['Cal'][0]), float(fs['Sharpe'][0]), float(fs['CAGR'][0]),
            float(fs['MDD'][0]), inf['dones'] / yrs, inf['ht95'], inf['maxht'],
            inf['ht95e'], inf['maxhte'], inf['cap_frac'], inf['park'],
            inf['arms'] / yrs]


def main() -> int:
    acc = {k: [] for k in CASES}
    for an, st in CELLS:
        axis, Rm, is_bd, _t, gate, mid = cell_inputs(an, st)
        tab = L.bd_tables(axis, is_bd, spans=SPANS)
        yrs = (axis[-1] - axis[0]).days / 365.25

        pv_ref, _ = run_one(axis, Rm, gate, tab, False, month_id=mid)
        pv_b, _ = run_ins(axis, Rm, gate, tab, 0, 5, 'none')
        if not np.array_equal(pv_ref, pv_b):
            raise AssertionError(f"앵커 {an} 비용 {st}: 무장없음 S=0 이 기준 팔과 어긋난다")

        for k in CASES:
            pv, inf = run_ins(axis, Rm, gate, tab, *k)
            acc[k].append(row(RUN.full_stats(pv[:, None], axis), inf, yrs))
        print(f"  앵커 {an} {'5배' if st else '실비용'} 끝")

    A = {k: np.asarray(v) for k, v in acc.items()}
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

    # ── 무장 정책별 효과 (같은 S·A 안에서 맨몸 대비)
    print("\n── 블록 A 효과 (맨몸 대비)")
    eff = {}
    for s in STARTS:
        off = A[(s, 5, 'none')]
        tag = '착수없음' if s is None else f'착수{s}bd'
        for a in ARMS[1:]:
            d = A[(s, 5, a)] - off
            m = d.mean(axis=0)
            bc, bm = int((d[:, 0] > 0).sum()), int((d[:, 3] > 0).sum())
            print(f"  {tag:9s} {arm_name(a):8s} 칼마 {m[0]:+.3f}({bc}/14)  "
                  f"연복리 {m[2]*100:+.2f}pp  최대낙폭 {m[3]*100:+.2f}pp({bm}/14)  "
                  f"상한일 {A[(s,5,a)][:,9].mean()*100:.1f}%")
            eff[f"{tag}/{arm_name(a)}"] = dict(zip(KEYS, [float(x) for x in m]),
                                               better_Cal=bc, better_MDD=bm)
    out['_effect_A'] = eff

    # ── 사전등록 판정 셋
    print("\n── 판정 (사전등록 alloc_insurance_grace_prereg.md)")
    verdict = {}
    for a in (5, 10):
        nm = arm_name(a)
        peace_ok, peace_why = True, []
        for s in (0, 5):
            dg = (A[(s, 5, a)] - A[(s, 5, 'none')]).mean(axis=0)
            dn = (A[(s, 5, 'now')] - A[(s, 5, 'none')]).mean(axis=0)
            capf = A[(s, 5, a)][:, 9].mean()
            ok = (dg[2] > dn[2]) and (capf < 0.01)
            peace_ok &= ok
            peace_why.append(f"착수{s}bd 연복리 {dg[2]*100:+.2f}pp vs 즉시 {dn[2]*100:+.2f}pp, "
                             f"상한일 {capf*100:.2f}%")
        keep_ok, keep_why = True, []
        for s in (20, 60, None):
            tag = '착수없음' if s is None else f'착수{s}bd'
            dg = (A[(s, 5, a)] - A[(s, 5, 'none')]).mean(axis=0)
            dn = (A[(s, 5, 'now')] - A[(s, 5, 'none')]).mean(axis=0)
            ratio = dg[3] / dn[3] if abs(dn[3]) > 1e-12 else float('nan')
            ok = bool(ratio >= 0.80) if dn[3] > 0 else True
            keep_ok &= ok
            keep_why.append(f"{tag} 낙폭개선 {dg[3]*100:+.2f}pp / 즉시 {dn[3]*100:+.2f}pp = {ratio:.2f}")
        harm_ok, worst = True, 0.0
        for k in CASES:
            if k[2] != a:
                continue
            base = A[(k[0], k[1], 'none')]
            d = (A[k] - base).mean(axis=0)[0]
            worst = min(worst, d)
            if d < -BAND['Cal']:
                harm_ok = False
        allp = peace_ok and keep_ok and harm_ok
        print(f"  {nm}: ⑴평시 {'통과' if peace_ok else '미달'} · "
              f"⑵방어 {'통과' if keep_ok else '미달'} · "
              f"⑶무해 {'통과' if harm_ok else '미달'} (최악 칼마차 {worst:+.3f}) "
              f"→ {'최종 규칙 후보' if allp else '탈락'}")
        for t in peace_why + keep_why:
            print(f"      {t}")
        verdict[nm] = dict(peace=bool(peace_ok), defend=bool(keep_ok),
                           harmless=bool(harm_ok),
                           pass_all=bool(allp), worst_Cal=float(worst),
                           why=peace_why + keep_why)
    out['_verdict'] = verdict

    cands = [n for n, v in verdict.items() if v['pass_all']]
    if len(cands) == 2:
        c5 = (A[(5, 5, 5)] - A[(5, 5, 'none')]).mean(axis=0)[2]
        c10 = (A[(5, 5, 10)] - A[(5, 5, 'none')]).mean(axis=0)[2]
        pick = '유예5bd' if (c5 > c10 or abs(c5 - c10) < 1e-9) else '유예10bd'
        note = f"둘 다 통과 — 평시 연복리 손실 유예5 {c5*100:+.2f}pp · 유예10 {c10*100:+.2f}pp → {pick}"
    elif len(cands) == 1:
        pick, note = cands[0], f"{cands[0]} 하나만 통과"
    else:
        pick, note = None, "유예 무장 탈락 — 즉시 무장을 쓰거나 보험을 안 올린다(운영자 판단)"
    print(f"\n결론: {note}")
    out['_pick'] = pick
    out['_note'] = note

    json.dump(out, open(RES, 'w'), ensure_ascii=False, indent=1)
    print(f"저장: {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
