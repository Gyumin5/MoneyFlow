#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""공중에 뜬 돈을 비중의 어디에 두나 — 두 규약의 차이를 잰다 (기술만 한다).

왜 재나. 토론이 "라이브 판정과 연구 장부의 정합성" 을 첫 단계로 지목했고, 하나씩
대조하다 실제로 갈리는 자리를 하나 찾았다. 비중의 분모·분자 처리다.

  라이브(recommend_personal.py 2711~2746행). 세 계좌 조회값을 그대로 더해 분모를 만들고
    각 계좌 평가액을 분자로 쓴다. 공중에 뜬 돈이라는 상태가 아예 없다 — 사람이 팔아서
    현금으로 들고 있는 동안 그 돈은 여전히 보내는 계좌의 평가액에 잡힌다.
  연구(alloc_ledger_sim.py do_request·simulate_ledger). 요청하는 순간 보내는 슬리브의
    v 에서 빼고 transit 에 올린다. 분모(tot)에는 들어가지만 어느 슬리브의 분자도 아니다.
    그래서 도착까지 닷새 동안 목표배분 이탈(ht)이 실제보다 크게 잡힌다.

이 스크립트가 하는 것. 현행 조합(C = T1 OR T3U) 한 줄만, 두 규약으로 각각 돌려
발화·완료 횟수와 이탈 지표·성과를 앵커 7 × 비용 2 = 14칸에서 비교한다.
안 하는 것. 규약 교체. 어느 쪽이 옳은지는 이 표가 아니라 운영 사실이 정한다.

쓰기: python3 alloc_transit_convention.py [--anchor N]
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

import alloc_grid_run as RUN
import alloc_ledger_sim as L
import alloc_trigger_space as S

OUT = RUN.OUT
TGT = L.TGT


def run_one(axis, R, gate, tab, keep_sender: bool, t1=None, t3u=None,
            month_id=None, cal_mask=None):
    """현행 조합 한 줄을 돌린다. keep_sender=True 면 뜬 돈을 보내는 쪽 비중에 남긴다.

    t1 / t3u 를 주면 그 문턱으로 돈다(기본은 모듈 상수 = 라이브값). 반환하는 info 에
    발화일 색인 목록 fire_idx 가 들어 있다 — 공식 시뮬은 날짜를 안 주기 때문이다.

    cal_mask 를 주면 문턱을 안 보고 그 날에만 쏜다 — 달력 리밸런싱 대조군용이다.
    편차가 얼마든 정해진 날에 목표로 되돌린다는 규칙이라 게이트도 안 본다.

    두 규약에서 뜬 돈의 수익 처리는 똑같다 — 요청하는 순간 v 에서 빠져 수익 0 으로
    닷새를 난다. 갈리는 것은 비중 w 의 분자뿐이다. 그래야 "분자를 어디에 두나" 하나만
    재고 수익 차이가 섞이지 않는다.
    """
    n = len(axis)
    T1 = L.T1 if t1 is None else float(t1)
    T3U = L.T3U if t3u is None else float(t3u)
    v = TGT.astype(float).copy()
    take = np.zeros(3)                       # 보내는 쪽에서 뺀 금액 (수익 0 으로 뜬다)
    share = np.zeros(3)                      # 마찰 뗀 뒤 받는 쪽에 들어갈 금액
    arrive = -1

    pv = np.empty(n)
    ht_hist = np.empty(n)
    fires = dones = 0
    max_ht = 0.0
    fire_idx = []
    mo_cnt = 0
    multi_mo = 0
    cur_mo = int(month_id[0]) if month_id is not None and n else 0
    lag = tab[L.LAG_BD]

    for i in range(n):
        if month_id is not None and int(month_id[i]) != cur_mo:
            multi_mo += int(mo_cnt >= 2)
            mo_cnt = 0
            cur_mo = int(month_id[i])
        v *= (1.0 + R[i])

        if arrive == i:
            v += share
            take = np.zeros(3)
            share = np.zeros(3)
            arrive = -1
            dones += 1
            mo_cnt += 1

        tot = v.sum() + take.sum()
        w = (v + take) / tot if keep_sender else v / tot
        ht = np.abs(w - TGT).sum() / 2
        pv[i] = tot
        ht_hist[i] = ht
        max_ht = max(max_ht, ht)

        if arrive >= 0:                      # 이미 떠 있는 동안은 새로 안 쏜다
            continue

        if cal_mask is not None:
            if not cal_mask[i]:
                continue
        else:
            under = (TGT - w) / TGT
            u = bool(((under >= T3U) & gate[i]).any())
            hit_t1 = bool(ht >= T1)
            if not (hit_t1 or u):
                continue

        fires += 1
        fire_idx.append(i)
        tgt_v = tot * TGT
        over = np.maximum(v - tgt_v, 0.0)
        short = np.maximum(tgt_v - v, 0.0)
        amt = min(short.sum(), over.sum())
        if amt <= 1e-12:
            continue
        take = over * (amt / max(over.sum(), 1e-12))
        share = short * (amt / max(short.sum(), 1e-12)) * (1.0 - L.FRIC)
        v -= take
        arrive = int(lag[i])
    if month_id is not None:
        multi_mo += int(mo_cnt >= 2)
    return pv, dict(fires=fires, dones=dones, multi=multi_mo,
                    ht95=float(np.percentile(ht_hist, 95)), maxht=float(max_ht),
                    fire_idx=fire_idx, ht_hist=ht_hist)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--anchor", type=int, default=None)
    a = ap.parse_args()
    anchors = [a.anchor] if a.anchor is not None else list(range(7))

    rows = []
    for an in anchors:
        for st in (False, True):
            eq_st, eq_sp, eq_fu = RUN.sleeve_curves(an, st)
            axis, Rm, is_bd = L.build_ledger(eq_st, eq_sp, eq_fu)
            tab = L.bd_tables(axis, is_bd)
            g = L.gate_series_ledger(axis)
            gate = np.column_stack([np.asarray(g[s].values, dtype=bool)
                                    for s in S.SLEEVES])
            yrs = (axis[-1] - axis[0]).days / 365.25
            for keep in (False, True):
                pv, inf = run_one(axis, Rm, gate, tab, keep)
                fs = RUN.full_stats(pv[:, None], axis)
                rows.append(dict(anchor=an, stress=bool(st),
                                 rule='보내는쪽유지' if keep else '연구현행',
                                 fires=inf['fires'], dones=inf['dones'],
                                 per_yr=inf['dones'] / yrs,
                                 ht95=inf['ht95'], maxht=inf['maxht'],
                                 Cal=float(fs['Cal'][0]), MDD=float(fs['MDD'][0]),
                                 Sharpe=float(fs['Sharpe'][0]),
                                 CAGR=float(fs['CAGR'][0])))
            print(f"  앵커 {an} {'5배' if st else '실비용'} 끝")

    df = pd.DataFrame(rows)
    p = os.path.join(OUT, "transit_convention.csv")
    df.to_csv(p, index=False)
    print(f"\n줄 {len(df)}  저장: {p}\n")

    g = df.groupby('rule').mean(numeric_only=True)
    print("칸 평균 (앵커 7 × 비용 2 = 14칸)")
    for r in ('연구현행', '보내는쪽유지'):
        x = g.loc[r]
        print(f"  {r:8s} 발화 {x.fires:6.2f}  완료 {x.dones:6.2f}  연 {x.per_yr:5.3f}회  "
              f"이탈95 {x.ht95*100:6.3f}pp  최대이탈 {x.maxht*100:6.3f}pp  "
              f"칼마 {x.Cal:5.3f}  낙폭 {x.MDD*100:7.3f}%")
    d = g.loc['보내는쪽유지'] - g.loc['연구현행']
    print(f"\n  차이(유지 - 현행) 완료 {d.dones:+.2f}회  이탈95 {d.ht95*100:+.3f}pp  "
          f"최대이탈 {d.maxht*100:+.3f}pp  칼마 {d.Cal:+.3f}  낙폭 {d.MDD*100:+.3f}pp")

    w = df.pivot_table(index=['anchor', 'stress'], columns='rule', values='dones')
    same = int((w['연구현행'] == w['보내는쪽유지']).sum())
    print(f"\n  완료 횟수가 같은 칸 {same}/{len(w)}")
    print("\n이 표로 규약을 바꾸지 않는다 — 어느 쪽이 옳은지는 운영 사실이 정한다.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
