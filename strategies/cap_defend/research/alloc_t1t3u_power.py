#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""송금 문턱 시험의 검출 능력 — 사전등록(alloc_t1t3u_power_prereg.md) 집행기.

묻는 것은 "어느 문턱이 좋은가" 가 아니라 "이 측정기가 문턱 차이를 가를 힘이 있나" 다.
어제의 국소 안정성 시험은 격자 어느 중심에 걸어도 실패했다. 그러면 그 실패는 문턱이
뾰족하다는 뜻이 아니라 시험이 아무것도 못 가른다는 뜻일 수 있다.

귀무는 "같은 규칙, 같은 사건, 시점만 조금 다른 세계" 다. 라이브 쌍이 실제로 쏜 날을
영업일 ±5 안에서 흔들어 날짜 가면으로 주입한다. 흔들지 않은 주입은 문턱 팔과 완전히
같아야 하고(멈춤 조건), 그래서 이 가면이 귀무를 만드는 자격을 얻는다.

통과 폭은 복제마다 열네 칸의 차이 최댓값을 취한 분포의 95분위다. 이렇게 잡아야
"한 칸이라도 넘으면 구분됐다" 는 규칙의 거짓경보율이 전체로 5% 가 된다.

쓰기:
  python3 alloc_t1t3u_power.py --reps 2000
  python3 alloc_t1t3u_power.py --reps 50 --pilot     # 시간 재기
"""
from __future__ import annotations

import argparse
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
RES = os.path.join(OUT, "t1t3u_power.json")
OPS = os.path.join(OUT, "t1t3u_stability_ops.csv")
PERF = os.path.join(OUT, "t1t3u_stability_perf.csv")

LIVE = (0.20, 0.20)
JIT_BD = 5                       # 발화일을 흔드는 폭 (영업일)
CELLS = [(an, st) for an in range(7) for st in (False, True)]
# 어제 손으로 정한 폭 — 거짓경보율을 재는 대상이지 이 측정의 판정 기준이 아니다
OLD_BAND = dict(ht95=0.02, maxht=0.02, per_yr=0.5)
# 지표 방향: True = 커지면 나쁨(한쪽), False = 절대값(양쪽)
ONE_SIDED = dict(ht95=True, maxht=True, per_yr=False, Cal=False, MDD=False)
POWER_TARGET = 0.8


def cell_inputs(anchor: int, stress: bool):
    eq = RUN.sleeve_curves(anchor, stress)
    axis, Rm, is_bd = L.build_ledger(*eq)
    tab = L.bd_tables(axis, is_bd)
    g = L.gate_series_ledger(axis)
    gate = np.column_stack([np.asarray(g[s].values, dtype=bool) for s in S.SLEEVES])
    mid = np.asarray(pd.DatetimeIndex(axis).year * 12
                     + pd.DatetimeIndex(axis).month, dtype=np.int64)
    return axis, Rm, is_bd, tab, gate, mid


def metrics(pv, inf, axis, yrs):
    fs = RUN.full_stats(pv[:, None], axis)
    return dict(ht95=inf['ht95'], maxht=inf['maxht'], per_yr=inf['dones'] / yrs,
                Cal=float(fs['Cal'][0]), MDD=float(fs['MDD'][0]))


def jitter_mask(fire_idx, bd_rank, bd_pos, n, rng):
    """발화일을 영업일 ±JIT_BD 안에서 독립적으로 흔든 날짜 가면.

    영업일 번호로 옮긴 뒤 다시 달력 색인으로 되돌린다 — 주말로 밀려나지 않게.
    같은 날에 둘이 겹치면 하나로 합쳐진다(집합). 떠 있는 돈과 겹쳐 사라지는 발화는
    시뮬레이터가 알아서 건너뛴다. 둘 다 시점 잡음의 일부로 본다.
    """
    m = np.zeros(n, dtype=bool)
    last = len(bd_pos) - 1
    for i in fire_idx:
        r = bd_rank[i] + int(rng.integers(-JIT_BD, JIT_BD + 1))
        m[bd_pos[min(max(r, 0), last)]] = True
    return m


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=2000)
    ap.add_argument("--pilot", action="store_true")
    a = ap.parse_args()
    cells = CELLS[:1] if a.pilot else CELLS
    rng = np.random.default_rng(20261016)

    t0 = time.time()
    base, null = {}, {}
    fidelity = []
    for an, st in cells:
        axis, Rm, is_bd, tab, gate, mid = cell_inputs(an, st)
        yrs = (axis[-1] - axis[0]).days / 365.25
        n = len(axis)
        bd_pos = np.flatnonzero(is_bd)                  # 영업일 번호 → 달력 색인
        bd_rank = np.maximum(np.cumsum(is_bd) - 1, 0)   # 달력 색인 → 영업일 번호

        pv_b, inf_b = run_one(axis, Rm, gate, tab, False, *LIVE, month_id=mid)
        mb = metrics(pv_b, inf_b, axis, yrs)
        base[(an, st)] = mb

        # 멈춤 조건 — 흔들지 않은 가면이 문턱 팔과 완전히 같은가
        m0 = np.zeros(n, dtype=bool)
        m0[inf_b['fire_idx']] = True
        pv_c, inf_c = run_one(axis, Rm, gate, tab, False, month_id=mid, cal_mask=m0)
        fidelity.append(bool(np.array_equal(pv_b, pv_c)
                             and inf_b['dones'] == inf_c['dones']))

        rows = []
        for _ in range(a.reps):
            m = jitter_mask(inf_b['fire_idx'], bd_rank, bd_pos, n, rng)
            pv_j, inf_j = run_one(axis, Rm, gate, tab, False, month_id=mid, cal_mask=m)
            mj = metrics(pv_j, inf_j, axis, yrs)
            rows.append([mj[k] - mb[k] for k in ONE_SIDED])
        null[(an, st)] = np.asarray(rows)
        print(f"  앵커 {an} {'5배' if st else '실비용'} 끝 ({time.time()-t0:.0f}초 누적)")

    if not all(fidelity):
        print("\n흔들지 않은 가면이 문턱 팔과 어긋난다 — 사전등록한 멈춤 조건. 판정 미확정.")
        return 1
    print(f"\n멈춤 조건 통과 — {len(cells)}칸 전부에서 가면 주입이 문턱 팔과 자산곡선까지 동일.")

    keys = list(ONE_SIDED)
    # 복제별 '열네 칸 최댓값' 분포
    D = np.stack([null[c] for c in cells])            # (칸, 복제, 지표)
    np.save(os.path.join(OUT, "t1t3u_power_null.npy"), D)   # 원시 차이 — 서술 통계용
    D_dir = np.where(np.array([ONE_SIDED[k] for k in keys]), D, np.abs(D))
    fam = D_dir.max(axis=0)                           # (복제, 지표)
    band = {k: float(np.percentile(fam[:, i], 95)) for i, k in enumerate(keys)}

    # 어제 폭의 거짓경보율 — 같은 귀무에서 '한 칸이라도 넘는' 비율
    old_fp = {}
    for i, k in enumerate(keys):
        if k in OLD_BAND:
            old_fp[k] = float((fam[:, i] > OLD_BAND[k]).mean())
    old_fp['셋 중 하나라도'] = float(np.mean(np.any(
        [fam[:, keys.index(k)] > v for k, v in OLD_BAND.items()], axis=0)))

    # 최소검출효과 — 참 효과 δ 를 모든 칸에 더하고 같은 규칙을 적용
    mde = {}
    for i, k in enumerate(keys):
        scale = band[k] if band[k] > 0 else 1e-6
        grid = np.linspace(0, 6 * scale, 241)[1:]
        found = None
        for d in grid:
            shifted = (D_dir[:, :, i] + d).max(axis=0)
            if (shifted > band[k]).mean() >= POWER_TARGET:
                found = float(d)
                break
        mde[k] = found
    power_curve = {k: [[float(x), float(((D_dir[:, :, keys.index(k)] + x).max(axis=0)
                                         > band[k]).mean())]
                       for x in np.linspace(0, 3 * max(band[k], 1e-9), 13)]
                   for k in keys}

    # 실측 — 이웃 여덟과 격자 36쌍이 라이브와 얼마나 다른가 (열네 칸 최댓값)
    ops = pd.read_csv(OPS)
    perf = pd.read_csv(PERF)
    mg = ops.merge(perf, on=['anchor', 'stress', 'name'])
    live_key = f"{LIVE[0]:.3f}/{LIVE[1]:.3f}"
    obs = {}
    for name, sub in mg.groupby('name'):
        if name in ("송금 없음", "T1만", "T3U만"):
            continue
        d = {}
        for k in keys:
            col = dict(ht95='ht95', maxht='maxht', per_yr='per_yr', Cal='Cal', MDD='MDD')[k]
            b = mg[mg.name == live_key].set_index(['anchor', 'stress'])[col]
            c = sub.set_index(['anchor', 'stress'])[col]
            dd = (c - b).dropna()
            d[k] = float(dd.max() if ONE_SIDED[k] else dd.abs().max())
        obs[name] = d

    step = 0.025
    nbs = [f"{round(LIVE[0]+dx,4):.3f}/{round(LIVE[1]+dy,4):.3f}"
           for dx in (-step, 0.0, step) for dy in (-step, 0.0, step)
           if not (dx == 0.0 and dy == 0.0)]

    def over(name, k):
        return mde[k] is not None and obs[name][k] > mde[k]

    nb_over = [(nb, k) for nb in nbs for k in ('ht95', 'maxht', 'per_yr') if over(nb, k)]
    grid_over = [(nm, k) for nm in obs for k in ('ht95', 'maxht', 'per_yr') if over(nm, k)]

    print("\n── 귀무 폭 (복제 {}개, 열네 칸 최댓값의 95분위)".format(a.reps))
    for k in keys:
        unit = 'pp' if k in ('ht95', 'maxht', 'MDD') else ''
        v = band[k] * 100 if k in ('ht95', 'maxht', 'MDD') else band[k]
        mv = (mde[k] * 100 if k in ('ht95', 'maxht', 'MDD') else mde[k]) if mde[k] is not None else None
        print(f"  {k:7s} 폭 {v:7.3f}{unit}   최소검출효과 "
              + (f"{mv:7.3f}{unit}" if mv is not None else "  없음(검출력 0.8 미달)"))

    print("\n── 어제 손으로 정한 폭의 거짓경보율 (같은 귀무)")
    for k, v in old_fp.items():
        print(f"  {k:12s} {v*100:5.1f}%")

    print("\n── 실측 차이 (열네 칸 최댓값) 대 최소검출효과")
    print(f"  이웃 여덟 중 최소검출효과를 넘는 자리: {len(nb_over)}건")
    for nb, k in nb_over:
        print(f"    {nb} {k} {obs[nb][k]:.4f} > {mde[k]:.4f}")
    print(f"  격자 36쌍 중 넘는 자리: {len(grid_over)}건")
    for nm, k in grid_over[:12]:
        print(f"    {nm} {k} {obs[nm][k]:.4f} > {mde[k]:.4f}")

    print("\n── 보조: 성과 축 (주 판정에 안 들어간다)")
    for k in ('Cal', 'MDD'):
        worst = max(obs, key=lambda nm: obs[nm][k])
        unit = 'pp' if k == 'MDD' else ''
        sc = 100 if k == 'MDD' else 1
        print(f"  {k}: 귀무 폭 {band[k]*sc:.3f}{unit}, 격자 최대 차이 {obs[worst][k]*sc:.3f}{unit} ({worst})")
        print(f"      15/15 대 라이브 차이 {obs['0.150/0.150'][k]*sc:.3f}{unit}"
              f" — {'폭 밖' if obs['0.150/0.150'][k] > band[k] else '폭 안'}")

    verdict = ("⑴ 이웃이 넘는다 — 라이브 주변이 평평하지 않다" if nb_over else
               ("⑵ 이웃도 격자도 못 넘는다 — 이 자료로는 문턱을 가를 수 없다"
                if not grid_over else
                "⑶ 이웃은 못 넘고 격자 멀리는 넘는다 — 라이브 주변은 평평하다"))
    print(f"\n판정 {verdict}")

    with open(RES, 'w') as fh:
        json.dump(dict(reps=a.reps, jit_bd=JIT_BD, cells=len(cells),
                       band=band, mde=mde, old_band=OLD_BAND, old_fp=old_fp,
                       power_curve=power_curve, obs=obs,
                       nb_over=nb_over, grid_over=grid_over, verdict=verdict,
                       base={f"{c[0]}_{int(c[1])}": v for c, v in base.items()}), fh,
                  ensure_ascii=False, indent=1)
    print(f"저장: {RES}   총 {time.time()-t0:.0f}초")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
