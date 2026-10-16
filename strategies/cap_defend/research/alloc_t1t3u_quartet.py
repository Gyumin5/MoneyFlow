#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""문턱 넷을 짝지어 T1 과 T3U 의 기여를 가른다 — 네 성과 지표 전부.

ai-debate run-20261016T021354Z 가 둘을 지적했다. ⑴ 15/15 대 20/20 보고에 샤프와
연복리수익률이 빠졌다. ⑵ 둘을 같이 조이면 어느 쪽이 무엇을 바꿨는지 모른다.

그래서 20/20 · 15/20 · 20/15 · 15/15 를 열네 칸에서 돌리고 칼마·샤프·연복리수익률·
최대낙폭에 운영 지표(연 완료 송금·이탈95·최대이탈)를 붙여 낸다. 이건 사후 분석이다 —
사전등록이 보호하지 않는다.

쓰기: python3 alloc_t1t3u_quartet.py
"""
from __future__ import annotations

import json
import os

import numpy as np

import alloc_grid_run as RUN
from alloc_t1t3u_power import CELLS, cell_inputs
from alloc_transit_convention import run_one

OUT = RUN.OUT
RES = os.path.join(OUT, "t1t3u_quartet.json")
PAIRS = [(0.20, 0.20), (0.15, 0.20), (0.20, 0.15), (0.15, 0.15)]


def main() -> int:
    acc = {p: [] for p in PAIRS}
    for an, st in CELLS:
        axis, Rm, is_bd, tab, gate, mid = cell_inputs(an, st)
        yrs = (axis[-1] - axis[0]).days / 365.25
        for p in PAIRS:
            pv, inf = run_one(axis, Rm, gate, tab, False, *p, month_id=mid)
            fs = RUN.full_stats(pv[:, None], axis)
            acc[p].append([float(fs['Cal'][0]), float(fs['Sharpe'][0]),
                           float(fs['CAGR'][0]), float(fs['MDD'][0]),
                           inf['dones'] / yrs, inf['ht95'], inf['maxht']])
        print(f"  앵커 {an} {'5배' if st else '실비용'} 끝")

    A = {p: np.asarray(v) for p, v in acc.items()}
    names = ['칼마', '샤프', '연복리', '최대낙폭', '연완료', '이탈95', '최대이탈']
    print("\n── 열네 칸 평균")
    print("  문턱        칼마    샤프   연복리   최대낙폭   연완료  이탈95   최대이탈")
    for p in PAIRS:
        m = A[p].mean(axis=0)
        print(f"  {p[0]*100:.0f}/{p[1]*100:.0f}   {m[0]:7.3f} {m[1]:6.3f} "
              f"{m[2]*100:7.2f}% {m[3]*100:8.2f}% {m[4]:7.3f} {m[5]*100:6.2f}pp {m[6]*100:7.2f}pp")

    base = A[(0.20, 0.20)]
    out = {}
    print("\n── 20/20 대비 차이 (평균, 괄호는 나아진 칸 수)")
    for p in PAIRS[1:]:
        d = A[p] - base
        m = d.mean(axis=0)
        better = [int((d[:, i] > 0).sum()) if i in (0, 1, 2, 3) else int((d[:, i] < 0).sum())
                  for i in range(7)]
        print(f"  {p[0]*100:.0f}/{p[1]*100:.0f}  칼마 {m[0]:+.3f}({better[0]}/14)  "
              f"샤프 {m[1]:+.3f}({better[1]}/14)  연복리 {m[2]*100:+.2f}pp({better[2]}/14)  "
              f"낙폭 {m[3]*100:+.2f}pp({better[3]}/14)")
        print(f"          연완료 {m[4]:+.3f}  이탈95 {m[5]*100:+.3f}pp({better[5]}/14)  "
              f"최대이탈 {m[6]*100:+.3f}pp({better[6]}/14)")
        out[f"{p[0]:.2f}/{p[1]:.2f}"] = dict(zip(names, [float(x) for x in m]),
                                             better=better)
    out['levels'] = {f"{p[0]:.2f}/{p[1]:.2f}": dict(zip(names,
                     [float(x) for x in A[p].mean(axis=0)])) for p in PAIRS}
    json.dump(out, open(RES, 'w'), ensure_ascii=False, indent=1)
    print(f"\n저장: {RES}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
