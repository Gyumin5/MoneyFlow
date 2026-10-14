"""주식 단독 곡선의 체크포인트 판본 정합 검사 (2026-10-15).

코인 쪽 검사(check_eq_checkpoint_version.py)에서 선물 곡선이 두 판본으로 갈려 있는
것이 드러났다. 주식 단독 곡선(state/20260925-stocksweep/eq/st__*.csv)도 앵커 0~6 은
2026-09-25, 7~10 은 2026-10-15 에 계산됐다. 주식 엔진은 그 사이 바뀐 커밋이 없지만
확인 없이 넘기지 않는다 — 저장본이 어느 판본인지 모른 채 매긴 순위는 아무것도
증명하지 못한다는 것이 코인 쪽에서 배운 것이다.

재는 법: 현행 팔의 앵커 0~6 을 지금 코드로 다시 계산해 저장본과 맞춰 본다.
"""
from __future__ import annotations

import os
import sys

import numpy as np

RES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(RES))
sys.path.insert(0, RES)

os.environ["ANCHORS_N"] = "11"
import bt_stock_param_resweep as S  # noqa: E402

bad = 0
for a in range(7):
    for stress in (False, True):
        key = S._key(S.CUR, a, stress)
        old = S._load_st(key)
        if old is None:
            print(f"  a{a} {'비용5배' if stress else '기본'}: 저장본 없음")
            bad += 1
            continue
        new = S.run_stock({}, a, stress)
        j = old.index.intersection(new.index)
        if len(j) < 100:
            print(f"  a{a}: 겹치는 날짜 {len(j)} — 비교 불가")
            bad += 1
            continue
        d = float(np.max(np.abs(old[j].values / new[j].values - 1.0)))
        same_len = len(old) == len(new)
        mark = "같다" if (d < 1e-9 and same_len) else "다르다"
        tag = "비용5배" if stress else "기본  "
        print(f"  a{a} {tag}: 최대 상대괴리 {d:.3e} · 길이 {len(old)}/{len(new)} → {mark}")
        if d >= 1e-9 or not same_len:
            bad += 1

print(f"\n어긋난 칸 {bad}/14")
