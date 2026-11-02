"""42일선 최고값이 정말 옮겨졌나 — 두 측정의 값 목록을 맞춰 놓고 다시 본다.

왜 따로 보나. 가드를 끄고 다시 잰 민감도(state/20261102-spot-long-noguard)에서
sma_days 최고값이 42일에서 48일로 옮겨진 것처럼 보였다. 그런데 두 측정의 값 목록이
달랐다 — 옛 6개(21·30·42·60·90·120) 대 새 13개(사이값 34·36·38·40·45·48·52 를
운영자 지적 9550 으로 채웠다). 목록이 넓어지면 최고값이 새로 들어온 값으로 가기
쉬우므로, 그 이동이 가드 탓인지 목록 탓인지 안 갈린다.

그래서 둘을 본다. ⑴ 공통 6값만으로 재면 최고가 어디인가. ⑵ 42일 둘레가 평평한가 —
평평하면 그 안의 어느 값을 고르든 같은 자리이고, 들쭉날쭉하면 특정 값을 고르는 것
자체가 과적합이다.

판정선은 기준선 앵커 폭이다. 두 값의 칼마 차이가 그보다 작으면 앵커를 바꾸면
사라지는 크기라 어느 쪽이 높다고 말하지 않는다.

실행: python3 spot_sma_peak_check.py
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))

OLD = os.path.join(ROOT, 'state', '20261102-spot-long', 'cells.jsonl')
NEW = os.path.join(ROOT, 'state', '20261102-spot-long-noguard')


def load(p):
    if os.path.isdir(p):
        p = os.path.join(p, 'cells.jsonl')
    if not os.path.exists(p):
        return None
    return [json.loads(l) for l in open(p, encoding='utf-8') if l.strip()]


def by_value(rows, axis):
    """축 하나의 값별 지표 평균. 앵커 여러 곳을 평균한다."""
    g = {}
    for r in rows:
        if r['axis'] != axis:
            continue
        g.setdefault(r['value'], []).append(r['metrics'])
    return {v: dict(cal=float(np.mean([m['calmar'] for m in ms])),
                    cagr=float(np.mean([m['cagr'] for m in ms])),
                    mdd=float(np.mean([m['mdd'] for m in ms])),
                    n=len(ms))
            for v, ms in g.items()}


def anchor_span(rows):
    """기준선 한 구성이 앵커 사이에서 벌어지는 칼마 폭. 판정선이다."""
    ms = [r['metrics'] for r in rows if r['axis'] == '_baseline']
    if not ms:
        return float('nan')
    c = [m['calmar'] for m in ms]
    return float(max(c) - min(c))


def main():
    ro, rn = load(OLD), load(NEW)
    if ro is None or rn is None:
        print(f"측정이 없다: {OLD if ro is None else NEW}")
        return 1

    so, sn = by_value(ro, 'sma_days'), by_value(rn, 'sma_days')
    span = anchor_span(rn)
    common = sorted(set(so) & set(sn))

    print(f"옛 값목록 {len(so)}개: {sorted(so)}")
    print(f"새 값목록 {len(sn)}개: {sorted(sn)}")
    print(f"기준선 앵커 폭(칼마) {span:.3f} — 이보다 작은 차이는 잡음으로 본다\n")

    print(f"공통 {len(common)}값만 (앵커 평균)")
    print(f"  {'값':>5}{'옛칼마':>9}{'새칼마':>9}")
    for v in common:
        print(f"  {v:>5}{so[v]['cal']:9.3f}{sn[v]['cal']:9.3f}")

    bo = max(common, key=lambda v: so[v]['cal'])
    bn = max(common, key=lambda v: sn[v]['cal'])
    moved = "옮겨졌다" if bo != bn else "그대로다"
    print(f"\n  공통값 최고: 옛 {bo}일({so[bo]['cal']:.3f}) · "
          f"새 {bn}일({sn[bn]['cal']:.3f}) → {moved}")

    ba = max(sn, key=lambda v: sn[v]['cal'])
    gap = sn[ba]['cal'] - sn[bn]['cal']
    print(f"  새 측정 전체 {len(sn)}값 최고: {ba}일({sn[ba]['cal']:.3f}), "
          f"공통값 최고와 차이 {gap:+.3f}")
    if gap == gap and span == span:
        tag = "앵커 폭보다 작다 — 잡음과 못 가른다" if abs(gap) < span else \
              "앵커 폭보다 크다 — 볼 만하다"
        print(f"    {tag}")

    print(f"\n42일 둘레가 평평한가 (새 측정, 30~60일)")
    print(f"  {'값':>5}{'칼마':>8}{'연복리':>9}{'낙폭':>9}")
    near = [v for v in sorted(sn) if 30 <= v <= 60]
    for v in near:
        print(f"  {v:>5}{sn[v]['cal']:8.3f}{sn[v]['cagr']:9.4f}{sn[v]['mdd']:9.4f}")
    if len(near) >= 3:
        cals = [sn[v]['cal'] for v in near]
        # 이웃끼리의 차이가 앵커 폭을 넘나드는지 본다. 넘으면 평평한 고원이 아니라
        # 톱니이고, 그런 축에서 특정 값을 고르는 것은 그 자체가 과적합이다.
        steps = [abs(cals[i + 1] - cals[i]) for i in range(len(cals) - 1)]
        big = sum(1 for s in steps if s > span)
        print(f"\n  이웃 간 칼마 차이 {len(steps)}개 중 앵커 폭을 넘는 것 {big}개")
        print("    " + ("톱니다 — 특정 값을 고르는 것 자체가 과적합이다"
                        if big > len(steps) / 2 else
                        "대체로 평평하다 — 이 구간 안이면 어느 값이든 같은 자리다"))
        best_cagr = max(near, key=lambda v: sn[v]['cagr'])
        print(f"  연복리 최고는 {best_cagr}일({sn[best_cagr]['cagr']:.4f}) — "
              "칼마만 보면 노출을 줄여 낙폭을 깎은 값이 좋아 보인다")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
