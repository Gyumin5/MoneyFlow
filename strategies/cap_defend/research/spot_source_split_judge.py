"""창 판정을 자료 출처별로 가른다 — 접합 구간이 결론을 만들고 있나.

토론(run-20261102T022434Z) action_step 4: "상한 0.5의 창 판정을 자료 출처별로
나눠 본다. 접합 구간을 포함하는 창과 2020-10 이후 자료만의 창을 따로 집계해,
우세가 접합부에서 나오는지 확인한다."

긴 구간 자료는 둘을 이어 붙인 것이다 — 2020-10 이전은 bitfinex 일봉, 이후는
기존 자료다. 이어 붙인 자리에서만 우세가 나면 그건 전략의 성질이 아니라 자료의
성질이다. 그래서 창을 세 묶음으로 가른다.
  옛 자료만  — 창이 경계보다 완전히 왼쪽 (2020-10 이전)
  걸친 창    — 경계를 가로지르는 창
  새 자료만  — 창이 경계보다 완전히 오른쪽 (2020-10 이후)
새 자료만의 묶음에서도 우세가 남으면 접합 의존이 아니다.

곡선은 다시 안 돈다. `spot_long_windows.py` 가 저장해 둔 curves.npz 를 읽어
창만 다시 가른다 — 같은 곡선, 같은 쌍대 판정, 묶음만 다르다.

경계 인덱스는 실측으로 구했다. 곡선 길이 3549 는 2017-02-13~2026-11-01 이고
2020-10-01 은 1326번째다(BTC 일봉 축 기준).

실행: python3 spot_source_split_judge.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

from spot_long_windows import (  # noqa: E402
    OUT, PHASES, WINDOWS, score_pair, win_metrics,
)

# 2020-10-01 의 곡선 인덱스. 곡선 길이가 3549 가 아니면 비율로 다시 잡는다.
SPLIT_LEN = 3549
SPLIT_IDX = 1326


def split_pairs(curves, name, ph, split):
    """한 앵커의 창을 세 묶음으로 가른다 — 옛 자료만 / 걸친 창 / 새 자료만."""
    if (name, ph) not in curves or ('BASE', ph) not in curves:
        return {}
    a, b = curves[(name, ph)], curves[('BASE', ph)]
    L = min(len(a), len(b))
    out = {'old': [], 'cross': [], 'new': []}
    for wlen, step in WINDOWS:
        if L < wlen:
            continue
        for s in range(0, L - wlen + 1, step):
            e = s + wlen
            if e <= split:
                tag = 'old'
            elif s >= split:
                tag = 'new'
            else:
                tag = 'cross'
            ma = win_metrics(a[s:e])
            mb = win_metrics(b[s:e])
            if ma and mb:
                out[tag].append((wlen, ma, mb))
    return out


def main():
    p = os.path.join(OUT, 'curves.npz')
    if not os.path.exists(p):
        print(f"곡선 캐시가 없다: {p}")
        print("먼저 spot_long_windows.py 를 돌린다.")
        return 1
    z = np.load(p, allow_pickle=True)
    curves, arms = {}, []
    for k in z.keys():
        n, ph = k.rsplit('|', 1)
        curves[(n, int(ph))] = z[k]
        if n != 'BASE' and n not in arms:
            arms.append(n)

    L = len(next(iter(curves.values())))
    split = SPLIT_IDX if L == SPLIT_LEN else int(round(L * SPLIT_IDX / SPLIT_LEN))
    print(f"곡선 길이 {L}, 경계 인덱스 {split} (2020-10-01)")
    if L != SPLIT_LEN:
        print(f"  주의: 기준 길이 {SPLIT_LEN} 과 달라 비율로 환산했다.")

    print("\n자료 출처별 쌍대 판정 (앵커 7곳 합산)")
    print("  순위합은 1.0 이 전승, 1.5 가 동률, 2.0 이 전패다. n 은 창 수.")
    print(f"  {'구성':<26}{'옛자료만':>20}{'걸친창':>20}{'새자료만':>20}")
    print(f"  {'':22}{'순위/승률/n':>20}{'순위/승률/n':>20}{'순위/승률/n':>20}")

    rows = []
    for name in arms:
        buckets = {'old': [], 'cross': [], 'new': []}
        for ph in PHASES:
            d = split_pairs(curves, name, ph, split)
            for k in buckets:
                buckets[k].extend(d.get(k, []))
        cells, sc = [], {}
        for k in ('old', 'cross', 'new'):
            s = score_pair(buckets[k])
            sc[k] = s
            cells.append(f"{s['rank']:.3f}/{s['winpct']:4.1f}%/{s['n']:4d}"
                         if s['n'] else "        -")
        rows.append((name, sc))
        print(f"  {name:<26}" + "".join(f"{c:>20}" for c in cells))

    print("\n읽는 법. 새 자료만 묶음에서 순위합이 1.5 보다 충분히 낮으면 그 우세는")
    print("  접합 자료에 기대지 않는다. 옛 자료만에서만 낮고 새 자료만에서 1.5 근처면")
    print("  결론을 만든 것은 전략이 아니라 이어 붙인 자리다.")
    print("  새 자료만 묶음은 2020-10 이후라 창 수가 적다 — 수가 적으면 그만큼 약한 근거다.")

    print("\n묶음 간 순위합 차이 (새자료만 - 옛자료만). 양수면 새 자료에서 더 나쁘다.")
    for name, sc in rows:
        if sc['old']['n'] and sc['new']['n']:
            d = sc['new']['rank'] - sc['old']['rank']
            print(f"  {name:<26}{d:+.3f}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
