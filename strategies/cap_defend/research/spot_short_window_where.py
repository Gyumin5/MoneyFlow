"""짧은 창에서 기준선이 더 자주 이기는 자리가 어디인가.

창 판정에서 드리프트 후보 셋(끔·0.40·0.50)은 창이 길수록 이기고 짧을수록 진다 —
504봉(약 2년) 승률이 38~41% 인데 1008봉(약 4년)에서는 62~71% 다. 이 비대칭이
채택을 막을 근거인지 보려면 "어느 2년에서 지나" 를 알아야 한다.

두 갈래로 갈린다.
  ⑴ 지는 2년이 한 시기에 몰려 있다 — 그 시기의 성질(예: 횡보장에서 되맞춤이 유리)이
     원인이고, 그 시기가 다시 오면 후보가 불리하다. 채택 판단에 들어가야 한다.
  ⑵ 지는 2년이 전 구간에 흩어져 있다 — 짧은 창은 표본이 짧아 흔들리는 것뿐이고,
     긴 창에서 뒤집히는 것은 그 흔들림이 상쇄되기 때문이다. 채택을 막지 않는다.

재는 법. 504봉 창마다 시작 날짜와 승패를 적고, 시작 연도별로 승률을 낸다.
승패는 창 판정과 같은 규칙이다 — 네 지표(칼마·샤프·연복리·낙폭) 중 엄격 우세 수가
엄격 열세 수보다 많으면 승.

같이 본다. 그 창에서 기준선이 전액 현금으로 보낸 비율. 카나리가 꺼져 둘 다 현금이면
무승부가 되므로, 승패가 갈리는 창은 실제로 들어가 있던 창이다.

곡선은 다시 안 돈다 — spot_long_windows.py 의 curves.npz 를 읽는다.

실행: python3 spot_short_window_where.py
"""
from __future__ import annotations

import os
import sys
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

from spot_long_windows import KEYS, OUT, PHASES, EPS, win_metrics  # noqa: E402

START, END = '2017-01-01', '2026-11-01'
WLEN, STEP = 504, 63
ARMS = ['n3-c33-d0-m127', 'n3-c33-d40-m127', 'n3-c33-d50-m127']


def load_dates(n):
    from long_history import load_extended
    bars, _fund = load_extended(verbose=False)
    idx = bars['BTC'].index
    idx = idx[(idx >= pd.Timestamp(START)) & (idx <= pd.Timestamp(END))]
    return idx[-n:]


def verdict(ma, mb):
    better = sum(1 for k in KEYS if ma[k] > mb[k] + EPS)
    worse = sum(1 for k in KEYS if ma[k] < mb[k] - EPS)
    if better > worse:
        return 1
    if better < worse:
        return -1
    return 0


def main():
    p = os.path.join(OUT, 'curves.npz')
    if not os.path.exists(p):
        print(f"곡선 캐시가 없다: {p}")
        return 1
    z = np.load(p, allow_pickle=True)
    curves = {}
    for k in z.files:
        n, ph = k.rsplit('|', 1)
        curves[(n, int(ph))] = z[k]

    L = len(curves[('BASE', 0)])
    dates = load_dates(L)
    print(f"곡선 {L}봉 {dates[0].date()}~{dates[-1].date()}, 창 {WLEN}봉 이동 {STEP}\n")

    for name in ARMS:
        by_year = defaultdict(lambda: [0, 0, 0])   # 승 무 패
        for ph in PHASES:
            if (name, ph) not in curves:
                continue
            a, b = curves[(name, ph)], curves[('BASE', ph)]
            n = min(len(a), len(b))
            for s in range(0, n - WLEN + 1, STEP):
                ma, mb = win_metrics(a[s:s + WLEN]), win_metrics(b[s:s + WLEN])
                if not (ma and mb):
                    continue
                y = dates[s].year
                v = verdict(ma, mb)
                by_year[y][0 if v > 0 else (1 if v == 0 else 2)] += 1
        print(f"{name} — 창 시작 연도별 (승/무/패, 승률은 무를 뺀 값)")
        tot = [0, 0, 0]
        for y in sorted(by_year):
            w, d, l = by_year[y]
            for i, x in enumerate((w, d, l)):
                tot[i] += x
            dec = w + l
            pct = f"{100.0*w/dec:5.1f}%" if dec else "    -"
            print(f"  {y}  {w:3d}/{d:3d}/{l:3d}  {pct}")
        dec = tot[0] + tot[2]
        pct = f"{100.0*tot[0]/dec:5.1f}%" if dec else "    -"
        print(f"  합계 {tot[0]:3d}/{tot[1]:3d}/{tot[2]:3d}  {pct}\n")

    print("읽는 법. 지는 해가 한두 해에 몰려 있으면 그 시기의 성질이 원인이라")
    print("  채택 판단에 들어가야 한다. 전 구간에 흩어져 있으면 짧은 창의 흔들림이다.")
    print("  무는 네 지표의 우세·열세 수가 같은 창이고, 둘 다 현금인 구간이 여기 모인다.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
