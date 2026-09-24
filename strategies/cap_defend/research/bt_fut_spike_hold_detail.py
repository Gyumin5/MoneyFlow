"""스윕 결과의 개선이 한 사건에서 온 것인지 본다 (2026-09-24).

bt_fut_spike_hold.py 가 저장한 자산곡선만 읽는다. 백테스트를 다시 돌리지 않는다.

묻는 것 셋:
  1. 슬리브 최대낙폭이 줄어든 자리가 언제인가 — 그 자리가 한 번뿐이면 개선도 한 번뿐이다.
  2. 윈도우 단위 칼마 차이가 어느 쪽으로 몰려 있나 — 몇 개 윈도우가 통째로 끌고 가는지.
  3. 연도별로 갈라 보면 어느 해가 좋아졌나.
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from bt_fut_spike_hold import (ANCHORS, CUR, WIN_SIZES, STRIDES, _key, _load,  # noqa: E402
                               build_alloc, metrics)

ARMS = [CUR, "E 감속 0.10", "E 감속 0.15", "E 감속 0.20", "D 보류 0.20"]


def dd(eq: pd.Series) -> pd.Series:
    return eq / eq.cummax() - 1


def main() -> None:
    print("[1] 선물 슬리브 낙폭 바닥 — 앵커 0, 기본 비용")
    print(f"  {'팔':<14s}{'최대낙폭':>9s}{'바닥일':>13s}{'2등 낙폭':>10s}{'2등 바닥':>13s}")
    for name in ARMS:
        eq = _load("fu", _key(name, 0, False))
        if eq is None:
            continue
        d = dd(eq)
        trough = d.idxmin()
        # 바닥 앞뒤 120일을 빼고 다시 최저를 찾으면 두 번째로 깊은 구간이 나온다
        mask = (d.index < trough - pd.Timedelta(days=120)) | (d.index > trough + pd.Timedelta(days=120))
        d2 = d[mask]
        print(f"  {name:<14s}{d.min():>8.1%}{str(trough.date()):>13s}"
              f"{d2.min():>10.1%}{str(d2.idxmin().date()):>13s}")

    print("\n[2] 연도별 선물 슬리브 수익률 — 앵커 0, 기본 비용")
    eqs = {n: _load("fu", _key(n, 0, False)) for n in ARMS}
    eqs = {k: v for k, v in eqs.items() if v is not None}
    years = sorted({d.year for d in list(eqs.values())[0].index})
    print("  연도  " + "".join(f"{n:>14s}" for n in eqs))
    for y in years:
        row = f"  {y}  "
        for _n, eq in eqs.items():
            s = eq[eq.index.year == y]
            row += f"{(s.iloc[-1]/s.iloc[0]-1):>13.1%} " if len(s) > 1 else f"{'-':>14s}"
        print(row)

    print("\n[3] 합성 60/25/15 윈도우별 칼마 차이 (팔 - 현행) · 기본 비용 · 앵커 7곳")
    print(f"  {'팔':<14s}{'이긴 윈도우':>12s}{'차이 중앙':>11s}{'차이 평균':>11s}"
          f"{'상위10% 기여':>13s}")
    allocs_by_anchor = {}
    for a in ANCHORS:
        st = _load("st", _key("주식", a, False))
        sp = _load("sp", _key(CUR, a, False))
        if st is None or sp is None:
            continue
        d = {}
        for name in ARMS:
            fu = _load("fu", _key(name, a, False))
            if fu is not None:
                d[name] = build_alloc(st, sp, fu)
        allocs_by_anchor[a] = d

    def wcal(eq, d0, d1):
        seg = eq.loc[d0:d1].dropna()
        if len(seg) < 30:
            return None
        yrs = (seg.index[-1] - seg.index[0]).days / 365.25
        if yrs <= 0:
            return None
        cagr = (seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
        m = float((seg / seg.cummax() - 1).min())
        return cagr / abs(m) if m < 0 else 0.0

    for name in ARMS:
        if name == CUR:
            continue
        diffs = []
        for _a, d in allocs_by_anchor.items():
            if name not in d or CUR not in d:
                continue
            common = d[name].index.intersection(d[CUR].index)
            common = sorted(common)
            for size in WIN_SIZES:
                for stride in STRIDES:
                    for i in range(0, len(common) - size, stride):
                        d0, d1 = common[i], common[i + size - 1]
                        ca, cb = wcal(d[name], d0, d1), wcal(d[CUR], d0, d1)
                        if ca is None or cb is None:
                            continue
                        diffs.append(ca - cb)
        v = np.array(diffs)
        if not len(v):
            continue
        top = np.sort(v)[-max(1, len(v) // 10):]
        share = top.sum() / v.sum() if v.sum() != 0 else float('nan')
        print(f"  {name:<14s}{(v>0).mean():>11.1%}{np.median(v):>11.3f}"
              f"{v.mean():>11.3f}{share:>13.1%}")

    print("\n[4] 합성 전체 구간 지표 — 앵커 7곳 평균, 기본 비용")
    print(f"  {'팔':<14s}{'칼마':>7s}{'연수익':>9s}{'낙폭':>9s}{'샤프':>7s}")
    for name in ARMS:
        ms = [metrics(d[name]) for d in allocs_by_anchor.values() if name in d]
        if not ms:
            continue
        print(f"  {name:<14s}{np.mean([m['Cal'] for m in ms]):>7.2f}"
              f"{np.mean([m['CAGR'] for m in ms]):>+8.1f}%"
              f"{np.mean([m['MDD'] for m in ms]):>+8.1f}%"
              f"{np.mean([m['Sharpe'] for m in ms]):>7.2f}")


if __name__ == '__main__':
    main()
