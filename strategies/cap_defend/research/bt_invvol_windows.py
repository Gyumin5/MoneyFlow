"""이득이 2025년 3월 한 사건에서만 나오나 (2026-09-25).

낙폭 구간별 비교에서 모양이 갈렸다. 2025-03 구간은 역수 20 이 0.83%p 얕고, 2026-08 은
1.45%p 얕은데, 2024-09 는 3.90%p 더 깊다. 최대낙폭 통계는 가장 깊은 한 구간만 기록하므로
다른 구간이 나빠져도 안 보인다. 초과수익 합은 5.9년에 걸쳐 사실상 0 이다.

그래서 결정층(윈도우 순위합)이 무엇을 보고 이겼는지를 창 단위로 쪼갠다. 2025-03 최저점을
품은 창과 안 품은 창을 갈라, 각 무리에서 칼마가 얼마나 자주 이겼는지 센다. 한쪽에서만
이기면 그 이득은 사건 하나다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_invvol_windows.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from bt_fut_spike_hold import ANCHORS, build_alloc  # noqa: E402
from bt_invvol_detail import _load, ARM, CURK  # noqa: E402

WIN_SIZES = [504, 756, 1008]
STRIDES = [63, 126, 252]
EVENT = pd.Timestamp("2025-03-10")


def _cal(eq: pd.Series) -> float:
    if len(eq) < 30 or eq.iloc[0] <= 0:
        return float('nan')
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0:
        return float('nan')
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    return float('nan') if mdd >= -1e-9 else cagr / abs(mdd)


def main() -> None:
    wi = {True: [], False: []}
    for a in ANCHORS:
        st = _load("st", f"주식__a{a}")
        fu = _load("fu", f"선물__a{a}")
        cur = _load("sp", f"{CURK}__a{a}")
        arm = _load("sp", f"{ARM}__a{a}")
        if any(x is None for x in (st, fu, cur, arm)):
            continue
        base, test = build_alloc(st, cur, fu), build_alloc(st, arm, fu)
        i = base.index.intersection(test.index)
        base, test = base.loc[i], test.loc[i]
        for w in WIN_SIZES:
            for s in STRIDES:
                for b in range(0, len(i) - w, s):
                    seg = i[b:b + w]
                    cb, ct = _cal(base.loc[seg]), _cal(test.loc[seg])
                    if np.isnan(cb) or np.isnan(ct):
                        continue
                    has = bool(seg[0] <= EVENT <= seg[-1])
                    wi[has].append(ct - cb)

    print(f"기준 사건: {EVENT.date()} (현행 최대낙폭 최저점)")
    for has in (True, False):
        d = np.array(wi[has], dtype=float)
        if not len(d):
            continue
        lbl = "그 날을 품은 창" if has else "그 날이 없는 창"
        print(f"\n[{lbl}]  창 {len(d)}개")
        print(f"  역수20 이 이긴 창 {int((d > 0).sum())}개 ({(d > 0).mean():.0%})")
        print(f"  칼마 차 평균 {d.mean():+.3f}  중앙 {np.median(d):+.3f}  "
              f"최소 {d.min():+.3f}  최대 {d.max():+.3f}")


if __name__ == '__main__':
    main()
