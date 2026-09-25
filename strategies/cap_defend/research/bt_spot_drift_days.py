"""느슨한 문턱의 차이가 몇 날에서 나오나 (2026-09-25).

문턱 0.15·0.20 이 두 비용 조건 모두에서 현행 0.10 을 이겼고, 평가창을 사건 유무로 갈라도
양쪽에서 이겼다. 그런데 기계적 차이가 아주 작다 — 5.9년 발화가 143회에서 137회로 여섯 번
줄고 현물 슬리브 최대낙폭은 소수점까지 같다.

그래서 급등 익일 규칙을 기각할 때 쓴 자로 다시 잰다. 두 곡선이 실제로 갈라진 날을 세고,
그 날들이 해마다 어떻게 퍼져 있는지, 기여가 몇 날에 몰려 있는지 본다. 몇 날에 몰려 있으면
순위 우세는 그 날들이 평가창에 겹쳐 들어간 결과다.

현물 슬리브 단독으로 본다(합성은 주식·선물이 희석해서 날을 못 가린다).

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_spot_drift_days.py
"""
from __future__ import annotations

import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402

R.OUT = "/home/gmoh/mon/251229/state/20260925-spotdrift-buf"
R.EQDIR = os.path.join(R.OUT, "eq")

ARMS = ["현물 0.15", "현물 0.20", "현물 0.30", "현물 끔"]
EPS = 1e-9


def main() -> None:
    for arm in ARMS:
        print(f"=== {arm} 대 현행 · 현물 슬리브 단독 ===")
        rows = []
        for a in R.ANCHORS:
            b = R._load("sp", R._key(R.CUR, a, False))
            t = R._load("sp", R._key(arm, a, False))
            if b is None or t is None:
                continue
            i = b.index.intersection(t.index)
            rb = b.loc[i].pct_change().fillna(0.0)
            rt = t.loc[i].pct_change().fillna(0.0)
            d = rt - rb
            rows.append((a, i, d, rb, rt))

        if not rows:
            print("  자료 없음\n")
            continue

        a0, i0, d0, rb0, rt0 = rows[0]
        diff = d0[d0.abs() > EPS]
        print(f"  앵커 0 — 전체 {len(i0):,}일 중 갈라진 날 {len(diff):,}일 "
              f"({len(diff)/len(i0):.1%})")
        yr = diff.groupby(diff.index.year).size()
        print("  연도별 갈라진 날: " + "  ".join(f"{y} {int(n)}" for y, n in yr.items()))

        exc = float(np.log1p(rt0).sum() - np.log1p(rb0).sum())
        top = diff.reindex(diff.abs().sort_values(ascending=False).index)[:5]
        print(f"  5.9년 초과 로그수익 {exc:+.4f}")
        print("  기여 상위 5일: " + "  ".join(
            f"{pd.Timestamp(x).date()} {v:+.2%}" for x, v in top.items()))
        cum = float(top.sum())
        print(f"  상위 5일 합 {cum:+.2%} (초과 로그수익 대비 "
              f"{cum/exc*100 if exc else float('nan'):.0f}%)")

        excs = []
        for a, i, d, rb, rt in rows:
            excs.append(float(np.log1p(rt).sum() - np.log1p(rb).sum()))
        print(f"  앵커별 초과 로그수익: {' '.join(f'{v:+.4f}' for v in excs)}")
        print(f"    앵커 7곳 전부 양수: {all(v > 0 for v in excs)}\n")


if __name__ == '__main__':
    main()
