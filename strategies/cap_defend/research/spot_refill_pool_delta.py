"""현물 BT 리필 후보 풀을 그 시점 유니버스로 좁힌 변경이 곡선을 바꾸는지 본다.

aligned3 폴더에 이미 저장된 현물 곡선(바꾸기 전 코드가 만든 것)과, 지금 코드로 다시
돌린 곡선을 같은 인자로 비교한다. 다르면 네 조합 측정을 다시 떠야 한다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 spot_refill_pool_delta.py
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
import flip_pfd_aligned3 as F  # noqa: E402

EQDIR = os.path.join(F.OUT, "eq")


def main() -> int:
    from unified_backtest import load_data
    print("코인 일봉 로드 중...", flush=True)
    R._bars, R._funding = load_data("D")

    worst = 0.0
    checked = 0
    for a in R.ANCHORS[:3]:
        for pfd in (0, 5):
            p = os.path.join(EQDIR, f"sp__pfd{pfd}__a{a}.csv")
            if not os.path.exists(p):
                print(f"없음: {p}")
                continue
            old = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
            new = F.run_spot(pfd, a, False)
            j = old.index.intersection(new.index)
            d = float(np.max(np.abs(new.loc[j].values / old.loc[j].values - 1.0)))
            worst = max(worst, d)
            checked += 1
            print(f"앵커 {a} pfd{pfd}: 길이 {len(old)}/{len(new)} 최대 상대차 {d:.3e}")

    print(f"\n비교 {checked}건, 최대 상대차 {worst:.3e}")
    if worst < 1e-12:
        print("곡선 동일 — 그 시점 유니버스로 좁혀도 역사 구간에서 리필이 안 바뀐다.")
        return 0
    print("곡선이 달라졌다 — 네 조합 측정을 다시 떠야 한다.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
