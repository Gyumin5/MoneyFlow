"""전환 후 5봉 재평가 — 네 조합을 한 표에 놓는다.

지금까지는 한쪽 슬리브만 바꿔서 쟀다. 실제로 올릴 모양은 둘 다 바꾸는 것이라
그 칸이 비어 있었다. 새로 돌리는 백테스트는 없다 — 이미 저장된 자산곡선 넷을
조합만 바꿔 합성한다.

  A 둘 다 있음      = 채택 백테스트 그대로
  B 둘 다 없음      = 지금 라이브 (실제 상태)
  C 현물만 없음
  D 선물만 없음

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 flip_pfd_compose.py
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402

OUT = "/home/gmoh/mon/251229/state/20261018-fliplock"
EQDIR = os.path.join(OUT, "eq")
STOCK_CACHE = "/home/gmoh/mon/251229/state/20260922-resweep/eq"

# 이름 → (현물 곡선 키, 선물 곡선 키)
ARMS = {
    "A 둘 다 있음": ("sp__현행", "fu"),
    "B 둘 다 없음(지금 라이브)": ("sp__라이브현행", "fu__fut_pfd0"),
    "C 현물만 없음": ("sp__라이브현행", "fu"),
    "D 선물만 없음": ("sp__현행", "fu__fut_pfd0"),
}


def load(prefix: str, a: int, stress: bool) -> pd.Series:
    p = os.path.join(EQDIR, f"{prefix}__a{a}{'__s5' if stress else ''}.csv")
    return pd.read_csv(p, index_col=0, parse_dates=True)["eq"]


def main() -> None:
    res = {}
    for stress in (False, True):
        rank = {k: [] for k in ARMS}
        met = {k: [] for k in ARMS}
        for a in R.ANCHORS:
            st = pd.read_csv(os.path.join(STOCK_CACHE, f"st__a{a}.csv"),
                             index_col=0, parse_dates=True)["eq"]
            comps = {}
            for name, (sk, fk) in ARMS.items():
                comps[name] = R.build_alloc(st, load(sk, a, stress), load(fk, a, stress))
                met[name].append(R.metrics(comps[name]))
            sums, _, nw = R.window_rs(comps)
            for name in ARMS:
                rank[name].append(sums[name] / max(nw, 1))

        tag = "비용 5배" if stress else "기본 비용"
        print(f"\n=== 합성 60/25/15 · {tag} · 앵커 {len(R.ANCHORS)}곳 · 창 순위합 ===")
        print(f"{'조합':<26}{'순위평균':>9}{'칼마':>7}{'샤프':>7}{'연복리':>8}{'최대낙폭':>9}")
        order = sorted(ARMS, key=lambda k: float(np.mean(rank[k])))
        for name in order:
            print(f"{name:<26}{float(np.mean(rank[name])):>9.3f}"
                  f"{float(np.mean([m['Cal'] for m in met[name]])):>7.2f}"
                  f"{float(np.mean([m['Sharpe'] for m in met[name]])):>7.2f}"
                  f"{float(np.mean([m['CAGR'] for m in met[name]])):>8.2f}"
                  f"{float(np.mean([m['MDD'] for m in met[name]])):>9.2f}")
        base = "B 둘 다 없음(지금 라이브)"
        print(f"  지금 라이브(B) 대비 A 가 이긴 앵커: "
              f"{sum(1 for x, y in zip(rank['A 둘 다 있음'], rank[base]) if x < y)}"
              f"/{len(R.ANCHORS)}")
        res[tag] = dict(rank=rank, metrics=met)

    with open(os.path.join(OUT, "compose_result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {OUT}/compose_result.json")


if __name__ == "__main__":
    main()
