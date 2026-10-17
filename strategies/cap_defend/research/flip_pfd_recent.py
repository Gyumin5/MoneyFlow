"""최근 구간만 떼어 본다 — 고치기 전과 후가 6월 이후에 얼마나 갈렸나.

저장된 자산곡선을 2026-06-01 부터 잘라 다시 잰다. 백테스트를 새로 돌리지 않는다.
곡선의 마지막 날이 측정 구간의 끝이고, 화면에 그대로 적는다.

구간이 넉 달이라 표본이 작다. 5.8년 판정을 뒤집는 근거로 쓰지 않는다 — 고친 쪽이
최근에 어떻게 움직였는지 보는 눈금이다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 flip_pfd_recent.py
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
SINCE = "2026-06-01"

ARMS = {
    "A 둘 다 넣음": ("sp__현행", "fu"),
    "B 지금 라이브": ("sp__라이브현행", "fu__fut_pfd0"),
    "C 선물만 넣음": ("sp__라이브현행", "fu"),
    "D 현물만 넣음": ("sp__현행", "fu__fut_pfd0"),
}


def load(prefix: str, a: int) -> pd.Series:
    return pd.read_csv(os.path.join(EQDIR, f"{prefix}__a{a}.csv"),
                       index_col=0, parse_dates=True)["eq"]


def seg(eq: pd.Series) -> pd.Series:
    s = eq.loc[SINCE:].dropna()
    return s / s.iloc[0]


def main() -> None:
    comp, sleeve = {k: [] for k in ARMS}, {k: [] for k in ARMS}
    end = None
    for a in R.ANCHORS:
        st = pd.read_csv(os.path.join(STOCK_CACHE, f"st__a{a}.csv"),
                         index_col=0, parse_dates=True)["eq"]
        for name, (sk, fk) in ARMS.items():
            c = seg(R.build_alloc(st, load(sk, a), load(fk, a)))
            end = c.index[-1]
            comp[name].append(dict(ret=(c.iloc[-1] - 1) * 100,
                                   mdd=float((c / c.cummax() - 1).min()) * 100))
            sp = seg(load(sk, a))
            sleeve[name].append((sp.iloc[-1] - 1) * 100)

    print(f"=== {SINCE} ~ {end.date()} · 합성 60/25/15 · 앵커 7곳 평균 ===")
    print(f"{'조합':<16}{'구간수익':>9}{'구간낙폭':>9}{'현물슬리브':>11}")
    for name in ARMS:
        print(f"{name:<16}{float(np.mean([x['ret'] for x in comp[name]])):>9.2f}"
              f"{float(np.mean([x['mdd'] for x in comp[name]])):>9.2f}"
              f"{float(np.mean(sleeve[name])):>11.2f}")
    b = float(np.mean([x["ret"] for x in comp["B 지금 라이브"]]))
    for name in ("A 둘 다 넣음", "C 선물만 넣음", "D 현물만 넣음"):
        d = float(np.mean([x["ret"] for x in comp[name]])) - b
        print(f"  {name} 은 지금 라이브보다 {d:+.2f}%포인트")

    with open(os.path.join(OUT, "recent_result.json"), "w") as f:
        json.dump({"since": SINCE, "end": str(end.date()),
                   "comp": comp, "spot_sleeve": sleeve}, f,
                  ensure_ascii=False, indent=1)
    print(f"\n저장: {OUT}/recent_result.json")


if __name__ == "__main__":
    main()
