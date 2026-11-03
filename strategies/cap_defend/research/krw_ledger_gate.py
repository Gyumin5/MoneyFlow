"""사전등록 통과 기준 다섯을 그대로 채점한다 (alloc_grid_run.py docstring 규약).

  ① 일대일 평균순위 1.40 이하
  ② 7앵커 중 6개 이상에서 1.5 미만
  ③ 앵커마다 칼마가 현행 이상이고 낙폭 악화가 1pp 이내
  ④ 평균 낙폭이 0.5pp 개선 또는 이탈 95분위가 2pp 개선
  ⑤ 송금이 현행의 1.5배 이하이면서 연 12회 이하

비용 기본과 5배 두 벌에서 모두 통과해야 후보로 센다.
"""
from __future__ import annotations

import json
import os

import numpy as np

HERE = "/home/gmoh/mon/251229/state/20261103-krw-ledger"
STAGES = ("달러", "+환율", "+통화별현금", "+김프")
CUR = "25/20"


def judge(d, tag, stage, c):
    keys = [f"a{a}_{tag}" for a in range(7)]
    rk = np.array([d[k][stage]["rows"][c]["rank"] for k in keys])
    cal = np.array([d[k][stage]["rows"][c]["Cal"] for k in keys])
    mdd = np.array([d[k][stage]["rows"][c]["MDD"] for k in keys])
    py = np.array([d[k][stage]["rows"][c]["per_yr"] for k in keys])
    h95 = np.array([d[k][stage]["rows"][c]["ht95"] for k in keys])
    bcal = np.array([d[k][stage]["rows"][CUR]["Cal"] for k in keys])
    bmdd = np.array([d[k][stage]["rows"][CUR]["MDD"] for k in keys])
    bpy = np.array([d[k][stage]["rows"][CUR]["per_yr"] for k in keys])
    bh95 = np.array([d[k][stage]["rows"][CUR]["ht95"] for k in keys])

    g1 = float(rk.mean()) <= 1.40
    g2 = int((rk < 1.5).sum()) >= 6
    g3 = bool(np.all(cal >= bcal) and np.all((mdd - bmdd) >= -0.01))
    g4 = bool((bmdd.mean() - mdd.mean()) >= 0.005
              or (bh95.mean() - h95.mean()) >= 0.02)
    g5 = bool(py.mean() <= bpy.mean() * 1.5 and py.mean() <= 12.0)
    return dict(rank=float(rk.mean()), win=int((rk < 1.5).sum()),
                g=(g1, g2, g3, g4, g5), ok=all((g1, g2, g3, g4, g5)),
                cal=float(cal.mean()), mdd=float(mdd.mean()),
                py=float(py.mean()), h95=float(h95.mean()))


def main() -> int:
    d = json.load(open(os.path.join(HERE, "score.json")))
    combos = sorted(d["a0_x1"]["달러"]["rows"],
                    key=lambda k: (int(k.split("/")[0]), int(k.split("/")[1])))

    for stage in STAGES:
        print(f"\n{'=' * 76}\n[{stage}] 사전등록 다섯 관문 — 두 벌 모두 통과해야 후보\n{'=' * 76}")
        print("문턱     비용기본 ①②③④⑤ 평균순위  승  │ 비용5배 ①②③④⑤ 평균순위  승  │ 판정")
        print("-" * 76)
        passed = []
        for c in combos:
            if c == CUR:
                print(f"{c:<8} {'— 현행(기준선)':<58}")
                continue
            a = judge(d, "x1", stage, c)
            b = judge(d, "x5", stage, c)
            fa = "".join("O" if x else "X" for x in a["g"])
            fb = "".join("O" if x else "X" for x in b["g"])
            both = a["ok"] and b["ok"]
            if both:
                passed.append(c)
            print(f"{c:<8}      {fa}  {a['rank']:.3f} {a['win']}/7 │"
                  f"      {fb}  {b['rank']:.3f} {b['win']}/7 │ "
                  f"{'통과' if both else ''}")
        print("-" * 76)
        print(f"  두 벌 통과: {', '.join(passed) if passed else '없음'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
