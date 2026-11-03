"""채점 결과를 사전등록 규약대로 읽는다 — 7앵커 평균, 비용 두 벌 따로."""
from __future__ import annotations

import json
import os

import numpy as np

HERE = "/home/gmoh/mon/251229/state/20261103-krw-ledger"
STAGES = ("달러", "+환율", "+통화별현금", "+김프")
CUR = "25/20"


def main() -> int:
    d = json.load(open(os.path.join(HERE, "score.json")))
    combos = sorted(d["a0_x1"]["달러"]["rows"],
                    key=lambda k: (int(k.split("/")[0]), int(k.split("/")[1])))

    for tag, label in (("x1", "비용 기본"), ("x5", "비용 5배")):
        keys = [f"a{a}_{tag}" for a in range(7)]
        print(f"\n{'=' * 78}\n[{label}] 7앵커 평균 — 현행 {CUR} 과 창마다 일대일"
              f" (1 에 가까울수록 현행보다 낫다)\n{'=' * 78}")
        hdr = "문턱".ljust(8) + "".join(s.ljust(17) for s in STAGES)
        print(hdr)
        print("-" * 78)
        best = {}
        for st in STAGES:
            r = np.array([[d[k][st]["rows"][c]["rank"] for c in combos] for k in keys])
            best[st] = r.mean(axis=0)
        for i, c in enumerate(combos):
            mark = " ←현행" if c == CUR else ""
            cells = ""
            for st in STAGES:
                v = best[st][i]
                win = sum(1 for k in keys if d[k][st]["rows"][c]["rank"] < 1.5)
                cells += f"{v:.3f} ({win}/7)".ljust(17)
            print(c.ljust(8) + cells + mark)

        print("-" * 78)
        for st in STAGES:
            order = np.argsort(best[st])
            top = combos[int(order[0])]
            print(f"  {st:<12} 1위 {top:<7} 평균순위 {best[st][order[0]]:.3f}"
                  f"   현행 {CUR} 평균순위 {best[st][combos.index(CUR)]:.3f}"
                  f"   {'같다' if top == CUR else '다르다'}")

        print(f"\n  현행 {CUR} 의 7앵커 평균 성과")
        for st in STAGES:
            cal = np.mean([d[k][st]["rows"][CUR]["Cal"] for k in keys])
            mdd = np.mean([d[k][st]["rows"][CUR]["MDD"] for k in keys])
            cg = np.mean([d[k][st]["rows"][CUR]["CAGR"] for k in keys])
            sh = np.mean([d[k][st]["rows"][CUR]["Sharpe"] for k in keys])
            py = np.mean([d[k][st]["rows"][CUR]["per_yr"] for k in keys])
            print(f"    {st:<12} 칼마 {cal:5.2f}  연복리 {cg * 100:6.1f}%"
                  f"  낙폭 {mdd * 100:6.1f}%  샤프 {sh:4.2f}  송금 {py:4.2f}회per년")

    m = d["a0_x1"]["_meta"]
    print(f"\n자재 — 창 {m['창']}개 · {m['연수']}년 · 영업일 {m['영업일']}"
          f" · 테더김프 실측 {m['테더김프_실측일']}일 · 현물 현금비중 평균 {m['현물현금_평균']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
