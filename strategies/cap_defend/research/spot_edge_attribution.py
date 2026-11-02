"""우세가 어느 날에서 나왔나 — 곡선 차이의 시점 분해.

앞선 자료 출처별 판정(`spot_source_split_judge.py`)에서 상한 0.5 가 경계를 걸친 창
126개를 하나도 안 지고 이겼다(승률 100%). 창 전부가 안 지는 것은 그 창들이 공통으로
품은 사건 하나가 결과를 만들었다는 뜻이다. 그 사건이 언제인지를 짚는다.

재는 법. 후보와 기준선의 일별 로그수익 차이 r = log(a_t/a_{t-1}) - log(b_t/b_{t-1})
를 쌓는다. 누적 차이가 계단처럼 오르는 자리가 우세가 생긴 날이다. 창 판정은 그
계단을 품은 창 전부를 후보 우세로 세므로, 계단 하나가 창 수백 개를 지배한다.

보는 것 셋.
  ⑴ 누적 차이 곡선에서 가장 크게 오른 연속 구간(최대 상승 구간)과 그 비중.
  ⑵ 일별 차이 상위 열흘이 전체 누적 차이의 몇 퍼센트인가. 집중도가 높을수록
     그 후보의 우세는 사건 하나짜리다.
  ⑶ 그 상위 열흘을 빼면(= 그날 차이를 0 으로 두면) 누적 차이가 얼마나 남나.
     남는 게 없으면 "전략이 낫다" 가 아니라 "그날 더 실려 있었다" 다.

곡선은 다시 안 돈다 — `spot_long_windows.py` 의 curves.npz 를 읽는다.
날짜 축은 BTC 일봉에서 복원한다(곡선 길이만큼 뒤에서 취한다).

실행: python3 spot_edge_attribution.py
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

from spot_long_windows import OUT, PHASES  # noqa: E402

START, END = '2017-01-01', '2026-11-01'
TOPN = 10


def load_dates(n):
    from long_history import load_extended
    bars, _fund = load_extended(verbose=False)
    idx = bars['BTC'].index
    idx = idx[(idx >= pd.Timestamp(START)) & (idx <= pd.Timestamp(END))]
    return idx[-n:]


def logret(eq):
    v = np.asarray(eq, dtype=float)
    v = np.where(v > 0, v, np.nan)
    return np.diff(np.log(v))


def attribute(a, b, dates):
    """일별 로그수익 차이와 그 집중도."""
    d = logret(a) - logret(b)
    d = np.nan_to_num(d, nan=0.0, posinf=0.0, neginf=0.0)
    cum = np.cumsum(d)
    total = float(cum[-1])

    # 최대 상승 구간 — 누적 차이의 최저점 이후 최고점까지
    lo_i = int(np.argmin(cum))
    hi_i = int(lo_i + np.argmax(cum[lo_i:]))
    rise = float(cum[hi_i] - cum[lo_i])

    # 열흘을 빼는 방식이 둘이고 재는 것이 다르다. 섞으면 안 된다.
    #  abs — 차이의 절댓값이 가장 큰 열흘. "변동이 큰 날" 을 본다.
    #  pos — 후보가 가장 많이 앞선 열흘. "양의 우위가 소수 사건에 몰렸나" 를 본다.
    # 판정에 쓰는 것은 pos 쪽이다. abs 로 재면 그 열흘에 큰 손실일이 섞여 들어가
    # 제외 후 잔여가 오히려 커지고, 그걸 "사건에 안 몰렸다" 로 읽게 된다
    # (2026-11-02 오후 판정에서 실제로 그렇게 읽었다, 토론 run-20261102T062159Z).
    abs_top = np.argsort(-np.abs(d))[:TOPN]
    abs_sum = float(np.sum(d[abs_top]))
    pos_top = np.argsort(-d)[:TOPN]
    pos_sum = float(np.sum(d[pos_top]))

    dd = dates[1:]          # 차이는 하루 밀린다
    return dict(
        total=total, rise=rise,
        rise_from=str(dd[lo_i].date()), rise_to=str(dd[hi_i].date()),
        abs_sum=abs_sum, abs_rest=total - abs_sum,
        pos_sum=pos_sum, pos_rest=total - pos_sum,
        top_days=[(str(dd[i].date()), float(d[i])) for i in pos_top[:5]],
    )


def main():
    p = os.path.join(OUT, 'curves.npz')
    if not os.path.exists(p):
        print(f"곡선 캐시가 없다: {p}")
        return 1
    z = np.load(p, allow_pickle=True)
    curves, arms = {}, []
    for k in z.keys():
        n, ph = k.rsplit('|', 1)
        curves[(n, int(ph))] = z[k]
        if n != 'BASE' and n not in arms:
            arms.append(n)

    L = len(next(iter(curves.values())))
    dates = load_dates(L)
    print(f"곡선 {L}봉 {dates[0].date()}~{dates[-1].date()}\n")

    print("우세의 시점 집중도 (앵커 7곳 평균, 로그수익 차이 누적)")
    print(f"  {'구성':<22}{'누적차':>9}{'최대상승':>10}{'상승비중':>9}"
          f"{'우위상위10':>11}{'우위제외':>10}{'변동제외':>10}")
    for name in arms:
        rs = [attribute(curves[(name, ph)], curves[('BASE', ph)], dates)
              for ph in PHASES if (name, ph) in curves]
        tot = float(np.mean([r['total'] for r in rs]))
        rise = float(np.mean([r['rise'] for r in rs]))
        psum = float(np.mean([r['pos_sum'] for r in rs]))
        prest = float(np.mean([r['pos_rest'] for r in rs]))
        arest = float(np.mean([r['abs_rest'] for r in rs]))
        share = (100.0 * rise / tot) if abs(tot) > 1e-9 else float('nan')
        print(f"  {name:<22}{tot:9.3f}{rise:10.3f}{share:8.0f}%"
              f"{psum:11.3f}{prest:10.3f}{arest:10.3f}")

    print("\n읽는 법. 누적차는 9.8년 로그수익 차이 합이다 — 양수면 후보가 더 벌었다.")
    print("  판정에 쓰는 칸은 우위제외다. 후보가 가장 많이 앞선 열흘을 빼고 남는 값이고,")
    print("  0 이하면 그 우세는 열흘짜리라 전략의 성질이 아니다.")
    print("  변동제외는 절댓값이 큰 열흘을 뺀 값이라 그 열흘에 손실일이 섞인다 — 참고만 한다.")
    print("  상승비중이 100% 를 크게 넘으면 나머지 구간에서는 오히려 지고 있었다는 뜻이다.")

    print("\n앵커 0 에서 후보가 가장 많이 앞선 닷새")
    for name in arms:
        if (name, 0) not in curves:
            continue
        r = attribute(curves[(name, 0)], curves[('BASE', 0)], dates)
        days = "  ".join(f"{d} {v:+.3f}" for d, v in r['top_days'])
        print(f"  {name:<22}{days}")
        print(f"  {'':22}최대상승 {r['rise_from']} ~ {r['rise_to']}")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
