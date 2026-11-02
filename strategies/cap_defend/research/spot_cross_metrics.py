"""교차 격자 축별 표 — 창 판정기가 받아 둔 곡선에서 바로 낸다.

왜 격자를 다시 안 도나. `spot_long_cross.py` 는 108 구성 × 앵커 7 을 돌려
전 구간 네 지표를 적는다. `spot_long_windows.py --all` 은 같은 108 구성 + 기준선을
돌려 전 구간 자산곡선 자체를 curves.npz 에 담는다. 곡선이 있으면 지표는 거기서
나오므로, 같은 측정을 두 번 돌릴 이유가 없다. 2026-11-02 에 둘 다 가드 결함으로
다시 돌려야 했을 때 한 번만 돌려 양쪽을 덮는 길로 골랐다(앞선 측정이 1.6시간).

지표 식은 `spot_long_windows.win_metrics` 를 그대로 쓴다 — 곡선 조각에 쓰던 식을
전 구간에 적용하는 것이다. 교차 격자가 쓰던 `ub.run` 반환값과 소수점 아래가 다를 수
있다(연수를 봉 수/365 로 잡는다). 모든 구성에 같은 식을 쓰므로 축별 비교와 순위는
그대로 유효하고, 절대값을 옛 격자 수치와 나란히 놓지만 않으면 된다.

구성 이름은 n{종목수}-c{상한}-d{드리프트}-m{장기모멘텀} 이고 기준선 BASE 는
n3-c33-d10-m127 과 같은 설정이라 둘을 대조해 식 차이를 눈으로 볼 수 있다.

실행: python3 spot_cross_metrics.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

from spot_long_windows import OUT, PHASES, win_metrics  # noqa: E402


def load():
    p = os.path.join(OUT, 'curves.npz')
    if not os.path.exists(p):
        print(f"곡선 캐시가 없다: {p}")
        print("창 판정기가 아직 도는 중이면 끝난 뒤에 다시 불러라 —"
              " 곡선은 전 구성이 끝나고 한 번에 저장된다.")
        return None
    z = np.load(p, allow_pickle=True)
    curves = {}
    for k in z.files:
        n, ph = k.rsplit('|', 1)
        curves[(n, int(ph))] = z[k]
    return curves


def main():
    curves = load()
    if curves is None:
        return 1

    names = sorted({n for n, _ph in curves})
    L = len(next(iter(curves.values())))
    print(f"구성 {len(names)}개 · 곡선 {L}봉 ({L/365.0:.2f}년)\n")

    out = []
    for name in names:
        ms = []
        for ph in PHASES:
            if (name, ph) not in curves:
                continue
            m = win_metrics(curves[(name, ph)])
            if m:
                ms.append(m)
        if len(ms) < len(PHASES):
            continue
        a = {k: float(np.mean([x[k] for x in ms]))
             for k in ('cal', 'cagr', 'mdd', 'sharpe')}
        a['cal_min'] = float(np.min([x['cal'] for x in ms]))
        a['cagr_min'] = float(np.min([x['cagr'] for x in ms]))
        out.append((name, a))

    if not out:
        print("앵커 일곱이 다 찬 구성이 없다")
        return 1

    def show(title, key, n=15):
        print(f"\n{title}")
        print("  구성                 칼마   연복리     낙폭    샤프  앵커최저칼마 앵커최저연복리")
        for name, a in sorted(out, key=lambda t: -t[1][key])[:n]:
            print(f"  {name:<20} {a['cal']:6.3f} {a['cagr']:8.4f} {a['mdd']:8.4f} "
                  f"{a['sharpe']:7.3f} {a['cal_min']:9.3f} {a['cagr_min']:11.4f}")

    show(f"칼마 상위 (완성 구성 {len(out)}개, 앵커 {len(PHASES)}곳 평균)", 'cal')
    show("연복리 상위", 'cagr')

    base = dict(out).get('BASE')
    cur = dict(out).get('n3-c33-d10-m127')
    if base and cur:
        print("\n기준선 대조 (같은 설정이라 수치가 같아야 한다)")
        print(f"  BASE              칼마 {base['cal']:6.3f}  연복리 {base['cagr']:8.4f}")
        print(f"  n3-c33-d10-m127   칼마 {cur['cal']:6.3f}  연복리 {cur['cagr']:8.4f}")
        gap = abs(base['cal'] - cur['cal'])
        print(f"  칼마 차이 {gap:.6f}" + ("  (같다)" if gap < 1e-6 else "  ← 벌어졌다. 설정을 다시 봐라"))

    print("\n축별 평균 칼마 / 연복리 / 낙폭")
    # 축은 자리가 아니라 조각의 머리글자로 가른다. 자리로 잡으면 1차 격자(조각 넷)에만
    # 맞아서, 2차 격자 이름(n2-c50-d10-m127-s217-v90)을 줬을 때 뒤의 앵커 간격·변동성
    # 기간 두 축이 에러 없이 집계에서 통째로 빠진다 — 인덱스 0~3 이 그대로 유효해서
    # 아무 경고도 안 난다. 머리글자로 가르면 두 격자가 같은 코드로 돈다.
    label = {'n': '종목수', 'c': '상한', 'd': '드리프트', 'm': '장기모멘텀',
             's': '앵커간격', 'v': '변동성기간'}
    grid = [(n, a) for n, a in out if n != 'BASE']
    axes = {}
    for name, a in grid:
        for piece in name.split('-'):
            axes.setdefault(label.get(piece[:1], piece[:1]), {}) \
                .setdefault(piece, []).append(a)
    for axis, buckets in axes.items():
        print(f"  [{axis}]")
        for k, v in sorted(buckets.items()):
            print(f"    {k:<8} 칼마 {np.mean([x['cal'] for x in v]):6.3f}  "
                  f"연복리 {np.mean([x['cagr'] for x in v]):8.4f}  "
                  f"낙폭 {np.mean([x['mdd'] for x in v]):8.4f}  "
                  f"(구성 {len(v)}개)")

    print("\n읽는 법. 축별 평균은 그 축의 값 하나를 고정하고 나머지 축을 전부 평균한 것이다.")
    print("  한 축이 혼자 결과를 끌면 거기서 보이고, 축끼리 얽히면 평균이 평평해진다.")
    print("  평균이 좋다고 채택 근거가 되지는 않는다 — 노출을 키우면 연복리가 오르고 낙폭도")
    print("  같이 커져서, 칼마만 보면 가려진다. 창 쌍대 판정을 먼저 통과한 구성만 본다.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
