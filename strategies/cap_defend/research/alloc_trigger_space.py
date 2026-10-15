"""토론이 동결한 전이 신호 E 141종을 실제로 생성하고, 같은 것끼리 합친다.

2026-10-15 ai-debate(run-20261015T005408Z)가 후보 문법을 동결하면서 "명목 수는 고유 후보
수가 아니다. 최종 고유 수와 제외 목록은 실행 담당자가 생성 결과로 확정한다" 고 넘겼다.
이 파일이 그 몫이다.

합치는 기준은 성과가 아니라 발화 날짜다. 두 명세가 이 구간에서 같은 날짜 집합에
발화하면 뒤에 무엇을 붙이든 같은 팔이다. 성과는 한 번도 안 본다 — 보고 줄이면 규약 위반이다.

E 141종의 문법 (토론 action_steps 2):
  ① 63종 = 슬리브 셋을 각각 미감시·꺼짐·켜짐·양방향 중 하나로 두고(4³−1),
          감시 슬리브 중 어느 하나라도 지정 전이를 보이면 발화.
  ② 54종 = 감시 슬리브 둘 이상이 같은 판단일에 모두 지정 전이를 보여야 발화(동시 전이).
          감시 집합 4가지(쌍 셋 + 셋 하나) × 각 감시 슬리브 방향 3가지.
  ③ 24종 = 슬리브 둘 이상의 카나리 상태를 OR 또는 AND 로 합성한 뒤,
          그 합성 상태 자체의 전이를 본다. 집합 4 × 합성식 2 × 방향 3.
  ①의 OR 와 ③의 합성 상태 전이는 다른 것이다 — 사건을 모으는 것과 상태를 모은 뒤
  그 상태가 바뀌는 것을 보는 것이 다르기 때문이다.

쓰는 계열은 라이브의 '자산배분 게이트' 다(alloc_canary 참조). 현물만 슬리브 카나리와
갈리고, 그 차이가 전이 횟수 87 대 154 다.
"""
from __future__ import annotations

import itertools
import os
import sys
from collections import defaultdict

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

SLEEVES = ('주식', '현물', '선물')
DIRS = ('꺼짐', '켜짐', '양방향')
MODES = ('미감시',) + DIRS


def gate_series() -> dict:
    """세 슬리브의 배분 게이트 계열. 공통 날짜로 맞춘다."""
    import alloc_canary as A
    from unified_backtest import load_data

    bars, _ = load_data("D")
    spot = A.spot_alloc_gate(bars)      # 단순 문턱 — 라이브 배분 게이트
    fut = A.coin_canary(bars)           # binance_state 의 슬리브 카나리 그대로
    stock = A.stock_canary()            # EEM 히스테리시스 — 슬리브와 같다

    idx = stock.index.intersection(spot.index).intersection(fut.index)
    idx = pd.DatetimeIndex(sorted(idx))
    return {'주식': stock.loc[idx], '현물': spot.loc[idx], '선물': fut.loc[idx]}


def _transitions(s: pd.Series, direction: str) -> pd.Series:
    """지정 방향의 전이만 True. 첫 날은 전이로 안 센다(직전 상태를 모른다)."""
    prev = s.shift(1)
    turned_on = (~prev.fillna(s.iloc[0])) & s
    turned_off = prev.fillna(s.iloc[0]) & (~s)
    out = {'켜짐': turned_on, '꺼짐': turned_off,
           '양방향': turned_on | turned_off}[direction]
    out = out.copy()
    out.iloc[0] = False
    return out.astype(bool)


def build_E(g: dict) -> dict:
    """E 141종. 값은 발화일 불리언 계열."""
    out: dict[str, pd.Series] = {}

    # ① 어느 감시 슬리브든 전이하면 (4^3 - 1 = 63)
    for combo in itertools.product(MODES, repeat=3):
        if all(m == '미감시' for m in combo):
            continue
        fire = None
        for sl, m in zip(SLEEVES, combo):
            if m == '미감시':
                continue
            t = _transitions(g[sl], m)
            fire = t if fire is None else (fire | t)
        name = "어느하나|" + ",".join(f"{sl}:{m}" for sl, m in zip(SLEEVES, combo)
                                      if m != '미감시')
        out[name] = fire

    # ② 감시 슬리브 둘 이상이 같은 날 모두 전이 (3 쌍 × 9 + 1 삼중 × 27 = 54)
    subsets = [c for r in (2, 3) for c in itertools.combinations(SLEEVES, r)]
    for sub in subsets:
        for dirs in itertools.product(DIRS, repeat=len(sub)):
            fire = None
            for sl, d in zip(sub, dirs):
                t = _transitions(g[sl], d)
                fire = t if fire is None else (fire & t)
            name = "동시|" + ",".join(f"{sl}:{d}" for sl, d in zip(sub, dirs))
            out[name] = fire

    # ③ 합성 상태를 먼저 만들고 그 상태의 전이 (4 × 2 × 3 = 24)
    for sub in subsets:
        for op in ('OR', 'AND'):
            st = None
            for sl in sub:
                s = g[sl]
                st = s if st is None else (st | s if op == 'OR' else st & s)
            for d in DIRS:
                name = f"합성{op}|" + ",".join(sub) + f"|{d}"
                out[name] = _transitions(st, d)
    return out


def main() -> int:
    g = gate_series()
    idx = g['주식'].index
    print(f"공통 구간 {idx[0].date()} ~ {idx[-1].date()} ({len(idx)}일)")
    for sl in SLEEVES:
        s = g[sl]
        print(f"  {sl}: 켜짐 {int(s.sum())} · 꺼짐 {int((~s).sum())}"
              f" · 꺼짐전이 {int(_transitions(s, '꺼짐').sum())}"
              f" · 켜짐전이 {int(_transitions(s, '켜짐').sum())}")

    E = build_E(g)
    print(f"\n생성한 E 명세 {len(E)}종 (토론 명목 141)")

    groups: dict[tuple, list[str]] = defaultdict(list)
    for name, s in E.items():
        key = tuple(s.values.nonzero()[0].tolist())
        groups[key].append(name)

    never = groups.get((), [])
    uniq = {k: v for k, v in groups.items() if k}
    print(f"  한 번도 발화 안 하는 명세 {len(never)}종 → 팔 하나로 합친다")
    print(f"  발화하는 고유 날짜집합 {len(uniq)}종")
    dupes = sum(len(v) - 1 for v in uniq.values())
    print(f"  날짜집합이 겹쳐 합쳐지는 명세 {dupes}종")
    print(f"  => 고유 E = {len(uniq) + (1 if never else 0)}종"
          f" (명목 {len(E)} 대비 {(len(uniq) + (1 if never else 0)) / len(E) * 100:.0f}%)")

    sizes = sorted(((len(k), v[0]) for k, v in uniq.items()), reverse=True)
    print("\n  발화 많은 쪽 5")
    for n, nm in sizes[:5]:
        print(f"    {n:4d}회  {nm}")
    print("  발화 적은 쪽 5")
    for n, nm in sizes[-5:]:
        print(f"    {n:4d}회  {nm}")

    if never:
        print(f"\n  무발화 명세 예시: {never[:3]}")

    multi = [v for v in uniq.values() if len(v) > 1]
    if multi:
        print(f"\n  합쳐진 묶음 {len(multi)}개 중 앞 3개")
        for v in multi[:3]:
            print(f"    {v[0]}  ←  {', '.join(v[1:4])}"
                  f"{' …' if len(v) > 4 else ''}")
    return 0


if __name__ == '__main__':
    sys.exit(main())
