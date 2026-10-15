"""토론이 세운 격자가 실제로 돌아가는 크기인지 잰다 — 성능이 아니라 예산 측정이다.

2026-10-15 ai-debate(run-20261015T005408Z)가 명목 후보 <금액>개를 확정했다. 그 수가
"고유 후보 수가 아니다" 라고 토론 자신이 적었고, 실행 담당자가 생성 결과로 확정하라고
넘겼다. 그래서 여기서 재는 것은 팔 하나를 한 번 도는 데 드는 시간이고, 그 값으로
전수가 몇 시간인지를 셈한다. 성과는 일절 안 본다 — 결과를 보고 격자를 줄이면 규약 위반이다.

도는 방식은 기존 송금 지연 하니스(bt_alloc_transfer_lag.py)의 simulate 와 같은 모양의
일별 루프다. 슬리브 곡선은 그 하니스가 쓰는 것과 같은 캐시에서 받아 온다.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import alloc_sim_batch as B  # noqa: E402
import bt_alloc_transfer_lag as H  # noqa: E402

# 토론이 동결한 명목 격자 (action_steps 2·3·4·5·6)
GRID = {
    "전이 신호 E": 141,
    "현행 결합": 6,
    "대기 W": 11,
    "휴지 R": 21,
    "최소 편차": 3,
    "송금 범위": 2,
}
BLOCK_CANDIDATES = 141 * 5 * 2 * 2      # 전이를 금지 신호로 쓰는 갈래
CONTROLS = 6                            # 현행·송금없음·T1만·U만·월고정·분기고정
ANCHORS = 7                             # 사전 지정 위상
COST_ARMS = 2                           # 기본 비용 + 비용 5배
PLACEBOS = 200                          # 음성 대조 최소 벌 수


def main() -> int:
    t0 = time.time()
    eq_sp = H.run_spot_live()
    eq_st = H.run_stock_live()
    eq_fu = H.run_fut('margin', 0.015)['_equity']
    load_s = time.time() - t0
    print(f"슬리브 로드 {load_s:.0f}s")

    common = sorted(eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index))
    n_days = len(common)

    reps = 40
    t0 = time.time()
    for _ in range(reps):
        H.simulate(eq_st, eq_sp, eq_fu, 'trigger', 5)
    one_by_one = (time.time() - t0) / reps

    # 배치 시뮬레이터 — 날짜가 아니라 팔을 벡터화한다(alloc_sim_batch, 정합 확인됨)
    Rm = np.column_stack([
        eq_st.loc[common].pct_change().fillna(0).values,
        eq_sp.loc[common].pct_change().fillna(0).values,
        eq_fu.loc[common].pct_change().fillna(0).values,
    ])
    A = 20000
    modes = np.full(A, B.MODE_TRIGGER, dtype=np.int64)
    lags = np.full(A, 5, dtype=np.int64)
    t0 = time.time()
    B.simulate_batch(Rm, modes, lags)
    per_arm = (time.time() - t0) / A

    print(f"공통 일수 {n_days}")
    print(f"  팔 하나씩 (기존 하니스)  {one_by_one * 1000:7.2f}ms  (반복 {reps}회 평균)")
    print(f"  팔 {A:,}개 묶어서       {per_arm * 1000:7.3f}ms  "
          f"({one_by_one / per_arm:.0f}배)  ← 아래 셈은 이 값 기준")

    nominal = 1
    for v in GRID.values():
        nominal *= v
    total_cand = nominal + BLOCK_CANDIDATES + CONTROLS
    print(f"\n명목 후보 {nominal:,} + 금지신호 {BLOCK_CANDIDATES:,}"
          f" + 대조 {CONTROLS} = {total_cand:,}")

    def hours(arms: int) -> float:
        return arms * per_arm / 3600

    main_runs = total_cand * ANCHORS * COST_ARMS
    print(f"\n주 채점  {total_cand:,} × 위상 {ANCHORS} × 비용 {COST_ARMS}"
          f" = {main_runs:,} 회  →  {hours(main_runs):,.0f} 시간"
          f" ({hours(main_runs) / 24 / 365:.1f} 년)")

    placebo_runs = main_runs * PLACEBOS
    cores = os.cpu_count() or 1
    print(f"음성 대조  위 × 위약 {PLACEBOS} = {placebo_runs:,} 회"
          f"  →  한 코어 {hours(placebo_runs):,.0f} 시간"
          f" ({hours(placebo_runs) / 24:.1f} 일)"
          f"  ·  {cores}코어로 나누면 {hours(placebo_runs) / cores:,.1f} 시간")

    # E 축은 발화 날짜집합으로 합치면 141 → 85 다 (alloc_trigger_space.py).
    uniq_E = 85
    shrink = uniq_E / GRID["전이 신호 E"]
    uniq_cand = int(total_cand * shrink)
    print(f"\n고유 E {uniq_E}종으로 합친 뒤 (alloc_trigger_space.py 측정):"
          f" 후보 {uniq_cand:,}")
    print(f"  주 채점 {hours(uniq_cand * ANCHORS * COST_ARMS):.1f} 시간"
          f"  ·  위약까지 {cores}코어로 "
          f"{hours(uniq_cand * ANCHORS * COST_ARMS * PLACEBOS) / cores:.1f} 시간")

    print("\n이 수치는 이 기계, 지금 구현 기준이다. 성과는 안 봤다.")
    return 0


if __name__ == '__main__':
    sys.exit(main())
