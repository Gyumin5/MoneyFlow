"""체크포인트 판본 정합 검사 (2026-10-15, 토론 run-20261014T173603Z 지적).

state/20260922-resweep/eq 의 자산곡선은 한 번에 계산된 게 아니다. 앵커 0~6 은
2026-09-23 01:45, 7~10 은 2026-10-15 다. 그 사이에 선물 엔진이 두 번 바뀌었다 —
같은 날 10:14 드리프트 문턱 0.03 → 0.015(b3e08b6), 10-02 리필 코인당 1/3 상한
(6b9ec3d). 두 무리가 다른 엔진이면 앵커 수 비교가 엔진 차이와 섞인다.

재는 법: 현행 팔의 앵커 0~6 을 지금 코드로 다시 계산해 저장본과 맞춰 본다.

2026-10-15 결과 = 어긋난 칸 6/14. 현물 일곱 칸은 전부 같고(상대괴리 2.2e-16),
선물은 a5 만 같고 나머지 여섯이 4.2e-2 ~ 2.7e-1 로 어긋났다. 그래서 저장돼 있던
선물 곡선 546개를 state/20261015-anchor11/stale-fu-20260923/ 로 치우고
recompute_fu_a0a6.sh 로 다시 계산했다. 다시 돌릴 때도 같은 순서로 한다 —
저장본이 어느 판본인지 모른 채로 매긴 순위는 아무것도 증명하지 못한다.
"""
from __future__ import annotations

import os
import sys

import numpy as np

RES = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(RES))
sys.path.insert(0, RES)

os.environ["ANCHORS_N"] = "11"
import bt_coin_param_resweep as R  # noqa: E402

from unified_backtest import load_data  # noqa: E402
from backtest_futures_v25 import build_K2_signal  # noqa: E402

print("코인 일봉 로드 중...")
R._bars, R._funding = load_data("D")
R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                        btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                        l_min=2.0, l_mid=3.0, l_max=4.0)
print(f"  심볼 {len(R._bars)}종")

bad = 0
for a in range(7):
    for kind, runner, phases in (("sp", R.run_spot, R.PH_SPOT),
                                 ("fu", R.run_fut, R.PH_FUT)):
        key = R._key(R.CUR, a, False)
        old = R._load(kind, key)
        if old is None:
            print(f"  a{a} {kind}: 저장본 없음")
            bad += 1
            continue
        new = runner({}, phases[a], False)
        j = old.index.intersection(new.index)
        if len(j) < 100:
            print(f"  a{a} {kind}: 겹치는 날짜 {len(j)} — 비교 불가")
            bad += 1
            continue
        d = float(np.max(np.abs(old[j].values / new[j].values - 1.0)))
        same_len = len(old) == len(new)
        mark = "같다" if (d < 1e-9 and same_len) else "다르다"
        print(f"  a{a} {kind}: 최대 상대괴리 {d:.3e} · 길이 {len(old)}/{len(new)} → {mark}")
        if d >= 1e-9 or not same_len:
            bad += 1

print(f"\n어긋난 칸 {bad}/14")
