"""현물 BT 를 라이브 체결 규칙으로 맞추기 전 확인 — 절대 최소주문·dust 정리의 영향 (2026-08-25).

계기: ai-debate run-20260825T071922Z 결론이 B(라이브는 그대로, BT 를 라이브에 맞춘다)였고,
중재자가 조건을 달았다 — "밴드만 0 으로 바꾸고 정합 끝났다고 선언하지 마라. 라이브의 실제
무거래 구간은 상대 밴드가 아니라 절대 최소주문 5,000원·dust 정리·반올림·수수료가 합쳐서 만든다."

라이브(trade/executor_coin.execute_delta)의 실제 판정:
  - 목표비중 0 이고 보유 있으면 전량매도 (문턱 없음)
  - |목표금액 - 현재금액| > 5,000원 이어야 주문. 상대 밴드 없음.
  - 부분매도 후 잔여가 5,000원 미만이면 전량매도(dust 정리)
  - 매도 추정액이 5,000원 미만이면 스킵
  - 매수는 가용현금 × 0.995 로 비례 축소, 축소 후 5,000원 미만이면 스킵

이 스크립트가 재는 것: 5,000원이라는 절대 문턱이 슬리브 규모 대비 몇 %냐에 따라
성과가 달라지는가. 달라지지 않으면 BT 기본값을 상대 밴드 0 으로 두는 것으로 충분하고,
달라지면 절대 문턱을 BT 에 모형화해야 한다.

MIN_ORDER_FRAC = 5,000원 / 슬리브 평가액. 슬리브 규모별로:
  1억 3,800만원 → 0.000036   (현행 규모 추정: 선물 15%가 약 6만 달러이므로 현물 25%는 이 근방)
  1,000만원     → 0.0005
  100만원       → 0.005
  (비교) 상대밴드 5% = 현행 BT

판정 (결과 보기 전 고정):
  세 규모 모두 밴드 0(문턱 없음)과 윈도우 평균순위 차 0.1 미만이면 → 절대 문턱은 무시해도 된다.
  하나라도 넘으면 → BT 에 절대 문턱을 남기고 슬리브 규모를 파라미터로 명시한다.

재현: git apply strategies/cap_defend/research/bt_spot_rebal_band.engine.patch 먼저.
실행: cd strategies/cap_defend/research && python3 bt_spot_min_order.py
"""
from __future__ import annotations
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

from bt_spot_rebal_band import window_rs, START, END, BASE_TX  # noqa: E402

ARMS = [
    ('밴드0 문턱없음',        0.0,  0.0),
    ('밴드0 + 5천원/1.38억',  0.0,  0.000036),
    ('밴드0 + 5천원/1000만',  0.0,  0.0005),
    ('밴드0 + 5천원/100만',   0.0,  0.005),
    ('밴드5%(현행BT)',        0.05, 0.0),
]


def run_arm(band, minord, tx=BASE_TX):
    from unified_backtest import run as bt_run, load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    os.environ['REBAL_BAND'] = f'{band}'
    os.environ['MIN_ORDER_FRAC'] = f'{minord}'
    os.environ['DUST_FRAC'] = f'{minord}'
    bars, funding = load_data('D')
    m = bt_run(bars, funding, interval='D', asset_type='spot', leverage=1.0,
               tx_cost=tx, start_date=START, end_date=END,
               sma_bars=42, mom_short_bars=20, mom_long_bars=127,
               vol_threshold=0.05, vol_mode='daily',
               n_snapshots=7, snap_interval_bars=217,
               canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
               universe_size=3, cap=1 / 3, health_mode='mom2vol')
    os.environ['REBAL_BAND'] = '0.05'
    os.environ['MIN_ORDER_FRAC'] = '0'
    os.environ['DUST_FRAC'] = '0'
    return m


def main():
    t0 = time.time()
    print(f"기간 {START} ~ {END} | 축: 절대 최소주문 문턱 (상대 밴드는 0 고정)")
    res = {}
    for tag, band, minord in ARMS:
        m = run_arm(band, minord)
        res[tag] = m
        print(f"  {tag:<22s} 완료 ({time.time()-t0:.0f}s)", flush=True)

    eqs = {k: v['_equity'] for k, v in res.items()}
    sums, wins, n = window_rs(eqs)
    print(f"\n윈도우 {n}개")
    print(f"  {'arm':<22s} {'순위':>6s} {'승률':>6s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} "
          f"{'Sharpe':>7s} {'체결':>6s}")
    for tag, _, _ in ARMS:
        m = res[tag]
        print(f"  {tag:<22s} {sums[tag]/n:>6.3f} {wins[tag]/n*100:>5.1f}% {m['Cal']:>6.2f} "
              f"{m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} {m['Sharpe']:>7.2f} {m['Trades']:>6d}")

    base = sums['밴드0 문턱없음'] / n
    print("\n[판정] 밴드0 문턱없음 대비 평균순위 차")
    for tag, _, _ in ARMS[1:4]:
        gap = sums[tag] / n - base
        print(f"  {tag:<22s} {gap:+.3f} → {'무시 가능' if abs(gap) < 0.1 else '유의차 — 모형화 필요'}")
    print(f"\n총 소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
