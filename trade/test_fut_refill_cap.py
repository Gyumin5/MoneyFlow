#!/usr/bin/env python3
"""선물 리필 경로의 코인당 상한(CAP) 검증 (2026-10-02).

고치기 전의 리필은 빈 비중을 "채울 코인 수" 로만 나눴다. 채울 후보가 빈 자리 수보다
적으면 한 코인이 빈 비중을 통째로 받아 CAP(1/3)을 넘었다. 선정 경로는 CAP 을 거는데
리필 경로만 안 걸던 비대칭이다.

확인하는 것 일곱.
  T1 후보가 모자랄 때 — 두 자리가 비고 후보가 하나면 그 코인은 CAP 까지만 받고
     나머지는 현금으로 간다(고치기 전에는 2/3 가 한 코인에 들어갔다).
  T2 후보가 충분할 때 — 결과가 고치기 전과 같다. 비중·종목·현금이 전부 그대로다.
  T3 후보가 하나도 없을 때 — 탈락 비중 전액이 현금으로 간다.
  T4 비중 합 보존 — 어느 경우에도 스냅 합이 1 이고 음수가 없다.
  T5 기존 현금 보존 — 스냅이 들고 있던 현금이 사라지거나 두 번 더해지지 않는다.
  T6 탈락이 없는 날 — 리필이 아무것도 안 바꾼다.
  T7 이미 CAP 을 넘겨 저장돼 있던 보유는 이 패치가 소급해서 줄이지 않는다.

네트워크를 타지 않는다. 봉은 전부 가짜로 만든다.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import auto_trade_binance as a  # noqa: E402

fails = []


def check(cond, msg):
    if not cond:
        fails.append(msg)


def _mk_df(index, closes):
    return pd.DataFrame({'Open': closes, 'High': closes, 'Low': closes,
                         'Close': closes, 'Volume': [1.0] * len(closes)}, index=index)


def _idx(n):
    now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    cur_open = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    return pd.to_datetime([cur_open - timedelta(days=(n - 1 - i)) for i in range(n)])


def build(n_healthy):
    """탈락 코인 둘 + 건강한 코인 n_healthy 개 + BTC(건강하지만 풀에서 제외) 를 만든다."""
    sp = a.STRATEGIES['D_SMA42']
    n = sp['mom_long_bars'] + 5
    idx = _idx(n)
    down = list(np.linspace(100.0, 70.0, n))      # 두 모멘텀 다 음수 = 탈락
    up = list(np.linspace(70.0, 140.0, n))        # 두 모멘텀 다 양수 = 건강
    bars = {'BTC': _mk_df(idx, up),
            'FAIL_A': _mk_df(idx, down),
            'FAIL_B': _mk_df(idx, down)}
    for i in range(n_healthy):
        bars[f'HEAL_{i}'] = _mk_df(idx, up)
    return {'D': bars}


def state(snaps):
    return {'strategies': {'D_SMA42': {'snapshots': [dict(s) for s in snaps]}}}


def snaps_of(st):
    return st['strategies']['D_SMA42']['snapshots']


T = 1.0 / 3.0
EPS = 1e-9


def main():
    cap = a.CAP

    # T1 — 두 자리가 비고 건강한 후보는 하나뿐.
    st = state([{'FAIL_A': T, 'FAIL_B': T, 'CASH': T}])
    a.apply_refill_v2_fut(st, build(1))
    sn = snaps_of(st)[0]
    coin_w = {k: v for k, v in sn.items() if k.upper() != 'CASH'}
    check(len(coin_w) == 1 and 'HEAL_0' in coin_w,
          f"T1 채운 종목이 HEAL_0 하나가 아니다: {coin_w}")
    check(abs(coin_w.get('HEAL_0', 0) - cap) < EPS,
          f"T1 상한을 안 지켰다: HEAL_0={coin_w.get('HEAL_0')} (기대 {cap})")
    check(abs(sn['CASH'] - (T + T)) < EPS,
          f"T1 남은 몫이 현금으로 안 갔다: CASH={sn['CASH']} (기대 {T + T})")

    # T2 — 후보가 충분하면 고치기 전과 같은 결과여야 한다.
    st = state([{'FAIL_A': T, 'FAIL_B': T, 'CASH': T}])
    a.apply_refill_v2_fut(st, build(4))
    sn = snaps_of(st)[0]
    coin_w = {k: v for k, v in sn.items() if k.upper() != 'CASH'}
    check(len(coin_w) == 2, f"T2 두 자리를 다 안 채웠다: {coin_w}")
    check(all(abs(v - T) < EPS for v in coin_w.values()),
          f"T2 비중이 1/3 이 아니다: {coin_w}")
    check(abs(sn['CASH'] - T) < EPS, f"T2 현금이 바뀌었다: {sn['CASH']} (기대 {T})")

    # T3 — 건강한 후보가 하나도 없으면 전액 현금.
    st = state([{'FAIL_A': T, 'FAIL_B': T, 'CASH': T}])
    a.apply_refill_v2_fut(st, build(0))
    sn = snaps_of(st)[0]
    coin_w = {k: v for k, v in sn.items() if k.upper() != 'CASH'}
    check(not coin_w, f"T3 후보가 없는데 코인이 들어갔다: {coin_w}")
    check(abs(sn['CASH'] - 1.0) < EPS, f"T3 전액 현금이 아니다: {sn['CASH']}")

    # T4 — 합 보존과 음수 없음. 후보 수를 0~4 로 바꿔가며 본다.
    for nh in range(5):
        st = state([{'FAIL_A': T, 'FAIL_B': T, 'CASH': T}])
        a.apply_refill_v2_fut(st, build(nh))
        sn = snaps_of(st)[0]
        tot = sum(sn.values())
        check(abs(tot - 1.0) < 1e-9, f"T4 후보 {nh}개: 합이 1 이 아니다 ({tot})")
        check(all(v >= -EPS for v in sn.values()), f"T4 후보 {nh}개: 음수 비중 {sn}")
        for k, v in sn.items():
            if k.upper() != 'CASH':
                check(v <= cap + EPS, f"T4 후보 {nh}개: {k} 가 상한을 넘었다 ({v})")

    # T5 — 기존 현금이 보존된다. 현금 0 인 스냅과 현금 1/3 인 스냅을 같이 돌린다.
    st = state([{'FAIL_A': T, 'FAIL_B': T, 'HEAL_0': T},
                {'FAIL_A': T, 'FAIL_B': T, 'CASH': T}])
    a.apply_refill_v2_fut(st, build(1))
    s0, s1 = snaps_of(st)
    # 스냅0 은 HEAL_0 를 이미 들고 있어 새로 채울 후보가 없다 → 두 자리 다 현금.
    check(abs(s0.get('CASH', 0) - (T + T)) < EPS,
          f"T5 스냅0 현금이 틀렸다: {s0}")
    check(abs(s0.get('HEAL_0', 0) - T) < EPS, f"T5 스냅0 기존 보유가 바뀌었다: {s0}")
    check(abs(s1.get('CASH', 0) - (T + T)) < EPS, f"T5 스냅1 현금이 틀렸다: {s1}")

    # T6 — 탈락이 없으면 아무것도 안 바뀐다.
    before = {'HEAL_0': T, 'HEAL_1': T, 'CASH': T}
    st = state([before])
    a.apply_refill_v2_fut(st, build(2))
    sn = snaps_of(st)[0]
    check({k: round(v, 9) for k, v in sn.items()} ==
          {k: round(v, 9) for k, v in before.items()},
          f"T6 탈락이 없는데 바뀌었다: {sn} (이전 {before})")

    # T7 — 이미 상한을 넘겨 저장된 보유는 소급해서 안 줄인다.
    st = state([{'HEAL_0': 0.667, 'FAIL_A': 0.333}])
    a.apply_refill_v2_fut(st, build(1))
    sn = snaps_of(st)[0]
    check(abs(sn.get('HEAL_0', 0) - 0.667) < EPS,
          f"T7 기존 초과 보유를 건드렸다: {sn}")
    check(abs(sn.get('CASH', 0) - 0.333) < EPS,
          f"T7 탈락 비중이 현금으로 안 갔다: {sn}")

    print("# 선물 리필 상한 검증")
    print(f"  CAP = {cap:.6f}")
    if fails:
        for f in fails:
            print("  FAIL:", f)
        print(f"FAILED ({len(fails)})")
        return 1
    print("ALL PASS (T1~T7)")
    return 0


if __name__ == '__main__':
    sys.exit(main())
