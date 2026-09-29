#!/usr/bin/env python3
"""신호 대조 감시의 데이터 지연 게이트와 비중 정규화 검사.

여기서 지키려는 것 하나 — 데이터가 밀렸는데 옛 날짜만 비교하고 "일치" 를 찍는 일이
없어야 한다. 그러면 감시는 매일 통과하면서 아무것도 안 본다.

네트워크도 거래소도 안 쓴다. 가짜 봉만 만든다.
"""
import os
import sys
from datetime import datetime, timedelta, timezone

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trade', 'ops'))

import signal_parity_daily as sp   # noqa: E402

FAILS = []


def check(name, cond, detail: object = ''):
    if cond:
        print(f"  ok   {name}")
    else:
        print(f"  FAIL {name} {detail}")
        FAILS.append(name)


def bars_ending(days_ago: int):
    """마지막 봉이 (기대 신호일 - days_ago) 인 가짜 BTC 일봉."""
    last = (datetime.now(timezone.utc) - timedelta(days=1 + days_ago)).date()
    idx = pd.to_datetime([last - timedelta(days=i) for i in range(5)][::-1])
    return {'BTC': pd.DataFrame({'Close': [1.0] * 5}, index=idx)}


print("[1] 최신 데이터는 통과한다")
last, expected, lag = sp._check_freshness(bars_ending(0), allow_stale_days=1)
check("지연 0일", lag == 0, lag)
check("마지막 봉과 기대 신호일이 같다", str(last)[:10] == str(expected)[:10], (last, expected))

print("\n[2] 허용치 안쪽의 지연은 통과한다")
_, _, lag = sp._check_freshness(bars_ending(1), allow_stale_days=1)
check("지연 1일, 허용 1일", lag == 1, lag)

print("\n[3] 허용치를 넘는 지연은 비교를 막는다")
for stale in (2, 3, 7, 30):
    try:
        sp._check_freshness(bars_ending(stale), allow_stale_days=1)
        check(f"{stale}일 밀림이 막힌다", False, "예외가 안 났다")
    except sp.ParityDataStale as exc:
        check(f"{stale}일 밀림이 막힌다", f"{stale}일 밀렸다" in str(exc), str(exc))
        check(f"{stale}일 밀림 사유에 갱신 명령이 있다",
              'refresh_backtest_data.py' in str(exc), str(exc))

print("\n[4] 음수 허용치는 0 으로 본다 (최신 데이터를 밀린 것으로 오판하지 않는다)")
_, _, lag = sp._check_freshness(bars_ending(0), allow_stale_days=-5)
check("최신인데 통과한다", lag == 0, lag)
try:
    sp._check_freshness(bars_ending(1), allow_stale_days=-5)
    check("음수 허용치에서도 1일 밀림은 막힌다", False, "예외가 안 났다")
except sp.ParityDataStale:
    check("음수 허용치에서도 1일 밀림은 막힌다", True)

print("\n[5] 데이터가 아예 없으면 비교를 막는다")
for label, bad in (("BTC 키 없음", {}),
                   ("BTC 빈 표", {'BTC': pd.DataFrame({'Close': []})})):
    try:
        sp._check_freshness(bad, allow_stale_days=1)
        check(f"{label} 이 막힌다", False, "예외가 안 났다")
    except sp.ParityDataStale as exc:
        check(f"{label} 이 막힌다", 'BTC' in str(exc), str(exc))

print("\n[6] 비중 정규화 — 현금 키 표기를 합치고 0 을 버린다")
check("Cash 가 CASH 로 합쳐진다", sp._norm({'Cash': 0.5, 'BTC': 0.5}) == {'CASH': 0.5, 'BTC': 0.5})
check("CASH 는 그대로", sp._norm({'CASH': 1.0}) == {'CASH': 1.0})
check("0 비중은 빠진다", sp._norm({'BTC': 0.0, 'ETH': 0.5}) == {'ETH': 0.5})
check("1e-12 도 0 으로 본다", sp._norm({'BTC': 1e-12, 'ETH': 0.5}) == {'ETH': 0.5})
check("소수점 6자리에서 끊는다",
      sp._norm({'BTC': 0.3333333333}) == {'BTC': 0.333333}, sp._norm({'BTC': 0.3333333333}))
check("표기만 다른 같은 배분은 같다고 나온다",
      sp._norm({'BTC': 1 / 3, 'Cash': 2 / 3}) == sp._norm({'BTC': 0.3333333333, 'CASH': 0.6666666667}))

print("\n[7] 고정 유니버스는 중복 없이 채워져 있다")
check("38종", len(sp.REF_UNIVERSE) == 38, len(sp.REF_UNIVERSE))
check("중복 없음", len(set(sp.REF_UNIVERSE)) == len(sp.REF_UNIVERSE))
check("BTC 포함", 'BTC' in sp.REF_UNIVERSE)

print()
if FAILS:
    print(f"==== 실패 {len(FAILS)}건: {FAILS}")
    sys.exit(1)
print("==== 전부 통과")
