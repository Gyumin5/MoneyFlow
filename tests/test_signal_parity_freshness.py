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

print("\n[8] 백테스트에 넘기는 전략값은 파라미터 대조표를 그대로 따라간다")
from check_params_drift import CANONICAL   # noqa: E402

_canon = CANONICAL['fut']['params']
check("SMA 일수", sp.BT_FUT_PARAMS['sma_days'] == _canon['sma_bars'], sp.BT_FUT_PARAMS)
check("단기 모멘텀 일수", sp.BT_FUT_PARAMS['mom_short_days'] == _canon['mom_short_bars'])
check("장기 모멘텀 일수", sp.BT_FUT_PARAMS['mom_long_days'] == _canon['mom_long_bars'])
check("변동성 문턱", sp.BT_FUT_PARAMS['vol_threshold'] == _canon['vol_threshold'])
check("카나리 히스테리시스", sp.BT_FUT_PARAMS['canary_hyst'] == _canon['canary_hyst'])
check("헬스 방식", sp.BT_FUT_PARAMS['health_mode'] == _canon['health_mode'])
check("값을 따로 베껴 적지 않았다",
      all(k not in sp.BT_FUT_PARAMS for k in ('snap_interval_bars', 'n_snapshots')),
      sp.BT_FUT_PARAMS)

print("\n[9] 판정 — 기대 신호일을 비교하지 못하면 통과를 찍지 않는다")
D1, D2, D3 = datetime(2026, 9, 26), datetime(2026, 9, 27), datetime(2026, 9, 28)
W = {'BTC': 0.333333, 'ETH': 0.333333, 'CASH': 0.333333}
X = {'BTC': 0.333333, 'SOL': 0.333333, 'CASH': 0.333333}

rc, compared, mism, why = sp.decide({D1: W, D2: W, D3: W}, {D1: W, D2: W, D3: W}, D3, 3)
check("전부 같으면 rc 0", (rc, len(compared), len(mism)) == (0, 3, 0), (rc, compared, mism))
check("기대 신호일이 비교 목록에 있다", D3 in compared, compared)

rc, _, mism, _ = sp.decide({D1: W, D2: W, D3: X}, {D1: W, D2: W, D3: W}, D3, 3)
check("기대 신호일이 다르면 rc 1", rc == 1 and mism == [D3], (rc, mism))

rc, _, mism, _ = sp.decide({D1: X, D2: W, D3: W}, {D1: W, D2: W, D3: W}, D3, 3)
check("옛 날짜가 달라도 rc 1", rc == 1 and mism == [D1], (rc, mism))

rc, _, _, why = sp.decide({D1: W, D2: W}, {D1: W, D2: W, D3: W}, D3, 3)
check("백테스트가 기대 신호일을 안 냈으면 rc 2", rc == 2 and '백테스트' in why, (rc, why))

rc, _, _, why = sp.decide({D1: W, D2: W, D3: W}, {D1: W, D2: W}, D3, 3)
check("라이브가 기대 신호일을 안 냈으면 rc 2", rc == 2 and '라이브' in why, (rc, why))

rc, _, _, why = sp.decide({D1: W}, {D1: W}, None, 3)
check("기대 신호일 자체가 없으면 rc 2", rc == 2, (rc, why))

rc, compared, _, _ = sp.decide({D1: W, D2: W, D3: W}, {D1: W, D2: W, D3: W}, D3, 1)
check("days 1 이어도 기대 신호일은 반드시 본다", compared == [D3], compared)

rc, compared, _, _ = sp.decide({D1: W, D2: W, D3: W}, {D2: W, D3: W}, D3, 3)
check("한쪽에만 있는 옛 날짜는 비교에서 빠진다", D1 not in compared and rc == 0, (rc, compared))

print("\n[10] 판정 — 비교 목록은 기대 신호일보다 뒤를 끌어오지 않는다")
D4 = datetime(2026, 9, 29)
rc, compared, _, _ = sp.decide({D2: W, D3: W, D4: X}, {D2: W, D3: W, D4: W}, D3, 3)
check("기대 신호일 이후 날짜는 무시한다", D4 not in compared and rc == 0, (rc, compared))

print("\n[11] 판본 대조 — 서버와 다르거나 확인 자체를 못 하면 통과가 아니다")
import subprocess as _sp   # noqa: E402

_real_run = _sp.run


class _Proc:
    def __init__(self, rc, out='', err=''):
        self.returncode, self.stdout, self.stderr = rc, out, err


def with_fake_ssh(fake):
    _sp.run = fake
    try:
        return sp.server_build_matches(timeout_sec=1)
    finally:
        _sp.run = _real_run


import hashlib   # noqa: E402

_local = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trade', 'auto_trade_binance.py')
with open(_local, 'rb') as _fh:
    _md5 = hashlib.md5(_fh.read()).hexdigest()

ok, why = with_fake_ssh(lambda *a, **k: _Proc(0, f'{_md5}  /home/ubuntu/auto_trade_binance.py\n'))
check("해시가 같으면 통과", ok, why)

ok, why = with_fake_ssh(lambda *a, **k: _Proc(0, 'deadbeef' * 4 + '  /x\n'))
check("해시가 다르면 막힌다", not ok and '다르다' in why, why)

ok, why = with_fake_ssh(lambda *a, **k: _Proc(255, '', 'Connection refused'))
check("접속 실패는 막힌다", not ok and '조회 실패' in why, why)


def _timeout(*a, **k):
    raise _sp.TimeoutExpired('ssh', 1)


ok, why = with_fake_ssh(_timeout)
check("시간 초과는 막힌다", not ok and 'TimeoutExpired' in why, why)

ok, why = with_fake_ssh(lambda *a, **k: _Proc(0, ''))
check("빈 응답은 막힌다", not ok, why)

print()
if FAILS:
    print(f"==== 실패 {len(FAILS)}건: {FAILS}")
    sys.exit(1)
print("==== 전부 통과")
