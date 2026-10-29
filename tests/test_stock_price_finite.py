#!/usr/bin/env python3
"""전략용 가격 수집 경로가 유한한 양수만 들여보내는지 본다 (2026-10-29).

왜 필요했나. 주문·평가 경로는 `_finite` 경계를 이미 지난다. 전략용 가격은 그 경계 밖에
있었다 — 야후 쪽은 `dropna()` 가 숫자 아님만 걸러서 무한대와 0 이하가 그대로 남고,
KIS 보강 쪽은 `float(cl)` 뒤에 `> 0` 만 봐서 `float('inf')` 가 통과한다.

한 칸이 무한대면 그 뒤 수익률·표준편차가 숫자 아님이 되고, Z점수 순위가 바뀌어 다른
종목이 뽑힌다. 주문은 정상처럼 나간다 — 어디서도 안 멈춘다. 열째 토론
(run-20261029T121329Z) 의 action_step 5 마지막 문장이 이 자리다.

여기서 고정하는 계약:
  · 야후 종가에서 무한대·0 이하는 버린다. 버린 뒤 두 줄이 안 남으면 그 종목을 안 쓴다.
  · KIS 보강 종가도 같다 — 유한한 양수만 채운다.
  · 멀쩡한 값은 그대로 들어온다(양성 대조).

망·자격증명 미사용. yfinance 와 KIS 조회를 둘 다 가짜로 바꿔 돌린다.
실행: python3 tests/test_stock_price_finite.py  (bash run_tests.sh 에 자동 포함)
"""
import datetime as _dt
import os
import sys
import types

import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trade'))
import executor_stock as ex  # noqa: E402

FAILED = []


def check(name, cond, detail=''):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILED.append(name)


INF = float('inf')
NAN = float('nan')


def _frame(values, last='2026-10-28'):
    """Close 한 칸짜리 일봉 표. 마지막 날짜가 last 다."""
    idx = pd.bdate_range(end=last, periods=len(values))
    return pd.DataFrame({'Close': values}, index=idx)


def fetch(frames, max_close_date=_dt.date(2026, 10, 28)):
    """_fetch_strategy_prices 를 yfinance 없이 돌린다. frames = {ticker: DataFrame|None}"""
    fake = types.ModuleType('yfinance')
    fake.download = lambda t, **kw: frames.get(t)
    old_mod = sys.modules.get('yfinance')
    old_log, old_fill = ex.log, ex.KIS_FILL_ENABLED
    sys.modules['yfinance'] = fake
    ex.log = lambda *a, **k: None
    ex.KIS_FILL_ENABLED = False        # 이 칸은 야후 쪽 경계만 본다
    try:
        return ex._fetch_strategy_prices(list(frames), max_close_date=max_close_date)
    finally:
        if old_mod is None:
            sys.modules.pop('yfinance', None)
        else:
            sys.modules['yfinance'] = old_mod
        ex.log, ex.KIS_FILL_ENABLED = old_log, old_fill


print("[1] 야후 종가 — 유한한 양수만 들어온다")

_ok = fetch({'SPY': _frame([100.0, 101.0, 102.0, 103.0, 104.0])})
check("[양성] 멀쩡한 다섯 줄은 그대로 들어온다",
      'SPY' in _ok and len(_ok['SPY']) == 5 and float(_ok['SPY'].iloc[-1]) == 104.0,
      str(_ok.get('SPY')))

for _label, _vals in (('무한대', [100.0, 101.0, INF, 103.0, 104.0]),
                      ('음의 무한대', [100.0, 101.0, -INF, 103.0, 104.0]),
                      ('0', [100.0, 101.0, 0.0, 103.0, 104.0]),
                      ('음수', [100.0, 101.0, -5.0, 103.0, 104.0])):
    _got = fetch({'SPY': _frame(_vals)})
    _s = _got.get('SPY')
    check(f"{_label} 한 칸은 버리고 나머지를 쓴다",
          _s is not None and len(_s) == 4 and 104.0 == float(_s.iloc[-1]),
          str(_s))

# 마지막 줄이 무한대면 기준일이 하루 앞으로 당겨진다 — 그 종목만 뒤처진 것으로 보여
# 기존 기준일 정렬 가드가 받는다. 무한대를 기준일로 삼지 않는 것이 핵심이다.
_tail = fetch({'SPY': _frame([100.0, 101.0, 102.0, 103.0, INF])})
check("마지막 줄이 무한대면 그 줄을 기준일로 안 삼는다",
      'SPY' in _tail and float(_tail['SPY'].iloc[-1]) == 103.0, str(_tail.get('SPY')))

# 쓸 수 있는 줄이 두 개 미만이면 그 종목을 통째로 뺀다. 한 줄로는 수익률이 안 나온다.
_thin = fetch({'SPY': _frame([INF, INF, INF, INF, 104.0])})
check("유한한 줄이 모자라면 그 종목을 안 쓴다", 'SPY' not in _thin, str(_thin))

# 숫자 아님은 원래 dropna 가 걸렀다 — 그 동작이 그대로인지 같이 본다.
_nan = fetch({'SPY': _frame([100.0, NAN, 102.0, 103.0, 104.0])})
check("숫자 아님은 예전처럼 걸러진다",
      'SPY' in _nan and len(_nan['SPY']) == 4, str(_nan.get('SPY')))

print("\n[2] KIS 보강 종가 — 같은 경계를 건다")


def fill(clos):
    """_kis_fill_recent 를 통신 없이 돌린다. clos = 응답의 종가 칸."""
    rows = [{'xymd': '20261028', 'clos': clos}]
    old_get, old_log = ex._get, ex.log
    ex._get = lambda *a, **k: {'output2': rows}
    ex.log = lambda *a, **k: None
    try:
        return ex._kis_fill_recent('SPY', _dt.date(2026, 10, 27), _dt.date(2026, 10, 28))
    finally:
        ex._get, ex.log = old_get, old_log


_fok = fill('104.25')
check("[양성] 멀쩡한 종가는 채운다",
      len(_fok) == 1 and list(_fok.values()) == [104.25], str(_fok))

for _label, _cl in (('무한대', 'inf'), ('음의 무한대', '-inf'), ('숫자 아님', 'nan'),
                    ('0', '0'), ('음수', '-3.5')):
    _got = fill(_cl)
    check(f"{_label} 은 안 채운다", _got == {}, str(_got))

print()
if FAILED:
    print(f"❌ {len(FAILED)}건 실패: {FAILED}")
    sys.exit(1)
print("✅ 전부 통과")
