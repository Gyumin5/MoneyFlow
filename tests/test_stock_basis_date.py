"""주식 신호의 종가 기준일 표시와 대체 신호 차단 검증 (2026-09-30).

왜 필요했나. 일일 리포트는 09:15 KST 에 도는데 야후 수정종가가 그 시점에 직전 세션을
아직 안 채우는 날이 있다. 그러면 아침에 보는 주식 종목이 그날 밤 23:35 에 실제로 체결될
종목과 다를 수 있다. 게다가 리포트가 보여주던 날짜는 비트코인 시계열의 마지막 날짜였다 —
주식이 쓴 종가의 날짜가 아니다. 2026-09-29 실측에서 표기는 09-28, 실제 계산은 09-25 였다.

매매에는 보통 영향이 없다(executor 가 23:35 에 가격을 다시 받는다). 다만 그 다시 받기가
실패하면 아침에 저장된 signal_state.json 이 매매 신호로 쓰이는 경로가 코드에 있다.
그 경로가 낡은 종가로 만든 목표를 그대로 주문으로 바꾼다. 그래서 둘을 같이 본다 —
리포트가 진짜 날짜를 보여주는가, 그리고 대체 신호가 낡았으면 막는가.
근거: ai-debate run-20260930T045429Z.

망·자격증명 미사용.
실행: cd /home/gmoh/mon/251229 && python3 tests/test_stock_basis_date.py
"""
import os
import sys

import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'strategies', 'cap_defend'))
sys.path.insert(0, os.path.join(_HERE, '..', 'trade'))
import stock_strategy_v25 as ss  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILS.append(name)


def prices(last, n=5, overrides=None):
    out = {t: pd.Series(range(n), dtype=float,
                        index=pd.bdate_range(end=last, periods=n))
           for t in ss.STOCK_ALL_TICKERS}
    for t, d in (overrides or {}).items():
        out[t] = pd.Series(range(n), dtype=float,
                           index=pd.bdate_range(end=d, periods=n))
    return out


def at(s):
    return pd.Timestamp(s, tz='Asia/Seoul')


# 2026-09-25 금 / 09-26 토 / 09-27 일 / 09-28 월 / 09-29 화
print("# 기준일 산출")

b = ss.stock_basis(prices('2026-09-28'), at('2026-09-29 09:15'))
check('T1 직전 세션까지 다 있으면 경고 없음', b['warn'] == '' and b['lag_bd'] == 0, b)
check('T1b 기준일이 실제 종가 날짜다', str(b['basis']) == '2026-09-28', b['basis'])

# 09-29 실측 재현 — 전 종목이 금요일 종가까지만 있다.
b = ss.stock_basis(prices('2026-09-25'), at('2026-09-29 09:15'))
check('T2 한 세션 지연을 잡는다', b['lag_bd'] == 1 and b['warn'], b)
check('T2b 비트코인 날짜가 아니라 주식 종가 날짜를 낸다',
      str(b['basis']) == '2026-09-25', b['basis'])
check('T2c 단정하지 않는다 — 휴장일 가능성을 같이 적는다',
      '휴장일' in b['warn'], b['warn'])

# 주말 아침은 지연이 아니다.
b = ss.stock_basis(prices('2026-09-25'), at('2026-09-26 09:15'))
check('T3 토요일 아침은 금요일 종가가 정상', b['warn'] == '', b)
b = ss.stock_basis(prices('2026-09-25'), at('2026-09-28 09:15'))
check('T4 월요일 아침은 금요일 종가가 정상', b['warn'] == '', b)

# 종목마다 날짜가 다르면 단일 기준일로 표시하지 않는다.
b = ss.stock_basis(prices('2026-09-28', overrides={'VNQ': '2026-09-25'}),
                   at('2026-09-29 09:15'))
check('T5 섞이면 단일 기준일을 안 준다', b['basis'] is None and b['mixed'], b)
check('T5b 계산을 지배하는 가장 오래된 날짜를 낸다',
      str(b['oldest']) == '2026-09-25', b['oldest'])
check('T5c 종목별 날짜를 같이 낸다',
      b['per_ticker'].get('VNQ') == '2026-09-25'
      and b['per_ticker'].get('SPY') == '2026-09-28', b['per_ticker'])

b = ss.stock_basis({}, at('2026-09-29 09:15'))
check('T6 가격이 하나도 없으면 확인 불가로 말한다',
      b['basis'] is None and '확인 불가' in b['warn'], b)

# ─────────────────────────────────────────────────────────────
print("\n# 대체 신호 차단")

os.environ.setdefault('KIS_APP_KEY', 'x')
import executor_stock as ex  # noqa: E402

NOW = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')


def signal(**stock_kw):
    stk = {'offense_picks': ['SPY'], 'offense_weights': {'SPY': 1.0},
           'defense_picks': [], 'defense_weights': {}, 'risk_on': True,
           'computed_side': 'offense', 'stale_side': 'defense',
           'basis_date': '2026-09-28', 'basis_oldest': '2026-09-28',
           'basis_mixed': False, 'basis_lag_bd': 0,
           'basis_per_ticker': {'SPY': '2026-09-28'}}
    stk.update(stock_kw)
    return {'stock': stk,
            'meta': {'strategy_version': 'V25', 'updated_at': NOW}}


check('T7 기준일이 싱싱하면 통과', ex._signal_state_fresh(signal()) is True)
check('T8 허용 밖 지연은 막는다',
      ex._signal_state_fresh(signal(basis_lag_bd=ex.DATE_ALIGN_TOLERANCE_BD + 1)) is False)
check('T9 허용 안 지연은 통과',
      ex._signal_state_fresh(signal(basis_lag_bd=ex.DATE_ALIGN_TOLERANCE_BD)) is True)
check('T10 날짜가 섞였으면 막는다',
      ex._signal_state_fresh(signal(basis_mixed=True)) is False)
check('T11 지연 값이 없으면 막는다',
      ex._signal_state_fresh(signal(basis_lag_bd=None)) is False)

# 기준일 자체가 없는 파일 = 리포트 판본이 낡았다. 추정해서 통과시키지 않는다.
_old = signal()
for k in ('basis_date', 'basis_oldest', 'basis_mixed', 'basis_lag_bd',
          'basis_per_ticker'):
    _old['stock'].pop(k)
check('T12 기준일이 아예 없으면 막는다', ex._signal_state_fresh(_old) is False)

# 24시간 신선도는 그대로 살아 있어야 한다 — 이 검사가 그걸 대체하지 않는다.
_stale = signal()
_stale['meta']['updated_at'] = '2020-01-01 09:15'
check('T13 24시간 신선도 검사는 그대로', ex._signal_state_fresh(_stale) is False)

print()
if FAILS:
    print(f"실패 {len(FAILS)}건: {FAILS}")
    sys.exit(1)
print("ALL PASS (T1~T13)")
