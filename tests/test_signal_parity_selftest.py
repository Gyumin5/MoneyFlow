#!/usr/bin/env python3
"""선물 신호 대조의 자기시험 검사 — 합성 일봉이 실제로 양성·음성 대조가 되는지 본다.

여기서 지키려는 것 하나 — 자기시험이 시장 상태에 따라 아무것도 증명하지 못하는 날이
없어야 한다. 옛 방식(라이브 단기 모멘텀 기간 흔들기)은 180 신호일 중 130일에서
목표가 그대로라 통과·실패를 가르지 못했고, 그날의 감시는 통째로 막혔다.

망도 거래소도 실데이터도 안 쓴다. 합성 일봉만 만든다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trade', 'ops'))

import signal_parity_daily as sp   # noqa: E402

FAILS = []


def check(name, cond, detail: object = ''):
    if cond:
        print(f'  ok   {name}')
    else:
        print(f'  FAIL {name} {detail}')
        FAILS.append(name)


UNIVERSE = list(sp.REF_UNIVERSE)
SIG = None   # 기대 신호일 (아래에서 채운다)


print('[1] 합성 일봉이 의도한 조건을 실제로 만든다')
syn = {}
for rising in (True, False):
    syn[rising] = sp.synthetic_bars(UNIVERSE, rising)
    ok, detail = sp.synthetic_facts(syn[rising], rising)
    check(f'{sp.SYN_NAME[rising]} 합성이 조건을 만족한다', ok, detail)
    print(f'       {detail}')

check('두 벌의 봉 개수가 같다',
      len(syn[True]['BTC']) == len(syn[False]['BTC']) == sp.SYN_BARS,
      (len(syn[True]['BTC']), len(syn[False]['BTC'])))
check('유니버스 전원에게 봉을 만든다',
      set(syn[True]) == set(UNIVERSE), set(UNIVERSE) - set(syn[True]))
check('워밍업이 장기 모멘텀 127봉 + 백테스트 시작 지연 43봉을 덮는다',
      sp.SYN_BARS > 127 + 43 + 2, sp.SYN_BARS)

SIG = syn[True]['BTC'].index[-2]
check('두 벌의 기대 신호일이 같다', syn[False]['BTC'].index[-2] == SIG)


print('\n[2] 세기를 반쯤 줄이면 조건 검사가 실패를 돌린다 (검사가 늘 참이 아니다)')
orig_rate, orig_flat = sp.SYN_RATE, sp.SYN_FLAT_RATE
try:
    sp.SYN_RATE = 0.0001           # 카나리 문턱 1.5% 를 못 넘는 세기
    weak = sp.synthetic_bars(UNIVERSE, True)
    ok, detail = sp.synthetic_facts(weak, True)
    check('약한 상승은 조건 불만족으로 잡힌다', not ok, detail)
    check('사유에 카나리가 나온다', '카나리' in detail, detail)

    sp.SYN_RATE = orig_rate
    sp.SYN_FLAT_RATE = -0.002      # 나머지 후보가 오히려 오르는 경우
    bad = sp.synthetic_bars(UNIVERSE, True)
    ok, detail = sp.synthetic_facts(bad, True)
    check('나머지 후보가 건강해지면 조건 불만족으로 잡힌다', not ok, detail)
finally:
    sp.SYN_RATE, sp.SYN_FLAT_RATE = orig_rate, orig_flat


print('\n[3] 두 엔진이 미리 정한 기대 목표를 낸다 (양쪽이 같이 틀리는 경우를 거른다)')
end_date = str(syn[True]['BTC'].index[-1])[:10]
bt, live = {}, {}
for rising in (True, False):
    bt[rising] = sp._run_bt(syn[rising], {}, UNIVERSE, end_date)
    live[rising] = sp._run_live(syn[rising], UNIVERSE, [SIG])
    name = sp.SYN_NAME[rising]
    want = sp.SYN_EXPECT[rising]
    check(f'{name} 합성 백테스트 목표 = {want}', bt[rising].get(SIG) == want, bt[rising].get(SIG))
    check(f'{name} 합성 라이브 목표 = {want}', live[rising].get(SIG) == want, live[rising].get(SIG))

check('상승과 하락의 기대 목표가 서로 다르다', sp.SYN_EXPECT[True] != sp.SYN_EXPECT[False])


print('\n[4] 네 갈래 판정 — 교란 없으면 0, 한쪽만 흔들면 1')
for live_r, bt_r, want in ((True, True, 0), (False, False, 0),
                           (True, False, 1), (False, True, 1)):
    rc, _c, _m, _w = sp.decide(bt[bt_r], live[live_r], SIG, 1)
    tag = f'라이브 {sp.SYN_NAME[live_r]} · 백테스트 {sp.SYN_NAME[bt_r]}'
    check(f'{tag} → {want}', rc == want, rc)


print('\n[5] self_check 자체가 통과하고, 실데이터·오늘 날짜를 안 읽는다')
ok, why = sp.self_check(UNIVERSE)
check('self_check 통과', ok, why)
check('설명에 네 갈래가 다 나온다', why.count('→') == 4, why)
ok2, why2 = sp.self_check(UNIVERSE)
check('두 번 불러도 설명이 같다 (시장·시각에 안 흔들린다)', why == why2)
check('self_check 는 인자로 유니버스만 받는다',
      sp.self_check.__code__.co_varnames[:sp.self_check.__code__.co_argcount] == ('universe',),
      sp.self_check.__code__.co_varnames[:sp.self_check.__code__.co_argcount])


print('\n[6] 기대 목표가 틀리면 self_check 가 실패를 돌린다 (늘 통과하는 검사가 아니다)')
orig_expect = dict(sp.SYN_EXPECT)
try:
    sp.SYN_EXPECT = {True: {'ETH': 1.0}, False: orig_expect[False]}
    ok, why = sp.self_check(UNIVERSE)
    check('기대 목표를 바꾸면 실패한다', not ok, why)
    check('사유에 미리 정한 값이 나온다', '미리 정한' in why, why)
finally:
    sp.SYN_EXPECT = orig_expect


print('\n[7] 보조 대조는 전액 현금인 날 해당 없음으로 적고 판정에 안 쓴다')
why = sp.aux_check({}, {}, UNIVERSE, SIG, 1, {'CASH': 1.0}, {})
check('전액 현금이면 해당 없음', why.startswith('해당 없음'), why)
check('사유에 이유가 적힌다', '전액 현금' in why, why)


print()
if FAILS:
    print(f'FAILED {len(FAILS)}: {FAILS}')
    sys.exit(1)
print('ALL PASS')
