"""서버가 집 감시의 침묵을 잡는지 검증 (2026-09-30, 토론 run-20260930T054956Z).

왜 필요했나. 신호 대조 감시는 집 크론에서 돌고 경보도 집 유저봇으로 나간다. 집 한 대가
꺼지면 감시와 경보가 같이 죽는데, 그 사실을 알릴 주체도 집이라 아무 일도 안 일어난다.
그동안 조용한 것이 이상 없음으로 읽힌다. 그래서 집이 매 실행마다 영수증을 서버에 올리고,
서버의 독립 크론이 유예시각 뒤에 그 영수증을 보고 없거나 낡았거나 판정이 나쁘면
서버 자기 통로로 알린다.

여기서 보는 것은 판정기(judge)와 그 자기 시험(self_check)이다. 망·자격증명·파일 미사용.
실행: cd /home/gmoh/mon/251229 && python3 trade/test_parity_watch.py
"""
import os
import sys
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'ops'))
import parity_watch as pw  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILS.append(name)


NOW = datetime(2026, 9, 30, 11, 40)          # 유예(10:00+90분) 뒤
TODAY = '2026-09-30'
OLD = '2026-09-27'


def healthy():
    return {
        'refresh': {'mode': 'refresh', 'date': TODAY, 'run_id': 'refresh-A',
                    'checks': {'선물 일봉': 0, '주식 일봉': 0}, 'worst': 0},
        'parity': {'mode': 'parity', 'date': TODAY, 'run_id': 'parity-A',
                   'checks': {'선물': 0, '주식': 0, '선물 스냅 상태': 0}, 'worst': 0},
    }


def alarms(receipts, state=None, now=NOW):
    return pw.judge(now, receipts, state or {})['alarms']


print("# 유예")
check('T1 유예 안에서는 판정하지 않는다',
      pw.judge(datetime(2026, 9, 30, 10, 30), {'refresh': None, 'parity': None},
               {})['waiting'] is True)
check('T1b 유예 안에서는 경보도 없다',
      alarms({'refresh': None, 'parity': None}, now=datetime(2026, 9, 30, 10, 30)) == [])
check('T2 유예가 지나면 판정한다',
      pw.judge(NOW, healthy(), {})['waiting'] is False)

print("\n# 정상")
check('T3 오늘 둘 다 있고 판정이 깨끗하면 조용하다', alarms(healthy()) == [])

print("\n# 침묵")
r = healthy(); r['parity'] = None
a = alarms(r)
check('T4 대조 영수증이 없으면 경보', len(a) == 1 and '신호 대조 영수증이 없다' in a[0], a)
r = healthy(); r['refresh'] = None
a = alarms(r)
check('T5 갱신 영수증만 없어도 경보', any('데이터 갱신 영수증이 없다' in x for x in a), a)
a = alarms({'refresh': None, 'parity': None})
check('T6 둘 다 없으면 경보 둘 (집이 통째로 안 돌았다)', len(a) == 2, a)

print("\n# 낡음")
r = healthy(); r['parity']['date'] = OLD
a = alarms(r)
check('T7 어제 영수증이 그대로 있으면 경보', any('오늘 것이 아니다' in x for x in a), a)
check('T7b 낡은 영수증의 판정은 통과로 안 읽는다',
      not any('일치' in x for x in pw.judge(NOW, r, {})['info']),
      pw.judge(NOW, r, {})['info'])

print("\n# 판정이 나쁨")
r = healthy(); r['parity']['checks']['주식'] = 2
a = alarms(r)
check('T8 검사 하나가 비교불능이면 경보',
      any('주식 비교 못 함' in x for x in a), a)
r = healthy(); r['parity']['checks']['선물'] = 1
check('T9 검사 하나가 불일치면 경보',
      any('선물 불일치' in x for x in alarms(r)), alarms(r))
r = healthy(); r['parity']['checks'] = {}
check('T10 판정이 하나도 없으면 경보 (파이썬이 죽고 껍데기가 적은 경우)',
      any('검사 판정이 하나도 없다' in x for x in alarms(r)), alarms(r))
r = healthy(); r['refresh']['worst'] = 124
check('T11 갱신이 시간초과로 끝났으면 경보',
      any('데이터 갱신이 실패로 끝났다' in x for x in alarms(r)), alarms(r))

print("\n# 실행 식별자")
r = healthy()
check('T12 옛 실행 식별자가 되풀이되면 경보 (옛 영수증 재업로드)',
      any('이미 쓰였다' in x for x in
          alarms(r, {'seen_run_ids': {'parity-A': OLD}})))
check('T13 오늘 이미 본 식별자는 경보가 아니다 (같은 날 두 번 봐도 조용)',
      alarms(r, {'seen_run_ids': {'parity-A': TODAY, 'refresh-A': TODAY}}) == [])
r = healthy(); r['parity'].pop('run_id')
check('T14 식별자가 없으면 경보',
      any('실행 식별자가 없다' in x for x in alarms(r)), alarms(r))
check('T15 갱신과 대조는 서로 다른 식별자다',
      healthy()['refresh']['run_id'] != healthy()['parity']['run_id'])

print("\n# 중복 억제 키")
r = healthy(); r['parity'] = None
k1 = pw.judge(NOW, r, {})['alarm_key']
k2 = pw.judge(NOW, r, {})['alarm_key']
check('T16 같은 이상이면 같은 억제 키', k1 == k2 and k1 != '')
r2 = healthy(); r2['refresh'] = None
check('T17 다른 이상이면 다른 키', pw.judge(NOW, r2, {})['alarm_key'] != k1)
check('T18 정상이면 키가 비어 있다', pw.judge(NOW, healthy(), {})['alarm_key'] == '')

print("\n# 양성 대조 (판정기가 살아 있나)")
check('T19 자기 시험이 통과한다', pw.self_check(NOW) == [], pw.self_check(NOW))
# 실측으로 잡은 구멍 — 유예 안에 돌면 judge 가 대기를 반환해서 자기 시험이 전부
# "경보가 안 났다" 로 실패했다(2026-09-30 서버 시험, rc 2). 판정 시각을 유예 뒤로
# 고정해 고쳤다. 이 줄이 그 회귀를 못박는다.
check('T19b 유예 안에 돌아도 자기 시험은 통과한다',
      pw.self_check(datetime(2026, 9, 30, 10, 30)) == [],
      pw.self_check(datetime(2026, 9, 30, 10, 30)))
check('T19c 자정 직전에 돌아도 통과한다',
      pw.self_check(datetime(2026, 9, 30, 23, 58)) == [])


# 판정기를 일부러 망가뜨리면 자기 시험이 그걸 잡아야 한다. 안 잡히면 자기 시험이
# 장식이라는 뜻이다 — 통과만 말하는 감시는 감시가 아니다.
_real = pw.judge
try:
    pw.judge = lambda now, receipts, state: {
        'alarms': [], 'info': [], 'waiting': False, 'alarm_key': ''}
    fails = pw.self_check(NOW)
    check('T20 판정기가 늘 조용하면 자기 시험이 잡는다', len(fails) >= 3, fails)
finally:
    pw.judge = _real

_real2 = pw.judge
try:
    pw.judge = lambda now, receipts, state: {
        'alarms': ['늘 경보'], 'info': [], 'waiting': False, 'alarm_key': 'x'}
    fails = pw.self_check(NOW)
    check('T21 판정기가 늘 경보면 자기 시험이 잡는다',
          any('건강한 영수증' in f for f in fails), fails)
finally:
    pw.judge = _real2

print("\n# 영수증이 깨졌을 때")
r = healthy(); r['parity'] = {'date': None, 'broken': 'JSONDecodeError: x'}
check('T22 깨진 영수증은 오늘 것이 아니라고 본다 (통과로 안 읽는다)',
      any('오늘 것이 아니다' in x for x in alarms(r)), alarms(r))

print("\n# 경계")
check('T23 유예 경계 직전은 대기',
      pw.judge(datetime(2026, 9, 30, 11, 29), {'parity': None}, {})['waiting'] is True)
check('T24 유예 경계에서 판정 시작',
      pw.judge(datetime(2026, 9, 30, 11, 30), {'parity': None}, {})['waiting'] is False)

print()
if FAILS:
    print(f"실패 {len(FAILS)}건: {FAILS}")
    sys.exit(1)
print("ALL PASS (T1~T24)")
