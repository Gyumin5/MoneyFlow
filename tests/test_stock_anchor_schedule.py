#!/usr/bin/env python3
"""주식 스냅샷 앵커 시간표가 위상을 잃지 않는지 확인한다.

2026-07-08 에 세 스냅샷이 같은 날 발화한 뒤로 영원히 같이 움직였다. 원인은 다음 앵커일을
"실제 발화일 + 69" 로 잡은 것이다 — 한 번 겹치면 스스로 못 풀린다. 채택 BT
(stock_engine_snap.py 의 snap_next_sched[i] += snap_days)처럼 예정일에서 세면 위상이 남는다.

여기서 보는 것:
  · 겹친 상태를 한 번에 23일씩 다시 벌린다
  · 실행이 밀려도(주말·휴일·장애) 위상이 안 밀린다
  · 오래 멈춰서 둘이 같은 날 발화해도 다음 예정일은 다시 23일 벌어져 있다
  · 예정일이 깨져 있으면 죽지 않고 다시 심는다
"""
import datetime as _dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trade'))
import executor_stock as ex  # noqa: E402

FAILED = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ''))
    if not cond:
        FAILED.append(name)


class _FrozenDatetime(_dt.datetime):
    """now() 만 고정한 datetime. fromisoformat 등 나머지는 그대로 쓴다."""
    _today = _dt.date(2026, 8, 28)

    @classmethod
    def now(cls, tz=None):
        return _dt.datetime.combine(cls._today, _dt.time(23, 35))


def run_anchor(state, today, picks=('SPY', 'QQQ', 'GLD')):
    """지정한 날짜에 check_anchors 를 한 번 태운다."""
    _FrozenDatetime._today = today
    saved = ex.datetime
    ex.datetime = _FrozenDatetime
    try:
        weights = {t: 1.0 / 3.0 for t in picks}
        ex.check_anchors({'stock': {'risk_on': True,
                                    'offense_picks': list(picks),
                                    'offense_weights': weights}}, state)
    finally:
        ex.datetime = saved
    return state


def sched(state):
    return [state['snapshots'][str(i)].get('next_rebal_date') for i in range(ex.N_SNAPS)]


def d(s):
    return _dt.date.fromisoformat(s)


P = ex.SNAP_PERIOD_DAYS
G = ex.SNAP_STAGGER_DAYS

print(f"\n[0] 상수 (period={P}, stagger={G}, n={ex.N_SNAPS})")
check("스태거가 주기보다 작다", 0 < G < P, f"{G}/{P}")

print("\n[1] 겹친 상태 재분리 — 실제 2026-08-28 state 모양")
today = _dt.date(2026, 8, 28)
collapsed = {'snapshots': {str(i): {'picks': ['VNQ', 'EEM', 'SPY'],
                                    'weights': {'VNQ': 1 / 3, 'EEM': 1 / 3, 'SPY': 1 / 3},
                                    'last_rebal_date': '2026-07-08'}
                           for i in range(ex.N_SNAPS)}}
run_anchor(collapsed, today)
want = [(today + _dt.timedelta(days=i * G)).isoformat() for i in range(ex.N_SNAPS)]
after = sched(collapsed)
# snap0 은 오늘이 예정일이라 바로 발화하고 다음은 오늘+69 로 밀린다.
want[0] = (today + _dt.timedelta(days=P)).isoformat()
check("셋의 예정일이 23일씩 벌어진다", after == want, f"{after} != {want}")
check("snap0 만 오늘 발화", collapsed['snapshots']['0']['last_rebal_date'] == today.isoformat()
      and collapsed['snapshots']['1']['last_rebal_date'] == '2026-07-08', str(collapsed['snapshots']))
check("발화했으니 리밸 대기", collapsed.get('rebalancing_needed') is True)
check("발화한 snap 만 새 picks", collapsed['snapshots']['0']['picks'] == ['SPY', 'QQQ', 'GLD']
      and collapsed['snapshots']['1']['picks'] == ['VNQ', 'EEM', 'SPY'])

print("\n[2] 실행이 밀려도 위상 유지")
base = _dt.date(2026, 9, 15)
st = {'snapshots': {'0': {'next_rebal_date': base.isoformat()},
                    '1': {'next_rebal_date': (base + _dt.timedelta(days=G)).isoformat()},
                    '2': {'next_rebal_date': (base + _dt.timedelta(days=2 * G)).isoformat()}}}
run_anchor(st, base + _dt.timedelta(days=3))  # 주말·휴일로 3일 밀려 실행
check("다음 예정일은 발화일이 아니라 예정일에서 센다",
      sched(st)[0] == (base + _dt.timedelta(days=P)).isoformat(), str(sched(st)))
check("안 온 snap 은 그대로", sched(st)[1:] == [(base + _dt.timedelta(days=G)).isoformat(),
                                                (base + _dt.timedelta(days=2 * G)).isoformat()])

print("\n[3] 장기 정지 후 재개 — 둘이 같은 날 발화해도 안 겹친다")
st = {'snapshots': {'0': {'next_rebal_date': base.isoformat()},
                    '1': {'next_rebal_date': (base + _dt.timedelta(days=G)).isoformat()},
                    '2': {'next_rebal_date': (base + _dt.timedelta(days=2 * G)).isoformat()}}}
late = base + _dt.timedelta(days=P + 10)  # 79일 정지 → snap0·snap1·snap2 전부 지남
run_anchor(st, late)
after = sched(st)
# 발화한 snap 은 한 주기 건너뛰므로 번호순이 곧 날짜순은 아니다. 보는 건 간격이다.
_srt = sorted(d(x) for x in after)
gaps = [(_srt[i + 1] - _srt[i]).days for i in range(ex.N_SNAPS - 1)]
check("셋 다 미래로 밀렸다", all(d(x) > late for x in after), str(after))
check("간격이 23일로 남는다", gaps == [G] * (ex.N_SNAPS - 1), f"{after} gaps={gaps}")
check("위상이 원래 시간표와 같다",
      all((d(after[i]) - (base + _dt.timedelta(days=i * G))).days % P == 0
          for i in range(ex.N_SNAPS)), str(after))

print("\n[4] 예정일이 깨져 있으면 다시 심는다")
st = {'snapshots': {'0': {'next_rebal_date': 'not-a-date'},
                    '1': {'next_rebal_date': None},
                    '2': {}}}
run_anchor(st, today)
after = sched(st)
check("죽지 않고 셋 다 예정일을 갖는다", all(after), str(after))
_srt = sorted(d(x) for x in after)
check("재분리 간격 23일",
      [(_srt[i + 1] - _srt[i]).days for i in range(ex.N_SNAPS - 1)] == [G] * (ex.N_SNAPS - 1),
      str(after))

print("\n[5] 예정일 전에는 아무 일도 없다")
st = {'snapshots': {str(i): {'picks': ['VNQ'], 'weights': {'VNQ': 1.0},
                             'next_rebal_date': (today + _dt.timedelta(days=1 + i * G)).isoformat()}
                    for i in range(ex.N_SNAPS)}}
run_anchor(st, today)
check("발화 없음", not st.get('rebalancing_needed'), str(st.get('rebalancing_needed')))
check("picks 보존", all(st['snapshots'][str(i)]['picks'] == ['VNQ'] for i in range(ex.N_SNAPS)))

print("\n[6] 회귀 — 발화일 기준으로 다음 앵커를 잡지 않는다")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'trade', 'executor_stock.py')).read()
check("고정 시간표 전진 규칙이 있다", 'while next_date <= today:' in src)
check("next_rebal_date 를 상태로 들고 다닌다", "snap['next_rebal_date'] = next_date.isoformat()" in src)

print()
if FAILED:
    print(f"{len(FAILED)} FAIL: {FAILED}")
    sys.exit(1)
print("ALL PASS")
