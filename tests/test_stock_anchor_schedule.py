#!/usr/bin/env python3
"""주식 스냅샷 앵커 시간표가 위상을 잃지 않는지 확인한다.

2026-07-08 에 세 스냅샷이 같은 날 발화한 뒤로 영원히 같이 움직였다. 원인은 다음 앵커일을
"실제 발화일 + 69" 로 잡은 것이다 — 한 번 겹치면 스스로 못 풀린다. 채택 BT
(stock_engine_snap.py 의 snap_next_sched[i] += snap_days)처럼 예정일에서 세면 위상이 남는다.

여기서 보는 것:
  · 실행이 밀려도(주말·휴일·장애) 위상이 안 밀린다
  · 오래 멈춰서 둘이 같은 날 발화해도 다음 예정일은 다시 23일 벌어져 있다
  · 겹친 상태를 마지막 발화일 기준으로 복원한다 — 배포한 날을 위상으로 삼지 않는다
  · 일부만 깨져도 살아있는 예정일 기준으로 복원한다
  · 날짜가 성해도 위상이 무너졌으면(같은 날 둘) 잡아낸다
  · 상태를 통째로 잃으면 셋을 다 채운다 — 하나만 채워 대량 매도하지 않는다
  · 빈 신호로는 발화하지 않는다
(ai-debate run-20260828T050415Z 의 NO-GO 지적 3건 반영)
"""
import datetime as _dt
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trade'))
import executor_stock as ex  # noqa: E402

FAILED = []
SENT = []


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


def run_anchor(state, today, picks=('SPY', 'QQQ', 'GLD'), weights=None, risk_on=True):
    """지정한 날짜에 check_anchors 를 한 번 태운다."""
    _FrozenDatetime._today = today
    saved_dt, saved_tg = ex.datetime, ex.send_telegram
    ex.datetime = _FrozenDatetime
    ex.send_telegram = lambda m: SENT.append(m)
    del SENT[:]
    try:
        w = weights if weights is not None else {t: 1.0 / 3.0 for t in picks}
        key = 'offense' if risk_on else 'defense'
        ex.check_anchors({'stock': {'risk_on': risk_on,
                                    f'{key}_picks': list(picks),
                                    f'{key}_weights': w}}, state)
    finally:
        ex.datetime, ex.send_telegram = saved_dt, saved_tg
    return state


def sched(state):
    return [state['snapshots'][str(i)].get('next_rebal_date') for i in range(ex.N_SNAPS)]


def d(s):
    return _dt.date.fromisoformat(s)


def gaps(dates):
    srt = sorted(d(x) for x in dates)
    return [(srt[i + 1] - srt[i]).days for i in range(len(srt) - 1)]


P = ex.SNAP_PERIOD_DAYS
G = ex.SNAP_STAGGER_DAYS
TODAY = _dt.date(2026, 8, 28)


def staggered(base):
    """base 부터 23일씩 벌어진 정상 시간표."""
    return {str(i): {'picks': ['VNQ', 'EEM', 'SPY'],
                     'weights': {'VNQ': 1 / 3, 'EEM': 1 / 3, 'SPY': 1 / 3},
                     'last_rebal_date': (base - _dt.timedelta(days=P)).isoformat(),
                     'next_rebal_date': (base + _dt.timedelta(days=i * G)).isoformat()}
            for i in range(ex.N_SNAPS)}


print(f"\n[0] 상수 (period={P}, stagger={G}, n={ex.N_SNAPS})")
check("스태거가 주기보다 작다", 0 < G < P, f"{G}/{P}")
check("세 스냅이 주기를 정확히 나눈다", ex.N_SNAPS * G == P, f"{ex.N_SNAPS}*{G} vs {P}")

print("\n[1] 겹친 상태 복원 — 실제 2026-08-28 서버 state 모양")
collapsed = {'snapshots': {str(i): {'picks': ['EEM', 'VNQ', 'SPY'],
                                    'weights': {'EEM': 1 / 3, 'VNQ': 1 / 3, 'SPY': 1 / 3},
                                    'last_rebal_date': '2026-07-08'}
                           for i in range(ex.N_SNAPS)}}
run_anchor(collapsed, TODAY)
after = sched(collapsed)
want = [(_dt.date(2026, 7, 8) + _dt.timedelta(days=P + i * G)).isoformat()
        for i in range(ex.N_SNAPS)]
check("마지막 발화일 + 69일 기준으로 심는다", after == want, f"{after} != {want}")
check("배포 당일 발화 없음", not collapsed.get('rebalancing_needed'), str(after))
check("보유 picks 그대로", collapsed['snapshots']['0']['picks'] == ['EEM', 'VNQ', 'SPY'])
check("시간표 정비를 알린다", len(SENT) == 1 and '앵커' in SENT[0], str(SENT))

print("\n[2] 실행이 밀려도 위상 유지")
base = _dt.date(2026, 9, 15)
st = {'snapshots': staggered(base)}
run_anchor(st, base + _dt.timedelta(days=3))  # 주말·휴일로 3일 밀려 실행
check("다음 예정일은 발화일이 아니라 예정일에서 센다",
      sched(st)[0] == (base + _dt.timedelta(days=P)).isoformat(), str(sched(st)))
check("안 온 snap 은 그대로", sched(st)[1:] == [(base + _dt.timedelta(days=G)).isoformat(),
                                                (base + _dt.timedelta(days=2 * G)).isoformat()])
check("정상 경로에선 알림 없음", SENT == [], str(SENT))

print("\n[3] 장기 정지 후 재개 — 둘이 같은 날 발화해도 안 겹친다")
st = {'snapshots': staggered(base)}
late = base + _dt.timedelta(days=P + 10)  # 79일 정지 → 셋 다 예정일 지남
run_anchor(st, late)
after = sched(st)
check("셋 다 미래로 밀렸다", all(d(x) > late for x in after), str(after))
check("간격이 23일로 남는다", gaps(after) == [G] * (ex.N_SNAPS - 1), f"{after} {gaps(after)}")
check("위상이 원래 시간표와 같다",
      all((d(after[i]) - (base + _dt.timedelta(days=i * G))).days % P == 0
          for i in range(ex.N_SNAPS)), str(after))

print("\n[4] 일부만 깨짐 — 살아있는 예정일 기준으로 복원")
st = {'snapshots': staggered(base)}
st['snapshots']['0']['next_rebal_date'] = 'not-a-date'
st['snapshots']['2']['next_rebal_date'] = None
run_anchor(st, TODAY)
after = sched(st)
check("살아있는 snap1 을 그대로 둔다",
      after[1] == (base + _dt.timedelta(days=G)).isoformat(), str(after))
check("나머지를 23일 간격으로 되살린다", gaps(after) == [G] * (ex.N_SNAPS - 1), str(after))
check("복원을 알린다(고친 snap 을 지목)",
      len(SENT) == 1 and '재구성' in SENT[0] and '0, 2' in SENT[0], str(SENT))

print("\n[5] 날짜는 성한데 위상이 무너짐 — 같은 날 둘")
st = {'snapshots': staggered(base)}
st['snapshots']['2']['next_rebal_date'] = st['snapshots']['1']['next_rebal_date']
run_anchor(st, TODAY)
after = sched(st)
check("중복을 잡아내 다시 벌린다", gaps(after) == [G] * (ex.N_SNAPS - 1), str(after))
check("가장 낮은 번호의 성한 날짜를 기준 삼는다", after[0] == base.isoformat(), str(after))

print("\n[6] 상태 전체 유실 — 셋을 다 채운다")
st = {'snapshots': {}}
run_anchor(st, TODAY)
after = sched(st)
check("셋 다 오늘 신호로 채워진다",
      all(st['snapshots'][str(i)]['picks'] == ['SPY', 'QQQ', 'GLD'] for i in range(ex.N_SNAPS)),
      str(st['snapshots']))
check("합산 목표가 3분의 1로 안 쪼그라든다",
      abs(sum(ex.merge_tranches(st).values()) - 1.0) < 1e-9, str(ex.merge_tranches(st)))
check("리밸 대기", st.get('rebalancing_needed') is True)
check("다음 시간표는 69일 뒤부터 23일 간격",
      after == [(TODAY + _dt.timedelta(days=P + i * G)).isoformat() for i in range(ex.N_SNAPS)],
      str(after))
check("유실을 알린다", len(SENT) == 1 and '비어' in SENT[0], str(SENT))

print("\n[7] 빈 신호로는 발화하지 않는다")
st = {'snapshots': staggered(base - _dt.timedelta(days=P))}  # 예정일이 이미 지난 상태
before = sched(st)
run_anchor(st, TODAY, picks=(), weights={})
check("보유 picks 보존", st['snapshots']['0']['picks'] == ['VNQ', 'EEM', 'SPY'],
      str(st['snapshots']['0']))
check("시간표도 안 건드린다", sched(st) == before, f"{sched(st)} != {before}")
check("리밸 대기 안 켬", not st.get('rebalancing_needed'))

print("\n[8] 전량 현금 신호(모멘텀 전멸)는 정상 발화")
st = {'snapshots': staggered(base)}
run_anchor(st, base, picks=(), weights={'Cash': 1.0})
check("현금 100% 로 갈아탄다", st['snapshots']['0']['weights'] == {'Cash': 1.0},
      str(st['snapshots']['0']))
check("snap0 만 발화", st['snapshots']['1']['weights'] != {'Cash': 1.0})

print("\n[9] 예정일 전에는 아무 일도 없다")
st = {'snapshots': staggered(TODAY + _dt.timedelta(days=1))}
run_anchor(st, TODAY)
check("발화 없음", not st.get('rebalancing_needed'), str(st.get('rebalancing_needed')))
check("picks 보존", all(st['snapshots'][str(i)]['picks'] == ['VNQ', 'EEM', 'SPY']
                        for i in range(ex.N_SNAPS)))

print("\n[10] 시계가 앞뒤로 튀어도 위상은 남는다")
st = {'snapshots': staggered(base)}
run_anchor(st, base - _dt.timedelta(days=400))  # 시계 역행
check("역행 시 발화 없음", not st.get('rebalancing_needed'), str(sched(st)))
check("역행 시 시간표 불변", sched(st) == [(base + _dt.timedelta(days=i * G)).isoformat()
                                            for i in range(ex.N_SNAPS)], str(sched(st)))
st = {'snapshots': staggered(base)}
run_anchor(st, base + _dt.timedelta(days=365))  # 시계 급전진
after = sched(st)
check("급전진 뒤에도 간격 23일", gaps(after) == [G] * (ex.N_SNAPS - 1), str(after))
check("급전진 뒤 예정일은 모두 미래",
      all(d(x) > base + _dt.timedelta(days=365) for x in after), str(after))

print("\n[11] 같은 날 두 번 실행해도 두 번 갈아타지 않는다")
st = {'snapshots': staggered(base)}
run_anchor(st, base)
first = sched(st)
st.pop('rebalancing_needed', None)
run_anchor(st, base)
check("두 번째 실행은 무발화", not st.get('rebalancing_needed'), str(sched(st)))
check("시간표 동일", sched(st) == first, f"{sched(st)} != {first}")

print("\n[13] 회전한 정상 시간표에서 하나만 잃어도 나머지를 안 끌고 간다")
# 돌아가는 중인 시간표: snap2 가 가장 이르고 snap0·snap1 은 이미 발화해 69일 건너뛴 상태.
rot = {0: _dt.date(2026, 10, 8), 1: _dt.date(2026, 10, 31), 2: _dt.date(2026, 9, 15)}
check("전제: 회전 상태도 23일 간격이다",
      ex._schedule_phase_ok(list(rot.values())))
for lost in range(ex.N_SNAPS):
    st = {'snapshots': {str(i): {'picks': ['VNQ'], 'weights': {'VNQ': 1.0},
                                 'next_rebal_date': rot[i].isoformat()}
                        for i in range(ex.N_SNAPS)}}
    del st['snapshots'][str(lost)]['next_rebal_date']
    run_anchor(st, TODAY)
    after = sched(st)
    want = [rot[i].isoformat() for i in range(ex.N_SNAPS)]
    check(f"snap{lost} 만 되살리고 나머지는 그대로", after == want, f"{after} != {want}")

print("\n[14] 마지막 발화일이 스태거돼 있으면 각자 자기 발화일 + 69일")
lasts = {0: _dt.date(2026, 8, 1), 1: _dt.date(2026, 8, 24), 2: _dt.date(2026, 9, 16)}
st = {'snapshots': {str(i): {'picks': ['VNQ'], 'weights': {'VNQ': 1.0},
                             'last_rebal_date': lasts[i].isoformat()}
                    for i in range(ex.N_SNAPS)}}
run_anchor(st, TODAY)
after = sched(st)
want = [(lasts[i] + _dt.timedelta(days=P)).isoformat() for i in range(ex.N_SNAPS)]
check("각 snap 이 자기 보유기간을 지킨다", after == want, f"{after} != {want}")

print("\n[15] 스냅샷 키가 통째로 없는 것도 결함이다")
st = {'snapshots': {'0': {'picks': ['VNQ'], 'weights': {'VNQ': 1.0}},
                    '1': {'picks': ['EEM'], 'weights': {'EEM': 1.0}}}}
check("결함으로 잡는다", ex.snapshot_payload_defect(st) is not None,
      str(ex.snapshot_payload_defect(st)))
check("온전하면 결함 아님",
      ex.snapshot_payload_defect({'snapshots': staggered(base)}) is None)

print("\n[16] 예정일끼리 위상이 갈리면 예정일을 믿지 않는다")
# snap0=09-15, snap1=09-20 — 23일 간격도 아니고 69일 배수도 아니다(위상 1:1).
lasts = {0: _dt.date(2026, 8, 1), 1: _dt.date(2026, 8, 24), 2: _dt.date(2026, 9, 16)}
st = {'snapshots': {'0': {'picks': ['VNQ'], 'weights': {'VNQ': 1.0},
                          'next_rebal_date': '2026-09-15',
                          'last_rebal_date': lasts[0].isoformat()},
                    '1': {'picks': ['VNQ'], 'weights': {'VNQ': 1.0},
                          'next_rebal_date': '2026-09-20',
                          'last_rebal_date': lasts[1].isoformat()},
                    '2': {'picks': ['VNQ'], 'weights': {'VNQ': 1.0},
                          'last_rebal_date': lasts[2].isoformat()}}}
run_anchor(st, TODAY)
after = sched(st)
want = [(lasts[i] + _dt.timedelta(days=P)).isoformat() for i in range(ex.N_SNAPS)]
check("마지막 발화일 쪽을 쓴다", after == want, f"{after} != {want}")
check("정비를 알린다", len(SENT) == 1 and '발화일' in SENT[0], str(SENT))

# 마지막 발화일도 없으면 예정일로 세우되 근거가 약하다고 알린다.
st = {'snapshots': {'0': {'picks': ['VNQ'], 'weights': {'VNQ': 1.0},
                          'next_rebal_date': '2026-09-15'},
                    '1': {'picks': ['VNQ'], 'weights': {'VNQ': 1.0},
                          'next_rebal_date': '2026-09-20'},
                    '2': {'picks': ['VNQ'], 'weights': {'VNQ': 1.0}}}}
run_anchor(st, TODAY)
after = sched(st)
check("그래도 23일 간격은 만든다", gaps(after) == [G] * (ex.N_SNAPS - 1), str(after))
check("근거가 약하다고 알린다", len(SENT) == 1 and '위상이 갈려' in SENT[0], str(SENT))

print("\n[17] 구조 검사 — 모양이 틀린 것만 잡고 정상은 통과")
check("정상 통과", ex.state_structure_defect({'snapshots': staggered(base)}) is None)
check("상태에 snapshots 가 없으면 통과", ex.state_structure_defect({}) is None)
check("빈 snapshots 는 통과(콜드스타트)", ex.state_structure_defect({'snapshots': {}}) is None)
for _bad in ([], [1], 'x', 3):
    check(f"snapshots 가 {type(_bad).__name__} 이면 결함",
          'dict 가 아니다' in (ex.state_structure_defect({'snapshots': _bad}) or ''),
          str(ex.state_structure_defect({'snapshots': _bad})))
check("잉여 키는 결함",
      '남아돈다' in (ex.state_structure_defect(
          {'snapshots': dict(staggered(base), **{'3': {}})}) or ''))
check("snap 이 dict 가 아니면 결함",
      'dict 가 아니다' in (ex.state_structure_defect(
          {'snapshots': dict(staggered(base), **{'0': []})}) or ''))

print("\n[12] 회귀 — 발화일 기준으로 다음 앵커를 잡지 않는다")
src = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '..', 'trade', 'executor_stock.py')).read()
check("고정 시간표 전진 규칙이 있다", 'while next_date <= today:' in src)
check("next_rebal_date 를 상태로 들고 다닌다", "snap['next_rebal_date'] = next_date.isoformat()" in src)
check("위상 불변식 검사가 있다", 'def _schedule_phase_ok(' in src)
check("신호 종목·비중 정합 가드가 세 곳에 있다",
      src.count('_picks_weights_ok(') >= 4)
check("스냅샷 결함 판정이 있다", 'def snapshot_payload_defect(' in src)
check("생존 예정일 복원 규칙이 있다", 'def _resolve_from_dates(' in src)
check("결함 검사가 앵커 체크보다 먼저다",
      src.index('_defect = snapshot_payload_defect(state)') < src.index('    check_anchors(signal, state)'))

print()
if FAILED:
    print(f"{len(FAILED)} FAIL: {FAILED}")
    sys.exit(1)
print("ALL PASS")
