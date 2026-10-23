#!/usr/bin/env python3
"""밤 조회 감시의 알림 규칙 — 몇 번 실패해야 사람을 깨우나, 언제 해소라고 하나.

감시 자체는 읽기 전용이라 틀려도 돈이 안 움직인다. 틀리면 생기는 손해는 둘이다.
밤새 같은 알림이 열한 번 쌓이거나, 반대로 조회가 죽었는데 아무 말도 없거나.
그 둘만 고정한다.

실행: python3 tests/test_balance_read_watch.py  (bash run_tests.sh 에 자동 포함)
"""
import json
import os
import sys
import tempfile

_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
sys.path.insert(0, os.path.join(_root, 'trade'))
sys.path.insert(0, os.path.join(_root, 'trade', 'ops'))
import balance_read_watch as w  # noqa: E402
import executor_stock as ex  # noqa: E402

FAILED = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ''))
    if not cond:
        FAILED.append(name)


class _FakeAPI:
    """부를 때마다 주어진 결과를 하나씩 돌려준다."""
    seq = []

    def __init__(self, dry_run=False):
        pass

    def get_balance(self):
        v = _FakeAPI.seq.pop(0)
        if isinstance(v, Exception):
            raise v
        return v


OK = ({'BIL': 1534, 'PDBC': 7041}, 304152.0, 1324.6)


def run(seq):
    """회차를 차례로 돌리고 (반환코드 목록, 보낸 알림 목록) 을 준다."""
    tmp = tempfile.mkdtemp()
    streak_path = os.path.join(tmp, 'streak.json')
    old = (w.STREAK_FILE, ex.KISAPI, ex.send_telegram, ex.log)
    alerts, rcs = [], []
    try:
        w.STREAK_FILE = streak_path
        ex.KISAPI = _FakeAPI
        ex.send_telegram = lambda m: alerts.append(m)
        _FakeAPI.seq = list(seq)
        for _ in seq:
            rcs.append(w.main())
    finally:
        w.STREAK_FILE, ex.KISAPI, ex.send_telegram, ex.log = old
    return rcs, alerts, streak_path


BAD = ex.BalanceUnavailable('환율을 어느 쪽에서도 못 받았다')

print(f"[설정] 연속 실패 {w.FAIL_ALERT_STREAK}회에서 알린다")

print("\n[1] 계속 정상이면 아무 말도 안 한다")
rcs, alerts, _ = run([OK, OK, OK])
check("반환코드 전부 0", rcs == [0, 0, 0], str(rcs))
check("알림 없음", alerts == [], str(alerts))

print("\n[2] 한 번 실패로는 안 깨운다 — 조회 실패는 일시적인 적이 있다")
rcs, alerts, _ = run([BAD])
check("반환코드 1", rcs == [1], str(rcs))
check("알림 없음", alerts == [], str(alerts))

print("\n[3] 연속 두 번이면 한 번만 알린다")
rcs, alerts, _ = run([BAD, BAD])
check("알림 한 건", len(alerts) == 1, str(alerts))
check("실패 횟수를 적는다", '2회 연속' in alerts[0] if alerts else False, str(alerts))

print("\n[4] 밤새 실패해도 알림은 그 한 건뿐이다")
rcs, alerts, _ = run([BAD] * 11)
check("알림 한 건", len(alerts) == 1, f'{len(alerts)}건 {alerts}')

print("\n[5] 다시 되면 해소를 한 번 알린다")
rcs, alerts, _ = run([BAD, BAD, OK])
check("알림 두 건", len(alerts) == 2, str(alerts))
check("둘째가 해소", '다시 된다' in alerts[1] if len(alerts) > 1 else False, str(alerts))
check("마지막 반환코드 0", rcs[-1] == 0, str(rcs))

print("\n[6] 알리기 전에 회복하면 해소 알림도 없다")
rcs, alerts, _ = run([BAD, OK])
check("알림 없음", alerts == [], str(alerts))

print("\n[7] 회복 뒤 다시 실패하면 처음부터 센다")
rcs, alerts, _ = run([BAD, OK, BAD])
check("알림 없음", alerts == [], str(alerts))

print("\n[8] 연속 횟수를 파일에 남긴다 — 회차가 따로 돌아도 이어 센다")
rcs, alerts, path = run([BAD, BAD])
with open(path) as f:
    saved = json.load(f)
check("fail_streak 2", saved.get('fail_streak') == 2, str(saved))
check("시각을 적는다", bool(saved.get('updated_at')), str(saved))

print("\n[9] 조회 실패가 아닌 예외도 같은 규칙으로 센다")
rcs, alerts, _ = run([RuntimeError('연결 끊김'), RuntimeError('연결 끊김')])
check("알림 한 건", len(alerts) == 1, str(alerts))
check("반환코드 1", rcs == [1, 1], str(rcs))

print()
if FAILED:
    print(f"❌ {len(FAILED)}건 실패: {FAILED}")
    sys.exit(1)
print("✅ 전부 통과")
