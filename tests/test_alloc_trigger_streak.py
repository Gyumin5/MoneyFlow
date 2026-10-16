#!/usr/bin/env python3
"""송금 트리거 지속일 검증 (2026-10-16).

무엇을 넣었나. 송금 트리거가 켜진 채로 며칠이 지났는지 세어, 길어질수록 텔레그램
알림 머리말을 세게 쓴다. 매매는 한 줄도 안 바뀐다 — 상한·보험은 안 올린다
(늦게 켜는 규칙 넷이 사전등록 판정에서 전부 탈락, history #69).

강도를 올리는 지점(5·20·60 영업일)은 측정한 착수 지연 팔과 같은 값이다.
20영업일까지의 최대낙폭 악화는 잡음 폭(7.280pp) 안이고 60영업일은 넘었다.

확인하는 것 아홉.
  T1 트리거가 꺼져 있으면 0영업일이고 시작일이 없다.
  T2 처음 켜진 날은 0영업일이고 그날을 시작일로 적는다.
  T3 이어서 켜져 있으면 달력이 아니라 영업일로 센다 (주말은 안 센다).
  T4 꺼지면 기록을 비우고, 다시 켜지면 그날부터 새로 센다.
  T5 파일이 없거나 깨졌어도 예외를 안 내고 오늘부터 센다.
  T6 저장이 실패해도 예외를 안 낸다 — 알림을 잃지 않는다.
  T7 미래 날짜가 적혀 있으면 오늘로 되돌린다 (음수 지속일 방지).
  T8 매매 상태파일(trade_state.json)을 쓰지 않는다. 전용 파일 하나만 쓴다.
  T9 알림 머리말이 지점마다 달라지고, 판정식·문턱·금액 계산은 그대로다.

네트워크·실계좌를 안 탄다. 파일은 임시 디렉토리에만 쓴다.
"""
import ast
import json
import os
import sys
import tempfile
from datetime import date, datetime

import numpy as np

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..')
PERSONAL = os.path.join(ROOT, 'strategies', 'cap_defend', 'recommend_personal.py')

FAILED = []


def check(name, cond, detail=''):
    mark = 'OK ' if cond else 'FAIL'
    print(f"  [{mark}] {name}" + (f" — {detail}" if detail and not cond else ''))
    if not cond:
        FAILED.append(name)


def grab(path, name):
    """모듈 전체를 import 하지 않고 함수 본문만 떼어 온다.

    recommend_personal.py 는 맨 위에서 config(비밀값)·pyupbit 를 import 한다.
    """
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            seg = ast.get_source_segment(src, node)
            if not seg:
                raise SystemExit(f"{path} 의 {name} 본문을 못 떼어 왔다")
            return seg, src
    raise SystemExit(f"{path} 에서 {name} 을 못 찾았다")


def load(fp, save=None):
    """지속일 함수를 떼어 와 임시 경로·가짜 저장기로 묶는다."""
    srcfn, _ = grab(PERSONAL, '_alloc_streak_update')
    saved = []

    def _save(obj, path):
        if save == 'raise':
            raise OSError('disk full')
        saved.append((dict(obj), path))
        tmp = path + '.tmp'
        with open(tmp, 'w') as f:
            json.dump(obj, f)
        os.replace(tmp, path)

    ns = {'json': json, 'np': np, 'datetime': datetime,
          'ALLOC_TRIGGER_FILE': fp, '_save_json_atomic': _save}
    exec(compile(srcfn, PERSONAL, 'exec'), ns)
    return ns['_alloc_streak_update'], saved


def read_since(fp):
    try:
        with open(fp) as f:
            return json.load(f).get('since')
    except Exception:
        return '<없음>'


def main():
    print("송금 트리거 지속일 검증")
    src = open(PERSONAL, encoding='utf-8').read()

    with tempfile.TemporaryDirectory() as d:
        fp = os.path.join(d, 'alloc_trigger_streak.json')

        # T1 — 꺼져 있음
        fn, _ = load(fp)
        bd, since = fn(False, date(2026, 10, 16))
        check('T1 꺼져 있으면 0영업일', bd == 0, f'got {bd}')
        check('T1 꺼져 있으면 시작일 없음', since is None, f'got {since}')
        check('T1 켜진 적 없으면 파일도 안 만든다', not os.path.exists(fp))

        # T2 — 처음 켜짐 (금요일)
        fn, saved = load(fp)
        bd, since = fn(True, date(2026, 10, 16))
        check('T2 첫날은 0영업일', bd == 0, f'got {bd}')
        check('T2 첫날을 시작일로 적는다', since == date(2026, 10, 16), f'got {since}')
        check('T2 파일에 남는다', read_since(fp) == '2026-10-16', f"got {read_since(fp)}")
        check('T2 저장은 한 번만', len(saved) == 1, f'got {len(saved)}')

        # T3 — 이어짐. 시작일이 영업일이면 그날이 지난 순간부터 1영업일로 센다
        # (np.busday_count 는 시작일 포함·오늘 제외). 10-16 금 시작 → 주말엔 1,
        # 10-19 월 = 1, 10-23 금 = 5.
        for day, want in ((date(2026, 10, 17), 1),   # 토 — 금요일 하루가 지났다
                          (date(2026, 10, 18), 1),   # 일
                          (date(2026, 10, 19), 1),   # 월
                          (date(2026, 10, 23), 5),   # 금
                          (date(2026, 11, 13), 20),
                          (date(2027, 1, 8), 60)):
            fn, saved = load(fp)
            bd, since = fn(True, day)
            check(f'T3 {day} → {want}영업일', bd == want, f'got {bd}')
            check(f'T3 {day} 시작일 보존', since == date(2026, 10, 16), f'got {since}')
            check(f'T3 {day} 같은 시작일이면 다시 안 쓴다', len(saved) == 0, f'got {len(saved)}')

        # T4 — 꺼졌다 다시 켜짐
        fn, saved = load(fp)
        bd, since = fn(False, date(2026, 10, 23))
        check('T4 꺼지면 0영업일', bd == 0 and since is None, f'got {bd}/{since}')
        check('T4 꺼지면 기록을 비운다', read_since(fp) is None, f"got {read_since(fp)}")
        check('T4 비울 때 한 번 쓴다', len(saved) == 1, f'got {len(saved)}')
        fn, _ = load(fp)
        bd, since = fn(True, date(2026, 10, 26))
        check('T4 다시 켜지면 그날부터', bd == 0 and since == date(2026, 10, 26), f'got {bd}/{since}')
        fn, _ = load(fp)
        bd, _ = fn(False, date(2026, 10, 27))
        check('T4 이미 비어 있으면 다시 안 쓴다', bd == 0)

        # T5 — 파일 없음 / 깨짐
        gone = os.path.join(d, 'none.json')
        fn, _ = load(gone)
        bd, since = fn(True, date(2026, 10, 16))
        check('T5 파일이 없어도 오늘부터', bd == 0 and since == date(2026, 10, 16), f'got {bd}/{since}')
        for label, body in (('깨진 JSON', '{not json'),
                            ('null', 'null'),
                            ('since 없음', '{"x": 1}'),
                            ('since 가 숫자', '{"since": 17}'),
                            ('since 형식 틀림', '{"since": "2026/10/16"}')):
            bad = os.path.join(d, 'bad.json')
            open(bad, 'w').write(body)
            fn, _ = load(bad)
            try:
                bd, since = fn(True, date(2026, 10, 20))
                ok = (bd == 0 and since == date(2026, 10, 20))
            except Exception as ex:
                ok = False
                bd, since = f'예외 {ex}', None
            check(f'T5 {label} → 오늘부터, 예외 없음', ok, f'got {bd}/{since}')

        # T6 — 저장 실패
        fn, _ = load(os.path.join(d, 'ro.json'), save='raise')
        try:
            bd, since = fn(True, date(2026, 10, 16))
            ok = (bd == 0 and since == date(2026, 10, 16))
        except Exception as ex:
            ok = False
            bd = f'예외 {ex}'
        check('T6 저장이 실패해도 지속일을 돌려준다', ok, f'got {bd}')

        # T7 — 미래 날짜
        fut = os.path.join(d, 'fut.json')
        open(fut, 'w').write('{"since": "2027-01-01"}')
        fn, _ = load(fut)
        bd, since = fn(True, date(2026, 10, 16))
        check('T7 미래 시작일은 오늘로 되돌린다', bd == 0 and since == date(2026, 10, 16), f'got {bd}/{since}')
        check('T7 되돌린 값을 적는다', read_since(fut) == '2026-10-16', f"got {read_since(fut)}")

    # T8 — 전용 파일 하나만 쓴다
    fnsrc, _ = grab(PERSONAL, '_alloc_streak_update')
    check('T8 trade_state.json 을 안 건드린다', 'trade_state.json' not in fnsrc)
    check('T8 signal_state.json 을 안 건드린다', 'signal_state.json' not in fnsrc)
    check('T8 쓰는 파일은 ALLOC_TRIGGER_FILE 뿐이다',
          fnsrc.count('_save_json_atomic(') == 2 and fnsrc.count('ALLOC_TRIGGER_FILE') == 3,
          f"save {fnsrc.count('_save_json_atomic(')} / file {fnsrc.count('ALLOC_TRIGGER_FILE')}")
    check('T8 전용 파일 상수가 선언돼 있다',
          "ALLOC_TRIGGER_FILE = os.path.join(\".\", \"alloc_trigger_streak.json\")" in src)
    check('T8 원자적 저장을 쓴다', '_save_json_atomic' in fnsrc and 'open(' in fnsrc)

    # T9 — 알림 머리말과 보존
    check('T9 강도 지점이 5·20·60 영업일이다', 'ALLOC_ESCALATE_BD = (5, 20, 60)' in src)
    check('T9 호출부가 지속일을 받는다',
          '_alloc_bd, _alloc_since = _alloc_streak_update(fire, datetime.now().date())' in src)
    for frag in ('_b5, _b20, _b60 = ALLOC_ESCALATE_BD',
                 'if _alloc_bd >= _b60:',
                 'elif _alloc_bd >= _b20:',
                 'elif _alloc_bd >= _b5:',
                 '_head = "⚠️ 자산배분 트리거 켜짐"'):
        check(f'T9 머리말 분기: {frag[:34]}', frag in src)
    check('T9 지속 줄은 5영업일부터', 'if _alloc_bd >= _b5 and _alloc_since is not None:' in src)
    check('T9 측정 근거 줄은 20영업일부터', 'if _alloc_bd >= _b20:' in src)
    check('T9 지속일이 매매에 안 쓰인다',
          '_alloc_bd' not in src.split('fire_reason = []')[1])
    for frag in ('t1_fire = ht >= REBAL_HT_THRESHOLD',
                 'fire = t1_fire or t3u_fire',
                 'REBAL_HT_THRESHOLD = 0.20',
                 'REBAL_T3U_REL = 0.20',
                 'tgt_stock_krw = alloc_total * STOCK_RATIO'):
        check(f'T9 보존: {frag[:32]}', frag in src)

    print()
    if FAILED:
        print(f"실패 {len(FAILED)}건: {FAILED}")
        return 1
    print("전부 통과")
    return 0


if __name__ == '__main__':
    sys.exit(main())
