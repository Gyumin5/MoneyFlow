#!/usr/bin/env python3
"""자산배분 알림 경로 검증 (2026-10-15).

고친 것 셋. ⑴ 24시간 중복 억제가 죽어 있었다 — 억제 여부를 적을 디스크 기록 조건이
영구히 거짓이라 last_alert_at 이 한 번도 저장되지 않았다. ⑵ 안 보낸 알림도 "발송" 이라고
리포트에 적었다 — 그 줄이 전송 블록 밖에 있었다. ⑶ 알림 머리말의 버전 표기가 낡았다.

고친 방향은 억제를 되살리는 게 아니라 걷어내는 것이다(토론 run-20261015T064950Z,
합의 1라운드). 이유 둘. send_telegram 이 실패를 삼켜서 "블록에 들어갔다" 를 전송 성공으로
쓸 수 없었고, 그 상태를 적을 자리인 trade_state.json 은 현물 실행기가 같이 쓰는 파일이라
통째 쓰기가 상대의 갱신을 덮을 수 있다. 크론이 하루 한 번이라 중복 피해는 사람이 손으로
다시 돌릴 때뿐이고, 그건 허용된 동작으로 둔다.

확인하는 것 여덟.
  T1 send_telegram 이 200 + 본문 ok 참이면 '접수' 를 돌려준다.
  T2 상태코드가 200 이 아니면 '실패' 다.
  T3 본문 ok 가 거짓이거나 본문이 JSON 이 아니면 각각 '실패'·'확인불가' 다.
  T4 시간 초과는 '확인불가', 연결 실패는 '실패' 다 (접수 여부를 아는 것과 모르는 것을 가른다).
  T5 알림 경로가 trade_state.json 을 읽지도 쓰지도 않는다.
  T6 억제 판정과 last_alert_at 기록이 남아 있지 않다.
  T7 리포트 줄이 전송 결과 셋을 각각 다르게 적는다 — 안 보낸 알림을 보냈다고 안 쓴다.
  T8 트리거 판정식·문턱·송금 제안 금액 계산은 안 바뀌었다 (보존 확인).

네트워크를 안 탄다. 전송 함수는 가짜 requests 로 바꿔 끼운다.
"""
import ast
import os
import sys

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
    시험이 그걸 끌고 오면 환경에 따라 실패하므로 AST 로 함수만 떼어 쓴다.
    """
    src = open(path, encoding='utf-8').read()
    tree = ast.parse(src)
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == name:
            return ast.get_source_segment(src, node), src
    raise SystemExit(f"{path} 에서 {name} 을 못 찾았다")


class FakeResp:
    def __init__(self, status, body=None, bad_json=False):
        self.status_code = status
        self._body = body
        self._bad = bad_json

    def json(self):
        if self._bad:
            raise ValueError('not json')
        return self._body


class FakeTimeout(Exception):
    pass


class FakeConnErr(Exception):
    pass


class FakeExceptions:
    Timeout = FakeTimeout


class FakeRequests:
    """post 가 미리 정한 것을 돌려주거나 던진다. 바깥으로 나가지 않는다."""
    exceptions = FakeExceptions

    def __init__(self, result):
        self.result = result
        self.calls = []

    def post(self, url, json=None, timeout=None):
        self.calls.append((url, json, timeout))
        if isinstance(self.result, Exception):
            raise self.result
        return self.result


def load_send(result):
    srcfn, _ = grab(PERSONAL, 'send_telegram')
    fake = FakeRequests(result)
    ns = {'requests': fake, 'TELEGRAM_CHAT_ID': 'x', 'TELEGRAM_BOT_TOKEN': 'y'}
    exec(compile(srcfn, PERSONAL, 'exec'), ns)
    return ns['send_telegram'], fake


def alert_block(src):
    """자산배분 알림 블록 본문만 떼어 온다 (주석 포함 원문 조각)."""
    start = src.index('# ── 자산배분 트리거 알림')
    end = src.index('fire_reason = []', start)
    return src[start:end]


def main():
    print("자산배분 알림 경로 검증")
    src = open(PERSONAL, encoding='utf-8').read()

    # T1
    fn, fake = load_send(FakeResp(200, {'ok': True, 'result': {}}))
    r = fn('hi')
    check('T1 200 + ok 참 → 접수', r == '접수', f'got {r}')
    check('T1 실제로 한 번 보낸다', len(fake.calls) == 1, f'calls {len(fake.calls)}')

    # T2
    for code in (400, 403, 429, 500):
        fn, _ = load_send(FakeResp(code, {'ok': False}))
        r = fn('hi')
        check(f'T2 상태 {code} → 실패', r == '실패', f'got {r}')

    # T3
    fn, _ = load_send(FakeResp(200, {'ok': False, 'description': 'blocked'}))
    r = fn('hi')
    check('T3 본문 ok 거짓 → 실패', r == '실패', f'got {r}')
    fn, _ = load_send(FakeResp(200, None, bad_json=True))
    r = fn('hi')
    check('T3 본문이 JSON 이 아니면 → 확인불가', r == '확인불가', f'got {r}')
    fn, _ = load_send(FakeResp(200, {}))
    r = fn('hi')
    check('T3 본문에 ok 가 없으면 → 실패', r == '실패', f'got {r}')

    # T4
    fn, _ = load_send(FakeTimeout('timed out'))
    r = fn('hi')
    check('T4 시간 초과 → 확인불가', r == '확인불가', f'got {r}')
    fn, _ = load_send(FakeConnErr('refused'))
    r = fn('hi')
    check('T4 연결 실패 → 실패', r == '실패', f'got {r}')

    blk = alert_block(src)
    # 주석은 빼고 본다 — 왜 안 건드리는지 적은 설명문이 블록 안에 있다.
    code = '\n'.join(l for l in blk.splitlines() if not l.strip().startswith('#'))

    # T5
    check('T5 알림 블록이 trade_state.json 을 안 건드린다',
          'trade_state.json' not in code and '_ts_obj' not in code and '_ts_path' not in code)
    check('T5 알림 블록에 파일 쓰기가 없다',
          'os.replace' not in code and '_atomic_tmp' not in code and 'json.dump' not in code)
    check('T5 알림 블록에 파일 열기가 없다', 'open(' not in code)

    # T6
    check('T6 억제 판정이 없다',
          '_alert_ok' not in blk and '86400' not in blk)
    check('T6 last_alert_at 기록이 없다', 'last_alert_at' not in blk)
    check('T6 폐지된 자동 배분의 강제해제 분기가 없다',
          '_was_active' not in blk and 'FORCE CLEAR' not in blk and
          '_alloc_transit_active' not in blk)
    check('T6 쓰이지 않던 cap_ratio 계산이 없다', '_compute_cap_ratios' not in blk)

    # T7
    check('T7 전송 결과를 받아서 쓴다', "_tg = send_telegram(" in blk)
    for word, label in (("'접수'", '접수'), ("'확인불가'", '확인불가')):
        check(f'T7 {label} 갈래가 있다', word in blk)
    check('T7 전송 실패 문구가 따로 있다', '텔레그램 알림 전송 실패' in blk)
    check('T7 결과와 무관한 "발송" 문구가 없다', '알림 발송' not in blk)

    # T8 — 판정식·문턱·금액 계산 보존
    check('T8 문턱 상수가 그대로다',
          'REBAL_HT_THRESHOLD = 0.20' in src and 'REBAL_T3U_REL = 0.20' in src)
    for frag in ('t1_fire = ht >= REBAL_HT_THRESHOLD',
                 'fire = t1_fire or t3u_fire',
                 't3u_fire = t3u_stock or t3u_spot or t3u_fut'):
        check(f'T8 판정식 보존: {frag[:28]}', frag in src)
    for frag in ("tgt_stock_krw = alloc_total * STOCK_RATIO",
                 "tgt_spot_krw = alloc_total * COIN_RATIO",
                 "tgt_fut_krw = alloc_total * FUTURES_RATIO",
                 "'stock': stock_krw - tgt_stock_krw",
                 "abs(_v) > alloc_total * 0.005"):
        check(f'T8 송금 금액 계산 보존: {frag[:30]}', frag in blk)

    # T9 — 못 믿을 계좌를 0원으로 받지 않는다 (2026-10-16, ai-debate run-20261016T032730Z)
    guard_src, _ = grab(PERSONAL, 'alloc_acct_errors')
    _ns = {}
    exec(guard_src, _ns)
    acct_errors = _ns['alloc_acct_errors']

    def _a(v):
        return {"total_krw": v, "holdings": []}

    t9 = [
        ('정상 셋', {"stock_kis": _a(6e8), "coin_upbit": _a(2e8), "coin_binance": _a(1e8)}, 0),
        ('error 키', {"stock_kis": {"error": "timeout"}, "coin_upbit": _a(2e8), "coin_binance": _a(1e8)}, 1),
        ('계좌 키 자체가 없음', {"coin_upbit": _a(2e8), "coin_binance": _a(1e8)}, 1),
        ('빈 사전', {"stock_kis": {}, "coin_upbit": _a(2e8), "coin_binance": _a(1e8)}, 1),
        ('총액 0 (error 없음)', {"stock_kis": _a(0), "coin_upbit": _a(2e8), "coin_binance": _a(1e8)}, 1),
        ('총액 None', {"stock_kis": {"total_krw": None}, "coin_upbit": _a(2e8), "coin_binance": _a(1e8)}, 1),
        ('총액이 숫자가 아님', {"stock_kis": {"total_krw": "abc"}, "coin_upbit": _a(2e8), "coin_binance": _a(1e8)}, 1),
        ('셋 다 0', {"stock_kis": _a(0), "coin_upbit": _a(0), "coin_binance": _a(0)}, 3),
    ]
    for label, accounts, want in t9:
        got = acct_errors(accounts)
        check(f'T9 {label} → 실패 {want}건', len(got) == want, f"실제 {len(got)}건 {got}")

    check('T9 호출부가 이 함수를 쓴다', '_acct_errors = alloc_acct_errors(accts)' in src)
    check('T9 실패가 있으면 트리거 평가를 건너뛴다',
          'if _acct_errors:' in src and '트리거 평가 스킵' in src)

    print()
    if FAILED:
        print(f"실패 {len(FAILED)}건: {FAILED}")
        return 1
    print("전부 통과")
    return 0


if __name__ == '__main__':
    sys.exit(main())
