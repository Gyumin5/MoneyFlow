"""선물 본 주문이 결과 불명인 주문을 다시 보내지 않는지 검증 (2026-09-27).

문제: create_order_with_retry 가 timeout·timed out·internal error·connection 을
재시도 대상으로 보고 같은 주문을 최대 3회 다시 보냈다. 이 오류들은 "거래소가 못 받았다"
가 아니라 "응답을 못 받았다" 다. 체결된 주문을 다시 보내면 같은 수량이 두 번 나간다.
reduceOnly 도 중복 체결을 막지 않는다 — 두 건 모두 포지션을 더 줄인다.

1차 수정은 이름표로 조회해서 "그런 주문 없다"(-2013) 면 다시 보내게 했다. 배포 토론
run-20260927T014854Z 가 그것을 C 판정으로 막았다: 거래소 반영이 늦으면 방금 받은 주문에도
없다고 답할 수 있고, 같은 이름표가 이미 체결된 시장가 주문의 재접수를 막는다는 보장도
문서에 없다. 그래서 없다는 답으로는 재전송하지 않는다.

지금 계약:
  - 거래소가 실행 전에 거절했다고 답한 경우(요청 한도 초과·recvWindow 이탈)만 재전송.
  - 그 밖의 일시 오류는 결과 불명. 읽기전용으로 접수·체결을 판정한다.
      체결 완료      → 그 주문을 결과로 돌려준다.
      거절·만료·미체결 → OrderNotCompleted. 재전송 안 함.
      확정 못 함      → OrderAcceptanceUnknown. 재전송 안 하고 그 실행의 남은 매매도 멈춘다.
  - 찾은 주문도 status 와 executedQty 를 본다. 거절·만료·미체결을 완료로 취급하지 않는다.

네트워크·라이브 state 미사용 (fake client).
실행: cd /home/gmoh/mon/251229 && python3 tests/test_fut_order_idempotency.py
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'trade'))
import auto_trade_binance as a  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILS.append(name)


def api_exc(code, msg):
    return a.BinanceAPIException(type('R', (), {'text': '', 'status_code': 400})(),
                                 400, '{"code":%d,"msg":"%s"}' % (code, msg))


NOT_EXIST = api_exc(-2013, 'Order does not exist')
TIMEOUT = api_exc(-1007, 'Timeout waiting for response from backend server')
INTERNAL = api_exc(-1001, 'Internal error; unable to process your request. Please try again')
RATE_LIMIT = api_exc(-1003, 'Too many requests; current limit is 2400 request weight per minute')
RECV_WINDOW = api_exc(-1021, 'Timestamp for this request is outside of the recvWindow')
MARGIN = api_exc(-2019, 'Margin is insufficient')

PARAMS = {'symbol': 'SOLUSDT', 'side': 'SELL', 'type': 'MARKET', 'quantity': '12.3'}
FILLED = {'orderId': 77, 'status': 'FILLED', 'executedQty': '12.3'}


class FakeClient:
    """send_results·lookup_results 를 앞에서부터 하나씩 쓴다. 예외면 raise."""

    def __init__(self, send_results, lookup_results=None):
        self.send_results = list(send_results)
        self.lookup_results = list(lookup_results or [])
        self.sent = []
        self.looked_up = []

    def futures_create_order(self, **params):
        self.sent.append(params)
        r = self.send_results.pop(0) if self.send_results else {'status': 'FILLED'}
        if isinstance(r, Exception):
            raise r
        return r

    def futures_get_order(self, **params):
        self.looked_up.append(params)
        r = self.lookup_results.pop(0) if self.lookup_results else NOT_EXIST
        if isinstance(r, Exception):
            raise r
        return r


# 재시도 대기를 없앤다. 이 시험이 재는 것은 대기 시간이 아니라 몇 번 보내는가다.
a.ORDER_RETRY_DELAYS = [0.0, 0.0, 0.0]
a.ORDER_LOOKUP_DELAYS = [0.0, 0.0]

print("== 정상 주문은 한 번만 나가고 이름표가 붙는다 ==")
c = FakeClient([{'orderId': 1, 'status': 'NEW'}])
out = a.create_order_with_retry(c, PARAMS)
check("주문 1건", len(c.sent) == 1, str(len(c.sent)))
check("조회 안 함", c.looked_up == [], str(c.looked_up))
check("결과 그대로 반환", out.get('orderId') == 1, str(out))
check("정상 응답에는 상태 검사를 걸지 않는다(시장가는 NEW 로 먼저 온다)",
      out.get('status') == 'NEW', str(out))
cid = c.sent[0].get('newClientOrderId')
check("이름표가 붙는다", bool(cid), str(c.sent[0]))
check("이름표가 거래소 규격을 지킨다",
      bool(cid) and len(cid) <= 36 and re.fullmatch(r'[.A-Za-z0-9_:/-]+', cid), repr(cid))
check("원래 파라미터를 건드리지 않는다", 'newClientOrderId' not in PARAMS, str(PARAMS))
check("수량·방향은 그대로", c.sent[0]['quantity'] == '12.3' and c.sent[0]['side'] == 'SELL',
      str(c.sent[0]))

print("\n== 응답만 유실됐는데 거래소에 체결로 남아 있다 → 그 주문을 결과로 쓴다 ==")
c = FakeClient([TIMEOUT], [FILLED])
out = a.create_order_with_retry(c, PARAMS)
check("주문은 1건에서 멈춘다", len(c.sent) == 1, str(len(c.sent)))
check("조회 1회", len(c.looked_up) == 1, str(c.looked_up))
check("조회는 그 심볼·그 이름표로", (c.looked_up[0].get('symbol') == 'SOLUSDT'
      and c.looked_up[0].get('origClientOrderId') == c.sent[0]['newClientOrderId']),
      str(c.looked_up[0]))
check("찾은 주문을 결과로 돌려준다", out.get('orderId') == 77, str(out))

print("\n== 거래소가 없다고 답해도 다시 보내지 않는다 (토론 C 판정의 핵심) ==")
for label, lookups in [
    ("-2013 한 번", [NOT_EXIST]),
    ("-2013 여러 번", [NOT_EXIST, NOT_EXIST, NOT_EXIST]),
]:
    c = FakeClient([TIMEOUT, FILLED], lookups)
    try:
        a.create_order_with_retry(c, PARAMS)
        check(f"{label}: 접수 불명 예외", False, '예외가 안 났다')
    except a.OrderAcceptanceUnknown as e:
        check(f"{label}: 접수 불명 예외", True)
        check(f"{label}: 사유에 반영 지연을 밝힌다", '반영 지연' in str(e), str(e))
    except Exception as e:
        check(f"{label}: 접수 불명 예외", False, f'다른 예외: {type(e).__name__} {e}')
    check(f"{label}: 주문은 1건에서 멈춘다", len(c.sent) == 1, str(len(c.sent)))
    check(f"{label}: 조회를 정해진 횟수만큼 한다",
          len(c.looked_up) == a.ORDER_LOOKUP_ATTEMPTS, str(len(c.looked_up)))

print("\n== 조회 자체가 안 돼도 다시 보내지 않는다 ==")
net = RuntimeError('connection aborted')
c = FakeClient([TIMEOUT, FILLED], [net, net, net])
try:
    a.create_order_with_retry(c, PARAMS)
    check("접수 불명 예외", False, '예외가 안 났다')
except a.OrderAcceptanceUnknown as e:
    check("접수 불명 예외", True)
    check("사유에 심볼이 남는다", 'SOLUSDT' in str(e), str(e))
except Exception as e:
    check("접수 불명 예외", False, f'다른 예외: {type(e).__name__} {e}')
check("주문은 1건에서 멈춘다", len(c.sent) == 1, str(len(c.sent)))

print("\n== 내부 오류도 결과 불명으로 본다 ==")
c = FakeClient([INTERNAL, FILLED], [NOT_EXIST, NOT_EXIST, NOT_EXIST])
try:
    a.create_order_with_retry(c, PARAMS)
    check("접수 불명 예외", False, '예외가 안 났다')
except a.OrderAcceptanceUnknown:
    check("접수 불명 예외", True)
check("주문은 1건에서 멈춘다", len(c.sent) == 1, str(len(c.sent)))

print("\n== 찾은 주문이 체결로 끝나지 않았으면 완료로 취급하지 않는다 ==")
cases = [
    ("거절", {'orderId': 3, 'status': 'REJECTED', 'executedQty': '0'}),
    ("만료", {'orderId': 4, 'status': 'EXPIRED', 'executedQty': '0'}),
    ("취소", {'orderId': 5, 'status': 'CANCELED', 'executedQty': '0'}),
    ("부분체결", {'orderId': 6, 'status': 'FILLED', 'executedQty': '4.0'}),
    ("상태 없음", {'orderId': 7, 'executedQty': '12.3'}),
]
for label, found in cases:
    c = FakeClient([TIMEOUT, FILLED], [found] * a.ORDER_LOOKUP_ATTEMPTS)
    try:
        a.create_order_with_retry(c, PARAMS)
        check(f"{label}: 완료로 안 본다", False, '예외가 안 났다')
    except a.OrderNotCompleted as e:
        check(f"{label}: 완료로 안 본다", True)
        check(f"{label}: 사유에 상태·체결량", 'status' in str(e) or '체결량' in str(e), str(e))
    except a.OrderAcceptanceUnknown:
        check(f"{label}: 완료로 안 본다", True)
    except Exception as e:
        check(f"{label}: 완료로 안 본다", False, f'다른 예외: {type(e).__name__} {e}')
    check(f"{label}: 재전송 없음", len(c.sent) == 1, str(len(c.sent)))

print("\n== 아직 미체결이면 다시 물어보고, 그 사이 체결되면 그 주문을 쓴다 ==")
c = FakeClient([TIMEOUT, FILLED],
               [{'orderId': 8, 'status': 'NEW', 'executedQty': '0'}, FILLED])
out = a.create_order_with_retry(c, PARAMS)
check("두 번째 조회에서 체결 확인", out.get('orderId') == 77, str(out))
check("재전송 없음", len(c.sent) == 1, str(len(c.sent)))
check("조회 2회", len(c.looked_up) == 2, str(len(c.looked_up)))

print("\n== 거래소가 실행 전에 거절했다고 답한 경우만 다시 보낸다 ==")
for label, exc in [("요청 한도 초과", RATE_LIMIT), ("recvWindow 이탈", RECV_WINDOW)]:
    c = FakeClient([exc, {'orderId': 9, 'status': 'NEW'}])
    out = a.create_order_with_retry(c, PARAMS)
    check(f"{label}: 재전송한다", len(c.sent) == 2, str(len(c.sent)))
    check(f"{label}: 같은 이름표",
          c.sent[0]['newClientOrderId'] == c.sent[1]['newClientOrderId'])
    check(f"{label}: 조회 안 함", c.looked_up == [], str(c.looked_up))
    check(f"{label}: 결과 반환", out.get('orderId') == 9, str(out))

print("\n== 실행 전 거절이 계속되면 상한에서 멈추고 올린다 ==")
c = FakeClient([RATE_LIMIT] * a.ORDER_MAX_RETRIES)
try:
    a.create_order_with_retry(c, PARAMS)
    check("예외 전달", False, '예외가 안 났다')
except a.BinanceAPIException:
    check("예외 전달", True)
check(f"주문은 {a.ORDER_MAX_RETRIES}건을 넘지 않는다",
      len(c.sent) == a.ORDER_MAX_RETRIES, str(len(c.sent)))

print("\n== 되돌릴 수 없는 거절은 재시도도 조회도 없다 ==")
c = FakeClient([MARGIN])
try:
    a.create_order_with_retry(c, PARAMS)
    check("예외 전달", False, '예외가 안 났다')
except a.BinanceAPIException as e:
    check("예외 전달", True)
    check("거래소 코드 보존", getattr(e, 'code', None) == -2019, str(e))
check("주문 1건", len(c.sent) == 1, str(len(c.sent)))
check("조회 안 함", c.looked_up == [], str(c.looked_up))

print("\n== 심볼이 없으면 조회도 재전송도 못 한다 ==")
c = FakeClient([TIMEOUT])
try:
    a.create_order_with_retry(c, {'side': 'SELL', 'type': 'MARKET', 'quantity': '1'})
    check("심볼 없음 → 접수 불명", False, '예외가 안 났다')
except a.OrderAcceptanceUnknown:
    check("심볼 없음 → 접수 불명", True)
except Exception as e:
    check("심볼 없음 → 접수 불명", False, f'다른 예외: {type(e).__name__} {e}')
check("주문 1건", len(c.sent) == 1, str(len(c.sent)))

print("\n== 분류 함수 ==")
check("timeout 은 재시도 대상", a._is_retryable_order_error(TIMEOUT) is True)
check("timeout 은 실행 전 거절이 아니다", a._is_rejected_before_execution(TIMEOUT) is False)
check("내부 오류도 실행 전 거절이 아니다", a._is_rejected_before_execution(INTERNAL) is False)
check("요청 한도 초과는 실행 전 거절", a._is_rejected_before_execution(RATE_LIMIT) is True)
check("recvWindow 이탈은 실행 전 거절", a._is_rejected_before_execution(RECV_WINDOW) is True)
check("증거금 부족은 재시도 대상 아님", a._is_retryable_order_error(MARGIN) is False)

print("\n== 완료 판정 함수 ==")
check("체결 완료", a._order_completion(FILLED, 12.3)[0] is True)
check("체결량 부족", a._order_completion({'status': 'FILLED', 'executedQty': '1'}, 12.3)[0] is False)
check("요청 수량을 모르면 상태만 본다",
      a._order_completion({'status': 'FILLED', 'executedQty': '1'}, None)[0] is True)
check("미체결", a._order_completion({'status': 'NEW', 'executedQty': '0'}, 12.3)[0] is False)
check("빈 주문", a._order_completion({}, 12.3)[0] is False)
check("반올림 오차는 통과",
      a._order_completion({'status': 'FILLED', 'executedQty': '12.2999'}, 12.3)[0] is True)

print("\n== 이름표는 주문마다 다르다 ==")
ids = {a._new_client_order_id() for _ in range(50)}
check("50개 전부 서로 다르다", len(ids) == 50, str(len(ids)))
check("전부 규격 안", all(len(x) <= 36 and re.fullmatch(r'[.A-Za-z0-9_:/-]+', x) for x in ids))

print("\n== 호출부가 이름표를 직접 주면 그걸 쓴다 ==")
c = FakeClient([{'orderId': 5}])
a.create_order_with_retry(c, dict(PARAMS, newClientOrderId='mine-001'))
check("준 이름표 유지", c.sent[0]['newClientOrderId'] == 'mine-001', str(c.sent[0]))

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL PASS")
