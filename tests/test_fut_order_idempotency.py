"""선물 본 주문 재전송이 중복 체결을 내지 않는지 검증 (2026-09-27).

문제: create_order_with_retry 가 timeout·timed out·internal error·connection 을
재시도 대상으로 보고 같은 주문을 최대 3회 다시 보냈다. 이 오류들은 "거래소가 못 받았다"
가 아니라 "응답을 못 받았다" 다. 체결된 주문을 다시 보내면 같은 수량이 두 번 나간다.
reduceOnly 도 중복 체결을 막지 않는다 — 두 건 모두 포지션을 더 줄인다.
사전매도에서 2026-09-26 에 없앤 것과 같은 결함이 본 주문 경로에 남아 있었다
(2026-09-27 배포 토론 run-20260927T010355Z 소수의견).

수정: 주문에 이름표(newClientOrderId)를 미리 붙이고, 재전송 직전에 그 이름표로
거래소에 물어본다. 답이 셋으로 갈린다.
  있다        → 그 주문을 결과로 돌려주고 재전송하지 않는다.
  확실히 없다 → 같은 이름표로 다시 보낸다(-2013 Order does not exist).
  모른다      → 보내지 않고 실패로 올린다(OrderAcceptanceUnknown).

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
MARGIN = api_exc(-2019, 'Margin is insufficient')

PARAMS = {'symbol': 'SOLUSDT', 'side': 'SELL', 'type': 'MARKET', 'quantity': '12.3'}


class FakeClient:
    """send_results 를 앞에서부터 하나씩 쓴다. 예외면 raise, 아니면 그 값을 반환.
    lookup_results 도 같다. 둘 다 호출 인자를 기록한다."""

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
c = FakeClient([{'orderId': 1, 'status': 'FILLED'}])
out = a.create_order_with_retry(c, PARAMS)
check("주문 1건", len(c.sent) == 1, str(len(c.sent)))
check("조회 안 함", c.looked_up == [], str(c.looked_up))
check("결과 그대로 반환", out.get('orderId') == 1, str(out))
cid = c.sent[0].get('newClientOrderId')
check("이름표가 붙는다", bool(cid), str(c.sent[0]))
check("이름표가 거래소 규격을 지킨다",
      bool(cid) and len(cid) <= 36 and re.fullmatch(r'[.A-Za-z0-9_:/-]+', cid), repr(cid))
check("원래 파라미터를 건드리지 않는다", 'newClientOrderId' not in PARAMS, str(PARAMS))
check("수량·방향은 그대로", c.sent[0]['quantity'] == '12.3' and c.sent[0]['side'] == 'SELL',
      str(c.sent[0]))

print("\n== 응답만 유실됐는데 거래소엔 있다 → 재전송하지 않는다 ==")
c = FakeClient([TIMEOUT], [{'orderId': 77, 'status': 'FILLED', 'executedQty': '12.3'}])
out = a.create_order_with_retry(c, PARAMS)
check("주문은 1건에서 멈춘다", len(c.sent) == 1, str(len(c.sent)))
check("조회 1회", len(c.looked_up) == 1, str(c.looked_up))
check("조회는 그 심볼·그 이름표로", (c.looked_up[0].get('symbol') == 'SOLUSDT'
      and c.looked_up[0].get('origClientOrderId') == c.sent[0]['newClientOrderId']),
      str(c.looked_up[0]))
check("찾은 주문을 결과로 돌려준다", out.get('orderId') == 77, str(out))

print("\n== 거래소가 확실히 못 받았다 → 같은 이름표로 다시 보낸다 ==")
c = FakeClient([TIMEOUT, {'orderId': 88, 'status': 'FILLED'}], [NOT_EXIST])
out = a.create_order_with_retry(c, PARAMS)
check("주문 2건", len(c.sent) == 2, str(len(c.sent)))
check("두 번 다 같은 이름표",
      c.sent[0]['newClientOrderId'] == c.sent[1]['newClientOrderId'],
      f"{c.sent[0]['newClientOrderId']} / {c.sent[1]['newClientOrderId']}")
check("두 번째가 성공하면 그 결과", out.get('orderId') == 88, str(out))

print("\n== 조회 자체가 안 되면 보내지 않는다 ==")
net = RuntimeError('connection aborted')
c = FakeClient([TIMEOUT], [net, net, net])
try:
    a.create_order_with_retry(c, PARAMS)
    check("접수 불명이면 예외", False, '예외가 안 났다')
except a.OrderAcceptanceUnknown as e:
    check("접수 불명이면 예외", True)
    check("사유에 심볼이 남는다", 'SOLUSDT' in str(e), str(e))
except Exception as e:
    check("접수 불명이면 예외", False, f'다른 예외: {type(e).__name__} {e}')
check("주문은 1건에서 멈춘다", len(c.sent) == 1, str(len(c.sent)))
check("조회는 정해진 횟수만큼 다시 묻는다",
      len(c.looked_up) == a.ORDER_LOOKUP_ATTEMPTS, str(len(c.looked_up)))

print("\n== 조회가 한 번 실패해도 다음에 답이 오면 그 답을 쓴다 ==")
c = FakeClient([TIMEOUT], [RuntimeError('connection reset'), NOT_EXIST])
c.send_results.append({'orderId': 99, 'status': 'FILLED'})
out = a.create_order_with_retry(c, PARAMS)
check("없다고 확인되면 다시 보낸다", len(c.sent) == 2, str(len(c.sent)))
check("결과 반환", out.get('orderId') == 99, str(out))

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

print("\n== 마지막 시도에서 실패하면 그대로 올린다 ==")
c = FakeClient([TIMEOUT] * a.ORDER_MAX_RETRIES, [NOT_EXIST] * a.ORDER_MAX_RETRIES)
try:
    a.create_order_with_retry(c, PARAMS)
    check("예외 전달", False, '예외가 안 났다')
except a.BinanceAPIException:
    check("예외 전달", True)
except a.OrderAcceptanceUnknown as e:
    check("예외 전달", False, f'접수 불명으로 잘못 분류: {e}')
check(f"주문은 {a.ORDER_MAX_RETRIES}건을 넘지 않는다",
      len(c.sent) == a.ORDER_MAX_RETRIES, str(len(c.sent)))

print("\n== 호출부가 이름표를 직접 주면 그걸 쓴다 ==")
c = FakeClient([{'orderId': 5}])
a.create_order_with_retry(c, dict(PARAMS, newClientOrderId='mine-001'))
check("준 이름표 유지", c.sent[0]['newClientOrderId'] == 'mine-001', str(c.sent[0]))

print("\n== 심볼이 없으면 재전송하지 않는다 ==")
c = FakeClient([TIMEOUT])
try:
    a.create_order_with_retry(c, {'side': 'SELL', 'type': 'MARKET', 'quantity': '1'})
    check("심볼 없음 → 예외", False, '예외가 안 났다')
except a.OrderAcceptanceUnknown:
    check("심볼 없음 → 예외", True)
except Exception as e:
    check("심볼 없음 → 예외", False, f'다른 예외: {type(e).__name__} {e}')
check("주문 1건", len(c.sent) == 1, str(len(c.sent)))

print("\n== 이름표는 주문마다 다르다 ==")
ids = set()
for _ in range(50):
    ids.add(a._new_client_order_id())
check("50개 전부 서로 다르다", len(ids) == 50, str(len(ids)))
check("전부 규격 안", all(len(x) <= 36 and re.fullmatch(r'[.A-Za-z0-9_:/-]+', x) for x in ids))

print("\n== 재시도 판정은 그대로 ==")
check("timeout 은 재시도 대상", a._is_retryable_order_error(TIMEOUT) is True)
check("증거금 부족은 아님", a._is_retryable_order_error(MARGIN) is False)

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL PASS")
