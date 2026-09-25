#!/usr/bin/env python3
"""전량청산 수량과 L 낮춤 사전매도가 계좌를 두 번 건드리지 않는지 확인한다.

ai-debate run-20260925T144408Z 가 정한 세 가지를 시험한다.
  (a) 전량청산은 그 심볼의 거래소 원문 수량으로만 나간다. 원문이 없으면 주문을 만들지
      않고 축소 실패로 표시한다 — 다른 심볼의 수량을 빌려 쓰면 남의 포지션 크기로
      판다(변수 재사용 결함).
  (b) 사전매도 실패는 "거래소가 접수하지 않았음" 이 포지션으로 확인될 때만 재시도한다.
      reduceOnly 는 중복 체결을 막지 않는다 — 두 번 나가면 두 번 줄어든다.
      거래정지·상장폐지면 그날 선물 매매 전체를 사유와 함께 차단한다.
  (c) 심볼 규격 조회 실패(MarketMetaUnavailable)는 사전매도 함수 안에서 끝내고 사유를
      돌려준다. 밖으로 올리면 호출부의 중단 기록·알림을 건너뛴 채 프로세스가 죽는다.
그리고 포지션·시세 조회 실패가 조용한 성공으로 끝나지 않는지 같이 본다.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trade'))
import auto_trade_binance as a  # noqa: E402

FAILED = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ''))
    if not cond:
        FAILED.append(name)


_KEEP = object()      # qty_raw=None 자체가 시험 대상이라 기본값과 구분한다


def pos(coin, notional, qty, qty_raw=_KEEP):
    return {'symbol': coin + 'USDT', 'notional': notional, 'qty': qty,
            'qty_raw': str(qty) if qty_raw is _KEEP else qty_raw,
            'weight': 0.1, 'pnl': 0.0, 'leverage': 3}


class Client:
    """주문을 통째로 받아 적는다. fail_on 에 든 심볼은 지정한 예외를 던진다."""

    def __init__(self, fail_on=None, price_fails=False):
        self.fail_on = fail_on or {}
        self.orders = []
        self.price_fails = price_fails

    def futures_create_order(self, **kw):
        self.orders.append(dict(kw))
        sym = kw['symbol']
        if sym in self.fail_on:
            raise self.fail_on[sym]
        return {'status': 'NEW', 'orderId': len(self.orders)}

    def futures_symbol_ticker(self, symbol=None):
        if self.price_fails:
            raise RuntimeError('ticker down')
        return {'symbol': symbol, 'price': '100.0'}

    def symbols(self):
        return [o['symbol'] for o in self.orders]

    def qty_of(self, symbol):
        return [o['quantity'] for o in self.orders if o['symbol'] == symbol]


def run_rebalance(positions, target, liquidation_only, listed=('AAAUSDT', 'BBBUSDT', 'CCCUSDT'),
                  positions_ok=True, price_fails=False, fail_on=None):
    """execute_rebalance 한 번. 전량청산은 원문 수량을 쓰므로 format_quantity 를 안 탄다."""
    saved = {k: getattr(a, k) for k in
             ('get_current_positions', 'cancel_stop_orders', '_trading_symbols',
              'get_symbol_constraints', 'format_quantity', 'UNIVERSE',
              '_v25_persist_abort_log')}
    errors, orders = [], []
    c = Client(fail_on, price_fails=price_fails)
    aborts = []
    try:
        a.get_current_positions = lambda _c: (positions, 1000.0, positions_ok)
        a.cancel_stop_orders = lambda _c, _s=None: None
        a._trading_symbols = lambda _c: set(listed)
        a.get_symbol_constraints = lambda _c, s: {
            'step_size': 0.001, 'min_qty': 0.0, 'min_notional': 5.0,
            'tick_size': 0.01, 'qty_precision': 3}
        a.format_quantity = lambda _c, s, q: f"{q:.3f}"
        a.UNIVERSE = list(listed)
        a._v25_persist_abort_log = lambda m: aborts.append(m)
        a._v25_touch_reset()
        a.execute_rebalance(c, target, 1000.0, {},
                            order_alerts=orders, error_alerts=errors,
                            liquidation_only=liquidation_only)
    finally:
        for k, v in saved.items():
            setattr(a, k, v)
    return c, errors, orders, aborts


print("[1] 전량청산은 심볼마다 자기 원문 수량으로 나간다")
# 소수 자릿수가 서로 다르고 format_quantity(3자리)와도 다른 값을 일부러 준다.
HELD2 = {'AAA': pos('AAA', 500.0, 10.12345678, '10.12345678'),
         'BBB': pos('BBB', 400.0, 20.5, '20.5')}
c, errors, orders, aborts = run_rebalance(HELD2, {'CASH': 1.0}, True)
check("두 심볼 다 주문이 나갔다", set(c.symbols()) == {'AAAUSDT', 'BBBUSDT'}, str(c.orders))
check("AAA 는 자기 원문 수량", c.qty_of('AAAUSDT') == ['10.12345678'], str(c.orders))
check("BBB 는 자기 원문 수량", c.qty_of('BBBUSDT') == ['20.5'], str(c.orders))
check("남의 수량을 빌려 쓰지 않는다", c.qty_of('BBBUSDT') != ['10.12345678'], str(c.orders))
check("수량을 내림하지 않는다(전량이 남지 않게)",
      all(q not in ('10.123', '20.500') for q in c.qty_of('AAAUSDT') + c.qty_of('BBBUSDT')),
      str(c.orders))
check("전부 reduceOnly 매도", all(o.get('reduceOnly') == 'true' and o['side'] == 'SELL'
                                  for o in c.orders), str(c.orders))
check("오류 없음", errors == [], str(errors))

print("\n[1b] 숏 포지션의 음수 원문도 부호만 떼어 그대로 쓴다")
c, errors, orders, aborts = run_rebalance(
    {'AAA': pos('AAA', 500.0, -10.12345678, '-10.12345678')}, {'CASH': 1.0}, True)
check("절대값 원문으로 나간다", c.qty_of('AAAUSDT') == ['10.12345678'], str(c.orders))

print("\n[2] 원문 수량이 없으면 그 심볼은 주문을 만들지 않는다")
for bad in (None, '', '   ', '0', '-0', 'abc', 0.0):
    held = {'AAA': pos('AAA', 500.0, 10.0, bad), 'BBB': pos('BBB', 400.0, 20.5, '20.5')}
    c, errors, orders, aborts = run_rebalance(held, {'CASH': 1.0}, True)
    check(f"원문 {bad!r} → AAA 주문 0건",
          'AAAUSDT' not in c.symbols(), str(c.orders))
    check(f"원문 {bad!r} → 차단 사유가 오류로 올라간다",
          any('FULL CLOSE BLOCKED' in m and 'AAAUSDT' in m for m in errors), str(errors))
    check(f"원문 {bad!r} → 정상 심볼은 그대로 나간다",
          'BBBUSDT' in c.symbols(), str(c.orders))

print("\n[2b] 원문 누락은 축소 실패라 뒤따르는 매수를 막는다")
c, errors, orders, aborts = run_rebalance(
    {'AAA': pos('AAA', 500.0, 10.0, '')},
    {'CCC': 0.5, 'CASH': 0.5}, False)
check("청산 주문을 안 보냈다", 'AAAUSDT' not in c.symbols(), str(c.orders))
check("매수도 안 보냈다", 'CCCUSDT' not in c.symbols(), str(c.orders))
check("매수 보류가 기록된다", a._skipped_buys == ['CCCUSDT'], str(a._skipped_buys))
check("차단 사유가 오류로 올라간다",
      any('FULL CLOSE BLOCKED' in m for m in errors), str(errors))

print("\n[2c] 대조 — 원문이 정상이면 매수는 그대로 나간다")
c, errors, orders, aborts = run_rebalance(
    {'AAA': pos('AAA', 500.0, 10.0, '10.0')},
    {'CCC': 0.5, 'CASH': 0.5}, False)
check("청산·매수 둘 다 나간다", {'AAAUSDT', 'CCCUSDT'} <= set(c.symbols()), str(c.orders))
check("오류 없음", errors == [], str(errors))

print("\n[3] 포지션 조회 실패는 조용한 성공으로 끝나지 않는다")
c, errors, orders, aborts = run_rebalance(
    HELD2, {'CASH': 1.0}, True, positions_ok=False)
check("주문 0건", c.orders == [], str(c.orders))
check("중단으로 올라간다",
      any(m.startswith('V25 ABORT: 포지션 조회 실패') for m in errors), str(errors))
check("중단 기록도 남는다", any('포지션 조회 실패' in m for m in aborts), str(aborts))

print("\n[4] 매수 계획의 시세·규격 조회 실패가 묻히지 않는다")
c, errors, orders, aborts = run_rebalance({}, {'CCC': 0.5, 'CASH': 0.5}, False, price_fails=True)
check("주문 0건", c.orders == [], str(c.orders))
check("계획 실패가 오류로 올라간다",
      any(m.startswith('BUY PLAN FAILED') and 'CCCUSDT' in m for m in errors), str(errors))


# ---------------------------------------------------------------- 사전매도
class PresellClient:
    """사전매도용. 주문을 받아 적고 포지션 수량을 대본대로 돌려준다.

    order_effects: 주문 시도마다 (예외 또는 None, 그 시도 뒤의 포지션 수량).
    pos_fail: futures_position_information 이 던질 예외.
    """

    def __init__(self, order_effects, start_amt=100.0, pos_fail=None):
        self.effects = list(order_effects)
        self.amt = start_amt
        self.pos_fail = pos_fail
        self.orders = []
        self.pos_calls = 0

    def futures_create_order(self, **kw):
        self.orders.append(dict(kw))
        exc, after = self.effects[min(len(self.orders) - 1, len(self.effects) - 1)]
        if after is not None:
            self.amt = after
        if exc is not None:
            raise exc
        return {'status': 'NEW', 'orderId': len(self.orders)}

    def futures_position_information(self, symbol=None):
        self.pos_calls += 1
        if self.pos_fail is not None:
            raise self.pos_fail
        return [{'symbol': symbol, 'positionSide': 'BOTH', 'positionAmt': str(self.amt)}]


def run_presell(client, step_size=0.001, meta_boom=False):
    """current=$1000 → target=$500, mark=$10 → 매도 수량 50."""
    saved = {k: getattr(a, k) for k in
             ('get_symbol_constraints', 'format_quantity', '_trading_symbols',
              'ORDER_RETRY_DELAYS')}
    alerts = []
    try:
        def _meta(_c, _s):
            if meta_boom:
                raise a.MarketMetaUnavailable('심볼정보 없음')
            return {'step_size': step_size, 'min_qty': 0.0, 'min_notional': 5.0,
                    'tick_size': 0.01, 'qty_precision': 3}
        a.get_symbol_constraints = _meta
        a.format_quantity = lambda _c, s, q: f"{q:.3f}"
        a._trading_symbols = lambda _c: {'AAAUSDT'}      # 기본은 거래 가능
        a.ORDER_RETRY_DELAYS = [0.0, 0.0, 0.0]
        a._v25_touch_reset()
        ok, why = a._v25_partial_sell_for_leverage_down(
            client, 'AAAUSDT', 1000.0, 500.0, 10.0, order_alerts=alerts)
    finally:
        for k, v in saved.items():
            setattr(a, k, v)
    return ok, why, alerts


def _timeout():
    return RuntimeError('Read timed out')


print("\n[5] 사전매도 — 정상")
cl = PresellClient([(None, 50.0)])
ok, why, alerts = run_presell(cl)
check("성공", ok is True and why == '', f"{ok} {why}")
check("주문 1건", len(cl.orders) == 1, str(cl.orders))
check("reduceOnly 매도", cl.orders[0]['side'] == 'SELL' and cl.orders[0]['reduceOnly'] is True,
      str(cl.orders))
check("규격 함수가 만든 수량 문자열로 나간다", cl.orders[0]['quantity'] == '50.000', str(cl.orders))
check("체결 알림에 남는다", any('L↓ 사전매도' in m for m in alerts), str(alerts))

print("\n[5b] 응답이 유실됐지만 포지션이 줄었으면 다시 팔지 않는다")
# 첫 주문은 거래소에 닿아 체결됐고(100 → 50) 응답만 잃었다. 재주문하면 50 을 또 판다.
cl = PresellClient([(_timeout(), 50.0), (None, 0.0)])
ok, why, alerts = run_presell(cl)
check("성공으로 본다", ok is True and why == '', f"{ok} {why}")
check("주문은 1건뿐(중복 매도 없음)", len(cl.orders) == 1, str(cl.orders))
check("계좌 변경 표시가 남는다", any('AAAUSDT' in t for t in a._V25_ACCOUNT_TOUCH),
      str(a._V25_ACCOUNT_TOUCH))

print("\n[5c] 미접수가 확인되면 재시도한다")
cl = PresellClient([(_timeout(), 100.0), (None, 50.0)])
ok, why, alerts = run_presell(cl)
check("성공", ok is True and why == '', f"{ok} {why}")
check("주문 2건(재시도 1회)", len(cl.orders) == 2, str(cl.orders))
check("두 번 다 같은 수량", {o['quantity'] for o in cl.orders} == {'50.000'}, str(cl.orders))

print("\n[5d] 접수 여부가 애매하면 재주문하지 않는다")
cl = PresellClient([(_timeout(), 90.0), (None, 40.0)])   # 10 만 줄었다 — 설명이 안 된다
ok, why, alerts = run_presell(cl)
check("차단", ok is False, f"{ok} {why}")
check("주문 1건", len(cl.orders) == 1, str(cl.orders))
check("사유에 포지션 변화가 남는다", '접수 여부 불명' in why, why)

print("\n[5e] 포지션을 못 읽으면 재주문하지 않는다")
cl = PresellClient([(_timeout(), None)], pos_fail=RuntimeError('position api down'))
ok, why, alerts = run_presell(cl)
check("차단", ok is False, f"{ok} {why}")
check("주문 1건", len(cl.orders) == 1, str(cl.orders))
check("사유가 조회 실패임을 밝힌다", '확인 불가' in why and '포지션 조회 실패' in why, why)

print("\n[5f] 재시도를 다 써도 안 나가면 차단한다")
cl = PresellClient([(_timeout(), 100.0)])
ok, why, alerts = run_presell(cl)
check("차단", ok is False, f"{ok} {why}")
check(f"주문은 {a.PRESELL_MAX_ATTEMPTS}건까지만",
      len(cl.orders) == a.PRESELL_MAX_ATTEMPTS, str(cl.orders))

print("\n[5g] 되돌릴 수 없는 거절은 재시도하지 않는다")
_marg = a.BinanceAPIException(type('R', (), {'text': '', 'status_code': 400})(),
                              400, '{"code":-2019,"msg":"Margin is insufficient"}')
cl = PresellClient([(_marg, 100.0)])
ok, why, alerts = run_presell(cl)
check("차단", ok is False, f"{ok} {why}")
check("주문 1건", len(cl.orders) == 1, str(cl.orders))
check("사유에 거래소 코드가 남는다", 'code=-2019' in why, why)

print("\n[5h] 거래정지·상장폐지면 사유를 밝히고 그날 전체를 막는다")
_invalid = a.BinanceAPIException(type('R', (), {'text': '', 'status_code': 400})(),
                                 400, '{"code":-1121,"msg":"Invalid symbol"}')
cl = PresellClient([(_invalid, 100.0)])
saved_ts = a._trading_symbols
try:
    a._trading_symbols = lambda _c: {'BBBUSDT'}      # AAAUSDT 가 목록에서 빠졌다
    saved2 = {k: getattr(a, k) for k in ('get_symbol_constraints', 'format_quantity')}
    try:
        a.get_symbol_constraints = lambda _c, s: {
            'step_size': 0.001, 'min_qty': 0.0, 'min_notional': 5.0,
            'tick_size': 0.01, 'qty_precision': 3}
        a.format_quantity = lambda _c, s, q: f"{q:.3f}"
        a._v25_touch_reset()
        ok, why = a._v25_partial_sell_for_leverage_down(cl, 'AAAUSDT', 1000.0, 500.0, 10.0)
    finally:
        for k, v in saved2.items():
            setattr(a, k, v)
finally:
    a._trading_symbols = saved_ts
check("차단", ok is False, f"{ok} {why}")
check("사유가 거래정지·상장폐지임을 밝힌다", '거래정지' in why and '상장폐지' in why, why)
check("거래불가로 따로 기록된다", a._untradable_orders == ['AAAUSDT'], str(a._untradable_orders))
check("재주문 없음", len(cl.orders) == 1, str(cl.orders))

print("\n[5i] 심볼 규격을 못 읽으면 주문 없이 사유를 돌려준다")
cl = PresellClient([(None, 50.0)])
ok, why, alerts = run_presell(cl, meta_boom=True)
check("차단", ok is False, f"{ok} {why}")
check("주문 0건", cl.orders == [], str(cl.orders))
check("사유가 규격 조회 실패임을 밝힌다", '심볼 규격 조회 실패' in why, why)
check("예외가 밖으로 안 새어나간다(위 호출이 끝났다)", True)

cl = PresellClient([(None, 50.0)])
ok, why, alerts = run_presell(cl, step_size=0.0)
check("step_size 를 못 읽어도 주문 0건", cl.orders == [], str(cl.orders))
check("차단 + 사유", ok is False and 'step_size' in why, f"{ok} {why}")

print("\n[5j] 줄일 게 없으면 성공이고 사유도 없다")
saved = {k: getattr(a, k) for k in ('get_symbol_constraints', 'format_quantity')}
try:
    a.get_symbol_constraints = lambda _c, s: {
        'step_size': 0.001, 'min_qty': 0.0, 'min_notional': 5.0,
        'tick_size': 0.01, 'qty_precision': 3}
    a.format_quantity = lambda _c, s, q: f"{q:.3f}"
    cl = PresellClient([(None, 50.0)])
    ok, why = a._v25_partial_sell_for_leverage_down(cl, 'AAAUSDT', 500.0, 1000.0, 10.0)
    check("불필요는 성공", ok is True and why == '', f"{ok} {why}")
    check("주문 0건", cl.orders == [], str(cl.orders))
    cl = PresellClient([(None, 50.0)])
    ok, why = a._v25_partial_sell_for_leverage_down(cl, 'AAAUSDT', 1000.0, 500.0, 0.0)
    check("mark_price 가 비정상이면 차단 + 사유", ok is False and 'mark_price' in why, f"{ok} {why}")
    check("주문 0건", cl.orders == [], str(cl.orders))
finally:
    for k, v in saved.items():
        setattr(a, k, v)

print("\n[6] 포지션 수량 조회는 모름과 0 을 구분한다")


class _PosOnly:
    def __init__(self, rows, boom=None):
        self.rows, self.boom = rows, boom

    def futures_position_information(self, symbol=None):
        if self.boom:
            raise self.boom
        return self.rows


check("정상이면 절대값",
      a._position_amt(_PosOnly([{'symbol': 'AAAUSDT', 'positionSide': 'BOTH',
                                 'positionAmt': '-7.5'}]), 'AAAUSDT') == 7.5)
check("조회 실패는 None(0 으로 속이지 않는다)",
      a._position_amt(_PosOnly([], boom=RuntimeError('down')), 'AAAUSDT') is None)
check("파싱 실패도 None",
      a._position_amt(_PosOnly([{'symbol': 'AAAUSDT', 'positionSide': 'BOTH',
                                 'positionAmt': 'x'}]), 'AAAUSDT') is None)
check("행이 없으면 0(실제로 무포지션)",
      a._position_amt(_PosOnly([]), 'AAAUSDT') == 0.0)
check("헤지 모드 행은 건너뛴다",
      a._position_amt(_PosOnly([{'symbol': 'AAAUSDT', 'positionSide': 'LONG',
                                 'positionAmt': '5'}]), 'AAAUSDT') == 0.0)

print("\n[7] 수량 문자열 판정")
for good in ('1', '0.001', '10.12345678', ' 2.5 '):
    check(f"{good!r} 는 유효", a._is_positive_number(good) is True)
for bad in (None, '', '  ', '0', '-1', 'abc', 1.0, 10, [], '1,5'):
    check(f"{bad!r} 는 무효", a._is_positive_number(bad) is False)

print("\n[8] 회귀")
_src = open(os.path.join(os.path.dirname(__file__), '..', 'trade',
                         'auto_trade_binance.py')).read()
check("전량청산이 float 로 되돌아가지 않는다",
      'qty_str = str(qty) if is_full_close' not in _src)
check("전량청산 수량을 주문 직전에 다시 검증한다",
      '전량청산 수량이 거래소 원문 문자열이 아니다' in _src)
check("원문 없는 전량청산은 축소 실패로 표시된다",
      'if _full_close_no_raw:' in _src and '_reduce_failed_hard = True' in _src)
check("포지션 조회 실패가 중단으로 올라간다",
      'V25 ABORT: 포지션 조회 실패' in _src)
check("매수 계획 실패가 알림으로 올라간다", 'BUY PLAN FAILED' in _src)
check("사전매도가 create_order_with_retry 를 쓰지 않는다",
      'create_order_with_retry(client, symbol=sym' not in _src
      and '_sent_ok(order)' in _src)
check("사전매도 재시도가 포지션으로 접수 여부를 확인한다",
      'qty_before = _position_amt(client, sym)' in _src
      and 'qty_now = _position_amt(client, sym)' in _src)
check("규격 조회 예외를 사전매도 안에서 처리한다",
      'except MarketMetaUnavailable as e:\n        log.error(f"_v25_partial_sell' in _src)
check("호출부가 사유를 받아 중단 기록·알림에 쓴다",
      '_presell_ok, _presell_why = _v25_partial_sell_for_leverage_down(' in _src
      and '_v25_persist_abort_log(err)' in _src)

print()
if FAILED:
    print(f"{len(FAILED)} FAIL: {FAILED}")
    sys.exit(1)
print("ALL PASS")
