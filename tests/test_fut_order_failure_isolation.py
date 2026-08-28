#!/usr/bin/env python3
"""주문 실패가 그 주문에서 끝나는지, 거래불가 심볼을 매일 오류로 올리지 않는지 확인한다.

두 구멍을 막은 변경(2026-08-28)의 회귀 시험.
  (a) 주문 루프가 BinanceAPIException 만 잡아서, 네트워크 오류 같은 다른 예외가 나면
      뒤 종목의 주문이 시도조차 안 됐다. create_order_with_retry 는 non-Binance 예외도
      그대로 올린다.
  (b) 보유 심볼이 거래불가가 되면 매일 같은 주문을 만들고 매일 거절당하고 매일 알림이
      왔다. 오류로 올리면 성공 판정까지 매일 실패가 된다(3일 연속이면 자동 lock).
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


def pos(coin, notional, qty):
    return {'symbol': coin + 'USDT', 'notional': notional, 'qty': qty,
            'qty_raw': str(qty), 'weight': 0.1, 'pnl': 0.0}


class Client:
    """주문을 받아 적는다. fail_on 에 든 심볼은 지정한 예외를 던진다."""

    def __init__(self, fail_on=None):
        self.fail_on = fail_on or {}
        self.attempted = []

    def futures_create_order(self, **kw):
        sym = kw['symbol']
        self.attempted.append(sym)
        if sym in self.fail_on:
            raise self.fail_on[sym]
        return {'status': 'NEW'}


def run_rebalance(positions, fail_on, listed):
    """execute_rebalance 를 목표 전액 CASH(=전량청산)로 한 번 돌린다."""
    saved = {k: getattr(a, k) for k in
             ('get_current_positions', 'cancel_stop_orders', 'fetch_binance_futures_listed',
              'get_symbol_constraints', 'format_quantity', 'UNIVERSE')}
    errors, orders = [], []
    c = Client(fail_on)
    try:
        a.get_current_positions = lambda _c: (positions, 1000.0, True)
        a.cancel_stop_orders = lambda _c, _s=None: None
        a.fetch_binance_futures_listed = lambda _c: set(listed)
        a.get_symbol_constraints = lambda _c, s: {
            'step_size': 0.001, 'min_qty': 0.0, 'min_notional': 5.0,
            'tick_size': 0.01, 'qty_precision': 3}
        a.format_quantity = lambda _c, s, q: f"{q:.3f}"
        a.UNIVERSE = list(listed)
        a._v25_touch_reset()
        a.execute_rebalance(c, {'CASH': 1.0}, 1000.0, {},
                            order_alerts=orders, error_alerts=errors,
                            liquidation_only=True)
    finally:
        for k, v in saved.items():
            setattr(a, k, v)
    return c, errors, orders


HELD = {'AAA': pos('AAA', 500.0, 10.0), 'BBB': pos('BBB', 400.0, 20.0)}
LISTED = {'AAAUSDT', 'BBBUSDT'}

print("[1] 거래소 외 예외가 뒤 주문을 막지 않는다")
c, errors, orders = run_rebalance(
    HELD, {'AAAUSDT': RuntimeError('network down')}, LISTED)
check("두 심볼 다 주문을 시도했다", set(c.attempted) == {'AAAUSDT', 'BBBUSDT'}, str(c.attempted))
check("성공한 쪽은 체결로 기록된다", any('BBBUSDT' in o for o in orders), str(orders))
check("실패한 쪽은 오류로 올라간다",
      any(m.startswith('ORDER FAILED') and 'AAAUSDT' in m for m in errors), str(errors))

print("\n[2] 거래불가 심볼의 거절은 오류로 올리지 않는다")
c, errors, orders = run_rebalance(
    HELD, {'AAAUSDT': a.BinanceAPIException(type('R', (), {'text': '', 'status_code': 400})(),
                                            400, '{"code":-1121,"msg":"Invalid symbol"}')},
    {'BBBUSDT'})                      # AAA 가 상장 목록에서 빠졌다
check("거래불가 심볼도 주문 시도는 한다(재개되면 그날 나간다)",
      'AAAUSDT' in c.attempted, str(c.attempted))
check("오류 목록에 안 올라간다(매일 알림·성공판정 실패 방지)", errors == [], str(errors))
check("거래불가로 따로 기록된다", a._untradable_orders == ['AAAUSDT'], str(a._untradable_orders))
check("다른 심볼은 정상 체결된다", any('BBBUSDT' in o for o in orders), str(orders))

print("\n[3] 거래 가능한 심볼의 거절은 그대로 오류다")
c, errors, orders = run_rebalance(
    HELD, {'AAAUSDT': a.BinanceAPIException(type('R', (), {'text': '', 'status_code': 400})(),
                                            400, '{"code":-2019,"msg":"Margin is insufficient"}')},
    LISTED)
check("오류로 올라간다", any(m.startswith('ORDER FAILED') for m in errors), str(errors))
check("거래불가로는 기록되지 않는다", a._untradable_orders == [], str(a._untradable_orders))

print("\n[4] 거래가능 여부를 못 물어보면 평범한 실패로 본다")
saved = a.fetch_binance_futures_listed
try:
    def _boom(_c):
        raise a.MarketMetaUnavailable('메타 없음')
    a.fetch_binance_futures_listed = _boom
    check("판단 불가는 False(=평범한 실패)", a._is_untradable_now(None, 'AAAUSDT') is False)
finally:
    a.fetch_binance_futures_listed = saved

print("\n[5] 회귀")
_src = open(os.path.join(os.path.dirname(__file__), '..', 'trade', 'auto_trade_binance.py')).read()
check("주문 루프가 예외 종류를 안 가린다",
      'except BinanceAPIException as e:\n            # 예외라도' not in _src)
check("거래불가 알림은 상태 변경 시에만", "state['untradable_symbols'] = sorted(_now_untradable)" in _src)

print()
if FAILED:
    print(f"{len(FAILED)} FAIL: {FAILED}")
    sys.exit(1)
print("ALL PASS")
