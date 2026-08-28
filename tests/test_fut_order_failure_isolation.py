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
        self.attempted.append((sym, kw.get('reduceOnly')))
        if sym in self.fail_on:
            raise self.fail_on[sym]
        return {'status': 'NEW'}

    def futures_symbol_ticker(self, symbol=None):
        return {'symbol': symbol, 'price': '100.0'}

    def symbols(self):
        return [s for s, _ in self.attempted]


def run_rebalance(positions, fail_on, listed, target=None, liquidation_only=True,
                  format_quantity=None):
    """execute_rebalance 를 한 번 돌린다. 기본은 목표 전액 CASH(=전량청산)."""
    saved = {k: getattr(a, k) for k in
             ('get_current_positions', 'cancel_stop_orders', '_trading_symbols',
              'get_symbol_constraints', 'format_quantity', 'UNIVERSE')}
    errors, orders = [], []
    c = Client(fail_on)
    try:
        a.get_current_positions = lambda _c: (positions, 1000.0, True)
        a.cancel_stop_orders = lambda _c, _s=None: None
        a._trading_symbols = lambda _c: set(listed)
        a.get_symbol_constraints = lambda _c, s: {
            'step_size': 0.001, 'min_qty': 0.0, 'min_notional': 5.0,
            'tick_size': 0.01, 'qty_precision': 3}
        a.format_quantity = format_quantity or (lambda _c, s, q: f"{q:.3f}")
        a.UNIVERSE = list(listed)
        a._v25_touch_reset()
        a.execute_rebalance(c, target or {'CASH': 1.0}, 1000.0, {},
                            order_alerts=orders, error_alerts=errors,
                            liquidation_only=liquidation_only)
    finally:
        for k, v in saved.items():
            setattr(a, k, v)
    return c, errors, orders


HELD = {'AAA': pos('AAA', 500.0, 10.0), 'BBB': pos('BBB', 400.0, 20.0)}
LISTED = {'AAAUSDT', 'BBBUSDT'}

print("[1] 거래소 외 예외가 뒤 주문을 막지 않는다")
c, errors, orders = run_rebalance(
    HELD, {'AAAUSDT': RuntimeError('network down')}, LISTED)
check("두 심볼 다 주문을 시도했다", set(c.symbols()) == {'AAAUSDT', 'BBBUSDT'}, str(c.attempted))
check("성공한 쪽은 체결로 기록된다", any('BBBUSDT' in o for o in orders), str(orders))
check("실패한 쪽은 오류로 올라간다",
      any(m.startswith('ORDER FAILED') and 'AAAUSDT' in m for m in errors), str(errors))

print("\n[2] 거래불가 심볼의 축소(reduceOnly) 거절은 오류로 올리지 않는다")
c, errors, orders = run_rebalance(
    HELD, {'AAAUSDT': a.BinanceAPIException(type('R', (), {'text': '', 'status_code': 400})(),
                                            400, '{"code":-1121,"msg":"Invalid symbol"}')},
    {'BBBUSDT'})                      # AAA 가 상장 목록에서 빠졌다
check("거래불가 심볼도 주문 시도는 한다(재개되면 그날 나간다)",
      'AAAUSDT' in c.symbols(), str(c.attempted))
check("거절된 주문은 reduceOnly 였다", ('AAAUSDT', 'true') in c.attempted, str(c.attempted))
check("오류 목록에 안 올라간다(매일 알림·성공판정 실패 방지)", errors == [], str(errors))
check("거래불가로 따로 기록된다", a._untradable_orders == ['AAAUSDT'], str(a._untradable_orders))
check("다른 심볼은 정상 체결된다", any('BBBUSDT' in o for o in orders), str(orders))

print("\n[2b] 늘리는 주문(비-reduceOnly)의 거절은 거래불가여도 오류다")
_rej = a.BinanceAPIException(type('R', (), {'text': '', 'status_code': 400})(),
                             400, '{"code":-1121,"msg":"Invalid symbol"}')
c, errors, orders = run_rebalance(
    {}, {'AAAUSDT': _rej}, {'BBBUSDT'},
    target={'AAA': 1.0}, liquidation_only=False)
check("매수를 시도했다", ('AAAUSDT', None) in c.attempted, str(c.attempted))
check("오류로 올라간다(살 수 없는 걸 사려 한 건 유니버스 문제다)",
      any(m.startswith('ORDER FAILED') for m in errors), str(errors))
check("거래불가로는 기록되지 않는다", a._untradable_orders == [], str(a._untradable_orders))

print("\n[2c] 주문을 만들다 터진 건 체결불명도 거래불가도 아니다")


def _boom_qty(_c, s, q):
    if s == 'AAAUSDT':
        raise ValueError('step_size 없음')
    return f"{q:.3f}"


# 전량청산은 보유수량 문자열을 그대로 쓰므로 format_quantity 를 안 탄다.
# 부분매도(목표비중 축소)여야 주문 생성 단계가 실제로 돈다.
c, errors, orders = run_rebalance(
    HELD, {}, {'AAAUSDT', 'BBBUSDT'},
    target={'AAA': 0.1, 'BBB': 0.1, 'CASH': 0.8}, liquidation_only=False,
    format_quantity=_boom_qty)
check("거래소에 주문을 안 보냈다", 'AAAUSDT' not in c.symbols(), str(c.attempted))
check("PREP 오류로 올라간다",
      any(m.startswith('ORDER PREP FAILED') for m in errors), str(errors))
check("거래불가로 분류되지 않는다", a._untradable_orders == [], str(a._untradable_orders))
check("계좌 변경 표시가 안 남는다(보낸 게 없다)",
      not any('AAAUSDT' in t for t in a._V25_ACCOUNT_TOUCH), str(a._V25_ACCOUNT_TOUCH))
check("뒤 종목은 정상 체결된다", any('BBBUSDT' in o for o in orders), str(orders))

print("\n[2d] 축소 주문이 실패하면 그 실행에서 노출을 늘리지 않는다")
# AAA 는 줄여야 하고(축소) CCC 는 새로 사야 한다(확대). 축소가 터지면 매수는 막혀야 한다.
c, errors, orders = run_rebalance(
    {'AAA': pos('AAA', 900.0, 10.0)}, {'AAAUSDT': RuntimeError('network down')},
    {'AAAUSDT', 'CCCUSDT'},
    target={'AAA': 0.1, 'CCC': 0.4, 'CASH': 0.5}, liquidation_only=False)
check("축소는 시도했다", 'AAAUSDT' in c.symbols(), str(c.attempted))
check("뒤따르는 매수를 안 보낸다", 'CCCUSDT' not in c.symbols(), str(c.attempted))
check("건너뛴 사실이 오류로 남는다",
      any('BUY SKIPPED' in m for m in errors), str(errors))

print("\n[2e] 축소가 정상이면 매수는 그대로 나간다")
c, errors, orders = run_rebalance(
    {'AAA': pos('AAA', 900.0, 10.0)}, {}, {'AAAUSDT', 'CCCUSDT'},
    target={'AAA': 0.1, 'CCC': 0.4, 'CASH': 0.5}, liquidation_only=False)
check("축소·확대 둘 다 나간다", {'AAAUSDT', 'CCCUSDT'} <= set(c.symbols()), str(c.attempted))
check("오류 없음", errors == [], str(errors))

print("\n[3] 거래 가능한 심볼의 거절은 그대로 오류다")
c, errors, orders = run_rebalance(
    HELD, {'AAAUSDT': a.BinanceAPIException(type('R', (), {'text': '', 'status_code': 400})(),
                                            400, '{"code":-2019,"msg":"Margin is insufficient"}')},
    LISTED)
check("오류로 올라간다", any(m.startswith('ORDER FAILED') for m in errors), str(errors))
check("거래불가로는 기록되지 않는다", a._untradable_orders == [], str(a._untradable_orders))

print("\n[4] 거래가능 여부를 못 물어보면 평범한 실패로 본다")
saved = a._trading_symbols
try:
    def _boom(_c):
        raise a.MarketMetaUnavailable('메타 없음')
    a._trading_symbols = _boom
    check("판단 불가는 False(=평범한 실패)", a._is_untradable_now(None, 'AAAUSDT') is False)
finally:
    a._trading_symbols = saved

print("\n[4b] 거래소가 거절한 게 아니면(네트워크 등) 억제하지 않는다")
c, errors, orders = run_rebalance(
    HELD, {'AAAUSDT': RuntimeError('timeout')}, {'BBBUSDT'})   # AAA 는 상장 목록에 없다
check("거래불가 심볼이어도 오류로 올라간다",
      any(m.startswith('ORDER FAILED') for m in errors), str(errors))
check("거래불가로 기록되지 않는다", a._untradable_orders == [], str(a._untradable_orders))

print("\n[4c] 전략 대상이 아닌 계약도 거래 중이면 거래불가가 아니다")
_saved_info = a.get_exchange_info
_saved_cache = a._exchange_info_cache
_saved_fresh0 = a._exchange_info_fresh
try:
    a._exchange_info_cache = None
    a._exchange_info_fresh = True
    a.get_exchange_info = lambda _c: {'symbols': [
        # 전략 유니버스(USDT 무기한)엔 안 들어가지만 거래소에선 정상 거래 중이다.
        {'symbol': 'BTCUSDC', 'status': 'TRADING', 'contractType': 'PERPETUAL',
         'quoteAsset': 'USDC'},
        {'symbol': 'BTCUSDT_260925', 'status': 'TRADING', 'contractType': 'CURRENT_QUARTER',
         'quoteAsset': 'USDT'},
        {'symbol': 'AAAUSDT', 'status': 'SETTLING', 'contractType': 'PERPETUAL',
         'quoteAsset': 'USDT'},
    ]}
    check("정상 거래 중인 USDC 무기한은 거래불가가 아니다",
          a._is_untradable_now(None, 'BTCUSDC') is False)
    check("정상 거래 중인 만기물도 거래불가가 아니다",
          a._is_untradable_now(None, 'BTCUSDT_260925') is False)
    check("거래정지된 USDT 무기한은 거래불가다",
          a._is_untradable_now(None, 'AAAUSDT') is True)
    check("심볼정보에 아예 없으면 거래불가다(상장폐지)",
          a._is_untradable_now(None, 'ZZZUSDT') is True)
finally:
    a.get_exchange_info = _saved_info
    a._exchange_info_cache = _saved_cache
    a._exchange_info_fresh = _saved_fresh0

print("\n[4d] 심볼정보가 캐시본이면 거래상태를 판정하지 않는다")
_saved_info = a.get_exchange_info
_saved_cache = a._exchange_info_cache
_saved_fresh = a._exchange_info_fresh
try:
    a._exchange_info_cache = None
    a.get_exchange_info = lambda _c: {'symbols': [
        {'symbol': 'BBBUSDT', 'status': 'TRADING', 'contractType': 'PERPETUAL',
         'quoteAsset': 'USDT'}]}
    a._exchange_info_fresh = False
    check("캐시본이면 미등재 심볼도 거래불가로 단정하지 않는다",
          a._is_untradable_now(None, 'AAAUSDT') is False)
    _saved_ts = a._trading_symbols
    try:
        a._trading_symbols = lambda _c: (_ for _ in ()).throw(
            a.MarketMetaUnavailable('캐시본'))
        keep, resumed, dropped = a._resolve_untradable(
            None, {'AAAUSDT'}, set(), {}, True)
        check("캐시본이면 해소하지 않고 목록을 보존한다",
              keep == {'AAAUSDT'} and not resumed and not dropped,
              f"keep={keep} resumed={resumed} dropped={dropped}")
    finally:
        a._trading_symbols = _saved_ts
    a._exchange_info_fresh = True
    check("방금 받은 응답이면 평소대로 판정한다",
          a._is_untradable_now(None, 'AAAUSDT') is True)
finally:
    a.get_exchange_info = _saved_info
    a._exchange_info_cache = _saved_cache
    a._exchange_info_fresh = _saved_fresh

print("\n[5] 해소는 적극적 증거가 있을 때만 한다")


class _ListedClient:
    def __init__(self, listed=None, boom=False):
        self.listed, self.boom = set(listed or []), boom


def _resolve(prev, now, listed, positions=None, positions_ok=True, boom=False):
    saved = a._trading_symbols
    try:
        if boom:
            def _f(_c):
                raise a.MarketMetaUnavailable('메타 없음')
        else:
            def _f(_c):
                return set(listed)
        a._trading_symbols = _f
        return a._resolve_untradable(None, set(prev), set(now), positions, positions_ok)
    finally:
        a._trading_symbols = saved


keep, resumed, dropped = _resolve(
    {'AAAUSDT'}, set(), {'BBBUSDT'}, positions={'AAA': pos('AAA', 500.0, 10.0)})
check("주문 안 낸 날은 해소로 보지 않는다", keep == {'AAAUSDT'} and not resumed and not dropped,
      f"keep={keep} resumed={resumed} dropped={dropped}")

keep, resumed, dropped = _resolve(
    {'AAAUSDT'}, set(), {'AAAUSDT', 'BBBUSDT'}, positions={'AAA': pos('AAA', 500.0, 10.0)})
check("상장 목록에 돌아오면 재개로 본다", keep == set() and resumed == {'AAAUSDT'},
      f"keep={keep} resumed={resumed}")

keep, resumed, dropped = _resolve({'AAAUSDT'}, set(), {'BBBUSDT'}, positions={})
check("보유가 없어졌으면 추적을 끝낸다", keep == set() and dropped == {'AAAUSDT'},
      f"keep={keep} dropped={dropped}")

keep, resumed, dropped = _resolve({'AAAUSDT'}, set(), {'BBBUSDT'}, positions={}, positions_ok=False)
check("포지션 조회 실패면 보유 없음으로 단정하지 않는다", keep == {'AAAUSDT'}, str(keep))

keep, resumed, dropped = _resolve({'AAAUSDT'}, {'CCCUSDT'}, None, boom=True)
check("거래가능 목록 조회 실패면 목록을 그대로 둔다",
      keep == {'AAAUSDT', 'CCCUSDT'} and not resumed and not dropped, str(keep))

print("\n[6] 회귀")
_src = open(os.path.join(os.path.dirname(__file__), '..', 'trade', 'auto_trade_binance.py')).read()
check("주문 루프가 예외 종류를 안 가린다",
      'except BinanceAPIException as e:\n            # 예외라도' not in _src)
check("거래불가 알림은 상태 변경 시에만",
      "state['untradable_symbols'] = sorted(_keep_untradable)" in _src)
check("알림 전송 실패면 상태를 갱신하지 않는다(첫 통보 유실 방지)",
      'if _notify_ok:' in _src)
check("억제는 축소 주문 + 거래소 거절에만 적용된다",
      'if reduce_only and isinstance(e, BinanceAPIException) '
      'and _is_untradable_now(client, symbol):' in _src)
check("주문 생성 단계는 별도 try 다", 'ORDER PREP FAILED' in _src)
check("축소 실패 뒤 확대 주문을 막는다", 'BUY SKIPPED' in _src)
check("캐시본으로 거래상태를 판정하지 않는다",
      'if not _exchange_info_fresh:' in _src)
check("거래가능 판정에 전략 유니버스 목록을 쓰지 않는다",
      'symbol not in _trading_symbols(client)' in _src
      and 'symbol not in fetch_binance_futures_listed' not in _src)
check("접수 뒤 기록 실패를 주문 실패로 되돌리지 않는다",
      '주문은 접수됐고 기록만 실패' in _src)
check("축소 준비 실패도 확대 차단 표시를 세운다",
      'if reduce_only:\n                _reduce_failed_hard = True' in _src)
check("취소 거절도 거래불가만 실패에서 뺀다",
      'isinstance(e, BinanceAPIException) and _is_untradable_now(client, symbol)' in _src
      and 'untradable.append((symbol, kind, str(e)))' in _src)
check("상태파일의 거래불가 목록 형태를 검증한다",
      'untradable_symbols 형태가 이상하다' in _src)

print("\n[6차 검토] 축소 주문을 만들다 실패해도 확대 주문을 막는다")


def _prep_fail_on(bad):
    def _fmt(_c, sym, q):
        if sym == bad:
            raise ValueError('LOT_SIZE 규격을 찾을 수 없다')
        return f"{q:.3f}"
    return _fmt


c, errors, orders = run_rebalance(
    HELD, {}, LISTED | {'CCCUSDT'},
    target={'AAA': 0.1, 'CCC': 0.4, 'CASH': 0.5}, liquidation_only=False,
    format_quantity=_prep_fail_on('AAAUSDT'))
check("축소 준비 실패를 오류로 올린다",
      any('ORDER PREP FAILED' in m and 'AAAUSDT' in m for m in errors), str(errors))
check("뒤따르는 확대 주문을 보내지 않는다",
      'CCCUSDT' not in c.symbols(), str(c.attempted))
check("확대 차단을 오류로 알린다",
      any('BUY SKIPPED' in m and 'CCCUSDT' in m for m in errors), str(errors))

# 대조: 확대 주문 쪽 준비가 실패한 것은 축소 실패가 아니다 — 다른 주문을 막지 않는다.
c, errors, orders = run_rebalance(
    HELD, {}, LISTED | {'CCCUSDT'},
    target={'AAA': 0.1, 'BBB': 0.1, 'CCC': 0.4, 'CASH': 0.4}, liquidation_only=False,
    format_quantity=_prep_fail_on('CCCUSDT'))
check("확대 준비 실패는 다른 주문을 막지 않는다",
      any('BUY SKIPPED' in m for m in errors) is False, str(errors))


print("\n[6차 검토] 거래불가 심볼의 취소 거절이 전체 매매를 막지 않는다")


class CancelClient:
    """cancel 을 받아 적고, reject 에 든 심볼은 지정한 예외를 던진다."""

    def __init__(self, reject):
        self.reject = reject
        self.done = []

    def futures_cancel_all_open_orders(self, symbol=None):
        if symbol in self.reject:
            raise self.reject[symbol]
        self.done.append(('open', symbol))

    def futures_cancel_all_algo_open_orders(self, symbol=None):
        if symbol in self.reject:
            raise self.reject[symbol]
        self.done.append(('algo', symbol))


def run_cancel(reject, listed):
    saved = {k: getattr(a, k) for k in ('_trading_symbols', 'UNIVERSE')}
    c = CancelClient(reject)
    try:
        a._trading_symbols = lambda _c: set(listed)
        a.UNIVERSE = ['AAA', 'BBB']
        ok = a.force_cancel_all_orders(c, ['AAAUSDT', 'BBBUSDT'])
    finally:
        for k, v in saved.items():
            setattr(a, k, v)
    return ok, c


_invalid = a.BinanceAPIException(type('R', (), {'text': '', 'status_code': 400})(),
                                 400, '{"code":-1121,"msg":"Invalid symbol."}')
ok, c = run_cancel({'AAAUSDT': _invalid}, {'BBBUSDT'})
check("거래불가 심볼의 취소 거절은 전체 중단이 아니다", ok is True)
check("나머지 심볼의 취소는 계속한다", ('open', 'BBBUSDT') in c.done, str(c.done))

ok, _ = run_cancel({'AAAUSDT': _invalid}, {'AAAUSDT', 'BBBUSDT'})
check("거래 가능한 심볼의 거절은 그대로 중단시킨다", ok is False)

ok, _ = run_cancel({'AAAUSDT': RuntimeError('네트워크 끊김')}, {'BBBUSDT'})
check("거래소 예외가 아니면(네트워크 등) 그대로 중단시킨다", ok is False)


def _raise_stale(_c):
    raise a.MarketMetaUnavailable('캐시본')


saved_ts = a._trading_symbols
try:
    a._trading_symbols = _raise_stale
    c = CancelClient({'AAAUSDT': _invalid})
    ok = a.force_cancel_all_orders(c, ['AAAUSDT'])
finally:
    a._trading_symbols = saved_ts
check("판정불가(캐시본)면 보수적으로 중단시킨다", ok is False)


print()
if FAILED:
    print(f"{len(FAILED)} FAIL: {FAILED}")
    sys.exit(1)
print("ALL PASS")
