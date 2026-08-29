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
check("억제는 축소 주문 + 시장상태 거절에만 적용된다",
      'if (reduce_only and isinstance(e, BinanceAPIException)' in _src
      and 'and _is_market_state_rejection(e)' in _src
      and 'and _is_untradable_now(client, symbol)):' in _src)
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
check("취소 거절도 시장상태 거절만 실패에서 뺀다",
      _src.count('and _is_market_state_rejection(e)') >= 2
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


def _rec(notional, qty, lev):
    """주문이 안 나간 심볼의 실행 전 상태(명목·부호 있는 수량·레버리지)."""
    return {'notional': notional, 'qty': qty, 'leverage': lev}


print("\n[7차 검토] 억제한 축소 실패가 종단 체결검사에서 되살아나지 않는다")
# 목표에서 빠진 심볼(intent 명목 0)이 거래불가로 남았다. 매일 불일치로 잡히면
# abort_streak 가 올라가 사흘째 자동 lock 이다 — 이 변경이 막으려던 결과 그대로다.
_intent = {'positions': {'BBBUSDT': {'notional': 400.0, 'leverage': 3}}, 'margin_type': 'CROSSED'}
_actual = {'positions': {'BBBUSDT': {'notional': 400.0, 'leverage': 3, 'qty': 20.0},
                         'AAAUSDT': {'notional': 500.0, 'leverage': 3, 'qty': 10.0}}}
check("빼주지 않으면 불일치로 잡힌다(전제)",
      any('AAAUSDT' in d for d in a._v25_exec_reconcile(_intent, _actual)))
check("거래불가 잔존은 불일치가 아니다",
      a._v25_exec_reconcile(_intent, _actual, {'AAAUSDT': _rec(500.0, 10.0, 3)}) == [])

# 노출이 늘었으면 빼주지 않는다.
_grown = {'positions': {'BBBUSDT': {'notional': 400.0, 'leverage': 3},
                        'AAAUSDT': {'notional': 900.0, 'leverage': 3, 'qty': 18.0}}}
check("잔존이 커졌으면 그대로 실패로 잡는다",
      any('AAAUSDT' in d for d in a._v25_exec_reconcile(_intent, _grown, {'AAAUSDT': _rec(500.0, 10.0, 3)})))

# 가격이 조금 움직인 정도는 통과한다(같은 실행 안 두 시점 비교).
_moved = {'positions': {'BBBUSDT': {'notional': 400.0, 'leverage': 3},
                        'AAAUSDT': {'notional': 512.0, 'leverage': 3, 'qty': 10.0}}}
check("가격 이동 정도는 통과", a._v25_exec_reconcile(_intent, _moved, {'AAAUSDT': _rec(500.0, 10.0, 3)}) == [])

# 다른 심볼의 불일치는 그대로 잡는다.
_other = {'positions': {'BBBUSDT': {'notional': 100.0, 'leverage': 3},
                        'AAAUSDT': {'notional': 500.0, 'leverage': 3, 'qty': 10.0}}}
_d = a._v25_exec_reconcile(_intent, _other, {'AAAUSDT': _rec(500.0, 10.0, 3)})
check("다른 심볼 불일치는 그대로 잡는다",
      len(_d) == 1 and 'BBBUSDT' in _d[0], str(_d))

# 사흘 연속으로 같은 상태가 와도 불일치가 0 이어야 자동 lock 이 안 걸린다.
_three = [a._v25_exec_reconcile(_intent, _actual, {'AAAUSDT': _rec(500.0, 10.0, 3)}) for _ in range(3)]
check("사흘 연속 실행에서도 불일치 0", all(d == [] for d in _three), str(_three))

check("체결검사 호출부가 미전송 목록을 넘긴다",
      '_v25_exec_reconcile(intent_snapshot, actual_snapshot,' in _src
      and '_unsent_before)' in _src)


print("\n[8차 검토] 주문이 안 나간 심볼은 그 사실대로 판정한다")
# 8차 반례 두 갈래.
#   (가) 거래불가 축소 거절이 뒤따르는 매수를 막는다(우리 결정). 그 매수는 거래소에
#        나가지도 않았는데 체결검사가 "목표와 다르다" 로 잡아 자동 lock 으로 간다.
#   (나) 전량청산이 아니라 부분축소를 거절당하면 의도 명목이 0 이 아니라서 7차 조건
#        (의도 명목 0)을 통과하지 못하고 매일 실패로 남는다.
# 규칙을 하나로 합쳤다 — 주문을 안 냈으면 포지션은 실행 전 그대로여야 한다.

# (가) 건너뛴 매수: 새로 사려던 심볼이라 보유가 없다.
_skip_intent = {'positions': {'BBBUSDT': {'notional': 400.0, 'leverage': 3}}, 'margin_type': 'CROSSED'}
_skip_actual = {'positions': {}}
check("빼주지 않으면 건너뛴 매수가 불일치로 잡힌다(전제)",
      any('BBBUSDT' in d for d in a._v25_exec_reconcile(_skip_intent, _skip_actual)))
check("건너뛴 매수는 불일치가 아니다",
      a._v25_exec_reconcile(_skip_intent, _skip_actual, {'BBBUSDT': _rec(0.0, 0.0, 0)}) == [])

# 건너뛴 매수인데 이미 보유가 있던 경우(비중을 늘리려던 것).
_add_actual = {'positions': {'BBBUSDT': {'notional': 100.0, 'leverage': 3, 'qty': 2.0}}}
check("보유를 늘리려다 건너뛴 것도 실행 전 그대로면 통과",
      a._v25_exec_reconcile(_skip_intent, _add_actual, {'BBBUSDT': _rec(100.0, 2.0, 3)}) == [])
_add_grown = {'positions': {'BBBUSDT': {'notional': 400.0, 'leverage': 3, 'qty': 8.0}}}
check("안 보냈다는데 포지션이 늘었으면 잡는다",
      any('BBBUSDT' in d for d in a._v25_exec_reconcile(_skip_intent, _add_grown, {'BBBUSDT': _rec(100.0, 2.0, 3)})))

# (나) 부분축소 거절: 의도 명목이 0 이 아니다.
_part_intent = {'positions': {'AAAUSDT': {'notional': 300.0, 'leverage': 3}}}
_part_actual = {'positions': {'AAAUSDT': {'notional': 500.0, 'leverage': 3, 'qty': 10.0}}}
check("빼주지 않으면 부분축소 거절이 불일치로 잡힌다(전제)",
      any('AAAUSDT' in d for d in a._v25_exec_reconcile(_part_intent, _part_actual)))
check("부분축소 거절도 실행 전 그대로면 통과",
      a._v25_exec_reconcile(_part_intent, _part_actual, {'AAAUSDT': _rec(500.0, 10.0, 3)}) == [])
_part_grown = {'positions': {'AAAUSDT': {'notional': 900.0, 'leverage': 3, 'qty': 18.0}}}
check("부분축소인데 노출이 늘었으면 잡는다",
      any('AAAUSDT' in d for d in a._v25_exec_reconcile(_part_intent, _part_grown, {'AAAUSDT': _rec(500.0, 10.0, 3)})))

# 안 보냈다는데 줄어든 것도 설명이 안 된다(외부 개입·부분체결). 조용히 넘기지 않는다.
_shrunk = {'positions': {'AAAUSDT': {'notional': 100.0, 'leverage': 3, 'qty': 2.0}}}
check("안 보냈다는데 포지션이 줄었으면 잡는다",
      any('AAAUSDT' in d for d in a._v25_exec_reconcile(_intent, _shrunk, {'AAAUSDT': _rec(500.0, 10.0, 3)})))

# 종단: 거래불가 축소 거절 → 매수 보류 → 체결검사까지 한 줄로 이었을 때 불일치 0.
_rej8 = a.BinanceAPIException(type('R', (), {'text': '', 'status_code': 400})(),
                              400, '{"code":-1121,"msg":"Invalid symbol"}')
c, errors, orders = run_rebalance(
    {'AAA': pos('AAA', 500.0, 10.0)}, {'AAAUSDT': _rej8}, {'BBBUSDT'},
    target={'BBB': 0.4, 'CASH': 0.6}, liquidation_only=False)
check("거래불가 축소는 시도했고 거절됐다", 'AAAUSDT' in c.symbols(), str(c.attempted))
check("뒤따르는 매수는 안 나갔다", 'BBBUSDT' not in c.symbols(), str(c.attempted))
check("거래불가 축소 거절은 오류로 안 올린다", errors == [], str(errors))
check("건너뛴 매수가 따로 기록된다", a._skipped_buys == ['BBBUSDT'], str(a._skipped_buys))
check("거래불가도 따로 기록된다", a._untradable_orders == ['AAAUSDT'], str(a._untradable_orders))

# 기준 상태는 주문 루프가 직접 남긴다(main 이 다시 만들지 않는다 — 10차 지적).
_unsent8 = dict(a._unsent_before)
_intent8 = {'positions': {'BBBUSDT': {'notional': 400.0, 'leverage': 3}}, 'margin_type': 'CROSSED'}
_actual8 = {'positions': {'AAAUSDT': {'notional': 500.0, 'leverage': 3, 'qty': 10.0}}}
check("주문 직전 상태를 두 갈래 다 모은다",
      _unsent8 == {'AAAUSDT': _rec(500.0, 10.0, 0.0), 'BBBUSDT': _rec(0.0, 0.0, 0.0)}, str(_unsent8))
check("종단 체결검사에서 불일치 0",
      a._v25_exec_reconcile(_intent8, _actual8, _unsent8) == [], str(_unsent8))
check("같은 실행이 사흘 연속이어도 불일치 0",
      all(a._v25_exec_reconcile(_intent8, _actual8, _unsent8) == [] for _ in range(3)))

# 안전장치: 예외 범위가 다른 심볼까지 덮지 않는다.
_actual8_other = {'positions': {'AAAUSDT': {'notional': 500.0, 'leverage': 3, 'qty': 10.0},
                                'CCCUSDT': {'notional': 700.0, 'leverage': 3}}}
_intent8_other = {'positions': {'BBBUSDT': {'notional': 400.0, 'leverage': 3},
                                'CCCUSDT': {'notional': 100.0, 'leverage': 3}}}
_d8 = a._v25_exec_reconcile(_intent8_other, _actual8_other, _unsent8)
check("목록 밖 심볼의 불일치는 그대로 잡는다", len(_d8) == 1 and 'CCCUSDT' in _d8[0], str(_d8))

a._v25_touch_reset()
check("실행 초기화가 건너뛴 매수 목록도 비운다", a._skipped_buys == [], str(a._skipped_buys))


print("\n[9차 검토] 미전송 심볼은 명목만으로 판정하지 않는다")
# 9차 반례.
#   (다) 순회 집합이 intent | actual 뿐이라, 전량청산 목표(intent 없음)인 심볼이
#        거래소 정산·ADL 로 사라지면(actual 없음) 검사가 통째로 생략된다. 강제청산은
#        상장폐지 대응의 핵심 실환경이라 "희박한 외부 개입" 으로 넘길 수 없다.
#   (라) 절대 명목만 저장·비교하면 같은 명목의 방향 반전과 레버리지 변경이 통과한다.
# 조치: 순회 집합에 unsent 를 넣고, 실행 전 기록에 부호 있는 수량과 레버리지를 담는다.

_gone_intent = {'positions': {}}
_gone_actual = {'positions': {}}
check("전량청산 목표인 심볼이 사라지면 검사한다(순회 집합)",
      any('AAAUSDT' in d for d in a._v25_exec_reconcile(
          _gone_intent, _gone_actual, {'AAAUSDT': _rec(500.0, 10.0, 3)})),
      "(intent·actual 어디에도 없어 옛 코드는 통째로 건너뛰었다)")

# 같은 명목 방향 반전 — 롱 10개에서 숏 10개로. 명목만 보면 500 그대로다.
_flip_actual = {'positions': {'AAAUSDT': {'notional': 500.0, 'leverage': 3, 'qty': -10.0}}}
check("같은 명목이어도 방향이 뒤집혔으면 잡는다",
      any('AAAUSDT' in d for d in a._v25_exec_reconcile(
          _part_intent, _flip_actual, {'AAAUSDT': _rec(500.0, 10.0, 3)})))

# 같은 명목인데 수량만 바뀐 경우(가격 급변 + 부분체결이 상쇄돼도 수량은 못 속인다).
_qty_actual = {'positions': {'AAAUSDT': {'notional': 500.0, 'leverage': 3, 'qty': 6.0}}}
check("명목이 같아도 수량이 달라졌으면 잡는다",
      any('AAAUSDT' in d for d in a._v25_exec_reconcile(
          _part_intent, _qty_actual, {'AAAUSDT': _rec(500.0, 10.0, 3)})))

# 레버리지만 외부에서 바뀐 경우.
_lev_actual = {'positions': {'AAAUSDT': {'notional': 500.0, 'leverage': 5, 'qty': 10.0}}}
check("명목·수량이 같아도 레버리지가 바뀌었으면 잡는다",
      any('AAAUSDT' in d for d in a._v25_exec_reconcile(
          _part_intent, _lev_actual, {'AAAUSDT': _rec(500.0, 10.0, 3)})))

# 반대로 먼지 수준 수량 오차(반올림)는 통과해야 한다.
_dust_actual = {'positions': {'AAAUSDT': {'notional': 500.0, 'leverage': 3, 'qty': 10.02}}}
check("0.2% 수량 오차는 통과", 
      a._v25_exec_reconcile(_part_intent, _dust_actual, {'AAAUSDT': _rec(500.0, 10.0, 3)}) == [])

# 레버리지를 모르는 기록(신규 매수 보류 등)은 레버리지로 잡지 않는다.
check("실행 전 레버리지가 0(미보유)이면 L 비교를 안 한다",
      a._v25_exec_reconcile(_skip_intent, {'positions': {}}, {'BBBUSDT': _rec(0.0, 0.0, 0)}) == [])

# 체결검사가 차이를 내면 그 실행은 실패로 판정된다(자동 lock 연쇄의 입구).
check("차이가 나면 성공 판정이 뒤집힌다(호출부)",
      'if recon_diffs:' in _src and 'v25_success = False' in _src)
check("체결검사가 주문 직전 수량·레버리지를 함께 넘긴다",
      "'qty': float(p.get('qty', 0.0) or 0.0)," in _src
      and "'leverage': float(p.get('leverage', 0.0) or 0.0)," in _src)
check("실체결 스냅샷에 수량이 들어간다",
      "'qty': pos.get('qty', 0.0)," in _src)


print("\n[10차 검토] 기준 상태는 주문 루프가 본 것이어야 한다")
# 10차 반례: main 의 positions_before 는 실행 첫머리에서 찍는다. 그 뒤에 우리가 하는
# 레버리지 변경과 L 낮춤 사전매도가 반영돼 있지 않다. 그걸 기준으로 삼으면 우리가 낸
# 변경분이 나중에 "주문을 안 냈는데 포지션이 움직였다" 로 잡혀 실행이 실패로 뒤집힌다.
# 조치: 기준을 execute_rebalance 안에서(레버리지·사전매도가 끝난 뒤 읽은 포지션으로)
# 남긴다. main 은 그걸 그대로 쓴다.

_rej10 = a.BinanceAPIException(type('R', (), {'text': '', 'status_code': 400})(),
                               400, '{"code":-1121,"msg":"Invalid symbol"}')


def _pos_lev(coin, notional, qty, lev):
    d = pos(coin, notional, qty)
    d['leverage'] = lev
    return d


# (가) L 3배→4배로 올린 뒤 매수를 보류한 경우. 주문 루프가 보는 BBB 의 L 은 이미 4다.
_after_lev = {'AAA': _pos_lev('AAA', 500.0, 10.0, 3),
              'BBB': _pos_lev('BBB', 300.0, 6.0, 4)}
c, errors, orders = run_rebalance(
    _after_lev, {'AAAUSDT': _rej10}, {'BBBUSDT'},
    target={'BBB': 0.6, 'CASH': 0.4}, liquidation_only=False)
check("L 변경 뒤 매수가 보류됐다", a._skipped_buys == ['BBBUSDT'], str(a._skipped_buys))
check("기준 레버리지는 변경 후 값(4)이다",
      a._unsent_before.get('BBBUSDT', {}).get('leverage') == 4.0, str(a._unsent_before))
_lev_actual10 = {'positions': {'AAAUSDT': {'notional': 500.0, 'leverage': 3, 'qty': 10.0},
                               'BBBUSDT': {'notional': 300.0, 'leverage': 4, 'qty': 6.0}}}
_lev_intent10 = {'positions': {'BBBUSDT': {'notional': 600.0, 'leverage': 4}}}
check("우리가 낸 L 변경은 불일치가 아니다",
      a._v25_exec_reconcile(_lev_intent10, _lev_actual10, dict(a._unsent_before)) == [],
      str(a._unsent_before))
# 옛 방식(실행 첫머리 스냅샷, L=3)을 기준으로 삼으면 잡힌다 — 반례가 실재했다는 확인.
_stale = dict(a._unsent_before)
_stale['BBBUSDT'] = _rec(300.0, 6.0, 3)
check("옛 기준(변경 전 L)이면 오탐이 난다(전제)",
      any('BBBUSDT' in d for d in a._v25_exec_reconcile(_lev_intent10, _lev_actual10, _stale)))

# (나) L 낮춤 사전매도로 명목이 줄어든 뒤 매수를 보류한 경우.
_after_presell = {'AAA': _pos_lev('AAA', 500.0, 10.0, 3),
                  'BBB': _pos_lev('BBB', 200.0, 4.0, 2)}
c, errors, orders = run_rebalance(
    _after_presell, {'AAAUSDT': _rej10}, {'BBBUSDT'},
    target={'BBB': 0.6, 'CASH': 0.4}, liquidation_only=False)
check("사전매도 뒤 매수가 보류됐다", a._skipped_buys == ['BBBUSDT'], str(a._skipped_buys))
check("기준 명목은 사전매도 후 값(200)이다",
      a._unsent_before.get('BBBUSDT', {}).get('notional') == 200.0, str(a._unsent_before))
_ps_actual = {'positions': {'AAAUSDT': {'notional': 500.0, 'leverage': 3, 'qty': 10.0},
                            'BBBUSDT': {'notional': 200.0, 'leverage': 2, 'qty': 4.0}}}
_ps_intent = {'positions': {'BBBUSDT': {'notional': 600.0, 'leverage': 2}}}
check("우리가 낸 사전매도는 불일치가 아니다",
      a._v25_exec_reconcile(_ps_intent, _ps_actual, dict(a._unsent_before)) == [])
_stale2 = dict(a._unsent_before)
_stale2['BBBUSDT'] = _rec(900.0, 18.0, 2)     # 사전매도 전 스냅샷
check("옛 기준(사전매도 전 명목)이면 오탐이 난다(전제)",
      any('BBBUSDT' in d for d in a._v25_exec_reconcile(_ps_intent, _ps_actual, _stale2)))

check("main 이 기준을 다시 만들지 않는다",
      'for _sym in list(_untradable_orders) + list(_skipped_buys):' not in _src
      and 'def _record_unsent(' in _src)
check("기준을 못 남긴 심볼은 빼주지 않는다(보수적 실패)",
      '_no_basis' in _src and '기준 상태 없음' in _src)
a._v25_touch_reset()
check("실행 초기화가 기준 기록도 비운다", a._unsent_before == {}, str(a._unsent_before))


print("\n[11차 검토] 거래불가는 시장 상태 거절에만 적용한다")
# 11차 반례: 억제 조건이 "심볼이 지금 거래불가인가" 뿐이라 오류 원인을 안 봤다.
# 거래불가 심볼에서 인증(-1022)·권한(-2015)·요청한도(-1003)·서버(5xx) 오류가 나면
# 그것까지 "상장폐지 거절" 로 흡수돼 error_alerts 에 안 남고 무기한 조용해진다.


def _api_err(code, status=400):
    return a.BinanceAPIException(type('R', (), {'text': '', 'status_code': status})(),
                                 status, '{"code":%d,"msg":"x"}' % code)


for _code, _name in ((-1022, '서명 오류'), (-2015, 'API 키 거부'), (-1003, '요청한도')):
    c, errors, orders = run_rebalance(
        HELD, {'AAAUSDT': _api_err(_code)}, {'BBBUSDT'})   # AAA 는 거래불가 상태
    check(f"{_name}({_code})는 거래불가여도 오류로 남는다",
          any(m.startswith('ORDER FAILED') for m in errors), str(errors))
    check(f"{_name}({_code})는 거래불가로 기록하지 않는다",
          a._untradable_orders == [], str(a._untradable_orders))

c, errors, orders = run_rebalance(
    HELD, {'AAAUSDT': _api_err(-1001, status=502)}, {'BBBUSDT'})
check("서버 장애(5xx)도 거래불가로 흡수하지 않는다",
      any(m.startswith('ORDER FAILED') for m in errors) and a._untradable_orders == [],
      f"{errors} {a._untradable_orders}")

# 대조: 진짜 시장 상태 거절은 그대로 억제된다.
c, errors, orders = run_rebalance(
    HELD, {'AAAUSDT': _api_err(-1121)}, {'BBBUSDT'})
check("Invalid symbol(-1121)은 그대로 억제된다",
      errors == [] and a._untradable_orders == ['AAAUSDT'],
      f"{errors} {a._untradable_orders}")


def run_cancel11(fail_on, listed):
    saved = (a._trading_symbols, a.UNIVERSE)
    try:
        a._trading_symbols = lambda _c: set(listed)
        a.UNIVERSE = ['AAAUSDT']
        return a.force_cancel_all_orders(CancelClient(fail_on), ['AAAUSDT'])
    finally:
        a._trading_symbols, a.UNIVERSE = saved


check("취소 거절도 인증 오류면 전체 중단시킨다",
      run_cancel11({'AAAUSDT': _api_err(-1022)}, {'BBBUSDT'}) is False)
check("취소 거절이 시장 상태면 중단시키지 않는다",
      run_cancel11({'AAAUSDT': _api_err(-1121)}, {'BBBUSDT'}) is True)

# 같은 심볼이 한 실행에서 매도와 매수로 동시에 실리지 않는다(11차 비판의 전제 반박).
# 이게 성립해야 _record_unsent 의 기준이 "루프 자신의 앞선 체결" 과 어긋나지 않는다.
_dup = []
for _w in (0.0, 0.1, 0.25, 0.5, 0.75, 1.0):
    c, errors, orders = run_rebalance(
        {'AAA': pos('AAA', 500.0, 10.0)}, {}, {'AAAUSDT'},
        target={'AAA': _w, 'CASH': 1.0 - _w}, liquidation_only=False)
    _syms = c.symbols()
    if len(_syms) != len(set(_syms)):
        _dup.append((_w, _syms))
check("한 심볼이 매도·매수로 두 번 실리지 않는다", _dup == [], str(_dup))

check("보류한 매수를 일일 보고에 남긴다", "'매수 보류'" in _src)

print("\n[12차 검토] 계정·인증·권한·한도 계열 전부가 억제에서 빠진다")
# 12차 반례: 11차에서 넣은 목록에 -1002(인증 안 됨)가 없어, 거래불가 심볼의 축소 주문이
# -1002 로 거절되면 그대로 "상장폐지 거절" 로 흡수됐다. 청산 대상이 그 심볼뿐이면
# 오류 알림도 실패 판정도 없이 성공으로 통과한다.
for _code, _name in ((-1002, '인증 안 됨'), (-1099, '권한 없음'), (-1015, '주문수 한도')):
    c, errors, orders = run_rebalance(
        HELD, {'AAAUSDT': _api_err(_code)}, {'BBBUSDT'})
    check(f"{_name}({_code})는 거래불가여도 오류로 남는다",
          any(m.startswith('ORDER FAILED') for m in errors), str(errors))
    check(f"{_name}({_code})는 거래불가로 기록하지 않는다",
          a._untradable_orders == [], str(a._untradable_orders))
    check(f"{_name}({_code})는 취소 경로도 중단시킨다",
          run_cancel11({'AAAUSDT': _api_err(_code)}, {'BBBUSDT'}) is False)

# 수량·정밀도 계열은 일부러 억제 대상으로 남긴다. 상장폐지 심볼은 거래소 심볼정보에서
# 사라져 format_quantity 가 소수점 8자리로 떨어지므로, 정밀도·필터 거절이 상장폐지
# 경로의 정상적인 결과다. 이걸 하드 오류로 올리면 3일 자동잠김이 되돌아온다.
for _code in (-1013, -1111):
    c, errors, orders = run_rebalance(
        HELD, {'AAAUSDT': _api_err(_code)}, {'BBBUSDT'})
    check(f"정밀도·필터 거절({_code})은 거래불가 심볼에서 억제된다",
          errors == [] and a._untradable_orders == ['AAAUSDT'],
          f"{errors} {a._untradable_orders}")

check("정밀도·수량 코드는 비시장 목록에 없다",
      not ({-1013, -1111, -4164} & a._NON_MARKET_ERROR_CODES),
      str(a._NON_MARKET_ERROR_CODES))
check("거래불가 신호(-1121)도 비시장 목록에 없다",
      -1121 not in a._NON_MARKET_ERROR_CODES)

print("\n[13차 검토] 전제조건 오류와 비정형 응답도 억제하지 않는다")
# 13차 반례 (a): 전제조건 계열(-4061 포지션 방향 불일치, -4068 포지션 모드, -4111)은
# 우리 요청 구성·계정 설정이 틀렸다는 뜻인데 거래불가 심볼에서 나면 흡수됐다.
for _code in sorted(a._BINANCE_PRECONDITION_CODES):
    c, errors, orders = run_rebalance(
        HELD, {'AAAUSDT': _api_err(_code)}, {'BBBUSDT'})
    check(f"전제조건 오류({_code})는 거래불가여도 오류로 남는다",
          any(m.startswith('ORDER FAILED') for m in errors) and a._untradable_orders == [],
          f"{errors} {a._untradable_orders}")
    check(f"전제조건 오류({_code})는 취소 경로도 중단시킨다",
          run_cancel11({'AAAUSDT': _api_err(_code)}, {'BBBUSDT'}) is False)

check("전제조건 계열 전부가 비시장 목록에 들어 있다",
      a._BINANCE_PRECONDITION_CODES <= a._NON_MARKET_ERROR_CODES,
      str(a._NON_MARKET_ERROR_CODES))

# 13차 반례 (b): 응답이 JSON 이 아니면 python-binance 가 code=0 으로 둔다(프록시·차단장비의
# HTML 오류 페이지). 거래소가 이 심볼을 거절했다는 근거가 없으므로 억제하면 안 된다.
_html_err = a.BinanceAPIException(
    type('R', (), {'text': '<html>403 Forbidden</html>', 'status_code': 403})(),
    403, '<html>403 Forbidden</html>')
check("비정형 응답은 code 0 으로 온다", _html_err.code == 0)
c, errors, orders = run_rebalance(HELD, {'AAAUSDT': _html_err}, {'BBBUSDT'})
check("비정형 응답(code 0)은 거래불가여도 오류로 남는다",
      any(m.startswith('ORDER FAILED') for m in errors) and a._untradable_orders == [],
      f"{errors} {a._untradable_orders}")
check("비정형 응답(code 0)은 취소 경로도 중단시킨다",
      run_cancel11({'AAAUSDT': _html_err}, {'BBBUSDT'}) is False)

_nocode = a.BinanceAPIException(type('R', (), {'text': '', 'status_code': 400})(),
                                400, '{"msg":"x"}')
check("code 필드가 없는 응답은 code None", _nocode.code is None)
check("code None 도 시장 상태 거절로 읽지 않는다",
      a._is_market_state_rejection(_nocode) is False)


print()
if FAILED:
    print(f"{len(FAILED)} FAIL: {FAILED}")
    sys.exit(1)
print("ALL PASS")
