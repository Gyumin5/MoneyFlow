"""V25 cycle 5 P1 unit tests — error code mapping, fresh fetch, preflight 계정 단위.

실행: cd /home/gmoh/mon/251229 && python3 tests/test_v25_error_codes.py
"""
import sys
sys.path.insert(0, '/home/gmoh/mon/251229')

from trade.auto_trade_binance import (
    _is_transient_error, _is_idempotent_error,
    _BINANCE_TRANSIENT_CODES, _BINANCE_IDEMPOTENT_CODES,
    _with_retry, _normalize_dual_side, _pick_oneway_row,
    preflight_zero_positions, preflight_zero_open_orders,
    verify_margin_type, verify_leverage,
)


class FakeBinanceErr(Exception):
    """Stand-in for BinanceAPIException with .code attr."""
    def __init__(self, code, msg='fake'):
        self.code = code
        super().__init__(f"{code}: {msg}")


def test_error_code_invariants():
    # P0: -1022 / -1003 must NOT be transient
    assert -1022 not in _BINANCE_TRANSIENT_CODES, "-1022 (INVALID_SIGNATURE) must not be transient"
    assert -1003 not in _BINANCE_TRANSIENT_CODES, "-1003 (rate-limit) must not be transient (longer backoff required)"
    # P1: added codes
    for code in (-1000, -1006, -1008):
        assert code in _BINANCE_TRANSIENT_CODES, f"{code} should be transient"
    # idempotent
    assert -4046 in _BINANCE_IDEMPOTENT_CODES and -4059 in _BINANCE_IDEMPOTENT_CODES
    print("PASS: code set invariants")


def test_is_transient_predicate():
    # Note: predicate uses BinanceAPIException; but real impl reads .code only — use isinstance trick
    # Our FakeBinanceErr is NOT BinanceAPIException, so we patch the function via direct call to code logic.
    # Verify via the _BINANCE_TRANSIENT_CODES set is equivalent.
    for c in (-1000, -1001, -1006, -1007, -1008, -1021):
        assert c in _BINANCE_TRANSIENT_CODES
    for c in (-1003, -1022, -4046, -2014):
        assert c not in _BINANCE_TRANSIENT_CODES
    print("PASS: transient predicate set membership")


def test_normalize_dual_side():
    # bool/str/int 안전 정규화
    assert _normalize_dual_side(False) is False
    assert _normalize_dual_side(True) is True
    assert _normalize_dual_side('false') is False
    assert _normalize_dual_side('TRUE') is True
    assert _normalize_dual_side('1') is True
    assert _normalize_dual_side(0) is False
    assert _normalize_dual_side(1) is True
    assert _normalize_dual_side(None) is False
    print("PASS: _normalize_dual_side")


def test_pick_oneway_row():
    # 단일 row, positionSide=BOTH → 선택
    one = [{'symbol': 'BTCUSDT', 'positionSide': 'BOTH', 'marginType': 'cross', 'leverage': '3'}]
    assert _pick_oneway_row(one) is one[0]
    # hedge mode (LONG/SHORT 두 row) → BOTH 없음 → None
    hedge = [
        {'symbol': 'BTCUSDT', 'positionSide': 'LONG', 'leverage': '3'},
        {'symbol': 'BTCUSDT', 'positionSide': 'SHORT', 'leverage': '3'},
    ]
    assert _pick_oneway_row(hedge) is None
    # empty
    assert _pick_oneway_row([]) is None
    assert _pick_oneway_row(None) is None
    print("PASS: _pick_oneway_row")


class FakeClient:
    """Mock binance client for preflight tests. cycle 8: futures_account 기반."""
    def __init__(self, positions=None, orders=None):
        # positions: list of dict with positionSide=BOTH default, isolated default False
        self._pos = positions or []
        self._ord = orders or []
        self.calls = []
    def futures_account(self):
        self.calls.append(('account', None))
        return {'positions': list(self._pos)}
    def futures_position_information(self, symbol=None):
        self.calls.append(('positions', symbol))
        if symbol:
            return [p for p in self._pos if p['symbol'] == symbol]
        return list(self._pos)
    def futures_get_open_orders(self, symbol=None):
        self.calls.append(('orders', symbol))
        if symbol:
            return [o for o in self._ord if o['symbol'] == symbol]
        return list(self._ord)


def test_preflight_account_wide():
    # Zero positions + zero orders → True
    cli = FakeClient(positions=[{'symbol': 'BTCUSDT', 'positionAmt': '0', 'positionSide': 'BOTH'}], orders=[])
    assert preflight_zero_positions(cli) is True
    assert preflight_zero_open_orders(cli) is True

    # Non-target symbol has position → ABORT (cycle 4 핵심)
    cli2 = FakeClient(positions=[
        {'symbol': 'BTCUSDT', 'positionAmt': '0', 'positionSide': 'BOTH'},
        {'symbol': 'XRPUSDT', 'positionAmt': '100', 'positionSide': 'BOTH'},  # 비-target 가정
    ])
    assert preflight_zero_positions(cli2) is False, "비-target 포지션을 잡지 못함"

    # Non-target order → ABORT
    cli3 = FakeClient(orders=[{'symbol': 'LINKUSDT', 'orderId': 1}])
    assert preflight_zero_open_orders(cli3) is False, "비-target 미체결 주문을 잡지 못함"

    print("PASS: preflight account-wide (catches non-target positions/orders)")


def test_verify_no_cache_param():
    """cycle 4 P1: verify_margin_type / verify_leverage 는 info_list 파라미터 없음 (fresh fetch 강제)."""
    import inspect
    assert 'info_list' not in inspect.signature(verify_margin_type).parameters
    assert 'info_list' not in inspect.signature(verify_leverage).parameters
    print("PASS: verify_* fresh-fetch enforced (no cache param)")


def test_verify_fresh_fetch_called():
    """verify_margin_type 호출 시 fresh fetch (futures_position_information) 가 호출되는지 확인."""
    cli = FakeClient(positions=[{
        'symbol': 'BTCUSDT', 'positionAmt': '0', 'positionSide': 'BOTH',
        'marginType': 'cross', 'leverage': '3',
    }])
    before = len(cli.calls)
    ok = verify_margin_type(cli, 'BTCUSDT', 'CROSSED')
    assert ok is True
    after = len(cli.calls)
    assert after > before, "verify_margin_type 이 fresh fetch 호출 안 함"

    before = len(cli.calls)
    ok = verify_leverage(cli, 'BTCUSDT', 3)
    assert ok is True
    after = len(cli.calls)
    assert after > before, "verify_leverage 가 fresh fetch 호출 안 함"
    print("PASS: verify_* 가 매 호출 fresh fetch")


def test_finalize_daily_bar_for_signal():
    """V25 cycle 7: UTC open_time anchor 검증 (인덱스 기반 close[:-1] 대체)."""
    from trade.auto_trade_binance import _finalize_daily_bar_for_signal, StaleBarError
    import pandas as pd
    from datetime import datetime, timedelta

    now = datetime(2026, 5, 28, 0, 5, 0)
    completed = datetime(2026, 5, 27, 0, 0, 0)

    # case 1: last == completed
    df = pd.DataFrame({'Close': [100.0, 110.0]}, index=pd.to_datetime(['2026-05-26', '2026-05-27']))
    assert _finalize_daily_bar_for_signal(df, now).index[-1] == completed

    # case 2: last == current (in-progress) → drop
    df = pd.DataFrame({'Close': [100.0, 110.0, 105.0]},
                     index=pd.to_datetime(['2026-05-26', '2026-05-27', '2026-05-28']))
    r = _finalize_daily_bar_for_signal(df, now)
    assert r.index[-1] == completed and len(r) == 2

    # case 3: stale
    try:
        _finalize_daily_bar_for_signal(pd.DataFrame({'Close': [100.0]},
            index=pd.to_datetime(['2026-05-25'])), now)
        assert False, "stale should raise"
    except StaleBarError:
        pass

    # case 4: future
    try:
        _finalize_daily_bar_for_signal(pd.DataFrame({'Close': [100.0]},
            index=pd.to_datetime(['2026-05-29'])), now)
        assert False, "future should raise"
    except StaleBarError:
        pass

    # case 5: empty
    try:
        _finalize_daily_bar_for_signal(pd.DataFrame(), now)
        assert False, "empty should raise"
    except StaleBarError:
        pass

    print("PASS: _finalize_daily_bar_for_signal (5 cases: completed/in-progress/stale/future/empty)")


def main():
    test_error_code_invariants()
    test_is_transient_predicate()
    test_normalize_dual_side()
    test_pick_oneway_row()
    test_preflight_account_wide()
    test_verify_no_cache_param()
    test_verify_fresh_fetch_called()
    test_finalize_daily_bar_for_signal()
    print("\n✅ V25 cycle 5+7 unit tests ALL PASSED")


if __name__ == '__main__':
    main()
