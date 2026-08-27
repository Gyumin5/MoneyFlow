#!/usr/bin/env python3
"""시장 메타를 못 얻은 실행이 주문을 한 건도 내지 않는지 프로세스 수준에서 확인한다.

단위시험(test_fut_market_meta_cache.py)은 refresh_universe 가 예외를 던지는 것까지만 본다.
여기서는 main(--trade) 를 실제로 태워서 다음 셋을 확인한다.
  · 거래소 주문 API 호출 0회
  · ABORT 가 기록으로 남는다
  · 텔레그램이 터져도 위 둘이 깨지지 않는다
ai-debate run-20260827T091013Z 의 검증계획 항목.
"""
import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'trade'))
import auto_trade_binance as a  # noqa: E402

FAILED = []


def check(name, cond, detail=''):
    print(f"  {'PASS' if cond else 'FAIL'}  {name}" + (f"  {detail}" if not cond else ''))
    if not cond:
        FAILED.append(name)


class FakeClient:
    """주문 계열 호출을 전부 잡아 세는 가짜 클라이언트."""

    ORDER_CALLS = (
        'futures_create_order', 'futures_cancel_order', 'futures_cancel_all_open_orders',
        'futures_change_leverage', 'futures_change_margin_type',
    )

    def __init__(self, *args, **kwargs):
        self.calls = []

    def __getattr__(self, name):
        def _rec(*args, **kwargs):
            self.calls.append(name)
            if name in self.ORDER_CALLS:
                raise AssertionError(f'차단됐어야 할 실행에서 {name} 호출')
            return {}
        return _rec

    def order_calls(self):
        return [c for c in self.calls if c in self.ORDER_CALLS]


def run_trade_with_meta_failure(tmp, telegram_explodes):
    """메타 실패 상황에서 main(--trade) 를 한 번 태우고 (주문호출, ABORT기록) 을 돌려준다."""
    aborts = []
    client_box = {}

    saved = {k: getattr(a, k) for k in (
        'load_config', 'Client', 'load_state', 'save_state', 'STATE_PATH',
        '_v25_check_lock', 'refresh_universe', 'send_telegram', '_v25_persist_abort_log',
    )}
    saved_argv = sys.argv

    def _client(*args, **kwargs):
        c = FakeClient()
        client_box['c'] = c
        return c

    def _telegram(msg):
        if telegram_explodes:
            raise RuntimeError('telegram down')
        return True

    def _refresh(_client_arg):
        raise a.MarketMetaUnavailable('거래소 심볼정보를 믿을 수 있게 못 얻었다')

    try:
        a.load_config = lambda: ('key', 'secret')
        a.Client = _client
        a.load_state = lambda: {}
        a.save_state = lambda s: None
        a.STATE_PATH = os.path.join(tmp, 'binance_state.json')
        a._v25_check_lock = lambda: None
        a.refresh_universe = _refresh
        a.send_telegram = _telegram
        a._v25_persist_abort_log = lambda msg: aborts.append(msg)
        sys.argv = ['auto_trade_binance.py', '--trade']
        a.CRON_START_JITTER_SECONDS = (0, 0)
        a.main()
    finally:
        for k, v in saved.items():
            setattr(a, k, v)
        sys.argv = saved_argv

    c = client_box.get('c')
    return (c.order_calls() if c else ['<클라이언트 생성 안 됨>']), aborts


with tempfile.TemporaryDirectory() as TMP:
    print("\n[1] 메타 실패 — 텔레그램 정상")
    orders, aborts = run_trade_with_meta_failure(TMP, telegram_explodes=False)
    check("주문 API 호출 0회", orders == [], str(orders))
    check("ABORT 를 기록에 남긴다", len(aborts) == 1, str(aborts))
    check("기록에 차단 사유가 들어간다",
          bool(aborts) and '심볼정보' in aborts[0], str(aborts))

    print("\n[2] 메타 실패 — 텔레그램까지 터짐")
    orders, aborts = run_trade_with_meta_failure(TMP, telegram_explodes=True)
    check("주문 API 호출 0회", orders == [], str(orders))
    check("알림이 터져도 ABORT 기록은 남는다", len(aborts) == 1, str(aborts))

print()
if FAILED:
    print(f"{len(FAILED)} FAIL: {FAILED}")
    sys.exit(1)
print("ALL PASS")
