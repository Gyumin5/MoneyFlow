"""L 낮춤 사전매도 수량 정밀도 회귀 — 2026-09-24 실사고.

사고: 2026-09-24 09:05 정규 실행에서 BNBUSDT 배수 3x → 2x 사전매도가
  code=-1111 Precision is over the maximum defined for this asset
로 거절돼 V25 ABORT(매매 차단)가 났다. 그날 포지션은 목표의 세 배 가까이 남았고
중단 연속 카운트가 1 로 올라갔다(3 이면 선물 슬리브 자동 잠금).

원인: 사전매도 수량을 `(raw // step) * step` 으로 만들어 부동소수 곱의 찌꺼기를
그대로 보냈다. step=0.01 에서 3181.0 * 0.01 = 31.810000000000002 이고, 거래소는
소수 두 자리만 받는 심볼에서 이를 거절한다. 본 주문 경로는 같은 자리에서
format_quantity 로 문자열까지 만들어 보내므로 이 결함이 없었다.

수정: 사전매도도 format_quantity 를 쓰고 문자열을 그대로 보낸다.

네트워크·라이브 state 미사용. 주문은 가짜 클라이언트가 받아 기록만 한다.
실행: cd /home/gmoh/mon/251229 && python3 tests/test_fut_presell_precision.py
"""
import os
import sys
from decimal import Decimal

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


class FakeClient:
    """주문을 받아 적기만 하는 클라이언트."""

    def __init__(self, fail_code=None, amt=1000.0):
        self.orders = []
        self.fail_code = fail_code
        self.amt = amt

    def futures_position_information(self, symbol=None):
        # 사전매도는 주문 전에 기준 수량을 읽는다. 못 읽으면 주문을 아예 안 보내므로
        # 정밀도 시험이 주문 단계까지 가려면 이 응답이 있어야 한다.
        # 실패 경로에서도 값은 그대로 — 수량이 안 변했다는 것이 미접수의 증거는 아니므로
        # 재주문 없이 차단되는 것이 맞는 동작이다.
        return [{'symbol': symbol, 'positionSide': 'BOTH', 'positionAmt': str(self.amt)}]

    def futures_create_order(self, **kw):
        if self.fail_code is not None:
            raise a.BinanceAPIException(_FakeResp(self.fail_code), 400,
                                        '{"code": %d, "msg": "Precision is over the maximum defined for this asset."}'
                                        % self.fail_code)
        self.orders.append(kw)
        return {'orderId': 1, 'status': 'FILLED'}


class _FakeResp:
    def __init__(self, code):
        self.status_code = 400
        self.text = '{"code": %d, "msg": "err"}' % code


def set_constraints(step, precision, min_qty=0.01, min_notional=5.0):
    a.get_symbol_constraints = lambda client, symbol: {
        'step_size': step, 'min_qty': min_qty, 'min_notional': min_notional,
        'tick_size': 0.01, 'qty_precision': precision,
    }


def decimals(s):
    return len(s.split('.')[1]) if '.' in s else 0


# ── 1. 사고 재현값 — 그날 로그의 숫자를 그대로 넣는다 ──────────────────────
print("[1] 2026-09-24 BNBUSDT 재현")
set_constraints(step=0.01, precision=2)
a._v25_reset_touch() if hasattr(a, '_v25_reset_touch') else None
cli = FakeClient()
alerts = []
ok, why = a._v25_partial_sell_for_leverage_down(
    cli, 'BNBUSDT', current_notional=73321.65, target_notional=48881.10,
    mark_price=768.2486, order_alerts=alerts)

check("사전매도 성공", ok is True, f"({why})")
check("성공은 사유 없음", why == '', f"({why})")
check("주문 1건", len(cli.orders) == 1, f"({len(cli.orders)})")
qty = cli.orders[0]['quantity'] if cli.orders else None
check("수량이 문자열", isinstance(qty, str), f"({type(qty).__name__}: {qty!r})")
check("소수 자릿수가 규격 이내", qty is not None and decimals(qty) <= 2, f"({qty})")
check("옛 결함값이 아니다", qty != '31.810000000000002' and str(qty) != str(31.810000000000002),
      f"({qty})")
check("step 배수", qty is not None and Decimal(qty) % Decimal('0.01') == 0, f"({qty})")
check("reduceOnly", cli.orders and cli.orders[0].get('reduceOnly') is True)
check("매도 방향", cli.orders and cli.orders[0].get('side') == 'SELL')
check("order_alerts 에 같은 수량", alerts and qty in alerts[0], f"({alerts})")

raw = (73321.65 - 48881.10) / 768.2486
check("바닥내림 — 필요량을 넘지 않는다", qty is not None and float(qty) <= raw,
      f"(qty={qty} raw={raw:.6f})")
check("과소매도 아님(1 step 이내)", qty is not None and float(qty) > raw - 0.01,
      f"(qty={qty} raw={raw:.6f})")


# ── 2. 여러 규격에서 자릿수가 규격을 넘지 않는다 ───────────────────────────
print("\n[2] 규격별 자릿수")
cases = [
    ('BTCUSDT', 0.001, 3, 60123.45, 40000.0, 84500.0),
    ('ETHUSDT', 0.001, 3, 36458.31, 24000.0, 2683.41),
    ('SOLUSDT', 1.0, 0, 72964.12, 48000.0, 114.95),
    ('ZECUSDT', 0.001, 3, 23320.77, 15000.0, 1498.0),
]
for sym, step, prec, cur, tgt, mark in cases:
    set_constraints(step=step, precision=prec)
    cli = FakeClient()
    a._v25_partial_sell_for_leverage_down(cli, sym, cur, tgt, mark)
    q = cli.orders[0]['quantity'] if cli.orders else None
    check(f"{sym} 자릿수 <= {prec}", q is not None and decimals(q) <= prec, f"({q})")
    check(f"{sym} step 배수", q is not None and Decimal(q) % Decimal(str(step)) == 0, f"({q})")


# ── 3. 경계 — 매도 불필요·최소단위 미달은 주문 없이 통과 ───────────────────
print("\n[3] 경계")
set_constraints(step=0.01, precision=2)
cli = FakeClient()
ok, _ = a._v25_partial_sell_for_leverage_down(cli, 'BNBUSDT', 48881.10, 48881.10, 768.2486)
check("차이 없음 → 주문 없이 True", ok is True and not cli.orders)

cli = FakeClient()
ok, _ = a._v25_partial_sell_for_leverage_down(cli, 'BNBUSDT', 48885.0, 48881.10, 768.2486)
check("1% 버퍼 안 → 주문 없이 True", ok is True and not cli.orders)

set_constraints(step=0.01, precision=2, min_qty=0.01, min_notional=5000.0)
cli = FakeClient()
ok, _ = a._v25_partial_sell_for_leverage_down(cli, 'BNBUSDT', 50000.0, 48881.10, 768.2486)
check("min_notional 미달 → 주문 없이 True(ABORT 아님)", ok is True and not cli.orders)

set_constraints(step=0.01, precision=2)
cli = FakeClient()
ok, why = a._v25_partial_sell_for_leverage_down(cli, 'BNBUSDT', 73321.65, 48881.10, 0.0)
check("mark_price 0 → False(ABORT)", ok is False and not cli.orders)
check("mark_price 실패 사유가 남는다", 'mark_price' in why, f"({why})")

# 규격 조회 실패(step=0)는 주문을 보내지 않고 실패로 올린다. 보내면 거래소가 다시
# 정밀도로 거절해 원인이 규격 조회 실패라는 사실이 묻힌다.
set_constraints(step=0.0, precision=8, min_qty=0.0)
cli = FakeClient()
ok, why = a._v25_partial_sell_for_leverage_down(cli, 'BNBUSDT', 73321.65, 48881.10, 768.2486)
check("step 조회 실패 → 주문 없이 False", ok is False and not cli.orders,
      f"(ok={ok} orders={cli.orders})")
check("step 실패 사유가 남는다", 'step_size' in why, f"({why})")


# ── 4. 실패는 여전히 ABORT 로 올라간다 ─────────────────────────────────────
print("\n[4] 실패 경로")
set_constraints(step=0.01, precision=2)
cli = FakeClient(fail_code=-1111)
alerts = []
ok, why = a._v25_partial_sell_for_leverage_down(cli, 'BNBUSDT', 73321.65, 48881.10,
                                                768.2486, order_alerts=alerts)
check("거절 → False(호출자 ABORT)", ok is False)
check("거절 사유에 코드가 남는다", '-1111' in why, f"({why})")
check("거절 시 order_alerts 비어 있음", alerts == [], f"({alerts})")
check("거절도 계좌 변경으로 표시(체결 불명)", a._v25_account_changed() is True)


# ── 5. 소스 불변식 ─────────────────────────────────────────────────────────
print("\n[5] 소스 불변식")
_src = open(os.path.join(_HERE, '..', 'trade', 'auto_trade_binance.py'), encoding='utf-8').read()
_fn = _src.split('def _v25_partial_sell_for_leverage_down')[1].split('\ndef ')[0]
check("사전매도가 format_quantity 를 쓴다", 'format_quantity(client, sym, sell_qty_raw)' in _fn)
check("옛 부동소수 바닥내림이 사라졌다", '(sell_qty_raw // step) * step' not in _fn)
check("주문 수량이 문자열 변수", 'quantity=qty_str' in _fn)
check("reduceOnly 고정 유지", 'reduceOnly=True' in _fn)

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL PASS")
