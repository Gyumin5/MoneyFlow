"""선물 청산 차단 버그(2026-08-15) 수정 오프라인 검증.

네트워크·라이브 state 미사용. fut_trade_gate 분기 + execute_rebalance 청산 불변식만 본다.
실행: cd /home/gmoh/mon/251229 && python3 tests/test_fut_liquidation_only.py
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'trade'))
import auto_trade_binance as m  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILS.append(name)


# ── 1. 게이트 분기 ───────────────────────────────────────────────────────────
print("[1] fut_trade_gate 분기")

# 사고 재현 케이스: 카나리 OFF → 전액 CASH, lev_map 정상적으로 빔
blocked, liq, reason = m.fut_trade_gate(False, {}, {'CASH': 1.0})
check("카나리 OFF 전액CASH → 통과 + liquidation_only", (not blocked) and liq, f"({reason})")

# 데이터/봉 산출 실패는 여전히 차단
blocked, liq, reason = m.fut_trade_gate(True, {}, {'CASH': 1.0})
check("lev_abort=True → 차단", blocked and not liq)
blocked, liq, reason = m.fut_trade_gate(True, {}, {'TRX': 1 / 3, 'CASH': 2 / 3})
check("lev_abort=True (코인 목표) → 차단", blocked and not liq)

# 목표 코인이 있는데 맵이 비면 차단 (구 가드의 정당한 용도)
blocked, liq, reason = m.fut_trade_gate(False, {}, {'TRX': 1 / 3, 'CASH': 2 / 3})
check("목표 코인 있는데 맵 빔 → 차단", blocked and not liq)

# 부분 일치(한 코인 누락)도 차단 — 부분 universe 매매 금지
blocked, liq, reason = m.fut_trade_gate(
    False, {'TRX': 2}, {'TRX': 1 / 3, 'XMR': 1 / 3, 'CASH': 1 / 3})
check("맵 키 부분누락 → 차단", blocked and not liq)

# 정상 케이스
blocked, liq, reason = m.fut_trade_gate(
    False, {'TRX': 2, 'XMR': 2}, {'TRX': 1 / 3, 'XMR': 1 / 3, 'CASH': 1 / 3})
check("정상 목표 + 맵 일치 → 통과, liquidation_only=False",
      (not blocked) and (not liq), f"({reason})")

# 비-CASH 비중이 남아있는데 코인 목록이 비는 기형 입력은 청산으로 보지 않는다
blocked, liq, reason = m.fut_trade_gate(False, {}, {'TRX': -0.0, 'CASH': 0.5, 'XXX': 0.5})
check("비-CASH 잔여비중 기형 입력 → 청산 아님(차단)", blocked and not liq)


# ── 2. execute_rebalance 청산 불변식 ────────────────────────────────────────
print("[2] execute_rebalance(liquidation_only=True) 불변식")

POSITIONS = {  # 사고 당시 4종 (2026-08-15 09:05 로그)
    'TRX': dict(symbol='TRXUSDT', notional=40183.0, qty=121000.0, qty_raw='121000', pnl=752.07),
    'UNI': dict(symbol='UNIUSDT', notional=7443.0, qty=2140.0, qty_raw='2140', pnl=-2004.89),
    'XMR': dict(symbol='XMRUSDT', notional=8156.0, qty=25.4, qty_raw='25.4', pnl=631.35),
    'HYPE': dict(symbol='HYPEUSDT', notional=31752.0, qty=1112.0, qty_raw='1112', pnl=-3142.74),
}


class FakeClient:
    def futures_symbol_ticker(self, symbol):
        raise AssertionError(f"청산 경로에서 가격 조회 발생 ({symbol}) — 매수 경로 진입 의심")


def run_exec(target, lev_map, liquidation_only):
    orders = []
    saved = {k: getattr(m, k) for k in
             ('get_current_positions', 'cancel_stop_orders', 'create_order_with_retry',
              'get_symbol_constraints', 'format_quantity')}
    m.get_current_positions = lambda c: (dict(POSITIONS), 60524.0, True)
    m.cancel_stop_orders = lambda c, syms: None
    m.get_symbol_constraints = lambda c, s: {'min_qty': 0.001, 'min_notional': 5.0}
    m.format_quantity = lambda c, s, q: str(q)
    m.create_order_with_retry = lambda c, p: orders.append(p) or {'status': 'NEW'}
    try:
        m.execute_rebalance(FakeClient(), target, 60524.0, lev_map,
                            liquidation_only=liquidation_only)
    finally:
        for k, v in saved.items():
            setattr(m, k, v)
    return orders


orders = run_exec({'CASH': 1.0}, {}, True)
check("주문 4건 생성", len(orders) == 4, f"(got {len(orders)})")
check("전부 SELL", all(o['side'] == 'SELL' for o in orders))
check("전부 reduceOnly", all(o.get('reduceOnly') == 'true' for o in orders))
check("전부 MARKET", all(o['type'] == 'MARKET' for o in orders))
check("심볼 4종 일치",
      sorted(o['symbol'] for o in orders) == sorted(p['symbol'] for p in POSITIONS.values()))
check("수량 = 보유 전량",
      all(float(o['quantity']) == abs(POSITIONS[o['symbol'][:-4]]['qty']) for o in orders))

# 불변식 위반 주입: liquidation_only 인데 목표에 코인이 섞여 있어도 매수는 안 나가야 한다
orders2 = run_exec({'CASH': 0.5, 'TRX': 0.5}, {'TRX': 2}, True)
check("liquidation_only 시 매수 주문 0건", all(o['side'] == 'SELL' for o in orders2))

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL PASS")
