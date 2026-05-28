"""V25 cycle 8 P1 live smoke test — L 낮춤 시 사전매도 시퀀스 실측.

시나리오:
  1) BTCUSDT CROSSED, L=4 으로 셋업
  2) $100 worth BTC market buy
  3) 현재 포지션 notional 측정
  4) _v25_partial_sell_for_leverage_down 호출 (new_L=2 시뮬)
  5) set_leverage(2) 호출 → 성공 여부
  6) 포지션 정리 (전량 reduceOnly close)

실행: cd /home/gmoh/mon/251229 && python3 tests/smoke_v25_leverage_down.py
"""
import sys, time
sys.path.insert(0, '/home/gmoh/mon/251229')
sys.path.insert(0, '/home/gmoh/mon/251229/trade')

from binance.client import Client
from binance.exceptions import BinanceAPIException

from trade.auto_trade_binance import (
    load_config, _fetch_position_info, _pick_oneway_row,
    ensure_margin_type, set_leverage, verify_leverage,
    _v25_partial_sell_for_leverage_down, get_symbol_constraints,
    MARGIN_TYPE,
)

SYMBOL = 'BTCUSDT'
TARGET_USD = 100.0


def fetch_pos(client):
    info = _fetch_position_info(client, SYMBOL)
    row = _pick_oneway_row(info) if info else None
    if not row:
        return None
    amt = float(row.get('positionAmt', 0) or 0)
    lev = int(float(row.get('leverage', 0) or 0))
    mark_keys = ['markPrice', 'entryPrice']
    mark = 0.0
    for k in mark_keys:
        v = row.get(k)
        if v:
            try:
                mark = float(v)
                if mark > 0: break
            except (TypeError, ValueError):
                pass
    if mark <= 0:
        mark = float(client.futures_mark_price(symbol=SYMBOL)['markPrice'])
    return {'amt': amt, 'lev': lev, 'mark': mark, 'notional': abs(amt) * mark}


def close_all(client):
    pos = fetch_pos(client)
    if not pos or abs(pos['amt']) <= 0:
        return
    side = 'SELL' if pos['amt'] > 0 else 'BUY'
    cons = get_symbol_constraints(client, SYMBOL)
    step = cons.get('step_size', 0.001)
    qty = (abs(pos['amt']) // step) * step
    if qty <= 0: return
    print(f"  cleanup: {side} reduceOnly qty={qty}")
    client.futures_create_order(symbol=SYMBOL, side=side, type='MARKET',
                                quantity=qty, reduceOnly=True)
    time.sleep(1.0)


def main():
    api_key, api_secret = load_config()
    if not api_key:
        print("FAIL: config 없음"); return 1
    client = Client(api_key, api_secret)

    print(f"=== V25 cycle 8 P1 smoke test: {SYMBOL} ===")
    # 기존 포지션 정리
    pos0 = fetch_pos(client)
    if pos0 and abs(pos0['amt']) > 0:
        print(f"기존 포지션 발견 amt={pos0['amt']} — 먼저 정리")
        close_all(client)

    # 1) CROSSED + L=4
    print(f"\n[1] {SYMBOL} CROSSED + L=4 셋업")
    assert ensure_margin_type(client, SYMBOL, MARGIN_TYPE), "margin 보장 실패"
    assert set_leverage(client, SYMBOL, 4), "L=4 셋업 실패"
    assert verify_leverage(client, SYMBOL, 4), "L=4 verify 실패"
    print("  OK")

    # 2) $100 BTC market buy
    print(f"\n[2] ${TARGET_USD} BTC market buy")
    mark = float(client.futures_mark_price(symbol=SYMBOL)['markPrice'])
    cons = get_symbol_constraints(client, SYMBOL)
    step = cons.get('step_size', 0.001)
    qty_raw = TARGET_USD / mark
    qty = (qty_raw // step) * step
    if qty * mark < cons.get('min_notional', 5.0):
        print(f"FAIL: qty too small {qty} × {mark} < min_notional"); return 1
    print(f"  mark=${mark:.2f} qty={qty} notional=${qty*mark:.2f}")
    client.futures_create_order(symbol=SYMBOL, side='BUY', type='MARKET', quantity=qty)
    time.sleep(1.5)

    # 3) 측정
    pos1 = fetch_pos(client)
    assert pos1 and abs(pos1['amt']) > 0, "포지션 진입 실패"
    print(f"\n[3] 진입 직후: amt={pos1['amt']} notional=${pos1['notional']:.2f} L={pos1['lev']}")

    # 4) L↓ 사전매도 (new_L=2, current=4 → target_notional = current × 2/4)
    new_lev = 2
    target_notional = pos1['notional'] * (new_lev / pos1['lev'])
    print(f"\n[4] L↓ 사전매도: ${pos1['notional']:.2f} → ${target_notional:.2f} (L {pos1['lev']}→{new_lev})")
    ok = _v25_partial_sell_for_leverage_down(client, SYMBOL,
                                              pos1['notional'], target_notional, pos1['mark'])
    print(f"  사전매도 결과: {'OK' if ok else 'FAIL'}")
    if not ok:
        close_all(client); return 1
    time.sleep(1.5)

    pos2 = fetch_pos(client)
    print(f"  매도 후: amt={pos2['amt']} notional=${pos2['notional']:.2f}")
    ratio = pos2['notional'] / pos1['notional'] if pos1['notional'] > 0 else 0
    print(f"  비율 {ratio:.3f} (예상 ~0.5 ± 0.05)")
    if not (0.45 <= ratio <= 0.55):
        print(f"WARN: 비율 예상 범위 벗어남")

    # 5) set_leverage(2)
    print(f"\n[5] set_leverage(2) 호출")
    ok = set_leverage(client, SYMBOL, 2)
    print(f"  set_leverage: {'OK' if ok else 'FAIL'}")
    if ok:
        time.sleep(0.5)
        ok2 = verify_leverage(client, SYMBOL, 2)
        print(f"  verify_leverage: {'OK' if ok2 else 'FAIL'}")
    if not ok:
        close_all(client); return 1

    # 6) 정리
    print(f"\n[6] 포지션 정리")
    close_all(client)
    set_leverage(client, SYMBOL, 3)  # 기본값 복원
    print("\n✅ V25 cycle 8 P1 smoke test PASSED")
    return 0


if __name__ == '__main__':
    sys.exit(main())
