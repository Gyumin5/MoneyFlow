"""선물 포지션 스냅샷 정합 검사 (2026-09-27).

문제: get_current_positions 는 예외가 안 나면 그 응답을 사실로 받았다. 두 갈래가 위험하다.
  (a) 포지션 행이 빈 배열로 오는데 계좌 합계는 포지션 증거금을 들고 있다.
      비었다고 믿으면 드리프트가 현금 100% 로 보이고, 이미 든 것 위에 목표를 또 산다.
      상시 무결성 검사가 보통은 잡지만 기준 스냅샷이 없는 구간(최초 실행, 주문 뒤
      사후조회 실패로 무효화된 다음 실행)에는 검사가 생략돼 구멍이 남는다.
  (b) 총자산이 0 으로 온다. 목표 명목이 전부 0 이 되어 반대로 전량청산이 된다.

수정: 두 경우를 조회 실패(ok=False)로 돌린다. 호출부는 pos_ok 가 False 면 그 실행을
거래 없이 건너뛴다. 재시도는 기존 조회 재시도 횟수를 그대로 쓴다.

주의: 문턱은 포지션만의 증거금으로 잰다. totalInitialMargin 은 미체결 주문 증거금까지
더해서 오므로, 그걸 쓰면 포지션이 정말 없고 주문만 걸린 계좌를 모순으로 오판한다.

네트워크·라이브 state 미사용 (fake client).
실행: cd /home/gmoh/mon/251229 && python3 tests/test_fut_position_snapshot_consistency.py
"""
import os
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'trade'))
import auto_trade_binance as a  # noqa: E402

FAILS = []


def check(name, cond, detail: object = ''):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILS.append(name)


# 2026-09-27 서버 실계좌 읽기전용 조회에서 받은 실제 값 형태.
# leverage 가 None 으로 오고 코드가 명목/증거금으로 되살린다는 점까지 그대로 옮겼다.
REAL_ROWS = [
    {'symbol': 'BNBUSDT', 'positionAmt': '63.63', 'positionSide': 'BOTH',
     'entryPrice': '700.0', 'markPrice': '772.6', 'unRealizedProfit': '4620.0',
     'liquidationPrice': '0', 'notional': '49160.5', 'leverage': None,
     'positionInitialMargin': '24580.3', 'marginType': 'cross'},
    {'symbol': 'ETHUSDT', 'positionAmt': '9.044', 'positionSide': 'BOTH',
     'entryPrice': '2500.0', 'markPrice': '2692.6', 'unRealizedProfit': '1741.0',
     'liquidationPrice': '0', 'notional': '24352.2', 'leverage': None,
     'positionInitialMargin': '12176.1', 'marginType': 'cross'},
]
REAL_ACCOUNT = {
    'totalWalletBalance': '81780.38793994',
    'totalUnrealizedProfit': '10037.81595446',
    'totalPositionInitialMargin': '36756.40',
    'totalOpenOrderInitialMargin': '0.00000000',
    'totalInitialMargin': '36756.40',
    'totalMaintMargin': '958.97289301',
}
TICKERS = [{'symbol': 'BNBUSDT', 'price': '772.6'},
           {'symbol': 'ETHUSDT', 'price': '2692.6'}]


class FakeClient:
    def __init__(self, account, rows):
        self.account = account
        self.rows = rows
        self.account_calls = 0
        self.row_calls = 0

    def futures_account(self):
        self.account_calls += 1
        return self.account

    def futures_position_information(self, symbol=None):
        self.row_calls += 1
        return self.rows

    def futures_symbol_ticker(self):
        return TICKERS


def acct(**over):
    out = dict(REAL_ACCOUNT)
    out.update(over)
    return out


print("== 정상 응답은 그대로 통과한다 ==")
c = FakeClient(REAL_ACCOUNT, REAL_ROWS)
pos, pv, ok = a.get_current_positions(c)
check("정상: ok=True", ok is True)
check("정상: 두 종목 모두 읽힌다", sorted(pos) == ['BNB', 'ETH'], sorted(pos))
check("정상: 총자산 = 지갑 + 미실현", abs(pv - 91818.20389440) < 0.01, pv)
# 허용폭이 넓은 이유: 여기 적은 명목·증거금이 실계좌 값을 소수 첫째 자리로 줄여 옮긴 것이라
# 되살린 배수에 그 반올림이 그대로 남는다. 배수를 정확히 재는 시험이 아니라 None 이 와도
# 0 으로 주저앉지 않는다는 것을 재는 시험이다.
check("정상: leverage None 이어도 명목/증거금으로 되살린다",
      abs(pos['BNB']['leverage'] - 2.0) < 1e-3, pos['BNB']['leverage'])
check("정상: 재시도 없이 1회 조회", c.row_calls == 1, c.row_calls)

print("\n== 포지션 행이 비었는데 계좌는 증거금을 들고 있다 → 조회 실패 ==")
c = FakeClient(REAL_ACCOUNT, [])
pos, pv, ok = a.get_current_positions(c)
check("빈 행 + 증거금: ok=False", ok is False)
check("빈 행 + 증거금: 보유 0 을 사실로 돌려주지 않는다", pos == {} and pv == 0.0, (pos, pv))
check("빈 행 + 증거금: 재시도 횟수만큼 다시 읽는다",
      c.row_calls == a.POSITION_FETCH_MAX_RETRIES, c.row_calls)

print("\n== 행은 있는데 전부 수량 0 이고 계좌는 증거금을 들고 있다 → 조회 실패 ==")
zero_rows = [dict(r, positionAmt='0', notional='0', positionInitialMargin='0')
             for r in REAL_ROWS]
c = FakeClient(REAL_ACCOUNT, zero_rows)
pos, pv, ok = a.get_current_positions(c)
check("수량 0 행 + 증거금: ok=False", ok is False)
check("수량 0 행 + 증거금: 빈 보유를 사실로 안 받는다", pos == {}, pos)

print("\n== 정말 아무것도 없는 계좌는 통과한다 ==")
flat = acct(totalPositionInitialMargin='0', totalInitialMargin='0',
            totalMaintMargin='0', totalUnrealizedProfit='0',
            totalWalletBalance='50000.0')
c = FakeClient(flat, [])
pos, pv, ok = a.get_current_positions(c)
check("전액현금: ok=True", ok is True)
check("전액현금: 보유 없음", pos == {}, pos)
check("전액현금: 총자산 그대로", abs(pv - 50000.0) < 0.01, pv)
check("전액현금: 재시도 안 한다", c.row_calls == 1, c.row_calls)

print("\n== 포지션은 없고 미체결 주문 증거금만 있는 계좌도 통과한다 ==")
# totalInitialMargin 으로 재면 여기서 오탐이 난다. 포지션 증거금만 봐야 한다.
open_only = acct(totalPositionInitialMargin='0', totalMaintMargin='0',
                 totalOpenOrderInitialMargin='8000.0', totalInitialMargin='8000.0',
                 totalUnrealizedProfit='0', totalWalletBalance='50000.0')
c = FakeClient(open_only, [])
pos, pv, ok = a.get_current_positions(c)
check("미체결 주문만: ok=True", ok is True)
check("미체결 주문만: 재시도 안 한다", c.row_calls == 1, c.row_calls)

print("\n== 먼지 수준 증거금은 모순으로 보지 않는다 ==")
dust = acct(totalPositionInitialMargin='0.4', totalMaintMargin='0.02',
            totalInitialMargin='0.4', totalUnrealizedProfit='0',
            totalWalletBalance='50000.0')
c = FakeClient(dust, [])
pos, pv, ok = a.get_current_positions(c)
check("먼지 증거금: ok=True", ok is True)
check("먼지 문턱 값 = 1.0 USDT", a.POSITION_CONSISTENCY_MARGIN_FLOOR == 1.0,
      a.POSITION_CONSISTENCY_MARGIN_FLOOR)

print("\n== 유지증거금만 잡히는 경우도 모순으로 본다 ==")
maint_only = acct(totalPositionInitialMargin='0', totalMaintMargin='958.97',
                  totalInitialMargin='0')
c = FakeClient(maint_only, [])
pos, pv, ok = a.get_current_positions(c)
check("유지증거금만: ok=False", ok is False)

print("\n== 총자산이 0 이하면 조회 실패 ==")
for label, over in [
    ("총자산 0", {'totalWalletBalance': '0', 'totalUnrealizedProfit': '0',
                  'totalPositionInitialMargin': '0', 'totalMaintMargin': '0'}),
    ("총자산 음수", {'totalWalletBalance': '100', 'totalUnrealizedProfit': '-200',
                     'totalPositionInitialMargin': '0', 'totalMaintMargin': '0'}),
    ("합계만 0 이고 포지션은 있음", {'totalWalletBalance': '0', 'totalUnrealizedProfit': '0'}),
]:
    c = FakeClient(acct(**over), REAL_ROWS if '포지션은 있음' in label else [])
    pos, pv, ok = a.get_current_positions(c)
    check(f"{label}: ok=False", ok is False)
    check(f"{label}: 총자산 0 을 사실로 안 돌려준다", pv == 0.0, pv)

print("\n== 전량청산을 막는가 — 총자산 0 이 목표 명목 0 으로 번지지 않는다 ==")
c = FakeClient(acct(totalWalletBalance='0', totalUnrealizedProfit='0'), REAL_ROWS)
pos, pv, ok = a.get_current_positions(c)
check("총자산 0 + 보유 있음: 호출부가 거래를 건너뛴다(ok=False)", ok is False)
check("총자산 0 + 보유 있음: 보유를 0 으로 넘겨주지 않는다", pos == {}, pos)

print("\n== 계좌 목록과 상세 조회의 보유 심볼이 어긋나면 조회 실패 ==")
# 합계만 보면 못 잡는 경우다 — 세 종목 중 하나만 빠져도 증거금 합계는 여전히 크다.
# 그래서 futures_account 의 포지션 행과 상세 조회를 심볼 단위로 대조한다.
ACCT_ROWS_3 = [
    {'symbol': 'BNBUSDT', 'positionAmt': '63.63', 'leverage': '2'},
    {'symbol': 'ETHUSDT', 'positionAmt': '9.044', 'leverage': '2'},
    {'symbol': 'SOLUSDT', 'positionAmt': '597.65', 'leverage': '3'},
    {'symbol': 'XRPUSDT', 'positionAmt': '0', 'leverage': '2'},
]
c = FakeClient(acct(positions=ACCT_ROWS_3), REAL_ROWS)  # 상세에는 SOL 이 없다
pos, pv, ok = a.get_current_positions(c)
check("일부 행 누락: ok=False", ok is False)
check("일부 행 누락: 남은 두 종목을 사실로 돌려주지 않는다", pos == {} and pv == 0.0, (pos, pv))
check("일부 행 누락: 재시도 횟수만큼 다시 읽는다",
      c.row_calls == a.POSITION_FETCH_MAX_RETRIES, c.row_calls)

print("\n== 상세에만 있고 계좌 목록에 없는 경우도 조회 실패 ==")
c = FakeClient(acct(positions=[{'symbol': 'BNBUSDT', 'positionAmt': '63.63'}]), REAL_ROWS)
pos, pv, ok = a.get_current_positions(c)
check("상세 초과: ok=False", ok is False)

print("\n== 두 목록이 맞으면 통과한다 (대조가 오탐을 만들지 않는다) ==")
ok_rows = [{'symbol': 'BNBUSDT', 'positionAmt': '63.63'},
           {'symbol': 'ETHUSDT', 'positionAmt': '9.044'},
           {'symbol': 'XRPUSDT', 'positionAmt': '0'}]
c = FakeClient(acct(positions=ok_rows), REAL_ROWS)
pos, pv, ok = a.get_current_positions(c)
check("일치: ok=True", ok is True)
check("일치: 두 종목 다 읽힌다", sorted(pos) == ['BNB', 'ETH'], sorted(pos))
check("일치: 재시도 안 한다", c.row_calls == 1, c.row_calls)

print("\n== 둘 다 비었고 증거금도 없으면 통과한다 ==")
c = FakeClient(acct(positions=[{'symbol': 'BNBUSDT', 'positionAmt': '0'}],
                    totalPositionInitialMargin='0', totalMaintMargin='0',
                    totalInitialMargin='0', totalUnrealizedProfit='0',
                    totalWalletBalance='50000.0'), [])
pos, pv, ok = a.get_current_positions(c)
check("양쪽 다 빔: ok=True", ok is True)

print("\n== 계좌 포지션 행의 수량이 비정상이면 조회 실패 ==")
for label, amt in [("빈 문자열", ''), ("숫자 아님", 'null'), ("없음", None),
                   ("무한대", 'Infinity')]:
    c = FakeClient(acct(positions=[{'symbol': 'BNBUSDT', 'positionAmt': amt}]), REAL_ROWS)
    pos, pv, ok = a.get_current_positions(c)
    check(f"행 수량 {label}: ok=False", ok is False)

print("\n== 필수 계좌 수치가 없거나 비정상이면 조회 실패 ==")
# _safe_float 로 읽으면 누락이 0 이 되어 "총자산 0" 과 구분되지 않고,
# 증거금 누락은 모순 검사를 통째로 무력화한다. 그래서 값 자체를 검사한다.
REQUIRED_KEYS = ['totalWalletBalance', 'totalUnrealizedProfit',
                 'totalPositionInitialMargin', 'totalMaintMargin']
for key in REQUIRED_KEYS:
    missing = acct()
    missing.pop(key)
    c = FakeClient(missing, REAL_ROWS)
    pos, pv, ok = a.get_current_positions(c)
    check(f"{key} 누락: ok=False", ok is False)
    check(f"{key} 누락: 보유·총자산을 사실로 안 돌려준다", pos == {} and pv == 0.0, (pos, pv))

for label, bad in [("빈 문자열", ''), ("None", None), ("숫자 아님", 'n/a'),
                   ("NaN", 'NaN'), ("무한대", 'Infinity')]:
    c = FakeClient(acct(totalWalletBalance=bad), REAL_ROWS)
    pos, pv, ok = a.get_current_positions(c)
    check(f"지갑잔고 {label}: ok=False", ok is False)
    c = FakeClient(acct(totalPositionInitialMargin=bad), [])
    pos, pv, ok = a.get_current_positions(c)
    check(f"포지션증거금 {label}: ok=False", ok is False)

print("\n== 미실현손익 0 은 정상값이다 (누락과 구분한다) ==")
c = FakeClient(acct(totalUnrealizedProfit='0'), REAL_ROWS)
pos, pv, ok = a.get_current_positions(c)
check("미실현 0: ok=True", ok is True)
check("미실현 0: 총자산 = 지갑", abs(pv - 81780.38793994) < 0.01, pv)

print("\n== 조회 예외는 기존 경로 그대로 ==")


class RaisingClient(FakeClient):
    def __init__(self):
        super().__init__(REAL_ACCOUNT, REAL_ROWS)

    def futures_position_information(self, symbol=None):
        self.row_calls += 1
        raise RuntimeError("connection reset by peer")


c = RaisingClient()
pos, pv, ok = a.get_current_positions(c)
check("네트워크 예외: ok=False", ok is False)
check("네트워크 예외: 재시도 횟수 유지",
      c.row_calls == a.POSITION_FETCH_MAX_RETRIES, c.row_calls)

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL PASS")
