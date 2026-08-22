"""선물 V25 margin preflight 검사 범위 회귀 검증 (2026-08-22 오ABORT 사고).

사고: 08-22 09:05 cron 이 앵커 갱신으로 BTC/ETH 를 새로 뽑았다. ETHUSDT 는 V24 시절 설정이
ISOLATED 로 남아 있었고(포지션 0) 코드는 "ETH 마진모드를 CROSSED 로 바꿔야 한다"고 정확히
판단했다. 그런데 preflight 를 대상 심볼 전체(BNB SOL BTC ETH)에 걸어서, 정상 보유 중인
SOLUSDT 포지션(amt=926.03, $86,739)에 막혀 매매 전체가 ABORT 됐다. BNB +18.5% 매수가
주문 0건으로 끝나고 abort_streak 1 이 됐다(3연속이면 lock).

Binance 규칙: 코인별 margin type 변경은 "그 심볼"만 포지션·미체결 0 이면 된다.
따라서 preflight 범위는 "변경이 실제로 필요한 심볼"이어야 하고, 대상 전체가 아니다.
같은 계열의 선행 사고: 07-15 무관 심볼 먼지 게이트, 08-15 빈 lev_map 청산차단.

네트워크·라이브 state 미사용 (fake client).
실행: cd /home/gmoh/mon/251229 && python3 tests/test_fut_margin_preflight_scope.py
"""
import os
import re
import sys

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


# 2026-08-22 09:05 실제 계좌 상태 (조회값 그대로)
ACCOUNT_0822 = {'positions': [
    {'symbol': 'BNBUSDT', 'positionAmt': '94.38', 'notional': '64725.80', 'isolated': False, 'leverage': '3'},
    {'symbol': 'SOLUSDT', 'positionAmt': '926.03', 'notional': '86739.39', 'isolated': False, 'leverage': '4'},
    {'symbol': 'BTCUSDT', 'positionAmt': '0.000', 'notional': '0', 'isolated': False, 'leverage': '3'},
    {'symbol': 'ETHUSDT', 'positionAmt': '0.000', 'notional': '0', 'isolated': True, 'leverage': '3'},
    # 무관 심볼 먼지 (07-15 결정: tolerate)
    {'symbol': 'TONUSDT', 'positionAmt': '0.5', 'notional': '1.20', 'isolated': True, 'leverage': '3'},
]}


class FakeClient:
    def __init__(self, account, open_orders=None, algo_orders=None, fail_syms=()):
        self._acc = account
        self._orders = open_orders or []
        self._algo = algo_orders
        self._fail = set(fail_syms)
        self.calls = []

    def futures_account(self):
        self.calls.append('account')
        if self._fail:
            # 조회 장애 시늉: 실패 심볼 row 를 지운 응답
            return {'positions': [r for r in self._acc['positions']
                                  if r['symbol'] not in self._fail]}
        return self._acc

    def futures_get_open_orders(self):
        return list(self._orders)

    def futures_get_open_algo_orders(self):
        if self._algo is None:
            raise a.BinanceAPIException(_FakeResp(), 400, '{"code":-1121,"msg":"unsupported"}')
        return {'orders': self._algo}


class _FakeResp:
    status_code = 400
    text = '{"code":-1121,"msg":"unsupported"}'

    def json(self):
        return {'code': -1121, 'msg': 'unsupported'}


def main():
    print(__doc__.splitlines()[0])

    print("\n[1] preflight 범위 — 사고 재현과 수정 후 동작")
    c = FakeClient(ACCOUNT_0822)
    # 사고 당시 호출: 대상 심볼 전체 → SOL 포지션에 걸려 False (=매매 전체 차단)
    check("대상 전체를 넘기면 여전히 False (사고 재현)",
          a.preflight_target_symbols_zero(
              c, ['BNBUSDT', 'SOLUSDT', 'BTCUSDT', 'ETHUSDT']) is False)
    # 수정 후 호출: 변경이 필요한 심볼만 → ETH 는 zero 이므로 True (=매매 진행)
    check("변경 필요 심볼(ETHUSDT)만 넘기면 True",
          a.preflight_target_symbols_zero(c, ['ETHUSDT']) is True)
    check("BTC/ETH 둘만 넘겨도 True (둘 다 포지션 0)",
          a.preflight_target_symbols_zero(c, ['BTCUSDT', 'ETHUSDT']) is True)

    print("\n[2] 좁혀도 막아야 하는 것은 막는다")
    check("변경 필요 심볼 자신에 포지션이 있으면 False",
          a.preflight_target_symbols_zero(c, ['SOLUSDT']) is False)
    c_ord = FakeClient(ACCOUNT_0822, open_orders=[{'symbol': 'ETHUSDT', 'orderId': 1}])
    check("변경 필요 심볼에 미체결 주문 있으면 False",
          a.preflight_target_symbols_zero(c_ord, ['ETHUSDT']) is False)
    c_ord2 = FakeClient(ACCOUNT_0822, open_orders=[{'symbol': 'BNBUSDT', 'orderId': 2}])
    check("무관 심볼 미체결 주문은 통과 (범위 밖)",
          a.preflight_target_symbols_zero(c_ord2, ['ETHUSDT']) is True)
    check("무관 심볼 먼지(TON $1.20)는 여전히 tolerate",
          a.preflight_target_symbols_zero(c, ['ETHUSDT']) is True)

    print("\n[3] 작업계획 — 조회 불명을 '변경 필요'와 섞지 않는다")
    plan, unknown = a.build_margin_plan(c, ['BNBUSDT', 'SOLUSDT', 'BTCUSDT', 'ETHUSDT'],
                                        'CROSSED', retries=0)
    check("계획이 4심볼 전부 담긴다", sorted(plan) == ['BNBUSDT', 'BTCUSDT', 'ETHUSDT', 'SOLUSDT'])
    check("ETH 만 변경 필요", sorted(s for s, v in plan.items() if v['need']) == ['ETHUSDT'])
    check("보유 코인은 cross 로 판정", plan['SOLUSDT']['cur'] == 'cross' and not plan['SOLUSDT']['need'])
    check("불명 없음", unknown == [])

    c_fail = FakeClient(ACCOUNT_0822, fail_syms={'BTCUSDT'})
    plan2, unknown2 = a.build_margin_plan(c_fail, ['SOLUSDT', 'BTCUSDT'], 'CROSSED',
                                          retries=1, sleep_s=0)
    check("조회 불명 심볼은 unknown 으로 분리", unknown2 == ['BTCUSDT'])
    check("불명 심볼은 변경 필요 목록에 안 들어감",
          [s for s, v in plan2.items() if v['need']] == [])
    check("불명 심볼은 계획에서도 빠짐", 'BTCUSDT' not in plan2)
    check("불명 판정 전 재시도한다", c_fail.calls.count('account') >= 3)

    print("\n[4] algo/조건부 주문")
    c_algo = FakeClient(ACCOUNT_0822, algo_orders=[{'symbol': 'ETHUSDT', 'algoId': 7}])
    check("변경 필요 심볼에 algo 주문 있으면 False",
          a.preflight_target_symbols_zero(c_algo, ['ETHUSDT']) is False)
    c_algo2 = FakeClient(ACCOUNT_0822, algo_orders=[{'symbol': 'BNBUSDT', 'algoId': 8}])
    check("무관 심볼 algo 주문은 통과",
          a.preflight_target_symbols_zero(c_algo2, ['ETHUSDT']) is True)
    check("algo 조회 미지원(예외)이면 일반 주문만 보고 통과",
          a.preflight_target_symbols_zero(c, ['ETHUSDT']) is True)

    print("\n[5] 소스 불변식 — 호출부가 좁힌 집합을 쓰는가")
    src = open(os.path.join(_HERE, '..', 'trade', 'auto_trade_binance.py')).read()
    check("작업계획을 build_margin_plan 으로 한 번 만든다",
          'margin_plan, margin_unknown = build_margin_plan(client, target_symbols, MARGIN_TYPE)' in src)
    check("preflight 는 margin_change_syms 로 호출",
          re.search(r'preflight_target_symbols_zero\(client, margin_change_syms\)', src) is not None)
    check("preflight 를 target_symbols 로 호출하는 경로 없음",
          re.search(r'preflight_target_symbols_zero\(client, target_symbols\)', src) is None)
    check("need_margin_change 는 계획 결과로만 결정",
          'need_margin_change = bool(margin_change_syms)' in src)
    check("조회 불명은 별도 ABORT 사유",
          '마진모드 조회 불명' in src)
    check("one-way 모드 preflight 는 계정 전체 엄격 유지",
          'preflight_zero_positions(client)' in src and 'preflight_zero_open_orders(client)' in src)

    print("\n[6] 하류 경로 — 계획과 같은 판정을 쓰는가")
    check("계획상 변경 불필요면 set 안 하고 검증만",
          re.search(r"if _mplan is not None and not _mplan\['need'\]:\s*\n\s*if not verify_margin_type",
                    src) is not None)
    check("계획 불일치는 ABORT", '작업계획과 불일치' in src)
    check("ensure_margin_type 은 verify 선행",
          re.search(r'def ensure_margin_type[\s\S]{0,200}?if verify_margin_type', src) is not None)
    check("사전매도는 마진 변경 대기 심볼을 건너뛴다",
          re.search(r'if sym in margin_change_syms:\s*\n\s*continue', src) is not None)

    print()
    if FAILS:
        print(f"FAIL {len(FAILS)} 건: {FAILS}")
        return 1
    print("전 항목 PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
