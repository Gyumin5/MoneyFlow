"""현금 키 정규화 회귀 검증 (2026-08-20 헛주문 사고).

사고: 엔진의 'CASH'→'Cash' 정규화가 refill v2 재계산보다 앞서 있어, drift 발화일에는
combined_target 이 대문자 CASH 로 되돌아갔고 executor 가 이를 코인 티커로 오인해
"매수 오류 CASH: Code not found" 헛주문을 냈다. 업비트가 거부해 결과 비중은 우연히 맞았지만
유령 수요가 노셔널 cap·완료판정을 오염시킨다.

네트워크·라이브 state 미사용.
실행: cd /home/gmoh/mon/251229 && python3 tests/test_cash_key_normalization.py
"""
import os
import re
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, '..', 'trade'))
import executor_coin as ec  # noqa: E402
import coin_live_engine as cle  # noqa: E402

FAILS = []


def check(name, cond, detail=''):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name} {detail}")
        FAILS.append(name)


# 사고 당일(2026-08-20) 엔진이 실제로 내보낸 combined_target
BUGGY = {'LINK': 1 / 3, 'SOL': 1 / 3, 'CASH': 1 / 3}
CLEAN = {'LINK': 1 / 3, 'SOL': 1 / 3, 'Cash': 1 / 3}

# ── 1. 엔진 반환 경계 ────────────────────────────────────────────────────────
print("[1] 엔진 반환 경계 정규화")

check("combine_ensemble 은 내부 규약대로 대문자 CASH 유지",
      'CASH' in cle.combine_ensemble({'M': {'CASH': 1.0}}, {'M': 1.0}))
check("normalize_cash_key 가 대문자를 병합",
      cle.normalize_cash_key({'SOL': 0.5, 'CASH': 0.3, 'Cash': 0.2}) == {'SOL': 0.5, 'Cash': 0.5})

# refill v2 재계산이 정규화를 덮어쓰지 않는지 = 소스 순서 불변식.
# (run() 전체는 거래소·시세 접근이 필요해 오프라인 재현 불가 → 순서로 회귀를 잡는다.)
_src = open(os.path.join(_HERE, '..', 'trade', 'coin_live_engine.py')).read()
_i_refill = _src.rindex('_apply_refill_v2_to_state(state')
_i_ret = _src.rindex('return EngineResult(')
_norm_calls = [m.start() for m in re.finditer(r'combined = normalize_cash_key\(combined\)', _src)]
check("refill 재계산 이후·EngineResult 반환 이전에 정규화가 있다",
      any(_i_refill < p < _i_ret for p in _norm_calls),
      "(정규화를 refill 앞으로만 되돌리면 사고가 재발한다)")
_i_snap = _src.rindex("state['last_target_snapshot']")
check("state 스냅샷도 그 정규화 이후에 기록된다",
      any(p < _i_snap < _i_ret for p in _norm_calls))

# ── 2. executor 정규화 유틸 ─────────────────────────────────────────────────
print("[2] _norm_cash_map / targets_equal")

check("대문자 CASH → Cash", ec._norm_cash_map(BUGGY) == CLEAN)
check("혼재 키 합산", ec._norm_cash_map({'CASH': 0.3, 'Cash': 0.2, 'cash': 0.1})['Cash'] == 0.6)
check("메타키 _ts 제외", '_ts' not in ec._norm_cash_map({'Cash': 1.0, '_ts': '2026-08-20T00:00:00Z'}))
check("비수치 값 제외", ec._norm_cash_map({'Cash': 1.0, 'X': None}) == {'Cash': 1.0})
check("합 보존", abs(sum(ec._norm_cash_map(BUGGY).values()) - 1.0) < 1e-12)

check("표기만 다른 목표는 동일 판정(허위 target_changed 방지)", ec.targets_equal(BUGGY, CLEAN))
check("메타키 붙은 legacy 스냅샷도 동일 판정",
      ec.targets_equal(CLEAN, {**BUGGY, '_ts': '2026-08-19T00:00:00Z'}))
check("실제로 다른 목표는 다르게 판정",
      not ec.targets_equal(CLEAN, {'LINK': 0.5, 'SOL': 0.5}))
check("빈 목표는 같다고 보지 않음", not ec.targets_equal({}, CLEAN))

# ── 3. cash buffer / notional cap ───────────────────────────────────────────
print("[3] buffer / cap 이 유령 코인을 만들지 않는다")

buf = ec.apply_cash_buffer(BUGGY, 0.01)
check("buffer 후 CASH 키 소멸", 'CASH' not in buf, f"({sorted(buf)})")
check("buffer 후 Cash = 33.3%*0.99 + 1%", abs(buf['Cash'] - (1 / 3 * 0.99 + 0.01)) < 1e-9)
check("buffer 후 합 = 1", abs(sum(buf.values()) - 1.0) < 1e-9)
check("표기와 무관하게 같은 결과", ec.apply_cash_buffer(BUGGY, 0.01) == ec.apply_cash_buffer(CLEAN, 0.01))

BAL = {'KRW': 55_000_000.0, 'LINK': 53_400_000.0, 'SOL': 53_400_000.0}
TOTAL = sum(BAL.values())
capped_u, gross_u = ec.apply_notional_cap(BUGGY, BAL, TOTAL, 0.5)
capped_c, gross_c = ec.apply_notional_cap(CLEAN, BAL, TOTAL, 0.5)
check("cap 결과가 현금 키 표기에 좌우되지 않음", capped_u == capped_c and abs(gross_u - gross_c) < 1e-12)
check("cap 결과에 CASH 키 없음", 'CASH' not in capped_u)

# ── 4. 완료 판정 ────────────────────────────────────────────────────────────
print("[4] coin_needs_rebalance — CASH 유령 수요로 고착되지 않는다")

check("목표 도달 상태에서 대문자 목표여도 재조정 불필요",
      not ec.coin_needs_rebalance(BUGGY, BAL, TOTAL),
      "(고착되면 rebalancing_needed 가 영구 True)")
check("정상 표기와 동일 판정",
      ec.coin_needs_rebalance(BUGGY, BAL, TOTAL) == ec.coin_needs_rebalance(CLEAN, BAL, TOTAL))
check("진짜 미달이면 True 유지",
      ec.coin_needs_rebalance({'LINK': 0.5, 'SOL': 0.5}, {'KRW': TOTAL}, TOTAL))

# ── 5. 주문 생성 — 현금은 어떤 표기로도 티커가 되지 않는다 ──────────────────
print("[5] execute_delta 주문 후보 하드 가드")


class RecordingAPI:
    def __init__(self, balance):
        self._balance = dict(balance)
        self.buys = []
        self.sells = []

    def get_balance(self):
        return dict(self._balance)

    def buy_limit(self, ticker, krw):
        self.buys.append((ticker, krw))
        return True

    def sell_market_robust(self, ticker, qty):
        self.sells.append((ticker, qty))
        return True, qty

    def get_current_price(self, ticker):
        return 100.0


def run_delta(target, balance):
    api = RecordingAPI(balance)
    ec.execute_delta(target, api, [], dry_run=False)
    return api


# 전액 현금(카나리 OFF) — 코인 보유 없음 → 아무 주문도 없어야 한다
api = run_delta({'CASH': 1.0}, {'KRW': TOTAL})
check("전액 CASH 목표 → 매수 0건", api.buys == [], f"({api.buys})")
check("전액 CASH 목표 → 매도 0건", api.sells == [], f"({api.sells})")

# 사고 재현 목표 — 현금 슬롯 1/3 이 티커로 새면 안 된다
api = run_delta(BUGGY, {'KRW': TOTAL})
bought = [t for t, _ in api.buys]
check("CASH 매수 주문 없음", not any(ec._is_cash_key(t) for t in bought), f"({bought})")
check("실제 코인만 매수", sorted(bought) == ['LINK', 'SOL'], f"({bought})")
check("현금 슬롯이 코인 매수액으로 새지 않음",
      all(abs(k - TOTAL / 3) < TOTAL * 0.005 for _, k in api.buys),
      f"({api.buys})")

# 소문자 변형도 동일
api = run_delta({'LINK': 0.5, 'cash': 0.5}, {'KRW': TOTAL})
check("소문자 cash 도 티커가 되지 않음",
      [t for t, _ in api.buys] == ['LINK'], f"({api.buys})")

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL PASS")
