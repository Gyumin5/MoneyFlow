"""선물 체결·상시 검사의 방향(롱 전용)과 마진모드 축 검증 (2026-09-30).

왜 필요했나. 두 검사 모두 "같은 크기의 숏" 을 통과시키고 있었다.
  · 체결 검사(_v25_exec_reconcile)는 명목을 절댓값으로 비교한다. 목표 <금액> 롱인데
    실제가 <금액> 숏이면 차이가 0 으로 나온다. 수량은 미전송 갈래에서만 봤다.
  · 상시 검사(_v25_standing_check)는 방향 반전을 보지만 기준 스냅샷이 있는 심볼만
    본다. 기준이 없는 날(최초 실행·무효화 후)과 기준에 없는 심볼은 통째로 생략됐다.
  · 마진모드 비교문은 있었지만 실제 스냅샷에 그 키가 들어간 적이 없어 안 돌았다.

수량 크기 축은 일부러 안 넣는다. 의도 수량을 저장하지 않아 목표 명목을 가격으로
나눠 만들어야 하는데, 그러면 5% 체결 밴드(정상적으로 주문을 생략하는 구간)와 겹쳐
정상 실행이 경보가 된다. 부호만 본다 — 부호에는 허용오차를 두지 않는다.
근거: ai-debate run-20260930T045424Z.

네트워크·라이브 state 미사용.
실행: cd /home/gmoh/mon/251229 && python3 tests/test_fut_direction_margin_check.py
"""
import os
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


def has(diffs, *words):
    return any(all(w in d for w in words) for d in diffs)


# ─────────────────────────────────────────────────────────────
print("# 체결 검사 — 방향")

INTENT = {
    'positions': {'BTCUSDT': {'notional': 1000.0, 'leverage': 3}},
    'margin_type': 'CROSSED',
}


def actual(notional, qty, margin_mode='cross', leverage=3):
    return {'positions': {'BTCUSDT': {'notional': notional, 'qty': qty,
                                      'leverage': leverage,
                                      'margin_mode': margin_mode}}}


d = a._v25_exec_reconcile(INTENT, actual(1000.0, 0.01))
check('T1 정상 롱은 통과', d == [], d)

d = a._v25_exec_reconcile(INTENT, actual(1000.0, -0.01))
check('T2 같은 명목의 숏을 잡는다', has(d, 'BTCUSDT', '숏'), d)

# 명목 축이 이 경우를 못 잡는다는 것 자체를 못박아 둔다.
check('T2b 명목 축만으로는 T2 가 안 걸린다',
      not has(d, 'notional'), d)

d = a._v25_exec_reconcile(INTENT, actual(1000.0, 0.0102))
check('T3 밴드 안 수량 차이는 경보가 아니다', d == [], d)

# ─────────────────────────────────────────────────────────────
print("\n# 체결 검사 — 마진모드")

d = a._v25_exec_reconcile(INTENT, actual(1000.0, 0.01, margin_mode='isolated'))
check('T4 ISOLATED 를 잡는다', has(d, 'BTCUSDT', '마진모드', 'isolated'), d)

d = a._v25_exec_reconcile(INTENT, actual(1000.0, 0.01, margin_mode=None))
check('T5 조회 불명을 통과시키지 않는다', has(d, '마진모드', '불명'), d)

# ─────────────────────────────────────────────────────────────
print("\n# 체결 검사 — 미전송 갈래에서도 방향을 본다")

unsent = {'BTCUSDT': {'notional': 1000.0, 'qty': 0.01, 'leverage': 3}}
d = a._v25_exec_reconcile(INTENT, actual(1000.0, 0.01), unsent)
check('T6 미전송 + 실행 전 그대로면 통과', d == [], d)

# 주문을 안 냈는데 방향이 뒤집혔다. 미전송 갈래는 continue 로 빠지므로,
# 방향 검사가 갈래 앞에 있지 않으면 이 경우가 "숏" 으로 안 잡힌다.
d = a._v25_exec_reconcile(INTENT, actual(1000.0, -0.01), unsent)
check('T7 미전송 심볼의 숏도 잡는다', has(d, '숏'), d)

# ─────────────────────────────────────────────────────────────
print("\n# 상시 검사 — 기준이 없어도 롱 전용은 본다")

crit, info = a._v25_standing_check({'BTC': {'qty': 0.01, 'leverage': 3}}, {})
check('T8 기준 없음 + 롱이면 critical 없음', crit == [], crit)
check('T8b 기준 없음은 info 로 남는다', any('기준 스냅샷 없음' in m for m in info), info)

crit, _ = a._v25_standing_check({'BTC': {'qty': -0.01, 'leverage': 3}}, {})
check('T9 기준이 없어도 숏은 잡는다', any('숏' in c for c in crit), crit)

BASE = {'positions': {'BTC': {'qty': 0.01, 'leverage': 3, 'isolated': False}}}
crit, _ = a._v25_standing_check({'BTC': {'qty': 0.01, 'leverage': 3},
                                 'SOL': {'qty': -5.0, 'leverage': 3}}, BASE)
check('T10 기준에 없는 심볼의 숏도 잡는다', any('SOL' in c and '숏' in c for c in crit), crit)

crit, _ = a._v25_standing_check({'BTC': {'qty': 0.01, 'leverage': 3}}, BASE)
check('T11 기준과 같은 정상 롱은 통과', crit == [], crit)

# ─────────────────────────────────────────────────────────────
print("\n# 의도 명목에 현금 버퍼가 반영되는가")

# 주문 크기는 execute_rebalance 가 total_pv * (1 - CASH_BUFFER) * w * L 로 잡는다.
# 의도 쪽이 같은 식이어야 정상 실행에서 차이가 0 이다.
PV, W, L = 60000.0, 1.0 / 3.0, 3
ordered = PV * (1 - a.CASH_BUFFER) * W * L
intent_buf = {'positions': {'BTCUSDT': {'notional': ordered, 'leverage': L}},
              'margin_type': 'CROSSED'}
d = a._v25_exec_reconcile(intent_buf, actual(ordered, 0.5, leverage=L))
check('T12 버퍼 반영 시 정상 체결은 차이 0', d == [], d)

# 버퍼를 빼먹으면 허용폭의 얼마를 까먹는지 수치로 남긴다(경보는 아니지만 폭 잠식).
naive = PV * W * L
gap = abs(naive - ordered)
tol = max(50.0, 0.05 * max(naive, ordered, 1.0))
check('T13 버퍼 누락분은 허용폭보다 작다(그래서 조용히 먹힌다)',
      0 < gap < tol, f'gap=${gap:.2f} tol=${tol:.2f}')
print(f"  버퍼 누락 시 상시 어긋남 ${gap:.2f} / 허용폭 ${tol:.2f} "
      f"= 폭의 {gap / tol:.0%}")

print()
if FAILS:
    print(f"실패 {len(FAILS)}건: {FAILS}")
    sys.exit(1)
print("ALL PASS (T1~T13)")
