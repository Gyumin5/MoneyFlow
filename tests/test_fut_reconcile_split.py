"""선물 V25 검사 분리 회귀 검증 (2026-08-21 오탐 사고).

사고: 무거래일에도 execution reconcile 이 돌았다. intent = 오늘 PV × 목표비중 × 목표L 로
매일 재계산되는 이론 명목이고 actual = 어제 체결분의 현재 명목이라, 가격이 움직인 만큼
항상 어긋난다(SOLUSDT intent $86,434 vs actual $81,139 = 6.1%). 그 차이는 drift threshold
0.03 이 이미 "안 건드린다"고 판정한 값인데도 v25_success=False → abort_streak +1 이 됐고,
3연속이면 ~/.binance_v25_lock 이 생겨 선물 매매 전체가 멈춘다.

수정: 체결 검사(_v25_exec_reconcile)는 매매를 시도한 실행에서만, 상시 무결성 검사
(_v25_standing_check)는 무거래일에도 돌되 가격·PnL·명목을 보지 않고 수량·방향·레버리지·
마진모드만 본다.

네트워크·라이브 state 미사용.
실행: cd /home/gmoh/mon/251229 && python3 tests/test_fut_reconcile_split.py
"""
import json
import os
import re
import sys
import tempfile

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


# 2026-08-21 실제 값 (사용자 알림에 찍힌 그대로)
INTENT_0821 = {
    'positions': {
        'SOLUSDT': {'notional': 86434.31, 'leverage': 4},
        'BNBUSDT': {'notional': 81000.00, 'leverage': 3},
    },
    'margin_type': 'CROSSED',
}
ACTUAL_0821 = {
    'positions': {
        'SOLUSDT': {'notional': 81138.75, 'leverage': 4},
        'BNBUSDT': {'notional': 80500.00, 'leverage': 3},
    },
}

# 무거래일이라 포지션은 어제 그대로. 수량·L·마진모드 불변, 명목만 시세로 움직였다.
BASELINE = {
    'ts': '2026-08-20T00:05:00+00:00',
    'positions': {
        'SOL': {'qty': 926.03, 'leverage': 4, 'isolated': False},
        'BNB': {'qty': 94.38, 'leverage': 3, 'isolated': False},
    },
}
POS_TODAY = {
    'SOL': {'qty': 926.03, 'leverage': 4.0, 'isolated': False, 'notional': 81138.75, 'symbol': 'SOLUSDT'},
    'BNB': {'qty': 94.38, 'leverage': 3.0, 'isolated': False, 'notional': 80500.00, 'symbol': 'BNBUSDT'},
}

# ── 1. 사고 재현: 그 차이는 execution reconcile 로는 여전히 "차이" 다 ──────────
print("[1] 2026-08-21 입력 재현")

diffs = a._v25_exec_reconcile(INTENT_0821, ACTUAL_0821)
check("무거래일 명목 차이는 exec reconcile 이 보면 diff 로 잡힌다(= 호출하면 안 된다)",
      any('SOLUSDT notional' in d for d in diffs), f"({diffs})")

crit, info = a._v25_standing_check(POS_TODAY, BASELINE)
check("같은 날 상시 무결성 검사는 통과", crit == [], f"({crit})")
check("정상 무거래일엔 info 도 없다", info == [], f"({info})")

# ── 2. 호출부 게이팅 (소스 불변식) ──────────────────────────────────────────
print("[2] 호출부 게이팅")

_src = open(os.path.join(_HERE, '..', 'trade', 'auto_trade_binance.py')).read()
check("exec reconcile 은 _v25_traded_path 로 게이팅된다",
      re.search(r'_v25_exec_reconcile\(intent_snapshot, actual_snapshot\)\s*\n\s*if \(v25_success and _v25_traded_path\)', _src) is not None,
      "(게이팅을 풀면 2026-08-21 오탐이 재발한다)")
check("옛 이름 _v25_reconcile( 은 남아 있지 않다",
      re.search(r'(?<!exec)_v25_reconcile\(', _src) is None)
check("_v25_traded_path 는 args.trade 분기에서만 True 가 된다",
      len(re.findall(r'_v25_traded_path = True', _src)) == 1)
check("상시 검사는 pos_after_ok 일 때만 돈다(조회 실패를 소실로 단정하지 않음)",
      re.search(r'if pos_after_ok:\s*\n\s*_base = _v25_read_health\(\)', _src) is not None)

# ── 3. streak 회계 ──────────────────────────────────────────────────────────
print("[3] abort_streak 회계")

_tmp = tempfile.mkdtemp(prefix='v25health_')
a.V25_HEALTH_FILE = os.path.join(_tmp, 'health.json')


def health():
    try:
        with open(a.V25_HEALTH_FILE) as f:
            return json.load(f)
    except FileNotFoundError:
        return {}


a._v25_write_health({'abort_streak': 1, 'last_abort_reason': 'reconcile diff'})
s = a._v25_record_cron_result(success=True, touch_streak=False)
check("무거래일 성공은 streak 을 리셋하지 않는다", s == 1 and health()['abort_streak'] == 1,
      f"(s={s})")
check("무거래일은 last_run_traded=False 로 남는다", health().get('last_run_traded') is False)

s = a._v25_record_cron_result(success=False, abort_reason='x', touch_streak=False)
check("무거래일 실패도 streak 을 올리지 않는다", s == 1 and health()['abort_streak'] == 1,
      f"(s={s})")
check("다만 사유는 남긴다", health().get('last_abort_reason') == 'x')

s = a._v25_record_cron_result(success=False, abort_reason='real', touch_streak=True)
check("매매 시도 실패는 streak 을 올린다", s == 2, f"(s={s})")

BASE_NEW = {'ts': 'now', 'positions': {'SOL': {'qty': 1.0, 'leverage': 4, 'isolated': False}}}
s = a._v25_record_cron_result(success=True, touch_streak=True, baseline=BASE_NEW)
check("매매 성공은 streak 을 0 으로 리셋", s == 0, f"(s={s})")
check("성공한 체결 실행만 기준 스냅샷을 갱신", health().get('post_trade_baseline') == BASE_NEW)

a._v25_record_cron_result(success=True, touch_streak=False)
check("무거래일은 기준 스냅샷을 건드리지 않는다", health().get('post_trade_baseline') == BASE_NEW)

# ── 4. 상시 검사가 실제 사고를 잡는가 ───────────────────────────────────────
print("[4] 상시 무결성 검사 — 진짜 이상만 잡는다")

crit, _ = a._v25_standing_check({'BNB': POS_TODAY['BNB']}, BASELINE)
check("포지션 소실 → critical", any('SOL' in c and '소실' in c for c in crit), f"({crit})")

lost = {**POS_TODAY, 'SOL': {**POS_TODAY['SOL'], 'qty': 500.0}}
crit, _ = a._v25_standing_check(lost, BASELINE)
check("부분청산(수량 46% 감소) → critical", any('수량 변동' in c for c in crit), f"({crit})")

flip = {**POS_TODAY, 'SOL': {**POS_TODAY['SOL'], 'qty': -926.03}}
crit, _ = a._v25_standing_check(flip, BASELINE)
check("방향 반전 → critical", any('방향 반전' in c for c in crit), f"({crit})")

iso = {**POS_TODAY, 'SOL': {**POS_TODAY['SOL'], 'isolated': True}}
crit, _ = a._v25_standing_check(iso, BASELINE)
check("마진모드 ISOLATED → critical", any('ISOLATED' in c for c in crit), f"({crit})")

lev = {**POS_TODAY, 'SOL': {**POS_TODAY['SOL'], 'leverage': 2.0}}
crit, _ = a._v25_standing_check(lev, BASELINE)
check("외부 레버리지 변경 → critical", any('레버리지' in c for c in crit), f"({crit})")

dust = {**POS_TODAY, 'DOGE': {'qty': 1.0, 'leverage': 3.0, 'isolated': False}}
crit, info = a._v25_standing_check(dust, BASELINE)
check("기준에 없는 포지션은 info 로만(자동 lock 금지)", crit == [] and len(info) == 1, f"({crit},{info})")

tiny = {**POS_TODAY, 'SOL': {**POS_TODAY['SOL'], 'qty': 926.03 * (1 + 0.004)}}
crit, _ = a._v25_standing_check(tiny, BASELINE)
check("0.4% 수량 오차(먼지/반올림)는 통과", crit == [], f"({crit})")

crit, info = a._v25_standing_check(POS_TODAY, {})
check("기준 스냅샷이 없으면 위반이 아니라 검사 생략", crit == [] and info and '생략' in info[0],
      f"({crit},{info})")

crit, _ = a._v25_standing_check({}, {'positions': {}})
check("양쪽 다 무포지션이면 조용하다", crit == [])

# 가격·PnL 로는 절대 발화하지 않는다 (오탐 원인 제거 확인)
moved = {'SOL': {**POS_TODAY['SOL'], 'notional': 40000.0, 'pnl': -12000.0, 'mark_price': 43.2},
         'BNB': {**POS_TODAY['BNB'], 'notional': 40000.0, 'pnl': -9000.0}}
crit, _ = a._v25_standing_check(moved, BASELINE)
check("명목 -50%·큰 평가손에도 발화하지 않는다(수량 불변이면 정상)", crit == [], f"({crit})")

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL PASS")
