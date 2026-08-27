"""선물 V25 상시 무결성 검사 ↔ 기준 스냅샷 순서 회귀 검증 (2026-08-22 자기잠김 사고).

사고: 08-22 09:46 수동 실행이 정상 체결(SOL 매도·BNB/BTC/ETH 매수, 목표 달성)됐는데
상시 무결성 검사가 그 체결을 '외부 변경' 으로 신고했다 — 기준 스냅샷이 08-20 체결분이라
우리 주문 결과(BNB 94.38→111.65, L3→L4, SOL 926.03→819.47)가 그대로 위반으로 잡혔다.
게다가 기준 갱신 조건이 v25_success 였으므로, 검사 실패가 기준 갱신을 막고
낡은 기준이 다음 실패를 다시 만드는 자기잠김 고리가 됐다(streak 2, lock 파일 생성).

수정 축 둘:
  (1) 상시 검사는 '주문을 내지 않은 실행' 에서만 돈다. 체결한 실행은 체결 검사가 맡는다.
      판정 축은 traded_path(시도)가 아니라 did_order(계좌 변경)다.
  (2) 기준 스냅샷은 주문이 나갔고 사후 조회가 성공하면 갱신한다 — v25_success 와 무관.
      사후 조회가 실패하면 기준을 무효화한다(검사 일시 중지 > 확정적 오탐).

네트워크·라이브 state 미사용 (fake client + 임시 health 파일).
실행: cd /home/gmoh/mon/251229 && python3 tests/test_fut_standing_baseline_order.py
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


# 08-20 09:05 체결 후 기준 스냅샷 (사고 당시 health 파일 값)
BASE_0820 = {
    'ts': '2026-08-20T00:05:26+00:00',
    'positions': {
        'BNB': {'qty': 94.38, 'leverage': 3, 'isolated': False},
        'SOL': {'qty': 926.03, 'leverage': 4, 'isolated': False},
    },
}
# 08-22 09:46 체결 후 실제 포지션
POS_AFTER_0822 = {
    'BNB': {'qty': 111.65, 'leverage': 4, 'isolated': False, 'symbol': 'BNBUSDT'},
    'BTC': {'qty': 0.246, 'leverage': 3, 'isolated': False, 'symbol': 'BTCUSDT'},
    'ETH': {'qty': 7.636, 'leverage': 3, 'isolated': False, 'symbol': 'ETHUSDT'},
    'SOL': {'qty': 819.47, 'leverage': 4, 'isolated': False, 'symbol': 'SOLUSDT'},
}


class FakeOrderClient:
    """사전매도 경로용 최소 fake."""

    def __init__(self, fail=False):
        self.fail = fail
        self.orders = []

    def futures_exchange_info(self):
        # 2026-08-27: contractType/status/quoteAsset 은 실제 응답에 항상 있다.
        # get_exchange_info 가 저장 전에 응답을 검증하므로 픽스처도 실제 형태를 갖춘다.
        return {'symbols': [{
            'symbol': 'SOLUSDT',
            'contractType': 'PERPETUAL',
            'status': 'TRADING',
            'quoteAsset': 'USDT',
            'filters': [
                {'filterType': 'LOT_SIZE', 'stepSize': '0.01', 'minQty': '0.01'},
                {'filterType': 'NOTIONAL', 'notional': '5'},
            ],
        }]}

    def futures_create_order(self, **kw):
        if self.fail:
            raise a.BinanceAPIException(_FakeResp(), 400, '{"code":-4131,"msg":"nope"}')
        self.orders.append(kw)
        return {'orderId': 1, 'status': 'FILLED'}


class FakeLevClient:
    """레버리지 변경용 최소 fake. err_code 지정 시 그 코드로 실패."""

    def __init__(self, err_code=None):
        self.err_code = err_code
        self.calls = []

    def futures_change_leverage(self, **kw):
        self.calls.append(kw)
        if self.err_code is not None:
            raise a.BinanceAPIException(
                _FakeResp(), 400, json.dumps({'code': self.err_code, 'msg': 'x'}))
        return {'leverage': kw.get('leverage')}


class _FakeResp:
    status_code = 400
    text = '{"code":-4131,"msg":"nope"}'

    def json(self):
        return {'code': -4131, 'msg': 'nope'}


def _with_temp_health(fn):
    """health 파일을 임시 경로로 바꿔 실행."""
    orig = a.V25_HEALTH_FILE
    fd, path = tempfile.mkstemp(prefix='v25_health_', suffix='.json')
    os.close(fd)
    os.unlink(path)
    a.V25_HEALTH_FILE = path
    try:
        return fn(path)
    finally:
        a.V25_HEALTH_FILE = orig
        for p in (path, path + '.tmp'):
            try:
                os.unlink(p)
            except OSError:
                pass


def main():
    print(__doc__.splitlines()[0])

    print("\n[1] 사고 재현 — 체결 후 포지션을 체결 전 기준과 비교하면 반드시 위반이 나온다")
    crit, _info = a._v25_standing_check(POS_AFTER_0822, BASE_0820)
    check("BNB 수량 변동이 위반으로 잡힌다",
          any('BNB' in c and '수량 변동' in c for c in crit))
    check("BNB 레버리지 변동이 위반으로 잡힌다",
          any('BNB' in c and '레버리지' in c for c in crit))
    check("SOL 수량 변동이 위반으로 잡힌다",
          any('SOL' in c and '수량 변동' in c for c in crit))
    check("즉 검사 자체는 정상 — 잘못은 부르는 시점이다", len(crit) >= 3)

    print("\n[2] 갱신된 기준으로는 같은 포지션이 깨끗하다 (고리가 닫힌다)")
    base_new = {'ts': '2026-08-22T00:46:29+00:00',
                'positions': {c: {'qty': p['qty'], 'leverage': p['leverage'], 'isolated': False}
                              for c, p in POS_AFTER_0822.items()}}
    crit2, _ = a._v25_standing_check(POS_AFTER_0822, base_new)
    check("갱신 후 위반 0건", crit2 == [], str(crit2))

    print("\n[3] 기준 갱신이 success 판정에 묶이지 않는다")

    def _t3(path):
        # 실패로 기록되는 실행이어도 baseline 은 들어가야 한다
        streak = a._v25_record_cron_result(
            success=False, abort_reason='reconcile diff',
            touch_streak=True, baseline=base_new)
        h = json.load(open(path))
        check("실패 기록에도 기준 스냅샷이 저장된다",
              (h.get('post_trade_baseline') or {}).get('ts') == base_new['ts'])
        check("실패이므로 streak 은 오른다", streak == 1)
        # 그 기준으로 다음 무거래일 검사는 통과
        crit3, _ = a._v25_standing_check(POS_AFTER_0822,
                                         h.get('post_trade_baseline') or {})
        check("저장된 기준으로 다음날 검사 통과", crit3 == [], str(crit3))
    _with_temp_health(_t3)

    print("\n[4] 사후 조회 실패 시 기준 무효화 → 검사 생략 (확정 오탐보다 공백을 택한다)")

    def _t4(path):
        a._v25_record_cron_result(success=True, touch_streak=True, baseline=BASE_0820)
        a._v25_invalidate_baseline("post-trade 포지션 조회 실패 — 기준 스냅샷 무효화")
        h = json.load(open(path))
        check("기준 스냅샷이 제거된다", 'post_trade_baseline' not in h)
        check("무효화 사유가 남는다", '조회 실패' in (h.get('baseline_invalidated_reason') or ''))
        crit4, info4 = a._v25_standing_check(POS_AFTER_0822,
                                             h.get('post_trade_baseline') or {})
        check("기준 없으면 위반 0건 + 생략 로그", crit4 == [] and any('생략' in m for m in info4))
        check("streak 은 건드리지 않는다", h.get('abort_streak') == 0)
    _with_temp_health(_t4)

    print("\n[5] L↓ 사전매도도 주문 기록에 남는다 (did_order 판정의 근거)")
    alerts = []
    c = FakeOrderClient()
    ok = a._v25_partial_sell_for_leverage_down(c, 'SOLUSDT', 90000.0, 60000.0, 200.0,
                                               order_alerts=alerts)
    check("사전매도 성공", ok is True)
    check("주문이 실제로 나갔다", len(c.orders) == 1 and c.orders[0]['reduceOnly'] is True)
    check("order_alerts 에 기록된다", len(alerts) == 1 and 'SOLUSDT' in alerts[0])
    check("사전매도임을 알 수 있다", '사전매도' in alerts[0])

    alerts_skip = []
    c_skip = FakeOrderClient()
    ok_skip = a._v25_partial_sell_for_leverage_down(c_skip, 'SOLUSDT', 60000.0, 60000.0, 200.0,
                                                    order_alerts=alerts_skip)
    check("매도 불필요면 주문도 기록도 없다",
          ok_skip is True and c_skip.orders == [] and alerts_skip == [])

    alerts_fail = []
    c_fail = FakeOrderClient(fail=True)
    ok_fail = a._v25_partial_sell_for_leverage_down(c_fail, 'SOLUSDT', 90000.0, 60000.0, 200.0,
                                                    order_alerts=alerts_fail)
    check("주문 실패는 False + 기록 없음",
          ok_fail is False and alerts_fail == [])

    print("\n[6] 소스 불변식 — 호출 시점과 갱신 조건")
    src = open(os.path.join(_HERE, '..', 'trade', 'auto_trade_binance.py')).read()
    check("did_order 는 계좌변경 표시로 판정 (보고 리스트가 근거가 아니다)",
          '_v25_did_order = _v25_account_changed()' in src
          and '_v25_did_order = bool(order_alerts)' not in src)
    check("상시 검사는 계좌를 건드리기 전에, 체결 전 포지션으로 돈다",
          re.search(r'_v25_standing_check\(positions_before, _sbase\)', src) is not None)
    check("체결 후 포지션으로 상시 검사하는 경로가 없다",
          '_v25_standing_check(positions_after' not in src)
    check("위반이면 그 실행이 매매를 안 한다 (선언과 동작 일치)",
          re.search(r'if _standing_block:\s*\n\s*log\.error\([^\n]*매매하지 않는다', src) is not None
          and re.search(r'_standing_block = True', src) is not None)
    check("검사가 매매 분기보다 앞에 있다",
          src.index('_v25_standing_check(positions_before') < src.index("elif not rebalance_needed:"))
    check("lock 은 검사 지점에서 한 번만 만든다",
          src.count('_v25_create_lock("standing integrity: ') == 1)
    check("기준 갱신 조건에서 v25_success 가 빠졌다",
          'if _v25_did_order and pos_after_ok:' in src)
    check("옛 자기잠김 조건이 남아있지 않다",
          'if v25_success and _v25_traded_path and pos_after_ok:' not in src)
    check("사후 조회 실패 시 기준 무효화",
          re.search(r'if not pos_after_ok:[\s\S]{0,500}?if _v25_did_order:\s*\n'
                    r'[\s\S]{0,400}?_v25_invalidate_baseline\(', src) is not None)
    check("무효화 전에 제한 재시도한다",
          re.search(r'if not pos_after_ok and _v25_did_order:\s*\n\s*for _try in range\(1, 4\)',
                    src) is not None)
    check("기준이 비어 있으면 재시딩해 감시를 재개한다",
          '_snap_baseline(reseeded=True)' in src)
    check("사전매도가 order_alerts 를 받는다",
          'order_alerts=order_alerts)' in src and
          re.search(r'def _v25_partial_sell_for_leverage_down[\s\S]{0,300}?order_alerts', src) is not None)

    print("\n[7] 계좌 변경 표시 — 주문이 아닌 변경과 결과 불명도 센다")
    check("표시 레지스트리와 헬퍼가 있다",
          '_V25_ACCOUNT_TOUCH' in src and 'def _v25_mark_touch' in src
          and 'def _v25_account_changed' in src)
    check("실행 시작 시 초기화",
          '_v25_touch_reset()  # 계좌 변경 표시 초기화' in src)
    for site, pat in (
        ("주문 성공", r'_v25_mark_touch\(f"order \{side\} \{symbol\} \{qty_str\}"\)'),
        ("주문 예외(체결 불명)", r'_v25_mark_touch\(f"order \{side\} \{symbol\} 시도 실패\(체결 불명\)"\)'),
        ("사전매도 성공", r'_v25_mark_touch\(f"presell SELL \{sym\} \{sell_qty\}"\)'),
        ("사전매도 예외", r'_v25_mark_touch\(f"presell SELL \{sym\} 시도 실패\(체결 불명\)"\)'),
        ("레버리지 변경", r'_v25_mark_touch\(f"leverage \{symbol\}=\{leverage\}x"\)'),
        ("레버리지 실패(적용 불명)", r'_v25_mark_touch\(f"leverage \{symbol\}=\{leverage\}x 시도 실패'),
        ("마진모드 변경", r'_v25_mark_touch\(f"margin \{symbol\}=\{margin_type\}"\)'),
        ("마진모드 실패(적용 불명)", r'_v25_mark_touch\(f"margin \{symbol\}=\{margin_type\} 시도 실패'),
    ):
        check(f"{site} 지점에 표시", re.search(pat, src) is not None)

    check("이미 목표 L 이면 set 을 부르지 않는다 (무의미한 변경표시 차단)",
          re.search(r'if verify_leverage\(client, sym, lev\):[\s\S]{0,200}?'
                    r'elif not set_leverage\(client, sym, lev\):', src) is not None)

    a._v25_touch_reset()
    check("초기 상태는 무변경", a._v25_account_changed() is False)
    c_lev = FakeLevClient()
    a.set_leverage(c_lev, 'SOLUSDT', 4)
    check("레버리지만 바꿔도 계좌 변경으로 잡힌다", a._v25_account_changed() is True)
    a._v25_touch_reset()
    c_idem = FakeLevClient(err_code=-4046)
    a.set_leverage(c_idem, 'SOLUSDT', 4)
    check("idempotent 응답(변경 불필요)은 변경으로 세지 않는다",
          a._v25_account_changed() is False)
    a._v25_touch_reset()
    c_err = FakeLevClient(err_code=-1001)
    ok_lev = a.set_leverage(c_err, 'SOLUSDT', 4)
    check("적용 불명 실패는 변경으로 센다", ok_lev is False and a._v25_account_changed() is True)
    a._v25_touch_reset()

    print("\n[8] 08-21 수정 회귀 — 무거래일 오탐 차단은 그대로 유지")
    check("체결 검사는 매매 시도 실행에서만",
          '_v25_exec_reconcile(intent_snapshot, actual_snapshot)' in src and
          'if (v25_success and _v25_traded_path) else []' in src)
    check("streak 은 무거래일에 움직이지 않는다",
          '_touch_streak = bool(_v25_traded_path or error_alerts or v25_abort or _standing_crit)' in src)
    check("상시 검사는 가격·명목을 안 본다 (수량·방향·L·마진만)",
          re.search(r"def _v25_standing_check[\s\S]{0,3000}?return crit, info", src) is not None and
          'notional' not in re.search(r"def _v25_standing_check([\s\S]{0,3000}?)return crit, info",
                                      src).group(1))

    print()
    if FAILS:
        print(f"FAIL {len(FAILS)} 건: {FAILS}")
        return 1
    print("전 항목 PASS")
    return 0


if __name__ == '__main__':
    sys.exit(main())
