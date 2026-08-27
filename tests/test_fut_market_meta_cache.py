"""거래소 심볼정보 fail-open 제거 + 36시간 캐시 검증 — 2026-08-27.

배경: refresh_universe 는 exchangeInfo 나 CoinGecko 조회가 실패하면 코드에 박아둔
고정 목록(HARDCODED_UNIVERSE_FALLBACK)으로 넘어갔다. 그 목록은 갱신되지 않아
상장폐지 심볼이 남아 있어도 걸러지지 않고 그대로 주문 경로에 들어갔다.
V25 매매경로의 다른 이상(시세조회 실패·마진모드 불일치·L 하향 사전매도 실패)은
전부 즉시 ABORT 인데 여기만 fail-open 이었다.

ai-debate run-20260827T084242Z 결론: 단순 fail-closed 도, 고정 목록 유지도 아니다.
직전 성공 응답을 36시간까지 쓰고(하루 한 번 cron 기준 일시 장애 한 번은 견딤),
그것도 없으면 주문을 만들기 전에 멈춘다.

중재자 verification_plan 이 지목한 경우를 그대로 옮겼다 — 정상 조회, 조회 실패 +
12시간 캐시, 40시간 캐시, 캐시 손상, 빈 응답, 심볼 누락, 비거래 상태.
핵심 불변식: 낡거나 검증 안 된 자료에서는 주문 경로가 한 번도 열리지 않는다.

네트워크·라이브 state 미사용. 캐시는 임시 디렉토리로 돌린다.
실행: cd /home/gmoh/mon/251229 && python3 tests/test_fut_market_meta_cache.py
"""
import json
import os
import shutil
import sys
import tempfile
from datetime import datetime, timedelta, timezone

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


def good_exchange_info(symbols=('BTCUSDT', 'ETHUSDT', 'SOLUSDT')):
    return {'symbols': [
        {'symbol': s, 'contractType': 'PERPETUAL', 'status': 'TRADING',
         'quoteAsset': 'USDT',
         'filters': [{'filterType': 'LOT_SIZE', 'stepSize': '0.001', 'minQty': '0.001'},
                     {'filterType': 'NOTIONAL', 'notional': '5.0'}]}
        for s in symbols]}


def cg_rows(symbols=('btc', 'eth', 'sol')):
    return [{'symbol': s, 'market_cap': 10 ** 12} for s in symbols]


class FakeClient:
    """exchangeInfo 만 흉내낸다. fail=True 면 조회가 터진다."""

    def __init__(self, info=None, fail=False):
        self.info = info if info is not None else good_exchange_info()
        self.fail = fail
        self.calls = 0

    def futures_exchange_info(self):
        self.calls += 1
        if self.fail:
            raise RuntimeError('binance down')
        return self.info


def reset(tmp, cg=None, cg_fail=False, wipe=True):
    """모듈 전역을 이 테스트용으로 갈아끼운다. wipe=False 면 디스크 캐시를 남긴다."""
    a._exchange_info_cache = None
    a._degraded_notified = False
    a.UNIVERSE = []
    a.EXCHANGE_INFO_CACHE_PATH = os.path.join(tmp, 'exinfo.json')
    a.UNIVERSE_CACHE_PATH = os.path.join(tmp, 'cg.json')
    if wipe:
        for p in (a.EXCHANGE_INFO_CACHE_PATH, a.UNIVERSE_CACHE_PATH):
            if os.path.isfile(p):
                os.remove(p)
    a.send_telegram = lambda msg: SENT.append(msg)
    a.fetch_coingecko_top_futures = (
        lambda limit=40, cache_path=None: [] if cg_fail else (cg or cg_rows()))


SENT = []
TMP = tempfile.mkdtemp(prefix='mktmeta-')

try:
    # ── 1. 정상 조회 ─────────────────────────────────────────────────────
    print("[1] 정상 조회")
    reset(TMP)
    c = FakeClient()
    info = a.get_exchange_info(c)
    check("정상 응답을 그대로 돌려준다", len(info['symbols']) == 3)
    check("캐시 파일이 생겼다", os.path.isfile(a.EXCHANGE_INFO_CACHE_PATH))
    blob = json.load(open(a.EXCHANGE_INFO_CACHE_PATH))
    check("캐시에 조회시각이 있다", 'fetched_at' in blob and 'data' in blob)
    check("정상 경로에서는 degraded 알림이 없다", SENT == [], str(SENT))
    a.get_exchange_info(c)
    check("한 실행 안에서는 API 를 한 번만 부른다", c.calls == 1, f"calls={c.calls}")

    reset(TMP)
    uni = a.refresh_universe(FakeClient())
    check("유니버스가 교집합으로 만들어진다",
          uni == ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], str(uni))

    # ── 2. 조회 실패 + 12시간 캐시 → 진행 ────────────────────────────────
    print("\n[2] 조회 실패 + 12시간 캐시")
    reset(TMP)
    a._save_market_cache(a.EXCHANGE_INFO_CACHE_PATH, good_exchange_info())
    blob = json.load(open(a.EXCHANGE_INFO_CACHE_PATH))
    blob['fetched_at'] = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    json.dump(blob, open(a.EXCHANGE_INFO_CACHE_PATH, 'w'))
    SENT.clear()
    info = a.get_exchange_info(FakeClient(fail=True))
    check("12시간 캐시로 진행한다", len(info['symbols']) == 3)
    check("degraded 알림을 한 번 보낸다", len(SENT) == 1, str(SENT))
    a._exchange_info_cache = None
    a.get_exchange_info(FakeClient(fail=True))
    check("같은 실행에서 알림이 두 번 가지 않는다", len(SENT) == 1, str(SENT))

    # ── 3. 40시간 캐시 → 중단 ────────────────────────────────────────────
    print("\n[3] 40시간 캐시 (36시간 초과)")
    reset(TMP)
    a._save_market_cache(a.EXCHANGE_INFO_CACHE_PATH, good_exchange_info())
    blob = json.load(open(a.EXCHANGE_INFO_CACHE_PATH))
    blob['fetched_at'] = (datetime.now(timezone.utc) - timedelta(hours=40)).isoformat()
    json.dump(blob, open(a.EXCHANGE_INFO_CACHE_PATH, 'w'))
    try:
        a.get_exchange_info(FakeClient(fail=True))
        check("40시간 캐시는 거부한다", False, '예외가 안 났다')
    except a.MarketMetaUnavailable:
        check("40시간 캐시는 거부한다", True)

    # ── 4. 캐시 없음 / 손상 / 나이 불명 ──────────────────────────────────
    print("\n[4] 캐시 없음·손상·나이 불명")
    reset(TMP)
    try:
        a.get_exchange_info(FakeClient(fail=True))
        check("캐시가 없으면 중단", False, '예외가 안 났다')
    except a.MarketMetaUnavailable:
        check("캐시가 없으면 중단", True)

    reset(TMP)
    open(a.EXCHANGE_INFO_CACHE_PATH, 'w').write('{ 깨진 json')
    try:
        a.get_exchange_info(FakeClient(fail=True))
        check("손상 캐시는 중단", False, '예외가 안 났다')
    except a.MarketMetaUnavailable:
        check("손상 캐시는 중단", True)

    reset(TMP)
    json.dump({'data': good_exchange_info()}, open(a.EXCHANGE_INFO_CACHE_PATH, 'w'))
    try:
        a.get_exchange_info(FakeClient(fail=True))
        check("조회시각 없는 캐시는 중단(나이를 모르면 낡은 것과 구분 불가)", False, '예외가 안 났다')
    except a.MarketMetaUnavailable:
        check("조회시각 없는 캐시는 중단(나이를 모르면 낡은 것과 구분 불가)", True)

    # ── 5. 빈 응답·심볼 누락·비거래 상태는 캐시에 저장하지 않는다 ────────
    print("\n[5] 못 쓸 응답")
    for label, bad in [
        ("빈 응답", {'symbols': []}),
        ("symbols 키 없음", {}),
        ("전부 비거래 상태(SETTLING)", {'symbols': [
            {'symbol': 'BTCUSDT', 'contractType': 'PERPETUAL', 'status': 'SETTLING',
             'quoteAsset': 'USDT', 'filters': [
                 {'filterType': 'LOT_SIZE', 'stepSize': '0.001'}]}]}),
        ("수량 단위 누락", {'symbols': [
            {'symbol': 'BTCUSDT', 'contractType': 'PERPETUAL', 'status': 'TRADING',
             'quoteAsset': 'USDT', 'filters': []}]}),
    ]:
        reset(TMP)
        check(f"{label} 은 검증에서 걸러진다", not a._valid_exchange_info(bad))
        try:
            a.get_exchange_info(FakeClient(info=bad))
            check(f"{label} 이면 중단", False, '예외가 안 났다')
        except a.MarketMetaUnavailable:
            check(f"{label} 이면 중단", True)
        check(f"{label} 은 캐시에 저장되지 않는다",
              not os.path.isfile(a.EXCHANGE_INFO_CACHE_PATH))

    # ── 6. 못 쓸 응답이 와도 기존 정상 캐시를 덮어쓰지 않는다 ────────────
    print("\n[6] 정상 캐시 보존")
    reset(TMP)
    a.get_exchange_info(FakeClient())                     # 정상 1회 → 캐시 생성
    reset(TMP, wipe=False)                                # 메모리 캐시만 비움
    info = a.get_exchange_info(FakeClient(info={'symbols': []}))
    check("빈 응답이 와도 직전 정상 캐시로 진행한다", len(info['symbols']) == 3)
    blob = json.load(open(a.EXCHANGE_INFO_CACHE_PATH))
    check("빈 응답이 캐시를 덮어쓰지 않았다", len(blob['data']['symbols']) == 3)

    # ── 7. CoinGecko 쪽도 같은 규칙 ──────────────────────────────────────
    print("\n[7] 시총 목록")
    reset(TMP, cg_fail=True)
    try:
        a.refresh_universe(FakeClient())
        check("시총 목록을 못 얻으면 중단", False, '예외가 안 났다')
    except a.MarketMetaUnavailable:
        check("시총 목록을 못 얻으면 중단", True)

    reset(TMP, cg=cg_rows(('doge', 'shib')))
    try:
        a.refresh_universe(FakeClient())
        check("교집합이 비면 중단(고정 목록으로 안 샌다)", False, '예외가 안 났다')
    except a.MarketMetaUnavailable:
        check("교집합이 비면 중단(고정 목록으로 안 샌다)", True)

    # ── 8. 상장폐지 심볼이 유니버스에 남지 않는다 ────────────────────────
    print("\n[8] 상장폐지 재현")
    delisted = good_exchange_info(('BTCUSDT', 'ETHUSDT'))
    delisted['symbols'].append(
        {'symbol': 'SOLUSDT', 'contractType': 'PERPETUAL', 'status': 'SETTLING',
         'quoteAsset': 'USDT',
         'filters': [{'filterType': 'LOT_SIZE', 'stepSize': '0.001'}]})
    reset(TMP)
    uni = a.refresh_universe(FakeClient(info=delisted))
    check("정산 단계 심볼은 유니버스에서 빠진다", 'SOLUSDT' not in uni, str(uni))
    check("나머지는 그대로 남는다", uni == ['BTCUSDT', 'ETHUSDT'], str(uni))

    # ── 9. 회귀 — 고정 목록이 코드에서 사라졌다 ──────────────────────────
    print("\n[9] 회귀")
    _src = open(os.path.join(_HERE, '..', 'trade', 'auto_trade_binance.py')).read()
    check("HARDCODED_UNIVERSE_FALLBACK 정의가 없다",
          'HARDCODED_UNIVERSE_FALLBACK = [' not in _src)
    check("매매 경로가 MarketMetaUnavailable 을 잡아 ABORT 한다",
          'except MarketMetaUnavailable as e:' in _src and 'V25 ABORT: {e} — 매매 차단' in _src)
    check("fetch_binance_futures_listed 가 실패를 빈 set 으로 삼키지 않는다",
          'return set()' not in _src)
    check("캐시 경로가 SCRIPT_DIR 아래다(재부팅에 안 날아감)",
          "EXCHANGE_INFO_CACHE_PATH = os.path.join(SCRIPT_DIR" in _src
          and "UNIVERSE_CACHE_PATH = os.path.join(SCRIPT_DIR" in _src)
    check("TTL 은 36시간", 'MARKET_META_MAX_AGE_H = 36.0' in _src)

finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL PASS")
