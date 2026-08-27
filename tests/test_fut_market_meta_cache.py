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


def _names(info):
    return {s['symbol'] for s in info.get('symbols', [])}


def sym_row(name, status='TRADING'):
    return {'symbol': name, 'contractType': 'PERPETUAL', 'status': status,
            'quoteAsset': 'USDT',
            'filters': [{'filterType': 'LOT_SIZE', 'stepSize': '0.001', 'minQty': '0.001'},
                        {'filterType': 'NOTIONAL', 'notional': '5.0'}]}


def filler(n):
    """최소 개수 문턱을 넘기기 위한 채움. 실제 바이낸스는 524종(2026-08-27 실측)이다."""
    return [sym_row(f'FILL{i}USDT') for i in range(n)]


def good_exchange_info(symbols=('BTCUSDT', 'ETHUSDT', 'SOLUSDT'), pad=240):
    return {'symbols': [sym_row(s) for s in symbols] + filler(pad)}


def cg_rows(symbols=('btc', 'eth', 'sol'), pad=37):
    """시총 목록. 채움은 거래소 채움과 같은 이름이라 교집합에 들어간다.

    유니버스 최소 크기(MIN_UNIVERSE_SIZE) 문턱이 있으므로 교집합이 실제로 커야 한다.
    상장 안 된 심볼을 섞는 검사는 cg_rows_unlisted 를 쓴다.
    """
    return ([{'symbol': s, 'market_cap': 10 ** 12} for s in symbols]
            + [{'symbol': f'FILL{i}', 'market_cap': 10 ** 9} for i in range(pad)])


def cg_rows_unlisted(pad=40):
    """형태는 멀쩡하지만 거래소에 하나도 상장 안 된 목록."""
    return [{'symbol': f'nolist{i}', 'market_cap': 10 ** 9} for i in range(pad)]


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
    a._degraded_causes.clear()
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


class _FailRequests:
    class exceptions:
        RequestException = Exception

    @staticmethod
    def get(*args, **kwargs):
        raise RuntimeError('network down')


class _NoSleep:
    @staticmethod
    def sleep(_):
        return None

    @staticmethod
    def time():
        return 0.0


SENT = []
_real_cg = a.fetch_coingecko_top_futures
_real_requests = a.requests
_real_time = a.time
TMP = tempfile.mkdtemp(prefix='mktmeta-')

try:
    # ── 1. 정상 조회 ─────────────────────────────────────────────────────
    print("[1] 정상 조회")
    reset(TMP)
    c = FakeClient()
    info = a.get_exchange_info(c)
    check("정상 응답을 그대로 돌려준다", _names(info) >= {'BTCUSDT', 'ETHUSDT', 'SOLUSDT'})
    check("캐시 파일이 생겼다", os.path.isfile(a.EXCHANGE_INFO_CACHE_PATH))
    blob = json.load(open(a.EXCHANGE_INFO_CACHE_PATH))
    check("캐시에 조회시각이 있다", 'fetched_at' in blob and 'data' in blob)
    check("정상 경로에서는 degraded 알림이 없다", SENT == [], str(SENT))
    a.get_exchange_info(c)
    check("한 실행 안에서는 API 를 한 번만 부른다", c.calls == 1, f"calls={c.calls}")

    reset(TMP)
    uni = a.refresh_universe(FakeClient())
    check("유니버스가 교집합으로 만들어진다",
          uni[:3] == ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'] and len(uni) >= a.MIN_UNIVERSE_SIZE,
          str(uni[:5]) + f' len={len(uni)}')

    # ── 2. 조회 실패 + 12시간 캐시 → 진행 ────────────────────────────────
    print("\n[2] 조회 실패 + 12시간 캐시")
    reset(TMP)
    a._save_market_cache(a.EXCHANGE_INFO_CACHE_PATH, good_exchange_info())
    blob = json.load(open(a.EXCHANGE_INFO_CACHE_PATH))
    blob['fetched_at'] = (datetime.now(timezone.utc) - timedelta(hours=12)).isoformat()
    json.dump(blob, open(a.EXCHANGE_INFO_CACHE_PATH, 'w'))
    SENT.clear()
    info = a.get_exchange_info(FakeClient(fail=True))
    check("12시간 캐시로 진행한다", 'BTCUSDT' in _names(info))
    a._flush_degraded()
    check("degraded 알림을 한 번 보낸다", len(SENT) == 1, str(SENT))

    # 두 소스가 동시에 degraded 면 한 건 안에 두 사유가 다 들어가야 한다.
    # 단일 불리언이면 두 번째 원인이 사람 화면에서 사라진다.
    reset(TMP, cg_fail=True)
    a._save_market_cache(a.EXCHANGE_INFO_CACHE_PATH, good_exchange_info())
    a._save_market_cache(a.UNIVERSE_CACHE_PATH, cg_rows())
    # 진짜 fetch_coingecko_top_futures 의 캐시 폴백 경로를 태운다.
    # 네트워크는 막고(requests.get 이 터지게) 재시도 대기는 없앤다.
    a.fetch_coingecko_top_futures = _real_cg
    a.requests = _FailRequests()
    a.time = _NoSleep()
    SENT.clear()
    uni = a.refresh_universe(FakeClient(fail=True))
    check("양쪽 degraded 여도 유니버스는 만들어진다",
          uni[:3] == ['BTCUSDT', 'ETHUSDT', 'SOLUSDT'], str(uni[:5]))
    check("알림은 한 건이다", len(SENT) == 1, str(SENT))
    check("그 한 건에 두 사유가 다 들어간다",
          len(SENT) == 1 and '심볼정보' in SENT[0] and '시총 목록' in SENT[0], str(SENT))
    a.requests, a.time = _real_requests, _real_time

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
    check("빈 응답이 와도 직전 정상 캐시로 진행한다", 'BTCUSDT' in _names(info))
    blob = json.load(open(a.EXCHANGE_INFO_CACHE_PATH))
    check("빈 응답이 캐시를 덮어쓰지 않았다", 'BTCUSDT' in _names(blob['data']))

    # ── 7. CoinGecko 쪽도 같은 규칙 ──────────────────────────────────────
    print("\n[7] 시총 목록")
    reset(TMP, cg_fail=True)
    try:
        a.refresh_universe(FakeClient())
        check("시총 목록을 못 얻으면 중단", False, '예외가 안 났다')
    except a.MarketMetaUnavailable:
        check("시총 목록을 못 얻으면 중단", True)

    reset(TMP, cg=cg_rows_unlisted())
    try:
        a.refresh_universe(FakeClient())
        check("교집합이 비면 중단(고정 목록으로 안 샌다)", False, '예외가 안 났다')
    except a.MarketMetaUnavailable:
        check("교집합이 비면 중단(고정 목록으로 안 샌다)", True)

    # 부분 성공 입력 — 형태는 멀쩡한데 내용이 모자란 응답들. 전부 거부해야 한다.
    # (ai-debate run-20260827T093109Z codex 지적: 개수만 세면 중복으로 채운 응답이 통과한다.)
    for label, rows in [
        ("고유 심볼이 문턱 미만", cg_rows(('btc',), pad=10)),
        ("같은 심볼로 채운 응답", [{'symbol': 'btc', 'market_cap': 1} for _ in range(40)]),
        ("빈 심볼이 섞임", cg_rows() + [{'symbol': '', 'market_cap': 1}]),
        ("심볼 타입이 문자열이 아님", cg_rows() + [{'symbol': 123, 'market_cap': 1}]),
    ]:
        reset(TMP, cg=rows)
        try:
            a.refresh_universe(FakeClient())
            check(f"{label} → 중단", False, '예외가 안 났다')
        except a.MarketMetaUnavailable:
            check(f"{label} → 중단", True)

    reset(TMP, cg=cg_rows(('btc', 'eth', 'sol'), pad=37))
    small = {'symbols': [sym_row(s) for s in ('BTCUSDT', 'ETHUSDT', 'SOLUSDT')] + filler(240)}
    # 거래소엔 많은데 시총 목록과 겹치는 게 적은 경우는 위 pad 로 이미 크다.
    # 여기서는 겹치는 걸 5종으로 줄여 유니버스 바닥을 확인한다.
    thin_cg = ([{'symbol': s, 'market_cap': 10 ** 12} for s in ('btc', 'eth', 'sol')]
               + [{'symbol': f'FILL{i}', 'market_cap': 10 ** 9} for i in range(2)]
               + [{'symbol': f'nolist{i}', 'market_cap': 10 ** 9} for i in range(35)])
    reset(TMP, cg=thin_cg)
    try:
        a.refresh_universe(FakeClient(info=small))
        check("교집합이 바닥 미만이면 중단", False, '예외가 안 났다')
    except a.MarketMetaUnavailable as e:
        check("교집합이 바닥 미만이면 중단", '너무 작다' in str(e), str(e))

    # ── 7b. 거래소 심볼정보 부분 응답 ────────────────────────────────────
    print("\n[7b] 거래소 부분 응답")
    reset(TMP)
    try:
        a.get_exchange_info(FakeClient(info=good_exchange_info(pad=100)))
        check("콜드 스타트에서 잘린 응답(103종)은 중단", False, '예외가 안 났다')
    except a.MarketMetaUnavailable:
        check("콜드 스타트에서 잘린 응답(103종)은 중단", True)
    check("잘린 응답은 캐시에 저장되지 않는다", not os.path.isfile(a.EXCHANGE_INFO_CACHE_PATH))

    reset(TMP)
    a.get_exchange_info(FakeClient(info=good_exchange_info(pad=500)))   # 503종 정상 캐시
    reset(TMP, wipe=False)
    info = a.get_exchange_info(FakeClient(info=good_exchange_info(pad=300)))  # 303종 = 60%
    check("직전 대비 급감한 응답은 캐시로 대체한다", len(_names(info)) == 503, str(len(_names(info))))
    blob = json.load(open(a.EXCHANGE_INFO_CACHE_PATH))
    check("급감 응답이 정상 캐시를 덮지 않았다", len(_names(blob['data'])) == 503)

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
    check("나머지는 그대로 남는다", uni[:2] == ['BTCUSDT', 'ETHUSDT'], str(uni[:5]))

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
    check("거래소 최소 심볼 문턱이 요청 top N 보다 훨씬 크다",
          a.MIN_EXCHANGE_INFO_SYMBOLS >= 200, str(a.MIN_EXCHANGE_INFO_SYMBOLS))
    check("시총 목록 문턱은 고유 심볼 수로 잰다", 'MIN_CG_UNIQUE' in _src and 'MIN_CG_ROWS' not in _src)

finally:
    shutil.rmtree(TMP, ignore_errors=True)

print()
if FAILS:
    print(f"FAILED {len(FAILS)}: {FAILS}")
    sys.exit(1)
print("ALL PASS")
