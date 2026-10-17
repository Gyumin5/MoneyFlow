#!/usr/bin/env python3
"""Cap Defend 현물 라이브 엔진 (V24).

V24 (2026-04-30): 1D 단일 멤버 D_SMA42 (snap=217봉, n_snap=7, drift_threshold=0.10).
바이낸스 spot kline에서 신호 계산 → 업비트 KRW 체결 (executor_coin.py에서 수행).

V24 변경 (vs V22):
- 1D + 4h 2멤버 → 1D 단일 멤버 (4h 멤버 제거)
- snap_interval_bars 60 → 217 (3.6배 확장, 7-tranche stagger)
- n_snapshots 3 → 7
- drift_threshold 0.0 → 0.10 (sleeve-level half-turnover 트리거 도입)
- cron 4h x 6 (9,13,17,21,1,5시) → 1d x 1 (09시)
- ensemble framework 유지: ENSEMBLE_WEIGHTS = {'D_SMA42': 1.0}

가드 없음: gap/exclusion 가드 전면 제거. 앙상블 분산 + Upbit warning/delisting 필터.

drift 트리거 (V24 신규):
- 매 D봉 닫힘 시각 cur_w 와 target_w 의 half_turnover 계산
- half_turnover >= drift_threshold(0.10) AND canary_on AND not crash_cooldown 시 추가 발화
- snap_fire OR drift_fire 면 리밸 발화 (기존: snap_fire 만)
- cur_w 는 자본금 기준 비중 (executor 가 잔고에서 산출 후 engine 에 주입)

설계:
- compute_member_target(): bar-idempotency + n-snap stagger 내장 (n=7 도 동일 메커니즘)
- 엔진 내부 현금 키는 'CASH' (대문자). executor로 넘어갈 때 'Cash' (소문자) 정규화.
- 유니버스: CoinGecko Top40 ∩ Binance spot TRADING ∩ Upbit KRW normal ∩ 253일 이상 ∩ 30일 평균 10억원 ≥ → Top 40
- Freshness: expected_last_bar_ts 일치 체크
- Schema version: V24 (state['schema_version'] = 'V24')

시총 목록 나이 제한 (2026-09-30, 토론 run-20260930T002529Z):
- 시총 목록은 마지막 유효 조회로부터 MARKET_META_MAX_AGE_H(36시간)까지만 쓴다.
- 만료되면 일반 매매를 멈추고 경보한다. 고정 목록으로 대신 채우지 않는다.
- 예외 하나 — BTC 완료봉으로 카나리 OFF 가 독립 확인되면 청산(매도)만 허용한다.
  fail-closed 가드가 매매가 꼭 필요한 유일한 경우를 막은 2026-08-15 사고 때문이다.
- 조회 실패를 빈 유니버스나 현금 100% 목표로 표현하지 않는다 — 그건 전량청산 지시로 읽힌다.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import requests

log = logging.getLogger('coin_live_engine')

# ─── 상수 ───
BINANCE_SPOT_BASE = 'https://api.binance.com'
BINANCE_EXCHANGEINFO_CACHE_SEC = 6 * 3600  # 6시간
COINGECKO_URL = 'https://api.coingecko.com/api/v3/coins/markets'
COINGECKO_FETCH_LIMIT = 40
UNIVERSE_TOP_N = 40
UNIVERSE_MIN_HISTORY_DAYS = 253
UNIVERSE_MIN_TRADE_VALUE_KRW = 1_000_000_000  # 30일 평균 ≥ 10억원

STABLECOINS = {
    'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'FDUSD', 'USDP', 'USDD',
    'GUSD', 'PYUSD', 'LUSD', 'FRAX', 'USTC',
}

UPBIT_MARKET_ALL_URL = 'https://api.upbit.com/v1/market/all?is_details=true'

# 시총 목록 나이 상한 (2026-09-30). 선물(auto_trade_binance.MARKET_META_MAX_AGE_H)과 같은 값.
# 하루 한 번 도는 크론이므로 36시간은 "첫 실패는 약 24시간 캐시로 넘기고 이틀째 실패에서 멈춘다".
MARKET_META_MAX_AGE_H = 36.0
CG_MIN_ROWS = 20          # 이보다 적은 응답은 형태 이상으로 본다 (요청은 40)
CG_SHRINK_RATIO = 0.6     # 직전 캐시 대비 이 비율 미만으로 줄면 급격한 축소로 보고 거부


class MarketMetaUnavailable(RuntimeError):
    """시총 목록을 신뢰할 수 있는 나이 안에서 얻지 못했다.

    조회 실패를 빈 유니버스로 돌려주면 엔진이 현금 100% 를 목표로 내고 그건 전량청산
    지시로 읽힌다. 그래서 오류는 오류로 올린다 — 호출자가 매매 차단인지 카나리 청산인지를
    따로 판정한다.
    """

# V24 멤버 설정 (1D 단일, snap=217봉×7, drift 0.10, 2026-04-30 확정)
# V24 (2026-05-27): refill v2 도입 — drift fire 시 mom2 음수 슬롯을 fresh healthy 로 교체
SCHEMA_VERSION = 'V24'
REFILL_ENABLED = True  # False 로 토글 시 V23 동작 (refill 비활성)
DRIFT_THRESHOLD_SPOT = 0.10  # half-turnover 트리거 임계값
DRIFT_ENABLED = True  # False 로 토글 시 drift_fire 강제 False (snap-only fallback)

MEMBER_D_SMA42 = {
    'interval': 'D',
    'sma_bars': 42,
    'mom_short_bars': 20,
    'mom_long_bars': 127,
    'snap_interval_bars': 217,   # V24: 7*31 (stagger 31)
    'n_snapshots': 7,            # V24: 3 → 7
    'canary_hyst': 0.015,
    'health_mode': 'mom2vol',
    'vol_mode': 'daily',
    'vol_threshold': 0.05,
    'vol_lookback_days': 90,
    'universe_size': 3,
    'cap': 1.0 / 3.0,
    # 전환 후 재평가 (2026-10-18). 카나리가 켜진 봉에서 이만큼 봉이 지난 첫 회차에
    # 전 스냅샷을 한 번 다시 고른다. 채택 BT 둘(unified_backtest.post_flip_delay,
    # backtest_futures_v25.post_flip_delay)이 쓰던 값과 같다. 0 으로 두면 꺼진다.
    'post_flip_delay': 5,
}

MEMBERS = {
    'D_SMA42': MEMBER_D_SMA42,
}

ENSEMBLE_WEIGHTS = {
    'D_SMA42': 1.0,
}

# 각 interval별 최소 가져올 봉 수
# V24: D 단일 (4h 제거). SMA42 + Mom127 + snap217 + vol90 → 최소 ~344. 500 충분
KLINE_LIMITS = {
    'D': 500,
}

# Binance interval 매핑
BINANCE_INTERVAL_MAP = {
    'D': '1d',
}


# ─── 유틸 ───
def utc_now() -> datetime:
    return datetime.now(timezone.utc)


def to_utc_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')


def parse_utc_iso(s: str) -> Optional[datetime]:
    if not s:
        return None
    try:
        s = s.replace('Z', '+00:00')
        return datetime.fromisoformat(s)
    except Exception:
        return None


def expected_last_closed_bar_ts(interval: str, now: Optional[datetime] = None) -> datetime:
    """현재 시점에서 기대되는 '직전 완성 봉'의 open_time (UTC).

    Binance 봉 timestamp는 open_time 기준. 예: 4h봉 '2026-04-13T00:00:00Z' 는
    2026-04-13 00:00~04:00 봉. 이 봉이 닫히려면 04:00 UTC 이상이어야 함.
    """
    now = now or utc_now()
    if interval == 'D':
        today_open = now.replace(hour=0, minute=0, second=0, microsecond=0)
        # 오늘 00:00 봉은 아직 진행중 → 마지막 완성봉은 어제 00:00
        return today_open - timedelta(days=1)
    elif interval == '4h':
        h = (now.hour // 4) * 4
        cur_bar_open = now.replace(hour=h, minute=0, second=0, microsecond=0)
        return cur_bar_open - timedelta(hours=4)
    elif interval == '1h':
        # 현재 시간 봉은 진행중. 마지막 완성봉은 직전 시간 open_time.
        cur_bar_open = now.replace(minute=0, second=0, microsecond=0)
        return cur_bar_open - timedelta(hours=1)
    raise ValueError(f'unsupported interval: {interval}')


def normalize_cash_key(target: Dict[str, float]) -> Dict[str, float]:
    """엔진 내부 'CASH' → executor 'Cash'로 통일."""
    out = {}
    for k, v in target.items():
        key = 'Cash' if k.upper() == 'CASH' else k
        out[key] = out.get(key, 0.0) + v
    return out


# ─── 데이터 수집: CoinGecko / Binance / Upbit ───
def _cg_auth_headers() -> Dict[str, str]:
    """CoinGecko 데모 키가 있으면 인증 헤더를 얹는다.

    2026-09-30: 키 없는 공개 호출이 403(CloudFront Request blocked)으로 막혔다.
    키가 없으면 예전처럼 키 없이 부른다 — 그 경로는 지금 막혀 있으므로 캐시로 떨어진다.
    """
    try:
        from config import COINGECKO_API_KEY
        key = (COINGECKO_API_KEY or '').strip()
    except ImportError:
        key = os.environ.get('COINGECKO_API_KEY', '').strip()
    return {'x-cg-demo-api-key': key} if key else {}


def validate_cg_rows(data, prev_count: int = 0) -> Tuple[bool, str]:
    """시총 응답이 캐시에 넣어도 되는 형태인가.

    형태·고유 종목 수·직전 대비 급격한 축소 셋을 본다. 망가진 응답으로 캐시를 덮어쓰면
    그 순간부터 나이 제한이 아무 의미가 없어진다 — 신선하지만 틀린 목록이 되기 때문이다.
    """
    if not isinstance(data, list):
        return False, f'목록이 아님({type(data).__name__})'
    if len(data) < CG_MIN_ROWS:
        return False, f'행 수 부족({len(data)} < {CG_MIN_ROWS})'
    syms = set()
    for item in data:
        if not isinstance(item, dict):
            return False, '행이 딕셔너리가 아님'
        sym = str(item.get('symbol', '') or '').strip().upper()
        if not sym:
            return False, '심볼 빈 행 있음'
        syms.add(sym)
    if len(syms) < CG_MIN_ROWS:
        return False, f'고유 심볼 부족({len(syms)} < {CG_MIN_ROWS})'
    if prev_count and len(data) < prev_count * CG_SHRINK_RATIO:
        return False, f'급격한 축소({prev_count} → {len(data)})'
    return True, ''


def _load_cg_cache(cache_path: Optional[str]) -> Tuple[Optional[List[Dict]], Optional[datetime], str]:
    """캐시 파일 → (데이터, 실제 성공 시각, 사유). 못 읽으면 (None, None, 사유)."""
    if not cache_path or not os.path.exists(cache_path):
        return None, None, '캐시 없음'
    try:
        with open(cache_path) as f:
            cached = json.load(f)
    except Exception as e:
        return None, None, f'캐시 로드 실패({e})'
    ts = parse_utc_iso(cached.get('ts', ''))
    if ts is None:
        return None, None, '캐시 시각 없음/손상'
    data = cached.get('data')
    if not isinstance(data, list) or not data:
        return None, ts, '캐시 내용 비어있음'
    return data, ts, ''


def fetch_coingecko_top(session: requests.Session, fetch_limit: int = COINGECKO_FETCH_LIMIT,
                         retries: int = 5, cache_path: Optional[str] = None,
                         max_age_h: float = MARKET_META_MAX_AGE_H,
                         now_utc: Optional[datetime] = None) -> Tuple[List[Dict], Dict]:
    """CoinGecko Top N 가져오기. 실패하면 캐시로 떨어지되 나이 상한을 지킨다.

    Returns: (데이터, meta) — meta = {'source': 'live'|'cache', 'ts': iso, 'age_h': float}
    Raises: MarketMetaUnavailable — 신선한 조회도 실패하고 쓸 수 있는 캐시도 없을 때.
    """
    now_utc = now_utc or utc_now()
    prev_data, prev_ts, _ = _load_cg_cache(cache_path)
    prev_count = len(prev_data) if prev_data else 0

    headers = {'accept': 'application/json', 'User-Agent': 'Mozilla/5.0 CoinExecutor/1.0'}
    headers.update(_cg_auth_headers())
    for attempt in range(1, retries + 1):
        try:
            params = {'vs_currency': 'usd', 'order': 'market_cap_desc',
                      'per_page': fetch_limit, 'page': 1}
            resp = session.get(COINGECKO_URL, params=params, headers=headers, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                ok, why = validate_cg_rows(data, prev_count)
                if not ok:
                    # 캐시를 덮어쓰지 않는다. 다음 시도로 넘긴다.
                    log.warning('CoinGecko 응답 검증 실패 (attempt %d): %s', attempt, why)
                    time.sleep(10)
                    continue
                fetched_at = utc_now()
                if cache_path:
                    try:
                        tmp = cache_path + '.tmp'
                        with open(tmp, 'w') as f:
                            json.dump({'ts': to_utc_iso(fetched_at), 'data': data}, f)
                        os.replace(tmp, cache_path)
                    except Exception as e:
                        log.warning('CoinGecko 캐시 저장 실패: %s', e)
                return data, {'source': 'live', 'ts': to_utc_iso(fetched_at), 'age_h': 0.0}
            elif resp.status_code == 429:
                wait = 30 * attempt
                log.warning('CoinGecko 429 rate limit, %ds 대기 (attempt %d)', wait, attempt)
                time.sleep(wait)
            else:
                log.warning('CoinGecko status %d (attempt %d)', resp.status_code, attempt)
                time.sleep(10)
        except Exception as e:
            log.warning('CoinGecko 호출 오류 (attempt %d): %s', attempt, e)
            time.sleep(10)

    # 캐시 fallback — 저장된 실제 성공 시각으로만 나이를 잰다.
    data, ts, why = _load_cg_cache(cache_path)
    if data is None or ts is None:
        raise MarketMetaUnavailable(f'시총 목록 조회 실패, 캐시 사용 불가 ({why})')
    age_h = (now_utc - ts).total_seconds() / 3600.0
    if age_h < 0:
        raise MarketMetaUnavailable(f'시총 캐시 시각이 미래 ({to_utc_iso(ts)})')
    if age_h > max_age_h:
        raise MarketMetaUnavailable(
            f'시총 캐시 {age_h:.1f}시간 경과 (상한 {max_age_h:.0f}시간, ts={to_utc_iso(ts)})')
    ok, why = validate_cg_rows(data)
    if not ok:
        raise MarketMetaUnavailable(f'시총 캐시 내용 이상 ({why})')
    log.warning('CoinGecko 캐시 fallback 사용 (ts=%s, %.1f시간 전)', to_utc_iso(ts), age_h)
    return data, {'source': 'cache', 'ts': to_utc_iso(ts), 'age_h': age_h}


def fetch_binance_exchange_info(session: requests.Session,
                                  cache_path: Optional[str] = None) -> Dict[str, str]:
    """Binance spot exchangeInfo → {BTC: TRADING, ETH: HALT, ...}
    6시간 캐시 유효. 신규 종목엔 캐시가 충분하지만 보유 종목의 TRADING 상태는 실조회 우선 (caller가 판단).
    """
    # 캐시 확인
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            cached_ts = parse_utc_iso(cached.get('ts', ''))
            if cached_ts and (utc_now() - cached_ts).total_seconds() < BINANCE_EXCHANGEINFO_CACHE_SEC:
                return cached.get('data', {})
        except Exception as e:
            log.warning('Binance exchangeInfo 캐시 로드 실패: %s', e)

    try:
        resp = session.get(f'{BINANCE_SPOT_BASE}/api/v3/exchangeInfo', timeout=20)
        if resp.status_code != 200:
            log.warning('Binance exchangeInfo status %d', resp.status_code)
            return {}
        data = resp.json()
        symbol_status = {}
        for s in data.get('symbols', []):
            if s.get('quoteAsset') == 'USDT':
                base = s.get('baseAsset', '')
                status = s.get('status', '')
                symbol_status[base] = status
        if cache_path:
            try:
                tmp = cache_path + '.tmp'
                with open(tmp, 'w') as f:
                    json.dump({'ts': to_utc_iso(utc_now()), 'data': symbol_status}, f)
                os.replace(tmp, cache_path)
            except Exception as e:
                log.warning('Binance exchangeInfo 캐시 저장 실패: %s', e)
        return symbol_status
    except Exception as e:
        log.warning('Binance exchangeInfo 호출 오류: %s', e)
        if cache_path and os.path.exists(cache_path):
            try:
                with open(cache_path) as f:
                    return json.load(f).get('data', {})
            except Exception:
                pass
        return {}


def fetch_upbit_market_status(session: requests.Session) -> Dict[str, Dict]:
    """Upbit 전체 마켓 상태 → {BTC: {warning, suspended, listed}, ...}"""
    try:
        resp = session.get(UPBIT_MARKET_ALL_URL, timeout=10)
        if resp.status_code != 200:
            log.warning('Upbit market/all status %d', resp.status_code)
            return {}
        data = resp.json()
        result = {}
        for m in data:
            market = m.get('market', '')
            if not market.startswith('KRW-'):
                continue
            coin = market[4:]
            warning = m.get('market_event', {}).get('warning', False)
            caution_any = any(m.get('market_event', {}).get('caution', {}).values())
            result[coin] = {
                'warning': bool(warning),
                'caution': bool(caution_any),
                'listed': True,
            }
        return result
    except Exception as e:
        log.warning('Upbit market/all 호출 오류: %s', e)
        return {}


def fetch_binance_klines(session: requests.Session, symbol: str, interval: str,
                          limit: int) -> pd.DataFrame:
    """Binance spot kline REST 호출. 최대 1000개씩 페이징.

    반환: DataFrame(index=DatetimeIndex UTC, columns=[Open, High, Low, Close, Volume])
    실패: 빈 DataFrame
    """
    b_iv = BINANCE_INTERVAL_MAP.get(interval)
    if b_iv is None:
        return pd.DataFrame()

    max_batch = 1000
    remaining = int(limit)
    end_time_ms: Optional[int] = None
    chunks: List[list] = []
    url = f'{BINANCE_SPOT_BASE}/api/v3/klines'

    try:
        while remaining > 0:
            batch = min(remaining, max_batch)
            params: Dict = {'symbol': symbol, 'interval': b_iv, 'limit': batch}
            if end_time_ms is not None:
                params['endTime'] = end_time_ms
            resp = session.get(url, params=params, timeout=15)
            if resp.status_code != 200:
                log.warning('klines %s %s status %d', symbol, interval, resp.status_code)
                break
            kls = resp.json()
            if not kls:
                break
            chunks.extend(kls)
            remaining -= len(kls)
            if len(kls) < batch:
                break
            end_time_ms = int(kls[0][0]) - 1
            time.sleep(0.05)

        if not chunks:
            return pd.DataFrame()

        dedup = {}
        for row in chunks:
            dedup[int(row[0])] = row
        rows = [dedup[k] for k in sorted(dedup.keys())]

        df = pd.DataFrame(rows, columns=[
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
            'CloseTime', 'QuoteVol', 'Trades', 'TakerBuy', 'TakerQuote', 'Ignore'
        ])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms', utc=True)
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[c] = df[c].astype(float)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].set_index('Date')
        return df.sort_index()
    except Exception as e:
        log.warning('klines %s %s 예외: %s', symbol, interval, e)
        return pd.DataFrame()


# ─── 유니버스 ───
def build_dynamic_universe(session: requests.Session, cache_dir: str,
                            upbit_price_fn=None) -> Tuple[List[str], Dict[str, Dict]]:
    """동적 유니버스 생성.

    Returns:
        (universe, meta) — universe = ['BTC', 'ETH', ...] (심볼 only, '-USD' 없음)
        meta = {
            'coingecko_count': int,
            'binance_trading_count': int,
            'upbit_listed_count': int,
            'filtered_count': int,
            'cg_ts': str, 'cg_age_h': float, 'cg_source': 'live'|'cache',
            'upbit_status': {coin: {warning, caution, listed}},
            'binance_status': {coin: 'TRADING' or other},
        }

    Raises:
        MarketMetaUnavailable — 시총 목록·거래소 상태 중 하나라도 못 얻었을 때. 고정 목록으로
        대신 채우지 않는다(2026-09-30 토론 run-20260930T002529Z): 그건 목록을 조용히 바꾸는
        것이라 조회 실패가 다른 전략으로 도는 결과가 된다.
    """
    cg_cache = os.path.join(cache_dir, 'universe_cg_cache.json')
    bn_cache = os.path.join(cache_dir, 'binance_exchinfo_cache.json')
    os.makedirs(cache_dir, exist_ok=True)

    meta: Dict = {'binance_status': {}, 'upbit_status': {}}

    cg_data, cg_meta = fetch_coingecko_top(session, cache_path=cg_cache)
    binance_status = fetch_binance_exchange_info(session, cache_path=bn_cache)
    upbit_status = fetch_upbit_market_status(session)

    meta['coingecko_count'] = len(cg_data)
    meta['cg_ts'] = cg_meta.get('ts')
    meta['cg_age_h'] = cg_meta.get('age_h')
    meta['cg_source'] = cg_meta.get('source')
    meta['binance_trading_count'] = sum(1 for s in binance_status.values() if s == 'TRADING')
    meta['upbit_listed_count'] = len(upbit_status)
    meta['binance_status'] = binance_status
    meta['upbit_status'] = upbit_status

    # 거래소 상태 조회 실패는 각각 따로 올린다 — 어느 쪽이 막혔는지가 알림에 남아야 한다.
    if not binance_status:
        raise MarketMetaUnavailable('바이낸스 거래상태 조회 실패')
    if not upbit_status:
        raise MarketMetaUnavailable('업비트 상장상태 조회 실패')

    final: List[str] = []
    # Top N 시총순으로 순회
    for item in cg_data:
        sym = item.get('symbol', '').upper()
        if not sym or sym in STABLECOINS:
            continue
        if binance_status.get(sym) != 'TRADING':
            continue
        upbit_info = upbit_status.get(sym)
        if not upbit_info or not upbit_info.get('listed', False):
            continue
        if upbit_info.get('warning', False):
            continue
        # 히스토리 + 거래대금 체크 (Upbit 기준)
        if upbit_price_fn is not None:
            try:
                df = upbit_price_fn(f'KRW-{sym}')
                if df is None or len(df) < UNIVERSE_MIN_HISTORY_DAYS:
                    continue
                # 30일 평균 거래대금
                if 'value' in df.columns:
                    avg_val = float(df['value'].iloc[-30:].mean())
                else:
                    continue
                if avg_val < UNIVERSE_MIN_TRADE_VALUE_KRW:
                    continue
            except Exception as e:
                log.debug('Upbit 히스토리 체크 실패 %s: %s', sym, e)
                continue
        final.append(sym)
        if len(final) >= UNIVERSE_TOP_N:
            break

    meta['filtered_count'] = len(final)
    return final, meta


# ─── 시그널 계산 ───
def _calc_sma(arr: np.ndarray, period: int) -> float:
    if len(arr) < period:
        return 0.0
    return float(np.mean(arr[-period:]))


def _calc_mom(arr: np.ndarray, period: int) -> float:
    if len(arr) < period + 1:
        return -999.0
    return float(arr[-1] / arr[-period - 1] - 1.0)


def _calc_vol_daily(arr: np.ndarray, bpd: int, lookback_bars: int) -> float:
    if len(arr) < lookback_bars + 1:
        return 999.0
    daily = arr[-lookback_bars::bpd]
    if len(daily) < 10:
        return 999.0
    return float(np.std(np.diff(np.log(daily))))


def _calc_vol_bars(arr: np.ndarray, lookback_bars: int, bars_per_year: int) -> float:
    """봉 기반 변동성 (연환산). backtest_futures_full.calc_vol_bars와 동일 정의."""
    if len(arr) < lookback_bars + 1:
        return 999.0
    rets = np.diff(np.log(arr[-lookback_bars - 1:]))
    return float(np.std(rets) * np.sqrt(bars_per_year))


def _bars_per_day(interval: str) -> int:
    return {'D': 1, '4h': 6, '2h': 12, '1h': 24}[interval]


def replay_canary_pfd(closes: np.ndarray, bar_ts: List[str], last_bar_ts: Optional[str],
                      prev_canary: bool, sma_p: int, hyst: float,
                      flip_bar_ts: Optional[str], pfd_done: bool,
                      pfd_bars: int, allow_arm: bool
                      ) -> Tuple[bool, bool, bool, Optional[str], bool, float]:
    """마지막으로 처리한 봉 다음 봉부터 이번 봉까지 차례로 카나리를 매긴다.

    크론을 걸러 봉이 여러 개 밀렸을 때 중간 봉의 전환(ON→OFF→ON)과 전환 후 재평가
    예약이 그대로 반영된다. 이번 봉만 보고 경과를 세면 그 사이의 전환이 통째로 사라진다.
    과거 주문은 내지 않는다 — 밀린 사이에 재평가가 예정돼 있었으면 이번 봉에서 지금
    자료로 한 번 수행한다.

    closes[i] 와 bar_ts[i] 는 같은 봉이고, 마지막 원소가 이번에 처리할 완성봉이다.

    반환 (canary_on, canary_flipped, pfd_fire, flip_bar_ts, pfd_done, sma_val)
    - canary_flipped: 재생 구간 어느 봉에서든 전환이 있었나. 밀린 사이에 전환이 끝나
      저장된 값과 결과가 같아도 스냅샷은 다시 골라야 하므로 '마지막 봉의 전환' 이 아니다.
    - pfd_fire: 이번 회차에 전 스냅샷 재평가를 수행해야 하나.
    """
    n = len(closes)
    start = n - 1
    if last_bar_ts is not None and last_bar_ts in bar_ts:
        start = bar_ts.index(last_bar_ts) + 1
    else:
        # 직전 상태를 모르는 회차 — 예약을 걸지 않는다(계약: 최초 초기화·상태 유실은
        # 전환으로 세지 않는다).
        allow_arm = False
    start = min(max(start, sma_p), n - 1)

    canary_on = prev_canary
    flipped_any = False
    pfd_fire = False
    sma_val = 0.0
    for i in range(start, n):
        c = closes[:i + 1]
        sma_val = _calc_sma(c, sma_p)
        cur = float(c[-1])
        prev = canary_on
        if prev:
            canary_on = not (cur < sma_val * (1 - hyst))
        else:
            canary_on = cur > sma_val * (1 + hyst)
        flipped = canary_on != prev
        flipped_any = flipped_any or flipped
        if flipped and canary_on:
            if allow_arm:
                flip_bar_ts = bar_ts[i]
                pfd_done = False
            else:
                flip_bar_ts = None
                pfd_done = True
        elif flipped and not canary_on:
            # 꺼지면 예약은 사라진다. 다시 켜지면 새 예약이 0 부터 센다.
            flip_bar_ts = None
            pfd_done = True
        elif pfd_bars > 0 and canary_on and flip_bar_ts is not None and not pfd_done:
            fi = bar_ts.index(flip_bar_ts) if flip_bar_ts in bar_ts else None
            if fi is not None and (i - fi) >= pfd_bars:
                pfd_done = True
                pfd_fire = True

    return canary_on, flipped_any, pfd_fire, flip_bar_ts, pfd_done, sma_val


@dataclass
class MemberState:
    canary_on: bool = False
    last_bar_ts: Optional[str] = None
    snapshots: List[Dict[str, float]] = field(default_factory=list)
    bar_counter: int = 0
    last_combined: Dict[str, float] = field(default_factory=lambda: {'CASH': 1.0})
    snap_id: int = 0
    # 전환 후 재평가 예약 (2026-10-18). flip_bar_ts 는 카나리가 켜진 봉의 타임스탬프,
    # pfd_done 은 그 예약이 이미 소비됐는지다. 옛 상태파일에는 둘 다 없으므로 기본값이
    # "예약 없음 + 소비됨" 이고, 그 회차는 발화하지 않는다.
    flip_bar_ts: Optional[str] = None
    pfd_done: bool = True

    @classmethod
    def from_dict(cls, d: Dict) -> 'MemberState':
        ms = cls()
        if d:
            ms.canary_on = bool(d.get('canary_on', False))
            ms.last_bar_ts = d.get('last_bar_ts')
            ms.snapshots = d.get('snapshots', []) or []
            ms.bar_counter = int(d.get('bar_counter', 0))
            ms.last_combined = d.get('last_combined', {'CASH': 1.0}) or {'CASH': 1.0}
            ms.snap_id = int(d.get('snap_id', 0))
            ms.flip_bar_ts = d.get('flip_bar_ts')
            ms.pfd_done = bool(d.get('pfd_done', True))
        return ms

    def to_dict(self) -> Dict:
        return {
            'canary_on': self.canary_on,
            'last_bar_ts': self.last_bar_ts,
            'snapshots': self.snapshots,
            'bar_counter': self.bar_counter,
            'last_combined': self.last_combined,
            'snap_id': self.snap_id,
            'flip_bar_ts': self.flip_bar_ts,
            'pfd_done': self.pfd_done,
        }


@dataclass
class MemberComputeResult:
    target: Dict[str, float]           # 'CASH' 포함
    new_state: MemberState
    fresh: bool                        # expected_last_bar_ts 일치 여부
    new_bar: bool                      # 새 봉이어서 재계산 수행했는지
    canary_flipped: bool
    alerts: List[str]                  # 텔레그램 알림용 메시지
    canary_info: Optional[Dict] = None  # {'on', 'ratio', 'cur', 'sma_val', 'sma_p'}
    pfd_fired: bool = False            # 이번 봉에 전환 후 재평가가 발화했나


def compute_member_target(member_name: str, cfg: Dict, bars: Dict[str, pd.DataFrame],
                          universe: List[str], state: MemberState,
                          now_utc: datetime) -> MemberComputeResult:
    """단일 멤버 target 계산 (auto_trade_binance.compute_strategy_target 이식).

    - bar-idempotency: latest_bar_ts == state.last_bar_ts 이면 재계산 스킵
    - 3-snapshot stagger: bar_counter % snap_interval == offset 마다 각 스냅샷 갱신
    - 가드 없음 (2026-04-21): gap/exclusion 필터 제거. Upbit warning/delisting은
      상위 호출자(compute_live_targets)에서 universe 레벨로 이미 걸러짐.

    bars 는 'Close' 등 포함 DataFrame. 사전에 "직전 완성봉까지만" slice 되어 있어야 함
    (즉 iloc[-1] 이 직전 완성봉, iloc[-2] 가 그 전 봉).
    """
    interval = cfg['interval']
    bpd = _bars_per_day(interval)
    sma_p = cfg['sma_bars']
    mom_s = cfg['mom_short_bars']
    mom_l = cfg['mom_long_bars']
    snap_iv = cfg['snap_interval_bars']
    n_snap = cfg['n_snapshots']
    cap = cfg['cap']
    hmode = cfg['health_mode']
    vol_mode = cfg['vol_mode']
    vol_th = cfg['vol_threshold']
    vol_lookback = cfg['vol_lookback_days'] * bpd
    hyst = cfg['canary_hyst']
    univ_size = cfg['universe_size']

    btc_df = bars.get('BTC')
    alerts: List[str] = []

    if btc_df is None or len(btc_df) < sma_p + 2:
        log.warning('%s: BTC 데이터 부족 → CASH 100%%', member_name)
        state.last_combined = {'CASH': 1.0}
        return MemberComputeResult(target={'CASH': 1.0}, new_state=state,
                                   fresh=False, new_bar=False,
                                   canary_flipped=False, alerts=alerts)

    # Freshness: 마지막 봉(iloc[-1])이 expected_last_closed_bar_ts와 일치해야
    # (bars는 사전에 '직전 완성봉까지'로 slice되어 있다고 가정)
    expected_ts = expected_last_closed_bar_ts(interval, now_utc)
    latest_ts = btc_df.index[-1].to_pydatetime()
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.replace(tzinfo=timezone.utc)
    fresh = latest_ts == expected_ts

    latest_bar_str = to_utc_iso(latest_ts)
    if latest_bar_str == state.last_bar_ts:
        # 같은 봉 — 이전 target 유지
        log.info('%s: 봉 동일 (%s) → 이전 target 유지', member_name, latest_bar_str)
        return MemberComputeResult(target=dict(state.last_combined or {'CASH': 1.0}),
                                   new_state=state, fresh=fresh, new_bar=False,
                                   canary_flipped=False, alerts=alerts)

    # 새 봉 — 재계산
    # 스냅샷 초기화 여부를 먼저 본다 (첫 실행 또는 n_snap 변경 시). 전환 후 재평가
    # 예약은 직전 상태를 아는 회차에서만 걸기 때문에 카나리 판정보다 앞에 둔다.
    freshly_initialized = (not state.snapshots or len(state.snapshots) != n_snap)
    if freshly_initialized:
        state.snapshots = [{'CASH': 1.0} for _ in range(n_snap)]

    # 카나리 (BTC 종가 vs SMA) — 마지막 처리 봉 다음 봉부터 차례로
    btc_close = btc_df['Close'].values
    prev_canary = state.canary_on
    bar_ts_list = [to_utc_iso(ts.to_pydatetime()) for ts in btc_df.index]
    pfd_bars = int(cfg.get('post_flip_delay', 0)) * bpd
    (canary_on, canary_flipped, pfd_fire,
     state.flip_bar_ts, state.pfd_done, sma_val) = replay_canary_pfd(
        btc_close, bar_ts_list, state.last_bar_ts, prev_canary, sma_p, hyst,
        state.flip_bar_ts, state.pfd_done, pfd_bars,
        allow_arm=not freshly_initialized)
    if pfd_fire and not fresh:
        # 자료가 밀린 회차에서 예약을 소비하지 않는다 — 다음 정상 회차에 발화한다.
        pfd_fire = False
        state.pfd_done = False
    cur = float(btc_close[-1])
    ratio = cur / sma_val if sma_val > 0 else 0.0
    log.info('%s: BTC=%.2f SMA%d=%.2f ratio=%.4f canary=%s%s',
             member_name, cur, sma_p, sma_val, ratio, 'ON' if canary_on else 'OFF',
             ' *FLIP*' if canary_flipped else '')
    if canary_flipped:
        alerts.append(f'{member_name} 카나리 {"ON 🟢" if canary_on else "OFF 🔴"} '
                      f'| BTC ${cur:,.0f} / SMA ${sma_val:,.0f} ({ratio:.3f})')

    def compute_weights(snap_idx: int) -> Dict[str, float]:
        if not canary_on:
            return {'CASH': 1.0}

        healthy: List[str] = []
        for coin in universe:
            df = bars.get(coin)
            if df is None or df.empty:
                continue
            c = df['Close'].values
            min_bars = max(mom_s, mom_l, sma_p, vol_lookback)
            if len(c) < min_bars + 1:
                continue

            m_short = _calc_mom(c, mom_s)
            m_long = _calc_mom(c, mom_l) if 'mom2' in hmode else 999.0
            if vol_mode == 'daily':
                vol = _calc_vol_daily(c, bpd, vol_lookback)
            elif vol_mode == 'bar':
                vol = _calc_vol_bars(c, vol_lookback, bpd * 365)
            else:
                vol = 999.0

            if hmode == 'mom2vol':
                ok = m_short > 0 and m_long > 0 and vol <= vol_th
            elif hmode == 'mom1vol':
                ok = m_short > 0 and vol <= vol_th
            elif hmode == 'mom1':
                ok = m_short > 0
            else:
                ok = True

            if ok:
                healthy.append(coin)

        picks = healthy[:univ_size]
        # Greedy absorption — 뒤쪽(낮은 시총)부터 앞쪽과 비교, 앞쪽 모멘텀이 더 높으면 뒤 제거
        if len(picks) > 1:
            for i in range(len(picks) - 1, 0, -1):
                df_a = bars.get(picks[i - 1])
                df_b = bars.get(picks[i])
                if df_a is None or df_b is None:
                    continue
                ca = df_a['Close'].values
                cb = df_b['Close'].values
                ma = _calc_mom(ca, mom_s)
                mb = _calc_mom(cb, mom_s)
                if ma >= mb:
                    picks.pop(i)

        if not picks:
            return {'CASH': 1.0}

        w = min(1.0 / len(picks), cap)
        weights = {coin: w for coin in picks}
        total = sum(weights.values())
        if total < 0.999:
            weights['CASH'] = 1.0 - total
        return weights

    # 스냅샷 갱신
    pfd_fired_now = False
    if canary_flipped or (freshly_initialized and canary_on):
        # 전 스냅샷 즉시 전환 (플립 또는 최초 초기화 시 전체 populate)
        for si in range(n_snap):
            state.snapshots[si] = compute_weights(si)
        state.snap_id += 1
    elif pfd_fire:
        # 전환 후 재평가 — 전환 봉에서 pfd_bars 봉이 지난 첫 회차에 한 번.
        # 전환일에 통과 코인이 0 이면 전 스냅샷이 현금으로 채워져 다음 스태거 자리까지
        # 슬리브가 잠긴다(최장 31일, history #72). 드리프트는 현금 100%→현금 100% 라
        # 반회전율이 0 이어서 못 깨운다. 채택 BT 둘의 pfd 분기와 같은 자리다 —
        # 전환 분기와 배타이고, 앵커보다 앞이다. 앵커를 삼키지는 않는다 — 이 분기가
        # 돈 봉의 앵커는 같은 자료로 같은 값을 다시 계산해 '변화 없음' 으로 끝난다.
        for si in range(n_snap):
            state.snapshots[si] = compute_weights(si)
        state.snap_id += 1
        pfd_fired_now = True
        log.info('%s: 전환 후 %d봉 재평가 발화 (전환봉=%s)', member_name, pfd_bars,
                 state.flip_bar_ts)
    elif canary_on:
        for si in range(n_snap):
            offset = int(si * snap_iv / n_snap)
            if state.bar_counter % snap_iv == offset:
                new_w = compute_weights(si)
                if new_w != state.snapshots[si]:
                    state.snapshots[si] = new_w
                    state.snap_id += 1
    # canary OFF: 스냅샷 건드리지 않음 (compute_weights는 어차피 CASH 반환)

    # 스냅샷 합산
    combined: Dict[str, float] = {}
    for snap in state.snapshots:
        for t, w in snap.items():
            combined[t] = combined.get(t, 0.0) + w / n_snap
    total = sum(combined.values())
    if total > 0:
        combined = {t: w / total for t, w in combined.items()}
    # canary OFF면 강제 CASH
    if not canary_on:
        combined = {'CASH': 1.0}

    # 상태 업데이트
    state.canary_on = canary_on
    state.bar_counter += 1
    state.last_bar_ts = latest_bar_str
    state.last_combined = combined

    return MemberComputeResult(target=dict(combined), new_state=state,
                               fresh=fresh, new_bar=True,
                               canary_flipped=canary_flipped, alerts=alerts,
                               pfd_fired=pfd_fired_now,
                               canary_info={'on': canary_on, 'ratio': ratio,
                                            'cur': cur, 'sma_val': sma_val, 'sma_p': sma_p})


# ─── 앙상블 + 데이터 슬라이스 ───
def slice_to_last_closed(bars: Dict[str, pd.DataFrame], interval: str,
                          now_utc: datetime) -> Dict[str, pd.DataFrame]:
    """bars를 '가장 최근 완성봉까지'로 잘라 반환.

    Binance kline open_time 기준. D봉 open_time=오늘 00:00이면 아직 진행중이므로 제외.
    4h봉 마찬가지.
    """
    expected_ts = expected_last_closed_bar_ts(interval, now_utc)
    sliced = {}
    for coin, df in bars.items():
        if df is None or df.empty:
            continue
        mask = df.index <= expected_ts
        sd = df[mask]
        if len(sd) > 0:
            sliced[coin] = sd
    return sliced


def combine_ensemble(member_targets: Dict[str, Dict[str, float]],
                      weights: Dict[str, float]) -> Dict[str, float]:
    """멤버별 target → 가중 평균 단일 target. 모두 'CASH' 대문자."""
    merged: Dict[str, float] = {}
    total_w = sum(weights.values())
    if total_w <= 0:
        return {'CASH': 1.0}
    for name, w in weights.items():
        tgt = member_targets.get(name)
        if not tgt:
            tgt = {'CASH': 1.0}
        nw = w / total_w
        for coin, cw in tgt.items():
            merged[coin] = merged.get(coin, 0.0) + cw * nw
    # 정규화
    s = sum(merged.values())
    if s > 0:
        merged = {k: v / s for k, v in merged.items()}
    return merged



# ─── V24 drift 트리거 유틸 ───
def half_turnover(cur_w: Dict[str, float], tgt_w: Dict[str, float]) -> float:
    """sleeve-level half-turnover. Cash/CASH 모두 동일 키로 정규화 후 계산.

    cur_w, tgt_w 는 같은 universe (자본금 기준 비중). 합 ≈ 1.0
    """
    def _norm(d: Dict[str, float]) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for k, v in d.items():
            kk = 'Cash' if k.upper() == 'CASH' else k
            out[kk] = out.get(kk, 0.0) + float(v)
        return out
    a, b = _norm(cur_w), _norm(tgt_w)
    keys = set(a) | set(b)
    return sum(abs(b.get(k, 0.0) - a.get(k, 0.0)) for k in keys) / 2


def _apply_refill_v2_to_state(state: Dict, members_cfg: Dict,
                                bars_by_member: Dict[str, Dict[str, pd.DataFrame]],
                                combined: Dict[str, float]) -> Dict[str, float]:
    """V24 refill v2 — drift fire 시점에 각 멤버 snapshot 의 mom2 음수 코인 자리를
    fresh healthy 코인으로 교체. 모든 멤버에 적용 후 combined 재계산해서 반환.

    Fail 기준: mom_short<0 AND mom_long<0 (vol 미포함).
    교체 대상: fresh healthy_pool (mom_short>0 AND mom_long>0) 중 snapshot 에 없는 종목.
    """
    ms_all = state.setdefault('members', {})
    n_members = len(members_cfg)
    new_combined: Dict[str, float] = {}
    log.info('refill v2: 시작. members=%d', n_members)

    for mname, cfg in members_cfg.items():
        bars = bars_by_member.get(mname, {})
        if not bars:
            log.warning('refill v2 [%s]: bars 없음 → skip', mname)
            continue
        mom_s = cfg['mom_short_bars']
        mom_l = cfg['mom_long_bars']
        mstate = ms_all.get(mname, {})
        snapshots = mstate.get('snapshots') or []
        if not snapshots:
            log.warning('refill v2 [%s]: snapshots 없음 → skip', mname)
            continue
        log.info('refill v2 [%s]: snapshots=%d mom_s=%d mom_l=%d', mname, len(snapshots), mom_s, mom_l)

        # fail 판정 (캐시: 같은 코인 중복 호출 방지)
        _fail_cache: Dict[str, Tuple[bool, float, float]] = {}

        def _is_failed(coin: str) -> bool:
            if coin in _fail_cache:
                return _fail_cache[coin][0]
            df = bars.get(coin)
            if df is None or df.empty:
                _fail_cache[coin] = (False, 0.0, 0.0)
                return False
            c = df['Close'].values
            if len(c) < max(mom_s, mom_l) + 1:
                _fail_cache[coin] = (False, 0.0, 0.0)
                return False
            ms = _calc_mom(c, mom_s)
            ml = _calc_mom(c, mom_l)
            fail = ms < 0 and ml < 0
            _fail_cache[coin] = (fail, ms, ml)
            return fail

        # fresh healthy pool (mom2 양수). tie-break: 심볼 오름차순으로 결정성 보장
        healthy_pool: List[Tuple[str, float]] = []
        for coin, df in bars.items():
            if coin == 'BTC':
                continue
            c = df['Close'].values
            if len(c) < max(mom_s, mom_l) + 1:
                continue
            ms = _calc_mom(c, mom_s)
            ml = _calc_mom(c, mom_l)
            if ms > 0 and ml > 0:
                healthy_pool.append((coin, ms))
        healthy_pool.sort(key=lambda x: (-x[1], x[0]))
        healthy_sorted = [c for c, _ in healthy_pool]
        log.info('refill v2 [%s]: healthy_pool top5=%s (total %d)', mname,
                 [(c, f'{m*100:.1f}%') for c, m in healthy_pool[:5]], len(healthy_pool))

        # 각 snapshot 처리
        new_snapshots = []
        per_snap_changes = []
        for si, snap in enumerate(snapshots):
            sn_coins = sorted([c for c in snap if c.upper() != 'CASH'])
            new_sn: Dict[str, float] = {}
            replaced = 0.0
            failed_coins: List[str] = []
            for c in sn_coins:
                if _is_failed(c):
                    replaced += float(snap.get(c, 0.0))
                    failed_coins.append(c)
                else:
                    new_sn[c] = float(snap.get(c, 0.0))
            cash_keys = [k for k in snap if k.upper() == 'CASH']
            cash_w = sum(float(snap.get(k, 0.0)) for k in cash_keys)
            new_sn['CASH'] = cash_w
            picks_made: List[str] = []
            if replaced > 0:
                fresh_picks = [c for c in healthy_sorted if c not in new_sn]
                n_failed = len(sn_coins) - sum(1 for k in new_sn if k.upper() != 'CASH')
                n_failed = max(1, n_failed)
                if fresh_picks:
                    picks_made = fresh_picks[:n_failed]
                    w_per = replaced / len(picks_made)
                    for c in picks_made:
                        new_sn[c] = new_sn.get(c, 0.0) + w_per
                else:
                    new_sn['CASH'] = new_sn.get('CASH', 0.0) + replaced
            new_snapshots.append(new_sn)
            if failed_coins:
                per_snap_changes.append((si, failed_coins, picks_made, replaced))
                fail_detail = ','.join(f'{c}(ms={_fail_cache[c][1]*100:+.1f},ml={_fail_cache[c][2]*100:+.1f})' for c in failed_coins)
                log.info('refill v2 [%s] snap[%d]: failed=%s replaced=%.4f → picks=%s', mname, si, fail_detail, replaced, picks_made or 'CASH')

        if not per_snap_changes:
            log.info('refill v2 [%s]: 모든 snapshot 변경 없음 (fail 없음)', mname)

        # state 에 반영
        mstate['snapshots'] = new_snapshots
        ms_all[mname] = mstate

        # combined 누적 (모든 멤버 동등 가중)
        for snap in new_snapshots:
            for t, w in snap.items():
                key = 'CASH' if t.upper() == 'CASH' else t
                new_combined[key] = new_combined.get(key, 0.0) + (w / len(new_snapshots)) / n_members

    # 정규화
    total = sum(new_combined.values())
    if total > 0:
        new_combined = {k: v / total for k, v in new_combined.items()}
    else:
        new_combined = {'CASH': 1.0}
    return new_combined


def evaluate_drift_fire(cur_w: Dict[str, float], tgt_w: Dict[str, float],
                         threshold: float = DRIFT_THRESHOLD_SPOT) -> Tuple[bool, float]:
    """drift trigger 평가. (fire, ht) 반환.

    fire = ht >= threshold. canary_on/crash_cooldown 같은 외부 조건은 호출자가 결합.
    """
    ht = half_turnover(cur_w, tgt_w)
    return (ht >= threshold, ht)


# ─── Top-level 진입점 ───
@dataclass
class EngineResult:
    combined_target: Dict[str, float]              # 'Cash' 키 정규화됨
    member_targets: Dict[str, Dict[str, float]]    # 'Cash' 키 정규화됨
    fresh: Dict[str, bool]                          # 멤버별 freshness
    new_bar: Dict[str, bool]
    canary_flipped: Dict[str, bool]
    all_fresh: bool                                 # 모든 멤버 fresh?
    any_new_bar: bool
    universe: List[str]
    universe_meta: Dict
    alerts: List[str]
    # V24 신규: drift 트리거 평가 결과 (cur_w 주어진 경우만)
    drift_fire: bool = False
    pfd_fire: bool = False   # 전환 후 재평가 발화 (target 불변이어도 리밸 필요)
    drift_half_turnover: float = 0.0
    drift_threshold: float = DRIFT_THRESHOLD_SPOT
    canary_info: Dict[str, Dict] = field(default_factory=dict)  # 멤버별 카나리 상세
    # 2026-09-30: 시총 목록 만료 시 — 카나리 OFF 가 독립 확인된 경우만 True (매도 전용)
    liquidation_only: bool = False
    market_meta: Optional[Dict] = None  # {'ok': bool, 'reason': str, ...}


def _independent_canary(session: requests.Session, state: Dict,
                        now_utc: datetime) -> Tuple[Optional[bool], Dict]:
    """시총 목록 없이 BTC 완료봉만으로 카나리 ON/OFF 를 판정한다.

    유니버스가 없어도 "지금 팔아야 하는 국면인가" 는 BTC 하나로 답이 난다. 이 판정이
    유니버스 생성보다 앞서야 2026-08-15 사고(매매가 꼭 필요한 유일한 경우를 가드가 막음)가
    안 되풀이된다. 상태는 건드리지 않는다 — 여기서 스냅샷·카나리 상태를 갱신하면 막힌
    실행이 전략을 조용히 전진시킨다.

    Returns: (canary_on 또는 None(판정 불가), 상세)
    """
    cfg = MEMBER_D_SMA42
    interval = cfg['interval']
    sma_p = cfg['sma_bars']
    hyst = cfg['canary_hyst']
    info: Dict = {'sma_bars': sma_p, 'hyst': hyst}
    try:
        df = fetch_binance_klines(session, 'BTCUSDT', interval, KLINE_LIMITS[interval])
    except Exception as e:
        info['reason'] = f'BTC 봉 조회 예외({e})'
        return None, info
    if df is None or df.empty:
        info['reason'] = 'BTC 봉 조회 실패'
        return None, info
    bars = slice_to_last_closed({'BTC': df}, interval, now_utc)
    btc = bars.get('BTC')
    if btc is None or len(btc) < sma_p + 2:
        info['reason'] = 'BTC 봉 부족'
        return None, info

    latest_ts = btc.index[-1].to_pydatetime()
    if latest_ts.tzinfo is None:
        latest_ts = latest_ts.replace(tzinfo=timezone.utc)
    expected_ts = expected_last_closed_bar_ts(interval, now_utc)
    info['bar_ts'] = to_utc_iso(latest_ts)
    if latest_ts != expected_ts:
        # 완료봉이 기대 봉과 다르면 믿을 수 있는 봉이 아니다 — 판정하지 않는다.
        info['reason'] = f'완료봉 불일치(기대 {to_utc_iso(expected_ts)})'
        return None, info

    close = np.asarray(btc['Close'].values, dtype=float)
    sma_val = _calc_sma(close, sma_p)
    if not sma_val or sma_val <= 0 or not math.isfinite(sma_val):
        info['reason'] = 'SMA 산출 실패'
        return None, info
    cur = float(close[-1])
    prev_canary = bool(((state.get('members', {}) or {}).get('D_SMA42', {}) or {})
                       .get('canary_on', False))
    if prev_canary:
        canary_on = not (cur < sma_val * (1 - hyst))
    else:
        canary_on = cur > sma_val * (1 + hyst)
    info.update({'cur': cur, 'sma_val': sma_val, 'ratio': cur / sma_val,
                 'prev': prev_canary, 'on': canary_on})
    return canary_on, info


def _market_meta_blocked(session: requests.Session, state: Dict, now_utc: datetime,
                          reason: str) -> EngineResult:
    """시총 목록을 못 얻었을 때의 결과. 카나리 OFF 가 확인될 때만 매도를 연다."""
    canary_on, cinfo = _independent_canary(session, state, now_utc)
    meta = {'ok': False, 'reason': reason, 'canary': cinfo}

    if canary_on is False:
        meta['action'] = 'liquidation_only'
        alert = (f'🔴 현물 시총 목록 사용 불가 — {reason}. '
                 f'BTC 완료봉으로 카나리 OFF 확인(BTC ${cinfo.get("cur", 0):,.0f} / '
                 f'SMA ${cinfo.get("sma_val", 0):,.0f}) → 청산만 수행한다.')
        log.error(alert)
        return EngineResult(combined_target={'Cash': 1.0}, member_targets={},
                            fresh={'MARKET_META': False}, new_bar={}, canary_flipped={},
                            all_fresh=True, any_new_bar=True,
                            universe=[], universe_meta={'market_meta_blocked': True},
                            alerts=[alert], liquidation_only=True, market_meta=meta)

    if canary_on is True:
        meta['action'] = 'block'
        alert = (f'🛑 현물 시총 목록 사용 불가 — {reason}. '
                 f'카나리 ON 이라 매매를 차단한다(전략 상태 무변경).')
    else:
        meta['action'] = 'block'
        alert = (f'🛑 현물 시총 목록 사용 불가 — {reason}. '
                 f'카나리도 판정 불가({cinfo.get("reason", "사유 불명")}) → 매매 차단.')
    log.error(alert)
    return EngineResult(combined_target={}, member_targets={},
                        fresh={'MARKET_META': False}, new_bar={}, canary_flipped={},
                        all_fresh=False, any_new_bar=False,
                        universe=[], universe_meta={'market_meta_blocked': True},
                        alerts=[alert], liquidation_only=False, market_meta=meta)


def compute_live_targets(state: Dict, session: requests.Session, cache_dir: str,
                          now_utc: Optional[datetime] = None,
                          upbit_price_fn=None,
                          upbit_status: Optional[Dict[str, Dict]] = None,
                          cur_w: Optional[Dict[str, float]] = None) -> EngineResult:
    """한 사이클의 전체 계산: 유니버스 → 데이터 → 멤버 target → 앙상블.

    state는 dict in/out (mutate됨). 호출자가 save_state로 저장.

    upbit_price_fn: 유니버스 히스토리/거래대금 필터용. (ticker) -> DataFrame 또는 None.
    upbit_status: executor에서 이미 조회한 market status. None이면 엔진이 재조회.
    """
    now_utc = now_utc or utc_now()
    alerts: List[str] = []

    # 1) 유니버스 (새 봉에서만 갱신, 그 외는 캐시 사용)
    cached_universe = state.get('universe_cache', {})
    cached_universe_ts = parse_utc_iso(cached_universe.get('ts', ''))
    # 유니버스 캐시 재사용도 원본 시총 조회 시각으로 나이를 잰다 (2026-09-30).
    # 유니버스 캐시의 ts 는 "이 목록을 만든 시각" 이라 재사용할 때마다 현재 시각으로
    # 갱신되면 원본이 아무리 낡아도 20시간마다 새것처럼 보인다.
    cached_cg_ts = parse_utc_iso((cached_universe.get('meta', {}) or {}).get('cg_ts', '') or '')
    cached_cg_age_h = ((now_utc - cached_cg_ts).total_seconds() / 3600.0
                       if cached_cg_ts else None)
    cg_stale = (cached_cg_age_h is None
                or cached_cg_age_h < 0
                or cached_cg_age_h > MARKET_META_MAX_AGE_H)
    need_universe_refresh = (
        not cached_universe.get('universe')
        or not cached_universe_ts
        or (now_utc - cached_universe_ts).total_seconds() > 20 * 3600
        or cg_stale
    )
    if need_universe_refresh:
        try:
            universe, u_meta = build_dynamic_universe(session, cache_dir,
                                                        upbit_price_fn=upbit_price_fn)
        except MarketMetaUnavailable as e:
            return _market_meta_blocked(session, state, now_utc, str(e))
        state['universe_cache'] = {
            'ts': to_utc_iso(now_utc),
            'universe': universe,
            'meta': {k: v for k, v in u_meta.items() if k not in ('binance_status', 'upbit_status')},
        }
    else:
        universe = cached_universe.get('universe', [])
        u_meta = {'upbit_status': {}, 'binance_status': {},
                  'filtered_count': len(universe),
                  'cg_ts': to_utc_iso(cached_cg_ts) if cached_cg_ts else None,
                  'cg_age_h': cached_cg_age_h,
                  'cg_source': 'universe_cache',
                  'cached': True}

    if not universe:
        log.error('유니버스 비어있음 → 강제 CASH 100%%')
        return EngineResult(combined_target={'Cash': 1.0}, member_targets={},
                            fresh={}, new_bar={}, canary_flipped={},
                            all_fresh=False, any_new_bar=False,
                            universe=[], universe_meta=u_meta, alerts=['유니버스 비어있음'])

    # 2) Upbit 상태 (executor에서 전달받거나 엔진에서 재조회)
    live_upbit_status = upbit_status if upbit_status is not None else fetch_upbit_market_status(session)
    if live_upbit_status:
        state['last_upbit_status'] = {
            'ts': to_utc_iso(now_utc),
            'status': {c: s for c, s in live_upbit_status.items() if c in universe or c in state.get('holdings_cache', {})},
        }

    # 투자유의(warning) / 상장폐지(!listed) 코인만 타겟에서 제외.
    # 투자주의(caution)는 단기 급변동 알림일 뿐 상장 리스크가 아니므로 허용.
    warning_coins = set()
    for coin, info in (live_upbit_status or {}).items():
        if bool(info.get('warning', False)) or not bool(info.get('listed', True)):
            warning_coins.add(coin)
    if warning_coins:
        log.warning('Upbit 유의/상폐 코인 %d: %s → 타겟 제외', len(warning_coins),
                    ','.join(sorted(warning_coins))[:200])
    # 2026-04-21: 유의/상폐 변경 알림 제거 (사용자 요청). 로그에만 기록.
    state['last_warning_coins_alerted'] = sorted(warning_coins)
    effective_universe = [c for c in universe if c not in warning_coins]
    if not effective_universe:
        log.error('경고 제외 후 유니버스 비어있음 → CASH 100%')
        return EngineResult(combined_target={'Cash': 1.0}, member_targets={},
                            fresh={}, new_bar={}, canary_flipped={},
                            all_fresh=False, any_new_bar=False,
                            universe=[], universe_meta=u_meta,
                            alerts=alerts + ['경고 제외 후 유니버스 비어있음'])

    # 경고 코인은 멤버별 snapshots/last_combined에서도 즉시 제거 — 새 봉 없이
    # 캐시된 target을 반환할 때 유의종목이 되살아나는 것을 차단.
    if warning_coins:
        ms_purge = state.get('members', {})
        for mname_p, mstate_p in ms_purge.items():
            if not isinstance(mstate_p, dict):
                continue
            snaps = mstate_p.get('snapshots', []) or []
            new_snaps = []
            for snap in snaps:
                if not isinstance(snap, dict):
                    new_snaps.append(snap)
                    continue
                removed_w = sum(v for k, v in snap.items() if k in warning_coins)
                cleaned = {k: v for k, v in snap.items() if k not in warning_coins}
                if removed_w > 0:
                    cleaned['CASH'] = cleaned.get('CASH', 0.0) + removed_w
                new_snaps.append(cleaned)
            mstate_p['snapshots'] = new_snaps
            lc = mstate_p.get('last_combined', {}) or {}
            if isinstance(lc, dict):
                removed_lc = sum(v for k, v in lc.items() if k in warning_coins)
                lc2 = {k: v for k, v in lc.items() if k not in warning_coins}
                if removed_lc > 0:
                    lc2['CASH'] = lc2.get('CASH', 0.0) + removed_lc
                mstate_p['last_combined'] = lc2

    # 3) 각 멤버 bars fetch + slice
    member_targets: Dict[str, Dict[str, float]] = {}
    fresh_map: Dict[str, bool] = {}
    new_bar_map: Dict[str, bool] = {}
    flipped_map: Dict[str, bool] = {}
    canary_info_map: Dict[str, Dict] = {}
    pfd_fired_any = False  # 전환 후 재평가가 이번 봉에 발화했나 (드리프트·리필 중복 방지)

    ms_all = state.setdefault('members', {})
    # 가드 제거 (2026-04-21): 기존 state의 excluded_coins는 이후 save 때 자연 소거.
    state.pop('excluded_coins', None)

    # V24 refill v2 용 — 각 멤버 bars 캐시 (drift fire 시점에 사용)
    member_bars_cache: Dict[str, Dict[str, pd.DataFrame]] = {}

    for mname, cfg in MEMBERS.items():
        interval = cfg['interval']
        # 모든 (warning 제외된) universe 코인 + BTC 데이터 fetch
        symbols = ['BTC'] + [c for c in effective_universe if c != 'BTC']
        bars: Dict[str, pd.DataFrame] = {}
        failed: List[str] = []
        for coin in symbols:
            df = fetch_binance_klines(session, f'{coin}USDT', interval, KLINE_LIMITS[interval])
            if df.empty:
                failed.append(coin)
            else:
                bars[coin] = df
        if failed:
            log.warning('%s interval=%s: Binance kline fetch 실패 %d종목: %s',
                        mname, interval, len(failed), ','.join(failed[:5]))
            alerts.append(f'{mname} Binance fetch 실패: {len(failed)}종목 ({",".join(failed[:5])})')

        # slice to last closed
        bars = slice_to_last_closed(bars, interval, now_utc)

        if 'BTC' not in bars:
            log.error('%s: BTC 데이터 없음 → CASH', mname)
            member_targets[mname] = {'CASH': 1.0}
            fresh_map[mname] = False
            new_bar_map[mname] = False
            flipped_map[mname] = False
            continue

        mem_state = MemberState.from_dict(ms_all.get(mname, {}))
        res = compute_member_target(mname, cfg, bars, effective_universe, mem_state, now_utc)

        # V24 refill v2 — bars 캐시
        member_bars_cache[mname] = bars

        member_targets[mname] = res.target
        fresh_map[mname] = res.fresh
        new_bar_map[mname] = res.new_bar
        flipped_map[mname] = res.canary_flipped
        if res.pfd_fired:
            pfd_fired_any = True
        if res.canary_info:
            canary_info_map[mname] = res.canary_info
        ms_all[mname] = res.new_state.to_dict()
        alerts.extend(res.alerts)

    # 4) 앙상블
    combined = combine_ensemble(member_targets, ENSEMBLE_WEIGHTS)
    combined = normalize_cash_key(combined)
    member_targets_norm = {n: normalize_cash_key(t) for n, t in member_targets.items()}

    # 5) state에 스냅샷 저장
    state['last_member_targets'] = {
        n: {**t, '_ts': to_utc_iso(now_utc)} for n, t in member_targets_norm.items()
    }
    state['last_target_snapshot'] = {**combined, '_ts': to_utc_iso(now_utc)}
    state['last_run_ts'] = to_utc_iso(now_utc)
    state['schema_version'] = SCHEMA_VERSION  # V24 마이그레이션 마크

    all_fresh = all(fresh_map.values()) if fresh_map else False
    any_new = any(new_bar_map.values()) if new_bar_map else False

    # 6) V24 drift 트리거 평가 (cur_w 주어진 경우만, 정보용; 실제 발화 결정은 호출자)
    drift_fire_b = False
    drift_ht = 0.0
    if cur_w is not None:
        drift_fire_b, drift_ht = evaluate_drift_fire(cur_w, combined, DRIFT_THRESHOLD_SPOT)
        log.info('drift eval: half_turnover=%.4f threshold=%.2f fire=%s enabled=%s',
                 drift_ht, DRIFT_THRESHOLD_SPOT, drift_fire_b, DRIFT_ENABLED)
        if not DRIFT_ENABLED:
            drift_fire_b = False  # snap-only fallback
        if drift_fire_b and pfd_fired_any:
            # 전환 후 재평가가 방금 전 스냅샷을 새로 고른 봉이다. 같은 봉에서 리필까지
            # 돌리면 갓 고른 자리를 다시 갈아끼운다 — 채택 BT 는 그 봉에 재평가 하나만 한다.
            log.info('전환 후 재평가 발화 봉 → 드리프트·리필 중복 적용 안 함 (ht=%.4f)', drift_ht)
            drift_fire_b = False

    # 6b) V24 refill v2 (drift fire 시점에 mom2 음수 슬롯 교체)
    if drift_fire_b and REFILL_ENABLED:
        try:
            combined_before = dict(combined)
            combined = _apply_refill_v2_to_state(state, MEMBERS, member_bars_cache, combined)
            # before/after diff (가장 큰 변화 top 5)
            all_keys = set(combined_before) | set(combined)
            diffs = sorted(
                [(k, combined.get(k, 0) - combined_before.get(k, 0)) for k in all_keys],
                key=lambda x: -abs(x[1])
            )[:5]
            log.info('refill v2 applied: combined %d→%d keys, top diffs=%s',
                     len(combined_before), len(combined),
                     [(k, f'{d*100:+.1f}%') for k, d in diffs if abs(d) > 1e-6])
            state['last_target_snapshot'] = {**combined, '_ts': to_utc_iso(now_utc)}
        except Exception as e:
            log.warning('refill v2 skipped due to error: %s', e, exc_info=True)

    # 6c) 반환 경계 정규화 — refill v2 재계산(combine_ensemble)은 내부 규약대로 대문자 CASH 를
    #     되돌려주므로, 모든 후처리가 끝난 여기서 한 번 더 'Cash' 로 통일한다. 이 값이
    #     EngineResult.combined_target 과 state 스냅샷의 단일 출처다.
    #     (2026-08-20: 정규화가 refill 앞에만 있어 drift 발화일에 executor 가 CASH 를
    #      코인 티커로 오인해 헛주문을 냈다.)
    combined = normalize_cash_key(combined)
    state['last_target_snapshot'] = {**combined, '_ts': to_utc_iso(now_utc)}

    return EngineResult(
        combined_target=combined,
        member_targets=member_targets_norm,
        fresh=fresh_map,
        new_bar=new_bar_map,
        canary_flipped=flipped_map,
        all_fresh=all_fresh,
        any_new_bar=any_new,
        universe=universe,
        universe_meta=u_meta,
        alerts=alerts,
        drift_fire=drift_fire_b,
        pfd_fire=pfd_fired_any,
        drift_half_turnover=drift_ht,
        drift_threshold=DRIFT_THRESHOLD_SPOT,
        canary_info=canary_info_map,
    )
