#!/usr/bin/env python3
"""Cap Defend 현물 라이브 엔진.

V21: D봉 3멤버 (D_SMA50 + D_SMA150 + D_SMA100) 1/3씩 EW 앙상블.
바이낸스 spot kline에서 신호 계산 → 업비트 KRW 체결 (executor_coin.py에서 수행).

멤버 D_SMA50: 일봉, SMA50, Mom30/90, snap 30봉, Top3
멤버 D_SMA150: 일봉, SMA150, Mom20/60, snap 90봉, Top3
멤버 D_SMA100: 일봉, SMA100, Mom20/120, snap 90봉, Top3

가드 없음 (2026-04-21 확정): gap/exclusion 가드 전면 제거.
앙상블 분산 + Upbit warning/delisting 필터만으로 방어.

설계:
- compute_strategy_target(): auto_trade_binance 패턴 이식 (bar-idempotency + 3-snap stagger 내장)
- 엔진 내부 현금 키는 'CASH' (대문자). executor로 넘어갈 때 'Cash' (소문자) 정규화.
- 유니버스: CoinGecko Top40 ∩ Binance spot TRADING ∩ Upbit KRW normal ∩ 253일 이상 ∩ 30일 평균 10억원 ≥ → Top 40
- Freshness: expected_last_bar_ts 일치 체크
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

HARDCODED_FALLBACK = ['BTC', 'ETH', 'SOL', 'XRP', 'DOGE', 'ADA', 'AVAX', 'LINK', 'SHIB', 'TRX']

UPBIT_MARKET_ALL_URL = 'https://api.upbit.com/v1/market/all?is_details=true'

# V21 멤버 설정 (ENS_spot_k3_4b270476, 2026-04-17 확정)
# 3멤버 D봉 EW 1/3씩. 4h봉 제거.
MEMBER_D_SMA50 = {
    'interval': 'D',
    'sma_bars': 50,
    'mom_short_bars': 20,
    'mom_long_bars': 90,
    'snap_interval_bars': 90,
    'n_snapshots': 3,
    'canary_hyst': 0.015,
    'health_mode': 'mom2vol',
    'vol_mode': 'daily',
    'vol_threshold': 0.05,
    'vol_lookback_days': 90,
    'universe_size': 3,
    'cap': 1.0 / 3.0,
}

MEMBER_D_SMA150 = {
    'interval': 'D',
    'sma_bars': 150,
    'mom_short_bars': 20,
    'mom_long_bars': 60,
    'snap_interval_bars': 90,
    'n_snapshots': 3,
    'canary_hyst': 0.015,
    'health_mode': 'mom2vol',
    'vol_mode': 'daily',
    'vol_threshold': 0.05,
    'vol_lookback_days': 90,
    'universe_size': 3,
    'cap': 1.0 / 3.0,
}

MEMBER_D_SMA100 = {
    'interval': 'D',
    'sma_bars': 100,
    'mom_short_bars': 20,
    'mom_long_bars': 120,
    'snap_interval_bars': 90,
    'n_snapshots': 3,
    'canary_hyst': 0.015,
    'health_mode': 'mom2vol',
    'vol_mode': 'daily',
    'vol_threshold': 0.05,
    'vol_lookback_days': 90,
    'universe_size': 3,
    'cap': 1.0 / 3.0,
}

MEMBERS = {
    'D_SMA50': MEMBER_D_SMA50,
    'D_SMA150': MEMBER_D_SMA150,
    'D_SMA100': MEMBER_D_SMA100,
}

ENSEMBLE_WEIGHTS = {
    'D_SMA50': 1.0 / 3.0,
    'D_SMA150': 1.0 / 3.0,
    'D_SMA100': 1.0 / 3.0,
}

# 각 interval별 최소 가져올 봉 수
KLINE_LIMITS = {
    'D': 500,    # SMA150 + Mom120 + vol90d → 500 충분
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
    raise ValueError(f'unsupported interval: {interval}')


def normalize_cash_key(target: Dict[str, float]) -> Dict[str, float]:
    """엔진 내부 'CASH' → executor 'Cash'로 통일."""
    out = {}
    for k, v in target.items():
        key = 'Cash' if k.upper() == 'CASH' else k
        out[key] = out.get(key, 0.0) + v
    return out


# ─── 데이터 수집: CoinGecko / Binance / Upbit ───
def fetch_coingecko_top(session: requests.Session, fetch_limit: int = COINGECKO_FETCH_LIMIT,
                         retries: int = 5, cache_path: Optional[str] = None) -> List[Dict]:
    """CoinGecko Top N 가져오기 (N=fetch_limit). 실패 시 cache fallback."""
    headers = {'accept': 'application/json', 'User-Agent': 'Mozilla/5.0 CoinExecutor/1.0'}
    for attempt in range(1, retries + 1):
        try:
            params = {'vs_currency': 'usd', 'order': 'market_cap_desc',
                      'per_page': fetch_limit, 'page': 1}
            resp = session.get(COINGECKO_URL, params=params, headers=headers, timeout=20)
            if resp.status_code == 200:
                data = resp.json()
                if cache_path:
                    try:
                        tmp = cache_path + '.tmp'
                        with open(tmp, 'w') as f:
                            json.dump({'ts': to_utc_iso(utc_now()), 'data': data}, f)
                        os.replace(tmp, cache_path)
                    except Exception as e:
                        log.warning('CoinGecko 캐시 저장 실패: %s', e)
                return data
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

    # Cache fallback
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path) as f:
                cached = json.load(f)
            log.warning('CoinGecko 캐시 fallback 사용 (ts=%s)', cached.get('ts'))
            return cached.get('data', [])
        except Exception as e:
            log.warning('CoinGecko 캐시 로드 실패: %s', e)
    return []


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
            'fallback_used': bool,
            'upbit_status': {coin: {warning, caution, listed}},
            'binance_status': {coin: 'TRADING' or other},
        }
    """
    cg_cache = os.path.join(cache_dir, 'universe_cg_cache.json')
    bn_cache = os.path.join(cache_dir, 'binance_exchinfo_cache.json')
    os.makedirs(cache_dir, exist_ok=True)

    meta = {'fallback_used': False, 'binance_status': {}, 'upbit_status': {}}

    cg_data = fetch_coingecko_top(session, cache_path=cg_cache)
    binance_status = fetch_binance_exchange_info(session, cache_path=bn_cache)
    upbit_status = fetch_upbit_market_status(session)

    meta['coingecko_count'] = len(cg_data)
    meta['binance_trading_count'] = sum(1 for s in binance_status.values() if s == 'TRADING')
    meta['upbit_listed_count'] = len(upbit_status)
    meta['binance_status'] = binance_status
    meta['upbit_status'] = upbit_status

    if not cg_data or not binance_status or not upbit_status:
        log.warning('외부 API 일부 실패 → 하드코딩 fallback 사용')
        meta['fallback_used'] = True
        # fallback도 최소한의 필터만 적용
        filtered = [c for c in HARDCODED_FALLBACK
                    if c not in STABLECOINS
                    and (not binance_status or binance_status.get(c) == 'TRADING')
                    and (not upbit_status or (upbit_status.get(c, {}).get('listed', False)
                                               and not upbit_status.get(c, {}).get('warning', False)))]
        meta['filtered_count'] = len(filtered)
        return filtered[:UNIVERSE_TOP_N], meta

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


@dataclass
class MemberState:
    canary_on: bool = False
    last_bar_ts: Optional[str] = None
    snapshots: List[Dict[str, float]] = field(default_factory=list)
    bar_counter: int = 0
    last_combined: Dict[str, float] = field(default_factory=lambda: {'CASH': 1.0})
    snap_id: int = 0

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
        return ms

    def to_dict(self) -> Dict:
        return {
            'canary_on': self.canary_on,
            'last_bar_ts': self.last_bar_ts,
            'snapshots': self.snapshots,
            'bar_counter': self.bar_counter,
            'last_combined': self.last_combined,
            'snap_id': self.snap_id,
        }


@dataclass
class MemberComputeResult:
    target: Dict[str, float]           # 'CASH' 포함
    new_state: MemberState
    fresh: bool                        # expected_last_bar_ts 일치 여부
    new_bar: bool                      # 새 봉이어서 재계산 수행했는지
    canary_flipped: bool
    alerts: List[str]                  # 텔레그램 알림용 메시지


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
    # 카나리 (BTC 종가 vs SMA)
    btc_close = btc_df['Close'].values
    sma_val = _calc_sma(btc_close, sma_p)
    prev_canary = state.canary_on
    cur = float(btc_close[-1])
    if prev_canary:
        canary_on = not (cur < sma_val * (1 - hyst))
    else:
        canary_on = cur > sma_val * (1 + hyst)
    canary_flipped = canary_on != prev_canary
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

    # 스냅샷 초기화 (첫 실행 또는 n_snap 변경 시)
    freshly_initialized = (not state.snapshots or len(state.snapshots) != n_snap)
    if freshly_initialized:
        state.snapshots = [{'CASH': 1.0} for _ in range(n_snap)]

    # 스냅샷 갱신
    if canary_flipped or (freshly_initialized and canary_on):
        # 전 스냅샷 즉시 전환 (플립 또는 최초 초기화 시 전체 populate)
        for si in range(n_snap):
            state.snapshots[si] = compute_weights(si)
        state.snap_id += 1
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
                               canary_flipped=canary_flipped, alerts=alerts)


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


def compute_live_targets(state: Dict, session: requests.Session, cache_dir: str,
                          now_utc: Optional[datetime] = None,
                          upbit_price_fn=None,
                          upbit_status: Optional[Dict[str, Dict]] = None) -> EngineResult:
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
    # 유니버스 캐시는 D봉 기준으로 하루 1회 갱신 (intrabar drift 방지)
    need_universe_refresh = (
        not cached_universe.get('universe')
        or not cached_universe_ts
        or (now_utc - cached_universe_ts).total_seconds() > 20 * 3600
    )
    if need_universe_refresh:
        universe, u_meta = build_dynamic_universe(session, cache_dir,
                                                    upbit_price_fn=upbit_price_fn)
        state['universe_cache'] = {
            'ts': to_utc_iso(now_utc),
            'universe': universe,
            'meta': {k: v for k, v in u_meta.items() if k not in ('binance_status', 'upbit_status')},
        }
    else:
        universe = cached_universe.get('universe', [])
        u_meta = {'fallback_used': False,
                  'upbit_status': {}, 'binance_status': {},
                  'filtered_count': len(universe),
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

    ms_all = state.setdefault('members', {})
    # 가드 제거 (2026-04-21): 기존 state의 excluded_coins는 이후 save 때 자연 소거.
    state.pop('excluded_coins', None)

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

        member_targets[mname] = res.target
        fresh_map[mname] = res.fresh
        new_bar_map[mname] = res.new_bar
        flipped_map[mname] = res.canary_flipped
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

    all_fresh = all(fresh_map.values()) if fresh_map else False
    any_new = any(new_bar_map.values()) if new_bar_map else False

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
    )
