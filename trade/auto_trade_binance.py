#!/usr/bin/env python3
"""
바이낸스 선물 자동매매 — V24 L3 단일 멤버 (2026-04-30 확정)
========================================
V24 신호:
- D_SMA42:    (interval=D, SMA=42, Mom=18/127, mom2vol, daily vol 5%, Snap=95 bars, n_snap=5, drift=0.03)

V24 변경 (vs V22):
- 1D + 4h 2멤버 → 1D 단일 멤버 (4h 제거, snap=90→57)
- drift_threshold = 0.03 (V24 갱신 05-04, 0.05 → 0.03 반응성 ↑)
- ENSEMBLE_WEIGHTS = {'D_SMA42': 1.0}
- cron 4h x 6 → 1d x 1 (09:05)

실행층:
- 고정 3배 레버리지 (L3)
- 스탑: 없음 (STOP_PCT=0, 가드 비활성)
- 캐시 게이트: 없음 (STOP_GATE_CASH_THRESHOLD=0)
- 앙상블 분산만으로 방어

실행: 1일 1회 (cron "5 9 * * *" 한국시간)
1. 바이낸스에서 D OHLCV 수집
2. 단일 전략 목표 비중 계산
3. 고정 3x 레버리지로 매핑
4. cur_w (자본금 기준 비중) vs target_w 의 half_turnover 계산
5. snap_fire OR (canary_on AND ht >= 0.05) 시 리밸 발화
6. 현재 포지션과 비교 → delta 리밸런싱
7. STOP 주문 없음 (sync_stop_orders는 early return)

Schema version: V24 (state['schema_version'])
"""

import argparse
import json
import math
import os
import random
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import requests
from binance.client import Client
from binance.exceptions import BinanceAPIException

from common.io import save_json as save_json_atomic
from common.notify import send_telegram as _send_tg_common

# ─── 설정 ───
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
CONFIG_PATH = os.path.join(SCRIPT_DIR, 'config.py')
STATE_PATH = os.path.join(SCRIPT_DIR, 'binance_state.json')
ALLOC_TRANSIT_STATE_PATH = os.path.join(SCRIPT_DIR, 'trade_state.json')  # 코인 state (alloc_transit flag)
LOG_PATH = os.path.join(SCRIPT_DIR, 'binance_trade.log')


CAP_RATIO_FLOOR = 0.10  # cap_ratio < floor → 거래 중단 fallback
ALLOC_TRANSIT_STALE_HOURS = 26
CAP_DEFEND_MIN_EXCESS = 0.01  # cap_ratio < 0.99 면 cap_defend 매도 발동


def _validate_cap_ratio(val, sleeve_name: str):
    """cap_ratio 검증. invalid → 1.0 fallback + ERROR 로그."""
    try:
        cr = float(val)
    except Exception:
        log.error(f'alloc_transit cap_ratio[{sleeve_name}] parse 실패 ({val!r}) → fallback 1.0')
        return 1.0
    if not math.isfinite(cr) or cr <= 0:
        log.error(f'alloc_transit cap_ratio[{sleeve_name}]={cr} invalid → fallback 1.0')
        return 1.0
    if cr < CAP_RATIO_FLOOR:
        log.error(f'alloc_transit cap_ratio[{sleeve_name}]={cr:.4f} < floor {CAP_RATIO_FLOOR} → SKIP (fallback 1.0)')
        return 1.0
    if cr > 1.0:
        return 1.0
    return cr


def _read_alloc_transit_cap_ratio_fut():
    """trade_state.json 의 alloc_transit active 면 fut cap_ratio (≤1.0) 반환. 아니면 None.

    schema/parse 실패 → fallback 1.0 + ERROR 로그.
    """
    for _p in (
        os.path.expanduser('~/trade_state.json'),
        ALLOC_TRANSIT_STATE_PATH,
    ):
        try:
            if not os.path.exists(_p):
                continue
            with open(_p, 'r') as f:
                obj = json.load(f)
            at = obj.get('alloc_transit')
            if not at or not at.get('active'):
                return None
            cr_raw = (at.get('cap_ratio') or {}).get('fut')
            if cr_raw is None:
                log.error('alloc_transit active 하나 cap_ratio[fut] missing → fallback 1.0')
                return 1.0
            cr = _validate_cap_ratio(cr_raw, 'fut')
            mtime_age = (time.time() - os.path.getmtime(_p)) / 3600 if os.path.exists(_p) else -1
            if mtime_age > ALLOC_TRANSIT_STALE_HOURS:
                log.error(f'alloc_transit state stale (age {mtime_age:.1f}h) → cap 무시')
                return None
            log.info(f'alloc_transit cap_ratio[fut]={cr:.4f} (mtime age {mtime_age:.1f}h)')
            return cr
        except json.JSONDecodeError as ex:
            log.error(f'alloc_transit JSON parse 실패 ({_p}): {ex} → fallback')
            return None
        except Exception:
            continue
    return None

# V25 전략/실행 파라미터 (2026-05-28 도입, K2 + 동적 L + CROSS)
# 동적 per-coin L: min(BTC_cap, per_coin_K2). Lmin=2, Lmid=3, Lmax=4
# 마진모드 CROSS (ISOLATED → CROSS)
# 가드 없음 (분산 + per-coin L 자체 방어)
LEVERAGE_FLOOR = 2        # V25: Lmin (V24 L3 → V25 L2)
LEVERAGE_MID = 3          # V25: Lmid (V24 와 동일)
LEVERAGE_CEILING = 4      # V25: Lmax (V24 L3 → V25 L4)
STOP_PCT = 0.0            # 가드 비활성 (V24 동일)
STOP_GATE_CASH_THRESHOLD = 0.0
LEVERAGE_MOM_LOOKBACK_BARS = 24 * 30  # legacy, V25 K2 SMA 가 대체

# V25 K2 (per-coin SMA-based) 파라미터
K2_SMA_PERIOD = 7         # 짧은 SMA — 빠른 trend 반영
K2_HYST = 0.025           # h=2.5% → thr_mid=1.025, thr_max=1.075

# V25 BTC cap 파라미터 (BTC SMA42 ratio 기반)
BTC_CAP_SMA_PERIOD = 42
BTC_CAP_THR_MID = 1.015
BTC_CAP_THR_MAX = 1.05

# V25 마진모드
MARGIN_TYPE = 'CROSSED'   # Binance API 표기 ('CROSSED' or 'ISOLATED')

# 디버그 로그 (V25 도입 — 동적 L 검증용)
DEBUG_LEVERAGE = True     # 매 cron L 결정 detail
DEBUG_MARGIN = True       # set_margin_type / set_leverage 결과

SCHEMA_VERSION = 'V25'
REFILL_ENABLED_FUT = True  # V24 refill v2 ON (drift fire 시 mom2 음수 슬롯 교체)
DRIFT_THRESHOLD_FUT = 0.03  # V24 갱신 (05-04): 0.05 → 0.03
DRIFT_ENABLED_FUT = True  # False 로 토글 시 drift_fire 강제 False (snap-only fallback)

ENSEMBLE_WEIGHTS = {'D_SMA42': 1.0}  # V24: 단일 멤버

STRATEGIES = {
    'D_SMA42': {
        'interval': 'D',
        'sma_bars': 42,
        'mom_short_bars': 18,
        'mom_long_bars': 127,
        'health_mode': 'mom2vol',
        'vol_mode': 'daily',
        'vol_threshold': 0.05,
        'snap_interval_bars': 95,    # V24 갱신 (05-04): 57 → 95 (5*19, stagger 19 유지)
        'canary_hyst': 0.015,
        'n_snapshots': 5,            # V24 갱신 (05-04): 3 → 5
    },
}

# 유니버스 (시총순). 매 매매 사이클마다 CoinGecko top40 + 바이낸스 USDT-M 선물 listing intersect로 동적 갱신.
# API 실패 시 fallback.
HARDCODED_UNIVERSE_FALLBACK = [
    'BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'SOLUSDT', 'XRPUSDT',
    'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'TRXUSDT', 'LINKUSDT',
    'DOTUSDT', 'UNIUSDT', 'NEARUSDT', 'LTCUSDT', 'BCHUSDT',
    'APTUSDT', 'ICPUSDT', 'FILUSDT', 'ATOMUSDT', 'ARBUSDT',
]
UNIVERSE: List[str] = list(HARDCODED_UNIVERSE_FALLBACK)
UNIVERSE_TARGET_SIZE = 40  # CoinGecko top N
COINGECKO_URL = 'https://api.coingecko.com/api/v3/coins/markets'
STABLECOINS = {'USDT', 'USDC', 'BUSD', 'DAI', 'TUSD', 'FDUSD', 'USDD', 'PYUSD', 'USDe'}

UNIVERSE_SIZE = 3  # 헬스 통과한 상위 N개 (전략별)
CAP = 1/3  # EW + 33% cap
CASH_BUFFER_DEFAULT = 0.02  # 현금 버퍼 기본값 (state에서 동적 읽기)
CASH_BUFFER = CASH_BUFFER_DEFAULT  # 런타임에 state에서 갱신
MIN_NOTIONAL = 5.0  # 최소 주문 금액 (USDT)
# V25 cycle 9: 코인별 margin 변경 preflight 에서 "무관 심볼 먼지"로 간주하는 상한.
# MIN_NOTIONAL 과 의미 분리 — 이 값 미만 잔존은 봇이 정상 주문으로 청산 불가(reduceOnly 도 min notional 미달).
DUST_NOTIONAL_LIMIT = MIN_NOTIONAL
DELTA_THRESHOLD = 0.01  # 리밸런싱 허용 편차 ±1% (MIN_NOTIONAL 미달 시 스킵)
DISPLAY_DUST_NOTIONAL = 1.0  # 알림/대시보드에서 숨길 최소 포지션 금액
ORDER_MAX_RETRIES = 3
ORDER_RETRY_DELAYS = [1.0, 2.0, 5.0]
POSITION_FETCH_MAX_RETRIES = 3
POSITION_FETCH_RETRY_DELAYS = [1.0, 2.0]
CRON_START_JITTER_SECONDS = (3, 17)

# 텔레그램
TELEGRAM_BOT_TOKEN = None
TELEGRAM_CHAT_ID = None

log = logging.getLogger('binance_trader')
log.setLevel(logging.INFO)
_fmt = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
_fh = logging.FileHandler(LOG_PATH)
_fh.setFormatter(_fmt)
log.addHandler(_fh)
# 터미널에서 실행 시에만 콘솔 출력
if sys.stderr.isatty():
    _sh = logging.StreamHandler()
    _sh.setFormatter(_fmt)
    log.addHandler(_sh)


# ─── 유틸리티 ───
def load_config():
    """config.py에서 API 키 로드."""
    global TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
    try:
        sys.path.insert(0, SCRIPT_DIR)
        import config
        api_key = getattr(config, 'BINANCE_API_KEY', '')
        api_secret = getattr(config, 'BINANCE_API_SECRET', '')
        TELEGRAM_BOT_TOKEN = getattr(config, 'TELEGRAM_BOT_TOKEN', None)
        TELEGRAM_CHAT_ID = getattr(config, 'TELEGRAM_CHAT_ID', None)
        return api_key, api_secret
    except ImportError:
        log.error("config.py not found")
        return '', ''


_DRY_RUN_SILENT = [False]  # main 에서 --trade 아니면 True 토글


def send_telegram(msg: str):
    """텔레그램 알림. dry-run 모드면 로그만 남기고 silent."""
    if _DRY_RUN_SILENT[0]:
        log.info(f'[DRY] telegram silent: {msg[:200]}')
        return
    _send_tg_common(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg, prefix='선물', timeout=10)


def _as_bool(value) -> bool:
    """state 파일의 문자열 True/False도 안전하게 bool로 정규화."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {'true', '1', 'yes', 'on'}:
            return True
        if v in {'false', '0', 'no', 'off', ''}:
            return False
    return bool(value)


def _normalize_state(state: dict) -> dict:
    strategies = state.get('strategies', {})
    if isinstance(strategies, dict):
        for _, ss in strategies.items():
            if isinstance(ss, dict) and 'canary_on' in ss:
                ss['canary_on'] = _as_bool(ss.get('canary_on'))
    state['rebalancing_needed'] = _as_bool(state.get('rebalancing_needed', False))
    state['kill_switch'] = _as_bool(state.get('kill_switch', False))
    return state


def load_state() -> dict:
    """상태 파일 로드."""
    if os.path.exists(STATE_PATH):
        with open(STATE_PATH) as f:
            state = _normalize_state(json.load(f))
            state.setdefault('rebalancing_needed', False)
            return state
    return {
        'strategies': {},  # 각 전략별 상태 (canary, snapshots, bar_counter 등)
        'last_target': {},  # 마지막 합산 목표 비중
        'last_run': None,
        'rebalancing_needed': False,
    }


def save_state(state: dict):
    """상태 파일 원자적 저장."""
    save_json_atomic(STATE_PATH, state, default=str)


# ─── 데이터 수집 ───
def fetch_klines(client: Client, symbol: str, interval: str, limit: int = 1500) -> pd.DataFrame:
    """바이낸스에서 OHLCV 가져오기.

    선물 klines는 한 번에 큰 limit를 받지 못하므로 1500봉 이하로 나눠서 뒤에서부터 이어붙인다.
    """
    max_batch = 1500
    remaining = max(1, int(limit))
    end_time = None
    chunks = []

    try:
        while remaining > 0:
            batch_limit = min(remaining, max_batch)
            params = {'symbol': symbol, 'interval': interval, 'limit': batch_limit}
            if end_time is not None:
                params['endTime'] = end_time
            klines = client.futures_klines(**params)
            if not klines:
                break
            chunks.extend(klines)
            remaining -= len(klines)
            if len(klines) < batch_limit:
                break
            oldest_open_ms = int(klines[0][0])
            end_time = oldest_open_ms - 1
            time.sleep(0.05)

        if not chunks:
            return pd.DataFrame()

        # 중복 제거 후 시간순 정렬
        dedup = {}
        for row in chunks:
            dedup[int(row[0])] = row
        rows = [dedup[k] for k in sorted(dedup.keys())]

        df = pd.DataFrame(rows, columns=[
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
            'CloseTime', 'QuoteVol', 'Trades', 'TakerBuy', 'TakerQuote', 'Ignore'
        ])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[c] = df[c].astype(float)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].set_index('Date')
        return df.sort_index()
    except Exception as e:
        log.error(f"fetch_klines {symbol} {interval}: {e}")
        return pd.DataFrame()


def fetch_spot_klines(client: Client, symbol: str, interval: str, limit: int = 500) -> pd.DataFrame:
    """Binance Spot OHLCV (BTC canary 용 — 글로벌 표준 가격)."""
    try:
        klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
        if not klines:
            return pd.DataFrame()
        df = pd.DataFrame(klines, columns=[
            'Date', 'Open', 'High', 'Low', 'Close', 'Volume',
            'CloseTime', 'QuoteVol', 'Trades', 'TakerBuy', 'TakerQuote', 'Ignore'
        ])
        df['Date'] = pd.to_datetime(df['Date'], unit='ms')
        for c in ['Open', 'High', 'Low', 'Close', 'Volume']:
            df[c] = df[c].astype(float)
        df = df[['Date', 'Open', 'High', 'Low', 'Close', 'Volume']].set_index('Date')
        return df.sort_index()
    except Exception as e:
        log.error(f"fetch_spot_klines {symbol} {interval}: {e}")
        return pd.DataFrame()


def fetch_coingecko_top_futures(limit: int = UNIVERSE_TARGET_SIZE,
                                 cache_path: Optional[str] = None) -> List[Dict]:
    """CoinGecko top N 시총순 fetch. 실패 시 cache fallback."""
    headers = {'accept': 'application/json', 'User-Agent': 'Mozilla/5.0 BinanceFut/1.0'}
    for attempt in range(1, 4):
        try:
            r = requests.get(COINGECKO_URL, params={
                'vs_currency': 'usd', 'order': 'market_cap_desc',
                'per_page': limit, 'page': 1,
            }, headers=headers, timeout=15)
            if r.status_code == 200:
                data = r.json()
                if cache_path:
                    try:
                        tmp = cache_path + '.tmp'
                        with open(tmp, 'w') as f:
                            json.dump({'ts': datetime.now(timezone.utc).isoformat(), 'data': data}, f)
                        os.replace(tmp, cache_path)
                    except Exception:
                        pass
                return data
            log.warning(f"coingecko status {r.status_code} attempt {attempt}")
        except Exception as e:
            log.warning(f"coingecko fail attempt {attempt}: {e}")
        time.sleep(5 * attempt)
    if cache_path and os.path.isfile(cache_path):
        try:
            with open(cache_path) as f:
                return json.load(f).get('data', [])
        except Exception:
            pass
    return []


def fetch_binance_futures_listed(client: Client) -> set:
    """바이낸스 USDT-M 선물 TRADING 중인 심볼 set."""
    try:
        info = client.futures_exchange_info()
        return {s['symbol'] for s in info.get('symbols', [])
                if s.get('contractType') == 'PERPETUAL'
                and s.get('status') == 'TRADING'
                and s.get('quoteAsset') == 'USDT'}
    except Exception as e:
        log.warning(f"binance futures exchangeInfo fail: {e}")
        return set()


def refresh_universe(client: Client, cache_dir: str = '/tmp') -> List[str]:
    """CoinGecko top40 + 바이낸스 USDT-M 선물 listing intersect, 시총순 정렬.

    실패 시 HARDCODED_UNIVERSE_FALLBACK 리턴. 글로벌 UNIVERSE도 갱신.
    """
    global UNIVERSE
    cg = fetch_coingecko_top_futures(cache_path=os.path.join(cache_dir, 'binfut_cg_cache.json'))
    listed = fetch_binance_futures_listed(client)
    if not cg or not listed:
        log.warning(f"universe API 실패 (cg={len(cg)} listed={len(listed)}) → fallback")
        UNIVERSE = list(HARDCODED_UNIVERSE_FALLBACK)
        return UNIVERSE
    out: List[str] = []
    for item in cg:
        sym = (item.get('symbol') or '').upper()
        if not sym or sym in STABLECOINS:
            continue
        full = sym + 'USDT'
        if full in listed:
            out.append(full)
    if not out:
        log.warning("universe intersect 비어있음 → fallback")
        UNIVERSE = list(HARDCODED_UNIVERSE_FALLBACK)
        return UNIVERSE
    UNIVERSE = out
    log.info(f"universe 갱신: {len(out)}개 (cg={len(cg)} listed={len(listed)}) head={out[:5]}")
    return UNIVERSE


def fetch_all_data(client: Client) -> Dict[str, Dict[str, pd.DataFrame]]:
    """모든 심볼의 D OHLCV 수집 + 1h (동적 레버리지 backup, 현재 비활성).
    V24: D 단일 전략 (SMA42+mom127+snap57+vol90 → 500 bars 충분).
    1h 는 동적 레버리지 백업용, 비활성이지만 호환 위해 fetch 유지.

    BTC 는 canary 용 spot 가격으로 override (글로벌 표준 통일).
    alt momentum/health 는 futures perp 유지.
    """
    data = {'1h': {}, 'D': {}}
    for sym in UNIVERSE:
        for iv in ['1h', 'D']:
            limit = {'1h': 1500, 'D': 500}[iv]
            iv_api = '1d' if iv == 'D' else iv
            df = fetch_klines(client, sym, iv_api, limit)
            if not df.empty:
                coin = sym.replace('USDT', '')
                data[iv][coin] = df
            time.sleep(0.05)  # rate limit

    # BTC 는 spot 으로 override (canary 글로벌 표준)
    for iv in ['D']:
        iv_api = '1d' if iv == 'D' else iv
        spot_df = fetch_spot_klines(client, 'BTCUSDT', iv_api, 500)
        if not spot_df.empty:
            data[iv]['BTC'] = spot_df
            log.info(f"BTC {iv} spot override OK (last close={spot_df['Close'].iloc[-1]:,.0f})")
        else:
            log.warning(f"BTC {iv} spot fetch 실패 — perp 유지")
    return data


# ─── 시그널 계산 ───
def calc_sma(arr, period):
    if len(arr) < period:
        return 0
    return float(np.mean(arr[-period:]))


def calc_mom(arr, period):
    if len(arr) < period + 1:
        return -999
    return arr[-1] / arr[-period - 1] - 1


def calc_vol_daily(arr, bpd, lookback_bars):
    """일봉 리샘플 변동성."""
    if len(arr) < lookback_bars + 1:
        return 999
    daily = arr[-lookback_bars::bpd]
    if len(daily) < 10:
        return 999
    return float(np.std(np.diff(np.log(daily))))


def calc_vol_bars(arr, lookback_bars, bars_per_year):
    """순수 봉 기반 연환산 변동성."""
    if len(arr) < lookback_bars + 1:
        return 999
    rets = np.diff(np.log(arr[-lookback_bars - 1:]))
    return float(np.std(rets) * np.sqrt(bars_per_year))


def get_target_coins(target: Dict[str, float]) -> List[str]:
    return [coin for coin, w in target.items() if coin != 'CASH' and w > 0]


def rank_coins_capmom(target: Dict[str, float], data_1h: Dict[str, pd.DataFrame]) -> List[str]:
    """실거래용 cap+momentum 순위.

    백테스트의 get_mcap(date)는 과거 시총 순위를 사용하지만,
    라이브에서는 현재 유니버스 순서(시총순)를 cap rank 대용으로 사용한다.
    """
    coins = get_target_coins(target)
    scored = []
    for coin in coins:
        df = data_1h.get(coin)
        if df is None or df.empty:
            continue
        close = df['Close'].values[:-1]  # 마지막 진행중 봉 제외
        if len(close) <= LEVERAGE_MOM_LOOKBACK_BARS:
            mom = -999.0
        else:
            mom = close[-1] / close[-LEVERAGE_MOM_LOOKBACK_BARS - 1] - 1.0
        try:
            cap_rank = UNIVERSE.index(coin + 'USDT')
        except ValueError:
            cap_rank = len(UNIVERSE)
        score = mom - cap_rank * 1e-4
        scored.append((coin, score))
    scored.sort(key=lambda x: (-x[1], x[0]))
    ranked = [coin for coin, _ in scored]
    for coin in coins:
        if coin not in ranked:
            ranked.append(coin)
    return ranked


def score_coins_capmom(target: Dict[str, float], data_1h: Dict[str, pd.DataFrame]):
    """디버그용 cap+mom 점수 상세."""
    coins = get_target_coins(target)
    rows = []
    for coin in coins:
        df = data_1h.get(coin)
        if df is None or df.empty:
            rows.append((coin, -999.0, None, None))
            continue
        close = df['Close'].values[:-1]
        if len(close) <= LEVERAGE_MOM_LOOKBACK_BARS:
            mom = -999.0
        else:
            mom = close[-1] / close[-LEVERAGE_MOM_LOOKBACK_BARS - 1] - 1.0
        try:
            cap_rank = UNIVERSE.index(coin + 'USDT')
        except ValueError:
            cap_rank = len(UNIVERSE)
        score = mom - cap_rank * 1e-4
        rows.append((coin, score, mom, cap_rank))
    rows.sort(key=lambda x: (-x[1], x[0]))
    return rows


def apply_cash_degrade(lev: int, cash_w: float) -> int:
    if cash_w < STOP_GATE_CASH_THRESHOLD:
        return lev
    if lev >= LEVERAGE_CEILING:
        return LEVERAGE_MID
    if lev >= LEVERAGE_MID:
        return LEVERAGE_FLOOR
    return LEVERAGE_FLOOR


class StaleBarError(Exception):
    """V25 cycle 7: 마지막 완성봉이 누락된 stale 상태. 신호 계산 불가 → ABORT 경로."""
    pass


def _finalize_daily_bar_for_signal(df: pd.DataFrame, now_utc=None) -> pd.DataFrame:
    """V25 cycle 7 — 날짜 기반 마지막 완성봉 검증 (인덱스 기반 close[:-1] 대체).

    Binance 1d 봉 open_time = UTC 00:00. cron 09:05 KST = 00:05 UTC.
    completed_open_utc = 어제 00:00 UTC. 신호 입력은 이 봉의 close.

    분기:
      last_open == completed_open_utc → 정상, df 그대로
      last_open == current_open_utc (오늘 진행중) → 마지막 행 drop 후 사용
      last_open <  completed_open_utc → stale, StaleBarError 발생 (cron 측 ABORT)
      last_open >  current_open_utc → future, StaleBarError 발생

    Returns: df 끝이 completed_open_utc 봉인 DataFrame.
    """
    from datetime import datetime, timedelta, timezone
    if df is None or df.empty:
        raise StaleBarError("빈 DataFrame")
    if now_utc is None:
        now_utc = datetime.now(timezone.utc).replace(tzinfo=None)
    current_open_utc = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
    completed_open_utc = current_open_utc - timedelta(days=1)
    # df.index 는 to_datetime(ms) 결과 — tz-naive UTC.
    last_open = df.index[-1]
    if hasattr(last_open, 'to_pydatetime'):
        last_open = last_open.to_pydatetime()
    if last_open == completed_open_utc:
        return df
    if last_open == current_open_utc:
        df2 = df.iloc[:-1]
        if df2.empty or df2.index[-1] != completed_open_utc:
            raise StaleBarError(
                f"진행중 봉 drop 후에도 completed bar({completed_open_utc}) 없음. last={df2.index[-1] if len(df2) else 'empty'}"
            )
        return df2
    if last_open < completed_open_utc:
        raise StaleBarError(f"stale: last_open={last_open} < completed={completed_open_utc}")
    raise StaleBarError(f"future: last_open={last_open} > current={current_open_utc}")


def _calc_btc_cap_lev(data_d: Dict[str, pd.DataFrame]) -> int:
    """V25 BTC cap: BTC SMA42 ratio 기반.
    > BTC_CAP_THR_MAX(1.05) → Lmax(4)
    > BTC_CAP_THR_MID(1.015) → Lmid(3)
    else → Lmin(2)
    V25 cycle 7: 날짜 기반 마지막 완성봉 (UTC open_time anchor) 사용.
    StaleBarError → 호출부에서 ABORT 처리.
    """
    btc_df = data_d.get('BTC')
    if btc_df is None or btc_df.empty:
        return LEVERAGE_FLOOR
    btc_df = _finalize_daily_bar_for_signal(btc_df)  # raises StaleBarError
    close = btc_df['Close'].values
    if len(close) < BTC_CAP_SMA_PERIOD + 1:
        return LEVERAGE_FLOOR
    sma = float(np.asarray(close[-BTC_CAP_SMA_PERIOD:], dtype=float).mean())
    cur = float(close[-1])
    if sma <= 0:
        return LEVERAGE_FLOOR
    ratio = cur / sma
    if ratio > BTC_CAP_THR_MAX:
        lev = LEVERAGE_CEILING
    elif ratio > BTC_CAP_THR_MID:
        lev = LEVERAGE_MID
    else:
        lev = LEVERAGE_FLOOR
    if DEBUG_LEVERAGE:
        log.info(f"  BTC_cap: prev_close=${cur:,.2f} SMA{BTC_CAP_SMA_PERIOD}=${sma:,.2f} "
                 f"ratio={ratio:.4f} → L={lev}")
    return lev


def _calc_percoin_k2_lev(coin: str, data_d: Dict[str, pd.DataFrame]) -> int:
    """V25 per-coin K2: close/SMA{K2_SMA_PERIOD} ratio 기반.
    > 1 + K2_HYST*3 (1.075) → Lmax(4)
    > 1 + K2_HYST (1.025) → Lmid(3)
    else → Lmin(2)
    prev_date(t-1) 기준.
    """
    df = data_d.get(coin)
    if df is None or df.empty:
        return LEVERAGE_FLOOR
    df = _finalize_daily_bar_for_signal(df)  # V25 cycle 7: UTC anchor, raises StaleBarError
    close = df['Close'].values
    if len(close) < K2_SMA_PERIOD + 1:
        return LEVERAGE_FLOOR
    sma = float(np.asarray(close[-K2_SMA_PERIOD:], dtype=float).mean())
    cur = float(close[-1])
    if sma <= 0:
        return LEVERAGE_FLOOR
    ratio = cur / sma
    thr_max = 1.0 + K2_HYST * 3
    thr_mid = 1.0 + K2_HYST
    if ratio > thr_max:
        lev = LEVERAGE_CEILING
    elif ratio > thr_mid:
        lev = LEVERAGE_MID
    else:
        lev = LEVERAGE_FLOOR
    if DEBUG_LEVERAGE:
        log.info(f"  K2[{coin}]: prev_close={cur:.4f} SMA{K2_SMA_PERIOD}={sma:.4f} "
                 f"ratio={ratio:.4f} → L={lev}")
    return lev


def get_coin_leverage_map(target: Dict[str, float], data_1h: Dict[str, pd.DataFrame],
                          data_d: Optional[Dict[str, pd.DataFrame]] = None) -> Dict[str, int]:
    """V25 동적 per-coin L. 최종 L = min(BTC_cap, per_coin_K2).

    data_d (1D 봉) 가 필요 (K2 + BTC cap 모두 1D 기반).
    data_d=None 이면 fallback (LEVERAGE_FLOOR).
    """
    coins = get_target_coins(target)
    if not coins:
        return {}
    if data_d is None:
        log.warning("V25 get_coin_leverage_map: data_d 없음 → 모든 코인 LEVERAGE_FLOOR fallback")
        return {coin: LEVERAGE_FLOOR for coin in coins}

    # V25 cycle 7: StaleBarError 가 발생하면 호출자 (main) 가 ABORT 처리.
    btc_cap = _calc_btc_cap_lev(data_d)
    lev_map = {}
    for coin in coins:
        pc_lev = _calc_percoin_k2_lev(coin, data_d)
        final_lev = min(btc_cap, pc_lev)
        lev_map[coin] = final_lev
        if DEBUG_LEVERAGE:
            log.info(f"  {coin} → final L = min(BTC_cap={btc_cap}, K2={pc_lev}) = {final_lev}")
    return lev_map


def compute_strategy_target(strat_name: str, strat_params: dict,
                             data: dict, state: dict,
                             alerts: Optional[List[str]] = None) -> Dict[str, float]:
    """단일 전략의 목표 비중 계산."""
    iv = strat_params['interval']
    bpd = {'D': 1, '4h': 6, '2h': 12, '1h': 24}[iv]
    bars_per_year = bpd * 365
    bars = data[iv]

    if 'BTC' not in bars or bars['BTC'].empty:
        return {'CASH': 1.0}

    btc_close = bars['BTC']['Close'].values
    sma_p = strat_params['sma_bars']

    if len(btc_close) < sma_p + 1:
        return {'CASH': 1.0}

    # 전략 상태 로드/초기화
    ss = state.get('strategies', {}).get(strat_name, {})
    prev_canary = _as_bool(ss.get('canary_on', False))
    last_bar_ts = ss.get('last_bar_ts', None)  # 마지막 처리된 봉 타임스탬프
    snapshots = ss.get('snapshots', [{'CASH': 1.0}] * strat_params['n_snapshots'])
    bar_counter = ss.get('bar_counter', 0)

    # 새 봉 확인: 마지막 완성봉의 타임스탬프
    btc_df = bars['BTC']
    latest_bar_ts = str(btc_df.index[-2])  # -1은 진행중, -2가 마지막 완성봉
    if latest_bar_ts == last_bar_ts:
        # 같은 봉 — 이미 처리됨, 이전 target 유지
        combined = ss.get('last_combined', {'CASH': 1.0})
        return combined

    # 카나리 (t-1 기준: 마지막 완성봉)
    c_prev = btc_close[:-1]  # 진행중 봉 제외
    sma_val = calc_sma(c_prev, sma_p)
    hyst = strat_params['canary_hyst']

    if prev_canary:
        canary_on = not (c_prev[-1] < sma_val * (1 - hyst))
    else:
        canary_on = c_prev[-1] > sma_val * (1 + hyst)

    canary_flipped = canary_on != prev_canary

    # 카나리 상세 로깅
    ratio = c_prev[-1] / sma_val if sma_val > 0 else 0
    log.info(f"  {strat_name} BTC=${c_prev[-1]:,.0f} SMA({sma_p})=${sma_val:,.0f}"
             f" ratio={ratio:.4f} canary={'ON' if canary_on else 'OFF'}"
             f"{'  *** FLIPPED ***' if canary_flipped else ''}")

    # 카나리 플립 텔레그램 알림
    if canary_flipped and alerts is not None:
        direction = "ON" if canary_on else "OFF"
        alerts.append(
            f"{strat_name} 카나리 {direction} | BTC ${c_prev[-1]:,.0f} / SMA ${sma_val:,.0f} ({ratio:.3f})"
        )

    # 헬스체크 + 종목 선정
    def compute_weights():
        if not canary_on:
            log.info(f"  {strat_name} weights: canary OFF -> CASH 100%")
            return {'CASH': 1.0}

        mom_s = strat_params['mom_short_bars']
        mom_l = strat_params['mom_long_bars']
        hmode = strat_params['health_mode']
        vol_mode = strat_params['vol_mode']
        vol_th = strat_params['vol_threshold']

        healthy = []
        debug_rows = []
        for sym in UNIVERSE:
            coin = sym.replace('USDT', '')
            df = bars.get(coin)
            if df is None or df.empty:
                debug_rows.append((coin, 'no_data', None, None, None))
                continue
            c = df['Close'].values[:-1]  # t-1
            min_bars = max(mom_s, mom_l, sma_p, 90 * bpd)
            if len(c) < min_bars:
                debug_rows.append((coin, 'short_data', None, None, None))
                continue

            m_short = calc_mom(c, mom_s)
            m_long = calc_mom(c, mom_l) if 'mom2' in hmode else 999
            if vol_mode == 'bar':
                vol = calc_vol_bars(c, 90 * bpd, bars_per_year)
            else:
                vol = calc_vol_daily(c, bpd, 90 * bpd)

            if hmode == 'mom2vol':
                ok = m_short > 0 and m_long > 0 and vol <= vol_th
            elif hmode == 'mom1vol':
                ok = m_short > 0 and vol <= vol_th
            elif hmode == 'mom1':
                ok = m_short > 0
            else:
                ok = True

            debug_rows.append((coin, 'ok' if ok else 'fail', m_short, m_long, vol))
            if ok:
                healthy.append(coin)

        if debug_rows:
            preview = []
            for coin, status, m_short, m_long, vol in debug_rows[:10]:
                if status in ('no_data', 'short_data'):
                    preview.append(f"{coin}:{status}")
                else:
                    long_str = f"{m_long:+.3f}" if m_long is not None and m_long != 999 else "-"
                    vol_str = f"{vol:.3f}" if vol is not None else "-"
                    preview.append(f"{coin}:{status}(m1={m_short:+.3f},m2={long_str},v={vol_str})")
            log.info(f"  {strat_name} health preview: {' | '.join(preview)}")
        log.info(f"  {strat_name} healthy_count={len(healthy)} healthy={healthy[:UNIVERSE_SIZE]}")

        # Greedy absorption
        picks = healthy[:UNIVERSE_SIZE]
        if len(picks) > 1:
            for i in range(len(picks) - 1, 0, -1):
                df_a = bars.get(picks[i-1])
                df_b = bars.get(picks[i])
                if df_a is None or df_b is None:
                    continue
                ca = df_a['Close'].values[:-1]
                cb = df_b['Close'].values[:-1]
                ma = calc_mom(ca, mom_s)
                mb = calc_mom(cb, mom_s)
                if ma >= mb:
                    picks.pop(i)

        log.info(f"  {strat_name} initial_top={healthy[:UNIVERSE_SIZE]} final_picks={picks}")

        if not picks:
            log.info(f"  {strat_name} picks empty -> CASH 100%")
            return {'CASH': 1.0}

        w = min(1.0 / len(picks), CAP)
        weights = {coin: w for coin in picks}
        total = sum(weights.values())
        if total < 0.999:
            weights['CASH'] = 1.0 - total
        log.info(f"  {strat_name} weights={weights}")
        return weights

    # 스냅샷 갱신
    need_update = False
    n_snap = strat_params['n_snapshots']
    snap_iv = strat_params['snap_interval_bars']

    if canary_flipped:
        for si in range(n_snap):
            snapshots[si] = compute_weights()
        need_update = True
        log.info(f"  {strat_name} EVENT: canary_flipped -> rebalancing_needed=true")
    elif canary_on:
        for si in range(n_snap):
            offset = int(si * snap_iv / n_snap)
            if bar_counter % snap_iv == offset:
                new_w = compute_weights()
                if new_w != snapshots[si]:
                    snapshots[si] = new_w
                    need_update = True
                    log.info(f"  {strat_name} EVENT: tranche_refresh[{si}] -> rebalancing_needed=true")

    # 스냅샷 합산
    combined = {}
    for snap in snapshots:
        for t, w in snap.items():
            combined[t] = combined.get(t, 0) + w / n_snap
    total = sum(combined.values())
    if total > 0:
        combined = {t: w / total for t, w in combined.items()}

    # 상태 저장 (봉 단위 카운터 — 새 봉일 때만 증가)
    bar_counter += 1
    ss_new = {
        'canary_on': canary_on,
        'bar_counter': bar_counter,
        'last_bar_ts': latest_bar_ts,
        'snapshots': snapshots,
        'last_combined': combined,
        'canary_info': {
            'on': canary_on,
            'flipped': canary_flipped,
            'ratio': ratio,
            'cur': float(c_prev[-1]),
            'sma_val': float(sma_val),
            'sma_p': sma_p,
        },
    }
    if 'strategies' not in state:
        state['strategies'] = {}
    state['strategies'][strat_name] = ss_new
    if need_update:
        state['rebalancing_needed'] = True

    return combined


def combine_ensemble(targets: Dict[str, Dict[str, float]],
                     weights: Dict[str, float]) -> Dict[str, float]:
    """여러 전략의 목표 비중을 가중 합산."""
    merged = {}
    for strat_name, w in weights.items():
        if strat_name not in targets:
            continue
        for coin, cw in targets[strat_name].items():
            merged[coin] = merged.get(coin, 0) + cw * w
    return merged


def apply_refill_v2_fut(state: Dict, data: Dict[str, Dict[str, pd.DataFrame]]) -> Dict[str, float]:
    """V24 fut refill v2 — drift fire 시 각 전략의 snapshot 중 mom2 음수 코인 자리를
    fresh healthy 로 교체. state['strategies'] 의 snapshots 를 in-place 수정한 후
    재합산된 combined 를 반환.

    Fail 기준: mom_short < 0 AND mom_long < 0 (vol 미포함, BT 일치).
    """
    bars = data.get('D', {})
    if not bars:
        return state.get('last_combined', {'CASH': 1.0})

    def _closed_close(df: pd.DataFrame):
        """진행중(미완성) 일봉 제외한 close 배열 — 시그널 경로(_finalize_daily_bar_for_signal)와
        동일한 t-1 완성봉 기준. look-ahead 방지 (V25 cycle 7 정합 수정 2026-06-06).
        finalize 실패 시 보수적으로 마지막 봉 제외."""
        try:
            return _finalize_daily_bar_for_signal(df)['Close'].values
        except Exception:
            v = df['Close'].values
            return v[:-1] if len(v) > 0 else v

    strats = state.get('strategies', {})
    new_combined: Dict[str, float] = {}
    n_strats = max(1, len(STRATEGIES))
    log.info(f"refill v2 fut: 시작. strategies={n_strats}")

    for strat_name, sp in STRATEGIES.items():
        ss = strats.get(strat_name, {})
        snaps = ss.get('snapshots') or []
        if not snaps:
            log.warning(f"refill v2 [{strat_name}]: snapshots 없음 → skip")
            continue
        mom_s = sp['mom_short_bars']
        mom_l = sp['mom_long_bars']
        log.info(f"refill v2 [{strat_name}]: snaps={len(snaps)} mom_s={mom_s} mom_l={mom_l}")

        _fail_cache: Dict[str, Tuple[bool, float, float]] = {}

        def _is_failed(coin: str) -> bool:
            if coin in _fail_cache:
                return _fail_cache[coin][0]
            df = bars.get(coin)
            if df is None or df.empty:
                _fail_cache[coin] = (False, 0.0, 0.0)
                return False
            c = _closed_close(df)
            if len(c) < max(mom_s, mom_l) + 1:
                _fail_cache[coin] = (False, 0.0, 0.0)
                return False
            ms = calc_mom(c, mom_s)
            ml = calc_mom(c, mom_l)
            fail = ms < 0 and ml < 0
            _fail_cache[coin] = (fail, ms, ml)
            return fail

        healthy_pool: List[Tuple[str, float]] = []
        for coin, df in bars.items():
            if coin == 'BTC':
                continue
            c = _closed_close(df)
            if len(c) < max(mom_s, mom_l) + 1:
                continue
            ms = calc_mom(c, mom_s)
            ml = calc_mom(c, mom_l)
            if ms > 0 and ml > 0:
                healthy_pool.append((coin, ms))
        # tie-break: 심볼 오름차순으로 결정성 보장
        healthy_pool.sort(key=lambda x: (-x[1], x[0]))
        healthy_sorted = [c for c, _ in healthy_pool]
        log.info(f"refill v2 [{strat_name}]: healthy_pool top5={[(c, f'{m*100:+.1f}%') for c, m in healthy_pool[:5]]} (total {len(healthy_pool)})")

        new_snaps = []
        change_count = 0
        for si, snap in enumerate(snaps):
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
            cash_w = sum(float(snap.get(k, 0.0)) for k in snap if k.upper() == 'CASH')
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
            new_snaps.append(new_sn)
            if failed_coins:
                fail_detail = ','.join(f'{c}(ms={_fail_cache[c][1]*100:+.1f},ml={_fail_cache[c][2]*100:+.1f})' for c in failed_coins)
                log.info(f"refill v2 [{strat_name}] snap[{si}]: failed=[{fail_detail}] replaced={replaced:.4f} → picks={picks_made or 'CASH'}")
                change_count += 1
        if change_count == 0:
            log.info(f"refill v2 [{strat_name}]: 모든 snapshot 변경 없음 (fail 없음)")

        ss['snapshots'] = new_snaps
        n_snap = len(new_snaps) or 1
        for snap in new_snaps:
            for t, w in snap.items():
                key = 'CASH' if t.upper() == 'CASH' else t
                new_combined[key] = new_combined.get(key, 0.0) + (w / n_snap) / n_strats

    total = sum(new_combined.values())
    if total > 0:
        new_combined = {k: v / total for k, v in new_combined.items()}
    else:
        new_combined = {'CASH': 1.0}
    return new_combined


# ─── 주문 실행 ───
def _safe_float(value, default: float = 0.0) -> float:
    """None/빈문자열에도 안전한 float 변환."""
    try:
        if value in (None, ''):
            return default
        return float(value)
    except Exception:
        return default


def _is_retryable_fetch_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    retry_markers = [
        'timeout',
        'timed out',
        'backend server',
        'service unavailable',
        'internal error',
        'server busy',
        'temporarily unavailable',
        'connection',
        'recvwindow',
        'too many requests',
        '-1007',
    ]
    return any(marker in msg for marker in retry_markers) or isinstance(exc, requests.exceptions.RequestException)


def get_current_positions(client: Client):
    """현재 선물 포지션 + PV 조회. returns (positions_dict, total_pv, ok)."""
    last_exc = None
    for attempt in range(1, POSITION_FETCH_MAX_RETRIES + 1):
        positions = {}
        try:
            info = client.futures_account()
            balance = _safe_float(info.get('totalWalletBalance'))
            unrealized = _safe_float(info.get('totalUnrealizedProfit'))
            total_pv = balance + unrealized

            # futures_account()['positions']는 markPrice/unRealizedProfit이 null로 오는 경우가 있어
            # 포지션 상세는 futures_position_information() 기준으로 읽는다.
            pos_rows = client.futures_position_information()
            tickers = {}
            for row in client.futures_symbol_ticker():
                sym = row.get('symbol')
                if sym:
                    tickers[sym] = _safe_float(row.get('price'))

            for p in pos_rows:
                amt = _safe_float(p.get('positionAmt'))
                if amt != 0:
                    sym = p.get('symbol')
                    if not sym:
                        continue
                    coin = sym.replace('USDT', '')
                    mark = _safe_float(p.get('markPrice'))
                    if mark <= 0:
                        mark = tickers.get(sym, 0.0)
                    notional = abs(_safe_float(p.get('notional')))
                    if notional <= 0 and mark > 0:
                        notional = abs(amt * mark)
                    lev = _safe_float(p.get('leverage')) or 0.0
                    margin = (_safe_float(p.get('isolatedMargin'))
                              or _safe_float(p.get('positionInitialMargin'))
                              or _safe_float(p.get('initialMargin'))
                              or 0.0)
                    if margin > 0:
                        real_notional = margin
                        if lev <= 0 and margin > 0:
                            lev = notional / margin if margin > 0 else 1.0
                    elif lev > 0:
                        real_notional = notional / lev
                    else:
                        real_notional = notional
                        lev = 1.0
                    positions[coin] = {
                        'qty': amt,
                        'qty_raw': str(p.get('positionAmt') or ''),
                        'symbol': sym,
                        'entry_price': _safe_float(p.get('entryPrice')),
                        'mark_price': mark,
                        'pnl': _safe_float(p.get('unRealizedProfit')),
                        'liquidation_price': _safe_float(p.get('liquidationPrice')),
                        'notional': notional,
                        'leverage': lev,
                        'real_notional': real_notional,
                        'weight': notional / total_pv if total_pv > 0 else 0,
                        'real_weight': real_notional / total_pv if total_pv > 0 else 0,
                    }

            return positions, total_pv, True
        except Exception as e:
            last_exc = e
            if attempt < POSITION_FETCH_MAX_RETRIES and _is_retryable_fetch_error(e):
                delay = POSITION_FETCH_RETRY_DELAYS[min(attempt - 1, len(POSITION_FETCH_RETRY_DELAYS) - 1)]
                log.warning(
                    f"get_positions retry {attempt}/{POSITION_FETCH_MAX_RETRIES} after error: {e}"
                )
                time.sleep(delay)
                continue
            log.error(f"get_positions error: {e}")
            return {}, 0.0, False
    log.error(f"get_positions error: {last_exc}")
    return {}, 0.0, False


_exchange_info_cache = None

def get_exchange_info(client: Client):
    """exchange_info 캐싱 (API 호출 최소화)."""
    global _exchange_info_cache
    if _exchange_info_cache is None:
        _exchange_info_cache = client.futures_exchange_info()
    return _exchange_info_cache


def get_symbol_constraints(client: Client, symbol: str) -> Dict[str, float]:
    """심볼별 주문 제약(step/minQty/minNotional/tick) 조회."""
    info = get_exchange_info(client)
    for s in info['symbols']:
        if s['symbol'] != symbol:
            continue
        out = {
            'step_size': 0.0,
            'min_qty': 0.0,
            'min_notional': MIN_NOTIONAL,
            'tick_size': 0.0,
            'qty_precision': 8,
        }
        for f in s.get('filters', []):
            ftype = f.get('filterType')
            if ftype == 'LOT_SIZE':
                step_str = f.get('stepSize', '0')
                out['step_size'] = float(step_str)
                out['min_qty'] = float(f.get('minQty', 0))
                out['qty_precision'] = len(step_str.rstrip('0').split('.')[-1]) if '.' in step_str else 0
            elif ftype == 'PRICE_FILTER':
                out['tick_size'] = float(f.get('tickSize', 0))
            elif ftype in {'MIN_NOTIONAL', 'NOTIONAL'}:
                out['min_notional'] = float(f.get('notional', f.get('minNotional', MIN_NOTIONAL)))
        return out
    return {
        'step_size': 0.0,
        'min_qty': 0.0,
        'min_notional': MIN_NOTIONAL,
        'tick_size': 0.0,
        'qty_precision': 8,
    }


def format_quantity(client: Client, symbol: str, qty: float) -> str:
    """심볼별 수량 정밀도 맞추기."""
    try:
        c = get_symbol_constraints(client, symbol)
        step = c['step_size']
        precision = int(c['qty_precision'])
        if step > 0:
            adjusted = int(qty / step) * step
            return f"{adjusted:.{precision}f}"
        return f"{qty:.8f}"
    except:
        return f"{qty:.8f}"


def format_price(client: Client, symbol: str, price: float) -> str:
    """심볼별 가격 정밀도 맞추기."""
    try:
        info = get_exchange_info(client)
        for s in info['symbols']:
            if s['symbol'] == symbol:
                for f in s['filters']:
                    if f['filterType'] == 'PRICE_FILTER':
                        tick = float(f['tickSize'])
                        precision = len(f['tickSize'].rstrip('0').split('.')[-1]) if '.' in f['tickSize'] else 0
                        adjusted = int(price / tick) * tick
                        return f"{adjusted:.{precision}f}"
        return f"{price:.8f}"
    except Exception:
        return f"{price:.8f}"


def _is_retryable_order_error(exc: Exception) -> bool:
    """일시적 주문 오류만 재시도한다."""
    msg = str(exc).lower()

    non_retry_markers = [
        'insufficient margin',
        'margin is insufficient',
        'reduceonly order is rejected',
        'reduceonly',
        'precision is over the maximum',
        'mandatory parameter',
        'invalid quantity',
        'min notional',
        'quantity less than or equal to zero',
        'order would immediately trigger',
        'unknown symbol',
        'invalid symbol',
        'parameter',
    ]
    if any(marker in msg for marker in non_retry_markers):
        return False

    retry_markers = [
        'timeout',
        'timed out',
        'internal error',
        'server busy',
        'service unavailable',
        'too many requests',
        'recvwindow',
        'connection',
        'temporarily unavailable',
        'try again',
    ]
    if any(marker in msg for marker in retry_markers):
        return True

    if isinstance(exc, requests.exceptions.RequestException):
        return True

    return False


def create_order_with_retry(client: Client, order_params: dict):
    """주문 실행. 일시 오류만 제한 횟수 재시도."""
    last_exc = None
    for attempt in range(1, ORDER_MAX_RETRIES + 1):
        try:
            return client.futures_create_order(**order_params)
        except BinanceAPIException as e:
            last_exc = e
            if not _is_retryable_order_error(e) or attempt >= ORDER_MAX_RETRIES:
                raise
            delay = ORDER_RETRY_DELAYS[min(attempt - 1, len(ORDER_RETRY_DELAYS) - 1)]
            log.warning(
                f"ORDER RETRY {attempt}/{ORDER_MAX_RETRIES} {order_params.get('side')} "
                f"{order_params.get('symbol')} after Binance error: {e}"
            )
            time.sleep(delay)
        except Exception as e:
            last_exc = e
            if not _is_retryable_order_error(e) or attempt >= ORDER_MAX_RETRIES:
                raise
            delay = ORDER_RETRY_DELAYS[min(attempt - 1, len(ORDER_RETRY_DELAYS) - 1)]
            log.warning(
                f"ORDER RETRY {attempt}/{ORDER_MAX_RETRIES} {order_params.get('side')} "
                f"{order_params.get('symbol')} after error: {e}"
            )
            time.sleep(delay)
    if last_exc is not None:
        raise last_exc


def execute_rebalance(client: Client, target: Dict[str, float], total_pv: float,
                      target_lev_map: Dict[str, int],
                      order_alerts: Optional[List[str]] = None,
                      error_alerts: Optional[List[str]] = None):
    """목표 비중으로 delta 리밸런싱. 매도 먼저, 매수 나중."""
    if total_pv <= 0:
        log.warning("PV <= 0, skip rebalance")
        return

    # 현재 포지션
    current_positions, _, current_ok = get_current_positions(client)
    if not current_ok:
        log.warning("REBALANCE skip: current positions fetch failed")
        return
    trades = []
    log.info(f"REBALANCE target={target}")
    log.info(f"REBALANCE target_lev_map={target_lev_map}")

    # 리밸런싱 전에 기존 closePosition 스탑/TP 주문을 정리한다.
    # 그렇지 않으면 leverage/margin 변경이나 새 스탑 등록이 막힐 수 있다.
    cancel_stop_orders(client, list(UNIVERSE))

    # 매도/청산 (보유 중이지만 target에 없거나 줄어야)
    for coin, pos in current_positions.items():
        target_w = target.get(coin, 0)
        target_lev = target_lev_map.get(coin, LEVERAGE_FLOOR)
        target_notional = total_pv * (1 - CASH_BUFFER) * target_w * target_lev
        current_notional = pos['notional']
        delta_pct = (target_notional - current_notional) / current_notional if current_notional > 0 else 999
        log.info(
            f"REBALANCE sell_check {coin}: current=${current_notional:.2f} "
            f"target_w={target_w:.1%} target_lev={target_lev}x target=${target_notional:.2f} "
            f"delta={delta_pct:+.1%}"
        )

        if target_w <= 0:
            # 전량 청산 (reduceOnly)
            qty_raw = str(pos.get('qty_raw') or '').strip()
            if qty_raw.startswith('-'):
                qty_raw = qty_raw[1:]
            trades.append(('SELL', pos['symbol'], qty_raw if qty_raw else abs(pos['qty']), True, True))
        elif delta_pct < -DELTA_THRESHOLD:
            sell_qty = abs(pos['qty']) * abs(delta_pct)
            trades.append(('SELL', pos['symbol'], sell_qty, True, False))

    # 매수 (target에 있지만 미보유거나 늘어야)
    for coin, w in target.items():
        if coin == 'CASH' or w <= 0:
            continue
        sym = coin + 'USDT'
        target_lev = target_lev_map.get(coin, LEVERAGE_FLOOR)
        target_notional = total_pv * (1 - CASH_BUFFER) * w * target_lev

        current_notional = current_positions[coin]['notional'] if coin in current_positions else 0
        if current_notional > 0:
            delta_pct = (target_notional - current_notional) / current_notional
            log.info(
                f"REBALANCE buy_check {coin}: current=${current_notional:.2f} "
                f"target_w={w:.1%} target_lev={target_lev}x target=${target_notional:.2f} "
                f"delta={delta_pct:+.1%}"
            )
            if delta_pct <= DELTA_THRESHOLD:
                continue
        buy_notional = target_notional - current_notional
        if buy_notional > MIN_NOTIONAL:
            try:
                price = float(client.futures_symbol_ticker(symbol=sym)['price'])
                buy_qty = buy_notional / price
                qty_str = format_quantity(client, sym, buy_qty)
                exec_qty = float(qty_str)
                constraints = get_symbol_constraints(client, sym)
                exec_notional = exec_qty * price
                log.info(
                    f"REBALANCE buy_plan {coin}: buy_notional=${buy_notional:.2f} "
                    f"price=${price:.2f} qty={buy_qty:.6f} exec_qty={exec_qty:.6f}"
                )
                if exec_qty <= 0 or exec_qty < constraints['min_qty'] or exec_notional < constraints['min_notional']:
                    log.info(
                        f"REBALANCE skip_small_buy {coin}: exec_qty={exec_qty:.6f} "
                        f"min_qty={constraints['min_qty']:.6f} exec_notional=${exec_notional:.2f} "
                        f"min_notional=${constraints['min_notional']:.2f}"
                    )
                    continue
                trades.append(('BUY', sym, buy_qty, False, False))
            except Exception as e:
                log.error(f"price fetch {sym}: {e}")

    log.info(f"REBALANCE planned_trades={trades}")

    # 매도 먼저 실행, 매수 나중
    for side, symbol, qty, reduce_only, is_full_close in sorted(trades, key=lambda x: 0 if x[0] == 'SELL' else 1):
        try:
            qty_str = str(qty) if is_full_close else format_quantity(client, symbol, qty)
            if float(qty_str) <= 0:
                continue

            order_params = dict(
                symbol=symbol,
                side=side,
                type='MARKET',
                quantity=qty_str,
            )
            if reduce_only:
                order_params['reduceOnly'] = 'true'

            order = create_order_with_retry(client, order_params)
            log.info(f"ORDER {side} {symbol} qty={qty_str}: {order.get('status', 'OK')}")
            if order_alerts is not None:
                order_alerts.append(f"{side} {symbol} {qty_str}")
            time.sleep(0.1)
        except BinanceAPIException as e:
            log.error(f"ORDER FAILED {side} {symbol} qty={qty_str}: {e}")
            if error_alerts is not None:
                error_alerts.append(f"ORDER FAILED {side} {symbol} {qty_str}: {e}")


def needs_rebalance(client: Client, target: Dict[str, float], current_positions: Dict[str, dict],
                    total_pv: float, target_lev_map: Dict[str, int]) -> bool:
    """현재 포지션이 목표와 충분히 다르면 리밸런싱 필요."""
    if total_pv <= 0:
        return False

    # 보유 중인데 목표에 없거나 많이 줄어야 하는 경우
    for coin, pos in current_positions.items():
        target_w = target.get(coin, 0.0)
        target_lev = target_lev_map.get(coin, LEVERAGE_FLOOR)
        target_notional = total_pv * (1 - CASH_BUFFER) * target_w * target_lev
        current_notional = pos.get('notional', 0.0)
        symbol = pos.get('symbol', coin + 'USDT')
        current_qty = abs(pos.get('qty', 0.0))
        constraints = get_symbol_constraints(client, symbol)
        if target_w <= 0 and current_notional > MIN_NOTIONAL:
            qty_raw = str(pos.get('qty_raw') or '').strip()
            if qty_raw.startswith('-'):
                qty_raw = qty_raw[1:]
            qty_str = qty_raw or format_quantity(client, symbol, current_qty)
            exec_qty = float(qty_str)
            if exec_qty > 0 and exec_qty >= constraints['min_qty']:
                log.info(f"REBALANCE_NEEDED {coin}: target_w=0 but current=${current_notional:.2f}")
                return True
            continue
        if current_notional > 0:
            delta_pct = (target_notional - current_notional) / current_notional
            if abs(delta_pct) > DELTA_THRESHOLD:
                delta_qty = current_qty * abs(delta_pct)
                qty_str = format_quantity(client, symbol, delta_qty)
                exec_qty = float(qty_str)
                price = current_notional / current_qty if current_qty > 0 else 0.0
                exec_notional = exec_qty * price
                if exec_qty <= 0 or exec_qty < constraints['min_qty'] or exec_notional < constraints['min_notional']:
                    log.info(
                        f"REBALANCE_TOLERATED {coin}: delta={delta_pct:+.1%} but "
                        f"exec_qty={exec_qty:.6f} exec_notional=${exec_notional:.2f} below exchange minimum"
                    )
                    continue
                log.info(
                    f"REBALANCE_NEEDED {coin}: current=${current_notional:.2f} "
                    f"target=${target_notional:.2f} delta={delta_pct:+.1%}"
                )
                return True

    # 목표에 있는데 미보유/증액 필요
    for coin, w in target.items():
        if coin == 'CASH' or w <= 0:
            continue
        target_lev = target_lev_map.get(coin, LEVERAGE_FLOOR)
        target_notional = total_pv * (1 - CASH_BUFFER) * w * target_lev
        current_notional = current_positions.get(coin, {}).get('notional', 0.0)
        if current_notional <= 0 and target_notional > MIN_NOTIONAL:
            symbol = coin + 'USDT'
            constraints = get_symbol_constraints(client, symbol)
            try:
                price = float(client.futures_symbol_ticker(symbol=symbol)['price'])
            except Exception:
                price = 0.0
            qty = (target_notional / price) if price > 0 else 0.0
            qty_str = format_quantity(client, symbol, qty)
            exec_qty = float(qty_str)
            exec_notional = exec_qty * price
            if exec_qty <= 0 or exec_qty < constraints['min_qty'] or exec_notional < constraints['min_notional']:
                log.info(
                    f"REBALANCE_TOLERATED {coin}: no position but exec_qty={exec_qty:.6f} "
                    f"exec_notional=${exec_notional:.2f} below exchange minimum"
                )
                continue
            log.info(f"REBALANCE_NEEDED {coin}: no position and target=${target_notional:.2f}")
            return True

    return False


def set_leverage(client: Client, symbol: str, leverage: int) -> bool:
    """레버리지 설정. V25 — bool 반환, 실패 시 False. transient retry."""
    try:
        resp = _with_retry(lambda: client.futures_change_leverage(symbol=symbol, leverage=leverage))
        if DEBUG_LEVERAGE:
            log.info(f"  set_leverage OK {symbol}={leverage}x resp={resp}")
        return True
    except BinanceAPIException as e:
        if _is_idempotent_error(e):
            if DEBUG_LEVERAGE:
                log.info(f"  set_leverage {symbol}={leverage}x idempotent (code={getattr(e,'code',None)})")
            return True
        log.error(f"set_leverage FAILED {symbol}={leverage}: code={getattr(e,'code',None)} msg={e}")
        return False


def set_margin_type(client: Client, symbol: str, margin_type: str = 'CROSSED') -> bool:
    """마진 모드 설정. V25 — bool 반환. transient retry."""
    try:
        resp = _with_retry(lambda: client.futures_change_margin_type(symbol=symbol, marginType=margin_type))
        if DEBUG_MARGIN:
            log.info(f"  set_margin OK {symbol}={margin_type} resp={resp}")
        return True
    except BinanceAPIException as e:
        if _is_idempotent_error(e):
            if DEBUG_MARGIN:
                log.info(f"  set_margin {symbol}={margin_type} idempotent (code={getattr(e,'code',None)})")
            return True
        log.error(f"set_margin FAILED {symbol}={margin_type}: code={getattr(e,'code',None)} msg={e}")
        return False


# V25: margin_type 정규화 명시 매핑 (cycle 2 fix — 'isolated'.replace('ed','') 취약 정규화 제거)
_MARGIN_NORM = {
    'crossed': 'cross', 'cross': 'cross',
    'isolated': 'isolated', 'isolate': 'isolated',
}

# V25 cycle 5 A: ABORT 알림 disk persist (process kill 대비 — Telegram flush 의존 제거)
V25_ABORT_LOG = os.path.expanduser('~/binance_abort.log')


def _v25_persist_abort_log(msg: str):
    """V25 cycle 5 A: ABORT 발생 시 즉시 disk flush. Telegram 실패/프로세스 kill 시에도 잔존."""
    try:
        from datetime import datetime
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        with open(V25_ABORT_LOG, 'a') as f:
            f.write(f"[{ts} KST] {msg}\n")
            f.flush()
            try:
                os.fsync(f.fileno())  # 강제 디스크 sync
            except Exception:
                pass
    except Exception as e:
        log.error(f"_v25_persist_abort_log 실패: {e}")


# V25 cycle 6: 건강성 / heartbeat / abort_streak / reconciliation
V25_HEALTH_FILE = os.path.expanduser('~/.binance_v25_health.json')
V25_LOCK_FILE = os.path.expanduser('~/.binance_v25_lock')  # 연속 ABORT 시 자동 lock
V25_ABORT_STREAK_LOCK_THRESHOLD = 3  # 3일 연속 ABORT → 자동 lock


def _v25_read_health() -> dict:
    try:
        with open(V25_HEALTH_FILE) as f:
            import json as _j
            return _j.load(f)
    except FileNotFoundError:
        return {}
    except Exception as e:
        log.warning(f"_v25_read_health 실패: {e}")
        return {}


def _v25_write_health(data: dict):
    import json as _j
    try:
        tmp = V25_HEALTH_FILE + '.tmp'
        with open(tmp, 'w') as f:
            _j.dump(data, f, indent=2, ensure_ascii=False)
            f.flush()
            try: os.fsync(f.fileno())
            except Exception: pass
        os.replace(tmp, V25_HEALTH_FILE)
    except Exception as e:
        log.error(f"_v25_write_health 실패: {e}")


def _v25_record_cron_result(success: bool, abort_reason: str = '',
                            intent: Optional[dict] = None,
                            actual: Optional[dict] = None) -> int:
    """cycle 6: cron 완료 시 헬스 상태 갱신.
    Returns: abort_streak (성공시 0).
    """
    from datetime import datetime
    now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    h = _v25_read_health()
    if success:
        h['last_success_at'] = now
        h['abort_streak'] = 0
        h['last_abort_reason'] = ''
        if intent is not None:
            h['last_intent'] = intent
        if actual is not None:
            h['last_actual'] = actual
    else:
        h['abort_streak'] = int(h.get('abort_streak', 0)) + 1
        h['last_abort_at'] = now
        h['last_abort_reason'] = abort_reason[:500]
    _v25_write_health(h)
    return int(h.get('abort_streak', 0))


def _v25_check_lock() -> Optional[str]:
    """lock file 존재 시 그 사유 반환. 없으면 None."""
    try:
        with open(V25_LOCK_FILE) as f:
            return f.read().strip() or 'locked'
    except FileNotFoundError:
        return None
    except Exception:
        return None


def _v25_create_lock(reason: str):
    """V25 cron 자동 정지 — 수동 해제 (`rm ~/.binance_v25_lock`) 필요."""
    try:
        with open(V25_LOCK_FILE, 'w') as f:
            from datetime import datetime
            f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} KST\n{reason}\n")
    except Exception as e:
        log.error(f"_v25_create_lock 실패: {e}")


def _v25_reconcile(intent: dict, actual: dict) -> List[str]:
    """주문 후 reconciliation — 의도 vs 실체결 차이 반환."""
    diffs = []
    intent_pos = intent.get('positions', {})
    actual_pos = actual.get('positions', {})
    for sym in set(intent_pos) | set(actual_pos):
        i = intent_pos.get(sym, {})
        a = actual_pos.get(sym, {})
        if abs(i.get('notional', 0) - a.get('notional', 0)) > max(50.0, 0.05 * max(abs(i.get('notional', 0)), abs(a.get('notional', 0)), 1.0)):
            diffs.append(f"{sym} notional: intent=${i.get('notional', 0):.2f} actual=${a.get('notional', 0):.2f}")
        if i.get('leverage') and a.get('leverage') and abs(float(i['leverage']) - float(a['leverage'])) > 0.01:
            diffs.append(f"{sym} L: intent={i['leverage']} actual={a['leverage']}")
    if intent.get('margin_type') and actual.get('margin_type') and intent['margin_type'].lower() != actual['margin_type'].lower():
        diffs.append(f"margin: intent={intent['margin_type']} actual={actual['margin_type']}")
    return diffs


# V25 Binance error code (cycle 3 P0 fix — 문자열 의존 제거)
_BINANCE_IDEMPOTENT_CODES = {-4046, -4059}  # 'No need to change margin type' / 'No need to change position side'
# V25 cycle 4 P0: -1022 (INVALID_SIGNATURE) 는 transient 아님 — 즉시 ABORT
# V25 cycle 4 P1: -1003 (rate limit/429) transient 제외 — 짧은 재시도 대신 ABORT+알림 (긴 backoff 정책)
# V25 cycle 4 P1: -1000/-1006/-1008 추가 (UNKNOWN/UNEXPECTED_RESP/SERVER_BUSY)
_BINANCE_TRANSIENT_CODES = {-1000, -1001, -1006, -1007, -1008, -1021}
_BINANCE_PRECONDITION_CODES = {-4061, -4068, -4111, -2014, -2015}  # position exist, isolate margin needed, mode dependency, api key invalid


def _is_idempotent_error(e):
    code = getattr(e, 'code', None)
    if code is not None:
        return code in _BINANCE_IDEMPOTENT_CODES
    return 'No need to change' in str(e)  # fallback for older python-binance


def _is_transient_error(e):
    code = getattr(e, 'code', None)
    if code is not None and code in _BINANCE_TRANSIENT_CODES:
        return True
    status = getattr(e, 'status_code', None)
    return status is not None and 500 <= status < 600


def _normalize_dual_side(val):
    """V25 cycle 3 P0: dualSidePosition bool/str/int 안전 정규화."""
    if isinstance(val, bool):
        return val
    if isinstance(val, (int, float)):
        return bool(val)
    if isinstance(val, str):
        return val.strip().lower() in ('true', '1', 'yes')
    return False


def _with_retry(fn, retries=3, base_delay=1.0):
    """V25 cycle 3 P1: transient error 시 bounded retry."""
    import time as _t
    last_exc = None
    for attempt in range(retries):
        try:
            return fn()
        except BinanceAPIException as e:
            last_exc = e
            if _is_transient_error(e) and attempt < retries - 1:
                _t.sleep(base_delay * (2 ** attempt))
                continue
            raise
    if last_exc:
        raise last_exc


def _pick_oneway_row(info_list):
    """V25: futures_position_information 응답에서 oneway 기준 positionSide=BOTH row 선택.

    one-way mode 면 row 1개, positionSide='BOTH'.
    hedge mode 면 row 다수, positionSide='LONG'/'SHORT' — V25 spec 위반. None 반환 → ABORT.
    """
    if not info_list:
        return None
    if len(info_list) == 1:
        side = (info_list[0].get('positionSide') or '').upper()
        if side in ('BOTH', ''):
            return info_list[0]
        return None
    # 다중 row → hedge mode. BOTH 가 있는지 확인 (없으면 None)
    for row in info_list:
        side = (row.get('positionSide') or '').upper()
        if side == 'BOTH':
            return row
    return None


def verify_position_mode_oneway(client: Client) -> bool:
    """V25: 계정 position mode 가 one-way 인지 검증."""
    try:
        mode = _with_retry(lambda: client.futures_get_position_mode())
        # V25 cycle 3 P0 fix: bool/str/int 안전 정규화
        dual = _normalize_dual_side(mode.get('dualSidePosition', False))
        ok = (not dual)
        if DEBUG_MARGIN:
            log.info(f"  position_mode: dualSidePosition={dual} (raw={mode.get('dualSidePosition')!r}) → {'oneway OK' if ok else 'hedge mode'}")
        return ok
    except BinanceAPIException as e:
        log.error(f"verify_position_mode_oneway: {e}")
        return False


def preflight_zero_positions(client: Client, symbols=None) -> bool:
    """V25 cycle 4 P0 + cycle 8: 계정 전체 심볼 zero 검증. futures_account positions 사용 (fresh account 도 정상).
    """
    try:
        acc = _with_retry(lambda: client.futures_account())
        info = acc.get('positions', [])
        non_zero = []
        for row in info:
            amt = abs(float(row.get('positionAmt', 0)))
            if amt > 1e-9:
                non_zero.append((row.get('symbol'), amt))
        if non_zero:
            log.error(f"preflight_zero_positions: 계정 전체 잔존 포지션 {non_zero} — 설정 변경 불가")
            return False
        if DEBUG_MARGIN:
            log.info(f"  preflight_zero_positions OK (account-wide, all zero, n={len(info)} symbols)")
        return True
    except BinanceAPIException as e:
        log.error(f"preflight_zero_positions: {e}")
        return False


def preflight_zero_open_orders(client: Client, symbols=None) -> bool:
    """V25 cycle 4 P0: 계정 전체 미체결 주문 0건 확인 (one-way / margin 변경 시 계정 전역 영향)."""
    try:
        orders = _with_retry(lambda: client.futures_get_open_orders())
        if orders:
            syms = sorted({o.get('symbol') for o in orders})
            log.error(f"preflight_zero_open_orders: 계정 미체결 주문 {len(orders)}건 ({syms})")
            return False
        if DEBUG_MARGIN:
            log.info(f"  preflight_zero_open_orders OK (account-wide, no orders)")
        return True
    except BinanceAPIException as e:
        log.error(f"preflight_zero_open_orders: {e}")
        return False


def preflight_target_symbols_zero(client: Client, symbols) -> bool:
    """V25 cycle 9: 코인별 margin type 변경용 preflight — 대상 심볼만 zero 확인.

    Binance 규칙상 코인별 margin type 변경은 "그 심볼"에 포지션·미체결이 없으면 된다(계정 전역 zero 불필요).
    따라서 무관 심볼의 잔존(특히 DUST_NOTIONAL_LIMIT 미만 먼지 — 봇이 정상 주문으로 청산 불가)은 허용한다.
    단 관측성 유지: 무관 먼지는 warning 으로 남긴다. 대상 심볼은 먼지라도 strict zero 요구(Binance 가 거부하므로).
    Returns: True (대상 심볼 모두 zero), False (대상 심볼에 포지션/주문 잔존 → 호출자 ABORT).
    """
    try:
        target = set(symbols)
        acc = _with_retry(lambda: client.futures_account())
        dust = []
        for row in acc.get('positions', []):
            sym = row.get('symbol')
            amt = abs(float(row.get('positionAmt', 0) or 0))
            if amt <= 1e-9:
                continue
            notional = abs(float(row.get('notional', 0) or 0))
            if sym in target:
                log.error(f"preflight_target_symbols_zero: 대상 심볼 {sym} 잔존 포지션 "
                          f"amt={amt} notional=${notional:.2f} — margin 변경 불가")
                return False
            if 0.0 < notional < DUST_NOTIONAL_LIMIT:
                dust.append((sym, round(notional, 4)))
            else:
                # 무관 심볼의 DUST 이상 포지션 — margin 변경 자체엔 영향 없으나 관측성 위해 기록
                log.warning(f"preflight_target_symbols_zero: 무관 심볼 {sym} 잔존 "
                            f"notional=${notional:.2f} (대상 아님, 허용)")
        if dust:
            log.warning(f"preflight_target_symbols_zero: 무관 심볼 먼지 {len(dust)}개 "
                        f"(합 ${sum(d[1] for d in dust):.2f}) 허용: {dust[:10]}")
        # 대상 심볼 미체결 주문도 0 이어야 margin 변경 가능
        orders = _with_retry(lambda: client.futures_get_open_orders())
        blocking = sorted({o.get('symbol') for o in orders if o.get('symbol') in target})
        if blocking:
            log.error(f"preflight_target_symbols_zero: 대상 심볼 미체결 주문 {blocking} — margin 변경 불가")
            return False
        if DEBUG_MARGIN:
            log.info(f"  preflight_target_symbols_zero OK (targets={sorted(target)} zero, "
                     f"무관 먼지 {len(dust)}개 허용)")
        return True
    except BinanceAPIException as e:
        log.error(f"preflight_target_symbols_zero: {e}")
        return False


def ensure_position_mode_oneway(client: Client) -> bool:
    """V25: one-way mode 검증 + 필요 시 자동 변경."""
    if verify_position_mode_oneway(client):
        return True
    log.info("position_mode = hedge → set one-way 시도")
    try:
        _with_retry(lambda: client.futures_change_position_mode(dualSidePosition=False))
        if DEBUG_MARGIN:
            log.info("  set_position_mode oneway OK")
    except BinanceAPIException as e:
        if _is_idempotent_error(e):
            return True
        log.error(f"set_position_mode oneway FAILED: code={getattr(e,'code',None)} msg={e}")
        return False
    # propagation 지연 대비
    import time as _t
    for _ in range(3):
        if verify_position_mode_oneway(client):
            return True
        _t.sleep(0.5)
    return False


def ensure_margin_type(client: Client, symbol: str, expected: str = 'CROSSED') -> bool:
    """V25: 마진모드 검증 + 필요 시 자동 변경. propagation 지연 대비 짧은 재조회."""
    if verify_margin_type(client, symbol, expected):
        return True
    log.info(f"{symbol} 마진모드 ≠ {expected} → set {expected} 시도")
    if not set_margin_type(client, symbol, expected):
        return False
    # P1: propagation 지연 대비 (Binance 변경 직후 즉시 조회 시 stale 가능)
    import time as _t
    for _ in range(3):
        if verify_margin_type(client, symbol, expected):
            return True
        _t.sleep(0.5)
    log.error(f"ensure_margin_type {symbol}: set 후에도 verify 실패")
    return False


def _fetch_position_info(client: Client, symbol: str = None):
    """V25 cycle 8 P0: futures_account()['positions'] 사용.

    futures_position_information 는 신규 계좌에서 0 rows 반환 → verify 가 false-negative.
    futures_account 는 모든 universe 심볼 1 entry 항상 반환 (positionAmt=0 포함).
    Returns: positions list (positionSide=BOTH row 포함). symbol=None 이면 전체, symbol 지정 시 1 entry list.
    """
    try:
        acc = _with_retry(lambda: client.futures_account())
        positions = acc.get('positions', [])
        if symbol:
            filtered = [p for p in positions if p.get('symbol') == symbol]
            return filtered
        return positions
    except BinanceAPIException as e:
        log.error(f"_fetch_position_info({symbol}): {e}")
        return None


def verify_margin_type(client: Client, symbol: str, expected: str = 'CROSSED') -> bool:
    """V25 cycle 8 P0: marginType 검증 — futures_account positions 사용.
    futures_account row 는 marginType 대신 isolated:bool. False=CROSS, True=ISOLATED.
    """
    info_list = _fetch_position_info(client, symbol)
    if info_list is None:
        return False
    row = _pick_oneway_row(info_list)
    if row is None:
        log.error(f"verify_margin_type {symbol}: oneway row 없음 (hedge mode 또는 빈 응답)")
        return False
    # futures_account: isolated (bool). futures_position_information: marginType (string).
    if 'isolated' in row:
        actual = 'isolated' if bool(row.get('isolated')) else 'cross'
    else:
        actual = (row.get('marginType') or '').lower()
    expected_norm = _MARGIN_NORM.get(expected.lower())
    if expected_norm is None:
        log.error(f"verify_margin_type {symbol}: 알 수 없는 expected={expected}")
        return False
    ok = (actual == expected_norm)
    if DEBUG_MARGIN:
        log.info(f"  verify_margin {symbol}: actual={actual} expected={expected_norm} → {'OK' if ok else 'MISMATCH'}")
    return ok


def verify_leverage(client: Client, symbol: str, expected: int) -> bool:
    """V25 cycle 4 P1: leverage 검증 — 항상 fresh fetch (set 직후 stale 방지)."""
    info_list = _fetch_position_info(client, symbol)
    if info_list is None:
        return False
    row = _pick_oneway_row(info_list)
    if row is None:
        log.error(f"verify_leverage {symbol}: oneway row 없음")
        return False
    actual = int(float(row.get('leverage', 0)))
    ok = (actual == expected)
    if DEBUG_LEVERAGE:
        log.info(f"  verify_leverage {symbol}: actual={actual} expected={expected} → {'OK' if ok else 'MISMATCH'}")
    return ok


def _v25_partial_sell_for_leverage_down(client: Client, sym: str, current_notional: float,
                                        target_notional: float, mark_price: float) -> bool:
    """V25 cycle 8 P1: L 낮춤 전 사전 부분 매도 (reduceOnly).

    new_L < current_L 인 코인은 used_margin 증가로 wallet 부족 (-4131) 위험.
    target_notional_new 까지 미리 줄임 → set_leverage 안전 통과.
    잔여 미세 조정은 execute_rebalance 가 처리.

    Returns: True (성공 또는 매도 불필요), False (실패 → 호출자 ABORT).
    """
    # 안전 버퍼 1% (가격 변동/슬리피지 대비)
    buffered_target = target_notional * 1.01
    if current_notional <= buffered_target + 1e-6:
        if DEBUG_LEVERAGE:
            log.info(f"  {sym} 사전매도 불필요: current={current_notional:.2f} <= target+buf={buffered_target:.2f}")
        return True
    if mark_price <= 0:
        log.error(f"_v25_partial_sell {sym}: mark_price={mark_price} 비정상")
        return False
    sell_notional = current_notional - target_notional
    sell_qty_raw = sell_notional / mark_price
    constraints = get_symbol_constraints(client, sym)
    step = constraints.get('step_size', 0.0) or 0.000001
    min_qty = constraints.get('min_qty', 0.0)
    min_notional = constraints.get('min_notional', MIN_NOTIONAL)
    sell_qty = (sell_qty_raw // step) * step
    if sell_qty < min_qty or sell_qty * mark_price < min_notional:
        log.warning(f"  {sym} 사전매도 skip: qty {sell_qty} 가 min_qty/min_notional 미달")
        return True  # 어차피 차이 작음. ABORT 안 함.
    log.info(f"  {sym} L 낮춤 사전매도: notional ${current_notional:.2f} → ${target_notional:.2f} "
             f"(qty={sell_qty}, mark=${mark_price:.4f})")
    try:
        order = client.futures_create_order(
            symbol=sym, side='SELL', type='MARKET', quantity=sell_qty, reduceOnly=True
        )
        if DEBUG_LEVERAGE:
            log.info(f"  사전매도 OK {sym} orderId={order.get('orderId')}")
        import time as _t
        _t.sleep(0.5)  # 체결 안정화
        return True
    except BinanceAPIException as e:
        log.error(f"  {sym} 사전매도 실패: code={getattr(e,'code',None)} msg={e}")
        return False
    except Exception as e:
        log.error(f"  {sym} 사전매도 예외: {e}")
        return False


def cancel_stop_orders(client: Client, symbols: Optional[List[str]] = None):
    """봇이 관리하는 일반/알고리즘 조건부 주문 정리."""
    symbol_set = set(symbols or UNIVERSE)

    # 1) 일반 open orders 에 노출되는 조건부 주문 정리
    try:
        orders = client.futures_get_open_orders()
    except Exception as e:
        log.warning(f"open_orders fetch failed: {e}")
    else:
        for order in orders:
            symbol = order.get('symbol')
            if symbol not in symbol_set:
                continue
            order_type = str(order.get('type') or '')
            is_close_position = str(order.get('closePosition')).lower() == 'true'
            is_conditional = order_type in {'STOP_MARKET', 'TAKE_PROFIT_MARKET', 'STOP', 'TAKE_PROFIT'}
            if not (is_close_position or is_conditional):
                continue
            try:
                client.futures_cancel_order(symbol=symbol, orderId=order['orderId'])
                log.info(
                    f"CANCEL CONDITIONAL {symbol} orderId={order['orderId']} "
                    f"type={order_type} closePosition={order.get('closePosition')}"
                )
            except Exception as e:
                log.warning(f"cancel conditional {symbol} orderId={order.get('orderId')}: {e}")

    # 2) 별도 algo/conditional 저장소에 있는 조건부 주문 정리
    try:
        algo_orders = client.futures_get_open_algo_orders()
    except Exception as e:
        log.warning(f"open_algo_orders fetch failed: {e}")
    else:
        for order in algo_orders:
            symbol = order.get('symbol')
            if symbol not in symbol_set:
                continue
            algo_id = order.get('algoId')
            if not algo_id:
                continue
            try:
                client.futures_cancel_algo_order(symbol=symbol, algoId=algo_id)
                log.info(
                    f"CANCEL ALGO {symbol} algoId={algo_id} "
                    f"reduceOnly={order.get('reduceOnly')} closePosition={order.get('closePosition')}"
                )
            except Exception as e:
                log.warning(f"cancel algo {symbol} algoId={algo_id}: {e}")


def force_cancel_all_orders(client: Client, symbols: Optional[List[str]] = None) -> bool:
    """심볼별 모든 열린 주문 강제 정리. V25 cycle 5 C: 실패 시 False 반환 → 호출자 ABORT.

    Binance가 조건부 closePosition 주문을 open_orders 목록에 노출하지 않는 경우가 있어,
    -4130 충돌 회피용으로 사용한다.
    """
    symbol_set = set(symbols or UNIVERSE)
    failures = []
    for symbol in symbol_set:
        try:
            client.futures_cancel_all_open_orders(symbol=symbol)
            log.info(f"CANCEL ALL OPEN ORDERS {symbol}")
        except Exception as e:
            log.warning(f"cancel all open orders {symbol}: {e}")
            failures.append((symbol, 'open', str(e)))
        try:
            client.futures_cancel_all_algo_open_orders(symbol=symbol)
            log.info(f"CANCEL ALL ALGO OPEN ORDERS {symbol}")
        except Exception as e:
            log.warning(f"cancel all algo open orders {symbol}: {e}")
            failures.append((symbol, 'algo', str(e)))
    if failures:
        log.error(f"force_cancel_all_orders: {len(failures)} 건 실패 — {failures[:5]}")
        return False
    return True


def count_stop_orders(client: Client, symbols: Optional[List[str]] = None) -> int:
    """현재 봇 대상 심볼의 활성 조건부 스탑 주문 개수."""
    symbol_set = set(symbols or UNIVERSE)
    count = 0

    try:
        orders = client.futures_get_open_orders()
    except Exception as e:
        log.warning(f"open_orders count failed: {e}")
    else:
        for order in orders:
            symbol = order.get('symbol')
            if symbol not in symbol_set:
                continue
            order_type = str(order.get('type') or '')
            is_close_position = str(order.get('closePosition')).lower() == 'true'
            is_conditional = order_type in {'STOP_MARKET', 'TAKE_PROFIT_MARKET', 'STOP', 'TAKE_PROFIT'}
            if is_close_position or is_conditional:
                count += 1

    try:
        algo_orders = client.futures_get_open_algo_orders()
    except Exception as e:
        log.warning(f"open_algo_orders count failed: {e}")
    else:
        for order in algo_orders:
            if order.get('symbol') in symbol_set:
                count += 1

    return count


def sync_stop_orders(client: Client, positions: Dict[str, dict], data_1h: Dict[str, pd.DataFrame],
                     target: Dict[str, float], order_alerts: Optional[List[str]] = None,
                     error_alerts: Optional[List[str]] = None):
    """V22/V24 (가드 완전 제거): 스탑 주문 로직 없음.

    이전 배포에서 남았을 수 있는 잔존 스탑을 정리하고 즉시 반환.
    현재 UNIVERSE 밖 심볼에도 stale 스탑이 있을 수 있으므로
    현재 포지션 심볼 + UNIVERSE 합집합을 대상으로 정리한다.
    앙상블 분산만으로 방어한다.
    """
    cleanup_symbols = set(UNIVERSE)
    for pos in (positions or {}).values():
        sym = pos.get('symbol')
        if sym:
            cleanup_symbols.add(sym)
    cancel_stop_orders(client, sorted(cleanup_symbols))
    log.info("STOP OFF (V24: 가드 완전 제거, 잔존 스탑 정리 %d심볼)", len(cleanup_symbols))


# ─── SQLite 자산 기록 ───
DB_PATH = os.path.join(SCRIPT_DIR, 'binance_history.db')

def _record_equity(pv: float, positions: dict, target: dict):
    """매 실행마다 SQLite에 자산 기록."""
    import sqlite3
    try:
        conn = sqlite3.connect(DB_PATH)
        conn.execute('''CREATE TABLE IF NOT EXISTS equity_history (
            ts TEXT PRIMARY KEY, pv REAL, n_positions INTEGER, target TEXT)''')
        conn.execute('''CREATE TABLE IF NOT EXISTS position_history (
            ts TEXT, coin TEXT, notional REAL, weight REAL)''')
        now = datetime.now(timezone.utc).isoformat()
        conn.execute('INSERT OR REPLACE INTO equity_history VALUES (?,?,?,?)',
                      (now, pv, len(positions), json.dumps(target)))
        for coin, pos in positions.items():
            conn.execute('INSERT INTO position_history VALUES (?,?,?,?)',
                          (now, coin, pos.get('real_notional', pos['notional']), pos.get('real_weight', pos['weight'])))
        conn.commit()
        conn.close()
    except Exception as e:
        log.warning(f"SQLite record error: {e}")


# ─── 메인 ───
def main():
    parser = argparse.ArgumentParser(description='바이낸스 선물 자동매매')
    parser.add_argument('--trade', action='store_true', help='실제 매매 실행')
    parser.add_argument('--dry-run', action='store_true', help='시뮬레이션 (매매 안 함)')
    parser.add_argument('--status', action='store_true', help='현재 상태 조회')
    parser.add_argument('--report', action='store_true', help='일일 리포트 (텔레그램)')

    args = parser.parse_args()

    # dry-run silent: --trade 없으면 텔레그램 silent (테스트용)
    _DRY_RUN_SILENT[0] = (not getattr(args, 'trade', False))

    api_key, api_secret = load_config()
    if not api_key:
        log.error("API key not configured")
        return

    client = Client(api_key, api_secret)
    state = load_state()

    # cash_buffer: V24 (2026-05-26): fut_cash_buffer 키 우선, 없으면 legacy cash_buffer
    global CASH_BUFFER
    CASH_BUFFER = float(state.get('fut_cash_buffer', state.get('cash_buffer', CASH_BUFFER_DEFAULT)))
    log.info(f"cash_buffer (fut): {CASH_BUFFER:.0%}")

    if args.status:
        positions, pv, ok = get_current_positions(client)
        if not ok:
            print("포지션 조회 실패")
            return
        print(f"총 자산: ${pv:.2f}")
        print(f"포지션: {positions}")
        print(f"마지막 실행: {state.get('last_run', 'N/A')}")
        return

    if args.report:
        positions, pv, ok = get_current_positions(client)
        if not ok:
            log.error("리포트 생성 중 포지션 조회 실패")
            return
        refresh_universe(client)
        data = fetch_all_data(client)
        initial = state.get('initial_capital', pv)
        if 'initial_capital' not in state:
            state['initial_capital'] = pv
            save_state(state)
        pnl_pct = (pv / initial - 1) * 100 if initial > 0 else 0
        now = datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')

        lines = [f"📊 바이낸스 선물 일일 리포트 ({now})"]
        lines.append(f"총 자산: ${pv:.2f} ({pnl_pct:+.1f}%)")
        lines.append(f"레버리지: 고정 3x (V24 1D 단일 D_SMA42 sn=95 n=5 drift=0.03)")

        if positions:
            lines.append("\n포지션 (실질금액, notional/lev):")
            for coin, pos in positions.items():
                if pos.get('notional', 0.0) < DISPLAY_DUST_NOTIONAL:
                    continue
                pnl = pos.get('pnl', 0.0)
                real_n = pos.get('real_notional', pos['notional'])
                real_w = pos.get('real_weight', pos['weight'])
                lev = pos.get('leverage', 1.0)
                lines.append(f"  {coin}: ${real_n:.0f} ({real_w:.1%}, {lev:.0f}x, PnL {pnl:+.2f})")
        else:
            lines.append("포지션: 없음 (현금)")

        stop_count = count_stop_orders(client, list(UNIVERSE))
        visible_positions = sum(1 for pos in positions.values() if pos.get('notional', 0.0) >= DISPLAY_DUST_NOTIONAL)
        last_run = state.get('last_run')
        last_run_str = last_run if not last_run else datetime.fromisoformat(last_run).astimezone(timezone(timedelta(hours=9))).strftime('%Y-%m-%d %H:%M:%S KST')
        lines.append("\n헬스:")
        lines.append(f"  마지막 실행: {last_run_str or '없음'}")
        lines.append(f"  리밸런싱 대기: {'예' if state.get('rebalancing_needed', False) else '아니오'}")
        lines.append(f"  포지션 수: {visible_positions}")
        lines.append(f"  활성 스탑 수: {stop_count}")

        # 전략 상태는 state에 저장된 실제 실행 결과를 보여준다.
        strategies_state = state.get('strategies', {})
        last_target = state.get('last_target', {})
        for sname in STRATEGIES:
            st = strategies_state.get(sname, {})
            canary = "ON" if st.get('canary_on', False) else "OFF"
            combined = st.get('last_combined', {})
            lines.append(f"{sname} 카나리: {canary}")
            if combined and combined.get('CASH', 0.0) < 0.999:
                coins = ', '.join(f"{k}:{v:.1%}" for k, v in combined.items() if k != 'CASH' and v > 0)
                cash_w = combined.get('CASH', 0.0)
                if cash_w > 0:
                    coins = f"{coins}, CASH:{cash_w:.1%}"
                lines.append(f"  목표: {coins}")
            else:
                lines.append("  목표: CASH")

        lines.append(f"\n마지막 매매: {state.get('last_run', 'N/A')}")
        msg = "\n".join(lines)
        print(msg)
        send_telegram(msg)

        # 자산 기록 (히스토리)
        history = state.get('daily_history', [])
        history.append({'date': now, 'pv': pv, 'positions': len(positions)})
        state['daily_history'] = history[-90:]  # 최근 90일만 보관
        save_state(state)
        return

    # 파일 락 (동시 실행 방지)
    import fcntl
    lock_path = STATE_PATH + '.lock'
    lock_fd = open(lock_path, 'w')
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except IOError:
        log.warning("다른 인스턴스 실행 중, 종료")
        return

    try:
        t_start = time.time()
        run_id = datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')
        canary_alerts: List[str] = []
        order_alerts: List[str] = []
        error_alerts: List[str] = []
        log.info(f"=== 바이낸스 선물 매매 시작 (run_id={run_id}) ===")

        if args.trade:
            jitter = random.randint(*CRON_START_JITTER_SECONDS)
            log.info(f"시작 지연: {jitter}s (크론 동시충돌 완화)")
            time.sleep(jitter)

        # V25 cycle 6: lock 파일 확인 (3일 연속 ABORT 시 자동 lock — 수동 해제 필요)
        lock_reason = _v25_check_lock()
        if lock_reason:
            err = f"V25 ABORT: lock 활성 — {lock_reason}. 수동 해제 필요 (rm {V25_LOCK_FILE})"
            log.error(err)
            send_telegram(f"🔒 {err}")
            _v25_persist_abort_log(err)
            return

        # 0. 유니버스 갱신 (CoinGecko top40 ∩ 바이낸스 USDT-M 선물 listing)
        refresh_universe(client)

        # 1. 데이터 수집
        log.info("데이터 수집...")
        data = fetch_all_data(client)
        n_1h = len(data['1h'])
        n_d = len(data['D'])
        log.info(f"수집 완료: 1h {n_1h}개, D {n_d}개 ({time.time()-t_start:.1f}s)")

        # 데이터 장애 방어
        if 'BTC' not in data['1h'] or 'BTC' not in data['D']:
            log.error("BTC 데이터 누락! 매매 중단. 이전 포지션 유지.")
            send_telegram("⚠️ BTC 데이터 누락 — 매매 중단")
            return
        if n_1h < len(UNIVERSE) // 2 or n_d < len(UNIVERSE) // 2:
            log.error(f"데이터 부족 ({n_1h}/{n_d}). 매매 중단.")
            send_telegram(f"⚠️ 데이터 부족 ({n_1h}/{n_d}) — 매매 중단")
            return

        # 2. 현재 포지션 (리밸런싱 전)
        positions_before, pv_before, pos_ok = get_current_positions(client)
        if not pos_ok:
            log.error("현재 포지션/PV 조회 실패. 이번 실행은 거래 없이 스킵.")
            send_telegram("⚠️ 바이낸스 포지션 조회 실패 — 이번 크론은 거래를 건너뜁니다.")
            return
        log.info(f"현재 PV: ${pv_before:.2f}")
        if positions_before:
            for coin, pos in positions_before.items():
                log.info(f"  보유: {coin} ${pos['notional']:.0f} ({pos['weight']:.1%})")

        # 3. 각 전략 시그널 계산
        targets = {}
        for strat_name, params in STRATEGIES.items():
            target = compute_strategy_target(strat_name, params, data, state, alerts=canary_alerts)
            targets[strat_name] = target
            # 종목 비중만 로깅 (CASH 제외)
            coins_only = {k: f"{v:.1%}" for k, v in target.items() if k != 'CASH' and v > 0}
            cash_pct = target.get('CASH', 0)
            log.info(f"  {strat_name} → {coins_only or 'CASH 100%'} (cash={cash_pct:.0%})")

        # 4. 앙상블 합산
        combined = combine_ensemble(targets, ENSEMBLE_WEIGHTS)
        coins_combined = {k: f"{v:.1%}" for k, v in combined.items() if k != 'CASH' and v > 0}
        log.info(f"합산: {coins_combined or 'CASH 100%'}")

        # V24 drift 트리거: cur_w (자본금 기준 비중, real_weight 사용) vs target_w
        # half_turnover >= DRIFT_THRESHOLD_FUT(0.05) 이면 rebalancing_needed=True
        # alloc_transit cap (옵션 D): real_weight 분모를 effective_pv 로 scale
        _drift_cap_ratio = _read_alloc_transit_cap_ratio_fut()
        _drift_scale = (1.0 / _drift_cap_ratio) if (_drift_cap_ratio is not None and 0 < _drift_cap_ratio < 1.0) else 1.0
        cur_w_fut: Dict[str, float] = {}
        for coin, pos in positions_before.items():
            cur_w_fut[coin] = float(pos.get('real_weight', 0.0)) * _drift_scale
        cash_w = max(0.0, 1.0 - sum(cur_w_fut.values()))
        cur_w_fut['CASH'] = cash_w
        # cash buffer 반영 — combined 은 risky-asset 100% 정규화, 실매매는 fut_cash_buffer KRW 유지
        try:
            _fut_buf = float(state.get('fut_cash_buffer', state.get('cash_buffer', CASH_BUFFER)))
        except (TypeError, ValueError):
            _fut_buf = CASH_BUFFER
        # canary OFF (CASH 100%) 시 buffer 적용 skip
        _cash_in_tgt = sum(float(v) for k, v in combined.items() if k.upper() == 'CASH')
        tgt_w_norm = {}
        if _fut_buf > 0 and _cash_in_tgt < 0.99:
            _has_cash_in_tgt = any(k.upper() == 'CASH' for k in combined)
            for k, v in combined.items():
                if k.upper() == 'CASH':
                    tgt_w_norm[k] = float(v)
                else:
                    tgt_w_norm[k] = float(v) * (1.0 - _fut_buf)
            if not _has_cash_in_tgt:
                tgt_w_norm['CASH'] = _fut_buf
        else:
            tgt_w_norm = {k: float(v) for k, v in combined.items()}
        _all_keys = set(cur_w_fut) | set(tgt_w_norm)
        ht_fut = sum(abs(tgt_w_norm.get(k, 0.0) - cur_w_fut.get(k, 0.0)) for k in _all_keys) / 2
        drift_fire_fut = ht_fut >= DRIFT_THRESHOLD_FUT
        if not DRIFT_ENABLED_FUT:
            drift_fire_fut = False  # snap-only fallback
        log.info(f"V24 drift eval: ht={ht_fut:.4f} threshold={DRIFT_THRESHOLD_FUT:.2f} fire={drift_fire_fut} enabled={DRIFT_ENABLED_FUT}")
        if drift_fire_fut and not state.get('rebalancing_needed', False):
            state['rebalancing_needed'] = True
            state['last_rebal_reason'] = 'drift'
            log.info(f"  🔔 V24 drift 발화 (silent) → rebalancing_needed=True. ht={ht_fut:.4f} >= {DRIFT_THRESHOLD_FUT:.2f}")

        # V24 refill v2 — drift fire 시 mom2 음수 슬롯 교체 후 combined 재계산
        if drift_fire_fut and REFILL_ENABLED_FUT:
            try:
                combined_before = dict(combined)
                combined = apply_refill_v2_fut(state, data)
                coins_after = {k: f"{v:.1%}" for k, v in combined.items() if k != 'CASH' and v > 0}
                all_keys = set(combined_before) | set(combined)
                diffs = sorted(
                    [(k, combined.get(k, 0) - combined_before.get(k, 0)) for k in all_keys],
                    key=lambda x: -abs(x[1])
                )[:5]
                top_diffs = [(k, f'{d*100:+.1f}%') for k, d in diffs if abs(d) > 1e-6]
                log.info(f"  🔁 V24 refill v2 적용: {coins_after or 'CASH 100%'} | top diffs={top_diffs}")
            except Exception as e:
                log.warning(f"refill v2 skipped: {e}", exc_info=True)

        # 옵션 Z: cap_defend trigger (drift 와 별개)
        if _drift_cap_ratio is not None and _drift_cap_ratio < (1.0 - CAP_DEFEND_MIN_EXCESS):
            if not state.get('rebalancing_needed', False):
                state['rebalancing_needed'] = True
                state['last_rebal_reason'] = 'cap_defend'
                log.info(f"  🛡️ cap_defend trigger: cap_ratio={_drift_cap_ratio:.4f} < {1.0-CAP_DEFEND_MIN_EXCESS:.2f} → rebal 강제")
            # V24: drift 알림 silent — Daily Report 09:15 가 통합 보고
        # schema_version 마크
        state['schema_version'] = SCHEMA_VERSION

        # 5. 리밸런싱
        # V25: data['D'] 필수 — K2 per-coin SMA(7) + BTC_cap SMA(42) 모두 1D 기반.
        # data['D'] 누락 또는 BTC 1D 길이 부족 시 abort (silent L2 fallback 차단)
        if 'D' not in data or 'BTC' not in data['D'] or data['D']['BTC'].empty:
            err = "V25 ABORT: data['D'] 또는 BTC 1D 누락 — K2/BTC_cap 평가 불가"
            log.error(err)
            error_alerts.append(err)
            target_lev_map = {}
        elif len(data['D']['BTC']) < BTC_CAP_SMA_PERIOD + 2:
            err = f"V25 ABORT: BTC 1D 길이 {len(data['D']['BTC'])} < SMA42+2"
            log.error(err)
            error_alerts.append(err)
            target_lev_map = {}
        else:
            try:
                target_lev_map = get_coin_leverage_map(combined, data['1h'], data['D'])
            except StaleBarError as e:
                err = f"V25 ABORT: 봉 정합성 실패 (StaleBarError) — {e}. 매매 차단, 다음 cron 재시도."
                log.error(err); error_alerts.append(err)
                _v25_persist_abort_log(err)
                send_telegram(f"⚠ {err}")
                target_lev_map = {}
        rebalance_needed = state.get('rebalancing_needed', False)
        v25_abort = False  # 버그 수정 2026-06-06: 비매매 경로에서도 항상 정의 (요약부 v25_success 참조)
        if not rebalance_needed:
            log.info("매매 스킵: rebalancing_needed=false")
            positions_after, pv_after = positions_before, pv_before
        elif not target_lev_map:
            # V25: leverage map 산출 실패 → 매매 안 함 (안전)
            err_msg = "V25 ABORT: target_lev_map 비어있음 — 매매 차단"
            log.error(err_msg)
            error_alerts.append(err_msg)
            positions_after, pv_after = positions_before, pv_before
        elif args.trade:
            # V25 정책: all-or-nothing — 한 코인이라도 verify 실패 시 매매 전체 ABORT
            # 부분 universe 매매 = BT 가정 (3 코인 균등) 위반. 다음 cron 에서 재시도.
            #
            # 순서: oneway 모드 검증 → 각 코인 margin → verify → leverage → verify → (모두 통과 시)
            #       force_cancel → execute_rebalance
            v25_abort = False
            target_symbols = [coin + 'USDT' for coin in target_lev_map.keys()]
            # V25 cycle 9 (ai-debate 20260715 보완): 설정변경 preflight 를 두 종류로 분리.
            #   need_mode_change: one-way/hedge 모드 변경 → Binance 계정단위 규칙 → 계정 전체 zero 엄격(먼지도 불가).
            #   need_margin_change: 코인별 margin type 변경 → 해당 심볼만 zero 면 됨 → 무관 심볼 먼지는 tolerate.
            # (기존엔 둘을 need_setup_change 로 묶어 계정전체 zero 를 강제 → 무관한 SOL 먼지에 margin 변경이 막혀 오ABORT.)
            need_mode_change = False
            need_margin_change = False
            try:
                cur_mode = client.futures_get_position_mode()
                if _normalize_dual_side(cur_mode.get('dualSidePosition', False)):
                    need_mode_change = True
            except BinanceAPIException:
                need_mode_change = True
            for sym in target_symbols:
                info = _fetch_position_info(client, sym)
                if info is None:
                    need_margin_change = True; break
                row = _pick_oneway_row(info)
                if row is None:
                    need_margin_change = True; break
                # cycle 8: futures_account 는 isolated(bool), futures_position_information 은 marginType(str)
                if 'isolated' in row:
                    if bool(row.get('isolated')):
                        need_margin_change = True; break
                elif (row.get('marginType') or '').lower() != 'cross':
                    need_margin_change = True; break
            # (a) one-way 모드 변경: 계정 전체 zero 엄격 (먼지 불허 — 계정단위 변경)
            if need_mode_change:
                if not preflight_zero_positions(client):
                    err = "V25 ABORT: one-way 모드 변경 필요한데 계정에 잔존 포지션 있음 — 매매 차단"
                    log.error(err); error_alerts.append(err); v25_abort = True; _v25_persist_abort_log(err)
                elif not preflight_zero_open_orders(client):
                    err = "V25 ABORT: one-way 모드 변경 필요한데 계정에 미체결 주문 있음 — 매매 차단"
                    log.error(err); error_alerts.append(err); v25_abort = True; _v25_persist_abort_log(err)
            # (b) 코인별 margin 변경: 대상 심볼만 zero 확인 (무관 심볼 먼지 tolerate)
            if not v25_abort and need_margin_change:
                if not preflight_target_symbols_zero(client, target_symbols):
                    err = "V25 ABORT: margin 변경 대상 코인에 잔존 포지션/주문 있음 — 매매 차단"
                    log.error(err); error_alerts.append(err); v25_abort = True; _v25_persist_abort_log(err)
            if not v25_abort:
                # one-way mode 자동 보장 (포지션 없을 때만)
                if not ensure_position_mode_oneway(client):
                    err = "V25 ABORT: one-way mode 보장 실패 — 매매 차단"
                    log.error(err); error_alerts.append(err); v25_abort = True; _v25_persist_abort_log(err)
            if not v25_abort:
                # cycle 8 P1: L 낮춤 전 사전 부분매도 — set_leverage 가 used_margin 증가로 -4131 거부하는 케이스 방지
                # 룰: new_L < current_L 이면 비율(new_L/current_L) 만큼 reduceOnly 매도 → set_leverage → 잔여 미세조정은 execute_rebalance
                # L 올림은 그대로 두고 execute_rebalance 가 매수 처리
                for coin, new_lev in target_lev_map.items():
                    sym = coin + 'USDT'
                    info = _fetch_position_info(client, sym)
                    if info is None:
                        continue
                    row = _pick_oneway_row(info)
                    if row is None:
                        continue
                    try:
                        pos_amt = abs(float(row.get('positionAmt', 0) or 0))
                        cur_lev = int(float(row.get('leverage', 0) or 0))
                    except (TypeError, ValueError):
                        continue
                    if pos_amt <= 0 or cur_lev <= 0 or new_lev >= cur_lev:
                        continue
                    # mark_price: futures_account row 의 entryPrice/markPrice 또는 ticker fallback
                    mark = 0.0
                    for k in ('markPrice', 'entryPrice'):
                        v = row.get(k)
                        if v:
                            try:
                                mark = float(v)
                                if mark > 0: break
                            except (TypeError, ValueError):
                                pass
                    if mark <= 0:
                        try:
                            mark = float(_with_retry(lambda: client.futures_mark_price(symbol=sym))['markPrice'])
                        except Exception as e:
                            log.error(f"V25 ABORT: {sym} mark_price 조회 실패: {e}")
                            v25_abort = True; _v25_persist_abort_log(f"V25 ABORT: {sym} mark_price 조회 실패"); break
                    current_notional = pos_amt * mark
                    target_notional = current_notional * (new_lev / cur_lev)
                    log.info(f"V25 L↓ {sym}: {cur_lev}x → {new_lev}x, 사전매도 notional ${current_notional:.2f} → ${target_notional:.2f}")
                    if not _v25_partial_sell_for_leverage_down(client, sym, current_notional, target_notional, mark):
                        err = f"V25 ABORT: {sym} L 낮춤 사전매도 실패 — 매매 차단"
                        log.error(err); error_alerts.append(err); v25_abort = True; _v25_persist_abort_log(err); break
            if not v25_abort:
                # 각 코인: 마진모드 자동 보장 → leverage 동적 변경
                for coin, lev in target_lev_map.items():
                    sym = coin + 'USDT'
                    if DEBUG_LEVERAGE:
                        log.info(f"V25 prep {sym}: ensure margin={MARGIN_TYPE}, leverage={lev}x")
                    if not ensure_margin_type(client, sym, MARGIN_TYPE):
                        err = f"V25 ABORT: {sym} 마진모드 {MARGIN_TYPE} 보장 실패 — 매매 차단"
                        log.error(err); error_alerts.append(err); v25_abort = True; _v25_persist_abort_log(err); break
                    if not set_leverage(client, sym, lev):
                        err = f"V25 ABORT: set_leverage({sym}={lev}) 실패 — 매매 차단"
                        log.error(err); error_alerts.append(err); v25_abort = True; _v25_persist_abort_log(err); break
                    # P1: propagation 짧은 재조회
                    import time as _t
                    lev_ok = False
                    for _ in range(3):
                        if verify_leverage(client, sym, lev):
                            lev_ok = True; break
                        _t.sleep(0.3)
                    if not lev_ok:
                        err = f"V25 ABORT: verify_leverage({sym}) ≠ {lev}x after retry — 매매 차단"
                        log.error(err); error_alerts.append(err); v25_abort = True; _v25_persist_abort_log(err); break
            if v25_abort:
                positions_after, pv_after = positions_before, pv_before
            else:
                # 모든 검증 통과 → 사전 주문 정리 후 매매 (cycle 2 fix: cancel 을 검증 후로 이동)
                prep_symbols = sorted({
                    *(coin + 'USDT' for coin in target_lev_map.keys()),
                    *(pos['symbol'] for pos in positions_before.values()),
                })
                # cycle 5 C: cancel 실패 시 ABORT (잔존 주문이 신규 주문과 충돌 위험)
                if not force_cancel_all_orders(client, prep_symbols):
                    err = "V25 ABORT: force_cancel_all_orders 실패 — 잔존 주문이 신규 주문과 충돌 위험, 매매 차단"
                    log.error(err); error_alerts.append(err); v25_abort = True
                    _v25_persist_abort_log(err)
            if not v25_abort:
                # alloc_transit cap (옵션 D, 2026-05-23): effective_pv = pv_before × cap_ratio (< 1.0)
                _fut_cap_ratio = _read_alloc_transit_cap_ratio_fut()
                if _fut_cap_ratio is not None and _fut_cap_ratio < 1.0:
                    _effective_pv = pv_before * _fut_cap_ratio
                    log.info(f"🔴 alloc_transit cap_ratio={_fut_cap_ratio:.3f} → pv ${pv_before:,.2f} → ${_effective_pv:,.2f}")
                else:
                    _effective_pv = pv_before
                execute_rebalance(
                    client, combined, _effective_pv, target_lev_map,
                    order_alerts=order_alerts, error_alerts=error_alerts,
                )

            # 리밸런싱 후 포지션
            positions_after, pv_after, pos_after_ok = get_current_positions(client)
            if not pos_after_ok:
                log.error("리밸런싱 후 포지션 조회 실패. kill-switch/미달 판정 없이 종료.")
                send_telegram("⚠️ 바이낸스 포지션 조회 실패 — 리밸런싱 후 상태 확인을 건너뜁니다.")
                return
            log.info(f"리밸런싱 완료: PV ${pv_before:.2f} → ${pv_after:.2f}")
            if positions_after:
                for coin, pos in positions_after.items():
                    pnl = pos.get('pnl', 0.0)
                    log.info(f"  보유: {coin} ${pos['notional']:.0f} ({pos['weight']:.1%}, PnL {pnl:+.2f})")
            # 이벤트가 발생해서 시작한 리밸런싱은 목표에 근접하면 종료, 아니면 다음 실행에서 재시도.
            if needs_rebalance(client, combined, positions_after, pv_after, target_lev_map):
                state['rebalancing_needed'] = True
                log.info("  ⏳ 미달, rebalancing_needed 유지")
            else:
                state['rebalancing_needed'] = False
                log.info("  ✅ 목표 달성, rebalancing_needed=false")
        else:
            log.info(f"DRY-RUN REBALANCE: {coins_combined or 'CASH'}")
            positions_after, pv_after = positions_before, pv_before

        if args.trade:
            # 매 실거래 실행마다 직전 완성봉 기준으로 스탑을 재동기화한다.
            sync_stop_orders(
                client, positions_after, data['1h'], combined,
                order_alerts=order_alerts, error_alerts=error_alerts,
            )

        # 6. 상태 저장
        if args.trade:
            state['last_target'] = combined
        state['prev_pv'] = pv_after
        state['kill_switch'] = False
        state.pop('kill_switch_reason', None)
        state.pop('kill_switch_time', None)
        state['last_run'] = datetime.now(timezone.utc).isoformat()
        save_state(state)

        # 7. SQLite 자산 기록
        _record_equity(pv_after, positions_after, combined)

        elapsed = time.time() - t_start
        if args.trade:
            # V24 통일 보고
            import sys as _sys, os as _os
            _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
            import v24_report as v24r

            kst_now = datetime.now(timezone.utc).astimezone(timezone(timedelta(hours=9)))
            target_norm = {('Cash' if k == 'CASH' else k): v
                           for k, v in (combined or {}).items() if v > 1e-4}

            holdings = []
            for coin, pos in positions_after.items():
                if pos.get('notional', 0.0) < DISPLAY_DUST_NOTIONAL:
                    continue
                margin = pos.get('real_notional', pos['notional'])
                holdings.append({
                    'ticker': coin,
                    'value_str': f"${margin:.0f}",
                    'weight': pos.get('real_weight', pos.get('weight', 0.0)),
                })

            if order_alerts:
                orders_text = '\n  - '.join([''] + list(order_alerts)).lstrip('\n').lstrip()
                orders_text = '\n  - ' + '\n  - '.join(order_alerts)
            else:
                orders_text = '없음'

            canary_lines = []
            for strat_name in STRATEGIES:
                ss = state.get('strategies', {}).get(strat_name, {})
                ci = ss.get('canary_info', {})
                if ci:
                    on = ci.get('on', False)
                    flip_mark = ' *FLIP*' if ci.get('flipped') else ''
                    canary_lines.append(
                        f"{strat_name}: {'ON 🟢' if on else 'OFF 🔴'} "
                        f"BTC ${ci.get('cur', 0):,.0f} vs SMA{ci.get('sma_p', 0)} "
                        f"${ci.get('sma_val', 0):,.0f} ratio={ci.get('ratio', 0):.4f}{flip_mark}"
                    )
                else:
                    canary_lines.append(f"{strat_name}: {'ON 🟢' if ss.get('canary_on') else 'OFF 🔴'}")

            stop_count = count_stop_orders(client, list(UNIVERSE))
            visible_positions = sum(1 for pos in positions_after.values()
                                     if pos.get('notional', 0.0) >= DISPLAY_DUST_NOTIONAL)
            status = {
                'schema': 'V24',
                '평가액': f'${pv_after:.2f}',
                'ht': f'{ht_fut:.4f}',
                'drift_threshold': f'{DRIFT_THRESHOLD_FUT:.2f}',
                'drift_fire': '예 🔔' if drift_fire_fut else '아니오',
                '리밸 대기': '예' if state.get('rebalancing_needed', False) else '아니오',
                '포지션 수': str(visible_positions),
                '활성 스탑 수': str(stop_count),
            }
            extra = None
            if error_alerts:
                extra = "⚠ 오류\n" + "\n".join(f"  - {msg}" for msg in error_alerts[:10])

            # V25 cycle 6: cron 결과 헬스 기록 + reconciliation
            v25_success = not (v25_abort or error_alerts)
            intent_snapshot = {
                'positions': {coin + 'USDT': {
                    'notional': pv_before * combined.get(coin, 0.0) * target_lev_map.get(coin, 1),
                    'leverage': target_lev_map.get(coin, 0),
                } for coin in target_lev_map.keys()},
                'margin_type': MARGIN_TYPE,
                'targets': dict(target_norm),
            }
            actual_snapshot = {
                'positions': {pos['symbol']: {
                    'notional': pos.get('notional', 0.0),
                    'leverage': pos.get('leverage', 0),
                } for pos in positions_after.values()},
                'pv': pv_after,
            }
            recon_diffs = _v25_reconcile(intent_snapshot, actual_snapshot) if v25_success else []
            if recon_diffs:
                recon_msg = "⚠ V25 reconciliation 차이:\n  - " + "\n  - ".join(recon_diffs[:5])
                log.warning(recon_msg)
                send_telegram(recon_msg)
                _v25_persist_abort_log(recon_msg)
                v25_success = False

            streak = _v25_record_cron_result(
                success=v25_success,
                abort_reason="; ".join(error_alerts[:3]) if error_alerts else ('reconcile diff' if recon_diffs else ''),
                intent=intent_snapshot,
                actual=actual_snapshot,
            )
            if not v25_success:
                if streak == 2:
                    send_telegram(f"⚠ V25 ABORT 2일 연속 — 다음 ABORT 시 자동 lock")
                elif streak >= V25_ABORT_STREAK_LOCK_THRESHOLD:
                    _v25_create_lock(f"abort_streak={streak} reached threshold")
                    send_telegram(f"🔒 V25 ABORT {streak}일 연속 — 자동 lock. 수동 해제: rm {V25_LOCK_FILE}")

            # V24: 정상 보고 silent — Daily Report 09:15 가 통합 보고. 오류만 즉시 알림.
            if error_alerts:
                send_telegram("⚠ 선물 오류\n" + "\n".join(f"  - {msg}" for msg in error_alerts[:10]))
            log.info(f"V25 fut: targets={target_norm} ht={ht_fut:.4f} fire={drift_fire_fut} pv=${pv_after:.2f} "
                     f"success={v25_success} streak={streak}")
        log.info(f"=== 완료 ({elapsed:.1f}s) ===")
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


if __name__ == '__main__':
    main()
