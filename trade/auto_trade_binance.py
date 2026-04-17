#!/usr/bin/env python3
"""
바이낸스 선물 자동매매 — V21 L3 앙상블 (ENS_fut_L3_k3_12652d57, 2026-04-17 확정)
========================================
확정 신호 조합:
- 4h_S240_SN120: (SMA=240, Mom=20/720, mom2vol, daily vol 5%, Snap=120)
- 4h_S240_SN30:  (SMA=240, Mom=20/480, mom2vol, daily vol 5%, Snap=30)
- 4h_S120_SN120: (SMA=120, Mom=20/720, mom2vol, daily vol 5%, Snap=120)

실행층:
- 고정 3배 레버리지 (L3)
- 스탑: 없음 (STOP_PCT=0, 가드 비활성)
- 캐시 게이트: 없음 (STOP_GATE_CASH_THRESHOLD=0)
- 앙상블 분산만으로 방어

실행: 4h마다 (cron "5 9,13,17,21,1,5 * * *" 한국시간)
1. 바이낸스에서 4h OHLCV 수집
2. 3전략 각각 목표 비중 계산
3. 가중 합산 (1/3씩) → 단일 포트폴리오
4. 고정 3x 레버리지로 매핑
5. 현재 포지션과 비교 → delta 리밸런싱
6. STOP 주문 없음 (sync_stop_orders는 early return)
"""

import argparse
import json
import os
import random
import sys
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional

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
LOG_PATH = os.path.join(SCRIPT_DIR, 'binance_trade.log')

# V21 전략/실행 파라미터 (ENS_fut_L3_k3_12652d57, 2026-04-17 확정)
# 고정 3배 레버리지, 가드 없음, 4h봉 3멤버 EW 1/3씩
LEVERAGE_FLOOR = 3
LEVERAGE_MID = 3
LEVERAGE_CEILING = 3
STOP_PCT = 0.0            # 가드 비활성
STOP_GATE_CASH_THRESHOLD = 0.0  # cash_guard 비활성
LEVERAGE_MOM_LOOKBACK_BARS = 24 * 30  # 동적 lev 비활성이지만 상수 유지

ENSEMBLE_WEIGHTS = {'4h_S240_SN120': 1/3, '4h_S240_SN30': 1/3, '4h_S120_SN120': 1/3}

STRATEGIES = {
    '4h_S240_SN120': {
        'interval': '4h',
        'sma_bars': 240,
        'mom_short_bars': 20,
        'mom_long_bars': 720,
        'health_mode': 'mom2vol',
        'vol_mode': 'daily',
        'vol_threshold': 0.05,
        'snap_interval_bars': 120,
        'canary_hyst': 0.015,
        'n_snapshots': 3,
    },
    '4h_S240_SN30': {
        'interval': '4h',
        'sma_bars': 240,
        'mom_short_bars': 20,
        'mom_long_bars': 480,
        'health_mode': 'mom2vol',
        'vol_mode': 'daily',
        'vol_threshold': 0.05,
        'snap_interval_bars': 30,
        'canary_hyst': 0.015,
        'n_snapshots': 3,
    },
    '4h_S120_SN120': {
        'interval': '4h',
        'sma_bars': 120,
        'mom_short_bars': 20,
        'mom_long_bars': 720,
        'health_mode': 'mom2vol',
        'vol_mode': 'daily',
        'vol_threshold': 0.05,
        'snap_interval_bars': 120,
        'canary_hyst': 0.015,
        'n_snapshots': 3,
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


def send_telegram(msg: str):
    """텔레그램 알림."""
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
    """모든 심볼의 1h, 2h, 4h OHLCV 수집."""
    data = {'1h': {}, '2h': {}, '4h': {}}
    for sym in UNIVERSE:
        for iv in ['1h', '2h', '4h']:
            # 2h 전략은 SMA240+mom720+vol90d가 필요 → 2h bars = 720+@
            # 4h 전략도 SMA240+mom720 → 4h bars = 1800
            # 1h는 레버리지 계산용 (mom 30d = 720 bars)
            limit = {'1h': 2500, '2h': 2000, '4h': 1800}[iv]
            df = fetch_klines(client, sym, iv, limit)
            if not df.empty:
                coin = sym.replace('USDT', '')
                data[iv][coin] = df
            time.sleep(0.05)  # rate limit
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


def get_coin_leverage_map(target: Dict[str, float], data_1h: Dict[str, pd.DataFrame]) -> Dict[str, int]:
    coins = get_target_coins(target)
    if not coins:
        return {}
    ranked = rank_coins_capmom(target, data_1h)
    cash_w = target.get('CASH', 0.0)
    lev_map = {}
    for idx, coin in enumerate(ranked):
        if idx == 0:
            lev = LEVERAGE_CEILING
        elif idx <= 2:
            lev = LEVERAGE_MID
        else:
            lev = LEVERAGE_FLOOR
        lev_map[coin] = apply_cash_degrade(lev, cash_w)
    return lev_map


def compute_strategy_target(strat_name: str, strat_params: dict,
                             data: dict, state: dict,
                             alerts: Optional[List[str]] = None) -> Dict[str, float]:
    """단일 전략의 목표 비중 계산."""
    iv = strat_params['interval']
    bpd = {'4h': 6, '2h': 12, '1h': 24}[iv]
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


def set_leverage(client: Client, symbol: str, leverage: int):
    """레버리지 설정."""
    try:
        client.futures_change_leverage(symbol=symbol, leverage=leverage)
    except BinanceAPIException as e:
        if 'No need to change' not in str(e):
            log.warning(f"set_leverage {symbol}: {e}")


def set_margin_type(client: Client, symbol: str, margin_type: str = 'ISOLATED'):
    """마진 모드 설정."""
    try:
        client.futures_change_margin_type(symbol=symbol, marginType=margin_type)
    except BinanceAPIException as e:
        if 'No need to change' not in str(e):
            log.warning(f"set_margin {symbol}: {e}")


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


def force_cancel_all_orders(client: Client, symbols: Optional[List[str]] = None):
    """심볼별 모든 열린 주문 강제 정리.

    Binance가 조건부 closePosition 주문을 open_orders 목록에 노출하지 않는 경우가 있어,
    -4130 충돌 회피용으로 사용한다.
    """
    symbol_set = set(symbols or UNIVERSE)
    for symbol in symbol_set:
        try:
            client.futures_cancel_all_open_orders(symbol=symbol)
            log.info(f"CANCEL ALL OPEN ORDERS {symbol}")
        except Exception as e:
            log.warning(f"cancel all open orders {symbol}: {e}")
        try:
            client.futures_cancel_all_algo_open_orders(symbol=symbol)
            log.info(f"CANCEL ALL ALGO OPEN ORDERS {symbol}")
        except Exception as e:
            log.warning(f"cancel all algo open orders {symbol}: {e}")


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
    """cash_guard 조건에 따라 reduce-only STOP_MARKET 주문 재등록."""
    cancel_stop_orders(client, list(UNIVERSE))

    # V21: STOP_PCT=0 이면 가드 완전 비활성 (앙상블 분산만으로 방어)
    if STOP_PCT <= 0.0 or STOP_GATE_CASH_THRESHOLD <= 0.0:
        log.info("STOP OFF (V21: 가드 비활성, STOP_PCT=0)")
        return

    cash_w = target.get('CASH', 0.0) if target else 0.0
    if cash_w < STOP_GATE_CASH_THRESHOLD:
        log.info(f"STOP OFF (cash={cash_w:.1%} < {STOP_GATE_CASH_THRESHOLD:.0%})")
        return

    log.info(f"STOP ON (cash={cash_w:.1%} >= {STOP_GATE_CASH_THRESHOLD:.0%}) positions={list(positions.keys())}")

    for coin, pos in positions.items():
        qty = abs(pos.get('qty', 0.0))
        if qty <= 0:
            continue
        df = data_1h.get(coin)
        if df is None or len(df) < 2:
            continue
        prev_close = float(df['Close'].iloc[-2])
        if prev_close <= 0:
            continue
        entry_price = float(pos.get('entry_price') or 0.0)
        entry_stop = entry_price * (1.0 - STOP_PCT) if entry_price > 0 else 0.0
        prev_close_stop = prev_close * (1.0 - STOP_PCT)
        stop_price = max(prev_close_stop, entry_stop)
        symbol = pos['symbol']
        stop_str = format_price(client, symbol, stop_price)
        qty_str = format_quantity(client, symbol, qty)
        log.info(
            f"STOP PLAN {symbol}: qty={qty:.6f} entry={entry_price:.4f} prev_close={prev_close:.4f} "
            f"entry_stop={entry_stop:.4f} prev_close_stop={prev_close_stop:.4f} stop={stop_str}"
        )
        try:
            order = create_order_with_retry(client, dict(
                symbol=symbol,
                side='SELL',
                type='STOP_MARKET',
                stopPrice=stop_str,
                quantity=qty_str,
                reduceOnly='true',
                workingType='CONTRACT_PRICE',
            ))
            log.info(f"STOP SELL {symbol} stop={stop_str}: {order.get('status', 'OK')}")
            if order_alerts is not None:
                order_alerts.append(f"STOP {symbol} {stop_str}")
        except BinanceAPIException as e:
            if 'code=-4130' in str(e) or 'closeposition in the direction is existing' in str(e).lower():
                try:
                    force_cancel_all_orders(client, [symbol])
                    order = create_order_with_retry(client, dict(
                        symbol=symbol,
                        side='SELL',
                        type='STOP_MARKET',
                        stopPrice=stop_str,
                        quantity=qty_str,
                        reduceOnly='true',
                        workingType='CONTRACT_PRICE',
                    ))
                    log.info(f"STOP SELL {symbol} stop={stop_str}: {order.get('status', 'OK')} (retry after cancel_all)")
                    if order_alerts is not None:
                        order_alerts.append(f"STOP {symbol} {stop_str}")
                    continue
                except Exception as e2:
                    e = e2
            log.error(f"STOP FAILED {symbol} stop={stop_str}: {e}")
            if error_alerts is not None:
                error_alerts.append(f"STOP FAILED {symbol} {stop_str}: {e}")


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

    api_key, api_secret = load_config()
    if not api_key:
        log.error("API key not configured")
        return

    client = Client(api_key, api_secret)
    state = load_state()

    # cash_buffer: state에서 동적 읽기
    global CASH_BUFFER
    CASH_BUFFER = state.get('cash_buffer', CASH_BUFFER_DEFAULT)
    log.info(f"cash_buffer: {CASH_BUFFER:.0%}")

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
        lines.append(f"레버리지: 고정 3x (V21 L3 ensemble)")

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

        # 0. 유니버스 갱신 (CoinGecko top40 ∩ 바이낸스 USDT-M 선물 listing)
        refresh_universe(client)

        # 1. 데이터 수집
        log.info("데이터 수집...")
        data = fetch_all_data(client)
        n_1h = len(data['1h'])
        n_2h = len(data['2h'])
        n_4h = len(data['4h'])
        log.info(f"수집 완료: 1h {n_1h}개, 2h {n_2h}개, 4h {n_4h}개 ({time.time()-t_start:.1f}s)")

        # 데이터 장애 방어
        if 'BTC' not in data['1h'] or 'BTC' not in data['2h'] or 'BTC' not in data['4h']:
            log.error("BTC 데이터 누락! 매매 중단. 이전 포지션 유지.")
            send_telegram("⚠️ BTC 데이터 누락 — 매매 중단")
            return
        if n_1h < len(UNIVERSE) // 2 or n_2h < len(UNIVERSE) // 2 or n_4h < len(UNIVERSE) // 2:
            log.error(f"데이터 부족 ({n_1h}/{n_2h}/{n_4h}). 매매 중단.")
            send_telegram(f"⚠️ 데이터 부족 ({n_1h}/{n_2h}/{n_4h}) — 매매 중단")
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

        # 5. 리밸런싱
        target_lev_map = get_coin_leverage_map(combined, data['1h'])
        rebalance_needed = state.get('rebalancing_needed', False)
        if not rebalance_needed:
            log.info("매매 스킵: rebalancing_needed=false")
            positions_after, pv_after = positions_before, pv_before
        elif args.trade:
            # leverage / margin 변경 전에 기존 조건부 주문을 정리한다.
            prep_symbols = sorted({
                *(coin + 'USDT' for coin in target_lev_map.keys()),
                *(pos['symbol'] for pos in positions_before.values()),
            })
            force_cancel_all_orders(client, prep_symbols)
            for coin, lev in target_lev_map.items():
                sym = coin + 'USDT'
                set_leverage(client, sym, lev)
                set_margin_type(client, sym, 'ISOLATED')

            execute_rebalance(
                client, combined, pv_before, target_lev_map,
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
            strat_lines = []
            for strat_name in STRATEGIES:
                target = targets.get(strat_name, {})
                if not target or target.get('CASH', 0.0) >= 0.999:
                    strat_lines.append(f"{strat_name}: CASH")
                else:
                    coins = ', '.join(f"{k}:{v:.1%}" for k, v in target.items() if k != 'CASH' and v > 0)
                    cash_w = target.get('CASH', 0.0)
                    if cash_w > 0:
                        coins = f"{coins}, CASH:{cash_w:.1%}"
                    strat_lines.append(f"{strat_name}: {coins}")

            summary = [
                f"📘 바이낸스 선물 실행 ({datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')})",
                *strat_lines,
                "",
                f"합산 목표: {', '.join(f'{k}:{v:.1%}' for k, v in combined.items() if v > 0) if combined else '없음'}",
            ]
            if canary_alerts:
                summary.append("")
                summary.append("카나리:")
                summary.extend(f"  - {msg}" for msg in canary_alerts)
            if order_alerts:
                summary.append("")
                summary.append("주문:")
                summary.extend(f"  - {msg}" for msg in order_alerts)
            else:
                summary.append("")
                summary.append("주문: 없음")
            if positions_after:
                summary.append("")
                summary.append("현재 포지션:")
                shown = 0
                for coin, pos in positions_after.items():
                    if pos.get('notional', 0.0) < DISPLAY_DUST_NOTIONAL:
                        continue
                    shown += 1
                    summary.append(f"  - {coin}: ${pos['notional']:.0f} ({pos['weight']:.1%}, PnL {pos.get('pnl', 0.0):+.2f})")
                if shown == 0:
                    summary[-1] = "현재 포지션: 없음 (현금)"
            else:
                summary.append("")
                summary.append("현재 포지션: 없음 (현금)")
            if error_alerts:
                summary.append("")
                summary.append("오류:")
                summary.extend(f"  - {msg}" for msg in error_alerts[:10])
            stop_count = count_stop_orders(client, list(UNIVERSE))
            visible_positions = sum(1 for pos in positions_after.values() if pos.get('notional', 0.0) >= DISPLAY_DUST_NOTIONAL)
            summary.append("")
            summary.append("헬스:")
            summary.append(f"  - 리밸런싱 대기: {'예' if state.get('rebalancing_needed', False) else '아니오'}")
            summary.append(f"  - 포지션 수: {visible_positions}")
            summary.append(f"  - 활성 스탑 수: {stop_count}")
            send_telegram("\n".join(summary))
        log.info(f"=== 완료 ({elapsed:.1f}s) ===")
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        lock_fd.close()


if __name__ == '__main__':
    main()
