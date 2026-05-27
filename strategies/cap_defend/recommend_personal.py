"""
Cap Defend V23 Recommendation Script
===================================
V23 (2026-04-30 확정): 모든 자산 1D 단일 + drift trigger.

Stock V23 (snap-based 3-tranche stagger sd=69, stagger=23):
  - signal 생성은 recommend_personal.py 단계, executor 가 snap-based 3-tranche stagger 적용.
  - 가드 없음.

Coin V23: 1D 단일 멤버 D_SMA42 (live engine: trade/coin_live_engine.py)
  - D_SMA42: 1D봉, SMA42, Mom20/127, snap 217봉×7, drift_threshold=0.10
  - 이전 H4_SMA240 멤버 제거 (4h 데이터 fetch 제거)

Futures V23: 1D 단일 멤버 D_SMA42 + L3 고정 (auto_trade_binance.py)
  - D_SMA42: 1D봉, SMA42, Mom18/127, snap 95봉×5, drift_threshold=0.03 (05-04 갱신)
  - 가드 없음, 스탑 없음

Asset Allocation: 60/25/15 (V23 갱신 2026-05-22), 리밸 트리거: T1(ht≥13pp) OR T3U_can(max rel-under≥20% & sleeve canary ON)
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import time
import json
import sqlite3
import requests
from datetime import datetime, timezone, timedelta
from typing import Tuple
import pyupbit

# ─── Telegram ─────────────────────────────────────────────
try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

try:
    from config import PORTFOLIO_PUBLIC_URL as CONFIG_PORTFOLIO_PUBLIC_URL
except Exception:
    CONFIG_PORTFOLIO_PUBLIC_URL = ""

def send_telegram(msg, button_text=None, button_url=None):
    """텔레그램 알림 전송. 실패해도 프로그램은 계속 진행."""
    try:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
        if button_text and button_url:
            payload["reply_markup"] = {
                "inline_keyboard": [[{"text": button_text, "url": button_url}]]
            }
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json=payload,
            timeout=10
        )
    except Exception:
        pass

# --- Configuration for Auto Turnover ---
# Try local config.py first (remote server), then ../../config/upbit.py (local dev)
ACCESS_KEY = ""
SECRET_KEY = ""
for _cfg_path in [os.path.dirname(os.path.abspath(__file__)),
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config')]:
    if _cfg_path not in sys.path:
        sys.path.insert(0, _cfg_path)
try:
    from config import UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY
    ACCESS_KEY = UPBIT_ACCESS_KEY
    SECRET_KEY = UPBIT_SECRET_KEY
except Exception:
    try:
        from upbit import UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY
        ACCESS_KEY = UPBIT_ACCESS_KEY
        SECRET_KEY = UPBIT_SECRET_KEY
    except Exception as e:
        print(f"⚠️ Upbit config import failed: {e}")

# --- 1. Constants & Configuration ---
DATA_DIR = "./data"
SIGNAL_STATE_FILE = os.path.join(".", "signal_state.json")
APP_HOME = os.environ.get("MONEYFLOW_APP_HOME", os.getcwd())
ASSETS_DB = os.environ.get("ASSETS_DB", os.path.join(APP_HOME, "assets.db"))
try:
    from config import PORTFOLIO_HTML_NAME as CONFIG_HTML_NAME
except ImportError:
    CONFIG_HTML_NAME = "portfolio_result_gmoh.html"
PORTFOLIO_HTML_NAME = os.environ.get("PORTFOLIO_HTML_NAME") or CONFIG_HTML_NAME
PORTFOLIO_PUBLIC_URL = os.environ.get("PORTFOLIO_PUBLIC_URL", "") or CONFIG_PORTFOLIO_PUBLIC_URL
STOCK_ANCHOR_DAYS = (1, 24, 47)  # V23: snap-based 3 staggered (offset days, period 69, stagger 23)
COIN_ANCHOR_DAYS = (1, 11, 21)   # legacy 표시용
FUTURES_TRANCHE_META = {
    "D_SMA42":   {"interval_hours": 24, "snap_interval_bars": 95,  "n_snapshots": 5},  # V23 갱신 05-04
}
COIN_MEMBER_META = {
    "D_SMA42":   {"interval_hours": 24, "snap_interval_bars": 217, "n_snapshots": 7},  # V23
}

def _save_signal_state(data):
    """signal_state.json 원자적 저장."""
    tmp = SIGNAL_STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, SIGNAL_STATE_FILE)


def save_daily_live_snapshot():
    """실계좌 기준 일간 스냅샷 저장. 하루 1건 upsert."""
    api = os.environ.get("TRADE_API_BASE", "http://127.0.0.1:5000")
    r = requests.get(f"{api}/api/assets/live_overview", timeout=60)
    r.raise_for_status()
    data = r.json()

    accounts = data.get("accounts", {})
    stock = accounts.get("stock_kis", {}) or {}
    upbit = accounts.get("coin_upbit", {}) or {}
    binance = accounts.get("coin_binance", {}) or {}

    stock_krw = float(stock.get("stock_eval_usd", 0.0)) * float(stock.get("exchange_rate", 0.0))
    fx_rate = float(stock.get("exchange_rate", 0.0) or binance.get("exchange_rate", 0.0) or 0.0)
    upbit_coin_krw = sum(float(row.get("value", 0.0)) for row in (upbit.get("holdings") or []))
    binance_total_krw = float(binance.get("total_krw", 0.0))
    binance_cash_krw = float(binance.get("cash_krw", 0.0))
    binance_invested_krw = max(binance_total_krw - binance_cash_krw, 0.0)  # 선물 포지션 + spot 토큰
    coin_krw = upbit_coin_krw + binance_invested_krw  # 업비트 holdings + 바이낸스 투자분
    cash_krw = (
        float(stock.get("cash_krw", 0.0))
        + float(upbit.get("krw_balance", 0.0))
        + binance_cash_krw
    )  # KIS 현금 + Upbit KRW + Binance(선물 free margin + spot USDT)
    total_krw = stock_krw + coin_krw + cash_krw
    usd_cash = float(stock.get("cash_usd", 0.0))
    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    accounts_json = json.dumps(accounts, ensure_ascii=False)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db_path = ASSETS_DB if os.path.isdir(os.path.dirname(ASSETS_DB)) else os.path.join(".", "assets.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT UNIQUE NOT NULL,
            stock_krw REAL DEFAULT 0,
            coin_krw REAL DEFAULT 0,
            futures_krw REAL DEFAULT 0,
            cash_krw REAL DEFAULT 0,
            total_krw REAL DEFAULT 0,
            fx_rate REAL DEFAULT 0,
            usd_cash REAL DEFAULT 0,
            memo TEXT DEFAULT '',
            accounts_json TEXT DEFAULT '{}',
            created_at TEXT
        )"""
    )
    # 기존 DB에 futures_krw 컬럼이 없을 수 있음
    try:
        conn.execute("ALTER TABLE snapshots ADD COLUMN futures_krw REAL DEFAULT 0")
    except Exception:
        pass  # 이미 존재
    conn.execute(
        """INSERT INTO snapshots
           (snapshot_date, stock_krw, coin_krw, futures_krw, cash_krw, total_krw, fx_rate, usd_cash, memo, accounts_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(snapshot_date) DO UPDATE SET
             stock_krw=excluded.stock_krw,
             coin_krw=excluded.coin_krw,
             futures_krw=excluded.futures_krw,
             cash_krw=excluded.cash_krw,
             total_krw=excluded.total_krw,
             fx_rate=excluded.fx_rate,
             usd_cash=excluded.usd_cash,
             memo=excluded.memo,
             accounts_json=excluded.accounts_json,
             created_at=excluded.created_at
        """,
        (
            snapshot_date,
            stock_krw,
            coin_krw,
            0.0,  # futures_krw: 코인에 합산됨
            cash_krw,
            total_krw,
            fx_rate,
            usd_cash,
            "Auto daily snapshot",
            accounts_json,
            created_at,
        ),
    )
    conn.commit()
    conn.close()
    return {
        "snapshot_date": snapshot_date,
        "stock_krw": stock_krw,
        "coin_krw": coin_krw,
        "cash_krw": cash_krw,
        "total_krw": total_krw,
    }
STRATEGY_VERSION = "V23"
VERSION_HISTORY = [
    ("V23", "2026-04-30",
     "전 자산 V23: 코인/선물 1D 단일 + drift trigger, 주식 snap-based stagger (sd=69, stagger=23). 가드 없음, 자산배분 60/25/15 (V23 갱신 2026-05-22, M 안).",
     """<b>▶ 코인 현물 (V23 — 1D 단일 D_SMA42 + drift)</b>
• <b>D_SMA42:</b> 일봉 · SMA42 · Mom20/127 · snap 217봉×7 · canary hyst 1.5% · drift_threshold 0.10
• <b>헬스:</b> Mom_short&gt;0 AND Mom_long&gt;0 AND daily Vol≤5%
• <b>실매매:</b> trade/coin_live_engine.py + trade/executor_coin.py

<b>▶ 선물 (V23 — 1D 단일 D_SMA42 + drift)</b>
• <b>D_SMA42:</b> 일봉 · SMA42 · Mom18/127 · snap 95봉×5 · drift_threshold 0.03
• 고정 3x · 가드 없음

<b>▶ 주식 (V23 — snap-based stagger sd=69)</b>
• <b>유니버스:</b> SPY, QQQ, VEA, EEM, EWJ, GLD, PDBC (7종, R7B)
• <b>카나리:</b> EEM &gt; SMA300 (2.0% hysteresis)
• <b>선정:</b> Z-score Top 3 (Mom + Sharpe126 합)
• <b>스냅:</b> 69일 주기 × 3 snap 스태거 (23일 오프셋), EW 평균
• <b>가드:</b> 없음 (앙상블 분산 단독 방어)

<b>▶ 자산배분:</b> 65/20/15 (주식계좌/현물/선물, V23 갱신 2026-05-26 B 안). 트리거: T1(ht≥13pp) OR T3U_can(max rel-under≥20% &amp; sleeve canary ON). 자산간 자동 rebal 제거 — 트리거 ON 시 텔레그램 알림만, 사용자 수동 송금. Per-sleeve cash buffer: stock 7% / spot 1% / fut 1% (total cash ≈ 5%)"""),
]

STOCK_RATIO, COIN_RATIO, FUTURES_RATIO = 0.65, 0.20, 0.15  # V23 갱신 (2026-05-26): 60/25/15 → 65/20/15 (B 안 채택, stock 계좌 65% 안에 cash 7%, 실투자 60.45%)
REBAL_HT_THRESHOLD = 0.13  # V23: T1 = half_turnover (sum|cur-tgt|/2) ≥ 13pp
REBAL_T3U_REL = 0.20  # V23: T3U_can = max((tgt - cur_w)/tgt) ≥ 20% AND 해당 sleeve canary ON
CASH_ASSET = 'Cash'
STOCK_CASH_BUFFER_DEFAULT = 0.07   # V23 (2026-05-26): stock 계좌 안에서 7% cash buffer (= 4.55% of total)
SPOT_CASH_BUFFER_DEFAULT = 0.01    # V23 (2026-05-26): Upbit 1%
FUT_CASH_BUFFER_DEFAULT = 0.01     # V23 (2026-05-26): Binance 1%
CASH_BUFFER_PERCENT_DEFAULT = STOCK_CASH_BUFFER_DEFAULT  # backward compat (stock 기준)
REBAL_BAND_PP = 0.08  # 8pp band — any asset drifts ≥8pp → full rebalance

def get_cash_buffer(sleeve: str = 'stock'):
    """sleeve 별 현금 버퍼 비율. V23 trade_state.json 에서 sleeve_cash_buffer 키 우선 읽기.

    sleeve: 'stock' | 'spot' | 'fut'.
    """
    default = {
        'stock': STOCK_CASH_BUFFER_DEFAULT,
        'spot': SPOT_CASH_BUFFER_DEFAULT,
        'fut': FUT_CASH_BUFFER_DEFAULT,
    }.get(sleeve, STOCK_CASH_BUFFER_DEFAULT)
    key = f'{sleeve}_cash_buffer'
    for _p in (
        os.path.join(APP_HOME, 'trade_state.json'),
        'trade_state.json',
    ):
        try:
            with open(_p, 'r') as f:
                d = json.load(f)
                # 신규 키 우선, 없으면 legacy cash_buffer (stock 호환)
                v = d.get(key)
                if v is None and sleeve == 'stock':
                    v = d.get('cash_buffer')
                if v is not None:
                    return float(v)
                return default
        except Exception:
            continue
    return default
STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'TUSD', 'PAX', 'GUSD', 'FRAX', 'LUSD', 'MIM', 'USDN', 'FDUSD']

# Stock Configuration (V23 R7B: B안 universe — 11yr rs=58, 5yr Cal 0.84)
OFFENSIVE_STOCK_UNIVERSE = ['SPY', 'QQQ', 'VEA', 'EEM', 'EWJ', 'GLD', 'PDBC']
DEFENSIVE_STOCK_UNIVERSE = ['IEF', 'BIL', 'BNDX', 'GLD', 'PDBC']
CANARY_ASSETS = ['EEM']
STOCK_CANARY_MA_PERIOD = 300   # V23
STOCK_CANARY_HYST = 0.020      # V23 (2%)
# V23: 가드 전면 제거. STOCK_CRASH_TICKER 만 universe 가격 다운로드용으로 보존.
STOCK_CRASH_TICKER = 'VT'

# Coin Configuration
COIN_CANARY_MA_PERIOD = 42  # V23: D_SMA42 (이전 V22 = 50)
COIN_CANARY_HYST = 0.015  # 1.5% Hysteresis: enter Risk-On at SMA*1.015, exit at SMA*0.985
N_SELECTED_COINS = 5
VOLATILITY_WINDOW = 90

# --- V15 Configuration ---
VOL_CAP_FILTER = 0.05
BL_THRESHOLD = -0.15
BL_DAYS = 7
DD_EXIT_LOOKBACK = 60
DD_EXIT_THRESHOLD = -0.25
CRASH_THRESHOLD = -0.10

def get_dynamic_coin_universe(log: list) -> (list, dict):
    print("\n--- 🛰️ Step 1: Coin Universe Selection (V15: LIVE CoinGecko + Upbit Filter) ---")
    log.append("<h2>🛰️ Step 1: 코인 유니버스 선정 (V23: Live CoinGecko Top 40)</h2>")
    
    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
    FETCH_LIMIT = 40
    MIN_TRADE_VALUE_KRW = 1_000_000_000 
    DAYS_TO_CHECK = 260 
    headers = {"accept": "application/json", "User-Agent": "Mozilla/5.0"}
    
    UNIVERSE_CACHE_FILE = os.path.join(DATA_DIR, "universe_cache.json")
    cg_data = []

    # 1. Try Fetching from CoinGecko (Retry 5 times)
    for attempt in range(1, 6):
        try:
            print(f"  - Fetching Top {FETCH_LIMIT} from CoinGecko (Attempt {attempt}/5)...")
            cg_params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': FETCH_LIMIT, 'page': 1}
            cg_response = requests.get(COINGECKO_URL, params=cg_params, headers=headers, timeout=20)
            
            if cg_response.status_code == 200:
                cg_data = cg_response.json()
                # Save to Cache
                with open(UNIVERSE_CACHE_FILE, 'w') as f:
                    json.dump(cg_data, f)
                print(f"    ✅ Got {len(cg_data)} coins from CoinGecko (Cached)")
                break
            elif cg_response.status_code == 429:
                wait_time = 30 * attempt # 30s, 60s, 90s...
                print(f"    ⚠️ Rate Limit (429). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    ⚠️ API Error: {cg_response.status_code}")
                time.sleep(10)
        except Exception as e:
            print(f"    ⚠️ Connection Error: {e}")
            time.sleep(10)
            
    # 2. If Failed, Load from Cache
    if not cg_data:
        if os.path.exists(UNIVERSE_CACHE_FILE):
            print(f"  ⚠️ Loading Universe from Local Cache (Fallback)...")
            log.append("<p class='warning'>⚠️ Loading Universe from Local Cache (Fallback).</p>")
            try:
                with open(UNIVERSE_CACHE_FILE, 'r') as f:
                    cg_data = json.load(f)
            except: pass

    if not cg_data:
        log.append("<p class='error'>❌ CoinGecko + cache 모두 실패. universe 없음 → 매매 skip.</p>")
        return [], {}

    cg_symbol_to_id_map = {f"{item['symbol'].upper()}-USD": item['id'] for item in cg_data}

    print("  - Fetching Upbit KRW Market...")
    try:
        upbit_krw_tickers = pyupbit.get_tickers(fiat="KRW")
        upbit_symbols = {t.split('-')[1] for t in upbit_krw_tickers}
        print(f"    ✅ Upbit has {len(upbit_symbols)} KRW markets")
    except Exception as e:
        print(f"    ⚠️ Upbit API Error: {e}")
        upbit_symbols = set()

    print("  - Fetching Binance Spot listed symbols...")
    binance_spot_symbols = set()
    try:
        bn_resp = requests.get("https://api.binance.com/api/v3/exchangeInfo", timeout=15)
        if bn_resp.status_code == 200:
            for s in bn_resp.json().get('symbols', []):
                if s.get('status') == 'TRADING' and s.get('quoteAsset') == 'USDT':
                    base = s.get('baseAsset', '').upper()
                    if base: binance_spot_symbols.add(base)
            print(f"    ✅ Binance Spot has {len(binance_spot_symbols)} USDT pairs")
    except Exception as e:
        print(f"    ⚠️ Binance Spot exchangeInfo Error: {e}")

    print("  - Filtering by Upbit/Binance Spot listing, liquidity, history...")
    final_universe = []

    for item in cg_data:
        symbol = item['symbol'].upper()
        if symbol in STABLECOINS: continue
        if symbol not in upbit_symbols:
            print(f"    ❌ {symbol}: Not in Upbit KRW")
            continue
        if binance_spot_symbols and symbol not in binance_spot_symbols:
            print(f"    ❌ {symbol}: Not in Binance Spot USDT")
            continue
        
        upbit_ticker = f"KRW-{symbol}"
        try:
            df = pyupbit.get_ohlcv(ticker=upbit_ticker, interval="day", count=DAYS_TO_CHECK)
            time.sleep(0.1)
            
            if df is None or len(df) < 253:
                print(f"    ❌ {symbol}: Insufficient History ({len(df) if df is not None else 0} days < 253)")
                continue
            
            avg_val = df['value'].iloc[-30:].mean()
            if avg_val < MIN_TRADE_VALUE_KRW:
                print(f"    ❌ {symbol}: Low Liquidity ({avg_val/100000000:.1f}억 < 10억)")
                continue
            
            ticker_usd = f"{symbol}-USD"
            if ticker_usd in final_universe: continue
                
            final_universe.append(ticker_usd)
            print(f"    ✅ {symbol}: Included (Rank {len(final_universe)}, Liquidity: {avg_val/100000000:.1f}억)")
            
        except Exception as e:
            print(f"    ⚠️ {symbol}: Upbit Check Error - {e}")
            continue
        
        if len(final_universe) >= 40: break

    log.append(f"<p>선정된 유니버스 ({len(final_universe)}개): Top 40 qualified</p>")
    return final_universe, cg_symbol_to_id_map


# --- Helper: Get Upbit Assets (Personal) ---
def get_current_upbit_holdings(log):
    """
    업비트 API를 사용하여 현재 보유 자산을 조회합니다.
    - holdings_qty: {ticker-USD: qty} (수량만 - 종가 평가용)
    - holdings_krw: {ticker-USD: krw_value} (실시간 KRW 가치 - 표시용)
    - unlisted coins are filtered out
    """
    if not ACCESS_KEY or not SECRET_KEY:
        print("⚠️ Upbit API keys not configured")
        log.append("<p class='error'>❌ Upbit API 키 미설정</p>")
        return {}, {}, 0.0

    try:
        upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

        krw = upbit.get_balance("KRW")
        if krw is None:
            print(f"⚠️ Upbit API Connection Failed (key prefix: {ACCESS_KEY[:6]}...)")
            log.append("<p class='error'>❌ Upbit API 연결 실패 (키 확인 필요)</p>")
            return {}, {}, 0.0
        
        # Get currently listed coins in Upbit KRW market
        try:
            upbit_krw_tickers = pyupbit.get_tickers(fiat="KRW")
            upbit_listed = {t.split('-')[1] for t in upbit_krw_tickers}
        except:
            upbit_listed = set()
            
        balances = upbit.get_balances()
        holdings_qty = {}   # {ticker-USD: qty}
        holdings_krw = {}   # {ticker-USD: krw_value} (real-time for display)
        
        my_cash = 0.0
        
        for b in balances:
            ticker = b['currency']
            if ticker == 'KRW': 
                my_cash = float(b['balance'])
                continue
            
            # Filter: only include coins currently listed in Upbit KRW market
            if ticker not in upbit_listed:
                print(f"    ⚠️ {ticker}: Not in Upbit KRW market (Skipped)")
                continue
            
            qty = float(b['balance']) + float(b['locked'])
            if qty > 0:
                try:
                    price_krw = pyupbit.get_current_price(f"KRW-{ticker}")
                    if price_krw:
                        val_krw = qty * price_krw
                        if val_krw > 1000:  # Exclude dust
                            holdings_qty[f"{ticker}-USD"] = qty
                            holdings_krw[f"{ticker}-USD"] = val_krw
                except: pass
                
        return holdings_qty, holdings_krw, my_cash
        
    except Exception as e:
        print(f"⚠️ Upbit Asset Load Error: {e}")
        log.append(f"<p class='error'>❌ 자산 조회 오류: {e}</p>")
        return {}, {}, 0.0

# --- 3. Data Download ---
def download_required_data(tickers: list, log: list, coin_id_map: dict):
    """
    데이터 다운로드 - Yahoo Finance 우선, CoinGecko 폴백
    [V12.3] 데이터 안정화: File Cache, Retry(10회), Stale Data Fallback 적용
    """
    print("\n--- 📥 Step 2: 데이터 다운로드 (Yahoo Priority + Quality Check) ---")
    log.append("<h2>📥 Step 2: 데이터 다운로드 (Yahoo 우선 + 품질검증)</h2>")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # USD-KRW 환율 조회 (품질 검증용)
    usd_krw_rate = 1450.0
    try:
        usdt_price = pyupbit.get_current_price("KRW-USDT")
        if usdt_price: usd_krw_rate = usdt_price
    except: pass
    
    # [V12.3] Robust Session with Retries
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    # Retry 10 times, backoff factor 0.5 (0.5, 1, 2, 4...)
    retries = Retry(total=10, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    today_str = datetime.now().date()
    
    for ticker in list(set(tickers)):
        if ticker == CASH_ASSET: continue
        fp = os.path.join(DATA_DIR, f"{ticker}.csv")
        success = False
        
        # 1. Fresh Cache Check
        # 오늘 다운로드한 파일이 있으면 재사용 (삭제하지 않음)
        if os.path.exists(fp):
            try:
                # 파일 수정 시간 확인
                mtime = datetime.fromtimestamp(os.path.getmtime(fp)).date()
                if mtime == today_str:
                    #print(f"  ✅ Using cached data for {ticker}")
                    # 캐시된 파일도 품질 검증을 통과했다고 가정하거나, 로드 시점에 다시 검증할 수 있음.
                    # 여기서는 '이미 검증되어 저장된 파일'이라고 가정하고 스킵.
                    success = True
            except: pass
        
        if success:
            continue

        # 2. Crypto (-USD) → Binance Spot. Else → Yahoo.
        is_crypto = ticker.endswith('-USD') and ticker.replace('-USD', '') not in {'EEM', 'SPY', 'QQQ', 'TLT', 'GLD', 'IEF', 'IWM', 'EFA', 'BIL'}
        if is_crypto:
            try:
                symbol = ticker.replace('-USD', '')
                bn_symbol = f"{symbol}USDT"
                url = f"https://api.binance.com/api/v3/klines"
                params = {'symbol': bn_symbol, 'interval': '1d', 'limit': 1000}
                resp = session.get(url, params=params, timeout=30)
                if resp.status_code == 200:
                    klines = resp.json()
                    if klines:
                        rows = [(pd.to_datetime(k[0], unit='ms').date(), float(k[4])) for k in klines]
                        df = pd.DataFrame(rows, columns=['Date', 'Adj_Close']).drop_duplicates('Date')
                        # 진행중 봉 (오늘) 제외: 마지막 row 가 오늘이면 drop
                        today_d = datetime.now().date()
                        if len(df) > 0 and df['Date'].iloc[-1] == today_d:
                            df = df.iloc[:-1]
                        if len(df) > 0:
                            df.to_csv(fp, index=False)
                            success = True
                            print(f"  - Downloaded {ticker} (Binance Spot {bn_symbol})")
                else:
                    print(f"  ⚠️ {ticker}: Binance Spot status {resp.status_code}")
            except Exception as e:
                print(f"  ⚠️ Binance Spot download failed for {ticker}: {e}")
            # crypto: Yahoo/CoinGecko fallback 사용 안 함 (source 통일)
            if not success:
                if os.path.exists(fp):
                    file_date = datetime.fromtimestamp(os.path.getmtime(fp)).date()
                    print(f"  ⚠️ {ticker} Binance fetch 실패 → STALE {file_date}")
                    log.append(f"<p class='warning'>Used stale data for {ticker} ({file_date})</p>")
                else:
                    log.append(f"<p class='error'>XXX Failed to download: {ticker}</p>")
            continue

        # Yahoo (주식)
        try:
            current_timestamp = int(datetime.now(timezone.utc).timestamp())
            start_timestamp = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp())
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            params = {"period1": start_timestamp, "period2": current_timestamp, "interval": "1d", "includeAdjustedClose": "true"}

            # [Stabilization] Timeout increased to 30s
            resp = session.get(url, params=params, timeout=30)

            if resp.status_code == 200:
                res = resp.json()['chart']['result'][0]
                df = pd.DataFrame({'Date': pd.to_datetime(res['timestamp'], unit='s').date, 'Adj_Close': res['indicators']['adjclose'][0]['adjclose']})
                df = df.dropna().drop_duplicates('Date')

                if False and ticker.endswith('-USD') and len(df) > 0:
                    symbol = ticker.replace('-USD', '')
                    try:
                        upbit_ohlcv = pyupbit.get_ohlcv(f"KRW-{symbol}", interval="day", count=1)
                        if upbit_ohlcv is not None and len(upbit_ohlcv) > 0:
                            upbit_close_krw = upbit_ohlcv['close'].iloc[-1]
                            yahoo_last_usd = df['Adj_Close'].iloc[-1]
                            yahoo_last_krw = yahoo_last_usd * usd_krw_rate

                            if upbit_close_krw > 0 and yahoo_last_krw > 0:
                                diff_pct = abs(yahoo_last_krw - upbit_close_krw) / upbit_close_krw
                                if diff_pct > 0.10:
                                    print(f"  ⚠️ {ticker}: 가격 불일치 (Yahoo {yahoo_last_krw:,.0f}원 vs Upbit {upbit_close_krw:,.0f}원, diff={diff_pct:.0%}) - 제외")
                                    # 저장하지 않고 스킵
                                    continue 
                    except: pass
                
                df.to_csv(fp, index=False)
                success = True
                print(f"  - Downloaded {ticker} (Yahoo)")
                
        except Exception as e:
            # print(f"  ⚠️ Yahoo download failed for {ticker}: {e}")
            pass
            
        # 3. Fallback to CoinGecko
        if not success and ticker in coin_id_map:
             try:
                cid = coin_id_map[ticker]
                url = f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart"
                # Timeout 30s
                resp = requests.get(url, params={'vs_currency':'usd','days':'500'}, timeout=30)
                if resp.status_code == 200:
                    data = resp.json().get('prices', [])
                    df = pd.DataFrame(data, columns=['ts', 'Adj_Close'])
                    df['Date'] = pd.to_datetime(df['ts'], unit='ms').dt.date
                    df[['Date','Adj_Close']].drop_duplicates('Date').to_csv(fp, index=False)
                    print(f"  - Downloaded {ticker} (CoinGecko)")
                    success = True
             except: pass
        
        # 4. Final Safety Net: Stale Data Fallback
        if not success:
            if os.path.exists(fp):
                # 기존 파일이 있으면 (어제 파일 등) 사용
                file_date = datetime.fromtimestamp(os.path.getmtime(fp)).date()
                print(f"  ⚠️ Failed to update {ticker}, using STALE data from {file_date}")
                log.append(f"<p class='warning'>Used stale data for {ticker} ({file_date})</p>")
                # success = True로 간주하지 않고, 에러 로그는 남기지 않음 (데이터가 있으므로)
            else:
                log.append(f"<p class='error'>XXX Failed to download: {ticker}</p>")


# --- 4. Logic Engines ---
def load_price(ticker):
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), parse_dates=['Date'])
        # [Validation] Check Data Recency
        if df.empty: return pd.Series(dtype=float)
        
        last_date = df['Date'].iloc[-1].date() if hasattr(df['Date'].iloc[-1], 'date') else df['Date'].iloc[-1]
        today = datetime.now().date()
        
        # 만약 데이터 마지막 날짜가 7일 이상 지났으면 쓸모없는 데이터 (상폐/티커변경 등)
        if (today - last_date).days > 7:
            return pd.Series(dtype=float)

        return df.set_index('Date')['Adj_Close'].sort_index()
    except: return pd.Series(dtype=float)

def calc_sma(s, w): return s.rolling(w).mean().iloc[-1] if len(s) >= w else np.nan
def calc_ret(s, d): return s.iloc[-1]/s.iloc[-1-d] - 1 if len(s) >= d+1 and s.iloc[-1-d]!=0 else np.nan
def calc_sharpe(s, d):
    if len(s) < d+1: return 0
    ret = s.pct_change().iloc[-d:]
    return (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0
def calc_weighted_mom(s):
    """V15: Pure 12-month momentum."""
    if len(s) < 253: return -np.inf
    return calc_ret(s, 252)

# --- V15 DD Exit / Blacklist ---
def check_dd_exit(s, lookback=DD_EXIT_LOOKBACK, threshold=DD_EXIT_THRESHOLD):
    """Check if coin should be exited: price / max(recent lookback days) - 1 < threshold."""
    if len(s) < lookback: return False, 0.0
    recent = s.iloc[-lookback:]
    peak = recent.max()
    if peak <= 0: return False, 0.0
    dd = s.iloc[-1] / peak - 1
    return dd <= threshold, dd

def check_blacklist(s, threshold=BL_THRESHOLD, lookback_days=BL_DAYS):
    """Check if coin had a daily drop worse than threshold in the last lookback_days."""
    if len(s) < lookback_days + 1: return False, 0.0
    recent = s.iloc[-(lookback_days + 1):]
    daily_rets = recent.pct_change().dropna()
    worst = daily_rets.min()
    return worst <= threshold, worst

def run_stock_strategy_v15(log, all_prices, target_date):
    """V23 Stock Strategy: R7B universe (SPY/QQQ/VEA/EEM/EWJ/GLD/PDBC) + EEM SMA300 canary 2.0% + Z-score3(Sh126) EW + Defense Top2. 가드 없음 (앙상블 분산 단독 방어)."""
    log.append("<h2>📈 주식 포트폴리오 분석 (V23: R7B + EEM SMA300 hyst2% + Zscore3 Sh126d, 가드 없음)</h2>")
    eem = all_prices.get('EEM')
    meta = {'signal_dist': {}, 'next_candidates': []}
    # 이전 추천 종목 (HTML 리포트 비교 표시용)
    stock_holdings: list = []
    signal_flipped = False
    try:
        with open(SIGNAL_STATE_FILE, 'r') as _sf:
            _prev = json.load(_sf)
        stock_holdings = _prev.get('stock', {}).get('offense_picks', [])
    except Exception:
        pass

    if eem is not None and len(eem) >= STOCK_CANARY_MA_PERIOD:
        eem_sma = eem.rolling(STOCK_CANARY_MA_PERIOD).mean().iloc[-1]
        eem_cur = eem.iloc[-1]
        dist = eem_cur / eem_sma - 1
        meta['signal_dist'] = {'EEM': dist}
        meta['canary_eem_cur'] = float(eem_cur)
        meta['canary_eem_sma'] = float(eem_sma)
        meta['canary_sma_period'] = STOCK_CANARY_MA_PERIOD

        # EEM-only canary with 0.5% hysteresis
        if dist > STOCK_CANARY_HYST:
            risk_on = True
        elif dist < -STOCK_CANARY_HYST:
            risk_on = False
        else:
            risk_on = eem_cur > eem_sma  # dead zone

        # 히스테리시스 dead zone → 이전 risk_on 참조
        prev_risk_on = None
        try:
            with open(SIGNAL_STATE_FILE, 'r') as _sf:
                prev_risk_on = json.load(_sf).get('stock', {}).get('risk_on')
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        # dead zone에서 이전 상태 유지
        if abs(dist) <= STOCK_CANARY_HYST and prev_risk_on is not None:
            risk_on = prev_risk_on
        signal_flipped = (prev_risk_on is not None and prev_risk_on != risk_on)

        log.append(f"<p><b>[Canary]</b> EEM: ${eem_cur:.2f} (MA{STOCK_CANARY_MA_PERIOD} ${eem_sma:.2f}, dist {dist:+.2%}, hyst ±{STOCK_CANARY_HYST:.1%})</p>")
        flip_info = " \U0001f504 <b>SIGNAL FLIP</b>" if signal_flipped else ""
        if risk_on: log.append(f"<p>\u2705 <b>Risk-On</b>{flip_info}</p>")
        else: log.append(f"<p>\U0001f6a8 <b>Risk-Off</b>{flip_info}</p>")
    else:
        risk_on = False
        log.append("<p class='error'>Canary Data Missing (EEM)</p>")

    if risk_on:
        log.append("<h4>🚀 공격 모드 (V23: Z-score Top 3 + Sharpe126d + EW)</h4>")
        scores = []
        for t in OFFENSIVE_STOCK_UNIVERSE:
            p = all_prices.get(t)
            if p is None or len(p) < 253: continue
            scores.append({'Ticker': t, 'Mom12M': calc_weighted_mom(p), 'Sharpe126': calc_sharpe(p, 126)})

        if not scores:
            log.append("<p class='warning'>공격 ETF 데이터 부족 → 수비 전환</p>")
        else:
            df = pd.DataFrame(scores).set_index('Ticker')

            # V23 Z-score composite: zscore(12M_mom) + zscore(Sharpe126d)
            m_std = df['Mom12M'].std()
            s_std = df['Sharpe126'].std()
            df['Z_Mom'] = (df['Mom12M'] - df['Mom12M'].mean()) / m_std if m_std > 0 else 0
            df['Z_Sh'] = (df['Sharpe126'] - df['Sharpe126'].mean()) / s_std if s_std > 0 else 0
            df['ZScore'] = df['Z_Mom'] + df['Z_Sh']

            try: log.append(f"<div class='table-wrap'>{df.to_html(classes='dataframe small-table', float_format='%.4f')}</div>")
            except: pass

            picks = df.nlargest(3, 'ZScore').index.tolist()
            meta['selection_reason'] = {'ZScore_Top3': picks}

            # Trigger detection: compare with actual holdings (signal_state.json의 stock_holdings)
            if stock_holdings:
                new_picks = sorted(set(picks) - set(stock_holdings))
                exit_picks = sorted(set(stock_holdings) - set(picks))
                n_changed = len(new_picks)
                is_monthly = (target_date.day <= 5)
                trigger_rebal = (n_changed >= 2 and not is_monthly)

                if trigger_rebal:
                    log.append(f"<p style='color:#d93025;font-size:1.1em'><b>🔄 TRIGGER REBALANCE: {n_changed}종목 변경</b></p>")
            else:
                log.append(f"<p style='color:#e37400'>⚠️ 보유종목 미설정</p>")

            log.append(f"<p>공격 {len(picks)}종목 선정 (Equal Weight)</p>")
            # NOTE: signal_state 저장은 main()에서 새 스키마로 일괄 저장
            return {t: 1.0/len(picks) for t in picks}, "공격 모드", meta

    # Defense mode: Top 3 by 6M return
    log.append("<h4>🛡️ 수비 모드 (Top 3 by 6M Return)</h4>")
    res = []
    for t in DEFENSIVE_STOCK_UNIVERSE:
        p = all_prices.get(t)
        if p is None: continue
        r = calc_ret(p, 126)
        if pd.notna(r): res.append({'Ticker': t, '6m Ret': r})

    try: log.append(f"<div class='table-wrap'>{pd.DataFrame(res).sort_values('6m Ret', ascending=False).to_html(classes='dataframe small-table')}</div>")
    except: pass

    if not res:
        return {CASH_ASSET: 1.0}, "수비 (데이터 없음)", meta

    res.sort(key=lambda x: x['6m Ret'], reverse=True)
    top3 = [r for r in res[:3] if r['6m Ret'] > 0]
    if not top3:
        return {CASH_ASSET: 1.0}, "수비 (전부 음수)", meta
    picks = [r['Ticker'] for r in top3]
    log.append(f"<p>Defense Picks: <b>{picks}</b> (Equal Weight)</p>")
    # NOTE: signal_state 저장은 main()에서 새 스키마로 일괄 저장
    return {t: 1.0/len(picks) for t in picks}, f"수비 ({', '.join(picks)})", meta

def _load_v20_state_personal():
    """V23 live state (trade_state.json)을 여러 경로에서 탐색해 로드."""
    candidates = [
        os.path.join(os.getcwd(), 'trade_state.json'),
        '/home/ubuntu/trade_state.json',
        os.path.join(os.path.dirname(os.path.abspath(__file__)), 'trade_state.json'),
    ]
    for path in candidates:
        try:
            with open(path, 'r') as f:
                return json.load(f), path
        except (FileNotFoundError, json.JSONDecodeError):
            continue
    return None, None

def run_coin_strategy_v20(coin_universe, all_prices, target_date, log, is_today=True):
    """V23 앙상블 표시: trade_state.json의 결합 타겟 + 멤버 상태 렌더링.

    시그니처는 5-tuple 언팩 caller 호환.
    coin_universe / all_prices / is_today는 현재 미사용(engine이 자체 데이터 사용).
    """
    date_str = target_date.date() if hasattr(target_date, 'date') else target_date
    log.append(f"<h3>🪙 코인 포트폴리오 (V23: 1D 단일 D_SMA42 sn=217×7 drift=0.10) ({date_str})</h3>")
    meta = {'signal_dist': {}, 'next_candidates': []}

    state, path = _load_v20_state_personal()
    if state is None:
        log.append("<p class='error'>V23 상태 파일(trade_state.json)을 찾을 수 없습니다. executor가 아직 실행되지 않았을 수 있습니다.</p>")
        return {CASH_ASSET: 1.0}, "상태 로드 실패", meta, log, []

    log.append(f"<p class='info'>상태: {path} · 마지막 실행 {state.get('last_run_ts', 'N/A')}</p>")

    members = state.get('members', {}) or {}
    excluded = state.get('excluded_coins', {}) or {}
    last_member_targets = state.get('last_member_targets', {}) or {}
    combined_snap = state.get('last_target_snapshot', {}) or {}

    # 멤버별 상태 테이블
    mrows = []
    healthy_union = []
    for m_name in ('D_SMA42',):  # V23: 단일 멤버
        m_st = members.get(m_name, {})
        m_tgt = {k: v for k, v in (last_member_targets.get(m_name, {}) or {}).items() if k != '_ts'}
        m_ex = list((excluded.get(m_name, {}) or {}).keys())
        picks = [k for k in m_tgt if k not in ('Cash', 'CASH')]
        healthy_union.extend(picks)
        inv_pct = sum(v for k, v in m_tgt.items() if k not in ('Cash', 'CASH'))
        mrows.append({
            '멤버': m_name,
            '카나리': '🟢 On' if m_st.get('canary_on') else '🔴 Off',
            '마지막 봉(UTC)': m_st.get('last_bar_ts', 'N/A'),
            '스냅 ID': m_st.get('snap_id', '-'),
            'Picks': ', '.join(picks) if picks else 'Cash 100%',
            '투자 비중': f"{inv_pct*100:.1f}%",
            '제외': ', '.join(m_ex) if m_ex else '-',
        })
    try:
        log.append(f"<div class='table-wrap'>{pd.DataFrame(mrows).to_html(classes='dataframe small-table', index=False)}</div>")
    except Exception:
        pass

    # 결합 타겟 (1/2씩 EW, V23)
    weights = {k: v for k, v in combined_snap.items() if k != '_ts'}
    if not weights:
        weights = {CASH_ASSET: 1.0}

    # Cash 키 정규화 (엔진은 'Cash' 사용)
    if 'CASH' in weights and 'Cash' not in weights:
        weights[CASH_ASSET] = weights.pop('CASH')

    w_rows = [{'자산': t, '비중': f"{w*100:.2f}%"} for t, w in sorted(weights.items(), key=lambda x: -x[1])]
    try:
        log.append("<p><b>[결합 타겟]</b> V23 단일 멤버 D_SMA42 (executor가 이 비중으로 실행)</p>")
        log.append(f"<div class='table-wrap'>{pd.DataFrame(w_rows).to_html(classes='dataframe small-table', index=False)}</div>")
    except Exception:
        pass

    # 카나리 상태 요약 (signal_dist meta 채우기 — 기존 UI 호환)
    d42_canary = members.get('D_SMA42', {}).get('canary_on')
    h4_canary = None  # V23: H4 멤버 제거
    meta['signal_dist'] = {
        'D_SMA42_canary': d42_canary,
        'H4_SMA240_canary': h4_canary,
    }

    invested = sum(v for k, v in weights.items() if k != CASH_ASSET)
    cash_pct = weights.get(CASH_ASSET, 0.0)
    if cash_pct >= 0.99:
        stat = "Risk-Off (Cash 100%)"
    elif cash_pct > 0.01:
        stat = f"투자 {invested:.0%} / 현금 {cash_pct:.0%}"
    else:
        stat = "Full Invest"

    # Deduplicate healthy_union while preserving order
    seen = set()
    healthy_union_unique = [c for c in healthy_union if not (c in seen or seen.add(c))]
    return weights, stat, meta, log, healthy_union_unique

def save_html(log_global, final_port, s_port, c_port, s_stat, c_stat, turnover, log_today, log_yesterday, date_today, asset_prices_krw, s_meta, c_meta, coin_health_status, cur_assets_raw=None, action_guide="", diff_table_rows=None, coin_total_krw=0):
    filepath = PORTFOLIO_HTML_NAME

    # 현금 버퍼 (trade_state.json)
    cash_buffer_pct = get_cash_buffer()
    try:
        pass
    except Exception:
        pass

    items = []
    for t, w in final_port.items(): items.append({'종목': t, '자산군': "현금" if t == CASH_ASSET else ("코인" if t in c_port else "주식"), '비중': w})
    items.sort(key=lambda x: (x['자산군']!='현금', x['비중']), reverse=True)
    
    tbody = "".join([f"<tr><td>{i['종목']}</td><td>{i['자산군']}</td><td>{i['비중']:.2%}</td></tr>" for i in items])
    
    # [Table] Integrated Portfolio (My vs Target)
    integrated_html = ""
    if diff_table_rows:
        integrated_html = f"<h3>Turnover: {turnover:.2%} ({action_guide})</h3>"
        integrated_html += "<table class='mobile-card-table'><thead><tr><th>Asset</th><th>My</th><th>Target</th><th>Diff</th><th>Action</th></tr></thead><tbody>"
        
        total_value_sum = sum(item['Value'] for item in diff_table_rows)
        
        for row in diff_table_rows:
            color = ""
            if "BUY" in row['Action']: color = "color:red; font-weight:bold;"
            elif "SELL" in row['Action']: color = "color:blue; font-weight:bold;"
            
            val_fmt = f"{int(row['Value']):,}" if row['Value'] > 0 else "0"
            
            integrated_html += f"<tr><td data-label='Asset'>{row['Asset']}</td><td data-label='My'>{row['My']:.1%}</td><td data-label='Target'>{row['Target']:.1%}</td><td data-label='Diff'>{row['Diff']:+.1%}</td><td data-label='Action' style='{color}'>{row['Action']}</td></tr>"
        
        integrated_html += f"<tr><td data-label='Total' style='font-weight:bold;'>Total</td><td data-label='My'></td><td data-label='Target'></td><td data-label='Diff'></td><td data-label='Action'></td></tr>"
        integrated_html += "</tbody></table>"

    # Strategy documentation link
    version_html = ""

    # Embed recommended stock tickers + prices for client-side calculation
    rec_stock_list = sorted([t for t in s_port.keys() if t != 'Cash'])
    rec_stock_json = json.dumps(rec_stock_list)

    # ETF 종가(USD) 임베딩 — 주수 계산용
    stock_prices_for_js = {}
    for t in rec_stock_list:
        p = prices.get(t)
        if p is not None and not p.empty:
            stock_prices_for_js[t] = round(float(p.iloc[-1]), 2)
    stock_prices_json = json.dumps(stock_prices_for_js)

    # Read signal state for UI
    signal_flipped = False
    current_risk_on = True
    saved_stock_holdings = []
    try:
        with open(SIGNAL_STATE_FILE, 'r') as _sf:
            _state = json.load(_sf)
            signal_flipped = False  # 플립 감지는 executor가 담당
            current_risk_on = _state.get('stock', {}).get('risk_on', True)
            saved_stock_holdings = _state.get('stock', {}).get('offense_picks', [])
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    saved_holdings_json = json.dumps(saved_stock_holdings)
    coin_total_krw_val = int(coin_total_krw)
    stock_holdings_js = """
            <script>
            const REC_STOCK_TICKERS = """ + rec_stock_json + """;
            const STOCK_PRICES_USD = """ + stock_prices_json + """;
            const COIN_TOTAL_KRW = """ + str(coin_total_krw_val) + """;
            const TARGET_STOCK_RATIO = 0.65;
            const TARGET_COIN_RATIO = 0.20;
            const TARGET_FUTURES_RATIO = 0.15;
            const REBAL_HT = 0.13;           // V23 갱신 (2026-05-22): T1 = half_turnover ≥ 13pp
            const REBAL_T3U_REL = 0.20;      // V23 추가 (2026-05-22): T3U_can = max rel underweight ≥ 20% + sleeve canary ON
            const SIGNAL_FLIPPED = """ + ("true" if signal_flipped else "false") + """;
            const RISK_ON = """ + ("true" if current_risk_on else "false") + """;
            const SAVED_STOCK_HOLDINGS = """ + saved_holdings_json + """;  // signal_state.json에서 로드

            function calcTrigger(myTickers, recTickers) {
                if (!myTickers.length || !recTickers.length) return null;
                const allT = [...new Set([...myTickers, ...recTickers])];
                let totalDiff = 0;
                allT.forEach(t => {
                    const myW = myTickers.includes(t) ? 1.0/myTickers.length : 0;
                    const recW = recTickers.includes(t) ? 1.0/recTickers.length : 0;
                    totalDiff += Math.abs(recW - myW);
                });
                const turnover = Math.round(totalDiff / 2 * 10000) / 10000;

                const added = recTickers.filter(t => !myTickers.includes(t));
                const removed = myTickers.filter(t => !recTickers.includes(t));
                return { turnover, added, removed };
            }

            function renderTrigger(myTickers) {
                const result = calcTrigger(myTickers, REC_STOCK_TICKERS);
                const el = document.getElementById('triggerResult');
                if (!el || !result) return;

                let changesHtml = '';
                if (result.added.length)
                    changesHtml += '<span style="color:#d93025; font-weight:600;">+ ' + result.added.join(', ') + '</span> ';
                if (result.removed.length)
                    changesHtml += '<span style="color:#1a73e8; font-weight:600;">- ' + result.removed.join(', ') + '</span>';
                if (!result.added.length && !result.removed.length)
                    changesHtml = '<span style="color:#0d904f;">\\u2705 \\ub3d9\\uc77c \\uc885\\ubaa9</span>';

                const pct = Math.round(result.turnover * 100);
                const regime = RISK_ON ? 'Risk-On &#x1F7E2;' : 'Risk-Off &#x1F534;';
                let statusHtml = '';
                if (SIGNAL_FLIPPED) {
                    statusHtml = '<div style="background:#fce8e6; border:2px solid #d93025; padding:12px; border-radius:8px; margin-top:10px;">'
                        + '&#x1F6A8; <b>Signal Flip \ubc1c\uc0dd (' + regime + ') \u2014 \uc989\uc2dc \ub9ac\ubc38\ub7f0\uc2f1 \ud544\uc694</b></div>';
                } else if (pct === 0) {
                    statusHtml = '<div style="background:#e8f5e9; padding:12px; border-radius:8px; margin-top:10px;">'
                        + '\u2705 ' + regime + ' | \ud604\uc7ac \ubcf4\uc720 = \ucd94\ucc9c \uc885\ubaa9 (\ub9ac\ubc38\ub7f0\uc2f1 \ubd88\ud544\uc694)</div>';
                } else if (result.added.length >= 2) {
                    statusHtml = '<div style="background:#fce8e6; border:2px solid #d93025; padding:12px; border-radius:8px; margin-top:10px;">'
                        + '&#x1F6A8; ' + regime + ' | <b>' + result.added.length + '\uc885\ubaa9 \ubcc0\uacbd \u2014 \ud2b8\ub9ac\uac70 \ub9ac\ubc38\ub7f0\uc2f1</b></div>';
                } else {
                    statusHtml = '<div style="background:#fff3e0; border:1px solid #ff9800; padding:12px; border-radius:8px; margin-top:10px;">'
                        + '&#x1F504; ' + regime + ' | ' + result.added.length + '\uc885\ubaa9 \ubcc0\uacbd \u2014 <b>\uc6d4\ucd08 \ub9ac\ubc38\ub7f0\uc2f1 \uc2dc \ubc18\uc601</b></div>';
                }

                el.innerHTML = '<div style="margin-top: 10px;">'
                    + '<div style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:8px;">'
                    + '<div><b>My:</b> ' + myTickers.join(', ') + ' (' + myTickers.length + ')</div>'
                    + '<div><b>Rec:</b> ' + REC_STOCK_TICKERS.join(', ') + ' (' + REC_STOCK_TICKERS.length + ')</div>'
                    + '</div>'
                    + '<div style="margin-bottom:8px;">Changes: ' + changesHtml + '</div>'
                    + statusHtml
                    + '</div>';
            }

            // Load on page init: localStorage 우선, 없으면 signal_state.json fallback
            (function() {
                try {
                    const saved = localStorage.getItem('cap_defend_stock_holdings');
                    if (saved) {
                        const data = JSON.parse(saved);
                        if (data.tickers && data.tickers.length > 0) {
                            document.getElementById('stockInput').value = data.tickers.join(' ');
                            document.getElementById('holdingsStatus').innerHTML =
                                '\u2705 \uc800\uc7a5\ub428: ' + data.tickers.join(', ') + ' (' + data.updated + ')';
                            renderTrigger(data.tickers);
                            return;
                        }
                    }
                    // Fallback: signal_state.json에서 가져온 값
                    if (SAVED_STOCK_HOLDINGS && SAVED_STOCK_HOLDINGS.length > 0) {
                        document.getElementById('stockInput').value = SAVED_STOCK_HOLDINGS.join(' ');
                        document.getElementById('holdingsStatus').innerHTML =
                            '\u2705 \uc11c\ubc84 \uc800\uc7a5\uac12: ' + SAVED_STOCK_HOLDINGS.join(', ');
                        renderTrigger(SAVED_STOCK_HOLDINGS);
                    }
                } catch(e) {}
            })();

            function saveHoldings() {
                const input = document.getElementById('stockInput').value.trim();
                const status = document.getElementById('holdingsStatus');
                if (!input) {
                    status.innerHTML = '\u274c \uc885\ubaa9\uc744 \uc785\ub825\ud574\uc8fc\uc138\uc694';
                    status.style.color = '#d93025';
                    return;
                }
                const tickers = input.toUpperCase().split(/\\s+/).filter(t => t.length > 0);
                const now = new Date().toLocaleString('ko-KR');
                const data = { tickers: tickers, updated: now };
                try {
                    localStorage.setItem('cap_defend_stock_holdings', JSON.stringify(data));
                    status.innerHTML = '\u2705 \uc800\uc7a5 \uc644\ub8cc: ' + tickers.join(', ');
                    status.style.color = '#0d904f';
                    renderTrigger(tickers);
                } catch(e) {
                    status.innerHTML = '\u274c \uc800\uc7a5 \uc2e4\ud328';
                    status.style.color = '#d93025';
                }
            }

            // === 통합 리밸런싱 계산기 (3자산: 주식/현물/선물) ===
            let liveCoinKRW = COIN_TOTAL_KRW;
            let liveFuturesKRW = 0;
            let coinFetchTime = '리포트 생성 시점';

            async function fetchCoinBalance() {
                const statusEl = document.getElementById('coinFetchStatus');
                try {
                    statusEl.innerHTML = '\\u23f3 \\ucf54\\uc778 \\uc794\\uace0 \\uc870\\ud68c \\uc911...';
                    statusEl.style.color = '#1967d2';
                    const resp = await fetch('http://' + window.location.hostname + ':5000/api/assets/coin_balance');
                    if (!resp.ok) throw new Error('API error');
                    const data = await resp.json();
                    liveCoinKRW = data.total_krw;
                    coinFetchTime = data.updated;
                    statusEl.innerHTML = '\\u2705 \\uc2e4\\uc2dc\\uac04 \\uc870\\ud68c \\uc644\\ub8cc (' + data.updated + ')';
                    statusEl.style.color = '#0d904f';
                    return true;
                } catch(e) {
                    statusEl.innerHTML = '\\u26a0\\ufe0f \\uc870\\ud68c \\uc2e4\\ud328 \\u2014 \\ub9ac\\ud3ec\\ud2b8 \\uc2dc\\uc810 \\uac12 \\uc0ac\\uc6a9';
                    statusEl.style.color = '#e37400';
                    return false;
                }
            }

            async function fetchFuturesBalance() {
                try {
                    const resp = await fetch('http://' + window.location.hostname + ':5000/api/assets/binance_balance');
                    if (!resp.ok) throw new Error('API error');
                    const data = await resp.json();
                    liveFuturesKRW = data.total_krw || 0;
                    return true;
                } catch(e) {
                    return false;
                }
            }

            async function calcRebalance() {
                // 주식 + 현물코인 + 선물 + 환율 동시 조회
                await Promise.all([fetchCoinBalance(), fetchFuturesBalance(), fetchStockBalance()]);

                const stockInput = String(getStockTotal());
                const resultEl = document.getElementById('rebalResult');

                const stockKRW = parseFloat(stockInput);
                const coinKRW = parseFloat(document.getElementById('snapCoin').value) || liveCoinKRW;
                const futuresKRW = parseFloat(document.getElementById('snapFutures').value) || liveFuturesKRW;
                const rate = parseFloat(document.getElementById('exchangeRate').value) || 0;
                if (!document.getElementById('snapCoin').value) document.getElementById('snapCoin').value = Math.round(coinKRW);
                if (!document.getElementById('snapFutures').value && liveFuturesKRW > 0) document.getElementById('snapFutures').value = Math.round(liveFuturesKRW);

                if (!stockKRW && stockKRW !== 0) { resultEl.innerHTML = '\\u274c \\uc8fc\\uc2dd \\ucd1d\\uc561\\uc744 \\uc785\\ub825\\ud574\\uc8fc\\uc138\\uc694'; return; }

                // 1. 3자산 비중 계산
                const totalAsset = stockKRW + coinKRW + futuresKRW;
                if (totalAsset <= 0) { resultEl.innerHTML = '\\u274c \\uc790\\uc0b0\\uc774 0\\uc785\\ub2c8\\ub2e4'; return; }

                const curStockPct = stockKRW / totalAsset;
                const curCoinPct = coinKRW / totalAsset;
                const curFuturesPct = futuresKRW / totalAsset;

                const driftStock = Math.abs(curStockPct - TARGET_STOCK_RATIO);
                const driftCoin = Math.abs(curCoinPct - TARGET_COIN_RATIO);
                const driftFutures = Math.abs(curFuturesPct - TARGET_FUTURES_RATIO);
                const halfTurnover = (driftStock + driftCoin + driftFutures) / 2;
                // T3U_can: max relative underweight per asset (canary ON 가정 — daily report 가 정확한 gate)
                const relUnderStock = Math.max(0, (TARGET_STOCK_RATIO - curStockPct) / TARGET_STOCK_RATIO);
                const relUnderCoin = Math.max(0, (TARGET_COIN_RATIO - curCoinPct) / TARGET_COIN_RATIO);
                const relUnderFut = Math.max(0, (TARGET_FUTURES_RATIO - curFuturesPct) / TARGET_FUTURES_RATIO);
                const maxRelUnder = Math.max(relUnderStock, relUnderCoin, relUnderFut);
                const t1Fire = halfTurnover >= REBAL_HT;
                const t3uFire = maxRelUnder >= REBAL_T3U_REL;  // canary ON 가정
                const needRebal = t1Fire || t3uFire;

                // 2. 목표 금액
                const targetStockKRW = totalAsset * TARGET_STOCK_RATIO;
                const targetCoinKRW = totalAsset * TARGET_COIN_RATIO;
                const targetFuturesKRW = totalAsset * TARGET_FUTURES_RATIO;

                // 3. 비중 카드 (3열, half_turnover 기준)
                const htColor = needRebal ? '#d93025' : '#0d904f';
                function assetCard(label, krw, cur, target) {
                    return '<div style="padding:14px; background:#f8f9fa; border-radius:10px;">'
                        + '<div style="font-size:0.85em; color:#666;">' + label + '</div>'
                        + '<div style="font-size:1.25em; font-weight:700;">' + Math.round(krw).toLocaleString() + '\\uc6d0</div>'
                        + '<div style="color:#444; font-weight:600;">'
                        + (cur * 100).toFixed(1) + '% (\\ubaa9\\ud45c ' + (target * 100).toFixed(0) + '%)</div></div>';
                }

                let html = '<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:16px;">';
                html += assetCard('\\uc8fc\\uc2dd', stockKRW, curStockPct, TARGET_STOCK_RATIO);
                html += assetCard('\\ud604\\ubb3c\\ucf54\\uc778', coinKRW, curCoinPct, TARGET_COIN_RATIO);
                html += assetCard('\\uc120\\ubb3c', futuresKRW, curFuturesPct, TARGET_FUTURES_RATIO);
                html += '</div>';

                html += '<div style="padding:8px 14px; background:#e8eaf6; border-radius:8px; margin-bottom:12px;">'
                    + '<b>\\ucd1d \\uc790\\uc0b0:</b> ' + Math.round(totalAsset).toLocaleString() + '\\uc6d0'
                    + ' &nbsp;|&nbsp; <span style="color:' + htColor + ';font-weight:600;">T1 ht ' + (halfTurnover * 100).toFixed(1) + 'pp / ' + (REBAL_HT * 100).toFixed(0) + 'pp</span>'
                    + ' &nbsp;|&nbsp; <span style="color:' + (t3uFire?'#d93025':'#0d904f') + ';font-weight:600;">T3U_can max-under ' + (maxRelUnder * 100).toFixed(0) + '% / ' + (REBAL_T3U_REL * 100).toFixed(0) + '%</span>'
                    + '</div>';

                // 4. 리밸런싱 판단
                if (needRebal) {
                    let moves = [];
                    const deltas = [
                        {name: '\\uc8fc\\uc2dd', delta: targetStockKRW - stockKRW},
                        {name: '\\ud604\\ubb3c', delta: targetCoinKRW - coinKRW},
                        {name: '\\uc120\\ubb3c', delta: targetFuturesKRW - futuresKRW},
                    ];
                    deltas.forEach(d => {
                        if (Math.abs(d.delta) > 10000) {
                            const sign = d.delta > 0 ? '+' : '';
                            moves.push(d.name + ' ' + sign + Math.round(d.delta).toLocaleString() + '\\uc6d0');
                        }
                    });
                    let reasons = [];
                    if (t1Fire) reasons.push('T1(ht ' + (halfTurnover*100).toFixed(1) + 'pp ≥ ' + (REBAL_HT*100).toFixed(0) + 'pp)');
                    if (t3uFire) reasons.push('T3U_can(max-under ' + (maxRelUnder*100).toFixed(0) + '% ≥ ' + (REBAL_T3U_REL*100).toFixed(0) + '% — canary 확인 필요)');
                    html += '<div style="padding:14px; background:#fce8e6; border:2px solid #d93025; border-radius:10px; margin-bottom:16px;">'
                        + '<div style="font-size:1.1em; font-weight:700; color:#d93025; margin-bottom:6px;">\\u26a0\\ufe0f \\ub9ac\\ubc38\\ub7f0\\uc2f1 \\ud544\\uc694 — ' + reasons.join(' | ') + '</div>'
                        + '<div style="font-size:1.05em;">' + moves.join('<br>') + '</div>'
                        + '</div>';
                } else {
                    html += '<div style="padding:14px; background:#e8f5e9; border:1px solid #0d904f; border-radius:10px; margin-bottom:16px;">'
                        + '<div style="font-size:1.1em; font-weight:700; color:#0d904f;">\\u2705 \\ub9ac\\ubc38\\ub7f0\\uc2f1 \\ubd88\\ud544\\uc694 (T1 ' + (halfTurnover*100).toFixed(1) + 'pp / T3U_can ' + (maxRelUnder*100).toFixed(0) + '%)</div>'
                        + '</div>';
                }

                // 5. ETF 주수 계산 (환율 입력 시만)
                const finalStockKRW = needRebal ? targetStockKRW : stockKRW;
                const tickers = REC_STOCK_TICKERS.filter(t => t in STOCK_PRICES_USD);
                if (tickers.length > 0 && rate > 0) {
                    const perETF = finalStockKRW / tickers.length;
                    let rows = '';
                    let totalUsed = 0;

                    tickers.forEach(t => {
                        const priceUSD = STOCK_PRICES_USD[t];
                        const priceKRW = priceUSD * rate;
                        const shares = Math.floor(perETF / priceKRW);
                        const usedKRW = shares * priceKRW;
                        totalUsed += usedKRW;
                        const kisQty = kisHoldings[t] || 0;
                        const shinhanQty = Math.max(0, shares - kisQty);
                        rows += '<tr>'
                            + '<td data-label="ETF" style="font-weight:600;">' + t + '</td>'
                            + '<td data-label="\\uac00\\uaca9">$' + priceUSD.toFixed(2) + '</td>'
                            + '<td data-label="\\ucd1d\\uc8fc\\uc218" style="font-size:1.2em;font-weight:700;color:#1a73e8;">' + shares + '\\uc8fc</td>'
                            + '<td data-label="\\ud55c\\ud22c">' + kisQty + '\\uc8fc</td>'
                            + '<td data-label="\\uc2e0\\ud55c">' + shinhanQty + '\\uc8fc</td>'
                            + '<td data-label="\\uae08\\uc561">' + Math.round(usedKRW).toLocaleString() + '\\uc6d0</td>'
                            + '</tr>';
                    });

                    const remainder = finalStockKRW - totalUsed;
                    html += '<h3 style="margin:16px 0 8px 0;">\\uc8fc\\uc2dd ETF \\ubcf4\\uc720 \\uac00\\uc774\\ub4dc' + (needRebal ? ' (\\ub9ac\\ubc38\\ub7f0\\uc2f1 \\ud6c4)' : '') + '</h3>';
                    html += '<table class="mobile-card-table">'
                        + '<thead><tr><th>ETF</th><th>\\uac00\\uaca9</th><th>\\ucd1d\\uc8fc\\uc218</th><th>\\ud55c\\ud22c</th><th>\\uc2e0\\ud55c</th><th>\\uae08\\uc561</th></tr></thead>'
                        + '<tbody>' + rows + '</tbody></table>';
                    html += '<div style="margin-top:10px;padding:10px;background:#f0f4ff;border-radius:8px;">'
                        + '<b>\\uc8fc\\uc2dd \\ud22c\\uc790:</b> ' + Math.round(finalStockKRW).toLocaleString() + '\\uc6d0'
                        + ' | <b>\\uc2e4\\uc81c \\ub9e4\\uc218:</b> ' + Math.round(totalUsed).toLocaleString() + '\\uc6d0'
                        + ' | <b>\\uc794\\uc5ec:</b> ' + Math.round(remainder).toLocaleString() + '\\uc6d0'
                        + '</div>';
                }

                if (tickers.length > 0 && rate <= 0) {
                    html += '<p style="color:#888; font-size:0.9em;">\\ud658\\uc728\\uc744 \\uc785\\ub825\\ud558\\uba74 ETF \\uc8fc\\uc218\\ub3c4 \\uacc4\\uc0b0\\ud569\\ub2c8\\ub2e4.</p>';
                }

                resultEl.innerHTML = html;
            }
            </script>
    """

    # 자산관리 UI는 실계좌 조회 전용으로 교체
    stock_holdings_js = ""

    # 실행 상태 요약 (상세 로그용) — 주식/현물/선물 공통 템플릿
    def _fmt_alloc(d: dict) -> str:
        if not isinstance(d, dict) or not d:
            return "없음"
        return ", ".join(
            f"{k}:{float(v):.1%}"
            for k, v in sorted(d.items(), key=lambda kv: float(kv[1]), reverse=True)
        )

    def _fmt_alloc_lines(d: dict) -> str:
        if not isinstance(d, dict) or not d:
            return "없음"
        return "<br>".join(
            f"{k}: {float(v):.1%}"
            for k, v in sorted(d.items(), key=lambda kv: float(kv[1]), reverse=True)
        )

    def _fmt_bool(v) -> str:
        return "예" if bool(v) else "아니오"

    def _load_first_json(candidates):
        for cand in candidates:
            if cand and os.path.exists(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        return json.load(f), cand
                except Exception:
                    continue
        return {}, ""

    def _card_list_html(rows, columns=None):
        if not rows:
            return "<p>기록 없음</p>"
        try:
            if columns:
                use_cols = columns
            else:
                use_cols = list(rows[0].keys())
            cards = []
            for row in rows:
                parts = []
                for col in use_cols:
                    val = row.get(col, "-")
                    parts.append(
                        "<div class='state-row'>"
                        f"<div class='state-label'>{col}</div>"
                        f"<div class='state-value'>{val}</div>"
                        "</div>"
                    )
                cards.append("<div class='state-card'>" + "".join(parts) + "</div>")
            return "<div class='state-cards'>" + "".join(cards) + "</div>"
        except Exception:
            return "<p>표 생성 실패</p>"

    def _strategy_block(title: str, summary_rows: list, tranche_rows: list, extra_html: str = "") -> str:
        parts = [f"<h2>{title}</h2>"]
        if summary_rows:
            parts.append("<div class='summary-list'>")
            for row in summary_rows:
                parts.append(f"<div class='summary-item'>{row}</div>")
            parts.append("</div>")
        if tranche_rows:
            parts.append("<h3>트랜치 상태</h3>")
            parts.append(_card_list_html(tranche_rows))
        if extra_html:
            parts.append(extra_html)
        return "".join(parts)

    def _merge_tranches(tranches: dict) -> dict:
        merged = {}
        if not isinstance(tranches, dict) or not tranches:
            return merged
        n = len(tranches)
        for tr in tranches.values():
            for ticker, weight in (tr.get('weights', {}) or {}).items():
                merged[ticker] = merged.get(ticker, 0.0) + float(weight) / n
        return {k: v for k, v in merged.items() if v > 0}

    def _merge_stock_state(state: dict) -> dict:
        """V23: snapshots 우선, 없으면 V21 tranches 폴백."""
        snapshots = state.get('snapshots', {}) or {}
        if snapshots:
            return _merge_tranches(snapshots)
        return _merge_tranches(state.get('tranches', {}) or {})

    def _next_anchor_str(anchor_days) -> str:
        now = datetime.now()
        year, month = now.year, now.month
        candidates = []
        for day in anchor_days:
            try:
                candidates.append(datetime(year, month, day))
            except ValueError:
                continue
        future = [dt for dt in candidates if dt.date() >= now.date()]
        if future:
            nxt = min(future)
        else:
            if month == 12:
                nxt = datetime(year + 1, 1, anchor_days[0])
            else:
                nxt = datetime(year, month + 1, anchor_days[0])
        return nxt.strftime('%Y-%m-%d')

    _this_month = datetime.now().strftime('%Y-%m')

    def _tranche_status(anchor_month: str, weights: dict, picks: list) -> Tuple[str, str, str]:
        has_alloc = isinstance(weights, dict) and len(weights) > 0
        has_picks = isinstance(picks, list) and len(picks) > 0
        if has_alloc or has_picks:
            if has_alloc and set(weights.keys()) == {'Cash'} and float(weights.get('Cash', 0)) >= 0.999:
                return "현금 유지", "없음", _fmt_alloc(weights)
            return "활성", ", ".join(picks or []) or "없음", _fmt_alloc(weights)
        if anchor_month and anchor_month < _this_month:
            return "미초기화", "다음 앵커 대기", "다음 앵커 대기"
        return "대기", "없음", "없음"

    def _next_tranche_refresh_str(last_bar_ts: str, interval_hours: int, bar_counter, snap_iv: int, n_snap: int) -> str:
        if not last_bar_ts or last_bar_ts == "-" or snap_iv <= 0 or n_snap <= 0:
            return "-"
        try:
            dt = pd.to_datetime(last_bar_ts)
            # Futures state stores bar timestamps as naive UTC strings.
            if getattr(dt, 'tzinfo', None) is None:
                dt = dt.tz_localize("UTC")
            c = int(bar_counter)
            offsets = [int(si * snap_iv / n_snap) for si in range(n_snap)]
            # state['bar_counter'] is incremented after the latest completed bar is processed.
            # The next refresh happens on the first future bar where the pre-increment counter
            # matches one of the tranche offsets.
            next_n = None
            for n in range(0, snap_iv + 1):
                if ((c + n) % snap_iv) in offsets:
                    next_n = n
                    break
            if next_n is None:
                next_n = 0
            nxt = dt + pd.Timedelta(hours=interval_hours * (next_n + 1))
            nxt = pd.Timestamp(nxt).tz_convert("Asia/Seoul")
            return nxt.strftime('%Y-%m-%d %H:%M:%S KST')
        except Exception:
            return "-"

    def _fmt_run_ts(ts: str) -> str:
        if not ts or ts == "-":
            return "-"
        try:
            dt = pd.to_datetime(ts)
            if getattr(dt, "tzinfo", None) is None:
                return pd.Timestamp(dt).strftime('%Y-%m-%d %H:%M:%S')
            dt = dt.tz_convert("Asia/Seoul")
            return pd.Timestamp(dt).strftime('%Y-%m-%d %H:%M:%S KST')
        except Exception:
            return str(ts)

    _base_dir = globals().get("BASE_DIR", os.getcwd())
    state_sections = []

    # ── 주식 실행 상태 ──
    try:
        _stock_state, _stock_path = _load_first_json([
            os.path.join(APP_HOME, "kis_trade_state.json"),
            os.path.join(_base_dir, "trade", "kis_trade_state.json"),
            "kis_trade_state.json",
        ])
        _stock_signal, _ = _load_first_json([
            SIGNAL_STATE_FILE,
            os.path.join(_base_dir, SIGNAL_STATE_FILE),
        ])
        _stk = _stock_signal.get("stock", {}) or {}
        _stock_exec_target = _merge_stock_state(_stock_state)
        _stock_summary = [
            f"<b>리밸런싱 대기:</b> {_fmt_bool(_stock_state.get('rebalancing_needed'))}",
            f"<b>마지막 실행:</b> {_stock_state.get('last_trade_date', '-')}",
            f"<b>합산 목표:</b><br>{_fmt_alloc_lines(_stock_exec_target)}",
            f"<b>전략 KIS_V23:</b> Canary {'ON' if _stk.get('risk_on', True) else 'OFF'}, "
            f"다음 트랜치 {_next_anchor_str(STOCK_ANCHOR_DAYS)}",
        ]
        _stock_tr_rows = []
        _v22_snaps = _stock_state.get("snapshots", {}) or {}
        if _v22_snaps:
            for _sid in sorted(_v22_snaps.keys(), key=lambda x: int(x)):
                _sn = _v22_snaps.get(_sid, {}) or {}
                _w = _sn.get("weights", {}) or {}
                _p = _sn.get("picks", []) or []
                _status, _picks_text, _weights_text = _tranche_status(
                    _sn.get("last_rebal_date", "-"), _w, _p,
                )
                _stock_tr_rows.append({
                    "스냅": f"S{_sid}",
                    "상태": _status,
                    "종목": _picks_text,
                    "비중": _weights_text,
                })
        else:
            for _anchor in sorted((_stock_state.get("tranches", {}) or {}).keys(), key=lambda x: int(x)):
                _tr = (_stock_state.get("tranches", {}) or {}).get(_anchor, {}) or {}
                _status, _picks_text, _weights_text = _tranche_status(
                    _tr.get("anchor_month", "-"),
                    _tr.get("weights", {}),
                    _tr.get("picks", []),
                )
                _stock_tr_rows.append({
                    "트랜치": f"D{_anchor}",
                    "상태": _status,
                    "기준월": _tr.get("anchor_month", "-"),
                    "종목": _picks_text,
                    "비중": _weights_text,
                })
        state_sections.append(_strategy_block("📘 주식 실행 상태", _stock_summary, _stock_tr_rows))
    except Exception as _e:
        state_sections.append(f"<h2>📘 주식 실행 상태</h2><p class='error'>상태 조회 실패: {_e}</p>")

    # ── 현물 코인 실행 상태 (V23 앙상블) ──
    try:
        _coin_state, _coin_path = _load_first_json([
            os.path.join(APP_HOME, "trade_state.json"),
            os.path.join(_base_dir, "trade", "trade_state.json"),
            "trade_state.json",
        ])
        _members = _coin_state.get("members", {}) or {}
        _last_target = _coin_state.get("last_target_snapshot", {}) or {}
        _last_target_clean = {k: v for k, v in _last_target.items() if not str(k).startswith('_')}
        _excl_all = _coin_state.get("excluded_coins", {}) or {}
        _upbit_status = _coin_state.get("last_upbit_status", {}) or {}
        _warning_coins = sorted((_upbit_status.get("warning", []) or []))
        _coin_summary = [
            f"<b>리밸런싱 대기:</b> {_fmt_bool(_coin_state.get('rebalancing_needed'))}",
            f"<b>마지막 실행:</b> {_fmt_run_ts(_coin_state.get('last_run_ts', '-'))}",
            f"<b>합산 목표:</b><br>{_fmt_alloc_lines(_last_target_clean)}",
        ]
        if _warning_coins:
            _coin_summary.append(f"<b>Upbit 유의/상폐:</b> {', '.join(_warning_coins)}")
        _coin_rows = []
        for _name in list(COIN_MEMBER_META.keys()):
            _ms = _members.get(_name, {}) or {}
            _meta = COIN_MEMBER_META.get(_name, {})
            _ex = (_excl_all.get(_name, {}) or {})
            _ex_text = ", ".join(sorted(_ex.keys())) if _ex else "없음"
            _coin_summary.append(
                f"<b>전략 {_name}:</b> Canary {'ON' if _ms.get('canary_on') else 'OFF'}, "
                f"다음 트랜치 {_next_tranche_refresh_str(_ms.get('last_bar_ts','-'), _meta.get('interval_hours',1), _ms.get('bar_counter',0), _meta.get('snap_interval_bars',0), _meta.get('n_snapshots',0))}, "
                f"제외 코인 {_ex_text}"
            )
            for _idx, _snap in enumerate((_ms.get("snapshots", []) or []), start=1):
                _w = _snap if isinstance(_snap, dict) else {}
                _picks = [k for k, v in _w.items() if k.lower() != 'cash' and float(v) > 1e-4]
                _status, _picks_text, _weights_text = _tranche_status("-", _w, _picks)
                _coin_rows.append({
                    "스냅": f"S{_idx}",
                    "상태": _status,
                    "종목": _picks_text,
                    "비중": _weights_text,
                })
        state_sections.append(_strategy_block("📘 업비트 실행 상태 (V23)", _coin_summary, _coin_rows))
    except Exception as _e:
        state_sections.append(f"<h2>📘 업비트 실행 상태</h2><p class='error'>상태 조회 실패: {_e}</p>")

    # ── 바이낸스 실행 상태 ──
    try:
        _fut, _fut_path = _load_first_json([
            os.path.join(APP_HOME, "binance_state.json"),
            os.path.join(_base_dir, "trade", "binance_state.json"),
            "binance_state.json",
        ])
        _strategies = _fut.get("strategies", {}) or {}
        _last_target = _fut.get("last_target", {}) or {}
        _fut_summary = [
            f"<b>리밸런싱 대기:</b> {_fmt_bool(_fut.get('rebalancing_needed'))}",
            f"<b>마지막 실행:</b> {_fmt_run_ts(_fut.get('last_run', '-'))}",
            f"<b>합산 목표:</b><br>{_fmt_alloc_lines(_last_target)}",
        ]
        if _fut.get('kill_switch'):
            _fut_summary.append(f"<b>⚠ 비상 정지:</b> {_fut.get('kill_switch_reason', 'unknown')}")
        _fut_tr_rows = []
        for _name in list(FUTURES_TRANCHE_META.keys()):
            _st = _strategies.get(_name, {}) or {}
            _meta = FUTURES_TRANCHE_META.get(_name, {})
            _fut_summary.append(
                f"<b>전략 {_name}:</b> Canary {'ON' if _st.get('canary_on') else 'OFF'}, "
                f"다음 트랜치 {_next_tranche_refresh_str(_st.get('last_bar_ts','-'), _meta.get('interval_hours',1), _st.get('bar_counter',0), _meta.get('snap_interval_bars',0), _meta.get('n_snapshots',0))}"
            )
            for _idx, _snap in enumerate((_st.get("snapshots", []) or []), start=1):
                _w = _snap if isinstance(_snap, dict) else {}
                _picks = [k for k, v in _w.items() if k.lower() != 'cash' and float(v) > 1e-4]
                _status, _picks_text, _weights_text = _tranche_status("-", _w, _picks)
                _fut_tr_rows.append({
                    "스냅": f"S{_idx}",
                    "상태": _status,
                    "종목": _picks_text,
                    "비중": _weights_text,
                })
        state_sections.append(_strategy_block("📘 바이낸스 실행 상태", _fut_summary, _fut_tr_rows))
    except Exception as _e:
        state_sections.append(f"<h2>📘 바이낸스 실행 상태</h2><p class='error'>상태 조회 실패: {_e}</p>")

    # 자산배분 BT 비교 표 제거 (사용자 요청 2026-05-08)

    execution_state_html = "".join(state_sections)

    html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cap Defend {STRATEGY_VERSION} (Personal)</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: -apple-system, sans-serif; background: #f0f2f5; padding: 10px; color: #333; }}
            .container {{ max-width: 800px; margin: 0 auto; background: #fff; padding: 20px; border-radius: 16px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 10px; }}
            th, td {{ padding: 10px; border-bottom: 1px solid #f1f3f4; text-align: left; font-size: 0.95em; }}
            th {{ background-color: #fafafa; font-weight: 600; color: #555; }}
            .card {{ background: #fff; padding: 15px; border-radius: 12px; border: 1px solid #e0e0e0; margin-bottom: 10px; }}
            .dataframe {{ width: 100%; border: 1px solid #ddd; border-collapse: collapse; margin: 10px 0; }}
            .dataframe th, .dataframe td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .dataframe tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .small-table {{ font-size: 0.9em; }}
            .table-wrap {{ overflow-x: auto; }}
            .summary-list {{ display: grid; gap: 6px; margin-bottom: 12px; }}
            .summary-item {{ padding: 8px 10px; background: #f8f9fa; border-radius: 8px; line-height: 1.45; font-size: 0.95em; }}
            .state-cards {{ display: grid; gap: 10px; margin-bottom: 12px; }}
            .state-card {{ border: 1px solid #e6e8eb; border-radius: 10px; padding: 10px 12px; background: #fff; }}
            .state-row {{ display: grid; grid-template-columns: 110px 1fr; gap: 10px; align-items: start; padding: 6px 0; border-bottom: 1px solid #f1f3f4; }}
            .state-row:last-child {{ border-bottom: none; }}
            .state-label {{ color: #5f6368; font-weight: 600; font-size: 0.9em; }}
            .state-value {{ color: #202124; line-height: 1.45; word-break: keep-all; overflow-wrap: anywhere; }}

            /* Collapsible sections */
            .section-header {{
                display: flex; justify-content: space-between; align-items: center;
                cursor: pointer; padding: 14px 16px; margin: 8px 0 0 0;
                background: #f8f9fa; border-radius: 10px; user-select: none;
            }}
            .section-header:hover {{ background: #eef1f5; }}
            .section-header h2 {{ margin: 0; font-size: 1.1em; }}
            .section-header .badge {{ font-size: 0.8em; color: #666; font-weight: 400; }}
            .section-header .arrow {{ transition: transform 0.2s; font-size: 0.8em; color: #999; }}
            .section-body {{ padding: 0 4px; }}
            .section-body.collapsed {{ display: none; }}

            /* Mobile */
            @media screen and (max-width: 600px) {{
                body {{ padding: 4px; }}
                .container {{ padding: 12px; border-radius: 10px; }}
                .mobile-card-table thead {{ display: none; }}
                .mobile-card-table tr {{ display: block; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 8px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
                .mobile-card-table td {{ display: flex; justify-content: space-between; gap: 10px; padding: 10px 12px; border-bottom: 1px solid #eee; text-align: right; font-size: 0.9em; }}
                .mobile-card-table td:last-child {{ border-bottom: none; }}
                .mobile-card-table td::before {{ content: attr(data-label); font-weight: 600; color: #555; text-align: left; flex: 0 0 38%; }}
                .mobile-card-table td:first-child {{ background: #f8f9fa; font-weight: bold; color: #1a73e8; border-radius: 8px 8px 0 0; }}
                .chart-container {{ height: 180px !important; }}
                /* 2열/3열 그리드 → 1열 */
                div[style*="grid-template-columns:1fr 1fr"] {{ grid-template-columns: 1fr !important; }}
                div[style*="grid-template-columns: 1fr 1fr"] {{ grid-template-columns: 1fr !important; }}
                div[style*="grid-template-columns:1fr 1fr 1fr"] {{ grid-template-columns: 1fr !important; }}
                div[style*="grid-template-columns: 1fr 1fr 1fr"] {{ grid-template-columns: 1fr !important; }}
                .section-header h2 {{ font-size: 1em; }}
                .summary-item {{ font-size: 0.9em; padding: 7px 9px; }}
                .table-wrap {{ overflow-x: visible; }}
                .state-card {{ padding: 8px 10px; }}
                .state-row {{ grid-template-columns: 92px 1fr; gap: 8px; padding: 5px 0; }}
                .state-label, .state-value {{ font-size: 0.88em; }}
                #historyTable th, #historyTable td {{ white-space: nowrap; font-size: 0.82em; padding: 8px 6px; }}
                .weights-table tr {{ display: table-row; margin-bottom: 0; border: none; border-radius: 0; background: transparent; box-shadow: none; }}
                .weights-table td {{ display: table-cell; padding: 10px 6px; border-bottom: 1px solid #eee; text-align: left; font-size: 0.88em; white-space: nowrap; }}
                .weights-table td::before {{ content: none; }}
                .weights-table td:first-child {{ background: transparent; font-weight: 600; color: #202124; border-radius: 0; }}
                .weights-table td:last-child {{ text-align: right; font-variant-numeric: tabular-nums; }}
            }}
            .chart-container {{ height: 250px; position: relative; }}
            .weights-table {{ table-layout: fixed; width: 100%; }}
            .weights-table th:first-child, .weights-table td:first-child {{ width: 62%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
            .weights-table th:last-child, .weights-table td:last-child {{ width: 38%; text-align: right; white-space: nowrap; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>\U0001f680 Cap Defend {STRATEGY_VERSION}</h1>
            <p style="color:#666; font-size:0.9em;">{datetime.now().strftime('%Y-%m-%d %H:%M')} | \uc885\uac00 {date_today.strftime('%Y-%m-%d')}</p>

            <!-- ===== 실계좌 현황 (기본 펼침) ===== -->
            <div class="section-header" onclick="toggleSection('secAsset')">
                <h2>\U0001f4bc \uc2e4\uacc4\uc88c \ud604\ud669</h2>
                <span class="badge" id="secAsset_badge"></span>
                <span class="arrow" id="secAsset_arrow">\u25bc</span>
            </div>
            <div class="section-body" id="secAsset">
                <div class="card">
                    <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:12px;">
                        <button onclick="fetchLiveOverview()" style="
                            background:linear-gradient(135deg,#0d904f 0%,#1a73e8 100%);
                            color:white; border:none; padding:12px 24px; border-radius:8px;
                            font-weight:600; font-size:1em; cursor:pointer;">\uc870\ud68c</button>
                        <span id="liveFetchStatus" style="font-size:0.9em; color:#666;"></span>
                    </div>
                    <div id="liveOverviewResult"></div>
                </div>
            </div>

            <!-- ===== 히스토리 (기본 접힘) ===== -->
            <div class="section-header" onclick="toggleSection('secHistory')">
                <h2>\U0001f4c8 \uae30\ub85d / \ucd94\uc774</h2>
                <span class="badge" id="secHistory_badge"></span>
                <span class="arrow" id="secHistory_arrow">\u25b6</span>
            </div>
            <div class="section-body collapsed" id="secHistory">
                <div class="card">
                    <div class="chart-container"><canvas id="chartTotal"></canvas></div>
                </div>
                <div id="historyContainer"></div>
            </div>

<!-- 추천 비중: 매일 확인에 통합됨 -->

            <!-- ===== 상세 로그 (기본 접힘) ===== -->
            <div class="section-header" onclick="toggleSection('secLog')">
                <h2>\U0001f4dc \uc0c1\uc138 \ub85c\uadf8</h2>
                <span class="arrow" id="secLog_arrow">\u25b6</span>
            </div>
            <div class="section-body collapsed" id="secLog">
                {execution_state_html}
            </div>

            {stock_holdings_js}

            <script>
            const API = 'http://' + window.location.hostname + ':5000';
            const fmt = n => (n/1e8).toFixed(2)+'\uc5b5';

            // === Section toggle ===
            function toggleSection(id) {{
                const body = document.getElementById(id);
                const arrow = document.getElementById(id + '_arrow');
                body.classList.toggle('collapsed');
                arrow.textContent = body.classList.contains('collapsed') ? '\u25b6' : '\u25bc';
            }}

            // === Buffer / Force Trade ===
            async function updateBuffer() {{
                const val = document.getElementById('bufferSelect').value;
                const status = document.getElementById('bufferStatus');
                const pwd = prompt('PIN 4\uc790\ub9ac:');
                if (!pwd) return;
                try {{
                    const resp = await fetch(API + '/api/cash_buffer', {{
                        method: 'POST', headers: {{'Content-Type':'application/json'}},
                        body: JSON.stringify({{cash_buffer: parseFloat(val), password: pwd}})
                    }});
                    const d = await resp.json();
                    if (resp.ok) {{ document.getElementById('bufferDisplay').textContent = Math.round((1-parseFloat(val))*100)+'%'; status.innerHTML='\u2705 \ubcc0\uacbd \uc644\ub8cc'; status.style.color='#0d904f'; }}
                    else {{ status.innerHTML='\u26a0\ufe0f '+(d.error||'Error'); status.style.color='#d93025'; }}
                }} catch(e) {{ status.innerHTML='\u274c API \uc2e4\ud328'; status.style.color='#d93025'; }}
                setTimeout(()=>{{status.innerHTML='';}}, 5000);
            }}

            async function forceTrade(exchange) {{
                const btn = document.getElementById('forceTradeUpbitBtn');
                const status = document.getElementById('tradeStatus');
                const pwd = prompt('PIN 4\uc790\ub9ac:');
                if (!pwd) return;
                const amtIn = prompt('\uc6b4\uc6a9 \uae08\uc561 (\uc6d0, 0=\uc804\uccb4):', '0');
                if (amtIn === null) return;
                const amt = parseInt(amtIn.replace(/,/g,'')) || 0;
                const amtText = amt > 0 ? amt.toLocaleString()+'\uc6d0' : '\uc804\uccb4';
                if (!confirm('Force Trade \uc2e4\ud589? ('+amtText+')')) return;
                btn.disabled=true; btn.style.opacity='0.6';
                status.innerHTML='\u23f3 \uc2e4\ud589 \uc911...'; status.style.color='#1967d2';
                try {{
                    const r = await fetch(API+'/api/trade/'+exchange, {{
                        method:'POST', headers:{{'Content-Type':'application/json'}},
                        body: JSON.stringify({{target_amount:amt, password:pwd}})
                    }});
                    const d = await r.json();
                    status.innerHTML = r.ok ? '\u2705 '+d.message : '\u26a0\ufe0f '+(d.error||'Error');
                    status.style.color = r.ok ? '#0d904f' : '#d93025';
                }} catch(e) {{ status.innerHTML='\u274c API \uc2e4\ud328'; status.style.color='#d93025'; }}
                setTimeout(()=>{{btn.disabled=false; btn.style.opacity='1';}}, 5000);
            }}

            function fmtKrwFull(n) {{
                return Math.round(Number(n || 0)).toLocaleString() + '\uc6d0';
            }}

            function fmtUsdFull(n) {{
                return '$' + Number(n || 0).toLocaleString(undefined, {{maximumFractionDigits: 2}});
            }}

            function fmtQty(n) {{
                return Number(n || 0).toLocaleString(undefined, {{maximumFractionDigits: 6}});
            }}

            function renderWeightsTable(title, weights) {{
                const parts = Object.entries(weights || {{}}).map(([ticker, weight]) => ({{
                    ticker,
                    weight: Number(weight || 0) * 100
                }}));
                if (!parts.length) {{
                    return '<div class="card"><h3 style="margin-top:0;">' + title + '</h3><div style="color:#777;">표시할 구성 없음</div></div>';
                }}
                const body = parts
                    .sort((a, b) => {{
                        const aCash = a.ticker === '현금' || a.ticker.toUpperCase() === 'CASH';
                        const bCash = b.ticker === '현금' || b.ticker.toUpperCase() === 'CASH';
                        if (aCash && !bCash) return 1;
                        if (!aCash && bCash) return -1;
                        return b.weight - a.weight;
                    }})
                    .map(row =>
                        '<tr>'
                        + '<td data-label="구성">' + row.ticker + '</td>'
                        + '<td data-label="비중">' + row.weight.toFixed(1) + '%</td>'
                        + '</tr>'
                    )
                    .join('');
                const head = '<tr><th>구성</th><th>계좌 내 비중</th></tr>';
                return '<div class="card"><h3 style="margin-top:0;">' + title + '</h3>'
                    + '<table class="mobile-card-table weights-table"><colgroup><col style="width:62%"><col style="width:38%"></colgroup><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>';
            }}

            function renderHoldingsTable(title, rows) {{
                if (!rows || !rows.length) {{
                    return '<div class="card"><h3 style="margin-top:0;">' + title + '</h3><div style="color:#777;">\ubcf4\uc720 \uc885\ubaa9 \uc5c6\uc74c</div></div>';
                }}
                let body = '';
                rows.forEach(row => {{
                    const price = row.price_krw || row.price || 0;
                    body += '<tr>'
                        + '<td data-label="종목">' + row.ticker + '</td>'
                        + '<td data-label="수량">' + fmtQty(row.qty) + '</td>'
                        + '<td data-label="현재가">' + fmtKrwFull(price) + '</td>'
                        + '</tr>';
                }});
                const head = '<tr><th>종목</th><th>수량</th><th>현재가(원화)</th></tr>';
                return '<div class="card"><h3 style="margin-top:0;">' + title + '</h3>'
                    + '<table class="mobile-card-table"><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>';
            }}

            function renderLiveOverview(data) {{
                const root = document.getElementById('liveOverviewResult');
                const badge = document.getElementById('secAsset_badge');
                const accounts = (data && data.accounts) ? data.accounts : {{}};
                const stock = accounts.stock_kis || {{}};
                const upbit = accounts.coin_upbit || {{}};
                const binance = accounts.coin_binance || {{}};
                const totalKrw = Number(data.total_krw || 0);
                if (badge) {{
                    badge.textContent = '실시간 총액: ' + fmt(Math.round(totalKrw));
                }}

                const shareText = (amount) => {{
                    if (!totalKrw) return '비중 -';
                    return '비중 ' + (Number(amount || 0) / totalKrw * 100).toFixed(1) + '%';
                }};

                const card = (title, total, cash, subline, error) => {{
                    if (error) {{
                        return '<div class="card"><div style="font-size:0.9em;color:#666;">' + title + '</div>'
                            + '<div style="margin-top:8px;color:#d93025;">조회 실패: ' + error + '</div></div>';
                    }}
                    return '<div class="card" style="margin-bottom:0;">'
                        + '<div style="font-size:0.9em;color:#666;">' + title + '</div>'
                        + '<div style="font-size:1.35em;font-weight:700;margin-top:6px;">' + total + '</div>'
                        + '<div style="margin-top:6px;color:#555;">남은 현금: <b>' + cash + '</b></div>'
                        + '<div style="margin-top:4px;font-size:0.85em;color:#777;">' + subline + '</div>'
                        + '</div>';
                }};

                let html = '<div class="card" style="background:#f8f9fa;">'
                    + '<div style="font-size:0.95em;color:#666;">전체 실시간 자산</div>'
                    + '<div style="font-size:1.55em;font-weight:700;margin-top:6px;">' + fmtKrwFull(data.total_krw || 0) + '</div>'
                    + '<div style="margin-top:4px;font-size:0.85em;color:#777;">업데이트: ' + (data.updated || '-') + '</div>'
                    + '</div>';

                html += '<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-top:12px;">';
                html += card('주식 - 한투', fmtKrwFull(stock.total_krw), fmtKrwFull(stock.cash_krw), shareText(stock.total_krw) + ' / 보유 ' + ((stock.holdings || []).length) + '종목', stock.error);
                html += card('업비트', fmtKrwFull(upbit.total_krw), fmtKrwFull(upbit.krw_balance), shareText(upbit.total_krw) + ' / 보유 ' + ((upbit.holdings || []).length) + '종목', upbit.error);
                html += card('바이낸스', fmtKrwFull(binance.total_krw), fmtKrwFull(binance.cash_krw), shareText(binance.total_krw) + ' / 보유 ' + ((binance.holdings || []).length) + '포지션', binance.error);
                html += '</div>';

                // === 3자산 배분 체크 (V23 갱신 2026-05-22: 60/25/15, T1(ht≥13pp) OR T3U_can(max rel-under≥20% + canary ON) 트리거 — 리밸런싱은 수동) ===
                const stockKrw = Number(stock.total_krw || 0);
                const spotKrw = Number(upbit.total_krw || 0);
                const futKrw = Number(binance.total_krw || 0);
                const allocTotal = stockKrw + spotKrw + futKrw;
                if (allocTotal > 0) {{
                    const T_STOCK = 0.65, T_SPOT = 0.20, T_FUT = 0.15;
                    const REBAL_HT = 0.13;
                    const REBAL_T3U = 0.20;
                    const pStock = stockKrw / allocTotal;
                    const pSpot = spotKrw / allocTotal;
                    const pFut = futKrw / allocTotal;
                    const dStock = Math.abs(pStock - T_STOCK);
                    const dSpot = Math.abs(pSpot - T_SPOT);
                    const dFut = Math.abs(pFut - T_FUT);
                    const halfTurnover = (dStock + dSpot + dFut) / 2;
                    const relUnderStock = Math.max(0, (T_STOCK - pStock) / T_STOCK);
                    const relUnderSpot = Math.max(0, (T_SPOT - pSpot) / T_SPOT);
                    const relUnderFut = Math.max(0, (T_FUT - pFut) / T_FUT);
                    const maxRelUnder = Math.max(relUnderStock, relUnderSpot, relUnderFut);
                    const t1Fire = halfTurnover >= REBAL_HT;
                    const t3uFire = maxRelUnder >= REBAL_T3U;  // canary ON 가정 — 정확한 gate 는 daily report
                    const need = t1Fire || t3uFire;

                    const pctBar = (label, cur, target, drift, relU) => {{
                        return '<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f0f0f0;">'
                            + '<span>' + label + '</span>'
                            + '<span style="color:#444;font-weight:600;">'
                            + (cur * 100).toFixed(1) + '% (목표 ' + (target * 100).toFixed(0) + '%'
                            + ', 편차 ' + (drift * 100).toFixed(1) + '%p, rel-under ' + (relU * 100).toFixed(0) + '%)</span></div>';
                    }};

                    html += '<div class="card" style="margin-top:12px;border:2px solid ' + (need ? '#d93025' : '#0d904f') + ';">';
                    let triggerLines = [];
                    if (t1Fire) triggerLines.push('T1(ht ' + (halfTurnover*100).toFixed(1) + 'pp ≥ ' + (REBAL_HT*100).toFixed(0) + 'pp)');
                    if (t3uFire) triggerLines.push('T3U_can(max-under ' + (maxRelUnder*100).toFixed(0) + '% ≥ ' + (REBAL_T3U*100).toFixed(0) + '% — canary 확인 필요)');
                    html += '<div style="font-weight:700;font-size:1.05em;color:' + (need ? '#d93025' : '#0d904f') + ';margin-bottom:8px;">'
                        + (need ? ('⚠️ 리밸런싱 필요 — ' + triggerLines.join(' | '))
                                : ('✅ 리밸런싱 불필요 — T1 ' + (halfTurnover*100).toFixed(1) + 'pp / T3U_can max-under ' + (maxRelUnder*100).toFixed(0) + '%'))
                        + '</div>';
                    html += pctBar('주식', pStock, T_STOCK, dStock, relUnderStock);
                    html += pctBar('업비트', pSpot, T_SPOT, dSpot, relUnderSpot);
                    html += pctBar('바이낸스', pFut, T_FUT, dFut, relUnderFut);

                    if (need) {{
                        const tgtStock = allocTotal * T_STOCK;
                        const tgtSpot = allocTotal * T_SPOT;
                        const tgtFut = allocTotal * T_FUT;
                        html += '<div style="margin-top:10px;padding:10px;background:#fce8e6;border-radius:8px;font-size:0.95em;">';
                        const moves = [
                            ['주식', tgtStock - stockKrw],
                            ['업비트', tgtSpot - spotKrw],
                            ['바이낸스', tgtFut - futKrw],
                        ];
                        moves.forEach(m => {{
                            if (Math.abs(m[1]) > 10000) {{
                                const sign = m[1] > 0 ? '+' : '';
                                html += '<div>' + m[0] + ': <b>' + sign + Math.round(m[1]).toLocaleString() + '원</b></div>';
                            }}
                        }});
                        html += '</div>';
                    }}
                    html += '</div>';
                }}

                html += '<div style="margin-top:16px;">';
                html += renderWeightsTable('한투 구성 비중', stock.weights || {{}});
                html += renderWeightsTable('업비트 구성 비중', upbit.weights || {{}});
                html += renderWeightsTable('바이낸스 구성 비중', binance.weights || {{}});
                html += '</div>';
                root.innerHTML = html;
            }}

            async function fetchLiveOverview() {{
                const statusEl = document.getElementById('liveFetchStatus');
                try {{
                    statusEl.innerHTML = '\u23f3 \uc2e4\uacc4\uc88c \uc870\ud68c \uc911...';
                    statusEl.style.color = '#1967d2';
                    const r = await fetch(API + '/api/assets/live_overview');
                    const data = await r.json();
                    if (!r.ok) throw new Error(data.error || 'API error');
                    renderLiveOverview(data);
                    statusEl.innerHTML = '\u2705 \uc870\ud68c \uc644\ub8cc (' + (data.updated || '-') + ')';
                    statusEl.style.color = '#0d904f';
                }} catch (e) {{
                    statusEl.innerHTML = '\u274c \uc870\ud68c \uc2e4\ud328: ' + e.message;
                    statusEl.style.color = '#d93025';
                }}
            }}

            // === History ===
            let chartTotal = null;
            async function loadHistory() {{
                try {{
                    const r = await fetch(API + '/api/assets/snapshots');
                    const rows = await r.json();
                    if (!rows.length) return;

                    // Badge
                    const badge2 = document.getElementById('secHistory_badge');
                    if (badge2) badge2.textContent = rows.length + '\uac74';

                    // 카드형 히스토리: 월별 접기/펼치기, 가로 넘침 없음
                    const container = document.getElementById('historyContainer');
                    container.innerHTML = '';
                    const sorted = [...rows].reverse();
                    let monthRows = {{}};
                    for (const s of sorted) {{
                        const ym = (s.snapshot_date||s.month).substring(0, 7);
                        if (!monthRows[ym]) monthRows[ym] = [];
                        monthRows[ym].push(s);
                    }}
                    const months = Object.keys(monthRows).sort().reverse();

                    function histItem(s, bold) {{
                        const dt = s.snapshot_date||s.month;
                        const fw = bold ? 'font-weight:700;' : 'color:#555;';
                        return '<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f0f0f0;'+fw+'">'
                            +'<span>'+dt.substring(5)+'</span>'
                            +'<span>'+fmt(s.total_krw)+'</span></div>'
                            +'<div style="display:flex;gap:8px;padding:0 0 4px 0;font-size:0.82em;color:#888;">'
                            +'<span>\uc8fc\uc2dd '+fmt(s.stock_krw)+'</span>'
                            +'<span>\ucf54\uc778 '+fmt(s.coin_krw)+'</span>'
                            +'<span>\ud604\uae08 '+fmt(s.cash_krw)+'</span></div>';
                    }}

                    for (const ym of months) {{
                        const items = monthRows[ym];
                        const last = items[0];
                        const monthId = 'hm_'+ym.replace('-','');
                        const hasMore = items.length > 1;
                        const arrow = hasMore ? ' <span style="color:#aaa;font-size:0.8em;">(\u25b6 '+items.length+'\uac74)</span>' : '';

                        let card = '<div class="card" style="margin-bottom:8px;padding:10px 12px;">';
                        card += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'+(hasMore?'cursor:pointer;':'')+ '"'
                            +(hasMore?' onclick="const el=document.getElementById(\\''+monthId+'\\');el.style.display=el.style.display===\\'none\\'?\\'\\':\\'none\\'"':'')+'>'
                            +'<span style="font-weight:700;font-size:1.05em;">'+ym+arrow+'</span>'
                            +'<span style="font-weight:700;font-size:1.05em;">'+fmt(last.total_krw)+'</span></div>';
                        card += '<div style="display:flex;gap:8px;font-size:0.82em;color:#888;">'
                            +'<span>\uc8fc\uc2dd '+fmt(last.stock_krw)+'</span>'
                            +'<span>\ucf54\uc778 '+fmt(last.coin_krw)+'</span>'
                            +'<span>\ud604\uae08 '+fmt(last.cash_krw)+'</span></div>';

                        if (hasMore) {{
                            card += '<div id="'+monthId+'" style="display:none;margin-top:6px;border-top:1px solid #eee;padding-top:6px;">';
                            for (let j = 1; j < items.length; j++) {{
                                card += histItem(items[j], false);
                            }}
                            card += '</div>';
                        }}
                        card += '</div>';
                        container.innerHTML += card;
                    }}

                    // Chart: 월별 1점, 그 달 마지막(가장 최근) snapshot 사용
                    const monthsAsc = [...months].sort();
                    const labels = monthsAsc.map(ym => ym);
                    const totals = monthsAsc.map(ym => monthRows[ym][0].total_krw / 1e8);
                    if (chartTotal) chartTotal.destroy();
                    chartTotal = new Chart(document.getElementById('chartTotal'), {{
                        type: 'line',
                        data: {{ labels, datasets: [{{ label: '\ucd1d\uc790\uc0b0(\uc5b5)', data: totals, borderColor: '#7627bb', borderWidth: 2, fill: true, backgroundColor: 'rgba(118,39,187,0.1)', tension: 0.3 }}] }},
                        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'top' }} }} }}
                    }});
                }} catch(e) {{}}
            }}

            // Init
            document.addEventListener('DOMContentLoaded', function() {{
                loadHistory();
            }});
            </script>
        </div>
    </body>
    </html>
    """
    with open(filepath, 'w') as f: f.write(html)
    print(f"Report saved to {filepath}")

# --- 6. Main Execution ---
# --- 6. Main Execution ---
if __name__ == "__main__":
    log = []
    
    # 0. Get My Holdings
    # my_holdings_krw should include KRW cash for correct portfolio calc
    my_holdings_qty, my_holdings_krw, my_cash = get_current_upbit_holdings(log)
    
    if my_holdings_qty or my_cash > 0:
        print(f"✅ Loaded Holdings: {len(my_holdings_qty)} coins + Cash (Total Real-time: {sum(my_holdings_krw.values()) + my_cash:,.0f} KRW)")
    
    c_univ, ids = get_dynamic_coin_universe(log)
    if 'BTC-USD' not in ids: ids['BTC-USD'] = 'bitcoin'
    
    # Also load my holdings' tickers for price data
    all_tickers = set(OFFENSIVE_STOCK_UNIVERSE + DEFENSIVE_STOCK_UNIVERSE + CANARY_ASSETS + [STOCK_CRASH_TICKER] + c_univ + ['BTC-USD'] + list(my_holdings_qty.keys()))
    download_required_data(list(all_tickers), log, ids)
    prices = {t: load_price(t) for t in all_tickers}
    
    if prices['BTC-USD'].empty: sys.exit(1)
    target_date = prices['BTC-USD'].index[-1]
    
    # Get USD-KRW Rate
    rate = 1450.0 
    try:
        usdt_price = pyupbit.get_current_price("KRW-USDT")
        if usdt_price: rate = usdt_price
    except: pass
    
    # Evaluate My Assets using Close Price for TURNOVER CALCULATION
    # [V12.1] 종가 데이터가 없는 경우 업비트 실시간 가격(my_holdings_krw)을 fallback으로 사용
    cur_assets_close = {}  # {ticker-USD: KRW value based on close price}
    for ticker, qty in my_holdings_qty.items():
        if ticker in prices and not prices[ticker].empty:
            close_usd = prices[ticker].iloc[-1]
            val_krw = close_usd * rate * qty
            if val_krw > 1000: 
                cur_assets_close[ticker] = val_krw
        else:
            # Fallback: 종가 데이터 없으면 업비트 실시간 가격 사용
            if ticker in my_holdings_krw and my_holdings_krw[ticker] > 1000:
                cur_assets_close[ticker] = my_holdings_krw[ticker]
                print(f"  ⚠️ {ticker}: 종가 데이터 없음 → 업비트 실시간 가격 사용 ({my_holdings_krw[ticker]:,.0f} KRW)")
    
    # [Fix] Include Cash in Total Value for correct weight calculation
    total_val_close = sum(cur_assets_close.values()) + my_cash
    
    # Current Portfolio Weights (Cash included in denominator)
    cur_coin_port = {k: v/total_val_close for k, v in cur_assets_close.items()} if total_val_close > 0 else {}
    
    if cur_assets_close or my_cash > 0:
        print(f"✅ Turnover Basis (Close Price + Fallback): {len(cur_assets_close)} coins + Cash, Total: {total_val_close:,.0f} KRW")
    
    s_port, s_stat, s_meta = run_stock_strategy_v15(log, prices, target_date)
    c_port, c_stat, c_meta, log, healthy_coins = run_coin_strategy_v20(c_univ, prices, target_date, log)
    
    # Calc Turnover with Threshold Guide (Close Price Based)
    turnover = 0.0
    action_guide = "N/A"
    diff_table_rows = []  # Pre-calculated diff table for HTML
    TURNOVER_THRESHOLD = 0.30  # 30% (Match V12 Backtest)
    
    # [V12 Health Check Enforcement]
    # 보유 중인 코인이 healthy_coins에 없고, target 비중이 0이면 (매도 대상)
    # 턴오버와 상관없이 즉시 리밸런싱해야 함.
    has_bad_coin = False
    bad_coins = []
    healthy_set = set(healthy_coins)
    
    for t in cur_assets_close.keys():
        # cur_assets_close 키: 'TRX-USD', healthy_set 키: 'TRX-USD'
        if t not in c_port and t not in healthy_set:
            has_bad_coin = True
            bad_coins.append(t.replace('-USD', ''))

    # [통합 테이블 데이터 생성]
    # 모든 자산 (내 보유 + 타겟) 합집합
    all_assets = set(cur_assets_close.keys()) | set(c_port.keys())
    
    # Apply Cash Buffer to Target Portfolio (Coin Part)
    # c_port sum is 1.0. We need to scale it down to (1.0 - BUFFER) and assign BUFFER to CASH.
    # But only if it's "Full Invest" mode. If it's Risk-Off (CASH=1.0), BUFFER logic is redundant but safe.
    cash_buf = get_cash_buffer()
    c_port_buffered = {}
    if CASH_ASSET in c_port and c_port[CASH_ASSET] == 1.0:
        c_port_buffered = {CASH_ASSET: 1.0}
    else:
        for t, w in c_port.items():
            c_port_buffered[t] = w * (1.0 - cash_buf)
        c_port_buffered[CASH_ASSET] = c_port_buffered.get(CASH_ASSET, 0.0) + cash_buf

    integrated_rows = []
    
    # 1. 코인 자산
    for k in all_assets:
        if k == CASH_ASSET: continue  # Skip 'Cash' key, handle below
        
        ticker = k.replace('-USD', '')
        val_krw = my_holdings_krw.get(k, 0)
        
        my_w = cur_coin_port.get(k, 0)
        tgt_w = c_port_buffered.get(k, 0)
        diff = tgt_w - my_w
        
        action = "-"
        if diff > 0.005: action = "🔺 BUY"
        elif diff < -0.005: action = "🔻 SELL"
        
        integrated_rows.append({
            'Asset': ticker,
            'Value': val_krw,
            'My': my_w,
            'Target': tgt_w,
            'Diff': diff,
            'Action': action
        })
        
    # 2. 현금 자산 (Cash)
    cash_w = my_cash / total_val_close if total_val_close > 0 else 0
    cash_tgt = c_port_buffered.get(CASH_ASSET, 0.0) 
    
    cash_diff = cash_tgt - cash_w
    cash_action = "-"
    if cash_diff > 0.005: cash_action = "🔻 SELL COINS" # 현금 부족 -> 코인 팔아야 함 (Target > My)
    elif cash_diff < -0.005: cash_action = "🔺 BUY COINS" # 현금 과다 -> 코인 사야 함 (Target < My)
    
    integrated_rows.append({
        'Asset': 'CASH',
        'Value': my_cash,
        'My': cash_w,
        'Target': cash_tgt,
        'Diff': cash_diff,
        'Action': cash_action
    })
    
    # 3. Calc Turnover (Based on diffs)
    turnover = sum(abs(r['Diff']) for r in integrated_rows) / 2
    
    # 4. Action Guide Update
    if has_bad_coin:
        bad_coins_str = ", ".join(bad_coins)
        action_guide = f"REBALANCE (Sick: {bad_coins_str})"
    elif turnover >= TURNOVER_THRESHOLD:
        action_guide = "REBALANCE"
    else:
        action_guide = "HOLD"

    integrated_rows.sort(key=lambda x: x['Target'], reverse=True)
        
    # [Restored] Final Portfolio (Stock + Spot Coin + Futures) for Report Table
    final_port = {CASH_ASSET: 0}
    for t, w in s_port.items():
        key = t if t!=CASH_ASSET else CASH_ASSET
        final_port[key] = final_port.get(key, 0) + w * STOCK_RATIO
    for t, w in c_port_buffered.items():
        key = t if t!=CASH_ASSET else CASH_ASSET
        final_port[key] = final_port.get(key, 0) + w * COIN_RATIO
    # 선물 15%는 바이낸스 자동매매에서 별도 관리 (여기서는 표시만)
    final_port['FUTURES'] = FUTURES_RATIO
    
    # Get KRW Prices (생략 가능, 위에서 integrated_rows에 다 넣음)
    krw_prices = {}
        
    # Pass my_holdings_krw, integrated_rows for unified display
    coin_total_krw = sum(my_holdings_krw.values()) + my_cash
    save_html(log, final_port, s_port, c_port, s_stat, c_stat, turnover, [], [], target_date, krw_prices, s_meta, c_meta, {}, my_holdings_krw, action_guide, integrated_rows, coin_total_krw=coin_total_krw)

    try:
        snap = save_daily_live_snapshot()
        print(
            f"✅ Daily snapshot saved: {snap['snapshot_date']} "
            f"(stock={snap['stock_krw']:,.0f}, coin={snap['coin_krw']:,.0f}, cash={snap['cash_krw']:,.0f}, total={snap['total_krw']:,.0f})"
        )
    except Exception as e:
        print(f"⚠️ daily snapshot 저장 실패: {e}")

    # ─── 새 스키마 signal_state.json 저장 ───
    try:
        # 주식: 공격/방어 picks (s_port는 현재 모드에 따라 하나만 반환됨)
        # 현재가 공격이면 s_port=공격, 방어이면 s_port=방어
        # 항상 둘 다 저장하기 위해 현재 모드 기반으로 분류
        s_picks = sorted([t for t in s_port.keys() if t != 'Cash'])
        s_weights_all = {t: w for t, w in s_port.items()}

        is_stock_risk_on = not s_stat.startswith('수비')
        if is_stock_risk_on:
            offense_picks = s_picks
            offense_weights = s_weights_all
            # 방어 종목은 이전 signal에서 가져오거나 기본값
            try:
                with open(SIGNAL_STATE_FILE, 'r') as _pf:
                    _ps = json.load(_pf)
                defense_picks = _ps.get('stock', {}).get('defense_picks', ['IEF', 'GLD', 'PDBC'])
                defense_weights = _ps.get('stock', {}).get('defense_weights', {})
                # 빈 경우 기본값 생성
                if not defense_weights and defense_picks:
                    dw = (1.0 - 0.02) / len(defense_picks)
                    defense_weights = {t: round(dw, 4) for t in defense_picks}
                    defense_weights['Cash'] = 0.02
            except Exception:
                defense_picks = ['IEF', 'GLD', 'PDBC']
                dw = (1.0 - 0.02) / 3
                defense_weights = {t: round(dw, 4) for t in defense_picks}
                defense_weights['Cash'] = 0.02
        else:
            defense_picks = s_picks
            defense_weights = s_weights_all
            # 공격 종목은 이전 signal에서 가져오거나 기본값
            try:
                with open(SIGNAL_STATE_FILE, 'r') as _pf:
                    _ps = json.load(_pf)
                offense_picks = _ps.get('stock', {}).get('offense_picks', [])
                offense_weights = _ps.get('stock', {}).get('offense_weights', {})
            except Exception:
                offense_picks = []
                offense_weights = {'Cash': 1.0}

        # 코인: guard_refs 계산 (KRW 기준 — executor가 KRW 현재가와 비교)
        # 보유 중인 코인 + 추천 코인 모두 포함 (보유중이지만 추천에서 빠진 종목도 가드 필요)
        coin_guard_refs = {}
        all_coin_tickers = set(t for t in c_port.keys() if t != 'Cash')
        # 기존 트랜치 보유 종목도 포함하면 좋지만, recommend는 trade_state를 안 읽으므로
        # 추천 종목 기반으로만 생성. executor가 잔고 기반으로 추가 체크함.
        for t in sorted(all_coin_tickers):
            p = prices.get(t)  # USD 시계열
            if p is not None and len(p) >= 60:
                # KRW 변환: Upbit 현재가 사용
                try:
                    krw_ticker = f"KRW-{t.replace('-USD', '')}"
                    krw_price = pyupbit.get_current_price(krw_ticker)
                    if krw_price and krw_price > 0:
                        # USD→KRW 비율로 과거 가격도 변환
                        usd_cur = float(p.iloc[-1])
                        ratio = krw_price / usd_cur if usd_cur > 0 else 1
                        coin_guard_refs[t.replace('-USD', '')] = {
                            'prev_close': round(float(p.iloc[-1]) * ratio),
                            'peak_60d': round(float(p.iloc[-60:].max()) * ratio),
                        }
                except Exception:
                    pass

        new_signal = {
            'stock': {
                'offense_picks': offense_picks,
                'offense_weights': offense_weights,
                'defense_picks': defense_picks,
                'defense_weights': defense_weights,
                'risk_on': is_stock_risk_on,
            },
            'coin': {
                'picks': sorted([t.replace('-USD', '') for t in c_port.keys() if t != 'Cash']),
                'weights': {t.replace('-USD', ''): w for t, w in c_port.items()},
                'risk_on': bool(not c_stat.startswith('Risk-Off')),
                'guard_refs': coin_guard_refs,
            },
            'meta': {
                'signal_date': str(target_date.date()) if hasattr(target_date, 'date') else str(target_date)[:10],
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            },
        }
        _save_signal_state(new_signal)
    except Exception as e:
        print(f"⚠️ signal_state 저장 실패: {e}")

    # ─── 텔레그램 일간 리포트 ───
    try:
        # 코인 카나리 (BTC vs SMA42, hyst 1.5%)
        btc_data = prices.get('BTC-USD')
        btc_cur = btc_sma = None; btc_ratio = None; btc_dist_pct = "n/a"
        if btc_data is not None and len(btc_data) >= COIN_CANARY_MA_PERIOD:
            btc_cur = float(btc_data.iloc[-1])
            btc_sma = float(btc_data.rolling(COIN_CANARY_MA_PERIOD).mean().iloc[-1])
            btc_ratio = btc_cur / btc_sma
            btc_dist_pct = f"{(btc_ratio-1)*100:+.2f}%"

        coin_picks_str = ', '.join(t.replace('-USD','') for t in c_port.keys() if t != 'Cash')

        # Helpers — 현재 보유 비중 + 목표 vs 보유 포맷
        def _cur_weights_early(acct):
            if not acct: return {}
            tot = float(acct.get('total_krw', 0) or 0)
            if tot <= 0: return {}
            w = {}
            invested = 0.0
            for h in (acct.get('holdings') or []):
                tk = (h.get('ticker') or h.get('symbol') or '').upper().replace('-USD', '')
                v = float(h.get('krw', h.get('value_krw', h.get('weight_value_krw', 0))) or 0)
                if tk and v > 0:
                    w[tk] = v / tot
                    invested += v
            cash_v = max(0.0, tot - invested)
            if cash_v > 0:
                w['Cash'] = cash_v / tot
            return w

        def _fmt_disp_early(disp_dict, cur_dict):
            def _norm(k):
                return 'Cash' if str(k).lower() == 'cash' else str(k).upper().replace('-USD', '')
            disp_n = {}
            for k, v in (disp_dict or {}).items():
                nk = _norm(k)
                disp_n[nk] = disp_n.get(nk, 0.0) + float(v or 0)
            cur_n = {}
            for k, v in (cur_dict or {}).items():
                nk = _norm(k)
                cur_n[nk] = cur_n.get(nk, 0.0) + float(v or 0)
            lines = []
            keys = sorted(set(disp_n) | set(cur_n), key=lambda k: -(disp_n.get(k, 0) or 0))
            for k in keys:
                tw = disp_n.get(k, 0.0)
                cw = cur_n.get(k, 0.0)
                if tw < 1e-4 and cw < 1e-4 and k.lower() != 'cash':
                    continue
                lines.append(f'  {k}: 목표 {tw*100:.1f}% / 보유 {cw*100:.1f}%')
            return lines

        # accts (live_overview) 를 여기서 미리 조회 — target lines 에 보유 % 표시용
        try:
            _api_early = os.environ.get("TRADE_API_BASE", "http://127.0.0.1:5000")
            _ov_early = requests.get(f"{_api_early}/api/assets/live_overview", timeout=60).json()
            accts = _ov_early.get("accounts", {}) or {}
        except Exception:
            accts = {}
        _cur_fut_early = _cur_weights_early(accts.get('coin_binance'))

        # 선물 state 읽기 (binance_state.json)
        fut_lines = ['🎯 바이낸스 목표']
        fut_canary_lines = []
        try:
            bn_state_path = os.environ.get('BINANCE_STATE', '/home/ubuntu/binance_state.json')
            if not os.path.exists(bn_state_path):
                bn_state_path = os.path.join(APP_HOME, 'binance_state.json')
            if os.path.exists(bn_state_path):
                with open(bn_state_path) as fh:
                    bn = json.load(fh)
                strat = bn.get('strategies', {}).get('D_SMA42', {})
                last_combined = strat.get('last_combined') or bn.get('last_target') or {}
                if last_combined:
                    _fut_buf = get_cash_buffer('fut')
                    f_disp = {}
                    _f_has_cash = any(str(k).lower() == 'cash' for k in last_combined.keys())
                    for k, w in last_combined.items():
                        if str(k).lower() == 'cash':
                            f_disp[k] = float(w)
                        else:
                            f_disp[k] = float(w) * (1.0 - _fut_buf)
                    if not _f_has_cash and _fut_buf > 0:
                        f_disp['Cash'] = _fut_buf
                    fut_lines.extend(_fmt_disp_early(f_disp, _cur_fut_early))
                else:
                    fut_lines.append('  (목표 없음)')
                # canary detail
                ci = strat.get('canary_info') or {}
                if ci:
                    fut_canary_lines.append(
                        f"  바이낸스 (BTC perp vs SMA{ci.get('sma_p',42)}): "
                        f"{'ON 공격' if ci.get('on') else 'OFF 캐시'} "
                        f"(ratio {ci.get('ratio',0):.4f}, cur ${ci.get('cur',0):,.0f}, sma ${ci.get('sma_val',0):,.0f})"
                    )
                else:
                    fut_canary_lines.append(f"  바이낸스 카나리: {'ON' if strat.get('canary_on') else 'OFF'} (info 없음)")
                fut_combined_str = ', '.join(t for t in last_combined.keys() if t.lower() != 'cash')
            else:
                fut_lines.append('  (state 없음)')
                fut_combined_str = ''
        except Exception as ex_fut:
            fut_lines.append(f'  (읽기 실패: {ex_fut})')
            fut_combined_str = ''

        # V23 통일 포맷 (Daily Report — 신호 요약, 실행 보고와 별개)
        date_str = target_date.strftime('%Y-%m-%d') if hasattr(target_date, 'strftime') else str(target_date)[:10]

        def _cur_weights(acct):
            return _cur_weights_early(acct)

        _cur_stock = _cur_weights(accts.get('stock_kis') if 'accts' in locals() else None)
        _cur_spot = _cur_weights(accts.get('coin_upbit') if 'accts' in locals() else None)
        _cur_fut = _cur_weights(accts.get('coin_binance') if 'accts' in locals() else None)

        def _fmt_disp(disp_dict, cur_dict):
            lines = []
            keys = sorted(set(disp_dict) | set(cur_dict), key=lambda k: -(disp_dict.get(k, 0) or 0))
            for k in keys:
                tw = disp_dict.get(k, 0.0)
                cw = cur_dict.get(k, 0.0)
                if tw < 1e-4 and cw < 1e-4 and k.lower() != 'cash':
                    continue
                tk = k.replace('-USD', '')
                lines.append(f'  {tk}: 목표 {tw*100:.1f}% / 보유 {cw*100:.1f}%')
            return lines

        c_lines = ['🎯 업비트 목표']
        _spot_buf = get_cash_buffer('spot')
        c_disp = {}
        _c_has_cash = any(str(k).lower() == 'cash' for k in (c_port or {}).keys())
        for k, w in (c_port or {}).items():
            if str(k).lower() == 'cash':
                c_disp[k] = float(w)
            else:
                c_disp[k] = float(w) * (1.0 - _spot_buf)
        if not _c_has_cash and _spot_buf > 0:
            c_disp['Cash'] = _spot_buf
        c_disp_n = {str(k).replace('-USD', ''): v for k, v in c_disp.items()}
        c_lines.extend(_fmt_disp(c_disp_n, _cur_spot))
        s_lines = ['🎯 주식 목표']
        _stock_buf = get_cash_buffer('stock')
        s_port_disp = {}
        _has_cash = any(str(k).lower() == 'cash' for k in (s_port or {}).keys())
        for k, w in (s_port or {}).items():
            if str(k).lower() == 'cash':
                s_port_disp[k] = float(w)
            else:
                s_port_disp[k] = float(w) * (1.0 - _stock_buf)
        if not _has_cash and _stock_buf > 0:
            s_port_disp['Cash'] = _stock_buf
        s_lines.extend(_fmt_disp(s_port_disp, _cur_stock))

        # 트리거 상태 등급 — verdict 산출용
        # status: OK (<50%), WATCH (50-80%), NEAR (80-100%), FIRE (>=100%)
        _verdict_signals = []
        def _trig_status(progress):
            if progress >= 1.0: return ('FIRE', '🔴')
            if progress >= 0.8: return ('NEAR', '🟠')
            if progress >= 0.5: return ('WATCH', '🟡')
            return ('OK', '🟢')

        def _canary_label(ratio):
            if ratio is None: return ('?', '')
            dist = (ratio - 1.0) * 100
            mode = 'BULL' if ratio >= 1.0 else 'BEAR'
            return (f"{dist:+.1f}%", mode)

        # 카나리 상세 — SMA 괴리율 + cur/sma/ratio
        canary_lines = ['🦅 카나리']
        if btc_cur is not None:
            _dist, _mode = _canary_label(btc_ratio)
            canary_lines.append(
                f"  업비트: SMA {_dist} {_mode} — BTC ${btc_cur:,.0f} / SMA{COIN_CANARY_MA_PERIOD} ${btc_sma:,.0f} (ratio {btc_ratio:.4f})"
            )
        else:
            canary_lines.append(f"  업비트: BTC 데이터 부족")
        try:
            ci = strat.get('canary_info') if 'strat' in locals() else None
        except Exception:
            ci = None
        if ci:
            _br = ci.get('ratio', 0)
            _bd, _bm = _canary_label(_br if _br else None)
            canary_lines.append(
                f"  바이낸스: SMA {_bd} {_bm} — BTC ${ci.get('cur',0):,.0f} / SMA{ci.get('sma_p',42)} ${ci.get('sma_val',0):,.0f} (ratio {_br:.4f})"
            )
        else:
            canary_lines.extend(fut_canary_lines)
        eem_cur_v = s_meta.get('canary_eem_cur')
        eem_sma_v = s_meta.get('canary_eem_sma')
        eem_p = s_meta.get('canary_sma_period', 300)
        if eem_cur_v and eem_sma_v:
            ratio_s = eem_cur_v / eem_sma_v if eem_sma_v else 0
            _ed, _em = _canary_label(ratio_s)
            canary_lines.append(
                f"  주식: SMA {_ed} {_em} — EEM ${eem_cur_v:.2f} / SMA{eem_p} ${eem_sma_v:.2f} (ratio {ratio_s:.4f})"
            )
        else:
            canary_lines.append(f"  주식: EEM 데이터 부족")

        # 보유/드리프트/자산배분 — live_overview + state 파일에서 조립
        holdings_lines = ['💼 보유']
        drift_lines = ['🌊 드리프트']
        alloc_lines = ['⚖️ 자산배분']
        try:
            api = os.environ.get("TRADE_API_BASE", "http://127.0.0.1:5000")
            ov = requests.get(f"{api}/api/assets/live_overview", timeout=60).json()
            accts = ov.get("accounts", {})
            stock_acct = accts.get("stock_kis") or {}
            spot_acct = accts.get("coin_upbit") or {}
            fut_acct = accts.get("coin_binance") or {}
            stock_krw = float(stock_acct.get("total_krw", 0))
            spot_krw = float(spot_acct.get("total_krw", 0))
            fut_krw = float(fut_acct.get("total_krw", 0))
            alloc_total = stock_krw + spot_krw + fut_krw

            # 보유 (티커 비중 — sleeve 내부)
            def _hold_lines(label, acct, total):
                lines = [f"  [{label}] 평가액 ₩{total:,.0f}"]
                hs = acct.get("holdings") or []
                if not hs:
                    return lines
                for h in sorted(hs, key=lambda x: -float(x.get("krw", 0) or 0)):
                    krw = float(h.get("krw", 0) or 0)
                    if krw < 5000:
                        continue
                    tk = h.get("ticker") or h.get("symbol") or "?"
                    w = krw / total if total > 0 else 0
                    lines.append(f"    {tk}: ₩{krw:,.0f} ({w*100:.1f}%)")
                return lines
            holdings_lines[0] = f"💼 보유 (총 ₩{alloc_total:,.0f})"
            holdings_lines.extend(_hold_lines("주식", stock_acct, stock_krw))
            holdings_lines.extend(_hold_lines("업비트", spot_acct, spot_krw))
            holdings_lines.extend(_hold_lines("바이낸스", fut_acct, fut_krw))

            # 자산배분
            if alloc_total > 0:
                p_stock = stock_krw / alloc_total
                p_spot = spot_krw / alloc_total
                p_fut = fut_krw / alloc_total
                d_stock = abs(p_stock - STOCK_RATIO)
                d_spot = abs(p_spot - COIN_RATIO)
                d_fut = abs(p_fut - FUTURES_RATIO)
                ht = (d_stock + d_spot + d_fut) / 2
                # T3U_can: max relative underweight per asset + sleeve canary gate
                rel_under_stock = max(0.0, (STOCK_RATIO - p_stock) / STOCK_RATIO)
                rel_under_spot = max(0.0, (COIN_RATIO - p_spot) / COIN_RATIO)
                rel_under_fut = max(0.0, (FUTURES_RATIO - p_fut) / FUTURES_RATIO)
                # sleeve canary states
                _spot_canary_on = (btc_ratio is not None and btc_ratio > 1.0 + COIN_CANARY_HYST)
                try:
                    _fut_canary_on = bool(strat.get('canary_on')) if 'strat' in locals() else _spot_canary_on
                except Exception:
                    _fut_canary_on = _spot_canary_on
                _stock_canary_on = bool(is_stock_risk_on) if 'is_stock_risk_on' in locals() else True
                t3u_stock = rel_under_stock >= REBAL_T3U_REL and _stock_canary_on
                t3u_spot = rel_under_spot >= REBAL_T3U_REL and _spot_canary_on
                t3u_fut = rel_under_fut >= REBAL_T3U_REL and _fut_canary_on
                t1_fire = ht >= REBAL_HT_THRESHOLD
                t3u_fire = t3u_stock or t3u_spot or t3u_fut
                fire = t1_fire or t3u_fire

                # ── 자산배분 트리거 (V23 B 안, 2026-05-26) ──
                # alloc_transit 자동 cap_ratio 시스템 폐지. read-only 평가 + 텔레그램 알림만.
                # 사용자 수동 송금. 각 executor 는 자기 계좌 안에서만 자동매매 (sleeve 내부).
                _alloc_transit_active = False  # 항상 False — 자동 cap 실행 X
                _KST = timezone(timedelta(hours=9))

                def _compute_cap_ratios(_tot, _sk, _spk, _fk):
                    tgt_stock_krw = _tot * STOCK_RATIO
                    tgt_spot_krw = _tot * COIN_RATIO
                    tgt_fut_krw = _tot * FUTURES_RATIO
                    return {
                        'stock': min(1.0, tgt_stock_krw / _sk) if _sk > 0 else 1.0,
                        'spot': min(1.0, tgt_spot_krw / _spk) if _spk > 0 else 1.0,
                        'fut': min(1.0, tgt_fut_krw / _fk) if _fk > 0 else 1.0,
                    }
                try:
                    _ts_path = None
                    for _p in (os.path.join(APP_HOME, 'trade_state.json'), 'trade_state.json'):
                        if os.path.exists(_p):
                            _ts_path = _p
                            break
                    _ts_obj = {}
                    if _ts_path:
                        with open(_ts_path, 'r') as _f:
                            _ts_obj = json.load(_f)
                    _at = _ts_obj.get('alloc_transit') or {}
                    _was_active = bool(_at.get('active', False))
                    _now_dt = datetime.now(_KST)
                    _now_str = _now_dt.strftime('%Y-%m-%d %H:%M KST')
                    _cap_ratios = _compute_cap_ratios(alloc_total, stock_krw, spot_krw, fut_krw)
                    # V23 (2026-05-26) B 안: 자동 cap_ratio 폐지. read-only 평가 + 알림만.
                    # legacy active=True 면 즉시 clear (마이그레이션).
                    if _was_active:
                        _ts_obj['alloc_transit'] = {
                            'active': False,
                            'cleared_at': _now_str,
                            'reason': 'V23 B 안 (2026-05-26) — 자동 cap_ratio 폐지, 수동 송금 모델로 전환',
                        }
                        _alloc_transit_active = False
                        alloc_lines.append(f"  🟢 alloc_transit FORCE CLEAR (V23 B 안 — 자동 cap_ratio 시스템 폐지)")
                    # 트리거 ON 시 텔레그램 알림 (read-only, 수동 송금 권장)
                    if fire:
                        # suggested_transfer 계산 (어느 sleeve → 어느 sleeve)
                        tgt_stock_krw = alloc_total * STOCK_RATIO
                        tgt_spot_krw = alloc_total * COIN_RATIO
                        tgt_fut_krw = alloc_total * FUTURES_RATIO
                        diffs = {
                            'stock': stock_krw - tgt_stock_krw,
                            'spot': spot_krw - tgt_spot_krw,
                            'fut': fut_krw - tgt_fut_krw,
                        }
                        reason = ' | '.join(([f'T1 ht {ht*100:.1f}pp'] if t1_fire else []) +
                                            ([f'T3U_can ≥ {REBAL_T3U_REL*100:.0f}%'] if t3u_fire else []))
                        # rate-limit: 같은 reason 24h 내 중복 알림 X
                        _last_alert = _at.get('last_alert_at', '') if isinstance(_at, dict) else ''
                        _alert_ok = True
                        try:
                            if _last_alert:
                                _la_dt = datetime.strptime(_last_alert.replace(' KST', ''), '%Y-%m-%d %H:%M').replace(tzinfo=_KST)
                                if (_now_dt - _la_dt).total_seconds() < 86400:
                                    _alert_ok = False
                        except Exception:
                            pass
                        if _alert_ok:
                            try:
                                _msg_lines = [
                                    f"⚠️ V23 자산배분 트리거 ON",
                                    f"트리거: {reason}",
                                    f"현재 비중: 주식 {p_stock*100:.1f}% / 업비트 {p_spot*100:.1f}% / 바이낸스 {p_fut*100:.1f}%",
                                    f"목표: 60/20/15 (B 안 + per-sleeve buffer stock 6/spot 1/fut 1)",
                                    f"현재 KRW: 주식 ₩{stock_krw:,.0f} / 업비트 ₩{spot_krw:,.0f} / 바이낸스 ₩{fut_krw:,.0f}",
                                    f"목표 KRW: 주식 ₩{tgt_stock_krw:,.0f} / 업비트 ₩{tgt_spot_krw:,.0f} / 바이낸스 ₩{tgt_fut_krw:,.0f}",
                                    f"송금 제안:",
                                ]
                                for _k, _v in diffs.items():
                                    if abs(_v) > alloc_total * 0.005:  # 0.5% 미만은 무시
                                        _kname = {'stock': '주식', 'spot': '업비트', 'fut': '바이낸스'}[_k]
                                        if _v > 0:
                                            _msg_lines.append(f"  {_kname}: 매도/출금 ₩{_v:,.0f}")
                                        else:
                                            _msg_lines.append(f"  {_kname}: 입금 ₩{-_v:,.0f}")
                                _msg_lines.append(f"※ 자동 rebal X — 수동 송금 필요. sleeve 내부 자동매매는 그대로 진행")
                                send_telegram('\n'.join(_msg_lines))
                                _ts_obj['alloc_transit'] = {
                                    'active': False,
                                    'last_alert_at': _now_str,
                                    'last_fire_reason': reason,
                                    'last_ht': float(ht),
                                    'last_total_krw': float(alloc_total),
                                    'target_ratios': {'stock': STOCK_RATIO, 'spot': COIN_RATIO, 'fut': FUTURES_RATIO},
                                }
                            except Exception as _ex_tg:
                                alloc_lines.append(f"  ⚠️ trigger alert 전송 실패: {_ex_tg}")
                        alloc_lines.append(f"  ⚠️ 트리거 ON ({reason}) — 텔레그램 알림 발송 (수동 송금 권장)")
                    if _ts_path and (_was_active or _alloc_transit_active):
                        _tmp = _ts_path + '.tmp'
                        with open(_tmp, 'w') as _f:
                            json.dump(_ts_obj, _f, ensure_ascii=False, indent=2)
                        os.replace(_tmp, _ts_path)
                except Exception as _ex_at:
                    alloc_lines.append(f"  ⚠️ alloc_transit 처리 실패: {_ex_at}")

                fire_reason = []
                if t1_fire: fire_reason.append(f"T1(ht {ht*100:.1f}pp ≥ {REBAL_HT_THRESHOLD*100:.0f}pp)")
                if t3u_fire:
                    parts = []
                    if t3u_stock: parts.append(f"주식 {rel_under_stock*100:.0f}%↓")
                    if t3u_spot: parts.append(f"업비트 {rel_under_spot*100:.0f}%↓")
                    if t3u_fut: parts.append(f"바이낸스 {rel_under_fut*100:.0f}%↓")
                    fire_reason.append(f"T3U_can({', '.join(parts)})")
                alloc_lines.append(f"  주식 {p_stock:.1%} (목표 {STOCK_RATIO:.0%}, 편차 {d_stock:.1%})")
                alloc_lines.append(f"  업비트 {p_spot:.1%} (목표 {COIN_RATIO:.0%}, 편차 {d_spot:.1%})")
                alloc_lines.append(f"  바이낸스 {p_fut:.1%} (목표 {FUTURES_RATIO:.0%}, 편차 {d_fut:.1%})")
                _t1_p = ht / REBAL_HT_THRESHOLD if REBAL_HT_THRESHOLD > 0 else 0
                _t1_lbl, _t1_ic = _trig_status(_t1_p)
                _verdict_signals.append(('T1', _t1_p, _t1_lbl))
                alloc_lines.append(f"  {_t1_ic} T1 {ht*100:.2f}/{REBAL_HT_THRESHOLD*100:.0f}pp {_t1_p*100:.0f}% — 남은 {(REBAL_HT_THRESHOLD-ht)*100:.2f}pp [{_t1_lbl}]")
                _max_ru = max(rel_under_stock, rel_under_spot, rel_under_fut)
                _t3u_p = _max_ru / REBAL_T3U_REL if REBAL_T3U_REL > 0 else 0
                _t3u_lbl, _t3u_ic = _trig_status(_t3u_p)
                _verdict_signals.append(('T3U_max', _t3u_p, _t3u_lbl))
                alloc_lines.append(f"  {_t3u_ic} T3U_can max-under {_max_ru*100:.0f}/{REBAL_T3U_REL*100:.0f}% {_t3u_p*100:.0f}% (주식 {rel_under_stock*100:.0f}% / 업비트 {rel_under_spot*100:.0f}% / 바이낸스 {rel_under_fut*100:.0f}%) [{_t3u_lbl}]")
                if fire:
                    alloc_lines.append(f"  🔔 리밸 필요 — {' | '.join(fire_reason)}")
                else:
                    alloc_lines.append(f"  ✅ 트리거 내")
                if fire:
                    moves = []
                    for name, cur, tgt in [("주식", stock_krw, alloc_total * STOCK_RATIO),
                                            ("업비트", spot_krw, alloc_total * COIN_RATIO),
                                            ("바이낸스", fut_krw, alloc_total * FUTURES_RATIO)]:
                        delta = tgt - cur
                        if abs(delta) > 10000:
                            moves.append(f"    {name} {delta:+,.0f}원")
                    if moves:
                        alloc_lines.append("  수동 조정:")
                        alloc_lines.extend(moves)
            else:
                alloc_lines.append("  (live_overview 0)")
        except Exception as ex_ov:
            holdings_lines.append(f"  (조회 실패: {ex_ov})")
            alloc_lines.append(f"  (조회 실패: {ex_ov})")

        # 드리프트 — 현재 보유 비중 vs 목표 비중에서 직접 계산 (state file 미사용)
        def _ht_from_holdings(acct, total_krw, target_dict):
            """half_turnover = sum(|cur_w - tgt_w|) / 2. cash 포함."""
            if total_krw <= 0:
                return None
            cur_w = {}
            for h in (acct.get('holdings') or []):
                tk = (h.get('ticker') or '').upper()
                v = float(h.get('value_krw', h.get('weight_value_krw', 0)) or 0)
                if tk and v > 0:
                    cur_w[tk] = v / total_krw
            cur_w_sum = sum(cur_w.values())
            cur_w['CASH'] = max(0.0, 1.0 - cur_w_sum)
            tgt = {}
            for k, v in target_dict.items():
                tk = 'CASH' if str(k).lower() == 'cash' else str(k).replace('-USD', '').upper()
                tgt[tk] = float(v)
            keys = set(cur_w) | set(tgt)
            return sum(abs(cur_w.get(k, 0) - tgt.get(k, 0)) for k in keys) / 2

        try:
            stock_acct_d = (accts.get("stock_kis") or {}) if 'accts' in locals() else {}
            stock_total_d = float(stock_acct_d.get("total_krw", 0))
            ht_st = _ht_from_holdings(stock_acct_d, stock_total_d, s_port or {})
            thr_st = 0.10
            if ht_st is None:
                drift_lines.append("  주식: 잔고 없음")
            else:
                fire_st = ht_st >= thr_st
                _p = ht_st / thr_st if thr_st > 0 else 0
                _st_lbl, _st_ic = _trig_status(_p)
                _verdict_signals.append(('주식 drift', _p, _st_lbl))
                drift_lines.append(f"  주식: {_st_ic} ht {ht_st*100:.2f}/{thr_st*100:.0f}pp {_p*100:.0f}% [{_st_lbl}]")
        except Exception as ex_dst:
            drift_lines.append(f"  주식: 계산 실패 ({ex_dst})")
        try:
            spot_acct_d = (accts.get("coin_upbit") or {}) if 'accts' in locals() else {}
            spot_total_d = float(spot_acct_d.get("total_krw", 0))
            ht_sp = _ht_from_holdings(spot_acct_d, spot_total_d, c_port or {})
            thr_sp = 0.10
            if ht_sp is None:
                drift_lines.append("  업비트: 잔고 없음")
            else:
                fire_sp = ht_sp >= thr_sp
                _p = ht_sp / thr_sp if thr_sp > 0 else 0
                _sp_lbl, _sp_ic = _trig_status(_p)
                _verdict_signals.append(('업비트 drift', _p, _sp_lbl))
                drift_lines.append(f"  업비트: {_sp_ic} ht {ht_sp*100:.2f}/{thr_sp*100:.0f}pp {_p*100:.0f}% [{_sp_lbl}]")
        except Exception as ex_ds:
            drift_lines.append(f"  업비트: 계산 실패 ({ex_ds})")
        try:
            fut_acct_d = (accts.get("coin_binance") or {}) if 'accts' in locals() else {}
            fut_total_d = float(fut_acct_d.get("total_krw", 0))
            fut_target = last_combined if 'last_combined' in locals() and last_combined else {}
            ht_f = _ht_from_holdings(fut_acct_d, fut_total_d, fut_target)
            thr_f = 0.03
            if ht_f is None:
                drift_lines.append("  바이낸스: 잔고 없음")
            elif not fut_target:
                drift_lines.append("  바이낸스: 목표 없음")
            else:
                fire_f = ht_f >= thr_f
                _p = ht_f / thr_f if thr_f > 0 else 0
                _f_lbl, _f_ic = _trig_status(_p)
                _verdict_signals.append(('바이낸스 drift', _p, _f_lbl))
                drift_lines.append(f"  바이낸스: {_f_ic} ht {ht_f*100:.2f}/{thr_f*100:.0f}pp {_p*100:.0f}% [{_f_lbl}]")
        except Exception as ex_df:
            drift_lines.append(f"  바이낸스: 계산 실패 ({ex_df})")

        # V23 params drift check (live ↔ canonical)
        params_drift_lines = ["🧭 params drift check"]
        try:
            import subprocess
            # 서버 (flat /home/ubuntu/) vs 로컬 (mon/251229/trade/ops/) 양쪽 지원
            _here = os.path.dirname(os.path.abspath(__file__))
            _candidates = [
                os.path.join(_here, 'check_params_drift.py'),  # 서버 flat
                os.path.abspath(os.path.join(_here, '..', '..', 'trade', 'ops', 'check_params_drift.py')),  # 로컬
            ]
            _check_path = next((p for p in _candidates if os.path.exists(p)), None)
            if _check_path is None:
                raise FileNotFoundError("check_params_drift.py not found")
            _r = subprocess.run(['python3', _check_path], capture_output=True, text=True, timeout=30)
            if _r.returncode == 0:
                params_drift_lines.append("  ✅ 3자산 canonical 일치")
            else:
                params_drift_lines.append("  ⚠️ drift 발견:")
                for ln in (_r.stdout or '').splitlines():
                    ln = ln.strip()
                    if ln and not ln.startswith('⚠️'):
                        params_drift_lines.append(f"  {ln}")
        except Exception as ex_pd:
            params_drift_lines.append(f"  (체크 실패: {ex_pd})")

        # verdict 산출 — 가장 높은 상태등급
        _rank = {'OK': 0, 'WATCH': 1, 'NEAR': 2, 'FIRE': 3}
        _max_lbl = 'OK'
        _max_sig = None
        _max_p = 0.0
        for _nm, _pp, _lbl in _verdict_signals:
            if _rank.get(_lbl, 0) > _rank.get(_max_lbl, 0):
                _max_lbl = _lbl
                _max_sig = _nm
                _max_p = _pp
            elif _rank.get(_lbl, 0) == _rank.get(_max_lbl, 0) and _pp > _max_p:
                _max_sig = _nm
                _max_p = _pp
        _params_fail = any('drift 발견' in ln or '⚠️' in ln for ln in params_drift_lines[1:])
        if _params_fail and _rank.get(_max_lbl, 0) < 1:
            _max_lbl = 'WATCH'
            _max_sig = 'params drift'
        _verdict_ic = {'OK': '🟢', 'WATCH': '🟡', 'NEAR': '🟠', 'FIRE': '🔴'}[_max_lbl]
        _verdict_msg = {'OK': 'NO ACTION', 'WATCH': 'WATCH', 'NEAR': 'NEAR — 발화 임박', 'FIRE': 'ACT — 트리거 ON'}[_max_lbl]
        _verdict_line = f"{_verdict_ic} {_verdict_msg}"
        if _max_sig and _max_lbl != 'OK':
            _verdict_line += f" ({_max_sig} {_max_p*100:.0f}%)"

        _now_kst = datetime.now().strftime('%Y-%m-%d %H:%M KST')
        _as_of_line = f"⏱ as-of {_now_kst}"

        _show_pd = _params_fail
        _pd_block = ('\n'.join(params_drift_lines) + "\n\n") if _show_pd else ""

        summary = (
                f"[Daily Report] 📊 V23 신호 ({date_str})\n"
                f"{_verdict_line}\n"
                f"{_as_of_line}\n\n"
                + '\n'.join(c_lines) + "\n\n"
                + '\n'.join(fut_lines) + "\n\n"
                + '\n'.join(s_lines) + "\n\n"
                + '\n'.join(holdings_lines) + "\n\n"
                + '\n'.join(canary_lines) + "\n\n"
                + '\n'.join(drift_lines) + "\n\n"
                + _pd_block
                + '\n'.join(alloc_lines)
            )
        if PORTFOLIO_PUBLIC_URL:
            send_telegram(summary, button_text="대시보드 열기", button_url=PORTFOLIO_PUBLIC_URL)
        else:
            send_telegram(summary)
        print("✅ 텔레그램 일간 리포트 전송 완료")
    except Exception as e:
        print(f"⚠️ 텔레그램 리포트 전송 실패: {e}")

    # 3자산 배분 체크는 위 Daily Report 안 ⚖️ 자산배분 섹션으로 통합됨 (별도 알림 제거)
