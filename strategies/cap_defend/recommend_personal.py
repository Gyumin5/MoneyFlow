"""
Cap Defend V13 Recommendation Script (Personal Version)
=====================================================
V13: Multi Bonus Scoring - RSI(45-70→+0.2), MACD hist>0→+0.2, BB %B>0.5→+0.2
- Includes Auto Turnover Calculation via Upbit API
- Asset Valuation: Uses Last Close Price (not real-time) for strategic consistency
- Detailed Logging Restored
- Price Formatting Fixed (e.g. PEPE)
- Allocation Calculator Added
- Report Layout Aligned (Port -> Calc -> Turnover -> Logs)
- Rebalancing Threshold Guide Added (30%)
- Generates 'portfolio_result_gmoh.html'
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import time
import json
import requests
from datetime import datetime, timezone, timedelta
import pyupbit

# --- Configuration for Auto Turnover ---
from config import UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY
ACCESS_KEY = UPBIT_ACCESS_KEY
SECRET_KEY = UPBIT_SECRET_KEY

# --- 1. Constants & Configuration ---
DATA_DIR = "./data"
SIGNAL_STATE_FILE = os.path.join(".", "signal_state.json")
STRATEGY_VERSION = "V13"
VERSION_HISTORY = [
    ("V13", "2026-03",
     "Multi Bonus scoring, 1% Hysteresis signal flip, T25%+C3d stock rebalancing trigger",
     """<b>자산배분:</b> 주식 60% / 코인 40% (현금 버퍼 2%)

<b>\u25b6 주식 전략 (V11 기반)</b>
\u2022 <b>유니버스:</b> 공격 12종 (SPY, QQQ, EFA, EEM, VT, VEA, GLD, PDBC, QUAL, MTUM, IQLT, IMTM) / 수비 5종 (IEF, BIL, BNDX, GLD, PDBC)
\u2022 <b>Canary Signal:</b> VT AND EEM \u003e SMA200 \u2192 Risk-On / \uc544\ub2c8\uba74 Risk-Off
\u2022 <b>Signal Flip Guard (V13 \uc2e0\uaddc):</b> <span style='color:#d93025;'>1% Hysteresis</span> \u2014 SMA200\xd71.01 \uc774\uc0c1\uc77c \ub54c Risk-On \uc9c4\uc785, SMA200\xd70.99 \uc774\ud558\uc77c \ub54c Risk-Off \uc9c4\uc785. \ud718\uc18c (\ube48\ubc88\ud55c \uc2e0\ud638 \uc804\ud658) 62% \uac10\uc18c
\u2022 <b>\uacf5\uaca9 \ubaa8\ub4dc:</b> \uac00\uc911 \ubaa8\uba58\ud140(50/30/20% for 3M/6M/12M) Top 3 + Sharpe(126d) Top 3 \ud569\uc9d1\ud569 \u2192 \uade0\ub4f1\ubc30\ubd84
\u2022 <b>\uc218\ube44 \ubaa8\ub4dc:</b> \uc218\ube44 5\uc885 \uc911 6\uac1c\uc6d4 \uc218\uc775\ub960 \ucd5c\uace0 1\uac1c (\uc74c\uc218\uba74 \ud604\uae08)
\u2022 <b>\uc885\ubaa9 \ub9ac\ubc38\ub7f0\uc2f1 (V13 \uc2e0\uaddc):</b> <span style='color:#d93025;'>T25%+C3d</span> \u2014 \ud134\uc624\ubc84 \u226525%\uac00 3\uc601\uc5c5\uc77c \uc5f0\uc18d \uc720\uc9c0\ub418\uba74 \ub9ac\ubc38\ub7f0\uc2f1 \uc2e0\ud638 \ubc1c\uc0dd

<b>\u25b6 \ucf54\uc778 \uc804\ub7b5 (V12 \uae30\ubc18)</b>
\u2022 <b>\uc720\ub2c8\ubc84\uc2a4:</b> CoinGecko Top 50 \uc2dc\uac00\ucd1d\uc561\uc21c \u2192 Upbit KRW \uc0c1\uc7a5 + 253\uc77c \ud788\uc2a4\ud1a0\ub9ac + 30\uc77c \ud3c9\uade0 \uac70\ub798\ub300\uae08 10\uc5b5\uc6d0\uc774\uc0c1 \ud544\ud130
\u2022 <b>Canary:</b> BTC \u003e SMA50 \u2192 \ud22c\uc790, \uc544\ub2c8\uba74 \ud604\uae08
\u2022 <b>Health Filter:</b> Price \u003e SMA30 AND Mom21 \u003e 0 AND Vol90 \u2264 10%
\u2022 <b>Scoring (V13 \uc2e0\uaddc):</b> <span style='color:#d93025;'>Multi Bonus</span> = Sharpe(126d)+Sharpe(252d) + RSI(45~70 \u2192 +0.2) + MACD hist\u003e0 \u2192 +0.2 + BB %B\u003e0.5 \u2192 +0.2
\u2022 <b>\uc120\uc815:</b> Multi Bonus Score Top 5
\u2022 <b>\ubc30\ubd84:</b> 90\uc77c \uc5ed\ubcc0\ub3d9\uc131 \uac00\uc911 (1/Vol)

<b>\u25b6 \ub9ac\ubc38\ub7f0\uc2f1 \uaddc\uce59</b>
\u2022 \ucf54\uc778: \uc6d4\uac04 \uc2a4\ucf00\uc904 + \ud134\uc624\ubc84 30%\u2191 \ub610\ub294 Health \uc2e4\ud328 \uc2dc \uc989\uc2dc
\u2022 \uc8fc\uc2dd: \uc6d4\uac04 \uc2a4\ucf00\uc904 + Signal Flip(1% Hysteresis) + T25%+C3d \uc885\ubaa9 \ub9ac\ubc38\ub7f0\uc2f1"""),

    ("V12", "2026-01",
     "Weighted momentum, Sharpe quality, inverse volatility weighting, health check",
     """<b>\uc790\uc0b0\ubc30\ubd84:</b> \uc8fc\uc2dd 60% / \ucf54\uc778 40% (\ud604\uae08 \ubc84\ud37c 2%)

<b>\u25b6 \uc8fc\uc2dd \uc804\ub7b5 (V11 \ub3d9\uc77c)</b>
\u2022 <b>Canary:</b> VT AND EEM \u003e SMA200 \u2192 Risk-On
\u2022 <b>\uacf5\uaca9:</b> 12\uc885 \uc720\ub2c8\ubc84\uc2a4, \uac00\uc911 \ubaa8\uba58\ud140(50/30/20%) Top 3 + Sharpe(126d) Top 3 \ud569\uc9d1\ud569, \uade0\ub4f1\ubc30\ubd84
\u2022 <b>\uc218\ube44:</b> 5\uc885 \uc911 6M \uc218\uc775\ub960 \ucd5c\uace0 1\uac1c (\uc74c\uc218\uba74 \ud604\uae08)
\u2022 Signal flip guard \uc5c6\uc74c (SMA200 \uc790\uccb4 \ub2e8\uc21c \ube44\uad50)

<b>\u25b6 \ucf54\uc778 \uc804\ub7b5 (V12 \uc2e0\uaddc)</b>
\u2022 <b>\uc720\ub2c8\ubc84\uc2a4:</b> CoinGecko Top 50 + Upbit KRW \ud544\ud130 (253\uc77c \ud788\uc2a4\ud1a0\ub9ac, 10\uc5b5\uc6d0 \uac70\ub798\ub300\uae08)
\u2022 <b>Canary:</b> BTC \u003e SMA50
\u2022 <b>Health Filter (V12 \uc2e0\uaddc):</b> Price \u003e SMA30 AND Mom21 \u003e 0 AND Vol90 \u2264 10%
\u2022 <b>Scoring:</b> Sharpe(126d) + Sharpe(252d) (\ubcf4\ub108\uc2a4 \uc5c6\uc74c)
\u2022 <b>\uc120\uc815:</b> Top 5
\u2022 <b>\ubc30\ubd84 (V12 \uc2e0\uaddc):</b> 90\uc77c \uc5ed\ubcc0\ub3d9\uc131 \uac00\uc911 (1/Vol)

<b>\u25b6 \ub9ac\ubc38\ub7f0\uc2f1</b>
\u2022 \uc6d4\uac04 + \ud134\uc624\ubc84 30%\u2191 \ub610\ub294 Health \uc2e4\ud328 \uc2dc"""),

    ("V11", "2025-12",
     "Dual canary, offensive Top3+Top3 union, defensive 6M best",
     """<b>\uc790\uc0b0\ubc30\ubd84:</b> \uc8fc\uc2dd 60% / \ucf54\uc778 40% (\ud604\uae08 \ubc84\ud37c 2%)

<b>\u25b6 \uc8fc\uc2dd \uc804\ub7b5 (V11 \uc2e0\uaddc)</b>
\u2022 <b>Canary (V11 \uc2e0\uaddc):</b> Dual Canary \u2014 VT AND EEM \ub458 \ub2e4 SMA200 \uc774\uc0c1\uc774\uc5b4\uc57c Risk-On
\u2022 <b>\uacf5\uaca9 \uc720\ub2c8\ubc84\uc2a4:</b> SPY, QQQ, EFA, EEM, VT, VEA, GLD, PDBC, QUAL, MTUM, IQLT, IMTM (12\uc885)
\u2022 <b>\uacf5\uaca9 \uc120\uc815 (V11 \uc2e0\uaddc):</b> \uac00\uc911\ubaa8\uba58\ud140(50/30/20%) Top 3 + Sharpe(126d) Top 3 \u2192 \ud569\uc9d1\ud569 (3~6\uc885), \uade0\ub4f1\ubc30\ubd84
\u2022 <b>\uc218\ube44 \uc720\ub2c8\ubc84\uc2a4:</b> IEF, BIL, BNDX, GLD, PDBC (5\uc885)
\u2022 <b>\uc218\ube44 \uc120\uc815 (V11 \uc2e0\uaddc):</b> 6\uac1c\uc6d4 \uc218\uc775\ub960 \ucd5c\uace0 1\uac1c (\uc74c\uc218\uba74 \ud604\uae08)

<b>\u25b6 \ucf54\uc778 \uc804\ub7b5</b>
\u2022 <b>\uc720\ub2c8\ubc84\uc2a4:</b> CoinGecko Top 50 + Upbit KRW
\u2022 <b>Canary:</b> BTC \u003e SMA50
\u2022 <b>Health Filter:</b> \uc5c6\uc74c (V12\uc5d0\uc11c \ucd94\uac00)
\u2022 <b>Scoring:</b> Sharpe(126d) + Sharpe(252d), Top 5 \uade0\ub4f1\ubc30\ubd84

<b>\u25b6 \ub9ac\ubc38\ub7f0\uc2f1</b>
\u2022 \uc6d4\uac04 \uc2a4\ucf00\uc904 \uae30\ubc18"""),
]

STOCK_RATIO, COIN_RATIO = 0.60, 0.40
CASH_ASSET = 'Cash'
CASH_BUFFER_PERCENT = 0.02 # 2% Cash Buffer
STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'TUSD', 'PAX', 'GUSD', 'FRAX', 'LUSD', 'MIM', 'USDN', 'FDUSD']

# Stock V10 Configuration
OFFENSIVE_STOCK_UNIVERSE = ['SPY', 'QQQ', 'EFA', 'EEM', 'VT', 'VEA', 'GLD', 'PDBC', 'QUAL', 'MTUM', 'IQLT', 'IMTM']
DEFENSIVE_STOCK_UNIVERSE = ['IEF', 'BIL', 'BNDX', 'GLD', 'PDBC']
CANARY_ASSETS = ['VT', 'EEM']
STOCK_CANARY_MA_PERIOD = 200
HYSTERESIS_BAND = 0.01  # 1% Hysteresis: enter Risk-On at SMA200*1.01, exit at SMA200*0.99
N_FACTOR_ASSETS = 3

# Coin V11 Configuration
COIN_CANARY_MA_PERIOD = 50
HEALTH_FILTER_MA_PERIOD = 30
HEALTH_FILTER_RETURN_PERIOD = 21
N_SELECTED_COINS = 5
VOLATILITY_WINDOW = 90

# --- 2. Dynamic Coin Universe (V12: LIVE CoinGecko Top 50 + Upbit Filter) ---
VOL_CAP_FILTER = 0.10  # V12: 10% daily vol cap

def get_dynamic_coin_universe(log: list) -> (list, dict):
    print("\n--- 🛰️ Step 1: Coin Universe Selection (V12: LIVE CoinGecko + Upbit Filter) ---")
    log.append("<h2>🛰️ Step 1: 코인 유니버스 선정 (V12: Live CoinGecko Top 50)</h2>")
    
    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
    FETCH_LIMIT = 100 
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
        log.append("<p class='error'>❌ All Methods Failed. Using Hardcoded Fallback.</p>")
        fallback = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD', 'AVAX-USD', 'LINK-USD', 'SHIB-USD', 'TRX-USD']
        return fallback, {}
    
    cg_symbol_to_id_map = {f"{item['symbol'].upper()}-USD": item['id'] for item in cg_data}
    
    print("  - Fetching Upbit KRW Market...")
    try:
        upbit_krw_tickers = pyupbit.get_tickers(fiat="KRW")
        upbit_symbols = {t.split('-')[1] for t in upbit_krw_tickers}
        print(f"    ✅ Upbit has {len(upbit_symbols)} KRW markets")
    except Exception as e:
        print(f"    ⚠️ Upbit API Error: {e}")
        upbit_symbols = set()
    
    print("  - Filtering by Upbit availability, liquidity, and history...")
    final_universe = []
    
    for item in cg_data:
        symbol = item['symbol'].upper()
        if symbol in STABLECOINS: continue
        if symbol not in upbit_symbols:
            print(f"    ❌ {symbol}: Not in Upbit KRW")
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
        
        if len(final_universe) >= 50: break
    
    log.append(f"<p>선정된 유니버스 ({len(final_universe)}개): Top 50 qualified</p>")
    return final_universe, cg_symbol_to_id_map


# --- Helper: Get Upbit Assets (Personal) ---
def get_current_upbit_holdings(log):
    """
    업비트 API를 사용하여 현재 보유 자산을 조회합니다.
    - holdings_qty: {ticker-USD: qty} (수량만 - 종가 평가용)
    - holdings_krw: {ticker-USD: krw_value} (실시간 KRW 가치 - 표시용)
    - unlisted coins are filtered out
    """
    try:
        upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)
        
        krw = upbit.get_balance("KRW")
        if krw is None:
            print("⚠️ Upbit API Connection Failed (Check Access/Secret Keys)")
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

        # 2. Download from Yahoo
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
                
                # [Quality Check V12.2] Yahoo 종가(USD→KRW) vs 업비트 종가(KRW) 비교
                if ticker.endswith('-USD') and len(df) > 0:
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
    if len(s) < 253: return -np.inf
    r3, r6, r12 = calc_ret(s, 63), calc_ret(s, 126), calc_ret(s, 252)
    return 0.5*r3 + 0.3*r6 + 0.2*r12

# --- Multi Bonus Indicators (V13) ---
def calc_rsi(s, period=14):
    """RSI(14) from closing prices"""
    if len(s) < period + 1: return np.nan
    delta = s.diff().iloc[-period-1:]
    gain = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
    if loss == 0: return 100.0
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calc_macd_hist(s):
    """MACD Histogram (12,26,9) from closing prices"""
    if len(s) < 35: return np.nan
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return (macd_line - signal).iloc[-1]

def calc_bb_pctb(s, period=20):
    """Bollinger Band %B from closing prices"""
    if len(s) < period: return np.nan
    sma = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    band_width = upper.iloc[-1] - lower.iloc[-1]
    if band_width == 0: return 0.5
    return (s.iloc[-1] - lower.iloc[-1]) / band_width

def calc_multi_bonus_score(s):
    """Base Sharpe score + RSI/MACD/BB bonus (+0.2 each)"""
    base = calc_sharpe(s, 126) + calc_sharpe(s, 252)
    rsi = calc_rsi(s)
    macd_h = calc_macd_hist(s)
    pctb = calc_bb_pctb(s)
    if pd.notna(rsi) and 45 <= rsi <= 70: base += 0.2
    if pd.notna(macd_h) and macd_h > 0: base += 0.2
    if pd.notna(pctb) and pctb > 0.5: base += 0.2
    return base, rsi, macd_h, pctb

def run_stock_strategy_v11(log, all_prices, target_date):
    log.append("<h2>📈 주식 포트폴리오 분석 (V11)</h2>")
    vt, eem = all_prices.get('VT'), all_prices.get('EEM')
    meta = {'signal_dist': {}, 'next_candidates': []}
    
    if len(vt) >= 200 and len(eem) >= 200:
        vt_sma, eem_sma = vt.rolling(200).mean().iloc[-1], eem.rolling(200).mean().iloc[-1]
        vt_cur, eem_cur = vt.iloc[-1], eem.iloc[-1]
        meta['signal_dist'] = {'VT': (vt_cur-vt_sma)/vt_sma, 'EEM': (eem_cur-eem_sma)/eem_sma}
        
        # Load previous signal state
        prev_risk_on = None
        try:
            with open(SIGNAL_STATE_FILE, 'r') as _sf:
                prev_risk_on = json.load(_sf).get('risk_on')
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        
        # 1% Hysteresis Signal Flip Guard
        upper = 1 + HYSTERESIS_BAND  # 1.01
        lower = 1 - HYSTERESIS_BAND  # 0.99
        
        if prev_risk_on is None:
            # First run: use simple comparison
            risk_on = vt_cur > vt_sma and eem_cur > eem_sma
        elif prev_risk_on:
            # Currently Risk-On: only flip to Off if BELOW lower band
            risk_on = not (vt_cur < vt_sma * lower or eem_cur < eem_sma * lower)
        else:
            # Currently Risk-Off: only flip to On if ABOVE upper band
            risk_on = vt_cur > vt_sma * upper and eem_cur > eem_sma * upper
        
        # Save new state
        with open(SIGNAL_STATE_FILE, 'w') as _sf:
            json.dump({'risk_on': bool(risk_on), 'updated': datetime.now().strftime('%Y-%m-%d %H:%M')}, _sf)
        
        # Logging
        hyst_info = f"(Hyst {HYSTERESIS_BAND:.0%}: enter >{upper:.2f}x, exit <{lower:.2f}x)"
        flip_info = ""
        if prev_risk_on is not None and prev_risk_on != risk_on:
            flip_info = " \U0001f504 <b>SIGNAL FLIP</b>"
        
        log.append(f"<p><b>[Canary]</b> VT: ${vt_cur:.2f} (MA200 ${vt_sma:.2f}, {meta['signal_dist']['VT']:+.2%}) | EEM: ${eem_cur:.2f} (MA200 ${eem_sma:.2f}, {meta['signal_dist']['EEM']:+.2%})</p>")
        log.append(f"<p><b>[Hysteresis]</b> {hyst_info} | Prev: {'On' if prev_risk_on else 'Off' if prev_risk_on is not None else 'N/A'}{flip_info}</p>")
        if risk_on: log.append("<p>\u2705 <b>Risk-On</b></p>")
        else: log.append("<p>\U0001f6a8 <b>Risk-Off</b></p>")
    else: 
        risk_on = False
        log.append("<p class='error'>Canary Data Missing</p>")

    if risk_on:
        log.append("<h4>🚀 공격 모드</h4>")
        scores = []
        for t in OFFENSIVE_STOCK_UNIVERSE:
            p = all_prices.get(t)
            if len(p) >= 253: scores.append({'Ticker': t, 'Mom': calc_weighted_mom(p), 'Qual': calc_sharpe(p, 126)})
        df = pd.DataFrame(scores).set_index('Ticker')
        
        try:
            log.append(f"<div class='table-wrap'>{df.to_html(classes='dataframe small-table')}</div>")
        except: pass

        top_m = df.sort_values('Mom', ascending=False).head(3).index.tolist()
        top_q = df.sort_values('Qual', ascending=False).head(3).index.tolist()
        picks = list(set(top_m + top_q))
        meta['selection_reason'] = {'Mom_Picks': top_m, 'Qual_Picks': top_q}
        
        log.append(f"<p>Final Picks: <b>{picks}</b></p>")
        return {t: 1.0/len(picks) for t in picks}, "공격 모드", meta
    else:
        log.append("<h4>🛡️ 수비 모드</h4>")
        res = []
        for t in DEFENSIVE_STOCK_UNIVERSE:
            r = calc_ret(all_prices.get(t), 126)
            if pd.notna(r): res.append({'Ticker': t, '6m Ret': r})
        
        try:
            log.append(f"<div class='table-wrap'>{pd.DataFrame(res).sort_values('6m Ret', ascending=False).to_html(classes='dataframe small-table')}</div>")
        except: pass
            
        best = sorted(res, key=lambda x: x['6m Ret'], reverse=True)[0]
        if best['6m Ret'] < 0: return {CASH_ASSET: 1.0}, "수비 (현금)", meta
        return {best['Ticker']: 1.0}, f"수비 ({best['Ticker']})", meta

def run_coin_strategy_v12(coin_universe, all_prices, target_date, log, is_today=True):
    date_str = target_date.date()
    log.append(f"<h3>🪙 코인 포트폴리오 (V12) ({date_str})</h3>")
    meta = {'signal_dist': {}, 'next_candidates': []}
    
    btc = all_prices.get('BTC-USD')
    if len(btc) < 50: return {CASH_ASSET: 1.0}, "데이터 부족", meta, log, []
    
    sma50 = btc.rolling(50).mean().iloc[-1]
    cur = btc.iloc[-1]
    meta['signal_dist'] = {'BTC': (cur - sma50) / sma50}
    
    # BTC 기준 Target Date와 비교
    tgt_dt = target_date.date() if hasattr(target_date, 'date') else target_date
    log.append(f"<p>[BTC Canary] ${cur:,.0f} vs MA50 ${sma50:,.0f} (Date: {tgt_dt})</p>")
    
    healthy = []
    rows = []
    
    def get_volatility(s, n=90): return s.pct_change().iloc[-n:].std()
    def fmt_price(p):
        if p < 1: return f"${p:,.8f}"
        if p < 100: return f"${p:,.4f}"
        return f"${p:,.2f}"
    
    for t in coin_universe:
        p = all_prices.get(t)
        if len(p) < 35: continue
        
        # [Strict Date Check]
        last_dt = p.index[-1].date() if hasattr(p.index[-1], 'date') else p.index[-1]
        diff_days = (tgt_dt - last_dt).days
        if diff_days != 0: # 날짜가 하루라도 다르면 데이터 믹스 위험 -> 제외
            continue
            
        sma30 = p.rolling(30).mean().iloc[-1]
        mom21 = calc_ret(p, 21)
        cur_p = p.iloc[-1]
        vol90 = get_volatility(p, 90)
        
        is_ok = (cur_p > sma30) and (mom21 > 0) and (vol90 <= VOL_CAP_FILTER)
        status = "🟢" if is_ok else "🔴"
        
        rows.append({'Coin': t, 'Price': fmt_price(cur_p), 'SMA30': fmt_price(sma30), 'Mom21': f"{mom21:.2%}", 'Vol90': f"{vol90:.4f}", 'Status': status})
        if is_ok: healthy.append(t)
            
    try:
        log.append(f"<div class='table-wrap'>{pd.DataFrame(rows).to_html(classes='dataframe small-table', index=False)}</div>")
    except: pass

    if cur <= sma50: return {CASH_ASSET: 1.0}, "Risk-Off", meta, log, []
    
    log.append(f"<p>🔍 Healthy Coins Found: <b>{len(healthy)}</b> items {healthy}</p>")
    if not healthy: return {CASH_ASSET: 1.0}, "No Healthy", meta, log, []
    
    # V13: Multi Bonus Scoring (Sharpe + RSI/MACD/BB bonuses)
    scores = []
    for t in healthy:
        score, rsi, macd_h, pctb = calc_multi_bonus_score(all_prices[t])
        bonus_flags = []
        if pd.notna(rsi) and 45 <= rsi <= 70: bonus_flags.append('RSI')
        if pd.notna(macd_h) and macd_h > 0: bonus_flags.append('MACD')
        if pd.notna(pctb) and pctb > 0.5: bonus_flags.append('BB')
        scores.append({
            'Coin': t, 'Score': score,
            'RSI': f"{rsi:.1f}" if pd.notna(rsi) else "-",
            'MACD_H': f"{macd_h:.4f}" if pd.notna(macd_h) else "-",
            'BB%B': f"{pctb:.2f}" if pd.notna(pctb) else "-",
            'Bonus': '+'.join(bonus_flags) if bonus_flags else '-'
        })
    score_df = pd.DataFrame(scores).sort_values('Score', ascending=False)

    try:
        log.append(f"<p><b>[V13 Multi Bonus]</b> RSI(45-70→+0.2) | MACD hist>0→+0.2 | BB %B>0.5→+0.2</p>")
        log.append(f"<div class='table-wrap'>{score_df.head(10).to_html(classes='dataframe small-table')}</div>")
    except: pass
    
    top5 = score_df.head(5)['Coin'].tolist()
    meta['next_candidates'] = score_df.iloc[5:10]['Coin'].tolist()
    
    vols = {t: all_prices[t].pct_change().iloc[-90:].std() for t in top5}
    inv_vols = {t: 1/v for t, v in vols.items() if v > 0}
    tot = sum(inv_vols.values())
    weights = {t: v/tot for t, v in inv_vols.items()} if tot > 0 else {t: 1/len(top5) for t in top5}
    
    # [Feedback] Show InvVol or Vol in the table
    w_rows = [{'Coin': t, 'Weight': f"{w:.2%}", 'Vol': f"{vols[t]:.4f}", 'InvVol': f"{inv_vols[t]:.2f}"} for t, w in weights.items()]
    try:
        log.append(f"<div class='table-wrap'>{pd.DataFrame(w_rows).to_html(classes='dataframe small-table', index=False)}</div>")
    except: pass
    
    return weights, "Full Invest", meta, log, healthy

def save_html(log_global, final_port, s_port, c_port, s_stat, c_stat, turnover, log_today, log_yesterday, date_today, asset_prices_krw, s_meta, c_meta, coin_health_status, cur_assets_raw=None, action_guide="", diff_table_rows=None, stock_trigger_info=None):
    filepath = "portfolio_result_gmoh.html"

    # Generate stock trigger HTML
    stock_trigger_html = ""
    if stock_trigger_info:
        ti = stock_trigger_info
        my_tickers = ti.get('my_tickers', [])
        rec_tickers = ti.get('rec_tickers', [])
        turnover_pct = ti.get('turnover', 0)
        consec_days = ti.get('consec_days', 0)
        should_rebal = ti.get('should_rebal', False)

        if my_tickers:
            added = sorted(set(rec_tickers) - set(my_tickers))
            removed = sorted(set(my_tickers) - set(rec_tickers))

            changes_html = ""
            if added:
                changes_html += f"<span style='color:#d93025; font-weight:600;'>+ {', '.join(added)}</span> "
            if removed:
                changes_html += f"<span style='color:#1a73e8; font-weight:600;'>- {', '.join(removed)}</span>"
            if not added and not removed:
                changes_html = "<span style='color:#0d904f;'>변동 없음</span>"

            alert_style = "background: #fce8e6; border: 2px solid #d93025; padding: 12px; border-radius: 8px; margin-top: 10px;" if should_rebal else "background: #e8f0fe; padding: 12px; border-radius: 8px; margin-top: 10px;"
            alert_icon = "\U0001f6a8" if should_rebal else "\u2139\ufe0f"
            alert_msg = f"<b>REBALANCE \uad8c\uace0</b> (Turnover {turnover_pct:.0%}, {consec_days}\uc77c \uc5f0\uc18d)" if should_rebal else f"Turnover {turnover_pct:.0%}, {consec_days}\uc77c \uc5f0\uc18d (3\uc77c \ubbf8\ub9cc \ub610\ub294 25% \ubbf8\ub9cc)"

            stock_trigger_html = f"""
                <div style="margin-top: 10px;">
                    <div style="display: flex; gap: 20px; flex-wrap: wrap; margin-bottom: 8px;">
                        <div><b>My:</b> {', '.join(my_tickers)} ({len(my_tickers)})</div>
                        <div><b>Rec:</b> {', '.join(rec_tickers)} ({len(rec_tickers)})</div>
                    </div>
                    <div style="margin-bottom: 8px;">Changes: {changes_html}</div>
                    <div style="{alert_style}">
                        {alert_icon} {alert_msg}
                    </div>
                </div>"""

    items = []
    for t, w in final_port.items(): items.append({'종목': t, '자산군': "현금" if t == CASH_ASSET else ("코인" if t in c_port else "주식"), '비중': w})
    items.sort(key=lambda x: (x['자산군']!='현금', x['비중']), reverse=True)
    
    tbody = "".join([f"<tr><td>{i['종목']}</td><td>{i['자산군']}</td><td>{i['비중']:.2%}</td></tr>" for i in items])
    
    # [Table] Integrated Portfolio (My vs Target)
    integrated_html = ""
    if diff_table_rows:
        integrated_html = f"<h3>Turnover: {turnover:.2%} ({action_guide})</h3>"
        integrated_html += "<table class='mobile-card-table'><thead><tr><th>Asset</th><th>Value (KRW)</th><th>My</th><th>Target</th><th>Diff</th><th>Action</th></tr></thead><tbody>"
        
        total_value_sum = sum(item['Value'] for item in diff_table_rows)
        
        for row in diff_table_rows:
            color = ""
            if "BUY" in row['Action']: color = "color:red; font-weight:bold;"
            elif "SELL" in row['Action']: color = "color:blue; font-weight:bold;"
            
            val_fmt = f"{int(row['Value']):,}" if row['Value'] > 0 else "0"
            
            integrated_html += f"<tr><td data-label='Asset'>{row['Asset']}</td><td data-label='Value (KRW)'>{val_fmt}</td><td data-label='My'>{row['My']:.1%}</td><td data-label='Target'>{row['Target']:.1%}</td><td data-label='Diff'>{row['Diff']:+.1%}</td><td data-label='Action' style='{color}'>{row['Action']}</td></tr>"
        
        integrated_html += f"<tr><td data-label='Total' style='font-weight:bold;'>Total</td><td data-label='Value (KRW)' style='font-weight:bold;'>{int(total_value_sum):,}</td><td data-label='My'></td><td data-label='Target'></td><td data-label='Diff'></td><td data-label='Action'></td></tr>"
        integrated_html += "</tbody></table>"

    # Strategy documentation link
    version_html = f"""
            <div style="margin:20px 0; text-align:center;">
                <a href="strategy.html" style="display:inline-block; padding:10px 24px; background:#f0f4ff; border:1px solid #1a73e8; border-radius:8px; color:#1a73e8; text-decoration:none; font-weight:600;">
                    \U0001f4cb \uc804\ub7b5 \ubc84\uc804 \ud788\uc2a4\ud1a0\ub9ac ({STRATEGY_VERSION})
                </a>
            </div>
    """

    # Embed recommended stock tickers for client-side calculation
    rec_stock_list = sorted([t for t in s_port.keys() if t != 'Cash'])
    rec_stock_json = json.dumps(rec_stock_list)

    # Embed consec days for client-side display
    _consec_days = 0
    _stock_turnover_pct = 0
    if stock_trigger_info:
        _consec_days = stock_trigger_info.get('consec_days', 0)
        _stock_turnover_pct = round(stock_trigger_info.get('turnover', 0) * 100)

    stock_holdings_js = """
            <script>
            const REC_STOCK_TICKERS = """ + rec_stock_json + """;
            const CONSEC_DAYS = """ + str(_consec_days) + """;
            const SERVER_TURNOVER = """ + str(_stock_turnover_pct) + """;

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
                    changesHtml = '<span style="color:#0d904f;">\\u2705 \\ubcc0\\ub3d9 \\uc5c6\\uc74c</span>';

                const pct = Math.round(result.turnover * 100);
                const isHigh = pct >= 25;
                const consec = CONSEC_DAYS;
                const needDays = Math.max(0, 3 - consec);
                const triggered = isHigh && consec >= 3;

                let statusHtml = '';
                if (!isHigh) {
                    statusHtml = '<div style="background:#e8f0fe; padding:12px; border-radius:8px; margin-top:10px;">'
                        + '\\u2139\\ufe0f Turnover ' + pct + '% (25% \\ubbf8\\ub9cc) \\u2014 <b>\\ub9ac\\ubc38\\ub7f0\\uc2f1 \\ubd88\\ud544\\uc694</b></div>';
                } else if (triggered) {
                    statusHtml = '<div style="background:#fce8e6; border:2px solid #d93025; padding:12px; border-radius:8px; margin-top:10px;">'
                        + '\\U0001f6a8 <b>Turnover ' + pct + '% / ' + consec + '\\uc77c \\uc5f0\\uc18d \\u2192 \\ub9ac\\ubc38\\ub7f0\\uc2f1 \\uad8c\\uace0</b></div>';
                } else {
                    statusHtml = '<div style="background:#fff3e0; border:1px solid #ff9800; padding:12px; border-radius:8px; margin-top:10px;">'
                        + '\\u23f3 Turnover ' + pct + '% / ' + consec + '\\uc77c\\uc9f8 (3\\uc77c \\uc5f0\\uc18d \\uc2dc \\ud2b8\\ub9ac\\uac70, <b>' + needDays + '\\uc77c \\ub0a8\\uc74c</b>)</div>';
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

            // Load on page init
            (async function() {
                try {
                    const res = await fetch('http://' + window.location.hostname + ':5000/api/holdings');
                    const data = await res.json();
                    if (data.tickers && data.tickers.length > 0) {
                        document.getElementById('stockInput').value = data.tickers.join(' ');
                        document.getElementById('holdingsStatus').innerHTML =
                            '\u2705 \uc800\uc7a5\ub428: ' + data.tickers.join(', ') + ' (' + data.updated + ')';
                        renderTrigger(data.tickers);
                    }
                } catch(e) {
                    document.getElementById('holdingsStatus').innerHTML = '\u26a0\ufe0f API \uc5f0\uacb0 \uc2e4\ud328';
                }
            })();

            async function saveHoldings() {
                const input = document.getElementById('stockInput').value.trim();
                const status = document.getElementById('holdingsStatus');
                try {
                    const res = await fetch('http://' + window.location.hostname + ':5000/api/holdings', {
                        method: 'POST',
                        headers: { 'Content-Type': 'application/json' },
                        body: JSON.stringify({ tickers: input })
                    });
                    const data = await res.json();
                    status.innerHTML = '\u2705 ' + data.message + ' (' + data.holdings.tickers.join(', ') + ')';
                    status.style.color = '#0d904f';
                    // Immediately calculate and show trigger
                    renderTrigger(data.holdings.tickers);
                } catch(e) {
                    status.innerHTML = '\u274c \uc800\uc7a5 \uc2e4\ud328';
                    status.style.color = '#d93025';
                }
            }
            </script>
    """

    html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cap Defend {STRATEGY_VERSION} Recommendation (Personal)</title>
         <style>
            body {{ font-family: -apple-system, sans-serif; background: #f0f2f5; padding: 10px; color: #333; }}
            .container {{ max-width: 800px; margin: 0 auto; background: #fff; padding: 20px; border-radius: 16px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 20px; }}
            th, td {{ padding: 10px; border-bottom: 1px solid #f1f3f4; text-align: left; font-size: 0.95em; }}
            th {{ background-color: #fafafa; font-weight: 600; color: #555; }}
            .card {{ background: #fff; padding: 15px; border-radius: 12px; border: 1px solid #e0e0e0; margin-bottom: 10px; }}
            .status-bar {{ display: flex; gap: 10px; background: #e8f0fe; padding: 15px; border-radius: 12px; margin-bottom: 20px; color: #1967d2; font-weight: 500; flex-wrap: wrap; }}
            .dataframe {{ width: 100%; border: 1px solid #ddd; border-collapse: collapse; margin: 10px 0; }}
            .dataframe th, .dataframe td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .dataframe tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .small-table {{ font-size: 0.9em; }}
            .table-wrap {{ overflow-x: auto; }}
            
            /* Mobile Responsive Table */
            @media screen and (max-width: 600px) {{
                .mobile-card-table thead {{ display: none; }}
                .mobile-card-table tr {{ 
                    display: block; 
                    margin-bottom: 15px; 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    background: #fff;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.05);
                }}
                .mobile-card-table td {{ 
                    display: flex; 
                    justify-content: space-between; 
                    padding: 12px; 
                    border-bottom: 1px solid #eee; 
                    text-align: right;
                }}
                .mobile-card-table td:last-child {{ border-bottom: none; }}
                .mobile-card-table td::before {{ 
                    content: attr(data-label); 
                    font-weight: 600; 
                    color: #555; 
                    text-align: left;
                }}
                /* Asset Name styling in card mode */
                .mobile-card-table td:first-child {{
                    background: #f8f9fa;
                    font-weight: bold;
                    color: #1a73e8;
                    border-radius: 8px 8px 0 0;
                }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚀 Cap Defend {STRATEGY_VERSION} (Personal)</h1>
            <p>기준일: {date_today.strftime('%Y-%m-%d')} | 종가 기준</p>
            
            <div class="status-bar">
                <div>📉 주식: {s_stat}</div>
                <div>🪙 코인: {c_stat}</div>
            </div>
            
            <!-- Force Trade Buttons -->
            <div style="margin-bottom: 20px; display: flex; gap: 10px; flex-wrap: wrap; align-items: center;">
                <button id="forceTradeUpbitBtn" onclick="forceTrade('upbit')" style="
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    color: white;
                    border: none;
                    padding: 12px 24px;
                    border-radius: 8px;
                    font-size: 1em;
                    font-weight: 600;
                    cursor: pointer;
                    box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4);
                    transition: all 0.3s ease;
                ">
                    ⚡ Force Trade (Upbit)
                </button>

                <span id="tradeStatus" style="margin-left: 10px; font-weight: 500;"></span>
            </div>
            
            <script>
            async function forceTrade(exchange) {{
                const btn = document.getElementById('forceTrade' + exchange.charAt(0).toUpperCase() + exchange.slice(1) + 'Btn');
                const status = document.getElementById('tradeStatus');
                const exchangeName = 'Upbit';

                // 암호 입력 → 서버에서 검증
                const inputPwd = prompt('거래 암호를 입력하세요:');
                if (!inputPwd) return;

                // 금액 입력 (0 또는 빈값: 전체 자산 운용)
                const amountInput = prompt('운용 금액을 입력하세요 (원):\\n(0 또는 빈값 입력 시 전체 자산 운용)', '0');
                if (amountInput === null) return;
                const amount = parseInt(amountInput.replace(/,/g, '')) || 0;

                const amountText = amount > 0 ? amount.toLocaleString() + '원' : '전체 자산';
                if (!confirm(exchangeName + ' Force Trade를 실행하시겠습니까?\\n운용 금액: ' + amountText + '\\n(실거래가 발생합니다!)')) {{
                    return;
                }}

                btn.disabled = true;
                btn.style.opacity = '0.6';
                status.innerHTML = '⏳ ' + exchangeName + ' 실행 중... (' + amountText + ')';
                status.style.color = '#1967d2';

                try {{
                    const response = await fetch('http://' + window.location.hostname + ':5000/api/trade/' + exchange, {{
                        method: 'POST',
                        headers: {{ 'Content-Type': 'application/json' }},
                        body: JSON.stringify({{ target_amount: amount, password: inputPwd }})
                    }});

                    const data = await response.json();

                    if (response.ok) {{
                        status.innerHTML = '✅ ' + data.message;
                        status.style.color = '#0d904f';
                    }} else {{
                        status.innerHTML = '⚠️ ' + (data.error || 'Error');
                        status.style.color = '#d93025';
                    }}
                }} catch (error) {{
                    status.innerHTML = '❌ API 연결 실패 (서버 확인 필요)';
                    status.style.color = '#d93025';
                }}

                setTimeout(() => {{
                    btn.disabled = false;
                    btn.style.opacity = '1';
                }}, 5000);
            }}
            </script>
            

            <!-- Stock Holdings & Rebalancing Trigger -->
            <h2>📈 주식 ETF 리밸런싱</h2>
            <div class="card">
                <div style="margin-bottom: 15px;">
                    <label style="font-weight: 600; color: #555;">현재 보유 주식 (띄어쓰기로 구분):</label>
                    <div style="display: flex; gap: 8px; margin-top: 8px; flex-wrap: wrap;">
                        <input id="stockInput" type="text" placeholder="예: SPY QQQ MTUM GLD EFA QUAL"
                            style="flex: 1; min-width: 200px; padding: 10px; border: 1px solid #ddd; border-radius: 8px; font-size: 1em; font-family: monospace;" />
                        <button onclick="saveHoldings()" style="
                            background: #1a73e8; color: white; border: none; padding: 10px 20px;
                            border-radius: 8px; font-weight: 600; cursor: pointer;">저장</button>
                    </div>
                    <div id="holdingsStatus" style="margin-top: 8px; font-size: 0.9em; color: #666;"></div>
                </div>
                <div id="triggerResult"></div>
            </div>

            {stock_holdings_js}

            <h2>🪙 통합 포트폴리오 현황</h2>
            <div class="card">
                {integrated_html}
            </div>

            <h2>📊 최종 추천 비중 (Stock + Coin)</h2>
            <table><thead><tr><th>종목</th><th>자산군</th><th>비중</th></tr></thead><tbody>{tbody}</tbody></table>
            
            {version_html}

            <h2>📜 상세 로그</h2>
            {''.join(log_global)}
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
    all_tickers = set(OFFENSIVE_STOCK_UNIVERSE + DEFENSIVE_STOCK_UNIVERSE + CANARY_ASSETS + c_univ + ['BTC-USD'] + list(my_holdings_qty.keys()))
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
    
    s_port, s_stat, s_meta = run_stock_strategy_v11(log, prices, target_date)
    c_port, c_stat, c_meta, log, healthy_coins = run_coin_strategy_v12(c_univ, prices, target_date, log)
    
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
    c_port_buffered = {}
    if CASH_ASSET in c_port and c_port[CASH_ASSET] == 1.0:
        c_port_buffered = {CASH_ASSET: 1.0}
    else:
        for t, w in c_port.items():
            c_port_buffered[t] = w * (1.0 - CASH_BUFFER_PERCENT)
        c_port_buffered[CASH_ASSET] = c_port_buffered.get(CASH_ASSET, 0.0) + CASH_BUFFER_PERCENT

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
        
    # [Restored] Final Portfolio (Stock + Coin) for Report Table
    final_port = {CASH_ASSET: 0}
    for t, w in s_port.items(): 
        key = t if t!=CASH_ASSET else CASH_ASSET
        final_port[key] = final_port.get(key, 0) + w * STOCK_RATIO
    for t, w in c_port_buffered.items():
        key = t if t!=CASH_ASSET else CASH_ASSET
        final_port[key] = final_port.get(key, 0) + w * COIN_RATIO
    
    # Get KRW Prices (생략 가능, 위에서 integrated_rows에 다 넣음)
    krw_prices = {}
        
    # Pass my_holdings_krw, integrated_rows for unified display
    # --- Stock Rebalancing Trigger (T25%+C3d) ---
    import json as _json
    STOCK_HOLDINGS_FILE = '/home/ubuntu/my_stock_holdings.json'
    STOCK_TRIGGER_FILE = '/home/ubuntu/stock_trigger_history.json'

    stock_trigger_info = None
    try:
        # Load my stock holdings
        with open(STOCK_HOLDINGS_FILE, 'r') as _f:
            _holdings = _json.load(_f)
        my_stock_tickers = sorted([t.upper() for t in _holdings.get('tickers', [])])

        # Get today's recommended stock tickers
        rec_stock_tickers = sorted([t for t in s_port.keys() if t != CASH_ASSET])

        if my_stock_tickers:
            # Calculate turnover (equal weight based)
            all_t = set(my_stock_tickers) | set(rec_stock_tickers)
            my_w = {t: 1.0/len(my_stock_tickers) if t in my_stock_tickers else 0 for t in all_t}
            rec_w = {t: 1.0/len(rec_stock_tickers) if t in rec_stock_tickers else 0 for t in all_t} if rec_stock_tickers else {t: 0 for t in all_t}
            stock_turnover = sum(abs(rec_w[t] - my_w[t]) for t in all_t) / 2

            # Load trigger history
            try:
                with open(STOCK_TRIGGER_FILE, 'r') as _f:
                    _hist = _json.load(_f)
            except:
                _hist = {"consec_days": 0, "last_rec": [], "last_date": ""}

            # Check consecutive days (only increment once per calendar day)
            last_rec = sorted(_hist.get("last_rec", []))
            last_date = _hist.get("last_date", "")
            today_str = target_date.strftime('%Y-%m-%d')
            
            if today_str == last_date:
                # Same day re-run: keep existing count, don't increment
                consec = _hist.get("consec_days", 1)
            elif rec_stock_tickers == last_rec:
                consec = _hist.get("consec_days", 0) + 1
            else:
                consec = 1

            # Save today's state
            _hist = {
                "consec_days": consec,
                "last_rec": rec_stock_tickers,
                "last_date": target_date.strftime('%Y-%m-%d'),
                "turnover": stock_turnover
            }
            with open(STOCK_TRIGGER_FILE, 'w') as _f:
                _json.dump(_hist, _f, indent=2)

            # Trigger: turnover >= 25% AND consec >= 3
            should_rebal = round(stock_turnover, 4) >= 0.25 and consec >= 3

            stock_trigger_info = {
                "my_tickers": my_stock_tickers,
                "rec_tickers": rec_stock_tickers,
                "turnover": stock_turnover,
                "consec_days": consec,
                "should_rebal": should_rebal
            }

            trigger_str = "REBALANCE" if should_rebal else "HOLD"
            print(f"[Stock Trigger] My: {my_stock_tickers} -> Rec: {rec_stock_tickers}")
            print(f"[Stock Trigger] Turnover: {stock_turnover:.1%}, Consec: {consec}d -> {trigger_str}")
    except FileNotFoundError:
        print("[Stock Trigger] No holdings file found - skipping trigger check")
    except Exception as _e:
        print(f"[Stock Trigger] Error: {_e}")

    save_html(log, final_port, s_port, c_port, s_stat, c_stat, turnover, [], [], target_date, krw_prices, s_meta, c_meta, {}, my_holdings_krw, action_guide, integrated_rows, stock_trigger_info)
