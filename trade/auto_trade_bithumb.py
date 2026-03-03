"""
V12 Bithumb Auto Trader (CoinGecko Global Universe)
===================================================
1. CoinGecko Global Top 100 조회
2. Stablecoin 제외
3. 빗썸 상장 여부 확인 (KRW 마켓)
4. V12 전략 필터 (SMA30, VolCap, Sharpe) 적용
5. 2% 현금 버퍼 유지 (수수료/슬리피지 방지)
6. **No Local History**: 직접 빗썸 잔고 조회하여 포트폴리오 계산
7. **Raw API Implementation**: pybithumb 래퍼 대신 Raw API 직접 구현하여 정교한 주문 (5600 에러 해결)
8. **Risk-Off Logic**: 리스크 오프 또는 타겟 공백 시 턴오버 무시하고 강제 청산
"""

import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))

import json
import time
import math
import base64
import hmac
import hashlib
import urllib.parse
import argparse
from datetime import datetime, timezone
from typing import Dict, List, Tuple

import pybithumb
import pyupbit
import pandas as pd
import numpy as np
import requests

from config.settings import (
    BITHUMB_API_KEY, BITHUMB_SECRET_KEY,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    DRY_RUN, MAX_BUDGET, TURNOVER_THRESHOLD,
    RETRY_COUNT, RETRY_DELAY
)

# --- 상수 ---
LOG_FILE = "auto_trade_v12.log"
API_URL = "https://api.bithumb.com"

# V12 전략 파라미터
N_SELECTED_COINS = 5
VOL_CAP_FILTER = 0.10
CASH_BUFFER_PERCENT = 0.02

# 제외할 스테이블코인 및 특수 코인 (래핑, 금 연동 등)
EXCLUDED_COINS = [
    # 스테이블코인
    'USDT', 'USDC', 'DAI', 'BUSD', 'UST', 'USDD', 'USDP', 'TUSD', 'GUSD', 'LUSD', 'FRAX', 'FDUSD', 'USDE',
    # 래핑/브릿지 토큰
    'WBTC', 'WETH', 'STETH', 'RETH', 'CBETH',
    # 금/원자재 연동 토큰 (정상 암호화폐 아님)
    'XAUT', 'PAXG', 'GLD', 'GOLD'
]

# --- Helper Utils for API Signing ---
def microtime(get_as_float=False):
    if get_as_float: return time.time()
    else: return '%f %d' % math.modf(time.time())

def usec_time():
    mt = microtime(False)
    mt_array = mt.split(" ")[:2]
    return mt_array[1] + mt_array[0][2:5]

class BithumbRaw:
    def __init__(self, key, secret):
        self.key = key
        self.secret = secret

    def xcoin_api_call(self, endpoint, rgParams):
        str_data = urllib.parse.urlencode(rgParams)
        nonce = usec_time()
        
        data = endpoint + chr(0) + str_data + chr(0) + nonce
        utf8_data = data.encode('utf-8')
        key = self.secret.encode('utf-8')
        
        signature = hmac.new(key, utf8_data, hashlib.sha512).hexdigest()
        signature64 = base64.b64encode(signature.encode('utf-8')).decode('utf-8')
        
        headers = {
            'Content-Type': 'application/x-www-form-urlencoded',
            'Api-Key': self.key,
            'Api-Sign': signature64,
            'Api-Nonce': nonce
        }
        
        url = API_URL + endpoint
        try:
            res = requests.post(url, headers=headers, data=rgParams, timeout=10)
            return res.json()
        except Exception as e:
            return {'status': '9999', 'message': str(e)}

class V12BithumbTrader:
    def __init__(self, is_live_trade: bool = False, is_force: bool = False, target_amount: int = 0):
        self.is_live_trade = is_live_trade
        self.is_force = is_force
        self.target_amount = target_amount
        self.dry_run = not is_live_trade # 내부 로직 호환성 유지
        
        self.bithumb = pybithumb.Bithumb(BITHUMB_API_KEY, BITHUMB_SECRET_KEY)
        self.raw_api = BithumbRaw(BITHUMB_API_KEY, BITHUMB_SECRET_KEY)
        self.usd_krw_rate = 1450.0
        
        # 환율 조회 (Upbit 참조)
        try:
             rate = pyupbit.get_current_price("KRW-USDT")
             if rate: self.usd_krw_rate = rate
        except: pass
        
        self.log_messages = []
        self.trade_history = [] 
        
        mode = "🔴 LIVE TRADE" if is_live_trade else "🔍 DRY-RUN (Analysis Only)"
        if is_force: mode += " (FORCE MODE)"
        self.log(f"[{mode}] V12 Bithumb Trader (Raw Limit Order + Yahoo USD Data)")
        
        self.usd_krw_rate = self.get_exchange_rate()
        self.log(f"USD-KRW 환율: {self.usd_krw_rate:,.0f}원")

    def get_exchange_rate(self) -> float:
        """업비트 KRW-USDT 가격을 환율로 사용"""
        try:
            url = "https://api.upbit.com/v1/ticker?markets=KRW-USDT"
            resp = requests.get(url, timeout=3).json()
            return float(resp[0]['trade_price'])
        except Exception as e:
            self.log(f"환율 조회 실패(기본값 사용): {e}")
            return 1450.0
    
    def log(self, message: str):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_line = f"[{timestamp}] {message}"
        print(log_line)
        self.log_messages.append(log_line)
        with open(LOG_FILE, 'a') as f:
            f.write(log_line + "\n")
            
    def send_telegram(self, message: str):
        if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
        try:
            url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
            requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=10)
        except Exception as e:
            self.log(f"텔레그램 전송 실패: {e}")

    # --- 유니버스 구성 (CoinGecko) ---
    # --- 유동성 필터 ---
    MIN_VOLUME_24H_KRW = 1_000_000_000  # 최소 24시간 거래대금 10억원
    
    def get_bithumb_24h_volume(self, ticker: str) -> float:
        """빗썸 24시간 거래대금(KRW) 조회"""
        try:
            url = f"https://api.bithumb.com/public/ticker/{ticker}_KRW"
            resp = requests.get(url, timeout=5)
            data = resp.json()
            if data.get('status') == '0000':
                # acc_trade_value_24H: 24시간 거래금액 (KRW)
                return float(data['data'].get('acc_trade_value_24H', 0))
        except: pass
        return 0

    def get_coingecko_top100(self) -> List[str]:
        self.log("CoinGecko Global Top 100 조회 중...")
        UNIVERSE_CACHE_FILE = "./universe_bithumb_v12_cache.json"
        
        cg_data = []
        max_retries = 3
        
        for attempt in range(max_retries):
            try:
                self.log(f"  - Fetching Top 100 from CoinGecko (Attempt {attempt+1}/{max_retries})...")
                url = "https://api.coingecko.com/api/v3/coins/markets"
                params = {
                    'vs_currency': 'usd',
                    'order': 'market_cap_desc',
                    'per_page': 100,
                    'page': 1,
                    'sparkline': 'false'
                }
                resp = requests.get(url, params=params, timeout=10)
                
                if resp.status_code == 200:
                    cg_data = resp.json()
                    # Cache Save
                    with open(UNIVERSE_CACHE_FILE, 'w') as f:
                        json.dump(cg_data, f)
                    self.log(f"    ✅ Got {len(cg_data)} coins (Cached)")
                    break
                elif resp.status_code == 429:
                    self.log("    ⚠️ Rate Limit (429). Waiting 15s...")
                    time.sleep(15)
                else:
                    self.log(f"    ⚠️ API Error: {resp.status_code}")
                    time.sleep(2)
            except Exception as e:
                self.log(f"    ⚠️ Connection Error: {e}")
                time.sleep(2)
        
        # Fallback: Load from Cache
        if not cg_data:
            if os.path.exists(UNIVERSE_CACHE_FILE):
                self.log("  ⚠️ Loading Universe from Local Cache (Fallback)...")
                try:
                    with open(UNIVERSE_CACHE_FILE, 'r') as f:
                        cg_data = json.load(f)
                except: pass

        if not cg_data:
            self.log("  ❌ CoinGecko Error => Fully Failed. Using Hardcoded Fallback")
            return ['BTC', 'ETH', 'XRP', 'SOL', 'BNB', 'DOGE', 'ADA', 'TRX', 'AVAX', 'LINK']
        global_symbols = [item['symbol'].upper() for item in cg_data]
        
        bithumb_tickers = pybithumb.get_tickers()
        
        universe = []
        low_volume_skipped = []
        
        for sym in global_symbols:
            if sym in EXCLUDED_COINS: continue
            if sym not in bithumb_tickers: continue
            
            # 유동성 필터: 24시간 거래대금 체크
            vol_24h = self.get_bithumb_24h_volume(sym)
            if vol_24h < self.MIN_VOLUME_24H_KRW:
                low_volume_skipped.append(f"{sym}({vol_24h/1e8:.1f}억)")
                continue
                
            universe.append(sym)
        
        if low_volume_skipped:
            self.log(f"⚠️ 유동성 부족 제외: {', '.join(low_volume_skipped[:5])}...")
                
        self.log(f"✅ 유니버스 선정: Global Top 100 ∩ 빗썸 ∩ 유동성 = {len(universe)}개")
        return universe


    # --- 잔고 관련 (Direct Check) ---
    def get_krw_balance(self) -> float:
        try:
            balance = self.bithumb.get_balance("BTC")
            if balance:
                total_krw = float(balance[2])
                in_use_krw = float(balance[3])
                return total_krw - in_use_krw
            return 0
        except Exception as e:
            self.log(f"잔고 조회 오류: {e}")
            return 0
    
    # --- 잔고 관련 (All Assets) ---
    def get_all_balances(self) -> Dict[str, float]:
        """빗썸 전체 잔고 조회 (API: /info/balance currency=ALL)"""
        try:
            res = self.raw_api.xcoin_api_call("/info/balance", {"currency": "ALL"})
            if res['status'] != '0000': return {}
            
            data = res['data']
            holdings = {}
            for k, v in data.items():
                if k.startswith("total_") and k != "total_krw":
                    ticker = k.replace("total_", "").upper()
                    qty = float(v)
                    if qty > 0: holdings[ticker] = qty
            return holdings
        except: return {}

    def get_current_portfolio(self, universe: List[str]) -> Dict[str, float]:
        portfolio_value = {}
        total_value = 0
        
        # 1. 전체 보유 코인 조회
        my_holdings = self.get_all_balances()
        
        # 2. 유니버스 + 보유분 합집합
        check_list = set(universe) | set(my_holdings.keys())
        
        for ticker in check_list:
            try:
                # 잔고는 이미 조회했거나, 없으면 다시 조회 (안전장치)
                qty = my_holdings.get(ticker, 0)
                if qty == 0:
                     # 혹시 모르니 개별 조회 시도 (유니버스에 있는데 get_all_balances에 없을 수도?)
                     bal = self.bithumb.get_balance(ticker)
                     if bal: qty = float(bal[0])
                
                if qty > 0:
                    price = pybithumb.get_current_price(ticker)
                    if price:
                        val = qty * price
                        if val > 1000: 
                            portfolio_value[ticker] = val
                            total_value += val
            except: pass
                
        krw = self.get_krw_balance()
        total_asset = total_value + krw
        self.log(f"💰 총 자산(KRW+Coin): {total_asset:,.0f}원 (보유코인: {len(portfolio_value)}개)")
        
        if total_asset <= 0: return {}
        portfolio_weight = {t: v / total_asset for t, v in portfolio_value.items()}
        portfolio_weight['Cash'] = krw / total_asset # 명시적 추가
        
        return portfolio_weight, total_asset

    # --- V12 지표 계산 (Yahoo Finance USD Data) ---
    
    def get_yahoo_ohlcv(self, ticker: str, days: int = 365) -> pd.DataFrame:
        try:
            yahoo_ticker = f"{ticker}-USD"
            
            period2 = int(datetime.now(timezone.utc).timestamp())
            period1 = period2 - (86400 * days)
            
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}"
            params = {
                "period1": period1,
                "period2": period2,
                "interval": "1d",
                "events": "history",
                "includeAdjustedClose": "true"
            }
            headers = {"User-Agent": "Mozilla/5.0"}
            
            resp = requests.get(url, params=params, headers=headers, timeout=5)
            data = resp.json()
            
            if 'chart' not in data or 'result' not in data['chart'] or not data['chart']['result']:
                return pd.DataFrame()
                
            timestamp = data['chart']['result'][0]['timestamp']
            quote = data['chart']['result'][0]['indicators']['quote'][0]
            closes = quote.get('close', [])
            
            # Zip and filter None values
            valid_data = [(ts, c) for ts, c in zip(timestamp, closes) if c is not None]
            
            if len(valid_data) < 30: return pd.DataFrame()
            
            df = pd.DataFrame(valid_data, columns=['Date', 'close'])
            df['Date'] = pd.to_datetime(df['Date'], unit='s')
            return df.set_index('Date')
            
        except Exception as e:
            return pd.DataFrame()

    def calc_sma(self, s, w): return s.rolling(w).mean().iloc[-1] if len(s)>=w else np.nan
    def calc_return(self, s, d): return s.iloc[-1]/s.iloc[-1-d]-1 if len(s)>d and s.iloc[-1-d]!=0 else 0
    def calc_sharpe(self, s, d):
        if len(s)<d+1: return 0
        ret = s.pct_change().iloc[-d:]
        return (ret.mean()/ret.std())*np.sqrt(252) if ret.std()!=0 else 0
    def calc_volatility(self, s, d=90):
        if len(s)<d+1: return np.inf
        ret = s.pct_change().iloc[-d:].dropna()
        return ret.std() if not ret.empty else np.inf

    def get_target_portfolio(self, universe: List[str]) -> Tuple[Dict[str, float], bool]:
        self.log("V12 전략 분석 시작 (Source: Yahoo Finance USD)...")
        
        # 1. BTC Risk Check (Yahoo USD)
        btc_df = self.get_yahoo_ohlcv('BTC', 100)
        
        if btc_df.empty:
            self.log("⚠️ BTC 데이터 조회 실패 (Yahoo) -> 안전 모드 발동 (Risk-Off 처리)")
            return {}, True
            
        # BTC 기준 Target Date
        tgt_dt = btc_df.index[-1].date() if hasattr(btc_df.index[-1], 'date') else btc_df.index[-1]
            
        btc_price = btc_df['close'].iloc[-1]
        btc_sma50 = self.calc_sma(btc_df['close'], 50)
        
        self.log(f"BTC(USD): ${btc_price:,.2f} vs SMA50: ${btc_sma50:,.2f} (Date: {tgt_dt})")
        
        if btc_price <= btc_sma50:
            self.log("🚨 Risk-Off: BTC < SMA50")
            return {}, True
            
        self.log("🟢 Risk-On: BTC > SMA50. 알트코인 분석 시작...")
        
        # 2. Universe Analysis (Yahoo USD)
        healthy_coins = []
        
        for idx, ticker in enumerate(universe):
            if ticker == 'BTC': continue
            time.sleep(0.1) 
            
            df = self.get_yahoo_ohlcv(ticker, 365)
            if df.empty or len(df) < 35: continue
            
            # [Strict Date Check]
            last_dt = df.index[-1].date() if hasattr(df.index[-1], 'date') else df.index[-1]
            diff_days = (tgt_dt - last_dt).days
            if diff_days != 0:
                # self.log(f"  ❌ {ticker}: 데이터 불일치 ({last_dt} vs {tgt_dt}) -> 제외")
                continue
            
            # [Quality Check V12.2] Yahoo vs Bithumb Price Check
            try:
                # Bithumb 종가 조회 (pybithumb는 KRW 마켓 기준 티커만 전달)
                # pybithumb.get_ohlcv("BTC")
                b_ohlcv = pybithumb.get_ohlcv(ticker)
                
                if b_ohlcv is not None and not b_ohlcv.empty:
                    b_close_krw = b_ohlcv['close'].iloc[-1]
                    y_last_usd = df['close'].iloc[-1]
                    y_last_krw = y_last_usd * self.usd_krw_rate
                    
                    if b_close_krw > 0 and y_last_krw > 0:
                        diff_pct = abs(y_last_krw - b_close_krw) / b_close_krw
                        if diff_pct > 0.10: # 10%
                            self.log(f"  ⚠️ {ticker}: 가격 불일치 (Yahoo {y_last_krw:,.0f}원 vs Bithumb {b_close_krw:,.0f}원, diff={diff_pct:.0%}) -> 제외")
                            continue
            except Exception as e:
                pass

            close = df['close']
            sma30 = self.calc_sma(close, 30)
            mom21 = self.calc_return(close, 21)
            vol90 = self.calc_volatility(close, 90)
            
            current_price = close.iloc[-1]
            
            if (current_price > sma30) and (mom21 > 0) and (vol90 <= VOL_CAP_FILTER):
                score = self.calc_sharpe(close, 126) + self.calc_sharpe(close, 252)
                healthy_coins.append({
                    'ticker': ticker, 
                    'score': score, 
                    'vol': vol90,
                    'debug': f"P:${current_price:.2f}/S30:${sma30:.2f}/V:{vol90:.1%}"
                })
        
        if not healthy_coins:
            self.log("건강한 코인 없음 (All filtered out)")
            return {}, False 
        
        # 3. Selection & Weighting
        healthy_coins.sort(key=lambda x: x['score'], reverse=True)
        top5 = healthy_coins[:N_SELECTED_COINS]
        
        debug_msg = ", ".join([f"{c['ticker']}({c['score']:.2f})" for c in top5])
        self.log(f"Top 5 Selected: {debug_msg}")
        
        inv_vols = {c['ticker']: 1/c['vol'] for c in top5 if c['vol']>0}
        total_inv = sum(inv_vols.values())
        
        if total_inv <= 0: return {}, False

        target_weights = {t: v/total_inv for t, v in inv_vols.items()}
        
        # [수정] Cash Buffer 2% 반영
        # 코인 비중 합 = 98% (Risk-On)
        buffered_weights = {t: w * (1.0 - CASH_BUFFER_PERCENT) for t, w in target_weights.items()}
        # 빗썸 봇은 target_weights를 매수 로직에서 순회하므로, CASH 키가 있어도 무방하거나 Skip 해야 함.
        # 기존 로직: for t, tw in target.items(): ... bithumb.get_balance(t)
        # CASH 키가 있으면 get_balance에서 에러 날 수 있으므로 예외 처리 필요하지만, 
        # 일단 로직의 일관성을 위해 반영하고, 매수 루프에서 CASH 스킵 처리 추가 필요.
        buffered_weights['Cash'] = CASH_BUFFER_PERCENT

        return buffered_weights, False

    def calculate_turnover(self, current, target) -> float:
        all_k = set(current.keys())|set(target.keys())
        return sum(abs(current.get(k,0)-target.get(k,0)) for k in all_k)/2

    # --- 실시간 호가 조회 ---
    def get_ask_price(self, ticker):
        try:
            url = f"https://api.bithumb.com/public/orderbook/{ticker}_KRW"
            res = requests.get(url, timeout=5).json()
            if res['status'] == '0000':
                return float(res['data']['asks'][0]['price'])
            return 0
        except: return 0

    # --- 매매 (Raw API Limit Order) ---
    def get_tick_size(self, price: float) -> float:
        if price < 1: return 0.0001 # 1원 미만
        if price < 10: return 0.01  # 10원 미만
        if price < 100: return 0.01 # 100원 미만 (빗썸 정책 확인 필요, 보통 0.01 or 0.1)
        if price < 1000: return 1
        if price < 5000: return 5
        if price < 10000: return 5
        if price < 50000: return 10
        if price < 100000: return 50
        if price < 500000: return 100
        if price < 1000000: return 500
        return 1000 

    def sell_coin(self, ticker, qty) -> bool:
        # 매도 최소 금액 체크 (안전하게 5000원)
        try:
            price = pybithumb.get_current_price(ticker)
            if price and qty * price < 5000:
                self.log(f"⚠️ {ticker} 매도 스킵: 금액({qty*price:,.0f}원)이 너무 작음")
                return False
        except: pass

        qty_str = f"{qty:.4f}"
        qty = float(qty_str)
        if qty <= 0.0001: return True

        for i in range(RETRY_COUNT):
            try:
                if self.dry_run:
                    self.log(f"[DRY-RUN] 매도: {ticker} {qty} (Str:{qty_str})")
                    return True
                
                result = self.bithumb.sell_market_order(ticker, qty)
                
                is_success = False
                order_id = "Unknown"
                
                if isinstance(result, str):
                    is_success = True
                    order_id = result
                elif isinstance(result, (tuple, list)) and len(result) >= 3:
                     is_success = True
                     order_id = result[2]
                elif isinstance(result, dict) and result.get('status') == '0000':
                     is_success = True
                     order_id = result.get('order_id')
                
                if is_success:
                    self.log(f"✅ 매도 성공: {ticker} {qty} (Order ID: {order_id})")
                    # 기록
                    try:
                        price = pybithumb.get_current_price(ticker)
                        val = price * qty if price else 0
                        self.trade_history.append(f"매도 {ticker}: {val:,.0f}원")
                    except: pass
                    return True
                else:
                    self.log(f"⚠️ 매도 실패: {result}")

            except Exception as e: self.log(f"매도 오류: {e}")
            time.sleep(RETRY_DELAY)
        return False

    def buy_coin(self, ticker, krw) -> bool:
        if krw < 5000: return True
        
        ask_price = self.get_ask_price(ticker)
        if ask_price <= 0:
             self.log(f"⚠️ 호가 조회 실패: {ticker}")
             return False
        
        limit_price = ask_price * 1.02
        tick_size = self.get_tick_size(limit_price)
        limit_price = math.floor(limit_price / tick_size) * tick_size
        
        qty = krw / limit_price 
        
        if ticker in ['BTC', 'ETH']:
            qty_str = f"{qty:.8f}"
        else:
            qty_str = f"{qty:.4f}"
            
        # [수정] 1원 미만 코인 정밀 호가 처리
        # 단순히 0.01 단위가 아니라, 현재 호가(ask_price) 그대로 주문하도록 변경
        # (빗썸은 호가창에 있는 가격이면 주문 가능)
        if limit_price < 1:
            price_str = f"{limit_price:.4f}" # 소수점 4자리까지 (SHIB 등)
        elif limit_price < 100:
            price_str = f"{limit_price:.2f}" # 100원 미만은 0.01 단위
        else:
            price_str = str(int(limit_price)) # 나머지는 정수
        
        for i in range(RETRY_COUNT):
            try:
                if self.dry_run:
                    self.log(f"[DRY-RUN] 매수(Marketable): {ticker} {price_str}원 * {qty_str}")
                    return True
                
                params = {
                    "order_currency": ticker,
                    "payment_currency": "KRW",
                    "units": qty_str,
                    "price": price_str,
                    "type": "bid"
                }
                
                result = self.raw_api.xcoin_api_call("/trade/place", params)
                
                if isinstance(result, dict) and result.get('status') == '0000':
                    order_id = result.get('order_id', 'Unknown')
                    self.log(f"✅ 매수 성공: {ticker} {krw:,.0f}원 (Order ID: {order_id})")
                    self.trade_history.append(f"매수 {ticker}: {krw:,.0f}원")
                    return True
                else:
                    self.log(f"⚠️ 매수 실패 ({result.get('status')}): {result.get('message')} | Params: {params}")

            except Exception as e: self.log(f"매수 오류: {e}")
            time.sleep(RETRY_DELAY)
        return False

    def cancel_all_orders(self, ticker):
        """미체결 주문 취소 (잔고 확보용)"""
        try:
            orders = self.bithumb.get_outstanding_order(ticker)
            if orders:
                self.log(f"🧹 {ticker} 미체결 주문 {len(orders)}건 취소 시도...")
                for order in orders:
                    self.bithumb.cancel_order(order)
                time.sleep(0.5)
        except Exception as e:
            # self.log(f"⚠️ 주문 취소 오류: {e}") 
            pass

    def rebalance(self):
        universe = self.get_coingecko_top100()
        curr, total_asset = self.get_current_portfolio(universe)
        
        # [추가] 리밸런싱 전 미체결 주문 취소하여 잔고 확보
        if not self.dry_run:
            for t in curr.keys():
                if t == 'Cash': continue
                self.cancel_all_orders(t)
        
        target, is_risk_off = self.get_target_portfolio(universe)
        
        turnover = self.calculate_turnover(curr, target)
        self.log(f"턴오버: {turnover:.2%}")

        # [상세 리포트 추가]
        self.log("\n============================================================")
        self.log("📋 현재 포트폴리오(빗썸) vs 목표")
        self.log("============================================================")
        all_tickers = sorted(set(curr.keys()) | set(target.keys()), key=lambda x: -target.get(x, 0))
        for t in all_tickers:
            if t == 'Cash': continue
            cw, tw = curr.get(t, 0), target.get(t, 0)
            diff = tw - cw
            mark = "✅" if abs(diff) < 0.005 else ("📈" if diff > 0 else "📉")
            self.log(f"  {mark} {t}: {cw:.1%} -> {tw:.1%} (Diff: {diff:+.1%})")
        self.log("")
        
        force_sell = False
        if is_risk_off:
            self.log("🚨 Risk-Off 상태: 턴오버 무시하고 강제 매도 진행(Target=0)")
            force_sell = True
        elif not target and len(curr) > 0:
            self.log("📉 타겟 없음(전량매도): 턴오버 무시하고 강제 청산")
            force_sell = True
            
        if not force_sell and turnover < TURNOVER_THRESHOLD and not self.is_force:
            self.log("턴오버 미달 - 스킵")
            return
            
        if self.is_force:
            self.log(f"💪 FORCE MODE: 턴오버({turnover:.2%}) 무시하고 리밸런싱 진행")
        
        if MAX_BUDGET and total_asset > MAX_BUDGET: 
            total_asset = MAX_BUDGET
            self.log(f"⚠️ 한도 적용 운용: {MAX_BUDGET:,.0f}원")
        
        # [수정] Target Amount 적용 (입력된 금액만큼만 운용, 나머지는 현금)
        investable_total = total_asset
        if self.target_amount > 0:
            investable_total = min(self.target_amount, total_asset)
            self.log(f"🎯 Target Amount 적용: {self.target_amount:,.0f}원 (실 운용액: {investable_total:,.0f}원)")
        
        # 1. Sell First (금액 기준 비교)
        for t, w in curr.items():
            if t == 'Cash': continue # Skip Cash
            
            # 현재 평가금
            bal = self.bithumb.get_balance(t)
            qty = float(bal[0]) if bal else 0
            p = pybithumb.get_current_price(t)
            current_val = qty * p if p else 0
            
            # 목표 평가금
            tw = target.get(t, 0)
            target_val = investable_total * tw
            
            self.log(f"[Check] {t}: 현재 {current_val:,.0f}원 vs 목표 {target_val:,.0f}원") 
            
            if target_val < current_val:
                sell_amt_krw = current_val - target_val
                
                sell_ratio = sell_amt_krw / current_val if current_val > 0 else 0
                sell_qty = qty * sell_ratio
                
                if tw == 0: sell_qty = qty # 전량 매도
                
                self.sell_coin(t, sell_qty)

                # [추가] 청산(Target=0)이고 실매매라면 잔고 확인 후 찌꺼기 정리
                if tw == 0 and not self.dry_run:
                    time.sleep(1)
                    rem_bal = self.bithumb.get_balance(t)
                    if rem_bal:
                        rem_qty = float(rem_bal[0])
                        p = pybithumb.get_current_price(t)
                        if p and rem_qty * p > 1000: 
                            self.log(f"  🧹 {t} 찌꺼기 정리 시도: {rem_qty}")
                            self.sell_coin(t, rem_qty)
        
        time.sleep(3)
        
        # 2. Buy Second
        # investable_total은 위에서 이미 설정됨.
        avail_krw = self.get_krw_balance()
        if MAX_BUDGET: avail_krw = min(avail_krw, MAX_BUDGET)
        
        if target:
            self.log(f"투자 가능 금액(98%): {investable_total:,.0f}원")
            
        for t, tw in target.items():
            if t == 'Cash': continue # Skip Cash
            target_amount = investable_total * tw
            
            bal = self.bithumb.get_balance(t)
            current_qty = float(bal[0]) if bal else 0
            p = pybithumb.get_current_price(t)
            current_amount = current_qty * p if p else 0

            self.log(f"[{t}] 목표: {target_amount:,.0f}원 | 현재: {current_amount:,.0f}원 | 갭: {(target_amount-current_amount):,.0f}원")

            if target_amount > current_amount:
                buy_amount = target_amount - current_amount
                
                real_avail_krw = self.get_krw_balance()
                safe_avail_krw = min(avail_krw, real_avail_krw)
                
                final_buy_amount = buy_amount * 0.995 
                if final_buy_amount < 5000:
                    self.log(f"⚠️ {t}: 주문금액({final_buy_amount:.0f})이 최소주문액(5000) 미만이라 건너뜀")
                    continue

                if final_buy_amount <= safe_avail_krw:
                    self.buy_coin(t, final_buy_amount)
                    avail_krw -= buy_amount
                    
                    # [추가] 잔고 확인 후 미달 시 재주문 (최대 2회)
                    if not self.dry_run:
                        time.sleep(2)  # 체결 대기
                        for retry in range(2):
                            bal = self.bithumb.get_balance(t)
                            cur_qty = float(bal[0]) if bal else 0
                            cur_p = pybithumb.get_current_price(t)
                            cur_val = cur_qty * cur_p if cur_p else 0
                            
                            # 목표의 99% 달성 시 성공
                            if cur_val >= target_amount * 0.99:
                                self.log(f"  ✅ {t} 매수 목표 달성 ({cur_val:,.0f}/{target_amount:,.0f})")
                                break
                            
                            # 미달 시 추가 주문
                            needed = target_amount - cur_val
                            if needed < 5000:
                                break
                            
                            # [제한] 재시도 시 추가 매수 금액을 목표의 2% 이내로 제한
                            max_retry = target_amount * 0.02
                            real_avail = self.get_krw_balance()
                            retry_amt = min(needed * 0.995, real_avail * 0.995, max_retry)
                            if retry_amt < 5000:
                                break
                            
                            self.log(f"  ⚠️ {t} 목표 미달 ({cur_val:,.0f}/{target_amount:,.0f}) -> 재매수 {retry+1}/2 (Max: {max_retry:,.0f})")
                            self.buy_coin(t, retry_amt)
                            time.sleep(2)
                else:
                    self.log(f"⚠️ 잔고 부족으로 매수 축소: {t}")
                    if safe_avail_krw > 5000:
                        final_scrape_amount = safe_avail_krw * 0.995 
                        self.buy_coin(t, final_scrape_amount)
                        avail_krw = 0

        # [수정] 텔레그램 알림 상세화
        msg = f"🤖 V12 Bithumb 리밸런싱 완료\n턴오버: {turnover:.1%}\nRisk-Off: {is_risk_off}"
        if self.is_force: msg += " (FORCE)"
        if self.target_amount > 0: msg += f"\nTarget: {self.target_amount:,.0f} KRW"
        if self.trade_history:
            msg += "\n\n[체결 내역]\n" + "\n".join(self.trade_history)
        else:
            msg += "\n\n(체결된 주문 없음)"
        self.send_telegram(msg)
        return turnover

    def run(self):
        self.log("------ 시작 (Raw API Mode) ------")
        
        MAX_LOOPS = 3 if self.is_live_trade else 1
        for i in range(MAX_LOOPS):
            if i > 0:
                self.log(f"\n⏳ 반복 수행 대기 중 ({i+1}/{MAX_LOOPS})...")
                time.sleep(10)
            
            turnover = self.rebalance()
            
            if turnover is not None and turnover < 0.02 and not self.is_force:
                self.log(f"✅ 턴오버 안정화({turnover:.2%}). 종료.")
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trade', action='store_true', help="실제 매매 수행")
    parser.add_argument('--force', action='store_true', help="턴오버 무시하고 강제 매매")
    parser.add_argument('--amount', type=int, default=0, help="목표 운용 금액 (0=전체)")
    args = parser.parse_args()
    
    # Config의 DRY_RUN이 True면 강제로 Dry Run (안전장치)
    is_live = args.trade
    if DRY_RUN: 
        is_live = False
        print("⚠️ Config 강제 DRY_RUN 설정됨")
        
    V12BithumbTrader(is_live_trade=is_live, is_force=args.force, target_amount=args.amount).run()
