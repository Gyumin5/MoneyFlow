"""
V16 Upbit Auto Trader (CoinGecko Global Universe)
==================================================
V16 코인 전략: BTC > SMA(60) + 1%hyst, Mom30+Mom90+Vol5%, 시총순 Top 5 EW + 20% Cap
- DD Exit: 60d peak -25% → cash
- Blacklist: -15% daily → 7d exclude
- Crash Breaker: BTC -10% daily → cash
- 턴오버 계산: Yahoo USD 종가 × 환율
- 헬스체크 실패 보유 코인 발견 시 강제 리밸런싱
- 실제 매매 기능 구현 (pyupbit Market Order)
"""

import time
import argparse
import json
import os
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Tuple

import pyupbit
import pandas as pd
import numpy as np
import requests

from config import (
    UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY,
    TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    TURNOVER_THRESHOLD, RETRY_COUNT, RETRY_DELAY
)

# --- 상수 ---
N_SELECTED_COINS = 5
VOL_CAP_FILTER = 0.05           # V15: 5% (was 10%)
CASH_BUFFER_PERCENT_DEFAULT = 0.02  # 최종 목표

def get_cash_buffer():
    """trade_state.json에서 cash_buffer 읽기. 없으면 기본값."""
    try:
        with open('trade_state.json', 'r') as f:
            import json
            return json.load(f).get('cash_buffer', CASH_BUFFER_PERCENT_DEFAULT)
    except Exception:
        return CASH_BUFFER_PERCENT_DEFAULT
MIN_ORDER_KRW = 5000

# V17 Canary / Protection
CANARY_SMA_PERIOD = 50           # V17: SMA(50)
CANARY_HYST = 0.015              # 1.5% hysteresis band
CRASH_THRESHOLD = -0.10          # BTC daily -10% → cash
BL_THRESHOLD = -0.15             # -15% daily → 7d exclude
BL_DAYS = 7
DD_EXIT_LOOKBACK = 60            # 60-day peak
DD_EXIT_THRESHOLD = -0.25        # -25% drawdown → sell

# Feature flag: True면 execution_plan 기반, False면 legacy
USE_EXECUTION_PLAN = True
SIGNAL_STATE_FILE = 'signal_state.json'

# 유니버스 선정 기준
MIN_TRADE_VALUE_KRW = 1_000_000_000
DAYS_TO_CHECK = 260
MIN_HISTORY_DAYS = 253

STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'TUSD', 'PAX', 'GUSD', 'FRAX', 'LUSD', 'MIM', 'USDN', 'FDUSD']

LOG_FILE = "auto_trade_v16_upbit.log"


def log(message: str):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_line = f"[{timestamp}] {message}"
    print(log_line)
    with open(LOG_FILE, 'a') as f:
        f.write(log_line + "\n")

def send_telegram(message: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID: return
    try:
        url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
        requests.post(url, data={"chat_id": TELEGRAM_CHAT_ID, "text": message}, timeout=5)
    except: pass


class V16UpbitTrader:
    def __init__(self, is_live_trade: bool = False, is_force: bool = False, target_amount: int = 0):
        self.is_live_trade = is_live_trade
        self.is_force = is_force
        self.target_amount = target_amount
        self.upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
        self.all_prices = {}
        self.usd_krw_rate = 1450.0
        self.trade_history = []  # 매매 내역 기록용 (매수/매도 성공 시 추가)
        
        mode = "🔴 LIVE TRADE" if is_live_trade else "🔍 ANALYSIS ONLY"
        if is_force: mode += " (FORCE MODE)"
        log("=" * 60)
        log(f"V16 Upbit Trader [{mode}]")
        log("전략: SMA50+1.5%hyst, Mom30+Mom90+Vol5%, 시총순 Top5 EW+20%Cap, DD/BL/Crash")
        log("=" * 60)
        
        try:
            usdt_price = pyupbit.get_current_price("KRW-USDT")
            if usdt_price: self.usd_krw_rate = usdt_price
        except: pass
        log(f"USD-KRW 환율: {self.usd_krw_rate:,.0f}")

    # --- 유니버스 ---
    def get_coingecko_top100(self) -> Tuple[List[str], dict]:
        log("")
        log("📡 CoinGecko Global Top 100 조회 중...")
        UNIVERSE_CACHE_FILE = "./universe_upbit_cache.json"
        
        cg_data = []
        max_retries = 3
        for attempt in range(max_retries):
            try:
                log(f"  - Fetching Top 100 from CoinGecko (Attempt {attempt+1}/{max_retries})...")
                url = "https://api.coingecko.com/api/v3/coins/markets"
                params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': 100, 'page': 1}
                headers = {"User-Agent": "Mozilla/5.0"}
                resp = requests.get(url, params=params, headers=headers, timeout=10)
                
                if resp.status_code == 200:
                    cg_data = resp.json()
                    # Cache Save
                    with open(UNIVERSE_CACHE_FILE, 'w') as f:
                        json.dump(cg_data, f)
                    log(f"    ✅ Got {len(cg_data)} coins (Cached)")
                    break
                elif resp.status_code == 429:
                    log("    ⚠️ Rate Limit (429). Waiting 15s...")
                    time.sleep(15)
                else:
                    log(f"    ⚠️ API Error: {resp.status_code}")
                    time.sleep(2)
            except Exception as e:
                log(f"    ⚠️ Conection Error: {e}")
                time.sleep(2)
        
        # Fallback: Load from Cache
        if not cg_data:
             if os.path.exists(UNIVERSE_CACHE_FILE):
                log("  ⚠️ Loading Universe from Local Cache (Fallback)...")
                try:
                    with open(UNIVERSE_CACHE_FILE, 'r') as f:
                        cg_data = json.load(f)
                except: pass

        if not cg_data:
            log("  ❌ CoinGecko 완전 실패 — 매매 중단 (포지션 유지)")
            return None, {}

        cg_symbol_to_id = {item['symbol'].upper(): item['id'] for item in cg_data}
        
        try:
            upbit_tickers = pyupbit.get_tickers(fiat="KRW")
            upbit_symbols = {t.split('-')[1] for t in upbit_tickers}
        except: upbit_symbols = set()
        
        universe = []
        for item in cg_data:
            symbol = item['symbol'].upper()
            if symbol in STABLECOINS: continue
            if symbol not in upbit_symbols: continue
            
            upbit_ticker = f"KRW-{symbol}"
            try:
                df = pyupbit.get_ohlcv(ticker=upbit_ticker, interval="day", count=DAYS_TO_CHECK)
                time.sleep(0.05)
                if df is None or len(df) < MIN_HISTORY_DAYS: continue
                if df['value'].iloc[-30:].mean() < MIN_TRADE_VALUE_KRW: continue
                
                universe.append(symbol)
            except: continue
            if len(universe) >= 40: break
        
        log(f"  🎯 유니버스: {len(universe)}개")
        return universe, cg_symbol_to_id

    # --- 잔고 ---
    def get_current_holdings_qty(self) -> Dict[str, float]:
        log("")
        log("💰 업비트 잔고 조회 중...")
        holdings_qty = {}
        try:
            upbit_tickers = pyupbit.get_tickers(fiat="KRW")
            upbit_listed = {t.split('-')[1] for t in upbit_tickers}
            
            balances = self.upbit.get_balances()
            if not isinstance(balances, list): balances = []
            
            for b in balances:
                if not isinstance(b, dict): continue
                ticker = b.get('currency', '')
                if ticker == 'KRW' or not ticker: continue
                if ticker not in upbit_listed: continue
                
                qty = float(b.get('balance', 0)) + float(b.get('locked', 0))
                if qty > 0: holdings_qty[ticker] = qty
        except Exception as e:
            log(f"  ❌ 잔고 조회 오류: {e}")
        return holdings_qty

    # --- Yahoo Data ---
    def get_yahoo_ohlcv(self, ticker: str, days: int = 365) -> pd.Series:
        try:
            yahoo_ticker = f"{ticker}-USD"
            
            p2 = int(datetime.now(timezone.utc).timestamp())
            p1 = p2 - (86400 * days)
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{yahoo_ticker}"
            params = {"period1": p1, "period2": p2, "interval": "1d"}
            resp = requests.get(url, params=params, headers={"User-Agent": "Mozilla/5.0"}, timeout=5)
            data = resp.json()
            
            if 'chart' not in data or not data['chart']['result']:
                return pd.Series(dtype=float)
            
            timestamp = data['chart']['result'][0]['timestamp']
            quote = data['chart']['result'][0]['indicators']['quote'][0]
            closes = quote.get('close', [])
            
            # Zip and filter None values
            valid_data = [(ts, c) for ts, c in zip(timestamp, closes) if c is not None]
            
            if len(valid_data) < 35: return pd.Series(dtype=float)
            
            df = pd.DataFrame(valid_data, columns=['Date', 'Close'])
            df['Date'] = pd.to_datetime(df['Date'], unit='s')
            return df.set_index('Date')['Close']
        except: return pd.Series(dtype=float)

    def calc_sma(self, s, w): return s.rolling(w).mean().iloc[-1] if len(s)>=w else np.nan
    def calc_ret(self, s, d): return s.iloc[-1]/s.iloc[-1-d]-1 if len(s)>d and s.iloc[-1-d]!=0 else 0
    def calc_sharpe(self, s, d):
        if len(s)<d+1: return 0
        ret = s.pct_change().iloc[-d:]
        return (ret.mean()/ret.std())*np.sqrt(252) if ret.std()!=0 else 0
    def calc_volatility(self, s, d=90):
        if len(s)<d+1: return np.inf
        ret = s.pct_change().iloc[-d:].dropna()
        return ret.std() if not ret.empty else np.inf

    def get_target_portfolio(self, universe: List[str]) -> Tuple[Dict[str, float], bool, str, List[str]]:
        log("")
        log("📊 V16 전략 분석 중...")

        # 1. BTC Check
        btc = self.get_yahoo_ohlcv('BTC', 365)
        self.all_prices['BTC'] = btc
        if len(btc) < CANARY_SMA_PERIOD:
            log("🚨 BTC 데이터 부족 — API 장애 의심, 매매 중단 (포지션 유지)")
            return None, False, "⚠️ DATA ERROR: 매매 중단", []  # None = abort signal

        # BTC 기준 Target Date
        tgt_dt = btc.index[-1].date() if hasattr(btc.index[-1], 'date') else btc.index[-1]

        # --- Crash Breaker 제거 (V17d): 과적합으로 판단. DD Exit + Blacklist + 카나리가 방어 ---
        # BTC -10% crash는 카나리/DD/Blacklist와 중복되어 효과 없음 (백테스트 확인)

        # --- Canary: BTC > SMA(60) with 1% Hysteresis ---
        sma = self.calc_sma(btc, CANARY_SMA_PERIOD)
        cur = btc.iloc[-1]
        dist = cur / sma - 1

        # Load previous canary state for hysteresis dead zone
        prev_coin_risk_on = None
        try:
            with open('trade_state.json', 'r') as _sf:
                prev_coin_risk_on = json.load(_sf).get('coin_risk_on')
        except Exception:
            try:
                with open('signal_state.json', 'r') as _sf:
                    prev_coin_risk_on = json.load(_sf).get('coin_risk_on')
            except Exception:
                pass

        if dist > CANARY_HYST:
            canary_on = True
        elif dist < -CANARY_HYST:
            canary_on = False
        elif prev_coin_risk_on is not None:
            canary_on = prev_coin_risk_on  # dead zone: maintain previous state
        else:
            canary_on = cur > sma  # no state: fallback
        log(f"  BTC: ${cur:,.0f} vs SMA({CANARY_SMA_PERIOD}): ${sma:,.0f} (dist {dist:+.2%}, hyst ±{CANARY_HYST:.0%})")

        if not canary_on:
            return {}, True, f"🚨 Risk-Off: BTC <= SMA{CANARY_SMA_PERIOD} (dist {dist:+.2%})", []

        # 2. Health Check: Mom(30)>0 AND Mom(90)>0 AND Vol(90)<=5%
        healthy = []
        blacklisted = []
        for t in universe:
            time.sleep(0.05)
            c = self.get_yahoo_ohlcv(t, 365)
            if len(c) < 91: continue

            # [Strict Date Check]
            last_dt = c.index[-1].date() if hasattr(c.index[-1], 'date') else c.index[-1]
            diff_days = (tgt_dt - last_dt).days
            if diff_days != 0: continue

            # [Quality Check] Yahoo vs Upbit price validation
            try:
                upbit_ohlcv = pyupbit.get_ohlcv(f"KRW-{t}", interval="day", count=1)
                if upbit_ohlcv is not None and len(upbit_ohlcv) > 0:
                    upbit_close_krw = upbit_ohlcv['close'].iloc[-1]
                    yahoo_last_krw = c.iloc[-1] * self.usd_krw_rate
                    if upbit_close_krw > 0 and yahoo_last_krw > 0:
                        diff_pct = abs(yahoo_last_krw - upbit_close_krw) / upbit_close_krw
                        if diff_pct > 0.10:
                            log(f"  ⚠️ {t}: 가격 불일치 ({diff_pct:.0%}) -> 제외")
                            continue
            except: pass

            self.all_prices[t] = c

            # --- Blacklist: -15% daily drop in last 7d → exclude ---
            if len(c) >= BL_DAYS + 1:
                recent_rets = c.iloc[-(BL_DAYS + 1):].pct_change().dropna()
                worst_ret = recent_rets.min()
                if worst_ret <= BL_THRESHOLD:
                    blacklisted.append(t)
                    log(f"  🚫 {t}: Blacklist (worst {worst_ret:+.1%} in {BL_DAYS}d)")
                    continue

            # --- Health: Mom(30)>0 AND Mom(90)>0 AND Vol(90)<=5% ---
            mom30 = self.calc_ret(c, 30)
            mom90 = self.calc_ret(c, 90)
            vol90 = self.calc_volatility(c, 90)

            if mom30 > 0 and mom90 > 0 and vol90 <= VOL_CAP_FILTER:
                healthy.append({'ticker': t, 'vol': vol90})

        healthy_tickers = [h['ticker'] for h in healthy]
        if blacklisted:
            log(f"  🚫 Blacklisted: {blacklisted}")
        if not healthy: return {}, False, "건강한 코인 없음", []

        # 3. Selection: 시총순 Top 5, Equal Weight (universe order = market cap)
        top5 = healthy[:N_SELECTED_COINS]
        log(f"  🎯 Top {len(top5)}: {[h['ticker'] for h in top5]}")

        # --- DD Exit: 60d peak -25% → sell ---
        dd_exits = []
        final_picks = []
        for h in top5:
            t = h['ticker']
            c = self.all_prices[t]
            if len(c) >= DD_EXIT_LOOKBACK:
                peak = c.iloc[-DD_EXIT_LOOKBACK:].max()
                dd = c.iloc[-1] / peak - 1 if peak > 0 else 0
                if dd <= DD_EXIT_THRESHOLD:
                    dd_exits.append(t)
                    log(f"  📉 {t}: DD Exit ({dd:+.1%} from {DD_EXIT_LOOKBACK}d peak)")
                    continue
            final_picks.append(t)

        if dd_exits:
            log(f"  📉 DD Exits: {dd_exits}")

        if not final_picks:
            return {}, False, "모든 코인 DD Exit", healthy_tickers

        # Equal Weight with 20% Cap (DD-exited weight → cash)
        COIN_WEIGHT_CAP = 0.20
        ew = min(1.0 / len(top5), COIN_WEIGHT_CAP)
        w = {t: ew for t in final_picks}
        cash_from_dd = ew * len(dd_exits)

        # Cash Buffer 2% + DD exit cash
        cash_buf = get_cash_buffer()
        buffered_w = {t: val * (1.0 - cash_buf) for t, val in w.items()}
        buffered_w['Cash'] = cash_buf + cash_from_dd

        return buffered_w, False, f"✅ Risk-On ({len(final_picks)}종 EW)", healthy_tickers

    # --- 실제 매매 ---
    def order(self, ticker: str, side: str, amount: float) -> bool:
        """
        side: 'buy' or 'sell'
        amount: buy는 krw 금액, sell은 코인 수량
        """
        try:
            # 1. 예상 주문 금액(KRW) 계산
            price = pyupbit.get_current_price(f"KRW-{ticker}") or 0
            if side == 'buy':
                est_val = amount
            else:
                est_val = amount * price

            # 2. 최소 주문 금액 체크 (5,000원)
            if est_val < 5000:
                log(f"  ⚠️ {side.upper()} {ticker}: 주문 금액({est_val:,.0f}원)이 최소 주문액(5,000원) 미만 -> 스킵")
                return False

            # 3. Dry Run 처리
            if not self.is_live_trade:
                log(f"  [DRY-RUN] {side.upper()} {ticker}: {amount:,.4f} (약 {est_val:,.0f}원)")
                # Dry Run은 기록하지 않음
                return True
            
            # 4. 실매매 실행
            tk = f"KRW-{ticker}"
            if side == 'buy':
                res = self.upbit.buy_market_order(tk, amount)
            else:
                res = self.upbit.sell_market_order(tk, amount)
            
            if res and 'uuid' in res:
                log(f"  ✅ {side.upper()} 주문 완료: {res['uuid']}")
                # 텔레그램용 내역 저장
                action = "매수" if side == 'buy' else "매도"
                self.trade_history.append(f"{action} {ticker}: {est_val:,.0f}원")
                return True
            else:
                log(f"  ⚠️ 주문 실패: {res}")
                return False
        except Exception as e:
            log(f"  ❌ 주문 에러: {e}")
            return False

    def ensure_sell(self, ticker: str, qty: float, is_clearance: bool = False):
        """매도 후 잔고 확인하여 확실히 청산 (is_clearance=True일 때만 잔고 0 확인)"""
        log(f"📉 매도 시도 {ticker}: {qty:.4f}")
        if not self.order(ticker, 'sell', qty): return
        
        if not self.is_live_trade: return
        
        # 부분 매도라면 여기서 종료 (남은 잔고는 정상)
        if not is_clearance:
            return

        time.sleep(2) # 체결 대기
        
        # 전량 청산 모드일 때만 잔고 확인 및 재시도 (최대 2회)
        for i in range(2):
            try:
                bal = self.upbit.get_balance(f"KRW-{ticker}")
                cur_price = pyupbit.get_current_price(f"KRW-{ticker}")
                
                # 잔고 가치가 5000원 미만이면 성공으로 간주 (먼지)
                if bal is None or (bal * cur_price) < 5000:
                    log(f"  ✅ {ticker} 전량 매도 확인 완료 (잔고: {bal})")
                    return
                
                log(f"  ⚠️ {ticker} 청산 실패 잔량 ({bal}) 발견 -> 재매도 시도 {i+1}/2")
                self.order(ticker, 'sell', bal)
                time.sleep(2)
            except Exception as e:
                log(f"  ❌ 재매도 중 에러: {e}")

    def ensure_buy(self, ticker: str, target_val: float):
        """매수 후 평가금액 확인하여 목표치 미달 시 재매수"""
        # 1차 시도
        krw_bal = self.upbit.get_balance("KRW") or 0
        investable = krw_bal * 0.99
        
        cur_price = pyupbit.get_current_price(f"KRW-{ticker}") or 0
        cur_bal = self.upbit.get_balance(f"KRW-{ticker}") or 0
        cur_val = cur_bal * cur_price
        
        needed = target_val - cur_val
        if needed < 5000:
            log(f"  ℹ️ {ticker} 매수 스킵: 부족분({needed:,.0f}원)이 최소주문액 미만")
            return
        
        buy_amt = min(needed, investable)
        if buy_amt < 5000:
            log(f"  ℹ️ {ticker} 매수 스킵: 가용현금 부족 또는 금액({buy_amt:,.0f}원) 미달")
            return

        log(f"📈 매수 시도 {ticker}: {buy_amt:,.0f}원")
        if not self.order(ticker, 'buy', buy_amt): return 
        
        if not self.is_live_trade: return
        
        time.sleep(2)
        
        # 확인 및 재시도 (최대 2회)
        for i in range(2):
            try:
                # 현재 상태 재조회
                krw_bal = self.upbit.get_balance("KRW") or 0
                investable = krw_bal * 0.99
                
                cur_price = pyupbit.get_current_price(f"KRW-{ticker}")
                cur_bal = self.upbit.get_balance(f"KRW-{ticker}") or 0
                cur_val_now = cur_bal * cur_price
                
                # 목표의 95% 이상 달성했으면 성공 (슬리피지 고려)
                # 또는 예산 부족하면 종료
                if cur_val_now >= target_val * 0.99:
                    log(f"  ✅ {ticker} 매수 목표 달성 ({cur_val_now:,.0f}/{target_val:,.0f})")
                    return
                
                needed_now = target_val - cur_val_now
                if needed_now < 5000: return
                
                # [제한] 재시도 시 추가 매수 금액을 목표의 2% 이내로 제한
                max_retry = target_val * 0.02
                retry_amt = min(needed_now, investable, max_retry)
                if retry_amt < 5000:
                    log(f"  ⚠️ 현금 부족 또는 2% 한도 도달로 매수 종료 ({investable:,.0f}원)")
                    return
                
                log(f"  ⚠️ {ticker} 목표 미달 ({cur_val_now:,.0f}/{target_val:,.0f}) -> 재매수 {i+1}/2: {retry_amt:,.0f}원 (Max: {max_retry:,.0f})")
                self.order(ticker, 'buy', retry_amt)
                time.sleep(2)
                
            except Exception as e:
                log(f"  ❌ 재매수 중 에러: {e}")

    # ═══ 분할 매매 엔진 ═══

    def calc_safe_amount(self, ticker, side='buy', max_slip=0.003):
        """슬리피지 0.1% 이내로 매매 가능한 최대 금액 (호가 기반)."""
        try:
            ob = pyupbit.get_orderbook(f"KRW-{ticker}")
            if not ob:
                return MIN_ORDER_KRW
            units = ob["orderbook_units"]
            if side == 'buy':
                best = units[0]["ask_price"]
                limit_price = best * (1 + max_slip)
                safe = sum(u["ask_size"] * u["ask_price"]
                           for u in units if u["ask_price"] <= limit_price)
            else:
                best = units[0]["bid_price"]
                limit_price = best * (1 - max_slip)
                safe = sum(u["bid_size"] * u["bid_price"]
                           for u in units if u["bid_price"] >= limit_price)
            return max(safe, MIN_ORDER_KRW)
        except Exception:
            return 0  # 호가 조회 실패 → 0 반환 → 주문 스킵

    def split_buy(self, ticker, target_krw, timeout_sec=180):
        """분할 매수: 슬리피지 0.3% 이내씩, 7초 간격, 타임아웃."""
        # 시작 전 미체결 주문 정리
        self.cancel_all_orders(ticker)
        time.sleep(1)
        # 시작 전 잔고 확인
        init_bal = self.upbit.get_balance(f"KRW-{ticker}") or 0
        init_price = pyupbit.get_current_price(f"KRW-{ticker}") or 0
        init_val = init_bal * init_price
        est_filled = 0  # dry-run용 추정 체결액

        start = time.time()
        attempt = 0
        while True:
            if time.time() - start > timeout_sec:
                log(f"  ⏰ {ticker} 분할매수 타임아웃 ({timeout_sec}초)")
                break
            # 실체결 금액 확인 (실거래) 또는 추정 (dry-run)
            if self.is_live_trade:
                cur_bal = self.upbit.get_balance(f"KRW-{ticker}") or 0
                cur_price = pyupbit.get_current_price(f"KRW-{ticker}") or 0
                filled = cur_bal * cur_price - init_val
            else:
                filled = est_filled
            remaining = target_krw - filled
            if remaining < 50000:
                break
            krw_bal = self.upbit.get_balance("KRW") or 0
            safe = self.calc_safe_amount(ticker, 'buy')
            chunk = min(remaining, safe, krw_bal * 0.99)
            if chunk < MIN_ORDER_KRW:
                log(f"  💤 {ticker} 매수 스킵: chunk {chunk:,.0f} < 최소주문")
                break
            attempt += 1
            log(f"  📈 [{attempt}] {ticker} 분할매수 {chunk:,.0f}원 (잔여 {remaining:,.0f})")
            success = self.order(ticker, 'buy', chunk)
            if not success:
                for retry in range(3):
                    log(f"  🔄 재시도 {retry+1}/3 (30초 대기)")
                    time.sleep(30)
                    success = self.order(ticker, 'buy', chunk)
                    if success:
                        break
                if not success:
                    log(f"  ❌ {ticker} 매수 실패")
                    break
            if not self.is_live_trade:
                est_filled += chunk  # dry-run: 추정 누적
            time.sleep(7)

        # 최종 체결액
        if self.is_live_trade:
            final_bal = self.upbit.get_balance(f"KRW-{ticker}") or 0
            final_price = pyupbit.get_current_price(f"KRW-{ticker}") or 0
            total_filled = final_bal * final_price - init_val
        else:
            total_filled = est_filled
        log(f"  💰 {ticker} 분할매수 완료: {total_filled:,.0f} / {target_krw:,.0f}")
        return max(total_filled, 0)

    def split_sell(self, ticker, target_krw, timeout_sec=180):
        """분할 매도: 슬리피지 0.3% 이내씩, 7초 간격."""
        # 시작 전 미체결 주문 정리
        self.cancel_all_orders(ticker)
        time.sleep(1)
        # 시작 전 잔고 확인
        init_bal = self.upbit.get_balance(f"KRW-{ticker}") or 0
        init_price = pyupbit.get_current_price(f"KRW-{ticker}") or 0
        init_val = init_bal * init_price
        est_sold = 0  # dry-run용

        start = time.time()
        attempt = 0
        while True:
            if time.time() - start > timeout_sec:
                log(f"  ⏰ {ticker} 분할매도 타임아웃 ({timeout_sec}초)")
                break
            # 실체결 확인
            if self.is_live_trade:
                cur_bal = self.upbit.get_balance(f"KRW-{ticker}") or 0
                cur_price = pyupbit.get_current_price(f"KRW-{ticker}") or 0
                sold = init_val - cur_bal * cur_price
            else:
                sold = est_sold
            remaining = target_krw - sold
            if remaining < 50000:
                break
            safe = self.calc_safe_amount(ticker, 'sell')
            chunk_krw = min(remaining, safe)
            if cur_price <= 0 or chunk_krw < MIN_ORDER_KRW:
                break
            chunk_qty = chunk_krw / cur_price
            chunk_qty = min(chunk_qty, cur_bal)
            if chunk_qty * cur_price < MIN_ORDER_KRW:
                break
            attempt += 1
            log(f"  📉 [{attempt}] {ticker} 분할매도 {chunk_qty:.4f}개 (~{chunk_qty*cur_price:,.0f}원)")
            if not self.order(ticker, 'sell', chunk_qty):
                for retry in range(3):
                    log(f"  🔄 재시도 {retry+1}/3 (30초 대기)")
                    time.sleep(30)
                    if self.order(ticker, 'sell', chunk_qty):
                        break
                else:
                    log(f"  ❌ {ticker} 매도 실패")
                    break
            if not self.is_live_trade:
                est_sold += chunk_qty * cur_price  # dry-run: 추정 누적
            time.sleep(7)

        # 최종 체결액
        if self.is_live_trade:
            final_bal = self.upbit.get_balance(f"KRW-{ticker}") or 0
            final_price = pyupbit.get_current_price(f"KRW-{ticker}") or 0
            total_sold = init_val - final_bal * final_price
        else:
            total_sold = est_sold
        log(f"  💰 {ticker} 분할매도 완료: {total_sold:,.0f} / {target_krw:,.0f}")
        return max(total_sold, 0)

    def emergency_sell_all(self, tickers):
        """긴급 전량 매도: 코인별 빠른 분할 (1~2초 간격, 2~3회)."""
        log("🚨 긴급 전량 매도 시작")
        # 미체결 주문 먼저 취소
        for t in tickers:
            self.cancel_all_orders(t)
        time.sleep(1)

        for t in tickers:
            bal = self.upbit.get_balance(f"KRW-{t}") or 0
            price = pyupbit.get_current_price(f"KRW-{t}") or 0
            if bal * price < MIN_ORDER_KRW:
                continue
            log(f"  🚨 {t} 긴급매도 ({bal:.4f}개, ~{bal*price:,.0f}원)")
            # 빠른 분할: 호가 기반, 1초 간격
            for i in range(5):  # 최대 5회
                cur_bal = self.upbit.get_balance(f"KRW-{t}") or 0
                cur_price = pyupbit.get_current_price(f"KRW-{t}") or 0
                if cur_bal * cur_price < MIN_ORDER_KRW:
                    break
                safe_krw = self.calc_safe_amount(t, 'sell')
                if safe_krw <= 0:
                    safe_krw = cur_bal * cur_price  # fallback: 전량
                sell_qty = min(cur_bal, safe_krw / cur_price if cur_price > 0 else cur_bal)
                if self.order(t, 'sell', sell_qty):
                    log(f"    ✅ {t} 긴급매도 {i+1}회 성공")
                else:
                    log(f"    ⚠️ {t} 긴급매도 {i+1}회 실패")
                time.sleep(1)
            # 잔량 확인 + 정리
            final_bal = self.upbit.get_balance(f"KRW-{t}") or 0
            final_price = pyupbit.get_current_price(f"KRW-{t}") or 0
            if final_bal * final_price >= MIN_ORDER_KRW:
                self.order(t, 'sell', final_bal)
        log("🚨 긴급 매도 완료")

    def _save_trade_state(self, state, filepath):
        """원자적 state 저장 (tmp + replace)."""
        try:
            tmp = filepath + '.tmp'
            with open(tmp, 'w') as f:
                json.dump(state, f, indent=2)
            os.replace(tmp, filepath)
        except Exception as e:
            log(f"⚠️ State 저장 실패: {e}")

    def _refresh_monitor_cache(self, trade_state):
        """Monitor용 캐시 갱신: USD 기준 BTC 데이터 + coin_peaks 정리.
        HOLD와 매매 완료 양쪽에서 호출."""
        # 레거시 KRW 캐시 키 정리
        trade_state.pop('btc_sma60', None)
        trade_state.pop('btc_prev_close', None)
        try:
            # BTC SMA50 (USD) — Yahoo 데이터 기준
            btc_data = self.all_prices.get('BTC', pd.Series())
            if len(btc_data) >= CANARY_SMA_PERIOD + 1:
                sma_usd = float(btc_data.rolling(CANARY_SMA_PERIOD).mean().iloc[-1])
                trade_state['btc_sma60_usd'] = sma_usd
                # BTC 전일 종가 (USD) — 완료된 봉 기준 (iloc[-2])
                # iloc[-1]은 당일 진행 봉일 수 있으므로 iloc[-2] 사용
                trade_state['btc_prev_close_usd'] = float(btc_data.iloc[-2])

            # coin_peaks 제거 — DD exit이 CSV 60일 rolling peak를 직접 조회
            trade_state.pop('coin_peaks', None)

        except Exception as e:
            log(f"⚠️ 캐싱 갱신 오류: {e}")

    def cancel_all_orders(self, ticker):
        """미체결 주문 취소 (잔고 확보용)"""
        try:
            # 업비트는 KRW- 접두사 필요
            orders = self.upbit.get_order(f"KRW-{ticker}", state='wait')
            if orders and isinstance(orders, list):
                log(f"🧹 {ticker} 미체결 주문 {len(orders)}건 취소 시도...")
                for order in orders:
                    self.upbit.cancel_order(order['uuid'])
                time.sleep(0.5)
        except Exception as e:
            # log(f"⚠️ 주문 취소 오류: {e}")
            pass

    def run(self):
        # 1. 미체결 주문 취소 (봇 시작 시 잔고 확보)
        # 잔고 조회 전에 수행하여 Locked Assets을 해제
        if not self.is_live_trade:
            log("🔍 Analysis Mode: Skip canceling orders.")
        else:
            log("🧹 기존 미체결 주문 정리 중...")
            # 보유 중인 코인 + 유니버스 대상 (유니버스는 아직 모르므로 전체 잔고 기준 우선)
            # 하지만 안전하게 get_balances로 확인
            try:
                balances = self.upbit.get_balances()
                for b in balances:
                    if b['currency'] == 'KRW': continue
                    # 보유 수량 있거나 Locked 있는 경우 취소 시도
                    if float(b['balance']) + float(b['locked']) > 0:
                        self.cancel_all_orders(b['currency'])
            except: pass

        # Execution plan 기반 (새 아키텍처)
        plan = None
        if USE_EXECUTION_PLAN:
            try:
                with open(SIGNAL_STATE_FILE, 'r') as _sf:
                    _sig = json.load(_sf)
                plan = _sig.get('execution_plan', {}).get('coin', {})
            except Exception:
                pass

        if plan and plan.get('ideal_picks') is not None:
            # Plan 기반: recommend가 이미 계산한 결과 사용
            coin_risk_on = plan.get('risk_on', True)
            is_risk_off = not coin_risk_on
            if coin_risk_on and plan.get('ideal_picks'):
                target_w = dict(plan.get('ideal_weights', {}))
                # Cash 비중 보정
                invested = sum(target_w.values())
                if invested < 1.0:
                    target_w['Cash'] = 1.0 - invested
                status = f"[PLAN] Risk-On: {list(target_w.keys())}"
                healthy_list = plan.get('ideal_picks', [])
            else:
                target_w = {'Cash': 1.0}
                status = "[PLAN] Risk-Off: 전량 현금"
                healthy_list = []
            log(f"\n🎯 [PLAN] risk_on={coin_risk_on}, picks={plan.get('ideal_picks')}, anchors={plan.get('today_anchors')}")
            # 데이터 다운로드는 여전히 필요 (DD/Blacklist 모니터용)
            universe, _ = self.get_coingecko_top100()
            if universe is None:
                universe = healthy_list  # fallback
        else:
            # Legacy: 자체 신호 계산
            universe, _ = self.get_coingecko_top100()
            if universe is None:
                log("🚨 CoinGecko 완전 실패 — 포지션 유지, 매매 없음")
                send_telegram("⚠️ V16: CoinGecko API 실패, 매매 중단")
                return
            target_w, is_risk_off, status, healthy_list = self.get_target_portfolio(universe)

            # API 장애 시 매매 중단 (포지션 유지)
            if target_w is None:
                log(f"\n🚨 {status} — 포지션 유지, 매매 없음")
                send_telegram(f"⚠️ V16 매매 중단: {status}")
                return

        log(f"\n🎯 상태: {status}")
        
        MAX_LOOPS = 3 if self.is_live_trade else 1
        for loop_i in range(MAX_LOOPS):
            if loop_i > 0:
                log(f"\n⏳ 반복 수행 대기 중 ({loop_i+1}/{MAX_LOOPS})...")
                time.sleep(10)

            turnover = self.run_iteration(universe, target_w, is_risk_off, healthy_list)

            if turnover is not None and turnover < 0.02 and not self.is_force:
                log(f"✅ 턴오버 안정화({turnover:.2%}). 종료.")
                break

    def run_iteration(self, universe, target_w, is_risk_off, healthy_list):
        holdings_qty = self.get_current_holdings_qty()

        # [변경] 가치 평가: 업비트 현재가 기준 (Real-time KRW)
        cur_assets_val = {}
        tickers = list(holdings_qty.keys())
        if tickers:
            krw_tickers = [f"KRW-{t}" for t in tickers]
            current_prices = pyupbit.get_current_price(krw_tickers)
            
            if not isinstance(current_prices, dict):
                current_prices = {krw_tickers[0]: current_prices}
                
            for t, q in holdings_qty.items():
                price = current_prices.get(f"KRW-{t}", 0)
                if price is not None:
                    val = price * q
                    if val > 1000: cur_assets_val[t] = val
        
        krw_bal = self.upbit.get_balance("KRW") or 0
        total_val = sum(cur_assets_val.values()) + krw_bal
        log(f"총 자산(현재가): {total_val:,.0f}원 (KRW: {krw_bal:,.0f}원)")
        if total_val <= 0: return
        
        curr_w = {t: v/total_val for t, v in cur_assets_val.items()}
        curr_w['Cash'] = krw_bal / total_val if total_val > 0 else 0
        
        # 턴오버 계산
        turnover = sum(abs(curr_w.get(k,0) - target_w.get(k,0)) for k in set(curr_w)|set(target_w)) / 2
        log(f"🔄 턴오버: {turnover:.1%} (기준: {TURNOVER_THRESHOLD:.0%})")
        
        # [상세 리포트]
        log("\n============================================================")
        log("📋 현재 포트폴리오(업비트) vs 목표")
        log("============================================================")
        for t in sorted(set(curr_w)|set(target_w), key=lambda x: -target_w.get(x,0)):
            cw, tw = curr_w.get(t,0), target_w.get(t,0)
            diff = tw - cw
            mark = "✅" if abs(diff) < 0.005 else ("📈" if diff > 0 else "📉")
            log(f"  {mark} {t}: {cw:.1%} -> {tw:.1%} (Diff: {diff:+.1%})")
            
        # Cash 비중 로그 별도 표시 (선택사항, 이미 위 루프에서 출력될 수 있음)
        # 하지만 target_w에 Cash가 잇으므로 위 루프에서 출력됨. 
        # 단, ensure_buy에서는 막아야 함.
            
        if holdings_qty.get('POL', 0) > 0 and 'POL' not in healthy_list:
             log(f"  🚨 POL: 헬스체크 탈락 (보유중) -> 전량 매도 필요")
        
        # 헬스체크 실패 코인 검사 (보유 + 트랜치 목표 모두 포함)
        bad_coins = []
        healthy_set = set(healthy_list)
        # 트랜치 목표 코인도 검사 범위에 포함
        all_tranche_coins_for_health = set(curr_w.keys())
        try:
            with open('trade_state.json', 'r') as _hf:
                _hstate = json.load(_hf)
            for _htr in _hstate.get('tranches', {}).values():
                all_tranche_coins_for_health.update(_htr.get('picks', []))
        except Exception:
            pass
        for t in all_tranche_coins_for_health:
            if t == 'Cash':
                continue
            if t not in healthy_set and t not in target_w:
                bad_coins.append(t)
        
        # ═══ V16 3트랜치 리밸런싱 ═══
        ANCHOR_DAYS = [1, 11, 21]  # 3트랜치 (10일 간격 균등)
        TRADE_STATE_FILE = 'trade_state.json'
        KST = timezone(timedelta(hours=9))
        now_kst = datetime.now(KST)
        today = now_kst.day
        current_month = now_kst.strftime('%Y-%m')

        # Load trade state
        trade_state = {}
        try:
            with open(TRADE_STATE_FILE, 'r') as _sf:
                trade_state = json.load(_sf)
        except Exception:
            pass

        prev_risk_on = trade_state.get('coin_risk_on', None)
        is_risk_on_now = not is_risk_off
        signal_flipped = (prev_risk_on is not None and prev_risk_on != is_risk_on_now)
        flip_on = signal_flipped and is_risk_on_now

        # Compute current signal (picks + weights) for rebalancing
        current_signal_w = target_w  # from get_target_portfolio()

        # Initialize tranches if missing — 전 트랜치를 현재 신호로 초기화
        is_first_run = 'tranches' not in trade_state
        if is_first_run:
            log("🆕 trade_state 초기화: 전 트랜치를 현재 신호로 설정")
            trade_state['tranches'] = {}
            for a in ANCHOR_DAYS:
                if is_risk_on_now:
                    init_picks = [t for t in current_signal_w if t != 'Cash']
                    init_weights = {t: w for t, w in current_signal_w.items() if t != 'Cash'}
                else:
                    init_picks = []
                    init_weights = {}
                trade_state['tranches'][str(a)] = {
                    'last_anchor_month': current_month, 'picks': init_picks, 'weights': init_weights
                }

        # ── 1. Crash: 전 트랜치 현금화 ──
        # (이미 get_target_portfolio에서 crash 시 빈 target_w 반환)

        # ── 2. 카나리아 플립 → 전 트랜치 갱신 ──
        if signal_flipped:
            log(f"🔄 카나리아 플립 ({'ON' if is_risk_on_now else 'OFF'}) → 전 트랜치 갱신")
            for a_str in trade_state['tranches']:
                tr = trade_state['tranches'][a_str]
                if is_risk_on_now:
                    # Remove Cash key for storage
                    tr['picks'] = [t for t in current_signal_w if t != 'Cash']
                    tr['weights'] = {t: w for t, w in current_signal_w.items() if t != 'Cash'}
                else:
                    tr['picks'] = []
                    tr['weights'] = {}
            if flip_on:
                trade_state['last_flip_date'] = datetime.now().strftime('%Y-%m-%d')
                trade_state['pfd_done'] = False

        # ── 3. DD Exit → 전 트랜치에서 해당 코인 제거 ──
        dd_removed = set()
        if is_risk_on_now:
            all_tranche_coins = set()
            for tr in trade_state['tranches'].values():
                all_tranche_coins.update(tr.get('picks', []))
            for t in all_tranche_coins:
                if t in self.all_prices and len(self.all_prices[t]) >= DD_EXIT_LOOKBACK:
                    c = self.all_prices[t]
                    peak = c.iloc[-DD_EXIT_LOOKBACK:].max()
                    if peak > 0 and (c.iloc[-1] / peak - 1) <= DD_EXIT_THRESHOLD:
                        log(f"📉 DD Exit: {t} ({c.iloc[-1]/peak-1:+.1%} from {DD_EXIT_LOOKBACK}d peak) → 전 트랜치 제거")
                        dd_removed.add(t)
            # Also remove health-failed coins from tranches
            for t in bad_coins:
                if t in all_tranche_coins:
                    log(f"🚨 헬스 실패: {t} → 전 트랜치 제거")
                    dd_removed.add(t)

            if dd_removed:
                for a_str in trade_state['tranches']:
                    tr = trade_state['tranches'][a_str]
                    for t in dd_removed:
                        if t in tr.get('weights', {}):
                            del tr['weights'][t]
                        if t in tr.get('picks', []):
                            tr['picks'].remove(t)

        # ── 4. PFD5: 플립 후 5일 ──
        if (is_risk_on_now and not trade_state.get('pfd_done', True)
                and trade_state.get('last_flip_date')):
            from datetime import datetime as dt2
            try:
                flip_dt = dt2.strptime(trade_state['last_flip_date'], '%Y-%m-%d')
                if (datetime.now() - flip_dt).days >= 5:
                    log(f"🔄 PFD5: 플립 후 5일 → 전 트랜치 갱신")
                    for a_str in trade_state['tranches']:
                        tr = trade_state['tranches'][a_str]
                        tr['picks'] = [t for t in current_signal_w if t != 'Cash']
                        tr['weights'] = {t: w for t, w in current_signal_w.items() if t != 'Cash'}
                    trade_state['pfd_done'] = True
                    trade_state['_pfd_triggered'] = True  # 트리거 플래그
            except Exception:
                pass

        # ── 5. 앵커일: 해당 트랜치만 갱신 ──
        anchors_triggered = []

        # Plan 기반: recommend가 앵커 대상을 알려줌
        plan_anchors = None
        if USE_EXECUTION_PLAN:
            try:
                with open(SIGNAL_STATE_FILE, 'r') as _sf:
                    _plan = json.load(_sf).get('execution_plan', {}).get('coin', {})
                plan_anchors = _plan.get('today_anchors', [])
            except Exception:
                pass

        if plan_anchors is not None and is_risk_on_now and not signal_flipped:
            for a in plan_anchors:
                a_str = str(a)
                tr = trade_state['tranches'].get(a_str, {})
                if tr.get('last_anchor_month', '') < current_month:
                    log(f"📅 [PLAN] 앵커 Day {a} → 트랜치 갱신")
                    tr['picks'] = [t for t in current_signal_w if t != 'Cash']
                    tr['weights'] = {t: w for t, w in current_signal_w.items() if t != 'Cash'}
                    tr['last_anchor_month'] = current_month
                    trade_state['tranches'][a_str] = tr
                    anchors_triggered.append(a)
        elif plan_anchors is None and is_risk_on_now and not signal_flipped:
            # Legacy fallback
            for a in ANCHOR_DAYS:
                a_str = str(a)
                tr = trade_state['tranches'].get(a_str, {})
                if today >= a and tr.get('last_anchor_month', '') < current_month:
                    log(f"📅 앵커일 Day {a} → 트랜치 갱신")
                    tr['picks'] = [t for t in current_signal_w if t != 'Cash']
                    tr['weights'] = {t: w for t, w in current_signal_w.items() if t != 'Cash'}
                    tr['last_anchor_month'] = current_month
                    trade_state['tranches'][a_str] = tr
                    anchors_triggered.append(a)
        elif is_risk_off and not signal_flipped:
            # Risk-Off 상태에서도 앵커일 마킹 (현금 유지)
            for a in ANCHOR_DAYS:
                a_str = str(a)
                tr = trade_state['tranches'].get(a_str, {})
                if today >= a and tr.get('last_anchor_month', '') < current_month:
                    tr['picks'] = []
                    tr['weights'] = {}
                    tr['last_anchor_month'] = current_month
                    trade_state['tranches'][a_str] = tr

        # ── 6. Force 실행 (앵커월 소모하지 않음) ──
        if self.is_force and not signal_flipped and not anchors_triggered:
            log(f"⚡ 강제 실행 → 전 트랜치 갱신 (앵커 미소모)")
            for a_str in trade_state['tranches']:
                tr = trade_state['tranches'][a_str]
                if is_risk_on_now:
                    tr['picks'] = [t for t in current_signal_w if t != 'Cash']
                    tr['weights'] = {t: w for t, w in current_signal_w.items() if t != 'Cash'}
                else:
                    tr['picks'] = []
                    tr['weights'] = {}
                # last_anchor_month는 변경하지 않음 — force가 앵커를 소모하면 안 됨

        # ── 합산 타겟 계산 ──
        combined_target = {}
        n_tranches = len(ANCHOR_DAYS)
        for a_str, tr in trade_state['tranches'].items():
            for t, w in tr.get('weights', {}).items():
                combined_target[t] = combined_target.get(t, 0) + w / n_tranches

        # Cash buffer
        invested = sum(combined_target.values())
        cash_pct = max(1.0 - invested, 0)
        cash_buf = get_cash_buffer()
        combined_target['Cash'] = cash_buf + (cash_pct - cash_buf if cash_pct > cash_buf else 0)

        log(f"\n📊 3트랜치 합산 타겟:")
        for a_str in sorted(trade_state['tranches'].keys()):
            tr = trade_state['tranches'][a_str]
            p = tr.get('picks', [])
            log(f"  트랜치 Day{a_str}: {p if p else '현금'} (last: {tr.get('last_anchor_month','-')})")
        log(f"  합산: {', '.join(f'{t}:{w:.1%}' for t, w in sorted(combined_target.items(), key=lambda x:-x[1]) if t != 'Cash')}")
        log(f"  현금: {combined_target.get('Cash', 0):.1%}")

        # Override target_w with combined
        target_w = combined_target

        # ── 트리거 기반 매매 판단 ──
        # 트리거가 발생한 경우만 매매, 가격 변동만으로는 매매 안 함
        trade_reasons = []
        if signal_flipped:
            trade_reasons.append(f"카나리아 플립 ({'ON' if is_risk_on_now else 'OFF'})")
        if anchors_triggered:
            trade_reasons.append(f"앵커일 Day {anchors_triggered}")
        if dd_removed:
            trade_reasons.append(f"DD Exit {list(dd_removed)}")
        bad_coins = [c for c in bad_coins if c != 'Cash']  # Cash 제외
        if bad_coins:
            trade_reasons.append(f"헬스 실패 {bad_coins}")
        if trade_state.get('_pfd_triggered'):
            trade_reasons.append("PFD5")
            del trade_state['_pfd_triggered']
        if is_first_run:
            trade_reasons.append("초기 진입")
        if self.is_force:
            trade_reasons.append("강제 실행")

        # rebalancing_needed 플래그: 이전 리밸런싱이 미완료면 재시도
        if trade_state.get('rebalancing_needed') and not trade_reasons:
            trade_reasons.append("리밸런싱 미완료 재시도")

        # pending이 남아있으면 트리거로 판단
        existing_pending = trade_state.get('pending_trades', {})
        if existing_pending and not trade_reasons:
            trade_reasons.append(f"pending 복구 ({list(existing_pending.keys())})")

        # cash_buffer 변경 시 트리거
        if trade_state.get('buffer_changed') and not trade_reasons:
            trade_reasons.append("buffer 변경")
            del trade_state['buffer_changed']

        # 트리거가 있으면 rebalancing_needed 설정
        if trade_reasons:
            trade_state['rebalancing_needed'] = True

        # Recalculate turnover for logging
        turnover = sum(abs(curr_w.get(k,0) - target_w.get(k,0))
                       for k in set(curr_w)|set(target_w)) / 2
        log(f"🔄 합산 턴오버: {turnover:.1%}")

        if not trade_reasons:
            log(f"✅ HOLD — 트리거 없음 (턴오버: {turnover:.1%})")
            # HOLD에서도 trade_state를 저장 (앵커 마킹 + monitor 캐시 갱신)
            trade_state['coin_risk_on'] = is_risk_on_now
            trade_state['updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            self._refresh_monitor_cache(trade_state)
            self._save_trade_state(trade_state, TRADE_STATE_FILE)
            return turnover

        log(f"⚡ 매매 트리거: {', '.join(trade_reasons)}")
        
    # 리밸런싱 실행
        # [수정] Target Amount 적용 (입력된 금액만큼만 운용, 나머지는 현금)
        investable_total = total_val
        if self.target_amount > 0:
            investable_total = min(self.target_amount, total_val)
            log(f"🎯 Target Amount 적용: {self.target_amount:,.0f}원 (실 운용액: {investable_total:,.0f}원)")
        
        # ── pending 초기화 ──
        pending = {}

        # ── 1. 매도 (시총 낮은 것 먼저) ──
        sell_list = []
        for t in curr_w.keys():
            if t == 'Cash':
                continue
            tgt = target_w.get(t, 0)
            target_amt_val = investable_total * tgt
            current_amt_val = cur_assets_val.get(t, 0)
            if target_amt_val < current_amt_val:
                sell_amt = current_amt_val - target_amt_val
                is_full = (tgt == 0)
                sell_list.append((t, sell_amt, is_full))

        # 시총 낮은 것 먼저 (유니버스 역순)
        sell_list.sort(key=lambda x: x[0], reverse=True)

        for t, sell_amt, is_full in sell_list:
            if is_full:
                log(f"📉 {t} 전량 매도")
                qty = holdings_qty.get(t, 0)
                self.ensure_sell(t, qty, is_clearance=True)
                # 전량 매도 후 잔량 체크 → 실패 시 pending에 저장
                remaining_bal = self.upbit.get_balance(f"KRW-{t}") or 0
                remaining_price = pyupbit.get_current_price(f"KRW-{t}") or 0
                if remaining_bal * remaining_price >= MIN_ORDER_KRW:
                    pending[t] = {'side': 'sell_all', 'target_krw': 0, 'filled_krw': 0,
                                  'created': datetime.now().strftime('%Y-%m-%d %H:%M')}
            else:
                log(f"📉 {t} 부분 매도 (-{sell_amt:,.0f}원)")
                sold = self.split_sell(t, sell_amt, timeout_sec=180)
                if sell_amt - sold > 50000:
                    pending[t] = {'side': 'sell', 'target_krw': sell_amt, 'filled_krw': sold,
                                  'created': datetime.now().strftime('%Y-%m-%d %H:%M')}

        time.sleep(2)

        # ── 2. 매수 (시총 높은 것 먼저) ──
        buy_list = []
        for t, w in target_w.items():
            if t == 'Cash':
                continue
            tgt_amt = investable_total * w
            cur_amt = cur_assets_val.get(t, 0)
            needed = tgt_amt - cur_amt
            if needed > MIN_ORDER_KRW:
                buy_list.append((t, needed))

        # 시총 높은 것 먼저 (유니버스 순서)
        # buy_list는 이미 target_w 순서 (시총순)

        krw_bal = self.upbit.get_balance("KRW") or 0
        log(f"\n💵 매수 시작 (보유 현금: {krw_bal:,.0f}원)")

        for t, needed in buy_list:
            krw_bal = self.upbit.get_balance("KRW") or 0
            buy_amt = min(needed, krw_bal * 0.99)
            if buy_amt < MIN_ORDER_KRW:
                log(f"  💤 {t} 매수 스킵: 현금 부족 ({krw_bal:,.0f}원)")
                pending[t] = {'side': 'buy', 'target_krw': needed, 'filled_krw': 0,
                              'created': datetime.now().strftime('%Y-%m-%d %H:%M')}
                continue
            log(f"  👉 {t} 분할매수 계획: +{buy_amt:,.0f}원")
            filled = self.split_buy(t, buy_amt, timeout_sec=180)
            if needed - filled > 50000:
                pending[t] = {'side': 'buy', 'target_krw': needed, 'filled_krw': filled,
                              'created': datetime.now().strftime('%Y-%m-%d %H:%M')}

        # pending 저장
        trade_state['pending_trades'] = pending
        if pending:
            log(f"📋 Pending: {list(pending.keys())}")

        # rebalancing_needed 플래그: pending이 없으면 완료
        if not pending:
            trade_state['rebalancing_needed'] = False
            log(f"✅ 리밸런싱 완료 — rebalancing_needed: false")
        else:
            trade_state['rebalancing_needed'] = True
            log(f"⏳ 리밸런싱 미완료 — rebalancing_needed: true (pending 있음)")

        self._refresh_monitor_cache(trade_state)

        # ── State 저장 (매매 실행 후) ──
        trade_state['coin_risk_on'] = is_risk_on_now
        trade_state['updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        trade_state['last_trade_reasons'] = trade_reasons
        self._save_trade_state(trade_state, TRADE_STATE_FILE)
        log(f"💾 trade_state 저장 완료")

        # 🎯 텔레그램 알림 전송 (상세 내역 포함)
        if self.is_live_trade:
             msg = f"🤖 V16 Upbit 리밸런싱 완료\n트리거: {', '.join(trade_reasons)}\n턴오버: {turnover:.1%}"
             if self.is_force: msg += " (FORCE)"
             if self.target_amount > 0: msg += f"\nTarget: {self.target_amount:,.0f} KRW"

             if self.trade_history:
                 msg += "\n\n[체결 내역]\n" + "\n".join(self.trade_history)
             else:
                 msg += "\n\n(체결된 주문 없음)"
             send_telegram(msg)
        return turnover

    def run_monitor(self):
        """--monitor 모드: 긴급 탈출 + pending 복구 (30분마다 실행).
        NOTE: 중복 실행 방지는 run_trade.sh의 flock이 담당. 여기선 락 불필요."""
        try:
            TRADE_STATE_FILE = 'trade_state.json'
            trade_state = {}
            try:
                with open(TRADE_STATE_FILE, 'r') as f:
                    trade_state = json.load(f)
            except Exception:
                return  # state 없으면 아무것도 안 함

            # ── 1. 긴급 탈출 체크 (pyupbit만) ──
            emergency = False
            emergency_reason = ''

            # ── BTC USD 가격 조회 (카나리/Crash용) ──
            btc_usd = 0
            try:
                btc_krw = pyupbit.get_current_price("KRW-BTC") or 0
                usdt_krw = pyupbit.get_current_price("KRW-USDT") or 0
                if usdt_krw > 0 and btc_krw > 0:
                    btc_usd = btc_krw / usdt_krw
                else:
                    log(f"⚠️ USDT 비정상 ({usdt_krw}), Binance fallback 시도")
                    try:
                        resp = requests.get("https://api.binance.com/api/v3/ticker/price?symbol=BTCUSDT", timeout=5)
                        btc_usd = float(resp.json()['price'])
                    except Exception:
                        log(f"⚠️ Binance도 실패, USD 신호 스킵")
            except Exception as e:
                log(f"⚠️ Monitor BTC 가격 조회 실패: {e}")

            # ── BTC 가격 로그 ──
            if btc_usd > 0:
                log(f"📡 BTC USD: ${btc_usd:,.0f}")

            # ── Blacklist 장중 체크: 보유 코인 전일 대비 -15% ──
            if not emergency:
                try:
                    active_coins = set()
                    for tr in trade_state.get('tranches', {}).values():
                        active_coins.update(tr.get('picks', []))
                    if active_coins:
                        for coin in active_coins:
                            try:
                                cur_price = pyupbit.get_current_price(f"KRW-{coin}") or 0
                                # 전일 종가: Yahoo CSV에서 가져오기
                                csv_path = f"data/{coin}-USD.csv"
                                if not os.path.exists(csv_path) or cur_price <= 0:
                                    continue
                                import pandas as _pd
                                df = _pd.read_csv(csv_path)
                                if len(df) < 2:
                                    continue
                                prev_close_usd = float(df['Adj_Close'].iloc[-1])
                                if prev_close_usd <= 0 or usdt_krw <= 0:
                                    continue
                                prev_close_krw = prev_close_usd * usdt_krw
                                ret = cur_price / prev_close_krw - 1
                                if ret <= BL_THRESHOLD:
                                    log(f"🚫 Blacklist 장중: {coin} {ret:+.1%} (현재 {cur_price:,.0f} vs 전일 {prev_close_krw:,.0f})")
                                    # 해당 코인만 매도
                                    bal = self.upbit.get_balance(f"KRW-{coin}") or 0
                                    if bal > 0:
                                        self.order(coin, 'sell', bal)
                                        send_telegram(f"🚫 Blacklist: {coin} {ret:+.1%}")
                                    for a_str in trade_state.get('tranches', {}):
                                        tr = trade_state['tranches'][a_str]
                                        if coin in tr.get('weights', {}):
                                            del tr['weights'][coin]
                                        if coin in tr.get('picks', []):
                                            tr['picks'].remove(coin)
                                    trade_state['rebalancing_needed'] = True
                            except Exception:
                                pass
                except Exception as e:
                    log(f"⚠️ Blacklist 장중 체크 실패: {e}")

            # ── 카나리아 OFF: BTC USD < SMA50 USD * 0.985 ──
            has_positions = any(t.get('picks') for t in trade_state.get('tranches', {}).values())
            if not emergency and btc_usd > 0:
                # 하위호환: USD 키 없으면 KRW 키로 fallback
                btc_sma60_usd = trade_state.get('btc_sma60_usd', 0)
                if btc_sma60_usd == 0:
                    btc_sma60_krw = trade_state.get('btc_sma60', 0)
                    if btc_sma60_krw > 0 and usdt_krw > 0:
                        btc_sma60_usd = btc_sma60_krw / usdt_krw
                if btc_sma60_usd > 0:
                    if trade_state.get('coin_risk_on', False) or has_positions:
                        if btc_usd < btc_sma60_usd * (1 - CANARY_HYST):
                            emergency = True
                            emergency_reason = f"카나리아 OFF (BTC ${btc_usd:,.0f} < SMA50*{1-CANARY_HYST:.3f} ${btc_sma60_usd*(1-CANARY_HYST):,.0f})"

            # ── DD Exit: 보유코인 USD 60일 고점 대비 -25% → 해당 코인만 매도 ──
            dd_exit_coins = []
            if not emergency:
                try:
                    active_coins = set()
                    for tr in trade_state.get('tranches', {}).values():
                        active_coins.update(tr.get('picks', []))
                    for coin in active_coins:
                        csv_path = f"data/{coin}-USD.csv"
                        if not os.path.exists(csv_path):
                            continue
                        df = pd.read_csv(csv_path)
                        if len(df) < DD_EXIT_LOOKBACK:
                            continue
                        peak_60d = df['Adj_Close'].iloc[-DD_EXIT_LOOKBACK:].max()
                        cur_usd = df['Adj_Close'].iloc[-1]
                        if peak_60d > 0 and cur_usd > 0:
                            dd = cur_usd / peak_60d - 1
                            if dd <= DD_EXIT_THRESHOLD:
                                dd_exit_coins.append((coin, dd, cur_usd, peak_60d))
                except Exception as e:
                    log(f"⚠️ DD Exit CSV 조회 실패: {e}")

            # ── DD Exit 발동: 해당 코인만 개별 매도 ──
            if dd_exit_coins:
                for coin, dd, cur, peak in dd_exit_coins:
                    reason = f"DD Exit {coin} ({dd:+.1%}, ${cur:,.1f} vs 60d peak ${peak:,.1f})"
                    log(f"📉 MONITOR DD: {reason}")
                    send_telegram(f"📉 DD Exit: {reason}")
                    # 해당 코인만 매도
                    try:
                        bal = self.upbit.get_balance(f"KRW-{coin}") or 0
                        if bal > 0:
                            self.order(coin, 'sell', bal)
                            log(f"  ✅ {coin} 매도 완료 ({bal:.4f}개)")
                    except Exception as e:
                        log(f"  ⚠️ {coin} 매도 실패: {e}")
                    # 전 트랜치에서 제거
                    for a_str in trade_state.get('tranches', {}):
                        tr = trade_state['tranches'][a_str]
                        if coin in tr.get('weights', {}):
                            del tr['weights'][coin]
                        if coin in tr.get('picks', []):
                            tr['picks'].remove(coin)
                trade_state['rebalancing_needed'] = True
                trade_state['updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                self._save_trade_state(trade_state, TRADE_STATE_FILE)

            # ── Crash/카나리OFF 긴급 발동 시 → 전량 매도 ──
            if emergency:
                log(f"🚨 MONITOR 긴급: {emergency_reason}")
                send_telegram(f"🚨 V16 긴급 탈출: {emergency_reason}")

                held_coins = []
                try:
                    balances = self.upbit.get_balances()
                    for b in balances:
                        if b['currency'] == 'KRW':
                            continue
                        bal = float(b['balance']) + float(b.get('locked', 0))
                        price = pyupbit.get_current_price(f"KRW-{b['currency']}") or 0
                        if bal * price >= MIN_ORDER_KRW:
                            held_coins.append(b['currency'])
                except Exception:
                    pass

                if held_coins:
                    self.emergency_sell_all(held_coins)

                trade_state['pending_trades'] = {}
                for a_str in trade_state.get('tranches', {}):
                    trade_state['tranches'][a_str]['picks'] = []
                    trade_state['tranches'][a_str]['weights'] = {}
                trade_state['coin_peaks'] = {}
                trade_state['coin_risk_on'] = False
                trade_state['rebalancing_needed'] = False
                trade_state['updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                trade_state['last_emergency'] = emergency_reason
                self._save_trade_state(trade_state, TRADE_STATE_FILE)
                return

            # ── 2. Pending 복구 ──
            pending = trade_state.get('pending_trades', {})
            if not pending:
                # peak만 갱신하고 종료
                self._save_trade_state(trade_state, TRADE_STATE_FILE)
                return

            log(f"📋 Monitor: pending {len(pending)}건 복구 시도")

            for ticker, p in list(pending.items()):
                side = p.get('side', '')
                if side == 'buy':
                    target = p.get('target_krw', 0)
                    filled = p.get('filled_krw', 0)
                    remaining = target - filled
                    if remaining < 50000:
                        del pending[ticker]
                        continue
                    added = self.split_buy(ticker, remaining, timeout_sec=180)
                    p['filled_krw'] = filled + added
                    if target - p['filled_krw'] < 50000:
                        del pending[ticker]
                        log(f"  ✅ {ticker} pending 매수 완료")

                elif side == 'sell':
                    target = p.get('target_krw', 0)
                    filled = p.get('filled_krw', 0)
                    remaining = target - filled
                    if remaining < 50000:
                        del pending[ticker]
                        continue
                    sold = self.split_sell(ticker, remaining, timeout_sec=180)
                    p['filled_krw'] = filled + sold
                    if target - p['filled_krw'] < 50000:
                        del pending[ticker]
                        log(f"  ✅ {ticker} pending 매도 완료")

                elif side == 'sell_all':
                    bal = self.upbit.get_balance(f"KRW-{ticker}") or 0
                    price = pyupbit.get_current_price(f"KRW-{ticker}") or 0
                    if bal * price < MIN_ORDER_KRW:
                        del pending[ticker]
                        log(f"  ✅ {ticker} pending 전량매도 완료")
                    else:
                        self.ensure_sell(ticker, bal, is_clearance=True)

            trade_state['pending_trades'] = pending
            # pending 모두 완료되면 rebalancing_needed 해제
            if not pending and trade_state.get('rebalancing_needed'):
                trade_state['rebalancing_needed'] = False
                log(f"✅ 모든 pending 완료 → rebalancing_needed: false")
            trade_state['updated'] = datetime.now().strftime('%Y-%m-%d %H:%M')
            self._save_trade_state(trade_state, TRADE_STATE_FILE)
            log(f"💾 Monitor 완료 (pending {len(pending)}건 남음)")

        finally:
            pass


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trade', action='store_true', help="실제 매매 수행")
    parser.add_argument('--force', action='store_true', help="턴오버 무시하고 강제 매매")
    parser.add_argument('--monitor', action='store_true', help="모니터링 모드 (긴급탈출+pending)")
    parser.add_argument('--amount', type=int, default=0, help="목표 운용 금액 (0=전체)")
    args = parser.parse_args()
    
    trader = V16UpbitTrader(is_live_trade=args.trade or args.monitor,
                            is_force=args.force, target_amount=args.amount)
    if args.monitor:
        trader.run_monitor()
    else:
        trader.run()
