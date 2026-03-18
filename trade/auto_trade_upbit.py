"""
V16 Upbit Auto Trader (CoinGecko Global Universe)
==================================================
V16 코인 전략: BTC > SMA(60) + 1%hyst, Mom30+Mom90+Vol5%, 시총순 Top 5 EW + 25% Cap
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
from datetime import datetime, timezone
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
CASH_BUFFER_PERCENT = 0.02
MIN_ORDER_KRW = 5000

# V15 Canary / Protection
CANARY_SMA_PERIOD = 60           # V15: SMA(60) (was 50)
CANARY_HYST = 0.01               # 1% hysteresis band
CRASH_THRESHOLD = -0.10          # BTC daily -10% → cash
BL_THRESHOLD = -0.15             # -15% daily → 7d exclude
BL_DAYS = 7
DD_EXIT_LOOKBACK = 60            # 60-day peak
DD_EXIT_THRESHOLD = -0.25        # -25% drawdown → sell

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
        log(f"V15 Upbit Trader [{mode}]")
        log("전략: SMA60+1%hyst, Mom21+Mom90+Vol5%, 시총순 Top5 EW, DD/BL/Crash")
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
            log("  ❌ CoinGecko Error => Fully Failed. Using Hardcoded Fallback")
            return ['BTC', 'ETH', 'XRP', 'SOL', 'KRW-SHIB'], {}

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
        if len(btc) < CANARY_SMA_PERIOD: return {}, True, "BTC 데이터 부족", []

        # BTC 기준 Target Date
        tgt_dt = btc.index[-1].date() if hasattr(btc.index[-1], 'date') else btc.index[-1]

        # --- Crash Breaker (G5): BTC daily -10% in last 3d → cash ---
        CRASH_COOLDOWN = 3
        if len(btc) >= CRASH_COOLDOWN + 1:
            recent_rets = btc.iloc[-(CRASH_COOLDOWN + 1):].pct_change().dropna()
            worst_crash = recent_rets.min()
            if worst_crash <= CRASH_THRESHOLD:
                log(f"🚨 CRASH BREAKER: BTC worst {worst_crash:+.1%} in {CRASH_COOLDOWN}d → 현금 대기")
                return {}, True, f"🚨 CRASH: BTC {worst_crash:+.1%}", []

        # --- Canary: BTC > SMA(60) with 1% Hysteresis ---
        sma = self.calc_sma(btc, CANARY_SMA_PERIOD)
        cur = btc.iloc[-1]
        dist = cur / sma - 1
        if dist > CANARY_HYST:
            canary_on = True
        elif dist < -CANARY_HYST:
            canary_on = False
        else:
            canary_on = cur > sma  # dead zone
        log(f"  BTC: ${cur:,.0f} vs SMA({CANARY_SMA_PERIOD}): ${sma:,.0f} (dist {dist:+.2%}, hyst ±{CANARY_HYST:.0%})")

        if not canary_on:
            return {}, True, f"🚨 Risk-Off: BTC <= SMA{CANARY_SMA_PERIOD} (dist {dist:+.2%})", []

        # 2. Health Check: Mom(21)>0 AND Mom(90)>0 AND Vol(90)<=5%
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

        # Equal Weight with 25% Cap (DD-exited weight → cash)
        COIN_WEIGHT_CAP = 0.25
        ew = min(1.0 / len(top5), COIN_WEIGHT_CAP)
        w = {t: ew for t in final_picks}
        cash_from_dd = ew * len(dd_exits)

        # Cash Buffer 2% + DD exit cash
        buffered_w = {t: val * (1.0 - CASH_BUFFER_PERCENT) for t, val in w.items()}
        buffered_w['Cash'] = CASH_BUFFER_PERCENT + cash_from_dd

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

        universe, _ = self.get_coingecko_top100()
        target_w, is_risk_off, status, healthy_list = self.get_target_portfolio(universe)
        
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
        
        # 헬스체크 실패 보유 코인 검사
        bad_coins = []
        healthy_set = set(healthy_list)
        for t in curr_w.keys():
            if t not in healthy_set and t not in target_w:
                bad_coins.append(t)
        
        do_rebalance = False
        if is_risk_off:
            log(f"🚨 Risk-Off -> 전량 매도")
            do_rebalance = True
        elif bad_coins:
            log(f"🚨 헬스체크 실패 코인 발견 ({', '.join(bad_coins)}) -> 강제 리밸런싱")
            do_rebalance = True
        elif turnover >= TURNOVER_THRESHOLD or self.is_force:
            log(f"⚡ 턴오버 초과 또는 강제 실행 -> 리밸런싱")
            do_rebalance = True
        else:
            log(f"✅ HOLD (턴오버 미달)")
            
        if not do_rebalance: return turnover
        
    # 리밸런싱 실행
        # [수정] Target Amount 적용 (입력된 금액만큼만 운용, 나머지는 현금)
        investable_total = total_val
        if self.target_amount > 0:
            investable_total = min(self.target_amount, total_val)
            log(f"🎯 Target Amount 적용: {self.target_amount:,.0f}원 (실 운용액: {investable_total:,.0f}원)")
        
        # 1. 매도 (확실하게 청산)
        for t in curr_w.keys():
            tgt = target_w.get(t, 0)
            target_amt_val = investable_total * tgt
            current_amt_val = cur_assets_val.get(t, 0)
            
            # 목표보다 많이 보유 중이면 매도
            if target_amt_val < current_amt_val: 
                sell_amt_krw = current_amt_val - target_amt_val
                
                # 수량으로 변환
                qty = holdings_qty[t]
                sell_ratio = sell_amt_krw / current_amt_val
                sell_qty = qty * sell_ratio
                
                if tgt == 0: sell_qty = qty # 전량
                
                self.ensure_sell(t, sell_qty, is_clearance=(tgt == 0))
        
        time.sleep(2)
        
        # 2. 매수 (확실하게 확보)
        krw_bal = self.upbit.get_balance("KRW") or 0
        log(f"\n💵 매수 시작 (보유 현금: {krw_bal:,.0f}원)")
        
        for t, w in target_w.items():
            if t == 'Cash': continue # Skip Cash
            tgt_amt = investable_total * w
            
            # 매수 계획 로그 (항상 출력)
            cur_amt = cur_assets_val.get(t, 0) 
            needed = tgt_amt - cur_amt
            if needed > 5000:
                log(f"  👉 {t} 매수 계획: +{needed:,.0f}원 (목표 비중: {w:.1%})")
            
            if tgt_amt > cur_amt:
                self.ensure_buy(t, tgt_amt)
        
        # 🎯 텔레그램 알림 전송 (상세 내역 포함)
        if self.is_live_trade:
             msg = f"🤖 V16 Upbit 리밸런싱 완료\n턴오버: {turnover:.1%}"
             if self.is_force: msg += " (FORCE)"
             if self.target_amount > 0: msg += f"\nTarget: {self.target_amount:,.0f} KRW"
             
             if self.trade_history:
                 msg += "\n\n[체결 내역]\n" + "\n".join(self.trade_history)
             else:
                 msg += "\n\n(체결된 주문 없음)"
             send_telegram(msg)
        return turnover

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--trade', action='store_true', help="실제 매매 수행")
    parser.add_argument('--force', action='store_true', help="턴오버 무시하고 강제 매매")
    parser.add_argument('--amount', type=int, default=0, help="목표 운용 금액 (0=전체)")
    args = parser.parse_args()
    
    V16UpbitTrader(is_live_trade=args.trade, is_force=args.force, target_amount=args.amount).run()
