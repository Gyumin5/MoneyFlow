"""
Cap Defend V17 — 한국투자증권 해외주식 자동매매
================================================
--trade   : 전략 신호 기반 매매 (cron 23:25 KST, 장 개시 전 주문)
--monitor : 체결 확인 + 미체결 재주문 (cron 23:35, 00:05, ... 장중)
--balance : 잔고 조회
--force   : 현재 신호대로 강제 매수 (최초 설정용)
--test    : API 연결 테스트

cron 예시:
  25 23 * * 1-5  cd /home/ubuntu && python3 auto_trade_kis.py --trade >> kis_trade.log 2>&1
  35 23 * * 1-5  cd /home/ubuntu && python3 auto_trade_kis.py --monitor >> kis_trade.log 2>&1
  05 0-5 * * 2-6 cd /home/ubuntu && python3 auto_trade_kis.py --monitor >> kis_trade.log 2>&1
"""

import os
import json
import time
import argparse
import logging
import requests
from datetime import datetime, timedelta
from typing import Tuple, List

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
log = logging.getLogger(__name__)

# ─── Config ──────────────────────────────────────────────
try:
    from config import KIS_APP_KEY, KIS_APP_SECRET, KIS_ACCOUNT, KIS_ACCOUNT_PROD
except ImportError:
    KIS_APP_KEY = os.environ.get('KIS_APP_KEY', '')
    KIS_APP_SECRET = os.environ.get('KIS_APP_SECRET', '')
    KIS_ACCOUNT = os.environ.get('KIS_ACCOUNT', '')
    KIS_ACCOUNT_PROD = os.environ.get('KIS_ACCOUNT_PROD', '01')

try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_BOT_TOKEN = ""
    TELEGRAM_CHAT_ID = ""

BASE_URL = "https://openapi.koreainvestment.com:9443"
TOKEN_FILE = os.path.expanduser("~/.kis_token.json")
SIGNAL_STATE_FILE = "signal_state.json"
KIS_TRADE_STATE_FILE = "kis_trade_state.json"

# ETF → 거래소 매핑
EXCHANGE_MAP = {
    # 한투 실전계좌: NASD = 미국 전체 (NASDAQ+NYSE+AMEX)
    'SPY': 'NASD', 'QQQ': 'NASD', 'VEA': 'NASD', 'EEM': 'NASD',
    'GLD': 'NASD', 'PDBC': 'NASD', 'VNQ': 'NASD',
    'IEF': 'NASD', 'BIL': 'NASD', 'BNDX': 'NASD', 'VT': 'NASD',
}


# ─── Telegram ────────────────────────────────────────────
def send_telegram(msg):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json={"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"},
            timeout=10
        )
    except Exception:
        pass


# ─── Token ───────────────────────────────────────────────
def _get_token() -> str:
    if os.path.exists(TOKEN_FILE):
        try:
            with open(TOKEN_FILE, 'r') as f:
                cache = json.load(f)
            expires = datetime.strptime(cache['expires'], '%Y-%m-%d %H:%M:%S')
            if datetime.now() < expires - timedelta(hours=1):
                return cache['token']
        except Exception:
            pass

    resp = requests.post(f"{BASE_URL}/oauth2/tokenP", json={
        "grant_type": "client_credentials",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET
    }, timeout=10)
    resp.raise_for_status()
    data = resp.json()
    token = data['access_token']
    expires = data.get('access_token_token_expired', '')

    os.makedirs(os.path.dirname(TOKEN_FILE) or '.', exist_ok=True)
    with open(TOKEN_FILE, 'w') as f:
        json.dump({'token': token, 'expires': expires}, f)
    try:
        os.chmod(TOKEN_FILE, 0o600)
    except Exception:
        pass
    log.info("새 토큰 발급 완료")
    return token


def _headers(tr_id: str) -> dict:
    return {
        "Content-Type": "application/json; charset=utf-8",
        "authorization": f"Bearer {_get_token()}",
        "appkey": KIS_APP_KEY,
        "appsecret": KIS_APP_SECRET,
        "tr_id": tr_id,
        "custtype": "P",
    }


def _get(path: str, tr_id: str, params: dict, retries: int = 3) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.get(f"{BASE_URL}{path}", headers=_headers(tr_id), params=params, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                log.warning(f"API GET 재시도 {attempt+1}/{retries}: {e}")
                time.sleep(2 * (attempt + 1))
            else:
                raise


def _post(path: str, tr_id: str, body: dict, retries: int = 2) -> dict:
    for attempt in range(retries):
        try:
            resp = requests.post(f"{BASE_URL}{path}", headers=_headers(tr_id), json=body, timeout=15)
            resp.raise_for_status()
            return resp.json()
        except Exception as e:
            if attempt < retries - 1:
                log.warning(f"API POST 재시도 {attempt+1}/{retries}: {e}")
                time.sleep(2 * (attempt + 1))
            else:
                raise


# ─── API Functions ───────────────────────────────────────
def get_balance() -> Tuple[list, dict]:
    """해외주식 잔고 → (holdings, summary)"""
    data = _get("/uapi/overseas-stock/v1/trading/inquire-balance", "TTTS3012R", {
        "CANO": KIS_ACCOUNT, "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
        "OVRS_EXCG_CD": "NASD", "TR_CRCY_CD": "USD",  # NASD=미국전체(실전)
        "CTX_AREA_FK200": "", "CTX_AREA_NK200": "",
    })
    holdings = []
    for item in data.get('output1', []):
        qty = float(item.get('ovrs_cblc_qty', 0))
        if qty <= 0:
            continue
        holdings.append({
            'ticker': item.get('ovrs_pdno', ''),
            'qty': int(qty),
            'avg_price': float(item.get('pchs_avg_pric', 0)),
            'current_price': float(item.get('now_pric2', 0)),
            'eval_amt': float(item.get('ovrs_stck_evlu_amt', 0)),
        })
    summary_raw = data.get('output2', {})
    if isinstance(summary_raw, list):
        summary_raw = summary_raw[0] if summary_raw else {}
    return holdings, summary_raw


def get_buying_power_usd() -> float:
    """매수가능 금액. 장중: psamount(통합증거금), 장외: foreign-margin."""
    # 1차: psamount (장중, 통합증거금 포함)
    try:
        data = _get("/uapi/overseas-stock/v1/trading/inquire-psamount", "TTTS3007R", {
            "CANO": KIS_ACCOUNT, "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
            "OVRS_EXCG_CD": "NASD", "OVRS_ORD_UNPR": "100",
            "ITEM_CD": "SPY",
        }, retries=1)
        output = data.get('output', {})
        integrated = float(output.get('frcr_ord_psbl_amt1', 0))
        if integrated > 0:
            return integrated
        basic = float(output.get('ord_psbl_frcr_amt', 0))
        if basic > 0:
            return basic
    except Exception:
        pass

    # 2차: foreign-margin (장외에도 동작)
    try:
        data = _get("/uapi/overseas-stock/v1/trading/foreign-margin", "TTTC2101R", {
            "CANO": KIS_ACCOUNT, "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
        }, retries=1)
        for item in data.get('output', []):
            if isinstance(item, dict) and item.get('natn_name') == '미국' and item.get('crcy_cd') == 'USD':
                integrated = float(item.get('itgr_ord_psbl_amt', 0))
                if integrated > 0:
                    return integrated
                basic = float(item.get('frcr_gnrl_ord_psbl_amt', 0))
                if basic > 0:
                    return basic
    except Exception:
        pass

    # 3차: present-balance (외화 출금가능)
    try:
        data = _get("/uapi/overseas-stock/v1/trading/inquire-present-balance", "CTRP6504R", {
            "CANO": KIS_ACCOUNT, "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
            "WCRC_FRCR_DVSN_CD": "01", "NATN_CD": "840",
            "TR_MKET_CD": "00", "INQR_DVSN_CD": "00",
        }, retries=1)
        for item in data.get('output2', []):
            if isinstance(item, dict) and item.get('crcy_cd') == 'USD':
                return float(item.get('frcr_drwg_psbl_amt_1', 0))
    except Exception:
        pass

    return 0.0


# 잔고 가격 캐시 (잔고 조회 시 자동 갱신)
_balance_price_cache = {}

def get_current_price(ticker: str) -> float:
    """현재가 조회. 장중: 시세API, 장외: 잔고 종가 fallback."""
    exchange = EXCHANGE_MAP.get(ticker, 'NASD')
    excd_map = {'NASD': 'NAS', 'AMEX': 'AMS', 'NYSE': 'NYS'}
    excd = excd_map.get(exchange, 'NAS')

    # 1차: 시세 API (올바른 경로: overseas-price)
    try:
        data = _get("/uapi/overseas-price/v1/quotations/price-detail", "HHDFS76200200", {
            "AUTH": "", "EXCD": excd, "SYMB": ticker,
        }, retries=1)
        price = float(data.get('output', {}).get('last', 0))
        if price > 0:
            return price
    except Exception:
        pass

    # 2차: 잔고 캐시에서 가져오기 (장외 시간)
    if ticker in _balance_price_cache and _balance_price_cache[ticker] > 0:
        log.info(f"  {ticker}: 잔고 종가 사용 ${_balance_price_cache[ticker]:.2f}")
        return _balance_price_cache[ticker]

    # 3차: 잔고 조회해서 캐시 갱신
    try:
        holdings, _ = get_balance()
        for h in holdings:
            _balance_price_cache[h['ticker']] = h['current_price']
            if h['ticker'] == ticker and h['current_price'] > 0:
                return h['current_price']
    except Exception:
        pass

    # 4차: KIS 기간별시세 API (장외에도 종가 반환)
    try:
        from datetime import date
        bymd = date.today().strftime('%Y%m%d')
        data = _get("/uapi/overseas-price/v1/quotations/dailyprice", "HHDFS76240000", {
            "AUTH": "", "EXCD": excd, "SYMB": ticker,
            "GUBN": "0", "BYMD": bymd, "MODP": "0",
        }, retries=1)
        for item in data.get('output2', []):
            if isinstance(item, dict):
                price = float(item.get('clos', 0))
                if price > 0:
                    log.info(f"  {ticker}: KIS 일봉 종가 ${price:.2f}")
                    _balance_price_cache[ticker] = price
                    return price
    except Exception:
        pass

    # 5차: Yahoo Finance 최종 fallback
    try:
        import yfinance as yf
        tk = yf.Ticker(ticker)
        hist = tk.history(period="5d")
        if not hist.empty:
            price = float(hist['Close'].iloc[-1])
            if price > 0:
                log.info(f"  {ticker}: Yahoo 종가 ${price:.2f}")
                _balance_price_cache[ticker] = price
                return price
    except Exception:
        pass

    log.error(f"{ticker} 현재가 조회 실패")
    return 0.0


def get_pending_orders() -> list:
    """미체결 내역 조회."""
    data = _get("/uapi/overseas-stock/v1/trading/inquire-nccs", "TTTS3018R", {
        "CANO": KIS_ACCOUNT, "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
        "OVRS_EXCG_CD": "NASD", "SORT_SQN": "DS",
        "CTX_AREA_FK200": "", "CTX_AREA_NK200": "",
    })
    orders = []
    for item in data.get('output', []):
        if isinstance(item, dict) and float(item.get('nccs_qty', 0)) > 0:
            orders.append({
                'order_no': item.get('odno', ''),
                'ticker': item.get('pdno', ''),
                'side': 'buy' if item.get('sll_buy_dvsn_cd') == '02' else 'sell',
                'qty': int(float(item.get('nccs_qty', 0))),
                'price': float(item.get('ft_ord_unpr3', 0)),
            })
    return orders


def place_order(ticker: str, qty: int, price: float, side: str = "buy",
                order_type: str = "00") -> dict:
    """주문 실행. order_type: 00=지정가, 32=LOO, 34=LOC"""
    exchange = EXCHANGE_MAP.get(ticker, 'NASD')
    if side == "buy":
        tr_id = "TTTT1002U" if exchange in ('NASD',) else "TTTT1002U"
    else:
        tr_id = "TTTT1006U" if exchange in ('NASD',) else "TTTT1006U"

    body = {
        "CANO": KIS_ACCOUNT, "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
        "OVRS_EXCG_CD": exchange, "PDNO": ticker,
        "ORD_QTY": str(qty), "OVRS_ORD_UNPR": str(price),
        "CTAC_TLNO": "", "MGCO_APTM_ODNO": "",
        "ORD_SVR_DVSN_CD": "0", "ORD_DVSN": order_type,
    }
    data = _post("/uapi/overseas-stock/v1/trading/order", tr_id, body)
    success = data.get('rt_cd') == '0'
    msg = data.get('msg1', '')
    order_no = data.get('output', {}).get('ODNO', '') if success else ''
    return {'success': success, 'message': msg, 'order_no': order_no}


def cancel_order(order_no: str, ticker: str, qty: int, side: str) -> dict:
    """주문 취소. KIS 문서: 취소 시 가격은 "0"."""
    exchange = EXCHANGE_MAP.get(ticker, 'NASD')
    tr_id = "TTTT1004U"

    body = {
        "CANO": KIS_ACCOUNT, "ACNT_PRDT_CD": KIS_ACCOUNT_PROD,
        "OVRS_EXCG_CD": exchange, "PDNO": ticker,
        "ORGN_ODNO": order_no,
        "RVSE_CNCL_DVSN_CD": "02",  # 02=취소
        "ORD_QTY": str(qty), "OVRS_ORD_UNPR": "0",
        "CTAC_TLNO": "", "MGCO_APTM_ODNO": "",
        "ORD_SVR_DVSN_CD": "0",
    }
    data = _post("/uapi/overseas-stock/v1/trading/order-rvsecncl", tr_id, body)
    return {'success': data.get('rt_cd') == '0', 'message': data.get('msg1', '')}


# ─── State Management ────────────────────────────────────
def load_signal_state() -> dict:
    try:
        with open(SIGNAL_STATE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def load_kis_state() -> dict:
    try:
        with open(KIS_TRADE_STATE_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_kis_state(state: dict):
    tmp = KIS_TRADE_STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, KIS_TRADE_STATE_FILE)


# ─── Trade Logic (4트랜치 + rebalancing_needed 플래그) ────
ANCHOR_DAYS = (1, 8, 15, 22)  # 4트랜치
REBAL_TOLERANCE_PCT = 0.05     # 목표 대비 5% 이내면 완료로 판단


def _calc_combined_target(kis_state, target_tickers):
    """트랜치별 목표를 합산하여 전체 목표 비중 계산."""
    tranches = kis_state.get('tranches', {})
    n = len(ANCHOR_DAYS)
    combined = {}
    for a_str, tr in tranches.items():
        for t, w in tr.get('weights', {}).items():
            combined[t] = combined.get(t, 0) + w / n
    # 트랜치가 비어있으면 signal 기반 초기화
    if not combined and target_tickers:
        w = 1.0 / len(target_tickers)
        combined = {t: w for t in target_tickers}
    return combined


def _check_rebal_complete(holdings, combined_target, total_asset):
    """목표 대비 보유가 허용 오차 이내인지 확인.
    금액 기준 ±5% OR 주수가 목표와 동일하면 완료."""
    if total_asset <= 0:
        return True
    current_map = {h['ticker']: h for h in holdings}
    invest = total_asset * 0.98
    for t, target_w in combined_target.items():
        target_val = invest * target_w
        current_val = current_map[t]['eval_amt'] if t in current_map else 0
        current_qty = current_map[t]['qty'] if t in current_map else 0
        # 목표 주수 계산
        price = current_map[t]['current_price'] if t in current_map and current_map[t]['current_price'] > 0 else 0
        target_qty = int(target_val / price) if price > 0 else 0
        # 주수가 같으면 OK (소액에서 금액 차이가 나도 더 살 수 없으므로)
        if target_qty > 0 and current_qty == target_qty:
            continue
        # 금액 기준 ±5%
        if target_val > 50 and abs(current_val - target_val) / target_val > REBAL_TOLERANCE_PCT:
            return False
    # 퇴출 종목이 남아있는지
    for h in holdings:
        if h['ticker'] not in combined_target and h['qty'] > 0:
            return False
    return True


def run_trade():
    """전략 신호 기반 자동 매매 (4트랜치 + rebalancing_needed).
    09:15에 recommend_personal.py가 signal_state.json 갱신.
    23:35에 이 함수가 실행.
    """
    signal = load_signal_state()
    kis_state = load_kis_state()

    stock_crash = signal.get('stock_crash', False)
    crash_cooldown = signal.get('stock_crash_cooldown', 0)
    risk_on = signal.get('risk_on', True)
    signal_flipped = signal.get('signal_flipped', False)
    target_tickers = signal.get('stock_holdings', [])
    current_month = datetime.now().strftime('%Y-%m')
    today = datetime.now().day

    holdings, _ = get_balance()
    current_map = {h['ticker']: h for h in holdings}

    # 미체결 주문 취소 (새로 계산해서 주문하므로)
    try:
        pending_orders = get_pending_orders()
        if pending_orders:
            log.info(f"미체결 {len(pending_orders)}건 취소:")
            for po in pending_orders:
                log.info(f"  취소: {po['side']} {po['ticker']} x{po['qty']}")
                cancel_order(po['order_no'], po['ticker'], po['qty'], po['side'])
                time.sleep(0.5)
            time.sleep(3)
    except Exception:
        pass

    log.info(f"=== KIS Trade (4트랜치) ===")
    log.info(f"Signal: crash={stock_crash}, cooldown={crash_cooldown}, risk_on={risk_on}, flipped={signal_flipped}")
    log.info(f"Target: {target_tickers}")
    log.info(f"Current: {[h['ticker'] for h in holdings]}")

    # ── 트랜치 초기화 ──
    if 'tranches' not in kis_state:
        kis_state['tranches'] = {}
        for a in ANCHOR_DAYS:
            kis_state['tranches'][str(a)] = {
                'anchor_month': '',
                'picks': target_tickers if target_tickers else [],
                'weights': {t: 1.0/len(target_tickers) for t in target_tickers} if target_tickers else {},
            }

    # ── 1. 트리거: Crash → 전 트랜치 즉시 매도 ──
    if stock_crash:
        if not holdings:
            log.info("Crash 상태, 보유 없음. 대기.")
        else:
            log.info(f"🚨 CRASH — 전량 매도")
            for h in holdings:
                _sell_all(h['ticker'], h['qty'], "CRASH")
            # 전 트랜치 현금화
            for a_str in kis_state['tranches']:
                kis_state['tranches'][a_str]['picks'] = []
                kis_state['tranches'][a_str]['weights'] = {}
        # 매도 후 잔고 확인 — 잔량 있으면 재시도 필요
        holdings_after, _ = get_balance()
        if any(h['qty'] > 0 for h in holdings_after):
            kis_state['rebalancing_needed'] = True
            log.warning("⏳ Crash 매도 미완료 — 잔량 있음")
        else:
            kis_state['rebalancing_needed'] = False
        kis_state['last_action'] = 'crash_sell'
        kis_state['last_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        save_kis_state(kis_state)
        return

    # ── 2. 트리거: 카나리 OFF → 전 트랜치 즉시 매도 ──
    if not target_tickers or target_tickers == ['Cash']:
        if holdings:
            log.info("카나리 OFF — 전량 매도")
            for h in holdings:
                _sell_all(h['ticker'], h['qty'], "CANARY_OFF")
            for a_str in kis_state['tranches']:
                kis_state['tranches'][a_str]['picks'] = []
                kis_state['tranches'][a_str]['weights'] = {}
        # 매도 후 잔고 확인
        holdings_after, _ = get_balance()
        if any(h['qty'] > 0 for h in holdings_after):
            kis_state['rebalancing_needed'] = True
            log.warning("⏳ 카나리OFF 매도 미완료 — 잔량 있음")
        else:
            kis_state['rebalancing_needed'] = False
        kis_state['last_action'] = 'canary_sell'
        kis_state['last_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
        save_kis_state(kis_state)
        return

    # ── 2.5. 트리거: Crash 복귀 → 전 트랜치 즉시 재진입 ──
    was_crash = kis_state.get('last_action') in ('crash_sell', 'canary_sell')
    if was_crash and not stock_crash and target_tickers and target_tickers != ['Cash']:
        log.info("📈 Crash/카나리 복귀 — 전 트랜치 즉시 재진입")
        for a_str in kis_state['tranches']:
            tr = kis_state['tranches'][a_str]
            tr['picks'] = target_tickers
            tr['weights'] = {t: 1.0/len(target_tickers) for t in target_tickers}
        kis_state['rebalancing_needed'] = True
        kis_state['last_action'] = 'recovery'

    # ── 3. 트리거: 카나리 플립 → 전 트랜치 즉시 전환 ──
    if signal_flipped:
        log.info("🔄 카나리 플립 — 전 트랜치 즉시 전환")
        for a_str in kis_state['tranches']:
            tr = kis_state['tranches'][a_str]
            tr['picks'] = target_tickers
            tr['weights'] = {t: 1.0/len(target_tickers) for t in target_tickers}
        kis_state['rebalancing_needed'] = True

    # ── 4. 앵커일 체크: 해당 트랜치만 목표 갱신 ──
    for a in ANCHOR_DAYS:
        a_str = str(a)
        tr = kis_state['tranches'].get(a_str, {})
        if today >= a and tr.get('anchor_month', '') < current_month:
            log.info(f"📅 앵커일 Day {a} → 트랜치 갱신")
            tr['picks'] = target_tickers
            tr['weights'] = {t: 1.0/len(target_tickers) for t in target_tickers}
            tr['anchor_month'] = current_month
            kis_state['tranches'][a_str] = tr
            kis_state['rebalancing_needed'] = True

    # ── 5. rebalancing_needed가 false면 종료 ──
    if not kis_state.get('rebalancing_needed', False):
        log.info("리밸런싱 불필요. 종료.")
        save_kis_state(kis_state)
        return

    # ── 6. 합산 목표 계산 + 매매 ──
    combined = _calc_combined_target(kis_state, target_tickers)
    log.info(f"합산 목표: {combined}")

    total_holdings = sum(h['eval_amt'] for h in holdings)
    available = get_buying_power_usd()
    total_asset = total_holdings + available
    invest_budget = total_asset * 0.98
    log.info(f"총 자산: ${total_asset:.2f}, 투자: ${invest_budget:.2f}")

    # 매도 (퇴출 + 초과)
    sells = []
    for h in holdings:
        if h['ticker'] not in combined and h['qty'] > 0:
            sells.append((h['ticker'], h['qty'], "REBALANCE"))
        elif h['ticker'] in combined:
            target_val = invest_budget * combined[h['ticker']]
            diff = h['eval_amt'] - target_val
            if diff > 50:
                price = h['current_price'] if h['current_price'] > 0 else get_current_price(h['ticker'])
                if price > 0:
                    sell_qty = int(diff / price)
                    if sell_qty > 0:
                        sells.append((h['ticker'], sell_qty, "EW_ADJUST"))

    for ticker, qty, reason in sells:
        _sell_all(ticker, qty, reason)

    # 매수 (부족)
    buys = []
    for t, w in combined.items():
        target_val = invest_budget * w
        current_val = current_map[t]['eval_amt'] if t in current_map else 0
        diff = target_val - current_val
        if diff > 50:
            buys.append((t, diff))

    if buys:
        time.sleep(3)
        for ticker, budget in buys:
            _buy_target(ticker, budget)
            time.sleep(0.5)

    # ── 7. 완료 체크 ──
    holdings_after, _ = get_balance()
    total_after = sum(h['eval_amt'] for h in holdings_after) + get_buying_power_usd()
    if _check_rebal_complete(holdings_after, combined, total_after):
        log.info("✅ 리밸런싱 완료 — rebalancing_needed: false")
        kis_state['rebalancing_needed'] = False
    else:
        log.info("⏳ 리밸런싱 미완료 — 다음 실행에서 재시도")
        kis_state['rebalancing_needed'] = True

    kis_state['last_action'] = 'trade'
    kis_state['last_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    save_kis_state(kis_state)
    log.info("=== Trade 완료 ===")


def run_force():
    """현재 신호대로 전 트랜치 강제 리밸런싱."""
    signal = load_signal_state()
    target_tickers = signal.get('stock_holdings', [])

    if not target_tickers or target_tickers == ['Cash']:
        log.info("현재 신호: Cash. 매수 대상 없음.")
        return

    kis_state = load_kis_state()

    # 전 트랜치 강제 갱신 (anchor_month는 변경 안 함)
    if 'tranches' not in kis_state:
        kis_state['tranches'] = {}
    for a in ANCHOR_DAYS:
        a_str = str(a)
        kis_state['tranches'][a_str] = {
            'anchor_month': kis_state.get('tranches', {}).get(a_str, {}).get('anchor_month', ''),
            'picks': target_tickers,
            'weights': {t: 1.0/len(target_tickers) for t in target_tickers},
        }
    kis_state['rebalancing_needed'] = True
    save_kis_state(kis_state)

    # run_trade 호출하여 실제 매매
    log.info("=== Force → Trade 실행 ===")
    run_trade()


def run_monitor():
    """장중 모니터: VT Crash 체크 + 리밸런싱 복구. 매시 실행."""
    kis_state = load_kis_state()

    # ── VT Crash 장중 체크 ──
    signal = load_signal_state()
    if not signal.get('stock_crash', False):
        # Crash 상태가 아닐 때만 장중 체크 (이미 Crash면 trade가 처리)
        try:
            vt_price = get_current_price('VT')
            vt_prev = signal.get('vt_prev_close', 0)
            if vt_price > 0 and vt_prev > 0:
                vt_ret = vt_price / vt_prev - 1
                if vt_ret <= -0.03:
                    log.info(f"🚨 장중 VT Crash! {vt_ret:+.1%} (${vt_price:.2f} vs prev ${vt_prev:.2f})")
                    send_telegram(f"🚨 <b>장중 VT Crash!</b> {vt_ret:+.1%}\n즉시 전량 매도")
                    # 전량 매도
                    holdings, _ = get_balance()
                    for h in holdings:
                        if h['qty'] > 0:
                            _sell_all(h['ticker'], h['qty'], "MONITOR_CRASH")
                    for a_str in kis_state.get('tranches', {}):
                        kis_state['tranches'][a_str]['picks'] = []
                        kis_state['tranches'][a_str]['weights'] = {}
                    kis_state['rebalancing_needed'] = False
                    kis_state['last_action'] = 'crash_sell'
                    kis_state['last_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
                    save_kis_state(kis_state)
                    return
        except Exception as e:
            log.info(f"⚠️ VT 장중 체크 실패: {e}")

    # rebalancing_needed가 false면 아무것도 안 함
    if not kis_state.get('rebalancing_needed', False):
        pending = get_pending_orders()
        if pending:
            log.info(f"미체결 {len(pending)}건 (rebal 완료 상태)")
        else:
            log.info("미체결 없음. rebal 완료.")
        return

    log.info("=== Monitor: rebalancing_needed=true ===")

    # 1. 미체결 전부 취소
    pending = get_pending_orders()
    if pending:
        log.info(f"미체결 {len(pending)}건 취소:")
        for order in pending:
            log.info(f"  취소: {order['side']} {order['ticker']} x{order['qty']} @ ${order['price']:.2f}")
            cancel_order(order['order_no'], order['ticker'], order['qty'], order['side'])
            time.sleep(0.5)
        time.sleep(3)  # 취소 반영 대기

    # 2. 현재 잔고 + 목표 계산
    signal = load_signal_state()
    target_tickers = signal.get('stock_holdings', [])
    if not target_tickers or target_tickers == ['Cash']:
        log.info("목표: Cash. 보유 있으면 매도.")
        holdings, _ = get_balance()
        for h in holdings:
            if h['qty'] > 0:
                _sell_all(h['ticker'], h['qty'], "MONITOR_SELL")
        kis_state['rebalancing_needed'] = False
        save_kis_state(kis_state)
        return

    combined = _calc_combined_target(kis_state, target_tickers)
    holdings, _ = get_balance()
    current_map = {h['ticker']: h for h in holdings}
    total_holdings = sum(h['eval_amt'] for h in holdings)
    available = get_buying_power_usd()
    total_asset = total_holdings + available
    invest_budget = total_asset * 0.98

    log.info(f"목표: {combined}")
    log.info(f"총 자산: ${total_asset:.2f}")

    # 3. 매도 (퇴출 + 초과)
    for h in holdings:
        if h['ticker'] not in combined and h['qty'] > 0:
            _sell_all(h['ticker'], h['qty'], "MONITOR_REBAL")
        elif h['ticker'] in combined:
            target_val = invest_budget * combined[h['ticker']]
            if h['eval_amt'] - target_val > 50:
                price = h['current_price'] if h['current_price'] > 0 else get_current_price(h['ticker'])
                if price > 0:
                    sell_qty = int((h['eval_amt'] - target_val) / price)
                    if sell_qty > 0:
                        _sell_all(h['ticker'], sell_qty, "MONITOR_EW")

    # 4. 매수 (부족)
    for t, w in combined.items():
        target_val = invest_budget * w
        current_val = current_map[t]['eval_amt'] if t in current_map else 0
        diff = target_val - current_val
        if diff > 50:
            _buy_target(t, diff)
            time.sleep(0.5)

    # 5. 완료 체크
    time.sleep(3)
    holdings_after, _ = get_balance()
    total_after = sum(h['eval_amt'] for h in holdings_after) + get_buying_power_usd()
    if _check_rebal_complete(holdings_after, combined, total_after):
        log.info("✅ Monitor: 리밸런싱 완료 — rebalancing_needed: false")
        kis_state['rebalancing_needed'] = False
    else:
        log.info("⏳ Monitor: 아직 미완료 — 다음 실행에서 재시도")

    kis_state['last_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    save_kis_state(kis_state)


# ─── Helper Functions ────────────────────────────────────
MAX_ORDER_ATTEMPTS = 5
ORDER_WAIT_SEC = 5


def _sell_all(ticker: str, qty: int, reason: str):
    """반복 매도: 주문 → 대기 → 미체결 취소 → 재시도.
    qty: 매도할 주수 (전량이 아닌 delta 기준)."""
    log.info(f"  SELL {ticker} x{qty} ({reason})")
    # 매도 전 보유수량 기록
    holdings_before, _ = get_balance()
    qty_before = 0
    for h in holdings_before:
        if h['ticker'] == ticker:
            qty_before = h['qty']
    target_qty_after = max(0, qty_before - qty)  # 매도 후 목표 보유수량

    for attempt in range(1, MAX_ORDER_ATTEMPTS + 1):
        holdings_now, _ = get_balance()
        current_qty = 0
        for h in holdings_now:
            if h['ticker'] == ticker:
                current_qty = h['qty']
        remaining = current_qty - target_qty_after
        if remaining <= 0:
            sold = qty_before - current_qty
            log.info(f"    ✅ {ticker} 매도 완료 ({sold}주)")
            if sold > 0:
                send_telegram(f"📉 <b>매도</b>: {ticker} x{sold}\n사유: {reason}")
            return
        price = get_current_price(ticker)
        if price <= 0:
            log.error(f"  {ticker}: 가격 조회 실패")
            break
        sell_price = round(price * 0.99, 2)
        log.info(f"    [{attempt}] 매도 {remaining}주 @ ${sell_price}")
        result = place_order(ticker, remaining, sell_price, side="sell")
        if not result['success']:
            log.warning(f"    주문 실패: {result['message']}")
            break
        time.sleep(ORDER_WAIT_SEC)
        # 미체결 취소
        pending = get_pending_orders()
        for p in pending:
            if p['ticker'] == ticker and p['side'] == 'sell':
                cancel_order(p['order_no'], p['ticker'], p['qty'], p['side'])
                time.sleep(0.5)
    # 최종 확인
    holdings_final, _ = get_balance()
    final_qty = 0
    for h in holdings_final:
        if h['ticker'] == ticker:
            final_qty = h['qty']
    sold_total = qty_before - final_qty
    if final_qty > target_qty_after:
        log.warning(f"    ⏳ {ticker} 매도 미완료 ({final_qty - target_qty_after}주 남음)")
        send_telegram(f"⚠️ <b>매도 미완료</b>: {ticker} {final_qty - target_qty_after}주 남음\n사유: {reason}")
    elif sold_total > 0:
        send_telegram(f"📉 <b>매도</b>: {ticker} x{sold_total}\n사유: {reason}")


def _buy_target(ticker: str, budget_usd: float):
    """반복 매수: 주문 → 대기 → 미체결 취소 → 재시도.
    budget_usd: 추가로 매수할 금액 (기존 보유분 제외)."""
    price = get_current_price(ticker)
    if price <= 0:
        log.error(f"  {ticker}: 가격 조회 실패, 매수 스킵")
        return
    add_qty = int(budget_usd / (price * 1.01))
    if add_qty <= 0:
        log.warning(f"  {ticker}: 수량 0 (budget ${budget_usd:.2f}, price ${price})")
        return

    # 매수 전 보유수량 기록
    holdings_before, _ = get_balance()
    qty_before = 0
    for h in holdings_before:
        if h['ticker'] == ticker:
            qty_before = h['qty']
    target_qty_after = qty_before + add_qty  # 매수 후 목표 보유수량

    log.info(f"  BUY {ticker} +{add_qty}주 (현재 {qty_before} → 목표 {target_qty_after})")

    for attempt in range(1, MAX_ORDER_ATTEMPTS + 1):
        holdings_now, _ = get_balance()
        current_qty = 0
        for h in holdings_now:
            if h['ticker'] == ticker:
                current_qty = h['qty']
        remaining = target_qty_after - current_qty
        if remaining <= 0:
            bought = current_qty - qty_before
            log.info(f"    ✅ {ticker} 매수 완료 (+{bought}주, 총 {current_qty}주)")
            if bought > 0:
                send_telegram(f"📈 <b>매수</b>: {ticker} +{bought}주 (총 {current_qty}주)")
            return
        price = get_current_price(ticker)
        if price <= 0:
            break
        buy_price = round(price * 1.01, 2)
        log.info(f"    [{attempt}] 매수 {remaining}주 @ ${buy_price}")
        result = place_order(ticker, remaining, buy_price, side="buy")
        if not result['success']:
            log.warning(f"    주문 실패: {result['message']}")
            break
        time.sleep(ORDER_WAIT_SEC)
        # 미체결 취소
        pending = get_pending_orders()
        for p in pending:
            if p['ticker'] == ticker and p['side'] == 'buy':
                cancel_order(p['order_no'], p['ticker'], p['qty'], p['side'])
                time.sleep(0.5)
    # 최종 확인
    holdings_final, _ = get_balance()
    final_qty = 0
    for h in holdings_final:
        if h['ticker'] == ticker:
            final_qty = h['qty']
    bought_total = final_qty - qty_before
    if final_qty < target_qty_after:
        log.warning(f"    ⏳ {ticker} 매수 미완료 (+{bought_total}/{add_qty}주)")
        send_telegram(f"⚠️ <b>매수 미완료</b>: {ticker} +{bought_total}/{add_qty}주")
    elif bought_total > 0:
        send_telegram(f"📈 <b>매수</b>: {ticker} +{bought_total}주 (총 {final_qty}주)")


# ─── CLI ─────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description='Cap Defend V17 - KIS Auto Trader')
    parser.add_argument('--trade', action='store_true', help='전략 신호 기반 매매 (23:25 실행)')
    parser.add_argument('--monitor', action='store_true', help='미체결 확인 + 재주문 (장중)')
    parser.add_argument('--balance', action='store_true', help='잔고 조회')
    parser.add_argument('--force', action='store_true', help='현재 신호대로 강제 매수')
    parser.add_argument('--test', action='store_true', help='API 연결 테스트')
    args = parser.parse_args()

    if args.test:
        print("=== KIS API 연결 테스트 ===")
        try:
            token = _get_token()
            print(f"✅ 토큰: {token[:20]}...")
            holdings, _ = get_balance()
            print(f"✅ 잔고: {len(holdings)}종목")
            for h in holdings:
                print(f"   {h['ticker']}: {h['qty']}주 @ ${h['current_price']:.2f}")
            avail = get_buying_power_usd()
            print(f"✅ 매수가능: ${avail:.2f}")
            signal = load_signal_state()
            print(f"✅ 신호: risk_on={signal.get('risk_on')}, crash={signal.get('stock_crash')}, target={signal.get('stock_holdings')}")
            print("\n✅ 모든 API 정상!")
        except Exception as e:
            print(f"❌ 오류: {e}")

    elif args.balance:
        holdings, _ = get_balance()
        avail = get_buying_power_usd()
        total = sum(h['eval_amt'] for h in holdings)
        print(f"\n=== 해외주식 잔고 ===")
        for h in holdings:
            print(f"  {h['ticker']:6s} {h['qty']:4d}주  avg ${h['avg_price']:8.2f}  now ${h['current_price']:8.2f}  eval ${h['eval_amt']:10.2f}")
        print(f"\n  총 평가: ${total:,.2f}")
        print(f"  매수가능: ${avail:,.2f}")
        print(f"  합계: ${total + avail:,.2f}")

    elif args.trade:
        run_trade()

    elif args.force:
        run_force()

    elif args.monitor:
        run_monitor()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
