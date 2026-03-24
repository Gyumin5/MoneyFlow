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
    'SPY': 'AMEX', 'VEA': 'AMEX', 'VNQ': 'AMEX', 'EEM': 'AMEX',
    'GLD': 'AMEX', 'PDBC': 'AMEX', 'IEF': 'NASD', 'BIL': 'AMEX',
    'BNDX': 'NASD', 'VT': 'AMEX', 'QQQ': 'NASD',
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

    # 1차: 시세 API (장중에만 동작)
    try:
        data = _get("/uapi/overseas-stock/v1/quotations/price-detail", "HHDFS76200200", {
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
        data = _get("/uapi/overseas-stock/v1/quotations/dailyprice", "HHDFS76240000", {
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
    """목표 대비 보유가 허용 오차 이내인지 확인."""
    if total_asset <= 0:
        return True
    current_map = {h['ticker']: h for h in holdings}
    for t, target_w in combined_target.items():
        target_val = total_asset * 0.98 * target_w
        current_val = current_map[t]['eval_amt'] if t in current_map else 0
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
    """미체결 확인 + 재주문. 장중 30분마다 실행."""
    pending = get_pending_orders()
    if not pending:
        log.info("미체결 없음.")
        return

    log.info(f"미체결 {len(pending)}건:")
    for order in pending:
        log.info(f"  {order['side']} {order['ticker']} x{order['qty']} @ ${order['price']:.2f} (#{order['order_no']})")

        # 현재가 조회 후 가격 괴리가 크면 취소 → 재주문
        current = get_current_price(order['ticker'])
        if current <= 0:
            continue

        if order['side'] == 'buy' and order['price'] < current * 0.99:
            # 매수 주문 가격이 현재가보다 1% 이상 낮으면 재주문
            log.info(f"  가격 조정: ${order['price']:.2f} → ${current * 1.005:.2f}")
            cancel_result = cancel_order(order['order_no'], order['ticker'],
                                         order['qty'], order['side'])
            if cancel_result['success']:
                time.sleep(0.5)
                new_price = round(current * 1.005, 2)
                place_order(order['ticker'], order['qty'], new_price, side='buy')
                log.info(f"  재주문: {order['ticker']} x{order['qty']} @ ${new_price}")

        elif order['side'] == 'sell' and order['price'] > current * 1.01:
            # 매도 주문 가격이 현재가보다 1% 이상 높으면 재주문
            log.info(f"  가격 조정: ${order['price']:.2f} → ${current * 0.995:.2f}")
            cancel_result = cancel_order(order['order_no'], order['ticker'],
                                         order['qty'], order['side'])
            if cancel_result['success']:
                time.sleep(0.5)
                new_price = round(current * 0.995, 2)
                place_order(order['ticker'], order['qty'], new_price, side='sell')
                log.info(f"  재주문: {order['ticker']} x{order['qty']} @ ${new_price}")


# ─── Helper Functions ────────────────────────────────────
def _sell_all(ticker: str, qty: int, reason: str):
    """전량 매도 (현재가 -0.5% 지정가)."""
    price = get_current_price(ticker)
    if price <= 0:
        log.error(f"  {ticker}: 가격 조회 실패, 매도 스킵")
        return
    sell_price = round(price * 0.995, 2)  # 현재가 -0.5%
    log.info(f"  SELL {ticker} x{qty} @ ${sell_price} ({reason})")
    result = place_order(ticker, qty, sell_price, side="sell")
    log.info(f"  → {result['message']}")
    if result['success']:
        send_telegram(f"📉 <b>매도</b>: {ticker} x{qty} @ ${sell_price}\n사유: {reason}")
    else:
        send_telegram(f"⚠️ <b>매도 실패</b>: {ticker}\n{result['message']}")


def _buy_target(ticker: str, budget_usd: float):
    """목표 금액만큼 매수 (현재가 +2% 지정가, 장외 종가 기반이면 여유있게)."""
    price = get_current_price(ticker)
    if price <= 0:
        log.error(f"  {ticker}: 가격 조회 실패, 매수 스킵")
        return
    buy_price = round(price * 1.02, 2)  # +2% 여유 (종가 기반일 수 있으므로)
    qty = int(budget_usd / buy_price)
    if qty <= 0:
        log.warning(f"  {ticker}: 수량 0 (budget ${budget_usd:.2f}, price ${buy_price})")
        return
    log.info(f"  BUY {ticker} x{qty} @ ${buy_price} (budget ${budget_usd:.2f})")
    result = place_order(ticker, qty, buy_price, side="buy")
    log.info(f"  → {result['message']}")
    if result['success']:
        send_telegram(f"📈 <b>매수</b>: {ticker} x{qty} @ ${buy_price}")
    else:
        send_telegram(f"⚠️ <b>매수 실패</b>: {ticker}\n{result['message']}")


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
