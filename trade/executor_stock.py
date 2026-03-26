#!/usr/bin/env python3
"""주식 Executor — 단일 모드 run_once().

설계 원칙:
  - signal_state.json 읽기 (recommend 출력)
  - kis_trade_state.json 읽기/쓰기
  - 전략 신호 계산 안 함
  - signal_state 수정 안 함

이벤트 우선순위:
  1. 가드: VT Crash (-3%) → 전량 매도, 3일+SMA10 대기
  2. 카나리 플립 → 전 트랜치 즉시 전환
  3. 앵커 체크 (해당 트랜치에 offense/defense picks 반영)
  4. Delta 매매

Usage:
  python3 executor_stock.py               # 실행
  python3 executor_stock.py --dry-run     # 주문 없이 로그만
"""

import json, os, time, argparse, logging, uuid, logging
from datetime import datetime, timedelta
from typing import Dict, List, Tuple

import requests

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

# ─── Config ───
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
    TELEGRAM_BOT_TOKEN = ''
    TELEGRAM_CHAT_ID = ''

# ─── 상수 ───
BASE_URL = 'https://openapi.koreainvestment.com:9443'
TOKEN_FILE = os.path.expanduser('~/.kis_token.json')
SIGNAL_STATE_FILE = 'signal_state.json'
TRADE_STATE_FILE = 'kis_trade_state.json'
LOG_FILE = 'executor_stock.log'

ANCHOR_DAYS = (1, 8, 15, 22)
CASH_BUFFER = 0.02
MAX_ORDER_ATTEMPTS = 5
ORDER_WAIT_SEC = 5
LIMIT_PRICE_SLIP = 0.003   # ±0.3%

# Crash 상수
CRASH_THRESHOLD = 0.97      # vt_prev_close × 0.97
CRASH_COOL_DAYS = 3

EXCHANGE_MAP = {
    'SPY': 'NASD', 'QQQ': 'NASD', 'VEA': 'NASD', 'EEM': 'NASD',
    'GLD': 'NASD', 'PDBC': 'NASD', 'VNQ': 'NASD',
    'IEF': 'NASD', 'BIL': 'NASD', 'BNDX': 'NASD', 'VT': 'NASD',
}

STALE_SIGNAL_HOURS = 24


# ═══ 유틸리티 ═══
RUN_ID = ''  # run_once()에서 설정

def _setup_logger():
    from logging.handlers import TimedRotatingFileHandler
    logger = logging.getLogger(LOG_FILE)
    if not logger.handlers:
        handler = TimedRotatingFileHandler(LOG_FILE, when='midnight', backupCount=14, encoding='utf-8')
        handler.setFormatter(logging.Formatter('%(message)s'))
        logger.addHandler(handler)
        logger.setLevel(logging.INFO)
    return logger

_logger = _setup_logger()

def log(msg: str):
    ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    line = f'[{ts}] [{RUN_ID[:8]}] {msg}'
    _logger.info(line)


def send_telegram(msg: str):
    if not TELEGRAM_BOT_TOKEN or not TELEGRAM_CHAT_ID:
        return
    try:
        url = f'https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage'
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': f'[주식] {msg}'}, timeout=5)
    except Exception:
        pass


def load_json(path: str) -> dict:
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def save_json(path: str, data: dict):
    tmp = path + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


# ═══ KIS API (기존 auto_trade_kis.py에서 가져옴) ═══

def _get_token() -> str:
    """OAuth 토큰 (캐시 사용)."""
    if os.path.exists(TOKEN_FILE):
        with open(TOKEN_FILE) as f:
            cached = json.load(f)
        expires = datetime.fromisoformat(cached.get('expires', '2000-01-01'))
        if datetime.now() < expires:
            return cached['token']

    resp = requests.post(f'{BASE_URL}/oauth2/tokenP', json={
        'grant_type': 'client_credentials',
        'appkey': KIS_APP_KEY, 'appsecret': KIS_APP_SECRET,
    }, timeout=10)
    data = resp.json()
    token = data['access_token']
    expires = datetime.now() + timedelta(hours=23)
    with open(TOKEN_FILE, 'w') as f:
        json.dump({'token': token, 'expires': expires.isoformat()}, f)
    return token


def _headers(tr_id: str) -> dict:
    return {
        'authorization': f'Bearer {_get_token()}',
        'appkey': KIS_APP_KEY, 'appsecret': KIS_APP_SECRET,
        'tr_id': tr_id, 'Content-Type': 'application/json; charset=UTF-8',
    }


def _get(path: str, tr_id: str, params: dict, retries=3) -> dict:
    for i in range(retries):
        try:
            resp = requests.get(f'{BASE_URL}{path}', headers=_headers(tr_id), params=params, timeout=15)
            data = resp.json()
            if data.get('rt_cd') == '0':
                return data
            if i < retries - 1:
                time.sleep(1)
        except Exception as e:
            if i < retries - 1:
                time.sleep(1)
    return {}


def _post(path: str, tr_id: str, body: dict, retries=2) -> dict:
    for i in range(retries):
        try:
            resp = requests.post(f'{BASE_URL}{path}', headers=_headers(tr_id), json=body, timeout=15)
            data = resp.json()
            if data.get('rt_cd') == '0':
                return data
            if i < retries - 1:
                time.sleep(1)
        except Exception:
            if i < retries - 1:
                time.sleep(1)
    return {}


class KISAPI:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run

    def get_balance(self) -> Tuple[Dict[str, int], float, float]:
        """잔고 → (holdings {ticker: qty}, total_usd, exchange_rate)."""
        data = _get('/uapi/overseas-stock/v1/trading/inquire-balance', 'TTTS3012R', {
            'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
            'AFHR_FLPR_YN': 'N', 'OFL_YN': '', 'INQR_DVSN': '01',
            'UNPR_DVSN': '01', 'FUND_STTL_ICLD_YN': 'N',
            'FNCG_AMT_AUTO_RDPT_YN': 'N', 'OVRS_EXCG_CD': 'NASD', 'TR_CRCY_CD': 'USD',
            'CTX_AREA_FK200': '', 'CTX_AREA_NK200': '',
        })
        holdings = {}
        total_usd = 0
        for item in data.get('output1', []):
            ticker = item.get('ovrs_pdno', '')
            qty = int(float(item.get('ovrs_cblc_qty', 0)))
            eval_amt = float(item.get('ovrs_stck_evlu_amt', 0))
            if qty > 0 and ticker:
                holdings[ticker] = qty
                total_usd += eval_amt

        # 환율
        exrt = 1350.0  # fallback
        fm_data = _get('/uapi/overseas-stock/v1/trading/foreign-margin', 'TTTC2101R', {
            'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
            'OVRS_EXCG_CD': 'NASD', 'CRCY_CD': 'USD',
        })
        if fm_data.get('output'):
            try:
                exrt = float(fm_data['output'].get('bass_exrt', 1350))
            except Exception:
                pass

        # USD 현금
        cash_usd = 0
        bp_data = _get('/uapi/overseas-stock/v1/trading/inquire-present-balance', 'CTRP6504R', {
            'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
            'WCRC_FRCR_DVSN_CD': '02', 'INQR_DVSN_CD': '00',
        })
        for item in bp_data.get('output3', []):
            if item.get('crcy_cd') == 'USD':
                try:
                    cash_usd = float(item.get('frcr_evlu_amt2', 0))
                except Exception:
                    pass

        total_usd += cash_usd
        return holdings, total_usd, exrt

    def get_current_price(self, ticker: str) -> float:
        """현재가 (USD)."""
        data = _get('/uapi/overseas-price/v1/quotations/price-detail', 'HHDFS76200200', {
            'AUTH': '', 'EXCD': EXCHANGE_MAP.get(ticker, 'NASD'), 'SYMB': ticker,
        })
        try:
            return float(data.get('output', {}).get('last', 0))
        except Exception:
            return 0

    def get_vt_price(self) -> float:
        """VT 현재가."""
        return self.get_current_price('VT')

    def cancel_all_pending(self):
        """미체결 전량 취소."""
        data = _get('/uapi/overseas-stock/v1/trading/inquire-nccs', 'TTTS3018R', {
            'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
            'OVRS_EXCG_CD': 'NASD', 'SORT_SQN': 'DS',
            'CTX_AREA_FK200': '', 'CTX_AREA_NK200': '',
        })
        for item in data.get('output', []):
            order_no = item.get('odno', '')
            ticker = item.get('pdno', '')
            qty = int(float(item.get('nccs_qty', 0)))
            side = 'buy' if item.get('sll_buy_dvsn_cd') == '02' else 'sell'
            if order_no and qty > 0:
                log(f'  미체결 취소: {ticker} {side} {qty}주')
                if not self.dry_run:
                    _post('/uapi/overseas-stock/v1/trading/order-rvsecncl', 'JTTT1004U', {
                        'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
                        'OVRS_EXCG_CD': EXCHANGE_MAP.get(ticker, 'NASD'),
                        'PDNO': ticker, 'ORGN_ODNO': order_no,
                        'RVSE_CNCL_DVSN_CD': '02',
                        'ORD_QTY': str(qty), 'OVRS_ORD_UNPR': '0',
                    })

    def place_order(self, ticker: str, qty: int, price: float, side: str = 'buy') -> bool:
        """지정가 주문."""
        if self.dry_run:
            log(f'  [DRY] {side} {ticker} {qty}주 @ ${price:.2f}')
            return True
        tr_id = 'JTTT1002U' if side == 'buy' else 'JTTT1006U'
        result = _post('/uapi/overseas-stock/v1/trading/order', tr_id, {
            'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
            'OVRS_EXCG_CD': EXCHANGE_MAP.get(ticker, 'NASD'),
            'PDNO': ticker,
            'ORD_QTY': str(qty),
            'OVRS_ORD_UNPR': f'{price:.2f}',
            'ORD_SVR_DVSN_CD': '0',
            'ORD_DVSN': '00',
        })
        success = bool(result.get('output', {}).get('ODNO'))
        if success:
            log(f'  주문 {side} {ticker} {qty}주 @ ${price:.2f}')
        else:
            log(f'  주문 실패 {side} {ticker}: {result}')
        return success


# ═══ 핵심 로직 ═══

def check_signal_freshness(signal: dict) -> bool:
    updated = signal.get('meta', {}).get('updated_at', '')
    if not updated:
        return False
    try:
        dt = datetime.strptime(updated, '%Y-%m-%d %H:%M')
        return (datetime.now() - dt).total_seconds() < STALE_SIGNAL_HOURS * 3600
    except Exception:
        return False


def check_crash(signal: dict, api: KISAPI, state: dict) -> bool:
    """VT Crash 체크 + 쿨다운 + SMA10 복귀 관리."""
    guard = state.setdefault('guard_state', {})

    # 이미 crash 중인 경우 → 복귀 조건 체크
    if guard.get('crash_active'):
        cooldown_until = guard.get('crash_cooldown_until', '')
        today = datetime.now().strftime('%Y-%m-%d')
        if today < cooldown_until:
            log(f'  Crash 쿨다운 중 ({cooldown_until}까지)')
            return True  # 아직 대기

        # 쿨다운 끝남 → VT > SMA10 체크
        vt_sma10 = signal.get('stock', {}).get('vt_sma10', 0)
        vt_current = api.get_vt_price()
        if vt_current > 0 and vt_sma10 > 0 and vt_current > vt_sma10:
            log(f'  ✅ Crash 복귀: VT ${vt_current:.2f} > SMA10 ${vt_sma10:.2f}')
            guard['crash_active'] = False
            guard['crash_date'] = None
            guard['crash_cooldown_until'] = None
            state['rebalancing_needed'] = True
            send_telegram(f'Crash 복귀: VT ${vt_current:.2f} > SMA10 ${vt_sma10:.2f}')
            return False
        else:
            # SMA10 아래 → 대기 연장
            guard['crash_cooldown_until'] = (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d')
            log(f'  Crash 대기 연장: VT ${vt_current:.2f} ≤ SMA10 ${vt_sma10:.2f}')
            return True

    # crash 아닌 경우 → 발동 체크
    vt_prev = signal.get('stock', {}).get('vt_prev_close', 0)
    if vt_prev <= 0:
        return False
    vt_current = api.get_vt_price()
    if vt_current <= 0:
        return False

    if vt_current <= vt_prev * CRASH_THRESHOLD:
        log(f'  🚨 VT CRASH: ${vt_current:.2f} ≤ ${vt_prev:.2f} × 0.97')
        guard['crash_active'] = True
        guard['crash_date'] = datetime.now().strftime('%Y-%m-%d')
        guard['crash_cooldown_until'] = (datetime.now() + timedelta(days=CRASH_COOL_DAYS)).strftime('%Y-%m-%d')
        state['rebalancing_needed'] = True
        send_telegram(f'🚨 VT CRASH: ${vt_current:.2f} (전일 ${vt_prev:.2f})')
        return True

    return False


def check_canary_flip(signal: dict, state: dict) -> bool:
    """카나리 플립 → 전 트랜치 즉시 전환."""
    risk_on = signal.get('stock', {}).get('risk_on', True)
    prev = state.get('prev_risk_on')

    if prev is not None and risk_on != prev:
        log(f'  🔄 카나리 플립: {prev} → {risk_on}')
        stock_sig = signal.get('stock', {})
        if risk_on:
            picks = stock_sig.get('offense_picks', [])
            weights = stock_sig.get('offense_weights', {})
        else:
            picks = stock_sig.get('defense_picks', [])
            weights = stock_sig.get('defense_weights', {})

        for tr in state.get('tranches', {}).values():
            tr['picks'] = list(picks)
            tr['weights'] = dict(weights)

        state['prev_risk_on'] = risk_on
        state['rebalancing_needed'] = True
        send_telegram(f'카나리 플립: {"Risk-On 🟢" if risk_on else "Risk-Off 🔴"}')
        return True

    state['prev_risk_on'] = risk_on
    return False


def check_anchors(signal: dict, state: dict):
    """앵커일 체크."""
    today = datetime.now().day
    this_month = datetime.now().strftime('%Y-%m')
    risk_on = signal.get('stock', {}).get('risk_on', True)
    stock_sig = signal.get('stock', {})

    if risk_on:
        picks = stock_sig.get('offense_picks', [])
        weights = stock_sig.get('offense_weights', {})
    else:
        picks = stock_sig.get('defense_picks', [])
        weights = stock_sig.get('defense_weights', {})

    for day in ANCHOR_DAYS:
        day_str = str(day)
        tr = state.setdefault('tranches', {}).setdefault(day_str, {})
        if today >= day and tr.get('anchor_month', '') < this_month:
            tr['picks'] = list(picks)
            tr['weights'] = dict(weights)
            tr['anchor_month'] = this_month
            state['rebalancing_needed'] = True
            log(f'  📅 앵커 Day {day}: {picks} (month={this_month})')


def merge_tranches(state: dict) -> Dict[str, float]:
    """전 트랜치 merge → 최종 target."""
    tranches = state.get('tranches', {})
    n = len(tranches) if tranches else 1
    merged = {}
    for tr in tranches.values():
        for ticker, w in tr.get('weights', {}).items():
            merged[ticker] = merged.get(ticker, 0) + w / n
    return merged


def execute_delta(target: Dict[str, float], api: KISAPI, state: dict):
    """target vs 현재 잔고 → Delta 매매 (정수 주 단위)."""
    holdings, total_usd, exrt = api.get_balance()
    if total_usd <= 0:
        log('  잔고 없음')
        return

    # 현재 비중 계산
    current_values = {}
    for ticker, qty in holdings.items():
        price = api.get_current_price(ticker)
        current_values[ticker] = qty * price if price else 0

    # 매도/매수 리스트
    sells = []
    buys = []
    target_usd = total_usd * (1 - CASH_BUFFER)

    for ticker, target_w in target.items():
        if ticker == 'Cash':
            continue
        target_val = target_usd * target_w
        current_val = current_values.get(ticker, 0)
        price = api.get_current_price(ticker)
        if price <= 0:
            continue

        delta_val = target_val - current_val
        delta_qty = int(delta_val / price)

        if delta_qty < 0 and abs(delta_qty) >= 1:
            sells.append((ticker, abs(delta_qty), price))
        elif delta_qty > 0:
            buys.append((ticker, delta_qty, price))

    # 보유 중이지만 target에 없는 종목 → 전량 매도
    for ticker, qty in holdings.items():
        if ticker not in target:  # target_w=0은 위 루프에서 이미 처리
            price = api.get_current_price(ticker)
            if price > 0 and qty > 0:
                sells.append((ticker, qty, price))

    # 매도 먼저 (delta 주수만큼만, target=0이면 전량)
    for ticker, qty, price in sells:
        sell_price = price * (1 - LIMIT_PRICE_SLIP)
        log(f'  매도: {ticker} {qty}주 @ ${sell_price:.2f}')
        for attempt in range(MAX_ORDER_ATTEMPTS):
            success = api.place_order(ticker, qty, sell_price, 'sell')
            if success:
                break
            time.sleep(ORDER_WAIT_SEC)

    if sells:
        time.sleep(ORDER_WAIT_SEC)
        api.cancel_all_pending()

    # 매수 (재시도 포함)
    for ticker, qty, price in buys:
        buy_price = price * (1 + LIMIT_PRICE_SLIP)
        log(f'  매수: {ticker} {qty}주 @ ${buy_price:.2f}')
        for attempt in range(MAX_ORDER_ATTEMPTS):
            success = api.place_order(ticker, qty, buy_price, 'buy')
            if success:
                break
            time.sleep(ORDER_WAIT_SEC)

    if buys:
        time.sleep(ORDER_WAIT_SEC)
        api.cancel_all_pending()

    # 목표 달성 확인
    holdings2, total2, _ = api.get_balance()
    if total2 > 0:
        max_diff = 0
        for ticker, target_w in target.items():
            if ticker == 'Cash':
                continue
            price = api.get_current_price(ticker)
            current_w = (holdings2.get(ticker, 0) * price / total2) if price > 0 else 0
            max_diff = max(max_diff, abs(target_w - current_w))
        if max_diff < 0.05:
            state['rebalancing_needed'] = False
            log('  ✅ 목표 달성 (±5% 이내)')
            send_telegram(f'✅ [주식] 리밸런싱 완료')
        else:
            log(f'  ⏳ 미달 (max diff={max_diff:.1%}), 다음 실행에서 재시도')


# ═══ run_once ═══

def run_once(dry_run=False):
    global RUN_ID
    RUN_ID = uuid.uuid4().hex
    _t0 = time.time()
    log('=' * 50)
    log('주식 executor 시작')

    signal = load_json(SIGNAL_STATE_FILE)
    state = load_json(TRADE_STATE_FILE)
    api = KISAPI(dry_run=dry_run)

    # ── 디버그 로그: 입력 데이터 ──
    log(f'  signal: {json.dumps(signal, ensure_ascii=False, default=str)[:500]}')
    log(f'  state: {json.dumps(state, ensure_ascii=False, default=str)[:500]}')
    
    if not signal:
        log('  signal_state.json 없음 — 스킵')
        return
    if not state.get('tranches'):
        log('  kis_trade_state.json 트랜치 없음 — 스킵')
        return

    # 첫 실행 초기화
    if state.get('prev_risk_on') is None:
        state['prev_risk_on'] = signal.get('stock', {}).get('risk_on', True)
        log('  첫 실행: prev_risk_on 초기화')

    _su = signal.get('meta', {}).get('updated_at', '')
    try:
        _age = (datetime.now() - datetime.strptime(_su, '%Y-%m-%d %H:%M')).total_seconds() / 3600
        log(f'  signal age: {_age:.1f}h (updated: {_su})')
    except Exception:
        log(f'  signal age: ? (updated: {_su})')
    is_fresh = check_signal_freshness(signal)
    if not is_fresh:
        log('  ⚠️ signal이 24시간 이상 오래됨 — 가드만 체크')
        send_telegram('⚠️ signal_state 갱신 안 됨 (24시간 초과)')

    # 1. 미체결 취소
    api.cancel_all_pending()

    # 2. Crash 체크
    crash = check_crash(signal, api, state)
    if crash:
        # crash_active → target = 전량 현금
        if state.get('rebalancing_needed'):
            target = {'Cash': 1.0}
            log('  Crash: target = 100% 현금')
            execute_delta(target, api, state)
        save_json(TRADE_STATE_FILE, state)
        log('주식 executor 완료 (Crash)')
        return

    if not is_fresh:
        save_json(TRADE_STATE_FILE, state)
        log('주식 executor 완료 (stale signal)')
        return

    # 3. 카나리 플립
    check_canary_flip(signal, state)

    # 4. 앵커 체크
    check_anchors(signal, state)

    # 5. Merge + Delta 매매
    if state.get('rebalancing_needed', False):
        if state.get('guard_state', {}).get('crash_active'):
            target = {'Cash': 1.0}
        else:
            target = merge_tranches(state)
        log(f'  Target: {target}')
        send_telegram(f'📊 [주식] 리밸런싱 시작: {target}')
        # 디버그: 현재 잔고 상세
        _bal = api.get_balance() if hasattr(api, 'get_balance') else {}
        log(f'  Balance: {_bal}')
        execute_delta(target, api, state)

    # 6. 저장
    state['last_action'] = 'executor'
    state['last_trade_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    save_json(TRADE_STATE_FILE, state)
    try:
        _h, _t, _e = api.get_balance()
        _elapsed = time.time() - _t0
        log(f'주식 executor 완료 ({_elapsed:.1f}s) | 총자산: ${_t:,.0f} (환율 {_e:,.0f}) | 잔고: {_h}')
    except Exception:
        _elapsed = time.time() - _t0
        log(f'주식 executor 완료 ({_elapsed:.1f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    try:
        run_once(dry_run=args.dry_run)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        log(f'🚨 FATAL ERROR: {e}')
        log(err)
        send_telegram(f'🚨 [주식] executor 비정상 종료: {e}')
