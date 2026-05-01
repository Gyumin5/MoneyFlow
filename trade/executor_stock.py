#!/usr/bin/env python3
"""주식 Executor — 단일 모드 run_once().

설계 원칙:
  - signal_state.json 읽기 (recommend 출력)
  - kis_trade_state.json 읽기/쓰기
  - 전략 신호 계산 안 함
  - signal_state 수정 안 함

이벤트 우선순위 (V23, 가드 없음):
  1. 카나리 플립 → 전 트랜치 즉시 전환
  2. 앵커 체크 (해당 snap 에 offense/defense picks 반영)
  3. Delta 매매

Usage:
  python3 executor_stock.py               # 실행
  python3 executor_stock.py --dry-run     # 주문 없이 로그만
"""

import json, os, sys, time, argparse, logging, uuid
from datetime import datetime, timedelta
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

from common.io import load_json, save_json
from common.notify import send_telegram as _send_tg
from common.logging_utils import setup_file_logger, make_log_fn

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

# V23 (2026-04-30): snap-based 3-tranche stagger (snap=69, stagger=23)
# - snap_id 0/1/2, 각 69일 주기로 rebal, stagger = 23일 (3*23, prime stagger)
# - V22(126/42) → V23(69/23): rank-sum BT 검증 결과 반영 (Cal 1.16 ymin +0.59)
SNAP_PERIOD_DAYS = 69
N_SNAPS = 3
SNAP_STAGGER_DAYS = 23  # SNAP_PERIOD_DAYS / N_SNAPS
CASH_BUFFER_DEFAULT = 0.02
MAX_ORDER_ATTEMPTS = 5
ORDER_WAIT_SEC = 5
LIMIT_PRICE_SLIP = 0.003   # ±0.3%
REBALANCE_TOLERANCE = 0.01  # 목표 달성 판정 허용 오차 ±1%

# Crash 상수
CRASH_THRESHOLD = 0.97      # vt_prev_close × 0.97
CRASH_COOL_DAYS = 3

EXCHANGE_MAP = {
    'BIL': 'AMEX',
    'BNDX': 'NASD',
    'EEM': 'AMEX',
    'GLD': 'AMEX',
    'IEF': 'NASD',
    'PDBC': 'NASD',
    'QQQ': 'NASD',
    'SPY': 'AMEX',
    'VEA': 'AMEX',
    'VNQ': 'AMEX',
    'VT': 'AMEX',
}

# KIS 시세 API(price-detail/dailyprice)는 주문용 EXCHANGE_MAP과 달리
# 실제 상장 거래소 코드(NAS/NYS/AMS)가 정확해야 종목별 가격이 정상 응답된다.
QUOTE_EXCD_MAP = {
    'BIL': 'AMS',
    'BNDX': 'NAS',
    'EEM': 'AMS',
    'GLD': 'AMS',
    'IEF': 'NAS',
    'PDBC': 'NAS',
    'QQQ': 'NAS',
    'SPY': 'AMS',
    'VEA': 'AMS',
    'VNQ': 'AMS',
    'VT': 'AMS',
}

STALE_SIGNAL_HOURS = 24


# ═══ 유틸리티 ═══
RUN_ID = ''  # run_once()에서 설정
_run_id_ref = ['']
_logger = setup_file_logger(LOG_FILE, LOG_FILE)
log = make_log_fn(_logger, _run_id_ref)


def send_telegram(msg: str):
    _send_tg(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg, prefix='주식')


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
    last_error = None
    for i in range(retries):
        try:
            resp = requests.post(f'{BASE_URL}{path}', headers=_headers(tr_id), json=body, timeout=15)
            data = resp.json()
            if data.get('rt_cd') == '0':
                return data
            last_error = {
                'status_code': resp.status_code,
                'body': data,
                'path': path,
                'tr_id': tr_id,
            }
            if i < retries - 1:
                time.sleep(1)
        except Exception as e:
            last_error = {
                'exception': repr(e),
                'path': path,
                'tr_id': tr_id,
            }
            if i < retries - 1:
                time.sleep(1)
    return {'_error': last_error} if last_error else {}


class KISAPI:
    def __init__(self, dry_run=False):
        self.dry_run = dry_run
        self._balance_price_cache: Dict[str, float] = {}

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
        holdings_usd = 0.0
        for item in data.get('output1', []):
            ticker = item.get('ovrs_pdno', '')
            qty = int(float(item.get('ovrs_cblc_qty', 0)))
            eval_amt = float(item.get('ovrs_stck_evlu_amt', 0))
            if qty > 0 and ticker:
                holdings[ticker] = qty
                holdings_usd += eval_amt
                try:
                    cur_price = float(item.get('now_pric2', 0) or item.get('ovrs_now_pric1', 0) or 0)
                    if cur_price > 0:
                        self._balance_price_cache[ticker] = cur_price
                except Exception:
                    pass

        # 환율 — API에서 조회, 실패 시 state 캐시 사용, 캐시도 없으면 None
        exrt = 0.0
        fm_data = _get('/uapi/overseas-stock/v1/trading/foreign-margin', 'TTTC2101R', {
            'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
            'OVRS_EXCG_CD': 'NASD', 'CRCY_CD': 'USD',
        })
        if fm_data.get('output'):
            try:
                exrt = float(fm_data['output'].get('bass_exrt', 0))
            except Exception:
                pass

        # 계좌 전체 자산(KRW 기준) + USD 예수금 fallback
        total_usd = 0.0
        cash_usd = 0.0
        bp_data = _get('/uapi/overseas-stock/v1/trading/inquire-present-balance', 'CTRP6504R', {
            'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
            'WCRC_FRCR_DVSN_CD': '02', 'NATN_CD': '840',
            'TR_MKET_CD': '00', 'INQR_DVSN_CD': '00',
        })
        try:
            output2 = bp_data.get('output2', [])
            if output2 and isinstance(output2[0], dict):
                exrt2 = float(output2[0].get('frst_bltn_exrt', 0))
                if exrt2 > 0:
                    exrt = exrt2
        except Exception:
            pass
        output2 = bp_data.get('output2', [])
        for item in output2:
            if isinstance(item, dict) and item.get('crcy_cd') == 'USD':
                try:
                    cash_usd = float(item.get('frcr_evlu_amt2', 0)) / exrt if exrt > 0 else 0.0
                except Exception:
                    pass

        try:
            total_krw = float(bp_data.get('output3', {}).get('tot_asst_amt', 0))
            if total_krw > 0 and exrt > 0:
                total_usd = total_krw / exrt
        except Exception:
            total_usd = 0.0

        if total_usd <= 0:
            total_usd = holdings_usd + cash_usd

        # 환율 캐싱: 유효한 환율이면 state에 저장, 없으면 캐시에서 복원
        if exrt > 0:
            _state = load_json(TRADE_STATE_FILE)
            _state['last_exchange_rate'] = exrt
            save_json(TRADE_STATE_FILE, _state)
        elif exrt <= 0:
            _state = load_json(TRADE_STATE_FILE)
            cached = _state.get('last_exchange_rate', 0)
            if cached > 0:
                exrt = cached
                log(f'  ⚠️ 환율 API 실패 — 캐시 사용: {exrt}')
            else:
                log('  ⚠️ 환율 API 실패 + 캐시 없음')

        return holdings, total_usd, exrt

    def get_current_price(self, ticker: str) -> float:
        """현재가 (USD). 장중 시세 → 잔고 캐시 → 일봉 종가 순으로 fallback."""
        excd = QUOTE_EXCD_MAP.get(ticker, 'NAS')

        # 1차: 실시간 시세
        try:
            data = _get('/uapi/overseas-price/v1/quotations/price-detail', 'HHDFS76200200', {
                'AUTH': '', 'EXCD': excd, 'SYMB': ticker,
            }, retries=1)
            price = float(data.get('output', {}).get('last', 0))
            if price > 0:
                return price
        except Exception:
            pass

        # 2차: 잔고 조회에서 얻은 가격 캐시
        cached = self._balance_price_cache.get(ticker, 0)
        if cached > 0:
            log(f'  {ticker}: 잔고 캐시 가격 사용 ${cached:.2f}')
            return cached

        # 3차: 잔고 재조회로 캐시 갱신
        try:
            self.get_balance()
            cached = self._balance_price_cache.get(ticker, 0)
            if cached > 0:
                log(f'  {ticker}: 잔고 재조회 가격 사용 ${cached:.2f}')
                return cached
        except Exception:
            pass

        # 4차: KIS 기간별시세 종가
        try:
            bymd = datetime.now().strftime('%Y%m%d')
            data = _get('/uapi/overseas-price/v1/quotations/dailyprice', 'HHDFS76240000', {
                'AUTH': '', 'EXCD': excd, 'SYMB': ticker,
                'GUBN': '0', 'BYMD': bymd, 'MODP': '0',
            }, retries=1)
            for item in data.get('output2', []):
                if isinstance(item, dict):
                    price = float(item.get('clos', 0))
                    if price > 0:
                        self._balance_price_cache[ticker] = price
                        log(f'  {ticker}: KIS 일봉 종가 사용 ${price:.2f}')
                        return price
        except Exception:
            pass

        # 5차: Yahoo 종가 fallback (VT처럼 비보유 종목의 장외 확인용)
        try:
            import yfinance as yf
            tk = yf.Ticker(ticker)
            hist = tk.history(period='5d')
            if not hist.empty:
                price = float(hist['Close'].iloc[-1])
                if price > 0:
                    self._balance_price_cache[ticker] = price
                    log(f'  {ticker}: Yahoo 종가 사용 ${price:.2f}')
                    return price
        except Exception:
            pass

        log(f'  ⚠️ {ticker} 가격 조회 실패')
        return 0.0

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
                    _post('/uapi/overseas-stock/v1/trading/order-rvsecncl', 'TTTT1004U', {
                        'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
                        'OVRS_EXCG_CD': EXCHANGE_MAP.get(ticker, 'NASD'),
                        'PDNO': ticker, 'ORGN_ODNO': order_no,
                        'RVSE_CNCL_DVSN_CD': '02',
                        'ORD_QTY': str(qty), 'OVRS_ORD_UNPR': '0',
                        'CTAC_TLNO': '', 'MGCO_APTM_ODNO': '',
                        'ORD_SVR_DVSN_CD': '0',
                    })

    def place_order(self, ticker: str, qty: int, price: float, side: str = 'buy') -> bool:
        """지정가 주문."""
        if self.dry_run:
            log(f'  [DRY] {side} {ticker} {qty}주 @ ${price:.2f}')
            return True
        tr_id = 'TTTT1002U' if side == 'buy' else 'TTTT1006U'
        result = _post('/uapi/overseas-stock/v1/trading/order', tr_id, {
            'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
            'OVRS_EXCG_CD': EXCHANGE_MAP.get(ticker, 'NASD'),
            'PDNO': ticker,
            'ORD_QTY': str(qty),
            'OVRS_ORD_UNPR': f'{price:.2f}',
            'CTAC_TLNO': '', 'MGCO_APTM_ODNO': '',
            'ORD_SVR_DVSN_CD': '0',
            'ORD_DVSN': '00',
        })
        success = bool(result.get('output', {}).get('ODNO'))
        if success:
            log(f'  주문 {side} {ticker} {qty}주 @ ${price:.2f}')
        else:
            log(f'  주문 실패 {side} {ticker}: {result}')
        return success

    def get_max_buy_qty(self, ticker: str, price: float) -> int:
        """매수 가능 수량 조회 (psamount API). -1이면 조회 실패.

        KIS 해외주식 inquire-psamount 필드 우선순위
        - ovrs_max_ord_psbl_qty: 외화 기준 최대 매수가능수량 (정확)
        - frcr_ord_psbl_amt1: 외화 매수가능금액 → /price 로 qty 환산
        - ord_psbl_qty / ovrs_ord_psbl_amt: 원화 환산 잔액 (외화 잔고는 미반영, 매우 작음)
        """
        if self.dry_run:
            return -1
        data = _get('/uapi/overseas-stock/v1/trading/inquire-psamount', 'TTTS3007R', {
            'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
            'OVRS_EXCG_CD': EXCHANGE_MAP.get(ticker, 'NASD'),
            'OVRS_ORD_UNPR': f'{price:.2f}',
            'ITEM_CD': ticker,
        }, retries=1)
        output = data.get('output', {})
        if not output:
            return -1
        try:
            qty = int(float(output.get('ovrs_max_ord_psbl_qty', 0)))
            if qty > 0:
                return qty
        except (ValueError, TypeError):
            pass
        try:
            amt = float(output.get('frcr_ord_psbl_amt1', 0))
            if amt > 0 and price > 0:
                return int(amt / price)
        except (ValueError, TypeError):
            pass
        try:
            qty = int(float(output.get('ord_psbl_qty', 0)))
            if qty > 0:
                return qty
        except (ValueError, TypeError):
            pass
        try:
            amt = float(output.get('ovrs_ord_psbl_amt', 0))
            if amt > 0 and price > 0:
                return int(amt / price)
        except (ValueError, TypeError):
            pass
        return 0


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
    """V23: 가드 전면 제거 (BT spec과 정합) — 항상 False."""
    guard = state.get('guard_state') or {}
    if guard.get('crash_active'):
        guard['crash_active'] = False
        guard['crash_date'] = None
        guard['crash_cooldown_until'] = None
        state['guard_state'] = guard
        log('  V23: 잔존 crash_active 정리 (가드 spec 제거)')
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


def _migrate_tranches_to_snaps(state: dict) -> None:
    """V21 monthly tranches (1/8/15/22) → V22/V23 snap-based (0/1/2) 1회 변환.
    기존 picks/weights 를 모두 보존 (4 → 3 평균 EW). 첫 snap-based 실행 시만 트리거.
    """
    if 'snapshots' in state:
        return
    old = state.get('tranches', {})
    today = datetime.now().date()
    if not old:
        # 빈 state — snapshots 만 빈 dict 로 init
        state['snapshots'] = {}
        return
    # 4 monthly tranche 의 picks 평균 → 모든 snap 에 동일 picks 인계
    merged = {}
    n_old = len(old)
    for tr in old.values():
        for tk, w in (tr.get('weights', {}) or {}).items():
            merged[tk] = merged.get(tk, 0.0) + w / n_old
    picks = sorted(merged.keys()) if merged else []
    weights = merged
    # snap_id 0/1/2 staggered last_rebal_date: today, today-42d, today-84d
    # → next rebal at today+126/+84/+42 (snap0 가 가장 늦게 rebal)
    snapshots = {}
    for snap_id in range(N_SNAPS):
        last_rebal = today - timedelta(days=snap_id * SNAP_STAGGER_DAYS)
        snapshots[str(snap_id)] = {
            'picks': list(picks),
            'weights': dict(weights),
            'last_rebal_date': last_rebal.isoformat(),
        }
    state['snapshots'] = snapshots
    state['_v22_migrated_from_tranches'] = today.isoformat()
    log(f'  🔁 V23 migration: monthly tranches → 3 snaps (last_rebal staggered today/-42d/-84d)')


def check_anchors(signal: dict, state: dict):
    """V23 snap-based 앵커 체크.
    각 snap 의 (today - last_rebal_date) >= SNAP_PERIOD_DAYS 면 신규 picks 으로 rebal.
    """
    _migrate_tranches_to_snaps(state)
    today = datetime.now().date()
    risk_on = signal.get('stock', {}).get('risk_on', True)
    stock_sig = signal.get('stock', {})

    if risk_on:
        picks = stock_sig.get('offense_picks', [])
        weights = stock_sig.get('offense_weights', {})
    else:
        picks = stock_sig.get('defense_picks', [])
        weights = stock_sig.get('defense_weights', {})

    snapshots = state.setdefault('snapshots', {})
    for snap_id in range(N_SNAPS):
        key = str(snap_id)
        snap = snapshots.setdefault(key, {})
        last_str = snap.get('last_rebal_date')
        if not last_str:
            # 첫 진입: snap_id 만큼 staggered last_rebal_date 부여 (즉시 rebal 트리거)
            snap['last_rebal_date'] = (today - timedelta(days=SNAP_PERIOD_DAYS + snap_id * SNAP_STAGGER_DAYS)).isoformat()
            last_str = snap['last_rebal_date']
        try:
            last_date = datetime.fromisoformat(last_str).date()
        except Exception:
            last_date = today - timedelta(days=SNAP_PERIOD_DAYS + 1)
        elapsed = (today - last_date).days
        if elapsed >= SNAP_PERIOD_DAYS:
            snap['picks'] = list(picks)
            snap['weights'] = dict(weights)
            snap['last_rebal_date'] = today.isoformat()
            state['rebalancing_needed'] = True
            log(f'  📅 V23 snap{snap_id} rebal (elapsed={elapsed}d ≥ {SNAP_PERIOD_DAYS}d): {picks}')


def merge_tranches(state: dict) -> Dict[str, float]:
    """V23 snapshots merge → 최종 target (3 snap EW 평균)."""
    snapshots = state.get('snapshots', {})
    if not snapshots:
        # 하위호환: 옛 tranches 가 남아있다면 그걸로 fallback
        old = state.get('tranches', {})
        if old:
            n = len(old)
            merged = {}
            for tr in old.values():
                for tk, w in (tr.get('weights', {}) or {}).items():
                    merged[tk] = merged.get(tk, 0) + w / n
            return merged
        return {}
    n = len(snapshots) or 1
    merged = {}
    for snap in snapshots.values():
        for ticker, w in (snap.get('weights', {}) or {}).items():
            merged[ticker] = merged.get(ticker, 0) + w / n
    return merged


def execute_delta(target: Dict[str, float], api: KISAPI, state: dict, cash_buffer: float = None):
    """target vs 현재 잔고 → Delta 매매 (정수 주 단위)."""
    if cash_buffer is None:
        cash_buffer = CASH_BUFFER_DEFAULT
    holdings, total_usd, exrt = api.get_balance()
    if total_usd <= 0:
        log('  잔고 없음')
        return {
            'holdings': {},
            'total_usd': 0.0,
            'exchange_rate': exrt,
            'sells': [],
            'buys': [],
            'max_diff': None,
            'completed': False,
        }

    # 현재 비중 계산
    current_values = {}
    for ticker, qty in holdings.items():
        price = api.get_current_price(ticker)
        current_values[ticker] = qty * price if price else 0
        log(f'    현재 {ticker}: ${current_values[ticker]:,.0f} ({(current_values[ticker] / total_usd):.1%})')

    # 매도/매수 리스트
    sells = []
    buys = []
    failed_orders = []
    target_usd = total_usd * (1 - cash_buffer)
    log(f'    target_usd: ${target_usd:,.0f} (buffer={cash_buffer:.2f})')

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
        success = False
        for attempt in range(MAX_ORDER_ATTEMPTS):
            success = api.place_order(ticker, qty, sell_price, 'sell')
            if success:
                break
            time.sleep(ORDER_WAIT_SEC)
        if not success:
            failed_orders.append(('sell', ticker, qty))
            send_telegram(f'⚠️ 주문 실패: sell {ticker} {qty}주')

    if sells:
        time.sleep(15)  # 매도 체결 + 주문가능금액 갱신 대기
        api.cancel_all_pending()

    # 매수 (psamount로 가능 수량 확인 후 실행)
    for ticker, qty, price in buys:
        buy_price = price * (1 + LIMIT_PRICE_SLIP)
        max_qty = api.get_max_buy_qty(ticker, buy_price)
        if max_qty == 0:
            log(f'  ⚠️ {ticker} 매수 불가 (주문가능금액 0) → 다음 실행에서 재시도')
            failed_orders.append(('buy', ticker, qty))
            continue
        actual_qty = min(qty, max_qty) if max_qty > 0 else qty
        if 0 < max_qty < qty:
            log(f'  ⚠️ {ticker} 매수 축소: {qty}→{actual_qty}주 (주문가능금액 제한)')
        log(f'  매수: {ticker} {actual_qty}주 @ ${buy_price:.2f}')
        success = False
        for attempt in range(MAX_ORDER_ATTEMPTS):
            success = api.place_order(ticker, actual_qty, buy_price, 'buy')
            if success:
                break
            time.sleep(ORDER_WAIT_SEC)
        if not success:
            failed_orders.append(('buy', ticker, actual_qty))
            send_telegram(f'⚠️ 주문 실패: buy {ticker} {actual_qty}주')

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
        if max_diff < REBALANCE_TOLERANCE and not failed_orders:
            state['rebalancing_needed'] = False
            log(f'  ✅ 목표 달성 (±{REBALANCE_TOLERANCE:.0%} 이내)')
            send_telegram('✅ 리밸런싱 완료')
        else:
            state['rebalancing_needed'] = True
            log(f'  ⏳ 미달 (max diff={max_diff:.1%}), 다음 실행에서 재시도')
        return {
            'holdings': holdings,
            'total_usd': total_usd,
            'exchange_rate': exrt,
            'sells': sells,
            'buys': buys,
            'failed_orders': failed_orders,
            'max_diff': max_diff,
            'completed': max_diff < REBALANCE_TOLERANCE and not failed_orders,
        }

    return {
        'holdings': holdings,
        'total_usd': total_usd,
        'exchange_rate': exrt,
        'sells': sells,
        'buys': buys,
        'failed_orders': failed_orders,
        'max_diff': None,
        'completed': False,
    }


# ═══ run_once ═══

def run_once(dry_run=False):
    global RUN_ID
    RUN_ID = uuid.uuid4().hex
    _run_id_ref[0] = RUN_ID
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
    # cash_buffer: state에서 읽기, 없으면 기본값 2%
    cash_buffer = state.get('cash_buffer', CASH_BUFFER_DEFAULT)
    log(f'  cash_buffer: {cash_buffer:.0%}')

    # V23: snapshots 또는 V21 tranches 둘 중 하나라도 없으면 첫 실행 — 마이그레이션이 자동 처리.
    # 단, 둘 다 없으면 신호도 없는 상태라 스킵.
    if not state.get('tranches') and not state.get('snapshots'):
        log('  kis_trade_state.json 트랜치/스냅 모두 없음 — 첫 실행, anchor 체크가 init')

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

    # 2. V23: 가드 없음. 잔존 crash_active 만 정리.
    check_crash(signal, api, state)

    if not is_fresh:
        if not dry_run:
            save_json(TRADE_STATE_FILE, state)
        log('주식 executor 완료 (stale signal)')
        return

    # 3. 카나리 플립
    check_canary_flip(signal, state)

    # 4. 앵커 체크
    check_anchors(signal, state)

    # 5. Merge + Delta 매매
    if state.get('rebalancing_needed', False):
        target = merge_tranches(state)
        log(f'  Target: {target}')
        mode = 'DRY-RUN' if dry_run else 'LIVE'
        summary_lines = [f'📊 리밸런싱 시작 ({mode})']
        summary_lines.append(f"Risk: {'ON' if signal.get('stock', {}).get('risk_on', True) else 'OFF'}")
        summary_lines.append(
            '목표: ' + ', '.join(f'{k}:{v:.1%}' for k, v in target.items())
        )
        # 디버그: 현재 잔고 상세
        _bal = api.get_balance() if hasattr(api, 'get_balance') else {}
        log(f'  Balance: {_bal}')
        holdings = _bal[0] if isinstance(_bal, tuple) and len(_bal) >= 1 else {}
        if holdings:
            summary_lines.append(
                '현재 보유: ' + ', '.join(f'{k}:{v}주' for k, v in holdings.items())
            )
        else:
            summary_lines.append('현재 보유: 없음')
        send_telegram('\n'.join(summary_lines))
        result = execute_delta(target, api, state, cash_buffer=cash_buffer)
        sells = result.get('sells', [])
        buys = result.get('buys', [])
        max_diff = result.get('max_diff')
        failed_orders = result.get('failed_orders', [])
        finish_lines = ['✅ 리밸런싱 완료' if result.get('completed') else '⏳ 리밸런싱 미완료']
        if not sells and not buys:
            finish_lines.append('주문 계산: 없음')
        else:
            if sells:
                finish_lines.append('매도 계획: ' + ', '.join(f'{t} {q}주' for t, q, _ in sells))
            if buys:
                finish_lines.append('매수 계획: ' + ', '.join(f'{t} {q}주' for t, q, _ in buys))
        if failed_orders:
            finish_lines.append(
                '실패 주문: ' + ', '.join(f'{side} {ticker} {qty}주' for side, ticker, qty in failed_orders)
            )
        if max_diff is not None:
            finish_lines.append(f'잔여 편차: {max_diff:.1%}')
        send_telegram('\n'.join(finish_lines))

    # 6. 저장
    state['last_action'] = 'executor'
    state['last_trade_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    if not dry_run:
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
