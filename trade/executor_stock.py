#!/usr/bin/env python3
"""주식 Executor — 단일 모드 run_once().

설계 원칙:
  - signal_state.json 읽기 (recommend 출력)
  - kis_trade_state.json 읽기/쓰기
  - 전략 신호 계산 안 함
  - signal_state 수정 안 함

이벤트 우선순위 (V24, 가드 없음):
  1. 카나리 플립 → 전 트랜치 즉시 전환
  2. 앵커 체크 (해당 snap 에 offense/defense picks 반영)
  3. Delta 매매

Usage:
  python3 executor_stock.py               # 실행
  python3 executor_stock.py --dry-run     # 주문 없이 로그만
"""

import json, os, sys, time, argparse, logging, uuid, math, fcntl
from datetime import datetime, timedelta
from typing import Dict, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import requests

from common.io import load_json, save_json
from common.notify import send_telegram as _send_tg
from common.logging_utils import setup_file_logger, make_log_fn
from common.health_guard import HealthGuard, UnknownExecutionError

# V25: strategy 직접 계산 (signal_state.json 의존 제거)
_strat_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           '..', 'strategies', 'cap_defend')
sys.path.insert(0, _strat_path)
try:
    import stock_strategy_v25 as stock_strat  # type: ignore
except Exception:
    stock_strat = None  # fallback to signal_state.json

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
ALLOC_TRANSIT_STATE_FILE = 'trade_state.json'  # 코인 state — alloc_transit flag 공유
ALLOC_STOCK_RATIO = 0.60
LOG_FILE = 'executor_stock.log'


CAP_RATIO_FLOOR = 0.10  # cap_ratio < floor 면 거래 중단 fallback (1.0 으로 처리)
ALLOC_TRANSIT_STALE_HOURS = 26  # state mtime > 26h 면 stale → cap 적용 안 함
CAP_DEFEND_MIN_EXCESS = 0.01  # cap_ratio < 0.99 (= cap_excess ≥ 1%) 면 cap_defend 매도 발동


def _validate_cap_ratio(val, sleeve_name: str):
    """cap_ratio 검증: 0 < cr ≤ 1, finite 보장. invalid 면 1.0 fallback + ERROR 로그."""
    try:
        cr = float(val)
    except Exception:
        log(f'  🚨 alloc_transit cap_ratio[{sleeve_name}] parse 실패 ({val!r}) → fallback 1.0')
        return 1.0
    if not math.isfinite(cr) or cr <= 0:
        log(f'  🚨 alloc_transit cap_ratio[{sleeve_name}]={cr} invalid (≤0/NaN/inf) → fallback 1.0')
        return 1.0
    if cr < CAP_RATIO_FLOOR:
        log(f'  🚨 alloc_transit cap_ratio[{sleeve_name}]={cr:.4f} < floor {CAP_RATIO_FLOOR} → SKIP (fallback 1.0)')
        return 1.0
    if cr > 1.0:
        return 1.0
    return cr


def _read_alloc_transit_cap_ratio(sleeve: str = 'stock'):
    """trade_state.json 의 alloc_transit active 면 sleeve cap_ratio (≤1.0) 반환. 아니면 None.

    cap_ratio = min(1.0, target_sleeve_krw / current_sleeve_krw)
    effective_pv_local = actual_pv_local × cap_ratio (cap_ratio < 1.0 일 때만 의미)

    schema 손상 / parse 실패 / cap_ratio invalid → fallback 1.0 (cap 없음과 동일) + ERROR 로그.
    """
    for _p in (
        os.path.expanduser('~/trade_state.json'),
        ALLOC_TRANSIT_STATE_FILE,
        os.path.join(os.getcwd(), ALLOC_TRANSIT_STATE_FILE),
    ):
        try:
            if not os.path.exists(_p):
                continue
            mtime = os.path.getmtime(_p)
            with open(_p, 'r') as f:
                obj = json.load(f)
            at = obj.get('alloc_transit')
            if not at or not at.get('active'):
                return None
            cr_raw = (at.get('cap_ratio') or {}).get(sleeve)
            if cr_raw is None:
                log(f'  🚨 alloc_transit active 하나 cap_ratio[{sleeve}] missing → fallback 1.0')
                return 1.0
            cr = _validate_cap_ratio(cr_raw, sleeve)
            age_h = (time.time() - mtime) / 3600
            if age_h > ALLOC_TRANSIT_STALE_HOURS:
                log(f'  🚨 alloc_transit state stale (age {age_h:.1f}h > {ALLOC_TRANSIT_STALE_HOURS}h) → cap 무시 (fallback 없음)')
                return None
            log(f'  alloc_transit cap_ratio[{sleeve}]={cr:.4f} (state mtime age {age_h:.1f}h)')
            return cr
        except json.JSONDecodeError as ex:
            log(f'  🚨 alloc_transit JSON parse 실패 ({_p}): {ex} → fallback (cap 없음)')
            return None
        except Exception as ex:
            log(f'  ⚠️ alloc_transit read 실패 ({_p}): {ex}')
            continue
    return None

# V24 (2026-04-30): snap-based 3-tranche stagger (snap=69, stagger=23)
# - snap_id 0/1/2, 각 69일 주기로 rebal, stagger = 23일 (3*23, prime stagger)
# - V22(126/42) → V24(69/23): rank-sum BT 검증 결과 반영 (Cal 1.16 ymin +0.59)
SNAP_PERIOD_DAYS = 69
N_SNAPS = 3
SNAP_STAGGER_DAYS = 23  # SNAP_PERIOD_DAYS / N_SNAPS
CASH_BUFFER_DEFAULT = 0.07  # V24 (2026-05-26): stock 계좌 안에 7% cash buffer (자산간 자동 rebal 제거, alloc 60/25/15)
MAX_ORDER_ATTEMPTS = 5
ORDER_WAIT_SEC = 5
LIMIT_PRICE_SLIP = 0.003   # ±0.3%
REBALANCE_TOLERANCE = 0.01  # 목표 달성 판정 허용 오차 ±1%
# V25 (2026-06-06): 드리프트 5pp 발화 시 그날 fresh 3-mom selection 으로 snap picks 리필.
# 채택 BT(bt_stock_mom3)와 정합 — 모멘텀 탈락 종목을 앵커(69일)까지 안 기다리고 발화일 교체.
# 비용 스트레스 1x/3x/5x 모두 anchor-only 대비 우위 (Cal 1.17 vs 1.05, 턴오버 오히려 낮음).
# 롤백: False 로 두면 anchor-only(종목 교체 앵커일에만) 로 복귀.
STOCK_DRIFT_REFILL = True

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


_DRY_RUN_SILENT = [False]  # mutable flag — run_once 에서 dry_run 시 True 로 토글


def send_telegram(msg: str):
    if _DRY_RUN_SILENT[0]:
        log(f'  [DRY] telegram silent: {msg}')
        return
    _send_tg(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, msg, prefix='주식')


# ═══ KIS API (기존 auto_trade_kis.py에서 가져옴) ═══

def _read_token_cache():
    """캐시에 살아 있는 토큰이 있으면 그 문자열, 없거나 깨졌으면 None.

    깨진 파일을 예외로 올리지 않는다 — 쓰다 만 캐시 하나가 그 뒤의 모든 회차를
    통째로 막으면 안 된다. 못 읽으면 없는 것으로 보고 새로 받는다.
    """
    try:
        with open(TOKEN_FILE) as f:
            cached = json.load(f)
        token = cached['token']
        expires = datetime.fromisoformat(cached['expires'])
    except (OSError, ValueError, KeyError, TypeError):
        return None
    return token if (token and datetime.now() < expires) else None


def _get_token() -> str:
    """OAuth 토큰 (캐시 사용).

    이 파일은 집행기와 밤 조회 감시가 같이 쓴다. 그래서 셋을 지킨다 —
    ⑴ 잠금 안에서 캐시를 다시 본다(앞사람이 방금 받아 뒀으면 또 안 받는다),
    ⑵ 임시파일에 쓰고 rename 한다(쓰다 만 파일이 안 남는다),
    ⑶ 깨진 캐시는 없는 것으로 보고 다시 받는다.
    잠금을 못 잡는 환경이면 잠금 없이 같은 순서로 진행한다 — 토큰을 한 번 더
    받는 것이 회차를 못 도는 것보다 낫다.
    """
    token = _read_token_cache()
    if token:
        return token

    lock_fd = None
    try:
        lock_fd = os.open(TOKEN_FILE + '.lock', os.O_CREAT | os.O_RDWR, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)
    except OSError:
        if lock_fd is not None:
            os.close(lock_fd)
        lock_fd = None

    try:
        # 잠금을 기다리는 동안 다른 쪽이 받아 뒀을 수 있다.
        token = _read_token_cache()
        if token:
            return token

        resp = requests.post(f'{BASE_URL}/oauth2/tokenP', json={
            'grant_type': 'client_credentials',
            'appkey': KIS_APP_KEY, 'appsecret': KIS_APP_SECRET,
        }, timeout=10)
        data = resp.json()
        token = data['access_token']
        expires = datetime.now() + timedelta(hours=23)
        tmp = f'{TOKEN_FILE}.tmp{os.getpid()}'
        with open(tmp, 'w') as f:
            json.dump({'token': token, 'expires': expires.isoformat()}, f)
        os.replace(tmp, TOKEN_FILE)
        return token
    finally:
        if lock_fd is not None:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)


def _headers(tr_id: str) -> dict:
    return {
        'authorization': f'Bearer {_get_token()}',
        'appkey': KIS_APP_KEY, 'appsecret': KIS_APP_SECRET,
        'tr_id': tr_id, 'Content-Type': 'application/json; charset=UTF-8',
    }


def _get(path: str, tr_id: str, params: dict, retries=3) -> dict:
    """성공({'rt_cd': '0', ...}) 또는 {} 를 돌려준다. 실패 사유는 로그에 남긴다.

    빈 dict 는 "조회가 안 됐다"는 뜻이지 "값이 0"이라는 뜻이 아니다. 둘을 같게 읽으면
    현금이 사라진 채로 매매가 돈다(2026-10-12·10-21 헛매매).
    """
    last = ''
    for i in range(retries):
        try:
            resp = requests.get(f'{BASE_URL}{path}', headers=_headers(tr_id), params=params, timeout=15)
            data = resp.json()
            if data.get('rt_cd') == '0':
                return data
            last = (f"rt_cd={data.get('rt_cd')} msg_cd={data.get('msg_cd')} "
                    f"msg={(data.get('msg1') or '').strip()[:80]}")
        except Exception as e:
            last = f'{type(e).__name__}: {str(e)[:80]}'
        if i < retries - 1:
            time.sleep(1)
    log(f'  ⚠️ 조회 실패 {tr_id} ({retries}회): {last}')
    return {}


class BalanceUnavailable(RuntimeError):
    """계좌 조회가 불완전하다. 없는 값을 0 으로 읽지 않고 그 회차를 멈추기 위한 신호."""


def _finite(v) -> float:
    """숫자로 못 읽거나 유한하지 않으면 nan. 정상적인 0 과 결측을 섞지 않으려는 것."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return float('nan')
    return f if math.isfinite(f) else float('nan')


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
        # 마지막으로 성공한 계좌 조회의 자산 구성(원화). 로그·보고용이고 매매 판단에 안 쓴다.
        self._last_asset: Dict[str, float] = {}

    def get_balance(self) -> Tuple[Dict[str, int], float, float]:
        """잔고 → (holdings {ticker: qty}, total_usd, exchange_rate).

        총자산은 투자계좌자산현황(CTRP6548R)의 tot_asst_amt 다 — 앱의 "계좌 총자산"과
        같은 값이고, 해외주식 평가 + 원화 예수금 + 외화 예수금 + 외화RP 를 모두 담는다.
        보유평가(해외주식 잔고)와 예수금을 따로 더해 만들지 않는다. 그렇게 만들면
        미결제 구간에서 한쪽은 아직 주식에 없고 한쪽은 이미 현금에서 빠져 총자산이
        미결제 매수대금만큼 작아진다(2026-10-23 실측 <금액>원, 계좌의 5.9%).

        조회가 하나라도 비면 BalanceUnavailable 을 올린다. 없는 값을 0 으로 읽으면
        현금이 사라져 드리프트가 현금버퍼 크기만큼 발화하고 멀쩡한 주식을 판다
        (2026-10-12·10-21, 체결액 약 <금액>).
        """
        data = _get('/uapi/overseas-stock/v1/trading/inquire-balance', 'TTTS3012R', {
            'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
            'AFHR_FLPR_YN': 'N', 'OFL_YN': '', 'INQR_DVSN': '01',
            'UNPR_DVSN': '01', 'FUND_STTL_ICLD_YN': 'N',
            'FNCG_AMT_AUTO_RDPT_YN': 'N', 'OVRS_EXCG_CD': 'NASD', 'TR_CRCY_CD': 'USD',
            'CTX_AREA_FK200': '', 'CTX_AREA_NK200': '',
        })
        if data.get('rt_cd') != '0':
            raise BalanceUnavailable('해외주식 잔고(TTTS3012R) 조회 실패')
        # 보유 목록이 통째로 빠진 응답을 "들고 있는 게 없다" 로 읽으면 안 된다 — 그러면
        # 이미 들고 있는 종목을 또 산다. 빈 계좌는 output1 이 빈 목록([])으로 온다.
        _o1 = data.get('output1')
        if not isinstance(_o1, list):
            raise BalanceUnavailable(
                f'해외주식 잔고(TTTS3012R) 보유 목록(output1)이 없다 (type={type(_o1).__name__})')
        holdings = {}
        holdings_usd = 0.0
        for item in _o1:
            if not isinstance(item, dict):
                raise BalanceUnavailable(f'보유 행이 딕셔너리가 아니다 ({type(item).__name__})')
            ticker = str(item.get('ovrs_pdno') or '').strip()
            raw_qty = item.get('ovrs_cblc_qty')
            # 종목이나 수량이 빠진 행은 0 으로 메우지 않는다. 그 한 줄이 비면 그 종목을
            # 안 들고 있는 것처럼 보여서 같은 종목을 또 사게 된다.
            if not ticker:
                raise BalanceUnavailable('보유 행에 종목(ovrs_pdno)이 없다')
            qty_f = _finite(raw_qty)
            if math.isnan(qty_f):
                raise BalanceUnavailable(f'{ticker} 보유 수량(ovrs_cblc_qty)을 숫자로 못 읽었다 ({raw_qty!r})')
            qty = int(qty_f)
            eval_amt = _finite(item.get('ovrs_stck_evlu_amt'))
            if math.isnan(eval_amt):
                eval_amt = 0.0   # 보고용 합계에만 쓴다 — 매매 판단에 안 들어간다
            if qty > 0 and ticker:
                holdings[ticker] = qty
                holdings_usd += eval_amt
                try:
                    cur_price = float(item.get('now_pric2', 0) or item.get('ovrs_now_pric1', 0) or 0)
                    if cur_price > 0:
                        self._balance_price_cache[ticker] = cur_price
                except Exception:
                    pass

        # 체결기준현재잔고 — 환율과 예수금·미결제의 출처. 비면 그 회차를 멈춘다.
        bp_data = _get('/uapi/overseas-stock/v1/trading/inquire-present-balance', 'CTRP6504R', {
            'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
            'WCRC_FRCR_DVSN_CD': '02', 'NATN_CD': '840',
            'TR_MKET_CD': '00', 'INQR_DVSN_CD': '00',
        })
        if bp_data.get('rt_cd') != '0':
            raise BalanceUnavailable('체결기준현재잔고(CTRP6504R) 조회 실패')

        # 환율 — 달러 행에서 먼저 찾고, 없으면 해외증거금에서 찾는다. 한쪽 실패로
        # 다른 쪽의 멀쩡한 값을 버리지 않는다. 둘 다 없으면 멈춘다.
        exrt = float('nan')
        for item in bp_data.get('output2') or []:
            if isinstance(item, dict) and item.get('crcy_cd') == 'USD':
                exrt = _finite(item.get('frst_bltn_exrt'))
                break
        if not (exrt > 0):
            fm_data = _get('/uapi/overseas-stock/v1/trading/foreign-margin', 'TTTC2101R', {
                'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
                'OVRS_EXCG_CD': 'NASD', 'CRCY_CD': 'USD',
            })
            _fm_out = fm_data.get('output')
            _fm_rows = _fm_out if isinstance(_fm_out, list) else [_fm_out] if isinstance(_fm_out, dict) else []
            for item in _fm_rows:
                if isinstance(item, dict) and item.get('crcy_cd') == 'USD':
                    exrt = _finite(item.get('bass_exrt'))
                    if exrt > 0:
                        break
        if not (exrt > 0):
            raise BalanceUnavailable('환율을 체결기준현재잔고·해외증거금 어느 쪽에서도 못 받았다')

        # 예수금·미결제 — 보고용. 칸이 통째로 없으면 조회가 빈 것이므로 멈춘다.
        _o3 = bp_data.get('output3')
        if not isinstance(_o3, dict) or 'tot_dncl_amt' not in _o3 or 'frcr_evlu_tota' not in _o3:
            raise BalanceUnavailable('예수금 칸(CTRP6504R output3)이 없다')
        _krw_cash = _finite(_o3.get('tot_dncl_amt'))
        _frcr_cash = _finite(_o3.get('frcr_evlu_tota'))
        _ustl_sll = _finite(_o3.get('ustl_sll_amt_smtl'))
        _ustl_buy = _finite(_o3.get('ustl_buy_amt_smtl'))
        if math.isnan(_krw_cash) or math.isnan(_frcr_cash):
            raise BalanceUnavailable('예수금 값을 숫자로 못 읽었다')

        # 계좌 총자산 — 앱의 "계좌 총자산"과 같은 값. 외화RP 스윕분도 이 칸에만 들어온다.
        _ab = _get('/uapi/domestic-stock/v1/trading/inquire-account-balance', 'CTRP6548R', {
            'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
            'INQR_DVSN_1': '', 'BSPR_BF_DT_APLY_YN': '',
        })
        if _ab.get('rt_cd') != '0':
            raise BalanceUnavailable('투자계좌자산현황(CTRP6548R) 조회 실패')
        _o2 = _ab.get('output2', {})
        if isinstance(_o2, list):
            _o2 = _o2[0] if _o2 else {}
        if not isinstance(_o2, dict):
            raise BalanceUnavailable('투자계좌자산현황 output2 가 없다')
        _tot_krw = _finite(_o2.get('tot_asst_amt'))
        if not (_tot_krw > 0):
            raise BalanceUnavailable('계좌 총자산(tot_asst_amt)을 못 읽었다')
        _stock_krw = _finite(_o2.get('ovrs_stck_evlu_amt1'))
        total_usd = _tot_krw / exrt

        # 외화RP 등 기타 = 총자산 − 주식 − 예수금. 이름 있는 칸으로만 도출한다.
        _rp_krw = _tot_krw - _stock_krw - _frcr_cash - _krw_cash if not math.isnan(_stock_krw) else float('nan')
        self._last_asset = {
            'exrt': exrt,
            'total_krw': _tot_krw,
            'settle_stock_krw': _stock_krw,
            'krw_cash_krw': _krw_cash,
            'frcr_cash_krw': _frcr_cash,
            'rp_krw': _rp_krw,
            'ustl_sll_krw': _ustl_sll,
            'ustl_buy_krw': _ustl_buy,
            'live_stock_usd': holdings_usd,
        }
        log(f'  계좌 총자산 {_tot_krw:,.0f}원 = 주식 {_stock_krw:,.0f} + 원화예수금 {_krw_cash:,.0f}'
            f' + 외화예수금 {_frcr_cash:,.0f} + 기타 {_rp_krw:,.0f} (환율 {exrt:,.2f}, ${total_usd:,.0f})')

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

        # 3차: KIS 기간별시세 종가
        # (잔고 재조회 단계는 뺐다 — 주문 루프 도중 계좌 조회가 실패하면 그 자리에서
        #  멈춰 이미 낸 주문의 뒤처리를 못 하고, 성공해도 평가 기준이 루프 중간에 바뀐다.)
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

    def cancel_all_pending(self) -> dict:
        """미체결 전량 취소. 결과를 돌려준다 — {'queried', 'found', 'failed'}.

        조회가 안 된 것을 빈 목록으로 읽으면 "미체결 없음"과 구분이 안 된다. 그 상태로
        사전 잔고를 잡으면 뒤늦게 체결되는 잔존 주문이 그 값을 거짓으로 만들고, 낡은
        수량으로 한 번 더 주문이 나간다. 부르는 쪽이 "확인됨"과 "모름"을 가르게 한다.
        """
        data = _get('/uapi/overseas-stock/v1/trading/inquire-nccs', 'TTTS3018R', {
            'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
            'OVRS_EXCG_CD': 'NASD', 'SORT_SQN': 'DS',
            'CTX_AREA_FK200': '', 'CTX_AREA_NK200': '',
        })
        if not data:
            log('  ⚠️ 미체결 조회 실패 — 미체결 없음과 구분할 수 없다')
            return {'queried': False, 'found': 0, 'failed': 0}
        found = 0
        failed = 0
        for item in data.get('output', []):
            order_no = item.get('odno', '')
            ticker = item.get('pdno', '')
            qty = int(float(item.get('nccs_qty', 0)))
            side = 'buy' if item.get('sll_buy_dvsn_cd') == '02' else 'sell'
            if order_no and qty > 0:
                found += 1
                log(f'  미체결 취소: {ticker} {side} {qty}주')
                if not self.dry_run:
                    res = _post('/uapi/overseas-stock/v1/trading/order-rvsecncl', 'TTTT1004U', {
                        'CANO': KIS_ACCOUNT, 'ACNT_PRDT_CD': KIS_ACCOUNT_PROD,
                        'OVRS_EXCG_CD': EXCHANGE_MAP.get(ticker, 'NASD'),
                        'PDNO': ticker, 'ORGN_ODNO': order_no,
                        'RVSE_CNCL_DVSN_CD': '02',
                        'ORD_QTY': str(qty), 'OVRS_ORD_UNPR': '0',
                        'CTAC_TLNO': '', 'MGCO_APTM_ODNO': '',
                        'ORD_SVR_DVSN_CD': '0',
                    })
                    if not res:
                        failed += 1
                        log(f'  ⚠️ 미체결 취소 실패: {ticker} {side} {qty}주')
        return {'queried': True, 'found': found, 'failed': failed}

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
    """V24: 가드 전면 제거 (BT spec과 정합) — 항상 False."""
    guard = state.get('guard_state') or {}
    if guard.get('crash_active'):
        guard['crash_active'] = False
        guard['crash_date'] = None
        guard['crash_cooldown_until'] = None
        state['guard_state'] = guard
        log('  V24: 잔존 crash_active 정리 (가드 spec 제거)')
    return False


def check_canary_flip(signal: dict, state: dict) -> bool:
    """카나리 플립 → 전 snapshot 즉시 전환 (offense ↔ defense).

    카나리 OFF 시 앵커(69일) 안 기다리고 그날 즉시 방어자산으로 교체 (전량 매도→방어).
    버그 수정(2026-06-06): 기존엔 구 state['tranches'](V24 마이그레이션 후 비어 있음)에
    써서 snapshot picks 가 안 바뀌어 방어 전환이 앵커까지 최대 69일 지연됐음.
    refill_snaps_fresh 로 모든 snapshot picks 를 새 레짐으로 교체. 앵커 시간표 보존
    (앵커 cadence 유지 = 채택 BT 정합).
    """
    risk_on = signal.get('stock', {}).get('risk_on', True)
    prev = state.get('prev_risk_on')

    if prev is not None and risk_on != prev:
        log(f'  🔄 카나리 플립: {prev} → {risk_on}')
        _migrate_tranches_to_snaps(state)
        if not _picks_weights_ok(*_signal_payload(signal)):
            # 비중이 온전치 않으면 교체를 못 한다. 이때 prev_risk_on 을 넘겨버리면 내일은
            # 플립이 아니게 되어 방어 전환이 영영 사라진다 — 플립을 소비하지 않고 남긴다.
            log('  ⚠️ 카나리 플립인데 신호가 온전치 않다 — 플립을 소비하지 않고 다음 실행에서 재시도')
            send_telegram('⚠️ 주식 카나리 전환 보류: 신호의 종목·비중이 온전하지 않다. '
                          '다음 실행에서 다시 시도한다.')
            return False
        swapped = refill_snaps_fresh(signal, state)
        state['prev_risk_on'] = risk_on
        state['rebalancing_needed'] = True
        log(f'카나리 플립 (silent): {"Risk-On 공격" if risk_on else "Risk-Off 방어"} swapped={swapped} → {merge_tranches(state)}')
        return True

    state['prev_risk_on'] = risk_on
    return False


def _migrate_tranches_to_snaps(state: dict) -> None:
    """V21 monthly tranches (1/8/15/22) → V22/V24 snap-based (0/1/2) 1회 변환.
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
    # 종목은 오늘 인계받은 것이니 보유기간 69일을 오늘부터 센다. 23일씩 벌려 둔다.
    snapshots = {}
    for snap_id in range(N_SNAPS):
        snapshots[str(snap_id)] = {
            'picks': list(picks),
            'weights': dict(weights),
            'last_rebal_date': today.isoformat(),
            'next_rebal_date': (today + timedelta(
                days=SNAP_PERIOD_DAYS + snap_id * SNAP_STAGGER_DAYS)).isoformat(),
        }
    state['snapshots'] = snapshots
    state['_v22_migrated_from_tranches'] = today.isoformat()
    log(f'  🔁 V24 migration: monthly tranches → 3 snaps '
        f'(next_rebal +{SNAP_PERIOD_DAYS}/+{SNAP_PERIOD_DAYS + SNAP_STAGGER_DAYS}/'
        f'+{SNAP_PERIOD_DAYS + 2 * SNAP_STAGGER_DAYS}d)')


def _parse_date(raw):
    """isoformat 날짜 문자열 → date. 못 읽으면 None."""
    if not raw:
        return None
    try:
        return datetime.fromisoformat(str(raw)).date()
    except Exception:
        return None


def _signal_payload(signal: dict):
    """지금 레짐에 해당하는 (종목, 비중). 공격이면 공격 쪽, 방어면 방어 쪽."""
    stock_sig = signal.get('stock', {}) or {}
    key = 'offense' if stock_sig.get('risk_on', True) else 'defense'
    return stock_sig.get(f'{key}_picks'), stock_sig.get(f'{key}_weights') or {}


def _weights_ok(w) -> bool:
    """스냅샷·신호의 비중이 쓸 수 있는 값인가.

    비중은 항상 위험자산 + Cash 로 합이 1 이다(compute_offense 의 cash_slot).
    합이 모자란 비중으로 발화하면 그 모자란 만큼이 현금 목표가 되어 멀쩡한 보유를 판다.
    종목 목록만 있고 비중이 비어 있는 신호도 여기서 걸린다.
    """
    if not isinstance(w, dict) or not w:
        return False
    total = 0.0
    for v in w.values():
        try:
            f = float(v)
        except (TypeError, ValueError):
            return False
        if f != f or f in (float('inf'), float('-inf')) or f < 0:
            return False
        total += f
    return 0.99 <= total <= 1.01


def _as_float(v) -> float:
    """숫자로 못 읽으면 0. 결함 판정 안에서 예외로 죽지 않으려고 쓴다."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return 0.0
    return f if f == f else 0.0


def state_structure_defect(state: dict):
    """상태의 모양 자체가 틀린 경우를 찾는다. 결함이면 사유 문자열, 아니면 None.

    비중 내용이 아니라 자료형과 키 집합만 본다. 이건 매매 판단보다 먼저 봐야 한다 —
    snapshots 가 dict 가 아니거나 snap 이 dict 가 아니면 카나리 검사부터 예외로 죽어서
    알림도 못 내보낸다. 키가 남아도는 것도 결함이다. merge_tranches 는 snapshots 개수로
    나누므로 쓰레기 키 하나가 합산 목표를 4분의 3으로 줄이고 그 차이가 매도 주문이 된다.
    (2026-08-28 4차 검토 지적.)
    """
    if 'snapshots' not in state:
        return None
    snapshots = state.get('snapshots')
    if snapshots is None or snapshots == {}:
        return None
    if not isinstance(snapshots, dict):
        return f'snapshots 가 dict 가 아니다({type(snapshots).__name__})'
    want = {str(i) for i in range(N_SNAPS)}
    extra = sorted(set(snapshots) - want)
    if extra:
        return f'스냅샷 키가 남아돈다: {", ".join(extra)} (있어야 할 키는 0~{N_SNAPS - 1})'
    bad_type = sorted(k for k, v in snapshots.items() if not isinstance(v, dict))
    if bad_type:
        return f'스냅샷 {", ".join(bad_type)} 가 dict 가 아니다'
    return None


def _picks_weights_ok(picks, weights) -> bool:
    """신호 한 쌍이 서로 맞는가. 상태에 쓰기 전 세 곳(앵커·전환·리필)에서 같이 쓴다.

    비중 합만 보면 종목과 비중이 서로 다른 신호가 그대로 통과한다 —
    picks=['QQQ'] / weights={'SPY': 1.0} 이면 SPY 를 사면서 QQQ 를 들고 있다고 적는다.
    그 어긋난 picks 는 다음 실행의 교체 판정 기준이 되므로 오염이 남는다.
    (2026-08-28 5차 검토 지적.) 전량 현금(weights={'Cash': 1.0}, picks=[])은 통과한다.
    """
    if not _weights_ok(weights):
        return False
    if not isinstance(picks, list) or not all(isinstance(t, str) for t in picks):
        return False
    held = {t for t, w in weights.items() if t != 'Cash' and _as_float(w) > 0}
    return not (held - set(picks))


def _schedule_phase_ok(dates) -> bool:
    """세 예정일이 23일씩 벌어져 있는가.

    건강한 시간표는 어느 시점에나 {p, p+23, p+46} 형태다 — 방금 발화해 69일 건너뛴
    snap 이 섞여 있어도 날짜순으로 세우면 간격이 23일씩이다. 그래서 정렬 후 인접 간격만
    보면 된다. 같은 날이 둘이면(사람이 손댔거나 옛 코드가 겹쳐놨거나) 간격 0 으로 걸리고,
    한 snap 만 69일 배수만큼 어긋난 경우도 간격 46 으로 걸린다.
    """
    srt = sorted(dates)
    return all((srt[i + 1] - srt[i]).days == SNAP_STAGGER_DAYS for i in range(len(srt) - 1))


def _resolve_from_dates(parsed: dict, today):
    """남아 있는 예정일에서 온전한 시간표를 되찾는다. (날짜 dict, 손댄 snap 목록).

    시간표는 돌아간다. 발화한 snap 은 69일을 건너뛰므로 dt_i = 기준 + i*23 + (회차)*69 다.
    그래서 "예정일 - 번호*23" 을 그냥 모아 다수결하면 회차가 다른 표가 갈라진다.
    (2026-08-28 3차 검토 반례: s0=11-23, s1=12-16, s2=10-31 은 멀쩡한 시간표인데
    s2 의 기준점만 69일 어긋나 보인다.)

    그래서 두 단계로 나눈다.
      1) 위상은 69일 나머지로 본다 — 회차가 달라도 나머지는 같다. 다수결로 위상을 정하면
         회전 때문에 표가 갈리는 일이 없다.
      2) 회차는 따로 고른다. snap 마다 오늘 이후 첫 해당 위상일을 잡고 ±69일 안에서
         조합을 훑어, 셋의 폭이 46일(=23×2)이 되는 배치를 고른다. 그 중에서
         (a) 저장된 날짜를 그대로 쓰는 게 많은 것, (b) 새로 채운 날이 오늘 이후인 것,
         (c) 가장 이른 예정일이 오늘+23일을 안 넘는 것(돌아가는 시간표의 성질),
         (d) 더 이른 것 순으로 고른다.

    저장된 날짜가 이미 온전하면 (a) 때문에 하나도 안 바뀐다. 실행이 며칠 밀려 예정일이
    과거로 내려간 경우도 (a) 가 지켜줘서 그대로 발화한다.
    """
    votes = {}
    for i, d in parsed.items():
        r = (d.toordinal() - i * SNAP_STAGGER_DAYS) % SNAP_PERIOD_DAYS
        votes.setdefault(r, []).append(i)
    r = max(votes, key=lambda k: (len(votes[k]), -min(votes[k])))

    base = {}
    for i in range(N_SNAPS):
        want = (r + i * SNAP_STAGGER_DAYS) % SNAP_PERIOD_DAYS
        first = today.toordinal() + 1
        base[i] = today + timedelta(days=1 + (want - first) % SNAP_PERIOD_DAYS)

    span = SNAP_STAGGER_DAYS * (N_SNAPS - 1)
    limit = today + timedelta(days=SNAP_STAGGER_DAYS)
    best = None
    cycles = [(-1,), (0,), (1,)]
    for _ in range(N_SNAPS - 1):
        cycles = [c + (k,) for c in cycles for k in (-1, 0, 1)]
    for ks in cycles:
        cand = {i: base[i] + timedelta(days=k * SNAP_PERIOD_DAYS) for i, k in enumerate(ks)}
        vals = sorted(cand.values())
        if (vals[-1] - vals[0]).days != span:
            continue
        kept = [i for i in cand if parsed.get(i) == cand[i]]
        filled = [i for i in range(N_SNAPS) if i not in kept]
        key = (-len(kept),
               0 if all(cand[i] > today for i in filled) else 1,
               0 if vals[0] <= limit else 1,
               vals[0])
        if best is None or key < best[0]:
            best = (key, cand, filled)
    if best is None:
        # k=(0,..,0) 은 항상 폭 46일이라 여기 올 수 없다. 그래도 남겨둔다.
        return dict(base), [i for i in range(N_SNAPS) if parsed.get(i) != base[i]]
    return best[1], best[2]


def _resolve_anchor_schedule(snapshots: dict, today):
    """세 snap 의 다음 앵커일을 확정한다. (날짜 리스트, 복구사유 or None) 반환.

    우선순위:
      1. 셋 다 성하고 위상(23일 간격)이 맞으면 그대로 쓴다. 평상시 경로다.
      2. 하나라도 성한 예정일이 있으면 그걸로 위상을 되찾는다(_resolve_from_dates).
         돌고 있는 위상을 살리는 게 새로 심는 것보다 낫다. 멀쩡한 날짜는 안 건드린다.
      3. 예정일이 하나도 없으면 각 snap 의 마지막 발화일 + 69일로 심는다. 이미 들고 있는
         종목의 보유기간을 존중한다 — 배포한 날을 전략 위상으로 삼지 않는다.
         발화일들이 겹쳐 있어(2026-07-08 사고 상태) 간격이 안 나오면 가장 늦은 발화일
         + 69일을 기준으로 23일씩 벌린다.
      4. 그것도 없으면 오늘 기준. (진짜 콜드스타트는 호출부가 따로 처리한다.)
    """
    parsed = {}
    for i in range(N_SNAPS):
        d = _parse_date((snapshots.get(str(i)) or {}).get('next_rebal_date'))
        if d is not None:
            parsed[i] = d

    if len(parsed) == N_SNAPS and _schedule_phase_ok(list(parsed.values())):
        return [parsed[i] for i in range(N_SNAPS)], None

    # 예정일이 둘 이상 남았는데 위상이 전부 제각각이면 다수결이 성립하지 않는다. 그때
    # 낮은 번호를 임의로 채택하면 근거 없는 시간표를 확정하는 것이다 — 마지막 발화일
    # 쪽을 먼저 본다. (2026-08-28 4차 검토 지적.)
    trusted = True
    if len(parsed) >= 2:
        seen = {}
        for i, d in parsed.items():
            r = (d.toordinal() - i * SNAP_STAGGER_DAYS) % SNAP_PERIOD_DAYS
            seen[r] = seen.get(r, 0) + 1
        trusted = max(seen.values()) >= 2

    if parsed and trusted:
        dates, filled = _resolve_from_dates(parsed, today)
        if not filled:
            return [dates[i] for i in range(N_SNAPS)], None
        note = ('앵커 시간표 재구성 — snap '
                + ', '.join(str(i) for i in filled)
                + ' 의 예정일만 다시 잡음(나머지는 그대로)')
        return [dates[i] for i in range(N_SNAPS)], note

    lasts = {}
    for i in range(N_SNAPS):
        d = _parse_date((snapshots.get(str(i)) or {}).get('last_rebal_date'))
        if d is not None:
            lasts[i] = d

    if len(lasts) == N_SNAPS:
        # 각자 자기 발화일 + 69일. 스태거가 살아 있으면 이게 정확하다.
        cand = [lasts[i] + timedelta(days=SNAP_PERIOD_DAYS) for i in range(N_SNAPS)]
        if _schedule_phase_ok(cand):
            return cand, '앵커 시간표 최초 부여 — 각 snap 의 마지막 발화일 + 69일'
    if parsed and not lasts:
        # 위상은 못 믿지만 달리 기준이 없다. 예정일로 세우고 그 사실을 알린다.
        dates, _filled = _resolve_from_dates(parsed, today)
        return ([dates[i] for i in range(N_SNAPS)],
                '앵커 시간표 재구성 — 예정일끼리 위상이 갈려 근거가 약하다(확인 필요)')
    if lasts:
        base = max(lasts.values()) + timedelta(days=SNAP_PERIOD_DAYS)
        note = (f'앵커 시간표 최초 부여 — 마지막 발화 {max(lasts.values()).isoformat()} '
                f'+ {SNAP_PERIOD_DAYS}일 기준으로 23일씩 벌림')
    else:
        base = today
        note = '앵커 시간표 최초 부여 — 기준 삼을 날짜가 없어 오늘 기준'
    return [base + timedelta(days=i * SNAP_STAGGER_DAYS) for i in range(N_SNAPS)], note


def check_anchors(signal: dict, state: dict):
    """snap 별 고정 시간표 앵커 체크.

    각 snap 은 자기 다음 앵커일(next_rebal_date)을 들고 있다. 그날이 되면 갈아타고,
    다음 앵커일은 그 예정일에 69일을 더해서 정한다 — 실제 발화일이 아니다.
    채택 BT(stock_engine_snap.py 의 snap_next_sched += snap_days)와 같은 규칙이고,
    이래야 세 snap 의 23일 간격이 영구히 유지된다.

    발화일을 앵커일로 덮어쓰던 옛 방식은 셋이 한 번 같은 날 발화하면(V24 마이그레이션이
    전부 "즉시 발화" 로 맞춰놨다) 그 뒤로 영원히 같이 움직였다. 실제로 2026-07-08 에
    셋이 겹친 채 굳어 스냅샷 1개짜리 전략으로 돌았다. (2026-08-28)

    상태가 온전하지 않을 때의 세 갈래(ai-debate run-20260828T050415Z 지적):
      · 신호에 종목·비중이 둘 다 없다 → 발화 안 한다. 보유를 지운다는 뜻이 되기 때문이다.
      · 스냅샷이 통째로 비었다 → 셋을 오늘 신호로 채운다. 하나만 채우면 목표가
        3분의 1로 쪼그라들어 멀쩡한 보유를 대량 매도한다.
      · 시간표가 깨졌거나 위상이 무너졌다 → _resolve_anchor_schedule 이 복원하고 알린다.
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

    if not _picks_weights_ok(picks, weights):
        # 비중이 비었거나 합이 안 맞는 신호로 발화하면 그 모자란 만큼이 현금 목표가 되어
        # 멀쩡한 보유를 판다. 종목 목록만 있고 비중이 빈 경우도 여기서 걸린다.
        # 모멘텀 전멸(전량 현금)은 weights={'Cash': 1.0} 이라 정상 통과한다.
        log(f'  ⚠️ 신호가 온전하지 않다(종목 {picks} / 비중 {weights}) — 앵커 발화 보류(보유 유지)')
        try:
            # 로그만 남기면 발화가 무기한 밀리는 걸 늦게 안다. 보류는 소리를 내야 한다.
            send_telegram(f'⚠️ 주식 앵커 발화 보류: 신호의 종목·비중이 온전하지 않다\n'
                          f'종목 {picks} / 비중 {weights}')
        except Exception as e:
            log(f'  ⚠️ 앵커 보류 알림 전송 실패: {e}')
        return

    snapshots = state.setdefault('snapshots', {})
    if all(not (snapshots.get(str(i)) or {}) for i in range(N_SNAPS)):
        # 상태를 통째로 잃은 경우. 한 snap 만 채우면 목표가 3분의 1로 쪼그라들어
        # 멀쩡한 보유를 대량 매도한다. 셋을 오늘 신호로 채우고 시간표만 벌려 둔다.
        for snap_id in range(N_SNAPS):
            snapshots[str(snap_id)] = {
                'picks': list(picks),
                'weights': dict(weights),
                'last_rebal_date': today.isoformat(),
                'next_rebal_date': (today + timedelta(
                    days=SNAP_PERIOD_DAYS + snap_id * SNAP_STAGGER_DAYS)).isoformat(),
            }
        state['rebalancing_needed'] = True
        msg = f'주식 스냅샷이 비어 있어 오늘 신호로 셋 다 초기화: {picks}'
        log(f'  🆕 {msg}')
        try:
            send_telegram(f'⚠️ {msg}')
        except Exception as e:
            log(f'  ⚠️ 초기화 알림 전송 실패: {e}')
        return

    schedule, repair = _resolve_anchor_schedule(snapshots, today)
    if repair:
        log(f'  🧭 {repair} → ' + ', '.join(d.isoformat() for d in schedule))
        try:
            send_telegram('주식 앵커 시간표 정비\n' + repair + '\n다음 앵커: '
                          + ', '.join(d.isoformat() for d in schedule))
        except Exception as e:
            log(f'  ⚠️ 앵커 시간표 알림 전송 실패: {e}')

    for snap_id in range(N_SNAPS):
        snap = snapshots.get(str(snap_id))
        next_date = schedule[snap_id]
        if today < next_date:
            # 없는 snap 을 여기서 만들지 않는다. 빈 껍데기를 만들어두면 merge 분모에는
            # 들어가고 비중은 0 이라 합산 목표가 쪼그라든다.
            if snap is not None:
                snap['next_rebal_date'] = next_date.isoformat()
            continue
        snap = snapshots.setdefault(str(snap_id), {})
        snap['picks'] = list(picks)
        snap['weights'] = dict(weights)
        snap['last_rebal_date'] = today.isoformat()
        # 다음 예정일은 오늘이 아니라 이번 예정일에서 센다. 주말·휴일·실행 누락으로
        # 하루 밀려도 위상이 안 밀린다.
        while next_date <= today:
            next_date += timedelta(days=SNAP_PERIOD_DAYS)
        snap['next_rebal_date'] = next_date.isoformat()
        state['rebalancing_needed'] = True
        log(f'  📅 snap{snap_id} 앵커 발화 → {picks} (다음 앵커 {next_date.isoformat()})')


def snapshot_payload_defect(state: dict):
    """일부 snap 만 비중을 잃은 상태를 찾는다. 결함이면 사유 문자열, 아니면 None.

    merge_tranches 는 snapshots 개수로 나눈다. 셋 중 하나가 비중을 잃으면 합산 목표가
    3분의 2로 줄고, 그 차이가 그대로 대량 매도 주문이 된다.

    키가 통째로 없는 것도 같은 결함이다. 분모도 같이 줄어 합은 1 이지만, 그 snap 이 들고
    있던 종목이 합산 목표에서 사라지므로 그 종목 전량 매도 + 남은 종목 매수가 된다.
    합이 맞는다는 건 산술적 축소를 피한다는 뜻이지 목표가 맞다는 뜻이 아니다.
    (2026-08-28 3차 검토 지적.)

    복구는 사람이 한다. 잃은 snap 의 종목은 다른 snap 이 알지 못하는 별개 시점의 선정이라
    코드가 지어낼 수 없다. 그래서 정규 리밸런싱만 멈추고 알린다.
    """
    snapshots = state.get('snapshots') or {}
    if not snapshots:
        return None
    missing = sorted(str(i) for i in range(N_SNAPS) if str(i) not in snapshots)
    bad = sorted(k for k, v in snapshots.items() if not _weights_ok((v or {}).get('weights')))
    mismatch = []
    for k in sorted(snapshots):
        snap = snapshots.get(k) or {}
        picks = snap.get('picks')
        if not isinstance(picks, list):
            continue
        held = {t for t, w in (snap.get('weights') or {}).items()
                if t != 'Cash' and _as_float(w) > 0}
        if held - set(picks):
            mismatch.append(k)
    if not bad and not missing and not mismatch:
        return None
    parts = []
    if bad:
        parts.append(f'스냅샷 {", ".join(bad)} 의 비중이 온전하지 않다')
    if mismatch:
        parts.append(f'스냅샷 {", ".join(mismatch)} 의 종목과 비중이 어긋난다')
    if missing:
        parts.append(f'스냅샷 {", ".join(missing)} 이(가) 통째로 없다')
    return ' / '.join(parts) + f' (전체 {N_SNAPS}개 기준)'


def merge_tranches(state: dict) -> Dict[str, float]:
    """V24 snapshots merge → 최종 target (3 snap EW 평균)."""
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


def refill_snaps_fresh(signal: dict, state: dict) -> bool:
    """V25 drift-refill: 드리프트 발화일에 모든 snap picks 를 그날 fresh 신호로 교체.

    채택 BT(bt_stock_mom3) 정합 — 발화일에 3-mom selection 으로 종목 교체.
    앵커 시간표(next_rebal_date)는 건드리지 않음 → 정규 cadence(69일) 유지 (BT 의 고정 phase 와 동일).
    같은 거래일 signal 은 _compute_signal_v25 로 1회만 계산되므로 picks 고정.
    반환: 종목집합이 실제로 바뀌었으면 True.
    """
    stock_sig = signal.get('stock', {})
    risk_on = stock_sig.get('risk_on', True)
    if risk_on:
        picks = stock_sig.get('offense_picks', [])
        weights = stock_sig.get('offense_weights', {})
    else:
        picks = stock_sig.get('defense_picks', [])
        weights = stock_sig.get('defense_weights', {})
    if not _picks_weights_ok(picks, weights):
        # 앵커와 같은 검증. 여기가 뚫리면 드리프트 발화일에 온전치 않은 비중이 세 snap 에
        # 통째로 들어가고, 그 실행에서 바로 전량 매도가 된다.
        log(f'  ⚠️ 신호가 온전하지 않다(종목 {picks} / 비중 {weights}) — 드리프트 리필 보류(보유 유지)')
        send_telegram(f'⚠️ 주식 신호 이상으로 종목 교체 보류: 종목 {picks} / 비중 {weights}')
        return False
    snapshots = state.setdefault('snapshots', {})
    changed = False
    for snap_id in range(N_SNAPS):
        snap = snapshots.setdefault(str(snap_id), {})
        if snap.get('picks') != list(picks):
            changed = True
        snap['picks'] = list(picks)
        snap['weights'] = dict(weights)
    return changed


def execute_delta(target: Dict[str, float], api: KISAPI, state: dict, cash_buffer: float = None,
                  effective_pv_usd: float = None, prefetched_balance=None, prefetched_at=None):
    """target vs 현재 잔고 → Delta 매매 (정수 주 단위).

    effective_pv_usd: alloc_transit cap 시 effective_pv (= min(actual, target_ratio×total)).
                     None 이면 actual total_usd 사용.
    prefetched_balance: 같은 회차에서 이미 읽은 (보유, 총자산USD, 환율). 주문을 내기 전이라
                     값이 같으므로 그대로 쓴다. None 이면 여기서 직접 읽는다.
                     주문 뒤 체결 확인은 이 값을 안 쓰고 항상 새로 묻는다.
    prefetched_at:   그 값을 읽은 시각(time.time()). 나이가 BALANCE_MAX_AGE_SEC 를 넘으면
                     그 값으로 주문하지 않는다 — 들어올 때 넘었으면 새로 읽고, 수량을
                     산출하는 동안 넘겼으면 주문 없이 돌아간다.
    """
    if cash_buffer is None:
        cash_buffer = CASH_BUFFER_DEFAULT
    _reuse_age = (time.time() - prefetched_at) if (prefetched_balance is not None
                                                   and prefetched_at is not None) else None
    if prefetched_balance is not None and (_reuse_age is None or _reuse_age <= BALANCE_MAX_AGE_SEC):
        holdings, total_usd, exrt = prefetched_balance
        _bal_at = prefetched_at if prefetched_at is not None else time.time()
        if _reuse_age is not None:
            log(f'    사전 조회 재사용 (나이 {_reuse_age:.1f}s / 허용 {BALANCE_MAX_AGE_SEC}s)')
    else:
        if prefetched_balance is not None:
            log(f'    ⚠️ 사전 조회가 {_reuse_age:.0f}초 전 값이다 — 낡은 값으로 주문하지 않고 새로 읽는다')
            holdings, total_usd, exrt = _read_balance_or_skip(api)
        else:
            holdings, total_usd, exrt = api.get_balance()
        _bal_at = time.time()

    # alloc_transit cap 적용
    pv_for_target = total_usd if effective_pv_usd is None else min(total_usd, effective_pv_usd)

    # 현재 비중 계산 — 보유 수량 × 현재가. 총자산도 같은 시점 기준이라 둘이 일관된다.
    current_values = {}
    for ticker, qty in holdings.items():
        price = api.get_current_price(ticker)
        current_values[ticker] = (qty * price) if price else 0
        log(f'    현재 {ticker}: ${current_values[ticker]:,.0f} ({(current_values[ticker] / total_usd):.1%})')

    # 매도/매수 리스트
    sells = []
    buys = []
    failed_orders = []
    target_usd = pv_for_target * (1 - cash_buffer)
    if effective_pv_usd is not None and effective_pv_usd < total_usd:
        log(f'    🔴 alloc_transit cap 적용: pv ${total_usd:,.0f} → ${pv_for_target:,.0f}')
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

    # 주문 직전 나이 검사 — 여기까지 오는 동안 종목마다 현재가 조회가 돌았다. 수량을
    # 산출한 계좌값이 허용 나이를 넘겼으면 주문을 내지 않고 돌아간다. rebalancing_needed
    # 가 True 로 남아 다음 회차가 그때의 실제 잔고로 다시 계산한다.
    _order_age = time.time() - _bal_at
    if sells or buys:
        log(f'    주문 직전 계좌값 나이: {_order_age:.1f}s (허용 {BALANCE_MAX_AGE_SEC}s)')
        if _order_age > BALANCE_MAX_AGE_SEC:
            log(f'    ⛔ 계좌값이 낡았다({_order_age:.0f}s) — 주문 없이 돌아간다')
            send_telegram(f'⛔ 주식: 계좌값이 {_order_age:.0f}초 전 것이라 주문을 내지 않았다. '
                          f'다음 회차에 다시 계산한다.')
            state['rebalancing_needed'] = True
            return {
                'holdings': holdings,
                'total_usd': total_usd,
                'exchange_rate': exrt,
                'sells': [],
                'buys': [],
                'failed_orders': [],
                'max_diff': None,
                'completed': False,
                'post_balance': None,
                'stale_abort': True,
            }

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

    # 목표 달성 확인 — 여기서 계좌 조회가 실패하면 주문은 이미 나갔다. 되돌리거나
    # 다시 내지 않고 "완료 판정 불가"로 남긴다(rebalancing_needed 유지 → 다음 회차가
    # 그때의 실제 잔고로 델타를 다시 계산한다).
    post_balance = None   # 체결 확인에 쓴 값. 보고도 이걸 쓴다 — 같은 시점을 두 번 묻지 않는다.
    try:
        holdings2, total2, exrt2 = api.get_balance()
        post_balance = (holdings2, total2, exrt2)
    except BalanceUnavailable as _e:
        log(f'  ⚠️ 주문 후 잔고 조회 실패 — 체결 확인 불가: {_e}')
        send_telegram(f'⚠️ 주식: 주문은 냈는데 잔고 조회가 안 된다 — 체결 확인 불가 ({_e})')
        holdings2, total2 = {}, 0.0
    if total2 > 0:
        max_diff = 0
        # alloc_transit cap 적용 시 denom 도 capped 기준
        denom_for_diff = min(total2, effective_pv_usd) if effective_pv_usd is not None else total2
        for ticker, target_w in target.items():
            if ticker == 'Cash':
                continue
            price = api.get_current_price(ticker)
            # cash buffer 반영 목표로 비교.
            # merge_tranches target 은 risky 합=1.0(버퍼 전)이라 실효 목표는 ×(1-buffer).
            # (미반영 시 현금이 정상적으로 buffer% 여도 max_diff 가 buffer 만큼 떠서
            #  rebalancing_needed 가 영구히 안 꺼지는 latch 버그 발생 — 2026-06-05 수정)
            current_w = (holdings2.get(ticker, 0) * price / denom_for_diff) if price > 0 else 0
            target_eff = target_w * (1 - cash_buffer)
            max_diff = max(max_diff, abs(target_eff - current_w))
        if max_diff < REBALANCE_TOLERANCE and not failed_orders:
            state['rebalancing_needed'] = False
            log(f'  ✅ 목표 달성 (±{REBALANCE_TOLERANCE:.0%} 이내)')
            log('✅ 리밸런싱 완료 (silent)')
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
            'post_balance': post_balance,
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
        'post_balance': post_balance,
    }


# ═══ run_once ═══

BALANCE_RETRY_ATTEMPTS = 3   # 계좌 조회를 몇 번까지 다시 볼까
BALANCE_RETRY_WAIT_SEC = 30  # 다시 보기 전에 쉬는 시간

# 사전에 읽은 계좌값으로 주문을 낼 수 있는 최대 나이. 넘기면 그 값으로 주문하지 않는다.
# 실측 회차 전체가 6.5~7.0초(2026-10-26·27·28)라 정상 흐름과는 거리가 멀고, 넘겼을 때의
# 대가는 조회 한 번 또는 그 회차 주문 보류라 작게 잡는다.
BALANCE_MAX_AGE_SEC = 60

PENDING_CONFIRM_ATTEMPTS = 3  # 취소를 건 뒤 미체결이 없어졌는지 다시 볼 횟수
PENDING_CONFIRM_WAIT_SEC = 5  # 그 사이 쉬는 시간


def _confirm_no_pending(api: 'KISAPI', cancel_result: dict):
    """취소를 건 뒤 미체결이 실제로 사라졌는지 확인한다. (확정됐나, 사유).

    확정 못 하면 부르는 쪽이 그 회차를 주문 없이 멈춘다. 잔존 주문이 뒤늦게 체결되면
    이어서 읽는 사전 잔고가 거짓이 되고, 그 수량으로 중복 주문이 나간다.
    """
    if not cancel_result.get('queried'):
        return False, '미체결 조회 실패'
    if not cancel_result.get('found'):
        return True, '미체결 없음'
    if cancel_result.get('failed'):
        return False, f"취소 실패 {cancel_result['failed']}건"
    for _ in range(PENDING_CONFIRM_ATTEMPTS):
        time.sleep(PENDING_CONFIRM_WAIT_SEC)
        again = api.cancel_all_pending()
        if not again.get('queried'):
            return False, '취소 뒤 미체결 재조회 실패'
        if not again.get('found'):
            return True, f"미체결 {cancel_result['found']}건 취소·종결 확인"
        log(f"  ⏳ 미체결 {again['found']}건이 남아 있다 — {PENDING_CONFIRM_WAIT_SEC}초 뒤 다시 본다")
    return False, '취소 뒤에도 미체결이 남아 있다'


def _read_balance_or_skip(api: 'KISAPI'):
    """계좌 조회를 몇 번 다시 본 뒤에도 안 되면 BalanceUnavailable 을 그대로 올린다.

    실패가 일시적인 것이 실측됐다 — 2026-10-21 23:35 회차는 두 번 실패하고 18초 뒤
    세 번째에 성공했다. 그 회차는 실패한 값으로 매매를 해버렸다.
    """
    last = None
    for attempt in range(BALANCE_RETRY_ATTEMPTS):
        try:
            return api.get_balance()
        except BalanceUnavailable as e:
            last = e
            if attempt < BALANCE_RETRY_ATTEMPTS - 1:
                log(f'  ⏳ 계좌 조회 실패({attempt + 1}/{BALANCE_RETRY_ATTEMPTS}) — '
                    f'{BALANCE_RETRY_WAIT_SEC}초 쉬고 다시 본다: {e}')
                time.sleep(BALANCE_RETRY_WAIT_SEC)
    raise last if last else BalanceUnavailable('계좌 조회 실패')


DATE_ALIGN_TOLERANCE_BD = 3  # 종목별 yfinance 지연 시 공통 기준일이 T-1 대비 최대 이만큼 과거여도 진행
KIS_FILL_ENABLED = True      # 야후 지연 종목을 KIS 해외 일봉 종가로 채워 진짜 T-1 정렬 (False 면 공통일자 가드만)


def _kis_fill_recent(ticker, have_last_date, target_date):
    """야후가 ticker 를 have_last_date 까지만 줬을 때 (have_last_date, target_date] 구간 종가를
    KIS dailyprice(HHDFS76240000) 로 가져와 {Timestamp: close} 반환. 실패/미커버 시 {}.

    KIS raw 종가 = 야후 수정종가(최신 bar 는 미래배당 미반영이라 동일, 실측 4자리 일치).
    target_date ≤ max_close_date(T-1) 라 장중 partial 미요청 — look-ahead 없음.
    """
    import pandas as pd
    excd = QUOTE_EXCD_MAP.get(ticker, 'NAS')
    bymd = target_date.strftime('%Y%m%d')
    filled = {}
    try:
        data = _get('/uapi/overseas-price/v1/quotations/dailyprice', 'HHDFS76240000', {
            'AUTH': '', 'EXCD': excd, 'SYMB': ticker,
            'GUBN': '0', 'BYMD': bymd, 'MODP': '0',
        }, retries=2)
        for it in data.get('output2', []) or []:
            if not isinstance(it, dict):
                continue
            xy, cl = it.get('xymd'), it.get('clos')
            if not xy or not cl:
                continue
            d = pd.to_datetime(xy, format='%Y%m%d')
            c = float(cl)
            if c > 0 and have_last_date < d.date() <= target_date:
                filled[d] = c
    except Exception as e:
        log(f'  ⚠️ KIS fill {ticker}: {e}')
    return filled


def _resolve_common_date(last_dates, max_close_date, tol_bd=DATE_ALIGN_TOLERANCE_BD):
    """종목별 마지막일이 불일치할 때 전 종목 공통 최신일(=min(last_dates))과 그 신선도 판정.

    반환 (common_date, gap_bd, ok). ok=True 면 common_date 로 정렬해 진행, False 면 SKIP.
    gap_bd = common_date 직후~max_close_date(T-1) 사이 영업일 수. tol_bd 이하면 진행.
    """
    import numpy as _np
    from datetime import timedelta as _td2
    common_last = min(last_dates.values())
    if max_close_date > common_last:
        gap_bd = int(_np.busday_count(common_last + _td2(days=1), max_close_date + _td2(days=1)))
    else:
        gap_bd = 0
    return common_last, gap_bd, (gap_bd <= tol_bd)


def _fetch_strategy_prices(tickers, days=400, max_close_date=None):
    """Yahoo Adj_Close (auto_adjust=True) fetch.

    max_close_date 이후 row 는 자르고, 마지막 row 날짜를 로그.
    23:35 KST = 14:35 UTC 시점에 US 장중 partial bar 가 yfinance 에 섞일 수 있어 명시적 차단.
    max_close_date 미지정 시 KST 기준 어제 (= 미국 거래일 T-1 이상) 로 자동 설정.
    종목별 마지막일 불일치 시: 전 종목 공통 최신일로 정렬해 진행(공통일이 T-1 대비 tol_bd 영업일
    이내). 너무 과거면 HARD FAIL (caller SKIP).
    """
    import yfinance as yf
    import pandas as pd
    from datetime import datetime as _dt, timedelta as _td, timezone as _tz
    if max_close_date is None:
        # KST 어제 = T-1 안전 (yfinance row index 는 UTC 자정 기준)
        kst_now = _dt.now(_tz(_td(hours=9)))
        max_close_date = (kst_now - _td(days=1)).date()
    end = _dt.now(); start = end - _td(days=days)
    out = {}
    last_dates = {}
    for t in tickers:
        try:
            df = yf.download(t, start=start.strftime('%Y-%m-%d'),
                            end=end.strftime('%Y-%m-%d'),
                            progress=False, auto_adjust=True)
            if df is None or len(df) == 0:
                continue
            s = df['Close'].squeeze().dropna()
            s.index = pd.to_datetime(s.index)
            # max_close_date 이후 row 제거 (장중 partial 차단)
            s = s[s.index.date <= max_close_date]
            if len(s) < 2:
                log(f'  ⚠️ {t}: T-1 까지 데이터 부족 ({len(s)} rows)')
                continue
            out[t] = s
            last_dates[t] = s.index[-1].date()
        except Exception as e:
            log(f'  ⚠️ Yahoo fetch {t}: {e}')
    if last_dates:
        dates_set = set(last_dates.values())
        if len(dates_set) > 1 and KIS_FILL_ENABLED:
            # 1차 복구: 야후 지연 종목을 KIS 최신 종가로 채워 진짜 T-1 정렬.
            target = min(max(last_dates.values()), max_close_date)
            for t in list(out.keys()):
                if last_dates[t] < target:
                    fill = _kis_fill_recent(t, last_dates[t], target)
                    if fill:
                        s = out[t]
                        for d, c in sorted(fill.items()):
                            s.loc[d] = c
                        s = s[s.index.date <= max_close_date].sort_index()
                        out[t] = s
                        last_dates[t] = s.index[-1].date()
                        log(f'  🔁 {t}: KIS 보강 {[str(d.date()) for d in sorted(fill)]} → last {last_dates[t]}')
            dates_set = set(last_dates.values())
        if len(dates_set) == 1:
            log(f'  📅 가격 기준일: {list(dates_set)[0]} (max_close_date={max_close_date})')
        else:
            # 2차(폴백): KIS 로도 못 맞춘 잔여 불일치 → 공통 최신일 정렬(≤tol) 또는 SKIP.
            common_last, gap_bd, ok = _resolve_common_date(last_dates, max_close_date)
            if ok:
                for t in list(out.keys()):
                    out[t] = out[t][out[t].index.date <= common_last]
                    if len(out[t]) < 2:
                        out.pop(t)
                log(f'  📅 종목별 마지막일 불일치 → 공통 기준일 {common_last} 정렬 진행 '
                    f'(gap {gap_bd}bd ≤ {DATE_ALIGN_TOLERANCE_BD}, last_dates={last_dates})')
            else:
                log(f'  ⚠️ 가격 마지막 date 불일치 + 공통일 {common_last} 너무 과거 '
                    f'(gap {gap_bd}bd > {DATE_ALIGN_TOLERANCE_BD}) — HARD FAIL: {last_dates}')
                out['__date_mismatch__'] = last_dates  # type: ignore
    return out


def _compute_signal_v25(state):
    """V25: stock_strategy_v25 직접 호출 → signal dict 생성.

    signal_state.json 의존 없음. 실패 시 None 반환 → 호출자가 fallback.
    prev_risk_on 추출: explicit None check (False 명시값 보존).
    가격 기준일 hard check + cold-start SKIP.
    """
    if stock_strat is None:
        return None
    try:
        import pandas as pd  # local ensure
        tickers = (stock_strat.OFFENSIVE_STOCK_UNIVERSE
                   + stock_strat.DEFENSIVE_STOCK_UNIVERSE
                   + [stock_strat.STOCK_CANARY_TICKER])
        tickers = sorted(set(tickers))
        prices = _fetch_strategy_prices(tickers)
        # Hard check: 가격 기준일 불일치 시 SKIP
        if '__date_mismatch__' in prices:
            dm = prices.pop('__date_mismatch__')
            send_telegram(f'⚠️ V25 stock: 가격 기준일 불일치 {dm} — 주문 SKIP')
            return None
        if len(prices) < 5:
            log(f'  ⚠️ V25 strategy: 가격 부족 ({len(prices)} 종목)')
            send_telegram(f'⚠️ V25 stock: 가격 데이터 부족 ({len(prices)}) — 주문 SKIP')
            return None
        # prev_risk_on: explicit None check (False 명시 보존)
        prev_risk_on = state.get('prev_risk_on')
        if prev_risk_on is None:
            prev_risk_on = state.get('canary_risk_on')
        # Cold-start SKIP: prev 도 None + EEM 결측이면 risk_on 추측 불가 → SKIP
        eem = prices.get(stock_strat.STOCK_CANARY_TICKER)
        if prev_risk_on is None and (eem is None or len(eem) < stock_strat.STOCK_CANARY_MA_PERIOD):
            log(f'  ⚠️ V25 strategy: cold-start (prev/canary 모두 None + EEM 결측) → SKIP')
            send_telegram('⚠️ V25 stock: cold-start 데이터 결측 → 주문 SKIP')
            return None
        off_picks, off_w, off_meta = stock_strat.compute_offense(prices)
        def_picks, def_w, def_meta = stock_strat.compute_defense(prices)
        risk_on, canary_meta = stock_strat.eem_canary(eem, prev_risk_on)
        signal = {
            'stock': {
                'risk_on': bool(risk_on),
                'offense_picks': list(off_picks),
                'offense_weights': dict(off_w),
                'defense_picks': list(def_picks),
                'defense_weights': dict(def_w),
            },
            'meta': {
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
                'source': 'executor_stock_v25',
                'strategy_version': 'V25',
                'canary': canary_meta,
                'offense_excluded_by_mom': off_meta.get('excluded_by_mom', []),
            },
        }
        log(f'  V25 strategy: risk_on={risk_on}, picks={off_picks if risk_on else def_picks}')
        return signal
    except Exception as e:
        log(f'  ⚠️ V25 strategy compute 실패: {e}')
        return None


V25_DEPLOY_DATE = '2026-05-29'   # V25 첫 배포일 (이 날부터 3 영업일 V24 fallback 금지)
V24_FALLBACK_BLOCK_DAYS = 3


def _signal_state_fresh(signal):
    """signal_state.json fallback 검증.

    - 24h 이내 + strategy_version V25 (V24 는 첫 3일 금지)
    - V25 fresh signal 만 fallback 허용
    """
    if not signal:
        return False
    meta = signal.get('meta', {}) or {}
    ver = meta.get('strategy_version', 'V24')
    if ver not in ('V24', 'V25'):
        log(f'  ⚠️ fallback signal: unknown strategy_version={ver}')
        return False
    # V25 첫 배포 후 V24_FALLBACK_BLOCK_DAYS 영업일 V24 fallback 금지
    if ver == 'V24':
        try:
            from datetime import date as _d
            dep = _d.fromisoformat(V25_DEPLOY_DATE)
            today = _d.today()
            days_since = (today - dep).days
            if days_since <= V24_FALLBACK_BLOCK_DAYS:
                log(f'  ⚠️ V24 fallback 금지 (V25 배포 후 {days_since}일, 블록 {V24_FALLBACK_BLOCK_DAYS}일)')
                return False
        except Exception as _ex:
            log(f'  ⚠️ V24 fallback 검증 예외: {_ex} → 차단')
            return False
    updated_at = meta.get('updated_at', '')
    try:
        age_h = (datetime.now() - datetime.strptime(updated_at, '%Y-%m-%d %H:%M')).total_seconds() / 3600
        if age_h > 24:
            log(f'  ⚠️ fallback signal: stale ({age_h:.1f}h)')
            return False
    except Exception:
        log(f'  ⚠️ fallback signal: updated_at 파싱 실패 ({updated_at})')
        return False

    # 주식 기준일 검증 (2026-09-30 토론 run-20260930T045429Z).
    # updated_at 은 리포트가 '언제 돌았나' 이지 '어느 종가를 썼나' 가 아니다. 아침
    # 09:15 에는 야후 수정종가가 직전 세션을 아직 안 채운 날이 있어, 24시간 안의
    # 싱싱한 파일이 한 세션 낡은 종가로 만든 신호를 들고 있을 수 있다.
    # 이 경로는 밤 가격 받기가 실패했을 때만 탄다 — 그때 낡은 종목으로 주문을 내느니
    # 그날은 안 내는 쪽이 맞다. meta.signal_date 는 비트코인 날짜라 여기 쓰지 않는다.
    stk = signal.get('stock', {}) or {}
    if 'basis_date' not in stk and 'basis_oldest' not in stk:
        log('  ⚠️ fallback signal: 주식 기준일이 기록돼 있지 않다 → 차단 '
            '(리포트 판본이 낡았다)')
        return False
    if stk.get('basis_mixed'):
        log(f"  ⚠️ fallback signal: 종목마다 종가 날짜가 다르다 "
            f"({stk.get('basis_per_ticker')}) → 차단")
        return False
    try:
        lag_bd = int(stk.get('basis_lag_bd'))
    except (TypeError, ValueError):
        log(f"  ⚠️ fallback signal: 주식 기준일 지연을 못 읽었다 "
            f"({stk.get('basis_lag_bd')!r}) → 차단")
        return False
    if lag_bd > DATE_ALIGN_TOLERANCE_BD:
        log(f"  ⚠️ fallback signal: 주식 종가가 {lag_bd}영업일 이전 "
            f"(허용 {DATE_ALIGN_TOLERANCE_BD}) → 차단")
        return False
    log(f"  📅 fallback signal 주식 기준일 {stk.get('basis_date')} "
        f"(지연 {lag_bd}영업일, 계산된 쪽 {stk.get('computed_side', '?')})")
    return True


def run_once(dry_run=False):
    global RUN_ID
    RUN_ID = uuid.uuid4().hex
    _run_id_ref[0] = RUN_ID
    _DRY_RUN_SILENT[0] = bool(dry_run)
    _t0 = time.time()
    log('=' * 50)
    log('주식 executor 시작')

    # 파일이 있는데 못 읽는 것과 아예 없는 것은 다르다. 공용 load_json 은 둘 다 {} 로
    # 돌려주는데, 그러면 손상된 상태파일이 콜드스타트로 둔갑해 실재하는 보유·앵커 이력을
    # 무시한 채 매매한다. (ai-debate run-20260828T051722Z)
    if os.path.exists(TRADE_STATE_FILE):
        try:
            with open(TRADE_STATE_FILE) as _f:
                state = json.load(_f)
            if not isinstance(state, dict):
                raise ValueError(f'최상위가 dict 가 아니다({type(state).__name__})')
        except Exception as _e:
            log(f'  ⛔ 상태파일을 읽을 수 없다: {_e}')
            send_telegram(f'⛔ 주식 상태파일 손상: {_e}\n'
                          f'{TRADE_STATE_FILE}\n매매를 멈췄다. 백업 복구가 필요하다.')
            return
    else:
        state = {}

    _shape = state_structure_defect(state)
    if _shape:
        # 모양이 틀린 상태로는 방어도 못 한다 — 카나리 검사부터 예외로 죽는다. 멈추고 알린다.
        log(f'  ⛔ 상태 구조 이상: {_shape}')
        send_telegram(f'⛔ 주식 상태 구조 이상: {_shape}\n'
                      f'{TRADE_STATE_FILE}\n매매를 멈췄다. 백업 복구가 필요하다.')
        return
    api = KISAPI(dry_run=dry_run)

    # V25: strategy 직접 계산 우선 — 실패 시 signal_state.json fallback (freshness 검증)
    signal = _compute_signal_v25(state)
    signal_source = 'v25_compute'
    if signal is None:
        fallback = load_json(SIGNAL_STATE_FILE)
        if _signal_state_fresh(fallback):
            signal = fallback
            signal_source = 'signal_state.json'
            log('  ⚠️ V25 compute 실패 → signal_state.json fallback (fresh)')
            send_telegram(f'⚠️ V25 stock: compute 실패 → signal_state fallback ({fallback.get("meta",{}).get("strategy_version","?")})')
        else:
            log('  ⚠️ V25 compute 실패 + fallback signal stale/missing → 주문 SKIP')
            send_telegram('⚠️ V25 stock: compute 실패 + fallback stale → 주문 SKIP')
            return
    else:
        # V25 정상 계산 → 텔레그램 알림 (첫 1~3 거래일 모니터링)
        _stock = signal.get('stock', {})
        _meta = signal.get('meta', {})
        _can = (_meta.get('canary', {}) or {})
        _mode = '공격' if _stock.get('risk_on') else '수비'
        _picks = _stock.get('offense_picks' if _stock.get('risk_on') else 'defense_picks', [])
        _excl = _meta.get('offense_excluded_by_mom', [])
        _dist = _can.get('canary_dist')
        try:
            _dist_s = f"{_dist*100:+.2f}%" if _dist is not None else 'NA'
        except Exception:
            _dist_s = 'NA'
        log(
            f"V25 stock signal 계산: mode={_mode} EEM_dist={_dist_s} "
            f"picks={_picks} mom_excl={_excl if _excl else '없음'}"
        )

    # ── 디버그 로그: 입력 데이터 ──
    log(f'  signal_source: {signal_source}')
    log(f'  signal: {json.dumps(signal, ensure_ascii=False, default=str)[:500]}')
    log(f'  state: {json.dumps(state, ensure_ascii=False, default=str)[:500]}')

    if not signal:
        log('  signal 없음 — 스킵')
        return
    # V24 (2026-05-26): stock_cash_buffer 키 우선, 없으면 legacy cash_buffer, 둘 다 없으면 default 6%
    cash_buffer = float(state.get('stock_cash_buffer', state.get('cash_buffer', CASH_BUFFER_DEFAULT)))
    log(f'  cash_buffer: {cash_buffer:.0%}')

    # V24: snapshots 또는 V21 tranches 둘 중 하나라도 없으면 첫 실행 — 마이그레이션이 자동 처리.
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

    # 1. 미체결 취소 — 결과를 확인한다. 조회가 안 됐거나 취소가 실패했거나 주문이 남아
    #    있으면 아래에서 읽을 사전 잔고를 믿을 수 없다(잔존 주문이 뒤늦게 체결되면 그
    #    수량이 틀린다). 그때는 주문 없이 회차를 멈춘다 — 상태를 저장하지 않으므로 다음
    #    회차가 같은 자리에서 잇는다. 모의 실행은 취소를 실제로 보내지 않아 검사하지 않는다.
    _cancel = api.cancel_all_pending()
    if not dry_run:
        _pending_ok, _pending_why = _confirm_no_pending(api, _cancel)
        log(f'  미체결 확인: {_pending_why}')
        if not _pending_ok:
            log(f'  ⛔ 미체결 상태를 확정하지 못했다 — 이번 회차 매매를 건너뛴다: {_pending_why}')
            send_telegram(f'⛔ 주식: 미체결 주문 상태를 확인하지 못해 이번 회차를 쉰다 '
                          f'({_pending_why}). 다음 회차에 다시 본다.')
            log('주식 executor 완료 (미체결 확정 불가)')
            return

    # 2. V24: 가드 없음. 잔존 crash_active 만 정리.
    check_crash(signal, api, state)

    if not is_fresh:
        if not dry_run:
            save_json(TRADE_STATE_FILE, state)
        log('주식 executor 완료 (stale signal)')
        return

    # 2.5 계좌 조회 사전 확인 — 카나리 소비·앵커 전진·드리프트·리필보다 앞이다.
    # 조회가 비면 그 회차는 쉰다. 상태를 저장하지 않으므로 다음 회차가 같은 자리에서 잇는다.
    # 일시적 실패가 실측돼서(2026-10-21 두 번 실패하고 세 번째 성공) 조금 쉬었다 다시 본다.
    # 여기서 읽은 값은 주문을 내기 전의 판단(드리프트·상한·델타 산출)에 그대로 재사용한다 —
    # 이 사이에 주문이 안 나가므로 다시 물어도 같은 값이고, 조회마다 다른 값이 오면 오히려
    # 한 회차의 판단이 서로 다른 시점을 섞는다. 체결 확인만 주문 뒤에 새로 묻는다.
    try:
        _bal_pre = _read_balance_or_skip(api)
        _bal_pre_at = time.time()   # 이 값의 나이를 주문 직전에 다시 잰다
    except BalanceUnavailable as _e:
        log(f'  ⛔ 계좌 조회 불가 — 이번 회차 매매를 건너뛴다: {_e}')
        send_telegram(f'⛔ 주식: 계좌 조회가 안 돼 이번 회차를 쉰다 ({_e}). 다음 회차에 다시 본다.')
        log('주식 executor 완료 (잔고 조회 불가)')
        return

    # 3. 카나리 플립
    _flipped = check_canary_flip(signal, state)

    # 3.5 스냅샷 온전성 — 일부가 비중을 잃거나 통째로 없으면 합산 목표의 구성이 바뀌어
    # 그 snap 이 들고 있던 종목을 전량 매도한다. 앵커 체크보다 먼저 본다 — 뒤에 두면
    # 앵커가 시간표를 전진시켜 저장한 뒤 매매만 멈춘다.
    # 방어로 넘어가는 플립(risk-off)이 있었던 실행만 통과시킨다. 플립은 모든 snap 을 같은
    # 비중으로 덮으므로 합산 목표가 어중간하게 섞이지 않는다 — 유일한 방어를 막지 않는다.
    # 공격으로 넘어가는 플립(risk-on)은 통과시키지 않는다. 방어를 막지 않는다는 명분이
    # 노출을 늘리는 데까지 가면 안 된다. 손상 상태에서 위험을 키우는 쪽은 멈춘다.
    # 다만 실제로는 소비된 플립이 세 snap 을 통째로 다시 쓰므로(refill_snaps_fresh 의
    # setdefault) 그 시점에 결함이 이미 사라져 여기서 걸릴 게 남지 않는 경우가 대부분이다.
    # 이 분기는 리필이 일부만 덮도록 바뀔 때를 위한 것이다(시험 [18] 이 현행을 고정한다).
    # (2026-08-28 4차 검토 지적.)
    _defense_flip = _flipped and not signal.get('stock', {}).get('risk_on', True)
    _defect = snapshot_payload_defect(state)
    if _defect and not _defense_flip:
        log(f'  ⛔ {_defect} — 정규 리밸런싱 중단(보유 유지)')
        send_telegram(f'⛔ 주식 상태 이상: {_defect}\n'
                      f'정규 리밸런싱을 멈췄다. 백업에서 복구가 필요하다.')
        log('주식 executor 완료 (스냅샷 결함)')
        return

    # 4. 앵커 체크
    check_anchors(signal, state)

    # 4.5. Drift trigger (2026-05-23): cur_w vs combined target half_turnover ≥ 0.10
    # alloc_transit cap (옵션 D): effective_pv = total_usd × cap_ratio (cap_ratio < 1.0 일 때만)
    _stock_cap_ratio = _read_alloc_transit_cap_ratio('stock')
    _stock_cap_usd = None
    if not state.get('rebalancing_needed', False):
        try:
            _merged_target = merge_tranches(state)
            if _merged_target:
                _holdings, _total_usd, _exrt = _bal_pre  # 2.5 의 사전 조회 재사용
                if _total_usd > 0:
                    if _stock_cap_ratio is not None and _stock_cap_ratio < 1.0:
                        _stock_cap_usd = _total_usd * _stock_cap_ratio
                    _pv_basis = _total_usd if _stock_cap_usd is None else _stock_cap_usd
                    if _stock_cap_usd is not None:
                        log(f'  🔴 alloc_transit cap_ratio={_stock_cap_ratio:.3f} → pv ${_total_usd:,.0f} → ${_pv_basis:,.0f}')
                    # 보유 수량 × 현재가 / 총자산. 총자산도 같은 시점 기준이다.
                    _cur_w = {}
                    for _tk, _qty in _holdings.items():
                        _px = api.get_current_price(_tk)
                        if _px and _qty > 0:
                            _cur_w[_tk] = (_qty * _px) / _pv_basis
                    _cash_w = max(0.0, 1.0 - sum(_cur_w.values()))
                    if _cash_w > 0:
                        _cur_w['Cash'] = _cash_w
                    # cash buffer 반영 — merged_target 은 risky-asset 100% 정규화이므로
                    # execute_delta 가 적용하는 cash buffer 만큼 target 도 스케일 다운 + Cash 추가
                    _stock_buf = float(state.get('stock_cash_buffer',
                                                 state.get('cash_buffer', CASH_BUFFER_DEFAULT)))
                    # V25: target 은 risky+Cash slot (sum=1.0 또는 risky 만 sum<=1.0).
                    # 모든 weight*(1-buf) 후 잔여 (buf 만큼)를 Cash 에 흡수.
                    _target_buf = {_k: float(_w) * (1.0 - _stock_buf)
                                   for _k, _w in _merged_target.items()}
                    _curr_sum = sum(_target_buf.values())
                    _target_buf['Cash'] = _target_buf.get('Cash', 0.0) + max(0.0, 1.0 - _curr_sum)
                    _all_keys = set(_cur_w.keys()) | set(_target_buf.keys())
                    _ht = sum(abs(_cur_w.get(_k, 0.0) - _target_buf.get(_k, 0.0))
                              for _k in _all_keys) / 2
                    if _ht >= 0.05:
                        state['rebalancing_needed'] = True
                        log(f'  🌊 V25 stock drift trigger: ht={_ht*100:.2f}pp ≥ 5pp → rebal 필요')
                        # V25 drift-refill (채택 BT 정합): 발화일에 그날 fresh 3-mom selection 으로
                        # snap picks 교체. _ht 는 위에서 옛 target 기준으로 산출됨(BT 와 동일 순서) →
                        # 이후 merge_tranches 가 새 picks 를 반영해 execute_delta 가 새 target 으로 체결.
                        if STOCK_DRIFT_REFILL:
                            _swapped = refill_snaps_fresh(signal, state)
                            if _swapped:
                                _new_t = merge_tranches(state)
                                log(f'  🔄 V25 drift-refill: snap picks → 그날 fresh 3-mom 으로 교체 (new target={_new_t})')
        except BalanceUnavailable:
            raise  # 계좌 조회 실패를 "드리프트 없음"으로 삼키지 않는다
        except Exception as _ex:
            log(f'  ⚠️ drift trigger 체크 실패: {_ex}')

    # alloc_transit cap (drift 체크 외에서도 execute_delta 에 적용)
    if _stock_cap_usd is None and _stock_cap_ratio is not None and _stock_cap_ratio < 1.0:
        _t2 = _bal_pre[1]  # 2.5 의 사전 조회 재사용
        _stock_cap_usd = (_t2 * _stock_cap_ratio) if _t2 > 0 else None

    # 옵션 Z: cap_defend trigger — cap_ratio < (1 - MIN_EXCESS) 면 rebal 강제 (drift 와 별개)
    if _stock_cap_ratio is not None and _stock_cap_ratio < (1.0 - CAP_DEFEND_MIN_EXCESS):
        if not state.get('rebalancing_needed', False):
            state['rebalancing_needed'] = True
            log(f'  🛡️ cap_defend trigger: cap_ratio={_stock_cap_ratio:.4f} < {1.0-CAP_DEFEND_MIN_EXCESS:.2f} (excess≥{CAP_DEFEND_MIN_EXCESS*100:.0f}%) → rebal 강제')

    # 5. Merge + Delta 매매
    _bal_after = None   # 주문 뒤 조회값. 리밸 분기를 안 타면 그대로 None 이다.
    _stale_abort = False  # 계좌값이 낡아 주문을 보류한 회차인가
    _rebal_ran = bool(state.get('rebalancing_needed', False))
    if state.get('rebalancing_needed', False):
        target = merge_tranches(state)
        log(f'  Target: {target}')
        mode = 'DRY-RUN' if dry_run else 'LIVE'
        try:
            result = execute_delta(target, api, state, cash_buffer=cash_buffer,
                                   effective_pv_usd=_stock_cap_usd, prefetched_balance=_bal_pre,
                                   prefetched_at=_bal_pre_at)
        except BalanceUnavailable as _e:
            # 사전 조회가 허용 나이를 넘겨 새로 읽었는데 그것도 실패했다. 주문은 아직
            # 안 나갔다 — 상태를 저장하지 않고 멈춘다.
            log(f'  ⛔ 계좌 재조회 실패 — 주문 없이 이번 회차를 멈춘다: {_e}')
            send_telegram(f'⛔ 주식: 계좌 조회가 안 돼 주문 없이 이번 회차를 멈춘다 ({_e}). '
                          f'다음 회차에 다시 본다.')
            log('주식 executor 완료 (잔고 재조회 실패)')
            return
        sells = result.get('sells', [])
        buys = result.get('buys', [])
        max_diff = result.get('max_diff')
        failed_orders = result.get('failed_orders', [])

        # V24 통일 보고
        import sys as _sys, os as _os
        _sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
        import v24_report as v24r
        from datetime import timezone, timedelta
        kst_now = datetime.now(timezone(timedelta(hours=9)))

        # 보고용 잔고는 execute_delta 가 체결 확인에 쓴 그 값이다. 같은 시점을 두 번 묻지 않는다.
        _bal_after = result.get('post_balance')
        _stale_abort = bool(result.get('stale_abort'))
        if _bal_after is None and not _stale_abort:
            log('  ⚠️ 보고용 잔고 없음 — 주문 뒤 조회가 실패한 회차다(체결 확인 불가)')
        if isinstance(_bal_after, tuple) and len(_bal_after) >= 2:
            holdings_q, total_usd, fx = _bal_after[0], _bal_after[1], (_bal_after[2] if len(_bal_after) > 2 else 1.0)
        else:
            holdings_q, total_usd, fx = {}, 0.0, 1.0

        holdings_list = []
        for tk, qty in holdings_q.items():
            try:
                p = api.get_current_price(tk)
            except Exception:
                p = 0.0
            v = qty * p
            if v < 100:  # $100 미만 dust
                continue
            holdings_list.append({
                'ticker': tk,
                'value_str': f'${v:,.0f}',
                'weight': (v / total_usd) if total_usd > 0 else 0.0,
            })
        holdings_list.sort(key=lambda h: -h['weight'])

        order_parts = []
        if sells:
            order_parts.append('매도: ' + ', '.join(f'{t} {q}주' for t, q, _ in sells))
        if buys:
            order_parts.append('매수: ' + ', '.join(f'{t} {q}주' for t, q, _ in buys))
        if failed_orders:
            order_parts.append('실패: ' + ', '.join(f'{side} {ticker} {qty}주'
                                                     for side, ticker, qty in failed_orders))
        orders_text = ('\n  ' + '\n  '.join(order_parts)) if order_parts else '없음'

        risk_on = signal.get('stock', {}).get('risk_on', True)
        canary_lines = [f"Stock canary: {'ON 🟢 (공격)' if risk_on else 'OFF 🔴 (방어)'}"]

        status = {
            'schema': 'V24',
            'mode': mode,
            '리밸 결과': '✅ 완료' if result.get('completed') else '⏳ 미완료',
            '잔여 편차': f'{max_diff:.1%}' if max_diff is not None else 'N/A',
            '총자산': f'${total_usd:,.0f} (환율 {fx:,.0f})',
        }
        # V24: 정상 보고 silent — Daily Report 09:15 가 통합 보고
        target_norm = {('Cash' if k == 'Cash' else k): v for k, v in target.items()}
        log(f'주식 보고 (silent): target={target_norm} max_diff={max_diff} mode={mode}')

    # 6. 저장
    state['last_action'] = 'executor'
    state['last_trade_date'] = datetime.now().strftime('%Y-%m-%d %H:%M')
    if not dry_run:
        save_json(TRADE_STATE_FILE, state)
    # 마무리 한 줄 — 이미 읽은 값을 쓴다. 매매가 없었으면 2.5 의 사전 조회가 지금 값이고,
    # 매매가 있었으면 주문 뒤에 읽은 값이 지금 값이다. 체결 확인이 실패한 회차에서만
    # 로그 한 줄을 위해 다시 묻는다 — 저장은 이미 위에서 끝났고, 이 조회가 성공해도
    # 체결 확인 상태를 바꾸지 않는다(rebalancing_needed 는 True 로 남는다). 그 사실이
    # 보이도록 로그에 표시를 붙인다.
    _tag = ''
    try:
        if isinstance(_bal_after, tuple) and len(_bal_after) >= 3:
            _h, _t, _e = _bal_after[0], _bal_after[1], _bal_after[2]
        elif _rebal_ran and not _stale_abort:
            _h, _t, _e = api.get_balance()
            _tag = ' | 체결 확인 불가(아래 잔고는 마무리 로그용 재조회, 판정 안 바꿈)'
        else:
            _h, _t, _e = _bal_pre
            if _stale_abort:
                _tag = ' | 계좌값 만료로 주문 보류(아래 잔고는 사전 조회값)'
        _elapsed = time.time() - _t0
        log(f'주식 executor 완료 ({_elapsed:.1f}s){_tag} | 총자산: ${_t:,.0f} (환율 {_e:,.0f}) | 잔고: {_h}')
    except Exception:
        _elapsed = time.time() - _t0
        log(f'주식 executor 완료 ({_elapsed:.1f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    hg = HealthGuard(name='stock')
    lock_reason = hg.is_locked()
    if lock_reason:
        log(f'🔒 health lock 활성 — {lock_reason}. 수동 해제 필요 (rm {hg.lock_file})')
        send_telegram(f'🔒 주식 lock 활성 — 수동 확인 후 rm {hg.lock_file}')
        sys.exit(0)
    try:
        run_once(dry_run=args.dry_run)
        if not args.dry_run:
            hg.record_success()
    except BalanceUnavailable as e:
        # 조회 불능은 고장이 아니라 쉬는 것이다 — health streak 을 올리지 않는다.
        log(f'⛔ 계좌 조회 불가로 이번 회차 종료: {e}')
        send_telegram(f'⛔ 주식: 계좌 조회가 안 돼 이번 회차를 쉰다 ({e}). 다음 회차에 다시 본다.')
        sys.exit(0)
    except UnknownExecutionError as e:
        import traceback
        log(f'⚠️ UNKNOWN_EXECUTION: {e}\n{traceback.format_exc()}')
        hg.lock(f'UNKNOWN_EXECUTION: {e}')
        send_telegram(f'⚠️ [주식] UNKNOWN_EXECUTION — 중복 주문 위험. 수동 확인 후 rm {hg.lock_file}: {e}')
        sys.exit(3)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        log(f'🚨 FATAL ERROR: {e}')
        log(err)
        streak = hg.record_abort(str(e)) if not args.dry_run else 0
        send_telegram(f'🚨 [주식] executor 비정상 종료 (streak={streak}): {e}')
