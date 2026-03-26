#!/usr/bin/env python3
"""코인 Executor — 단일 모드 run_once().

설계 원칙:
  - signal_state.json 읽기 (recommend 출력)
  - coin_trade_state.json 읽기/쓰기
  - 전략 신호 계산 안 함 (카나리/Z-score/헬스)
  - signal_state 수정 안 함

이벤트 우선순위:
  1. 가드: DD Exit / Blacklist
  2. 카나리 플립 → 전 트랜치 즉시 전환
  3. PFD (플립 후 5달력일 재평가)
  4. 앵커 체크 (해당 트랜치에 signal.picks 반영)
  5. Delta 매매

Usage:
  python3 executor_coin.py                # 실행
  python3 executor_coin.py --dry-run      # 주문 없이 로그만
"""

import json, os, time, argparse, logging, uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import pyupbit
import requests

try:
    from config import UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    UPBIT_ACCESS_KEY = os.environ.get('UPBIT_ACCESS_KEY', '')
    UPBIT_SECRET_KEY = os.environ.get('UPBIT_SECRET_KEY', '')
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

# ═══ 상수 ═══
SIGNAL_STATE_FILE = 'signal_state.json'
TRADE_STATE_FILE = 'coin_trade_state.json'
LOG_FILE = 'executor_coin.log'

ANCHOR_DAYS = (1, 11, 21)
CASH_BUFFER_DEFAULT = 0.02
MIN_ORDER_KRW = 5000
MAX_ORDER_ATTEMPTS = 5
ORDER_WAIT_SEC = 5
LIMIT_PRICE_SLIP = 0.003    # ±0.3% 지정가

# 가드 임계값
DD_EXIT_THRESHOLD = 0.75     # peak_60d × 0.75
BL_THRESHOLD = 0.85          # prev_close × 0.85
BL_EXCLUDE_DAYS = 7

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
        requests.post(url, data={'chat_id': TELEGRAM_CHAT_ID, 'text': f'[코인] {msg}'}, timeout=5)
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


# ═══ 거래소 API ═══
class UpbitAPI:
    def __init__(self, dry_run=False):
        self.upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
        self.dry_run = dry_run

    def get_balance(self) -> Dict[str, float]:
        """잔고 조회 → {티커: KRW 평가액}. 현금은 'KRW' 키."""
        result = {}
        balances = self.upbit.get_balances()
        if not isinstance(balances, list):
            log(f'  잔고 조회 실패: {balances}')
            return result
        for b in balances:
            try:
                currency = b['currency']
                qty = float(b['balance']) + float(b.get('locked', 0))
                if currency == 'KRW':
                    result['KRW'] = qty
                elif qty > 0:
                    ticker = f'KRW-{currency}'
                    try:
                        price = pyupbit.get_current_price(ticker)
                        if price and isinstance(price, (int, float)):
                            result[currency] = qty * price
                    except Exception:
                        pass  # 거래 불가 코인 무시
            except Exception:
                continue
        return result

    def get_current_price(self, coin: str) -> float:
        """현재가 조회."""
        try:
            ticker = f'KRW-{coin}'
            price = pyupbit.get_current_price(ticker)
            return float(price) if price and isinstance(price, (int, float)) else 0
        except Exception:
            return 0

    def cancel_all(self):
        """미체결 전량 취소."""
        try:
            orders = self.upbit.get_order('', state='wait')
            if isinstance(orders, list) and orders:
                for o in orders:
                    if isinstance(o, dict) and 'uuid' in o:
                        self.upbit.cancel_order(o['uuid'])
                log(f'  미체결 {len(orders)}건 취소')
            elif isinstance(orders, dict) and orders.get('error'):
                pass  # 미체결 없음
        except Exception as e:
            log(f'  미체결 취소 오류: {e}')

    def sell(self, coin: str, qty: float) -> bool:
        """시장가 매도."""
        if self.dry_run:
            log(f'  [DRY] 매도 {coin} qty={qty:.6f}')
            return True
        try:
            ticker = f'KRW-{coin}'
            result = self.upbit.sell_market_order(ticker, qty)
            if result and 'uuid' in str(result):
                log(f'  매도 {coin} qty={qty:.6f} → {result}')
                return True
            send_telegram(f'⚠️ [코인] 매도 실패: {coin}')
            log(f'  매도 실패 {coin}: {result}')
            return False
        except Exception as e:
            log(f'  매도 오류 {coin}: {e}')
            return False

    def buy(self, coin: str, krw_amount: float) -> bool:
        """지정가 매수 (현재가 + 0.3%)."""
        if self.dry_run:
            log(f'  [DRY] 매수 {coin} ₩{krw_amount:,.0f}')
            return True
        try:
            ticker = f'KRW-{coin}'
            price = pyupbit.get_current_price(ticker)
            if not price:
                return False
            limit_price = price * (1 + LIMIT_PRICE_SLIP)
            qty = krw_amount / limit_price
            result = self.upbit.buy_limit_order(ticker, limit_price, qty)
            if result and 'uuid' in str(result):
                log(f'  매수 {coin} ₩{krw_amount:,.0f} @ {limit_price:,.0f} → {result}')
                time.sleep(ORDER_WAIT_SEC)
                # 미체결 취소
                try:
                    self.upbit.cancel_order(result['uuid'])
                except Exception:
                    pass
                return True
            send_telegram(f'⚠️ [코인] 매수 실패: {coin}')
            log(f'  매수 실패 {coin}: {result}')
            return False
        except Exception as e:
            log(f'  매수 오류 {coin}: {e}')
            return False


# ═══ 핵심 로직 ═══

def check_signal_freshness(signal: dict) -> bool:
    """signal이 24시간 이내인지 확인."""
    updated = signal.get('meta', {}).get('updated_at', '')
    if not updated:
        return False
    try:
        dt = datetime.strptime(updated, '%Y-%m-%d %H:%M')
        return (datetime.now() - dt).total_seconds() < STALE_SIGNAL_HOURS * 3600
    except Exception:
        return False


def filter_exclusions(picks, weights, exclusions):
    """exclusions에 있는 종목 제거 + Cash로 전환 (공통)."""
    filtered_picks = [p for p in picks if p not in exclusions]
    filtered_weights = {k: v for k, v in weights.items() if k not in exclusions or k == 'Cash'}
    removed_w = sum(v for k, v in weights.items() if k in exclusions and k != 'Cash')
    if removed_w > 0:
        filtered_weights['Cash'] = filtered_weights.get('Cash', 0) + removed_w
    return filtered_picks, filtered_weights


def check_guards(signal: dict, api: UpbitAPI, state: dict) -> Dict:
    """DD Exit / Blacklist 체크. 현재가 vs signal 기준값."""
    result = {'dd_exits': [], 'bl_exits': []}
    guard_refs = signal.get('coin', {}).get('guard_refs', {})
    exclusions = state.get('guard_state', {}).get('exclusions', {})

    # 현재 보유 코인 확인 (실잔고 기준 — state가 아닌 거래소 잔고)
    balance = api.get_balance()
    held_coins = set(k for k, v in balance.items() if k != 'KRW' and v > MIN_ORDER_KRW)

    for coin in held_coins:
        if coin in exclusions:
            continue  # 이미 제외됨
        refs = guard_refs.get(coin, {})
        if not refs:
            continue

        current_price = api.get_current_price(coin)
        if not current_price:
            continue

        peak_60d = refs.get('peak_60d', 0)
        prev_close = refs.get('prev_close', 0)

        log(f'    가드 체크 {coin}: 현재가={current_price:,.0f}, peak_60d={peak_60d:,.0f}, prev_close={prev_close:,.0f}')
        # DD Exit: 현재가 ≤ peak_60d × 0.75
        if peak_60d > 0 and current_price <= peak_60d * DD_EXIT_THRESHOLD:
            log(f'  🚨 DD Exit: {coin} ₩{current_price:,.0f} ≤ peak ₩{peak_60d:,.0f} × 0.75')
            result['dd_exits'].append(coin)

        # Blacklist: 현재가 ≤ prev_close × 0.85
        elif prev_close > 0 and current_price <= prev_close * BL_THRESHOLD:
            log(f'  🚨 Blacklist: {coin} ₩{current_price:,.0f} ≤ prev ₩{prev_close:,.0f} × 0.85')
            result['bl_exits'].append(coin)

    return result


def handle_guard_exits(guard_result: dict, state: dict, api: UpbitAPI):
    """DD/BL 발동 종목 매도 + exclusions 기록."""
    today = datetime.now().strftime('%Y-%m-%d')
    exclusions = state.setdefault('guard_state', {}).setdefault('exclusions', {})

    for coin in guard_result['dd_exits']:
        balance = api.get_balance()
        if coin in balance and balance[coin] > MIN_ORDER_KRW:
            raw_bal = api.upbit.get_balance(coin)
            qty = float(raw_bal) if raw_bal and not isinstance(raw_bal, (dict, list, tuple)) else 0
            if qty > 0:
                api.sell(coin, qty)
                send_telegram(f'DD Exit: {coin} 매도')
        exclusions[coin] = {'reason': 'dd', 'until_date': None}
        # 트랜치에서 해당 코인 → Cash
        for tr in state.get('tranches', {}).values():
            if coin in tr.get('picks', []):
                tr['picks'] = [p for p in tr['picks'] if p != coin]
                w = tr['weights'].pop(coin, 0)
                tr['weights']['Cash'] = tr['weights'].get('Cash', 0) + w
        state['rebalancing_needed'] = True

    for coin in guard_result['bl_exits']:
        balance = api.get_balance()
        if coin in balance and balance[coin] > MIN_ORDER_KRW:
            raw_bal = api.upbit.get_balance(coin)
            qty = float(raw_bal) if raw_bal and not isinstance(raw_bal, (dict, list, tuple)) else 0
            if qty > 0:
                api.sell(coin, qty)
                send_telegram(f'Blacklist: {coin} 매도 (7일 제외)')
        until = (datetime.now() + timedelta(days=BL_EXCLUDE_DAYS)).strftime('%Y-%m-%d')
        exclusions[coin] = {'reason': 'bl', 'until_date': until}
        for tr in state.get('tranches', {}).values():
            if coin in tr.get('picks', []):
                tr['picks'] = [p for p in tr['picks'] if p != coin]
                w = tr['weights'].pop(coin, 0)
                tr['weights']['Cash'] = tr['weights'].get('Cash', 0) + w
        state['rebalancing_needed'] = True


def clean_expired_exclusions(state: dict):
    """만료된 exclusions 제거."""
    exclusions = state.get('guard_state', {}).get('exclusions', {})
    today = datetime.now().strftime('%Y-%m-%d')
    expired = [coin for coin, info in exclusions.items()
               if info.get('until_date') and info['until_date'] <= today]
    for coin in expired:
        del exclusions[coin]
        log(f'  Exclusion 해제: {coin}')


def check_canary_flip(signal: dict, state: dict) -> bool:
    """카나리 플립 감지 → 전 트랜치 즉시 전환."""
    signal_risk_on = signal.get('coin', {}).get('risk_on', True)
    prev_risk_on = state.get('prev_risk_on')

    if prev_risk_on is not None and signal_risk_on != prev_risk_on:
        log(f'  🔄 카나리 플립: {prev_risk_on} → {signal_risk_on}')
        if signal_risk_on:
            # Risk-Off → On: 전 트랜치를 signal.picks로
            picks = signal.get('coin', {}).get('picks', [])
            weights = signal.get('coin', {}).get('weights', {})
        else:
            # Risk-On → Off: 전 트랜치를 현금으로
            picks = []
            weights = {'Cash': 1.0}

        exclusions = state.get('guard_state', {}).get('exclusions', {})
        filtered_picks, filtered_weights = filter_exclusions(picks, weights, exclusions)
        for tr in state.get('tranches', {}).values():
            tr['picks'] = list(filtered_picks)
            tr['weights'] = dict(filtered_weights)

        state['prev_risk_on'] = signal_risk_on
        state['flip_date'] = datetime.now().strftime('%Y-%m-%d')
        state['pfd_done'] = False
        state['rebalancing_needed'] = True
        send_telegram(f'카나리 플립: {"Risk-On 🟢" if signal_risk_on else "Risk-Off 🔴"}')
        return True

    state['prev_risk_on'] = signal_risk_on
    return False


def check_pfd(signal: dict, state: dict) -> bool:
    """PFD: 플립 후 5달력일 재평가."""
    flip_date = state.get('flip_date')
    pfd_done = state.get('pfd_done', True)

    if not flip_date or pfd_done:
        return False

    try:
        flip_dt = datetime.strptime(flip_date, '%Y-%m-%d')
        if (datetime.now() - flip_dt).days >= 5:
            log('  📋 PFD: 플립 후 5일 경과 → 전 트랜치 재평가')
            picks = signal.get('coin', {}).get('picks', [])
            weights = signal.get('coin', {}).get('weights', {})
            risk_on = signal.get('coin', {}).get('risk_on', True)

            exclusions = state.get('guard_state', {}).get('exclusions', {})
            if risk_on:
                filtered_picks, filtered_weights = filter_exclusions(picks, weights, exclusions)
                for tr in state.get('tranches', {}).values():
                    tr['picks'] = list(filtered_picks)
                    tr['weights'] = dict(filtered_weights)
            # risk_off든 on이든 pfd_done=True (재평가 1회 완료)
            state['pfd_done'] = True
            state['rebalancing_needed'] = True
            return True
    except Exception:
        pass
    return False


def check_anchors(signal: dict, state: dict):
    """앵커일 체크 → 해당 트랜치에 signal.picks 반영."""
    today = datetime.now().day
    this_month = datetime.now().strftime('%Y-%m')
    risk_on = signal.get('coin', {}).get('risk_on', True)

    for day in ANCHOR_DAYS:
        day_str = str(day)
        tr = state.setdefault('tranches', {}).setdefault(day_str, {})
        if today >= day and tr.get('anchor_month', '') < this_month:
            if risk_on:
                picks = signal.get('coin', {}).get('picks', [])
                weights = signal.get('coin', {}).get('weights', {})
            else:
                picks = []
                weights = {'Cash': 1.0}

            exclusions = state.get('guard_state', {}).get('exclusions', {})
            filtered_picks, filtered_weights = filter_exclusions(picks, weights, exclusions)

            tr['picks'] = filtered_picks
            tr['weights'] = filtered_weights
            tr['anchor_month'] = this_month
            state['rebalancing_needed'] = True
            log(f'  📅 앵커 Day {day}: {filtered_picks} weights={filtered_weights} (month={this_month})')


def merge_tranches(state: dict) -> Dict[str, float]:
    """전 트랜치 merge → 최종 target (비중 합산)."""
    tranches = state.get('tranches', {})
    n = len(tranches) if tranches else 1
    merged = {}
    for tr in tranches.values():
        for ticker, w in tr.get('weights', {}).items():
            merged[ticker] = merged.get(ticker, 0) + w / n
    return merged


def execute_delta(target: Dict[str, float], api: UpbitAPI, state: dict):
    """target vs 현재 잔고 비교 → Delta 매매."""
    balance = api.get_balance()
    total = sum(balance.values())
    if total <= 0:
        log('  잔고 없음')
        return

    cash_available = balance.get('KRW', 0)

    # 현재 비중
    current_weights = {k: v / total for k, v in balance.items() if k != 'KRW'}

    # Delta 계산 (매도 먼저)
    sells = []
    buys = []

    for ticker, target_w in target.items():
        if ticker == 'Cash':
            continue
        current_w = current_weights.get(ticker, 0)
        delta_w = target_w - current_w
        delta_krw = delta_w * total

        if delta_krw < -MIN_ORDER_KRW:
            sells.append((ticker, abs(delta_krw)))
        elif delta_krw > MIN_ORDER_KRW:
            buys.append((ticker, delta_krw))

    # 보유 중이지만 target에 없는 종목 → 매도
    for ticker in list(current_weights.keys()):
        if ticker not in target or target.get(ticker, 0) == 0:
            val = balance.get(ticker, 0)
            if val > MIN_ORDER_KRW:
                sells.append((ticker, val))

    # 매도 실행 (부분 매도: delta 금액만큼만)
    for ticker, sell_krw in sells:
        current_price = api.get_current_price(ticker)
        if current_price <= 0:
            continue
        raw_bal = api.upbit.get_balance(ticker) if not api.dry_run else 0
        total_qty = float(raw_bal) if raw_bal and not isinstance(raw_bal, (dict, list, tuple)) else 0
        # target에 없으면 전량, 있으면 delta만큼만
        if ticker not in target or target.get(ticker, 0) == 0:
            sell_qty = total_qty  # 전량 매도
        else:
            sell_qty = min(total_qty, sell_krw / current_price)  # 부분 매도
        if sell_qty > 0 or api.dry_run:
            log(f'  매도: {ticker} qty={sell_qty:.6f} (₩{sell_qty * current_price:,.0f})')
            api.sell(ticker, sell_qty)

    time.sleep(1)

    # 매수 실행 (매도 후 현금 재확인)
    if buys:
        balance = api.get_balance()
        cash_available = balance.get('KRW', 0) * (1 - CASH_BUFFER)
        total_buy = sum(b[1] for b in buys)

        for ticker, amount in buys:
            # 현금 부족 시 비중 비례 축소
            actual = amount * min(1.0, cash_available / max(total_buy, 1))
            if actual >= MIN_ORDER_KRW:
                log(f'  매수: {ticker} ₩{actual:,.0f}')
                api.buy(ticker, actual)
                cash_available -= actual

    # 목표 달성 여부
    balance = api.get_balance()
    total = sum(balance.values())
    if total > 0:
        max_diff = 0
        for ticker, target_w in target.items():
            if ticker == 'Cash':
                continue
            current_w = balance.get(ticker, 0) / total
            max_diff = max(max_diff, abs(target_w - current_w))
        if max_diff < 0.05:  # ±5% 이내
            state['rebalancing_needed'] = False
            log('  ✅ 목표 달성 (±5% 이내)')
            send_telegram(f'✅ [코인] 리밸런싱 완료')
        else:
            log(f'  ⏳ 미달 (max diff={max_diff:.1%}), 다음 실행에서 재시도')


# ═══ run_once ═══

def run_once(dry_run=False):
    """단일 실행. cron에서 매 30분마다 호출."""
    global RUN_ID, CASH_BUFFER
    RUN_ID = uuid.uuid4().hex
    _t0 = time.time()
    log('=' * 50)
    log('코인 executor 시작')

    # 1. 데이터 읽기
    signal = load_json(SIGNAL_STATE_FILE)
    state = load_json(TRADE_STATE_FILE)
    api = UpbitAPI(dry_run=dry_run)

    # cash_buffer: state에서 읽기, 없으면 기본값 2%
    CASH_BUFFER = state.get('cash_buffer', CASH_BUFFER_DEFAULT)
    log(f'  cash_buffer: {CASH_BUFFER:.0%}')

    # ── 디버그 로그: 입력 데이터 ──
    log(f'  signal: {json.dumps(signal, ensure_ascii=False, default=str)[:500]}')
    log(f'  state: {json.dumps(state, ensure_ascii=False, default=str)[:500]}')
    
    if not signal:
        log('  signal_state.json 없음 — 스킵')
        return
    if not state.get('tranches'):
        log('  coin_trade_state.json 트랜치 없음 — 스킵')
        return

    # 첫 실행 초기화: prev_risk_on이 없으면 현재 signal로 설정
    if state.get('prev_risk_on') is None:
        state['prev_risk_on'] = signal.get('coin', {}).get('risk_on', True)
        log('  첫 실행: prev_risk_on 초기화')

    # signal 신선도 + 경과 시간
    _su = signal.get('meta', {}).get('updated_at', '')
    try:
        _age = (datetime.now() - datetime.strptime(_su, '%Y-%m-%d %H:%M')).total_seconds() / 3600
        log(f'  signal age: {_age:.1f}h (updated: {_su})')
    except Exception:
        log(f'  signal age: ? (updated: {_su})')
    is_fresh = check_signal_freshness(signal)
    if not is_fresh:
        log('  ⚠️ signal이 24시간 이상 오래됨 — 가드만 체크, 매매 보류')
        send_telegram('⚠️ signal_state 갱신 안 됨 (24시간 초과)')

    # 2. 미체결 취소
    api.cancel_all()

    # 3. 만료된 exclusions 정리
    clean_expired_exclusions(state)

    log('  [2] 가드 체크')
    # 4. 가드 체크 (최우선)
    guards = check_guards(signal, api, state)
    if guards['dd_exits'] or guards['bl_exits']:
        handle_guard_exits(guards, state, api)
        save_json(TRADE_STATE_FILE, state)
        log('코인 executor 완료 (가드 발동)')
        return

    if not is_fresh:
        save_json(TRADE_STATE_FILE, state)
        log('코인 executor 완료 (stale signal)')
        return

    log('  [3] 카나리 플립 체크')
    # 5. 카나리 플립
    flipped = check_canary_flip(signal, state)

    # 6. PFD
    if not flipped:
        check_pfd(signal, state)

    log('  [5] 앵커 체크')
    # 7. 앵커 체크
    check_anchors(signal, state)

    log('  [6] Merge + Delta')
    # 8. Merge + Delta 매매
    if not state.get('rebalancing_needed', False):
        log('  매매 스킵: rebalancing_needed=false')
    if state.get('rebalancing_needed', False):
        target = merge_tranches(state)
        log(f'  Target: {target}')
        send_telegram(f'📊 [코인] 리밸런싱 시작: {target}')
        # 디버그: 현재 잔고 상세
        _bal = api.get_balance() if hasattr(api, 'get_balance') else {}
        log(f'  Balance: {_bal}')
        execute_delta(target, api, state)

    # 9. 저장
    state['last_action'] = 'executor'
    state['last_trade_date'] = datetime.now().strftime('%Y-%m-%d')
    save_json(TRADE_STATE_FILE, state)
    # 종료 시 자산 요약
    try:
        _final_bal = api.get_balance()
        _total = sum(_final_bal.values())
        _elapsed = time.time() - _t0
        log(f'코인 executor 완료 ({_elapsed:.1f}s) | 총자산: ₩{_total:,.0f} | 잔고: {_final_bal}')
    except Exception:
        _elapsed = time.time() - _t0
        log(f'코인 executor 완료 ({_elapsed:.1f}s)')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dry-run', action='store_true', help='주문 없이 로그만')
    args = parser.parse_args()
    try:
        run_once(dry_run=args.dry_run)
    except Exception as e:
        import traceback
        err = traceback.format_exc()
        log(f'🚨 FATAL ERROR: {e}')
        log(err)
        send_telegram(f'🚨 [코인] executor 비정상 종료: {e}')
