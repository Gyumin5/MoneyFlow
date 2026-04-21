#!/usr/bin/env python3
"""Cap Defend 코인 현물 Executor.

구조:
  - 신호: coin_live_engine.compute_live_targets
          (Binance spot kline → V21 D봉 3멤버 D_SMA50/150/100 1/3씩 EW 앙상블)
  - 체결: Upbit KRW (pyupbit)
  - 상태: trade_state.json

실행 순서:
  1. flock /tmp/coin_executor.lock
  2. UpbitAPI + 미체결 취소
  3. 유의종목/거래정지 감지 → 즉시 시장가 청산 (freshness 무관, try/except + 3회 재시도 + permanent_block)
  4. compute_live_targets() 호출 (엔진 내부에서 freshness / 카나리 / 갭 / 앙상블 처리)
  5. all_fresh=False 면 리밸런싱 스킵 (청산만 수행)
  6. Cash buffer 2% 적용
  7. Notional cap 비활성 (필요 시 상수로 재활성)
  8. Delta 매매 (매도 → 매수), dust <5000 KRW → 전량 매도
  9. 상태 저장 + 텔레그램 사전/사후 알림

Usage:
  python3 executor_coin.py
  python3 executor_coin.py --dry-run
"""

from __future__ import annotations

import argparse
import fcntl
import logging
import math
import os
import sys
import time
import traceback
from typing import Dict, List, Optional, Tuple

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pyupbit
import requests
from pyupbit import request_api as pyupbit_request_api

from common.io import load_json, save_json
from common.notify import send_telegram as _send_tg

try:
    from common.logging_utils import setup_file_logger, make_log_fn
except ImportError:
    setup_file_logger = None  # type: ignore
    make_log_fn = None  # type: ignore

import coin_live_engine as cle

try:
    from config import (
        UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY,
        TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID,
    )
except ImportError:
    UPBIT_ACCESS_KEY = os.environ.get('UPBIT_ACCESS_KEY', '')
    UPBIT_SECRET_KEY = os.environ.get('UPBIT_SECRET_KEY', '')
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')


# ═══ 상수 ═══
STATE_FILE = 'trade_state.json'
LOCK_FILE = '/tmp/coin_executor.lock'
LOG_FILE = 'executor_coin.log'
CACHE_DIR = os.path.dirname(os.path.abspath(__file__))

CASH_BUFFER_DEFAULT = 0.02          # 총자산의 2%는 KRW 유지
NOTIONAL_CAP_FRACTION = 1.00        # 1.0이면 비활성
MIN_ORDER_KRW = 5000                # 업비트 최소주문
DUST_KRW = 5000                     # 이보다 작은 잔여는 전량 매도
LIMIT_PRICE_SLIP = 0.003            # 매수 지정가 +0.3%
ORDER_WAIT_SEC = 5                  # 매수 후 미체결 취소 대기
LIQUIDATION_MAX_RETRIES = 3


def _patch_pyupbit_remaining_req_parser():
    """pyupbit Remaining-Req 파싱 실패를 무해화.

    Upbit 응답 헤더 형식이 약간 달라질 때 pyupbit가 예외를 던지고
    private API 결과를 None으로 삼켜 버리는 문제가 있어, 파싱 실패 시에도
    기본 구조를 반환하도록 패치한다.
    """
    orig_parse = pyupbit_request_api._parse

    def _safe_parse(remaining_req: str):
        try:
            return orig_parse(remaining_req)
        except Exception:
            return {"group": "unknown", "min": 0, "sec": 0}

    pyupbit_request_api._parse = _safe_parse


_patch_pyupbit_remaining_req_parser()


# ═══ 로거 ═══
LOG_PATH = os.path.join(CACHE_DIR, LOG_FILE)
if setup_file_logger and make_log_fn:
    _logger = setup_file_logger(LOG_FILE, LOG_PATH)
    _run_id_ref = ['']
    log = make_log_fn(_logger, _run_id_ref)
else:
    logging.basicConfig(level=logging.INFO,
                        format='%(asctime)s [%(levelname)s] %(message)s')
    _logger = logging.getLogger('executor_coin')
    def log(msg: str, level: str = 'info'):  # type: ignore
        getattr(_logger, level, _logger.info)(msg)


# ═══ 텔레그램 버퍼 ═══
_tg_events: List[str] = []

def _tg(msg: str):
    _tg_events.append(msg)

def _flush_telegram(dry_run: bool = False):
    if not _tg_events:
        return
    prefix = '[DRY] ' if dry_run else ''
    payload = prefix + '[코인]\n' + '\n'.join(_tg_events)
    try:
        _send_tg(TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID, payload)
    except Exception as e:
        log(f'텔레그램 전송 실패: {e}')
    _tg_events.clear()


# ═══ 호가 단위 ═══
def _round_price_up(price: float) -> float:
    if price >= 2_000_000: tick = 1000
    elif price >= 1_000_000: tick = 500
    elif price >= 500_000: tick = 100
    elif price >= 100_000: tick = 50
    elif price >= 10_000: tick = 10
    elif price >= 1_000: tick = 5
    elif price >= 100: tick = 1
    elif price >= 10: tick = 0.1
    elif price >= 1: tick = 0.01
    else: tick = 0.001
    return math.ceil(price / tick) * tick


# ═══ Upbit API 래퍼 ═══
class UpbitAPI:
    def __init__(self, dry_run: bool = False):
        self.upbit = pyupbit.Upbit(UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY)
        self.dry_run = dry_run
        self._krw_markets: Optional[set] = None

    def _get_krw_markets(self) -> set:
        if self._krw_markets is None:
            try:
                tickers = pyupbit.get_tickers(fiat='KRW')
                self._krw_markets = set(tickers) if tickers else set()
            except Exception as e:
                log(f'  Upbit 마켓 조회 오류: {e}')
                self._krw_markets = set()
        return self._krw_markets

    def get_balance(self) -> Dict[str, float]:
        """{currency: KRW 평가액}. 현금은 'KRW' 키."""
        result: Dict[str, float] = {}
        try:
            balances = self.upbit.get_balances()
        except Exception as e:
            log(f'  잔고 조회 오류: {e}')
            return result
        if not isinstance(balances, list):
            log(f'  잔고 조회 실패: {balances}')
            return result
        coin_rows: List[Dict] = []
        for b in balances:
            try:
                currency = b['currency']
                qty = float(b['balance']) + float(b.get('locked', 0))
                if currency == 'KRW':
                    result['KRW'] = qty
                elif qty > 0:
                    coin_rows.append({'currency': currency, 'qty': qty,
                                       'avg_buy': float(b.get('avg_buy_price', 0) or 0)})
            except Exception:
                continue

        if coin_rows:
            krw_markets = self._get_krw_markets()
            all_tickers = [f'KRW-{r["currency"]}' for r in coin_rows]
            valid_tickers = [t for t in all_tickers if not krw_markets or t in krw_markets]
            price_map: Dict[str, float] = {}
            if valid_tickers:
                try:
                    prices = pyupbit.get_current_price(valid_tickers)
                    if isinstance(prices, dict):
                        for t, p in prices.items():
                            if p and isinstance(p, (int, float)):
                                price_map[t] = float(p)
                    elif isinstance(prices, (int, float)) and len(valid_tickers) == 1:
                        price_map[valid_tickers[0]] = float(prices)
                except Exception as e:
                    log(f'  가격 일괄조회 오류: {e}')

            for row in coin_rows:
                ticker = f'KRW-{row["currency"]}'
                price_val = price_map.get(ticker, 0.0)
                if price_val > 0:
                    result[row['currency']] = row['qty'] * price_val
                else:
                    fallback = row['qty'] * row['avg_buy'] if row['avg_buy'] > 0 else 0.0
                    result[row['currency']] = fallback
                    if krw_markets and ticker not in krw_markets:
                        pass
                    else:
                        log(f'  가격조회 실패 {row["currency"]}: qty={row["qty"]} fallback={fallback:,.0f}')
        return result

    def get_coin_qty(self, coin: str) -> float:
        """코인 수량 (locked 포함)."""
        try:
            balances = self.upbit.get_balances()
            if not isinstance(balances, list):
                return 0.0
            for b in balances:
                if b.get('currency') == coin and b.get('unit_currency') == 'KRW':
                    return float(b.get('balance', 0) or 0) + float(b.get('locked', 0) or 0)
            return 0.0
        except Exception:
            return 0.0

    def get_current_price(self, coin: str) -> float:
        try:
            price = pyupbit.get_current_price(f'KRW-{coin}')
            return float(price) if price and isinstance(price, (int, float)) else 0.0
        except Exception:
            return 0.0

    def cancel_all(self):
        try:
            orders = self.upbit.get_order('', state='wait')
        except Exception as e:
            log(f'  미체결 조회 오류: {e}')
            return
        if isinstance(orders, list) and orders:
            for o in orders:
                if isinstance(o, dict) and 'uuid' in o:
                    try:
                        self.upbit.cancel_order(o['uuid'])
                    except Exception as e:
                        log(f'  취소 오류 {o.get("uuid")}: {e}')
            log(f'  미체결 {len(orders)}건 취소')

    def sell_market(self, coin: str, qty: float) -> bool:
        if qty <= 0:
            return True
        if self.dry_run:
            log(f'  [DRY] 시장가 매도 {coin} qty={qty:.8f}')
            return True
        try:
            ticker = f'KRW-{coin}'
            result = self.upbit.sell_market_order(ticker, qty)
            if result and 'uuid' in str(result):
                log(f'  매도 {coin} qty={qty:.8f} → ok')
                return True
            log(f'  매도 실패 {coin}: {result}')
            return False
        except Exception as e:
            log(f'  매도 오류 {coin}: {e}')
            return False

    def buy_limit(self, coin: str, krw_amount: float) -> bool:
        if krw_amount < MIN_ORDER_KRW:
            return True
        if self.dry_run:
            log(f'  [DRY] 지정가 매수 {coin} ₩{krw_amount:,.0f}')
            return True
        try:
            ticker = f'KRW-{coin}'
            price = pyupbit.get_current_price(ticker)
            if not price or not isinstance(price, (int, float)):
                log(f'  매수 실패 {coin}: 현재가 조회 실패')
                return False
            limit_price = _round_price_up(float(price) * (1 + LIMIT_PRICE_SLIP))
            qty = krw_amount / limit_price
            result = self.upbit.buy_limit_order(ticker, limit_price, qty)
            if not isinstance(result, dict) or 'uuid' not in result:
                log(f'  매수 실패 {coin}: {result}')
                return False
            uuid = result['uuid']
            log(f'  매수 {coin} ₩{krw_amount:,.0f} @ {limit_price:,.0f}')
            time.sleep(ORDER_WAIT_SEC)
            try:
                self.upbit.cancel_order(uuid)
            except Exception:
                pass
            return True
        except Exception as e:
            log(f'  매수 오류 {coin}: {e}')
            return False


# ═══ 유의종목/거래정지 청산 ═══
def detect_warning_suspended(upbit_status: Dict[str, Dict]) -> List[str]:
    """투자유의(warning) 또는 상장폐지(!listed) coin 목록.
    투자주의(caution)는 단기 급변동 알림이라 청산 대상에서 제외.
    스키마: {coin: {warning: bool, caution: bool, listed: bool}}."""
    out: List[str] = []
    for coin, info in upbit_status.items():
        warning = bool(info.get('warning', False))
        listed = bool(info.get('listed', True))
        if warning or not listed:
            out.append(coin)
    return out


def liquidate_coins(coins: List[str], reason: str, api: UpbitAPI,
                    state: Dict) -> Tuple[List[str], List[str]]:
    """시장가 전량 매도 (3회 재시도). 실패 시 permanent_block 등록.
    Returns: (liquidated, failed)."""
    permanent = state.setdefault('permanent_block', [])
    liquidated: List[str] = []
    failed: List[str] = []
    for coin in coins:
        qty = api.get_coin_qty(coin)
        if qty <= 0:
            continue
        success = False
        last_err: Optional[str] = None
        for attempt in range(1, LIQUIDATION_MAX_RETRIES + 1):
            try:
                ok = api.sell_market(coin, qty)
                if ok:
                    success = True
                    break
                last_err = 'sell returned False'
            except Exception as e:
                last_err = str(e)
            time.sleep(2 * attempt)
        if success:
            liquidated.append(coin)
            log(f'  🧹 {reason} 청산: {coin} qty={qty:.8f}')
            _tg(f'{reason} 청산: {coin}')
        else:
            if coin not in permanent:
                permanent.append(coin)
            failed.append(coin)
            log(f'  🚨 {reason} 청산 실패 {coin} (err={last_err}) → permanent_block 등록')
            _tg(f'🚨 {reason} 청산 실패 {coin} → 수동 확인 필요 (permanent_block)')
    return liquidated, failed


# ═══ Cash Buffer / Notional Cap ═══
def apply_cash_buffer(target: Dict[str, float], buffer_pct: float) -> Dict[str, float]:
    """최종 target × (1-buffer) 후 Cash += buffer."""
    if buffer_pct <= 0:
        return dict(target)
    out: Dict[str, float] = {}
    for k, v in target.items():
        if k == 'Cash':
            continue
        out[k] = v * (1 - buffer_pct)
    out['Cash'] = target.get('Cash', 0.0) * (1 - buffer_pct) + buffer_pct
    return out


def apply_notional_cap(target: Dict[str, float], balance: Dict[str, float],
                       total_krw: float, cap_fraction: float) -> Tuple[Dict[str, float], float]:
    """이번 실행 Σ|delta| ≤ cap_fraction으로 제한. 잔여는 다음 실행에서 자연 재계산(carryover 저장 없음)."""
    if total_krw <= 0 or cap_fraction <= 0 or cap_fraction >= 1:
        return dict(target), 0.0

    current_w: Dict[str, float] = {}
    for k, v in balance.items():
        if k == 'KRW':
            continue
        current_w[k] = v / total_krw
    current_w['Cash'] = balance.get('KRW', 0.0) / total_krw

    all_keys = (set(target.keys()) | set(current_w.keys())) - {'KRW'}
    deltas = {k: target.get(k, 0.0) - current_w.get(k, 0.0) for k in all_keys}
    gross_delta = sum(abs(v) for v in deltas.values())

    if gross_delta <= cap_fraction + 1e-9:
        return dict(target), gross_delta

    shrink = cap_fraction / gross_delta
    scaled: Dict[str, float] = {}
    for k in all_keys:
        cw = current_w.get(k, 0.0)
        dw = deltas.get(k, 0.0)
        new_w = cw + dw * shrink
        if new_w > 1e-9 or k == 'Cash':
            scaled[k] = max(new_w, 0.0)
    s = sum(scaled.values())
    if s > 0:
        scaled = {k: v / s for k, v in scaled.items()}
    return scaled, cap_fraction


# ═══ Delta 매매 ═══
def execute_delta(target: Dict[str, float], api: UpbitAPI,
                   permanent_block: List[str], dry_run: bool):
    """target vs 현재 잔고 비교 → 매도 먼저, 매수 나중.
    - dust (<5000 KRW) 잔여 → 비율 매도 대신 전량 매도
    - permanent_block 코인은 신규 매수 금지
    """
    balance = api.get_balance()
    total = sum(balance.values())
    if total <= 0:
        log('  잔고 없음')
        return

    current_value: Dict[str, float] = {k: v for k, v in balance.items() if k != 'KRW'}

    sells: List[Tuple[str, float, bool]] = []  # (coin, sell_krw, sell_all)
    buys: List[Tuple[str, float]] = []

    all_tickers = set(current_value.keys()) | set(target.keys())
    for ticker in all_tickers:
        if ticker == 'Cash':
            continue
        tgt_w = target.get(ticker, 0.0)
        cur_v = current_value.get(ticker, 0.0)
        tgt_v = tgt_w * total
        delta_v = tgt_v - cur_v

        if tgt_w <= 0 and cur_v > 0:
            sells.append((ticker, cur_v, True))
        elif delta_v < -MIN_ORDER_KRW:
            remainder = cur_v - abs(delta_v)
            if remainder < DUST_KRW:
                sells.append((ticker, cur_v, True))
            else:
                sells.append((ticker, abs(delta_v), False))
        elif delta_v > MIN_ORDER_KRW:
            if ticker in permanent_block:
                log(f'  ⚠ permanent_block {ticker} 매수 스킵')
                continue
            buys.append((ticker, delta_v))

    # 매도 — sell_all은 시세 API 장애와 무관하게 전량 청산 (fail-closed)
    for coin, sell_krw, sell_all in sells:
        qty_owned = api.get_coin_qty(coin)
        if qty_owned <= 0 and not dry_run:
            continue
        price = api.get_current_price(coin)

        if sell_all:
            if qty_owned <= 0 and dry_run:
                qty_owned = sell_krw / price if price > 0 else 0.0
            sell_qty = qty_owned
            est_krw = sell_qty * price if price > 0 else sell_krw
            if price <= 0:
                log(f'  ⚠ 전량매도 {coin}: 현재가 0 → qty 기반 시장가 강행')
        else:
            if price <= 0:
                log(f'  부분매도 스킵 {coin}: 현재가 0')
                continue
            if dry_run and qty_owned <= 0:
                qty_owned = sell_krw / price
            sell_qty = min(qty_owned, sell_krw / price)
            est_krw = sell_qty * price

        if sell_qty <= 0:
            continue
        if est_krw < MIN_ORDER_KRW:
            log(f'  매도 스킵 {coin}: est_krw=₩{est_krw:,.0f} < 최소주문 ₩{MIN_ORDER_KRW:,} (Upbit 거부)')
            continue
        log(f'  매도 {coin} qty={sell_qty:.8f} ≈ ₩{est_krw:,.0f} ({"전량" if sell_all else "부분"})')
        api.sell_market(coin, sell_qty)

    if buys:
        time.sleep(1)
        balance = api.get_balance() if not dry_run else balance
        cash_avail = balance.get('KRW', 0.0) * 0.995
        total_buy = sum(amt for _, amt in buys)
        scale = min(1.0, cash_avail / max(total_buy, 1.0))
        for coin, amt in buys:
            actual = amt * scale
            if actual < MIN_ORDER_KRW:
                continue
            log(f'  매수 {coin} ₩{actual:,.0f}')
            if api.buy_limit(coin, actual):
                cash_avail -= actual


# ═══ 사전 알림 ═══
def format_target_summary(combined: Dict[str, float],
                           member_targets: Dict[str, Dict[str, float]]) -> str:
    lines = ['목표 (앙상블):']
    for k, v in sorted(combined.items(), key=lambda kv: -kv[1]):
        if k.startswith('_'):
            continue
        if v < 1e-4 and k != 'Cash':
            continue
        lines.append(f'  {k}: {v*100:.2f}%')
    for mname, mt in member_targets.items():
        tokens = [f'{k}={v*100:.1f}%' for k, v in sorted(mt.items(), key=lambda kv: -kv[1])
                  if v > 1e-4 and not k.startswith('_')]
        lines.append(f'  [{mname}] ' + ', '.join(tokens[:6]))
    return '\n'.join(lines)


def format_delta_preview(target: Dict[str, float], balance: Dict[str, float],
                          total: float) -> str:
    if total <= 0:
        return '잔고 없음'
    lines = ['예상 Delta:']
    current_v = {k: v for k, v in balance.items() if k != 'KRW'}
    all_keys = set(current_v.keys()) | set(target.keys())
    rows = []
    for k in all_keys:
        if k == 'Cash':
            continue
        tgt_v = target.get(k, 0.0) * total
        cur_v = current_v.get(k, 0.0)
        d = tgt_v - cur_v
        if abs(d) < MIN_ORDER_KRW:
            continue
        rows.append((k, d))
    rows.sort(key=lambda x: -abs(x[1]))
    for k, d in rows[:10]:
        sign = '+' if d > 0 else ''
        lines.append(f'  {k}: {sign}₩{d:,.0f}')
    return '\n'.join(lines) if len(lines) > 1 else '  (변화 없음)'


# ═══ run_once ═══
def run_once(dry_run: bool = False) -> int:
    """한 사이클 실행. 리턴: 0=정상, 1=freshness 스킵, 2=에러."""
    state_path = os.path.join(CACHE_DIR, STATE_FILE)
    state = load_json(state_path, default={})
    if 'cash_buffer' in state and 'buffer_pct' not in state:
        state['buffer_pct'] = state['cash_buffer']
    elif 'buffer_pct' in state and 'cash_buffer' not in state:
        state['cash_buffer'] = state['buffer_pct']
    now = cle.utc_now()

    # 이전 사이클의 combined target 캡처 (engine 이 덮어쓰기 전)
    _prev_snap = state.get('last_target_snapshot') or {}
    prev_combined = {k: float(v) for k, v in _prev_snap.items()
                     if k != '_ts' and isinstance(v, (int, float))}

    log(f'═══ 코인 Executor 시작 (dry_run={dry_run}, now={cle.to_utc_iso(now)}) ═══')

    session = requests.Session()
    api = UpbitAPI(dry_run=dry_run)

    if not dry_run:
        api.cancel_all()

    # 유의종목/거래정지 감지 (freshness 무관, 매번 수행)
    upbit_status = cle.fetch_upbit_market_status(session)
    holdings = api.get_balance()
    held_coins = [k for k, v in holdings.items() if k != 'KRW' and v > MIN_ORDER_KRW]
    warn_or_susp = detect_warning_suspended(upbit_status)
    to_liquidate = [c for c in held_coins if c in warn_or_susp]
    if to_liquidate:
        log(f'  🚨 유의/정지 보유: {to_liquidate}')
        _, failed_liq = liquidate_coins(to_liquidate, 'Upbit 경고/정지', api, state)
        holdings = api.get_balance()
        if failed_liq:
            log(f'❌ 청산 실패 {failed_liq} → fail-closed (리밸런싱 스킵)')
            _tg(f'❌ 코인 청산 실패 {failed_liq} → 실행 중단')
            save_json(state_path, state)
            _flush_telegram(dry_run)
            return 3

    def _upbit_ohlcv(ticker: str):
        try:
            return pyupbit.get_ohlcv(ticker, interval='day', count=260)
        except Exception:
            return None

    # 엔진 호출
    try:
        result = cle.compute_live_targets(
            state, session, CACHE_DIR, now_utc=now,
            upbit_price_fn=_upbit_ohlcv,
            upbit_status=upbit_status,
        )
    except Exception as e:
        log(f'❌ 엔진 호출 실패: {e}\n{traceback.format_exc()}')
        _tg(f'❌ 엔진 호출 실패: {e}')
        save_json(state_path, state)
        _flush_telegram(dry_run)
        return 2

    for a in result.alerts:
        _tg(a)

    # Freshness 판정
    if not result.all_fresh:
        fresh_str = ', '.join(f'{k}={"✓" if v else "✗"}' for k, v in result.fresh.items())
        log(f'  ⚠ Freshness 미달 ({fresh_str}) → 리밸런싱 스킵. 상태만 저장.')
        _tg(f'⚠ Freshness 미달: {fresh_str} → 스킵')
        save_json(state_path, state)
        _flush_telegram(dry_run)
        return 1

    if not result.any_new_bar:
        log('  ℹ 새 봉 없음 (idempotent) → 리밸런싱 스킵')
        save_json(state_path, state)
        _flush_telegram(dry_run)
        return 0

    # 멤버/합산 target 로깅
    for mname, mt in result.member_targets.items():
        coins = ', '.join(f'{k}:{v:.1%}' for k, v in mt.items() if k != 'Cash' and v > 0)
        cash_w = result.member_targets.get(mname, {}).get('Cash', 0.0)
        log(f'  {mname} target: {coins or "CASH only"} (cash={cash_w:.1%})')
    combined_coins = ', '.join(f'{k}:{v:.1%}' for k, v in result.combined_target.items() if k != 'Cash' and v > 0)
    combined_cash = result.combined_target.get('Cash', 0.0)
    log(f'  combined target: {combined_coins or "CASH only"} (cash={combined_cash:.1%})')

    # Cash buffer
    buffer_pct = float(state.get('cash_buffer', state.get('buffer_pct', CASH_BUFFER_DEFAULT)))
    state['cash_buffer'] = buffer_pct
    state['buffer_pct'] = buffer_pct
    target = apply_cash_buffer(result.combined_target, buffer_pct)
    log(f'  Cash buffer {buffer_pct*100:.1f}% 적용 후 target Cash={target.get("Cash",0)*100:.2f}%')

    # Notional cap
    balance = api.get_balance()
    total_krw = sum(balance.values())
    effective_target = dict(target)
    if total_krw > 0 and 0 < NOTIONAL_CAP_FRACTION < 1:
        effective_target, gross = apply_notional_cap(target, balance, total_krw, NOTIONAL_CAP_FRACTION)
        log(f'  Notional cap {NOTIONAL_CAP_FRACTION*100:.0f}% 적용 (gross_delta={gross*100:.1f}%)')

    # Anchor-only 가드: 이전 combined_target 과 사실상 동일하면 drift 리밸런싱 스킵
    # (V21: 스냅샷 회전(anchor)이 실제로 일어났을 때만 매매)
    # 2026-04-21: 사전 알림보다 먼저 판정 — 타겟 불변이면 텔레그램 메시지도 보내지 않음.
    def _targets_equal(a: Dict[str, float], b: Dict[str, float], tol: float = 0.005) -> bool:
        if not a or not b:
            return False
        keys = set(a.keys()) | set(b.keys())
        for k in keys:
            if abs(a.get(k, 0.0) - b.get(k, 0.0)) > tol:
                return False
        return True

    if _targets_equal(result.combined_target, prev_combined):
        log(f'  ℹ target 불변 (앵커 미발생) → 리밸런싱 스킵. prev={prev_combined}')
        state['last_krw_balance'] = total_krw
        save_json(state_path, state)
        _flush_telegram(dry_run)
        return 0

    # 사전 알림 (타겟 변화가 실제로 있을 때만)
    if state.get('pretrade_alert', True):
        summary = format_target_summary(result.combined_target, result.member_targets)
        delta_preview = format_delta_preview(effective_target, balance, total_krw)
        universe_sample = ', '.join(result.universe[:8])
        if len(result.universe) > 8:
            universe_sample += f' ... (+{len(result.universe) - 8})'
        _tg(summary)
        _tg(delta_preview)
        _tg(f'유니버스 {len(result.universe)}: {universe_sample}')
        flips = [m for m, f in result.canary_flipped.items() if f]
        if flips:
            _tg(f'🔄 카나리 플립: {flips}')

    # Delta 매매
    permanent_block = state.get('permanent_block', [])
    execute_delta(effective_target, api, permanent_block, dry_run)

    # 상태 저장
    state['last_krw_balance'] = total_krw
    save_json(state_path, state)
    log(f'  상태 저장: {STATE_FILE}')

    _tg(f'✅ 실행 완료 ({"DRY" if dry_run else "LIVE"}) total=₩{total_krw:,.0f}')
    _flush_telegram(dry_run)
    return 0


# ═══ 진입점 ═══
def main():
    parser = argparse.ArgumentParser(description='Cap Defend 코인 현물 Executor')
    parser.add_argument('--dry-run', action='store_true', help='주문 없이 target/delta만 로그+텔레그램')
    args = parser.parse_args()

    lock_f = None
    try:
        lock_f = open(LOCK_FILE, 'w')
        try:
            fcntl.flock(lock_f.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            log('🔒 다른 인스턴스 실행 중 (lock 충돌) → 종료')
            _tg('🔒 코인 락 충돌 → 스킵')
            _flush_telegram(args.dry_run)
            return

        rc = run_once(dry_run=args.dry_run)
        sys.exit(rc)
    except SystemExit:
        raise
    except Exception as e:
        log(f'❌ 치명 오류: {e}\n{traceback.format_exc()}')
        _tg(f'❌ 코인 치명 오류: {e}')
        _flush_telegram(args.dry_run)
        sys.exit(2)
    finally:
        if lock_f is not None:
            try:
                fcntl.flock(lock_f.fileno(), fcntl.LOCK_UN)
                lock_f.close()
            except Exception:
                pass


if __name__ == '__main__':
    main()
