#!/usr/bin/env python3
"""V25 params drift check — live params 가 canonical 과 일치하는지 확인.

코인 spot (coin_live_engine.py), 선물 (auto_trade_binance.py), 주식 (executor_stock.py)
의 핵심 파라미터를 canonical 과 비교. 불일치 발견 시 텔레그램 알림.

사용
- 수동: python3 check_params_drift.py
- cron: Daily Report 직전 또는 매일 별도 cron

config.py 경로:
- 1순위: CONFIG_PATH 환경변수
- 2순위: TELEGRAM_CONFIG_PATH 환경변수
- 3순위: /home/ubuntu/config.py (운영 fallback)
- 모두 실패 시 명확한 에러 로그 (silent pass 금지)
"""
import math, os, sys
HERE = os.path.dirname(os.path.abspath(__file__))
TRADE_DIR = os.path.abspath(os.path.join(HERE, '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(TRADE_DIR, '..'))
sys.path.insert(0, HERE)
sys.path.insert(0, TRADE_DIR)
sys.path.insert(0, PROJECT_ROOT)
# 집 저장소는 전략 모듈이 하위 폴더에 있다(서버는 전부 평면이라 이 경로가 없고 무해하다).
# executor_stock 을 먼저 import 하면 그 쪽 부수효과로 경로가 붙지만, 그 순서에 기대면
# import 하나만 바뀌어도 조용히 확인을 건너뛴다.
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'strategies', 'cap_defend'))

FLOAT_TOL = 1e-9

CANONICAL = {
    'fut': {
        'source': 'auto_trade_binance.py::STRATEGIES["D_SMA42"] + module-level V25 상수',
        'params': dict(interval='D', sma_bars=42, mom_short_bars=18, mom_long_bars=127,
                       health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                       snap_interval_bars=95, canary_hyst=0.015, n_snapshots=5,
                       post_flip_delay=5),
        'extra': dict(
            DRIFT_THRESHOLD_FUT=0.015,
            LEVERAGE_FLOOR=2, LEVERAGE_MID=3, LEVERAGE_CEILING=4,
            K2_SMA_PERIOD=7, K2_HYST=0.025,
            BTC_CAP_SMA_PERIOD=42, BTC_CAP_THR_MID=1.015, BTC_CAP_THR_MAX=1.05,
            MARGIN_TYPE='CROSSED',
            SCHEMA_VERSION='V25',
        ),
    },
    'spot': {
        'source': 'coin_live_engine.py::MEMBER_D_SMA42',
        'params': dict(interval='D', sma_bars=42, mom_short_bars=20, mom_long_bars=127,
                       snap_interval_bars=217, n_snapshots=7, canary_hyst=0.015,
                       health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                       vol_lookback_days=90, universe_size=3, cap=1.0/3.0,
                       post_flip_delay=5),
    },
    'stock': {
        'source': 'executor_stock.py',
        'params': dict(SNAP_PERIOD_DAYS=69, N_SNAPS=3, SNAP_STAGGER_DAYS=23),
        # 전략 상수는 stock_strategy_v25.py 에 있다. 여기 적어 두는 이유는 둘이다 —
        # 하나는 이 값이 바뀌면 매일 경고가 뜨게 하는 것이고(2026-06-06 사고가 바로
        # 이 계열이다: 라이브 Z랭킹 모멘텀이 순수 252일로 들어가 채택 백테스트와
        # 종목 선정이 16.8% 날 어긋났는데 아무 경보도 없었다), 다른 하나는 주식
        # 신호 대조(trade/ops/signal_parity_daily.py)가 백테스트 쪽 3-모멘텀 기간을
        # 라이브 모듈에서 읽지 않게 하는 것이다. 라이브에서 읽으면 한쪽을 고칠 때
        # 양쪽이 같이 움직여서 대조가 늘 통과한다.
        'strategy_source': 'stock_strategy_v25.py',
        'strategy': dict(
            MOM_SHORT=30, MOM_MID=72, MOM_LONG=230,
            SHARPE_LB=126, MOM12M_DAYS=252,
            STOCK_CANARY_TICKER='EEM', STOCK_CANARY_MA_PERIOD=200,
            STOCK_CANARY_HYST=0.005,
            OFFENSIVE_STOCK_UNIVERSE=['SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC', 'VNQ'],
            DEFENSIVE_STOCK_UNIVERSE=['IEF', 'BIL', 'BNDX', 'GLD', 'PDBC'],
        ),
    },
}


def _eq(a, b):
    """표현 오차 허용 비교. float/int 는 isclose, 그 외는 ==."""
    if a is None or b is None:
        return a == b
    if isinstance(a, float) or isinstance(b, float):
        try:
            return math.isclose(float(a), float(b), rel_tol=FLOAT_TOL, abs_tol=FLOAT_TOL)
        except (TypeError, ValueError):
            return False
    return a == b


def _import(name):
    try:
        return __import__(name)
    except Exception as e:
        print(f"  (import {name} 실패: {e})", file=sys.stderr)
        return None


def check_fut(_):
    mod = _import('auto_trade_binance')
    if mod is None:
        return ['fut: auto_trade_binance import 실패']
    live = mod.STRATEGIES.get('D_SMA42', {})
    diffs = []
    for k, v in CANONICAL['fut']['params'].items():
        if not _eq(live.get(k), v):
            diffs.append(f"  fut/{k}: live={live.get(k)} expected={v}")
    for k, v in CANONICAL['fut']['extra'].items():
        live_v = getattr(mod, k, None)
        if not _eq(live_v, v):
            diffs.append(f"  fut/{k}: live={live_v} expected={v}")
    return diffs


def check_spot(_):
    mod = _import('coin_live_engine')
    if mod is None:
        return ['spot: coin_live_engine import 실패']
    live = mod.MEMBER_D_SMA42
    diffs = []
    for k, v in CANONICAL['spot']['params'].items():
        if not _eq(live.get(k), v):
            diffs.append(f"  spot/{k}: live={live.get(k)} expected={v}")
    return diffs


def check_stock(_):
    mod = _import('executor_stock')
    if mod is None:
        return ['stock: executor_stock import 실패']
    diffs = []
    for k, v in CANONICAL['stock']['params'].items():
        live_v = getattr(mod, k, None)
        if not _eq(live_v, v):
            diffs.append(f"  stock/{k}: live={live_v} expected={v}")

    smod = _import('stock_strategy_v25')
    if smod is None:
        return diffs + ['stock: stock_strategy_v25 import 실패']
    for k, v in CANONICAL['stock']['strategy'].items():
        live_v = getattr(smod, k, None)
        if not _eq(live_v, v):
            diffs.append(f"  stock/{k}: live={live_v} expected={v}")
    return diffs


def _resolve_config_path():
    """env 우선 → /home/ubuntu/config.py fallback. 없으면 None + 명확한 사유."""
    for env_key in ('CONFIG_PATH', 'TELEGRAM_CONFIG_PATH'):
        p = os.environ.get(env_key)
        if p:
            if os.path.isfile(p):
                return p, None
            return None, f"{env_key}={p} 지정됐으나 파일 없음"
    fallback = '/home/ubuntu/config.py'
    if os.path.isfile(fallback):
        return fallback, None
    return None, f"CONFIG_PATH/TELEGRAM_CONFIG_PATH 환경변수 없음 + fallback {fallback} 없음"


def _send_drift_alert(all_diffs):
    """텔레그램 push. 실패 사유는 silent 가 아닌 명시적 로그."""
    try:
        from common.notify import send_telegram
    except Exception as e:
        print(f"(텔레그램 알림 실패: notify 모듈 import 실패: {e})", file=sys.stderr)
        return False
    cfg_path, err = _resolve_config_path()
    if cfg_path is None:
        print(f"(텔레그램 알림 실패: config 경로 미해결 — {err})", file=sys.stderr)
        return False
    try:
        import importlib.util
        spec = importlib.util.spec_from_file_location('config', cfg_path)
        if spec is None or spec.loader is None:
            print(f"(텔레그램 알림 실패: spec load 실패 — {cfg_path})", file=sys.stderr)
            return False
        cfg = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(cfg)
        token = getattr(cfg, 'TELEGRAM_BOT_TOKEN', None)
        chat = getattr(cfg, 'TELEGRAM_CHAT_ID', None)
        if not token or not chat:
            print("(텔레그램 알림 실패: TELEGRAM_BOT_TOKEN 또는 TELEGRAM_CHAT_ID 누락)", file=sys.stderr)
            return False
        msg = "⚠️ V25 params drift 발견\n" + "\n".join(all_diffs)
        send_telegram(token, chat, msg, prefix='경고', timeout=10)
        return True
    except Exception as e:
        print(f"(텔레그램 알림 실패: 전송 중 예외 — {e})", file=sys.stderr)
        return False


def main():
    here = os.path.dirname(os.path.abspath(__file__))
    all_diffs = []
    all_diffs += check_fut(here)
    all_diffs += check_spot(here)
    all_diffs += check_stock(here)

    if not all_diffs:
        print("✅ V25 params drift check OK — 3자산 모두 canonical 일치")
        sys.exit(0)
    print("⚠️ V25 params drift 발견:")
    for d in all_diffs:
        print(d)
    sent = _send_drift_alert(all_diffs)
    if not sent:
        print("⚠️ drift 발견되었으나 텔레그램 푸시 전송 실패 — 위 stderr 확인", file=sys.stderr)
    sys.exit(1)


if __name__ == '__main__':
    main()
