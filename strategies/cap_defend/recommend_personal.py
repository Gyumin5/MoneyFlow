"""
Cap Defend V19 Recommendation Script
===================================
Stock V17: R7 + EEM canary + Z-score3(Sh252) EW + Defense Top3 + VT Crash(-3%/3d)
Coin V18: K:SMA(50)+1.5%hyst + H:Mom30+Mom90+Vol5% + Greedy Absorption + EW+33%Cap + DD Exit + Blacklist
Futures: d005 4전략 앙상블 EW + 5x 동적 레버리지
Asset Allocation: 주식60 / 코인25 / 선물15, 8pp 밴드
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd
import numpy as np
import time
import json
import sqlite3
import requests
from datetime import datetime, timezone, timedelta
from typing import Tuple
import pyupbit

# ─── Telegram ─────────────────────────────────────────────
try:
    from config import TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
except ImportError:
    TELEGRAM_BOT_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN', '')
    TELEGRAM_CHAT_ID = os.environ.get('TELEGRAM_CHAT_ID', '')

try:
    from config import PORTFOLIO_PUBLIC_URL as CONFIG_PORTFOLIO_PUBLIC_URL
except Exception:
    CONFIG_PORTFOLIO_PUBLIC_URL = ""

def send_telegram(msg, button_text=None, button_url=None):
    """텔레그램 알림 전송. 실패해도 프로그램은 계속 진행."""
    try:
        payload = {"chat_id": TELEGRAM_CHAT_ID, "text": msg, "parse_mode": "HTML"}
        if button_text and button_url:
            payload["reply_markup"] = {
                "inline_keyboard": [[{"text": button_text, "url": button_url}]]
            }
        requests.post(
            f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage",
            json=payload,
            timeout=10
        )
    except Exception:
        pass

# --- Configuration for Auto Turnover ---
# Try local config.py first (remote server), then ../../config/upbit.py (local dev)
ACCESS_KEY = ""
SECRET_KEY = ""
for _cfg_path in [os.path.dirname(os.path.abspath(__file__)),
                   os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'config')]:
    if _cfg_path not in sys.path:
        sys.path.insert(0, _cfg_path)
try:
    from config import UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY
    ACCESS_KEY = UPBIT_ACCESS_KEY
    SECRET_KEY = UPBIT_SECRET_KEY
except Exception:
    try:
        from upbit import UPBIT_ACCESS_KEY, UPBIT_SECRET_KEY
        ACCESS_KEY = UPBIT_ACCESS_KEY
        SECRET_KEY = UPBIT_SECRET_KEY
    except Exception as e:
        print(f"⚠️ Upbit config import failed: {e}")

# --- 1. Constants & Configuration ---
DATA_DIR = "./data"
SIGNAL_STATE_FILE = os.path.join(".", "signal_state.json")
APP_HOME = os.environ.get("MONEYFLOW_APP_HOME", os.getcwd())
ASSETS_DB = os.environ.get("ASSETS_DB", os.path.join(APP_HOME, "assets.db"))
try:
    from config import PORTFOLIO_HTML_NAME as CONFIG_HTML_NAME
except ImportError:
    CONFIG_HTML_NAME = "portfolio_result_gmoh.html"
PORTFOLIO_HTML_NAME = os.environ.get("PORTFOLIO_HTML_NAME") or CONFIG_HTML_NAME
PORTFOLIO_PUBLIC_URL = os.environ.get("PORTFOLIO_PUBLIC_URL", "") or CONFIG_PORTFOLIO_PUBLIC_URL
STOCK_ANCHOR_DAYS = (1, 8, 15, 22)
COIN_ANCHOR_DAYS = (1, 11, 21)
FUTURES_TRANCHE_META = {
    "4h_d005": {"interval_hours": 4, "snap_interval_bars": 60, "n_snapshots": 3},
    "2h_S240": {"interval_hours": 2, "snap_interval_bars": 120, "n_snapshots": 3},
    "2h_S120": {"interval_hours": 2, "snap_interval_bars": 120, "n_snapshots": 3},
    "4h_M20":  {"interval_hours": 4, "snap_interval_bars": 21, "n_snapshots": 3},
}

def _save_signal_state(data):
    """signal_state.json 원자적 저장."""
    tmp = SIGNAL_STATE_FILE + '.tmp'
    with open(tmp, 'w') as f:
        json.dump(data, f)
    os.replace(tmp, SIGNAL_STATE_FILE)


def save_daily_live_snapshot():
    """실계좌 기준 일간 스냅샷 저장. 하루 1건 upsert."""
    api = os.environ.get("TRADE_API_BASE", "http://127.0.0.1:5000")
    r = requests.get(f"{api}/api/assets/live_overview", timeout=60)
    r.raise_for_status()
    data = r.json()

    accounts = data.get("accounts", {})
    stock = accounts.get("stock_kis", {}) or {}
    upbit = accounts.get("coin_upbit", {}) or {}
    binance = accounts.get("coin_binance", {}) or {}

    stock_krw = float(stock.get("stock_eval_usd", 0.0)) * float(stock.get("exchange_rate", 0.0))
    upbit_coin_krw = sum(float(row.get("value", 0.0)) for row in (upbit.get("holdings") or []))
    binance_coin_krw = sum(float(row.get("value_krw", 0.0)) for row in (binance.get("holdings") or []))
    coin_krw = upbit_coin_krw + binance_coin_krw
    cash_krw = (
        float(stock.get("cash_krw", 0.0))
        + float(upbit.get("krw_balance", 0.0))
        + float(binance.get("cash_krw", 0.0))
    )
    total_krw = stock_krw + coin_krw + cash_krw
    fx_rate = float(stock.get("exchange_rate", 0.0) or binance.get("exchange_rate", 0.0) or 0.0)
    usd_cash = float(stock.get("cash_usd", 0.0))
    snapshot_date = datetime.now().strftime("%Y-%m-%d")
    accounts_json = json.dumps(accounts, ensure_ascii=False)
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    db_path = ASSETS_DB if os.path.isdir(os.path.dirname(ASSETS_DB)) else os.path.join(".", "assets.db")
    conn = sqlite3.connect(db_path)
    conn.execute(
        """CREATE TABLE IF NOT EXISTS snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT UNIQUE NOT NULL,
            stock_krw REAL DEFAULT 0,
            coin_krw REAL DEFAULT 0,
            cash_krw REAL DEFAULT 0,
            total_krw REAL DEFAULT 0,
            fx_rate REAL DEFAULT 0,
            usd_cash REAL DEFAULT 0,
            memo TEXT DEFAULT '',
            accounts_json TEXT DEFAULT '{}',
            created_at TEXT
        )"""
    )
    conn.execute(
        """INSERT INTO snapshots
           (snapshot_date, stock_krw, coin_krw, cash_krw, total_krw, fx_rate, usd_cash, memo, accounts_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(snapshot_date) DO UPDATE SET
             stock_krw=excluded.stock_krw,
             coin_krw=excluded.coin_krw,
             cash_krw=excluded.cash_krw,
             total_krw=excluded.total_krw,
             fx_rate=excluded.fx_rate,
             usd_cash=excluded.usd_cash,
             memo=excluded.memo,
             accounts_json=excluded.accounts_json,
             created_at=excluded.created_at
        """,
        (
            snapshot_date,
            stock_krw,
            coin_krw,
            cash_krw,
            total_krw,
            fx_rate,
            usd_cash,
            "Auto daily snapshot",
            accounts_json,
            created_at,
        ),
    )
    conn.commit()
    conn.close()
    return {
        "snapshot_date": snapshot_date,
        "stock_krw": stock_krw,
        "coin_krw": coin_krw,
        "cash_krw": cash_krw,
        "total_krw": total_krw,
    }
STRATEGY_VERSION = "V19"
VERSION_HISTORY = [
    ("V19", "2026-04",
     "선물 추가 + 자산배분 60/25/15 + 밴드 8pp",
     """<b>▶ 자산배분 (V19)</b>
• 주식 <span style='color:#d93025;'>60%</span> / 현물코인 <span style='color:#d93025;'>25%</span> / 선물 <span style='color:#d93025;'>15%</span>
• 밴드 리밸런싱: 편차 <span style='color:#d93025;'>±8%p</span> 초과 시만 (매일 자동체크)
• 포트폴리오: Sharpe 2.12, CAGR +39%, MDD -12.2%

<b>▶ 선물 (V19 신규)</b>
• d005 4전략 앙상블 (EW 25%씩)
• 4h_d005 / 2h_S240 / 2h_S120 / 4h_M20
• 5x 동적 레버리지 (cap_mom_blend_543_cash)
• 단독: Sharpe 2.08, CAGR +227%, MDD -34%

<b>▶ 주식: V17 동일, 코인: V18 동일</b>"""),

    ("V18", "2026-03",
     "코인: SMA50+1.5%hyst, Greedy Absorption, EW+33%Cap",
     """<b>▶ 코인 카나리 (V18)</b>
• BTC > SMA(<span style='color:#d93025;'>50</span>) + <span style='color:#d93025;'>1.5%</span> Hysteresis
• MDD 4.4%p 개선, Calmar 27% 개선

<b>▶ 코인 선정 (V18)</b>
• 시총순 Top 5 → <span style='color:#d93025;'>Greedy Absorption</span>
• 시총 큰 코인이 Mom30 높으면 작은 코인 제거
• 비중: EW + <span style='color:#d93025;'>33% Cap</span> (V17: 20%)
• 통합 MDD -24.8% (V17: -29.0%)"""),

    ("V17", "2026-03",
     "주식: Zscore3+Sh252+VT Crash, 코인: 모니터 USD 전환",
     """<b>▶ 주식 변경 (V17)</b>
• <b>선정:</b> Z-score <span style='color:#d93025;'>Top 3</span> (Sharpe <span style='color:#d93025;'>252d</span>)
• <b>Crash:</b> <span style='color:#d93025;'>VT -3%</span> daily → 3일 현금
• <b>백테스트:</b> 11-anchor 평균 Sharpe 1.037, CAGR +12.1%, MDD -13.2%, Calmar 0.94

<b>▶ 코인 모니터 수정 (V17)</b>
• 카나리/Crash를 <span style='color:#d93025;'>USD 기준</span>으로 비교 (환율 왜곡 방지)
• DD Exit: CSV 60일 rolling peak 직접 조회
• HOLD 시에도 캐시 갱신, cron :05/:35"""),

    ("V16", "2026-03",
     "코인: Mom30+25%Cap, 주식: V15 동일, hysteresis 수정",
     """<b>▶ 코인 변경 (V16)</b>
• <b>Health:</b> <span style='color:#d93025;'>Mom(30)</span>>0 AND Mom(90)>0 AND Vol(90)≤5% (Mom30→30으로 변경)
• <b>비중 캡:</b> <span style='color:#d93025;'>20% Cap</span> — 1종목 최대 20%, 나머지 현금 (붕괴 리스크 방어)
• <b>백테스트:</b> 10-anchor 3트랜치 평균 Sharpe 1.451, CAGR +64.5%, MDD -33.2%, Calmar 1.94
• <b>카나리아:</b> Hysteresis dead zone에서 signal_state.json 이전 상태 참조 (stateless 수정)

<b>▶ 주식: V15 동일</b>"""),

    ("V15", "2026-03",
     "R7 universe, Z-score4 selection, M+D2 trigger rebalancing, delta trading",
     """<b>자산배분:</b> 주식 60% / 코인 40% (현금 버퍼 2%)

<b>▶ 주식 전략 (V15 변경)</b>
• <b>유니버스:</b> <span style='color:#d93025;'>R7</span> (SPY, QQQ, VEA, EEM, GLD, PDBC, <b>VNQ</b> — REIT 추가로 사이클 분산)
• <b>Canary:</b> EEM > SMA200 (0.5% Hysteresis)
• <b>Health:</b> 없음 (카나리아가 시장 방어 담당)
• <b>공격:</b> <span style='color:#d93025;'>Z-score Top 4</span> = zscore(12M Mom) + zscore(Sharpe63), 균등배분
• <b>수비:</b> 5종 중 6M 수익률 Top 3 (음수면 현금), 균등배분
• <b>백테스트:</b> 11-anchor 평균 Sharpe 1.281 (σ=0.033), CAGR +15.2%, MDD -13.2%

<b>▶ 코인 전략 (V14 동일)</b>
• <b>유니버스:</b> CoinGecko Top 40 시총순 → Upbit KRW 필터
• <b>Canary:</b> BTC > SMA(50) → 투자, 아니면 현금 (1.5% Hysteresis)
• <b>Health Filter:</b> Mom(30)>0 AND Mom(90)>0 AND Vol(90)≤5%
• <b>선정:</b> 시총순 Top 5, 균등배분 (EW)
• <b>DD Exit:</b> 60일 고점 대비 -25% 하락 → 매도 (매일 체크)
• <b>Blacklist:</b> 일일 -15% 하락 → 7일 제외
• <b>Crash Breaker:</b> BTC 일일 -10% → 3일 현금

<b>▶ 리밸런싱</b>
• 주식: <span style='color:#d93025;'>M+D2</span> — 월간 정기 + 매일 Z-score Top4 체크, 2종목 이상 변경 시 즉시 리밸런싱
• 코인: 월간 정기 + DD Exit/Blacklist/Crash 시 즉시
• <span style='color:#d93025;'>Delta-based trading</span>: 변경된 비중만 거래 (TX 비용 61% 절감)"""),

    ("V14", "2026-03",
     "SMA(50) canary, Mom+Mom+Vol5% health, EW, DD Exit, Blacklist, Crash Breaker",
     """<b>▶ 코인:</b> K:SMA(50)+1.5%hyst, H:Mom30+Mom90+Vol5%, Greedy Absorption EW+33%Cap, DD Exit(-25%), Blacklist(-15%), Crash(-10%)
<b>▶ 주식:</b> R6 (SPY,QQQ,VEA,EEM,GLD,PDBC), EEM>SMA200, Mom3+Sh3 union EW, Defense Top3"""),

    ("V13", "2026-03",
     "Multi Bonus scoring, 1% Hysteresis signal flip, monthly rebalancing",
     """<b>자산배분:</b> 주식 60% / 코인 40% (현금 버퍼 2%)

<b>\u25b6 주식 전략 (V11 기반)</b>
\u2022 <b>유니버스:</b> 공격 12종 (SPY, QQQ, EFA, EEM, VT, VEA, GLD, PDBC, QUAL, MTUM, IQLT, IMTM) / 수비 5종 (IEF, BIL, BNDX, GLD, PDBC)
\u2022 <b>Canary Signal:</b> VT AND EEM \u003e SMA200 \u2192 Risk-On / \uc544\ub2c8\uba74 Risk-Off
\u2022 <b>Signal Flip Guard (V13 \uc2e0\uaddc):</b> <span style='color:#d93025;'>1% Hysteresis</span> \u2014 SMA200\xd71.01 \uc774\uc0c1\uc77c \ub54c Risk-On \uc9c4\uc785, SMA200\xd70.99 \uc774\ud558\uc77c \ub54c Risk-Off \uc9c4\uc785. \ud718\uc18c (\ube48\ubc88\ud55c \uc2e0\ud638 \uc804\ud658) 62% \uac10\uc18c
\u2022 <b>\uacf5\uaca9 \ubaa8\ub4dc:</b> \uac00\uc911 \ubaa8\uba58\ud140(50/30/20% for 3M/6M/12M) Top 3 + Sharpe(126d) Top 3 \ud569\uc9d1\ud569 \u2192 \uade0\ub4f1\ubc30\ubd84
\u2022 <b>\uc218\ube44 \ubaa8\ub4dc:</b> \uc218\ube44 5\uc885 \uc911 6\uac1c\uc6d4 \uc218\uc775\ub960 \ucd5c\uace0 1\uac1c (\uc74c\uc218\uba74 \ud604\uae08)
\u2022 <b>\uc885\ubaa9 \ub9ac\ubc38\ub7f0\uc2f1:</b> \uc6d4\uac04 \uc815\uae30 \ub9ac\ubc38\ub7f0\uc2f1 (\uc6d4\ub9d0 \uad8c\uc7a5). \ubc31\ud14c\uc2a4\ud2b8 \uacb0\uacfc \uc6d4\ub9d0 \ub9ac\ubc38\ub7f0\uc2f1\uc774 Sharpe 1.001\ub85c \ucd5c\uc801

<b>\u25b6 \ucf54\uc778 \uc804\ub7b5 (V12 \uae30\ubc18)</b>
\u2022 <b>\uc720\ub2c8\ubc84\uc2a4:</b> CoinGecko Top 50 \uc2dc\uac00\ucd1d\uc561\uc21c \u2192 Upbit KRW \uc0c1\uc7a5 + 253\uc77c \ud788\uc2a4\ud1a0\ub9ac + 30\uc77c \ud3c9\uade0 \uac70\ub798\ub300\uae08 10\uc5b5\uc6d0\uc774\uc0c1 \ud544\ud130
\u2022 <b>Canary:</b> BTC \u003e SMA50 \u2192 \ud22c\uc790, \uc544\ub2c8\uba74 \ud604\uae08
\u2022 <b>Health Filter:</b> Price \u003e SMA30 AND Mom30 \u003e 0 AND Vol90 \u2264 10%
\u2022 <b>Scoring (V13 \uc2e0\uaddc):</b> <span style='color:#d93025;'>Multi Bonus</span> = Sharpe(126d)+Sharpe(252d) + RSI(45~70 \u2192 +0.2) + MACD hist\u003e0 \u2192 +0.2 + BB %B\u003e0.5 \u2192 +0.2
\u2022 <b>\uc120\uc815:</b> Multi Bonus Score Top 5
\u2022 <b>\ubc30\ubd84:</b> 90\uc77c \uc5ed\ubcc0\ub3d9\uc131 \uac00\uc911 (1/Vol)

<b>\u25b6 \ub9ac\ubc38\ub7f0\uc2f1 \uaddc\uce59</b>
\u2022 \ucf54\uc778: \uc6d4\uac04 \uc2a4\ucf00\uc904 + \ud134\uc624\ubc84 30%\u2191 \ub610\ub294 Health \uc2e4\ud328 \uc2dc \uc989\uc2dc
\u2022 \uc8fc\uc2dd: \uc6d4\uac04 \uc815\uae30 \ub9ac\ubc38\ub7f0\uc2f1 (\uc6d4\ub9d0 \uad8c\uc7a5) + Signal Flip(1% Hysteresis)"""),

    ("V12", "2026-01",
     "Weighted momentum, Sharpe quality, inverse volatility weighting, health check",
     """<b>\uc790\uc0b0\ubc30\ubd84:</b> \uc8fc\uc2dd 60% / \ucf54\uc778 40% (\ud604\uae08 \ubc84\ud37c 2%)

<b>\u25b6 \uc8fc\uc2dd \uc804\ub7b5 (V11 \ub3d9\uc77c)</b>
\u2022 <b>Canary:</b> VT AND EEM \u003e SMA200 \u2192 Risk-On
\u2022 <b>\uacf5\uaca9:</b> 12\uc885 \uc720\ub2c8\ubc84\uc2a4, \uac00\uc911 \ubaa8\uba58\ud140(50/30/20%) Top 3 + Sharpe(126d) Top 3 \ud569\uc9d1\ud569, \uade0\ub4f1\ubc30\ubd84
\u2022 <b>\uc218\ube44:</b> 5\uc885 \uc911 6M \uc218\uc775\ub960 \ucd5c\uace0 1\uac1c (\uc74c\uc218\uba74 \ud604\uae08)
\u2022 Signal flip guard \uc5c6\uc74c (SMA200 \uc790\uccb4 \ub2e8\uc21c \ube44\uad50)

<b>\u25b6 \ucf54\uc778 \uc804\ub7b5 (V12 \uc2e0\uaddc)</b>
\u2022 <b>\uc720\ub2c8\ubc84\uc2a4:</b> CoinGecko Top 50 + Upbit KRW \ud544\ud130 (253\uc77c \ud788\uc2a4\ud1a0\ub9ac, 10\uc5b5\uc6d0 \uac70\ub798\ub300\uae08)
\u2022 <b>Canary:</b> BTC \u003e SMA50
\u2022 <b>Health Filter (V12 \uc2e0\uaddc):</b> Price \u003e SMA30 AND Mom30 \u003e 0 AND Vol90 \u2264 10%
\u2022 <b>Scoring:</b> Sharpe(126d) + Sharpe(252d) (\ubcf4\ub108\uc2a4 \uc5c6\uc74c)
\u2022 <b>\uc120\uc815:</b> Top 5
\u2022 <b>\ubc30\ubd84 (V12 \uc2e0\uaddc):</b> 90\uc77c \uc5ed\ubcc0\ub3d9\uc131 \uac00\uc911 (1/Vol)

<b>\u25b6 \ub9ac\ubc38\ub7f0\uc2f1</b>
\u2022 \uc6d4\uac04 + \ud134\uc624\ubc84 30%\u2191 \ub610\ub294 Health \uc2e4\ud328 \uc2dc"""),

    ("V11", "2025-12",
     "Dual canary, offensive Top3+Top3 union, defensive 6M best",
     """<b>\uc790\uc0b0\ubc30\ubd84:</b> \uc8fc\uc2dd 60% / \ucf54\uc778 40% (\ud604\uae08 \ubc84\ud37c 2%)

<b>\u25b6 \uc8fc\uc2dd \uc804\ub7b5 (V11 \uc2e0\uaddc)</b>
\u2022 <b>Canary (V11 \uc2e0\uaddc):</b> Dual Canary \u2014 VT AND EEM \ub458 \ub2e4 SMA200 \uc774\uc0c1\uc774\uc5b4\uc57c Risk-On
\u2022 <b>\uacf5\uaca9 \uc720\ub2c8\ubc84\uc2a4:</b> SPY, QQQ, EFA, EEM, VT, VEA, GLD, PDBC, QUAL, MTUM, IQLT, IMTM (12\uc885)
\u2022 <b>\uacf5\uaca9 \uc120\uc815 (V11 \uc2e0\uaddc):</b> \uac00\uc911\ubaa8\uba58\ud140(50/30/20%) Top 3 + Sharpe(126d) Top 3 \u2192 \ud569\uc9d1\ud569 (3~6\uc885), \uade0\ub4f1\ubc30\ubd84
\u2022 <b>\uc218\ube44 \uc720\ub2c8\ubc84\uc2a4:</b> IEF, BIL, BNDX, GLD, PDBC (5\uc885)
\u2022 <b>\uc218\ube44 \uc120\uc815 (V11 \uc2e0\uaddc):</b> 6\uac1c\uc6d4 \uc218\uc775\ub960 \ucd5c\uace0 1\uac1c (\uc74c\uc218\uba74 \ud604\uae08)

<b>\u25b6 \ucf54\uc778 \uc804\ub7b5</b>
\u2022 <b>\uc720\ub2c8\ubc84\uc2a4:</b> CoinGecko Top 50 + Upbit KRW
\u2022 <b>Canary:</b> BTC \u003e SMA50
\u2022 <b>Health Filter:</b> \uc5c6\uc74c (V12\uc5d0\uc11c \ucd94\uac00)
\u2022 <b>Scoring:</b> Sharpe(126d) + Sharpe(252d), Top 5 \uade0\ub4f1\ubc30\ubd84

<b>\u25b6 \ub9ac\ubc38\ub7f0\uc2f1</b>
\u2022 \uc6d4\uac04 \uc2a4\ucf00\uc904 \uae30\ubc18"""),
]

STOCK_RATIO, COIN_RATIO, FUTURES_RATIO = 0.60, 0.25, 0.15
CASH_ASSET = 'Cash'
CASH_BUFFER_PERCENT_DEFAULT = 0.02 # 2% Cash Buffer
REBAL_BAND_PP = 0.08  # 8pp band — any asset drifts ≥8pp → full rebalance

def get_cash_buffer():
    """현금 버퍼 비율. coin_trade_state에서 읽기 (HTML 표시용)."""
    try:
        with open('coin_trade_state.json', 'r') as f:
            return json.load(f).get('cash_buffer', CASH_BUFFER_PERCENT_DEFAULT)
    except Exception:
        return CASH_BUFFER_PERCENT_DEFAULT
STABLECOINS = ['USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'TUSD', 'PAX', 'GUSD', 'FRAX', 'LUSD', 'MIM', 'USDN', 'FDUSD']

# Stock Configuration (V15: R7 Universe + EEM-only Canary)
OFFENSIVE_STOCK_UNIVERSE = ['SPY', 'QQQ', 'VEA', 'EEM', 'GLD', 'PDBC', 'VNQ']
DEFENSIVE_STOCK_UNIVERSE = ['IEF', 'BIL', 'BNDX', 'GLD', 'PDBC']
CANARY_ASSETS = ['EEM']
STOCK_CANARY_MA_PERIOD = 200
STOCK_CANARY_HYST = 0.005  # 0.5% hysteresis
STOCK_CRASH_TICKER = 'VT'
STOCK_CRASH_THRESHOLD = -0.03  # VT daily -3%
STOCK_CRASH_COOL_DAYS = 3

# Coin Configuration
COIN_CANARY_MA_PERIOD = 50
COIN_CANARY_HYST = 0.015  # 1.5% Hysteresis: enter Risk-On at SMA*1.015, exit at SMA*0.985
N_SELECTED_COINS = 5
VOLATILITY_WINDOW = 90

# --- V15 Configuration ---
VOL_CAP_FILTER = 0.05
BL_THRESHOLD = -0.15
BL_DAYS = 7
DD_EXIT_LOOKBACK = 60
DD_EXIT_THRESHOLD = -0.25
CRASH_THRESHOLD = -0.10

def get_dynamic_coin_universe(log: list) -> (list, dict):
    print("\n--- 🛰️ Step 1: Coin Universe Selection (V15: LIVE CoinGecko + Upbit Filter) ---")
    log.append("<h2>🛰️ Step 1: 코인 유니버스 선정 (V19: Live CoinGecko Top 40)</h2>")
    
    COINGECKO_URL = "https://api.coingecko.com/api/v3/coins/markets"
    FETCH_LIMIT = 100 
    MIN_TRADE_VALUE_KRW = 1_000_000_000 
    DAYS_TO_CHECK = 260 
    headers = {"accept": "application/json", "User-Agent": "Mozilla/5.0"}
    
    UNIVERSE_CACHE_FILE = os.path.join(DATA_DIR, "universe_cache.json")
    cg_data = []

    # 1. Try Fetching from CoinGecko (Retry 5 times)
    for attempt in range(1, 6):
        try:
            print(f"  - Fetching Top {FETCH_LIMIT} from CoinGecko (Attempt {attempt}/5)...")
            cg_params = {'vs_currency': 'usd', 'order': 'market_cap_desc', 'per_page': FETCH_LIMIT, 'page': 1}
            cg_response = requests.get(COINGECKO_URL, params=cg_params, headers=headers, timeout=20)
            
            if cg_response.status_code == 200:
                cg_data = cg_response.json()
                # Save to Cache
                with open(UNIVERSE_CACHE_FILE, 'w') as f:
                    json.dump(cg_data, f)
                print(f"    ✅ Got {len(cg_data)} coins from CoinGecko (Cached)")
                break
            elif cg_response.status_code == 429:
                wait_time = 30 * attempt # 30s, 60s, 90s...
                print(f"    ⚠️ Rate Limit (429). Waiting {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"    ⚠️ API Error: {cg_response.status_code}")
                time.sleep(10)
        except Exception as e:
            print(f"    ⚠️ Connection Error: {e}")
            time.sleep(10)
            
    # 2. If Failed, Load from Cache
    if not cg_data:
        if os.path.exists(UNIVERSE_CACHE_FILE):
            print(f"  ⚠️ Loading Universe from Local Cache (Fallback)...")
            log.append("<p class='warning'>⚠️ Loading Universe from Local Cache (Fallback).</p>")
            try:
                with open(UNIVERSE_CACHE_FILE, 'r') as f:
                    cg_data = json.load(f)
            except: pass

    if not cg_data:
        log.append("<p class='error'>❌ All Methods Failed. Using Hardcoded Fallback.</p>")
        fallback = ['BTC-USD', 'ETH-USD', 'SOL-USD', 'XRP-USD', 'DOGE-USD', 'ADA-USD', 'AVAX-USD', 'LINK-USD', 'SHIB-USD', 'TRX-USD']
        return fallback, {}
    
    cg_symbol_to_id_map = {f"{item['symbol'].upper()}-USD": item['id'] for item in cg_data}
    
    print("  - Fetching Upbit KRW Market...")
    try:
        upbit_krw_tickers = pyupbit.get_tickers(fiat="KRW")
        upbit_symbols = {t.split('-')[1] for t in upbit_krw_tickers}
        print(f"    ✅ Upbit has {len(upbit_symbols)} KRW markets")
    except Exception as e:
        print(f"    ⚠️ Upbit API Error: {e}")
        upbit_symbols = set()
    
    print("  - Filtering by Upbit availability, liquidity, and history...")
    final_universe = []
    
    for item in cg_data:
        symbol = item['symbol'].upper()
        if symbol in STABLECOINS: continue
        if symbol not in upbit_symbols:
            print(f"    ❌ {symbol}: Not in Upbit KRW")
            continue
        
        upbit_ticker = f"KRW-{symbol}"
        try:
            df = pyupbit.get_ohlcv(ticker=upbit_ticker, interval="day", count=DAYS_TO_CHECK)
            time.sleep(0.1)
            
            if df is None or len(df) < 253:
                print(f"    ❌ {symbol}: Insufficient History ({len(df) if df is not None else 0} days < 253)")
                continue
            
            avg_val = df['value'].iloc[-30:].mean()
            if avg_val < MIN_TRADE_VALUE_KRW:
                print(f"    ❌ {symbol}: Low Liquidity ({avg_val/100000000:.1f}억 < 10억)")
                continue
            
            ticker_usd = f"{symbol}-USD"
            if ticker_usd in final_universe: continue
                
            final_universe.append(ticker_usd)
            print(f"    ✅ {symbol}: Included (Rank {len(final_universe)}, Liquidity: {avg_val/100000000:.1f}억)")
            
        except Exception as e:
            print(f"    ⚠️ {symbol}: Upbit Check Error - {e}")
            continue
        
        if len(final_universe) >= 40: break

    log.append(f"<p>선정된 유니버스 ({len(final_universe)}개): Top 40 qualified</p>")
    return final_universe, cg_symbol_to_id_map


# --- Helper: Get Upbit Assets (Personal) ---
def get_current_upbit_holdings(log):
    """
    업비트 API를 사용하여 현재 보유 자산을 조회합니다.
    - holdings_qty: {ticker-USD: qty} (수량만 - 종가 평가용)
    - holdings_krw: {ticker-USD: krw_value} (실시간 KRW 가치 - 표시용)
    - unlisted coins are filtered out
    """
    if not ACCESS_KEY or not SECRET_KEY:
        print("⚠️ Upbit API keys not configured")
        log.append("<p class='error'>❌ Upbit API 키 미설정</p>")
        return {}, {}, 0.0

    try:
        upbit = pyupbit.Upbit(ACCESS_KEY, SECRET_KEY)

        krw = upbit.get_balance("KRW")
        if krw is None:
            print(f"⚠️ Upbit API Connection Failed (key prefix: {ACCESS_KEY[:6]}...)")
            log.append("<p class='error'>❌ Upbit API 연결 실패 (키 확인 필요)</p>")
            return {}, {}, 0.0
        
        # Get currently listed coins in Upbit KRW market
        try:
            upbit_krw_tickers = pyupbit.get_tickers(fiat="KRW")
            upbit_listed = {t.split('-')[1] for t in upbit_krw_tickers}
        except:
            upbit_listed = set()
            
        balances = upbit.get_balances()
        holdings_qty = {}   # {ticker-USD: qty}
        holdings_krw = {}   # {ticker-USD: krw_value} (real-time for display)
        
        my_cash = 0.0
        
        for b in balances:
            ticker = b['currency']
            if ticker == 'KRW': 
                my_cash = float(b['balance'])
                continue
            
            # Filter: only include coins currently listed in Upbit KRW market
            if ticker not in upbit_listed:
                print(f"    ⚠️ {ticker}: Not in Upbit KRW market (Skipped)")
                continue
            
            qty = float(b['balance']) + float(b['locked'])
            if qty > 0:
                try:
                    price_krw = pyupbit.get_current_price(f"KRW-{ticker}")
                    if price_krw:
                        val_krw = qty * price_krw
                        if val_krw > 1000:  # Exclude dust
                            holdings_qty[f"{ticker}-USD"] = qty
                            holdings_krw[f"{ticker}-USD"] = val_krw
                except: pass
                
        return holdings_qty, holdings_krw, my_cash
        
    except Exception as e:
        print(f"⚠️ Upbit Asset Load Error: {e}")
        log.append(f"<p class='error'>❌ 자산 조회 오류: {e}</p>")
        return {}, {}, 0.0

# --- 3. Data Download ---
def download_required_data(tickers: list, log: list, coin_id_map: dict):
    """
    데이터 다운로드 - Yahoo Finance 우선, CoinGecko 폴백
    [V12.3] 데이터 안정화: File Cache, Retry(10회), Stale Data Fallback 적용
    """
    print("\n--- 📥 Step 2: 데이터 다운로드 (Yahoo Priority + Quality Check) ---")
    log.append("<h2>📥 Step 2: 데이터 다운로드 (Yahoo 우선 + 품질검증)</h2>")
    os.makedirs(DATA_DIR, exist_ok=True)
    
    # USD-KRW 환율 조회 (품질 검증용)
    usd_krw_rate = 1450.0
    try:
        usdt_price = pyupbit.get_current_price("KRW-USDT")
        if usdt_price: usd_krw_rate = usdt_price
    except: pass
    
    # [V12.3] Robust Session with Retries
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    
    # Retry 10 times, backoff factor 0.5 (0.5, 1, 2, 4...)
    retries = Retry(total=10, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    
    today_str = datetime.now().date()
    
    for ticker in list(set(tickers)):
        if ticker == CASH_ASSET: continue
        fp = os.path.join(DATA_DIR, f"{ticker}.csv")
        success = False
        
        # 1. Fresh Cache Check
        # 오늘 다운로드한 파일이 있으면 재사용 (삭제하지 않음)
        if os.path.exists(fp):
            try:
                # 파일 수정 시간 확인
                mtime = datetime.fromtimestamp(os.path.getmtime(fp)).date()
                if mtime == today_str:
                    #print(f"  ✅ Using cached data for {ticker}")
                    # 캐시된 파일도 품질 검증을 통과했다고 가정하거나, 로드 시점에 다시 검증할 수 있음.
                    # 여기서는 '이미 검증되어 저장된 파일'이라고 가정하고 스킵.
                    success = True
            except: pass
        
        if success:
            continue

        # 2. Download from Yahoo
        try:
            current_timestamp = int(datetime.now(timezone.utc).timestamp())
            start_timestamp = int(datetime(2018, 1, 1, tzinfo=timezone.utc).timestamp())
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            params = {"period1": start_timestamp, "period2": current_timestamp, "interval": "1d", "includeAdjustedClose": "true"}
            
            # [Stabilization] Timeout increased to 30s
            resp = session.get(url, params=params, timeout=30)
            
            if resp.status_code == 200:
                res = resp.json()['chart']['result'][0]
                df = pd.DataFrame({'Date': pd.to_datetime(res['timestamp'], unit='s').date, 'Adj_Close': res['indicators']['adjclose'][0]['adjclose']})
                df = df.dropna().drop_duplicates('Date')
                
                # [Quality Check V12.2] Yahoo 종가(USD→KRW) vs 업비트 종가(KRW) 비교
                if ticker.endswith('-USD') and len(df) > 0:
                    symbol = ticker.replace('-USD', '')
                    try:
                        upbit_ohlcv = pyupbit.get_ohlcv(f"KRW-{symbol}", interval="day", count=1)
                        if upbit_ohlcv is not None and len(upbit_ohlcv) > 0:
                            upbit_close_krw = upbit_ohlcv['close'].iloc[-1]
                            yahoo_last_usd = df['Adj_Close'].iloc[-1]
                            yahoo_last_krw = yahoo_last_usd * usd_krw_rate
                            
                            if upbit_close_krw > 0 and yahoo_last_krw > 0:
                                diff_pct = abs(yahoo_last_krw - upbit_close_krw) / upbit_close_krw
                                if diff_pct > 0.10:
                                    print(f"  ⚠️ {ticker}: 가격 불일치 (Yahoo {yahoo_last_krw:,.0f}원 vs Upbit {upbit_close_krw:,.0f}원, diff={diff_pct:.0%}) - 제외")
                                    # 저장하지 않고 스킵
                                    continue 
                    except: pass
                
                df.to_csv(fp, index=False)
                success = True
                print(f"  - Downloaded {ticker} (Yahoo)")
                
        except Exception as e:
            # print(f"  ⚠️ Yahoo download failed for {ticker}: {e}")
            pass
            
        # 3. Fallback to CoinGecko
        if not success and ticker in coin_id_map:
             try:
                cid = coin_id_map[ticker]
                url = f"https://api.coingecko.com/api/v3/coins/{cid}/market_chart"
                # Timeout 30s
                resp = requests.get(url, params={'vs_currency':'usd','days':'500'}, timeout=30)
                if resp.status_code == 200:
                    data = resp.json().get('prices', [])
                    df = pd.DataFrame(data, columns=['ts', 'Adj_Close'])
                    df['Date'] = pd.to_datetime(df['ts'], unit='ms').dt.date
                    df[['Date','Adj_Close']].drop_duplicates('Date').to_csv(fp, index=False)
                    print(f"  - Downloaded {ticker} (CoinGecko)")
                    success = True
             except: pass
        
        # 4. Final Safety Net: Stale Data Fallback
        if not success:
            if os.path.exists(fp):
                # 기존 파일이 있으면 (어제 파일 등) 사용
                file_date = datetime.fromtimestamp(os.path.getmtime(fp)).date()
                print(f"  ⚠️ Failed to update {ticker}, using STALE data from {file_date}")
                log.append(f"<p class='warning'>Used stale data for {ticker} ({file_date})</p>")
                # success = True로 간주하지 않고, 에러 로그는 남기지 않음 (데이터가 있으므로)
            else:
                log.append(f"<p class='error'>XXX Failed to download: {ticker}</p>")


# --- 4. Logic Engines ---
def load_price(ticker):
    try:
        df = pd.read_csv(os.path.join(DATA_DIR, f"{ticker}.csv"), parse_dates=['Date'])
        # [Validation] Check Data Recency
        if df.empty: return pd.Series(dtype=float)
        
        last_date = df['Date'].iloc[-1].date() if hasattr(df['Date'].iloc[-1], 'date') else df['Date'].iloc[-1]
        today = datetime.now().date()
        
        # 만약 데이터 마지막 날짜가 7일 이상 지났으면 쓸모없는 데이터 (상폐/티커변경 등)
        if (today - last_date).days > 7:
            return pd.Series(dtype=float)

        return df.set_index('Date')['Adj_Close'].sort_index()
    except: return pd.Series(dtype=float)

def calc_sma(s, w): return s.rolling(w).mean().iloc[-1] if len(s) >= w else np.nan
def calc_ret(s, d): return s.iloc[-1]/s.iloc[-1-d] - 1 if len(s) >= d+1 and s.iloc[-1-d]!=0 else np.nan
def calc_sharpe(s, d):
    if len(s) < d+1: return 0
    ret = s.pct_change().iloc[-d:]
    return (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() != 0 else 0
def calc_weighted_mom(s):
    """V15: Pure 12-month momentum."""
    if len(s) < 253: return -np.inf
    return calc_ret(s, 252)

# --- V15 DD Exit / Blacklist ---
def check_dd_exit(s, lookback=DD_EXIT_LOOKBACK, threshold=DD_EXIT_THRESHOLD):
    """Check if coin should be exited: price / max(recent lookback days) - 1 < threshold."""
    if len(s) < lookback: return False, 0.0
    recent = s.iloc[-lookback:]
    peak = recent.max()
    if peak <= 0: return False, 0.0
    dd = s.iloc[-1] / peak - 1
    return dd <= threshold, dd

def check_blacklist(s, threshold=BL_THRESHOLD, lookback_days=BL_DAYS):
    """Check if coin had a daily drop worse than threshold in the last lookback_days."""
    if len(s) < lookback_days + 1: return False, 0.0
    recent = s.iloc[-(lookback_days + 1):]
    daily_rets = recent.pct_change().dropna()
    worst = daily_rets.min()
    return worst <= threshold, worst

def run_stock_strategy_v15(log, all_prices, target_date):
    """V17 Stock Strategy: R7 + EEM canary + Z-score3(Sh252) EW + Defense Top3 + VT Crash"""
    log.append("<h2>📈 주식 포트폴리오 분석 (V17: R7+EEM+Zscore3+Sh252+VT Crash)</h2>")

    # --- VT Crash Breaker (V17d: 동적 복귀) ---
    # 최근 60일을 시뮬레이션하여 오늘의 Crash 상태를 stateless로 도출
    vt = all_prices.get(STOCK_CRASH_TICKER)
    stock_crash = False
    crash_days_remaining = 0
    crash_trigger_day_idx = -1
    recovered_today = False

    if vt is not None and len(vt) >= 10:
        vt_rets = vt.pct_change()
        vt_sma10 = vt.rolling(10).mean()

        sim_len = min(60, len(vt))
        for i in range(len(vt) - sim_len, len(vt)):
            ret = vt_rets.iloc[i]
            cur_vt = vt.iloc[i]
            cur_sma = vt_sma10.iloc[i]
            is_today = (i == len(vt) - 1)

            # 새 Crash 발생 → 쿨다운 리셋
            if not np.isnan(ret) and ret <= STOCK_CRASH_THRESHOLD:
                stock_crash = True
                crash_days_remaining = STOCK_CRASH_COOL_DAYS
                crash_trigger_day_idx = i
                if is_today:
                    recovered_today = False

            # Crash 상태 → 쿨다운 처리 + V17d 동적 연장
            elif stock_crash:
                if crash_days_remaining > 0:
                    crash_days_remaining -= 1
                if crash_days_remaining == 0:
                    if not np.isnan(cur_sma) and cur_vt > cur_sma:
                        stock_crash = False
                        if is_today:
                            recovered_today = True
                    else:
                        crash_days_remaining = 1  # VT ≤ SMA10 → 1일 연장

        # 로그
        vt_ret_today = vt_rets.iloc[-1] if not np.isnan(vt_rets.iloc[-1]) else 0
        vt_cur_today = vt.iloc[-1]
        sma_cur_today = vt_sma10.iloc[-1] if not np.isnan(vt_sma10.iloc[-1]) else 0

        log.append(f"<p><b>[Crash Check]</b> {STOCK_CRASH_TICKER}: 일간 수익률 {vt_ret_today:+.2%} (임계 {STOCK_CRASH_THRESHOLD:.0%})</p>")

        if stock_crash:
            days_ago = (len(vt) - 1) - crash_trigger_day_idx if crash_trigger_day_idx >= 0 else 0
            if days_ago == 0:
                log.append(f"<div style='background:#fce8e6;border:2px solid #d93025;padding:16px;border-radius:8px;margin:12px 0'>"
                           f"<h3 style='color:#d93025;margin:0'>🚨 VT CRASH 발동! ({STOCK_CRASH_TICKER} {vt_ret_today:+.2%})</h3>"
                           f"<p style='font-size:1.2em;margin:8px 0'><b>즉시 행동:</b> 주식 전량 매도</p>"
                           f"<p><b>최소 {STOCK_CRASH_COOL_DAYS}영업일 + VT &gt; SMA10 회복 시 재진입</b></p></div>")
            elif days_ago < STOCK_CRASH_COOL_DAYS:
                log.append(f"<div style='background:#fef7e0;border:2px solid #f9ab00;padding:16px;border-radius:8px;margin:12px 0'>"
                           f"<h3 style='color:#e37400;margin:0'>⏸️ Crash 쿨다운 중</h3>"
                           f"<p>{days_ago}일 전 Crash 발동. 잔여 {crash_days_remaining}영업일 대기</p></div>")
            else:
                log.append(f"<div style='background:#fef7e0;border:2px solid #f9ab00;padding:16px;border-radius:8px;margin:12px 0'>"
                           f"<h3 style='color:#e37400;margin:0'>⏸️ Crash 동적 대기 (V17d)</h3>"
                           f"<p>VT ${vt_cur_today:.2f} &le; SMA10 ${sma_cur_today:.2f} → 현금 유지</p>"
                           f"<p style='font-size:1.2em'><b>VT &gt; SMA10 회복 시 재진입</b></p></div>")
        elif recovered_today:
            log.append(f"<p style='color:#0d904f'><b>✅ Crash 복귀 (V17d):</b> VT ${vt_cur_today:.2f} &gt; SMA10 ${sma_cur_today:.2f}</p>")

    # VT 기준가 계산 (executor가 Crash 판단에 사용)
    _vt_prev_close = float(vt.iloc[-1]) if vt is not None and len(vt) >= 1 else 0
    _vt_sma10 = float(vt.rolling(10).mean().iloc[-1]) if vt is not None and len(vt) >= 10 else 0

    # NOTE: Crash 판단은 executor가 함. recommend는 기준가만 제공.
    # stock_crash 변수는 HTML 리포트 표시용으로만 유지.
    if stock_crash:
        return {CASH_ASSET: 1.0}, "🚨 CRASH (전량 현금)", {'signal_dist': {}, 'next_candidates': [], '_vt_prev_close': _vt_prev_close, '_vt_sma10': _vt_sma10}

    eem = all_prices.get('EEM')
    meta = {'signal_dist': {}, 'next_candidates': [], '_vt_prev_close': _vt_prev_close, '_vt_sma10': _vt_sma10}
    # 이전 추천 종목 (HTML 리포트 비교 표시용)
    stock_holdings: list = []
    signal_flipped = False
    try:
        with open(SIGNAL_STATE_FILE, 'r') as _sf:
            _prev = json.load(_sf)
        stock_holdings = _prev.get('stock', {}).get('offense_picks', [])
    except Exception:
        pass

    if eem is not None and len(eem) >= STOCK_CANARY_MA_PERIOD:
        eem_sma = eem.rolling(STOCK_CANARY_MA_PERIOD).mean().iloc[-1]
        eem_cur = eem.iloc[-1]
        dist = eem_cur / eem_sma - 1
        meta['signal_dist'] = {'EEM': dist}

        # EEM-only canary with 0.5% hysteresis
        if dist > STOCK_CANARY_HYST:
            risk_on = True
        elif dist < -STOCK_CANARY_HYST:
            risk_on = False
        else:
            risk_on = eem_cur > eem_sma  # dead zone

        # 히스테리시스 dead zone → 이전 risk_on 참조
        prev_risk_on = None
        try:
            with open(SIGNAL_STATE_FILE, 'r') as _sf:
                prev_risk_on = json.load(_sf).get('stock', {}).get('risk_on')
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        # dead zone에서 이전 상태 유지
        if abs(dist) <= STOCK_CANARY_HYST and prev_risk_on is not None:
            risk_on = prev_risk_on
        signal_flipped = (prev_risk_on is not None and prev_risk_on != risk_on)

        log.append(f"<p><b>[Canary]</b> EEM: ${eem_cur:.2f} (MA{STOCK_CANARY_MA_PERIOD} ${eem_sma:.2f}, dist {dist:+.2%}, hyst ±{STOCK_CANARY_HYST:.1%})</p>")
        flip_info = " \U0001f504 <b>SIGNAL FLIP</b>" if signal_flipped else ""
        if risk_on: log.append(f"<p>\u2705 <b>Risk-On</b>{flip_info}</p>")
        else: log.append(f"<p>\U0001f6a8 <b>Risk-Off</b>{flip_info}</p>")
    else:
        risk_on = False
        log.append("<p class='error'>Canary Data Missing (EEM)</p>")

    if risk_on:
        log.append("<h4>🚀 공격 모드 (Z-score Top 3 + Sharpe252d + EW)</h4>")
        scores = []
        for t in OFFENSIVE_STOCK_UNIVERSE:
            p = all_prices.get(t)
            if p is None or len(p) < 253: continue
            scores.append({'Ticker': t, 'Mom12M': calc_weighted_mom(p), 'Sharpe252': calc_sharpe(p, 252)})

        if not scores:
            log.append("<p class='warning'>공격 ETF 데이터 부족 → 수비 전환</p>")
        else:
            df = pd.DataFrame(scores).set_index('Ticker')

            # Z-score composite: zscore(12M_mom) + zscore(Sharpe252d)
            m_std = df['Mom12M'].std()
            s_std = df['Sharpe252'].std()
            df['Z_Mom'] = (df['Mom12M'] - df['Mom12M'].mean()) / m_std if m_std > 0 else 0
            df['Z_Sh'] = (df['Sharpe252'] - df['Sharpe252'].mean()) / s_std if s_std > 0 else 0
            df['ZScore'] = df['Z_Mom'] + df['Z_Sh']

            try: log.append(f"<div class='table-wrap'>{df.to_html(classes='dataframe small-table', float_format='%.4f')}</div>")
            except: pass

            picks = df.nlargest(3, 'ZScore').index.tolist()
            meta['selection_reason'] = {'ZScore_Top3': picks}

            # Trigger detection: compare with actual holdings (signal_state.json의 stock_holdings)
            if stock_holdings:
                new_picks = sorted(set(picks) - set(stock_holdings))
                exit_picks = sorted(set(stock_holdings) - set(picks))
                n_changed = len(new_picks)
                is_monthly = (target_date.day <= 5)
                trigger_rebal = (n_changed >= 2 and not is_monthly)

                if trigger_rebal:
                    log.append(f"<p style='color:#d93025;font-size:1.1em'><b>🔄 TRIGGER REBALANCE: {n_changed}종목 변경</b></p>")
            else:
                log.append(f"<p style='color:#e37400'>⚠️ 보유종목 미설정</p>")

            log.append(f"<p>공격 {len(picks)}종목 선정 (Equal Weight)</p>")
            # NOTE: signal_state 저장은 main()에서 새 스키마로 일괄 저장
            return {t: 1.0/len(picks) for t in picks}, "공격 모드", meta

    # Defense mode: Top 3 by 6M return
    log.append("<h4>🛡️ 수비 모드 (Top 3 by 6M Return)</h4>")
    res = []
    for t in DEFENSIVE_STOCK_UNIVERSE:
        p = all_prices.get(t)
        if p is None: continue
        r = calc_ret(p, 126)
        if pd.notna(r): res.append({'Ticker': t, '6m Ret': r})

    try: log.append(f"<div class='table-wrap'>{pd.DataFrame(res).sort_values('6m Ret', ascending=False).to_html(classes='dataframe small-table')}</div>")
    except: pass

    if not res:
        return {CASH_ASSET: 1.0}, "수비 (데이터 없음)", meta

    res.sort(key=lambda x: x['6m Ret'], reverse=True)
    top3 = [r for r in res[:3] if r['6m Ret'] > 0]
    if not top3:
        return {CASH_ASSET: 1.0}, "수비 (전부 음수)", meta
    picks = [r['Ticker'] for r in top3]
    log.append(f"<p>Defense Picks: <b>{picks}</b> (Equal Weight)</p>")
    # NOTE: signal_state 저장은 main()에서 새 스키마로 일괄 저장
    return {t: 1.0/len(picks) for t in picks}, f"수비 ({', '.join(picks)})", meta

def run_coin_strategy_v15(coin_universe, all_prices, target_date, log, is_today=True):
    date_str = target_date.date()
    log.append(f"<h3>🪙 코인 포트폴리오 (V19) ({date_str})</h3>")
    meta = {'signal_dist': {}, 'next_candidates': []}

    btc = all_prices.get('BTC-USD')
    if len(btc) < COIN_CANARY_MA_PERIOD: return {CASH_ASSET: 1.0}, "데이터 부족", meta, log, []

    sma_val = btc.rolling(COIN_CANARY_MA_PERIOD).mean().iloc[-1]
    cur = btc.iloc[-1]
    meta['signal_dist'] = {'BTC': (cur - sma_val) / sma_val}
    tgt_dt = target_date.date() if hasattr(target_date, 'date') else target_date

    # --- Crash Breaker (G5): BTC daily -10% in last 3d → cash ---
    CRASH_COOLDOWN = 3
    if len(btc) >= CRASH_COOLDOWN + 1:
        crash_rets = btc.iloc[-(CRASH_COOLDOWN + 1):].pct_change().dropna()
        worst_crash = crash_rets.min()
        if worst_crash <= CRASH_THRESHOLD:
            log.append(f"<p class='error'><b>[CRASH BREAKER]</b> BTC worst {worst_crash:+.1%} in {CRASH_COOLDOWN}d — 현금 대기</p>")
            return {CASH_ASSET: 1.0}, "CRASH (BTC -10%)", meta, log, []

    # Load previous coin canary state
    prev_coin_risk_on = None
    try:
        with open(SIGNAL_STATE_FILE, 'r') as _sf:
            _prev_sig = json.load(_sf)
            prev_coin_risk_on = _prev_sig.get('coin', {}).get('risk_on', _prev_sig.get('coin_risk_on'))
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    # 1% Hysteresis Signal Flip Guard
    upper = 1 + COIN_CANARY_HYST  # 1.01
    lower = 1 - COIN_CANARY_HYST  # 0.99

    if prev_coin_risk_on is None:
        # First run: use simple comparison
        coin_risk_on = cur > sma_val
    elif prev_coin_risk_on:
        # Currently Risk-On: only flip to Off if BELOW lower band
        coin_risk_on = not (cur < sma_val * lower)
    else:
        # Currently Risk-Off: only flip to On if ABOVE upper band
        coin_risk_on = cur > sma_val * upper

    canary_off = not coin_risk_on

    # Save new coin state (merge with existing stock state)
    coin_signal_flipped = (prev_coin_risk_on is not None and prev_coin_risk_on != coin_risk_on)
    try:
        with open(SIGNAL_STATE_FILE, 'r') as _sf:
            _state = json.load(_sf)
    except (FileNotFoundError, json.JSONDecodeError):
        _state = {}
    # NOTE: signal_state 저장은 main()에서 새 스키마로 일괄 저장
    # trade_state 동기화 제거 (executor가 signal_state에서 직접 읽음)

    # Logging
    hyst_info = f"(Hyst {COIN_CANARY_HYST:.0%}: enter >{upper:.2f}x, exit <{lower:.2f}x)"
    flip_info = ""
    if coin_signal_flipped:
        flip_info = " 🔄 <b>SIGNAL FLIP</b>"

    canary_label = "🔴 Risk-Off (현금)" if canary_off else "🟢 Risk-On"
    log.append(f"<p><b>[Canary]</b> BTC ${cur:,.0f} vs SMA({COIN_CANARY_MA_PERIOD}) ${sma_val:,.0f} → {canary_label} (Date: {tgt_dt})</p>")
    log.append(f"<p><b>[Hysteresis]</b> {hyst_info} | Prev: {'On' if prev_coin_risk_on else 'Off' if prev_coin_risk_on is not None else 'N/A'}{flip_info}</p>")

    # --- Blacklist: -15% daily drop → 7d exclude ---
    blacklisted = []
    for t in coin_universe:
        p = all_prices.get(t)
        if p is None or len(p) < 2: continue
        is_bl, daily_ret = check_blacklist(p)
        if is_bl:
            blacklisted.append((t, daily_ret))
    if blacklisted:
        bl_text = ', '.join([f"{t} ({r:+.1%})" for t, r in blacklisted])
        log.append(f"<p class='warning'><b>[Blacklist]</b> {bl_text} — 7일 제외</p>")
    bl_set = {t for t, _ in blacklisted}
    filtered_universe = [t for t in coin_universe if t not in bl_set]

    # --- Health: Mom(30)>0 AND Mom(90)>0 AND Vol(90)<=5% ---
    healthy = []
    rows = []

    def fmt_price(p):
        if p < 1: return f"${p:,.8f}"
        if p < 100: return f"${p:,.4f}"
        return f"${p:,.2f}"

    for t in filtered_universe:
        p = all_prices.get(t)
        if p is None or len(p) < 91: continue

        last_dt = p.index[-1].date() if hasattr(p.index[-1], 'date') else p.index[-1]
        if (tgt_dt - last_dt).days != 0: continue

        cur_p = p.iloc[-1]
        mom30 = calc_ret(p, 30)
        mom90 = calc_ret(p, 90)
        vol90 = p.pct_change().iloc[-90:].std()

        is_ok = (pd.notna(mom30) and mom30 > 0 and
                 pd.notna(mom90) and mom90 > 0 and
                 vol90 <= VOL_CAP_FILTER)
        status = "🟢" if is_ok else "🔴"

        rows.append({'Coin': t, 'Price': fmt_price(cur_p),
                     'Mom30': f"{mom30:.2%}" if pd.notna(mom30) else "-",
                     'Mom90': f"{mom90:.2%}" if pd.notna(mom90) else "-",
                     'Vol90': f"{vol90:.4f}", 'Status': status})
        if is_ok:
            healthy.append(t)

    try: log.append(f"<div class='table-wrap'>{pd.DataFrame(rows).to_html(classes='dataframe small-table', index=False)}</div>")
    except: pass

    log.append(f"<p>🔍 Healthy Coins: <b>{len(healthy)}</b> {healthy}</p>")

    if canary_off:
        log.append(f"<p><b>→ 카나리아 OFF: 전량 현금 대기</b></p>")
        return {CASH_ASSET: 1.0}, "Risk-Off", meta, log, healthy

    if not healthy:
        return {CASH_ASSET: 1.0}, "No Healthy", meta, log, []

    # --- Selection: V19 Greedy Absorption ---
    top5 = healthy[:N_SELECTED_COINS]
    log.append(f"<p><b>[Selection]</b> 시총순 Top {N_SELECTED_COINS}: {top5}</p>")

    # Greedy: 시총 큰 코인이 Mom30 높으면 작은 코인 제거
    picks = list(top5)
    for i in range(len(picks) - 1, 0, -1):
        p_above = all_prices.get(picks[i-1])
        p_below = all_prices.get(picks[i])
        if p_above is not None and p_below is not None and len(p_above) > 30 and len(p_below) > 30:
            mom_above = float(p_above.iloc[-1]) / float(p_above.iloc[-31]) - 1
            mom_below = float(p_below.iloc[-1]) / float(p_below.iloc[-31]) - 1
            if mom_above >= mom_below:
                absorbed = picks.pop(i)
                log.append(f"<p>  ↑ {picks[i-1].replace('-USD','')}(Mom {mom_above:+.1%}) absorbs {absorbed.replace('-USD','')}(Mom {mom_below:+.1%})</p>")
    meta['next_candidates'] = healthy[N_SELECTED_COINS:N_SELECTED_COINS+5]
    log.append(f"<p><b>[Greedy]</b> 생존: {picks} ({len(picks)}개)</p>")

    top5 = picks  # 이후 로직은 picks 사용

    # --- Weighting: EW + Cap 33% ---
    COIN_WEIGHT_CAP = 1.0 / 3  # 33%
    w = min(1.0 / len(top5), COIN_WEIGHT_CAP)
    weights = {t: w for t in top5}
    w_rows = [{'Coin': t, 'Weight': f"{w:.2%}"} for t, w in weights.items()]
    try: log.append(f"<div class='table-wrap'>{pd.DataFrame(w_rows).to_html(classes='dataframe small-table', index=False)}</div>")
    except: pass

    # --- DD Exit Warning: 60d peak -25% ---
    dd_warnings = []
    for t in top5:
        p = all_prices.get(t)
        if p is None: continue
        is_exit, dd = check_dd_exit(p)
        if is_exit:
            dd_warnings.append((t, dd))
    if dd_warnings:
        dd_text = ', '.join([f"<b>{t}</b> ({dd:+.1%})" for t, dd in dd_warnings])
        log.append(f"<p class='error'><b>[DD EXIT]</b> {dd_text} — 매도 필요 (60d 고점 대비 -25% 초과)</p>")
        # Remove DD-exited coins: their weight goes to CASH (not redistributed)
        cash_weight = 0.0
        for t, _ in dd_warnings:
            if t in weights:
                cash_weight += weights[t]
                del weights[t]
        if cash_weight > 0:
            weights[CASH_ASSET] = weights.get(CASH_ASSET, 0.0) + cash_weight
        if not weights:
            weights = {CASH_ASSET: 1.0}

    invested_pct = sum(v for k, v in weights.items() if k != CASH_ASSET)
    cash_pct = 1.0 - invested_pct
    if cash_pct > 0.01:
        weights[CASH_ASSET] = weights.get(CASH_ASSET, 0) + cash_pct
        stat = f"투자 {invested_pct:.0%} / 현금 {cash_pct:.0%}"
    else:
        stat = "Full Invest"
    return weights, stat, meta, log, healthy

def save_html(log_global, final_port, s_port, c_port, s_stat, c_stat, turnover, log_today, log_yesterday, date_today, asset_prices_krw, s_meta, c_meta, coin_health_status, cur_assets_raw=None, action_guide="", diff_table_rows=None, coin_total_krw=0):
    filepath = PORTFOLIO_HTML_NAME

    # 현금 버퍼 (coin_trade_state에서 읽기)
    cash_buffer_pct = get_cash_buffer()
    try:
        pass
    except Exception:
        pass

    items = []
    for t, w in final_port.items(): items.append({'종목': t, '자산군': "현금" if t == CASH_ASSET else ("코인" if t in c_port else "주식"), '비중': w})
    items.sort(key=lambda x: (x['자산군']!='현금', x['비중']), reverse=True)
    
    tbody = "".join([f"<tr><td>{i['종목']}</td><td>{i['자산군']}</td><td>{i['비중']:.2%}</td></tr>" for i in items])
    
    # [Table] Integrated Portfolio (My vs Target)
    integrated_html = ""
    if diff_table_rows:
        integrated_html = f"<h3>Turnover: {turnover:.2%} ({action_guide})</h3>"
        integrated_html += "<table class='mobile-card-table'><thead><tr><th>Asset</th><th>My</th><th>Target</th><th>Diff</th><th>Action</th></tr></thead><tbody>"
        
        total_value_sum = sum(item['Value'] for item in diff_table_rows)
        
        for row in diff_table_rows:
            color = ""
            if "BUY" in row['Action']: color = "color:red; font-weight:bold;"
            elif "SELL" in row['Action']: color = "color:blue; font-weight:bold;"
            
            val_fmt = f"{int(row['Value']):,}" if row['Value'] > 0 else "0"
            
            integrated_html += f"<tr><td data-label='Asset'>{row['Asset']}</td><td data-label='My'>{row['My']:.1%}</td><td data-label='Target'>{row['Target']:.1%}</td><td data-label='Diff'>{row['Diff']:+.1%}</td><td data-label='Action' style='{color}'>{row['Action']}</td></tr>"
        
        integrated_html += f"<tr><td data-label='Total' style='font-weight:bold;'>Total</td><td data-label='My'></td><td data-label='Target'></td><td data-label='Diff'></td><td data-label='Action'></td></tr>"
        integrated_html += "</tbody></table>"

    # Strategy documentation link
    version_html = ""

    # Embed recommended stock tickers + prices for client-side calculation
    rec_stock_list = sorted([t for t in s_port.keys() if t != 'Cash'])
    rec_stock_json = json.dumps(rec_stock_list)

    # ETF 종가(USD) 임베딩 — 주수 계산용
    stock_prices_for_js = {}
    for t in rec_stock_list:
        p = prices.get(t)
        if p is not None and not p.empty:
            stock_prices_for_js[t] = round(float(p.iloc[-1]), 2)
    stock_prices_json = json.dumps(stock_prices_for_js)

    # Read signal state for UI
    signal_flipped = False
    current_risk_on = True
    saved_stock_holdings = []
    try:
        with open(SIGNAL_STATE_FILE, 'r') as _sf:
            _state = json.load(_sf)
            signal_flipped = False  # 플립 감지는 executor가 담당
            current_risk_on = _state.get('stock', {}).get('risk_on', True)
            saved_stock_holdings = _state.get('stock', {}).get('offense_picks', [])
    except (FileNotFoundError, json.JSONDecodeError):
        pass

    saved_holdings_json = json.dumps(saved_stock_holdings)
    coin_total_krw_val = int(coin_total_krw)
    stock_holdings_js = """
            <script>
            const REC_STOCK_TICKERS = """ + rec_stock_json + """;
            const STOCK_PRICES_USD = """ + stock_prices_json + """;
            const COIN_TOTAL_KRW = """ + str(coin_total_krw_val) + """;
            const TARGET_STOCK_RATIO = 0.60;
            const TARGET_COIN_RATIO = 0.25;
            const TARGET_FUTURES_RATIO = 0.15;
            const REBAL_BAND = 0.08;  // ±8%p
            const SIGNAL_FLIPPED = """ + ("true" if signal_flipped else "false") + """;
            const RISK_ON = """ + ("true" if current_risk_on else "false") + """;
            const SAVED_STOCK_HOLDINGS = """ + saved_holdings_json + """;  // signal_state.json에서 로드

            function calcTrigger(myTickers, recTickers) {
                if (!myTickers.length || !recTickers.length) return null;
                const allT = [...new Set([...myTickers, ...recTickers])];
                let totalDiff = 0;
                allT.forEach(t => {
                    const myW = myTickers.includes(t) ? 1.0/myTickers.length : 0;
                    const recW = recTickers.includes(t) ? 1.0/recTickers.length : 0;
                    totalDiff += Math.abs(recW - myW);
                });
                const turnover = Math.round(totalDiff / 2 * 10000) / 10000;

                const added = recTickers.filter(t => !myTickers.includes(t));
                const removed = myTickers.filter(t => !recTickers.includes(t));
                return { turnover, added, removed };
            }

            function renderTrigger(myTickers) {
                const result = calcTrigger(myTickers, REC_STOCK_TICKERS);
                const el = document.getElementById('triggerResult');
                if (!el || !result) return;

                let changesHtml = '';
                if (result.added.length)
                    changesHtml += '<span style="color:#d93025; font-weight:600;">+ ' + result.added.join(', ') + '</span> ';
                if (result.removed.length)
                    changesHtml += '<span style="color:#1a73e8; font-weight:600;">- ' + result.removed.join(', ') + '</span>';
                if (!result.added.length && !result.removed.length)
                    changesHtml = '<span style="color:#0d904f;">\\u2705 \\ub3d9\\uc77c \\uc885\\ubaa9</span>';

                const pct = Math.round(result.turnover * 100);
                const regime = RISK_ON ? 'Risk-On &#x1F7E2;' : 'Risk-Off &#x1F534;';
                let statusHtml = '';
                if (SIGNAL_FLIPPED) {
                    statusHtml = '<div style="background:#fce8e6; border:2px solid #d93025; padding:12px; border-radius:8px; margin-top:10px;">'
                        + '&#x1F6A8; <b>Signal Flip \ubc1c\uc0dd (' + regime + ') \u2014 \uc989\uc2dc \ub9ac\ubc38\ub7f0\uc2f1 \ud544\uc694</b></div>';
                } else if (pct === 0) {
                    statusHtml = '<div style="background:#e8f5e9; padding:12px; border-radius:8px; margin-top:10px;">'
                        + '\u2705 ' + regime + ' | \ud604\uc7ac \ubcf4\uc720 = \ucd94\ucc9c \uc885\ubaa9 (\ub9ac\ubc38\ub7f0\uc2f1 \ubd88\ud544\uc694)</div>';
                } else if (result.added.length >= 2) {
                    statusHtml = '<div style="background:#fce8e6; border:2px solid #d93025; padding:12px; border-radius:8px; margin-top:10px;">'
                        + '&#x1F6A8; ' + regime + ' | <b>' + result.added.length + '\uc885\ubaa9 \ubcc0\uacbd \u2014 \ud2b8\ub9ac\uac70 \ub9ac\ubc38\ub7f0\uc2f1</b></div>';
                } else {
                    statusHtml = '<div style="background:#fff3e0; border:1px solid #ff9800; padding:12px; border-radius:8px; margin-top:10px;">'
                        + '&#x1F504; ' + regime + ' | ' + result.added.length + '\uc885\ubaa9 \ubcc0\uacbd \u2014 <b>\uc6d4\ucd08 \ub9ac\ubc38\ub7f0\uc2f1 \uc2dc \ubc18\uc601</b></div>';
                }

                el.innerHTML = '<div style="margin-top: 10px;">'
                    + '<div style="display:flex; gap:20px; flex-wrap:wrap; margin-bottom:8px;">'
                    + '<div><b>My:</b> ' + myTickers.join(', ') + ' (' + myTickers.length + ')</div>'
                    + '<div><b>Rec:</b> ' + REC_STOCK_TICKERS.join(', ') + ' (' + REC_STOCK_TICKERS.length + ')</div>'
                    + '</div>'
                    + '<div style="margin-bottom:8px;">Changes: ' + changesHtml + '</div>'
                    + statusHtml
                    + '</div>';
            }

            // Load on page init: localStorage 우선, 없으면 signal_state.json fallback
            (function() {
                try {
                    const saved = localStorage.getItem('cap_defend_stock_holdings');
                    if (saved) {
                        const data = JSON.parse(saved);
                        if (data.tickers && data.tickers.length > 0) {
                            document.getElementById('stockInput').value = data.tickers.join(' ');
                            document.getElementById('holdingsStatus').innerHTML =
                                '\u2705 \uc800\uc7a5\ub428: ' + data.tickers.join(', ') + ' (' + data.updated + ')';
                            renderTrigger(data.tickers);
                            return;
                        }
                    }
                    // Fallback: signal_state.json에서 가져온 값
                    if (SAVED_STOCK_HOLDINGS && SAVED_STOCK_HOLDINGS.length > 0) {
                        document.getElementById('stockInput').value = SAVED_STOCK_HOLDINGS.join(' ');
                        document.getElementById('holdingsStatus').innerHTML =
                            '\u2705 \uc11c\ubc84 \uc800\uc7a5\uac12: ' + SAVED_STOCK_HOLDINGS.join(', ');
                        renderTrigger(SAVED_STOCK_HOLDINGS);
                    }
                } catch(e) {}
            })();

            function saveHoldings() {
                const input = document.getElementById('stockInput').value.trim();
                const status = document.getElementById('holdingsStatus');
                if (!input) {
                    status.innerHTML = '\u274c \uc885\ubaa9\uc744 \uc785\ub825\ud574\uc8fc\uc138\uc694';
                    status.style.color = '#d93025';
                    return;
                }
                const tickers = input.toUpperCase().split(/\\s+/).filter(t => t.length > 0);
                const now = new Date().toLocaleString('ko-KR');
                const data = { tickers: tickers, updated: now };
                try {
                    localStorage.setItem('cap_defend_stock_holdings', JSON.stringify(data));
                    status.innerHTML = '\u2705 \uc800\uc7a5 \uc644\ub8cc: ' + tickers.join(', ');
                    status.style.color = '#0d904f';
                    renderTrigger(tickers);
                } catch(e) {
                    status.innerHTML = '\u274c \uc800\uc7a5 \uc2e4\ud328';
                    status.style.color = '#d93025';
                }
            }

            // === 통합 리밸런싱 계산기 (3자산: 주식/현물/선물) ===
            let liveCoinKRW = COIN_TOTAL_KRW;
            let liveFuturesKRW = 0;
            let coinFetchTime = '리포트 생성 시점';

            async function fetchCoinBalance() {
                const statusEl = document.getElementById('coinFetchStatus');
                try {
                    statusEl.innerHTML = '\\u23f3 \\ucf54\\uc778 \\uc794\\uace0 \\uc870\\ud68c \\uc911...';
                    statusEl.style.color = '#1967d2';
                    const resp = await fetch('http://' + window.location.hostname + ':5000/api/assets/coin_balance');
                    if (!resp.ok) throw new Error('API error');
                    const data = await resp.json();
                    liveCoinKRW = data.total_krw;
                    coinFetchTime = data.updated;
                    statusEl.innerHTML = '\\u2705 \\uc2e4\\uc2dc\\uac04 \\uc870\\ud68c \\uc644\\ub8cc (' + data.updated + ')';
                    statusEl.style.color = '#0d904f';
                    return true;
                } catch(e) {
                    statusEl.innerHTML = '\\u26a0\\ufe0f \\uc870\\ud68c \\uc2e4\\ud328 \\u2014 \\ub9ac\\ud3ec\\ud2b8 \\uc2dc\\uc810 \\uac12 \\uc0ac\\uc6a9';
                    statusEl.style.color = '#e37400';
                    return false;
                }
            }

            async function fetchFuturesBalance() {
                try {
                    const resp = await fetch('http://' + window.location.hostname + ':5000/api/assets/binance_balance');
                    if (!resp.ok) throw new Error('API error');
                    const data = await resp.json();
                    liveFuturesKRW = data.total_krw || 0;
                    return true;
                } catch(e) {
                    return false;
                }
            }

            async function calcRebalance() {
                // 주식 + 현물코인 + 선물 + 환율 동시 조회
                await Promise.all([fetchCoinBalance(), fetchFuturesBalance(), fetchStockBalance()]);

                const stockInput = String(getStockTotal());
                const resultEl = document.getElementById('rebalResult');

                const stockKRW = parseFloat(stockInput);
                const coinKRW = parseFloat(document.getElementById('snapCoin').value) || liveCoinKRW;
                const futuresKRW = parseFloat(document.getElementById('snapFutures').value) || liveFuturesKRW;
                const rate = parseFloat(document.getElementById('exchangeRate').value) || 0;
                if (!document.getElementById('snapCoin').value) document.getElementById('snapCoin').value = Math.round(coinKRW);
                if (!document.getElementById('snapFutures').value && liveFuturesKRW > 0) document.getElementById('snapFutures').value = Math.round(liveFuturesKRW);

                if (!stockKRW && stockKRW !== 0) { resultEl.innerHTML = '\\u274c \\uc8fc\\uc2dd \\ucd1d\\uc561\\uc744 \\uc785\\ub825\\ud574\\uc8fc\\uc138\\uc694'; return; }

                // 1. 3자산 비중 계산
                const totalAsset = stockKRW + coinKRW + futuresKRW;
                if (totalAsset <= 0) { resultEl.innerHTML = '\\u274c \\uc790\\uc0b0\\uc774 0\\uc785\\ub2c8\\ub2e4'; return; }

                const curStockPct = stockKRW / totalAsset;
                const curCoinPct = coinKRW / totalAsset;
                const curFuturesPct = futuresKRW / totalAsset;

                const driftStock = Math.abs(curStockPct - TARGET_STOCK_RATIO);
                const driftCoin = Math.abs(curCoinPct - TARGET_COIN_RATIO);
                const driftFutures = Math.abs(curFuturesPct - TARGET_FUTURES_RATIO);
                const maxDrift = Math.max(driftStock, driftCoin, driftFutures);
                const needRebal = maxDrift >= REBAL_BAND;

                // 2. 목표 금액
                const targetStockKRW = totalAsset * TARGET_STOCK_RATIO;
                const targetCoinKRW = totalAsset * TARGET_COIN_RATIO;
                const targetFuturesKRW = totalAsset * TARGET_FUTURES_RATIO;

                // 3. 비중 카드 (3열)
                function pctColor(cur, target) {
                    return Math.abs(cur - target) >= REBAL_BAND ? '#d93025' : '#0d904f';
                }
                function assetCard(label, krw, cur, target) {
                    return '<div style="padding:14px; background:#f8f9fa; border-radius:10px;">'
                        + '<div style="font-size:0.85em; color:#666;">' + label + '</div>'
                        + '<div style="font-size:1.25em; font-weight:700;">' + Math.round(krw).toLocaleString() + '\\uc6d0</div>'
                        + '<div style="color:' + pctColor(cur, target) + '; font-weight:600;">'
                        + (cur * 100).toFixed(1) + '% (\\ubaa9\\ud45c ' + (target * 100).toFixed(0) + '%)</div></div>';
                }

                let html = '<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:10px; margin-bottom:16px;">';
                html += assetCard('\\uc8fc\\uc2dd', stockKRW, curStockPct, TARGET_STOCK_RATIO);
                html += assetCard('\\ud604\\ubb3c\\ucf54\\uc778', coinKRW, curCoinPct, TARGET_COIN_RATIO);
                html += assetCard('\\uc120\\ubb3c', futuresKRW, curFuturesPct, TARGET_FUTURES_RATIO);
                html += '</div>';

                html += '<div style="padding:8px 14px; background:#e8eaf6; border-radius:8px; margin-bottom:12px;">'
                    + '<b>\\ucd1d \\uc790\\uc0b0:</b> ' + Math.round(totalAsset).toLocaleString() + '\\uc6d0</div>';

                // 4. 리밸런싱 판단
                if (needRebal) {
                    let moves = [];
                    const deltas = [
                        {name: '\\uc8fc\\uc2dd', delta: targetStockKRW - stockKRW},
                        {name: '\\ud604\\ubb3c', delta: targetCoinKRW - coinKRW},
                        {name: '\\uc120\\ubb3c', delta: targetFuturesKRW - futuresKRW},
                    ];
                    deltas.forEach(d => {
                        if (Math.abs(d.delta) > 10000) {
                            const sign = d.delta > 0 ? '+' : '';
                            moves.push(d.name + ' ' + sign + Math.round(d.delta).toLocaleString() + '\\uc6d0');
                        }
                    });
                    html += '<div style="padding:14px; background:#fce8e6; border:2px solid #d93025; border-radius:10px; margin-bottom:16px;">'
                        + '<div style="font-size:1.1em; font-weight:700; color:#d93025; margin-bottom:6px;">\\u26a0\\ufe0f \\ub9ac\\ubc38\\ub7f0\\uc2f1 \\ud544\\uc694 (\\ucd5c\\ub300\\ud3b8\\ucc28 ' + (maxDrift * 100).toFixed(1) + '%p \\u2265 ' + (REBAL_BAND * 100).toFixed(0) + '%p)</div>'
                        + '<div style="font-size:1.05em;">' + moves.join('<br>') + '</div>'
                        + '</div>';
                } else {
                    html += '<div style="padding:14px; background:#e8f5e9; border:1px solid #0d904f; border-radius:10px; margin-bottom:16px;">'
                        + '<div style="font-size:1.1em; font-weight:700; color:#0d904f;">\\u2705 \\ub9ac\\ubc38\\ub7f0\\uc2f1 \\ubd88\\ud544\\uc694 (\\ucd5c\\ub300\\ud3b8\\ucc28 ' + (maxDrift * 100).toFixed(1) + '%p < ' + (REBAL_BAND * 100).toFixed(0) + '%p)</div>'
                        + '</div>';
                }

                // 5. ETF 주수 계산 (환율 입력 시만)
                const finalStockKRW = needRebal ? targetStockKRW : stockKRW;
                const tickers = REC_STOCK_TICKERS.filter(t => t in STOCK_PRICES_USD);
                if (tickers.length > 0 && rate > 0) {
                    const perETF = finalStockKRW / tickers.length;
                    let rows = '';
                    let totalUsed = 0;

                    tickers.forEach(t => {
                        const priceUSD = STOCK_PRICES_USD[t];
                        const priceKRW = priceUSD * rate;
                        const shares = Math.floor(perETF / priceKRW);
                        const usedKRW = shares * priceKRW;
                        totalUsed += usedKRW;
                        const kisQty = kisHoldings[t] || 0;
                        const shinhanQty = Math.max(0, shares - kisQty);
                        rows += '<tr>'
                            + '<td data-label="ETF" style="font-weight:600;">' + t + '</td>'
                            + '<td data-label="\\uac00\\uaca9">$' + priceUSD.toFixed(2) + '</td>'
                            + '<td data-label="\\ucd1d\\uc8fc\\uc218" style="font-size:1.2em;font-weight:700;color:#1a73e8;">' + shares + '\\uc8fc</td>'
                            + '<td data-label="\\ud55c\\ud22c">' + kisQty + '\\uc8fc</td>'
                            + '<td data-label="\\uc2e0\\ud55c">' + shinhanQty + '\\uc8fc</td>'
                            + '<td data-label="\\uae08\\uc561">' + Math.round(usedKRW).toLocaleString() + '\\uc6d0</td>'
                            + '</tr>';
                    });

                    const remainder = finalStockKRW - totalUsed;
                    html += '<h3 style="margin:16px 0 8px 0;">\\uc8fc\\uc2dd ETF \\ubcf4\\uc720 \\uac00\\uc774\\ub4dc' + (needRebal ? ' (\\ub9ac\\ubc38\\ub7f0\\uc2f1 \\ud6c4)' : '') + '</h3>';
                    html += '<table class="mobile-card-table">'
                        + '<thead><tr><th>ETF</th><th>\\uac00\\uaca9</th><th>\\ucd1d\\uc8fc\\uc218</th><th>\\ud55c\\ud22c</th><th>\\uc2e0\\ud55c</th><th>\\uae08\\uc561</th></tr></thead>'
                        + '<tbody>' + rows + '</tbody></table>';
                    html += '<div style="margin-top:10px;padding:10px;background:#f0f4ff;border-radius:8px;">'
                        + '<b>\\uc8fc\\uc2dd \\ud22c\\uc790:</b> ' + Math.round(finalStockKRW).toLocaleString() + '\\uc6d0'
                        + ' | <b>\\uc2e4\\uc81c \\ub9e4\\uc218:</b> ' + Math.round(totalUsed).toLocaleString() + '\\uc6d0'
                        + ' | <b>\\uc794\\uc5ec:</b> ' + Math.round(remainder).toLocaleString() + '\\uc6d0'
                        + '</div>';
                }

                if (tickers.length > 0 && rate <= 0) {
                    html += '<p style="color:#888; font-size:0.9em;">\\ud658\\uc728\\uc744 \\uc785\\ub825\\ud558\\uba74 ETF \\uc8fc\\uc218\\ub3c4 \\uacc4\\uc0b0\\ud569\\ub2c8\\ub2e4.</p>';
                }

                resultEl.innerHTML = html;
            }
            </script>
    """

    # 자산관리 UI는 실계좌 조회 전용으로 교체
    stock_holdings_js = ""

    # 실행 상태 요약 (상세 로그용) — 주식/현물/선물 공통 템플릿
    def _fmt_alloc(d: dict) -> str:
        if not isinstance(d, dict) or not d:
            return "없음"
        return ", ".join(
            f"{k}:{float(v):.1%}"
            for k, v in sorted(d.items(), key=lambda kv: float(kv[1]), reverse=True)
        )

    def _fmt_alloc_lines(d: dict) -> str:
        if not isinstance(d, dict) or not d:
            return "없음"
        return "<br>".join(
            f"{k}: {float(v):.1%}"
            for k, v in sorted(d.items(), key=lambda kv: float(kv[1]), reverse=True)
        )

    def _fmt_bool(v) -> str:
        return "예" if bool(v) else "아니오"

    def _load_first_json(candidates):
        for cand in candidates:
            if cand and os.path.exists(cand):
                try:
                    with open(cand, "r", encoding="utf-8") as f:
                        return json.load(f), cand
                except Exception:
                    continue
        return {}, ""

    def _card_list_html(rows, columns=None):
        if not rows:
            return "<p>기록 없음</p>"
        try:
            if columns:
                use_cols = columns
            else:
                use_cols = list(rows[0].keys())
            cards = []
            for row in rows:
                parts = []
                for col in use_cols:
                    val = row.get(col, "-")
                    parts.append(
                        "<div class='state-row'>"
                        f"<div class='state-label'>{col}</div>"
                        f"<div class='state-value'>{val}</div>"
                        "</div>"
                    )
                cards.append("<div class='state-card'>" + "".join(parts) + "</div>")
            return "<div class='state-cards'>" + "".join(cards) + "</div>"
        except Exception:
            return "<p>표 생성 실패</p>"

    def _strategy_block(title: str, summary_rows: list, tranche_rows: list, extra_html: str = "") -> str:
        parts = [f"<h2>{title}</h2>"]
        if summary_rows:
            parts.append("<div class='summary-list'>")
            for row in summary_rows:
                parts.append(f"<div class='summary-item'>{row}</div>")
            parts.append("</div>")
        if tranche_rows:
            parts.append("<h3>트랜치 상태</h3>")
            parts.append(_card_list_html(tranche_rows))
        if extra_html:
            parts.append(extra_html)
        return "".join(parts)

    def _merge_tranches(tranches: dict) -> dict:
        merged = {}
        if not isinstance(tranches, dict) or not tranches:
            return merged
        n = len(tranches)
        for tr in tranches.values():
            for ticker, weight in (tr.get('weights', {}) or {}).items():
                merged[ticker] = merged.get(ticker, 0.0) + float(weight) / n
        return {k: v for k, v in merged.items() if v > 0}

    def _next_anchor_str(anchor_days) -> str:
        now = datetime.now()
        year, month = now.year, now.month
        candidates = []
        for day in anchor_days:
            try:
                candidates.append(datetime(year, month, day))
            except ValueError:
                continue
        future = [dt for dt in candidates if dt.date() >= now.date()]
        if future:
            nxt = min(future)
        else:
            if month == 12:
                nxt = datetime(year + 1, 1, anchor_days[0])
            else:
                nxt = datetime(year, month + 1, anchor_days[0])
        return nxt.strftime('%Y-%m-%d')

    _this_month = datetime.now().strftime('%Y-%m')

    def _tranche_status(anchor_month: str, weights: dict, picks: list) -> Tuple[str, str, str]:
        has_alloc = isinstance(weights, dict) and len(weights) > 0
        has_picks = isinstance(picks, list) and len(picks) > 0
        if has_alloc or has_picks:
            if has_alloc and set(weights.keys()) == {'Cash'} and float(weights.get('Cash', 0)) >= 0.999:
                return "현금 유지", "없음", _fmt_alloc(weights)
            return "활성", ", ".join(picks or []) or "없음", _fmt_alloc(weights)
        if anchor_month and anchor_month < _this_month:
            return "미초기화", "다음 앵커 대기", "다음 앵커 대기"
        return "대기", "없음", "없음"

    def _next_tranche_refresh_str(last_bar_ts: str, interval_hours: int, bar_counter, snap_iv: int, n_snap: int) -> str:
        if not last_bar_ts or last_bar_ts == "-" or snap_iv <= 0 or n_snap <= 0:
            return "-"
        try:
            dt = pd.to_datetime(last_bar_ts)
            # Futures state stores bar timestamps as naive UTC strings.
            if getattr(dt, 'tzinfo', None) is None:
                dt = dt.tz_localize("UTC")
            c = int(bar_counter)
            offsets = [int(si * snap_iv / n_snap) for si in range(n_snap)]
            # state['bar_counter'] is incremented after the latest completed bar is processed.
            # The next refresh happens on the first future bar where the pre-increment counter
            # matches one of the tranche offsets.
            next_n = None
            for n in range(0, snap_iv + 1):
                if ((c + n) % snap_iv) in offsets:
                    next_n = n
                    break
            if next_n is None:
                next_n = 0
            nxt = dt + pd.Timedelta(hours=interval_hours * (next_n + 1))
            nxt = pd.Timestamp(nxt).tz_convert("Asia/Seoul")
            return nxt.strftime('%Y-%m-%d %H:%M:%S KST')
        except Exception:
            return "-"

    def _fmt_run_ts(ts: str) -> str:
        if not ts or ts == "-":
            return "-"
        try:
            dt = pd.to_datetime(ts)
            if getattr(dt, "tzinfo", None) is None:
                return pd.Timestamp(dt).strftime('%Y-%m-%d %H:%M:%S')
            dt = dt.tz_convert("Asia/Seoul")
            return pd.Timestamp(dt).strftime('%Y-%m-%d %H:%M:%S KST')
        except Exception:
            return str(ts)

    _base_dir = globals().get("BASE_DIR", os.getcwd())
    state_sections = []

    # ── 주식 실행 상태 ──
    try:
        _stock_state, _stock_path = _load_first_json([
            os.path.join(APP_HOME, "kis_trade_state.json"),
            os.path.join(_base_dir, "trade", "kis_trade_state.json"),
            "kis_trade_state.json",
        ])
        _stock_signal, _ = _load_first_json([
            SIGNAL_STATE_FILE,
            os.path.join(_base_dir, SIGNAL_STATE_FILE),
        ])
        _stk = _stock_signal.get("stock", {}) or {}
        _stock_exec_target = _merge_tranches(_stock_state.get("tranches", {}) or {})
        _stock_summary = [
            f"<b>리스크 상태:</b> {'Risk-On' if _stk.get('risk_on', True) else 'Risk-Off'}",
            f"<b>리밸런싱 대기:</b> {_fmt_bool(_stock_state.get('rebalancing_needed'))}",
            f"<b>마지막 실행일:</b> {_stock_state.get('last_trade_date', '-')}",
            f"<b>실행 목표:</b> {_fmt_alloc(_stock_exec_target)}",
            f"<b>다음 앵커:</b> {_next_anchor_str(STOCK_ANCHOR_DAYS)}",
        ]
        if s_meta.get('signal_dist'):
            _dist = s_meta['signal_dist']
            _stock_summary.append(
                "<b>카나리 거리:</b> " + ", ".join(f"{k}:{float(v):+.2%}" for k, v in _dist.items())
            )
        _stock_tr_rows = []
        for _anchor in sorted((_stock_state.get("tranches", {}) or {}).keys(), key=lambda x: int(x)):
            _tr = (_stock_state.get("tranches", {}) or {}).get(_anchor, {}) or {}
            _status, _picks_text, _weights_text = _tranche_status(
                _tr.get("anchor_month", "-"),
                _tr.get("weights", {}),
                _tr.get("picks", []),
            )
            _stock_tr_rows.append({
                "트랜치": f"D{_anchor}",
                "상태": _status,
                "기준월": _tr.get("anchor_month", "-"),
                "종목": _picks_text,
                "비중": _weights_text,
            })
        state_sections.append(_strategy_block("📘 주식 실행 상태", _stock_summary, _stock_tr_rows))
    except Exception as _e:
        state_sections.append(f"<h2>📘 주식 실행 상태</h2><p class='error'>상태 조회 실패: {_e}</p>")

    # ── 현물 코인 실행 상태 ──
    try:
        _coin_state, _coin_path = _load_first_json([
            os.path.join(APP_HOME, "coin_trade_state.json"),
            os.path.join(_base_dir, "trade", "coin_trade_state.json"),
            "coin_trade_state.json",
        ])
        _coin_signal = _stock_signal if '_stock_signal' in locals() else load_json(SIGNAL_STATE_FILE)
        _coin_sig = _coin_signal.get("coin", {}) or {}
        _coin_risk_on = bool(_coin_sig.get('risk_on', True))
        _coin_exec_target_text = "Cash:100.0%" if not _coin_risk_on else _fmt_alloc(_merge_tranches(_coin_state.get("tranches", {}) or {}))
        _coin_summary = [
            f"<b>리스크 상태:</b> {'Risk-On' if _coin_risk_on else 'Risk-Off'}",
            f"<b>리밸런싱 대기:</b> {_fmt_bool(_coin_state.get('rebalancing_needed'))}",
            f"<b>마지막 플립:</b> {_coin_state.get('last_flip_date', '-')}",
            f"<b>마지막 실행일:</b> {_coin_state.get('last_trade_date', '-')}",
            f"<b>실행 목표:</b> {_coin_exec_target_text}",
            f"<b>다음 앵커:</b> {_next_anchor_str(COIN_ANCHOR_DAYS)}",
        ]
        if c_meta.get('signal_dist'):
            _dist = c_meta['signal_dist']
            _coin_summary.append(
                "<b>카나리 거리:</b> " + ", ".join(f"{k}:{float(v):+.2%}" for k, v in _dist.items())
            )
        _coin_exclusions = ((_coin_state.get("guard_state", {}) or {}).get("exclusions", {}) or {})
        if _coin_exclusions:
            _coin_summary.append(
                "<b>가드 제외:</b> " + ", ".join(
                    f"{k}({v.get('reason','-')}, until={v.get('until_date') or '-'})"
                    for k, v in _coin_exclusions.items()
                )
            )
        _coin_tr_rows = []
        for _anchor in sorted((_coin_state.get("tranches", {}) or {}).keys(), key=lambda x: int(x)):
            _tr = (_coin_state.get("tranches", {}) or {}).get(_anchor, {}) or {}
            _status, _picks_text, _weights_text = _tranche_status(
                _tr.get("anchor_month", "-"),
                _tr.get("weights", {}),
                _tr.get("picks", []),
            )
            if not _coin_risk_on and not (_tr.get("weights", {}) or {}):
                _status, _picks_text, _weights_text = ("현금 유지", "없음", "Cash:100.0%")
            _coin_tr_rows.append({
                "트랜치": f"D{_anchor}",
                "상태": _status,
                "기준월": _tr.get("anchor_month", "-"),
                "종목": _picks_text,
                "비중": _weights_text,
            })
        state_sections.append(_strategy_block("📘 현물 코인 실행 상태", _coin_summary, _coin_tr_rows))
    except Exception as _e:
        state_sections.append(f"<h2>📘 현물 코인 실행 상태</h2><p class='error'>상태 조회 실패: {_e}</p>")

    # ── 선물 실행 상태 ──
    try:
        _fut, _fut_path = _load_first_json([
            os.path.join(APP_HOME, "binance_state.json"),
            os.path.join(_base_dir, "trade", "binance_state.json"),
            "binance_state.json",
        ])
        _strategies = _fut.get("strategies", {}) or {}
        _last_target = _fut.get("last_target", {}) or {}
        _fut_summary = [
            f"<b>리밸런싱 대기:</b> {_fmt_bool(_fut.get('rebalancing_needed'))}",
            f"<b>마지막 실행:</b> {_fmt_run_ts(_fut.get('last_run', '-'))}",
            f"<b>비상 정지:</b> {_fmt_bool(_fut.get('kill_switch'))}" + (f" ({_fut.get('kill_switch_reason')})" if _fut.get('kill_switch') else ""),
            f"<b>합산 목표:</b><br>{_fmt_alloc_lines(_last_target)}",
        ]
        _fut_tr_rows = []
        for _name in list(FUTURES_TRANCHE_META.keys()):
            _st = _strategies.get(_name, {}) or {}
            _meta = FUTURES_TRANCHE_META.get(_name, {})
            _snapshots_text = "<br><br>".join(
                f"<b>S{idx}</b><br>{_fmt_alloc_lines(snap)}"
                for idx, snap in enumerate((_st.get("snapshots", []) or []), start=1)
            ) or "없음"
            _fut_tr_rows.append({
                "전략": _name,
                "Canary": "ON" if _st.get("canary_on") else "OFF",
                "다음 트랜치 일정": _next_tranche_refresh_str(
                    _st.get("last_bar_ts", "-"),
                    _meta.get("interval_hours", 1),
                    _st.get("bar_counter", 0),
                    _meta.get("snap_interval_bars", 0),
                    _meta.get("n_snapshots", 0),
                ),
                "현재 합산": _fmt_alloc_lines(_st.get("last_combined", {})),
                "트랜치": _snapshots_text,
            })
        state_sections.append(_strategy_block("📘 선물 실행 상태", _fut_summary, _fut_tr_rows))
    except Exception as _e:
        state_sections.append(f"<h2>📘 선물 실행 상태</h2><p class='error'>상태 조회 실패: {_e}</p>")

    execution_state_html = "".join(state_sections)

    html = f"""
    <!DOCTYPE html>
    <html lang="ko">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Cap Defend {STRATEGY_VERSION} (Personal)</title>
        <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
        <style>
            body {{ font-family: -apple-system, sans-serif; background: #f0f2f5; padding: 10px; color: #333; }}
            .container {{ max-width: 800px; margin: 0 auto; background: #fff; padding: 20px; border-radius: 16px; }}
            table {{ width: 100%; border-collapse: collapse; margin-bottom: 10px; }}
            th, td {{ padding: 10px; border-bottom: 1px solid #f1f3f4; text-align: left; font-size: 0.95em; }}
            th {{ background-color: #fafafa; font-weight: 600; color: #555; }}
            .card {{ background: #fff; padding: 15px; border-radius: 12px; border: 1px solid #e0e0e0; margin-bottom: 10px; }}
            .dataframe {{ width: 100%; border: 1px solid #ddd; border-collapse: collapse; margin: 10px 0; }}
            .dataframe th, .dataframe td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
            .dataframe tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .small-table {{ font-size: 0.9em; }}
            .table-wrap {{ overflow-x: auto; }}
            .summary-list {{ display: grid; gap: 6px; margin-bottom: 12px; }}
            .summary-item {{ padding: 8px 10px; background: #f8f9fa; border-radius: 8px; line-height: 1.45; font-size: 0.95em; }}
            .state-cards {{ display: grid; gap: 10px; margin-bottom: 12px; }}
            .state-card {{ border: 1px solid #e6e8eb; border-radius: 10px; padding: 10px 12px; background: #fff; }}
            .state-row {{ display: grid; grid-template-columns: 110px 1fr; gap: 10px; align-items: start; padding: 6px 0; border-bottom: 1px solid #f1f3f4; }}
            .state-row:last-child {{ border-bottom: none; }}
            .state-label {{ color: #5f6368; font-weight: 600; font-size: 0.9em; }}
            .state-value {{ color: #202124; line-height: 1.45; word-break: keep-all; overflow-wrap: anywhere; }}

            /* Collapsible sections */
            .section-header {{
                display: flex; justify-content: space-between; align-items: center;
                cursor: pointer; padding: 14px 16px; margin: 8px 0 0 0;
                background: #f8f9fa; border-radius: 10px; user-select: none;
            }}
            .section-header:hover {{ background: #eef1f5; }}
            .section-header h2 {{ margin: 0; font-size: 1.1em; }}
            .section-header .badge {{ font-size: 0.8em; color: #666; font-weight: 400; }}
            .section-header .arrow {{ transition: transform 0.2s; font-size: 0.8em; color: #999; }}
            .section-body {{ padding: 0 4px; }}
            .section-body.collapsed {{ display: none; }}

            /* Mobile */
            @media screen and (max-width: 600px) {{
                body {{ padding: 4px; }}
                .container {{ padding: 12px; border-radius: 10px; }}
                .mobile-card-table thead {{ display: none; }}
                .mobile-card-table tr {{ display: block; margin-bottom: 15px; border: 1px solid #ddd; border-radius: 8px; background: #fff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
                .mobile-card-table td {{ display: flex; justify-content: space-between; gap: 10px; padding: 10px 12px; border-bottom: 1px solid #eee; text-align: right; font-size: 0.9em; }}
                .mobile-card-table td:last-child {{ border-bottom: none; }}
                .mobile-card-table td::before {{ content: attr(data-label); font-weight: 600; color: #555; text-align: left; flex: 0 0 38%; }}
                .mobile-card-table td:first-child {{ background: #f8f9fa; font-weight: bold; color: #1a73e8; border-radius: 8px 8px 0 0; }}
                .chart-container {{ height: 180px !important; }}
                /* 2열/3열 그리드 → 1열 */
                div[style*="grid-template-columns:1fr 1fr"] {{ grid-template-columns: 1fr !important; }}
                div[style*="grid-template-columns: 1fr 1fr"] {{ grid-template-columns: 1fr !important; }}
                div[style*="grid-template-columns:1fr 1fr 1fr"] {{ grid-template-columns: 1fr !important; }}
                div[style*="grid-template-columns: 1fr 1fr 1fr"] {{ grid-template-columns: 1fr !important; }}
                .section-header h2 {{ font-size: 1em; }}
                .summary-item {{ font-size: 0.9em; padding: 7px 9px; }}
                .table-wrap {{ overflow-x: visible; }}
                .state-card {{ padding: 8px 10px; }}
                .state-row {{ grid-template-columns: 92px 1fr; gap: 8px; padding: 5px 0; }}
                .state-label, .state-value {{ font-size: 0.88em; }}
                #historyTable th, #historyTable td {{ white-space: nowrap; font-size: 0.82em; padding: 8px 6px; }}
                .weights-table tr {{ display: table-row; margin-bottom: 0; border: none; border-radius: 0; background: transparent; box-shadow: none; }}
                .weights-table td {{ display: table-cell; padding: 10px 6px; border-bottom: 1px solid #eee; text-align: left; font-size: 0.88em; white-space: nowrap; }}
                .weights-table td::before {{ content: none; }}
                .weights-table td:first-child {{ background: transparent; font-weight: 600; color: #202124; border-radius: 0; }}
                .weights-table td:last-child {{ text-align: right; font-variant-numeric: tabular-nums; }}
            }}
            .chart-container {{ height: 250px; position: relative; }}
            .weights-table {{ table-layout: fixed; width: 100%; }}
            .weights-table th:first-child, .weights-table td:first-child {{ width: 62%; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}
            .weights-table th:last-child, .weights-table td:last-child {{ width: 38%; text-align: right; white-space: nowrap; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>\U0001f680 Cap Defend {STRATEGY_VERSION}</h1>
            <p style="color:#666; font-size:0.9em;">{datetime.now().strftime('%Y-%m-%d %H:%M')} | \uc885\uac00 {date_today.strftime('%Y-%m-%d')}</p>

            <!-- ===== 실계좌 현황 (기본 펼침) ===== -->
            <div class="section-header" onclick="toggleSection('secAsset')">
                <h2>\U0001f4bc \uc2e4\uacc4\uc88c \ud604\ud669</h2>
                <span class="badge" id="secAsset_badge"></span>
                <span class="arrow" id="secAsset_arrow">\u25bc</span>
            </div>
            <div class="section-body" id="secAsset">
                <div class="card">
                    <div style="display:flex; gap:10px; align-items:center; flex-wrap:wrap; margin-bottom:12px;">
                        <button onclick="fetchLiveOverview()" style="
                            background:linear-gradient(135deg,#0d904f 0%,#1a73e8 100%);
                            color:white; border:none; padding:12px 24px; border-radius:8px;
                            font-weight:600; font-size:1em; cursor:pointer;">\uc870\ud68c</button>
                        <span id="liveFetchStatus" style="font-size:0.9em; color:#666;"></span>
                    </div>
                    <div id="liveOverviewResult"></div>
                </div>
            </div>

            <!-- ===== 히스토리 (기본 접힘) ===== -->
            <div class="section-header" onclick="toggleSection('secHistory')">
                <h2>\U0001f4c8 \uae30\ub85d / \ucd94\uc774</h2>
                <span class="badge" id="secHistory_badge"></span>
                <span class="arrow" id="secHistory_arrow">\u25b6</span>
            </div>
            <div class="section-body collapsed" id="secHistory">
                <div class="card">
                    <div class="chart-container"><canvas id="chartTotal"></canvas></div>
                </div>
                <div id="historyContainer"></div>
            </div>

<!-- 추천 비중: 매일 확인에 통합됨 -->

            <!-- ===== 상세 로그 (기본 접힘) ===== -->
            <div class="section-header" onclick="toggleSection('secLog')">
                <h2>\U0001f4dc \uc0c1\uc138 \ub85c\uadf8</h2>
                <span class="arrow" id="secLog_arrow">\u25b6</span>
            </div>
            <div class="section-body collapsed" id="secLog">
                {execution_state_html}
            </div>

            {stock_holdings_js}

            <script>
            const API = 'http://' + window.location.hostname + ':5000';
            const fmt = n => (n/1e8).toFixed(2)+'\uc5b5';

            // === Section toggle ===
            function toggleSection(id) {{
                const body = document.getElementById(id);
                const arrow = document.getElementById(id + '_arrow');
                body.classList.toggle('collapsed');
                arrow.textContent = body.classList.contains('collapsed') ? '\u25b6' : '\u25bc';
            }}

            // === Buffer / Force Trade ===
            async function updateBuffer() {{
                const val = document.getElementById('bufferSelect').value;
                const status = document.getElementById('bufferStatus');
                const pwd = prompt('PIN 4\uc790\ub9ac:');
                if (!pwd) return;
                try {{
                    const resp = await fetch(API + '/api/cash_buffer', {{
                        method: 'POST', headers: {{'Content-Type':'application/json'}},
                        body: JSON.stringify({{cash_buffer: parseFloat(val), password: pwd}})
                    }});
                    const d = await resp.json();
                    if (resp.ok) {{ document.getElementById('bufferDisplay').textContent = Math.round((1-parseFloat(val))*100)+'%'; status.innerHTML='\u2705 \ubcc0\uacbd \uc644\ub8cc'; status.style.color='#0d904f'; }}
                    else {{ status.innerHTML='\u26a0\ufe0f '+(d.error||'Error'); status.style.color='#d93025'; }}
                }} catch(e) {{ status.innerHTML='\u274c API \uc2e4\ud328'; status.style.color='#d93025'; }}
                setTimeout(()=>{{status.innerHTML='';}}, 5000);
            }}

            async function forceTrade(exchange) {{
                const btn = document.getElementById('forceTradeUpbitBtn');
                const status = document.getElementById('tradeStatus');
                const pwd = prompt('PIN 4\uc790\ub9ac:');
                if (!pwd) return;
                const amtIn = prompt('\uc6b4\uc6a9 \uae08\uc561 (\uc6d0, 0=\uc804\uccb4):', '0');
                if (amtIn === null) return;
                const amt = parseInt(amtIn.replace(/,/g,'')) || 0;
                const amtText = amt > 0 ? amt.toLocaleString()+'\uc6d0' : '\uc804\uccb4';
                if (!confirm('Force Trade \uc2e4\ud589? ('+amtText+')')) return;
                btn.disabled=true; btn.style.opacity='0.6';
                status.innerHTML='\u23f3 \uc2e4\ud589 \uc911...'; status.style.color='#1967d2';
                try {{
                    const r = await fetch(API+'/api/trade/'+exchange, {{
                        method:'POST', headers:{{'Content-Type':'application/json'}},
                        body: JSON.stringify({{target_amount:amt, password:pwd}})
                    }});
                    const d = await r.json();
                    status.innerHTML = r.ok ? '\u2705 '+d.message : '\u26a0\ufe0f '+(d.error||'Error');
                    status.style.color = r.ok ? '#0d904f' : '#d93025';
                }} catch(e) {{ status.innerHTML='\u274c API \uc2e4\ud328'; status.style.color='#d93025'; }}
                setTimeout(()=>{{btn.disabled=false; btn.style.opacity='1';}}, 5000);
            }}

            function fmtKrwFull(n) {{
                return Math.round(Number(n || 0)).toLocaleString() + '\uc6d0';
            }}

            function fmtUsdFull(n) {{
                return '$' + Number(n || 0).toLocaleString(undefined, {{maximumFractionDigits: 2}});
            }}

            function fmtQty(n) {{
                return Number(n || 0).toLocaleString(undefined, {{maximumFractionDigits: 6}});
            }}

            function renderWeightsTable(title, weights) {{
                const parts = Object.entries(weights || {{}}).map(([ticker, weight]) => ({{
                    ticker,
                    weight: Number(weight || 0) * 100
                }}));
                if (!parts.length) {{
                    return '<div class="card"><h3 style="margin-top:0;">' + title + '</h3><div style="color:#777;">표시할 구성 없음</div></div>';
                }}
                const body = parts
                    .sort((a, b) => {{
                        const aCash = a.ticker === '현금' || a.ticker.toUpperCase() === 'CASH';
                        const bCash = b.ticker === '현금' || b.ticker.toUpperCase() === 'CASH';
                        if (aCash && !bCash) return 1;
                        if (!aCash && bCash) return -1;
                        return b.weight - a.weight;
                    }})
                    .map(row =>
                        '<tr>'
                        + '<td data-label="구성">' + row.ticker + '</td>'
                        + '<td data-label="비중">' + row.weight.toFixed(1) + '%</td>'
                        + '</tr>'
                    )
                    .join('');
                const head = '<tr><th>구성</th><th>계좌 내 비중</th></tr>';
                return '<div class="card"><h3 style="margin-top:0;">' + title + '</h3>'
                    + '<table class="mobile-card-table weights-table"><colgroup><col style="width:62%"><col style="width:38%"></colgroup><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>';
            }}

            function renderHoldingsTable(title, rows) {{
                if (!rows || !rows.length) {{
                    return '<div class="card"><h3 style="margin-top:0;">' + title + '</h3><div style="color:#777;">\ubcf4\uc720 \uc885\ubaa9 \uc5c6\uc74c</div></div>';
                }}
                let body = '';
                rows.forEach(row => {{
                    const price = row.price_krw || row.price || 0;
                    body += '<tr>'
                        + '<td data-label="종목">' + row.ticker + '</td>'
                        + '<td data-label="수량">' + fmtQty(row.qty) + '</td>'
                        + '<td data-label="현재가">' + fmtKrwFull(price) + '</td>'
                        + '</tr>';
                }});
                const head = '<tr><th>종목</th><th>수량</th><th>현재가(원화)</th></tr>';
                return '<div class="card"><h3 style="margin-top:0;">' + title + '</h3>'
                    + '<table class="mobile-card-table"><thead>' + head + '</thead><tbody>' + body + '</tbody></table></div>';
            }}

            function renderLiveOverview(data) {{
                const root = document.getElementById('liveOverviewResult');
                const badge = document.getElementById('secAsset_badge');
                const accounts = (data && data.accounts) ? data.accounts : {{}};
                const stock = accounts.stock_kis || {{}};
                const upbit = accounts.coin_upbit || {{}};
                const binance = accounts.coin_binance || {{}};
                const totalKrw = Number(data.total_krw || 0);
                if (badge) {{
                    badge.textContent = '실시간 총액: ' + fmt(Math.round(totalKrw));
                }}

                const shareText = (amount) => {{
                    if (!totalKrw) return '비중 -';
                    return '비중 ' + (Number(amount || 0) / totalKrw * 100).toFixed(1) + '%';
                }};

                const card = (title, total, cash, subline, error) => {{
                    if (error) {{
                        return '<div class="card"><div style="font-size:0.9em;color:#666;">' + title + '</div>'
                            + '<div style="margin-top:8px;color:#d93025;">조회 실패: ' + error + '</div></div>';
                    }}
                    return '<div class="card" style="margin-bottom:0;">'
                        + '<div style="font-size:0.9em;color:#666;">' + title + '</div>'
                        + '<div style="font-size:1.35em;font-weight:700;margin-top:6px;">' + total + '</div>'
                        + '<div style="margin-top:6px;color:#555;">남은 현금: <b>' + cash + '</b></div>'
                        + '<div style="margin-top:4px;font-size:0.85em;color:#777;">' + subline + '</div>'
                        + '</div>';
                }};

                let html = '<div class="card" style="background:#f8f9fa;">'
                    + '<div style="font-size:0.95em;color:#666;">전체 실시간 자산</div>'
                    + '<div style="font-size:1.55em;font-weight:700;margin-top:6px;">' + fmtKrwFull(data.total_krw || 0) + '</div>'
                    + '<div style="margin-top:4px;font-size:0.85em;color:#777;">업데이트: ' + (data.updated || '-') + '</div>'
                    + '</div>';

                html += '<div style="display:grid; grid-template-columns:1fr 1fr 1fr; gap:12px; margin-top:12px;">';
                html += card('주식 - 한투', fmtKrwFull(stock.total_krw), fmtKrwFull(stock.cash_krw), shareText(stock.total_krw) + ' / 보유 ' + ((stock.holdings || []).length) + '종목', stock.error);
                html += card('현물코인 - 업비트', fmtKrwFull(upbit.total_krw), fmtKrwFull(upbit.krw_balance), shareText(upbit.total_krw) + ' / 보유 ' + ((upbit.holdings || []).length) + '종목', upbit.error);
                html += card('선물 - 바이낸스', fmtKrwFull(binance.total_krw), fmtKrwFull(binance.cash_krw), shareText(binance.total_krw) + ' / 보유 ' + ((binance.holdings || []).length) + '포지션', binance.error);
                html += '</div>';

                // === 3자산 배분 체크 (60/25/15, ±8pp 밴드) ===
                const stockKrw = Number(stock.total_krw || 0);
                const spotKrw = Number(upbit.total_krw || 0);
                const futKrw = Number(binance.total_krw || 0);
                const allocTotal = stockKrw + spotKrw + futKrw;
                if (allocTotal > 0) {{
                    const T_STOCK = 0.60, T_SPOT = 0.25, T_FUT = 0.15, BAND = 0.08;
                    const pStock = stockKrw / allocTotal;
                    const pSpot = spotKrw / allocTotal;
                    const pFut = futKrw / allocTotal;
                    const dStock = Math.abs(pStock - T_STOCK);
                    const dSpot = Math.abs(pSpot - T_SPOT);
                    const dFut = Math.abs(pFut - T_FUT);
                    const maxD = Math.max(dStock, dSpot, dFut);
                    const need = maxD >= BAND;

                    const pctColor = (d) => d >= BAND ? '#d93025' : '#0d904f';
                    const pctBar = (label, cur, target, drift) => {{
                        return '<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f0f0f0;">'
                            + '<span>' + label + '</span>'
                            + '<span style="color:' + pctColor(drift) + ';font-weight:600;">'
                            + (cur * 100).toFixed(1) + '% (목표 ' + (target * 100).toFixed(0) + '%'
                            + ', 편차 ' + (drift * 100).toFixed(1) + '%p)</span></div>';
                    }};

                    html += '<div class="card" style="margin-top:12px;border:2px solid ' + (need ? '#d93025' : '#0d904f') + ';">';
                    html += '<div style="font-weight:700;font-size:1.05em;color:' + (need ? '#d93025' : '#0d904f') + ';margin-bottom:8px;">'
                        + (need ? '⚠️ 리밸런싱 필요' : '✅ 리밸런싱 불필요')
                        + ' (최대편차 ' + (maxD * 100).toFixed(1) + '%p ' + (need ? '≥' : '<') + ' ±8%p)</div>';
                    html += pctBar('주식', pStock, T_STOCK, dStock);
                    html += pctBar('현물코인', pSpot, T_SPOT, dSpot);
                    html += pctBar('선물', pFut, T_FUT, dFut);

                    if (need) {{
                        const tgtStock = allocTotal * T_STOCK;
                        const tgtSpot = allocTotal * T_SPOT;
                        const tgtFut = allocTotal * T_FUT;
                        html += '<div style="margin-top:10px;padding:10px;background:#fce8e6;border-radius:8px;font-size:0.95em;">';
                        const moves = [
                            ['주식', tgtStock - stockKrw],
                            ['현물코인', tgtSpot - spotKrw],
                            ['선물', tgtFut - futKrw],
                        ];
                        moves.forEach(m => {{
                            if (Math.abs(m[1]) > 10000) {{
                                const sign = m[1] > 0 ? '+' : '';
                                html += '<div>' + m[0] + ': <b>' + sign + Math.round(m[1]).toLocaleString() + '원</b></div>';
                            }}
                        }});
                        html += '</div>';
                    }}
                    html += '</div>';
                }}

                html += '<div style="margin-top:16px;">';
                html += renderWeightsTable('한투 구성 비중', stock.weights || {{}});
                html += renderWeightsTable('업비트 구성 비중', upbit.weights || {{}});
                html += renderWeightsTable('바이낸스 구성 비중', binance.weights || {{}});
                html += '</div>';
                root.innerHTML = html;
            }}

            async function fetchLiveOverview() {{
                const statusEl = document.getElementById('liveFetchStatus');
                try {{
                    statusEl.innerHTML = '\u23f3 \uc2e4\uacc4\uc88c \uc870\ud68c \uc911...';
                    statusEl.style.color = '#1967d2';
                    const r = await fetch(API + '/api/assets/live_overview');
                    const data = await r.json();
                    if (!r.ok) throw new Error(data.error || 'API error');
                    renderLiveOverview(data);
                    statusEl.innerHTML = '\u2705 \uc870\ud68c \uc644\ub8cc (' + (data.updated || '-') + ')';
                    statusEl.style.color = '#0d904f';
                }} catch (e) {{
                    statusEl.innerHTML = '\u274c \uc870\ud68c \uc2e4\ud328: ' + e.message;
                    statusEl.style.color = '#d93025';
                }}
            }}

            // === History ===
            let chartTotal = null;
            async function loadHistory() {{
                try {{
                    const r = await fetch(API + '/api/assets/snapshots');
                    const rows = await r.json();
                    if (!rows.length) return;

                    // Badge
                    const badge2 = document.getElementById('secHistory_badge');
                    if (badge2) badge2.textContent = rows.length + '\uac74';

                    // 카드형 히스토리: 월별 접기/펼치기, 가로 넘침 없음
                    const container = document.getElementById('historyContainer');
                    container.innerHTML = '';
                    const sorted = [...rows].reverse();
                    let monthRows = {{}};
                    for (const s of sorted) {{
                        const ym = (s.snapshot_date||s.month).substring(0, 7);
                        if (!monthRows[ym]) monthRows[ym] = [];
                        monthRows[ym].push(s);
                    }}
                    const months = Object.keys(monthRows).sort().reverse();

                    function histItem(s, bold) {{
                        const dt = s.snapshot_date||s.month;
                        const fw = bold ? 'font-weight:700;' : 'color:#555;';
                        return '<div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #f0f0f0;'+fw+'">'
                            +'<span>'+dt.substring(5)+'</span>'
                            +'<span>'+fmt(s.total_krw)+'</span></div>'
                            +'<div style="display:flex;gap:8px;padding:0 0 4px 0;font-size:0.82em;color:#888;">'
                            +'<span>\uc8fc\uc2dd '+fmt(s.stock_krw)+'</span>'
                            +'<span>\ucf54\uc778 '+fmt(s.coin_krw)+'</span>'
                            +'<span>\ud604\uae08 '+fmt(s.cash_krw)+'</span></div>';
                    }}

                    for (const ym of months) {{
                        const items = monthRows[ym];
                        const last = items[0];
                        const monthId = 'hm_'+ym.replace('-','');
                        const hasMore = items.length > 1;
                        const arrow = hasMore ? ' <span style="color:#aaa;font-size:0.8em;">(\u25b6 '+items.length+'\uac74)</span>' : '';

                        let card = '<div class="card" style="margin-bottom:8px;padding:10px 12px;">';
                        card += '<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px;'+(hasMore?'cursor:pointer;':'')+ '"'
                            +(hasMore?' onclick="const el=document.getElementById(\\''+monthId+'\\');el.style.display=el.style.display===\\'none\\'?\\'\\':\\'none\\'"':'')+'>'
                            +'<span style="font-weight:700;font-size:1.05em;">'+ym+arrow+'</span>'
                            +'<span style="font-weight:700;font-size:1.05em;">'+fmt(last.total_krw)+'</span></div>';
                        card += '<div style="display:flex;gap:8px;font-size:0.82em;color:#888;">'
                            +'<span>\uc8fc\uc2dd '+fmt(last.stock_krw)+'</span>'
                            +'<span>\ucf54\uc778 '+fmt(last.coin_krw)+'</span>'
                            +'<span>\ud604\uae08 '+fmt(last.cash_krw)+'</span></div>';

                        if (hasMore) {{
                            card += '<div id="'+monthId+'" style="display:none;margin-top:6px;border-top:1px solid #eee;padding-top:6px;">';
                            for (let j = 1; j < items.length; j++) {{
                                card += histItem(items[j], false);
                            }}
                            card += '</div>';
                        }}
                        card += '</div>';
                        container.innerHTML += card;
                    }}

                    // Chart
                    const labels = rows.map(r => (r.snapshot_date||r.month));
                    const totals = rows.map(r => r.total_krw / 1e8);
                    if (chartTotal) chartTotal.destroy();
                    chartTotal = new Chart(document.getElementById('chartTotal'), {{
                        type: 'line',
                        data: {{ labels, datasets: [{{ label: '\ucd1d\uc790\uc0b0(\uc5b5)', data: totals, borderColor: '#7627bb', borderWidth: 2, fill: true, backgroundColor: 'rgba(118,39,187,0.1)', tension: 0.3 }}] }},
                        options: {{ responsive: true, maintainAspectRatio: false, plugins: {{ legend: {{ position: 'top' }} }} }}
                    }});
                }} catch(e) {{}}
            }}

            // Init
            document.addEventListener('DOMContentLoaded', function() {{
                loadHistory();
            }});
            </script>
        </div>
    </body>
    </html>
    """
    with open(filepath, 'w') as f: f.write(html)
    print(f"Report saved to {filepath}")

# --- 6. Main Execution ---
# --- 6. Main Execution ---
if __name__ == "__main__":
    log = []
    
    # 0. Get My Holdings
    # my_holdings_krw should include KRW cash for correct portfolio calc
    my_holdings_qty, my_holdings_krw, my_cash = get_current_upbit_holdings(log)
    
    if my_holdings_qty or my_cash > 0:
        print(f"✅ Loaded Holdings: {len(my_holdings_qty)} coins + Cash (Total Real-time: {sum(my_holdings_krw.values()) + my_cash:,.0f} KRW)")
    
    c_univ, ids = get_dynamic_coin_universe(log)
    if 'BTC-USD' not in ids: ids['BTC-USD'] = 'bitcoin'
    
    # Also load my holdings' tickers for price data
    all_tickers = set(OFFENSIVE_STOCK_UNIVERSE + DEFENSIVE_STOCK_UNIVERSE + CANARY_ASSETS + [STOCK_CRASH_TICKER] + c_univ + ['BTC-USD'] + list(my_holdings_qty.keys()))
    download_required_data(list(all_tickers), log, ids)
    prices = {t: load_price(t) for t in all_tickers}
    
    if prices['BTC-USD'].empty: sys.exit(1)
    target_date = prices['BTC-USD'].index[-1]
    
    # Get USD-KRW Rate
    rate = 1450.0 
    try:
        usdt_price = pyupbit.get_current_price("KRW-USDT")
        if usdt_price: rate = usdt_price
    except: pass
    
    # Evaluate My Assets using Close Price for TURNOVER CALCULATION
    # [V12.1] 종가 데이터가 없는 경우 업비트 실시간 가격(my_holdings_krw)을 fallback으로 사용
    cur_assets_close = {}  # {ticker-USD: KRW value based on close price}
    for ticker, qty in my_holdings_qty.items():
        if ticker in prices and not prices[ticker].empty:
            close_usd = prices[ticker].iloc[-1]
            val_krw = close_usd * rate * qty
            if val_krw > 1000: 
                cur_assets_close[ticker] = val_krw
        else:
            # Fallback: 종가 데이터 없으면 업비트 실시간 가격 사용
            if ticker in my_holdings_krw and my_holdings_krw[ticker] > 1000:
                cur_assets_close[ticker] = my_holdings_krw[ticker]
                print(f"  ⚠️ {ticker}: 종가 데이터 없음 → 업비트 실시간 가격 사용 ({my_holdings_krw[ticker]:,.0f} KRW)")
    
    # [Fix] Include Cash in Total Value for correct weight calculation
    total_val_close = sum(cur_assets_close.values()) + my_cash
    
    # Current Portfolio Weights (Cash included in denominator)
    cur_coin_port = {k: v/total_val_close for k, v in cur_assets_close.items()} if total_val_close > 0 else {}
    
    if cur_assets_close or my_cash > 0:
        print(f"✅ Turnover Basis (Close Price + Fallback): {len(cur_assets_close)} coins + Cash, Total: {total_val_close:,.0f} KRW")
    
    s_port, s_stat, s_meta = run_stock_strategy_v15(log, prices, target_date)
    c_port, c_stat, c_meta, log, healthy_coins = run_coin_strategy_v15(c_univ, prices, target_date, log)
    
    # Calc Turnover with Threshold Guide (Close Price Based)
    turnover = 0.0
    action_guide = "N/A"
    diff_table_rows = []  # Pre-calculated diff table for HTML
    TURNOVER_THRESHOLD = 0.30  # 30% (Match V12 Backtest)
    
    # [V12 Health Check Enforcement]
    # 보유 중인 코인이 healthy_coins에 없고, target 비중이 0이면 (매도 대상)
    # 턴오버와 상관없이 즉시 리밸런싱해야 함.
    has_bad_coin = False
    bad_coins = []
    healthy_set = set(healthy_coins)
    
    for t in cur_assets_close.keys():
        # cur_assets_close 키: 'TRX-USD', healthy_set 키: 'TRX-USD'
        if t not in c_port and t not in healthy_set:
            has_bad_coin = True
            bad_coins.append(t.replace('-USD', ''))

    # [통합 테이블 데이터 생성]
    # 모든 자산 (내 보유 + 타겟) 합집합
    all_assets = set(cur_assets_close.keys()) | set(c_port.keys())
    
    # Apply Cash Buffer to Target Portfolio (Coin Part)
    # c_port sum is 1.0. We need to scale it down to (1.0 - BUFFER) and assign BUFFER to CASH.
    # But only if it's "Full Invest" mode. If it's Risk-Off (CASH=1.0), BUFFER logic is redundant but safe.
    cash_buf = get_cash_buffer()
    c_port_buffered = {}
    if CASH_ASSET in c_port and c_port[CASH_ASSET] == 1.0:
        c_port_buffered = {CASH_ASSET: 1.0}
    else:
        for t, w in c_port.items():
            c_port_buffered[t] = w * (1.0 - cash_buf)
        c_port_buffered[CASH_ASSET] = c_port_buffered.get(CASH_ASSET, 0.0) + cash_buf

    integrated_rows = []
    
    # 1. 코인 자산
    for k in all_assets:
        if k == CASH_ASSET: continue  # Skip 'Cash' key, handle below
        
        ticker = k.replace('-USD', '')
        val_krw = my_holdings_krw.get(k, 0)
        
        my_w = cur_coin_port.get(k, 0)
        tgt_w = c_port_buffered.get(k, 0)
        diff = tgt_w - my_w
        
        action = "-"
        if diff > 0.005: action = "🔺 BUY"
        elif diff < -0.005: action = "🔻 SELL"
        
        integrated_rows.append({
            'Asset': ticker,
            'Value': val_krw,
            'My': my_w,
            'Target': tgt_w,
            'Diff': diff,
            'Action': action
        })
        
    # 2. 현금 자산 (Cash)
    cash_w = my_cash / total_val_close if total_val_close > 0 else 0
    cash_tgt = c_port_buffered.get(CASH_ASSET, 0.0) 
    
    cash_diff = cash_tgt - cash_w
    cash_action = "-"
    if cash_diff > 0.005: cash_action = "🔻 SELL COINS" # 현금 부족 -> 코인 팔아야 함 (Target > My)
    elif cash_diff < -0.005: cash_action = "🔺 BUY COINS" # 현금 과다 -> 코인 사야 함 (Target < My)
    
    integrated_rows.append({
        'Asset': 'CASH',
        'Value': my_cash,
        'My': cash_w,
        'Target': cash_tgt,
        'Diff': cash_diff,
        'Action': cash_action
    })
    
    # 3. Calc Turnover (Based on diffs)
    turnover = sum(abs(r['Diff']) for r in integrated_rows) / 2
    
    # 4. Action Guide Update
    if has_bad_coin:
        bad_coins_str = ", ".join(bad_coins)
        action_guide = f"REBALANCE (Sick: {bad_coins_str})"
    elif turnover >= TURNOVER_THRESHOLD:
        action_guide = "REBALANCE"
    else:
        action_guide = "HOLD"

    integrated_rows.sort(key=lambda x: x['Target'], reverse=True)
        
    # [Restored] Final Portfolio (Stock + Spot Coin + Futures) for Report Table
    final_port = {CASH_ASSET: 0}
    for t, w in s_port.items():
        key = t if t!=CASH_ASSET else CASH_ASSET
        final_port[key] = final_port.get(key, 0) + w * STOCK_RATIO
    for t, w in c_port_buffered.items():
        key = t if t!=CASH_ASSET else CASH_ASSET
        final_port[key] = final_port.get(key, 0) + w * COIN_RATIO
    # 선물 15%는 바이낸스 자동매매에서 별도 관리 (여기서는 표시만)
    final_port['FUTURES'] = FUTURES_RATIO
    
    # Get KRW Prices (생략 가능, 위에서 integrated_rows에 다 넣음)
    krw_prices = {}
        
    # Pass my_holdings_krw, integrated_rows for unified display
    coin_total_krw = sum(my_holdings_krw.values()) + my_cash
    save_html(log, final_port, s_port, c_port, s_stat, c_stat, turnover, [], [], target_date, krw_prices, s_meta, c_meta, {}, my_holdings_krw, action_guide, integrated_rows, coin_total_krw=coin_total_krw)

    try:
        snap = save_daily_live_snapshot()
        print(
            f"✅ Daily snapshot saved: {snap['snapshot_date']} "
            f"(stock={snap['stock_krw']:,.0f}, coin={snap['coin_krw']:,.0f}, cash={snap['cash_krw']:,.0f}, total={snap['total_krw']:,.0f})"
        )
    except Exception as e:
        print(f"⚠️ daily snapshot 저장 실패: {e}")

    # ─── 새 스키마 signal_state.json 저장 ───
    try:
        # 주식: 공격/방어 picks (s_port는 현재 모드에 따라 하나만 반환됨)
        # 현재가 공격이면 s_port=공격, 방어이면 s_port=방어
        # 항상 둘 다 저장하기 위해 현재 모드 기반으로 분류
        s_picks = sorted([t for t in s_port.keys() if t != 'Cash'])
        s_weights_all = {t: w for t, w in s_port.items()}

        is_stock_risk_on = not s_stat.startswith('수비')
        if is_stock_risk_on:
            offense_picks = s_picks
            offense_weights = s_weights_all
            # 방어 종목은 이전 signal에서 가져오거나 기본값
            try:
                with open(SIGNAL_STATE_FILE, 'r') as _pf:
                    _ps = json.load(_pf)
                defense_picks = _ps.get('stock', {}).get('defense_picks', ['IEF', 'GLD', 'PDBC'])
                defense_weights = _ps.get('stock', {}).get('defense_weights', {})
                # 빈 경우 기본값 생성
                if not defense_weights and defense_picks:
                    dw = (1.0 - 0.02) / len(defense_picks)
                    defense_weights = {t: round(dw, 4) for t in defense_picks}
                    defense_weights['Cash'] = 0.02
            except Exception:
                defense_picks = ['IEF', 'GLD', 'PDBC']
                dw = (1.0 - 0.02) / 3
                defense_weights = {t: round(dw, 4) for t in defense_picks}
                defense_weights['Cash'] = 0.02
        else:
            defense_picks = s_picks
            defense_weights = s_weights_all
            # 공격 종목은 이전 signal에서 가져오거나 기본값
            try:
                with open(SIGNAL_STATE_FILE, 'r') as _pf:
                    _ps = json.load(_pf)
                offense_picks = _ps.get('stock', {}).get('offense_picks', [])
                offense_weights = _ps.get('stock', {}).get('offense_weights', {})
            except Exception:
                offense_picks = []
                offense_weights = {'Cash': 1.0}

        # VT 기준가
        vt_prev = s_meta.get('_vt_prev_close', 0) if isinstance(s_meta, dict) else 0
        vt_sma10_val = s_meta.get('_vt_sma10', 0) if isinstance(s_meta, dict) else 0

        # 코인: guard_refs 계산 (KRW 기준 — executor가 KRW 현재가와 비교)
        # 보유 중인 코인 + 추천 코인 모두 포함 (보유중이지만 추천에서 빠진 종목도 가드 필요)
        coin_guard_refs = {}
        all_coin_tickers = set(t for t in c_port.keys() if t != 'Cash')
        # 기존 트랜치 보유 종목도 포함하면 좋지만, recommend는 trade_state를 안 읽으므로
        # 추천 종목 기반으로만 생성. executor가 잔고 기반으로 추가 체크함.
        for t in sorted(all_coin_tickers):
            p = prices.get(t)  # USD 시계열
            if p is not None and len(p) >= 60:
                # KRW 변환: Upbit 현재가 사용
                try:
                    krw_ticker = f"KRW-{t.replace('-USD', '')}"
                    krw_price = pyupbit.get_current_price(krw_ticker)
                    if krw_price and krw_price > 0:
                        # USD→KRW 비율로 과거 가격도 변환
                        usd_cur = float(p.iloc[-1])
                        ratio = krw_price / usd_cur if usd_cur > 0 else 1
                        coin_guard_refs[t.replace('-USD', '')] = {
                            'prev_close': round(float(p.iloc[-1]) * ratio),
                            'peak_60d': round(float(p.iloc[-60:].max()) * ratio),
                        }
                except Exception:
                    pass

        new_signal = {
            'stock': {
                'offense_picks': offense_picks,
                'offense_weights': offense_weights,
                'defense_picks': defense_picks,
                'defense_weights': defense_weights,
                'risk_on': is_stock_risk_on,
                'vt_prev_close': vt_prev,
                'vt_sma10': vt_sma10_val,
            },
            'coin': {
                'picks': sorted([t.replace('-USD', '') for t in c_port.keys() if t != 'Cash']),
                'weights': {t.replace('-USD', ''): w for t, w in c_port.items()},
                'risk_on': bool(not c_stat.startswith('Risk-Off')),
                'guard_refs': coin_guard_refs,
            },
            'meta': {
                'signal_date': str(target_date.date()) if hasattr(target_date, 'date') else str(target_date)[:10],
                'updated_at': datetime.now().strftime('%Y-%m-%d %H:%M'),
            },
        }
        _save_signal_state(new_signal)
    except Exception as e:
        print(f"⚠️ signal_state 저장 실패: {e}")

    # ─── 텔레그램 일간 리포트 ───
    try:
        # 요약 텍스트
        btc_data = prices.get('BTC-USD')
        btc_dist_pct = ""
        if btc_data is not None and len(btc_data) >= COIN_CANARY_MA_PERIOD:
            btc_cur = float(btc_data.iloc[-1])
            btc_sma = float(btc_data.rolling(COIN_CANARY_MA_PERIOD).mean().iloc[-1])
            btc_dist_pct = f"{(btc_cur/btc_sma-1)*100:+.1f}%"

        coin_picks_str = ', '.join(t.replace('-USD','') for t in c_port.keys() if t != 'Cash')
        coin_weights_str = ' / '.join(f"{t.replace('-USD','')} {w:.0%}" for t, w in c_port.items())

        summary = (
            f"📊 Daily Report ({target_date.strftime('%m/%d') if hasattr(target_date, 'strftime') else str(target_date)[:10]})\n"
            f"\n🪙 코인: {c_stat}\n"
            f"  BTC vs SMA50: {btc_dist_pct}\n"
            f"  선정: {coin_picks_str or '없음'}\n"
            f"  비중: {coin_weights_str}\n"
            f"\n📈 주식: {s_stat}\n"
            f"  비중: {', '.join(f'{t} {w:.0%}' for t,w in s_port.items())}"
        )
        if PORTFOLIO_PUBLIC_URL:
            send_telegram(summary, button_text="대시보드 열기", button_url=PORTFOLIO_PUBLIC_URL)
        else:
            send_telegram(summary)
        print("✅ 텔레그램 일간 리포트 전송 완료")
    except Exception as e:
        print(f"⚠️ 텔레그램 리포트 전송 실패: {e}")

    # ─── 3자산 배분 체크 (60/25/15, ±8pp 밴드) ───
    try:
        api = os.environ.get("TRADE_API_BASE", "http://127.0.0.1:5000")
        ov = requests.get(f"{api}/api/assets/live_overview", timeout=60).json()
        accts = ov.get("accounts", {})
        stock_krw = float((accts.get("stock_kis") or {}).get("total_krw", 0))
        spot_krw = float((accts.get("coin_upbit") or {}).get("total_krw", 0))
        fut_krw = float((accts.get("coin_binance") or {}).get("total_krw", 0))
        alloc_total = stock_krw + spot_krw + fut_krw

        if alloc_total > 0:
            p_stock = stock_krw / alloc_total
            p_spot = spot_krw / alloc_total
            p_fut = fut_krw / alloc_total
            d_stock = abs(p_stock - STOCK_RATIO)
            d_spot = abs(p_spot - COIN_RATIO)
            d_fut = abs(p_fut - FUTURES_RATIO)
            max_drift = max(d_stock, d_spot, d_fut)

            alloc_line = (
                f"⚖️ 자산배분: 주식 {p_stock:.1%} / 현물 {p_spot:.1%} / 선물 {p_fut:.1%}"
                f" (목표 60/25/15, 최대편차 {max_drift:.1%})"
            )
            print(alloc_line)

            if max_drift >= REBAL_BAND_PP:
                moves = []
                for name, cur, tgt in [("주식", stock_krw, alloc_total * STOCK_RATIO),
                                        ("현물", spot_krw, alloc_total * COIN_RATIO),
                                        ("선물", fut_krw, alloc_total * FUTURES_RATIO)]:
                    delta = tgt - cur
                    if abs(delta) > 10000:
                        moves.append(f"  {name}: {delta:+,.0f}원")
                alert_msg = (
                    f"⚠️ 자산배분 리밸런싱 필요!\n"
                    f"최대편차 {max_drift:.1%} ≥ ±8%p\n\n"
                    f"주식 {p_stock:.1%} (목표 60%)\n"
                    f"현물코인 {p_spot:.1%} (목표 25%)\n"
                    f"선물 {p_fut:.1%} (목표 15%)\n\n"
                    f"조정 필요:\n" + "\n".join(moves)
                )
                send_telegram(alert_msg)
                print("🚨 자산배분 리밸런싱 알림 전송")
            else:
                print("✅ 자산배분 밴드 내 (리밸런싱 불필요)")
    except Exception as e:
        print(f"⚠️ 자산배분 체크 실패: {e}")
