"""Cap Defend V17 — 새 아키텍처 스키마 정의.

signal_state.json: recommend만 씀, executor는 읽기만
coin_trade_state.json: 코인 executor만 씀
kis_trade_state.json: 주식 executor만 씀
"""

# ═══════════════════════════════════════════════════════════════════
# signal_state.json — recommend가 생성하는 순수 신호
# ═══════════════════════════════════════════════════════════════════

SIGNAL_STATE_EXAMPLE = {
    "stock": {
        "offense_picks": ["GLD", "EEM", "PDBC"],
        "offense_weights": {"GLD": 0.327, "EEM": 0.327, "PDBC": 0.327, "Cash": 0.02},
        "defense_picks": ["IEF", "GLD", "PDBC"],
        "defense_weights": {"IEF": 0.327, "GLD": 0.327, "PDBC": 0.327, "Cash": 0.02},
        "risk_on": True,
        "vt_prev_close": 137.9,     # Crash 발동 기준 (VT 전일 종가)
        "vt_sma10": 136.5,           # Crash 복귀 기준 (VT SMA10)
    },
    "coin": {
        "picks": ["BTC", "ETH", "XRP", "SOL", "ADA"],
        "weights": {"BTC": 0.196, "ETH": 0.196, "XRP": 0.196, "SOL": 0.196, "ADA": 0.196, "Cash": 0.02},
        "risk_on": True,
        "guard_refs": {
            # 각 코인별 DD Exit / Blacklist 기준값
            "BTC": {"prev_close": 87500, "peak_60d": 95000},
            "ETH": {"prev_close": 3200, "peak_60d": 3800},
            "XRP": {"prev_close": 2.1, "peak_60d": 2.5},
            "SOL": {"prev_close": 180, "peak_60d": 210},
            "ADA": {"prev_close": 0.85, "peak_60d": 1.05},
        },
    },
    "meta": {
        "signal_date": "2026-03-24",   # 종가 기준일
        "updated_at": "2026-03-25 09:15",  # 생성 시각
    },
}

# signal_state 필수 키
SIGNAL_REQUIRED_KEYS = {
    "stock": {"offense_picks", "offense_weights", "defense_picks", "defense_weights", "risk_on", "vt_prev_close", "vt_sma10"},
    "coin": {"picks", "weights", "risk_on", "guard_refs"},
    "meta": {"signal_date", "updated_at"},
}


# ═══════════════════════════════════════════════════════════════════
# coin_trade_state.json — 코인 executor가 관리
# ═══════════════════════════════════════════════════════════════════

COIN_TRADE_STATE_EXAMPLE = {
    "tranches": {
        "1": {
            "picks": ["BTC", "ETH", "XRP", "SOL", "ADA"],
            "weights": {"BTC": 0.196, "ETH": 0.196, "XRP": 0.196, "SOL": 0.196, "ADA": 0.196, "Cash": 0.02},
            "anchor_month": "2026-03",
        },
        "11": {
            "picks": ["BTC", "ETH", "SOL", "ADA", "DOGE"],
            "weights": {"BTC": 0.196, "ETH": 0.196, "SOL": 0.196, "ADA": 0.196, "DOGE": 0.196, "Cash": 0.02},
            "anchor_month": "2026-03",
        },
        "21": {
            "picks": ["BTC", "ETH", "XRP", "SOL", "LINK"],
            "weights": {"BTC": 0.196, "ETH": 0.196, "XRP": 0.196, "SOL": 0.196, "LINK": 0.196, "Cash": 0.02},
            "anchor_month": "2026-02",
        },
    },
    "rebalancing_needed": False,
    "guard_state": {
        "crash_active": False,
        "crash_date": None,
        "crash_cooldown_until": None,
        "exclusions": {},
        # exclusions 예시: {"LUNA": {"reason": "dd", "until_date": null},
        #                   "DOGE": {"reason": "bl", "until_date": "2026-03-26"}}
    },
    "prev_risk_on": True,
    "flip_date": None,
    "pfd_done": True,
    "last_action": "trade",
    "last_trade_date": "2026-03-25",
}


# ═══════════════════════════════════════════════════════════════════
# kis_trade_state.json — 주식 executor가 관리
# ═══════════════════════════════════════════════════════════════════

KIS_TRADE_STATE_EXAMPLE = {
    "tranches": {
        "1": {
            "picks": ["GLD", "EEM", "PDBC"],
            "weights": {"GLD": 0.327, "EEM": 0.327, "PDBC": 0.327, "Cash": 0.02},
            "anchor_month": "2026-03",
        },
        "8": {
            "picks": ["GLD", "EEM", "PDBC"],
            "weights": {"GLD": 0.327, "EEM": 0.327, "PDBC": 0.327, "Cash": 0.02},
            "anchor_month": "2026-03",
        },
        "15": {
            "picks": ["GLD", "EEM", "PDBC"],
            "weights": {"GLD": 0.327, "EEM": 0.327, "PDBC": 0.327, "Cash": 0.02},
            "anchor_month": "2026-03",
        },
        "22": {
            "picks": ["GLD", "EEM", "PDBC"],
            "weights": {"GLD": 0.327, "EEM": 0.327, "PDBC": 0.327, "Cash": 0.02},
            "anchor_month": "2026-02",
        },
    },
    "rebalancing_needed": False,
    "guard_state": {
        "crash_active": False,
        "crash_date": None,
        "crash_cooldown_until": None,
        # 주식은 exclusions 없음 (DD/BL은 코인만)
    },
    "prev_risk_on": True,
    "last_action": "trade",
    "last_trade_date": "2026-03-25",
}
