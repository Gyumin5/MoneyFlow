#!/usr/bin/env python3
"""V25 실거래 선물 전략 설정 (1D 단일 멤버 + 동적 L per-coin + CROSS margin, 2026-05-28).

V25 도입 (2026-05-28):
- 마진모드: ISOLATED → CROSS (wallet 전체 cushion)
- 레버리지: 고정 L3 → 동적 per-coin L (Lmin=2, Lmid=3, Lmax=4)
  · BTC cap: BTC/SMA42 > 1.05 → Lmax(4), > 1.015 → Lmid(3), else Lmin(2)
  · per-coin K2 (SMA-based winner): close/SMA7 > 1.075 → Lmax(4), > 1.025 → Lmid(3), else Lmin(2)
  · final L per coin = min(BTC cap, per-coin K2)
- shift(1) lag (전일 신호 → 당일 적용)
- BT: 5.6yr Cal 8.12 / CAGR 312% / MDD -38.3% / Sharpe 1.90 (단독 sleeve)
- alloc 60/25/15 + T1=20pp/T3U=25pp: Cal 5.72 / MDD -18.6% / CAGR 106%
- 채택 근거: window rank-sum 1/25 plateau (K2 SMA=7 h=2.5%), J(mom-based) 7.45 보다 Cal +0.67 + MDD 7pp 개선

V24 (2026-04-30~2026-05-28, deprecated):
- 1D 단일 + drift trigger + 고정 L3 ISO
- BT: Cal 4.05 / MDD -63.1% / CAGR 256%

V22 (2026-04-27~2026-04-30, deprecated):
- 1D + 4h 2멤버 EW
"""

START = '2020-10-01'
END = '2026-04-27'

# Sleeve 시그널 (V24 와 동일) — universe 선정 + drift trigger 용
CURRENT_STRATEGIES = {
    "D_SMA42": dict(
        interval="D",
        sma_bars=42,
        mom_short_bars=18,
        mom_long_bars=127,
        canary_hyst=0.015,
        drift_threshold=0.03,
        dd_threshold=0,
        dd_lookback=0,
        bl_drop=0,
        bl_days=0,
        health_mode="mom2vol",
        vol_mode="daily",
        vol_threshold=0.05,
        n_snapshots=5,
        snap_interval_bars=95,
    ),
}

CURRENT_LIVE_COMBO = {k: 1.0 for k in CURRENT_STRATEGIES}


# === V25 동적 레버리지 spec ===
# 각 코인 L 결정: min(BTC_cap, per_coin_K2)
LEVERAGE_MIN = 2       # Lmin
LEVERAGE_MID = 3       # Lmid (V24 와 동일)
LEVERAGE_MAX = 4       # Lmax (V24 L3 → V25 L4 확대)

# BTC cap (BTC SMA42 ratio 기반)
BTC_CAP_SMA_PERIOD = 42
BTC_CAP_THR_MID = 1.015   # > 1.015 → Lmid 이상
BTC_CAP_THR_MAX = 1.05    # > 1.05 → Lmax

# per-coin K2 (SMA-based winner)
K2_SMA_PERIOD = 7         # 짧은 SMA — 빠른 trend 반영
K2_HYST = 0.025           # h=2.5%, thr_mid=1.025, thr_max=1.075

# 마진모드 (V24 ISOLATED → V25 CROSS)
MARGIN_TYPE = 'CROSSED'   # Binance API 표기

# 디버그 로그 레벨 (V25 도입 — 동적 L 검증)
DEBUG_LEVERAGE = True     # 매 cron L 결정 detail 로그
DEBUG_MARGIN = True       # set_margin_type / set_leverage 호출 결과
