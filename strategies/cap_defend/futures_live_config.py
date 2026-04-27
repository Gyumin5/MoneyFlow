#!/usr/bin/env python3
"""V22 실거래 선물 전략 설정 (1D+4h 2멤버, 2026-04-27 확정)."""

START = '2020-10-01'
END = '2026-03-28'

CURRENT_STRATEGIES = {
    "D_SMA42": dict(
        interval="D",
        sma_bars=42,
        mom_short_bars=18,
        mom_long_bars=127,
        canary_hyst=0.015,
        drift_threshold=0.0,
        dd_threshold=0,
        dd_lookback=0,
        bl_drop=0,
        bl_days=0,
        health_mode="mom2vol",
        vol_mode="daily",
        vol_threshold=0.05,
        n_snapshots=3,
        snap_interval_bars=90,
    ),
    "4h_SMA240": dict(
        interval="4h",
        sma_bars=240,
        mom_short_bars=12,
        mom_long_bars=180,
        canary_hyst=0.015,
        drift_threshold=0.0,
        dd_threshold=0,
        dd_lookback=0,
        bl_drop=0,
        bl_days=0,
        health_mode="mom2vol",
        vol_mode="daily",
        vol_threshold=0.05,
        n_snapshots=3,
        snap_interval_bars=540,    # 90일 동기 (4h × 6 × 90)
    ),
}

CURRENT_LIVE_COMBO = {k: 1 / 2 for k in CURRENT_STRATEGIES}
