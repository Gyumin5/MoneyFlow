#!/usr/bin/env python3
"""V23 실거래 선물 전략 설정 (1D 단일 멤버, 2026-04-30 확정).

V23 변경 (vs V22):
- 1D + 4h 2멤버 → 1D 단일 멤버 (4h 제거)
- snap_interval_bars 90 → 57 (3*19, stagger 19 prime)
- drift_threshold 0.0 → 0.05 (sleeve-level half-turnover 트리거)
- ENSEMBLE 단일 (weight = 1.0)
"""

START = '2020-10-01'
END = '2026-04-27'

CURRENT_STRATEGIES = {
    "D_SMA42": dict(
        interval="D",
        sma_bars=42,
        mom_short_bars=18,
        mom_long_bars=127,
        canary_hyst=0.015,
        drift_threshold=0.05,        # V23: 0.0 → 0.05
        dd_threshold=0,
        dd_lookback=0,
        bl_drop=0,
        bl_days=0,
        health_mode="mom2vol",
        vol_mode="daily",
        vol_threshold=0.05,
        n_snapshots=3,
        snap_interval_bars=57,       # V23: 90 → 57 (3*19)
    ),
}

CURRENT_LIVE_COMBO = {k: 1.0 for k in CURRENT_STRATEGIES}
