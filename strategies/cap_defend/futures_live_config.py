#!/usr/bin/env python3
"""V23 실거래 선물 전략 설정 (1D 단일 멤버, 2026-05-04 갱신).

V23 갱신 (2026-05-04, n=5 sn=95 d=0.03):
- n_snapshots 3 → 5 (universe 분산 ↑)
- snap_interval_bars 57 → 95 (5*19, stagger 19 유지)
- drift_threshold 0.05 → 0.03 (반응성 ↑)
- BT: sleeve Cal 10.45→13.41 (+28%), MDD -41.9%→-37.7%, ymin -7.5%→-3.9%

V23 (2026-04-30):
- 1D + 4h 2멤버 → 1D 단일 멤버 (4h 제거)
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
        drift_threshold=0.03,        # V23 갱신 (05-04): 0.05 → 0.03
        dd_threshold=0,
        dd_lookback=0,
        bl_drop=0,
        bl_days=0,
        health_mode="mom2vol",
        vol_mode="daily",
        vol_threshold=0.05,
        n_snapshots=5,               # V23 갱신 (05-04): 3 → 5
        snap_interval_bars=95,       # V23 갱신 (05-04): 57 → 95 (5*19)
    ),
}

CURRENT_LIVE_COMBO = {k: 1.0 for k in CURRENT_STRATEGIES}
