"""변동성 가중이 투자비율을 얼마나 깎나 (2026-09-25).

ai-debate run-20260924T154821Z 가 짚은 구멍을 잰다. 종목별 상한이 1/3 이고 3종을 고르면
등분은 정확히 100% 가 투자된다. 변동성 역수로 나누면 변동성이 낮은 종목의 몫이 1/3 을
넘어 상한에 잘리고, 잘린 몫은 현금으로 남는다. 즉 배분만 바뀌는 게 아니라 노출이 준다.
그러면 관측된 낙폭 개선이 배분 효과인지 노출 축소 효과인지 갈리지 않는다.

여기서는 날짜별 투자비율 자체만 센다. 대조군 성과 비교는 bt_coin_invvol_control.py 다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_spot_exposure.py
"""
from __future__ import annotations

import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from bt_fut_spike_hold import START, END, SPOT_BASE, TX_SPOT, PH_SPOT  # noqa: E402

WINDOWS = [10, 20, 30, 120]


def _run(mode: str, days: int, bars, funding) -> list:
    from unified_backtest import run as bt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    os.environ["CD_WEIGHT_MODE_SPOT"] = mode
    os.environ["CD_WEIGHT_VOL_DAYS"] = str(days)
    trace: list = []
    try:
        bt_run(bars, funding, interval="D", asset_type="spot", leverage=1.0,
               tx_cost=TX_SPOT, start_date=START, end_date=END,
               vol_mode="daily", phase_offset_bars=PH_SPOT[0],
               post_flip_delay=5, selection="greedy",
               stop_kind="none", stop_pct=0.0,
               dd_lookback=60, dd_threshold=-99.0,
               bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
               health_mode="mom2vol", _trace=trace, **SPOT_BASE)
    finally:
        os.environ["CD_WEIGHT_MODE_SPOT"] = 'ew'
    return trace


def _invested(trace: list) -> list:
    out = []
    for row in trace:
        tgt = row.get("target") or {}
        s = sum(v for k, v in tgt.items()
                if k not in ("CASH", "Cash") and v and v > 1e-9)
        n = sum(1 for k, v in tgt.items()
                if k not in ("CASH", "Cash") and v and v > 1e-9)
        out.append((n, s))
    return out


def main() -> None:
    from unified_backtest import load_data
    bars, funding = load_data("D")

    base = _invested(_run('ew', 90, bars, funding))
    print(f"추적한 날 {len(base)}일")
    print("\n[등분 = 현행]")
    _report(base)
    for d in WINDOWS:
        print(f"\n[변동성 역수 · 창 {d}일]")
        _report(_invested(_run('invvol', d, bars, funding)), base)


def _report(rows: list, base: list | None = None) -> None:
    held = [(n, s) for n, s in rows if n >= 2]
    allv = np.array([s for _n, s in rows], dtype=float)
    hv = np.array([s for _n, s in held], dtype=float)
    print(f"  2종 이상 보유한 날 {len(held)}일")
    if len(hv):
        print(f"  그 날의 투자비율 — 평균 {hv.mean():.3f} 중앙 {np.median(hv):.3f} "
              f"최소 {hv.min():.3f} 최대 {hv.max():.3f}")
        print(f"  투자비율 100% 미만인 날 {int((hv < 0.999).sum())}일 "
              f"({(hv < 0.999).mean():.0%} of 2종 이상)")
    print(f"  전체 날 평균 투자비율 {allv.mean():.3f}")
    if base is not None:
        bv = np.array([s for _n, s in base], dtype=float)
        n = min(len(allv), len(bv))
        print(f"  현행 대비 평균 노출차 {allv[:n].mean() - bv[:n].mean():+.4f}")


if __name__ == '__main__':
    main()
