"""현물 슬리브가 하루에 몇 종목을 들고 있었나 (2026-09-25).

쓰임 하나다. 변동성 가중 실험에서 자산곡선이 갈라진 날이 2136일 중 837일(39%)이었다.
매일 작동하는 규칙인데 절반이 안 된다 — 이게 사건 의존이라서인지, 아니면 나눌 종목이
2종 미만인 날에는 구조적으로 갈릴 수가 없어서인지를 가른다. 후자면 "2종 이상 보유한 날"
수가 837일 근처여야 한다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_spot_hold_count.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from bt_fut_spike_hold import START, END, SPOT_BASE, TX_SPOT, PH_SPOT  # noqa: E402


def main() -> None:
    from unified_backtest import load_data, run as bt_run
    bars, funding = load_data("D")
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    os.environ["CD_WEIGHT_MODE_SPOT"] = "ew"
    trace: list = []
    bt_run(bars, funding, interval="D", asset_type="spot", leverage=1.0,
           tx_cost=TX_SPOT, start_date=START, end_date=END,
           vol_mode="daily", phase_offset_bars=PH_SPOT[0],
           post_flip_delay=5, selection="greedy",
           stop_kind="none", stop_pct=0.0,
           dd_lookback=60, dd_threshold=-99.0,
           bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
           health_mode="mom2vol", _trace=trace, **SPOT_BASE)

    cnt = Counter()
    for row in trace:
        tgt = row.get("target") or {}
        n = sum(1 for k, v in tgt.items()
                if k not in ("CASH", "Cash") and v and v > 1e-9)
        cnt[n] += 1
    tot = sum(cnt.values())
    ge2 = sum(v for k, v in cnt.items() if k >= 2)
    print(f"추적한 날 {tot}일")
    for k in sorted(cnt):
        print(f"  보유 {k}종: {cnt[k]}일 ({cnt[k]/tot:.0%})")
    print(f"2종 이상 보유한 날 {ge2}일 ({ge2/tot:.0%}) — 변동성 가중이 갈릴 수 있는 최대치")


if __name__ == '__main__':
    main()
