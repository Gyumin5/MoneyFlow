"""선물 종목 교체가 실제로 며칠 일어나나 — 문턱별로 센다 (2026-09-23).

앞선 측정(bt_fut_refill_vs_anchor.py)에서 선물은 현물과 달리 드리프트 발화일 교체가
실제로 일어난다는 것까지 봤다(refill 과 앵커 전용의 자산곡선이 갈린다). 남은 물음은
"문턱을 조이면 그 교체가 몇 번에서 몇 번으로 느나" 다.

세는 법. 엔진이 원래 받아 주는 _trace(매 봉의 목표 구성과 발화 여부)를 쓴다. 엔진은 한 줄도
안 고친다. 교체일 = 발화한 날에 목표 종목 집합에 전날 없던 코인이 들어온 날. 이 셈에는
앵커(95봉) 교체도 같이 들어가므로, 같은 문턱에서 교체를 끈 쪽(DRIFT_HEALTH_MODE=off)을
같이 세서 뺀다. 앵커 교체는 양쪽에 공통이므로 차이가 발화일 교체 쪽이다.

한계를 먼저 적는다. 교체가 한 번 일어나면 그 뒤 보유가 갈려서 두 모드의 경로가 달라진다.
그래서 이 차이는 정확한 건수가 아니라 규모다. 정확한 건수는 엔진 안에 계수기를 넣어야 한다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_fut_refill_count.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')

START, END = '2020-10-01', '2026-09-18'
TX_FUT = 0.0006
THRESHOLDS = [0.005, 0.01, 0.02, 0.03]
FUT_BASE = dict(sma_days=42, mom_short_days=18, mom_long_days=127,
                vol_threshold=0.05, n_snapshots=5, snap_interval_bars=95,
                canary_hyst=0.015, universe_size=3, cap=1 / 3)

_bars = None
_funding = None
_k2 = None


def run_traced(mode: str, thr: float, phase: int = 0):
    from backtest_futures_v25 import run as fbt_run
    os.environ['DRIFT_HEALTH_MODE'] = mode
    os.environ['DAILY_LEV'] = 'off'
    trace: list = []
    kw = dict(FUT_BASE)
    kw['drift_threshold'] = thr
    m = fbt_run(_bars, _funding, interval='D', leverage=_k2,
                tx_cost=TX_FUT, maint_rate=0.004,
                vol_days=90, vol_mode='daily', post_flip_delay=5,
                phase_offset_bars=phase, health_mode='mom2vol',
                start_date=START, end_date=END, _trace=trace, **kw)
    return m, trace


def count_swaps(trace: list) -> tuple[int, int]:
    """(발화일 수, 새 코인이 들어온 발화일 수)."""
    fires = swaps = 0
    prev: set = set()
    for row in trace:
        cur = {c for c in row['target'] if c != 'CASH'}
        if row['rebal']:
            fires += 1
            if cur - prev:
                swaps += 1
        prev = cur
    return fires, swaps


def main() -> None:
    global _bars, _funding, _k2
    t0 = time.time()
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print('코인 일봉 로드 중...')
    _bars, _funding = load_data('D')
    _k2 = build_K2_signal(_bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                          btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                          l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f'  심볼 {len(_bars)}종 ({time.time()-t0:.0f}초)\n')

    print('선물 종목 교체일 수 — 앵커 위상 0, 기본 비용, 기간 5.97년')
    print(f"  {'문턱':>6} {'모드':<7} {'발화일':>7} {'교체일':>7} {'교체÷발화':>10}")
    res = {}
    for thr in THRESHOLDS:
        for mode in ('off', 'refill'):
            _m, tr = run_traced(mode, thr)
            f, s = count_swaps(tr)
            res[(thr, mode)] = (f, s)
            ratio = s / f if f else float('nan')
            print(f"  {thr:>6.4f} {mode:<7} {f:>7d} {s:>7d} {ratio:>10.3f}")
    print('\n  교체 차이 — 발화일 교체가 만든 몫(앵커 교체는 양쪽 공통이라 상쇄)')
    print(f"  {'문턱':>6} {'교체(refill)':>13} {'교체(off)':>10} {'차이':>7}")
    for thr in THRESHOLDS:
        sr = res[(thr, 'refill')][1]
        so = res[(thr, 'off')][1]
        print(f"  {thr:>6.4f} {sr:>13d} {so:>10d} {sr-so:>+7d}")
    print(f'\n총 소요: {time.time()-t0:.0f}초')


if __name__ == '__main__':
    main()
