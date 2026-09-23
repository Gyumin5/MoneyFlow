"""재조정 직후에도 드리프트가 현금버퍼만큼 남나 — 바닥 실측 (2026-09-23).

bt_fut_drift_parity.py 가 "좌표계를 라이브와 맞춰도 발화 수가 거의 안 변한다" 를 보였다.
그러면 "백테스트에는 버퍼 크기의 구조적 바닥이 있다" 는 지적이 왜 안 나타나는지가 남는다.
이 스크립트는 드리프트 값 자체의 분포를 두 좌표계에서 직접 잰다. 특히 재조정한 바로 다음 날의
값이 0 근처인지 0.01 근처인지가 답이다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 -u bt_fut_drift_floor.py
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')

START, END = '2020-10-01', '2026-09-18'
TX_FUT = 0.0006
FUT_BASE = dict(sma_days=42, mom_short_days=18, mom_long_days=127,
                vol_threshold=0.05, n_snapshots=5, snap_interval_bars=95,
                canary_hyst=0.015, universe_size=3, cap=1 / 3)

_bars = None
_funding = None
_k2 = None


def run_one(parity: bool, thr: float = 0.03):
    os.environ['FUT_DRIFT_PARITY'] = 'on' if parity else 'off'
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    import importlib
    import backtest_futures_v25 as F
    importlib.reload(F)
    trace: list = []
    kw = dict(FUT_BASE)
    kw['drift_threshold'] = thr
    F.run(_bars, _funding, interval='D', leverage=_k2,
          tx_cost=TX_FUT, maint_rate=0.004,
          vol_days=90, vol_mode='daily', post_flip_delay=5,
          phase_offset_bars=0, health_mode='mom2vol',
          start_date=START, end_date=END, _trace=trace, **kw)
    return list(F._HT_LOG), trace


def pct(vals, q):
    if not vals:
        return float('nan')
    s = sorted(vals)
    i = max(0, min(len(s) - 1, int(round(q * (len(s) - 1)))))
    return s[i]


def summarize(name: str, ht_log, trace) -> None:
    rebal_days = {row['date'] for row in trace if row['rebal']}
    dates = [d for d, _ in ht_log]
    after = []          # 직전 봉이 재조정이었던 날의 드리프트
    idx = {d: i for i, d in enumerate([row['date'] for row in trace])}
    order = [row['date'] for row in trace]
    for d, ht in ht_log:
        i = idx.get(d)
        if i is not None and i > 0 and order[i - 1] in rebal_days:
            after.append(ht)
    allv = [ht for _, ht in ht_log]
    print(f"\n  [{name}]  드리프트를 잰 날 {len(allv)}일, 그중 직전날 재조정 {len(after)}일")
    print(f"    전체   최소 {min(allv):.4f} · 10% {pct(allv,0.10):.4f} · 중앙 {pct(allv,0.5):.4f} "
          f"· 90% {pct(allv,0.90):.4f} · 최대 {max(allv):.4f}")
    if after:
        print(f"    직후   최소 {min(after):.4f} · 10% {pct(after,0.10):.4f} · 중앙 {pct(after,0.5):.4f} "
              f"· 90% {pct(after,0.90):.4f} · 최대 {max(after):.4f}")
        for edge in (0.001, 0.005, 0.0075, 0.01, 0.02):
            n = sum(1 for v in after if v < edge)
            print(f"    직후 {edge:.4f} 미만 {n:4d}일 ({n/len(after)*100:5.1f}%)")
    del dates


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
    print(f'  심볼 {len(_bars)}종 ({time.time()-t0:.0f}초)')

    print('\n드리프트 값 분포 — 앵커 위상 0, 문턱 0.03(현행), 기본 비용')
    for parity, name in ((False, '현행 좌표계(목표 합 1)'), (True, '라이브 좌표계(목표에 현금 1%)')):
        ht_log, trace = run_one(parity)
        summarize(name, ht_log, trace)

    print(f'\n총 소요: {time.time()-t0:.0f}초')
    os.environ['FUT_DRIFT_PARITY'] = 'off'


if __name__ == '__main__':
    main()
