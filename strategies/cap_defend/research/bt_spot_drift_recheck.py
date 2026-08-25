"""현물 드리프트 문턱 0.10 재확인 — 체결 밴드를 라이브(0)로 맞춘 뒤 (2026-08-25).

계기: ai-debate run-20260825T071922Z 가 B(백테스트를 라이브에 맞춤)를 결론내면서 조건을 달았다 —
"밴드를 0 으로 바꾸면 5% 밴드에서 산출했던 과거 채택값이 그대로인지 확인하라. 가장 직접 얽힌
것은 드리프트 문턱 0.10 이다." 드리프트가 리밸런싱을 켤지를 정하고 밴드가 그 안에서 주문을
낼지를 정하므로 두 값은 순서로 결합돼 있다.

축: 드리프트 문턱 하나. 후보군은 원래 채택 때 쓴 것 그대로(0.02~0.12), 위상 5종도 그대로.
나머지는 채택 V24 현물 설정 고정. 밴드는 이제 엔진 기본값 0(현물)이라 따로 지정하지 않는다.

판정 (결과 보기 전 고정):
  0.10 이 윈도우 rank-sum 최상위이거나 최상위와 평균순위 차 0.2 이내면 → 채택값 유지.
  다른 값이 뚜렷하게 앞서면 → 그 사실만 보고하고 값은 자동으로 바꾸지 않는다(별도 결정).
  거래비용 5배에서 결론이 뒤집히면 유지 쪽으로 판정한다(비용 민감 개선은 채택 안 함).

실행: cd strategies/cap_defend/research && python3 bt_spot_drift_recheck.py
"""
from __future__ import annotations
import os
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')

import unified_backtest as ub                     # noqa: E402
from bt_spot_rebal_band import window_rs          # noqa: E402

BASE_TX = 0.004
START = '2020-10-01'
END = '2026-08-24'
PHASES = [0, 43, 87, 130, 173]
DRIFTS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12]


def run_cfg(drift, phase, tx):
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    return ub.run(BARS, FUNDING, interval='D', asset_type='spot', leverage=1.0,
                  sma_days=42, mom_short_days=20, mom_long_days=127,
                  vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=7,
                  universe_size=3, cap=1 / 3, tx_cost=tx,
                  health_mode='mom2vol', vol_mode='daily', drift_threshold=drift,
                  snap_interval_bars=217, phase_offset_bars=phase,
                  start_date=START, end_date=END)


def main():
    global BARS, FUNDING
    t0 = time.time()
    BARS, FUNDING = ub.load_data('D')
    print(f"데이터 로드 완료 ({time.time()-t0:.0f}s) | {START}~{END} | 위상 {len(PHASES)}종")
    print("체결 밴드 = 엔진 기본값(현물 0 = 라이브 규칙)\n")

    for mult in (1, 5):
        tx = BASE_TX * mult
        sums = defaultdict(float)
        wins = defaultdict(int)
        n_all = 0
        agg = defaultdict(list)
        for phase in PHASES:
            eqs = {}
            for d in DRIFTS:
                m = run_cfg(d, phase, tx)
                eqs[f"{d:.2f}"] = m['_equity']
                agg[f"{d:.2f}"].append(m)
            s, w, n = window_rs(eqs)
            for k, v in s.items():
                sums[k] += v
            for k, v in w.items():
                wins[k] += v
            n_all += n
        print(f"=== 거래비용 {mult}배 (편도 {tx*100:.1f}%) · 윈도우 {n_all}개 ===")
        print(f"  {'drift':>6} {'순위':>6} {'승률':>6} {'Cal':>6} {'CAGR':>8} {'MDD':>8} "
              f"{'Rebal':>6} {'체결':>6}")
        best = min(DRIFTS, key=lambda d: sums[f"{d:.2f}"])
        for d in DRIFTS:
            k = f"{d:.2f}"
            r = agg[k]
            mark = " <= 현행" if d == 0.10 else (" (최상위)" if d == best else "")
            print(f"  {k:>6} {sums[k]/n_all:>6.3f} {wins[k]/n_all*100:>5.1f}% "
                  f"{np.mean([x['Cal'] for x in r]):>6.2f} "
                  f"{np.mean([x['CAGR'] for x in r]):>+7.1%} "
                  f"{np.mean([x['MDD'] for x in r]):>+7.1%} "
                  f"{np.mean([x['Rebal'] for x in r]):>6.0f} "
                  f"{np.mean([x['Trades'] for x in r]):>6.0f}{mark}")
        gap = (sums['0.10'] - sums[f"{best:.2f}"]) / n_all
        print(f"  현행 0.10 과 최상위({best:.2f}) 평균순위 차 {gap:+.3f} → "
              f"{'채택값 유지' if gap <= 0.2 else '재검토 필요'}\n", flush=True)

    print(f"총 소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
