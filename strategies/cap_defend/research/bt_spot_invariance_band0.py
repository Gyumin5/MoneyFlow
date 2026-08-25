"""현물 채택값 불변성 확인 — 체결 밴드를 라이브(0)로 맞춘 뒤 (2026-08-25).

재최적화가 아니다. ai-debate run-20260825T071922Z 중재자 조건 — "밴드를 0 으로 바꿨으면
5% 밴드에서 고른 다른 채택값들이 여전히 같은 자리인지 확인하라" — 를 채우는 확인이다.
드리프트 문턱은 bt_spot_drift_recheck.py 에서 따로 봤고, 여기서는 나머지 둘을 함께 본다.

축 둘을 같이 도는 이유: 변동성 상한(vol_threshold)은 어떤 코인이 들어오나를 정하고
스냅 간격(snap_interval_bars)은 언제 갈아타나를 정하는데, 둘 다 회전수를 통해 밴드와 얽힌다.
따로 보면 한 축을 고정한 채 다른 축만 흔드는 게 되어 상호작용을 놓친다.

후보는 채택값과 그 인접값만 — 새 최적점을 찾는 게 아니라 채택값이 여전히 봉우리 안인지만 본다.
  vol_threshold 0.04 / 0.05(채택) / 0.06
  snap_interval_bars 186 / 217(채택) / 248   (전부 31 의 배수 = 스태거 소수 관계 유지)

판정 (결과 보기 전 고정):
  채택 조합 (0.05, 217) 이 9개 중 상위권이고 인접값 대비 Calmar 급락이 없으면 → 불변 확인.
  다른 조합이 뚜렷하게 앞서면 사실만 보고하고 값은 바꾸지 않는다(별도 결정 사안).

실행: cd strategies/cap_defend/research && python3 bt_spot_invariance_band0.py
"""
from __future__ import annotations
import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import unified_backtest as ub                       # noqa: E402
from bt_spot_rebal_band import window_rs, START, END, BASE_TX  # noqa: E402

VOLS = [0.04, 0.05, 0.06]
SNAPS = [186, 217, 248]
PHASES = [0, 87, 173]
ADOPTED = (0.05, 217)


def run_cfg(vol, snap, phase, tx):
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    return ub.run(BARS, FUNDING, interval='D', asset_type='spot', leverage=1.0,
                  sma_bars=42, mom_short_bars=20, mom_long_bars=127,
                  vol_threshold=vol, vol_mode='daily', vol_days=90,
                  n_snapshots=7, snap_interval_bars=snap, phase_offset_bars=phase,
                  canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
                  universe_size=3, cap=1 / 3, health_mode='mom2vol',
                  tx_cost=tx, start_date=START, end_date=END)


def main():
    global BARS, FUNDING
    t0 = time.time()
    BARS, FUNDING = ub.load_data('D')
    print(f"기간 {START} ~ {END} | 체결 밴드 = 엔진 기본값(현물 0 = 라이브)")
    print(f"vol {VOLS} × snap {SNAPS} × 위상 {PHASES} | tx {BASE_TX}\n")

    agg = defaultdict(list)
    sums = defaultdict(float)
    wins = defaultdict(int)
    n_all = 0
    for phase in PHASES:
        eqs = {}
        for vol in VOLS:
            for snap in SNAPS:
                m = run_cfg(vol, snap, phase, BASE_TX)
                k = f"vol{vol:.2f}/snap{snap}"
                eqs[k] = m['_equity']
                agg[k].append(m)
        s, w, n = window_rs(eqs)
        for k, v in s.items():
            sums[k] += v
        for k, v in w.items():
            wins[k] += v
        n_all += n
        print(f"  위상 {phase:>3d} 완료 ({time.time()-t0:.0f}s)", flush=True)

    print(f"\n윈도우 {n_all}개 (위상 3종 합산)")
    print(f"  {'조합':<18s} {'순위':>6s} {'승률':>6s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} "
          f"{'체결':>6s}")
    ranked = sorted(sums, key=lambda k: sums[k])
    ak = f"vol{ADOPTED[0]:.2f}/snap{ADOPTED[1]}"
    for k in ranked:
        r = agg[k]
        mark = " <= 채택" if k == ak else ""
        print(f"  {k:<18s} {sums[k]/n_all:>6.3f} {wins[k]/n_all*100:>5.1f}% "
              f"{np.mean([x['Cal'] for x in r]):>6.2f} "
              f"{np.mean([x['CAGR'] for x in r]):>+7.1%} "
              f"{np.mean([x['MDD'] for x in r]):>+7.1%} "
              f"{np.mean([x['Trades'] for x in r]):>6.0f}{mark}")

    pos = ranked.index(ak) + 1
    gap = (sums[ak] - sums[ranked[0]]) / n_all
    cal_a = np.mean([x['Cal'] for x in agg[ak]])
    cal_best = np.mean([x['Cal'] for x in agg[ranked[0]]])
    print(f"\n[판정] 채택 조합 {ak} 은 9개 중 {pos}위, 최상위와 평균순위 차 {gap:+.3f}, "
          f"Calmar {cal_a:.2f} vs 최상위 {cal_best:.2f}")
    print(f"  {'불변 확인 — 값 유지' if pos <= 3 and cal_best - cal_a < 0.3 else '재검토 필요 — 사실만 보고, 자동 변경 없음'}")
    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
