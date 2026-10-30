"""비중 조합을 이 저장소 기준(창 순위합)으로 겨룬다 (2026-10-30).

왜 재나. 운영자가 물었다 — 70/15/15 가 현행 60/25/15 보다 더 좋은가. 앵커 평균
한 번으로는 칼마가 3.80 대 3.56 이라 앞서 보이는데, 이 저장소에서 자산배분을 고를 때
쓰는 기준은 그게 아니다. 겹치는 창을 여러 길이로 잘라 순위를 매기고, 비용 5배에서도
같은 순서가 나오는지 본다. 창을 자르면 전 구간 칼마 하나로 본 결론이 뒤집힌 사례가
이 저장소에 여러 번 있다(history #75 계열).

재는 것은 창 순위합(낮을수록 우세)과 창 1위 횟수다. 기본 비용과 비용 5배 두 조건에서
각각 잰다. 앵커는 7곳이고 창 길이·이동폭은 bt_coin_param_resweep 의 값을 그대로 쓴다.

성과만 본다 — 거래소 파산 노출은 alloc_exchange_cap.py 가 따로 잰다.

실행: python3 alloc_mix_window_rs.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# (이름, 주식, 현물, 선물)
MIXES = [
    ('현행 60/25/15', 0.60, 0.25, 0.15),
    ('70/15/15', 0.70, 0.15, 0.15),
    ('70/20/10', 0.70, 0.20, 0.10),
    ('75/15/10', 0.75, 0.15, 0.10),
    ('80/10/10', 0.80, 0.10, 0.10),
    ('85/8/7', 0.85, 0.08, 0.07),
]


def main() -> int:
    import bt_coin_param_resweep as R
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal

    t0 = time.time()
    print('코인 일봉 로드 중...')
    R._bars, R._funding = load_data('D')
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f'  심볼 {len(R._bars)}종 ({time.time() - t0:.0f}초)')
    print(f'  창 길이 {R.WIN_SIZES} · 이동폭 {R.STRIDES}')

    for stress in (False, True):
        label = '비용 5배' if stress else '기본 비용'
        sums: dict[str, float] = defaultdict(float)
        wins: dict[str, int] = defaultdict(int)
        total_n = 0
        cal_rows: dict[str, list[float]] = defaultdict(list)
        anchors = 0
        for i in range(7):
            st = R.run_stock(i, stress)
            sp = R.run_spot({}, R.PH_SPOT[i], stress)
            fu = R.run_fut({}, R.PH_FUT[i], stress)
            if st is None or sp is None or fu is None:
                continue
            eqs = {name: R.build_alloc(st, sp, fu, w=(ws, wsp, wfu))
                   for name, ws, wsp, wfu in MIXES}
            for name, eq in eqs.items():
                cal_rows[name].append(R.metrics(eq)['Cal'])
            s, w, n = R.window_rs(eqs)
            for k, v in s.items():
                sums[k] += v
            for k, v in w.items():
                wins[k] += v
            total_n += n
            anchors += 1
            print(f'  [{label}] 앵커 {i} 끝 (창 {n}, {time.time() - t0:.0f}초)')

        if not total_n:
            print(f'=== {label}: 창을 못 만들었다')
            continue
        print()
        print(f'=== {label} — 창 순위합 (앵커 {anchors}곳 합산, 창 {total_n}개)')
        print(f'  {"비중":>14} {"평균순위":>9} {"창 1위":>8} {"1위 비율":>9} {"칼마 평균":>10}')
        for name, avg in sorted(((k, sums[k] / total_n) for k in sums), key=lambda x: x[1]):
            print(f'  {name:>14} {avg:>9.3f} {wins[name]:>7}회'
                  f' {wins[name] / total_n * 100:>8.1f}%'
                  f' {float(np.mean(cal_rows[name])):>10.2f}')
        print()

    print(f'끝 ({time.time() - t0:.0f}초)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
