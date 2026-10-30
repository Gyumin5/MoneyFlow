"""코인 총량을 고정하고 현물:선물 비율만 바꿔 겨룬다 (2026-10-30).

왜 재나. 앞선 측정(alloc_mix_window_rs.py)에서 창 순위합 상위 둘이 현물과 선물을 같은
비중으로 둔 조합이었다. 그래서 신호가 "코인을 줄여라" 가 아니라 "현물보다 선물을
키워라" 일 수 있는데, 그 격자에는 선물이 현물보다 큰 조합이 하나도 없었다. 한쪽만
있는 격자에서 1위가 나오면 그게 봉우리인지 격자 끝인지 구분이 안 된다.

그래서 코인 총량을 30%와 40% 두 값으로 고정하고 현물:선물을 2:1 · 1:1 · 1:2 로
갈라 여섯을 만든다. 현행 60/25/15 는 코인 40% 의 5:3 이라 같은 표에 기준선으로 둔다.
토론(run-20261030T123227Z)이 이름을 댄 70/10/20 · 60/20/20 · 60/15/25 가 모두 들어온다.

판정은 앞선 측정과 같다 — 겹치는 창 순위합을 기본 비용과 비용 5배 두 조건에서 잰다.

실행: python3 alloc_mix_ratio.py
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

# (이름, 주식, 현물, 선물). 코인 총량 30% 와 40% 를 각각 2:1 · 1:1 · 1:2 로 가른다.
MIXES = [
    ('현행 60/25/15', 0.60, 0.25, 0.15),
    ('코인30 2:1 70/20/10', 0.70, 0.20, 0.10),
    ('코인30 1:1 70/15/15', 0.70, 0.15, 0.15),
    ('코인30 1:2 70/10/20', 0.70, 0.10, 0.20),
    ('코인40 2:1 60/27/13', 0.60, 0.2667, 0.1333),
    ('코인40 1:1 60/20/20', 0.60, 0.20, 0.20),
    ('코인40 1:2 60/13/27', 0.60, 0.1333, 0.2667),
    ('코인40 1:2.5 60/15/25', 0.60, 0.15, 0.25),
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
        head: dict[str, int] = defaultdict(int)   # 현행과의 직접 승 (토론 보조 검증)
        total_n = 0
        cal_rows: dict[str, list[float]] = defaultdict(list)
        cagr_rows: dict[str, list[float]] = defaultdict(list)
        mdd_rows: dict[str, list[float]] = defaultdict(list)
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
                m = R.metrics(eq)
                cal_rows[name].append(m['Cal'])
                cagr_rows[name].append(m['CAGR'])
                mdd_rows[name].append(m['MDD'])
            s, w, n = R.window_rs(eqs)
            for k, v in s.items():
                sums[k] += v
            for k, v in w.items():
                wins[k] += v
            total_n += n
            anchors += 1
            # 창별 직접 승률: 같은 창에서 현행보다 칼마가 높았나
            for size in R.WIN_SIZES:
                for stride in R.STRIDES:
                    idx = sorted(set.intersection(*[set(e.index) for e in eqs.values()]))
                    for j in range(0, len(idx) - size, stride):
                        d0, d1 = idx[j], idx[j + size - 1]
                        cal = {}
                        for k, e in eqs.items():
                            seg = e.loc[d0:d1].dropna()
                            if len(seg) < 30:
                                cal = {}
                                break
                            yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                            c = (seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
                            md = float((seg / seg.cummax() - 1).min())
                            cal[k] = c / abs(md) if md < 0 else 0.0
                        if not cal:
                            continue
                        base = cal['현행 60/25/15']
                        for k, v in cal.items():
                            if v > base:
                                head[k] += 1
            print(f'  [{label}] 앵커 {i} 끝 (창 {n}, {time.time() - t0:.0f}초)')

        if not total_n:
            print(f'=== {label}: 창을 못 만들었다')
            continue
        print()
        print(f'=== {label} — 창 순위합 (앵커 {anchors}곳 합산, 창 {total_n}개)')
        print(f'  {"비중":>22} {"평균순위":>9} {"창 1위":>8} {"현행 직접승":>11}'
              f' {"칼마":>7} {"연복리":>8} {"낙폭":>8}')
        for name, avg in sorted(((k, sums[k] / total_n) for k in sums), key=lambda x: x[1]):
            print(f'  {name:>22} {avg:>9.3f} {wins[name]:>7}회'
                  f' {head[name] / total_n * 100:>10.1f}%'
                  f' {float(np.mean(cal_rows[name])):>7.2f}'
                  f' {float(np.mean(cagr_rows[name])):>7.1f}%'
                  f' {float(np.mean(mdd_rows[name])):>7.1f}%')
        print()

    print(f'끝 ({time.time() - t0:.0f}초)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
