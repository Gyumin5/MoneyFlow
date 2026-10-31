"""비중마다 자기 최적 송금 타이밍을 찾아 공정하게 겨룬다 (2026-10-31).

왜 재나. 운영자 지적 — 지금 트리거(반회전율 20퍼센트포인트, 상대부족 20퍼센트 + 그 슬리브
카나리 ON)는 60/25/15 에서 고른 값이다. 다른 비중에 그대로 씌우면 현행에 유리한 비교다.
비중이 달라지면 세 슬리브가 벌어지는 속도도 달라지므로 되돌릴 시점도 달라져야 한다.

두 단계로 한다.
  1단계 — 비중마다 트리거 격자(반회전율 5칸 × 상대부족 4칸 = 20칸)를 그 비중 안에서만
          창 순위합으로 겨뤄 최적을 고른다. 같은 비중이면 슬리브 곡선이 같으므로 차이는
          오직 타이밍이다.
  2단계 — 열두 비중을 각자의 최적 타이밍으로 다시 겨룬다. 비교용으로 전부 라이브 값
          (20·20)으로 고정한 표도 같이 찍는다 — 그게 불공정했던 쪽이다.

고르는 자유도만큼 좋아 보이는 쪽이 생기므로 봉우리 하나를 믿지 않는다. 비중마다 20칸
평균순위를 격자 모양 그대로 찍어 평지인지 봉우리인지 눈으로 보게 하고, 비용 5배에서도
같은 자리가 뽑히는지 본다.

바꾼 축은 트리거 하나뿐이다. 슬리브 백테스트·앵커·비용 조건·창 길이·이동폭·합성 모형은
alloc_mix_real_rebal.py 와 같은 것을 그대로 쓴다(그 모듈의 함수를 불러 쓴다 — 베끼지 않는다).

실행: python3 alloc_mix_trigger_opt.py
"""
from __future__ import annotations

import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from alloc_mix_real_rebal import (MIXES, BASE, canary_from_trace,
                                  build_alloc_real, table)

GRID_HT = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]    # T1 반회전율
GRID_T3U = [0.15, 0.20, 0.25, 0.30, 0.40]               # T3U_can 상대부족
COMBOS = [(h, t) for h in GRID_HT for t in GRID_T3U]
LIVE = (0.20, 0.20)                          # 라이브 값 = 불공정 기준선


def cname(c: tuple[float, float]) -> str:
    return f'{c[0] * 100:.0f}/{c[1] * 100:.0f}'


def print_grid(name: str, avg: dict[tuple[float, float], float],
               best: tuple[float, float]) -> None:
    print(f'  [{name}] 평균순위 격자 (행=반회전율 %p, 열=상대부족 %) — 낮을수록 우세')
    print('        ' + ''.join(f'{t * 100:>8.0f}' for t in GRID_T3U))
    for h in GRID_HT:
        cells = []
        for t in GRID_T3U:
            v = avg[(h, t)]
            cells.append(f'{v:>7.2f}' + ('*' if (h, t) == best else ' '))
        print(f'  {h * 100:>5.0f} ' + ''.join(cells))


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

    print('주식 카나리(EEM) 로드 중...')
    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    pdf = pd.DataFrame(load_prices(ALL_TICKERS, start="2005-01-01"))
    pdf = pdf[~pdf.index.duplicated(keep="first")].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    can_st = precompute(pdf, [30, 72, 230], [42, 63, 126])[3].astype(bool)
    print(f'  주식 카나리 {len(can_st)}일, ON 비율 {can_st.mean() * 100:.1f}%'
          f' ({time.time() - t0:.0f}초)')
    print(f'  창 길이 {R.WIN_SIZES} · 이동폭 {R.STRIDES}')
    print(f'  트리거 격자 {len(COMBOS)}칸 × 비중 {len(MIXES)}칸')

    for stress in (False, True):
        label = '비용 5배' if stress else '기본 비용'
        # 1단계 누적: 비중별로 20칸을 그 안에서만 겨룬다
        s1_sums = {m[0]: defaultdict(float) for m in MIXES}
        s1_n = {m[0]: 0 for m in MIXES}
        store: dict = {}
        fires: dict = defaultdict(list)
        anchors = []
        for i in range(7):
            st = R.run_stock(i, stress)
            tr_sp: list = []
            sp = R.run_spot({}, R.PH_SPOT[i], stress, trace=tr_sp)
            tr_fu: list = []
            fu = R.run_fut({}, R.PH_FUT[i], stress, trace=tr_fu)
            if st is None or sp is None or fu is None:
                continue
            can_sp = canary_from_trace(tr_sp)
            can_fu = canary_from_trace(tr_fu)
            for name, ws, wsp, wfu in MIXES:
                eqs = {}
                for c in COMBOS:
                    eq, nf = build_alloc_real(st, sp, fu, can_st, can_sp, can_fu,
                                              (ws, wsp, wfu), ht_thr=c[0], t3u_rel=c[1])
                    eqs[cname(c)] = eq
                    store[(i, name, c)] = eq
                    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
                    fires[(name, c)].append(nf / yrs)
                s, _w, n = R.window_rs(eqs)
                for k, v in s.items():
                    s1_sums[name][k] += v
                s1_n[name] += n
            anchors.append(i)
            print(f'  [{label}] 앵커 {i} 끝 ({time.time() - t0:.0f}초)')

        if not anchors:
            print(f'=== {label}: 앵커를 못 돌렸다')
            continue

        print()
        print(f'=== {label} — 1단계: 비중마다 자기 최적 타이밍 (앵커 {len(anchors)}곳)')
        best: dict[str, tuple[float, float]] = {}
        for name, *_ in MIXES:
            avg = {c: s1_sums[name][cname(c)] / s1_n[name] for c in COMBOS}
            bc = min(COMBOS, key=lambda c: avg[c])
            best[name] = bc
            print_grid(name, avg, bc)
            print(f'      최적 = 반회전율 {bc[0] * 100:.0f}%p · 상대부족 {bc[1] * 100:.0f}%'
                  f' (평균순위 {avg[bc]:.2f}, 라이브값 {avg[LIVE]:.2f},'
                  f' 송금 연 {float(np.mean(fires[(name, bc)])):.1f}회)')

        # 2단계: 각자의 최적으로 열두 칸을 겨룬다 + 라이브 고정값 표
        for tag, pick in (('각자 최적 타이밍', best),
                          ('라이브 고정 20/20 (불공정 기준선)',
                           {m[0]: LIVE for m in MIXES})):
            A = dict(sums=defaultdict(float), wins=defaultdict(int),
                     head=defaultdict(int), n=0, cal=defaultdict(list),
                     cagr=defaultdict(list), mdd=defaultdict(list),
                     fire=defaultdict(list))
            for i in anchors:
                eqs = {m[0]: store[(i, m[0], pick[m[0]])] for m in MIXES}
                for name, eq in eqs.items():
                    m = R.metrics(eq)
                    A['cal'][name].append(m['Cal'])
                    A['cagr'][name].append(m['CAGR'])
                    A['mdd'][name].append(m['MDD'])
                    A['fire'][name].append(fires[(name, pick[name])][anchors.index(i)])
                s, w, n = R.window_rs(eqs)
                for k, v in s.items():
                    A['sums'][k] += v
                for k, v in w.items():
                    A['wins'][k] += v
                A['n'] += n
                idx = sorted(set.intersection(*[set(e.index) for e in eqs.values()]))
                for size in R.WIN_SIZES:
                    for stride in R.STRIDES:
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
                            b = cal[BASE]
                            for k, v in cal.items():
                                if v > b:
                                    A['head'][k] += 1
            table(f'{label} — 2단계: {tag}', A['sums'], A['wins'], A['head'],
                  A['n'], A['cal'], A['cagr'], A['mdd'], A['fire'], len(anchors))
            if tag.startswith('각자'):
                print('  (쓰인 타이밍: '
                      + ', '.join(f'{n} {best[n][0] * 100:.0f}/{best[n][1] * 100:.0f}'
                                  for n, *_ in MIXES) + ')')

        # 3단계: 타이밍을 지워서 겨룬다.
        # 비중마다 자기 최적을 골라 겨루면 고르는 자유도만큼 운이 섞인다 — 1단계 격자가
        # 평지가 아니라 들쭉날쭉하면 그 최적은 신호가 아니라 운이다. 그래서 타이밍 칸마다
        # 열두 비중을 겨룬 뒤 비중별 평균순위를 모든 칸에 걸쳐 평균한다. 모든 비중이 모든
        # 타이밍을 똑같이 겪으므로 불공정도 운도 안 남는다.
        m3: dict[str, list[float]] = defaultdict(list)
        m3_cal: dict[str, list[float]] = defaultdict(list)
        m3_cagr: dict[str, list[float]] = defaultdict(list)
        m3_mdd: dict[str, list[float]] = defaultdict(list)
        for c in COMBOS:
            cs: dict[str, float] = defaultdict(float)
            cn = 0
            for i in anchors:
                eqs = {m[0]: store[(i, m[0], c)] for m in MIXES}
                for name, eq in eqs.items():
                    mm = R.metrics(eq)
                    m3_cal[name].append(mm['Cal'])
                    m3_cagr[name].append(mm['CAGR'])
                    m3_mdd[name].append(mm['MDD'])
                s, _w, n = R.window_rs(eqs)
                for k, v in s.items():
                    cs[k] += v
                cn += n
            for k, v in cs.items():
                m3[k].append(v / cn)
        print()
        print(f'=== {label} — 3단계: 타이밍을 지우고 겨룬다'
              f' (타이밍 {len(COMBOS)}칸 평균, 앵커 {len(anchors)}곳)')
        print(f'  {"비중":>16} {"평균순위":>9} {"칸간 표준편차":>13}'
              f' {"칼마":>7} {"연복리":>8} {"낙폭":>8}')
        for name, v in sorted(((k, float(np.mean(m3[k]))) for k in m3),
                              key=lambda x: x[1]):
            print(f'  {name:>16} {v:>9.3f} {float(np.std(m3[name])):>13.3f}'
                  f' {float(np.mean(m3_cal[name])):>7.2f}'
                  f' {float(np.mean(m3_cagr[name])):>7.1f}%'
                  f' {float(np.mean(m3_mdd[name])):>7.1f}%')

    print()
    print(f'끝 ({time.time() - t0:.0f}초)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
