"""P1-b — 합성 60/25/15 가 송금 가정에 얼마나 기대나 + 세 슬리브 꼬리동조 (2026-08-25).

왜 재나: 지금까지 인용해 온 합성 수치(Cal 4.5~4.7)는 build_alloc, 즉 매일 60/25/15 로
되돌리는 계산이다. 실제로는 계좌가 셋으로 갈려 있고 자금 이동은 사람이 손으로 한다 —
증권계좌 T+2 결제 + 출금 + 환전 + 입금이라 트리거가 떠도 며칠 뒤에나 반영된다.
즉 인용치는 실행 불가능한 상한이다. 얼마나 상한인지를 재는 게 이 측정의 목적이다.

축은 송금 현실성 하나. 슬리브 전략·비용·기간은 라이브 설정 그대로 고정한다.

  L0 매일 재조정      매일 60/25/15 로 되돌린다. 지금 인용해 온 계산 = 상한.
  L1 트리거 즉시      T1(half_turnover 20pp) 또는 T3U_can(상대 미달 20% + 그 슬리브 카나리 ON)
                     발화 당일 100% 송금. 2026-05-26 alloc 결정 때 쓴 가정.
  L2 트리거 + 3영업일  발화 후 3거래일 뒤 반영 (T+2 결제 + 당일 이체를 가정한 최선)
  L3 트리거 + 5영업일  현실적 기본값 (주말·환전 포함)
  L4 트리거 + 10영업일 사람이 늦게 볼 때
  L5 송금 없음        한 번 넣고 방치. 비중이 무한정 흘러간다 = 하한.

무엇을 보나 (결과 보기 전 고정):
  1. L0 대 L3 의 Calmar 격차 = 인용치에 섞인 가정 프리미엄. 이게 크면 인용치를 고쳐야 한다.
  2. L1 대 L3~L4 격차 = 사람이 늦는 값. 작으면 송금 지연은 걱정거리가 아니다.
  3. L5 가 L3 과 비슷하면 alloc 트리거 자체가 값을 못 하는 것이다(별도 결정 사안).
  파라미터를 고르는 측정이 아니다 — 어떤 결론이 나와도 라이브 트리거 값은 안 건드린다.

꼬리동조: 합성의 방어는 세 슬리브가 같이 안 무너진다는 전제에 기댄다. 전체 상관이 낮아도
급락 국면에서만 같이 떨어지면 그 전제가 깨진다. 전구간 상관과 하위 5% 국면 상관을 같이 본다.

실행: cd strategies/cap_defend/research && python3 bt_alloc_transfer_lag.py
"""
from __future__ import annotations
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

from bt_fut_rebal_band import (  # noqa: E402
    run_spot_live, run_stock_live, metrics, window_rs, WIN_SIZES, STRIDES,
)
from bt_fut_drift_def import run_fut  # noqa: E402  (드리프트 정의 A = 현행)

TGT = np.array([0.60, 0.25, 0.15])        # stock / spot / fut
T1 = 0.20                                  # half_turnover 발화선
T3U = 0.20                                 # 상대 미달 발화선
ARMS = [
    ("L0 매일 재조정", 'daily', 0),
    ("L1 트리거 즉시", 'trigger', 0),
    ("L2 트리거+3일", 'trigger', 3),
    ("L3 트리거+5일", 'trigger', 5),
    ("L4 트리거+10일", 'trigger', 10),
    ("L5 송금 없음", 'none', 0),
]


def _canary_on(rets, i, lookback=20):
    """그 슬리브가 위험선호 국면인지 근사 — 최근 20거래일 평균 수익 > 0.

    라이브 카나리(주식 EEM SMA200 / 코인 BTC SMA42)를 그대로 못 가져오는 자리라
    기존 bt_v25_t1_t3u.py 와 같은 근사를 쓴다. 이 근사는 팔 6개에 똑같이 적용되므로
    팔 사이 비교에는 영향이 없다(수준값 해석에만 주의).
    """
    if i < lookback:
        return False
    return float(rets[i - lookback:i].mean()) > 0


def simulate(eq_st, eq_sp, eq_fu, mode, lag):
    """일별 합성 시뮬레이션. mode=daily/trigger/none, lag=영업일 지연."""
    common = sorted(eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index))
    R = np.column_stack([
        eq_st.loc[common].pct_change().fillna(0).values,
        eq_sp.loc[common].pct_change().fillna(0).values,
        eq_fu.loc[common].pct_change().fillna(0).values,
    ])
    n = len(common)
    v = TGT.copy()                      # 초기 자본 1, 목표비중대로 배분
    pv = np.empty(n)
    pending = -1                        # 예약된 송금 실행일 인덱스
    fires = 0
    resets = 0
    max_ht = 0.0
    for i in range(n):
        v = v * (1 + R[i])
        tot = v.sum()
        pv[i] = tot
        w = v / tot if tot > 0 else TGT.copy()
        ht = float(np.abs(w - TGT).sum() / 2)
        max_ht = max(max_ht, ht)

        if mode == 'daily':
            v = tot * TGT
            continue
        if mode == 'none':
            continue

        if pending == i:                # 오늘이 송금 반영일
            v = tot * TGT
            resets += 1
            pending = -1
            continue
        if pending >= 0:                # 이미 예약돼 있으면 새로 안 잡는다
            continue

        fire = ht >= T1
        if not fire:
            rel = (TGT - w) / TGT       # 목표 대비 상대 미달
            for s in range(3):
                if rel[s] >= T3U and _canary_on(R[:, s], i):
                    fire = True
                    break
        if fire:
            fires += 1
            pending = min(i + lag, n - 1) if lag > 0 else i
            if lag == 0:
                v = tot * TGT
                resets += 1
                pending = -1
    eq = pd.Series(pv, index=pd.DatetimeIndex(common))
    return eq, dict(fires=fires, resets=resets, max_ht=max_ht,
                    end_w=(v / v.sum()) if v.sum() > 0 else TGT)


def tail_table(eq_st, eq_sp, eq_fu):
    common = sorted(eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index))
    df = pd.DataFrame({
        '주식': eq_st.loc[common].pct_change(),
        '현물': eq_sp.loc[common].pct_change(),
        '선물': eq_fu.loc[common].pct_change(),
    }).dropna()
    print("\n[꼬리동조] 일별 수익 상관 — 전구간 vs 하위 5% 국면")
    pairs = [('주식', '현물'), ('주식', '선물'), ('현물', '선물')]
    print(f"  {'쌍':<12s} {'전구간':>8s} {'각자 하위5%일':>14s} {'합성 하위5%일':>14s}")
    comp = df.mul([0.60, 0.25, 0.15]).sum(axis=1)
    worst = comp <= comp.quantile(0.05)
    for a, b in pairs:
        thr_a = df[a].quantile(0.05)
        m = df[a] <= thr_a
        print(f"  {a}-{b:<9s} {df[a].corr(df[b]):>8.3f} {df.loc[m, a].corr(df.loc[m, b]):>14.3f} "
              f"{df.loc[worst, a].corr(df.loc[worst, b]):>14.3f}")
    print(f"  (하위5% 기준: 왼쪽 자산의 최악 5% 일 {int(len(df)*0.05)}일 / 합성 최악 5% 일 {int(worst.sum())}일)")

    print("\n[동시 하락] 하루에 몇 개 슬리브가 같이 떨어졌나")
    neg = (df < 0).sum(axis=1)
    for k in range(4):
        print(f"  {k}개 하락: {int((neg == k).sum()):>5d}일 ({(neg == k).mean()*100:>4.1f}%)  "
              f"그날 합성 평균 {comp[neg == k].mean()*100:>+6.2f}%")

    print("\n[합성 최악 20일] 각 슬리브가 그날 얼마였나")
    w20 = comp.nsmallest(20)
    print(f"  {'날짜':<12s} {'합성':>8s} {'주식':>8s} {'현물':>8s} {'선물':>8s}")
    for d, c in w20.items():
        print(f"  {d.date()!s:<12s} {c*100:>+7.2f}% {df.loc[d, '주식']*100:>+7.2f}% "
              f"{df.loc[d, '현물']*100:>+7.2f}% {df.loc[d, '선물']*100:>+7.2f}%")
    n_all3 = int((df.loc[w20.index] < 0).sum(axis=1).eq(3).sum())
    print(f"  최악 20일 중 세 슬리브가 모두 하락한 날: {n_all3}일")


def main():
    t0 = time.time()
    print("P1-b 합성 송금 현실성 + 꼬리동조")
    print(f"목표비중 주식 {TGT[0]:.0%} / 현물 {TGT[1]:.0%} / 선물 {TGT[2]:.0%} | "
          f"T1 {T1:.0%} · T3U_can {T3U:.0%} (라이브 값, 고정)")

    eq_sp = run_spot_live()
    eq_st = run_stock_live()
    eq_fu = run_fut('margin', 0.03)['_equity']     # 현행 드리프트 정의
    print(f"슬리브 로드 완료 ({time.time()-t0:.0f}s)\n")

    print(f"  {'arm':<16s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} "
          f"{'발화':>5s} {'송금':>5s} {'최대이탈':>8s} {'최종배수':>9s}")
    eqs = {}
    stats = {}
    for name, mode, lag in ARMS:
        eq, st = simulate(eq_st, eq_sp, eq_fu, mode, lag)
        eqs[name] = eq
        stats[name] = st
        m = metrics(eq)
        print(f"  {name:<16s} {m['Cal']:>6.2f} {m['CAGR']:>+7.1f}% {m['MDD']:>+7.1f}% "
              f"{m['Sharpe']:>7.2f} {st['fires']:>5d} {st['resets']:>5d} "
              f"{st['max_ht']*100:>7.1f}pp {eq.iloc[-1]/eq.iloc[0]:>8.1f}x")

    print("\n[윈도우 rank-sum]")
    sums, wins, n = window_rs(eqs)
    for k in sorted(sums, key=lambda k: sums[k]):
        print(f"  {k:<16s} avg_rank {sums[k]/n:.3f}  wins {wins[k]:>4d}/{n}")
    print(f"  windows n={n} (sizes {WIN_SIZES} × strides {STRIDES})")

    print("\n[연도별 수익률]")
    keys = list(eqs)
    print("  year  " + "".join(f"{k:>16s}" for k in keys))
    for year in sorted({d.year for d in eqs[keys[0]].index}):
        cells = []
        for k in keys:
            seg = eqs[k][eqs[k].index.year == year].dropna()
            cells.append(f"{'-':>16s}" if len(seg) < 30
                         else f"{seg.iloc[-1]/seg.iloc[0]-1:>+15.1%} ")
        print(f"  {year:<5d} " + "".join(cells))

    base, real = "L0 매일 재조정", "L3 트리거+5일"
    c0, c3 = metrics(eqs[base])['Cal'], metrics(eqs[real])['Cal']
    c1, c5 = metrics(eqs["L1 트리거 즉시"])['Cal'], metrics(eqs["L5 송금 없음"])['Cal']
    print(f"\n[판정] 가정 프리미엄 = L0 {c0:.2f} - L3 {c3:.2f} = {c0-c3:+.2f} "
          f"({'인용치 정정 필요' if abs(c0-c3) >= 0.3 else '인용치 유지 가능'})")
    print(f"  사람이 늦는 값 = L1 {c1:.2f} - L3 {c3:.2f} = {c1-c3:+.2f}")
    print(f"  트리거의 값 = L3 {c3:.2f} - L5 {c5:.2f} = {c3-c5:+.2f} "
          f"({'트리거가 값을 한다' if c3-c5 >= 0.3 else '트리거 기여 미미 — 별도 결정 사안'})")

    tail_table(eq_st, eq_sp, eq_fu)
    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
