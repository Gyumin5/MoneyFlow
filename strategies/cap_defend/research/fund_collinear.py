"""펀딩비가 현행 신호와 얼마나 겹치는가 (2026-09-02).

bt_fut_funding.py 에서 펀딩비를 배수 상한으로 넣은 세 설계가 전부 현행 이하로 나왔다.
그리고 세 계열 모두 "상한이 덜 물수록 좋아진다" 는 한 방향으로만 움직였다 — 규칙이 물 때마다
손해라는 뜻이다. 왜 그런지를 확정한다.

가설: 펀딩비는 가격이 오른 뒤에 따라 오르는 값이라, 현행이 이미 쓰는 모멘텀·단기선 이격과
같은 것을 늦게 다시 말한다. 그렇다면 배수축이든 선정층이든 새로 줄 정보가 없다.

재는 것은 순위상관(스피어만)이다. 값의 크기가 아니라 순서만 보므로 단위가 달라도 비교된다.
  - 펀딩 7일 평균  vs  단기 모멘텀(18일 수익)      ← 선물 헬스 필터가 쓰는 값
  - 펀딩 7일 평균  vs  장기 모멘텀(127일 수익)     ← 선물 헬스 필터가 쓰는 값
  - 펀딩 7일 평균  vs  종가/SMA7 이격             ← K2 배수가 쓰는 값
  - 펀딩 7일 평균  vs  종가/SMA42 이격            ← 카나리·BTC 상한이 쓰는 값
모든 값은 같은 날 기준으로 맞춘다(신호 시점 동일). 상관이 높으면 독립 입력이 아니다.

같이 본다: 배수가 3·4배인 날에 펀딩 상위 구간이 몰려 있는가. 몰려 있으면 펀딩 과열 상한은
하필 배수를 올려둔 날만 골라 깎는 규칙이 된다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 fund_collinear.py
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_fut_funding as ff                                            # noqa: E402

START, END = ff.START, ff.END


def main():
    from unified_backtest import load_data
    bars, funding = load_data('D')

    rows = []
    for coin, df in bars.items():
        close = pd.Series(df['Close'].values, index=df.index)
        f = ff._fund_series(coin, close.index, funding)
        d = pd.DataFrame({
            'fund': f,
            'mom18': close.pct_change(18),
            'mom127': close.pct_change(127),
            'sma7': close / close.rolling(7).mean() - 1,
            'sma42': close / close.rolling(42).mean() - 1,
        }).loc[START:END].dropna()
        if len(d) > 200:
            d['coin'] = coin
            rows.append(d)

    all_ = pd.concat(rows)
    print(f"표본 {len(all_):,} 코인-일 / {all_['coin'].nunique()}종  ({START}~{END})\n")

    print("[펀딩 7일 평균과 현행 신호의 순위상관 — 전체 풀]")
    for k, label in (('mom18', '단기 모멘텀 18일'), ('mom127', '장기 모멘텀 127일'),
                     ('sma7', '종가/SMA7 이격'), ('sma42', '종가/SMA42 이격')):
        r = all_['fund'].corr(all_[k], method='spearman')
        print(f"  {label:<20s} {r:+.3f}")

    print("\n[코인별 순위상관 분포 — 한두 종목이 끄는 값이 아닌지]")
    for k, label in (('mom18', '단기 모멘텀 18일'), ('sma7', '종가/SMA7 이격')):
        per = all_.groupby('coin').apply(
            lambda g: g['fund'].corr(g[k], method='spearman'))
        print(f"  {label:<20s} 중앙값 {per.median():+.3f}  "
              f"최소 {per.min():+.3f}  최대 {per.max():+.3f}  "
              f"양수 {int((per > 0).sum())}/{len(per)}종")

    print("\n[배수 3·4배인 날에 펀딩 상위 구간이 몰려 있나]")
    lev = ff.build_lev(bars, funding, None, None)
    fu, lv = [], []
    for coin, s in lev.items():
        a = s.loc[START:END]
        f = ff._fund_series(coin, a.index, funding)
        ok = a.notna() & f.notna()
        fu.append(f[ok])
        lv.append(a[ok])
    fu = pd.concat(fu)
    lv = pd.concat(lv)
    for q in (0.90, 0.95, 0.99):
        thr = fu.quantile(q)
        hi = fu > thr
        base = float(hi.mean())
        cond = float(hi[lv > 2].mean())
        print(f"  펀딩 상위 {int((1-q)*100)}% (문턱 {thr:+.6f}): "
              f"전체 {base:.1%} → 배수 3·4배인 날 {cond:.1%}  "
              f"(집중도 {cond/base:.2f}배)")
    print(f"  배수 3·4배인 날 비율 {float((lv > 2).mean()):.1%}")

    print("\n[펀딩 구간별 평균 배수 — 펀딩이 높을 때 이미 배수가 높은가]")
    b = pd.qcut(fu, 5, labels=['q1 최저', 'q2', 'q3', 'q4', 'q5 최고'])
    for name, g in lv.groupby(b, observed=True):
        print(f"  {str(name):<8s} 평균 배수 {g.mean():.3f}  "
              f"(3·4배 비율 {float((g > 2).mean()):.1%})")


if __name__ == '__main__':
    main()
