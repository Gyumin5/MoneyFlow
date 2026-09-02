"""펀딩비 분포 실사 — 문턱을 임의로 고르지 않기 위한 사전 관측 (2026-09-02).

배수축에 펀딩비를 넣기 전에 먼저 본다.
  1. 어느 코인에 파일이 있고 없나 (없으면 그 코인은 신호 결측 = 상한 없음으로 처리해야 한다)
  2. 일 단위로 합친 펀딩비의 분위수 — 문턱 후보를 데이터에서 뽑는다
  3. 롤링 평균(3·7·14일)의 분위수 — 스윕 축 선택용
  4. 펀딩비가 실제로 다음날 수익과 관계가 있나 (부호만 본다, 채택 근거 아님)

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 fund_profile.py
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

START = '2020-10-01'
END = '2026-08-25'
QS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]


def daily_funding(fs: pd.Series) -> pd.Series:
    """8시간 정산 3회를 하루 합으로. 인덱스는 자정 정규화."""
    s = fs.copy()
    s.index = pd.to_datetime(s.index).normalize()
    return s.groupby(level=0).sum()


def main():
    from unified_backtest import load_data
    bars, funding = load_data('D')

    have = sorted(funding)
    miss = sorted(set(bars) - set(funding))
    print(f"바 {len(bars)}종 / 펀딩 {len(have)}종 / 펀딩 없음 {len(miss)}종: {miss}")

    daily = {}
    for c, fs in funding.items():
        d = daily_funding(fs).loc[START:END]
        if len(d) > 100:
            daily[c] = d
    print(f"기간 {START}~{END} 안에 100일 이상인 코인 {len(daily)}종\n")

    allv = pd.concat(daily.values())
    print(f"[일 합계 펀딩비 분포]  표본 {len(allv):,} 코인-일")
    print("  " + "  ".join(f"q{int(q*100):02d}={allv.quantile(q):+.6f}" for q in QS))
    print(f"  평균 {allv.mean():+.6f}  표준편차 {allv.std():.6f}  "
          f"음수비율 {(allv < 0).mean():.1%}")
    print(f"  연율 환산 중앙값 {allv.quantile(0.5)*365:+.1%}  "
          f"q90 {allv.quantile(0.90)*365:+.1%}  q99 {allv.quantile(0.99)*365:+.1%}\n")

    for win in (3, 7, 14):
        roll = pd.concat([d.rolling(win).mean() for d in daily.values()]).dropna()
        print(f"[{win}일 롤링 평균 분포]  표본 {len(roll):,}")
        print("  " + "  ".join(f"q{int(q*100):02d}={roll.quantile(q):+.6f}" for q in QS))

    print("\n[다음날 수익과의 관계 — 부호 확인용, 채택 근거 아님]")
    print(f"  {'창':>4s} {'상위10%다음날':>14s} {'하위10%다음날':>14s} "
          f"{'음수구간':>12s} {'전체':>10s}")
    for win in (3, 7, 14):
        xs, ys = [], []
        for c, d in daily.items():
            close = pd.Series(bars[c]['Close'].values,
                              index=pd.to_datetime(bars[c].index).normalize())
            close = close[~close.index.duplicated(keep='first')].loc[START:END]
            f = d.rolling(win).mean().reindex(close.index)
            r = close.pct_change().shift(-1)          # 다음날 수익
            ok = f.notna() & r.notna()
            xs.append(f[ok])
            ys.append(r[ok])
        x = pd.concat(xs)
        y = pd.concat(ys)
        hi, lo = x.quantile(0.90), x.quantile(0.10)
        print(f"  {win:>4d} {y[x >= hi].mean():>+13.4%} {y[x <= lo].mean():>+13.4%} "
              f"{y[x < 0].mean():>+11.4%} {y.mean():>+9.4%}")

    print("\n[코인별 커버리지 상위/하위 5]")
    lens = sorted(((len(d), c) for c, d in daily.items()), reverse=True)
    for n, c in lens[:5] + lens[-5:]:
        print(f"  {c:<8s} {n:>5d}일  첫날 {daily[c].index[0].date()}")


if __name__ == '__main__':
    main()
