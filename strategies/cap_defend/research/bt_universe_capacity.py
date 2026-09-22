"""넓어진 유니버스에서 후보와 건강도 통과가 실제로 몇 종인지 진단한다 (2026-09-22).

코인 파라미터 재검증의 첫 단계다. 상위 몇 종을 담을지(universe_size)와 변동성 문턱을
건드리기 전에 물어야 할 것이 있다 — 그날 고를 수 있는 코인이 애초에 몇 종인가.
건강도를 통과하는 종이 평소 3종을 못 넘으면 상위 4종으로 넓히는 것은 아무 일도 안 한다.
반대로 늘 6~7종이 통과하고 있으면 상한 3종이 진짜 구속조건이다.

엔진의 선정층을 그대로 다시 적지 않고 unified_backtest 의 함수를 가져다 쓴다
(get_mcap, calc_mom, calc_vol_daily). 봉수 바닥도 엔진과 같은 식으로 계산한다.

재는 것:
  [1] 시점별 상위 목록 크기, 봉 127개 이상 쌓인 후보 수, 건강도 통과 수 (연도별)
  [2] 건강도 통과 수의 분포 — 통과가 0·1·2·3·4종 이상인 날이 각각 며칠인가
  [3] 변동성 문턱 0.03~0.08 에서 [2] 가 어떻게 움직이는가
  [4] 옛 40종 목록만 썼을 때와의 대조 (같은 날 같은 기준)

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_universe_capacity.py
"""
from __future__ import annotations

import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import unified_backtest as ub

START = pd.Timestamp("2020-10-01")
END = pd.Timestamp("2026-09-18")

SMA = 42
MOM_SHORT = 20      # 현물 기준. 선물은 18 이라 바닥이 같다(127 이 지배).
MOM_LONG = 127
VOL_BARS = 90
VOL_GRID = [0.03, 0.04, 0.05, 0.06, 0.07, 0.08]
CUR_VOL = 0.05

FLOOR = max(MOM_SHORT, MOM_LONG, VOL_BARS, SMA)

# 결함 수정 전 download_futures_data.py 에 손으로 적혀 있던 목록 (커밋 64843f3 이전).
OLD40 = {
    "BTC", "ETH", "BNB", "XRP", "ADA", "DOGE", "SOL", "TRX", "DOT", "MATIC",
    "LTC", "SHIB", "AVAX", "LINK", "ATOM", "XLM", "BCH", "NEAR", "APT", "FIL",
    "ICP", "HBAR", "VET", "ARB", "OP", "AAVE", "ALGO", "SAND", "MANA", "AXS",
    "EOS", "THETA", "FTM", "EGLD", "XTZ", "CHZ", "GRT", "CRV", "SUI", "INJ",
}


def health(c: np.ndarray, vol_th: float) -> bool:
    """엔진 health_mode='mom2vol' 과 같은 판정."""
    ms = ub.calc_mom(c, MOM_SHORT)
    ml = ub.calc_mom(c, MOM_LONG)
    vol = ub.calc_vol_daily(c, 1, lookback_bars=VOL_BARS)
    return ms > 0 and ml > 0 and vol <= vol_th


def main() -> None:
    print("코인 일봉 로드 중...")
    bars, _ = ub.load_data("D")
    print(f"  심볼 {len(bars)}종\n")

    # 공통 날짜 축 = BTC 일봉
    dates = [d for d in bars["BTC"].index if START <= d <= END]
    print(f"기간 {START.date()} ~ {END.date()} · {len(dates)}일 · 봉수 바닥 {FLOOR}\n")

    rows = []
    for d in dates:
        order = ub.get_mcap(d)
        n_listed = len(order)
        cands = []
        for coin in order:
            df = bars.get(coin)
            if df is None:
                continue
            ci = df.index.get_indexer([d], method="ffill")[0]
            if ci < 0 or ci < FLOOR:
                continue
            cands.append((coin, df["Close"].values[: ci + 1]))
        rec = dict(date=d, 상위목록=n_listed, 후보=len(cands),
                   후보_옛40=sum(1 for c, _ in cands if c in OLD40))
        for vt in VOL_GRID:
            rec[f"건강_{vt}"] = sum(1 for _, c in cands if health(c, vt))
        rec[f"건강_옛40_{CUR_VOL}"] = sum(
            1 for cn, c in cands if cn in OLD40 and health(c, CUR_VOL))
        rows.append(rec)

    df = pd.DataFrame(rows).set_index("date")

    print("[1] 연도별 평균 — 상위 목록 크기 · 봉수 충족 후보 · 건강도 통과(문턱 0.05)")
    print(f"{'연도':<6}{'상위목록':>9}{'후보':>7}{'건강도통과':>11}"
          f"{'후보(옛40)':>11}{'건강도(옛40)':>13}")
    for y, g in df.groupby(df.index.year):
        print(f"{y:<6}{g['상위목록'].mean():>9.1f}{g['후보'].mean():>7.1f}"
              f"{g[f'건강_{CUR_VOL}'].mean():>11.2f}"
              f"{g['후보_옛40'].mean():>11.1f}{g[f'건강_옛40_{CUR_VOL}'].mean():>13.2f}")
    print(f"{'전체':<6}{df['상위목록'].mean():>9.1f}{df['후보'].mean():>7.1f}"
          f"{df[f'건강_{CUR_VOL}'].mean():>11.2f}"
          f"{df['후보_옛40'].mean():>11.1f}{df[f'건강_옛40_{CUR_VOL}'].mean():>13.2f}")

    print("\n[2] 건강도 통과 종 수의 분포 (문턱 0.05, 전 기간)")
    cnt = Counter(df[f"건강_{CUR_VOL}"])
    tot = len(df)
    print(f"{'통과 종수':<10}{'일수':>7}{'비율':>8}{'누적(이상)':>11}")
    cum = tot
    for k in sorted(cnt):
        print(f"{k:<10}{cnt[k]:>7}{cnt[k]/tot*100:>7.1f}%{cum/tot*100:>10.1f}%")
        cum -= cnt[k]
    over3 = int((df[f"건강_{CUR_VOL}"] > 3).sum())
    print(f"  상한 3종이 실제로 구속한 날 = 통과가 4종 이상인 날 {over3}일 ({over3/tot*100:.1f}%)")
    empty = int((df[f"건강_{CUR_VOL}"] == 0).sum())
    print(f"  한 종도 못 고른 날 {empty}일 ({empty/tot*100:.1f}%) — 그날은 전액 현금이다")

    print("\n[3] 변동성 문턱을 움직이면 통과 종 수가 어떻게 되나")
    print(f"{'문턱':<8}{'평균 통과':>10}{'0종인 날':>10}{'4종+ 날':>10}{'중앙':>7}{'최대':>7}")
    for vt in VOL_GRID:
        s = df[f"건강_{vt}"]
        mark = "  <- 현행" if vt == CUR_VOL else ""
        print(f"{vt:<8.2f}{s.mean():>10.2f}{(s==0).sum():>10}{(s>3).sum():>10}"
              f"{s.median():>7.0f}{s.max():>7}{mark}")

    print("\n[4] 유니버스가 넓어져 실제로 늘어난 몫")
    print(f"  후보 평균 {df['후보_옛40'].mean():.1f}종 → {df['후보'].mean():.1f}종 "
          f"(+{df['후보'].mean()-df['후보_옛40'].mean():.1f})")
    print(f"  건강도 통과 평균 {df[f'건강_옛40_{CUR_VOL}'].mean():.2f}종 → "
          f"{df[f'건강_{CUR_VOL}'].mean():.2f}종 "
          f"(+{df[f'건강_{CUR_VOL}'].mean()-df[f'건강_옛40_{CUR_VOL}'].mean():.2f})")
    gain = df[f"건강_{CUR_VOL}"] - df[f"건강_옛40_{CUR_VOL}"]
    print(f"  통과 종이 실제로 늘어난 날 {int((gain>0).sum())}일 ({(gain>0).mean()*100:.1f}%)")
    b3_old = int((df[f"건강_옛40_{CUR_VOL}"] > 3).sum())
    print(f"  4종 이상 통과한 날: 옛 40종 {b3_old}일 → 전체 {over3}일")


if __name__ == "__main__":
    main()
