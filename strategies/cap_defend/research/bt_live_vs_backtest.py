"""실제 계좌가 백테스트가 그리는 분포 안에 있나 (2026-10-15).

백테스트가 일반화된 전략인지 묻는 자리에서 한 번도 안 해 본 것이 이것이다 — 지금까지는
백테스트끼리만 비교했다. 실제로 돌아간 계좌가 그 백테스트가 말하는 범위 안에서 움직였는지
를 처음 잰다.

## 왜 수익률을 맞대 보지 않나

같은 구간의 백테스트 수익률과 라이브 수익률을 1대1로 비교하면 안 된다. 라이브의 스냅샷
일정은 실제 가동 시점에서 시작했고 백테스트 앵커 일곱 곳 중 어느 것과도 안 맞는다. 스냅
간격이 현물 217봉·선물 95봉·주식 69일이라 넉 달 반이면 보유 종목 자체가 다르다. 그래서
"맞나 틀리나" 가 아니라 "그 분포 안에 있나" 를 묻는다.

## 재는 것

  라이브 = 주식 + 코인 현물 + 선물(달러×환율) + 현금, 일별 총액. 자산 스냅샷(assets.db)과
  선물 자산기록(binance_history.db)을 합친다. 입금은 보정한다 — 총액이 하루에 5% 넘게
  뛴 날을 입금으로 보고 그날 수익률을 빼고 이어 붙인다. 실측에서 그런 날은 하나였다
  (2026-08-21→22, +6.25%).

  백테스트 = 같은 합성 60/25/15 를 앵커 일곱 곳에서 돌리고, 라이브 구간과 같은 길이의
  창을 전 기간에 걸쳐 민다. 그 창들의 수익률·낙폭·칼마 분포를 만든다.

  비교 = 라이브 값이 그 분포의 몇 번째 백분위인가.

## 판정 (결과 보기 전에 못박는다)

  - 분포 안(5~95 백분위): 라이브가 백테스트가 말하는 범위 안이다. 일반화를 증명하는 건
    아니지만 어긋남의 증거도 아니다.
  - 아래쪽 꼬리(5 백분위 미만): 실제가 백테스트가 그린 최악보다도 나쁘다는 뜻이고 그대로
    적는다.
  - 위쪽 꼬리(95 백분위 초과): 운이 좋았던 구간이다. 이것도 그대로 적는다 — 좋은 쪽으로
    벗어난 것을 검증 통과로 쓰지 않는다.

표본이 하나다. 넉 달 반짜리 창 하나로 전략을 판정하지 않는다 — 이 측정이 답하는 것은
"지금까지의 실제가 백테스트와 모순되나" 하나뿐이다.

## 안 하는 것

슬리브별 비교는 안 한다. 계좌 사이 송금은 총액을 안 바꾸면서 슬리브 금액을 바꾸는데,
송금 기록이 따로 없어 시장 움직임과 못 가른다. 총액은 외부 입금만 걸러내면 돼서 그쪽만
잰다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research
      python3 bt_live_vs_backtest.py
"""
from __future__ import annotations

import csv
import os
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import bt_coin_param_resweep as R  # noqa: E402

LIVE = "/home/gmoh/mon/251229/state/20261015-livecheck"
DEPOSIT_THR = 0.05       # 하루 총액 변화가 이보다 크면 입금으로 본다
LIVE_START = "2026-05-28"  # 선물 V25 가동일. 그 전은 전략이 다르다.
BT_END = "2026-10-14"


def live_series() -> tuple[pd.Series, list[tuple[str, float]]]:
    """입금 보정한 라이브 총액 곡선과 걸러낸 날 목록."""
    rows = list(csv.DictReader(open(os.path.join(LIVE, "assets_snapshots.csv"))))
    fut = {r["date"]: float(r["pv"])
           for r in csv.DictReader(open(os.path.join(LIVE, "fut_live_pv.csv")))}
    dates, totals = [], []
    for r in rows:
        d = r["date"]
        if d < "2026-03-25":
            continue
        fx = float(r["fx_rate"]) or 1340.0
        if d not in fut:
            continue
        dates.append(pd.Timestamp(d))
        totals.append(float(r["total_krw"]) + fut[d] * fx)

    rets, skipped = [0.0], []
    for i in range(1, len(totals)):
        rr = totals[i] / totals[i - 1] - 1
        if abs(rr) > DEPOSIT_THR:
            skipped.append((str(dates[i].date()), rr))
            rr = 0.0
        rets.append(rr)
    eq = pd.Series((1 + np.array(rets)).cumprod(), index=pd.DatetimeIndex(dates))
    return eq, skipped


def window_metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 10:
        return {}
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    tot = eq.iloc[-1] / eq.iloc[0] - 1
    cagr = (1 + tot) ** (1 / yrs) - 1 if yrs > 0 else 0.0
    mdd = float((eq / eq.cummax() - 1).min())
    return dict(총수익=tot * 100, 연복리=cagr * 100, 최대낙폭=mdd * 100,
                칼마=cagr / abs(mdd) if mdd < 0 else float("nan"))


def pct_rank(value: float, pool: list[float], higher_better: bool) -> float:
    arr = np.array([v for v in pool if v == v])
    if len(arr) == 0:
        return float("nan")
    p = float((arr < value).mean() * 100)
    return p if higher_better else 100 - p


def main() -> int:
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal

    eq_live, skipped = live_series()
    eq_live = eq_live.loc[pd.Timestamp(LIVE_START):]
    eq_live = eq_live / eq_live.iloc[0]
    print(f"라이브 {eq_live.index[0].date()} ~ {eq_live.index[-1].date()} "
          f"· 달력일 {len(eq_live)}일")
    if skipped:
        for d, rr in skipped:
            print(f"  입금으로 보고 뺀 날 {d} ({rr:+.2%})")
    else:
        print("  입금으로 걸러낸 날 없음")

    print("\n코인 일봉 로드 중...", flush=True)
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    R.END = BT_END
    print(f"  심볼 {len(R._bars)}종 · 백테스트 끝 {R.END}", flush=True)

    # 두 벌을 같이 센다 — 시점 고친 엔진이 실제를 재는 정본이고, 현행 엔진은
    # 지금까지 숫자를 내던 쪽이라 둘 다 있어야 "부풀었나" 와 "모순인가" 가 갈린다.
    engines = {"수정 엔진": ("prev", "1"), "현행 엔진": ("same", "0")}
    comp_by: dict[str, list[pd.Series]] = {k: [] for k in engines}
    for a in R.ANCHORS:
        fu = R.run_fut({}, R.PH_FUT[a], False)
        for label, (pv, lag) in engines.items():
            os.environ["SPOT_PV_DATE"] = pv
            os.environ["STOCK_SIG_LAG"] = lag
            try:
                sp = R.run_spot({}, R.PH_SPOT[a], False)
                st = R.run_stock(a, False)
            finally:
                os.environ["SPOT_PV_DATE"] = "same"
                os.environ["STOCK_SIG_LAG"] = "0"
            comp_by[label].append(R.build_alloc(st, sp, fu))
        print(f"  앵커 {a} 완료", flush=True)

    g = float((comp_by["수정 엔진"][0] - comp_by["현행 엔진"][0]).abs().max())
    print(f"\n사전 확인 — 두 엔진 합성 곡선 최대 차이 {g:.6g}")
    if g <= 1e-9:
        print("  멈춘다 — 토글이 안 걸려 두 벌이 같은 것을 재고 있다.")
        return 1

    comps = comp_by["수정 엔진"]
    # 라이브 곡선을 백테스트 달력(거래일)으로 맞춘다.
    cal = comps[0].index
    eq_live = eq_live.reindex(cal.intersection(eq_live.index)).dropna()
    eq_live = eq_live / eq_live.iloc[0]
    n = len(eq_live)
    print(f"\n거래일로 맞춘 라이브 창 길이 {n}일 "
          f"({eq_live.index[0].date()} ~ {eq_live.index[-1].date()})")

    m_live = window_metrics(eq_live)
    print("\n라이브 실적")
    print(f"  총수익 {m_live['총수익']:+.2f}%  연복리 {m_live['연복리']:+.2f}%  "
          f"최대낙폭 {m_live['최대낙폭']:.2f}%  칼마 {m_live['칼마']:.3f}")

    verdict = []
    for label in ("수정 엔진", "현행 엔진"):
        pools: dict[str, list[float]] = {k: [] for k in m_live}
        for eq in comp_by[label]:
            for i in range(0, len(eq) - n, 5):
                m = window_metrics(eq.iloc[i:i + n])
                if not m:
                    continue
                for k, v in m.items():
                    pools[k].append(v)
        print(f"\n[{label}] 같은 길이 창 {len(pools['총수익'])}개 "
              f"(앵커 {len(comp_by[label])}곳 × 5일 간격)")
        print("  지표      라이브      백테스트 5%     중앙      95%    라이브 백분위")
        for k in ("총수익", "연복리", "최대낙폭", "칼마"):
            arr = np.array([v for v in pools[k] if v == v])
            p = pct_rank(m_live[k], list(arr), True)
            print(f"  {k:8s} {m_live[k]:+9.2f}  {np.percentile(arr,5):+9.2f}  "
                  f"{np.percentile(arr,50):+9.2f}  {np.percentile(arr,95):+9.2f}  {p:8.1f}")
            if label == "수정 엔진":
                verdict.append((k, p))

    print("\n판정은 수정 엔진 기준이다 — 그쪽이 실제를 재는 정본이다.")
    low = [k for k, p in verdict if p < 5]
    high = [k for k, p in verdict if p > 95]
    if low:
        print(f"  아래쪽 꼬리 — {', '.join(low)} 가 백테스트 5 백분위 미만이다.")
        print("  실제가 백테스트가 그린 범위보다 나쁘다는 뜻이다.")
    if high:
        print(f"  위쪽 꼬리 — {', '.join(high)} 가 95 백분위를 넘는다. 운이 좋은 구간이고")
        print("  이것을 검증 통과로 쓰지 않는다.")
    if not low and not high:
        print("  네 지표 전부 분포 안(5~95 백분위). 라이브가 백테스트와 모순되지 않는다.")
    print("\n  표본은 창 하나다. 이 측정은 모순 여부만 답하고 일반화를 증명하지 않는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
