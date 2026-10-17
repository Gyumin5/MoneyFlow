"""선물 쪽 어긋남의 값 — 전환 후 5봉 재평가를 뺐을 때 얼마나 나빠지나.

현물에서 쟀던 "라이브현행" 진단을 선물에 그대로 건다. 후보 측정이 아니다 — 채택
백테스트(backtest_futures_v25.py)에는 post_flip_delay=5 가 있고 라이브
(trade/auto_trade_binance.py)에는 없으니, 그 차이가 선물에서 얼마인지만 본다.

현물·주식 팔은 고정이고 선물만 바꾼다. 현물 곡선은 spot_flip_retry_measure.py 가
남긴 현행 캐시를 그대로 읽는다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 fut_flip_retry_measure.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402

OUT = "/home/gmoh/mon/251229/state/20261018-fliplock"
EQDIR = os.path.join(OUT, "eq")
STOCK_CACHE = "/home/gmoh/mon/251229/state/20260922-resweep/eq"

CUR = "현행"
LIVE = "라이브현행"


def _load(kind: str, key: str):
    p = os.path.join(EQDIR, f"{kind}__{key}.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def run_fut(pfd: int, phase: int, stress: bool) -> pd.Series:
    from backtest_futures_v25 import run as fbt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    m = fbt_run(R._bars, R._funding, interval="D", leverage=R._k2,
                tx_cost=R.TX_FUT * (R.STRESS if stress else 1.0), maint_rate=0.004,
                vol_days=90, vol_mode="daily", post_flip_delay=pfd,
                phase_offset_bars=phase, health_mode="mom2vol",
                start_date=R.START, end_date=R.END, **R.FUT_BASE)
    return m.get("_equity")


def main() -> None:
    t0 = time.time()
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...", flush=True)
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    stock_eq, spot_eq, fut_eq = {}, {}, {}
    for a in R.ANCHORS:
        stock_eq[a] = pd.read_csv(os.path.join(STOCK_CACHE, f"st__a{a}.csv"),
                                  index_col=0, parse_dates=True)["eq"]
        for st in (False, True):
            k = f"현행__a{a}{'__s5' if st else ''}"
            s = _load("sp", k)
            if s is None:
                raise SystemExit(f"현물 현행 곡선이 없다: {k} — spot_flip_retry_measure.py 먼저")
            spot_eq[(a, st)] = s
            fut_eq[(CUR, a, st)] = _load("fu", f"a{a}{'__s5' if st else ''}")
            kk = f"fut_pfd0__a{a}{'__s5' if st else ''}"
            s2 = _load("fu", kk)
            if s2 is None:
                s2 = run_fut(0, R.PH_FUT[a], st)
                s2.to_csv(os.path.join(EQDIR, f"fu__{kk}.csv"), header=["eq"])
                print(f"  선물 pfd0 a{a}{' 5배' if st else ''} ({time.time()-t0:.0f}초)",
                      flush=True)
            fut_eq[(LIVE, a, st)] = s2

    res = {}
    for st in (False, True):
        rank = {CUR: [], LIVE: []}
        met = {CUR: [], LIVE: []}
        fcal = {CUR: [], LIVE: []}
        for a in R.ANCHORS:
            comps = {n: R.build_alloc(stock_eq[a], spot_eq[(a, st)], fut_eq[(n, a, st)])
                     for n in (CUR, LIVE)}
            for n in (CUR, LIVE):
                met[n].append(R.metrics(comps[n]))
                fcal[n].append(R.metrics(fut_eq[(n, a, st)])["Cal"])
            sums, _, nw = R.window_rs(comps)
            for n in (CUR, LIVE):
                rank[n].append(sums[n] / max(nw, 1))

        tag = "비용 5배" if st else "기본 비용"
        print(f"\n=== 선물만 바꾼 합성 60/25/15 · {tag} · 앵커 {len(R.ANCHORS)}곳 ===")
        print(f"{'팔':<12}{'순위평균':>9}{'칼마':>7}{'샤프':>7}{'연복리':>8}"
              f"{'최대낙폭':>9}{'선물칼마':>9}")
        for n in (CUR, LIVE):
            print(f"{n:<12}{float(np.mean(rank[n])):>9.3f}"
                  f"{float(np.mean([m['Cal'] for m in met[n]])):>7.2f}"
                  f"{float(np.mean([m['Sharpe'] for m in met[n]])):>7.2f}"
                  f"{float(np.mean([m['CAGR'] for m in met[n]])):>8.2f}"
                  f"{float(np.mean([m['MDD'] for m in met[n]])):>9.2f}"
                  f"{float(np.mean(fcal[n])):>9.2f}")
        d = [x - y for x, y in zip(rank[LIVE], rank[CUR])]
        print("  앵커별 순위 차이(라이브현행 - 현행) " + " ".join(f"{x:+6.2f}" for x in d)
              + f"   라이브가 이긴 앵커 {sum(1 for x in d if x < 0)}/{len(d)}")
        res[tag] = dict(rank=rank, fut_cal=fcal,
                        metrics={n: met[n] for n in (CUR, LIVE)})

    with open(os.path.join(OUT, "fut_result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {OUT}/fut_result.json  ({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
