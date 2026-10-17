"""정합을 맞춘 엔진에서 네 조합을 다시 잰다 (2026-10-18 운영자 지시).

앞선 측정은 엔진 둘이 라이브와 어긋난 채로 돌았다. 그 둘을 고친 뒤 같은 표를
다시 만든다 — 고친 것은 이 두 가지다.

  ① 선물 증액 증거금 — backtest_futures_v25.py 가 보유분 증액에 드는 증거금을
     진입 시점 레버리지로 나누던 것을 그 시점 목표 레버리지로 바꿨다. 라이브는
     레버리지를 목표값으로 먼저 바꾼 뒤 목표 명목금액으로 주문한다.
  ② 보유분 증거금 재표시 — 같은 이유의 나머지 절반. 레버리지가 바뀌면 그
     포지션이 묶고 있던 개시증거금이 풀리거나 더 묶인다. 지갑 총액은 그대로고
     살 수 있는 현금만 움직인다.
  ③ 현물 주문 크기 평가액 — unified_backtest.py 가 체결봉(t) 종가로 크기를
     정하던 것을 신호봉(t-1) 종가로 바꿨다. 체결가는 t 봉 시가다.
  ④ ②의 자리 — 재표시를 매도보다 먼저 돌리면 현금이 모자라 덜 묶인 상태에서
     매도가 또 비례 차감해 매도 뒤 증거금이 라이브보다 적게 남는다. 자리를
     매도와 매수 사이로 옮겼다(토론 run-20261017T204756Z 반례, 검사
     trade/test_fut_margin_order.py).
  ⑤ 선물 드리프트 리필 후보 풀 — BT 가 상위 선발 3개(변동성까지 걸러 고른 것)를
     심볼순으로 세워 쓰던 것을, 라이브 apply_refill_v2_fut 와 같이 그날 유니버스
     전체에서 두 모멘텀 양수인 코인을 단기 모멘텀 내림차순으로 세우도록 바꿨다
     (토론 run-20261017T210537Z, 검사 trade/test_fut_refill_pool.py).
     그래서 곡선 폴더를 aligned3 로 다시 뗀다.

조합은 전과 같다.
  A 둘 다 켬 / B 둘 다 끔(지금 라이브) / C 선물만 켬 / D 현물만 켬

곡선 캐시는 새 폴더에 쓴다 — 옛 폴더의 곡선은 고치기 전 엔진이 만든 것이라
섞으면 안 된다. 호출 인자는 앞선 측정기에서 글자 그대로 옮겼다(history #74).

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 flip_pfd_aligned.py
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

OUT = "/home/gmoh/mon/251229/state/20261018-aligned3"
EQDIR = os.path.join(OUT, "eq")
STOCK_CACHE = "/home/gmoh/mon/251229/state/20260922-resweep/eq"

ARMS = {
    "A 둘 다 켬": (5, 5),
    "B 둘 다 끔(지금 라이브)": (0, 0),
    "C 선물만 켬": (0, 5),
    "D 현물만 켬": (5, 0),
}


def _p(kind: str, key: str) -> str:
    return os.path.join(EQDIR, f"{kind}__{key}.csv")


def _load(kind: str, key: str):
    p = _p(kind, key)
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def _save(kind: str, key: str, eq: pd.Series) -> None:
    eq.to_csv(_p(kind, key), header=["eq"])


def run_spot(pfd: int, phase: int, stress: bool) -> pd.Series:
    """spot_flip_retry_measure.run_spot 의 인자를 그대로 옮겼다 (pfd 만 바뀜)."""
    from unified_backtest import run as bt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    os.environ["FLIP_ALL_CASH_RETRY"] = "off"
    kw = dict(R.SPOT_BASE)
    m = bt_run(R._bars, R._funding, interval="D", asset_type="spot", leverage=1.0,
               tx_cost=R.TX_SPOT * (R.STRESS if stress else 1.0),
               start_date=R.START, end_date=R.END,
               vol_mode="daily", phase_offset_bars=phase,
               post_flip_delay=pfd, selection="greedy",
               stop_kind="none", stop_pct=0.0,
               dd_lookback=60, dd_threshold=-99.0,
               bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
               health_mode="mom2vol", **kw)
    return m.get("_equity")


def run_fut(pfd: int, phase: int, stress: bool) -> pd.Series:
    """fut_flip_retry_measure.run_fut 의 인자를 그대로 옮겼다 (pfd 만 바뀜)."""
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
    os.makedirs(EQDIR, exist_ok=True)

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...", flush=True)
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    curves = {}
    for a in R.ANCHORS:
        for stress in (False, True):
            tag = f"a{a}{'__s5' if stress else ''}"
            for pfd in (0, 5):
                k = f"pfd{pfd}__{tag}"
                s = _load("sp", k)
                if s is None:
                    s = run_spot(pfd, a, stress)
                    _save("sp", k, s)
                curves[("sp", pfd, a, stress)] = s
                f = _load("fu", k)
                if f is None:
                    f = run_fut(pfd, a, stress)
                    _save("fu", k, f)
                curves[("fu", pfd, a, stress)] = f
        print(f"  앵커 {a} 완료 ({time.time()-t0:.0f}초)", flush=True)

    res = {}
    for stress in (False, True):
        rank = {k: [] for k in ARMS}
        met = {k: [] for k in ARMS}
        for a in R.ANCHORS:
            st = pd.read_csv(os.path.join(STOCK_CACHE, f"st__a{a}.csv"),
                             index_col=0, parse_dates=True)["eq"]
            comps = {}
            for name, (sp_pfd, fu_pfd) in ARMS.items():
                comps[name] = R.build_alloc(st, curves[("sp", sp_pfd, a, stress)],
                                            curves[("fu", fu_pfd, a, stress)])
                met[name].append(R.metrics(comps[name]))
            sums, _, nw = R.window_rs(comps)
            for name in ARMS:
                rank[name].append(sums[name] / max(nw, 1))

        tag = "비용 5배" if stress else "기본 비용"
        print(f"\n=== 정합 수정 후 · 합성 60/25/15 · {tag} · 앵커 {len(R.ANCHORS)}곳 ===")
        print(f"{'조합':<26}{'순위평균':>9}{'칼마':>7}{'샤프':>7}{'연복리':>8}{'최대낙폭':>9}")
        for name in sorted(ARMS, key=lambda k: float(np.mean(rank[k]))):
            print(f"{name:<26}{float(np.mean(rank[name])):>9.3f}"
                  f"{float(np.mean([m['Cal'] for m in met[name]])):>7.2f}"
                  f"{float(np.mean([m['Sharpe'] for m in met[name]])):>7.2f}"
                  f"{float(np.mean([m['CAGR'] for m in met[name]])):>8.2f}"
                  f"{float(np.mean([m['MDD'] for m in met[name]])):>9.2f}")
        base = "B 둘 다 끔(지금 라이브)"
        for nm in ("A 둘 다 켬", "C 선물만 켬", "D 현물만 켬"):
            print(f"  B 대비 {nm} 이 이긴 앵커: "
                  f"{sum(1 for x, y in zip(rank[nm], rank[base]) if x < y)}/{len(R.ANCHORS)}")
        res[tag] = dict(rank=rank, metrics=met)

    with open(os.path.join(OUT, "aligned_result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {OUT}/aligned_result.json  ({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
