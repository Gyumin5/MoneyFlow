"""전환 후 재평가 지연을 몇 봉으로 둘지 격자로 잰다 (2026-10-18 운영자 물음 "5가 제일좋아?").

지금까지 잰 것은 두 자리뿐이다 — 끔(0)과 다섯 봉(5). 그 사이와 바깥은 한 번도 안 쟀고,
5 라는 값은 채택 백테스트가 원래 들고 있던 숫자다. 앞서 기각된 "다음 봉"(= 1)과 "매 봉"은
라이브 쪽 재시도 장치로 잰 것이라 여기 격자와 같은 축이 아니다 — 여기서 1 을 다시 잰다.

재는 축은 하나다: 두 슬리브에 같은 값을 넣었을 때 합성 60/25/15 가 어떤 값에서 제일 좋나.
값 = 0, 1, 2, 3, 4, 5, 6, 7, 10 (아홉). 선물 스태거가 19봉이라 10 이면 그 절반이고,
그보다 길게 미루면 다음 앵커 자리가 먼저 와서 재평가 자체가 뜻을 잃는다.

판정 규칙은 돌리기 전에 적는다(결과를 보고 고치지 않는다).
  ① 기본 비용과 비용 5배 두 칸 모두에서 창 순위합 평균이 5 보다 낮고,
  ② 5 와 머리를 맞댄 앵커 일곱 곳 중 다섯 곳 이상에서 앞서는
  값이 있을 때만 그 값으로 바꾼다. 둘 중 하나라도 못 채우면 5 를 그대로 둔다.
  ③ 바꾸는 경우에도 그 값의 양 옆(±1)이 같이 좋은 고원인지 본다. 한 점만 솟으면 안 바꾼다.

곡선 캐시는 aligned3 폴더를 그대로 쓴다 — 0 과 5 는 이미 거기 있고, 같은 엔진·같은
호출 인자다. 호출 인자는 flip_pfd_aligned3.py 에서 글자 그대로 옮겼다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 flip_pfd_grid.py
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

EQDIR = "/home/gmoh/mon/251229/state/20261018-aligned3/eq"
OUT = "/home/gmoh/mon/251229/state/20261018-pfdgrid"
STOCK_CACHE = "/home/gmoh/mon/251229/state/20260922-resweep/eq"

PFDS = [0, 1, 2, 3, 4, 5, 6, 7, 10]
CURRENT = 5


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
    """flip_pfd_aligned3.run_spot 의 인자를 그대로 옮겼다 (pfd 만 바뀜)."""
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
    """flip_pfd_aligned3.run_fut 의 인자를 그대로 옮겼다 (pfd 만 바뀜)."""
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
    os.makedirs(OUT, exist_ok=True)
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
            for pfd in PFDS:
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
        tag = "비용 5배" if stress else "기본 비용"
        # (1) 두 슬리브 같은 값
        rank = {p: [] for p in PFDS}
        met = {p: [] for p in PFDS}
        # (2) 슬리브별 한 축만 흔들기 (상대 슬리브는 5 고정)
        rank_sp = {p: [] for p in PFDS}
        rank_fu = {p: [] for p in PFDS}

        for a in R.ANCHORS:
            st = pd.read_csv(os.path.join(STOCK_CACHE, f"st__a{a}.csv"),
                             index_col=0, parse_dates=True)["eq"]

            comps = {p: R.build_alloc(st, curves[("sp", p, a, stress)],
                                      curves[("fu", p, a, stress)]) for p in PFDS}
            for p in PFDS:
                met[p].append(R.metrics(comps[p]))
            sums, _, nw = R.window_rs({str(p): c for p, c in comps.items()})
            for p in PFDS:
                rank[p].append(sums[str(p)] / max(nw, 1))

            c_sp = {str(p): R.build_alloc(st, curves[("sp", p, a, stress)],
                                          curves[("fu", CURRENT, a, stress)]) for p in PFDS}
            s2, _, n2 = R.window_rs(c_sp)
            for p in PFDS:
                rank_sp[p].append(s2[str(p)] / max(n2, 1))

            c_fu = {str(p): R.build_alloc(st, curves[("sp", CURRENT, a, stress)],
                                          curves[("fu", p, a, stress)]) for p in PFDS}
            s3, _, n3 = R.window_rs(c_fu)
            for p in PFDS:
                rank_fu[p].append(s3[str(p)] / max(n3, 1))

        print(f"\n=== 전환 후 재평가 지연 격자 · 합성 60/25/15 · {tag} · 앵커 {len(R.ANCHORS)}곳 ===")
        print(f"{'지연(봉)':<10}{'순위평균':>9}{'칼마':>7}{'샤프':>7}{'연복리':>8}{'최대낙폭':>9}{'5 대비 승':>10}")
        for p in sorted(PFDS, key=lambda k: float(np.mean(rank[k]))):
            wins = sum(1 for x, y in zip(rank[p], rank[CURRENT]) if x < y)
            label = f"{p}" + (" (현행)" if p == CURRENT else "")
            print(f"{label:<10}{float(np.mean(rank[p])):>9.3f}"
                  f"{float(np.mean([m['Cal'] for m in met[p]])):>7.2f}"
                  f"{float(np.mean([m['Sharpe'] for m in met[p]])):>7.2f}"
                  f"{float(np.mean([m['CAGR'] for m in met[p]])):>8.2f}"
                  f"{float(np.mean([m['MDD'] for m in met[p]])):>9.2f}"
                  f"{wins:>7}/{len(R.ANCHORS)}")

        print(f"  현물만 흔들기(선물 {CURRENT} 고정) 순위평균: "
              + "  ".join(f"{p}={float(np.mean(rank_sp[p])):.2f}" for p in PFDS))
        print(f"  선물만 흔들기(현물 {CURRENT} 고정) 순위평균: "
              + "  ".join(f"{p}={float(np.mean(rank_fu[p])):.2f}" for p in PFDS))

        res[tag] = dict(rank={str(k): v for k, v in rank.items()},
                        metrics={str(k): v for k, v in met.items()},
                        rank_spot_only={str(k): v for k, v in rank_sp.items()},
                        rank_fut_only={str(k): v for k, v in rank_fu.items()})

    with open(os.path.join(OUT, "grid_result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {OUT}/grid_result.json  ({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
