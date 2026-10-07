"""현행 선정 규칙의 우연 여부 — 본 측정 (2026-10-07).

사전등록은 b0_robust_prereg.md 다. 이 파일은 거기 고정한 설정대로 백테스트를 돌리고
자산곡선을 캐시에 쌓기만 한다. 판정은 b0_robust_report.py 가 한다.

돌리는 것 셋.
  1 기준선과 귀무 대조 — 슬리브마다 B0 + 고정형 위약 19벌 + 갱신형 위약 19벌 +
    단순 대조 둘(top1, fullpool). 41팔 × 앵커 7 × 비용 2.
  2 슬리브별 제외 — 현물 10조합(단일 7 + 집단 3), 선물 9조합(단일 6 + 집단 3).
    조합마다 B0 와 같은 제외를 적용한 위약 5벌(pfix:0,1,2 · pref:0,1).
  3 두 슬리브 동시 제외 — 공통 3집합을 양쪽에 같이 건다.

총 3,248회. 사전등록 계획 4,172회 안이다(실제 대상 수가 상한 12보다 적어 줄었다).
현물 1회 약 4초, 선물 약 11초라 대략 6시간 45분.

캐시가 있으면 건너뛴다 — 중간에 끊겨도 다시 돌리면 이어간다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_b0_robust.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402

OUT = "/home/gmoh/mon/251229/state/20261007-b0robust"
EQDIR = os.path.join(OUT, "eq")

DRIFT_SPOT = 0.10
DRIFT_FUT = 0.015
SP_OVER = dict(drift_threshold=DRIFT_SPOT)
FU_OVER = dict(drift_threshold=DRIFT_FUT)

# 사전등록에서 못박은 목록. 결과를 보고 바꾸지 않는다.
TARGETS = {
    "sp": ["BTC", "ETH", "BNB", "SOL", "XRP", "LINK", "TRX"],
    "fu": ["BTC", "ETH", "BNB", "SOL", "DOGE", "TRX"],
}
GROUPS = {
    "sp": [("상위2", ["BNB", "SOL"]), ("상위3", ["BNB", "SOL", "XRP"]),
           ("BTC+ETH", ["BTC", "ETH"])],
    "fu": [("상위2", ["BNB", "SOL"]), ("상위3", ["BNB", "SOL", "DOGE"]),
           ("BTC+ETH", ["BTC", "ETH"])],
}
COMMON = [("공통상위2", ["BNB", "SOL"]), ("공통상위3", ["BNB", "SOL", "BTC"]),
          ("공통BTC+ETH", ["BTC", "ETH"])]

N_SEED = 19
PLACEBO_5 = ["pfix:0", "pfix:1", "pfix:2", "pref:0", "pref:1"]
BASE_ARMS = (["b0"]
             + [f"pfix:{i}" for i in range(N_SEED)]
             + [f"pref:{i}" for i in range(N_SEED)]
             + ["top1", "fullpool"])


def excl_key(ex) -> str:
    return "none" if not ex else "-".join(sorted(ex))


def key(sleeve: str, arm: str, ex, anchor: int, stress: bool) -> str:
    a = arm.replace(":", "")
    return f"{sleeve}__{a}__{excl_key(ex)}__a{anchor}{'__s5' if stress else ''}"


def load(k: str):
    p = os.path.join(EQDIR, f"{k}.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def ensure(sleeve: str, arm: str, ex, anchor: int, stress: bool):
    k = key(sleeve, arm, ex, anchor, stress)
    eq = load(k)
    if eq is not None:
        return eq, False
    os.environ["B0_ARM"] = "" if arm == "b0" else arm
    os.environ["SELECT_NO_ABSORB"] = "1" if arm == "fullpool" else "0"
    os.environ["B0_EXCLUDE"] = ",".join(ex) if ex else ""
    try:
        if sleeve == "fu":
            eq = R.run_fut(FU_OVER, R.PH_FUT[anchor], stress)
        else:
            eq = R.run_spot(SP_OVER, R.PH_SPOT[anchor], stress)
    finally:
        os.environ["B0_ARM"] = ""
        os.environ["SELECT_NO_ABSORB"] = "0"
        os.environ.pop("B0_EXCLUDE", None)
    eq.to_csv(os.path.join(EQDIR, f"{k}.csv"), header=["eq"])
    return eq, True


def plan():
    """(sleeve, arm, excl) 작업 목록. 중복 조합은 한 번만."""
    jobs = []
    seen = set()

    def add(sleeve, arm, ex):
        t = (sleeve, arm, excl_key(ex))
        if t in seen:
            return
        seen.add(t)
        jobs.append((sleeve, arm, list(ex)))

    # 1 기준선·귀무·단순 대조
    for sleeve in ("sp", "fu"):
        for arm in BASE_ARMS:
            add(sleeve, arm, [])
    # 2 슬리브별 제외
    for sleeve in ("sp", "fu"):
        combos = [[c] for c in TARGETS[sleeve]] + [g for _n, g in GROUPS[sleeve]]
        for ex in combos:
            for arm in ["b0"] + PLACEBO_5:
                add(sleeve, arm, ex)
    # 3 두 슬리브 동시
    for _n, ex in COMMON:
        for sleeve in ("sp", "fu"):
            for arm in ["b0"] + PLACEBO_5:
                add(sleeve, arm, ex)
    return jobs


def main() -> None:
    t0 = time.time()
    os.makedirs(EQDIR, exist_ok=True)
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"데이터 로드 {time.time() - t0:.0f}s · 종목 {len(R._bars)}", flush=True)

    # 주식 슬리브는 축이 아니다. 기존 하니스 캐시를 그대로 쓴다.
    stock_dir = R.EQDIR
    for a in R.ANCHORS:
        p = os.path.join(stock_dir, f"st__a{a}.csv")
        if not os.path.exists(p):
            R.run_stock(a, False).to_csv(p, header=["eq"])
    print(f"주식 슬리브 7벌 확인 {time.time() - t0:.0f}s", flush=True)

    jobs = plan()
    total = len(jobs) * len(R.ANCHORS) * 2
    print(f"계획 {len(jobs)}조합 × 앵커 7 × 비용 2 = {total}회", flush=True)
    done = ran = 0
    for i, (sleeve, arm, ex) in enumerate(jobs):
        for a in R.ANCHORS:
            for st in (False, True):
                _eq, fresh = ensure(sleeve, arm, ex, a, st)
                done += 1
                ran += 1 if fresh else 0
        if (i + 1) % 10 == 0 or i + 1 == len(jobs):
            el = time.time() - t0
            eta = el / max(1, ran) * (total - done) if ran else 0
            print(f"  {i+1}/{len(jobs)}조합 · {done}/{total}회 (새로 {ran}) "
                  f"· {el/60:.0f}분 경과 · 남은 추정 {eta/60:.0f}분", flush=True)

    with open(os.path.join(OUT, "run_done.json"), "w") as f:
        json.dump(dict(jobs=len(jobs), runs=total, fresh=ran,
                       seconds=time.time() - t0), f, ensure_ascii=False, indent=1)
    print(f"\n완료 {total}회 (새로 {ran}) · 총 {(time.time()-t0)/60:.0f}분", flush=True)


if __name__ == "__main__":
    main()
