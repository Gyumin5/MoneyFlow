"""현물 백테스트가 주문 크기를 그날 종가로 정한다 — 그래서 수치가 얼마나 부풀었나.

unified_backtest.py 의 `_execute_rebalance` 는 체결가로 그 봉의 시가를 쓰는데, 주문
크기를 정하는 자산 평가액(`_port_val`)은 같은 봉의 종가로 잰다. 그날 오르는 것을 알고
크기를 정하는 셈이다. 선물 엔진은 이 자리에 이미 `pv_date=prev_date` 가 들어가 있어
전일 종가로 잰다(backtest_futures_v25.py:1088). 현물에만 빠져 있다.

여기서는 현물만 전일 종가 기준으로 바꿔(PV_FILL_BASIS=prev) 다시 돌리고, 전환 후 5봉
재평가를 켠 쪽과 끈 쪽이 각각 얼마나 움직이는지 본다. 매매가 많은 쪽이 앞날 참조의 덕을
더 보므로, 켠 쪽의 우위가 이 참조에서 나온 것인지가 이 측정의 쟁점이다.

선물·주식 곡선은 기존 캐시를 그대로 쓴다(이 결함과 무관).

엔진 스위치가 필요하다(research/pv_fill_basis.engine.patch). 기본값은 기존 동작이고,
측정이 끝나면 엔진을 되돌린다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 pv_fill_basis_measure.py
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402

OUT = "/home/gmoh/mon/251229/state/20261018-pvbasis"
EQDIR = os.path.join(OUT, "eq")
FLIP = "/home/gmoh/mon/251229/state/20261018-fliplock/eq"
STOCK_CACHE = "/home/gmoh/mon/251229/state/20260922-resweep/eq"

# 팔 이름 → (post_flip_delay, 평가 기준)
ARMS = {
    "B 끔 · 종가(현행 수치)": (0, "close"),
    "B 끔 · 전일종가(고침)": (0, "prev"),
    "A 켬 · 종가(현행 수치)": (5, "close"),
    "A 켬 · 전일종가(고침)": (5, "prev"),
}
# 고치기 전 값은 이미 돌려 둔 것을 그대로 쓴다.
CACHED = {
    "B 끔 · 종가(현행 수치)": "sp__라이브현행",
    "A 켬 · 종가(현행 수치)": "sp__현행",
}


def run_spot(pfd: int, basis: str, phase: int, stress: bool) -> dict:
    """인자는 bt_coin_param_resweep.run_spot 과 글자까지 같게 둔다 — 캐시된 비교군이
    그걸로 만들어졌다. post_flip_delay 와 평가 기준만 바꾼다."""
    from unified_backtest import run as sbt_run
    old = os.environ.get("PV_FILL_BASIS")
    os.environ["PV_FILL_BASIS"] = basis
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    try:
        return sbt_run(R._bars, R._funding, interval="D", asset_type="spot",
                       leverage=1.0,
                       tx_cost=R.TX_SPOT * (R.STRESS if stress else 1.0),
                       start_date=R.START, end_date=R.END,
                       vol_mode="daily", phase_offset_bars=phase,
                       post_flip_delay=pfd, selection="greedy",
                       stop_kind="none", stop_pct=0.0,
                       dd_lookback=60, dd_threshold=-99.0,
                       bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
                       health_mode="mom2vol", **R.SPOT_BASE)
    finally:
        if old is None:
            os.environ.pop("PV_FILL_BASIS", None)
        else:
            os.environ["PV_FILL_BASIS"] = old


def main() -> None:
    os.makedirs(EQDIR, exist_ok=True)
    t0 = time.time()
    from unified_backtest import load_data
    print("코인 일봉 로드 중...", flush=True)
    R._bars, R._funding = load_data("D")
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    eq, counts = {}, {}
    for a in R.ANCHORS:
        for stress in (False, True):
            key = f"a{a}{'__s5' if stress else ''}"
            for name, (pfd, basis) in ARMS.items():
                cache = CACHED.get(name)
                if cache is not None:
                    eq[(name, a, stress)] = pd.read_csv(
                        os.path.join(FLIP, f"{cache}__{key}.csv"),
                        index_col=0, parse_dates=True)["eq"]
                    continue
                p = os.path.join(EQDIR, f"sp_pfd{pfd}_{basis}__{key}.csv")
                if os.path.exists(p):
                    eq[(name, a, stress)] = pd.read_csv(
                        p, index_col=0, parse_dates=True)["eq"]
                    continue
                m = run_spot(pfd, basis, R.PH_SPOT[a], stress)
                eq[(name, a, stress)] = m["_equity"]
                m["_equity"].to_csv(p, header=["eq"])
        print(f"  앵커 {a} 완료 ({time.time()-t0:.0f}초)", flush=True)

    print("\n슬리브 단독 칼마(앵커 0 · 기본 비용)", flush=True)
    for name, (pfd, basis) in ARMS.items():
        m = run_spot(pfd, basis, R.PH_SPOT[0], False)
        counts[name] = dict(Trades=m["Trades"], Rebal=m["Rebal"], Cal=m["Cal"],
                            CAGR=m["CAGR"], MDD=m["MDD"])
        print(f"  {name:<22} 거래 {m['Trades']:>5}  리밸 {m['Rebal']:>5}  "
              f"칼마 {m['Cal']:>6.2f}  연복리 {m['CAGR']:>6.2f}", flush=True)

    res = {"counts": counts}
    for stress in (False, True):
        rank = {k: [] for k in ARMS}
        met = {k: [] for k in ARMS}
        for a in R.ANCHORS:
            st = pd.read_csv(os.path.join(STOCK_CACHE, f"st__a{a}.csv"),
                             index_col=0, parse_dates=True)["eq"]
            fu = pd.read_csv(
                os.path.join(FLIP, f"fu__a{a}{'__s5' if stress else ''}.csv"),
                index_col=0, parse_dates=True)["eq"]
            comps = {n: R.build_alloc(st, eq[(n, a, stress)], fu) for n in ARMS}
            for n in ARMS:
                met[n].append(R.metrics(comps[n]))
            sums, _, nw = R.window_rs(comps)
            for n in ARMS:
                rank[n].append(sums[n] / max(nw, 1))

        tag = "비용 5배" if stress else "기본 비용"
        print(f"\n=== 주문크기 평가 기준 · 합성 60/25/15 · {tag} · 앵커 7곳 ===")
        print(f"{'팔':<24}{'순위평균':>9}{'칼마':>7}{'샤프':>7}{'연복리':>8}{'최대낙폭':>9}")
        for n in ARMS:
            print(f"{n:<24}{float(np.mean(rank[n])):>9.3f}"
                  f"{float(np.mean([m['Cal'] for m in met[n]])):>7.2f}"
                  f"{float(np.mean([m['Sharpe'] for m in met[n]])):>7.2f}"
                  f"{float(np.mean([m['CAGR'] for m in met[n]])):>8.2f}"
                  f"{float(np.mean([m['MDD'] for m in met[n]])):>9.2f}")

        def cagr(n):
            return float(np.mean([m["CAGR"] for m in met[n]]))

        b_drop = cagr("B 끔 · 전일종가(고침)") - cagr("B 끔 · 종가(현행 수치)")
        a_drop = cagr("A 켬 · 전일종가(고침)") - cagr("A 켬 · 종가(현행 수치)")
        gap_before = cagr("A 켬 · 종가(현행 수치)") - cagr("B 끔 · 종가(현행 수치)")
        gap_after = cagr("A 켬 · 전일종가(고침)") - cagr("B 끔 · 전일종가(고침)")
        print(f"  앞날 참조를 빼면 연복리: 끔 {b_drop:+.2f}%p, 켬 {a_drop:+.2f}%p")
        print(f"  켬-끔 격차: 고치기 전 {gap_before:+.2f}%p → 고친 뒤 {gap_after:+.2f}%p")
        # 앵커별로 켠 쪽이 끈 쪽을 이기는지(고친 뒤)
        wins = sum(1 for i in range(len(R.ANCHORS))
                   if met["A 켬 · 전일종가(고침)"][i]["Cal"] > met["B 끔 · 전일종가(고침)"][i]["Cal"])
        print(f"  고친 뒤 칼마 기준 켬 우세 앵커: {wins}/{len(R.ANCHORS)}곳")
        res[tag] = dict(rank=rank, metrics=met, b_drop=b_drop, a_drop=a_drop,
                        gap_before=gap_before, gap_after=gap_after, wins=wins)

    with open(os.path.join(OUT, "result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {OUT}/result.json  ({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
