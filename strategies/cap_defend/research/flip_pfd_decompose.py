"""현물 쪽 차이를 분해한다 — 잠김 해소 몫과 정상 전환 재선정 몫.

현물에 전환 후 5봉 재평가를 넣으면 성과가 오른다. 그 몫이 어디서 오는지 가른다.

  없음       pfd=0            지금 라이브
  현금만     pfd=5 + 전환일 전부 현금이었던 전환에만
  전체       pfd=5 모든 전환   채택 백테스트

없음 → 현금만 = 전환일에 현금으로 잠겼던 걸 푸는 몫
현금만 → 전체 = 정상 진입한 전환에서도 5봉 뒤 한 번 더 갈아타는 몫

선물 팔은 채택 백테스트 그대로(pfd=5) 고정이다. 매매량(거래 수·리밸 수)도 같이 적는다.

엔진에 PFD_SCOPE 스위치가 필요하다(research/flip_pfd_scope.engine.patch). 기본값은
기존 동작이고, 측정이 끝나면 엔진을 되돌린다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 flip_pfd_decompose.py
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

OUT = "/home/gmoh/mon/251229/state/20261018-fliplock"
EQDIR = os.path.join(OUT, "eq")
STOCK_CACHE = "/home/gmoh/mon/251229/state/20260922-resweep/eq"

NONE, CASHONLY, ALL = "없음(지금 라이브)", "현금 전환만", "전체 전환"
# 이름 → (pfd 봉수, PFD_SCOPE, 캐시 접두 — None 이면 새로 돌린다)
SPOT_ARMS = {
    NONE: (0, "all", "sp__라이브현행"),
    CASHONLY: (5, "cashonly", None),
    ALL: (5, "all", "sp__현행"),
}


def run_spot(pfd: int, scope: str, phase: int, stress: bool) -> dict:
    from unified_backtest import run as sbt_run
    old = os.environ.get("PFD_SCOPE")
    os.environ["PFD_SCOPE"] = scope
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    try:
        # 인자는 bt_coin_param_resweep.run_spot 과 글자까지 같게 둔다 — 캐시된
        # 비교군이 그걸로 만들어졌다. 2026-10-18 첫 판에서 가드 끄는 인자
        # (stop/dd/bl/crash)와 leverage·selection 을 빠뜨려 이 팔만 가드가 켜진 채
        # 돌았고, 그래서 분해 결과가 무효였다. post_flip_delay 만 바꾼다.
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
            os.environ.pop("PFD_SCOPE", None)
        else:
            os.environ["PFD_SCOPE"] = old


def main() -> None:
    t0 = time.time()
    from unified_backtest import load_data
    print("코인 일봉 로드 중...", flush=True)
    R._bars, R._funding = load_data("D")
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    eq, counts = {}, {}
    for a in R.ANCHORS:
        for stress in (False, True):
            for name, (pfd, scope, cache) in SPOT_ARMS.items():
                key = f"a{a}{'__s5' if stress else ''}"
                p = os.path.join(EQDIR, f"sp__cashonly__{key}.csv")
                if cache is not None:
                    eq[(name, a, stress)] = pd.read_csv(
                        os.path.join(EQDIR, f"{cache}__{key}.csv"),
                        index_col=0, parse_dates=True)["eq"]
                    continue
                if os.path.exists(p):
                    eq[(name, a, stress)] = pd.read_csv(
                        p, index_col=0, parse_dates=True)["eq"]
                    continue
                m = run_spot(pfd, scope, R.PH_SPOT[a], stress)
                eq[(name, a, stress)] = m["_equity"]
                m["_equity"].to_csv(p, header=["eq"])
                print(f"  현금만 a{a}{' 5배' if stress else ''} "
                      f"거래 {m['Trades']} 리밸 {m['Rebal']} ({time.time()-t0:.0f}초)",
                      flush=True)

    # 매매량 비교는 앵커 0 · 기본 비용 한 칸에서만 본다(진단).
    print("\n매매량(앵커 0 · 기본 비용, 현물 슬리브 단독)", flush=True)
    for name, (pfd, scope, _c) in SPOT_ARMS.items():
        m = run_spot(pfd, scope, R.PH_SPOT[0], False)
        counts[name] = dict(Trades=m["Trades"], Rebal=m["Rebal"], Cal=m["Cal"])
        print(f"  {name:<14} 거래 {m['Trades']:>5}  리밸 {m['Rebal']:>5}  "
              f"슬리브칼마 {m['Cal']:>6.2f}", flush=True)

    res = {"counts": counts}
    for stress in (False, True):
        rank = {k: [] for k in SPOT_ARMS}
        met = {k: [] for k in SPOT_ARMS}
        for a in R.ANCHORS:
            st = pd.read_csv(os.path.join(STOCK_CACHE, f"st__a{a}.csv"),
                             index_col=0, parse_dates=True)["eq"]
            fu = pd.read_csv(
                os.path.join(EQDIR, f"fu__a{a}{'__s5' if stress else ''}.csv"),
                index_col=0, parse_dates=True)["eq"]
            comps = {n: R.build_alloc(st, eq[(n, a, stress)], fu) for n in SPOT_ARMS}
            for n in SPOT_ARMS:
                met[n].append(R.metrics(comps[n]))
            sums, _, nw = R.window_rs(comps)
            for n in SPOT_ARMS:
                rank[n].append(sums[n] / max(nw, 1))

        tag = "비용 5배" if stress else "기본 비용"
        print(f"\n=== 현물 재평가 범위 분해 · 합성 60/25/15 · {tag} · 앵커 7곳 ===")
        print(f"{'현물 재평가':<16}{'순위평균':>9}{'칼마':>7}{'샤프':>7}{'연복리':>8}{'최대낙폭':>9}")
        for n in (NONE, CASHONLY, ALL):
            print(f"{n:<16}{float(np.mean(rank[n])):>9.3f}"
                  f"{float(np.mean([m['Cal'] for m in met[n]])):>7.2f}"
                  f"{float(np.mean([m['Sharpe'] for m in met[n]])):>7.2f}"
                  f"{float(np.mean([m['CAGR'] for m in met[n]])):>8.2f}"
                  f"{float(np.mean([m['MDD'] for m in met[n]])):>9.2f}")
        g1 = float(np.mean([m["CAGR"] for m in met[CASHONLY]])) - \
            float(np.mean([m["CAGR"] for m in met[NONE]]))
        g2 = float(np.mean([m["CAGR"] for m in met[ALL]])) - \
            float(np.mean([m["CAGR"] for m in met[CASHONLY]]))
        print(f"  연복리 분해: 잠김 해소 {g1:+.2f}%p, 정상 전환 재선정 {g2:+.2f}%p")
        res[tag] = dict(rank=rank, metrics=met, gap_lock=g1, gap_resel=g2)

    with open(os.path.join(OUT, "decompose_result.json"), "w") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    print(f"\n저장: {OUT}/decompose_result.json  ({time.time()-t0:.0f}초)")


if __name__ == "__main__":
    main()
