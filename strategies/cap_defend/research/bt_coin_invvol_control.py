"""변동성 가중의 이득이 배분 때문인가 노출 축소 때문인가 (2026-09-25).

계기: ai-debate run-20260924T154821Z. 종목 상한이 1/3 이라 변동성 역수 배분은 변동성이
낮은 종목의 몫을 상한에서 잘라 현금으로 남긴다. 실측(bt_spot_exposure.py)으로 확인했다 —
2종 이상 보유한 784일의 평균 투자비율이 등분 0.647, 변동성 역수(창 20) 0.593 이다.
즉 관측된 낙폭 개선에 배분 효과와 노출 축소 효과가 섞여 있다.

여기서 그 둘을 가른다. 같은 엔진·같은 날에 세 가지를 비교한다.
  invvol   변동성 역수 배분 (노출도 같이 준다)
  ewmatch  같은 날 invvol 과 투자비율만 똑같이 맞추고 종목 사이는 등분 (배분 효과만 제거)
  ewfixed  등분에 고정 배율만 곱한다 (배분은 그대로, 노출만 줄이는 단순 대조군)

배율 0.917 은 창 20 의 상대 노출(0.593/0.647)을 맞춘 값이다.

판정 기준 (결과 보기 전 고정):
  1. invvol 이 같은 창의 ewmatch 를 두 비용 조건 모두에서 이겨야 배분 효과가 실재한다.
     못 이기면 이 축은 기각한다 — 이득의 정체가 노출 축소이기 때문이다.
  2. invvol 이 ewfixed 도 이겨야 한다. ewfixed 가 비슷하거나 더 좋으면, 할 일은
     비중 규칙을 바꾸는 것이 아니라 노출을 줄이는 것이고 그 축은 이미 기각돼 있다
     (변동성 목표 2026-08-31).
  3. 두 비용 조건에서 순서가 뒤집히면 채택하지 않는다.
  4. 결정층은 합성 60/25/15 윈도우 순위합, 앵커 7곳.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_coin_invvol_control.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

from bt_fut_spike_hold import (ANCHORS, PH_SPOT, PH_FUT, START, END, SPOT_BASE,  # noqa: E402
                               FUT_BASE, TX_SPOT, TX_FUT, STRESS, CUR,
                               build_alloc, metrics, window_rs, run_stock)

OUT = "/home/gmoh/mon/251229/state/20260925-invvol-control"
EQDIR = os.path.join(OUT, "eq")

_bars = None
_funding = None
_k2 = None

# (이름, 현물 모드, 창, 고정배율)
ARMS = [
    (CUR, 'ew', 90, 1.0),
    ("역수 10", 'invvol', 10, 1.0),
    ("역수 20", 'invvol', 20, 1.0),
    ("역수 30", 'invvol', 30, 1.0),
    ("노출맞춤 10", 'ewmatch', 10, 1.0),
    ("노출맞춤 20", 'ewmatch', 20, 1.0),
    ("노출맞춤 30", 'ewmatch', 30, 1.0),
    ("등분×0.917", 'ewfixed', 90, 0.917),
]


def _key(name: str, anchor: int, stress: bool) -> str:
    safe = name.replace("/", "_").replace(" ", "_").replace(".", "p").replace("×", "x")
    return f"{safe}__a{anchor}{'__s5' if stress else ''}"


def _save(kind: str, key: str, eq: pd.Series) -> None:
    eq.to_csv(os.path.join(EQDIR, f"{kind}__{key}.csv"), header=["eq"])


def _load(kind: str, key: str):
    p = os.path.join(EQDIR, f"{kind}__{key}.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def run_spot(mode: str, days: int, scale: float, phase: int, stress: bool) -> pd.Series:
    from unified_backtest import run as bt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    os.environ["CD_WEIGHT_MODE_SPOT"] = mode
    os.environ["CD_WEIGHT_VOL_DAYS"] = str(days)
    os.environ["CD_WEIGHT_FIXED_SCALE"] = str(scale)
    try:
        m = bt_run(_bars, _funding, interval="D", asset_type="spot", leverage=1.0,
                   tx_cost=TX_SPOT * (STRESS if stress else 1.0),
                   start_date=START, end_date=END,
                   vol_mode="daily", phase_offset_bars=phase,
                   post_flip_delay=5, selection="greedy",
                   stop_kind="none", stop_pct=0.0,
                   dd_lookback=60, dd_threshold=-99.0,
                   bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
                   health_mode="mom2vol", **SPOT_BASE)
    finally:
        os.environ["CD_WEIGHT_MODE_SPOT"] = 'ew'
        os.environ["CD_WEIGHT_FIXED_SCALE"] = '1.0'
    return m.get("_equity")


def run_fut(phase: int, stress: bool) -> pd.Series:
    from backtest_futures_v25 import run as fbt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    os.environ["CD_WEIGHT_MODE_FUT"] = 'ew'
    m = fbt_run(_bars, _funding, interval="D", leverage=_k2,
                tx_cost=TX_FUT * (STRESS if stress else 1.0), maint_rate=0.004,
                vol_days=90, vol_mode="daily", post_flip_delay=5,
                phase_offset_bars=phase, health_mode="mom2vol",
                start_date=START, end_date=END, **FUT_BASE)
    return m.get("_equity")


def main() -> None:
    global _bars, _funding, _k2
    os.makedirs(EQDIR, exist_ok=True)
    t0 = time.time()

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...", flush=True)
    _bars, _funding = load_data("D")
    _k2 = build_K2_signal(_bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                          btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                          l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(_bars)}종 ({time.time()-t0:.0f}초)", flush=True)
    print(f"\n기간 {START} ~ {END} | 묻는 것: 이득이 배분 때문인가 노출 축소 때문인가")
    print(f"팔 {len(ARMS)}개 · 앵커 {len(ANCHORS)}곳 · 비용 2조건\n", flush=True)

    print("주식 슬리브(합성용 고정 팔) 계산 중...", flush=True)
    stock_eq = {}
    for a in ANCHORS:
        k = _key("주식", a, False)
        eq = _load("st", k)
        if eq is None:
            eq = run_stock(a)
            _save("st", k, eq)
        stock_eq[a] = eq
    print(f"  완료 ({time.time()-t0:.0f}초)", flush=True)

    fut_eq = {}
    for stress in (False, True):
        for a in ANCHORS:
            k = _key("선물", a, stress)
            eq = _load("fu", k)
            if eq is None:
                eq = run_fut(PH_FUT[a], stress)
                _save("fu", k, eq)
                print(f"  fu a{a}{' s5' if stress else ''} ({time.time()-t0:.0f}초)", flush=True)
            fut_eq[(a, stress)] = eq

    total = len(ARMS) * len(ANCHORS) * 2
    done = 0
    for stress in (False, True):
        for a in ANCHORS:
            for name, sm, days, scale in ARMS:
                k = _key(name, a, stress)
                done += 1
                if _load("sp", k) is None:
                    t1 = time.time()
                    _save("sp", k, run_spot(sm, days, scale, PH_SPOT[a], stress))
                    print(f"  [{done}/{total}] sp {name} a{a}"
                          f"{' s5' if stress else ''} {time.time()-t1:.0f}초", flush=True)

    print("\n[현물 슬리브 단독 — 앵커 0, 기본 비용]")
    print(f"  {'팔':<14s}{'칼마':>8s}{'연수익':>10s}{'낙폭':>9s}")
    for name, _sm, _d, _sc in ARMS:
        sp = _load("sp", _key(name, 0, False))
        if sp is None:
            continue
        m = metrics(sp)
        print(f"  {name:<14s}{m['Cal']:>8.2f}{m['CAGR']:>+9.1f}%{m['MDD']:>+8.1f}%")

    for stress in (False, True):
        names = [n for n, _, _, _ in ARMS]
        per_rank = {n: [] for n in names}
        per_cal = {n: [] for n in names}
        per_mdd = {n: [] for n in names}
        for a in ANCHORS:
            allocs = {}
            for name in names:
                sp = _load("sp", _key(name, a, stress))
                if sp is None:
                    continue
                al = build_alloc(stock_eq[a], sp, fut_eq[(a, stress)])
                allocs[name] = al
                mm = metrics(al)
                per_cal[name].append(mm["Cal"])
                per_mdd[name].append(mm["MDD"])
            if len(allocs) == len(names):
                sums, _w, n = window_rs(allocs)
                for name in names:
                    per_rank[name].append(sums[name] / max(n, 1))
        tag = "비용 5배" if stress else "기본 비용"
        print(f"\n=== 합성 60/25/15 · {tag} · 앵커 {len(ANCHORS)}곳 ===")
        print(f"  {'팔':<14s}{'평균순위':>9s}{'중앙':>8s}{'최악':>8s}{'현행대비':>10s}"
              f"{'칼마평균':>10s}{'낙폭평균':>10s}{'승앵커':>8s}")
        base = per_rank.get(CUR, [])
        for name in sorted(names, key=lambda x: float(np.mean(per_rank[x])) if per_rank[x] else 9):
            r = per_rank[name]
            if not r:
                print(f"  {name:<14s}  (자료 없음)")
                continue
            wins = sum(1 for i in range(len(r)) if base and r[i] < base[i])
            gap = float(np.mean(r)) - (float(np.mean(base)) if base else 0.0)
            print(f"  {name:<14s}{float(np.mean(r)):>9.3f}{float(np.median(r)):>8.3f}"
                  f"{float(np.max(r)):>8.3f}{gap:>+10.3f}"
                  f"{float(np.mean(per_cal[name])):>10.2f}"
                  f"{float(np.mean(per_mdd[name])):>10.2f}{wins:>6d}/{len(r)}")

    print(f"\n소요 {time.time()-t0:.0f}초")


if __name__ == '__main__':
    main()
