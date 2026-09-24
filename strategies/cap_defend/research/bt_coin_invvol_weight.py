"""고른 종목 사이 비중을 변동성으로 나눈다 (2026-09-25).

계기: 배수 축이 2026-09-24 에 닫혔다(급등 익일 억제 22갈래 전량 기각 — 이득이 두 날에서
나왔다). 운영자 물음은 "드물게 발생하는 사건 말고 자주 발생하면서 괜찮은 방법은 없나" 였다.
비중 축은 매일 작동하고 문턱도 사건도 없어서 과적합이 낄 자리가 적다.

무엇을 바꾸나: 지금은 고른 세 종목에 1/3 씩 똑같이 넣는다. 대신 각 종목의 실현변동성에
반비례해 나눈다 — 변동성이 큰 종목에 덜, 작은 종목에 더. 총 노출은 그대로 둔다
(합계는 현행과 같고 종목별 상한 cap=1/3 도 그대로). 즉 바꾸는 것은 배분뿐이다.

이미 기각한 것과 다른 축이다:
  · 보유 종수 2·4·5종(2026-09-?) — 칸 수를 바꾸는 축이지 같은 칸 안의 배분이 아니다.
  · 변동성 목표(2026-08-31 C) — 총 노출 자체를 변동성으로 정한다. 여기는 총 노출 불변.
  · 배수 계열 전부 — 레버리지를 건드린다. 여기는 안 건드린다.

팔 여섯:
  현행, 선물만, 현물만, 둘 다(창 90일), 둘 다(창 30일), 둘 다(창 60일).
  자산군을 갈라 본 이유는 어디서 효과가 나는지 알아야 채택 범위를 정할 수 있어서다.

엔진 주입은 환경변수 토글이다(CD_WEIGHT_MODE_SPOT / CD_WEIGHT_MODE_FUT /
CD_WEIGHT_VOL_DAYS). 정본 엔진의 토글 블록은 결론 후 revert 하고 패치만 남긴다.

판정 기준 (결과 보기 전 고정):
  1. 결정층은 합성 60/25/15 윈도우 순위합, 앵커 7곳 평균·중앙·최악.
  2. 비용 5배에서 순서가 뒤집히면 현행 유지.
  3. 창 길이 30/60/90 에 플래토가 있어야 한다. 한 값만 좋으면 채택하지 않는다.
  4. 이득이 몇 날에 몰려 있는지 먼저 본다 — 2026-09-24 기각 사유가 그것이었다.
     자산곡선이 갈라진 날이 전체의 절반 미만이면 이 축도 사건 의존으로 본다
     (매일 작동하는 규칙이므로 갈라진 날이 대부분이어야 정상이다).

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_coin_invvol_weight.py
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

OUT = "/home/gmoh/mon/251229/state/20260925-invvol"
EQDIR = os.path.join(OUT, "eq")

_bars = None
_funding = None
_k2 = None

# (이름, 현물 모드, 선물 모드, 변동성 창)
ARMS = [
    (CUR, 'ew', 'ew', 90),
    ("선물만 90", 'ew', 'invvol', 90),
    ("현물만 90", 'invvol', 'ew', 90),
    ("둘다 90", 'invvol', 'invvol', 90),
    ("둘다 60", 'invvol', 'invvol', 60),
    ("둘다 30", 'invvol', 'invvol', 30),
    # 2차 — 1차에서 현물만 90 이 유일하게 현행을 이겼다(순위 1.850/1.830, 6/7 앵커).
    # 90 이 그리드 끝이라 플래토를 못 본다. 창 길이를 양옆으로 넓혀 확인한다.
    ("현물만 30", 'invvol', 'ew', 30),
    ("현물만 60", 'invvol', 'ew', 60),
    ("현물만 120", 'invvol', 'ew', 120),
    # 3차 — 2차에서 현물만 네 갈래가 전부 현행을 이겼지만 창 30 이 다시 그리드 끝이다.
    # 아래로 더 내려가도 좋아지기만 하면 창 길이는 잡음이라는 뜻이고, 꺾이면 봉우리가 있다.
    ("현물만 10", 'invvol', 'ew', 10),
    ("현물만 20", 'invvol', 'ew', 20),
    # 4차 — 3차에서 10 이 또 끝이다. 5 로 한 번 더 내려가 바닥을 보고, 45 로 30~60 사이를
    # 메워 모양을 본다. 목적은 최고점 찾기가 아니라 어디가 봉우리고 어디가 평지인지 보는 것.
    ("현물만 5", 'invvol', 'ew', 5),
    ("현물만 45", 'invvol', 'ew', 45),
]


def _key(name: str, anchor: int, stress: bool) -> str:
    safe = name.replace("/", "_").replace(" ", "_").replace(".", "p")
    return f"{safe}__a{anchor}{'__s5' if stress else ''}"


def _save(kind: str, key: str, eq: pd.Series) -> None:
    eq.to_csv(os.path.join(EQDIR, f"{kind}__{key}.csv"), header=["eq"])


def _load(kind: str, key: str):
    p = os.path.join(EQDIR, f"{kind}__{key}.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def run_spot(mode: str, days: int, phase: int, stress: bool) -> pd.Series:
    from unified_backtest import run as bt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    os.environ["CD_WEIGHT_MODE_SPOT"] = mode
    os.environ["CD_WEIGHT_VOL_DAYS"] = str(days)
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
    return m.get("_equity")


def run_fut(mode: str, days: int, phase: int, stress: bool) -> pd.Series:
    from backtest_futures_v25 import run as fbt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    os.environ["CD_WEIGHT_MODE_FUT"] = mode
    os.environ["CD_WEIGHT_VOL_DAYS"] = str(days)
    try:
        m = fbt_run(_bars, _funding, interval="D", leverage=_k2,
                    tx_cost=TX_FUT * (STRESS if stress else 1.0), maint_rate=0.004,
                    vol_days=90, vol_mode="daily", post_flip_delay=5,
                    phase_offset_bars=phase, health_mode="mom2vol",
                    start_date=START, end_date=END, **FUT_BASE)
    finally:
        os.environ["CD_WEIGHT_MODE_FUT"] = 'ew'
    return m.get("_equity")


def divergence(arm: str) -> None:
    """매일 작동하는 규칙이면 자산곡선이 대부분의 날에 갈라져야 한다."""
    for kind in ("sp", "fu"):
        b = _load(kind, _key(CUR, 0, False))
        x = _load(kind, _key(arm, 0, False))
        if b is None or x is None:
            continue
        i = b.index.intersection(x.index)
        d = (x.loc[i].pct_change() - b.loc[i].pct_change()).fillna(0)
        nz = d[d.abs() > 1e-9]
        lbl = "현물" if kind == "sp" else "선물"
        yrs = nz.groupby(nz.index.year).size().to_dict() if len(nz) else {}
        print(f"  {lbl}: 갈라진 날 {len(nz)}일 / {len(i)}일 ({len(nz)/max(len(i),1):.0%})  {yrs}")


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
    print(f"\n기간 {START} ~ {END} | 축: 고른 종목 사이 비중을 변동성 역수로 나눈다")
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

    total = len(ARMS) * len(ANCHORS) * 2 * 2
    done = 0
    for stress in (False, True):
        for a in ANCHORS:
            for name, sm, fm, days in ARMS:
                k = _key(name, a, stress)
                done += 2
                if _load("sp", k) is None:
                    t1 = time.time()
                    _save("sp", k, run_spot(sm, days, PH_SPOT[a], stress))
                    print(f"  [{done}/{total}] sp {name} a{a}"
                          f"{' s5' if stress else ''} {time.time()-t1:.0f}초", flush=True)
                if _load("fu", k) is None:
                    t1 = time.time()
                    if fm == 'ew':
                        # 선물을 안 건드리는 팔은 현행 선물곡선과 정의상 같다. 재계산 안 한다.
                        _save("fu", k, _load("fu", _key(CUR, a, stress)))
                    else:
                        _save("fu", k, run_fut(fm, days, PH_FUT[a], stress))
                    print(f"  [{done}/{total}] fu {name} a{a}"
                          f"{' s5' if stress else ''} {time.time()-t1:.0f}초", flush=True)

    print("\n[슬리브 단독 — 앵커 0, 기본 비용]")
    print(f"  {'팔':<12s}{'현물칼마':>9s}{'현물연수익':>11s}{'현물낙폭':>10s}"
          f"{'선물칼마':>10s}{'선물연수익':>11s}{'선물낙폭':>10s}")
    for name, _sm, _fm, _d in ARMS:
        sp = _load("sp", _key(name, 0, False))
        fu = _load("fu", _key(name, 0, False))
        if sp is None or fu is None:
            continue
        a_, b_ = metrics(sp), metrics(fu)
        print(f"  {name:<12s}{a_['Cal']:>9.2f}{a_['CAGR']:>+10.1f}%{a_['MDD']:>+9.1f}%"
              f"{b_['Cal']:>10.2f}{b_['CAGR']:>+10.1f}%{b_['MDD']:>+9.1f}%")

    print("\n[자산곡선이 갈라진 날 — 앵커 0, 기본 비용]")
    for name, _sm, _fm, _d in ARMS:
        if name == CUR:
            continue
        print(f" {name}")
        divergence(name)

    for stress in (False, True):
        names = [n for n, _, _, _ in ARMS]
        per_rank = {n: [] for n in names}
        per_cal = {n: [] for n in names}
        per_mdd = {n: [] for n in names}
        for a in ANCHORS:
            allocs = {}
            for name in names:
                k = _key(name, a, stress)
                sp, fu = _load("sp", k), _load("fu", k)
                if sp is None or fu is None:
                    continue
                al = build_alloc(stock_eq[a], sp, fu)
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
        print(f"  {'팔':<12s}{'평균순위':>9s}{'중앙':>8s}{'최악':>8s}{'현행대비':>10s}"
              f"{'칼마평균':>10s}{'낙폭평균':>10s}{'승앵커':>8s}")
        base = per_rank.get(CUR, [])
        for name in sorted(names, key=lambda x: float(np.mean(per_rank[x])) if per_rank[x] else 9):
            r = per_rank[name]
            if not r:
                print(f"  {name:<12s}  (자료 없음)")
                continue
            wins = sum(1 for i in range(len(r)) if base and r[i] < base[i])
            gap = float(np.mean(r)) - (float(np.mean(base)) if base else 0.0)
            print(f"  {name:<12s}{float(np.mean(r)):>9.3f}{float(np.median(r)):>8.3f}"
                  f"{float(np.max(r)):>8.3f}{gap:>+10.3f}"
                  f"{float(np.mean(per_cal[name])):>10.2f}"
                  f"{float(np.mean(per_mdd[name])):>10.2f}{wins:>6d}/{len(r)}")

    print(f"\n소요 {time.time()-t0:.0f}초")


if __name__ == '__main__':
    main()
