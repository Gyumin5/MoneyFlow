"""시점을 고치면 파라미터 사이의 순위가 바뀌나 (2026-10-15).

운영자 물음 그대로다 — "중요한건 전략간 순위가 변했는가".

숫자가 부풀어 있었다는 것과 고른 설정이 틀렸다는 것은 다른 얘기다. 모든 후보가
같은 만큼 부풀면 순위는 그대로고, 그러면 지금 쓰는 값은 그대로 두면 된다. 후보마다
다르게 부풀었으면 지금 값은 시점 착오가 뽑아 준 것이고 다시 골라야 한다.

## 재는 것

코인 파라미터 재검증(bt_coin_param_resweep.py)이 쓰던 축과 팔을 그대로 가져온다.
축마다 현행 값을 가운데 두고 인접값을 둔 묶음이고, 그 묶음 안의 순위가 결정층이다.

두 벌을 돌린다.

  현행 엔진 — 현물 평가 당일 종가(SPOT_PV_DATE=same), 주식 당일 신호(STOCK_SIG_LAG=0)
  수정 엔진 — 현물 평가 전일 종가(prev), 주식 전 거래일 신호(1)

선물 엔진은 두 벌이 같다. 주문 크기를 전일 종가로 정하는 자리가 이미 고쳐져 있어서
이 토글이 안 걸린다(남아 있는 0.001퍼센트 차이는 펀딩비가 당일 종가로 계산되는
별도 기제이고 bt_fut_funding_cause.py 가 따로 지목한다). 그래서 선물 곡선은 축별로
한 번만 계산하고 두 벌이 같이 쓴다.

결정층은 합성 60/25/15 의 윈도우 순위합이고, 앵커 일곱 곳에서 각각 낸 뒤 평균을 쓴다
(채택 규약 그대로 — history #49).

## 판정 (결과 보기 전에 못박는다)

축마다 두 벌의 순위를 비교한다.

  (가) 1등이 같은가. 이게 제일 중요하다 — 1등이 바뀐 축은 그 파라미터를 시점 착오가
       뽑았다는 뜻이다.
  (나) 뒤바뀐 쌍이 몇 개인가. 팔 네 개면 쌍이 여섯이다. 0 이면 순위 완전 보존.
  (다) 현행 값의 등수가 어떻게 움직였나.

채택 판정이 아니다. 순위가 바뀐 축이 나오면 그 축을 다시 고르자는 제안이 아니라
"이 축은 시점 착오 위에서 골랐다" 는 사실을 적는 것이고, 재선정은 그 다음 결정이다.

## 캐시 판본 표시

곡선 캐시는 이 측정 전용 디렉토리에 판본 꼬리표를 붙여 쓴다. 옛 재검증 캐시
(state/20260922-resweep/eq)는 안 읽는다 — 그 사이 엔진이 바뀌었는데 캐시에 판본
표시가 없어 조용히 섞인 사고가 있었다(history #50). 두 벌 다 새로 센다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research
      python3 bt_rank_shift_check.py
      SHARD=0/3 … SHARD=2/3 으로 셋을 같이 돌리면 계산만 나눠 한다(보고 건너뜀).
      다 모이면 SHARD 없이 한 번 더 돌려 보고만 받는다.
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))

import bt_coin_param_resweep as R  # noqa: E402

OUT = "/home/gmoh/mon/251229/state/20261015-rankshift"
EQDIR = os.path.join(OUT, "eq")
ARMS = (("현행 엔진", "same", "0"), ("수정 엔진", "prev", "1"))


def _key(name: str) -> str:
    return name.replace("/", "_").replace(" ", "_").replace("±", "pm")


def _path(kind: str, name: str, anchor: int) -> str:
    return os.path.join(EQDIR, f"{kind}__{_key(name)}__a{anchor}.csv")


def _load(kind: str, name: str, anchor: int):
    p = _path(kind, name, anchor)
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def _save(kind: str, name: str, anchor: int, eq: pd.Series) -> None:
    eq.to_csv(_path(kind, name, anchor), header=["eq"])


def spot_curve(name: str, over: dict, anchor: int, pv: str) -> pd.Series:
    kind = f"sp_{pv}"
    got = _load(kind, name, anchor)
    if got is not None:
        return got
    os.environ["SPOT_PV_DATE"] = pv
    try:
        eq = R.run_spot(over, R.PH_SPOT[anchor], False)
    finally:
        os.environ["SPOT_PV_DATE"] = "same"
    _save(kind, name, anchor, eq)
    return eq


def fut_curve(name: str, over: dict, anchor: int) -> pd.Series:
    got = _load("fu", name, anchor)
    if got is not None:
        return got
    eq = R.run_fut(over, R.PH_FUT[anchor], False)
    _save("fu", name, anchor, eq)
    return eq


def stock_curve(anchor: int, lag: str) -> pd.Series:
    kind = f"st_lag{lag}"
    got = _load(kind, "주식", anchor)
    if got is not None:
        return got
    os.environ["STOCK_SIG_LAG"] = lag
    try:
        eq = R.run_stock(anchor, False)
    finally:
        os.environ["STOCK_SIG_LAG"] = "0"
    _save(kind, "주식", anchor, eq)
    return eq


def jobs() -> list[tuple[str, dict, dict]]:
    """계산해야 하는 (이름, 현물 덮어쓰기, 선물 덮어쓰기) 를 중복 없이 모은다."""
    seen, out = set(), []
    for arms in R.AXES.values():
        for name, sp, fu in arms:
            if name in seen:
                continue
            seen.add(name)
            out.append((name, sp, fu))
    return out


def ranks(cals: dict[str, float]) -> dict[str, float]:
    """칼마가 아니라 평균순위를 받는다 — 낮을수록 우세. 동률은 평균순위."""
    order = sorted(cals.items(), key=lambda x: x[1])
    out, i0 = {}, 0
    while i0 < len(order):
        i1 = i0
        while i1 + 1 < len(order) and order[i1 + 1][1] == order[i0][1]:
            i1 += 1
        avg = (i0 + i1) / 2 + 1
        for j in range(i0, i1 + 1):
            out[order[j][0]] = avg
        i0 = i1 + 1
    return out


def axis_ranking(arms, stock_eq, pv: str, lag: str) -> dict[str, float]:
    """축 하나의 팔별 평균 윈도우 순위(앵커 일곱 곳 평균). 낮을수록 우세."""
    acc: dict[str, list[float]] = {a[0]: [] for a in arms}
    for a in R.ANCHORS:
        comps = {}
        for name, sp_over, fu_over in arms:
            sp = _load(f"sp_{pv}", name, a)
            fu = _load("fu", name, a)
            if sp is None or fu is None:
                continue
            comps[name] = R.build_alloc(stock_eq[(lag, a)], sp, fu)
        if len(comps) < 2:
            continue
        sums, _w, n = R.window_rs(comps)
        if n == 0:
            continue
        for k, v in sums.items():
            acc[k].append(v / n)
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def main() -> int:
    os.makedirs(EQDIR, exist_ok=True)
    t0 = time.time()
    shard = os.environ.get("SHARD", "")

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...", flush=True)
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    todo = jobs()
    print(f"서로 다른 설정 {len(todo)}개 × 앵커 {len(R.ANCHORS)}곳", flush=True)

    units = [(i, nm, sp, fu, a) for i, (nm, sp, fu) in enumerate(todo) for a in R.ANCHORS]
    if shard:
        si, sn = (int(x) for x in shard.split("/"))
        units = [u for k, u in enumerate(units) if k % sn == si]
        print(f"  샤드 {si}/{sn} — 이 벌이 맡은 단위 {len(units)}개", flush=True)

    for k, (_i, nm, sp_over, fu_over, a) in enumerate(units):
        fut_curve(nm, fu_over, a)
        for _label, pv, _lag in ARMS:
            spot_curve(nm, sp_over, a, pv)
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(units)} ({time.time()-t0:.0f}초)", flush=True)

    if shard:
        print(f"계산 끝 ({time.time()-t0:.0f}초)")
        print("샤드 실행이라 주식 곡선과 보고는 건너뛴다. SHARD 없이 한 번 더 돌려라.")
        print("  (주식 곡선을 샤드마다 만들면 같은 파일을 동시에 쓴다.)")
        return 0

    stock_eq = {}
    for _label, _pv, lag in ARMS:
        for a in R.ANCHORS:
            stock_eq[(lag, a)] = stock_curve(a, lag)
    print(f"계산 끝 ({time.time()-t0:.0f}초)", flush=True)

    base = todo[0][0]
    g_sp = float((_load("sp_same", base, 0) - _load("sp_prev", base, 0)).abs().max())
    c0, c1 = stock_eq[("0", 0)], stock_eq[("1", 0)]
    ci = c0.index.intersection(c1.index)
    g_st = float((c0.loc[ci] - c1.loc[ci]).abs().max())
    print(f"\n사전 확인 — 현물 토글 차이 {g_sp:.6g} · 주식 토글 차이 {g_st:.6g}")
    if g_sp <= 1e-9 or g_st <= 1e-9:
        print("  멈춘다 — 토글 하나가 안 걸렸다. 두 벌이 같은 것을 재고 있다.")
        return 1
    print("  둘 다 갈린다. 비교로 간다.")

    flipped, total = [], 0
    for axis, arms in R.AXES.items():
        r_cur = axis_ranking(arms, stock_eq, "same", "0")
        r_fix = axis_ranking(arms, stock_eq, "prev", "1")
        if len(r_cur) < 2 or len(r_fix) < 2:
            print(f"\n[{axis}] 곡선이 모자라 건너뜀")
            continue
        total += 1
        o_cur = ranks(r_cur)
        o_fix = ranks(r_fix)
        w_cur = min(r_cur, key=lambda k: r_cur[k])
        w_fix = min(r_fix, key=lambda k: r_fix[k])

        names = [a[0] for a in arms if a[0] in r_cur and a[0] in r_fix]
        swaps = 0
        for i in range(len(names)):
            for j in range(i + 1, len(names)):
                a, b = names[i], names[j]
                if (r_cur[a] - r_cur[b]) * (r_fix[a] - r_fix[b]) < 0:
                    swaps += 1
        pairs = len(names) * (len(names) - 1) // 2

        mark = "바뀜" if w_cur != w_fix else "같음"
        print(f"\n[{axis}] 1등 {mark} · 뒤바뀐 쌍 {swaps}/{pairs}")
        print(f"  현행 엔진 1등 {w_cur}   수정 엔진 1등 {w_fix}")
        for nm in names:
            star = " ←현행값" if nm == R.CUR else ""
            print(f"    {nm:14s} 평균순위 {r_cur[nm]:.3f} → {r_fix[nm]:.3f}"
                  f"   등수 {o_cur[nm]:.1f} → {o_fix[nm]:.1f}{star}")
        if w_cur != w_fix or swaps:
            flipped.append((axis, mark, swaps, pairs))

    print(f"\n{'=' * 66}\n정리 — 축 {total}개\n{'=' * 66}")
    if not flipped:
        print("  순위가 바뀐 축 없음. 시점 결함은 모든 후보를 같은 방향으로 부풀렸다.")
        print("  지금 쓰는 파라미터는 시점을 고쳐도 그대로 뽑힌다.")
        return 0
    print(f"  순위가 움직인 축 {len(flipped)}개:")
    for axis, mark, swaps, pairs in flipped:
        print(f"    {axis:20s} 1등 {mark} · 뒤바뀐 쌍 {swaps}/{pairs}")
    print("\n  1등이 바뀐 축은 그 값을 시점 착오 위에서 골랐다는 뜻이다.")
    print("  재선정 여부는 이 측정이 정하지 않는다 — 사실만 적는다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
