"""선물 리필의 탈락조건과 채움순서를 바꿔 본다 — 끔까지 포함해 60/25/15 로 잰다 (2026-10-01).

운영자 물음 셋.
  ⑴ 탈락조건을 이동평균 교차로 바꾸면 어떤가. 18일선이 127일선 아래로 내려가면 탈락.
     다른 기간 조합도 같이 본다.
  ⑵ 채움순서가 왜 모멘텀순인가. 시총순으로도 본다.
  ⑶ 끄는 것도 본다 — 켜는 게 좋아서 들어가 있을 텐데 그게 맞는지.
  셋을 따로 잰 뒤 좋은 것끼리 조합한다(조합 팔은 2차 실행에서 넣는다).

재는 자. 결정층은 합성 60/25/15 윈도우 순위합이고 앵커 7곳 각각에서 낸다. 비용 5배
스트레스를 같은 축에 붙인다. 주식·현물 슬리브는 이 축이 아니라 고정 팔이다 —
선물만 바뀐다.

기준선 주의. bt_coin_param_resweep 의 FUT_BASE 는 문턱 0.03 이다(2026-09-22 측정 당시 값).
지금 라이브는 0.015 이므로 여기서는 0.015 로 덮어쓴다. 그래서 캐시도 따로 쓴다.

엔진은 환경변수로만 바뀌어 있고 측정이 끝나면 되돌린다. 정본에 실험 코드를 남기지 않는다.
  REFILL_FAIL_MODE = mom2 | smacross,  REFILL_FAIL_SMA_S / _L
  REFILL_ORDER     = bt | alpha | mom | mcap
  DRIFT_HEALTH_MODE = refill | off   (기존 토글)

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_fut_refill_variants.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402  (하니스·순위합·주식 슬리브 재사용)

OUT = "/home/gmoh/mon/251229/state/20261001-futrefill"
EQDIR = os.path.join(OUT, "eq")

DRIFT_FUT = 0.015  # 라이브 현행 (2026-09-23 배포)

CUR = "현행(둘 다 음수)"

# (이름, 환경변수 덮어쓰기)
ARMS_FAIL: list[tuple[str, dict]] = [
    ("끔", dict(DRIFT_HEALTH_MODE="off")),
    (CUR, {}),
    ("교차 18/127", dict(REFILL_FAIL_MODE="smacross",
                        REFILL_FAIL_SMA_S="18", REFILL_FAIL_SMA_L="127")),
    ("교차 20/120", dict(REFILL_FAIL_MODE="smacross",
                        REFILL_FAIL_SMA_S="20", REFILL_FAIL_SMA_L="120")),
    ("교차 20/60", dict(REFILL_FAIL_MODE="smacross",
                       REFILL_FAIL_SMA_S="20", REFILL_FAIL_SMA_L="60")),
    ("교차 10/120", dict(REFILL_FAIL_MODE="smacross",
                        REFILL_FAIL_SMA_S="10", REFILL_FAIL_SMA_L="120")),
    ("교차 30/90", dict(REFILL_FAIL_MODE="smacross",
                       REFILL_FAIL_SMA_S="30", REFILL_FAIL_SMA_L="90")),
    ("교차 50/200", dict(REFILL_FAIL_MODE="smacross",
                        REFILL_FAIL_SMA_S="50", REFILL_FAIL_SMA_L="200")),
]

ARMS_ORDER: list[tuple[str, dict]] = [
    (CUR, {}),
    ("알파벳 전체풀", dict(REFILL_ORDER="alpha")),
    ("모멘텀순 전체풀", dict(REFILL_ORDER="mom")),
    ("시총순 전체풀", dict(REFILL_ORDER="mcap")),
]

AXES: dict[str, list[tuple[str, dict]]] = {
    "탈락조건": ARMS_FAIL,
    "채움순서": ARMS_ORDER,
}

ENV_KEYS = ("DRIFT_HEALTH_MODE", "REFILL_FAIL_MODE",
            "REFILL_FAIL_SMA_S", "REFILL_FAIL_SMA_L", "REFILL_ORDER")
ENV_DEFAULT = dict(DRIFT_HEALTH_MODE="refill", REFILL_FAIL_MODE="mom2",
                   REFILL_FAIL_SMA_S="18", REFILL_FAIL_SMA_L="127",
                   REFILL_ORDER="bt")


def set_env(over: dict) -> None:
    for k in ENV_KEYS:
        os.environ[k] = over.get(k, ENV_DEFAULT[k])


def _key(name: str, anchor: int, stress: bool) -> str:
    safe = name.replace("/", "_").replace(" ", "_").replace("(", "").replace(")", "")
    return f"{safe}__a{anchor}{'__s5' if stress else ''}"


def _save(kind: str, key: str, eq: pd.Series) -> None:
    eq.to_csv(os.path.join(EQDIR, f"{kind}__{key}.csv"), header=["eq"])


def _load(kind: str, key: str):
    p = os.path.join(EQDIR, f"{kind}__{key}.csv")
    if not os.path.exists(p):
        return None
    s = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    return s if len(s) > 100 else None


def run_fut_arm(over_env: dict, anchor: int, stress: bool, want_counts=False):
    from backtest_futures_v25 import run as fbt_run
    set_env(over_env)
    kw = dict(R.FUT_BASE)
    kw["drift_threshold"] = DRIFT_FUT
    m = fbt_run(R._bars, R._funding, interval="D", leverage=R._k2,
                tx_cost=R.TX_FUT * (R.STRESS if stress else 1.0), maint_rate=0.004,
                vol_days=90, vol_mode="daily", post_flip_delay=5,
                phase_offset_bars=R.PH_FUT[anchor], health_mode="mom2vol",
                start_date=R.START, end_date=R.END, **kw)
    set_env({})
    if want_counts:
        return m
    return m.get("_equity")


def ensure_spot(anchor: int, stress: bool) -> pd.Series:
    """현물은 이 축이 아니다 — 현행 한 벌만 돌려 모든 팔에 공유한다."""
    k = _key("현물현행", anchor, stress)
    eq = _load("sp", k)
    if eq is None:
        os.environ["DRIFT_HEALTH_MODE"] = "refill"
        eq = R.run_spot({}, R.PH_SPOT[anchor], stress)
        _save("sp", k, eq)
    return eq


def report(axis: str, arms, stress: bool, stock_eq, spot_eq) -> None:
    names = [a[0] for a in arms]
    rank = {n: [] for n in names}
    comp = {n: [] for n in names}      # 합성 지표 dict 목록
    fut = {n: [] for n in names}       # 선물 단독 지표 dict 목록

    for a in R.ANCHORS:
        curves = {}
        for name in names:
            fu = _load("fu", _key(name, a, stress))
            if fu is None:
                continue
            curves[name] = R.build_alloc(stock_eq[a], spot_eq[(a, stress)], fu)
            comp[name].append(R.metrics(curves[name]))
            fut[name].append(R.metrics(fu))
        if len(curves) < 2:
            continue
        sums, _w, n = R.window_rs(curves)
        for name in curves:
            rank[name].append(sums[name] / max(n, 1))

    tag = "비용 5배" if stress else "기본 비용"
    print(f"\n=== {axis} · {tag} · 앵커 {len(R.ANCHORS)}곳 · 합성 60/25/15 ===")
    print(f"{'팔':<18}{'순위평균':>9}{'중앙':>7}{'최악':>7}{'현행대비':>9}"
          f"{'합성Cal':>8}{'합성Sharpe':>11}{'합성CAGR':>10}{'합성MDD':>9}"
          f"{'선물Cal':>8}{'선물CAGR':>10}{'선물MDD':>9}{'이긴앵커':>9}")
    base = rank.get(CUR, [])
    base_mean = float(np.mean(base)) if base else float("nan")
    rows = sorted((float(np.mean(r)), n, r) for n, r in rank.items() if r)
    for mean_r, name, r in rows:
        c = comp[name]
        f = fut[name]
        wins = ("" if name == CUR or not base
                else f"{sum(1 for x, y in zip(r, base) if x < y)}/{len(base)}")
        mark = "  <- 현행" if name == CUR else ""
        print(f"{name:<18}{mean_r:>9.3f}{float(np.median(r)):>7.3f}{max(r):>7.3f}"
              f"{mean_r-base_mean:>+9.3f}"
              f"{np.mean([x['Cal'] for x in c]):>8.2f}"
              f"{np.mean([x['Sharpe'] for x in c]):>11.2f}"
              f"{np.mean([x['CAGR'] for x in c]):>9.1f}%"
              f"{np.mean([x['MDD'] for x in c]):>8.2f}%"
              f"{np.mean([x['Cal'] for x in f]):>8.2f}"
              f"{np.mean([x['CAGR'] for x in f]):>9.1f}%"
              f"{np.mean([x['MDD'] for x in f]):>8.2f}%"
              f"{wins:>9}{mark}")


def main() -> None:
    os.makedirs(EQDIR, exist_ok=True)
    os.makedirs(R.EQDIR, exist_ok=True)
    t0 = time.time()
    set_env({})

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...")
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)", flush=True)

    print("주식 슬리브(고정 팔) 준비 중...")
    stock_eq = {}
    for a in R.ANCHORS:
        p = os.path.join(R.EQDIR, f"st__a{a}.csv")
        if os.path.exists(p):
            stock_eq[a] = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
        else:
            stock_eq[a] = R.run_stock(a, False)
            stock_eq[a].to_csv(p, header=["eq"])
    print(f"  완료 ({time.time()-t0:.0f}초)", flush=True)

    print("현물 슬리브(고정 팔) 준비 중...")
    spot_eq = {}
    for a in R.ANCHORS:
        for s in (False, True):
            spot_eq[(a, s)] = ensure_spot(a, s)
    print(f"  완료 ({time.time()-t0:.0f}초)", flush=True)

    uniq = {}
    for _axis, arms in AXES.items():
        for name, env in arms:
            uniq[name] = env
    todo = [(n, e, a, s) for n, e in uniq.items()
            for a in R.ANCHORS for s in (False, True)]
    print(f"\n선물 계산 조합 {len(todo)}건 (이미 있는 것은 건너뜀)", flush=True)
    done = 0
    for name, env, a, s in todo:
        k = _key(name, a, s)
        if _load("fu", k) is None:
            _save("fu", k, run_fut_arm(env, a, s))
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}초)", flush=True)
    print(f"  전부 완료 ({time.time()-t0:.0f}초)", flush=True)

    for axis, arms in AXES.items():
        for stress in (False, True):
            report(axis, arms, stress, stock_eq, spot_eq)

    print("\n=== 리필 발화·탈락 건수 (앵커 0 · 기본 비용 · 선물 슬리브 단독) ===")
    print("발화 = 드리프트가 켜진 날. 탈락일 = 리필이 코인을 하나라도 떨어뜨린 날.")
    print("탈락수 = 떨어뜨린 스냅-코인 수(스냅 5개라 하루 최대 5).")
    print(f"  {'팔':<18}{'발화':>7}{'탈락일':>8}{'탈락수':>8}{'체결':>8}")
    for name, env in uniq.items():
        m = run_fut_arm(env, 0, False, want_counts=True)
        print(f"  {name:<18}{int(m.get('Rebal', 0)):>7}"
              f"{int(m.get('RefillDays', 0)):>8}{int(m.get('RefillDrops', 0)):>8}"
              f"{int(m.get('Trades', 0)):>8}", flush=True)

    print(f"\n총 {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
