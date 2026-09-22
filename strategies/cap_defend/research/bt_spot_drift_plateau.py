"""현물 드리프트 문턱의 조인 쪽을 끝까지 내려본다 (2026-09-22).

10축 재검증(ddf5301)에서 현물 드리프트 축을 "방향이 없는 노이즈" 로 닫았다. 그 판정에
구멍이 있다. 그때 잰 값은 0.06 · 0.10(현행) · 0.15 · 0.20 이고 조인 쪽 끝점이 0.06 이었는데,
합성 칼마로 보면 0.06 이 두 비용 조건 모두 1위였다(기본 3.32 대 현행 3.28, 5배 2.61 대 2.49).
순위합이 0.06 을 3등으로 놓아서 노이즈로 읽었지만, 이건 선물에서 0.015 를 끝점으로 보고
"절벽인지 고원인지 모른다" 고 했던 것과 같은 모양이다. 선물은 아래를 채우니 고원이 나왔다.

그래서 현물도 아래를 채운다. 0.01 · 0.02 · 0.03 · 0.04 를 넣어 곡선을 본다.

반대 예상도 같이 적어둔다 — 현물 거래비용은 0.4% 로 선물(0.06%)의 일곱 배이고 레버리지가
없다. 조이는 대가가 선물보다 훨씬 비싸야 정상이다. 그 예상이 맞으면 조인 쪽이 무너지고,
틀리면 선물과 같은 고원이 나온다. 어느 쪽이든 축이 닫힌다.

선물 슬리브는 이 축이 아니라 손대지 않는다 — 앞선 스윕의 현행 선물 곡선을 그대로 쓴다.
엔진은 한 줄도 안 고친다(Rebal·Trades 는 원래 반환값이다).

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_spot_drift_plateau.py
"""
from __future__ import annotations

import os
import sys
import time

import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402  (하니스·캐시·순위합 재사용)

# 이름이 앞선 스윕과 같은 팔은 캐시를 그대로 쓴다(현물 0.06 · 0.15 · 현행).
ARMS: list[tuple[str, float]] = [
    ("현물 0.01", 0.01),
    ("현물 0.02", 0.02),
    ("현물 0.03", 0.03),
    ("현물 0.04", 0.04),
    ("현물 0.06", 0.06),
    (R.CUR, 0.10),
    ("현물 0.15", 0.15),
]


def ensure_spot(name: str, thr: float, anchor: int, stress: bool) -> None:
    """현물만 돌린다. 선물은 이 축이 아니므로 현행 곡선을 재사용한다."""
    k = R._key(name, anchor, stress)
    if R._load("sp", k) is None:
        R._save("sp", k, R.run_spot(dict(drift_threshold=thr), R.PH_SPOT[anchor], stress))
    if R._load("fu", k) is None:
        cur = R._load("fu", R._key(R.CUR, anchor, stress))
        if cur is None:
            cur = R.run_fut({}, R.PH_FUT[anchor], stress)
            R._save("fu", R._key(R.CUR, anchor, stress), cur)
        R._save("fu", k, cur)


def count_fires(thr: float, anchor: int = 0) -> dict:
    """발화(Rebal)와 체결(Trades). 엔진 반환값이라 계측 코드가 필요 없다."""
    from unified_backtest import run as bt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    kw = dict(R.SPOT_BASE)
    kw["drift_threshold"] = thr
    m = bt_run(R._bars, R._funding, interval="D", asset_type="spot", leverage=1.0,
               tx_cost=R.TX_SPOT, start_date=R.START, end_date=R.END,
               vol_mode="daily", phase_offset_bars=R.PH_SPOT[anchor],
               post_flip_delay=5, selection="greedy",
               stop_kind="none", stop_pct=0.0,
               dd_lookback=60, dd_threshold=-99.0,
               bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
               health_mode="mom2vol", **kw)
    mm = R.metrics(m.get("_equity"))
    return dict(rebal=int(m.get("Rebal", 0)), trades=int(m.get("Trades", 0)),
                cal=mm["Cal"], cagr=mm["CAGR"], mdd=mm["MDD"])


def main() -> None:
    os.makedirs(R.EQDIR, exist_ok=True)
    t0 = time.time()

    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print("코인 일봉 로드 중...")
    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  심볼 {len(R._bars)}종 ({time.time()-t0:.0f}초)")

    print("주식 슬리브(합성용 고정 팔) 앵커별 불러오는 중...")
    stock_eq = {}
    for a in R.ANCHORS:
        p = os.path.join(R.EQDIR, f"st__a{a}.csv")
        if os.path.exists(p):
            stock_eq[a] = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
        else:
            stock_eq[a] = R.run_stock(a, False)
            stock_eq[a].to_csv(p, header=["eq"])
    print(f"  완료 ({time.time()-t0:.0f}초)")

    todo = [(n, t, a, s) for n, t in ARMS for a in R.ANCHORS for s in (False, True)]
    print(f"\n계산할 조합 {len(todo)}건 (이미 있는 것은 건너뜀)")
    done = 0
    for name, thr, a, stress in todo:
        ensure_spot(name, thr, a, stress)
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}초)")
    print(f"  전부 완료 ({time.time()-t0:.0f}초)")

    arms_for_report = [(n, dict(drift_threshold=t), {}) for n, t in ARMS]
    R.report("현물 드리프트 문턱(플래토)", arms_for_report, False, stock_eq)
    R.report("현물 드리프트 문턱(플래토)", arms_for_report, True, stock_eq)

    print("\n=== 발화 대비 체결 (앵커 0 · 기본 비용 · 현물 슬리브 단독) ===")
    print(f"{'문턱':<12}{'발화':>8}{'체결':>8}{'체결/발화':>10}"
          f"{'현물Cal':>9}{'현물CAGR':>10}{'현물MDD':>9}")
    counts = {name: count_fires(thr) for name, thr in ARMS}
    for name, _ in ARMS:
        c = counts[name]
        ratio = c["trades"] / c["rebal"] if c["rebal"] else float("nan")
        mark = "  <- 현행" if name == R.CUR else ""
        print(f"{name:<12}{c['rebal']:>8}{c['trades']:>8}{ratio:>10.2f}"
              f"{c['cal']:>9.2f}{c['cagr']:>10.1f}{c['mdd']:>9.2f}{mark}")
    base = counts.get(R.CUR)
    if base:
        print("\n  현행 대비 증가율 — 발화가 늘어난 만큼 체결도 늘었나")
        print(f"{'문턱':<12}{'발화배수':>10}{'체결배수':>10}{'차이':>10}")
        for name, _ in ARMS:
            if name == R.CUR:
                continue
            c = counts[name]
            fr = c["rebal"] / base["rebal"] if base["rebal"] else float("nan")
            tr = c["trades"] / base["trades"] if base["trades"] else float("nan")
            print(f"{name:<12}{fr:>10.2f}{tr:>10.2f}{tr-fr:>+10.2f}")

    print(f"\n총 {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
