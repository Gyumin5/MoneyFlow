"""선물 드리프트 문턱 0.015 의 두 미확인 사항을 닫는다 (2026-09-22).

10축 재검증(커밋 ddf5301)에서 선물 드리프트 0.015 만 비용 5배까지 살아남았다.
그때 못 닫은 것이 둘이고 여기서 그 둘만 잰다.

하나 — 플래토. 0.015 는 그 격자의 끝점이었다. 끝점이 1위인 것은 고원의 가운데일 수도,
더 가면 무너지는 절벽 가장자리일 수도 있다. 0.005·0.0075·0.01·0.02 를 채워 곡선을 본다.

둘 — 체결 밴드와의 상호작용. 라이브 선물은 명목 편차 ±5% 밴드를 쓰고
backtest_futures_v25.py 도 같은 밴드를 건다(증분 매도 538행 delta < -holdings*0.05,
증분 매수 573행 delta > holdings*0.05. 전량청산·신규진입은 밴드를 안 탄다 — 라이브와 같다).
그러므로 밴드는 이미 앞선 측정에 들어가 있다. 남은 물음은 "문턱을 조이면 발화는 늘어나는데
그 발화가 실제 주문으로 이어지나" 이고, 엔진이 돌려주는 Rebal(발화)·Trades(체결)로 잰다.
발화만 늘고 체결이 안 늘면 밴드가 삼킨 것이다.

현물 슬리브는 이 축이 아니라 손대지 않는다 — 앞선 스윕의 현행 현물 곡선을 그대로 쓴다.
엔진은 한 줄도 안 고친다(Rebal·Trades 는 원래 반환값이다).

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_fut_drift_plateau.py
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

# 문턱 격자. 이름이 R.CUR 인 팔은 앞선 스윕의 캐시를 그대로 쓴다.
ARMS: list[tuple[str, float]] = [
    ("선물 0.005", 0.005),
    ("선물 0.0075", 0.0075),
    ("선물 0.01", 0.010),
    ("선물 0.015", 0.015),
    ("선물 0.02", 0.020),
    (R.CUR, 0.030),
    ("선물 0.05", 0.050),
]


def ensure_fut(name: str, thr: float, anchor: int, stress: bool) -> None:
    """선물만 돌린다. 현물은 이 축이 아니므로 현행 곡선을 재사용한다."""
    k = R._key(name, anchor, stress)
    if R._load("fu", k) is None:
        R._save("fu", k, R.run_fut(dict(drift_threshold=thr), R.PH_FUT[anchor], stress))
    # 합성에 쓸 현물은 현행 캐시를 그 이름으로 복사해 둔다(같은 곡선임을 명시적으로).
    if R._load("sp", k) is None:
        cur = R._load("sp", R._key(R.CUR, anchor, stress))
        if cur is None:
            cur = R.run_spot({}, R.PH_SPOT[anchor], stress)
            R._save("sp", R._key(R.CUR, anchor, stress), cur)
        R._save("sp", k, cur)


def count_fires(thr: float, anchor: int = 0) -> dict:
    """발화(Rebal)와 체결(Trades)을 센다. 엔진 반환값이라 계측 코드가 필요 없다."""
    from backtest_futures_v25 import run as fbt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    kw = dict(R.FUT_BASE)
    kw["drift_threshold"] = thr
    m = fbt_run(R._bars, R._funding, interval="D", leverage=R._k2,
                tx_cost=R.TX_FUT, maint_rate=0.004,
                vol_days=90, vol_mode="daily", post_flip_delay=5,
                phase_offset_bars=R.PH_FUT[anchor], health_mode="mom2vol",
                start_date=R.START, end_date=R.END, **kw)
    eq = m.get("_equity")
    mm = R.metrics(eq)
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
        ensure_fut(name, thr, a, stress)
        done += 1
        if done % 10 == 0:
            print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}초)")
    print(f"  전부 완료 ({time.time()-t0:.0f}초)")

    # ── 1. 플래토: 문턱 곡선 ──
    arms_for_report = [(n, {}, dict(drift_threshold=t)) for n, t in ARMS]
    R.report("선물 드리프트 문턱(플래토)", arms_for_report, False, stock_eq)
    R.report("선물 드리프트 문턱(플래토)", arms_for_report, True, stock_eq)

    # ── 2. 체결 밴드: 발화 대비 체결 ──
    print("\n=== 발화 대비 체결 (앵커 0 · 기본 비용 · 선물 슬리브 단독) ===")
    print("체결 밴드는 이미 엔진에 들어 있다(증분 매매 ±5%, 전량청산·신규진입은 제외).")
    print(f"{'문턱':<12}{'발화':>8}{'체결':>8}{'체결/발화':>10}"
          f"{'선물Cal':>9}{'선물CAGR':>10}{'선물MDD':>9}")
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
