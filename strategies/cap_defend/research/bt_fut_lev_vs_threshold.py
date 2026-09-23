"""선물 문턱 이득의 정체 — 비중을 다시 맞춘 값인가, 배수를 제때 맞춘 값인가 (2026-09-23).

운영자 물음. 지금 구조는 배수를 발화한 날에만 맞춘다(라이브 auto_trade_binance.py 는
rebalancing_needed 가 거짓이면 매매 경로 자체를 건너뛰고, 백테스트도 배수를 다시 잡는 자리가
_execute_rebalance 안에만 있다). 그러면 문턱을 조여서 얻은 이득이 비중 재조정이 아니라
배수가 어긋난 채 버티는 구간이 짧아진 것일 수 있다. 둘을 분리해서 본다.

팔 넷. 문턱 축과 배수 축을 직교로 둔다.
  현행            문턱 0.03, 배수는 발화일에만 (라이브 그대로)
  선물 0.01       문턱 0.01, 배수는 발화일에만
  현행+배수매일   문턱 0.03, 배수는 매일
  0.01+배수매일   문턱 0.01, 배수도 매일
넷째 팔이 있어야 두 축이 겹치는지 더해지는지가 갈린다.

왜 다시 도나(기각된 실험의 의도적 재현). 2026-08-21 에 "매일 배수 재조정" 을 이미 재서
기각했다(history.md 같은 날 항목, 재현 패치 research/bt_fut_daily_lev.engine.patch).
다시 도는 이유는 둘이다. 하나, 그때 기준선은 현금버퍼가 0.95 로 틀려 있던 엔진이다.
둘, 그때 쓴 자는 선물 슬리브 단독 윈도우 순위합이고 지금 채택 판정에 쓰는 자는
앵커 7곳 곱하기 비용 2조건의 합성 60/25/15 다. 같은 자로 재야 문턱 결과와 나란히 놓인다.
문턱 축을 같이 놓고 재는 것도 그때는 안 한 것이다.

배수 매일 조정은 공짜가 아니다. 바이낸스에서 배수를 바꾸는 것 자체는 포지션 크기를 안 바꾸고
필요 증거금만 바꾼다. 목표 명목이 총자산×(1-버퍼)×비중×배수라서 배수가 오르면 그만큼 사야
실제로 그 배수로 돈다. 그래서 변형은 증거금을 고정하고 노출만 바꾸되 델타 명목에 거래비용과
슬리피지를 물린다(8월 패치의 _relever 그대로).

엔진은 환경변수 DAILY_LEV 스위치로만 바뀌어 있고 측정이 끝나면 되돌린다. 정본에 실험 코드를
남기지 않는다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_fut_lev_vs_threshold.py
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

# (이름, 문턱, 배수를 매일 맞추나)
ARMS: list[tuple[str, float, bool]] = [
    (R.CUR, 0.030, False),
    ("선물 0.01", 0.010, False),
    ("현행+배수매일", 0.030, True),
    ("0.01+배수매일", 0.010, True),
]


def _set_daily(daily: bool) -> None:
    os.environ["DAILY_LEV"] = "on" if daily else "off"


def ensure_fut(name: str, thr: float, daily: bool, anchor: int, stress: bool) -> None:
    """선물만 돌린다. 현물은 이 축이 아니므로 현행 곡선을 그대로 쓴다."""
    k = R._key(name, anchor, stress)
    if R._load("fu", k) is None:
        _set_daily(daily)
        R._save("fu", k, R.run_fut(dict(drift_threshold=thr), R.PH_FUT[anchor], stress))
        _set_daily(False)
    if R._load("sp", k) is None:
        cur = R._load("sp", R._key(R.CUR, anchor, stress))
        if cur is None:
            cur = R.run_spot({}, R.PH_SPOT[anchor], stress)
            R._save("sp", R._key(R.CUR, anchor, stress), cur)
        R._save("sp", k, cur)


def count_fires(thr: float, daily: bool, anchor: int = 0, stress: bool = False) -> dict:
    """발화·체결·배수재조정 건수. 셋 다 엔진 반환값이라 별도 계측이 필요 없다."""
    from backtest_futures_v25 import run as fbt_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"
    _set_daily(daily)
    kw = dict(R.FUT_BASE)
    kw["drift_threshold"] = thr
    m = fbt_run(R._bars, R._funding, interval="D", leverage=R._k2,
                tx_cost=R.TX_FUT * (R.STRESS if stress else 1.0), maint_rate=0.004,
                vol_days=90, vol_mode="daily", post_flip_delay=5,
                phase_offset_bars=R.PH_FUT[anchor], health_mode="mom2vol",
                start_date=R.START, end_date=R.END, **kw)
    _set_daily(False)
    mm = R.metrics(m.get("_equity"))
    return dict(rebal=int(m.get("Rebal", 0)), trades=int(m.get("Trades", 0)),
                relev=int(m.get("Relev", 0)), opp=int(m.get("RelevOpp", 0)),
                bars=int(m.get("RelevBars", 0)),
                cal=mm["Cal"], cagr=mm["CAGR"], mdd=mm["MDD"])


def main() -> None:
    os.makedirs(R.EQDIR, exist_ok=True)
    t0 = time.time()
    _set_daily(False)

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

    todo = [(n, t, d, a, s) for n, t, d in ARMS for a in R.ANCHORS for s in (False, True)]
    print(f"\n계산할 조합 {len(todo)}건 (이미 있는 것은 건너뜀)")
    done = 0
    for name, thr, daily, a, stress in todo:
        ensure_fut(name, thr, daily, a, stress)
        done += 1
        if done % 8 == 0:
            print(f"  {done}/{len(todo)} ({time.time()-t0:.0f}초)")
    print(f"  전부 완료 ({time.time()-t0:.0f}초)")

    arms_for_report = [(n, {}, dict(drift_threshold=t)) for n, t, _ in ARMS]
    R.report("문턱 대 배수 동기화", arms_for_report, False, stock_eq)
    R.report("문턱 대 배수 동기화", arms_for_report, True, stock_eq)

    print("\n=== 발화·체결·배수 재조정 (앵커 0 · 선물 슬리브 단독) ===")
    print("배수재조정 = 발화하지 않은 날에 배수가 바뀌어 명목을 맞춘 체결 건수(변형에서만 발생).")
    print("기회 = 그런 날 보유 코인-일 수. 무리밸봉 = 발화 없이 보유한 봉 수.")
    for stress in (False, True):
        tag = "비용 5배" if stress else "기본 비용"
        print(f"\n  [{tag}]")
        print(f"  {'팔':<16}{'발화':>7}{'체결':>7}{'배수재조정':>11}{'기회':>7}"
              f"{'무리밸봉':>9}{'선물Cal':>9}{'선물CAGR':>10}{'선물MDD':>9}")
        for name, thr, daily in ARMS:
            c = count_fires(thr, daily, 0, stress)
            print(f"  {name:<16}{c['rebal']:>7}{c['trades']:>7}{c['relev']:>11}"
                  f"{c['opp']:>7}{c['bars']:>9}{c['cal']:>9.2f}"
                  f"{c['cagr']:>10.1f}{c['mdd']:>9.2f}")

    print(f"\n총 {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
