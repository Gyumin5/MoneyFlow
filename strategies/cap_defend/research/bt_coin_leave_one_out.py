"""종목 하나씩 빼보며 성과가 어디서 나오는지 본다 (2026-09-22).

운영자 가설: 지금 성적이 BTC 보다 더 번 소수 종목에 얹혀 있다. BNB·SOL 을 그 대표로 지목했다.

전 기간(2020-10-01~2026-09-18) 자료가 있는 41종 중 BTC 를 이긴 것은 여섯뿐이다.
  SOL 연 85.1% · DOGE 80.0% · BNB 74.6% · ZEC 70.5% · FTM 66.4% · TRX 54.0% (BTC 40.6%)

그래서 실제로 많이 들었던 종목을 하나씩 빼고 같은 전략을 다시 돌린다. 앵커 하나·기본 비용만
쓰는 가벼운 진단이다 — 순위합 게이트를 대신하지 않는다. 목적은 순위를 매기는 게 아니라
"성과가 어느 종목에 얼마나 묶여 있나" 를 눈으로 보는 것이다.

제외는 bars 에서 키를 지워서 한다. SSoT 엔진은 안 고친다.
BTC 는 카나리 기준 자산이라 빼면 전략 자체가 달라진다 — 진단 대상에서 뺀다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_coin_leave_one_out.py
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import pandas as pd  # noqa: E402

import bt_coin_param_resweep as R  # noqa: E402

ANCHOR = 0
# 보유 비중일 상위(BTC 제외) + 운영자가 지목한 둘. BTC 는 카나리라 제외 대상에서 뺀다.
TARGETS = ["BNB", "SOL", "TRX", "ETH", "XMR", "XRP", "LTC", "KAS"]
PAIRS = [("BNB", "SOL"), ("TRX", "ETH"), ("BNB", "TRX"), ("SOL", "TRX")]


def main() -> None:
    t0 = time.time()
    from unified_backtest import load_data, run as sp_run
    from backtest_futures_v25 import build_K2_signal, run as fu_run
    os.environ["DRIFT_HEALTH_MODE"] = "refill"

    bars_full, funding = load_data("D")
    stock = pd.read_csv(os.path.join(R.EQDIR, f"st__a{ANCHOR}.csv"),
                        index_col=0, parse_dates=True)["eq"]
    print(f"로드 완료 {len(bars_full)}종 ({time.time()-t0:.0f}초)")

    def measure(drop: tuple[str, ...]) -> dict:
        bars = {k: v for k, v in bars_full.items() if k not in drop}
        k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                             btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                             l_min=2.0, l_mid=3.0, l_max=4.0)
        sp = sp_run(bars, funding, interval="D", asset_type="spot", leverage=1.0,
                    tx_cost=R.TX_SPOT, start_date=R.START, end_date=R.END,
                    vol_mode="daily", phase_offset_bars=R.PH_SPOT[ANCHOR],
                    post_flip_delay=5, selection="greedy", stop_kind="none", stop_pct=0.0,
                    dd_lookback=60, dd_threshold=-99.0, bl_drop=-99.0, bl_days=7,
                    crash_threshold=-99.0, health_mode="mom2vol", **R.SPOT_BASE)["_equity"]
        fu = fu_run(bars, funding, interval="D", leverage=k2, tx_cost=R.TX_FUT,
                    maint_rate=0.004, vol_days=90, vol_mode="daily", post_flip_delay=5,
                    phase_offset_bars=R.PH_FUT[ANCHOR], health_mode="mom2vol",
                    start_date=R.START, end_date=R.END, **R.FUT_BASE)["_equity"]
        comp = R.metrics(R.build_alloc(stock, sp, fu))
        return dict(cal=comp["Cal"], cagr=comp["CAGR"], mdd=comp["MDD"],
                    sp=R.metrics(sp)["Cal"], fu=R.metrics(fu)["Cal"])

    base = measure(())
    print(f"\n기준선(아무것도 안 뺌) 합성 칼마 {base['cal']:.2f} · 연 {base['cagr']:.1f}% · "
          f"낙폭 {base['mdd']:.2f}% · 현물 {base['sp']:.2f} · 선물 {base['fu']:.2f}")

    print(f"\n=== 한 종목씩 제외 (앵커 {ANCHOR} · 기본 비용) ===")
    print(f"{'뺀 종목':<10}{'합성Cal':>9}{'기준대비':>10}{'합성CAGR':>10}"
          f"{'합성MDD':>9}{'현물Cal':>9}{'선물Cal':>9}")
    rows = []
    for c in TARGETS:
        if c not in bars_full:
            print(f"{c:<10}  자료 없음")
            continue
        m = measure((c,))
        rows.append((c, m))
        print(f"{c:<10}{m['cal']:>9.2f}{m['cal']-base['cal']:>+10.2f}{m['cagr']:>10.1f}"
              f"{m['mdd']:>9.2f}{m['sp']:>9.2f}{m['fu']:>9.2f}")
    print(f"  ({time.time()-t0:.0f}초)")

    print(f"\n=== 두 종목씩 제외 (앵커 {ANCHOR} · 기본 비용) ===")
    print(f"{'뺀 쌍':<14}{'합성Cal':>9}{'기준대비':>10}{'합성CAGR':>10}"
          f"{'합성MDD':>9}{'현물Cal':>9}{'선물Cal':>9}")
    for pair in PAIRS:
        if any(c not in bars_full for c in pair):
            continue
        m = measure(pair)
        name = "+".join(pair)
        print(f"{name:<14}{m['cal']:>9.2f}{m['cal']-base['cal']:>+10.2f}{m['cagr']:>10.1f}"
              f"{m['mdd']:>9.2f}{m['sp']:>9.2f}{m['fu']:>9.2f}")

    print(f"\n총 {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
