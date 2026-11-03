"""세 슬리브 달러 곡선을 지금 엔진으로 다시 만든다 (7앵커 × 비용 2벌).

state/20261015-allocrebal/eq/ 의 캐시는 2026-10-15 에 만들어졌는데 엔진이 그 뒤
두 번 바뀌었다 — 10-18 체결층 정합(cf766bd·163e382)과 11-02 전략 가드 기본값
(7de5788). 그 캐시로 재면 지금 전략이 아닌 것을 잰다. 같은 자리에 덮어쓰지 않고
새 디렉토리에 만든다 — 옛 캐시는 10-31 격자 결과의 재현 근거라 그대로 둔다.

실행: python3 make_curves.py   (순차, 백그라운드로 띄울 것)
"""
from __future__ import annotations

import os
import sys
import time

RES = "/home/gmoh/mon/251229/strategies/cap_defend/research"
PARENT = os.path.dirname(RES)
for _p in (RES, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

OUT = "/home/gmoh/mon/251229/state/20261103-krw-ledger/eq"
STOCK_TX = 0.001
STRESS_MULT = 5.0


def main() -> int:
    os.makedirs(OUT, exist_ok=True)
    import bt_coin_param_resweep as R
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal

    R._bars, R._funding = load_data("D")
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print("자료 로드 완료", flush=True)

    for anchor in range(7):
        for stress in (False, True):
            for kind in ("stock", "spot", "fut"):
                p = os.path.join(OUT, f"{kind}_a{anchor}_{'x5' if stress else 'x1'}.csv")
                if os.path.exists(p):
                    continue
                t0 = time.time()
                if kind == "stock":
                    import bt_stock_mom3 as bsm3
                    old = bsm3.TX
                    bsm3.TX = STOCK_TX * (STRESS_MULT if stress else 1.0)
                    try:
                        s = R.run_stock(anchor, stress)
                    finally:
                        bsm3.TX = old
                elif kind == "spot":
                    s = R.run_spot({}, R.PH_SPOT[anchor], stress)
                else:
                    s = R.run_fut({}, R.PH_FUT[anchor], stress)
                s = s.dropna()
                tmp = f"{p}.tmp.{os.getpid()}"
                s.rename("eq").to_frame().to_csv(tmp)
                os.replace(tmp, p)
                print(f"{kind} a{anchor} {'x5' if stress else 'x1'}"
                      f" · {len(s)}봉 · {time.time() - t0:.0f}초", flush=True)
    print("끝", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
