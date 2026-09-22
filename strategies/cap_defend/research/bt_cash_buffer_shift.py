"""현금 버퍼를 라이브와 맞췄을 때 기준선이 얼마나 움직이나 (2026-09-22).

배경. 백테스트 두 엔진의 배분 규모가 라이브와 달랐다.
  선물: pv × 비중 × 0.95  (라이브는 총자산 × (1-0.01) × 비중 × 배수) → 4퍼센트포인트 차이
  현물: pv × 비중 × 0.99602  (1/(1+leverage×tx_cost) — 거래비용 예약분이지 버퍼가 아니다)
        라이브는 1퍼센트 버퍼 → 0.6퍼센트포인트 차이
운영자 판정은 "라이브의 현금 버퍼가 맞다" 였고 두 엔진을 1퍼센트로 고쳤다.

이 스크립트가 재는 것은 하나다 — 그 교정이 기준선을 얼마나 움직이나.
지금까지 채택한 파라미터가 전부 옛 세계에서 고른 것이라, 기준선이 크게 움직이면
채택 판단을 그 위에서 다시 봐야 한다.

방법. 옛 값은 이미 돌아 있는 자산곡선 캐시에서 읽고(그 캐시가 옛 버퍼로 계산된 것이다),
새 값은 지금 엔진으로 메모리에서 계산해 맞붙인다. 캐시에 쓰지 않는다 — 이름이 같은 채로
옛 곡선과 새 곡선이 섞이면 뒤의 모든 측정이 오염된다.
앵커 0 · 기본 비용 한 점만 본다. 버퍼는 두 팔에 똑같이 곱해지는 상수라
전체 게이트를 다시 도는 건 이 물음에 필요 없다(그건 별도 작업이다).

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 -u bt_cash_buffer_shift.py
"""
from __future__ import annotations

import os
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

import bt_coin_param_resweep as R  # noqa: E402

ANCHOR = 0
STRESS = False


def show(label: str, old, new) -> None:
    mo, mn = R.metrics(old), R.metrics(new)
    print(f"\n{label}")
    print(f"{'':<10}{'수익%':>10}{'낙폭%':>10}{'칼마':>9}{'누적배수':>11}")
    for tag, m, eq in (("옛 버퍼", mo, old), ("새 버퍼", mn, new)):
        print(f"{tag:<10}{m['CAGR']:>10.1f}{m['MDD']:>10.2f}{m['Cal']:>9.2f}"
              f"{eq.iloc[-1]/eq.iloc[0]:>11.2f}")
    print(f"{'차이':<10}{mn['CAGR']-mo['CAGR']:>+10.1f}{mn['MDD']-mo['MDD']:>+10.2f}"
          f"{mn['Cal']-mo['Cal']:>+9.2f}")
    common = old.index.intersection(new.index)
    ratio = (new.reindex(common) / old.reindex(common))
    print(f"자산곡선 비율 최소 {ratio.min():.4f} 최대 {ratio.max():.4f} "
          f"(상수배면 버퍼가 규모만 바꾼 것, 흔들리면 체결 경로가 갈린 것)")


def main() -> None:
    t0 = time.time()

    from unified_backtest import load_data, CASH_BUFFER as SPOT_BUF
    from backtest_futures_v25 import build_K2_signal, CASH_BUFFER as FUT_BUF
    print(f"현재 엔진 버퍼 — 현물 {SPOT_BUF:g} · 선물 {FUT_BUF:g}")

    print("코인 일봉 로드 중...")
    bars, funding = load_data("D")
    R._bars, R._funding = bars, funding
    R._k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f"  {len(bars)}종 ({time.time()-t0:.0f}초)")

    k = R._key(R.CUR, ANCHOR, STRESS)
    sp_old, fu_old = R._load("sp", k), R._load("fu", k)
    st = R._load("st", f"_a{ANCHOR}")
    if st is None:
        import pandas as pd
        p = os.path.join(R.EQDIR, f"st__a{ANCHOR}.csv")
        st = pd.read_csv(p, index_col=0, parse_dates=True)["eq"]
    if sp_old is None or fu_old is None:
        print("옛 버퍼 캐시가 없다 — 비교 불가")
        return

    print("새 버퍼로 현물 재계산...")
    sp_new = R.run_spot({}, R.PH_SPOT[ANCHOR], STRESS)
    print(f"  ({time.time()-t0:.0f}초)  새 버퍼로 선물 재계산...")
    fu_new = R.run_fut({}, R.PH_FUT[ANCHOR], STRESS)
    print(f"  ({time.time()-t0:.0f}초)")

    print("\n=== 앵커 0 · 기본 비용 · 슬리브 단독 ===")
    show("현물 슬리브 (0.99602 → 0.99)", sp_old, sp_new)
    show("선물 슬리브 (0.95 → 0.99)", fu_old, fu_new)

    print("\n=== 합성 60/25/15 ===")
    show("주식은 양쪽 같은 곡선", R.build_alloc(st, sp_old, fu_old),
         R.build_alloc(st, sp_new, fu_new))

    print(f"\n총 {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
