"""거래소 한 곳에 두는 비중을 전략 낙폭 아래로 맞추면 성과가 어떻게 되나 (2026-10-30).

왜 재나. 운영자가 물었다 — 거래소가 파산하면 그 거래소 돈이 전액 날아간다고 보고,
그 손실을 전략이 이미 감수하는 최대낙폭 수준으로 묶으려면 현물·선물을 15퍼센트씩
하면 되나. 지금은 주식 60 · 현물 25 · 선물 15 라 업비트 쪽만 선을 넘는다.

그래서 재는 것은 둘이다. 바꾼 비중의 성과(연복리·낙폭·칼마·샤프)와, 거래소 한 곳이
통째로 날아갔을 때 남는 자산이다. 뒤쪽은 전략 계산이 아니라 비중 곱셈이다 —
파산은 시장 사건이 아니라서 자산곡선에 안 들어온다.

앵커는 7곳 평균이다. 하나로 재면 그 자리만의 우연이 결론이 된다.

실행: python3 alloc_exchange_cap.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

# (이름, 주식, 현물, 선물)
MIXES = [
    ('현행 60/25/15', 0.60, 0.25, 0.15),
    ('70/15/15', 0.70, 0.15, 0.15),
    ('70/20/10', 0.70, 0.20, 0.10),
    ('75/15/10', 0.75, 0.15, 0.10),
    ('80/10/10', 0.80, 0.10, 0.10),
    ('85/8/7', 0.85, 0.08, 0.07),
]


def main() -> int:
    import bt_coin_param_resweep as R
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal

    t0 = time.time()
    print('코인 일봉 로드 중...')
    R._bars, R._funding = load_data('D')
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f'  심볼 {len(R._bars)}종 ({time.time() - t0:.0f}초)')

    # 앵커 7곳. 세 슬리브를 같은 자리끼리 짝지어 합성한다.
    sleeves = []
    for i in range(7):
        st = R.run_stock(i, False)
        sp = R.run_spot({}, R.PH_SPOT[i], False)
        fu = R.run_fut({}, R.PH_FUT[i], False)
        if st is None or sp is None or fu is None:
            continue
        sleeves.append((st, sp, fu))
    print(f'  앵커 {len(sleeves)}곳 준비 ({time.time() - t0:.0f}초)')
    if not sleeves:
        print('자산곡선을 못 얻었다')
        return 1

    print()
    print('=== 비중별 성과 (앵커 7곳 평균)')
    print(f'  {"비중":>14} {"연복리":>9} {"최대낙폭":>9} {"칼마":>7} {"샤프":>7}'
          f' {"칼마 우세 앵커":>13}')
    base_cal = None
    rows = []
    for name, ws, wsp, wfu in MIXES:
        ms = [R.metrics(R.build_alloc(st, sp, fu, w=(ws, wsp, wfu)))
              for st, sp, fu in sleeves]
        avg = {k: float(np.mean([m[k] for m in ms])) for k in ('CAGR', 'MDD', 'Cal', 'Sharpe')}
        cals = [m['Cal'] for m in ms]
        if base_cal is None:
            base_cal = cals
            win = ''
        else:
            win = f'{sum(1 for a, b in zip(cals, base_cal) if a >= b)}/{len(cals)}'
        rows.append((name, ws, wsp, wfu, avg))
        print(f'  {name:>14} {avg["CAGR"]:>8.2f}% {avg["MDD"]:>8.2f}%'
              f' {avg["Cal"]:>7.2f} {avg["Sharpe"]:>7.2f} {win:>13}')

    print()
    print('=== 거래소 한 곳이 통째로 날아갔을 때 (비중 곱셈, 전략 계산 아님)')
    print(f'  {"비중":>14} {"업비트 파산":>11} {"바이낸스 파산":>13} {"둘 다":>8}'
          f' {"전략 최대낙폭":>13} {"낙폭 안에 드나":>14}')
    for name, ws, wsp, wfu, avg in rows:
        mdd = abs(avg['MDD'])  # metrics() 가 이미 퍼센트다
        both = (wsp + wfu) * 100
        worst = max(wsp, wfu) * 100
        ok = '예' if worst <= mdd else '아니오'
        print(f'  {name:>14} {wsp * 100:>10.1f}% {wfu * 100:>12.1f}% {both:>7.1f}%'
              f' {mdd:>12.2f}% {ok:>14}')

    print()
    print('읽는 법: 오른쪽 칸은 한 거래소가 날아간 손실이 전략이 이미 겪은 최대낙폭 안에')
    print('드는지만 본다. 파산 손실은 회복되지 않으므로 같은 숫자라도 낙폭보다 나쁘다.')
    print('빼 놓는 운영을 하면 실효 노출은 비중 × 노출 시간(53.0%)이다.')
    print()
    print(f'끝 ({time.time() - t0:.0f}초)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
