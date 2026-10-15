"""선물 슬리브의 아주 작은 주문 수량 차이가 어디서 오는지 (2026-10-15).

bt_all_sleeves_causality.py 가 선물을 실패로 찍었다. 다만 크기가 현물과 천 배 이상
다르다 — 현물 1.29퍼센트, 선물 0.0011퍼센트. 음성 대조군으로 세운 슬리브가 같이
실패했으므로 검사 자체를 먼저 의심해야 하고, 그걸 푸는 길은 선물 쪽 차이의 원인을
지목하는 것이다. 원인이 현물과 다른 기제로 지목되면 "아무거나 실패로 찍는 검사" 가
아니라는 것이 선다.

## 가설

backtest_futures_v25.run 의 주문 크기 산출 자체는 깨끗하다 — `_execute_rebalance` 를
`pv_date=prev_date` 로 부르는 자리가 하나뿐이다(1105줄). 그런데 그 호출 직전,
같은 날 처리 안에서 펀딩비가 capital 을 깎는다(861~876줄).

    cur = get_close(bars, coin, <date 의 봉>)
    capital -= holdings[coin] * cur * fr_sum

`_port_val(prev_date)` 는 가격만 전일 것을 쓰고 capital 은 지금 값을 쓴다. 그래서
그날 종가가 흔들리면 펀딩비가 흔들리고, 그만큼 평가액이 흔들려 주문 수량이 흔들린다.
현물의 누수(평가 시점 자체가 당일 종가)와는 다른 기제다.

## 판정 (결과 보기 전에 못박는다)

같은 날을 펀딩비 없이(funding={}) 두 번 재생한다. 다른 입력은 안 바꾼다.

  - 가설 맞음: 펀딩비를 빼면 두 팔의 주문 수량이 허용오차(1e-10 상대) 안으로 일치한다.
  - 가설 틀림: 펀딩비를 빼도 차이가 남는다. 그러면 다른 원인이 있다는 뜻이고 그대로
    적는다.

펀딩비를 끄는 것은 원인 지목용 재생이지 전략 변경 제안이 아니다. 펀딩비는 실제로
나가는 비용이라 백테스트에서 뺄 대상이 아니다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research
      python3 bt_fut_funding_cause.py
"""
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import backtest_futures_v25 as fbt  # noqa: E402
import bt_coin_param_resweep as R  # noqa: E402
from bt_all_sleeves_causality import compare, fut_runner, perturb  # noqa: E402

DAY = '2020-11-18'


def arm(bars, funding, date, coins, frac):
    fbt.ORDER_PROBE.clear()
    os.environ['ORDER_PROBE_DATE'] = str(date.date())
    try:
        fut_runner(perturb(bars, date, coins, frac), funding)
    finally:
        os.environ.pop('ORDER_PROBE_DATE', None)
    return list(fbt.ORDER_PROBE)


def main() -> int:
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal
    print('코인 일봉 로드 중...', flush=True)
    bars, funding = load_data('D')
    R._bars, R._funding = bars, funding
    R._k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f'  심볼 {len(bars)}종', flush=True)

    date = bars['BTC'].index[bars['BTC'].index.get_indexer(
        [__import__('pandas').Timestamp(DAY)], method='ffill')[0]]
    coins = sorted(bars.keys())
    print(f'  고른 날 {date.date()} (bt_all_sleeves_causality 와 같은 날)\n', flush=True)

    out = {}
    for tag, fd in (('펀딩비 있음(현행)', funding), ('펀딩비 없음', {})):
        a = arm(bars, fd, date, coins, 0.25)
        b = arm(bars, fd, date, coins, 0.75)
        if not a or not b:
            print(f'[{tag}] 측정불가 — probe 비었음')
            out[tag] = '측정불가'
            continue
        print(f'[{tag}]')
        print(f'  평가액 {a[0]["pv_used"]:.9f} vs {b[0]["pv_used"]:.9f}')
        diffs = compare(a, b)
        if not diffs:
            print('  차이 없음 — 주문 수량이 그날 종가와 무관하다')
            out[tag] = '차이없음'
        else:
            print(f'  차이 {len(diffs)} 건')
            for nm, va, vb in diffs[:4]:
                rel = abs(va - vb) / abs(va) if va else float('nan')
                print(f'    {nm}: {va!r} vs {vb!r}  (상대 {rel:.3e})')
            out[tag] = '차이있음'
        print(flush=True)

    print('=' * 60)
    if out.get('펀딩비 있음(현행)') == '차이있음' and out.get('펀딩비 없음') == '차이없음':
        print('가설 맞음 — 선물의 차이는 전부 펀딩비가 그날 종가로 계산되는 데서 온다.')
        print('현물의 누수(주문 평가 시점 자체가 당일 종가)와 다른 기제다.')
        print('검사는 두 슬리브를 같은 이유로 찍은 게 아니다 — 원인이 서로 다르게 지목됐다.')
        return 0
    print(f'가설 틀림 — 결과 {out}. 다른 원인이 있다.')
    return 1


if __name__ == '__main__':
    sys.exit(main())
