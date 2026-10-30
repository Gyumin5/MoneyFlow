"""거래소에 자금을 상주시키지 않을 때의 전략 비용을 잰다 (2026-10-30).

왜 재나. 2026-10-29 바이낸스가 63억 달러 규모로 털렸다. 운영자가 물은 것은 둘이다 —
바이낸스에 테더를 상주시키는 게 맞나, 그리고 "평소엔 빼 놓고 카나리가 켜질 때만 넣어
사는" 운영이 비효율인가.

전략 쪽 비용은 하나로 모인다. 카나리가 꺼져 있는 동안은 두 슬리브가 전액 현금이라
빼 놓아도 전략이 바뀌지 않는다. 비용이 생기는 자리는 카나리가 켜지는 날 하나뿐이다 —
돈이 아직 거래소에 안 닿아 있으면 그날 못 산다.

그래서 재는 것은 "진입이 D일 늦으면 성과가 얼마나 깎이나" 다. 청산은 늦지 않는다
(파는 것은 거래소 안에서 즉시 되고, 빼는 것은 판 뒤의 일이다). 즉 비대칭 지연이다.

모델. 켜짐 전환 뒤 첫 D일의 일간 수익을 0 으로 둔다 — 그 며칠을 현금으로 보냈다는
뜻이고, 그 뒤에는 그때의 가격에서 복리가 다시 붙는다. 늦게 들어가서 더 싸게 사는
경우도 이 방식에 그대로 들어온다(그 구간 수익이 음수였으면 지연이 이득으로 잡힌다).

근사인 자리 하나. 지연된 회차는 종목 선정도 D일 뒤 자료로 했을 것이다. 여기서는
선정은 그대로 두고 노출만 끊는다. 선정이 바뀌는 효과는 이 측정에 안 들어간다.

실행: python3 exchange_custody_cost.py
"""
from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
for _p in (HERE, PARENT):
    if _p not in sys.path:
        sys.path.insert(0, _p)

DELAYS = [0, 1, 2, 3, 5]


def metrics(eq: pd.Series) -> dict:
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0.0
    mdd = (eq / eq.cummax() - 1).min()
    return {'CAGR': cagr, 'MDD': mdd, 'Cal': cagr / abs(mdd) if mdd else 0.0,
            'Sharpe': sh, 'Mult': eq.iloc[-1] / eq.iloc[0]}


def delayed_equity(eq: pd.Series, on_days: pd.DatetimeIndex, delay: int) -> pd.Series:
    """켜짐 전환 뒤 첫 delay 일의 일간 수익을 0 으로 두고 자산곡선을 다시 쌓는다."""
    if delay <= 0:
        return eq
    ret = eq.pct_change().fillna(0.0)
    pos = {d: i for i, d in enumerate(eq.index)}
    blocked = np.zeros(len(eq), dtype=bool)
    for d in on_days:
        i = pos.get(d)
        if i is None:
            continue
        blocked[i:i + delay] = True
    ret = ret.where(~blocked, 0.0)
    return (1.0 + ret).cumprod() * float(eq.iloc[0])


def main() -> int:
    import bt_coin_param_resweep as R
    from alloc_canary import coin_canary
    from unified_backtest import load_data
    from backtest_futures_v25 import build_K2_signal

    t0 = time.time()
    print('코인 일봉 로드 중...')
    R._bars, R._funding = load_data('D')
    R._k2 = build_K2_signal(R._bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                            btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                            l_min=2.0, l_mid=3.0, l_max=4.0)
    print(f'  심볼 {len(R._bars)}종 ({time.time() - t0:.0f}초)')

    can = coin_canary(R._bars)
    can = can.loc[R.START:R.END]
    flip_on = can & ~can.shift(1).fillna(False)
    on_days = can.index[flip_on]

    # ── 카나리가 꺼져 있는 비율과 입출금 횟수 ──
    seg, cur = [], None
    for d, v in can.items():
        if v and cur is None:
            cur = d
        elif not v and cur is not None:
            seg.append((cur, d))
            cur = None
    if cur is not None:
        seg.append((cur, can.index[-1]))
    lens = [int((b - a).days) for a, b in seg]
    yrs = (can.index[-1] - can.index[0]).days / 365.25
    print()
    print(f'카나리 계열 {can.index[0].date()} ~ {can.index[-1].date()} ({len(can)}일 · {yrs:.2f}년)')
    print(f'  꺼진 날 {int((~can).sum())}일 ({(~can).mean() * 100:.1f}%)'
          f' · 켜진 날 {int(can.sum())}일 ({can.mean() * 100:.1f}%)')
    print(f'  켜짐 전환 {len(on_days)}회 (연 {len(on_days) / yrs:.1f}회)'
          f' — 넣고 빼는 왕복이 그만큼 필요하다')
    if lens:
        print(f'  켜짐 구간 길이 중앙값 {int(np.median(lens))}일 · 평균 {np.mean(lens):.0f}일'
              f' · 최단 {min(lens)}일 · 최장 {max(lens)}일')
        short = sum(1 for x in lens if x <= 7)
        print(f'  7일 이하로 끝난 구간 {short}개 — 짧은 구간일수록 지연이 통째로 먹는다')

    # ── 꺼진 뒤 유예: 왕복 횟수와 거래소 체류 비율 ──
    # 유예는 전략을 안 바꾼다. 파는 것은 거래소 안에서 즉시 하고 돈만 더 오래 둔다.
    # 그래서 여기서 바뀌는 것은 성과가 아니라 손이 가는 횟수와 노출 시간뿐이다.
    print()
    print('=== 꺼진 뒤 며칠 더 두고 뺄 때 — 왕복 횟수와 노출')
    print(f'  {"유예":>6} {"왕복":>7} {"연 왕복":>8} {"거래소 체류":>11} {"없어진 왕복":>11}')
    arr = can.to_numpy()
    base_trips = None
    for g in (0, 1, 2, 3, 5, 7, 10, 14):
        # 켜진 날은 반드시 두고, 꺼진 구간이 g 일 이하면 그대로 둔 채로 넘긴다.
        held = arr.copy()
        i = 0
        n = len(arr)
        while i < n:
            if arr[i]:
                i += 1
                continue
            j = i
            while j < n and not arr[j]:
                j += 1
            # [i, j) 가 꺼진 구간. 양쪽이 다 켜짐이고 길이가 g 이하면 유지한다.
            if i > 0 and j < n and (j - i) <= g:
                held[i:j] = True
            i = j
        trips = int(np.sum(held[1:] & ~held[:-1])) + (1 if held[0] else 0)
        if base_trips is None:
            base_trips = trips
        print(f'  {g:>4}일 {trips:>6}회 {trips / yrs:>7.1f}회'
              f' {held.mean() * 100:>10.1f}%'
              f' {base_trips - trips:>10}회')

    # ── 슬리브별 진입 지연 비용 (앵커 7곳) ──
    # 앵커 하나로 재면 그 자리만의 우연이 결론이 된다 — 이 저장소 기준은 앵커 평균이다.
    for kind, runner, phases in (('코인 현물 25%', R.run_spot, R.PH_SPOT[:7]),
                                 ('코인 선물 15%', R.run_fut, R.PH_FUT[:7])):
        print()
        print(f'=== {kind} — 진입이 늦을 때 (앵커 {len(phases)}곳)')
        eqs = []
        for ph in phases:
            e = runner({}, ph, False)
            if e is not None and len(e) > 100:
                eqs.append(e)
        if not eqs:
            print('  자산곡선을 못 얻었다')
            continue
        base = [metrics(e) for e in eqs]
        print(f'  {"지연":>6} {"연복리":>9} {"최대낙폭":>9} {"칼마":>7} {"샤프":>7}'
              f'  {"연복리 손실":>11} {"칼마 변화":>10} {"칼마 개선 앵커":>12}')
        for d in DELAYS:
            ms = [metrics(delayed_equity(e, on_days.intersection(e.index), d)) for e in eqs]
            avg = {k: float(np.mean([m[k] for m in ms])) for k in ('CAGR', 'MDD', 'Cal', 'Sharpe')}
            gap = np.mean([m['CAGR'] - b['CAGR'] for m, b in zip(ms, base)]) * 100
            cgap = np.mean([m['Cal'] - b['Cal'] for m, b in zip(ms, base)])
            win = sum(1 for m, b in zip(ms, base) if m['Cal'] >= b['Cal'])
            print(f'  {d:>4}일 {avg["CAGR"] * 100:>8.2f}% {avg["MDD"] * 100:>8.2f}%'
                  f' {avg["Cal"]:>7.2f} {avg["Sharpe"]:>7.2f}'
                  f'  {gap:>+10.2f}%p {cgap:>+10.2f} {win:>8}/{len(eqs)}')
    print()
    print(f'끝 ({time.time() - t0:.0f}초)')
    return 0


if __name__ == '__main__':
    sys.exit(main())
