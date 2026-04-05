#!/usr/bin/env python3
"""서브계정 다배수 조합 백테스트.

아이디어: 서브계정 여러 개, 각각 다른 배수로 동일 전략 운영.
카나리 전체 OFF(현금) 시점에 서브계정 간 자금 균등 재분배.

시뮬레이션:
1. 각 서브계정을 독립 실행 → equity curve 생성
2. "전체 카나리 OFF" 시점 감지 (합산 target == CASH)
3. 해당 시점에 모든 서브의 equity를 합산 → 균등 재분배
4. 이어서 각자 독립 실행
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ensemble import combine_targets, STRATEGIES, load_data
from backtest_futures_full import run
import numpy as np, pandas as pd

START = '2020-10-01'
END = '2026-03-28'


def run_multi_leverage(combined_targets, bars_1h, funding_1h, leverages, initial_capital=10000.0):
    """다배수 서브계정 시뮬레이션.

    leverages: [2.0, 5.0] → 2개 서브계정
    카나리 전체 OFF 시 자금 균등 재분배.
    """
    from run_ensemble import SingleAccountEngine

    n_subs = len(leverages)
    sub_capital = initial_capital / n_subs  # 균등 분배

    # 각 서브를 독립적으로 실행하되, 카나리 OFF 시점에 동기화
    # Step 1: 카나리 OFF 시점 찾기
    off_dates = set()
    prev_is_cash = False
    for date, target in combined_targets:
        is_cash = target.get('CASH', 0) > 0.95
        if is_cash and not prev_is_cash:
            off_dates.add(date)  # OFF 전환 시점
        prev_is_cash = is_cash

    # Step 2: OFF 구간 사이를 "세그먼트"로 분할, 각 세그먼트를 독립 실행
    # 간단한 방법: 전체를 한 번에 실행하고, OFF 시점에 equity 재분배

    # 각 서브별 equity를 날짜별로 추적
    sub_equities = [[] for _ in range(n_subs)]
    sub_engines = []
    for i, lev in enumerate(leverages):
        eng = SingleAccountEngine(bars_1h, funding_1h, leverage=lev,
                                   initial_capital=sub_capital)
        # run()에서 초기화되는 속성을 미리 설정
        eng.capital = sub_capital
        eng.holdings = {}
        eng.entry_prices = {}
        eng.entry_bar_index = {}
        eng.margins = {}
        eng.reentry_cooldown = {}
        sub_engines.append(eng)

    # 타겟을 순차 실행하면서, OFF 시점에 재분배
    prev_target = {}
    segment_start = 0

    for idx, (date, target) in enumerate(combined_targets):
        is_cash = target.get('CASH', 0) > 0.95

        # 각 서브 엔진에 1봉씩 실행
        for i, engine in enumerate(sub_engines):
            # 청산 체크
            if engine.leverage > 1:
                for coin in list(engine.holdings.keys()):
                    df = bars_1h.get(coin)
                    if df is None: continue
                    ci = df.index.get_indexer([date], method='ffill')[0]
                    if ci < 0: continue
                    low = float(df['Low'].iloc[ci])
                    if low <= 0: continue
                    pnl = engine.holdings[coin] * (low - engine.entry_prices[coin])
                    eq = engine.margins[coin] + pnl
                    maint = engine.holdings[coin] * low * engine.maint_rate
                    if eq <= maint:
                        returned = max(eq - max(eq, 0) * 0.015, 0)
                        engine.capital += returned
                        del engine.holdings[coin]
                        del engine.entry_prices[coin]
                        del engine.margins[coin]
                        engine.entry_bar_index.pop(coin, None)

            # 펀딩비
            for coin in list(engine.holdings.keys()):
                fr_series = engine.funding.get(coin)
                if fr_series is None: continue
                if date in fr_series.index:
                    fr = float(fr_series.loc[date])
                    if fr != 0 and not np.isnan(fr):
                        cur = engine._get_price(coin, date)
                        if cur > 0:
                            engine.capital -= engine.holdings[coin] * cur * fr
            engine.capital = max(engine.capital, 0)

            # 리밸런싱
            if target != prev_target and target:
                engine._execute_rebalance(target, date)

            # PV 기록
            pv = engine.capital
            for coin in engine.holdings:
                cur = engine._get_price(coin, date)
                if cur > 0:
                    pv += engine.margins[coin] + engine.holdings[coin] * (cur - engine.entry_prices[coin])
            sub_equities[i].append({'Date': date, 'Value': max(pv, 0)})

        prev_target = target

        # 카나리 OFF 전환 시 재분배
        if date in off_dates:
            total = 0
            for i in range(n_subs):
                if sub_equities[i]:
                    total += sub_equities[i][-1]['Value']
            if total > 0:
                per_sub = total / n_subs
                for i, engine in enumerate(sub_engines):
                    # 포지션 전부 청산 (카나리 OFF니까 어차피 CASH)
                    engine.holdings.clear()
                    engine.entry_prices.clear()
                    engine.margins.clear()
                    engine.entry_bar_index.clear()
                    engine.reentry_cooldown.clear()
                    engine.capital = per_sub
                    # equity 마지막 값 수정
                    if sub_equities[i]:
                        sub_equities[i][-1]['Value'] = per_sub

    # 합산 equity
    total_equity = []
    for idx in range(len(sub_equities[0])):
        date = sub_equities[0][idx]['Date']
        total_val = sum(sub_equities[i][idx]['Value'] for i in range(n_subs))
        total_equity.append({'Date': date, 'Value': total_val})

    eq = pd.DataFrame(total_equity).set_index('Date')['Value']
    eq_daily = eq.resample('D').last().dropna()

    yrs = (eq_daily.index[-1] - eq_daily.index[0]).days / 365.25
    if eq_daily.iloc[-1] <= 0 or yrs <= 0:
        return {'Sharpe': 0, 'CAGR': -1, 'MDD': -1, 'Cal': 0}
    cagr = (eq_daily.iloc[-1] / eq_daily.iloc[0]) ** (1 / yrs) - 1
    dr = eq_daily.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()  # 1h 해상도 MDD
    cal = cagr / abs(mdd) if mdd != 0 else 0
    final = eq.iloc[-1]
    return {'Sharpe': sh, 'CAGR': cagr, 'MDD': mdd, 'Cal': cal, 'Final': final}


if __name__ == '__main__':
    data = {}
    for iv in ['4h', '1h']:
        print(f"Loading {iv}...")
        data[iv] = load_data(iv)

    traces = {}
    for key in ['4h1', '4h2', '1h1']:
        spec = STRATEGIES[key]
        iv = spec['interval']
        bars, funding = data[iv]
        trace = []
        run(bars, funding, interval=iv, leverage=1.0,
            start_date=START, end_date=END, _trace=trace, **spec['config'])
        traces[key] = trace
        print(f"  {key}: {len(trace)} bars")

    bars_1h, funding_1h = data['1h']
    btc_1h = bars_1h['BTC']
    all_dates = btc_1h.index[(btc_1h.index >= START) & (btc_1h.index <= END)]

    weights = {'4h1': 1/3, '4h2': 1/3, '1h1': 1/3}

    # Normalize funding index
    norm_funding = {}
    for coin, fr in funding_1h.items():
        nfr = fr.copy()
        nfr.index = nfr.index.floor('h')
        norm_funding[coin] = nfr

    combined = combine_targets(traces, weights, all_dates)

    # 단일 배수 (비교 기준)
    print("\n━━━ 단일 배수 (비교 기준) ━━━")
    print(f"  {'Config':<20s} {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'최종':>12s}")
    print(f"  {'-'*60}")
    for lev in [1.5, 2.0, 2.5, 3.0, 4.0, 5.0]:
        m = run_multi_leverage(combined, bars_1h, norm_funding, [lev], 10000)
        mult = m['Final'] / 10000
        print(f"  {lev:.1f}x 단일{'':<13s} {m['Sharpe']:>5.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} ${m['Final']:>10,.0f} ({mult:.0f}x)")

    # 2계정 조합
    print("\n━━━ 2계정 조합 (카나리 OFF 시 5:5 재분배) ━━━")
    print(f"  {'Config':<20s} {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'최종':>12s}")
    print(f"  {'-'*60}")
    pairs = [(1.5, 3), (1.5, 5), (2, 3), (2, 4), (2, 5), (3, 5)]
    for a, b in pairs:
        m = run_multi_leverage(combined, bars_1h, norm_funding, [a, b], 10000)
        mult = m['Final'] / 10000
        print(f"  {a}x+{b}x{'':<13s} {m['Sharpe']:>5.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} ${m['Final']:>10,.0f} ({mult:.0f}x)")

    # 3계정 조합
    print("\n━━━ 3계정 조합 (카나리 OFF 시 1/3씩 재분배) ━━━")
    print(f"  {'Config':<20s} {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'최종':>12s}")
    print(f"  {'-'*60}")
    triples = [(1.5, 2.5, 5), (2, 3, 5), (1.5, 3, 5), (2, 3, 4)]
    for a, b, c in triples:
        m = run_multi_leverage(combined, bars_1h, norm_funding, [a, b, c], 10000)
        mult = m['Final'] / 10000
        print(f"  {a}x+{b}x+{c}x{'':<8s} {m['Sharpe']:>5.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} ${m['Final']:>10,.0f} ({mult:.0f}x)")

    print("\n완료!")
