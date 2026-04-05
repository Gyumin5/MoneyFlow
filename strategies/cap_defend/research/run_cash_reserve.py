#!/usr/bin/env python3
"""현금 비중 + 고배수 조합 백테스트.

"자금의 X%만 Nx로 운용, 나머지 현금"
실효 레버리지 = X% × N

타겟별 조합:
- 타겟 2x: 100%×2x, 67%×3x, 50%×4x, 40%×5x
- 타겟 2.5x: 100%×2.5x(추정), 83%×3x, 63%×4x, 50%×5x
- 타겟 3x: 100%×3x, 75%×4x, 60%×5x
- 타겟 4x: 100%×4x, 80%×5x
- 타겟 5x: 100%×5x
"""
import sys, os
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_ensemble import SingleAccountEngine, combine_targets, STRATEGIES, load_data
from backtest_futures_full import run
import numpy as np, pandas as pd

START = '2020-10-01'
END = '2026-03-28'


def run_with_cash_reserve(combined_targets, bars_1h, funding_1h,
                           leverage, invest_pct, initial_capital=10000.0):
    """자금의 invest_pct%만 leverage로 운용, 나머지 현금.

    청산 시 투입분만 손실, 현금은 보존.
    카나리 OFF 시 전액 현금 → 다시 ON 시 invest_pct로 재투입.
    """
    invest_capital = initial_capital * invest_pct
    cash_reserve = initial_capital * (1 - invest_pct)

    engine = SingleAccountEngine(bars_1h, funding_1h, leverage=leverage,
                                  initial_capital=invest_capital)
    engine.capital = invest_capital
    engine.holdings = {}
    engine.entry_prices = {}
    engine.margins = {}

    # Normalize funding
    norm_funding = {}
    for coin, fr in funding_1h.items():
        nfr = fr.copy()
        nfr.index = nfr.index.floor('h')
        norm_funding[coin] = nfr

    pv_list = []
    prev_target = {}
    prev_was_cash = True

    for date, target in combined_targets:
        is_cash = target.get('CASH', 0) > 0.95

        # 카나리 OFF → ON 전환 시: 현금 재투입 비율 적용
        if not is_cash and prev_was_cash and not engine.holdings:
            total = engine.capital + cash_reserve
            invest_capital = total * invest_pct
            cash_reserve = total * (1 - invest_pct)
            engine.capital = invest_capital

        # 카나리 ON → OFF 전환 시: 투입분 회수
        if is_cash and not prev_was_cash:
            # 포지션 청산은 엔진이 처리 (target=CASH)
            pass

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

        # 펀딩비
        for coin in list(engine.holdings.keys()):
            fr_series = norm_funding.get(coin)
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

        # PV = 투입분 + 현금 예비
        pv_invest = engine.capital
        for coin in engine.holdings:
            cur = engine._get_price(coin, date)
            if cur > 0:
                pv_invest += engine.margins[coin] + engine.holdings[coin] * (cur - engine.entry_prices[coin])

        total_pv = max(pv_invest, 0) + cash_reserve
        pv_list.append({'Date': date, 'Value': total_pv})

        prev_target = target
        prev_was_cash = is_cash

    if not pv_list:
        return {}
    eq = pd.DataFrame(pv_list).set_index('Date')['Value']
    eq_daily = eq.resample('D').last().dropna()
    yrs = (eq_daily.index[-1] - eq_daily.index[0]).days / 365.25
    if eq_daily.iloc[-1] <= 0 or yrs <= 0:
        return {'Sharpe': 0, 'CAGR': -1, 'MDD': -1, 'Cal': 0, 'Final': 0}
    cagr = (eq_daily.iloc[-1] / eq_daily.iloc[0]) ** (1 / yrs) - 1
    dr = eq_daily.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return {'Sharpe': sh, 'CAGR': cagr, 'MDD': mdd, 'Cal': cal, 'Final': eq.iloc[-1]}


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
    combined = combine_targets(traces, weights, all_dates)

    # 다양한 조합: 저배수~초고배수 + 현금
    combos = []
    # (레버리지, 투입비율) → 실효 = 레버리지 × 투입비율
    for lev in [2, 3, 4, 5, 7, 10, 15, 20]:
        for pct in [1.0, 0.80, 0.60, 0.50, 0.40, 0.30, 0.25, 0.20, 0.15, 0.10]:
            effective = lev * pct
            if 1.5 <= effective <= 10.0:
                combos.append((lev, pct, effective))

    # 중복 제거 + 실효 레버리지 순 정렬
    combos.sort(key=lambda x: (x[2], x[0]))

    print(f"\n총 {len(combos)}개 조합\n")
    print(f"  {'레버리지':>6s} {'투입':>5s} {'현금':>5s} {'실효':>5s} {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'최종':>12s}")
    print(f"  {'-'*75}")

    results = []
    for lev, pct, eff in combos:
        m = run_with_cash_reserve(combined, bars_1h, funding_1h, lev, pct)
        if not m or 'Final' not in m:
            continue
        mult = m['Final'] / 10000
        cash_pct = (1-pct)*100
        results.append((lev, pct, eff, m))
        print(f"  {lev:>5d}x {pct:>5.0%} {cash_pct:>4.0f}% {eff:>5.1f}x {m['Sharpe']:>5.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} ${m['Final']:>10,.0f} ({mult:.0f}x)")

    # 실효 레버리지별 최고 Calmar
    print(f"\n━━━ 실효 레버리지별 최적 조합 ━━━")
    by_eff = {}
    for lev, pct, eff, m in results:
        eff_r = round(eff * 2) / 2  # 0.5 단위 반올림
        if eff_r not in by_eff or m['Cal'] > by_eff[eff_r][3]['Cal']:
            by_eff[eff_r] = (lev, pct, eff, m)

    print(f"  {'실효':>5s} {'최적조합':>15s} {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'최종':>12s}")
    print(f"  {'-'*65}")
    for eff_r in sorted(by_eff.keys()):
        lev, pct, eff, m = by_eff[eff_r]
        mult = m['Final'] / 10000
        print(f"  {eff_r:>5.1f}x {lev}x×{pct:.0%}{'':<8s} {m['Sharpe']:>5.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} ${m['Final']:>10,.0f} ({mult:.0f}x)")

    for target_lev, combos in targets.items():
        print(f"\n━━━ 타겟 실효 {target_lev}x ━━━")
        print(f"  {'Config':<20s} {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'최종':>12s}")
        print(f"  {'-'*60}")
        for lev, pct in combos:
            cash_pct = (1 - pct) * 100
            m = run_with_cash_reserve(combined, bars_1h, funding_1h, lev, pct)
            if not m or 'Final' not in m:
                continue
            mult = m['Final'] / 10000
            label = f"{lev}x×{pct:.0%} (현금{cash_pct:.0f}%)"
            print(f"  {label:<20s} {m['Sharpe']:>5.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f} ${m['Final']:>10,.0f} ({mult:.0f}x)")

    print("\n완료!")
