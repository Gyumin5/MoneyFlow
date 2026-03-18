#!/usr/bin/env python3
"""비중 캡 과적합 검증.

올바른 캡 구현: n_picks=5 유지, 개별 비중만 cap% 제한, 초과분은 현금.
예: HC=3, Cap 40% → 각 33% (캡 안걸림, 33<40)
    HC=2, Cap 40% → 각 40%, 현금 20%
    HC=1, Cap 40% → 40%, 현금 60%
    HC=1, Cap 50% → 50%, 현금 50%
    HC=2, Cap 50% → 각 50% (= EW와 동일)

검증:
1. 올바른 캡 비교 (30%, 40%, 50%, 60%, 80%, 100%)
2. 상위 N개월 제거 (Top 1/3/5 month drop)
3. 특정 코인 제거 (BNB, HT, FTT, REP)
4. 연도 제외 (2021 제외)
5. Walk-forward OOS (2018-2020 train → 2021-2025 test)
"""

import os, sys, copy
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    Params, load_data, _close_to, calc_ret, get_vol, get_sma,
    get_price, load_universe, filter_universe, load_all_prices,
    calc_metrics
)


def B(**kw):
    base = dict(
        canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1,
        health='HK', health_sma=2, health_mom_short=21,
        health_mom_long=90, vol_cap=0.05,
        risk='G5', selection='S8', n_picks=5,
        dd_exit_lookback=60, dd_exit_threshold=-0.25,
        bl_threshold=-0.15, bl_days=7,
        start_date='2018-01-01', end_date='2025-06-30',
    )
    base.update(kw)
    return Params(**base)


# ── 올바른 캡 적용 백테스트 ──
# strategy_engine의 run_backtest를 사용하되, weighting 결과를 후처리

def run_bt_with_cap(prices, universe, params, cap_pct=1.0):
    """Run backtest with proper weight cap.

    After the engine computes weights (EW baseline),
    cap each position at cap_pct and put excess in cash.
    This is done by monkey-patching the weight computation.
    """
    from strategy_engine import run_backtest as _run_bt

    if cap_pct >= 1.0:
        return _run_bt(prices, universe, params)

    # We need to intercept the weighting step.
    # Simplest: run with EW, then in the rebalance step, cap weights.
    # Since we can't easily intercept, we'll re-implement the core loop
    # with cap applied.

    # Actually, let's use a simpler approach:
    # Run the engine but patch the compute_weights function
    import strategy_engine as se
    original_compute = se.compute_weights

    def capped_compute(picks, prices_arg, date_arg, params, state):
        weights = original_compute(picks, prices_arg, date_arg, params, state)
        if not weights:
            return weights

        # Apply cap
        capped = {}
        for t, w in weights.items():
            if t == 'CASH':
                capped[t] = w
            else:
                capped[t] = min(w, cap_pct)

        # Normalize: excess goes to cash
        total_invested = sum(w for t, w in capped.items() if t != 'CASH')
        cash_from_cap = 1.0 - total_invested - capped.get('CASH', 0)
        if cash_from_cap > 0.001:
            capped['CASH'] = capped.get('CASH', 0) + cash_from_cap

        return capped

    se.compute_weights = capped_compute
    try:
        result = _run_bt(prices, universe, params)
    finally:
        se.compute_weights = original_compute

    return result


def get_monthly_returns(pv):
    """Extract monthly returns from equity curve."""
    if pv is None or len(pv) < 2:
        return {}
    values = pv['Value']
    months = {}
    for ym in sorted(set(d.strftime('%Y-%m') for d in values.index)):
        mask = [d.strftime('%Y-%m') == ym for d in values.index]
        m_vals = values[mask]
        if len(m_vals) >= 2:
            months[ym] = m_vals.iloc[-1] / m_vals.iloc[0] - 1
    return months


def calc_from_monthly(monthly_rets):
    """Approximate Sharpe/CAGR/MDD from monthly returns."""
    if not monthly_rets:
        return {'Sharpe': 0, 'CAGR': 0, 'MDD': 0}
    rets = list(monthly_rets.values())
    # Compound
    equity = [1.0]
    for r in rets:
        equity.append(equity[-1] * (1 + r))
    eq = np.array(equity)
    years = len(rets) / 12
    cagr = (eq[-1] / eq[0]) ** (1/years) - 1 if years > 0 else 0
    peak = np.maximum.accumulate(eq)
    mdd = np.min(eq / peak - 1)
    # Monthly Sharpe → annualized
    mr = np.array(rets)
    sharpe = mr.mean() / mr.std() * np.sqrt(12) if mr.std() > 0 else 0
    return {'Sharpe': sharpe, 'CAGR': cagr, 'MDD': mdd}


def remove_top_months(monthly_rets, n_remove):
    """Remove top N performing months (set to 0)."""
    sorted_months = sorted(monthly_rets.items(), key=lambda x: -x[1])
    removed = set()
    for i in range(min(n_remove, len(sorted_months))):
        removed.add(sorted_months[i][0])
    return {ym: (0.0 if ym in removed else r) for ym, r in monthly_rets.items()}, removed


def remove_year(monthly_rets, year):
    """Remove all months of a specific year."""
    return {ym: r for ym, r in monthly_rets.items() if not ym.startswith(str(year))}


def main():
    print('Loading data...')
    prices, universe = load_data()
    print(f'  {len(prices)} tickers loaded\n')

    # ====================================================================
    #  Test 1: 올바른 캡 비교
    # ====================================================================
    print('=' * 120)
    print('  TEST 1: 올바른 비중 캡 비교 (n_picks=5 유지, 개별 비중만 제한)')
    print('=' * 120)

    CAPS = [1.0, 0.80, 0.60, 0.50, 0.40, 0.33, 0.25, 0.20]
    cap_results = {}

    for cap in CAPS:
        label = 'EW (100%)' if cap >= 1.0 else f'Cap {cap:.0%}'
        print(f'  Running {label}...')
        result = run_bt_with_cap(prices, universe, B(), cap)
        m = result.get('metrics', {})
        pv = result.get('pv')
        monthly = get_monthly_returns(pv)
        cap_results[cap] = {'metrics': m, 'pv': pv, 'monthly': monthly, 'label': label}

    print(f'\n{"Config":<14} {"Sharpe":>8} {"CAGR":>8} {"MDD":>8} {"Calmar":>8}')
    print('-' * 50)
    for cap in CAPS:
        r = cap_results[cap]
        m = r['metrics']
        sh, cagr, mdd = m.get('Sharpe',0), m.get('CAGR',0), m.get('MDD',0)
        cal = abs(cagr/mdd) if mdd else 0
        print(f'{r["label"]:<14} {sh:>8.3f} {cagr:>+7.1%} {mdd:>7.1%} {cal:>8.2f}')

    # ====================================================================
    #  Test 2: 상위 N개월 제거
    # ====================================================================
    print(f'\n{"=" * 120}')
    print('  TEST 2: 상위 대박 월 제거 후 재비교')
    print('=' * 120)

    ew_monthly = cap_results[1.0]['monthly']
    cap40_monthly = cap_results[0.40]['monthly']
    cap50_monthly = cap_results[0.50]['monthly']

    for n_drop in [0, 1, 3, 5]:
        ew_clean, removed = remove_top_months(ew_monthly, n_drop)
        c40_clean, _ = remove_top_months(cap40_monthly, n_drop)
        c50_clean, _ = remove_top_months(cap50_monthly, n_drop)

        m_ew = calc_from_monthly(ew_clean)
        m_40 = calc_from_monthly(c40_clean)
        m_50 = calc_from_monthly(c50_clean)

        label = f'Top {n_drop} 제거' if n_drop > 0 else '원본'
        removed_str = ', '.join(sorted(removed)[:3]) if removed else ''
        print(f'\n  [{label}] {removed_str}')
        print(f'    {"EW":<10} Sharpe {m_ew["Sharpe"]:>6.3f}  CAGR {m_ew["CAGR"]:>+7.1%}  MDD {m_ew["MDD"]:>7.1%}')
        print(f'    {"Cap 50%":<10} Sharpe {m_50["Sharpe"]:>6.3f}  CAGR {m_50["CAGR"]:>+7.1%}  MDD {m_50["MDD"]:>7.1%}')
        print(f'    {"Cap 40%":<10} Sharpe {m_40["Sharpe"]:>6.3f}  CAGR {m_40["CAGR"]:>+7.1%}  MDD {m_40["MDD"]:>7.1%}')
        d_50 = m_ew['Sharpe'] - m_50['Sharpe']
        d_40 = m_ew['Sharpe'] - m_40['Sharpe']
        print(f'    ΔSharpe: EW-Cap50={d_50:>+.3f}, EW-Cap40={d_40:>+.3f}')

    # ====================================================================
    #  Test 3: 특정 코인 제거
    # ====================================================================
    print(f'\n{"=" * 120}')
    print('  TEST 3: 문제 코인 제거 후 재비교')
    print('=' * 120)

    EXCLUDE_SETS = [
        ('원본', []),
        ('BNB 제거', ['BNB-USD']),
        ('HT 제거', ['HT-USD']),
        ('FTT 제거', ['FTT-USD']),
        ('BNB+HT+FTT 제거', ['BNB-USD', 'HT-USD', 'FTT-USD']),
        ('BNB+HT+FTT+REP 제거', ['BNB-USD', 'HT-USD', 'FTT-USD', 'REP-USD']),
    ]

    for exc_label, exclude_coins in EXCLUDE_SETS:
        # Filter prices
        filtered_prices = {t: p for t, p in prices.items() if t not in exclude_coins}

        ew_r = run_bt_with_cap(filtered_prices, universe, B(), 1.0)
        c50_r = run_bt_with_cap(filtered_prices, universe, B(), 0.50)
        c40_r = run_bt_with_cap(filtered_prices, universe, B(), 0.40)

        m_ew = ew_r.get('metrics', {})
        m_50 = c50_r.get('metrics', {})
        m_40 = c40_r.get('metrics', {})

        sh_ew = m_ew.get('Sharpe', 0)
        sh_50 = m_50.get('Sharpe', 0)
        sh_40 = m_40.get('Sharpe', 0)
        cagr_ew = m_ew.get('CAGR', 0)
        cagr_50 = m_50.get('CAGR', 0)
        cagr_40 = m_40.get('CAGR', 0)
        mdd_ew = m_ew.get('MDD', 0)

        d_50 = sh_ew - sh_50
        d_40 = sh_ew - sh_40

        print(f'\n  [{exc_label}]')
        print(f'    EW     Sharpe {sh_ew:>6.3f}  CAGR {cagr_ew:>+7.1%}  MDD {mdd_ew:>7.1%}')
        print(f'    Cap50  Sharpe {sh_50:>6.3f}  CAGR {cagr_50:>+7.1%}')
        print(f'    Cap40  Sharpe {sh_40:>6.3f}  CAGR {cagr_40:>+7.1%}')
        print(f'    ΔSharpe: EW-Cap50={d_50:>+.3f}, EW-Cap40={d_40:>+.3f}')

    # ====================================================================
    #  Test 4: 연도 제외
    # ====================================================================
    print(f'\n{"=" * 120}')
    print('  TEST 4: 연도별 제외 (특히 2021)')
    print('=' * 120)

    for exclude_year in [None, 2021, 2019]:
        if exclude_year:
            ew_clean = remove_year(cap_results[1.0]['monthly'], exclude_year)
            c50_clean = remove_year(cap_results[0.50]['monthly'], exclude_year)
            c40_clean = remove_year(cap_results[0.40]['monthly'], exclude_year)
            label = f'{exclude_year} 제외'
        else:
            ew_clean = cap_results[1.0]['monthly']
            c50_clean = cap_results[0.50]['monthly']
            c40_clean = cap_results[0.40]['monthly']
            label = '전체'

        m_ew = calc_from_monthly(ew_clean)
        m_50 = calc_from_monthly(c50_clean)
        m_40 = calc_from_monthly(c40_clean)

        d_50 = m_ew['Sharpe'] - m_50['Sharpe']
        d_40 = m_ew['Sharpe'] - m_40['Sharpe']

        print(f'\n  [{label}]')
        print(f'    EW     Sharpe {m_ew["Sharpe"]:>6.3f}  CAGR {m_ew["CAGR"]:>+7.1%}  MDD {m_ew["MDD"]:>7.1%}')
        print(f'    Cap50  Sharpe {m_50["Sharpe"]:>6.3f}  CAGR {m_50["CAGR"]:>+7.1%}  MDD {m_50["MDD"]:>7.1%}')
        print(f'    Cap40  Sharpe {m_40["Sharpe"]:>6.3f}  CAGR {m_40["CAGR"]:>+7.1%}  MDD {m_40["MDD"]:>7.1%}')
        print(f'    ΔSharpe: EW-Cap50={d_50:>+.3f}, EW-Cap40={d_40:>+.3f}')

    # ====================================================================
    #  Test 5: Walk-Forward OOS
    # ====================================================================
    print(f'\n{"=" * 120}')
    print('  TEST 5: Walk-Forward (전반 vs 후반)')
    print('=' * 120)

    PERIODS = [
        ('전반 2018-2021', '2018-01-01', '2021-12-31'),
        ('후반 2022-2025', '2022-01-01', '2025-06-30'),
        ('최근 2023-2025', '2023-01-01', '2025-06-30'),
    ]

    for period_label, start, end in PERIODS:
        ew_r = run_bt_with_cap(prices, universe, B(start_date=start, end_date=end), 1.0)
        c50_r = run_bt_with_cap(prices, universe, B(start_date=start, end_date=end), 0.50)
        c40_r = run_bt_with_cap(prices, universe, B(start_date=start, end_date=end), 0.40)

        m_ew = ew_r.get('metrics', {})
        m_50 = c50_r.get('metrics', {})
        m_40 = c40_r.get('metrics', {})

        sh_ew = m_ew.get('Sharpe', 0)
        sh_50 = m_50.get('Sharpe', 0)
        sh_40 = m_40.get('Sharpe', 0)

        print(f'\n  [{period_label}]')
        print(f'    EW     Sharpe {sh_ew:>6.3f}  CAGR {m_ew.get("CAGR",0):>+7.1%}  MDD {m_ew.get("MDD",0):>7.1%}')
        print(f'    Cap50  Sharpe {sh_50:>6.3f}  CAGR {m_50.get("CAGR",0):>+7.1%}  MDD {m_50.get("MDD",0):>7.1%}')
        print(f'    Cap40  Sharpe {sh_40:>6.3f}  CAGR {m_40.get("CAGR",0):>+7.1%}  MDD {m_40.get("MDD",0):>7.1%}')
        d_50 = sh_ew - sh_50
        d_40 = sh_ew - sh_40
        print(f'    ΔSharpe: EW-Cap50={d_50:>+.3f}, EW-Cap40={d_40:>+.3f}')

    # ====================================================================
    #  Summary
    # ====================================================================
    print(f'\n{"=" * 120}')
    print('  종합 요약')
    print('=' * 120)
    print('  EW vs Cap50 ΔSharpe가 0에 가까울수록 = 집중투자 효과가 적음 (캡 무관)')
    print('  EW vs Cap50 ΔSharpe가 클수록 = 집중투자에 의존한 성과 (과적합 위험)')


if __name__ == '__main__':
    main()
