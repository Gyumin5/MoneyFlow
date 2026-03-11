#!/usr/bin/env python3
"""Compare V12 vs V13 vs V14 strategy performance via backtest."""

import os, sys
sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, run_backtest


def make_v12():
    """V12: SMA(50) + SMA30+Mom21+Vol10% + Sharpe선정 + InvVol가중"""
    return Params(
        canary='K8', vote_smas=(50,), vote_moms=(), vote_threshold=1,
        health='baseline',    # SMA30 + Mom21 + Vol
        health_sma=30,
        health_mom_short=21,
        health_mom_long=0,    # no Mom90
        vol_cap=0.10,
        selection='S6',       # Sharpe(126)+Sharpe(252)
        weighting='W6',       # inverse volatility
        risk='baseline',      # no crash breaker
        top_n=50,
        dd_exit_lookback=0,
        bl_threshold=0.0,
        start_date='2018-01-01',
    )


def make_v13():
    """V13: SMA(50) + SMA30+Mom21+Vol10% + Sharpe선정 + InvVol가중
    V13 added multi-bonus (RSI/MACD/BB) to selection — not in engine,
    so we use S6 (Sharpe) as closest approximation."""
    return Params(
        canary='K8', vote_smas=(50,), vote_moms=(), vote_threshold=1,
        health='baseline',    # SMA30 + Mom21 + Vol
        health_sma=30,
        health_mom_short=21,
        health_mom_long=0,    # no Mom90
        vol_cap=0.10,
        selection='S6',       # Sharpe (closest to multi-bonus)
        weighting='W6',       # inverse volatility
        risk='baseline',      # no crash breaker
        top_n=50,
        dd_exit_lookback=0,
        bl_threshold=0.0,
        start_date='2018-01-01',
    )


def make_v14():
    """V14: SMA(60) + Mom21+Mom90+Vol5% + 시총순 + EW + G5 + DD/BL"""
    return Params(
        canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1,
        health='HK',          # configurable health
        health_sma=0,         # no SMA
        health_mom_short=21,
        health_mom_long=90,
        vol_cap=0.05,
        selection='baseline', # market cap order
        weighting='baseline', # equal weight
        risk='G5',            # crash breaker
        top_n=40,
        dd_exit_lookback=60,
        dd_exit_threshold=-0.25,
        bl_threshold=-0.15,
        bl_days=7,
        start_date='2018-01-01',
    )


def main():
    print("Loading data...")
    prices, universe = load_data(top_n=50)  # need 50 for V12/V13
    print(f"  {len(prices)} tickers loaded\n")

    versions = {
        'V12': make_v12(),
        'V13': make_v13(),
        'V14': make_v14(),
    }

    results = {}
    for name, params in versions.items():
        print(f"Running {name}: {params.label}...")
        r = run_backtest(prices, universe, params)
        results[name] = r
        m = r['metrics']
        print(f"  CAGR: {m['CAGR']:+.1%}  MDD: {m['MDD']:.1%}  Sharpe: {m['Sharpe']:.3f}  Final: ${m['Final']:,.0f}")

    # Summary table
    print(f"\n{'='*90}")
    print(f"  V12 vs V13 vs V14 백테스트 비교 (2018-01 ~ 2025-06)")
    print(f"{'='*90}")
    print(f"  {'':>6} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'Rebals':>7} {'DD Exit':>8} {'Final($)':>10}")
    print(f"  {'─'*80}")
    for name in ['V12', 'V13', 'V14']:
        m = results[name]['metrics']
        rc = results[name]['rebal_count']
        dd = results[name].get('dd_exit_count', 0)
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        print(f"  {name:>6} {m['CAGR']:>+7.1%} {m['MDD']:>7.1%} {m['Sharpe']:>8.3f} {m['Sortino']:>9.3f} {calmar:>8.2f} {rc:>7} {dd:>8} ${m['Final']:>9,.0f}")

    # V14 vs V12 improvement
    m12 = results['V12']['metrics']
    m14 = results['V14']['metrics']
    print(f"\n  V14 vs V12 변화:")
    print(f"    Sharpe: {m14['Sharpe'] - m12['Sharpe']:+.3f}")
    print(f"    MDD:    {m14['MDD'] - m12['MDD']:+.1%} ({'개선' if m14['MDD'] > m12['MDD'] else '악화'})")
    print(f"    CAGR:   {m14['CAGR'] - m12['CAGR']:+.1%}")

    # Yearly comparison
    all_years = set()
    for r in results.values():
        all_years.update(r['yearly'].keys())

    print(f"\n{'='*90}")
    print(f"  연도별 수익률 비교")
    print(f"{'='*90}")
    print(f"  {'Year':>6}  {'V12':>10}  {'V13':>10}  {'V14':>10}  {'Best':>6}")
    print(f"  {'─'*50}")
    for y in sorted(all_years):
        vals = {}
        for name in ['V12', 'V13', 'V14']:
            ym = results[name]['yearly'].get(y, {})
            if ym:
                vals[name] = ym['CAGR']
            else:
                vals[name] = None

        parts = []
        for name in ['V12', 'V13', 'V14']:
            v = vals[name]
            parts.append(f"{v:>+9.1%}" if v is not None else f"{'N/A':>10}")

        valid = {k: v for k, v in vals.items() if v is not None}
        best = max(valid, key=lambda k: valid[k]) if valid else ""
        print(f"  {y:>6}  {'  '.join(parts)}  {best:>6}")

    print(f"\n  연도별 MDD 비교")
    print(f"  {'Year':>6}  {'V12':>10}  {'V13':>10}  {'V14':>10}  {'Best':>6}")
    print(f"  {'─'*50}")
    for y in sorted(all_years):
        vals = {}
        for name in ['V12', 'V13', 'V14']:
            ym = results[name]['yearly'].get(y, {})
            if ym:
                vals[name] = ym['MDD']
            else:
                vals[name] = None

        parts = []
        for name in ['V12', 'V13', 'V14']:
            v = vals[name]
            parts.append(f"{v:>9.1%}" if v is not None else f"{'N/A':>10}")

        valid = {k: v for k, v in vals.items() if v is not None}
        best = max(valid, key=lambda k: valid[k]) if valid else ""  # higher = less bad
        print(f"  {y:>6}  {'  '.join(parts)}  {best:>6}")

    # Strategy parameter summary
    print(f"\n{'='*90}")
    print(f"  전략 파라미터 비교")
    print(f"{'='*90}")
    print(f"  {'Layer':<12} {'V12':<25} {'V13':<25} {'V14':<25}")
    print(f"  {'─'*85}")
    print(f"  {'Canary':<12} {'SMA(50)':<25} {'SMA(50)':<25} {'SMA(60)':<25}")
    print(f"  {'Health':<12} {'SMA30+Mom21+Vol10%':<25} {'SMA30+Mom21+Vol10%':<25} {'Mom21+Mom90+Vol5%':<25}")
    print(f"  {'Selection':<12} {'Sharpe Score':<25} {'Multi-Bonus':<25} {'시총순 (Market Cap)':<25}")
    print(f"  {'Weighting':<12} {'Inverse Vol':<25} {'Inverse Vol':<25} {'Equal Weight':<25}")
    print(f"  {'Risk Mgmt':<12} {'없음':<25} {'없음':<25} {'G5+DD+BL':<25}")
    print(f"  {'Universe':<12} {'Top 50':<25} {'Top 50':<25} {'Top 40':<25}")


if __name__ == '__main__':
    main()
