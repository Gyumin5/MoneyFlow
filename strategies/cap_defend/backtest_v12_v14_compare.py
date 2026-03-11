#!/usr/bin/env python3
"""
V12 vs V14 Strategy Comparison Backtest
========================================
Compare stock strategy, coin strategy, and 60/40 combined for both versions.

Stock V12: 12 ETF, VT+EEM dual canary SMA200, weighted mom(50/30/20), Mom3+Sh3 EW, defense top1
Stock V14: R8 (8 ETF), EEM-only canary SMA200 (0.5% hyst), no health, 12M Mom3+Sh3 EW, defense top3

Coin V12: BTC>SMA50, SMA30+Mom21+Vol10%, Sharpe scoring, InvVol
Coin V14: BTC>SMA60+1%hyst, Mom21+Mom90+Vol5%, market cap, EW, DD exit, blacklist, crash breaker
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, BASE_DIR)

# ============================================================
# PART 1: STOCK BACKTEST (using test_stock_improve infrastructure)
# ============================================================

from test_stock_improve import (
    SP, load_prices, precompute, run_bt, metrics,
    OFF_CURRENT, DEF_CURRENT, CANARY_DEFAULT, ALL_TICKERS
)

# R8 universe (V14)
OFF_R8 = ('SPY', 'QQQ', 'VGK', 'EWJ', 'EEM', 'VWO', 'GLD', 'PDBC')

# Ensure VGK, EWJ, VWO are in the ticker list
EXTRA_TICKERS = ['VGK', 'EWJ', 'VWO']

def run_stock_comparison():
    print("=" * 70)
    print("📈 STOCK STRATEGY: V12 vs V14")
    print("=" * 70)

    # Load all needed tickers
    all_needed = sorted(set(list(ALL_TICKERS) + EXTRA_TICKERS))
    print(f"Loading {len(all_needed)} tickers...")
    prices = load_prices(all_needed)
    ind = precompute(prices)
    print(f"Loaded {len(prices)} tickers with data")

    # V12 Stock: 12 ETF, VT+EEM dual canary, weighted mom, Mom3+Sh3 EW, defense top1
    v12_params = SP(
        offensive=OFF_CURRENT,       # 12 ETFs
        defensive=DEF_CURRENT,
        canary_assets=('VT', 'EEM'), # dual canary
        canary_sma=200,
        canary_hyst=0.0,             # no hysteresis
        health='none',               # no health filter
        defense='top1',              # defense best 1
        select='mom3_sh3',
        mom_style='default',         # 50/30/20
        weight='ew',
        tx_cost=0.001,
    )

    # V14 Stock: R8, EEM-only canary (0.5% hyst), no health, 12M Mom3+Sh3 EW, defense top3
    v14_params = SP(
        offensive=OFF_R8,            # 8 ETFs
        defensive=DEF_CURRENT,
        canary_assets=('EEM',),      # EEM only
        canary_sma=200,
        canary_hyst=0.005,           # 0.5% hysteresis
        health='none',               # No health filter (anchor-day robust)
        defense='top3',              # defense top 3
        select='mom3_sh3',
        mom_style='lh',              # 20/30/50 (pure 12M approximation is '12m')
        weight='ew',
        tx_cost=0.001,
    )

    # Actually, V14 uses pure 12M momentum, not lh (20/30/50)
    v14_params.mom_style = '12m'

    # Run anchor-day average (days 1-4, as validated)
    anchor_days = [1, 2, 3, 4]
    results = {}

    for label, params in [('V12_Stock', v12_params), ('V14_Stock', v14_params)]:
        anchor_results = []
        for day in anchor_days:
            p = SP(**{f.name: getattr(params, f.name) for f in params.__dataclass_fields__.values()})
            p._anchor = day
            df = run_bt(prices, ind, p)
            m = metrics(df)
            if m:
                anchor_results.append(m)

        if anchor_results:
            avg = {}
            for key in anchor_results[0]:
                vals = [r[key] for r in anchor_results]
                avg[key] = np.mean(vals)
            results[label] = avg
            results[f'{label}_curves'] = []
            # Also store individual day results
            for i, day in enumerate(anchor_days):
                results[f'{label}_d{day}'] = anchor_results[i] if i < len(anchor_results) else None

    return results


# ============================================================
# PART 2: COIN BACKTEST (using strategy_engine infrastructure)
# ============================================================

from strategy_engine import Params, load_data, run_backtest

def run_coin_comparison():
    print("\n" + "=" * 70)
    print("🪙 COIN STRATEGY: V12 vs V14")
    print("=" * 70)

    print("Loading coin data...")
    prices, universe_map = load_data(top_n=50)
    print(f"Loaded {len(prices)} coin tickers")

    # V12 Coin: BTC>SMA50, SMA30+Mom21+Vol10%, Sharpe scoring, InvVol
    v12_params = Params(
        canary='baseline',           # BTC > SMA(sma_period)
        sma_period=50,               # SMA(50)
        health='baseline',           # SMA30 + Mom21 + Vol
        vol_cap=0.10,                # 10%
        selection='S6',              # Sharpe(126)+Sharpe(252) scoring
        weighting='W6',              # InvVol
        n_picks=5,
        tx_cost=0.004,
        top_n=50,
        # No DD exit, blacklist, crash breaker
        dd_exit_lookback=0,
        bl_threshold=0.0,
        risk='baseline',
    )

    # V14 Coin: BTC>SMA60+1%hyst, Mom21+Mom90+Vol5%, market cap, EW, DD exit, BL, crash
    v14_params = Params(
        canary='K8',                 # K8 vote system with single SMA
        vote_smas=(60,),             # SMA(60)
        vote_threshold=1,
        canary_band=1.0,             # 1% hysteresis (engine divides by 100)
        health='HK',                 # Configurable health
        health_sma=0,                # No SMA in health
        health_mom_short=21,         # Mom(21)
        health_mom_long=90,          # Mom(90)
        health_vol_window=90,
        vol_cap=0.05,                # 5%
        selection='baseline',        # Market cap order (top N)
        weighting='baseline',        # Equal Weight
        n_picks=5,
        tx_cost=0.004,
        top_n=40,                    # T40
        # V14 additions
        dd_exit_lookback=60,         # DD 60-day
        dd_exit_threshold=-0.25,     # -25%
        bl_threshold=-0.15,          # Blacklist -15%
        bl_days=7,
        risk='G5',                   # Crash breaker
    )

    results = {}

    for label, params in [('V12_Coin', v12_params), ('V14_Coin', v14_params)]:
        print(f"\nRunning {label}...")
        result = run_backtest(prices, universe_map, params)
        if result and result.get('metrics'):
            m = result['metrics']
            results[label] = {
                'cagr': m.get('CAGR', 0),
                'mdd': m.get('MDD', 0),
                'sharpe': m.get('Sharpe', 0),
                'sortino': m.get('Sortino', 0),
                'calmar': m.get('CAGR', 0) / abs(m.get('MDD', -1)) if m.get('MDD', 0) != 0 else 0,
                'rebals': result.get('rebal_count', 0),
                'dd_exits': result.get('dd_exit_count', 0),
            }
            print(f"  {label}: Sharpe={m['Sharpe']:.3f}  CAGR={m['CAGR']:+.1%}  MDD={m['MDD']:.1%}")

    return results


# ============================================================
# PART 3: COMBINED 60/40
# ============================================================

def combine_6040(stock_metrics, coin_metrics):
    """Estimate 60/40 combined portfolio metrics."""
    # Simple weighted average (approximate — real combination needs daily curves)
    combined = {}
    for key in ['CAGR', 'Sharpe', 'Sortino']:
        s_val = stock_metrics.get(key, 0)
        c_val = coin_metrics.get(key, 0) if isinstance(coin_metrics.get(key), (int, float)) else 0
        combined[key] = 0.6 * s_val + 0.4 * c_val

    # MDD: worst case is sum (not average), but 60/40 provides diversification
    s_mdd = stock_metrics.get('MDD', 0)
    c_mdd = coin_metrics.get('MDD', 0) if isinstance(coin_metrics.get('MDD'), (int, float)) else 0
    combined['MDD'] = 0.6 * s_mdd + 0.4 * c_mdd  # approximate

    # Calmar from combined
    combined['Calmar'] = combined['CAGR'] / abs(combined['MDD']) if combined['MDD'] != 0 else 0

    return combined


# ============================================================
# PART 4: DISPLAY
# ============================================================

def fmt(v, fmt_str):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return '-'
    return fmt_str.format(v)

def print_comparison(stock_results, coin_results):
    print("\n" + "=" * 80)
    print("📊 V12 vs V14 전략 비교 결과")
    print("=" * 80)

    # Stock comparison
    print("\n┌─── 주식 전략 ───────────────────────────────────────────────┐")
    print(f"│ {'':>20} │ {'V12 (12ETF+VT&EEM)':>20} │ {'V14 (R8+EEM+12M)':>20} │")
    print(f"│ {'─'*20} │ {'─'*20} │ {'─'*20} │")

    s12 = stock_results.get('V12_Stock', {})
    s14 = stock_results.get('V14_Stock', {})

    rows = [
        ('Universe', '12 ETF', 'R8 (8 ETF)'),
        ('Canary', 'VT+EEM SMA200', 'EEM SMA200 0.5%h'),
        ('Health', 'None', 'None'),
        ('Momentum', 'Wgt 50/30/20', 'Pure 12M'),
        ('Defense', 'Top 1', 'Top 3 EW'),
        ('', '', ''),
        ('CAGR', fmt(s12.get('CAGR'), '{:+.1%}'), fmt(s14.get('CAGR'), '{:+.1%}')),
        ('MDD', fmt(s12.get('MDD'), '{:.1%}'), fmt(s14.get('MDD'), '{:.1%}')),
        ('Sharpe', fmt(s12.get('Sharpe'), '{:.3f}'), fmt(s14.get('Sharpe'), '{:.3f}')),
        ('Sortino', fmt(s12.get('Sortino'), '{:.3f}'), fmt(s14.get('Sortino'), '{:.3f}')),
        ('Calmar', fmt(s12.get('Calmar'), '{:.2f}'), fmt(s14.get('Calmar'), '{:.2f}')),
        ('Rebals', fmt(s12.get('Rebals'), '{:.0f}'), fmt(s14.get('Rebals'), '{:.0f}')),
        ('Flips', fmt(s12.get('Flips'), '{:.0f}'), fmt(s14.get('Flips'), '{:.0f}')),
    ]

    for label, v12, v14 in rows:
        print(f"│ {label:>20} │ {v12:>20} │ {v14:>20} │")
    print(f"└{'─'*22}┴{'─'*22}┴{'─'*22}┘")

    # Coin comparison
    print("\n┌─── 코인 전략 ───────────────────────────────────────────────┐")
    print(f"│ {'':>20} │ {'V12 (SMA50+InvVol)':>20} │ {'V14 (SMA60+EW+DD)':>20} │")
    print(f"│ {'─'*20} │ {'─'*20} │ {'─'*20} │")

    c12 = coin_results.get('V12_Coin', {})
    c14 = coin_results.get('V14_Coin', {})

    coin_rows = [
        ('Canary', 'BTC>SMA50', 'BTC>SMA60 1%h'),
        ('Health', 'SMA30+M21+V10%', 'M21+M90+V5%'),
        ('Selection', 'Sharpe Top5', 'MarketCap Top5'),
        ('Weighting', 'InvVol', 'Equal Weight'),
        ('DD Exit', 'None', '60d -25%'),
        ('Blacklist', 'None', '-15% 7d'),
        ('Crash', 'None', 'BTC -10%'),
        ('', '', ''),
        ('CAGR', fmt(c12.get('cagr'), '{:+.1%}'), fmt(c14.get('cagr'), '{:+.1%}')),
        ('MDD', fmt(c12.get('mdd'), '{:.1%}'), fmt(c14.get('mdd'), '{:.1%}')),
        ('Sharpe', fmt(c12.get('sharpe'), '{:.3f}'), fmt(c14.get('sharpe'), '{:.3f}')),
        ('Calmar', fmt(c12.get('calmar'), '{:.2f}'), fmt(c14.get('calmar'), '{:.2f}')),
        ('Rebals', fmt(c12.get('rebals'), '{:.0f}'), fmt(c14.get('rebals'), '{:.0f}')),
    ]

    for label, v12, v14 in coin_rows:
        print(f"│ {label:>20} │ {v12:>20} │ {v14:>20} │")
    print(f"└{'─'*22}┴{'─'*22}┴{'─'*22}┘")

    # 60/40 Combined
    print("\n┌─── 60/40 합산 (주식60%+코인40%) ─────────────────────────┐")
    print(f"│ {'':>20} │ {'V12 Combined':>20} │ {'V14 Combined':>20} │")
    print(f"│ {'─'*20} │ {'─'*20} │ {'─'*20} │")

    # Map coin keys to stock-style keys
    c12_mapped = {
        'CAGR': c12.get('cagr', 0), 'MDD': c12.get('mdd', 0),
        'Sharpe': c12.get('sharpe', 0), 'Sortino': c12.get('sortino', 0),
        'Calmar': c12.get('calmar', 0)
    }
    c14_mapped = {
        'CAGR': c14.get('cagr', 0), 'MDD': c14.get('mdd', 0),
        'Sharpe': c14.get('sharpe', 0), 'Sortino': c14.get('sortino', 0),
        'Calmar': c14.get('calmar', 0)
    }

    comb12 = combine_6040(s12, c12_mapped)
    comb14 = combine_6040(s14, c14_mapped)

    comb_rows = [
        ('CAGR', fmt(comb12.get('CAGR'), '{:+.1%}'), fmt(comb14.get('CAGR'), '{:+.1%}')),
        ('MDD (approx)', fmt(comb12.get('MDD'), '{:.1%}'), fmt(comb14.get('MDD'), '{:.1%}')),
        ('Sharpe (wgt)', fmt(comb12.get('Sharpe'), '{:.3f}'), fmt(comb14.get('Sharpe'), '{:.3f}')),
        ('Calmar (wgt)', fmt(comb12.get('Calmar'), '{:.2f}'), fmt(comb14.get('Calmar'), '{:.2f}')),
    ]

    for label, v12, v14 in comb_rows:
        print(f"│ {label:>20} │ {v12:>20} │ {v14:>20} │")
    print(f"└{'─'*22}┴{'─'*22}┴{'─'*22}┘")

    # Delta summary
    print("\n📈 V14 개선폭:")
    for key, label in [('Sharpe', 'Sharpe'), ('CAGR', 'CAGR'), ('MDD', 'MDD')]:
        s_delta = (s14.get(key, 0) or 0) - (s12.get(key, 0) or 0)
        c_delta = (c14_mapped.get(key, 0) or 0) - (c12_mapped.get(key, 0) or 0)
        combo_delta = (comb14.get(key, 0) or 0) - (comb12.get(key, 0) or 0)
        if key == 'MDD':
            # For MDD, positive delta means improvement (less negative)
            print(f"  {label:>8}: Stock {s_delta:+.3f}  Coin {c_delta:+.3f}  60/40 {combo_delta:+.3f}  {'✅' if combo_delta > 0 else '⚠️'}")
        else:
            print(f"  {label:>8}: Stock {s_delta:+.3f}  Coin {c_delta:+.3f}  60/40 {combo_delta:+.3f}  {'✅' if combo_delta > 0 else '⚠️'}")

    # Per-anchor day stock detail
    print("\n📅 주식 전략 앵커별 상세:")
    print(f"  {'Day':>4} │ {'V12 Sharpe':>12} {'V12 CAGR':>10} {'V12 MDD':>10} │ {'V14 Sharpe':>12} {'V14 CAGR':>10} {'V14 MDD':>10}")
    for d in range(1, 5):
        s12d = stock_results.get(f'V12_Stock_d{d}', {})
        s14d = stock_results.get(f'V14_Stock_d{d}', {})
        if s12d and s14d:
            print(f"  {d:>4} │ {s12d.get('Sharpe',0):>12.3f} {s12d.get('CAGR',0):>+10.1%} {s12d.get('MDD',0):>10.1%} │ {s14d.get('Sharpe',0):>12.3f} {s14d.get('CAGR',0):>+10.1%} {s14d.get('MDD',0):>10.1%}")


if __name__ == '__main__':
    stock_results = run_stock_comparison()
    coin_results = run_coin_comparison()
    print_comparison(stock_results, coin_results)
