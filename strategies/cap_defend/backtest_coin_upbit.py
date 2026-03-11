#!/usr/bin/env python3
"""
Upbit KRW Market Only Backtest with Trigger-Based Rebalancing
- Daily monitoring: Health failure → immediate rebal, Turnover 30%+ → immediate rebal
- Monthly scheduled rebalancing
- Excludes stablecoins and real-asset-backed tokens
"""

import os, sys, warnings, random
import numpy as np
import pandas as pd
from collections import defaultdict

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from backtest_coin_strategy import (
    load_universe, load_all_prices, calc_metrics,
    calc_sharpe, calc_rsi, calc_macd_hist, calc_bb_pctb,
    calc_ret, get_volatility, STABLECOINS
)

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')

EXCLUDE_SYMBOLS = STABLECOINS | {
    'PAXG', 'XAUT', 'WBTC', 'USD1', 'USDE',
}


def get_upbit_krw_symbols():
    try:
        import requests
        url = 'https://api.upbit.com/v1/market/all?is_details=false'
        r = requests.get(url, timeout=10)
        markets = r.json()
        symbols = {m['market'].replace('KRW-', '') for m in markets if m['market'].startswith('KRW-')}
        symbols -= EXCLUDE_SYMBOLS
        return symbols
    except Exception as e:
        print(f"  Warning: Could not fetch Upbit markets: {e}")
        return None


def filter_universe(universe_map, allowed_symbols):
    filtered = {}
    for month_key, tickers in universe_map.items():
        filtered_tickers = []
        for t in tickers:
            sym = t.replace('-USD', '')
            if sym in allowed_symbols and sym not in EXCLUDE_SYMBOLS:
                filtered_tickers.append(t)
        filtered[month_key] = filtered_tickers
    return filtered


def check_health(ticker, prices, global_idx, health_type, short_ma, long_ma):
    """Check if a single coin passes health filter."""
    if ticker not in prices:
        return False
    close = prices[ticker]['Close'].iloc[:global_idx+1]

    if health_type == 'baseline':
        if len(close) < 90: return False
        cur = close.iloc[-1]
        sma30 = close.rolling(30).mean().iloc[-1]
        mom21 = calc_ret(close, 21)
        vol90 = get_volatility(close, 90)
        return cur > sma30 and mom21 > 0 and vol90 <= 0.10
    elif health_type == 'dual_ma':
        if len(close) < long_ma: return False
        cur = close.iloc[-1]
        sma_s = close.rolling(short_ma).mean().iloc[-1]
        sma_l = close.rolling(long_ma).mean().iloc[-1]
        return cur > sma_s and sma_s > sma_l
    return False


def compute_target_portfolio(prices, universe_map, date, global_idx,
                              canary_type, health_type, breadth_thr,
                              short_ma, long_ma, n_picks,
                              exclude_top_pct):
    """Compute target portfolio weights for a given date."""
    btc = prices.get('BTC-USD')
    if btc is None:
        return None, []

    month_key = date.strftime('%Y-%m') + '-01'
    uni_tickers = []
    for mk in sorted(universe_map.keys(), reverse=True):
        if mk <= month_key:
            uni_tickers = universe_map[mk]
            break
    if not uni_tickers and universe_map:
        uni_tickers = list(universe_map.values())[0]
    uni_clean = [t.replace('-USD', '') for t in uni_tickers
                if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]

    # Canary
    risk_on = False
    if canary_type == 'btc_sma50':
        btc_close = btc['Close'].iloc[:global_idx+1]
        if len(btc_close) >= 50:
            risk_on = btc_close.iloc[-1] > btc_close.rolling(50).mean().iloc[-1]
    elif canary_type == 'market_breadth':
        count_above = count_total = 0
        for sym in uni_clean:
            ticker = f"{sym}-USD"
            if ticker not in prices: continue
            p = prices[ticker]['Close'].iloc[:global_idx+1]
            if len(p) < 50: continue
            count_total += 1
            if p.iloc[-1] > p.rolling(50).mean().iloc[-1]:
                count_above += 1
        if count_total > 0:
            risk_on = (count_above / count_total) > breadth_thr

    if not risk_on:
        return {'CASH': 1.0}, []

    # Health filter
    healthy = []
    for sym in uni_clean:
        ticker = f"{sym}-USD"
        if check_health(ticker, prices, global_idx, health_type, short_ma, long_ma):
            healthy.append(ticker)

    # Outlier exclusion
    if exclude_top_pct > 0 and healthy:
        returns_list = []
        for t in healthy:
            close = prices[t]['Close'].iloc[:global_idx+1]
            ret90 = calc_ret(close, 90) if len(close) >= 90 else 0
            returns_list.append((t, ret90))
        returns_list.sort(key=lambda x: x[1], reverse=True)
        n_exclude = max(1, int(len(returns_list) * exclude_top_pct))
        excluded = set(t for t, _ in returns_list[:n_exclude])
        healthy = [t for t in healthy if t not in excluded]

    if not healthy:
        return {'CASH': 1.0}, healthy

    # Scoring
    scores = []
    for t in healthy:
        close = prices[t]['Close'].iloc[:global_idx+1]
        if len(close) < 252: continue
        base = calc_sharpe(close, 126) + calc_sharpe(close, 252)
        rsi_val = calc_rsi(close)
        macd_h = calc_macd_hist(close)
        pctb = calc_bb_pctb(close)
        if pd.notna(rsi_val) and 45 <= rsi_val <= 70: base += 0.2
        if pd.notna(macd_h) and macd_h > 0: base += 0.2
        if pd.notna(pctb) and pctb > 0.5: base += 0.2
        scores.append((t, base))

    scores.sort(key=lambda x: x[1], reverse=True)
    picks = [t for t, _ in scores[:n_picks]]

    if not picks:
        return {'CASH': 1.0}, healthy

    # InvVol weighting
    vols = {}
    for t in picks:
        close = prices[t]['Close'].iloc[:global_idx+1]
        v = get_volatility(close, 90)
        if v > 0: vols[t] = v
    if vols:
        inv = {t: 1/v for t, v in vols.items()}
        tot = sum(inv.values())
        weights = {t: w/tot for t, w in inv.items()}
    else:
        weights = {t: 1/len(picks) for t in picks}

    return weights, healthy


def execute_rebalance(holdings, cash, weights, prices, date, tx_cost):
    """Sell all holdings and buy new portfolio according to weights."""
    sell_val = 0
    for t, units in holdings.items():
        if t in prices:
            idx = prices[t].index.get_indexer([date], method='ffill')[0]
            if idx >= 0:
                sell_val += units * prices[t]['Close'].iloc[idx]

    total_val = (cash + sell_val) * (1 - tx_cost)
    new_holdings = {}
    new_cash = 0

    if 'CASH' in weights and weights['CASH'] == 1.0:
        return {}, total_val

    for t, w in weights.items():
        if t == 'CASH':
            continue
        if t in prices:
            idx2 = prices[t].index.get_indexer([date], method='ffill')[0]
            if idx2 >= 0:
                price = prices[t]['Close'].iloc[idx2]
                if price > 0:
                    alloc = total_val * w * (1 - tx_cost)
                    new_holdings[t] = alloc / price
                    new_cash += total_val * w - alloc

    return new_holdings, new_cash


def get_current_weights(holdings, cash, prices, date):
    """Calculate current portfolio weights."""
    port_val = cash
    holding_vals = {}
    for t, units in holdings.items():
        if t in prices:
            idx = prices[t].index.get_indexer([date], method='ffill')[0]
            if idx >= 0:
                val = units * prices[t]['Close'].iloc[idx]
                holding_vals[t] = val
                port_val += val

    if port_val <= 0:
        return {}, 0

    weights = {t: v / port_val for t, v in holding_vals.items()}
    weights['CASH'] = cash / port_val
    return weights, port_val


def calc_turnover(current_weights, target_weights):
    """Calculate turnover between current and target weights."""
    all_keys = set(current_weights.keys()) | set(target_weights.keys())
    turnover = sum(abs(target_weights.get(k, 0) - current_weights.get(k, 0)) for k in all_keys) / 2
    return turnover


def run_backtest(prices, universe_map, canary_type='btc_sma50',
                 health_type='baseline', breadth_thr=0.40,
                 short_ma=15, long_ma=50,
                 start_date='2019-01-01', end_date='2025-12-31',
                 tx_cost=0.002, n_picks=5,
                 exclude_top_pct=0.0,
                 # Rebalancing mode
                 rebal_mode='trigger',  # 'monthly' or 'trigger'
                 turnover_thr=0.30,
                 trigger_cooldown=0,  # Min days between trigger rebalances
                 # Tracking
                 track_picks=False, track_rebals=False):
    """
    Backtest runner with trigger-based rebalancing.
    rebal_mode='trigger': daily health check + turnover check + monthly scheduled
    rebal_mode='monthly': monthly only (old behavior)
    """
    btc = prices.get('BTC-USD')
    if btc is None: return None

    all_dates = btc.index[(btc.index >= start_date) & (btc.index <= end_date)]
    if len(all_dates) == 0: return None

    holdings = {}
    cash = 10000
    portfolio_values = []
    prev_month = None
    pick_history = []
    rebal_log = []  # Track rebalancing events
    last_target = None  # Cache target portfolio for turnover check
    last_healthy = []
    last_canary = None  # Track canary state for flip detection
    last_trigger_date = None  # Cooldown tracking

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)

        # Portfolio value
        port_val = cash
        for t, units in holdings.items():
            if t in prices:
                idx = prices[t].index.get_indexer([date], method='ffill')[0]
                if idx >= 0:
                    port_val += units * prices[t]['Close'].iloc[idx]

        current_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and current_month != prev_month)

        # Determine if we should rebalance
        do_rebal = False
        rebal_reason = None

        # Check current canary state (needed for flip detection)
        cur_canary = None
        if canary_type == 'btc_sma50':
            btc_close = btc['Close'].iloc[:global_idx+1]
            if len(btc_close) >= 50:
                cur_canary = btc_close.iloc[-1] > btc_close.rolling(50).mean().iloc[-1]
        elif canary_type == 'market_breadth':
            month_key = date.strftime('%Y-%m') + '-01'
            uni_tickers = []
            for mk in sorted(universe_map.keys(), reverse=True):
                if mk <= month_key:
                    uni_tickers = universe_map[mk]
                    break
            if not uni_tickers and universe_map:
                uni_tickers = list(universe_map.values())[0]
            uni_clean_tmp = [t.replace('-USD', '') for t in uni_tickers
                            if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]
            count_above = count_total = 0
            for sym in uni_clean_tmp:
                ticker = f"{sym}-USD"
                if ticker not in prices: continue
                p = prices[ticker]['Close'].iloc[:global_idx+1]
                if len(p) < 50: continue
                count_total += 1
                if p.iloc[-1] > p.rolling(50).mean().iloc[-1]:
                    count_above += 1
            if count_total > 0:
                cur_canary = (count_above / count_total) > breadth_thr

        # Cooldown check helper
        def cooldown_ok():
            if trigger_cooldown <= 0: return True
            if last_trigger_date is None: return True
            return (date - last_trigger_date).days >= trigger_cooldown

        if i == 0:
            do_rebal = True
            rebal_reason = 'init'
        elif rebal_mode == 'monthly':
            if is_month_change:
                do_rebal = True
                rebal_reason = 'monthly'
        elif rebal_mode == 'trigger':
            # Monthly scheduled (always, no cooldown)
            if is_month_change:
                do_rebal = True
                rebal_reason = 'monthly'

            # Trigger 1: Canary signal flip (risk-on ↔ risk-off)
            if not do_rebal and last_canary is not None and cur_canary is not None:
                if cur_canary != last_canary and cooldown_ok():
                    do_rebal = True
                    rebal_reason = f'canary_flip:{"ON" if cur_canary else "OFF"}'

            # Trigger 2: Held coin fails Health AND not in new target
            if not do_rebal and holdings and cooldown_ok():
                sick_coins = []
                for t in list(holdings.keys()):
                    if not check_health(t, prices, global_idx, health_type, short_ma, long_ma):
                        sick_coins.append(t)
                if sick_coins:
                    peek_target, _ = compute_target_portfolio(
                        prices, universe_map, date, global_idx,
                        canary_type, health_type, breadth_thr,
                        short_ma, long_ma, n_picks, exclude_top_pct
                    )
                    if peek_target is not None:
                        for t in sick_coins:
                            if t not in peek_target:
                                do_rebal = True
                                rebal_reason = f'health_fail:{t.replace("-USD","")}'
                                break

            # Trigger 3: Turnover exceeds threshold
            if not do_rebal and last_target is not None and cooldown_ok():
                cur_w, _ = get_current_weights(holdings, cash, prices, date)
                turnover = calc_turnover(cur_w, last_target)
                if turnover >= turnover_thr:
                    do_rebal = True
                    rebal_reason = f'turnover:{turnover:.1%}'

        if do_rebal:
            target, healthy = compute_target_portfolio(
                prices, universe_map, date, global_idx,
                canary_type, health_type, breadth_thr,
                short_ma, long_ma, n_picks, exclude_top_pct
            )
            if target is not None:
                last_target = target
                last_healthy = healthy

                if track_picks:
                    coins = [t for t in target if t != 'CASH']
                    if coins:
                        pick_history.append({'date': date, 'picks': coins})

                if track_rebals:
                    rebal_log.append({'date': date, 'reason': rebal_reason,
                                      'target': dict(target)})

                holdings, cash = execute_rebalance(
                    holdings, cash, target, prices, date, tx_cost
                )
                if rebal_reason not in ('init', 'monthly'):
                    last_trigger_date = date

        last_canary = cur_canary
        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    result = pd.DataFrame(portfolio_values).set_index('Date')
    if track_picks:
        result.attrs['pick_history'] = pick_history
    if track_rebals:
        result.attrs['rebal_log'] = rebal_log
    return result


def main():
    print("=" * 100)
    print("  UPBIT KRW MARKET BACKTEST: TRIGGER vs MONTHLY REBALANCING")
    print("  (Excluding stablecoins & real-asset-backed tokens)")
    print("=" * 100)

    # 1. Get Upbit KRW market list
    print("\n[1] Fetching Upbit KRW market list...")
    upbit_symbols = get_upbit_krw_symbols()
    if upbit_symbols is None:
        print("  FAILED. Exiting.")
        return
    print(f"  {len(upbit_symbols)} tradable symbols")

    # 2. Load and filter universe
    print("\n[2] Loading universe & prices...")
    universe_map = load_universe()
    filtered_map = filter_universe(universe_map, upbit_symbols)

    all_tickers = set()
    for mt in universe_map.values():
        for t in mt:
            s = t.replace('-USD', '')
            if s not in EXCLUDE_SYMBOLS and s in upbit_symbols:
                all_tickers.add(t)
    all_tickers.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(all_tickers)
    print(f"  {len(prices)} tickers loaded")

    baseline_kw = dict(canary_type='btc_sma50', health_type='baseline')
    proposed_kw = dict(canary_type='market_breadth', health_type='dual_ma',
                       breadth_thr=0.40, short_ma=15, long_ma=50)

    years = list(range(2019, 2026))

    # ═══════════════════════════════════════════════════════════════════
    # TEST 1: TRIGGER vs MONTHLY (Baseline)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST 1: REBALANCING MODE COMPARISON")
    print("  Monthly Only vs Trigger-Based (Health fail + Turnover 30%)")
    print("=" * 100)

    configs = [
        ('BL Monthly',      {**baseline_kw, 'rebal_mode': 'monthly'}),
        ('BL Trig(0d)',     {**baseline_kw, 'rebal_mode': 'trigger', 'trigger_cooldown': 0}),
        ('BL Trig(7d)',     {**baseline_kw, 'rebal_mode': 'trigger', 'trigger_cooldown': 7}),
        ('BL Trig(14d)',    {**baseline_kw, 'rebal_mode': 'trigger', 'trigger_cooldown': 14}),
        ('B40 Monthly',     {**proposed_kw, 'rebal_mode': 'monthly'}),
        ('B40 Trig(0d)',    {**proposed_kw, 'rebal_mode': 'trigger', 'trigger_cooldown': 0}),
        ('B40 Trig(7d)',    {**proposed_kw, 'rebal_mode': 'trigger', 'trigger_cooldown': 7}),
        ('B40 Trig(14d)',   {**proposed_kw, 'rebal_mode': 'trigger', 'trigger_cooldown': 14}),
    ]

    print(f"\n  {'Strategy':<16}", end='')
    for y in years:
        print(f" {y:>8}", end='')
    print(f" {'Overall':>10} {'MDD':>8} {'CAGR':>8}")
    print(f"  {'-'*115}")

    overall = {}
    for name, kw in configs:
        print(f"  {name:<16}", end='')
        pv_full = run_backtest(prices, filtered_map, **kw)
        for y in years:
            pv = run_backtest(prices, filtered_map, start_date=f'{y}-01-01', end_date=f'{y}-12-31', **kw)
            if pv is not None and len(pv) > 10:
                m = calc_metrics(pv)
                print(f" {m['Sharpe']:>8.3f}", end='')
            else:
                print(f" {'N/A':>8}", end='')
        if pv_full is not None and len(pv_full) > 10:
            m_full = calc_metrics(pv_full)
            overall[name] = m_full
            print(f" {m_full['Sharpe']:>10.3f} {m_full['MDD']:>7.1%} {m_full['CAGR']:>+7.1%}")
        else:
            print()

    # BTC B&H reference
    btc = prices.get('BTC-USD')
    if btc is not None:
        btc_dates = btc.index[(btc.index >= '2019-01-01') & (btc.index <= '2025-12-31')]
        if len(btc_dates) > 10:
            btc_pv = pd.DataFrame({
                'Date': btc_dates,
                'Value': 10000 * btc.loc[btc_dates, 'Close'] / btc.loc[btc_dates, 'Close'].iloc[0]
            }).set_index('Date')
            btc_m = calc_metrics(btc_pv)
            print(f"  {'BTC B&H':<16}", end='')
            for y in years:
                mask = btc_pv.index.year == y
                if mask.sum() > 10:
                    ym = calc_metrics(btc_pv[mask])
                    print(f" {ym['Sharpe']:>8.3f}", end='')
                else:
                    print(f" {'N/A':>8}", end='')
            print(f" {btc_m['Sharpe']:>10.3f} {btc_m['MDD']:>7.1%} {btc_m['CAGR']:>+7.1%}")

    # ═══════════════════════════════════════════════════════════════════
    # TEST 2: REBALANCING EVENT ANALYSIS (with cooldown variants)
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  TEST 2: REBALANCING EVENT ANALYSIS")
    print("  Triggers: canary_flip + health_fail + turnover30% + monthly")
    print("=" * 100)

    event_configs = [
        ('BL 0d-cool',  {**baseline_kw, 'rebal_mode': 'trigger', 'trigger_cooldown': 0}),
        ('BL 7d-cool',  {**baseline_kw, 'rebal_mode': 'trigger', 'trigger_cooldown': 7}),
        ('B40 0d-cool', {**proposed_kw, 'rebal_mode': 'trigger', 'trigger_cooldown': 0}),
        ('B40 7d-cool', {**proposed_kw, 'rebal_mode': 'trigger', 'trigger_cooldown': 7}),
    ]

    for name, kw in event_configs:
        pv = run_backtest(prices, filtered_map, track_rebals=True, **kw)
        if pv is None: continue
        log = pv.attrs.get('rebal_log', [])
        total = len(log)
        by_reason = defaultdict(int)
        for entry in log:
            reason = entry['reason']
            if reason.startswith('health_fail'): reason = 'health_fail'
            elif reason.startswith('turnover'): reason = 'turnover'
            elif reason.startswith('canary_flip'): reason = 'canary_flip'
            by_reason[reason] += 1

        print(f"\n  {name}: {total} total rebalances over {len(years)} years")
        for reason, count in sorted(by_reason.items(), key=lambda x: x[1], reverse=True):
            pct = count / total * 100 if total else 0
            print(f"    {reason:<16} {count:>4} ({pct:>5.1f}%)")

        yearly_counts = defaultdict(lambda: defaultdict(int))
        for entry in log:
            yr = entry['date'].year
            reason = entry['reason']
            if reason.startswith('health_fail'): reason = 'health_fail'
            elif reason.startswith('turnover'): reason = 'turnover'
            elif reason.startswith('canary_flip'): reason = 'canary_flip'
            yearly_counts[yr][reason] += 1

        print(f"\n  {'Year':<8}", end='')
        reasons = ['monthly', 'canary_flip', 'health_fail', 'turnover', 'init']
        for r in reasons:
            print(f" {r:>12}", end='')
        print(f" {'Total':>8}")
        print(f"  {'-'*75}")
        for y in years:
            print(f"  {y:<8}", end='')
            yr_total = 0
            for r in reasons:
                cnt = yearly_counts[y].get(r, 0)
                yr_total += cnt
                print(f" {cnt:>12}", end='')
            print(f" {yr_total:>8}")

    # ═══════════════════════════════════════════════════════════════════
    # FINAL SUMMARY
    # ═══════════════════════════════════════════════════════════════════
    print("\n" + "=" * 100)
    print("  FINAL SUMMARY")
    print("=" * 100)

    print(f"\n  {'Strategy':<16} {'Sharpe':>8} {'MDD':>8} {'CAGR':>8}")
    print(f"  {'-'*45}")
    for name in sorted(overall.keys()):
        m = overall[name]
        print(f"  {name:<16} {m['Sharpe']:>8.3f} {m['MDD']:>7.1%} {m['CAGR']:>+7.1%}")

    # Find best config for each strategy type
    print(f"\n  Best configs:")
    for prefix in ['BL', 'B40']:
        best_name = max((n for n in overall if n.startswith(prefix)),
                       key=lambda n: overall[n]['Sharpe'], default=None)
        if best_name:
            m = overall[best_name]
            print(f"    {prefix}: {best_name} (Sharpe={m['Sharpe']:.3f} MDD={m['MDD']:.1%} CAGR={m['CAGR']:+.1%})")


if __name__ == '__main__':
    main()
