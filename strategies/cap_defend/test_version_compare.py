#!/usr/bin/env python3
"""V12~V15 코인 전략 비교 — 단일슬롯 + 3트랜치.

버전별 핵심 차이 (코인만):
  V12: SMA50, SMA30+Mom21+Vol10%, Sharpe scoring, 1/Vol, Top50, 월간
  V13: SMA50, SMA30+Mom21+Vol10%, Multi Bonus, 1/Vol, Top50, 월간
  V14: SMA60+1%hyst, Mom21+Mom90+Vol5%, 시총순 Top5 EW, DD/BL/Crash, Top40, 3트랜치
  V15: V14 코인과 동일 (주식만 변경)

비교:
  A. 단일슬롯 월간 (10-anchor avg)
  B. 3트랜치 Method B (10-anchor avg)
"""

import os, sys, time
import numpy as np, pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine_v15 import (
    V15Params, Tranche, load_data,
    resolve_canary, check_crash, get_blacklist, check_health,
    check_dd_exit_tranche, execute_rebalance,
    sell_coin_from_tranche, sell_all_from_tranche, check_drift,
    get_universe_for_date, _close_to, calc_ret, get_vol, get_sma,
    get_price, calc_metrics, calc_yearly_metrics,
    _get_daily_returns, _bl_cache,
)


# ─── V12/V13 Specific Functions ───────────────────────────────────

def check_health_v12(ticker, prices, date):
    """V12/V13 Health: Price > SMA(30) AND Mom(21) > 0 AND Vol(90) <= 10%."""
    s = _close_to(ticker, prices, date)
    if len(s) < 90:
        return False
    price = s.iloc[-1]
    sma30 = s.rolling(30).mean().iloc[-1] if len(s) >= 30 else price
    mom21 = calc_ret(s, 21)
    vol90 = get_vol(s, 90)
    return price > sma30 and mom21 > 0 and vol90 <= 0.10


def score_sharpe_v12(ticker, prices, date):
    """V12 Scoring: Sharpe(126d) + Sharpe(252d)."""
    s = _close_to(ticker, prices, date)
    sh126 = _sharpe(s, 126)
    sh252 = _sharpe(s, 252)
    return sh126 + sh252


def score_multi_bonus_v13(ticker, prices, date):
    """V13 Multi Bonus: Sharpe(126)+Sharpe(252) + RSI/MACD/BB bonuses."""
    s = _close_to(ticker, prices, date)
    base = _sharpe(s, 126) + _sharpe(s, 252)
    if len(s) < 252:
        return base

    # RSI(14) bonus
    delta = s.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] > 0 else 100
    rsi = 100 - 100 / (1 + rs)
    bonus = 0
    if 45 <= rsi <= 70:
        bonus += 0.2

    # MACD bonus
    ema12 = s.ewm(span=12).mean()
    ema26 = s.ewm(span=26).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9).mean()
    hist = macd.iloc[-1] - signal.iloc[-1]
    if hist > 0:
        bonus += 0.2

    # BB %B bonus
    sma20 = s.rolling(20).mean()
    std20 = s.rolling(20).std()
    upper = sma20 + 2 * std20
    lower = sma20 - 2 * std20
    bb_range = upper.iloc[-1] - lower.iloc[-1]
    if bb_range > 0:
        pct_b = (s.iloc[-1] - lower.iloc[-1]) / bb_range
        if pct_b > 0.5:
            bonus += 0.2

    return base + bonus


def _sharpe(s, d):
    if len(s) < d + 1:
        return 0
    ret = s.pct_change().iloc[-d:]
    return (ret.mean() / ret.std()) * np.sqrt(365) if ret.std() > 0 else 0


def weight_inv_vol(picks, prices, date):
    """Inverse volatility weighting."""
    vols = {}
    for t in picks:
        s = _close_to(t, prices, date)
        v = get_vol(s, 90) if len(s) >= 91 else 0.01
        vols[t] = max(v, 0.001)
    inv = {t: 1.0 / v for t, v in vols.items()}
    total = sum(inv.values())
    return {t: w / total for t, w in inv.items()} if total > 0 else {t: 1.0/len(picks) for t in picks}


# ─── Generic Backtest Runner ──────────────────────────────────────

def run_generic_backtest(prices, universe_map, version='V15',
                         n_tranches=3, tranche_days=(1, 10, 19),
                         start='2018-01-01', end='2025-06-30',
                         tx_cost=0.004, capital=10000.0):
    """Run backtest for any version with configurable tranche count."""

    # Version-specific params
    if version in ('V12', 'V13'):
        canary_sma, canary_hyst = 50, 0.0
        top_n, n_picks = 50, 5
        has_dd, has_bl, has_crash = False, False, False
        vol_cap_health = 0.10
    else:  # V14/V15
        canary_sma, canary_hyst = 60, 0.01
        top_n, n_picks = 40, 5
        has_dd, has_bl, has_crash = True, True, True
        vol_cap_health = 0.05

    # Load universe with correct top_n
    from strategy_engine import load_universe, filter_universe, load_all_prices
    um = load_universe()
    fm = filter_universe(um, top_n)
    # Ensure we have all needed prices (use passed prices)

    capital_per = capital / n_tranches
    tranches = [Tranche(i, tranche_days[i % len(tranche_days)], capital_per)
                for i in range(n_tranches)]

    # Global state
    prev_canary = False
    canary_on = False
    canary_on_date = None
    crash_cooldown = 0
    prev_month = None

    rebal_count = 0
    dd_exit_count = 0

    btc = prices.get('BTC-USD')
    if btc is None:
        return None
    all_dates = btc.index[(btc.index >= start) & (btc.index <= end)]
    if len(all_dates) < 2:
        return None

    portfolio_values = []

    for date in all_dates:
        cur_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and cur_month != prev_month)
        is_first = (prev_month is None)

        if is_month_change:
            for tr in tranches:
                tr.anchor_done_this_month = False

        # Crash Breaker (V14/V15 only)
        crash_just_ended = False
        if has_crash:
            if crash_cooldown > 0:
                crash_cooldown -= 1
                if crash_cooldown == 0:
                    if check_crash(prices, date, V15Params()):
                        crash_cooldown = 3
                    else:
                        crash_just_ended = True
            elif not is_first and check_crash(prices, date, V15Params()):
                for tr in tranches:
                    sell_all_from_tranche(tr, prices, date, tx_cost)
                crash_cooldown = 3

        # Canary (always update)
        btc_s = _close_to('BTC-USD', prices, date)
        if len(btc_s) >= canary_sma:
            price = btc_s.iloc[-1]
            sma = btc_s.rolling(canary_sma).mean().iloc[-1]
            if not np.isnan(sma):
                if canary_hyst > 0:
                    if prev_canary:
                        canary_on = not (price < sma * (1 - canary_hyst))
                    else:
                        canary_on = price > sma * (1 + canary_hyst)
                else:
                    canary_on = price > sma

        signal_flipped = (prev_canary is not None and canary_on != prev_canary)
        flip_on = signal_flipped and canary_on

        if flip_on:
            canary_on_date = date
            for tr in tranches:
                tr.post_flip_refreshed = False

        # Skip if crash cooldown
        if crash_cooldown > 0 and not crash_just_ended:
            pv = sum(tr.value(prices, date) for tr in tranches)
            portfolio_values.append({'Date': date, 'Value': pv})
            prev_canary = canary_on
            prev_month = cur_month
            continue

        # Blacklist (V14/V15 only)
        universe = get_universe_for_date(fm, date)
        blacklist = set()
        if has_bl:
            for t in universe:
                rets = _get_daily_returns(t, prices)
                if len(rets) == 0:
                    continue
                mask = rets.index <= date
                if mask.sum() < 7:
                    continue
                recent = rets.loc[mask].iloc[-7:]
                if (recent <= -0.15).any():
                    blacklist.add(t)

        # Per-tranche operations
        for tr in tranches:
            # DD Exit (V14/V15 only)
            tr.dd_exited_today = set()
            if has_dd and canary_on:
                for t in list(tr.holdings.keys()):
                    if tr.holdings.get(t, 0) <= 0:
                        continue
                    s = _close_to(t, prices, date)
                    if len(s) < 60:
                        continue
                    peak = s.iloc[-60:].max()
                    if peak > 0 and (s.iloc[-1] / peak - 1) <= -0.25:
                        sell_coin_from_tranche(tr, t, prices, date, tx_cost)
                        tr.dd_exited_today.add(t)
                        dd_exit_count += 1

            # Rebalance triggers
            do_rebal = False
            reason = 'none'

            if is_first:
                do_rebal = True; reason = 'init'
            if not do_rebal and signal_flipped:
                do_rebal = True; reason = 'flip'
            if not do_rebal and crash_just_ended:
                do_rebal = True; reason = 'crash_end'

            # PFD5 (V14/V15)
            if (not do_rebal and has_dd and canary_on
                    and not tr.post_flip_refreshed and canary_on_date):
                if (date - canary_on_date).days >= 5:
                    do_rebal = True; reason = 'pfd5'
                    tr.post_flip_refreshed = True

            # Monthly anchor
            if (not do_rebal and not tr.anchor_done_this_month
                    and date.day >= tr.anchor_day and not is_first):
                do_rebal = True; reason = 'monthly'

            if do_rebal:
                if canary_on:
                    # Compute targets based on version
                    candidates = [t for t in universe
                                  if t not in blacklist
                                  and t not in tr.dd_exited_today]

                    # Health filter
                    healthy = []
                    for t in candidates:
                        if version in ('V12', 'V13'):
                            ok = check_health_v12(t, prices, date)
                        else:
                            ok = check_health(t, prices, date, V15Params())
                        if ok:
                            healthy.append(t)

                    # Selection
                    if version == 'V12':
                        scored = [(t, score_sharpe_v12(t, prices, date)) for t in healthy]
                        scored.sort(key=lambda x: -x[1])
                        picks = [t for t, _ in scored[:n_picks]]
                    elif version == 'V13':
                        scored = [(t, score_multi_bonus_v13(t, prices, date)) for t in healthy]
                        scored.sort(key=lambda x: -x[1])
                        picks = [t for t, _ in scored[:n_picks]]
                    else:  # V14/V15: market cap order
                        picks = healthy[:n_picks]

                    if picks:
                        # Weighting
                        if version in ('V12', 'V13'):
                            target_w = weight_inv_vol(picks, prices, date)
                        else:
                            target_w = {t: 1.0 / len(picks) for t in picks}

                        execute_rebalance(tr, target_w, prices, date, tx_cost)
                        tr.prev_picks = list(target_w.keys())
                    else:
                        sell_all_from_tranche(tr, prices, date, tx_cost)
                        tr.prev_picks = []
                else:
                    sell_all_from_tranche(tr, prices, date, tx_cost)
                    tr.prev_picks = []

                if reason == 'monthly':
                    tr.anchor_done_this_month = True
                rebal_count += 1

        pv = sum(tr.value(prices, date) for tr in tranches)
        portfolio_values.append({'Date': date, 'Value': pv})
        prev_canary = canary_on
        prev_month = cur_month

    if not portfolio_values:
        return None
    pvdf = pd.DataFrame(portfolio_values).set_index('Date')
    m = calc_metrics(pvdf)
    ym = calc_yearly_metrics(pvdf)
    return {'metrics': m, 'yearly': ym, 'pv': pvdf,
            'rebal_count': rebal_count, 'dd_exit_count': dd_exit_count}


def main():
    print('V12~V15 코인 전략 비교 — 단일슬롯 + 3트랜치')
    print('=' * 100)

    print('Loading data...')
    prices, universe = load_data(top_n=50)  # Top 50 for V12/V13 compatibility
    print(f'  {len(prices)} tickers loaded\n')

    global _bl_cache
    _bl_cache.clear()

    VERSIONS = ['V12', 'V13', 'V14']
    MODES = [
        ('Single', 1, (1,)),
        ('3-Tranche', 3, (1, 10, 19)),
    ]
    ANCHOR_OFFSETS = list(range(10))

    results = {}

    for ver in VERSIONS:
        for mode_name, n_tr, base_days in MODES:
            key = f'{ver} {mode_name}'
            print(f'Running {key}...')
            sharpes, cagrs, mdds = [], [], []

            for offset in ANCHOR_OFFSETS:
                _bl_cache.clear()
                days = tuple((d - 1 + offset) % 28 + 1 for d in base_days)
                r = run_generic_backtest(
                    prices, universe, version=ver,
                    n_tranches=n_tr, tranche_days=days)
                if r:
                    sharpes.append(r['metrics']['Sharpe'])
                    cagrs.append(r['metrics']['CAGR'])
                    mdds.append(r['metrics']['MDD'])

            if sharpes:
                results[key] = {
                    'Sharpe': np.mean(sharpes), 'σ': np.std(sharpes),
                    'CAGR': np.mean(cagrs), 'MDD': np.mean(mdds),
                    'Calmar': abs(np.mean(cagrs) / np.mean(mdds)) if np.mean(mdds) != 0 else 0,
                }
                # Get yearly from last run
                if r and 'yearly' in r:
                    results[key]['yearly'] = {yr: ym['CAGR'] for yr, ym in r['yearly'].items()}
                    results[key]['rebal'] = r['rebal_count']

    # Print results
    print(f'\n{"=" * 120}')
    print(f'  V12~V15 코인 전략 비교 (10-anchor 평균)')
    print(f'{"=" * 120}')
    print(f'{"Config":<20} {"Sharpe":>8} {"σ(S)":>6} {"CAGR":>8} {"MDD":>8} {"Calmar":>8} {"Rebal":>6}  {"2019":>7} {"2021":>7} {"2022":>7} {"2024":>7}')
    print('-' * 120)

    for key in sorted(results.keys()):
        r = results[key]
        yr = r.get('yearly', {})
        yr_str = ''
        for y in [2019, 2021, 2022, 2024]:
            yr_str += f' {yr.get(y, 0):>+6.1%}' if y in yr else f' {"---":>6}'
        rebal = r.get('rebal', 0)
        print(f'{key:<20} {r["Sharpe"]:>8.3f} {r["σ"]:>5.3f} {r["CAGR"]:>+7.1%} '
              f'{r["MDD"]:>7.1%} {r["Calmar"]:>8.2f} {rebal:>6}{yr_str}')

    # BTC B&H reference
    btc = prices['BTC-USD']['Close']
    btc = btc[(btc.index >= '2018-01-01') & (btc.index <= '2025-06-30')]
    years = (btc.index[-1] - btc.index[0]).days / 365.25
    cagr = (btc.iloc[-1] / btc.iloc[0]) ** (1/years) - 1
    mdd = (btc / btc.cummax() - 1).min()
    ret = btc.pct_change().dropna()
    sharpe = ret.mean() / ret.std() * np.sqrt(365)
    print(f'{"BTC B&H":<20} {sharpe:>8.3f} {"":>6} {cagr:>+7.1%} {mdd:>7.1%} {abs(cagr/mdd):>8.2f}')


if __name__ == '__main__':
    main()
