#!/usr/bin/env python3
"""V15 Coin Strategy Backtest Engine — True 3-Tranche Implementation.

Strategy Spec (V15 Final):
  Universe: T40 (시총 Top 40, stablecoins/wrapped excluded)
  Canary (K): BTC > SMA(60) with 1% hysteresis
  Health (H): Mom(21)>0 AND Mom(90)>0 AND Vol(90)≤5%
  Crash Breaker (G5): BTC daily -10% → 3 days all-cash
  Selection: 시총순 Top 5, Equal Weight
  Snapshot: Method B — 3 tranches, days 1/10/19 (9-day stagger)
  PFD5: Post-flip 5-day refresh per tranche
  Drift: 10% half-turnover → rebalance per tranche
  Blacklist: -15% daily drop → 7 days excluded (stateless lookback)
  DD Exit: 60-day high -25% → sell that coin to cash (daily check, per tranche)
  TX Cost: 0.4%

Architecture:
  - 3 independent tranches, each with own holdings/cash/anchor day
  - Shared global state: canary, crash breaker cooldown, blacklist
  - Daily loop: global checks first, then per-tranche operations
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional

# Reuse data loading from existing engine
sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    load_universe, filter_universe, load_all_prices,
    get_universe_for_date, _close_to, calc_ret, get_vol,
    get_sma, get_price, calc_metrics, calc_yearly_metrics,
)


# ─── Configuration ─────────────────────────────────────────────────

@dataclass
class V15Params:
    # Universe
    top_n: int = 40

    # Canary
    canary_sma: int = 60
    canary_hyst: float = 0.01       # 1% hysteresis

    # Health
    mom_short: int = 21
    mom_long: int = 90
    vol_cap: float = 0.05

    # Crash Breaker (G5)
    crash_threshold: float = -0.10
    crash_cool_days: int = 3

    # Selection
    n_picks: int = 5

    # Tranche
    n_tranches: int = 3
    tranche_days: Tuple = (1, 10, 19)  # anchor days for each tranche

    # PFD (Post-Flip Delay)
    post_flip_delay: int = 5

    # Drift
    drift_threshold: float = 0.10

    # DD Exit
    dd_lookback: int = 60
    dd_threshold: float = -0.25

    # Blacklist (stateless lookback)
    bl_threshold: float = -0.15
    bl_days: int = 7

    # TX Cost
    tx_cost: float = 0.004

    # Backtest window
    start_date: str = '2018-01-01'
    end_date: str = '2025-06-30'
    initial_capital: float = 10000.0


# ─── Tranche State ─────────────────────────────────────────────────

class Tranche:
    def __init__(self, idx: int, anchor_day: int, cash: float):
        self.idx = idx
        self.anchor_day = anchor_day
        self.holdings: Dict[str, float] = {}  # {ticker: units}
        self.cash = cash
        self.anchor_done_this_month = False  # FIX #2: only for monthly anchor
        self.post_flip_refreshed = True  # starts True (no pending PFD)
        self.prev_picks: List[str] = []
        self.dd_exited_today: set = set()  # FIX #5: DD exit coins excluded from same-day rebal

    def value(self, prices, date) -> float:
        v = self.cash
        for t, units in self.holdings.items():
            p = get_price(t, prices, date)
            if p > 0:
                v += units * p
        return v

    def current_weights(self, prices, date) -> Dict[str, float]:
        pv = self.value(prices, date)
        if pv <= 0:
            return {}
        w = {}
        for t, units in self.holdings.items():
            p = get_price(t, prices, date)
            if p > 0:
                w[t] = (units * p) / pv
        return w


# ─── Signal Functions ──────────────────────────────────────────────

def resolve_canary(prices, date, params: V15Params, prev_on: Optional[bool]) -> bool:
    """BTC > SMA(60) with 1% hysteresis."""
    btc = _close_to('BTC-USD', prices, date)
    if len(btc) < params.canary_sma:
        return False
    price = btc.iloc[-1]
    sma = btc.rolling(params.canary_sma).mean().iloc[-1]
    if np.isnan(sma):
        return False

    hyst = params.canary_hyst
    if prev_on is None:
        return price > sma
    elif prev_on:
        # Currently ON: need price < sma * (1 - hyst) to turn OFF
        return not (price < sma * (1 - hyst))
    else:
        # Currently OFF: need price > sma * (1 + hyst) to turn ON
        return price > sma * (1 + hyst)


def check_crash(prices, date, params: V15Params) -> bool:
    """Check if BTC daily return < crash_threshold (-10%)."""
    btc = _close_to('BTC-USD', prices, date)
    if len(btc) < 2:
        return False
    ret = btc.iloc[-1] / btc.iloc[-2] - 1
    return ret <= params.crash_threshold


_bl_cache: Dict[str, pd.Series] = {}  # ticker → daily returns cache

def _get_daily_returns(ticker, prices) -> pd.Series:
    """Cached daily returns for blacklist check."""
    if ticker not in _bl_cache:
        if ticker in prices:
            _bl_cache[ticker] = prices[ticker]['Close'].pct_change()
        else:
            _bl_cache[ticker] = pd.Series(dtype=float)
    return _bl_cache[ticker]

def get_blacklist(prices, universe_tickers, date, params: V15Params) -> set:
    """Stateless blacklist: coins with daily return <= -15% in last 7 days."""
    blacklisted = set()
    for t in universe_tickers:
        rets = _get_daily_returns(t, prices)
        if len(rets) == 0:
            continue
        # Get last bl_days returns up to date
        mask = rets.index <= date
        if mask.sum() < params.bl_days:
            continue
        recent = rets.loc[mask].iloc[-params.bl_days:]
        if (recent <= params.bl_threshold).any():
            blacklisted.add(t)
    return blacklisted


def check_health(ticker, prices, date, params: V15Params) -> bool:
    """Mom(21)>0 AND Mom(90)>0 AND Vol(90)<=vol_cap."""
    s = _close_to(ticker, prices, date)
    if len(s) < params.mom_long + 1:
        return False
    mom_s = calc_ret(s, params.mom_short)
    mom_l = calc_ret(s, params.mom_long)
    vol = get_vol(s, 90)
    return mom_s > 0 and mom_l > 0 and vol <= params.vol_cap


def compute_target_weights(prices, universe_map, date, params: V15Params,
                           blacklist: set,
                           dd_exited: Optional[set] = None) -> Dict[str, float]:
    """Compute target weights: health filter → top N by market cap → EW."""
    universe = get_universe_for_date(universe_map, date)
    # Filter: remove blacklisted + DD-exited today (FIX #5)
    exclude = blacklist | (dd_exited or set())
    candidates = [t for t in universe if t not in exclude]

    # Health check
    healthy = []
    for t in candidates:
        if check_health(t, prices, date, params):
            healthy.append(t)
        if len(healthy) >= params.n_picks:
            break  # market cap order, take first N healthy

    if not healthy:
        return {}  # no healthy coins → cash

    # Equal weight
    w = 1.0 / len(healthy)
    return {t: w for t in healthy}


# ─── DD Exit (per tranche, per coin) ──────────────────────────────

def check_dd_exit_tranche(tranche: Tranche, prices, date,
                          params: V15Params) -> List[str]:
    """Check DD exit for each coin in tranche. Return list of tickers to sell."""
    exits = []
    for t in list(tranche.holdings.keys()):
        if tranche.holdings[t] <= 0:
            continue
        s = _close_to(t, prices, date)
        if len(s) < params.dd_lookback:
            continue
        peak = s.iloc[-params.dd_lookback:].max()
        if peak <= 0:
            continue
        dd = s.iloc[-1] / peak - 1
        if dd <= params.dd_threshold:
            exits.append(t)
    return exits


# ─── Rebalance Execution ──────────────────────────────────────────

def execute_rebalance(tranche: Tranche, target_weights: Dict[str, float],
                      prices, date, tx_cost: float):
    """Sell-first-then-buy rebalance for a tranche."""
    pv = tranche.value(prices, date)
    if pv <= 0:
        return

    # Determine target positions
    target_value = {}
    for t, w in target_weights.items():
        target_value[t] = w * pv

    # SELL: reduce or close positions
    for t in list(tranche.holdings.keys()):
        p = get_price(t, prices, date)
        if p <= 0:
            continue
        cur_val = tranche.holdings[t] * p
        tgt_val = target_value.get(t, 0)
        if cur_val > tgt_val:
            sell_val = cur_val - tgt_val
            sell_units = sell_val / p
            actual_sell = min(sell_units, tranche.holdings[t])
            tx = actual_sell * p * tx_cost
            tranche.cash += actual_sell * p - tx
            tranche.holdings[t] -= actual_sell
            if tranche.holdings[t] < 1e-10:
                del tranche.holdings[t]

    # BUY: increase or open positions
    for t, tgt_val in target_value.items():
        p = get_price(t, prices, date)
        if p <= 0:
            continue
        cur_val = tranche.holdings.get(t, 0) * p
        if tgt_val > cur_val:
            buy_val = tgt_val - cur_val
            buy_val = min(buy_val, tranche.cash / (1 + tx_cost))
            if buy_val <= 0:
                continue
            buy_units = buy_val / p
            tx = buy_val * tx_cost
            tranche.holdings[t] = tranche.holdings.get(t, 0) + buy_units
            tranche.cash -= (buy_val + tx)


def sell_coin_from_tranche(tranche: Tranche, ticker: str, prices, date,
                           tx_cost: float):
    """Sell a single coin from tranche (DD exit / crash)."""
    units = tranche.holdings.get(ticker, 0)
    if units <= 0:
        return
    p = get_price(ticker, prices, date)
    if p <= 0:
        return  # FIX #4: don't pop if price unavailable — keep position
    del tranche.holdings[ticker]
    tx = units * p * tx_cost
    tranche.cash += units * p - tx


def sell_all_from_tranche(tranche: Tranche, prices, date, tx_cost: float):
    """Liquidate all holdings in tranche to cash."""
    for t in list(tranche.holdings.keys()):
        sell_coin_from_tranche(tranche, t, prices, date, tx_cost)


# ─── Drift Check ──────────────────────────────────────────────────

def check_drift(tranche: Tranche, target_weights: Dict[str, float],
                prices, date, threshold: float) -> bool:
    """Check if half-turnover exceeds drift threshold."""
    cur_w = tranche.current_weights(prices, date)
    all_tickers = set(list(cur_w.keys()) + list(target_weights.keys()))
    half_to = sum(abs(cur_w.get(t, 0) - target_weights.get(t, 0))
                  for t in all_tickers) / 2
    return half_to >= threshold


# ─── Main Backtest ─────────────────────────────────────────────────

def run_v15_backtest(prices, universe_map, params: V15Params,
                     tranche_days: Optional[Tuple] = None) -> dict:
    """Run V15 coin backtest with true N-tranche architecture.

    Returns dict with: metrics, yearly, pv, rebal_count, dd_exit_count,
                       crash_count, flip_count, tranche_rebal_counts
    """
    global _bl_cache
    _bl_cache = {}  # clear blacklist cache for each run

    if tranche_days is None:
        tranche_days = params.tranche_days

    n_tr = len(tranche_days)
    capital_per = params.initial_capital / n_tr

    # Initialize tranches
    tranches = [Tranche(i, tranche_days[i], capital_per) for i in range(n_tr)]

    # Global state
    prev_canary: Optional[bool] = False  # FIX #4: start OFF (not None) for consistent hysteresis
    canary_on = False
    canary_on_date = None
    crash_cooldown = 0
    prev_month = None

    # Counters
    rebal_count = 0
    dd_exit_count = 0
    crash_count = 0
    flip_count = 0
    tranche_rebal_counts = [0] * n_tr

    # Get trading dates from BTC
    btc = prices.get('BTC-USD')
    if btc is None:
        return _empty_result()
    all_dates = btc.index[
        (btc.index >= params.start_date) & (btc.index <= params.end_date)]
    if len(all_dates) < 2:
        return _empty_result()

    portfolio_values = []

    for date in all_dates:
        cur_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and cur_month != prev_month)
        is_first = (prev_month is None)

        # New month: reset tranche anchor flags
        if is_month_change:
            for tr in tranches:
                tr.anchor_done_this_month = False  # FIX #2: separate flag for monthly anchor

        # ── 1. Combined portfolio value ──
        pv = sum(tr.value(prices, date) for tr in tranches)

        # ── 2. Crash Breaker (G5) — GLOBAL ──
        crash_just_ended = False
        if crash_cooldown > 0:
            crash_cooldown -= 1
            if crash_cooldown == 0:
                if check_crash(prices, date, params):
                    crash_cooldown = params.crash_cool_days
                else:
                    crash_just_ended = True
        elif not is_first and check_crash(prices, date, params):
            # Sell ALL tranches to cash
            for tr in tranches:
                sell_all_from_tranche(tr, prices, date, params.tx_cost)
            crash_cooldown = params.crash_cool_days
            crash_count += 1

        # ── 3. Resolve Canary — GLOBAL (always, even during crash cooldown) ──
        # FIX #1: canary must be updated every day for correct flip detection
        canary_on = resolve_canary(prices, date, params, prev_canary)

        # ── 4. If in crash cooldown, skip tranche operations ──
        if crash_cooldown > 0 and not crash_just_ended:
            pv = sum(tr.value(prices, date) for tr in tranches)
            portfolio_values.append({'Date': date, 'Value': pv})
            prev_canary = canary_on  # FIX #1: update even during cooldown
            prev_month = cur_month
            continue
        signal_flipped = (prev_canary is not None and canary_on != prev_canary)
        flip_off = signal_flipped and not canary_on
        flip_on = signal_flipped and canary_on

        if flip_on:
            canary_on_date = date
            flip_count += 1
            # Mark all tranches for PFD
            for tr in tranches:
                tr.post_flip_refreshed = False
        if flip_off:
            flip_count += 1

        # ── 5. Blacklist — GLOBAL (stateless) ──
        universe = get_universe_for_date(universe_map, date)
        blacklist = get_blacklist(prices, universe, date, params)

        # ── 6~8. Per-Tranche Operations ──
        for tr in tranches:
            # ── 6. DD Exit (daily, per coin) ──
            tr.dd_exited_today = set()  # reset daily
            if canary_on and not crash_just_ended:
                dd_exits = check_dd_exit_tranche(tr, prices, date, params)
                for t in dd_exits:
                    sell_coin_from_tranche(tr, t, prices, date, params.tx_cost)
                    tr.dd_exited_today.add(t)  # FIX #5: prevent same-day rebuy
                    dd_exit_count += 1

            # ── 7. Determine if rebalance needed ──
            do_rebal = False
            reason = 'none'

            # 7a. First day
            if is_first:
                do_rebal = True
                reason = 'init'

            # 7b. Canary flip → immediate rebalance
            if not do_rebal and signal_flipped:
                do_rebal = True
                reason = 'flip_on' if flip_on else 'flip_off'

            # 7c. Crash just ended → re-enter
            if not do_rebal and crash_just_ended:
                do_rebal = True
                reason = 'crash_end'

            # 7d. PFD5: 5 days after OFF→ON flip
            if (not do_rebal and canary_on and not tr.post_flip_refreshed
                    and canary_on_date is not None):
                days_since = (date - canary_on_date).days
                if days_since >= params.post_flip_delay:
                    do_rebal = True
                    reason = 'pfd5'
                    tr.post_flip_refreshed = True

            # 7e. Monthly anchor day — independent from other rebal types (FIX #2)
            if (not do_rebal and not tr.anchor_done_this_month
                    and date.day >= tr.anchor_day and not is_first):
                do_rebal = True
                reason = 'monthly'

            # 7f. Drift check
            if (not do_rebal and canary_on and tr.holdings
                    and params.drift_threshold > 0):
                target_w = compute_target_weights(
                    prices, universe_map, date, params, blacklist,
                    tr.dd_exited_today)
                if target_w and check_drift(
                        tr, target_w, prices, date, params.drift_threshold):
                    do_rebal = True
                    reason = 'drift'

            # ── 8. Execute rebalance ──
            if do_rebal:
                if canary_on:
                    target_w = compute_target_weights(
                        prices, universe_map, date, params, blacklist,
                        tr.dd_exited_today)
                    if target_w:
                        execute_rebalance(tr, target_w, prices, date,
                                          params.tx_cost)
                        tr.prev_picks = list(target_w.keys())
                    else:
                        # No healthy coins → go to cash
                        sell_all_from_tranche(tr, prices, date, params.tx_cost)
                        tr.prev_picks = []
                else:
                    # Risk-off → all cash
                    sell_all_from_tranche(tr, prices, date, params.tx_cost)
                    tr.prev_picks = []

                if reason == 'monthly':
                    tr.anchor_done_this_month = True  # FIX #2: only set for monthly anchor
                rebal_count += 1
                tranche_rebal_counts[tr.idx] += 1

        # ── 9. Record combined value ──
        pv = sum(tr.value(prices, date) for tr in tranches)
        portfolio_values.append({'Date': date, 'Value': pv})

        prev_canary = canary_on
        prev_month = cur_month

    # ── Metrics ──
    if not portfolio_values:
        return _empty_result()

    pvdf = pd.DataFrame(portfolio_values).set_index('Date')
    m = calc_metrics(pvdf)
    ym = calc_yearly_metrics(pvdf)

    return {
        'metrics': m,
        'yearly': ym,
        'pv': pvdf,
        'rebal_count': rebal_count,
        'dd_exit_count': dd_exit_count,
        'crash_count': crash_count,
        'flip_count': flip_count,
        'tranche_rebal_counts': tranche_rebal_counts,
    }


def _empty_result():
    return {
        'metrics': {'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'Final': 0},
        'yearly': {}, 'pv': pd.DataFrame(),
        'rebal_count': 0, 'dd_exit_count': 0,
        'crash_count': 0, 'flip_count': 0,
        'tranche_rebal_counts': [],
    }


# ─── Multi-Anchor Runner ──────────────────────────────────────────

def run_multi_anchor(prices, universe_map, params: V15Params,
                     n_offsets: int = 10) -> list:
    """Run backtest with different tranche anchor offsets."""
    results = []
    spacing = params.tranche_days[1] - params.tranche_days[0]  # default 9

    for offset in range(n_offsets):
        days = tuple((params.tranche_days[i] - 1 + offset) % 28 + 1
                     for i in range(len(params.tranche_days)))
        r = run_v15_backtest(prices, universe_map, params, tranche_days=days)
        results.append({'offset': offset, 'days': days, **r})
    return results


# ─── Data Loading ──────────────────────────────────────────────────

def load_data(top_n=40):
    """Load price data and universe map."""
    um = load_universe()
    fm = filter_universe(um, top_n)
    tickers = set()
    for ts in fm.values():
        tickers.update(ts)
    tickers.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(tickers)
    return prices, fm


# ─── Main ──────────────────────────────────────────────────────────

def main():
    print('V15 Coin Strategy Backtest — True 3-Tranche Engine')
    print('=' * 80)

    print('Loading data...')
    prices, universe = load_data()
    print(f'  {len(prices)} tickers loaded\n')

    params = V15Params()

    # Single run with default tranche days
    print(f'Running single backtest (tranches: {params.tranche_days})...')
    r = run_v15_backtest(prices, universe, params)
    m = r['metrics']
    print(f'  Sharpe: {m["Sharpe"]:.3f}')
    print(f'  CAGR:   {m["CAGR"]:+.1%}')
    print(f'  MDD:    {m["MDD"]:.1%}')
    print(f'  Rebals: {r["rebal_count"]}')
    print(f'  DD Exits: {r["dd_exit_count"]}')
    print(f'  Crashes: {r["crash_count"]}')
    print(f'  Flips: {r["flip_count"]}')
    print(f'  Per-tranche rebals: {r["tranche_rebal_counts"]}')

    # Multi-anchor sensitivity
    print(f'\nRunning 10-anchor sensitivity...')
    results = run_multi_anchor(prices, universe, params, n_offsets=10)

    print(f'\n{"Offset":>6} {"Days":>16} {"Sharpe":>8} {"CAGR":>8} {"MDD":>8} {"Rebals":>7} {"DD":>4}')
    print('-' * 65)
    sharpes = []
    for r in results:
        m = r['metrics']
        sharpes.append(m['Sharpe'])
        print(f'{r["offset"]:>6} {str(r["days"]):>16} {m["Sharpe"]:>8.3f} '
              f'{m["CAGR"]:>+7.1%} {m["MDD"]:>7.1%} {r["rebal_count"]:>7} '
              f'{r["dd_exit_count"]:>4}')

    avg_sh = np.mean(sharpes)
    std_sh = np.std(sharpes)
    avg_cagr = np.mean([r['metrics']['CAGR'] for r in results])
    avg_mdd = np.mean([r['metrics']['MDD'] for r in results])

    print(f'\n{"=" * 65}')
    print(f'  10-anchor avg: Sharpe {avg_sh:.3f} (σ={std_sh:.3f})  '
          f'CAGR {avg_cagr:+.1%}  MDD {avg_mdd:.1%}')

    # Yearly breakdown (from first offset)
    print(f'\n  연도별 (offset=0):')
    for yr, ym in sorted(results[0]['yearly'].items()):
        print(f'    {yr}: CAGR {ym["CAGR"]:>+7.1%}  MDD {ym["MDD"]:>6.1%}  '
              f'Sharpe {ym["Sharpe"]:.3f}')


if __name__ == '__main__':
    main()
