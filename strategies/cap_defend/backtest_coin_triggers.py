#!/usr/bin/env python3
"""
Trigger-Based Rebalancing Test v2 (2018-2025, CoinGecko Top 50)
Baseline(monthly) + daily triggers: both OFFENSIVE (cash->buy) and DEFENSIVE (hold->sell).

Offensive Triggers (enter mid-month when in cash):
  E1: BTC > SMA50 for 3 consecutive days + volume > 1.5x 20d avg (Trend Confirmation)
  E2: BTC > SMA50 + Breadth >= 50% + 5d breadth delta >= 10% (Market Breadth Thrust)
  E3: After BTC -12% in 10d, BTC > SMA20 for 2d + ret5 > 3% (Post-Shock Recovery)

Defensive Triggers (exit mid-month when holding):
  D1: BTC daily -7% + volume > 2x 20d avg (Panic Crash, proven A3)
  D2: BTC < SMA50 for 2d + ETH < SMA50 + portfolio DD > -10% (Trend Breakdown)
  D3: BTC < SMA50 + Breadth <= 35% + 5d breadth delta <= -10% (Breadth Collapse)

Combined Strategies:
  C1: E1 + D1 (Wave Rider + Panic Exit)
  C2: E2 + D3 (Breadth Entry + Breadth Exit)
  C3: E3 + D1 (Post-Shock + Panic Exit)

Design principles:
  - Asymmetric: enter carefully (3d confirm), exit fast (immediate)
  - Frequency target: 3-8 fires/year
  - Monthly rebalancing always runs; triggers add extra rebalancing
  - On trigger fire: run full strategy (canary -> health -> score -> pick -> execute)
"""

import os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.dirname(__file__))
from backtest_coin_strategy import (
    load_universe, load_all_prices, calc_metrics,
    calc_sharpe, calc_rsi, calc_macd_hist, calc_bb_pctb,
    calc_ret, get_volatility, STABLECOINS
)

EXCLUDE_SYMBOLS = STABLECOINS | {'PAXG', 'XAUT', 'WBTC', 'USD1', 'USDE'}
TOP_N = 50  # CoinGecko Top N (excluding stablecoins/real-asset)
START_DATE = '2018-01-01'
END_DATE = '2025-12-31'


def filter_universe_topn(universe_map, top_n=TOP_N):
    """Filter to top N coins by market cap rank (excluding stablecoins)."""
    filtered = {}
    for month_key, tickers in universe_map.items():
        clean = [t for t in tickers if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]
        filtered[month_key] = clean[:top_n]
    return filtered


def get_universe_for_date(universe_map, date):
    month_key = date.strftime('%Y-%m') + '-01'
    for mk in sorted(universe_map.keys(), reverse=True):
        if mk <= month_key:
            return universe_map[mk]
    if universe_map:
        return list(universe_map.values())[0]
    return []


def check_canary(prices, global_idx):
    """BTC > SMA50?"""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < 50: return False
    return close.iloc[-1] > close.rolling(50).mean().iloc[-1]


def check_health_baseline(ticker, prices, global_idx):
    if ticker not in prices: return False
    close = prices[ticker]['Close'].iloc[:global_idx+1]
    if len(close) < 90: return False
    cur = close.iloc[-1]
    sma30 = close.rolling(30).mean().iloc[-1]
    mom21 = calc_ret(close, 21)
    vol90 = get_volatility(close, 90)
    return cur > sma30 and mom21 > 0 and (vol90 is not None and vol90 <= 0.10)


def compute_target(prices, universe_map, date, global_idx, n_picks=5):
    """Full baseline strategy: canary -> health -> score -> pick -> weight."""
    risk_on = check_canary(prices, global_idx)
    if not risk_on:
        return {'CASH': 1.0}

    uni_tickers = get_universe_for_date(universe_map, date)
    uni_clean = [t.replace('-USD', '') for t in uni_tickers
                if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]

    healthy = []
    for sym in uni_clean:
        ticker = f"{sym}-USD"
        if check_health_baseline(ticker, prices, global_idx):
            healthy.append(ticker)

    if not healthy:
        return {'CASH': 1.0}

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
        return {'CASH': 1.0}

    vols = {}
    for t in picks:
        close = prices[t]['Close'].iloc[:global_idx+1]
        v = get_volatility(close, 90)
        if v and v > 0: vols[t] = v
    if vols:
        inv = {t: 1/v for t, v in vols.items()}
        tot = sum(inv.values())
        return {t: w/tot for t, w in inv.items()}
    else:
        return {t: 1/len(picks) for t in picks}


def execute_rebalance(holdings, cash, weights, prices, date, tx_cost):
    """Rebalance with tx costs only on actual trade deltas.
    Same positions kept → no cost. Only sells/buys of the difference are charged."""

    def _get_price(ticker):
        if ticker not in prices: return 0
        idx = prices[ticker].index.get_indexer([date], method='ffill')[0]
        return prices[ticker]['Close'].iloc[idx] if idx >= 0 else 0

    # Current position values
    current_values = {}
    port_val = cash
    for t, units in holdings.items():
        p = _get_price(t)
        v = units * p
        current_values[t] = v
        port_val += v

    # Target: all cash
    if 'CASH' in weights and weights['CASH'] == 1.0:
        sell_total = sum(current_values.values())
        return {}, cash + sell_total * (1 - tx_cost)

    # Target position values
    target_values = {t: port_val * w for t, w in weights.items() if t != 'CASH'}

    new_holdings = {}
    new_cash = cash
    all_tickers = set(list(current_values.keys()) + list(target_values.keys()))

    # Pass 1: Sells and holds
    for t in all_tickers:
        cur_val = current_values.get(t, 0)
        tgt_val = target_values.get(t, 0)
        p = _get_price(t)
        if p <= 0: continue

        if tgt_val >= cur_val:
            # Keep current position (buy more in pass 2)
            if cur_val > 0:
                new_holdings[t] = holdings[t]
        else:
            # Sell the excess, tx_cost on sold amount only
            sell_amount = cur_val - tgt_val
            new_cash += sell_amount * (1 - tx_cost)
            if tgt_val > 0:
                new_holdings[t] = tgt_val / p

    # Pass 2: Buys (using available cash after sells)
    buys = {}
    for t in all_tickers:
        cur_val = current_values.get(t, 0)
        tgt_val = target_values.get(t, 0)
        if tgt_val > cur_val:
            buys[t] = tgt_val - cur_val

    total_buy = sum(buys.values())
    if total_buy > 0:
        # Scale down if not enough cash (due to sell-side tx costs)
        scale = min(1.0, new_cash / total_buy)
        for t, buy_val in buys.items():
            p = _get_price(t)
            if p <= 0: continue
            actual_spend = buy_val * scale
            bought_value = actual_spend * (1 - tx_cost)
            new_cash -= actual_spend
            cur_units = new_holdings.get(t, 0)
            new_holdings[t] = cur_units + bought_value / p

    return new_holdings, new_cash


def get_portfolio_value(holdings, cash, prices, date):
    val = cash
    for t, units in holdings.items():
        if t in prices:
            idx = prices[t].index.get_indexer([date], method='ffill')[0]
            if idx >= 0:
                val += units * prices[t]['Close'].iloc[idx]
    return val


def calc_breadth(prices, universe_map, date, global_idx, ma_period=50):
    """Fraction of universe coins above their SMA."""
    uni_tickers = get_universe_for_date(universe_map, date)
    uni_clean = [t.replace('-USD', '') for t in uni_tickers
                if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]
    count_above = count_total = 0
    for sym in uni_clean:
        ticker = f"{sym}-USD"
        if ticker not in prices: continue
        close = prices[ticker]['Close'].iloc[:global_idx+1]
        if len(close) < ma_period: continue
        count_total += 1
        if close.iloc[-1] > close.rolling(ma_period).mean().iloc[-1]:
            count_above += 1
    return count_above / count_total if count_total > 0 else 0.0


# ─── Offensive Triggers (cash -> buy) ───

def trigger_E1(prices, global_idx, state):
    """BTC > SMA50 for 3 consecutive days + volume > 1.5x 20d avg on breakout day.
    Edge-triggered: fires only on the 3rd day of a new breakout."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < 53: return False
    sma50 = close.rolling(50).mean()

    # Check last 3 days all above SMA50
    for offset in range(3):
        idx = len(close) - 1 - offset
        if close.iloc[idx] <= sma50.iloc[idx]:
            return False

    # Edge: 4th day back must have been below (new breakout)
    if len(close) >= 4:
        idx_4 = len(close) - 4
        if close.iloc[idx_4] > sma50.iloc[idx_4]:
            return False  # Already above for 4+ days, not new

    # Volume confirmation: breakout day (3 days ago) volume > 1.5x
    if 'Volume' not in btc.columns: return True
    vol = btc['Volume'].iloc[:global_idx+1]
    if len(vol) < 21: return True
    breakout_day = len(vol) - 3  # The first day of the breakout
    if breakout_day >= 0:
        avg_vol = vol.iloc[max(0, breakout_day-20):breakout_day].mean()
        if avg_vol > 0 and vol.iloc[breakout_day] < 1.5 * avg_vol:
            return False  # No volume confirmation

    return True


def trigger_E2(prices, global_idx, state, universe_map, date):
    """BTC > SMA50 + Breadth >= 50% + 5d breadth improvement >= 10%.
    Edge-triggered: 2d consecutive + not true 3rd day back."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < 50: return False
    if close.iloc[-1] <= close.rolling(50).mean().iloc[-1]:
        return False

    breadth_today = calc_breadth(prices, universe_map, date, global_idx)
    state['breadth_history'] = state.get('breadth_history', [])
    state['breadth_history'].append(breadth_today)
    # Keep only last 10
    if len(state['breadth_history']) > 10:
        state['breadth_history'] = state['breadth_history'][-10:]

    bh = state['breadth_history']
    if len(bh) < 6:
        return False

    # Today: breadth >= 50% and 5d delta >= 10%
    today_ok = bh[-1] >= 0.50 and (bh[-1] - bh[-6]) >= 0.10
    yesterday_ok = bh[-2] >= 0.50 and (bh[-2] - bh[-7]) >= 0.10 if len(bh) >= 7 else False
    day_before_ok = bh[-3] >= 0.50 and (bh[-3] - bh[-8]) >= 0.10 if len(bh) >= 8 else False

    # Edge: 2d consecutive, not 3rd day back
    return today_ok and yesterday_ok and not day_before_ok


def trigger_E3(prices, global_idx, state):
    """Post-shock recovery: after BTC -12% in 10d, BTC > SMA20 for 2d + ret5 > 3%.
    Edge-triggered: fires once per shock episode."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < 25: return False

    sma20 = close.rolling(20).mean()

    # Check if there was a -12% shock in the last 10-20 days
    had_shock = False
    for lookback in range(10, 21):
        if len(close) >= lookback + 1:
            ret = close.iloc[-1] / close.iloc[-lookback-1] - 1
            # Actually check if at some point in the last 20 days, 10d return was <= -12%
    # Simpler: check if min of last 20 days close was 12%+ below 10d-earlier price
    for day_offset in range(1, 15):
        idx_now = len(close) - 1 - day_offset
        idx_prev = idx_now - 10
        if idx_prev >= 0:
            ret_10d = close.iloc[idx_now] / close.iloc[idx_prev] - 1
            if ret_10d <= -0.12:
                had_shock = True
                break

    if not had_shock:
        state['e3_shock_fired'] = False
        return False

    # Already fired for this shock episode?
    if state.get('e3_shock_fired', False):
        return False

    # BTC > SMA20 for 2 consecutive days
    if close.iloc[-1] <= sma20.iloc[-1] or close.iloc[-2] <= sma20.iloc[-2]:
        return False

    # 5d return > 3%
    ret5 = close.iloc[-1] / close.iloc[-6] - 1 if len(close) >= 6 else 0
    if ret5 <= 0.03:
        return False

    state['e3_shock_fired'] = True
    return True


def trigger_E4(prices, global_idx, state):
    """20-Day Range Breakout: BTC closes above 20d high for 2 consecutive days + vol > 1.3x.
    Edge-triggered: 3rd day back must not be above 20d high."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < 23: return False

    # Last 2 days: close > prior 20d high
    for offset in range(2):
        idx = len(close) - 1 - offset
        high_20 = close.iloc[max(0, idx-20):idx].max()
        if close.iloc[idx] <= high_20:
            return False

    # Edge: 3rd day back must NOT be above
    idx_3 = len(close) - 3
    if idx_3 >= 20:
        high_20_3 = close.iloc[max(0, idx_3-20):idx_3].max()
        if close.iloc[idx_3] > high_20_3:
            return False

    # Volume > 1.3x (required)
    if 'Volume' not in btc.columns: return False
    vol = btc['Volume'].iloc[:global_idx+1]
    if len(vol) < 21: return False
    avg_vol = vol.iloc[-21:-1].mean()
    return avg_vol > 0 and vol.iloc[-1] > 1.3 * avg_vol


def trigger_E5(prices, global_idx, state):
    """Volatility Squeeze Breakout: BTC 10d HV in bottom 20% of 120d range,
    then close > 15d high + vol > 1.5x. Edge: fire once per squeeze episode."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close_full = btc['Close'].iloc[:global_idx+1]
    if len(close_full) < 135: return False
    # Only use last 135 bars for performance
    close = close_full.iloc[-135:]

    log_ret = np.log(close / close.shift(1)).dropna()
    if len(log_ret) < 120: return False

    hv_series = log_ret.rolling(10).std() * np.sqrt(365)
    # Exclude today from the baseline distribution
    hv_baseline = hv_series.iloc[-121:-1].dropna()
    if len(hv_baseline) < 20: return False

    hv_now = hv_series.iloc[-1]
    in_squeeze = hv_now <= hv_baseline.quantile(0.20)

    if not in_squeeze:
        state['e5_fired'] = False
        return False

    if state.get('e5_fired', False):
        return False

    # Breakout: close > 15d high
    if close.iloc[-1] <= close.iloc[-16:-1].max():
        return False

    # Volume > 1.5x (required)
    if 'Volume' not in btc.columns: return False
    vol = btc['Volume'].iloc[:global_idx+1]
    if len(vol) < 21: return False
    avg_vol = vol.iloc[-21:-1].mean()
    if avg_vol <= 0 or vol.iloc[-1] <= 1.5 * avg_vol:
        return False

    state['e5_fired'] = True
    return True


def trigger_E6(prices, global_idx, state):
    """Liquidation Wick Recovery: yesterday low < BB lower but close inside band,
    today close > yesterday high + vol > 1.5x. Edge: fire once, reset above SMA20."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < 22: return False
    if 'Low' not in btc.columns or 'High' not in btc.columns:
        return False

    sma20 = close.iloc[-22:].rolling(20).mean()

    # Edge: reset when price recovers above SMA20
    if state.get('e6_fired', False):
        if close.iloc[-1] > sma20.iloc[-1]:
            state['e6_fired'] = False
        return False

    low = btc['Low'].iloc[:global_idx+1]
    high = btc['High'].iloc[:global_idx+1]
    std20 = close.iloc[-22:].rolling(20).std()
    bb_lower = sma20 - 2 * std20

    # Yesterday: low pierced BB lower but close recovered above it
    if low.iloc[-2] >= bb_lower.iloc[-2] or close.iloc[-2] <= bb_lower.iloc[-2]:
        return False

    # Today: close > yesterday's high
    if close.iloc[-1] <= high.iloc[-2]:
        return False

    # Volume > 1.5x
    if 'Volume' not in btc.columns: return False
    vol = btc['Volume'].iloc[:global_idx+1]
    if len(vol) < 21: return False
    avg_vol = vol.iloc[-21:-1].mean()
    if avg_vol <= 0 or vol.iloc[-1] <= 1.5 * avg_vol:
        return False

    state['e6_fired'] = True
    return True


def trigger_E7(prices, global_idx, state):
    """Golden Cross Entry: BTC SMA20 crosses above SMA50, held 2 days + vol > 1.2x.
    Edge-triggered: 3rd day back SMA20 <= SMA50."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < 53: return False

    sma20 = close.rolling(20).mean()
    sma50 = close.rolling(50).mean()

    # SMA20 > SMA50 for last 2 days
    if sma20.iloc[-1] <= sma50.iloc[-1] or sma20.iloc[-2] <= sma50.iloc[-2]:
        return False

    # Edge: 3rd day back must be SMA20 <= SMA50 (fresh cross)
    if sma20.iloc[-3] > sma50.iloc[-3]:
        return False

    # Volume > 1.2x (required)
    if 'Volume' not in btc.columns: return False
    vol = btc['Volume'].iloc[:global_idx+1]
    if len(vol) < 21: return False
    avg_vol = vol.iloc[-21:-1].mean()
    return avg_vol > 0 and vol.iloc[-1] > 1.2 * avg_vol


# ─── Defensive Triggers (hold -> sell) ───

def trigger_D1(prices, global_idx, state):
    """BTC daily -7% + volume > 2x 20d avg (proven A3)."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < 22: return False
    daily_ret = close.iloc[-1] / close.iloc[-2] - 1 if close.iloc[-2] > 0 else 0
    if daily_ret >= -0.07: return False
    if 'Volume' not in btc.columns: return False
    vol = btc['Volume'].iloc[:global_idx+1]
    if len(vol) < 21: return False
    avg_vol = vol.iloc[-21:-1].mean()
    return avg_vol > 0 and vol.iloc[-1] > 2 * avg_vol


def trigger_D2(prices, global_idx, state, holdings, cash):
    """BTC < SMA50 for 2d + ETH < SMA50 + portfolio DD > -10%.
    Edge-triggered: fires once per drawdown episode."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close_b = btc['Close'].iloc[:global_idx+1]
    if len(close_b) < 52: return False
    sma50_b = close_b.rolling(50).mean()

    # BTC < SMA50 for 2 consecutive days
    if close_b.iloc[-1] >= sma50_b.iloc[-1] or close_b.iloc[-2] >= sma50_b.iloc[-2]:
        state['d2_fired'] = False
        return False

    # ETH < SMA50
    eth = prices.get('ETH-USD')
    if eth is not None:
        close_e = eth['Close'].iloc[:global_idx+1]
        if len(close_e) >= 50:
            sma50_e = close_e.rolling(50).mean()
            if close_e.iloc[-1] >= sma50_e.iloc[-1]:
                return False

    # Portfolio DD > -10%
    date = btc.index[global_idx]
    port_val = get_portfolio_value(holdings, cash, prices, date)
    peak = state.get('peak_value', port_val)
    if peak <= 0: return False
    dd = (port_val - peak) / peak
    if dd >= -0.10:
        state['d2_fired'] = False
        return False

    # Edge: fire once per episode
    if state.get('d2_fired', False):
        return False
    state['d2_fired'] = True
    return True


def trigger_D3(prices, global_idx, state, universe_map, date):
    """BTC < SMA50 + Breadth <= 35% + 5d breadth delta <= -10%.
    Edge-triggered: 2d consecutive."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < 50: return False

    # Always append breadth to history (even when BTC > SMA50)
    breadth_today = calc_breadth(prices, universe_map, date, global_idx)
    state['d3_breadth_history'] = state.get('d3_breadth_history', [])
    state['d3_breadth_history'].append(breadth_today)
    if len(state['d3_breadth_history']) > 10:
        state['d3_breadth_history'] = state['d3_breadth_history'][-10:]

    if close.iloc[-1] >= close.rolling(50).mean().iloc[-1]:
        return False

    bh = state['d3_breadth_history']
    if len(bh) < 6:
        return False

    today_ok = bh[-1] <= 0.35 and (bh[-1] - bh[-6]) <= -0.10
    yesterday_ok = bh[-2] <= 0.35 and (bh[-2] - bh[-7]) <= -0.10 if len(bh) >= 7 else False
    day_before_ok = bh[-3] <= 0.35 and (bh[-3] - bh[-8]) <= -0.10 if len(bh) >= 8 else False

    return today_ok and yesterday_ok and not day_before_ok


def trigger_D4(prices, global_idx, state):
    """Downside Volatility Shock: BTC 5d downside vol > 60d 90th percentile + 3d ret < -5%.
    Edge: fire once, reset when 3d return turns positive."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close_full = btc['Close'].iloc[:global_idx+1]
    if len(close_full) < 70: return False
    close = close_full.iloc[-70:]

    ret_3d = close.iloc[-1] / close.iloc[-4] - 1

    # Reset edge when conditions clear
    if ret_3d >= 0:
        state['d4_fired'] = False
    if state.get('d4_fired', False):
        return False
    if ret_3d >= -0.05:
        return False

    # Downside volatility: only negative returns
    rets = close.pct_change().dropna()
    down_rets = rets.copy()
    down_rets[down_rets > 0] = 0

    dv_5 = down_rets.iloc[-5:].std()
    dv_series = down_rets.rolling(5).std()
    # Exclude today from baseline distribution
    dv_baseline = dv_series.iloc[-61:-1].dropna()
    if len(dv_baseline) < 10: return False

    if dv_5 <= dv_baseline.quantile(0.90):
        return False

    state['d4_fired'] = True
    return True


def trigger_D5(prices, global_idx, state, holdings, cash):
    """Consecutive Bleed Out: 4+ down days in last 5 + portfolio DD > -10% + close < SMA20.
    Edge: fire once per drawdown episode."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < 25: return False

    # Count down days in last 5
    down_count = sum(1 for i in range(1, 6) if close.iloc[-i] < close.iloc[-i-1])

    # Portfolio DD
    date = btc.index[global_idx]
    port_val = get_portfolio_value(holdings, cash, prices, date)
    peak = state.get('peak_value', port_val)
    dd = (port_val - peak) / peak if peak > 0 else 0

    # Reset when conditions clear (fewer down days OR DD recovers)
    if down_count < 4 or dd >= -0.05:
        state['d5_fired'] = False

    if down_count < 4:
        return False
    if dd >= -0.10:
        return False

    # Close < SMA20
    sma20 = close.rolling(20).mean()
    if close.iloc[-1] >= sma20.iloc[-1]:
        return False

    if state.get('d5_fired', False):
        return False
    state['d5_fired'] = True
    return True


def trigger_D6(prices, global_idx, state):
    """Blow-off Top Reversal: BTC +20% in 5d + Vol×2, then negative candle
    with close in lower 25% of range. Defends against euphoria top."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    close = btc['Close'].iloc[:global_idx+1]
    if len(close) < 8: return False

    # Yesterday was a +20% 5d surge
    ret_5d_prev = close.iloc[-2] / close.iloc[-7] - 1 if len(close) >= 7 else 0
    if ret_5d_prev < 0.20:
        return False

    # Yesterday vol > 2x avg
    if 'Volume' not in btc.columns: return False
    vol = btc['Volume'].iloc[:global_idx+1]
    if len(vol) < 22: return False
    avg_vol = vol.iloc[-22:-2].mean()
    if avg_vol <= 0 or vol.iloc[-2] <= 2 * avg_vol:
        return False

    # Today: close < yesterday's close (negative)
    if close.iloc[-1] >= close.iloc[-2]:
        return False

    # Close in lower 25% of today's range
    if 'High' in btc.columns and 'Low' in btc.columns:
        high_t = btc['High'].iloc[global_idx]
        low_t = btc['Low'].iloc[global_idx]
        if high_t > low_t:
            pos = (close.iloc[-1] - low_t) / (high_t - low_t)
            if pos > 0.25:
                return False

    return True


# ─── Rotation Triggers (swap coins while staying invested) ───

def compute_scores(prices, universe_map, date, global_idx):
    """Compute scores for all healthy coins. Returns dict {ticker: score}."""
    uni_tickers = get_universe_for_date(universe_map, date)
    uni_clean = [t.replace('-USD', '') for t in uni_tickers
                if t.replace('-USD', '') not in EXCLUDE_SYMBOLS]
    scores = {}
    for sym in uni_clean:
        ticker = f"{sym}-USD"
        if not check_health_baseline(ticker, prices, global_idx):
            continue
        close = prices[ticker]['Close'].iloc[:global_idx+1]
        if len(close) < 252: continue
        base = calc_sharpe(close, 126) + calc_sharpe(close, 252)
        rsi_val = calc_rsi(close)
        macd_h = calc_macd_hist(close)
        pctb = calc_bb_pctb(close)
        if pd.notna(rsi_val) and 45 <= rsi_val <= 70: base += 0.2
        if pd.notna(macd_h) and macd_h > 0: base += 0.2
        if pd.notna(pctb) and pctb > 0.5: base += 0.2
        scores[ticker] = base
    return scores


def get_current_weights(holdings, cash, prices, date):
    port_val = get_portfolio_value(holdings, cash, prices, date)
    if port_val <= 0: return {}
    weights = {}
    for t, units in holdings.items():
        if t in prices:
            idx = prices[t].index.get_indexer([date], method='ffill')[0]
            if idx >= 0:
                weights[t] = units * prices[t]['Close'].iloc[idx] / port_val
    weights['CASH'] = cash / port_val
    return weights


def calc_turnover(w1, w2):
    all_keys = set(w1.keys()) | set(w2.keys())
    return sum(abs(w2.get(k, 0) - w1.get(k, 0)) for k in all_keys) / 2


def trigger_R1(prices, global_idx, state, holdings, universe_map, date):
    """Any held coin fails health check -> rebalance.
    Edge-triggered: 2 consecutive days of failure to avoid whipsaw."""
    if not holdings: return False

    any_fail = False
    for t in holdings:
        if not check_health_baseline(t, prices, global_idx):
            any_fail = True
            break

    consec = state.get('r1_consec', 0)
    if any_fail:
        consec += 1
    else:
        consec = 0
    state['r1_consec'] = consec

    if consec >= 2:
        state['r1_consec'] = 0  # reset after fire
        return True
    return False


def trigger_R2(prices, global_idx, state, holdings, cash, universe_map, date):
    """Shadow target turnover > 30% for 2 consecutive days -> rebalance."""
    if not holdings: return False

    target = compute_target(prices, universe_map, date, global_idx)
    cur_w = get_current_weights(holdings, cash, prices, date)
    turnover = calc_turnover(cur_w, target)

    consec = state.get('r2_consec', 0)
    if turnover >= 0.30:
        consec += 1
    else:
        consec = 0
    state['r2_consec'] = consec

    if consec >= 2:
        state['r2_consec'] = 0
        return True
    return False


def trigger_R3(prices, global_idx, state, holdings, universe_map, date):
    """Worst held coin score < best non-held candidate by 0.5+ -> rebalance.
    Edge-triggered: 2 consecutive days."""
    if not holdings: return False

    all_scores = compute_scores(prices, universe_map, date, global_idx)
    if not all_scores: return False

    held_scores = {t: all_scores.get(t, -999) for t in holdings if t in all_scores}
    non_held_scores = {t: s for t, s in all_scores.items() if t not in holdings}

    gap_detected = False
    if held_scores and non_held_scores:
        worst_held = min(held_scores.values())
        best_non_held = max(non_held_scores.values())
        if best_non_held - worst_held >= 0.5:
            gap_detected = True

    consec = state.get('r3_consec', 0)
    if gap_detected:
        consec += 1
    else:
        consec = 0
    state['r3_consec'] = consec

    if consec >= 2:
        state['r3_consec'] = 0
        return True
    return False


def trigger_R4(prices, global_idx, state, holdings, universe_map, date):
    """Leadership Collapse: held coins that were top-20% by 20d return
    now underperform universe median by 8%+ in 5d. Edge: 2 consecutive days."""
    if not holdings: return False

    uni_tickers = get_universe_for_date(universe_map, date)
    rets = {}
    for t in uni_tickers:
        sym = t.replace('-USD', '')
        if sym in EXCLUDE_SYMBOLS: continue
        ticker = f"{sym}-USD"
        if ticker not in prices: continue
        close = prices[ticker]['Close'].iloc[:global_idx+1]
        if len(close) < 25: continue
        rets[ticker] = {
            'ret20': close.iloc[-1] / close.iloc[-21] - 1,
            'ret5': close.iloc[-1] / close.iloc[-6] - 1 if len(close) >= 6 else 0,
        }

    if len(rets) < 10:
        state['r4_consec'] = 0
        return False

    # Top 20% by 20d return
    sorted_by_20d = sorted(rets.keys(), key=lambda x: rets[x]['ret20'], reverse=True)
    top_n = max(1, len(sorted_by_20d) // 5)
    leaders = set(sorted_by_20d[:top_n])

    # Any held coin is a recent leader?
    held_leaders = [t for t in holdings if t in leaders and t in rets]
    if not held_leaders:
        state['r4_consec'] = 0
        return False

    # Median 5d return
    all_ret5 = [rets[t]['ret5'] for t in rets]
    median_5d = np.median(all_ret5)

    # Check if any held leader underperforms by 8%+
    collapse = any(rets[t]['ret5'] - median_5d <= -0.08 for t in held_leaders)

    consec = state.get('r4_consec', 0)
    consec = consec + 1 if collapse else 0
    state['r4_consec'] = consec

    if consec >= 2:
        state['r4_consec'] = 0
        return True
    return False


def trigger_R5(prices, global_idx, state, holdings, universe_map, date):
    """Altseason Decoupling: Top50 alt avg 20d return - BTC 20d return > +15%p
    for 2 consecutive days + BTC > SMA50. Edge: 2 consecutive days."""
    if not holdings: return False

    btc = prices.get('BTC-USD')
    if btc is None: return False
    close_btc = btc['Close'].iloc[:global_idx+1]
    if len(close_btc) < 50: return False

    # BTC must be in uptrend
    if close_btc.iloc[-1] <= close_btc.rolling(50).mean().iloc[-1]:
        state['r5_consec'] = 0
        return False

    btc_ret20 = close_btc.iloc[-1] / close_btc.iloc[-21] - 1

    # Alt 20d returns
    uni_tickers = get_universe_for_date(universe_map, date)
    alt_rets = []
    for t in uni_tickers:
        sym = t.replace('-USD', '')
        if sym in EXCLUDE_SYMBOLS or sym == 'BTC': continue
        ticker = f"{sym}-USD"
        if ticker not in prices: continue
        close = prices[ticker]['Close'].iloc[:global_idx+1]
        if len(close) < 21: continue
        alt_rets.append(close.iloc[-1] / close.iloc[-21] - 1)

    if len(alt_rets) < 10:
        state['r5_consec'] = 0
        return False

    decouple = (np.mean(alt_rets) - btc_ret20) > 0.15

    consec = state.get('r5_consec', 0)
    consec = consec + 1 if decouple else 0
    state['r5_consec'] = consec

    if consec >= 2:
        state['r5_consec'] = 0
        return True
    return False


# ─── Backtest Engine ───

TRIGGER_DEFS = {
    # Individual triggers
    'E1': {'type': 'offensive', 'label': 'E1: BTC3d>SMA50+Vol'},
    'E2': {'type': 'offensive', 'label': 'E2: Breadth50%+Thrust'},
    'E3': {'type': 'offensive', 'label': 'E3: PostShock Recovery'},
    'D1': {'type': 'defensive', 'label': 'D1: BTC-7%+Vol2x'},
    'D2': {'type': 'defensive', 'label': 'D2: BTC+ETH<SMA50+DD'},
    'D3': {'type': 'defensive', 'label': 'D3: BreadthCollapse'},
    # Combined strategies
    'C1': {'type': 'combined', 'label': 'C1: E1+D1 WaveRider',  'offense': 'E1', 'defense': 'D1'},
    'C2': {'type': 'combined', 'label': 'C2: E2+D3 Breadth',    'offense': 'E2', 'defense': 'D3'},
    'C3': {'type': 'combined', 'label': 'C3: E3+D1 PostShock',  'offense': 'E3', 'defense': 'D1'},
}


def check_trigger(trigger_name, prices, global_idx, state, holdings, cash,
                  universe_map, date):
    """Check a single trigger. Returns (fired, trigger_type)."""
    is_in_cash = not holdings

    if trigger_name == 'E1':
        if is_in_cash:
            return trigger_E1(prices, global_idx, state), 'offensive'
        # Still update state even if holding
        trigger_E1(prices, global_idx, state)
        return False, None
    elif trigger_name == 'E2':
        if is_in_cash:
            return trigger_E2(prices, global_idx, state, universe_map, date), 'offensive'
        trigger_E2(prices, global_idx, state, universe_map, date)
        return False, None
    elif trigger_name == 'E3':
        if is_in_cash:
            return trigger_E3(prices, global_idx, state), 'offensive'
        trigger_E3(prices, global_idx, state)
        return False, None
    elif trigger_name == 'D1':
        if not is_in_cash:
            return trigger_D1(prices, global_idx, state), 'defensive'
        return False, None
    elif trigger_name == 'D2':
        if not is_in_cash:
            return trigger_D2(prices, global_idx, state, holdings, cash), 'defensive'
        state['d2_fired'] = False
        return False, None
    elif trigger_name == 'D3':
        if not is_in_cash:
            return trigger_D3(prices, global_idx, state, universe_map, date), 'defensive'
        trigger_D3(prices, global_idx, state, universe_map, date)
        return False, None
    elif trigger_name == 'R1':
        if not is_in_cash:
            return trigger_R1(prices, global_idx, state, holdings, universe_map, date), 'rotation'
        return False, None
    elif trigger_name == 'R2':
        if not is_in_cash:
            return trigger_R2(prices, global_idx, state, holdings, cash, universe_map, date), 'rotation'
        return False, None
    elif trigger_name == 'R3':
        if not is_in_cash:
            return trigger_R3(prices, global_idx, state, holdings, universe_map, date), 'rotation'
        return False, None
    # ─── New triggers ───
    elif trigger_name == 'E4':
        if is_in_cash:
            return trigger_E4(prices, global_idx, state), 'offensive'
        trigger_E4(prices, global_idx, state)
        return False, None
    elif trigger_name == 'E5':
        if is_in_cash:
            return trigger_E5(prices, global_idx, state), 'offensive'
        trigger_E5(prices, global_idx, state)
        return False, None
    elif trigger_name == 'E6':
        if is_in_cash:
            return trigger_E6(prices, global_idx, state), 'offensive'
        trigger_E6(prices, global_idx, state)
        return False, None
    elif trigger_name == 'E7':
        if is_in_cash:
            return trigger_E7(prices, global_idx, state), 'offensive'
        trigger_E7(prices, global_idx, state)
        return False, None
    elif trigger_name == 'D4':
        if not is_in_cash:
            return trigger_D4(prices, global_idx, state), 'defensive'
        state['d4_fired'] = False
        return False, None
    elif trigger_name == 'D5':
        if not is_in_cash:
            return trigger_D5(prices, global_idx, state, holdings, cash), 'defensive'
        state['d5_fired'] = False
        return False, None
    elif trigger_name == 'D6':
        if not is_in_cash:
            return trigger_D6(prices, global_idx, state), 'defensive'
        return False, None
    elif trigger_name == 'R4':
        if not is_in_cash:
            return trigger_R4(prices, global_idx, state, holdings, universe_map, date), 'rotation'
        return False, None
    elif trigger_name == 'R5':
        if not is_in_cash:
            return trigger_R5(prices, global_idx, state, holdings, universe_map, date), 'rotation'
        return False, None
    return False, None


def run_backtest(prices, universe_map, trigger_names=None, tx_cost=0.004,
                 start_date=None, end_date=None):
    """Run backtest with monthly + optional daily triggers.
    trigger_names: list of trigger names to apply (e.g. ['E1', 'D1'] for combined)
    """
    if trigger_names is None:
        trigger_names = []

    btc = prices.get('BTC-USD')
    if btc is None: return None

    sd = start_date or START_DATE
    ed = end_date or END_DATE
    all_dates = btc.index[(btc.index >= sd) & (btc.index <= ed)]
    if len(all_dates) == 0: return None

    holdings = {}
    cash = 10000
    portfolio_values = []
    prev_month = None
    rebal_count = {'monthly': 0, 'trigger_off': 0, 'trigger_def': 0, 'trigger_rot': 0, 'init': 0}
    state = {}

    for i, date in enumerate(all_dates):
        global_idx = btc.index.get_loc(date)
        port_val = get_portfolio_value(holdings, cash, prices, date)

        # Track peak for D2
        peak = state.get('peak_value', port_val)
        if port_val > peak:
            state['peak_value'] = port_val

        current_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and current_month != prev_month)

        do_rebal = False
        reason = 'monthly'

        if i == 0:
            do_rebal = True
            reason = 'init'
        elif is_month_change:
            do_rebal = True
            reason = 'monthly'

        # Always run trigger checks to keep state continuous
        trigger_fired = False
        trigger_type = None
        for tname in trigger_names:
            fired, ttype = check_trigger(tname, prices, global_idx, state,
                                         holdings, cash, universe_map, date)
            if fired and not trigger_fired:
                trigger_fired = True
                trigger_type = ttype

        if trigger_fired and not do_rebal:
            do_rebal = True
            if trigger_type == 'offensive':
                reason = 'trigger_off'
            elif trigger_type == 'rotation':
                reason = 'trigger_rot'
            else:
                reason = 'trigger_def'

        if do_rebal:
            target = compute_target(prices, universe_map, date, global_idx)
            holdings, cash = execute_rebalance(holdings, cash, target, prices, date, tx_cost)
            rebal_count[reason] = rebal_count.get(reason, 0) + 1

            # Update peak and port_val after rebalance
            new_val = get_portfolio_value(holdings, cash, prices, date)
            if new_val > state.get('peak_value', 0):
                state['peak_value'] = new_val
            port_val = new_val

        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    result = pd.DataFrame(portfolio_values).set_index('Date')
    result.attrs['rebal_count'] = rebal_count
    return result


def main():
    print("=" * 105)
    print(f"  PHASE 2: E×D COMBINATION TEST ({START_DATE[:4]}-{END_DATE[:4]}, Top {TOP_N})")
    print("  Offensive (E1/E2/E5) × Defensive (D1/D2/D5) combinations")
    print("=" * 105)

    universe_map = load_universe()
    filtered_map = filter_universe_topn(universe_map, TOP_N)

    all_tickers = set()
    for mt in filtered_map.values():
        for t in mt:
            all_tickers.add(t)
    all_tickers.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(all_tickers)
    print(f"  {len(prices)} tickers loaded\n")

    # Phase 2: E×D combination test (22 configs)
    configs = [
        ('Baseline (Monthly)',       []),
        # ── Group 1: Single E × Single D (8) ──
        ('E1+D1',                    ['E1', 'D1']),
        ('E1+D2',                    ['E1', 'D2']),
        ('E1+D5',                    ['E1', 'D5']),
        ('E5+D1',                    ['E5', 'D1']),
        ('E5+D2',                    ['E5', 'D2']),
        ('E5+D5',                    ['E5', 'D5']),
        ('E2+D2',                    ['E2', 'D2']),
        ('E2+D5',                    ['E2', 'D5']),
        # ── Group 2: Multi E + Single D (5) ──
        ('E1E5+D1',                  ['E1', 'E5', 'D1']),
        ('E1E5+D2',                  ['E1', 'E5', 'D2']),
        ('E1E5+D5',                  ['E1', 'E5', 'D5']),
        ('E1E2+D2',                  ['E1', 'E2', 'D2']),
        ('E2E5+D2',                  ['E2', 'E5', 'D2']),
        # ── Group 3: E + Multi D (5) ──
        ('E1+D1D2',                  ['E1', 'D1', 'D2']),
        ('E1+D2D5',                  ['E1', 'D2', 'D5']),
        ('E5+D1D2',                  ['E5', 'D1', 'D2']),
        ('E5+D2D5',                  ['E5', 'D2', 'D5']),
        ('E1E5+D1D2',               ['E1', 'E5', 'D1', 'D2']),
        # ── Group 4: Full + Control (3) ──
        ('E1E5+D2D5',               ['E1', 'E5', 'D2', 'D5']),
        ('AllIn E1E2E5+D1D2D5',     ['E1', 'E2', 'E5', 'D1', 'D2', 'D5']),
        ('E1E5 (atk only)',          ['E1', 'E5']),
    ]

    # ═══ TEST 1: Overall Performance ═══
    print("=" * 105)
    print(f"  TEST 1: OVERALL PERFORMANCE ({START_DATE[:4]}-{END_DATE[:4]})")
    print("=" * 105)

    years = list(range(int(START_DATE[:4]), int(END_DATE[:4]) + 1))
    print(f"\n  {'Strategy':<24}", end='')
    for y in years:
        print(f" {y:>8}", end='')
    print(f" {'Overall':>10} {'MDD':>8} {'CAGR':>8} {'#Off':>5} {'#Def':>5} {'#Rot':>5}")
    print(f"  {'-'*110}")

    overall_results = {}
    all_rebal_counts = {}

    for name, trigs in configs:
        print(f"  {name:<24}", end='')
        pv_full = run_backtest(prices, filtered_map, trigger_names=trigs)

        for y in years:
            if pv_full is not None and len(pv_full) > 0:
                mask = (pv_full.index >= f'{y}-01-01') & (pv_full.index <= f'{y}-12-31')
                pv_slice = pv_full[mask]
                if len(pv_slice) > 10:
                    m = calc_metrics(pv_slice)
                    print(f" {m['Sharpe']:>8.3f}", end='')
                else:
                    print(f" {'N/A':>8}", end='')
            else:
                print(f" {'N/A':>8}", end='')

        if pv_full is not None and len(pv_full) > 10:
            m_full = calc_metrics(pv_full)
            rc = pv_full.attrs.get('rebal_count', {})
            n_off = rc.get('trigger_off', 0)
            n_def = rc.get('trigger_def', 0)
            n_rot = rc.get('trigger_rot', 0)
            overall_results[name] = m_full
            all_rebal_counts[name] = rc
            print(f" {m_full['Sharpe']:>10.3f} {m_full['MDD']:>7.1%} {m_full['CAGR']:>+7.1%} {n_off:>5} {n_def:>5} {n_rot:>5}")
        else:
            print()

    # ═══ TEST 2: Trigger Frequency ═══
    print("\n" + "=" * 105)
    print("  TEST 2: TRIGGER FREQUENCY ANALYSIS")
    print("=" * 105)

    print(f"\n  {'Strategy':<24} {'Monthly':>8} {'Off':>8} {'Def':>8} {'Rot':>8} {'Total':>8} {'Trig/Yr':>8}")
    print(f"  {'-'*78}")

    n_years = len(years)
    for name, trigs in configs:
        rc = all_rebal_counts.get(name, {})
        monthly = rc.get('monthly', 0) + rc.get('init', 0)
        off = rc.get('trigger_off', 0)
        defen = rc.get('trigger_def', 0)
        rot = rc.get('trigger_rot', 0)
        total = monthly + off + defen + rot
        per_yr = (off + defen + rot) / n_years if n_years > 0 else 0
        print(f"  {name:<24} {monthly:>8} {off:>8} {defen:>8} {rot:>8} {total:>8} {per_yr:>8.1f}")

    # ═══ SUMMARY ═══
    print("\n" + "=" * 105)
    print("  SUMMARY: Impact vs Monthly-Only Baseline")
    print("=" * 105)

    baseline_m = overall_results.get('Baseline (Monthly)') or overall_results.get('Monthly Only')
    if baseline_m:
        bl_sharpe = baseline_m['Sharpe']
        bl_mdd = baseline_m['MDD']
        bl_cagr = baseline_m['CAGR']

        print(f"\n  {'Strategy':<24} {'dSharpe':>10} {'dMDD':>10} {'dCAGR':>10} {'Verdict':>10}")
        print(f"  {'-'*70}")

        for name, trigs in configs:
            if name in ('Monthly Only', 'Baseline (Monthly)'): continue
            if name not in overall_results: continue
            m = overall_results[name]
            ds = m['Sharpe'] - bl_sharpe
            dm = (m['MDD'] - bl_mdd) * 100  # positive = less severe MDD
            dc = (m['CAGR'] - bl_cagr) * 100
            # Verdict: BETTER if Sharpe improved AND (CAGR up OR MDD improved)
            if ds > 0.03 and (dc > 0 or dm > 0):
                verdict = 'BETTER'
            elif ds < -0.05 or (dc < -5 and dm < 0):
                verdict = 'WORSE'
            else:
                verdict = 'NEUTRAL'
            print(f"  {name:<24} {ds:>+10.3f} {dm:>+9.1f}pp {dc:>+9.1f}pp {verdict:>10}")

    print()


if __name__ == '__main__':
    main()
