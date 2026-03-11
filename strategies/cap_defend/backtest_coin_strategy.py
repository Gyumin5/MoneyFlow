#!/usr/bin/env python3
"""
Coin Strategy Backtest Framework
- Baseline: Current V13 (Sharpe + Multi Bonus, BTC>SMA50, Health, InvVol90)
- 16 strategy variants: scoring, canary, health filter, weighting combinations
- Monthly rebalancing on historical CoinGecko universe
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional
from datetime import datetime

warnings.filterwarnings('ignore')

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', '..', 'data')
UNIVERSE_FILE = os.path.join(DATA_DIR, 'historical_universe.json')

STABLECOINS = {'USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'TUSD', 'PAX', 'GUSD', 'FRAX', 'LUSD', 'MIM', 'USDN', 'FDUSD'}

# ─── Technical Indicators ───────────────────────────────────────────

def calc_ret(s, d):
    if len(s) < d + 1: return 0
    return s.iloc[-1] / s.iloc[-d-1] - 1

def calc_sharpe(s, d):
    if len(s) < d + 1: return 0
    ret = s.pct_change().iloc[-d:]
    return (ret.mean() / ret.std()) * np.sqrt(252) if ret.std() > 0 else 0

def calc_sortino(s, d):
    if len(s) < d + 1: return 0
    ret = s.pct_change().iloc[-d:]
    down = ret[ret < 0]
    dd = down.std() if len(down) > 1 else ret.std()
    return (ret.mean() / dd) * np.sqrt(252) if dd > 0 else 0

def calc_rsi(s, period=14):
    if len(s) < period + 1: return np.nan
    delta = s.diff().iloc[-period-1:]
    gain = delta.clip(lower=0).rolling(period).mean().iloc[-1]
    loss = (-delta.clip(upper=0)).rolling(period).mean().iloc[-1]
    if loss == 0: return 100.0
    return 100 - (100 / (1 + gain / loss))

def calc_macd_hist(s):
    if len(s) < 35: return np.nan
    ema12 = s.ewm(span=12, adjust=False).mean()
    ema26 = s.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal = macd_line.ewm(span=9, adjust=False).mean()
    return (macd_line - signal).iloc[-1]

def calc_bb_pctb(s, period=20):
    if len(s) < period: return np.nan
    sma = s.rolling(period).mean()
    std = s.rolling(period).std()
    upper = sma + 2 * std
    lower = sma - 2 * std
    bw = upper.iloc[-1] - lower.iloc[-1]
    if bw == 0: return 0.5
    return (s.iloc[-1] - lower.iloc[-1]) / bw

def calc_adx(high, low, close, period=14):
    """Calculate ADX from high/low/close series."""
    if len(close) < period * 2 + 1: return np.nan
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)

    plus_dm = high.diff()
    minus_dm = -low.diff()
    plus_dm = plus_dm.where((plus_dm > minus_dm) & (plus_dm > 0), 0)
    minus_dm = minus_dm.where((minus_dm > plus_dm) & (minus_dm > 0), 0)

    atr = tr.ewm(span=period, adjust=False).mean()
    plus_di = 100 * (plus_dm.ewm(span=period, adjust=False).mean() / atr)
    minus_di = 100 * (minus_dm.ewm(span=period, adjust=False).mean() / atr)

    dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.ewm(span=period, adjust=False).mean()
    return adx.iloc[-1] if pd.notna(adx.iloc[-1]) else np.nan

def calc_cmf(high, low, close, volume, period=20):
    """Chaikin Money Flow."""
    if len(close) < period: return np.nan
    hl = high - low
    mfm = ((close - low) - (high - close)) / hl.replace(0, np.nan)
    mfv = mfm * volume
    cmf = mfv.rolling(period).sum() / volume.rolling(period).sum()
    return cmf.iloc[-1] if pd.notna(cmf.iloc[-1]) else 0

def calc_drawdown(s, days=14):
    """Max drawdown from recent peak over last N days."""
    if len(s) < days: return 0
    recent = s.iloc[-days:]
    peak = recent.cummax()
    dd = (recent / peak - 1).min()
    return dd

def get_volatility(s, d):
    if len(s) < d + 1: return 1.0
    return s.pct_change().iloc[-d:].std()


# ─── Data Loading ───────────────────────────────────────────────────

def load_universe():
    with open(UNIVERSE_FILE) as f:
        return json.load(f)

def load_price(ticker):
    """Load OHLCV CSV for a ticker, return DataFrame with Date index."""
    fpath = os.path.join(DATA_DIR, f"{ticker}.csv")
    if not os.path.exists(fpath):
        return None
    df = pd.read_csv(fpath, parse_dates=['Date'], index_col='Date')
    df = df.sort_index()
    # Ensure columns
    for col in ['Open', 'High', 'Low', 'Close', 'Volume']:
        if col not in df.columns:
            if col == 'Volume':
                df[col] = 0
            else:
                df[col] = df.get('Close', df.iloc[:, 0])
    return df

def load_all_prices(tickers):
    """Load price data for all tickers. Returns dict of DataFrames."""
    prices = {}
    for t in tickers:
        df = load_price(t)
        if df is not None and len(df) > 30:
            prices[t] = df
    return prices


# ─── Strategy Configurations ────────────────────────────────────────

@dataclass
class StrategyConfig:
    name: str
    # Scoring
    scoring: str = 'sharpe_bonus'  # sharpe_bonus, sortino_short, sortino_mid, sharpe_cmf, pure_momentum
    # Canary
    canary: str = 'btc_sma50'  # btc_sma50, dual_momentum, market_breadth, btc_eth_sma50
    # Health filter
    health: str = 'baseline'  # baseline, adx, drawdown, dual_ma
    # Weighting
    weighting: str = 'inv_vol90'  # inv_vol90, inv_vol30, inv_vol20, equal
    # Selection
    n_picks: int = 5


def get_all_strategies():
    strategies = []

    # Baseline
    strategies.append(StrategyConfig('Baseline'))

    # A. Scoring variants
    strategies.append(StrategyConfig('A1_Sortino30+90', scoring='sortino_short'))
    strategies.append(StrategyConfig('A2_Sortino60+126', scoring='sortino_mid'))
    strategies.append(StrategyConfig('A3_Sharpe+CMF', scoring='sharpe_cmf'))
    strategies.append(StrategyConfig('A4_PureMomentum', scoring='pure_momentum'))

    # B. Canary variants
    strategies.append(StrategyConfig('B1_DualMomentum', canary='dual_momentum'))
    strategies.append(StrategyConfig('B2_MarketBreadth', canary='market_breadth'))
    strategies.append(StrategyConfig('B3_BTC+ETH', canary='btc_eth_sma50'))

    # C. Health filter variants
    strategies.append(StrategyConfig('C1_ADX', health='adx'))
    strategies.append(StrategyConfig('C2_Drawdown', health='drawdown'))
    strategies.append(StrategyConfig('C3_DualMA', health='dual_ma'))

    # D. Weighting variants
    strategies.append(StrategyConfig('D1_InvVol30', weighting='inv_vol30'))
    strategies.append(StrategyConfig('D2_InvVol20', weighting='inv_vol20'))
    strategies.append(StrategyConfig('D3_EqualWeight', weighting='equal'))

    # E. Composite best-of
    strategies.append(StrategyConfig('E1_Sort+ETH+DD+Vol30', scoring='sortino_short', canary='btc_eth_sma50', health='drawdown', weighting='inv_vol30'))
    strategies.append(StrategyConfig('E2_Sort+Breadth+ADX+Vol30', scoring='sortino_short', canary='market_breadth', health='adx', weighting='inv_vol30'))
    strategies.append(StrategyConfig('E3_SortMid+Dual+DualMA+Vol30', scoring='sortino_mid', canary='dual_momentum', health='dual_ma', weighting='inv_vol30'))
    strategies.append(StrategyConfig('E4_PureMom+Dual+ADX+EW', scoring='pure_momentum', canary='dual_momentum', health='adx', weighting='equal'))

    # F. Optimized combinations of top components
    strategies.append(StrategyConfig('F1_DualMA+Breadth', scoring='sharpe_bonus', canary='market_breadth', health='dual_ma', weighting='inv_vol90'))
    strategies.append(StrategyConfig('F2_PureMom+Breadth', scoring='pure_momentum', canary='market_breadth', health='baseline', weighting='inv_vol90'))
    strategies.append(StrategyConfig('F3_PureMom+Breadth+DualMA', scoring='pure_momentum', canary='market_breadth', health='dual_ma', weighting='inv_vol90'))
    strategies.append(StrategyConfig('F4_PureMom+Breadth+DualMA+EW', scoring='pure_momentum', canary='market_breadth', health='dual_ma', weighting='equal'))
    strategies.append(StrategyConfig('F5_Baseline+Breadth+DualMA', scoring='sharpe_bonus', canary='market_breadth', health='dual_ma', weighting='equal'))
    strategies.append(StrategyConfig('F6_CMF+Breadth+DualMA', scoring='sharpe_cmf', canary='market_breadth', health='dual_ma', weighting='inv_vol90'))

    return strategies


# ─── Strategy Logic ─────────────────────────────────────────────────

def check_canary(cfg: StrategyConfig, prices: dict, universe: list, date_idx: int) -> bool:
    """Check canary signal. Returns True if Risk-On."""
    btc = prices.get('BTC-USD')
    if btc is None: return False
    btc_close = btc['Close'].iloc[:date_idx+1]
    if len(btc_close) < 50: return False

    btc_cur = btc_close.iloc[-1]
    btc_sma50 = btc_close.rolling(50).mean().iloc[-1]

    if cfg.canary == 'btc_sma50':
        return btc_cur > btc_sma50

    elif cfg.canary == 'dual_momentum':
        # BTC > SMA50 AND BTC 1-month return > 0
        if btc_cur <= btc_sma50: return False
        ret_1m = calc_ret(btc_close, 21)
        return ret_1m > 0

    elif cfg.canary == 'market_breadth':
        # % of universe coins above SMA50 > 40%
        count_above = 0
        count_total = 0
        for t in universe:
            if t in STABLECOINS or f"{t}-USD" not in prices: continue
            ticker = f"{t}-USD" if not t.endswith('-USD') else t
            if ticker not in prices: continue
            p = prices[ticker]['Close'].iloc[:date_idx+1]
            if len(p) < 50: continue
            count_total += 1
            if p.iloc[-1] > p.rolling(50).mean().iloc[-1]:
                count_above += 1
        if count_total == 0: return False
        breadth = count_above / count_total
        return breadth > 0.40

    elif cfg.canary == 'btc_eth_sma50':
        eth = prices.get('ETH-USD')
        if eth is None: return False
        eth_close = eth['Close'].iloc[:date_idx+1]
        if len(eth_close) < 50: return False
        eth_cur = eth_close.iloc[-1]
        eth_sma50 = eth_close.rolling(50).mean().iloc[-1]
        return btc_cur > btc_sma50 and eth_cur > eth_sma50

    return False


def check_health(cfg: StrategyConfig, prices: dict, ticker: str, date_idx: int) -> bool:
    """Check health filter for a single coin. Returns True if healthy."""
    if ticker not in prices: return False
    df = prices[ticker]
    close = df['Close'].iloc[:date_idx+1]
    if len(close) < 90: return False

    cur = close.iloc[-1]
    sma30 = close.rolling(30).mean().iloc[-1]
    mom21 = calc_ret(close, 21)
    vol90 = get_volatility(close, 90)

    if cfg.health == 'baseline':
        return cur > sma30 and mom21 > 0 and vol90 <= 0.10

    elif cfg.health == 'adx':
        # Baseline + ADX > 25
        if not (cur > sma30 and mom21 > 0 and vol90 <= 0.10): return False
        high = df['High'].iloc[:date_idx+1]
        low = df['Low'].iloc[:date_idx+1]
        adx = calc_adx(high, low, close)
        return pd.notna(adx) and adx > 25

    elif cfg.health == 'drawdown':
        # Price > SMA30 AND Mom21 > 0 AND DD14 > -20% (no vol cap)
        dd14 = calc_drawdown(close, 14)
        return cur > sma30 and mom21 > 0 and dd14 > -0.20

    elif cfg.health == 'dual_ma':
        # SMA10 > SMA30 AND Price > SMA10
        sma10 = close.rolling(10).mean().iloc[-1]
        return cur > sma10 and sma10 > sma30

    return False


def score_coin(cfg: StrategyConfig, prices: dict, ticker: str, date_idx: int) -> float:
    """Score a single coin. Returns score value."""
    if ticker not in prices: return -999
    df = prices[ticker]
    close = df['Close'].iloc[:date_idx+1]
    if len(close) < 252: return -999

    if cfg.scoring == 'sharpe_bonus':
        # Baseline: Sharpe(126) + Sharpe(252) + RSI/MACD/BB bonuses
        base = calc_sharpe(close, 126) + calc_sharpe(close, 252)
        rsi = calc_rsi(close)
        macd_h = calc_macd_hist(close)
        pctb = calc_bb_pctb(close)
        if pd.notna(rsi) and 45 <= rsi <= 70: base += 0.2
        if pd.notna(macd_h) and macd_h > 0: base += 0.2
        if pd.notna(pctb) and pctb > 0.5: base += 0.2
        return base

    elif cfg.scoring == 'sortino_short':
        # Sortino(30) + Sortino(90)
        return calc_sortino(close, 30) + calc_sortino(close, 90)

    elif cfg.scoring == 'sortino_mid':
        # Sortino(60) + Sortino(126)
        return calc_sortino(close, 60) + calc_sortino(close, 126)

    elif cfg.scoring == 'sharpe_cmf':
        # Baseline Sharpe + CMF bonus
        base = calc_sharpe(close, 126) + calc_sharpe(close, 252)
        rsi = calc_rsi(close)
        macd_h = calc_macd_hist(close)
        pctb = calc_bb_pctb(close)
        if pd.notna(rsi) and 45 <= rsi <= 70: base += 0.2
        if pd.notna(macd_h) and macd_h > 0: base += 0.2
        if pd.notna(pctb) and pctb > 0.5: base += 0.2
        # Add CMF bonus
        high = df['High'].iloc[:date_idx+1]
        low = df['Low'].iloc[:date_idx+1]
        vol = df['Volume'].iloc[:date_idx+1]
        cmf = calc_cmf(high, low, close, vol)
        if pd.notna(cmf) and cmf > 0.05: base += 0.2
        return base

    elif cfg.scoring == 'pure_momentum':
        # Pure return: Ret(30) + Ret(90)
        return calc_ret(close, 30) + calc_ret(close, 90)

    return 0


def compute_weights(cfg: StrategyConfig, prices: dict, picks: list, date_idx: int) -> dict:
    """Compute portfolio weights for selected coins."""
    if not picks: return {}

    if cfg.weighting == 'equal':
        return {t: 1.0/len(picks) for t in picks}

    vol_window = 90
    if cfg.weighting == 'inv_vol30': vol_window = 30
    elif cfg.weighting == 'inv_vol20': vol_window = 20

    vols = {}
    for t in picks:
        if t in prices:
            close = prices[t]['Close'].iloc[:date_idx+1]
            v = get_volatility(close, vol_window)
            if v > 0: vols[t] = v

    if not vols:
        return {t: 1.0/len(picks) for t in picks}

    inv_vols = {t: 1.0/v for t, v in vols.items()}
    total = sum(inv_vols.values())
    return {t: w/total for t, w in inv_vols.items()}


# ─── Backtest Engine ────────────────────────────────────────────────

def run_backtest(cfg: StrategyConfig, prices: dict, universe_map: dict,
                 start_date: str = '2019-01-01', end_date: str = '2025-12-31',
                 initial_capital: float = 10000, tx_cost: float = 0.002):
    """
    Run monthly rebalancing backtest.
    Returns DataFrame of daily portfolio values.
    """
    # Get BTC trading days as date index
    btc = prices.get('BTC-USD')
    if btc is None:
        return pd.Series(dtype=float)

    all_dates = btc.index[(btc.index >= start_date) & (btc.index <= end_date)]
    if len(all_dates) == 0:
        return pd.Series(dtype=float)

    # Track portfolio
    capital = initial_capital
    holdings = {}  # ticker -> units
    cash = initial_capital
    portfolio_values = []
    rebal_count = 0

    prev_month = None

    for i, date in enumerate(all_dates):
        # Get global date index for this date
        global_idx = btc.index.get_loc(date)

        # 1. Mark to market
        port_val = cash
        for t, units in holdings.items():
            if t in prices:
                idx = prices[t].index.get_indexer([date], method='ffill')[0]
                if idx >= 0:
                    port_val += units * prices[t]['Close'].iloc[idx]

        # 2. Monthly rebalancing check
        current_month = date.strftime('%Y-%m')
        should_rebal = (prev_month is not None and current_month != prev_month)

        if should_rebal or (i == 0):
            # Determine universe for this month
            month_key = date.strftime('%Y-%m') + '-01'
            # Find closest universe month
            uni_tickers = []
            for mk in sorted(universe_map.keys(), reverse=True):
                if mk <= month_key:
                    uni_tickers = universe_map[mk]
                    break
            if not uni_tickers and universe_map:
                uni_tickers = list(universe_map.values())[0]

            # Filter stablecoins
            uni_clean = [t.replace('-USD', '') for t in uni_tickers if t.replace('-USD', '') not in STABLECOINS]

            # Check canary
            risk_on = check_canary(cfg, prices, uni_clean, global_idx)

            if risk_on:
                # Health filter
                healthy = []
                for sym in uni_clean:
                    ticker = f"{sym}-USD"
                    if check_health(cfg, prices, ticker, global_idx):
                        healthy.append(ticker)

                if healthy:
                    # Score and rank
                    scores = []
                    for t in healthy:
                        sc = score_coin(cfg, prices, t, global_idx)
                        if sc > -999:
                            scores.append((t, sc))

                    scores.sort(key=lambda x: x[1], reverse=True)
                    picks = [t for t, _ in scores[:cfg.n_picks]]

                    if picks:
                        # Compute weights
                        weights = compute_weights(cfg, prices, picks, global_idx)

                        # Rebalance: sell everything, buy new
                        # Liquidate
                        sell_value = 0
                        for t, units in holdings.items():
                            if t in prices:
                                idx2 = prices[t].index.get_indexer([date], method='ffill')[0]
                                if idx2 >= 0:
                                    sell_value += units * prices[t]['Close'].iloc[idx2]

                        total_value = cash + sell_value
                        total_value *= (1 - tx_cost)  # sell cost

                        # Buy new
                        holdings = {}
                        cash = 0
                        for t, w in weights.items():
                            if t in prices:
                                idx2 = prices[t].index.get_indexer([date], method='ffill')[0]
                                if idx2 >= 0:
                                    price = prices[t]['Close'].iloc[idx2]
                                    if price > 0:
                                        alloc = total_value * w * (1 - tx_cost)  # buy cost
                                        holdings[t] = alloc / price
                                        cash += total_value * w - alloc  # leftover from cost

                        rebal_count += 1
                    else:
                        # No valid picks - go to cash
                        sell_value = 0
                        for t, units in holdings.items():
                            if t in prices:
                                idx2 = prices[t].index.get_indexer([date], method='ffill')[0]
                                if idx2 >= 0:
                                    sell_value += units * prices[t]['Close'].iloc[idx2]
                        cash = (cash + sell_value) * (1 - tx_cost)
                        holdings = {}
                        rebal_count += 1
                else:
                    # No healthy coins
                    sell_value = 0
                    for t, units in holdings.items():
                        if t in prices:
                            idx2 = prices[t].index.get_indexer([date], method='ffill')[0]
                            if idx2 >= 0:
                                sell_value += units * prices[t]['Close'].iloc[idx2]
                    cash = (cash + sell_value) * (1 - tx_cost)
                    holdings = {}
                    rebal_count += 1
            else:
                # Risk-Off: all cash
                sell_value = 0
                for t, units in holdings.items():
                    if t in prices:
                        idx2 = prices[t].index.get_indexer([date], method='ffill')[0]
                        if idx2 >= 0:
                            sell_value += units * prices[t]['Close'].iloc[idx2]
                if sell_value > 0 or holdings:
                    cash = (cash + sell_value) * (1 - tx_cost)
                    holdings = {}
                    rebal_count += 1

        prev_month = current_month
        portfolio_values.append({'Date': date, 'Value': port_val})

    result = pd.DataFrame(portfolio_values).set_index('Date')
    result.attrs['rebal_count'] = rebal_count
    return result


# ─── Metrics ────────────────────────────────────────────────────────

def calc_metrics(pv: pd.Series) -> dict:
    """Calculate performance metrics from portfolio value series."""
    if len(pv) < 2:
        return {'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'WinRate': 0}

    values = pv['Value']
    days = (pv.index[-1] - pv.index[0]).days
    years = days / 365.25

    # CAGR
    total_ret = values.iloc[-1] / values.iloc[0]
    cagr = total_ret ** (1/years) - 1 if years > 0 else 0

    # MDD
    peak = values.cummax()
    dd = (values / peak - 1)
    mdd = dd.min()

    # Daily returns
    daily_ret = values.pct_change().dropna()

    # Sharpe (annualized)
    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(365)
    else:
        sharpe = 0

    # Sortino
    down_ret = daily_ret[daily_ret < 0]
    if len(down_ret) > 1 and down_ret.std() > 0:
        sortino = (daily_ret.mean() / down_ret.std()) * np.sqrt(365)
    else:
        sortino = sharpe

    # Win Rate (monthly)
    monthly = values.resample('M').last().pct_change().dropna()
    win_rate = (monthly > 0).mean() if len(monthly) > 0 else 0

    # Final value
    final = values.iloc[-1]

    return {
        'Final': final,
        'CAGR': cagr,
        'MDD': mdd,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'WinRate': win_rate,
    }


def calc_yearly_metrics(pv: pd.DataFrame) -> dict:
    """Calculate per-year metrics."""
    yearly = {}
    for year in range(pv.index[0].year, pv.index[-1].year + 1):
        mask = pv.index.year == year
        if mask.sum() < 10: continue
        year_data = pv[mask].copy()
        yearly[year] = calc_metrics(year_data)
    return yearly


# ─── Main ───────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Coin Strategy Backtest')
    parser.add_argument('--start', default='2019-01-01', help='Start date')
    parser.add_argument('--end', default='2025-12-31', help='End date')
    parser.add_argument('--download', action='store_true', help='Download missing data')
    parser.add_argument('--strategy', default=None, help='Run specific strategy only')
    args = parser.parse_args()

    print("=" * 90)
    print(f"  COIN STRATEGY BACKTEST  ({args.start} ~ {args.end})")
    print("=" * 90)

    # Load universe
    print("\n[1/4] Loading historical universe...")
    universe_map = load_universe()
    print(f"  {len(universe_map)} months loaded ({sorted(universe_map.keys())[0]} ~ {sorted(universe_map.keys())[-1]})")

    # Collect all unique tickers
    all_tickers = set()
    for month_tickers in universe_map.values():
        for t in month_tickers:
            sym = t.replace('-USD', '')
            if sym not in STABLECOINS:
                all_tickers.add(t)
    all_tickers.add('BTC-USD')
    all_tickers.add('ETH-USD')
    print(f"  {len(all_tickers)} unique tickers across all months")

    # Download missing data if requested
    if args.download:
        print("\n[DOWNLOAD] Downloading missing price data...")
        import yfinance as yf
        missing = [t for t in all_tickers if not os.path.exists(os.path.join(DATA_DIR, f"{t}.csv"))]
        print(f"  {len(missing)} tickers to download")
        for i, t in enumerate(missing):
            try:
                df = yf.download(t, start='2016-01-01', progress=False)
                if len(df) > 0:
                    df.to_csv(os.path.join(DATA_DIR, f"{t}.csv"))
                    if (i+1) % 50 == 0:
                        print(f"  ... {i+1}/{len(missing)}")
            except: pass
        print(f"  Download complete")

    # Load prices
    print("\n[2/4] Loading price data...")
    prices = load_all_prices(all_tickers)
    print(f"  {len(prices)} tickers loaded successfully")

    # BTC benchmark
    print("\n[3/4] Running backtests...")
    btc_pv = None
    if 'BTC-USD' in prices:
        btc = prices['BTC-USD']
        btc_dates = btc.index[(btc.index >= args.start) & (btc.index <= args.end)]
        btc_pv = pd.DataFrame({
            'Date': btc_dates,
            'Value': 10000 * btc.loc[btc_dates, 'Close'] / btc.loc[btc_dates, 'Close'].iloc[0]
        }).set_index('Date')

    # Run all strategies
    strategies = get_all_strategies()
    if args.strategy:
        strategies = [s for s in strategies if args.strategy.lower() in s.name.lower()]

    results = {}
    for si, cfg in enumerate(strategies):
        print(f"  [{si+1}/{len(strategies)}] {cfg.name}...", end='', flush=True)
        pv = run_backtest(cfg, prices, universe_map, args.start, args.end)
        if len(pv) > 0:
            metrics = calc_metrics(pv)
            metrics['Rebals'] = pv.attrs.get('rebal_count', 0)
            results[cfg.name] = {'pv': pv, 'metrics': metrics, 'yearly': calc_yearly_metrics(pv)}
            print(f" CAGR={metrics['CAGR']:+.1%} MDD={metrics['MDD']:.1%} Sharpe={metrics['Sharpe']:.3f}")
        else:
            print(" NO DATA")

    # BTC B&H benchmark
    if btc_pv is not None and len(btc_pv) > 0:
        btc_metrics = calc_metrics(btc_pv)
        results['BTC_BuyHold'] = {'pv': btc_pv, 'metrics': btc_metrics, 'yearly': calc_yearly_metrics(btc_pv)}

    # ─── Output ─────────────────────────────────────────────────────
    print("\n" + "=" * 90)
    print(f"{'Strategy':<32} {'Final($)':>10} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>8} {'WinRate':>8} {'Rebals':>7}")
    print("-" * 90)

    # Sort by Sharpe
    sorted_results = sorted(results.items(), key=lambda x: x[1]['metrics']['Sharpe'], reverse=True)

    for name, data in sorted_results:
        m = data['metrics']
        print(f"{name:<32} {m.get('Final', 10000):>10,.0f} {m['CAGR']:>+7.1%} {m['MDD']:>7.1%} {m['Sharpe']:>8.3f} {m['Sortino']:>8.3f} {m['WinRate']:>7.0%} {m.get('Rebals', 0):>7}")

    # Yearly breakdown for top 5 + baseline
    print("\n" + "=" * 90)
    print("  YEARLY BREAKDOWN (Top 5 + Baseline + BTC)")
    print("=" * 90)

    show_strategies = []
    # Always include baseline
    if 'Baseline' in results: show_strategies.append('Baseline')
    # Top 5 by Sharpe (excluding baseline if already added)
    for name, _ in sorted_results[:5]:
        if name not in show_strategies and name != 'BTC_BuyHold':
            show_strategies.append(name)
    if 'BTC_BuyHold' in results: show_strategies.append('BTC_BuyHold')

    for name in show_strategies:
        if name not in results: continue
        data = results[name]
        print(f"\n  {name}")
        print(f"  {'Year':<8} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8}")
        print(f"  {'-'*32}")
        for year, ym in sorted(data['yearly'].items()):
            print(f"  {year:<8} {ym['CAGR']:>+7.1%} {ym['MDD']:>7.1%} {ym['Sharpe']:>8.3f}")
        m = data['metrics']
        print(f"  {'OVERALL':<8} {m['CAGR']:>+7.1%} {m['MDD']:>7.1%} {m['Sharpe']:>8.3f}")

    # Component comparison
    print("\n" + "=" * 90)
    print("  COMPONENT ANALYSIS (avg improvement vs Baseline)")
    print("=" * 90)

    baseline_sharpe = results.get('Baseline', {}).get('metrics', {}).get('Sharpe', 0)

    groups = {
        'A. Scoring': ['A1_Sortino30+90', 'A2_Sortino60+126', 'A3_Sharpe+CMF', 'A4_PureMomentum'],
        'B. Canary': ['B1_DualMomentum', 'B2_MarketBreadth', 'B3_BTC+ETH'],
        'C. Health': ['C1_ADX', 'C2_Drawdown', 'C3_DualMA'],
        'D. Weighting': ['D1_InvVol30', 'D2_InvVol20', 'D3_EqualWeight'],
        'E. Composite': ['E1_Sort+ETH+DD+Vol30', 'E2_Sort+Breadth+ADX+Vol30', 'E3_SortMid+Dual+DualMA+Vol30', 'E4_PureMom+Dual+ADX+EW'],
    }

    for group_name, members in groups.items():
        print(f"\n  {group_name}:")
        best_name, best_sharpe = 'None', -999
        for name in members:
            if name in results:
                m = results[name]['metrics']
                diff = m['Sharpe'] - baseline_sharpe
                marker = '***' if diff > 0.1 else '**' if diff > 0.05 else '*' if diff > 0 else ''
                print(f"    {name:<32} Sharpe={m['Sharpe']:>7.3f} ({diff:>+.3f}) MDD={m['MDD']:>7.1%} {marker}")
                if m['Sharpe'] > best_sharpe:
                    best_sharpe = m['Sharpe']
                    best_name = name
        print(f"    >> Best: {best_name}")

    print("\n" + "=" * 90)
    print("  RECOMMENDATION")
    print("=" * 90)

    top3 = sorted_results[:3]
    for i, (name, data) in enumerate(top3):
        m = data['metrics']
        print(f"  #{i+1} {name}: Sharpe={m['Sharpe']:.3f}, CAGR={m['CAGR']:+.1%}, MDD={m['MDD']:.1%}")

    print(f"\n  Baseline: Sharpe={baseline_sharpe:.3f}")
    if top3[0][1]['metrics']['Sharpe'] > baseline_sharpe:
        best = top3[0]
        print(f"  Best improvement: {best[0]} (+{best[1]['metrics']['Sharpe'] - baseline_sharpe:.3f} Sharpe)")
    else:
        print(f"  Baseline is already optimal!")

    print()


if __name__ == '__main__':
    main()
