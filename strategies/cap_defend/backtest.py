"""
Cap Defend V12 Backtest Framework
==================================
- 1년 단위 구간별 성과 + 전체 성과 출력
- CoinMarketCap 히스토리컬 데이터로 시점별 코인 유니버스 구성
- 파라미터 A/B 테스트 지원
- 벤치마크 (SPY B&H, BTC B&H) 동시 계산

Usage:
    python3 strategies/cap_defend/backtest.py                        # 기본 실행
    python3 strategies/cap_defend/backtest.py --start 2019-01-01     # 기간 변경
    python3 strategies/cap_defend/backtest.py --compare vol_cap=0.04 # A/B 테스트
    python3 strategies/cap_defend/backtest.py --download              # 데이터 다운로드
"""

import os
import sys
import json
import argparse
import warnings
from dataclasses import dataclass, replace
from datetime import datetime, timezone
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# 1. Params dataclass — 모든 파라미터 중앙 관리
# ---------------------------------------------------------------------------

@dataclass
class Params:
    # Portfolio allocation
    stock_ratio: float = 0.60
    coin_ratio: float = 0.40
    cash_buffer: float = 0.02

    # Stock canary
    stock_canary_ma: int = 200

    # Coin canary
    coin_canary_ma: int = 50

    # Momentum weights (3m, 6m, 12m)
    mom_weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)

    # Volatility / health filters
    vol_cap: float = 0.10
    vol_window: int = 90
    health_ma: int = 30
    health_mom: int = 21

    # Coin selection
    n_coin_picks: int = 5
    sharpe_windows: Tuple[int, int] = (126, 252)

    # Rebalancing
    turnover_threshold: float = 0.30
    rebal_freq: str = 'M'   # 'M' = monthly

    # Costs
    tx_cost: float = 0.001
    initial_capital: float = 10000.0

    # Date range
    start_date: str = '2021-01-01'
    end_date: str = '2025-12-31'

    # Paths
    data_dir: str = './data'
    universe_file: str = './data/historical_universe.json'

    # Universes (stock side is fixed)
    offensive_stocks: Tuple[str, ...] = (
        'SPY', 'QQQ', 'EFA', 'EEM', 'VT', 'VEA',
        'GLD', 'PDBC', 'QUAL', 'MTUM', 'IQLT', 'IMTM',
    )
    defensive_stocks: Tuple[str, ...] = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')
    canary_assets: Tuple[str, ...] = ('VT', 'EEM')

    stablecoins: Tuple[str, ...] = (
        'USDT', 'USDC', 'BUSD', 'DAI', 'UST', 'TUSD', 'PAX',
        'GUSD', 'FRAX', 'LUSD', 'MIM', 'USDN', 'FDUSD', 'USDS', 'PYUSD', 'USDE',
    )


# ---------------------------------------------------------------------------
# 2. Data pipeline
# ---------------------------------------------------------------------------

def load_historical_universe(path: str) -> dict:
    if os.path.exists(path):
        with open(path, 'r') as f:
            return json.load(f)
    return {}


def collect_all_tickers(params: Params, hist_universe: dict) -> set:
    """Scan the historical universe JSON and collect every ticker we'll need."""
    tickers = set(params.offensive_stocks) | set(params.defensive_stocks) | set(params.canary_assets)
    tickers.add('BTC-USD')
    tickers.add('SPY')  # benchmark
    for symbols in hist_universe.values():
        for s in symbols:
            t = s if s.endswith('-USD') else f"{s}-USD"
            sym = t.replace('-USD', '')
            if sym not in params.stablecoins:
                tickers.add(t)
    return tickers


def download_data(tickers: set, params: Params):
    """Download price data from Yahoo Finance for all tickers."""
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry

    os.makedirs(params.data_dir, exist_ok=True)

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[500, 502, 503, 504])
    session.mount('https://', HTTPAdapter(max_retries=retries))

    downloaded, skipped, failed = 0, 0, 0
    total = len(tickers)

    for i, ticker in enumerate(sorted(tickers), 1):
        fp = os.path.join(params.data_dir, f"{ticker}.csv")

        # Skip if file exists and is recent enough (< 1 day old)
        if os.path.exists(fp):
            mtime = datetime.fromtimestamp(os.path.getmtime(fp))
            if (datetime.now() - mtime).total_seconds() < 86400:
                skipped += 1
                continue

        try:
            end_ts = int(datetime.now(timezone.utc).timestamp())
            start_ts = int(datetime(2014, 1, 1, tzinfo=timezone.utc).timestamp())
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            resp_params = {
                "period1": start_ts, "period2": end_ts,
                "interval": "1d", "includeAdjustedClose": "true"
            }
            resp = session.get(url, params=resp_params, timeout=30)

            if resp.status_code == 200:
                res = resp.json()['chart']['result'][0]
                ts = res.get('timestamp')
                adj = res['indicators']['adjclose'][0]['adjclose']
                if ts and adj:
                    df = pd.DataFrame({
                        'Date': pd.to_datetime(ts, unit='s').date,
                        'Adj_Close': adj
                    })
                    df = df.dropna().drop_duplicates('Date')
                    if len(df) > 10:
                        df.to_csv(fp, index=False)
                        downloaded += 1
                        if downloaded % 50 == 0:
                            print(f"  [{i}/{total}] Downloaded {downloaded} tickers...")
                        continue

            failed += 1
        except Exception:
            failed += 1

    print(f"Download complete: {downloaded} new, {skipped} cached, {failed} failed (total {total})")


def load_all_data(tickers: set, params: Params) -> pd.DataFrame:
    """Load all CSV files into a single DataFrame, forward-filled."""
    # Buffer: load 2 years before start_date for indicator warm-up
    buffer_start = (pd.to_datetime(params.start_date) - pd.DateOffset(years=2)).strftime('%Y-%m-%d')

    data_dict = {}
    for ticker in tickers:
        fp = os.path.join(params.data_dir, f"{ticker}.csv")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_csv(fp, parse_dates=['Date'])
            df = df.drop_duplicates(subset=['Date'], keep='first').set_index('Date')
            col = 'Adj Close' if 'Adj Close' in df else ('Adj_Close' if 'Adj_Close' in df else 'Close')
            if col in df:
                data_dict[ticker] = df[col]
        except Exception:
            pass

    if not data_dict:
        print("ERROR: No data loaded!")
        sys.exit(1)

    full_index = pd.date_range(start=buffer_start, end=params.end_date, freq='D')
    result = pd.DataFrame(data_dict).reindex(full_index).ffill()
    print(f"Loaded {len(data_dict)} tickers, date range: {result.index[0].date()} ~ {result.index[-1].date()}")
    return result


# ---------------------------------------------------------------------------
# 3. Indicator helpers
# ---------------------------------------------------------------------------

def get_return(s: pd.Series, n: int) -> float:
    s = s.dropna()
    if len(s) > n and s.iloc[-n - 1] != 0:
        return s.iloc[-1] / s.iloc[-n - 1] - 1
    return -np.inf


def get_sharpe(s: pd.Series, n: int) -> float:
    clean = s.dropna()
    if len(clean) < n + 1:
        return -np.inf
    ret = s.pct_change().iloc[-n:].dropna()
    if ret.empty or ret.std() == 0:
        return 0.0
    return (ret.mean() / ret.std()) * np.sqrt(252)


def calc_weighted_momentum(s: pd.Series, weights: Tuple[float, float, float] = (0.5, 0.3, 0.2)) -> float:
    if len(s.dropna()) < 253:
        return -np.inf
    r3 = get_return(s, 63)
    r6 = get_return(s, 126)
    r12 = get_return(s, 252)
    if any(r == -np.inf for r in [r3, r6, r12]):
        return -np.inf
    return weights[0] * r3 + weights[1] * r6 + weights[2] * r12


# ---------------------------------------------------------------------------
# 4. Strategy logic — exact replica of live code
# ---------------------------------------------------------------------------

def get_coin_universe(date: pd.Timestamp, hist_universe: dict, params: Params, top_n: int = 50) -> List[str]:
    """Get coin universe for a given date from historical JSON."""
    key = date.strftime("%Y-%m") + "-01"
    symbols = hist_universe.get(key, [])
    if not symbols:
        available = sorted([k for k in hist_universe.keys() if k <= key], reverse=True)
        if available:
            symbols = hist_universe[available[0]]

    final = []
    for s in symbols:
        ticker = s if s.endswith('-USD') else f"{s}-USD"
        sym = ticker.replace('-USD', '')
        if sym in params.stablecoins:
            continue
        final.append(ticker)
        if len(final) >= top_n:
            break
    return final


def get_stock_portfolio(date: pd.Timestamp, data: pd.DataFrame, params: Params) -> Tuple[Dict[str, float], str]:
    """Stock strategy: canary → offensive or defensive."""
    prices = data.loc[:date]
    if len(prices) < params.stock_canary_ma:
        return {}, "No Data"

    # Canary check: VT AND EEM > MA200
    canary_ok = True
    for c in params.canary_assets:
        if c not in prices or prices[c].isna().all():
            canary_ok = False
            break
        p = prices[c].dropna()
        if len(p) < params.stock_canary_ma:
            canary_ok = False
            break
        if p.iloc[-1] <= p.rolling(params.stock_canary_ma).mean().iloc[-1]:
            canary_ok = False
            break

    if canary_ok:
        # Offensive: top 3 momentum + top 3 quality (Sharpe 126d), equal weight
        candidates = []
        for t in params.offensive_stocks:
            if t not in prices or prices[t].isna().all():
                continue
            p = prices[t].dropna()
            if len(p) < 253:
                continue
            mom = calc_weighted_momentum(p, params.mom_weights)
            qual = get_sharpe(p, params.sharpe_windows[0])
            if mom != -np.inf and qual != -np.inf:
                candidates.append({'Ticker': t, 'Mom': mom, 'Qual': qual})

        if not candidates:
            return {}, "Risk-On (No Data)"

        df = pd.DataFrame(candidates).set_index('Ticker')
        top_m = df.nlargest(3, 'Mom').index.tolist()
        top_q = df.nlargest(3, 'Qual').index.tolist()
        picks = list(set(top_m + top_q))
        return {t: 1.0 / len(picks) for t in picks}, "Risk-On"
    else:
        # Defensive: best 6m return among defensive assets
        best_t, best_r = None, -999.0
        for t in params.defensive_stocks:
            if t not in prices or prices[t].isna().all():
                continue
            r = get_return(prices[t], 126)
            if r != -np.inf and r > best_r:
                best_r, best_t = r, t
        if best_t is None or best_r < 0:
            return {}, "Risk-Off (Cash)"
        return {best_t: 1.0}, "Risk-Off"


def check_coin_health(coin: str, date: pd.Timestamp, data: pd.DataFrame, params: Params) -> bool:
    """V12 Health Check: Price > SMA(health_ma) AND mom(health_mom) > 0 AND vol(vol_window) <= vol_cap."""
    if coin not in data or data[coin].isna().all():
        return False
    p = data[coin].loc[:date].dropna()
    if len(p) < max(params.vol_window + 1, params.health_ma + 1):
        return False

    # Technical health
    sma = p.rolling(params.health_ma).mean().iloc[-1]
    ret = get_return(p, params.health_mom)
    if p.iloc[-1] <= sma or ret <= 0:
        return False

    # Volatility cap
    vol = p.pct_change().iloc[-params.vol_window:].std()
    return vol <= params.vol_cap


def get_coin_portfolio(date: pd.Timestamp, data: pd.DataFrame, hist_universe: dict,
                       params: Params) -> Tuple[Dict[str, float], str, List[str]]:
    """Coin strategy: BTC canary → health filter → Sharpe scoring → inverse vol weighting."""
    if 'BTC-USD' not in data or data['BTC-USD'].isna().all():
        return {}, "No BTC Data", []

    btc = data['BTC-USD'].loc[:date].dropna()
    if len(btc) < params.coin_canary_ma:
        return {}, "No Data", []

    # Canary: BTC > MA(coin_canary_ma)
    if btc.iloc[-1] <= btc.rolling(params.coin_canary_ma).mean().iloc[-1]:
        return {}, "Risk-Off", []

    univ = get_coin_universe(date, hist_universe, params)

    # Health filter
    healthy = [c for c in univ if c in data and check_coin_health(c, date, data, params)]
    if not healthy:
        return {}, "No Healthy", []

    # Sharpe scoring: sum of sharpe(window1) + sharpe(window2)
    scores = {}
    for c in healthy:
        p = data[c].loc[:date].dropna()
        if len(p) < max(params.sharpe_windows) + 1:
            continue
        s1 = get_sharpe(p, params.sharpe_windows[0])
        s2 = get_sharpe(p, params.sharpe_windows[1])
        if s1 != -np.inf and s2 != -np.inf:
            scores[c] = s1 + s2

    if not scores:
        return {}, "No Scores", healthy

    top = pd.Series(scores).nlargest(params.n_coin_picks).index.tolist()

    # Inverse volatility weighting
    vols = {}
    for c in top:
        p = data[c].loc[:date].dropna()
        v = p.pct_change().iloc[-params.vol_window:].std()
        vols[c] = v

    inv = {c: 1.0 / v if v > 0 else 0 for c, v in vols.items()}
    tot = sum(inv.values())
    if tot > 0:
        port = {c: v / tot for c, v in inv.items()}
    else:
        port = {c: 1.0 / len(top) for c in top}

    return port, "Risk-On", healthy


def calc_turnover(holding: dict, target: dict, prices: dict, total_val: float) -> float:
    """Calculate portfolio turnover between current holding and target."""
    if total_val <= 0:
        return 1.0
    cur_w = {}
    for t, units in holding.items():
        p = prices.get(t, 0)
        if p > 0:
            cur_w[t] = (units * p) / total_val
    all_t = set(cur_w.keys()) | set(target.keys())
    return sum(abs(target.get(t, 0) - cur_w.get(t, 0)) for t in all_t) / 2


# ---------------------------------------------------------------------------
# 5. Backtest engine
# ---------------------------------------------------------------------------

def run_backtest(data: pd.DataFrame, params: Params, hist_universe: dict, label: str = "Strategy") -> pd.DataFrame:
    """
    Daily backtest loop:
      1. Mark-to-market
      2. Generate signals
      3. Check rebalance triggers (monthly / turnover / health ejection)
      4. Execute rebalance (with tx cost)
      5. Record history
    """
    dates = pd.date_range(start=params.start_date, end=params.end_date, freq='D')
    dates = dates[dates.isin(data.index)]
    month_ends = set(pd.date_range(start=params.start_date, end=params.end_date, freq='M'))

    s_cash = params.initial_capital * params.stock_ratio
    c_cash = params.initial_capital * params.coin_ratio
    s_hold: Dict[str, float] = {}  # ticker -> units
    c_hold: Dict[str, float] = {}

    history = []
    prev_s_stat = None
    rebal_count = 0

    for today in dates:
        row = data.loc[today]

        # 1. Mark-to-market
        s_val = s_cash + sum(h * (row.get(t, 0) if pd.notna(row.get(t, 0)) else 0) for t, h in s_hold.items())
        c_val = c_cash + sum(h * (row.get(t, 0) if pd.notna(row.get(t, 0)) else 0) for t, h in c_hold.items())
        t_val = s_val + c_val

        history.append({'Date': today, 'Value': t_val, 'StockVal': s_val, 'CoinVal': c_val})

        # 2. Signals
        s_port, s_stat = get_stock_portfolio(today, data, params)
        c_port, c_stat, healthy_coins = get_coin_portfolio(today, data, hist_universe, params)

        # 3. Rebalance triggers
        is_s_flip = prev_s_stat is not None and s_stat != prev_s_stat
        prev_s_stat = s_stat

        # Coin turnover
        c_prices = {t: (row.get(t, 0) if pd.notna(row.get(t, 0)) else 0)
                     for t in set(list(c_hold.keys()) + list(c_port.keys()))}
        c_turn = calc_turnover(c_hold, c_port, c_prices, c_val)
        is_c_turn = c_turn > params.turnover_threshold

        # Health ejection: any held coin no longer healthy
        is_health_ejection = False
        if c_hold:
            for held_coin in c_hold:
                if not check_coin_health(held_coin, today, data, params):
                    is_health_ejection = True
                    break

        is_monthly = today in month_ends
        do_global = is_monthly or is_s_flip
        do_coin_only = is_c_turn or is_health_ejection

        # 4. Rebalance execution
        if do_global or do_coin_only:
            rebal_count += 1
            cost_mult = 1 - params.tx_cost

            if do_global:
                # Full rebalance: both stock and coin
                s_tgt = t_val * params.stock_ratio * cost_mult
                c_tgt = t_val * params.coin_ratio * cost_mult

                s_cash = s_tgt
                s_hold = {}
                for t, w in s_port.items():
                    p = row.get(t, 0) if pd.notna(row.get(t, 0)) else 0
                    if p > 0:
                        amt = s_tgt * w
                        s_hold[t] = amt / p
                        s_cash -= amt

                c_cash = c_tgt
                c_hold = {}
                for t, w in c_port.items():
                    p = row.get(t, 0) if pd.notna(row.get(t, 0)) else 0
                    if p > 0:
                        amt = c_tgt * w
                        c_hold[t] = amt / p
                        c_cash -= amt

            elif do_coin_only:
                # Coin-only rebalance
                c_tgt = c_val * cost_mult
                c_cash = c_tgt
                c_hold = {}
                for t, w in c_port.items():
                    p = row.get(t, 0) if pd.notna(row.get(t, 0)) else 0
                    if p > 0:
                        amt = c_tgt * w
                        c_hold[t] = amt / p
                        c_cash -= amt

    df = pd.DataFrame(history).set_index('Date')
    df.attrs['rebal_count'] = rebal_count
    df.attrs['label'] = label
    return df


# ---------------------------------------------------------------------------
# 6. Metrics & reporting
# ---------------------------------------------------------------------------

def calc_metrics(values: pd.Series) -> dict:
    """Calculate performance metrics for a value series."""
    if len(values) < 2:
        return {'cagr': 0, 'mdd': 0, 'sharpe': 0, 'sortino': 0, 'win_rate': 0}

    days = (values.index[-1] - values.index[0]).days
    if days <= 0:
        return {'cagr': 0, 'mdd': 0, 'sharpe': 0, 'sortino': 0, 'win_rate': 0}

    final_val = values.iloc[-1]
    init_val = values.iloc[0]

    cagr = (final_val / init_val) ** (365.25 / days) - 1
    mdd = (values / values.cummax() - 1).min()

    daily_ret = values.pct_change().dropna()
    if daily_ret.std() > 0:
        sharpe = (daily_ret.mean() / daily_ret.std()) * np.sqrt(252)
    else:
        sharpe = 0.0

    downside = daily_ret[daily_ret < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = (daily_ret.mean() / downside.std()) * np.sqrt(252)
    else:
        sortino = 0.0

    win_rate = (daily_ret > 0).sum() / len(daily_ret) if len(daily_ret) > 0 else 0

    return {
        'cagr': cagr,
        'mdd': mdd,
        'sharpe': sharpe,
        'sortino': sortino,
        'win_rate': win_rate,
        'final': final_val,
    }


def calc_benchmark(data: pd.DataFrame, ticker: str, start: str, end: str) -> dict:
    """Calculate B&H benchmark metrics."""
    if ticker not in data:
        return {'cagr': 0, 'mdd': 0, 'sharpe': 0}

    s = data[ticker].loc[start:end].dropna()
    if len(s) < 2:
        return {'cagr': 0, 'mdd': 0, 'sharpe': 0}

    return calc_metrics(s)


def print_report(results: pd.DataFrame, params: Params, bench_spy: dict, bench_btc: dict):
    """Print yearly + overall performance table."""
    rebal_count = results.attrs.get('rebal_count', 0)
    label = results.attrs.get('label', 'Strategy')

    years = sorted(results.index.year.unique())

    print(f"\n{'=' * 78}")
    print(f"  CAP DEFEND V12 BACKTEST — {label}")
    print(f"  Period: {params.start_date} ~ {params.end_date} | "
          f"Capital: ${params.initial_capital:,.0f} | "
          f"Stock/Coin: {params.stock_ratio:.0%}/{params.coin_ratio:.0%}")
    print(f"{'=' * 78}")
    print(f"{'Period':<12} {'Final($)':>12} {'CAGR':>9} {'MDD':>9} {'Sharpe':>8} {'Sortino':>8} {'WinRate':>8} {'Rebals':>7}")
    print(f"{'-' * 78}")

    for year in years:
        mask = results.index.year == year
        yr_data = results.loc[mask, 'Value']
        if len(yr_data) < 2:
            continue

        m = calc_metrics(yr_data)

        # Count rebals for this year (approximate from total)
        yr_rebals = '-'

        print(f"{year:<12} {m['final']:>12,.0f} {m['cagr']:>8.1%} {m['mdd']:>8.1%} "
              f"{m['sharpe']:>8.2f} {m['sortino']:>8.2f} {m['win_rate']:>7.1%} {yr_rebals:>7}")

    # Overall
    m_all = calc_metrics(results['Value'])
    print(f"{'-' * 78}")
    print(f"{'OVERALL':<12} {m_all['final']:>12,.0f} {m_all['cagr']:>8.1%} {m_all['mdd']:>8.1%} "
          f"{m_all['sharpe']:>8.2f} {m_all['sortino']:>8.2f} {m_all['win_rate']:>7.1%} {rebal_count:>7}")

    # Benchmarks
    print(f"\n  Benchmarks:  "
          f"SPY B&H  CAGR={bench_spy.get('cagr', 0):+.1%}  MDD={bench_spy.get('mdd', 0):.1%}  Sharpe={bench_spy.get('sharpe', 0):.2f}  |  "
          f"BTC B&H  CAGR={bench_btc.get('cagr', 0):+.1%}  MDD={bench_btc.get('mdd', 0):.1%}  Sharpe={bench_btc.get('sharpe', 0):.2f}")
    print(f"{'=' * 78}\n")


def print_comparison(res_a: pd.DataFrame, res_b: pd.DataFrame, params_a: Params, params_b: Params,
                     diff_desc: str, bench_spy: dict, bench_btc: dict):
    """Print side-by-side A/B comparison."""
    m_a = calc_metrics(res_a['Value'])
    m_b = calc_metrics(res_b['Value'])
    label_a = res_a.attrs.get('label', 'Baseline')
    label_b = res_b.attrs.get('label', 'Variant')
    rebal_a = res_a.attrs.get('rebal_count', 0)
    rebal_b = res_b.attrs.get('rebal_count', 0)

    print(f"\n{'=' * 78}")
    print(f"  A/B TEST COMPARISON: {diff_desc}")
    print(f"  Period: {params_a.start_date} ~ {params_a.end_date}")
    print(f"{'=' * 78}")
    print(f"{'Metric':<16} {'[A] ' + label_a:>28} {'[B] ' + label_b:>28}")
    print(f"{'-' * 78}")
    print(f"{'Final Value':<16} {m_a['final']:>28,.0f} {m_b['final']:>28,.0f}")
    print(f"{'CAGR':<16} {m_a['cagr']:>27.1%} {m_b['cagr']:>27.1%}")
    print(f"{'MDD':<16} {m_a['mdd']:>27.1%} {m_b['mdd']:>27.1%}")
    print(f"{'Sharpe':<16} {m_a['sharpe']:>28.2f} {m_b['sharpe']:>28.2f}")
    print(f"{'Sortino':<16} {m_a['sortino']:>28.2f} {m_b['sortino']:>28.2f}")
    print(f"{'Win Rate':<16} {m_a['win_rate']:>27.1%} {m_b['win_rate']:>27.1%}")
    print(f"{'Rebalances':<16} {rebal_a:>28} {rebal_b:>28}")
    print(f"{'-' * 78}")

    print(f"\n  Benchmarks:  "
          f"SPY B&H  CAGR={bench_spy.get('cagr', 0):+.1%}  MDD={bench_spy.get('mdd', 0):.1%}  |  "
          f"BTC B&H  CAGR={bench_btc.get('cagr', 0):+.1%}  MDD={bench_btc.get('mdd', 0):.1%}")
    print(f"{'=' * 78}\n")


# ---------------------------------------------------------------------------
# 7. CLI
# ---------------------------------------------------------------------------

def parse_param_override(s: str) -> Tuple[str, object]:
    """Parse 'key=value' string into (field_name, typed_value)."""
    if '=' not in s:
        raise ValueError(f"Invalid parameter format: {s} (expected key=value)")
    key, val = s.split('=', 1)
    key = key.strip()

    # Type inference based on Params dataclass fields
    field_types = {f.name: f.type for f in Params.__dataclass_fields__.values()}
    if key not in field_types:
        raise ValueError(f"Unknown parameter: {key}")

    ftype = field_types[key]
    if ftype == float or ftype == 'float':
        return key, float(val)
    elif ftype == int or ftype == 'int':
        return key, int(val)
    elif ftype == str or ftype == 'str':
        return key, val
    else:
        # Try float, then int, then string
        try:
            return key, float(val)
        except ValueError:
            return key, val


def main():
    parser = argparse.ArgumentParser(description='Cap Defend V12 Backtest')
    parser.add_argument('--start', type=str, help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, help='End date (YYYY-MM-DD)')
    parser.add_argument('--capital', type=float, help='Initial capital')
    parser.add_argument('--download', action='store_true', help='Download/refresh price data')
    parser.add_argument('--compare', type=str, nargs='+', metavar='key=value',
                        help='A/B test: compare baseline vs variant with changed params')
    parser.add_argument('--data-dir', type=str, default='./data', help='Data directory')
    parser.add_argument('--universe-file', type=str, default='./data/historical_universe.json',
                        help='Historical universe JSON file')
    args = parser.parse_args()

    # Base params
    params = Params(data_dir=args.data_dir, universe_file=args.universe_file)
    if args.start:
        params.start_date = args.start
    if args.end:
        params.end_date = args.end
    if args.capital:
        params.initial_capital = args.capital

    # Load universe
    hist_universe = load_historical_universe(params.universe_file)
    if not hist_universe:
        print(f"WARNING: No historical universe data at {params.universe_file}")

    # Collect tickers
    all_tickers = collect_all_tickers(params, hist_universe)
    print(f"Total tickers to process: {len(all_tickers)}")

    # Download if requested
    if args.download:
        print("\n--- Downloading price data ---")
        download_data(all_tickers, params)
        print("Done. Run again without --download to backtest.\n")
        return

    # Load data
    print("\n--- Loading price data ---")
    data = load_all_data(all_tickers, params)

    # Benchmarks
    bench_spy = calc_benchmark(data, 'SPY', params.start_date, params.end_date)
    bench_btc = calc_benchmark(data, 'BTC-USD', params.start_date, params.end_date)

    if args.compare:
        # A/B test mode
        diff_parts = []
        variant_params = replace(params)  # shallow copy
        for override in args.compare:
            key, val = parse_param_override(override)
            setattr(variant_params, key, val)
            diff_parts.append(f"{key}={val}")
        diff_desc = ', '.join(diff_parts)

        print(f"\n--- Running A/B test: {diff_desc} ---")

        print("  [A] Baseline...")
        res_a = run_backtest(data, params, hist_universe, label="Baseline")

        print("  [B] Variant...")
        res_b = run_backtest(data, variant_params, hist_universe, label=f"Variant ({diff_desc})")

        print_comparison(res_a, res_b, params, variant_params, diff_desc, bench_spy, bench_btc)
    else:
        # Single run
        print("\n--- Running backtest ---")
        results = run_backtest(data, params, hist_universe, label="Cap Defend V12")
        print_report(results, params, bench_spy, bench_btc)


if __name__ == '__main__':
    main()
