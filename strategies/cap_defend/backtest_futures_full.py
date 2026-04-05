#!/usr/bin/env python3
"""선물 백테스트 — 시간봉 완전 엔진.

시그널+체결 모두 시간봉 단위. 현물 V18의 모든 로직을 포팅:
- 카나리: BTC > SMA(50일) + 1.5% hyst (매 바마다)
- 헬스: Mom30>0 AND Mom90>0 AND Vol90≤5%
- 선정: 시총순 Top N → Greedy Absorption
- 비중: EW + Cap 33%
- 3-snapshot merge (앵커일: Day 1/10/19)
- DD Exit (60d peak -25%)
- Blacklist (-15% daily → 7일 제외)
- Crash Breaker (BTC -10% → 3일 현금)
- Drift 리밸런싱 (10% half-turnover)
- PFD (플립 후 5일 재평가)
- 격리마진 청산 (Low/High)
- 실제 바이낸스 펀딩레이트
- 시총 기반 슬리피지

지원: 1h / 4h / D (bars_per_day 자동 조정)

Usage:
  python3 backtest_futures_full.py
"""

import numpy as np, pandas as pd, os, sys, json, time

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'futures')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─── 심볼/슬리피지 ───
TICKER_MAP = {
    'BTC': 'BTCUSDT', 'ETH': 'ETHUSDT', 'SOL': 'SOLUSDT',
    'BNB': 'BNBUSDT', 'XRP': 'XRPUSDT', 'DOGE': 'DOGEUSDT',
    'ADA': 'ADAUSDT', 'AVAX': 'AVAXUSDT', 'TRX': 'TRXUSDT',
    'LINK': 'LINKUSDT', 'DOT': 'DOTUSDT', 'MATIC': 'MATICUSDT',
    'UNI': 'UNIUSDT', 'NEAR': 'NEARUSDT', 'LTC': 'LTCUSDT',
    'BCH': 'BCHUSDT', 'APT': 'APTUSDT', 'ICP': 'ICPUSDT',
    'FIL': 'FILUSDT', 'ATOM': 'ATOMUSDT', 'ARB': 'ARBUSDT',
    'OP': 'OPUSDT', 'SUI': 'SUIUSDT', 'SHIB': 'SHIBUSDT',
    'PEPE': 'PEPEUSDT', 'XLM': 'XLMUSDT', 'VET': 'VETUSDT',
    'ALGO': 'ALGOUSDT', 'FTM': 'FTMUSDT', 'GRT': 'GRTUSDT',
    'AAVE': 'AAVEUSDT', 'SAND': 'SANDUSDT', 'MANA': 'MANAUSDT',
    'AXS': 'AXSUSDT', 'THETA': 'THETAUSDT', 'EOS': 'EOSUSDT',
    'FLOW': 'FLOWUSDT', 'CHZ': 'CHZUSDT', 'APE': 'APEUSDT',
    'GALA': 'GALAUSDT',
}
# 슬리피지: 시총 기반 tier (자동 할당)
def _default_slippage(coin):
    tier1 = {'BTC', 'ETH'}  # 최소
    tier2 = {'BNB', 'SOL', 'XRP', 'DOGE'}
    tier3 = {'ADA', 'AVAX', 'TRX', 'LINK', 'DOT', 'LTC', 'BCH', 'SHIB'}
    if coin in tier1: return 0.0002
    if coin in tier2: return 0.0003
    if coin in tier3: return 0.0004
    return 0.0005  # 나머지
SLIPPAGE_MAP = {coin: _default_slippage(coin) for coin in TICKER_MAP}

# ─── 월별 시총 순위 ───
def _load_mcap():
    paths = [
        os.path.join(BASE_DIR, '..', '..', 'backup_20260125', 'historical_universe.json'),
        os.path.join(BASE_DIR, '..', '..', 'data', 'historical_universe.json'),
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                raw = json.load(f)
            result = {}
            for ds, tickers in raw.items():
                # 'BTC-USD' → 'BTC'
                result[ds] = [t.replace('-USD', '') for t in tickers if t.replace('-USD', '') in TICKER_MAP]
            return result
    return {}

_MCAP = None
def get_mcap(date):
    global _MCAP
    if _MCAP is None:
        _MCAP = _load_mcap()
    mk = date.strftime('%Y-%m') + '-01'
    if mk in _MCAP:
        return _MCAP[mk]
    keys = sorted(k for k in _MCAP if k <= mk)
    return _MCAP[keys[-1]] if keys else list(TICKER_MAP.keys())


# ─── 데이터 로드 ───
def _resample_to_daily(df):
    """1h/4h OHLCV → daily OHLCV 리샘플링."""
    return df.resample('D').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna(subset=['Close'])


def _resample_to_4h(df):
    """1h OHLCV → 4h OHLCV 리샘플링."""
    return df.resample('4h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna(subset=['Close'])


def _resample_to_2h(df):
    """1h OHLCV → 2h OHLCV 리샘플링."""
    return df.resample('2h').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna(subset=['Close'])


def _resample_to_30m(df):
    """15m OHLCV → 30m OHLCV 리샘플링."""
    return df.resample('30min').agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna(subset=['Close'])


def load_data(interval='1h'):
    """바이낸스 OHLCV + 펀딩 로드. 키는 심볼('BTC' 등).

    원본 데이터는 1h(또는 15m)를 기준으로 관리한다.
    - 1h: 네이티브 CSV 직접 로드
    - 4h: 1h에서 리샘플링
    - 2h: 1h에서 리샘플링
    - D: 1h에서 리샘플링
    - 15m: 네이티브 CSV 직접 로드
    - 30m: 15m에서 리샘플링
    """
    _RESAMPLE = {
        '4h': ('1h', _resample_to_4h),
        '2h': ('1h', _resample_to_2h),
        'D':  ('1h', _resample_to_daily),
        '30m': ('15m', _resample_to_30m),
    }

    bars = {}
    funding = {}
    for coin, sym in TICKER_MAP.items():
        fpath = os.path.join(DATA_DIR, f'{sym}_{interval}.csv')

        if os.path.exists(fpath) and os.path.getsize(fpath) > 1000:
            # CSV가 이미 있으면 직접 로드 (리샘플 불필요)
            df = pd.read_csv(fpath, parse_dates=['Date'], index_col='Date')
            bars[coin] = df
        elif interval in _RESAMPLE:
            src_iv, resample_fn = _RESAMPLE[interval]
            fpath_src = os.path.join(DATA_DIR, f'{sym}_{src_iv}.csv')
            if os.path.exists(fpath_src):
                df_src = pd.read_csv(fpath_src, parse_dates=['Date'], index_col='Date')
                bars[coin] = resample_fn(df_src)

        fpath_f = os.path.join(DATA_DIR, f'{sym}_funding.csv')
        if os.path.exists(fpath_f):
            funding[coin] = pd.read_csv(fpath_f, parse_dates=['Date'], index_col='Date')['fundingRate']
    return bars, funding


# ─── 헬퍼 ───
def get_close(bars, coin, idx):
    df = bars.get(coin)
    if df is None or idx < 0 or idx >= len(df):
        return 0
    return float(df['Close'].iloc[idx])

def get_low(bars, coin, idx):
    df = bars.get(coin)
    if df is None or idx < 0 or idx >= len(df):
        return 0
    return float(df['Low'].iloc[idx])

def get_high(bars, coin, idx):
    df = bars.get(coin)
    if df is None or idx < 0 or idx >= len(df):
        return 0
    return float(df['High'].iloc[idx])

def calc_sma(close_arr, period):
    if len(close_arr) < period:
        return 0
    return float(np.mean(close_arr[-period:]))

def calc_mom(close_arr, period):
    if len(close_arr) < period + 1:
        return -999
    return close_arr[-1] / close_arr[-period - 1] - 1

def calc_vol_daily(close_arr, bars_per_day, lookback_days=90, lookback_bars=0):
    """일봉 리샘플 기준 변동성 (calendar mode)."""
    n = lookback_bars if lookback_bars > 0 else lookback_days * bars_per_day
    if len(close_arr) < n + 1:
        return 999
    daily = close_arr[-n::bars_per_day]
    if len(daily) < 10:
        return 999
    rets = np.diff(np.log(daily))
    return float(np.std(rets))

def calc_vol_bars(close_arr, lookback_bars, bars_per_year=8760):
    """순수 봉 기반 변동성 (bar mode). 연환산."""
    if len(close_arr) < lookback_bars + 1:
        return 999
    rets = np.diff(np.log(close_arr[-lookback_bars - 1:]))
    return float(np.std(rets) * np.sqrt(bars_per_year))


# ─── 메인 엔진 ───
def run(bars, funding, interval='1h', leverage=1.0,
        universe_size=5, selection='greedy', cap=1/3,
        tx_cost=0.0004, maint_rate=0.004,
        # 윈도우: _days 또는 _bars 중 하나만 지정. _bars 우선.
        sma_days=50, mom_short_days=30, mom_long_days=90, vol_days=90,
        sma_bars=0, mom_short_bars=0, mom_long_bars=0, vol_bars=0,
        canary_hyst=0.015,
        dd_lookback=60, dd_threshold=-0.25,
        dd_bars_override=0,  # 0이면 dd_lookback*bpd 사용
        bl_drop=-0.15, bl_days=7,
        bl_bars_override=0,
        drift_threshold=0.10, post_flip_delay=5,
        daily_gate=False,
        health_mode='mom2vol',  # 'mom2vol'(기본), 'mom1vol', 'mom1', 'mom2', 'vol', 'none'
        vol_mode='daily',  # 'daily'(일봉 리샘플) or 'bar'(순수 봉 기반 연환산)
        vol_threshold=0.05,  # vol_mode='daily' 기본. bar mode면 연환산 기준 (예: 0.80)
        n_snapshots=3,  # 스냅샷 수 (3=월3회, 6=월6회, 12=거의매일)
        snap_interval_bars=0,  # 0=달력 기반, >0=봉 기반 앵커 간격
        crash_threshold=-0.10,
        crash_lookback_bars=0,  # 0=bpd(일간), >0=N봉 누적 수익률
        crash_cool_override=0,  # 0=3*bpd, >0=직접 봉 수
        bl_lookback_bars=0,  # 0=bpd(일간), >0=N봉 누적 수익률
        pfd_bars_override=0,  # 0=post_flip_delay*bpd, >0=직접 봉 수
        stop_kind='none',  # none, prev_close_pct, highest_close_since_entry_pct, highest_high_since_entry_pct, rolling_high_close_pct, rolling_high_high_pct
        stop_pct=0.0,
        stop_lookback_bars=0,
        initial_capital=10000.0,
        start_date='2020-10-01', end_date='2026-03-28',
        _trace=None):  # list를 넘기면 매 봉 {'date':..., 'target':..., 'rebal':bool} 기록
    """시간봉 완전 선물 백테스트. 봉 단위 파라미터 지원."""

    bpd = {'D': 1, '4h': 6, '2h': 12, '1h': 24, '30m': 48, '15m': 96}[interval]
    bars_per_year = bpd * 365

    btc_df = bars.get('BTC')
    if btc_df is None:
        return {}

    all_dates = btc_df.index[(btc_df.index >= start_date) & (btc_df.index <= end_date)]
    if len(all_dates) == 0:
        return {}

    # 봉 단위 우선, 없으면 일 단위 × bpd
    sma_period = sma_bars if sma_bars > 0 else sma_days * bpd
    mom30 = mom_short_bars if mom_short_bars > 0 else mom_short_days * bpd
    mom90 = mom_long_bars if mom_long_bars > 0 else mom_long_days * bpd
    _vol_bars = vol_bars if vol_bars > 0 else vol_days * bpd
    dd_bars = dd_bars_override if dd_bars_override > 0 else dd_lookback * bpd
    bl_cooldown = bl_bars_override if bl_bars_override > 0 else bl_days * bpd
    pfd_bars = pfd_bars_override if pfd_bars_override > 0 else post_flip_delay * bpd
    crash_cool_bars = crash_cool_override if crash_cool_override > 0 else 3 * bpd
    _crash_lookback = crash_lookback_bars if crash_lookback_bars > 0 else bpd
    _bl_lookback = bl_lookback_bars if bl_lookback_bars > 0 else bpd

    # State
    capital = initial_capital
    holdings = {}  # {coin: qty}  — spot-equivalent units (leverage는 PnL에만 적용)
    entry_prices = {}  # {coin: entry_price}
    entry_bar_index = {}  # {coin: first entry bar index}
    margins = {}  # {coin: margin}

    snapshots = [{'CASH': 1.0} for _ in range(n_snapshots)]
    # 앵커일: n_snapshots에 따라 균등 배분
    if n_snapshots <= 3:
        snap_days = [1, 10, 19][:n_snapshots]
    else:
        snap_days = [1 + int(i * 28 / n_snapshots) for i in range(n_snapshots)]
    snap_done = {}
    blacklist = {}  # {coin: remaining_bars}

    prev_canary = False
    canary_on_bar = None
    pfd_done = True
    crash_cooldown = 0
    rebal_count = 0
    liq_count = 0
    trade_count = 0
    stop_count = 0
    pv_list = []

    btc_close = btc_df['Close'].values
    btc_idx_map = {d: i for i, d in enumerate(btc_df.index)}

    def _port_val(date):
        """포트폴리오 가치."""
        pv = capital
        for coin in holdings:
            df = bars.get(coin)
            if df is None: continue
            ci = df.index.get_indexer([date], method='ffill')[0]
            cur = float(df['Close'].iloc[ci]) if ci >= 0 else 0
            if cur > 0:
                pnl = holdings[coin] * (cur - entry_prices[coin])
                pv += margins[coin] + pnl
        return pv

    def _get_price(coin, date):
        """date 기준 종가 (ffill)."""
        df = bars.get(coin)
        if df is None:
            return 0
        ci = df.index.get_indexer([date], method='ffill')[0]
        if ci < 0:
            return 0
        return float(df['Close'].iloc[ci])

    def _get_bar_index(coin, date):
        df = bars.get(coin)
        if df is None:
            return -1
        return df.index.get_indexer([date], method='ffill')[0]

    def _get_stop_price(coin, date):
        if stop_kind == 'none' or stop_pct <= 0:
            return None
        df = bars.get(coin)
        if df is None:
            return None
        ci = _get_bar_index(coin, date)
        if ci <= 0:
            return None

        if stop_kind == 'prev_close_pct':
            ref = float(df['Close'].iloc[ci - 1])
        elif stop_kind == 'highest_close_since_entry_pct':
            start_ci = entry_bar_index.get(coin, -1)
            if start_ci < 0:
                return None
            ref = float(np.max(df['Close'].iloc[start_ci:ci]))
        elif stop_kind == 'highest_high_since_entry_pct':
            start_ci = entry_bar_index.get(coin, -1)
            if start_ci < 0:
                return None
            ref = float(np.max(df['High'].iloc[start_ci:ci]))
        elif stop_kind == 'rolling_high_close_pct':
            if stop_lookback_bars <= 0 or ci < stop_lookback_bars:
                return None
            ref = float(np.max(df['Close'].iloc[ci - stop_lookback_bars:ci]))
        elif stop_kind == 'rolling_high_high_pct':
            if stop_lookback_bars <= 0 or ci < stop_lookback_bars:
                return None
            ref = float(np.max(df['High'].iloc[ci - stop_lookback_bars:ci]))
        else:
            return None

        if ref <= 0:
            return None
        return ref * (1.0 - stop_pct)

    def _execute_stop_exit(coin, date, stop_price):
        nonlocal capital, trade_count, stop_count
        ci = _get_bar_index(coin, date)
        if ci < 0:
            return False
        low = get_low(bars, coin, ci)
        if low <= 0 or low > stop_price:
            return False
        slip = SLIPPAGE_MAP.get(coin, 0.0005)
        cur_open = float(bars[coin]['Open'].iloc[ci])
        exit_p = min(cur_open, stop_price) * (1 - slip)
        pnl = holdings[coin] * (exit_p - entry_prices[coin])
        tx = holdings[coin] * exit_p * tx_cost
        capital += margins[coin] + pnl - tx
        del holdings[coin]; del entry_prices[coin]; del margins[coin]
        entry_bar_index.pop(coin, None)
        trade_count += 1
        stop_count += 1
        return True

    def _get_liq_price(coin):
        qty = holdings.get(coin, 0)
        entry = entry_prices.get(coin, 0)
        margin = margins.get(coin, 0)
        if qty <= 0 or entry <= 0:
            return None
        denom = qty * (1.0 - maint_rate)
        if denom <= 0:
            return None
        liq_price = (qty * entry - margin) / denom
        if liq_price <= 0:
            return None
        return liq_price

    def _execute_rebalance(target_weights, date):
        """Delta 리밸런싱 with 선물 비용."""
        nonlocal capital, trade_count
        pv = _port_val(date)
        if pv <= 0:
            return

        # 목표 포지션
        target_qty = {}
        target_margin = {}
        for coin, w in target_weights.items():
            if coin == 'CASH' or w <= 0:
                continue
            cur = _get_price(coin, date)
            if cur <= 0:
                continue
            tmgn = pv * w * 0.95
            tnotional = tmgn * leverage
            tqty = tnotional / cur
            target_qty[coin] = tqty
            target_margin[coin] = tmgn

        # 매도 (보유 중이지만 target에 없거나 줄어야)
        for coin in list(holdings.keys()):
            cur = _get_price(coin, date)
            if cur <= 0:
                continue
            slip = SLIPPAGE_MAP.get(coin, 0.0005)
            if coin not in target_qty:
                # 전량 매도
                exit_p = cur * (1 - slip)
                pnl = holdings[coin] * (exit_p - entry_prices[coin])
                tx = holdings[coin] * cur * tx_cost
                capital += margins[coin] + pnl - tx
                del holdings[coin]; del entry_prices[coin]; del margins[coin]
                entry_bar_index.pop(coin, None)
                trade_count += 1
            else:
                delta = target_qty[coin] - holdings[coin]
                if delta < -holdings[coin] * 0.05:
                    sell_qty = -delta
                    sell_frac = sell_qty / holdings[coin]
                    sell_margin = margins[coin] * sell_frac
                    exit_p = cur * (1 - slip)
                    pnl = sell_qty * (exit_p - entry_prices[coin])
                    tx = sell_qty * cur * tx_cost
                    capital += sell_margin + pnl - tx
                    holdings[coin] -= sell_qty
                    margins[coin] -= sell_margin
                    trade_count += 1

        # 매수 (target에 있지만 미보유거나 늘어야)
        for coin, tqty in target_qty.items():
            cur = _get_price(coin, date)
            if cur <= 0:
                continue
            slip = SLIPPAGE_MAP.get(coin, 0.0005)
            if coin not in holdings:
                entry_p = cur * (1 + slip)
                margin = target_margin[coin]
                notional = margin * leverage
                qty = notional / entry_p
                tx = notional * tx_cost
                if capital >= margin + tx:
                    capital -= margin + tx
                    holdings[coin] = qty
                    entry_prices[coin] = entry_p
                    entry_bar_index[coin] = _get_bar_index(coin, date)
                    margins[coin] = margin
                    trade_count += 1
            else:
                delta = tqty - holdings[coin]
                if delta > holdings[coin] * 0.05:
                    entry_p = cur * (1 + slip)
                    add_notional = delta * entry_p
                    add_margin = add_notional / leverage
                    tx = add_notional * tx_cost
                    if capital >= add_margin + tx:
                        capital -= add_margin + tx
                        total = holdings[coin] + delta
                        entry_prices[coin] = (entry_prices[coin] * holdings[coin] + entry_p * delta) / total
                        holdings[coin] = total
                        margins[coin] += add_margin
                        trade_count += 1

    def _compute_weights(sig_date):
        """V18 선정 + 비중 계산. sig_date=시그널 기준(t-1)."""
        mcap_order = get_mcap(sig_date)  # t-1 기준 시총 (look-ahead 방지)

        healthy = []
        min_bars = max(mom30, mom90, _vol_bars, sma_period)
        for coin in mcap_order:
            if coin in blacklist:
                continue
            df = bars.get(coin)
            if df is None:
                continue
            ci = df.index.get_indexer([sig_date], method='ffill')[0]
            if ci < 0 or ci < min_bars:
                continue
            c = df['Close'].values[:ci + 1]

            # health_mode에 따라 헬스 체크
            if health_mode == 'none':
                healthy.append(coin)
                continue
            m_short = calc_mom(c, mom30) if 'mom' in health_mode else 999
            m_long = calc_mom(c, mom90) if 'mom2' in health_mode else 999
            if 'vol' in health_mode:
                if vol_mode == 'bar':
                    vol = calc_vol_bars(c, _vol_bars, bars_per_year)
                else:
                    vol = calc_vol_daily(c, bpd, lookback_bars=_vol_bars)
            else:
                vol = 0

            if health_mode == 'mom2vol':
                ok = m_short > 0 and m_long > 0 and vol <= vol_threshold
            elif health_mode == 'mom1vol':
                ok = m_short > 0 and vol <= vol_threshold
            elif health_mode == 'mom1':
                ok = m_short > 0
            elif health_mode == 'mom2':
                ok = m_short > 0 and m_long > 0
            elif health_mode == 'vol':
                ok = vol <= vol_threshold
            else:
                ok = True
            if ok:
                healthy.append(coin)

        picks = healthy[:universe_size]

        # Greedy absorption
        if selection == 'greedy' and len(picks) > 1:
            for i in range(len(picks) - 1, 0, -1):
                df_a = bars.get(picks[i - 1])
                df_b = bars.get(picks[i])
                if df_a is None or df_b is None:
                    continue
                ci_a = df_a.index.get_indexer([sig_date], method='ffill')[0]
                ci_b = df_b.index.get_indexer([sig_date], method='ffill')[0]
                ca = df_a['Close'].values[:ci_a + 1]
                cb = df_b['Close'].values[:ci_b + 1]
                ma = calc_mom(ca, mom30)
                mb = calc_mom(cb, mom30)
                if ma >= mb:
                    picks.pop(i)

        if not picks:
            return {'CASH': 1.0}

        w = min(1.0 / len(picks), cap)
        weights = {coin: w for coin in picks}
        total = sum(weights.values())
        if total < 0.999:
            weights['CASH'] = 1.0 - total
        return weights

    def _merge_snapshots():
        combined = {}
        n = len(snapshots)
        for snap in snapshots:
            for t, w in snap.items():
                combined[t] = combined.get(t, 0) + w / n
        total = sum(combined.values())
        if total > 0:
            return {t: w / total for t, w in combined.items()}
        return {'CASH': 1.0}

    def _half_turnover(cur_w, tgt_w):
        all_k = set(cur_w.keys()) | set(tgt_w.keys())
        return sum(abs(tgt_w.get(k, 0) - cur_w.get(k, 0)) for k in all_k) / 2

    def _current_weights(date):
        pv = _port_val(date)
        if pv <= 0:
            return {'CASH': 1.0}
        w = {}
        for coin in holdings:
            cur = _get_price(coin, date)
            val = margins[coin] + holdings[coin] * (cur - entry_prices[coin])
            if val > 0:
                w[coin] = val / pv
        cash_w = capital / pv
        if cash_w > 0.001:
            w['CASH'] = cash_w
        return w

    # ═══ 메인 루프 ═══
    for bar_i in range(sma_period + 1, len(all_dates)):
        date = all_dates[bar_i]
        prev_date = all_dates[bar_i - 1]
        btc_i = btc_idx_map.get(date, -1)
        btc_i_prev = btc_idx_map.get(prev_date, -1)
        if btc_i < sma_period or btc_i_prev < sma_period:
            continue

        # daily_gate: 일간 개념 체크(DD/BL/drift)를 UTC 00시 바에서만 실행
        is_daily_bar = (not daily_gate) or (bpd == 1) or (hasattr(date, 'hour') and date.hour == 0)

        cur_month = date.strftime('%Y-%m')

        # ── 청산/스탑 체크 (롱 전용) ──
        for coin in list(holdings.keys()):
            ci = btc_idx_map.get(date, -1) if coin == 'BTC' else bars[coin].index.get_indexer([date], method='ffill')[0] if coin in bars else -1
            low = get_low(bars, coin, ci)
            if low <= 0:
                continue

            liq_price = _get_liq_price(coin) if leverage > 1 else None
            stop_price = _get_stop_price(coin, date) if stop_kind != 'none' and stop_pct > 0 else None
            hit_liq = liq_price is not None and low <= liq_price
            hit_stop = stop_price is not None and low <= stop_price

            if hit_stop and (not hit_liq or stop_price > liq_price):
                _execute_stop_exit(coin, date, stop_price)
                continue

            if hit_liq:
                pnl_at_low = holdings[coin] * (low - entry_prices[coin])
                eq = margins[coin] + pnl_at_low
                liq_fee = max(eq, 0) * 0.015
                returned = max(eq - liq_fee, 0)
                capital += returned
                del holdings[coin]; del entry_prices[coin]; del margins[coin]
                entry_bar_index.pop(coin, None)
                liq_count += 1

        # ── 펀딩비 ──
        for coin in list(holdings.keys()):
            fr_series = funding.get(coin)
            if fr_series is None:
                continue
            if date in fr_series.index:
                fr = float(fr_series.loc[date])
                if fr != 0 and not np.isnan(fr):
                    cur = get_close(bars, coin, btc_idx_map.get(date, -1) if coin == 'BTC' else
                                    bars[coin].index.get_indexer([date], method='ffill')[0] if coin in bars else -1)
                    if cur > 0:
                        # 롱: notional × rate
                        capital -= holdings[coin] * cur * fr
        capital = max(capital, 0)

        # ── 카나리 (t-1 기준) ──
        btc_c_prev = btc_close[:btc_i_prev + 1]
        sma_val = calc_sma(btc_c_prev, sma_period)
        cur_btc_prev = btc_c_prev[-1] if len(btc_c_prev) > 0 else 0

        if prev_canary:
            canary_on = not (cur_btc_prev < sma_val * (1 - canary_hyst))
        else:
            canary_on = cur_btc_prev > sma_val * (1 + canary_hyst)

        canary_flipped = canary_on != prev_canary
        if canary_on and canary_flipped:
            canary_on_bar = bar_i
            pfd_done = False
        elif not canary_on and canary_flipped:
            canary_on_bar = None

        # ── Blacklist 감소 ──
        for coin in list(blacklist.keys()):
            blacklist[coin] -= 1
            if blacklist[coin] <= 0:
                del blacklist[coin]

        # ── Crash Breaker (봉 기반: _crash_lookback봉 전 대비) ──
        if crash_cooldown > 0:
            crash_cooldown -= 1
        elif btc_i_prev >= _crash_lookback:
            btc_ret = btc_close[btc_i_prev] / btc_close[btc_i_prev - _crash_lookback] - 1
            if btc_ret <= crash_threshold:
                # 전량 청산
                for coin in list(holdings.keys()):
                    ci = bars[coin].index.get_indexer([date], method='ffill')[0] if coin in bars else -1
                    cur = get_close(bars, coin, ci)
                    if cur > 0:
                        pnl = holdings[coin] * (cur - entry_prices[coin])
                        capital += margins[coin] + pnl - holdings[coin] * cur * tx_cost
                holdings.clear(); entry_prices.clear(); margins.clear(); entry_bar_index.clear()
                for si in range(n_snapshots):
                    snapshots[si] = {'CASH': 1.0}
                crash_cooldown = crash_cool_bars
                rebal_count += 1

        # ── DD Exit (daily_gate 시 일 1회만) ──
        if dd_lookback > 0 and canary_on and holdings and not canary_flipped and is_daily_bar:
            for coin in list(holdings.keys()):
                df = bars.get(coin)
                if df is None:
                    continue
                ci = df.index.get_indexer([date], method='ffill')[0]
                if ci < dd_bars:
                    continue
                c = df['Close'].values
                peak = np.max(c[ci - dd_bars:ci])
                if peak > 0 and (c[ci] / peak - 1) <= dd_threshold:
                    cur = c[ci]
                    slip = SLIPPAGE_MAP.get(coin, 0.0005)
                    pnl = holdings[coin] * (cur * (1 - slip) - entry_prices[coin])
                    capital += margins[coin] + pnl - holdings[coin] * cur * tx_cost
                    del holdings[coin]; del entry_prices[coin]; del margins[coin]
                    entry_bar_index.pop(coin, None)
                    trade_count += 1

        # ── Blacklist (봉 기반: _bl_lookback봉 전 대비) ──
        if bl_drop < 0 and canary_on and holdings and is_daily_bar:
            for coin in list(holdings.keys()):
                df = bars.get(coin)
                if df is None:
                    continue
                ci = df.index.get_indexer([date], method='ffill')[0]
                if ci < _bl_lookback:
                    continue
                c = df['Close'].values
                bar_ret = c[ci] / c[ci - _bl_lookback] - 1
                if bar_ret <= bl_drop:
                    cur = c[ci]
                    slip = SLIPPAGE_MAP.get(coin, 0.0005)
                    pnl = holdings[coin] * (cur * (1 - slip) - entry_prices[coin])
                    capital += margins[coin] + pnl - holdings[coin] * cur * tx_cost
                    del holdings[coin]; del entry_prices[coin]; del margins[coin]
                    entry_bar_index.pop(coin, None)
                    blacklist[coin] = bl_cooldown
                    trade_count += 1

        # ── 시그널 → 스냅샷 갱신 ──
        need_rebal = False

        if bar_i <= sma_period + 1:
            # 첫 바
            for si in range(n_snapshots):
                if canary_on:
                    snapshots[si] = _compute_weights(prev_date)
                else:
                    snapshots[si] = {'CASH': 1.0}
            need_rebal = True
        elif canary_flipped:
            for si in range(n_snapshots):
                if canary_on:
                    snapshots[si] = _compute_weights(prev_date)
                else:
                    snapshots[si] = {'CASH': 1.0}
            need_rebal = True
        elif pfd_bars > 0 and canary_on:
            if canary_on_bar and not pfd_done and (bar_i - canary_on_bar) >= pfd_bars:
                pfd_done = True
                for si in range(n_snapshots):
                    snapshots[si] = _compute_weights(prev_date)
                need_rebal = True

        # ── 앵커 리밸런싱 (Risk-On + 미플립) ──
        if canary_on and not canary_flipped:
            if snap_interval_bars > 0:
                # 봉 기반 앵커: bar_i % interval로 트리거
                for si in range(n_snapshots):
                    offset = int(si * snap_interval_bars / n_snapshots)
                    if bar_i % snap_interval_bars == offset:
                        new_w = _compute_weights(prev_date)
                        if new_w != snapshots[si]:
                            snapshots[si] = new_w
                            need_rebal = True
            else:
                # 달력 기반 앵커 (기존)
                day_of_month = date.day
                for si, anchor in enumerate(snap_days):
                    key = f"{cur_month}_snap{si}"
                    if day_of_month >= anchor and key not in snap_done:
                        snap_done[key] = True
                        new_w = _compute_weights(prev_date)
                        if new_w != snapshots[si]:
                            snapshots[si] = new_w
                            need_rebal = True

        # ── Drift (daily_gate 시 일 1회만) ──
        combined = _merge_snapshots()
        if not need_rebal and canary_on and drift_threshold > 0 and holdings and is_daily_bar:
            if _half_turnover(_current_weights(date), combined) >= drift_threshold:
                need_rebal = True

        if crash_cooldown > 0:
            need_rebal = False

        # ── trace 기록 ──
        if _trace is not None:
            _trace.append({
                'date': date,
                'target': dict(combined),
                'rebal': need_rebal,
                'stop_kind': stop_kind,
                'stop_pct': stop_pct,
            })

        # ── 리밸런싱 실행 ──
        if need_rebal:
            _execute_rebalance(combined, date)
            rebal_count += 1

        pv_list.append({'Date': date, 'Value': _port_val(date)})
        prev_canary = canary_on

    # ── 결과 ──
    if not pv_list:
        return {}
    pvdf = pd.DataFrame(pv_list).set_index('Date')
    eq = pvdf['Value']
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if eq.iloc[-1] <= 0 or yrs <= 0:
        return {
            'Sharpe': 0, 'CAGR': -1, 'MDD': -1, 'Cal': 0,
            'Liq': liq_count, 'Trades': trade_count, 'Rebal': rebal_count,
            'Stops': stop_count,
        }
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(bars_per_year) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    result = {
        'Sharpe': sh, 'CAGR': cagr, 'MDD': mdd, 'Cal': cal,
        'Liq': liq_count, 'Trades': trade_count, 'Rebal': rebal_count,
        'Stops': stop_count,
    }
    result['_equity'] = eq  # equity curve (pd.Series)
    return result


# ═══ 메인 ═══
if __name__ == '__main__':
    t0 = time.time()

    print("전략: V18 Cap Defend 코인 (바이낸스 선물 포팅)")
    print("  카나리: BTC > SMA(50일) + 1.5% hyst")
    print("  헬스: Mom30>0 AND Mom90>0 AND Vol90≤5%")
    print("  선정: 시총순 Top5 → Greedy Absorption")
    print("  비중: EW + Cap 33%")
    print("  리스크: DD -25%(60d), BL -15%(7d), Crash BTC-10%(3d)")
    print("  3-snapshot Day 1/10/19, Drift 10%, PFD 5d")
    print("  윈도우: 동일 기간 유지 (SMA 50일, Mom 30/90일 → 바 단위 자동 변환)")
    print("  비용: tx 0.04%, 시총별 슬리피지, 실제 펀딩레이트")
    print(f"  기간: 2020-10-01 ~ 2026-03-28")

    for interval in ['D', '4h', '1h']:
        bars, funding = load_data(interval)
        if 'BTC' not in bars:
            print(f"\n  {interval}: BTC 데이터 없음, skip")
            continue

        bpd_label = {'D': '1', '4h': '6', '1h': '24'}[interval]
        print(f"\n{'='*70}")
        print(f"  {interval} (bpd={bpd_label}) | V18 완전 엔진")
        print(f"{'='*70}")
        print(f"  {'Leverage':<10s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s} {'Liq':>4s} {'Rebal':>6s}")
        print(f"  {'-'*55}")

        for lev in [1.0, 1.5, 2.0, 3.0]:
            m = run(bars, funding, interval=interval, leverage=lev,
                    start_date='2020-10-01')
            if not m:
                print(f"  {lev}x: NO DATA")
                continue
            liq = f"💀{m['Liq']}" if m['Liq'] > 0 else ""
            print(f"  {lev:<10.1f} {m['Sharpe']:>7.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>7.2f} {liq:>4s} {m['Rebal']:>6d}")
        sys.stdout.flush()

    print(f"\n소요: {time.time() - t0:.0f}s")
