#!/usr/bin/env python3
"""V25 선물 백테스트 — 동적 per-coin L + CROSS 마진 + K2 시그널 지원.

V25 추가 기능 (2026-05-28):
- leverage 파라미터: scalar / pd.Series / dict[coin -> pd.Series]
- CROSS 마진모드: wallet 전체 cushion (worst-case all-coins low 가정)
- entry_levs[coin] 저장 — 진입 시점 L 보존
- mom_stop_threshold: per-bar mom 기반 stop (optional, default disabled)
- catastrophic_stop_pct + cooldown: entry -X% 시 강제 청산
- gross_exposure_cap: sum L_i × w_i 상한
- cushion_buffer_mult: wallet < mult × maint 시 강제 축소

build_K2_signal(bars, ...) helper 제공 — V25 라이브 spec 동일 시그널.

기존 backtest_futures_full.py 는 V24 reference 로 유지.

원본 docstring:
선물 백테스트 — 시간봉 완전 엔진.

시그널+체결 모두 시간봉 단위. 현물 V18의 모든 로직을 포팅:
- 카나리: BTC > SMA(50일) + 1.5% hyst (매 바마다)
- 헬스: Mom30>0 AND Mom90>0 AND Vol90≤5%
- 선정: 시총순 Top N → Greedy Absorption
- 비중: EW + Cap 33%
- 3-snapshot merge (앵커일: Day 1/10/19)
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

DATA_DIR = '/home/gmoh/mon/251229/data/futures'
BASE_DIR = '/home/gmoh/mon/251229/strategies/cap_defend'

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
# ⚠ V21 (ENS_fut_L3_k3_12652d57) 재현 시 반드시 universe_size=3 명시할 것 (실전 live: UNIVERSE_SIZE=3).
# 기본값 5는 V18/V19 시절 호환. V21 관련 스크립트에서 3으로 override 필수.
def run(bars, funding, interval='D', leverage=3.0,
        universe_size=3, selection='greedy', cap=1/3,
        tx_cost=0.0006, maint_rate=0.004,
        # V24 라이브 spec default (2026-05-27 정정)
        sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
        sma_bars=0, mom_short_bars=0, mom_long_bars=0, vol_bars=0,
        canary_hyst=0.015,
        drift_threshold=0.03, post_flip_delay=5,
        daily_gate=False,
        health_mode='mom2vol',
        vol_mode='daily',
        vol_threshold=0.05,
        n_snapshots=5,  # V24 spec
        snap_interval_bars=95,  # V24 spec
        phase_offset_bars=0,  # bar_i 에 더할 위상 오프셋 (멤버별 비동기화 테스트용, 기본 0 = 기존 동작)
        pfd_bars_override=0,  # 0=post_flip_delay*bpd, >0=직접 봉 수
        stop_kind='none',  # none, prev_close_pct, highest_close_since_entry_pct, highest_high_since_entry_pct, rolling_high_close_pct, rolling_high_high_pct
        stop_pct=0.0,
        stop_lookback_bars=0,
        mom_stop_threshold=-999.0,  # >-999 enable. min(ms,ml) < -mom_stop_threshold => 매도. -1 sentinel: 헬스체크와 동일 sign-only (ms<0 AND ml<0)
        catastrophic_stop_pct=0.0,  # A: entry 가 대비 -X% 도달 시 강제 매도 (low 기준). 0=disabled. cooldown_bars 동안 그 코인 재진입 차단.
        catastrophic_cooldown_bars=0,  # A 발동 후 그 코인 재진입 차단 봉 수
        gross_exposure_cap=0.0,  # C: 총 gross exposure (sum L_i × w_i) 상한. 0=disabled. 초과 시 비율 축소.
        cushion_buffer_mult=0.0,  # G: cushion < mult × total_maint 이면 강제 축소. 0=disabled. 1.5 권장.
        initial_capital=10000.0,
        start_date='2020-10-01', end_date='2026-05-13',
        fill_mode='open',  # 'open'(unified_backtest 기준) | 'close'(legacy)
        external_target_schedule=None,  # dict[date -> {'target':dict, 'rebal':bool}] (앙상블 BT 용)
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
    pfd_bars = pfd_bars_override if pfd_bars_override > 0 else post_flip_delay * bpd

    # State
    capital = initial_capital
    holdings = {}  # {coin: qty}  — spot-equivalent units (leverage는 PnL에만 적용)
    entry_prices = {}  # {coin: entry_price}
    entry_bar_index = {}  # {coin: first entry bar index}
    margins = {}  # {coin: margin}
    entry_levs = {}  # {coin: leverage at entry}  — dynamic L 시 entry 시점 L 보존
    no_reentry = {}  # {coin: bar_i until} — catastrophic stop cooldown (A 안)
    cat_stop_count = 0
    cushion_force_count = 0
    gross_cap_count = 0

    # leverage 처리: scalar / pd.Series (date->L, 공통) / dict {coin: pd.Series} (per-coin)
    if isinstance(leverage, dict) and leverage and hasattr(next(iter(leverage.values())), 'asof'):
        # per-coin Series dict
        _lev_per_coin = leverage
        _default_series = next(iter(_lev_per_coin.values()))
        _lev_default = float(_default_series.iloc[0])
        def _cur_lev(d, coin=None):
            if coin is None:
                return _lev_default
            s = _lev_per_coin.get(coin)
            if s is None:
                return _lev_default
            try:
                return float(s.asof(d))
            except Exception:
                return _lev_default
    elif hasattr(leverage, 'loc'):
        _lev_series = leverage
        _lev_default = float(_lev_series.iloc[0])
        def _cur_lev(d, coin=None):
            try:
                return float(_lev_series.asof(d))
            except Exception:
                return _lev_default
    elif isinstance(leverage, dict):
        _lev_dict = leverage
        _lev_default = next(iter(_lev_dict.values())) if _lev_dict else 3.0
        def _cur_lev(d, coin=None):
            return float(_lev_dict.get(d, _lev_default))
    else:
        _lev_scalar = float(leverage)
        def _cur_lev(d, coin=None):
            return _lev_scalar

    snapshots = [{'CASH': 1.0} for _ in range(n_snapshots)]
    # 앵커일: n_snapshots에 따라 균등 배분
    if n_snapshots <= 3:
        snap_days = [1, 10, 19][:n_snapshots]
    else:
        snap_days = [1 + int(i * 28 / n_snapshots) for i in range(n_snapshots)]
    snap_done = {}

    prev_canary = False
    canary_on_bar = None
    pfd_done = True
    rebal_count = 0
    liq_count = 0
    liq_log = []
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
        """date 기준 종가 (ffill). pv 평가 전용."""
        df = bars.get(coin)
        if df is None:
            return 0
        ci = df.index.get_indexer([date], method='ffill')[0]
        if ci < 0:
            return 0
        return float(df['Close'].iloc[ci])

    def _get_fill_price(coin, date):
        """체결가: fill_mode='open'이면 해당 봉 Open (exact match),
        'close'면 Close(legacy)."""
        df = bars.get(coin)
        if df is None:
            return 0
        if fill_mode == 'close':
            return _get_price(coin, date)
        try:
            ci = df.index.get_loc(date)
        except KeyError:
            return 0
        return float(df['Open'].iloc[ci])

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
        del holdings[coin]; del entry_prices[coin]; del margins[coin]; entry_levs.pop(coin, None)
        entry_bar_index.pop(coin, None)
        trade_count += 1
        stop_count += 1
        # FIX: snapshots 와 combined 에서 그 코인 제거 → 다음 봉 refill v2 자동 작동
        for sn in snapshots:
            if coin in sn:
                w = sn.pop(coin)
                sn['CASH'] = sn.get('CASH', 0) + w
        return True

    def _get_liq_price(coin, _date=None):
        """CROSS 모드 worst-case: 모든 코인 low 동시 도달 가정 (보수적).

        cross_equity_worst = wallet + sum(qty_c × (low_c - entry_c))
        그 코인 X 의 liq 시점 = cross_equity_worst < total_maint (low 기준)
        즉 wallet + sum_other(qty × (low - entry)) + qty_X × (p - entry_X) < total_maint_low
        liq_price = entry_X - (wallet + sum_other_low_pnl - total_maint_low) / qty_X
        """
        qty = holdings.get(coin, 0)
        entry = entry_prices.get(coin, 0)
        if qty <= 0 or entry <= 0:
            return None
        if _date is None:
            return None
        cur_X = _get_price(coin, _date)
        if cur_X <= 0:
            return None
        wallet = capital + sum(margins.values())
        sum_other_low_pnl = 0.0
        total_maint_low = 0.0
        for c in holdings:
            if c == coin:
                continue
            ci = btc_idx_map.get(_date, -1) if c == 'BTC' else (
                bars[c].index.get_indexer([_date], method='ffill')[0] if c in bars else -1)
            low_c = get_low(bars, c, ci)
            if low_c <= 0:
                continue
            sum_other_low_pnl += holdings[c] * (low_c - entry_prices[c])
            total_maint_low += holdings[c] * low_c * maint_rate
        ci_X = btc_idx_map.get(_date, -1) if coin == 'BTC' else (
            bars[coin].index.get_indexer([_date], method='ffill')[0] if coin in bars else -1)
        low_X = get_low(bars, coin, ci_X)
        if low_X > 0:
            total_maint_low += qty * low_X * maint_rate
        cushion = wallet + sum_other_low_pnl - total_maint_low
        if cushion <= 0:
            return cur_X * 1.0
        liq_price = entry - cushion / qty
        # 단 entry 위로 가는 비현실적 liq_price 차단
        if liq_price <= 0 or liq_price > cur_X:
            return None
        return liq_price

    def _execute_rebalance(target_weights, date, pv_date=None):
        """Delta 리밸런싱 with 선물 비용. pv_date=None 이면 date 사용 (legacy)."""
        nonlocal capital, trade_count
        pv = _port_val(pv_date if pv_date is not None else date)
        if pv <= 0:
            return

        # 목표 포지션 (체결가는 t 봉 Open)
        target_qty = {}
        target_margin = {}
        for coin, w in target_weights.items():
            if coin == 'CASH' or w <= 0:
                continue
            cur = _get_fill_price(coin, date)
            if cur <= 0:
                continue
            tmgn = pv * w * 0.95
            cur_L = _cur_lev(date, coin)
            tnotional = tmgn * cur_L
            tqty = tnotional / cur
            target_qty[coin] = tqty
            target_margin[coin] = tmgn

        # 매도 (보유 중이지만 target에 없거나 줄어야)
        for coin in list(holdings.keys()):
            cur = _get_fill_price(coin, date)
            if cur <= 0:
                continue
            slip = SLIPPAGE_MAP.get(coin, 0.0005)
            if coin not in target_qty:
                # 전량 매도
                exit_p = cur * (1 - slip)
                pnl = holdings[coin] * (exit_p - entry_prices[coin])
                tx = holdings[coin] * cur * tx_cost
                capital += margins[coin] + pnl - tx
                del holdings[coin]; del entry_prices[coin]; del margins[coin]; entry_levs.pop(coin, None)
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
            cur = _get_fill_price(coin, date)
            if cur <= 0:
                continue
            slip = SLIPPAGE_MAP.get(coin, 0.0005)
            if coin not in holdings:
                entry_p = cur * (1 + slip)
                margin = target_margin[coin]
                cur_L = _cur_lev(date, coin)
                notional = margin * cur_L
                qty = notional / entry_p
                tx = notional * tx_cost
                if capital >= margin + tx:
                    capital -= margin + tx
                    holdings[coin] = qty
                    entry_prices[coin] = entry_p
                    entry_bar_index[coin] = _get_bar_index(coin, date)
                    margins[coin] = margin
                    entry_levs[coin] = cur_L
                    trade_count += 1
            else:
                delta = tqty - holdings[coin]
                if delta > holdings[coin] * 0.05:
                    entry_p = cur * (1 + slip)
                    add_notional = delta * entry_p
                    add_margin = add_notional / entry_levs.get(coin, _cur_lev(date, coin))
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

        # daily_gate: 일간 개념 체크(BL/drift)를 UTC 00시 바에서만 실행
        is_daily_bar = (not daily_gate) or (bpd == 1) or (hasattr(date, 'hour') and date.hour == 0)

        cur_month = date.strftime('%Y-%m')

        # ── 청산/스탑 체크 (롱 전용) ──
        for coin in list(holdings.keys()):
            ci = btc_idx_map.get(date, -1) if coin == 'BTC' else bars[coin].index.get_indexer([date], method='ffill')[0] if coin in bars else -1
            low = get_low(bars, coin, ci)
            if low <= 0:
                continue

            liq_price = _get_liq_price(coin, date) if entry_levs.get(coin, _cur_lev(date, coin)) > 1 else None
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
                pv_pre = _port_val(date)
                liq_log.append(dict(date=date, coin=coin, entry=entry_prices[coin],
                                    low=low, liq_price=liq_price,
                                    qty=holdings[coin], margin=margins[coin],
                                    pnl_at_low=pnl_at_low, returned=returned,
                                    loss_pct=(low/entry_prices[coin]-1),
                                    sleeve_pv_pre=pv_pre,
                                    sleeve_loss_pct=(margins[coin]-returned)/pv_pre if pv_pre>0 else 0))
                capital += returned
                del holdings[coin]; del entry_prices[coin]; del margins[coin]; entry_levs.pop(coin, None)
                entry_bar_index.pop(coin, None)
                liq_count += 1

        # ── mom-stop (헬스체크 필터 그대로 = 매 봉 보유 코인 mom 평가, 탈락 코인 매도) ──
        if mom_stop_threshold > -999.0 and holdings:
            for coin in list(holdings.keys()):
                df_c = bars.get(coin)
                if df_c is None: continue
                ci_c = df_c.index.get_indexer([prev_date], method='ffill')[0]
                if ci_c < 0 or ci_c < max(mom30, mom90): continue
                c_arr = df_c['Close'].values[:ci_c+1]
                ms = calc_mom(c_arr, mom30)
                ml = calc_mom(c_arr, mom90)
                if mom_stop_threshold < 0:
                    # sentinel -1: sign-only (헬스체크 동일)
                    fail = (ms < 0 and ml < 0)
                else:
                    # min(ms, ml) < -threshold
                    fail = min(ms, ml) < -mom_stop_threshold
                if not fail: continue
                # 현재 가 매도
                ci_d = bars[coin].index.get_indexer([date], method='ffill')[0]
                if ci_d < 0: continue
                slip = SLIPPAGE_MAP.get(coin, 0.0005)
                cur_open = float(bars[coin]['Open'].iloc[ci_d])
                exit_p = cur_open * (1 - slip)
                pnl = holdings[coin] * (exit_p - entry_prices[coin])
                tx = holdings[coin] * exit_p * tx_cost
                capital += margins[coin] + pnl - tx
                del holdings[coin]; del entry_prices[coin]; del margins[coin]; entry_levs.pop(coin, None)
                entry_bar_index.pop(coin, None)
                trade_count += 1
                stop_count += 1
                # snapshots 비움 → 다음 drift fire 시 refill v2 자동
                for sn in snapshots:
                    if coin in sn:
                        w = sn.pop(coin)
                        sn['CASH'] = sn.get('CASH', 0) + w

        # ── A: catastrophic stop (entry 가 대비 -X% 도달 시 강제 청산) ──
        if catastrophic_stop_pct > 0 and holdings:
            for coin in list(holdings.keys()):
                ci_d = bars[coin].index.get_indexer([date], method='ffill')[0] if coin in bars else -1
                if ci_d < 0: continue
                low_d = get_low(bars, coin, ci_d)
                entry = entry_prices.get(coin, 0)
                if entry <= 0 or low_d <= 0: continue
                trigger_price = entry * (1 - catastrophic_stop_pct)
                if low_d > trigger_price: continue
                # 강제 매도 (mark/trigger 중 큰 쪽 = 보수적)
                slip = SLIPPAGE_MAP.get(coin, 0.0005)
                cur_open = float(bars[coin]['Open'].iloc[ci_d])
                exit_p = min(cur_open, trigger_price) * (1 - slip)
                pnl = holdings[coin] * (exit_p - entry)
                tx = holdings[coin] * exit_p * tx_cost
                capital += margins[coin] + pnl - tx
                del holdings[coin]; del entry_prices[coin]; del margins[coin]; entry_levs.pop(coin, None)
                entry_bar_index.pop(coin, None)
                trade_count += 1
                cat_stop_count += 1
                no_reentry[coin] = bar_i + catastrophic_cooldown_bars
                # snapshots 비움 + cooldown 동안 재진입 차단 (compute_weights 단에서도 처리)
                for sn in snapshots:
                    if coin in sn:
                        w = sn.pop(coin)
                        sn['CASH'] = sn.get('CASH', 0) + w

        # ── G: cushion buffer (cushion < mult × maint 이면 가장 약한 코인 매도) ──
        if cushion_buffer_mult > 0 and holdings:
            wallet = capital + sum(margins.values())
            total_maint = 0.0
            for c in holdings:
                ci_c = bars[c].index.get_indexer([date], method='ffill')[0] if c in bars else -1
                if ci_c < 0: continue
                cur_c = float(bars[c]['Open'].iloc[ci_c])
                if cur_c <= 0: continue
                total_maint += holdings[c] * cur_c * maint_rate
            if total_maint > 0 and wallet < cushion_buffer_mult * total_maint:
                # 가장 큰 unrealized loss% 코인 매도
                worst_coin = None; worst_loss = 0
                for c in holdings:
                    ci_c = bars[c].index.get_indexer([date], method='ffill')[0] if c in bars else -1
                    if ci_c < 0: continue
                    cur_c = float(bars[c]['Open'].iloc[ci_c])
                    loss = (cur_c - entry_prices[c]) / entry_prices[c] if entry_prices[c] > 0 else 0
                    if loss < worst_loss:
                        worst_loss = loss
                        worst_coin = c
                if worst_coin is not None:
                    ci_d = bars[worst_coin].index.get_indexer([date], method='ffill')[0]
                    slip = SLIPPAGE_MAP.get(worst_coin, 0.0005)
                    exit_p = float(bars[worst_coin]['Open'].iloc[ci_d]) * (1 - slip)
                    pnl = holdings[worst_coin] * (exit_p - entry_prices[worst_coin])
                    tx = holdings[worst_coin] * exit_p * tx_cost
                    capital += margins[worst_coin] + pnl - tx
                    del holdings[worst_coin]; del entry_prices[worst_coin]; del margins[worst_coin]; entry_levs.pop(worst_coin, None)
                    entry_bar_index.pop(worst_coin, None)
                    trade_count += 1
                    cushion_force_count += 1
                    for sn in snapshots:
                        if worst_coin in sn:
                            w = sn.pop(worst_coin)
                            sn['CASH'] = sn.get('CASH', 0) + w

        # ── 펀딩비 (prev_date < t <= date 윈도우 합산: D봉=3회, 4h=1~2회, 1h=0~1회) ──
        for coin in list(holdings.keys()):
            fr_series = funding.get(coin)
            if fr_series is None:
                continue
            window = fr_series.loc[(fr_series.index > prev_date) & (fr_series.index <= date)]
            if len(window) == 0:
                continue
            fr_sum = float(window.sum())
            if fr_sum != 0 and not np.isnan(fr_sum):
                cur = get_close(bars, coin, btc_idx_map.get(date, -1) if coin == 'BTC' else
                                bars[coin].index.get_indexer([date], method='ffill')[0] if coin in bars else -1)
                if cur > 0:
                    # 롱: notional × sum(rate)
                    capital -= holdings[coin] * cur * fr_sum
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

        # ── 시그널 → 스냅샷 갱신 ──
        need_rebal = False

        if external_target_schedule is None:
            # 내부 시그널 모드 (기존)
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
            import os as _osa
            _anchor_defer = _osa.environ.get('ANCHOR_TRADE_MODE', 'on') == 'defer'
            if canary_on and not canary_flipped:
                if snap_interval_bars > 0:
                    for si in range(n_snapshots):
                        offset = int(si * snap_interval_bars / n_snapshots)
                        if (bar_i + phase_offset_bars) % snap_interval_bars == offset:
                            new_w = _compute_weights(prev_date)
                            if new_w != snapshots[si]:
                                snapshots[si] = new_w
                                if not _anchor_defer:
                                    need_rebal = True
                else:
                    day_of_month = date.day
                    for si, anchor in enumerate(snap_days):
                        key = f"{cur_month}_snap{si}"
                        if day_of_month >= anchor and key not in snap_done:
                            snap_done[key] = True
                            new_w = _compute_weights(prev_date)
                            if new_w != snapshots[si]:
                                snapshots[si] = new_w
                                if not _anchor_defer:
                                    need_rebal = True

            combined = _merge_snapshots()
        else:
            # 외부 schedule 모드 (앙상블 BT 용)
            raw = external_target_schedule.get(date, None)
            if raw is None:
                combined = {'CASH': 1.0}
                ext_rebal = False
            else:
                raw_target = dict(raw.get('target', {}))
                ext_rebal = bool(raw.get('rebal', False))
                # NaN/음수/미존재 코인 제거 (제거분은 CASH 로)
                # 참고: BL 필터는 적용 안함 — 기존 run() 도 snapshot 에 BL 코인 잔존 허용
                bad = 0.0
                for coin in list(raw_target.keys()):
                    v = raw_target[coin]
                    if v is None or (isinstance(v, float) and (v != v)) or v < 0:
                        bad += max(0.0, raw_target.pop(coin))
                    elif coin != 'CASH' and coin not in bars:
                        bad += raw_target.pop(coin)
                if bad > 0:
                    raw_target['CASH'] = raw_target.get('CASH', 0.0) + bad
                # 정규화 (합이 0이면 CASH 1)
                s = sum(raw_target.values())
                if s > 0:
                    combined = {k: v/s for k, v in raw_target.items() if v > 0}
                else:
                    combined = {'CASH': 1.0}
            need_rebal = ext_rebal

        # ── Drift (daily_gate 시 일 1회만) ──
        # 변종: DRIFT_HEALTH_MODE = 'off' | 'refill' | 'cash'
        if not need_rebal and canary_on and drift_threshold > 0 and holdings and is_daily_bar:
            if _half_turnover(_current_weights(prev_date), combined) >= drift_threshold:
                need_rebal = True
                # 변종: 헬스 fail 슬롯 처리 (mode != off 일 때만)
                import os as _os
                _drift_mode = _os.environ.get('DRIFT_HEALTH_MODE', 'refill')  # V24 default ON
                _drift_refill = _drift_mode in ('refill', 'cash')
                _drift_cash_only = _drift_mode == 'cash'
                if _drift_refill:
                    # 사용자 결정: 두 mom 모두 음수면 fail (vol 무시).
                    # 메모리 Do-Not-Repeat: health-fail 에 vol 포함 금지.
                    def _is_failed(coin):
                        df_c = bars.get(coin)
                        if df_c is None: return True
                        ci_c = df_c.index.get_indexer([prev_date], method='ffill')[0]
                        if ci_c < 0 or ci_c < max(mom30, mom90): return True
                        c_arr = df_c['Close'].values[:ci_c+1]
                        ms = calc_mom(c_arr, mom30)
                        ml = calc_mom(c_arr, mom90)
                        return ms < 0 and ml < 0
                    fresh = _compute_weights(prev_date)
                    healthy_coins = sorted([c for c in fresh.keys() if c != 'CASH'])
                    for si in range(n_snapshots):
                        sn = snapshots[si]
                        sn_coins = sorted([c for c in sn.keys() if c != 'CASH'])
                        new_sn = {}
                        replaced_w = 0.0
                        for c in sn_coins:
                            if not _is_failed(c):
                                new_sn[c] = sn[c]
                            else:
                                replaced_w += sn[c]
                        new_sn['CASH'] = sn.get('CASH', 0)
                        if replaced_w > 0:
                            if _drift_cash_only:
                                new_sn['CASH'] = new_sn.get('CASH', 0) + replaced_w
                            else:
                                already = set(new_sn.keys())
                                fresh_picks = [c for c in healthy_coins if c not in already]
                                n_failed = max(1, len(sn_coins) - (len(new_sn) - 1))
                                if fresh_picks:
                                    picks = fresh_picks[:n_failed]
                                    w_per = replaced_w / len(picks)
                                    for c in picks:
                                        new_sn[c] = new_sn.get(c, 0) + w_per
                                else:
                                    new_sn['CASH'] = new_sn.get('CASH', 0) + replaced_w
                        snapshots[si] = new_sn
                    # combined 재계산
                    combined = _merge_snapshots()
                # V25 옵션 B: CASH 슬롯 refill (drift 발화 + 빈 CASH 슬롯 + fresh healthy 존재 시)
                if _os.environ.get('DRIFT_CASH_REFILL', 'off') == 'on':
                    _fresh_full = _compute_weights(prev_date)
                    _fresh_pool = sorted([c for c in _fresh_full.keys() if c != 'CASH'])
                    if _fresh_pool:
                        for si in range(n_snapshots):
                            sn = snapshots[si]
                            cash_w = sn.get('CASH', 0)
                            if cash_w <= 0.001:
                                continue
                            already = set(c for c in sn.keys() if c != 'CASH')
                            candidates = [c for c in _fresh_pool if c not in already]
                            if not candidates:
                                continue
                            # 빈 슬롯 수 = round(cash_w / cap)
                            n_slots = max(1, int(round(cash_w / cap)))
                            picks = candidates[:n_slots]
                            w_per = cash_w / len(picks)
                            new_sn = dict(sn)
                            del new_sn['CASH']
                            for c in picks:
                                new_sn[c] = new_sn.get(c, 0) + w_per
                            snapshots[si] = new_sn
                        combined = _merge_snapshots()

        # ── A 안 cooldown: no_reentry 코인 combined 에서 제거 ──
        if no_reentry:
            for c in list(combined.keys()):
                if c == 'CASH': continue
                if c in no_reentry and bar_i < no_reentry[c]:
                    w = combined.pop(c)
                    combined['CASH'] = combined.get('CASH', 0) + w
                elif c in no_reentry and bar_i >= no_reentry[c]:
                    del no_reentry[c]

        # ── C: gross exposure cap (sum L_i × w_i ≤ gross_exposure_cap) ──
        if gross_exposure_cap > 0:
            non_cash = {c: w for c, w in combined.items() if c != 'CASH'}
            if non_cash:
                gross = sum(w * _cur_lev(date, c) for c, w in non_cash.items())
                if gross > gross_exposure_cap:
                    scale = gross_exposure_cap / gross
                    new_combined = {}
                    cash_add = 0
                    for c, w in non_cash.items():
                        new_combined[c] = w * scale
                        cash_add += w * (1 - scale)
                    new_combined['CASH'] = combined.get('CASH', 0) + cash_add
                    combined = new_combined
                    gross_cap_count += 1

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
            _execute_rebalance(combined, date, pv_date=prev_date)
            rebal_count += 1
            # 신규/추가된 포지션 same-bar liq 검사 (entry 시 같은 봉 Low 이미 알려진 경우 fairness)
            for coin in list(holdings.keys()):
                ci2 = btc_idx_map.get(date, -1) if coin == 'BTC' else (bars[coin].index.get_indexer([date], method='ffill')[0] if coin in bars else -1)
                low2 = get_low(bars, coin, ci2)
                if low2 <= 0: continue
                liq_p = _get_liq_price(coin, date) if entry_levs.get(coin, _cur_lev(date, coin)) > 1 else None
                if liq_p is not None and low2 <= liq_p:
                    pnl_at_low = holdings[coin] * (low2 - entry_prices[coin])
                    eq_at = margins[coin] + pnl_at_low
                    liq_fee = max(eq_at, 0) * 0.015
                    returned = max(eq_at - liq_fee, 0)
                    capital += returned
                    del holdings[coin]; del entry_prices[coin]; del margins[coin]; entry_levs.pop(coin, None)
                    entry_bar_index.pop(coin, None)
                    liq_count += 1

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
        'Stops': stop_count, 'Stop': stop_count,
        'Cat': cat_stop_count, 'Cushion': cushion_force_count, 'GrossCap': gross_cap_count,
    }
    result['_equity'] = eq  # equity curve (pd.Series)
    result['_liq_log'] = liq_log
    return result


# ═══ 메인 ═══
if __name__ == '__main__':
    t0 = time.time()

    print("전략: V18 Cap Defend 코인 (바이낸스 선물 포팅)")
    print("  카나리: BTC > SMA(50일) + 1.5% hyst")
    print("  헬스: Mom30>0 AND Mom90>0 AND Vol90≤5%")
    print("  선정: 시총순 Top5 → Greedy Absorption  (V18 기본값, V21은 Top3 필수 → 이 main 블록 아님)")
    print("  비중: EW + Cap 33%")
    print("  리스크: 가드 없음 (V24)")
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


# ─── V25 K2 시그널 빌더 (라이브 spec 동일) ───
def build_K2_signal(bars,
                    btc_cap_sma_period=42,
                    btc_cap_thr_mid=1.015,
                    btc_cap_thr_max=1.05,
                    k2_sma_period=7,
                    k2_hyst=0.025,
                    l_min=2.0,
                    l_mid=3.0,
                    l_max=4.0):
    """V25 동적 per-coin L 시그널.

    각 코인 L = min(BTC_cap, per_coin_K2).
    - BTC_cap: BTC/SMA{btc_cap_sma_period} ratio → l_min/l_mid/l_max
    - per_coin_K2: close/SMA{k2_sma_period} ratio → l_min/l_mid/l_max (hyst band)
    - shift(1) lag

    Returns: dict[coin -> pd.Series] (per-date L per coin)
    """
    import numpy as _np, pandas as _pd
    btc_df = bars.get('BTC')
    if btc_df is None:
        return {}
    btc_close = _pd.Series(btc_df['Close'].values, index=btc_df.index)
    btc_sma = btc_close.rolling(btc_cap_sma_period).mean()
    btc_ratio = btc_close / btc_sma
    btc_cap = _pd.Series(_np.where(btc_ratio > btc_cap_thr_max, l_max,
                          _np.where(btc_ratio > btc_cap_thr_mid, l_mid, l_min)),
                          index=btc_ratio.index).shift(1).ffill().fillna(l_min)
    out = {}
    for coin in bars:
        close = bars[coin]['Close']
        sma = close.rolling(k2_sma_period).mean()
        ratio = close / sma
        thr_max = 1.0 + k2_hyst * 3
        thr_mid = 1.0 + k2_hyst
        pc = _pd.Series(_np.where(ratio > thr_max, l_max,
                         _np.where(ratio > thr_mid, l_mid, l_min)),
                         index=close.index).shift(1).ffill().fillna(l_min)
        idx = pc.index.intersection(btc_cap.index)
        out[coin] = _pd.Series(_np.minimum(pc.loc[idx].values, btc_cap.loc[idx].values), index=idx)
    return out
