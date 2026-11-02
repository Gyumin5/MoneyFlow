#!/usr/bin/env python3
"""Unified backtest — 현물/선물 공통 시간봉 엔진.

asset_type="spot"|"fut" 플래그로 분기. 차이는 파라미터로만:
- tx_cost (spot 0.004 / fut 0.0004)
- leverage, funding, liquidation (fut only)
- cash_buffer_pct (spot 0.02 / fut 0.05 idle margin)
- slippage_mode ("none" / "tier")

데이터 소스: Binance 1h perp (spot proxy). KRW-USD 환율·Upbit 상폐 편향 무시.
실매매 parity 불필요 — 백테스트 전용. 실매매는 별도 live engine 사용.

시그널+체결 모두 시간봉 단위. 원 backtest_futures_full 로직 전부 포팅:
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

# 이 파일은 research/ 한 칸 아래에 있다. 경로를 __file__ 기준으로 두면 자료 디렉토리를
# strategies/data/ 로 찾아 못 찾고, 유니버스가 빈 채로 조용히 돌아 거래가 0 이 된다
# (2026-11-02 실측: 성과가 전부 0 인데 예외도 경고도 없다). 정본 엔진이 보는 자리를 그대로 쓴다.
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
DATA_DIR = os.path.join(BASE_DIR, '..', '..', 'data', 'futures')

# 현금 버퍼 — 라이브 정본과 같은 값이어야 한다.
# 현물 라이브는 trade_state.json 의 spot_cash_buffer 를, 선물 라이브는 binance_state.json 의
# fut_cash_buffer 를 읽고 둘 다 1% 다. 목표 규모를 총자산×(1-버퍼)×비중으로 잡는 식도 같다.
# 여기 값이 라이브와 다르면 백테스트가 라이브와 다른 규모로 돌아서 절대 수치를 그대로 읽을 수 없다.
# 2026-09-22 까지는 버퍼 대신 1/(1+leverage×tx_cost)=0.99602 를 썼다 — 마지막 매수 leg 가
# 자본 부족으로 빠지지 않게 거래비용분만 예약한 값이라 라이브 버퍼와 목적이 다르다.
# 1% 는 그 예약분보다 크므로 그 미체결 방지 성질은 그대로 유지된다.
CASH_BUFFER = 0.01

# ─── 심볼/슬리피지 ───
def _discover_tickers():
    """가지고 있는 선물 데이터에서 심볼표를 만든다.

    고정 목록으로 두면 조용히 낡는다 — 2026-09-22 실측에서 시점별 상위 40 중
    스물한 종이 이 표에 없어서, 라이브는 살 수 있는 코인을 백테스트는 후보로
    보지도 못했다. 데이터 폴더는 download_futures_data.py 가 시점별 유니버스로
    채우므로, 여기서 폴더를 읽으면 유니버스 갱신이 그대로 따라온다.
    """
    if not os.path.isdir(DATA_DIR):
        return {}
    out = {}
    for f in sorted(os.listdir(DATA_DIR)):
        if not f.endswith('_1h.csv'):
            continue
        sym = f[:-7]
        if not sym.endswith('USDT'):
            continue
        coin = sym[:-4]
        # 1000SHIBUSDT 처럼 1000배 단위 계약은 같은 자산이다. 유니버스는 SHIB 로
        # 부르므로 접두를 떼어 키를 맞춘다.
        if coin.startswith('1000') and len(coin) > 4:
            coin = coin[4:]
        out[coin] = sym
    return out


TICKER_MAP = _discover_tickers()
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
    # 갱신되는 파일은 data/ 하나다. backup_20260125/ 사본을 먼저 보던 경로는
    # 2025-12-01 에서 멈춰 있어서, 그 뒤 구간이 통째로 2025년 12월 순위로 돌았다.
    p = os.path.join(BASE_DIR, '..', '..', 'data', 'historical_universe.json')
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        raw = json.load(f)
    result = {}
    for ds, tickers in raw.items():
        # 'BTC-USD' → 'BTC'
        result[ds] = [t.replace('-USD', '') for t in tickers if t.replace('-USD', '') in TICKER_MAP]
    return result

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
        # ─── 숏 연구용 (short_design_spec.md). 기본값은 전부 꺼짐이라
        #     enable_short=False 면 이 엔진은 원본과 같은 경로로 돈다(= 구성 B). ───
        enable_short=False,
        short_entry_discount=0.03,    # a(x) = min(0.985, 1 - x)
        short_slope_days=10,          # s = SMA42_t / SMA42_(t-N) - 1
        short_slope_gate=True,        # 진입 때 s < 0 요구. 해제는 청산 신호가 아니다.
        short_exit_margin=0.015,      # 청산 문턱 = a(x) + 이 값
        short_health_mode='inv_mom2vol',  # 'inv_mom2vol' | 'failed_rally'
        short_universe_size=5,
        short_cap=0.20,               # 1/5
        short_leverage=1.0,
        short_gross_exposure_cap=0.75,
        short_per_coin_cap=0.20,      # 종목별 상한 = E × min(이 값, G_cap/N)
        short_stop_kind='entry_price_pct',  # 'none' | 'entry_price_pct' | 'lowest_close_since_entry_pct'
        short_stop_pct=0.12,
        short_catastrophic_stop_pct=0.25,
        short_catastrophic_cooldown_bars=5,
        short_post_flip_delay=5,
        short_rebalance_mode='anchor_fixed_qty',  # | 'target_weight'
        short_cooldown_bars=5,        # 손절 뒤 그 종목 숏 재진입 금지 기간
        funding_veto=False,
        funding_veto_rate=-0.0001,    # 8시간당 환산 기준
        cushion_buffer_mult=1.5,
        cash_transit_bars=1,          # 방향 전환 사이에 강제로 비워 두는 봉 수
        universe_size=5, selection='greedy', cap=1/3,
        tx_cost=0.0004, maint_rate=0.004,
        asset_type='fut',  # 'spot' | 'fut'
        slippage_mode=None,  # None=asset_type 기본 (spot 'none', fut 'tier')
        # 윈도우: _days 또는 _bars 중 하나만 지정. _bars 우선.
        sma_days=50, mom_short_days=30, mom_long_days=90, vol_days=90,
        sma_bars=0, mom_short_bars=0, mom_long_bars=0, vol_bars=0,
        mom_mid_days=0, mom_mid_bars=0,  # V25: 3-mom 지원 (mom_mid 추가)
        zscore_mom_bars=0, zscore_sharpe_bars=0,  # V25 selection='zscore' 룩백
        canary_hyst=0.015,
        dd_lookback=60, dd_threshold=-0.25,
        dd_bars_override=0,  # 0이면 dd_lookback*bpd 사용
        bl_drop=-0.15, bl_days=7,
        bl_bars_override=0,
        drift_threshold=0.10, post_flip_delay=5,
        daily_gate=False,
        health_mode='mom2vol',  # 'mom2vol'(기본), 'mom1vol', 'mom1', 'mom2', 'vol', 'sma', 'smavol', 'none'
        health_sma_days=0,  # 0이면 sma_days(카나리) 와 동일. 헬스 SMA 와 카나리 SMA 분리용.
        vol_mode='daily',  # 'daily'(일봉 리샘플) or 'bar'(순수 봉 기반 연환산)
        vol_threshold=0.05,  # vol_mode='daily' 기본. bar mode면 연환산 기준 (예: 0.80)
        n_snapshots=3,  # 스냅샷 수 (3=월3회, 6=월6회, 12=거의매일)
        snap_interval_bars=0,  # 0=달력 기반, >0=봉 기반 앵커 간격
        phase_offset_bars=0,  # 봉 기반 앵커 위상 오프셋 (0=기존 동작)
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
        # --- 2026-04-24 추가 (Step 7 stress 용) ---
        execution_delay_bars=0,  # 0=즉시 체결, N>0=signal 에서 N bar 뒤 체결
        exclude_assets=None,  # {asset_name} 제외할 자산 set (drop_top_contributor 용)
        daily_health_exit=False,  # True 시 매일 보유 코인 헬스 체크. 탈락 코인은 combined target 에서 0 → CASH
        # ------------------------------------------
        _trace=None):  # list를 넘기면 매 봉 {'date':..., 'target':..., 'rebal':bool} 기록
    """시간봉 통합 백테스트. spot/fut 공통."""

    # asset_type 기본값 dispatch
    if asset_type not in ('spot', 'fut'):
        raise ValueError(f"asset_type must be 'spot' or 'fut', got {asset_type!r}")
    if slippage_mode is None:
        slippage_mode = 'none' if asset_type == 'spot' else 'tier'
    is_spot = (asset_type == 'spot')
    if is_spot:
        leverage = 1.0  # spot은 무조건 lev=1

    # slippage 조회 헬퍼 (tier vs none)
    def _slip(coin):
        return 0.0 if slippage_mode == 'none' else SLIPPAGE_MAP.get(coin, 0.0005)

    # 배분 규모는 라이브와 같은 현금 버퍼로 잡는다(모듈 상단 CASH_BUFFER).
    # 예약 성질 확인: sum(margin_i + tx_i) = pv*(1-buffer)*(1 + leverage*tx_cost) < pv 이므로
    # 마지막 leg 가 자본 부족으로 미체결되는 일은 여전히 없다.
    margin_factor = 1.0 - CASH_BUFFER

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
    health_sma_period = (health_sma_days * bpd) if health_sma_days and health_sma_days > 0 else sma_period
    mom30 = mom_short_bars if mom_short_bars > 0 else mom_short_days * bpd
    mom90 = mom_long_bars if mom_long_bars > 0 else mom_long_days * bpd
    mom60 = mom_mid_bars if mom_mid_bars > 0 else (mom_mid_days * bpd if mom_mid_days > 0 else 0)
    _vol_bars = vol_bars if vol_bars > 0 else vol_days * bpd
    dd_bars = dd_bars_override if dd_bars_override > 0 else dd_lookback * bpd
    bl_cooldown = bl_bars_override if bl_bars_override > 0 else bl_days * bpd
    pfd_bars = pfd_bars_override if pfd_bars_override > 0 else post_flip_delay * bpd
    crash_cool_bars = crash_cool_override if crash_cool_override > 0 else 3 * bpd
    _crash_lookback = crash_lookback_bars if crash_lookback_bars > 0 else bpd
    _bl_lookback = bl_lookback_bars if bl_lookback_bars > 0 else bpd

    # State
    capital = initial_capital
    holdings = {}  # {coin: qty}  — 항상 양수. 방향은 아래 pos_dir 하나가 들고 있다.

    # ── 슬리브 방향 ──
    # 운영자가 롱숏 동시 보유를 금지했으므로(2026-11-01) 슬리브 전체가 한 방향이다.
    # 그래서 수량에 부호를 주는 대신 방향 변수 하나를 둔다. 손대는 자리가 줄고,
    # 명세 2절이 경고한 "부호 반전 이중 적용" 이 구조적으로 불가능해진다 —
    # pos_dir 은 손익식과 펀딩식에 각각 정확히 한 번씩만 곱해진다.
    # 수수료는 방향과 무관한 비용이라 pos_dir 을 안 곱한다(곱하면 숏에서 수수료가 수입이 된다).
    pos_dir = 1          # +1 롱, -1 숏. 보유가 없으면 의미 없음.
    sleeve_state = 'CASH'  # 'LONG' | 'CASH' | 'SHORT'
    cash_until_bar = -1    # 이 봉까지는 어느 방향으로도 진입하지 않는다(방향 전환 유예)
    short_cooldown = {}    # {coin: 남은 봉 수} 손절 뒤 숏 재진입 금지
    short_entry_bar = None
    short_pfd_done = True
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
    pending_rebalance = None  # execution_delay_bars 용 (signal bar 에서 N bar 뒤 체결)
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
                pnl = pos_dir * holdings[coin] * (cur - entry_prices[coin])
                pv += margins[coin] + pnl
        return pv

    def _get_price(coin, date):
        """date 기준 종가 (ffill). 평가·펀딩·DD/BL용."""
        df = bars.get(coin)
        if df is None:
            return 0
        ci = df.index.get_indexer([date], method='ffill')[0]
        if ci < 0:
            return 0
        return float(df['Close'].iloc[ci])

    def _get_fill_price(coin, date):
        """체결가: 해당 봉 Open (signal=t-1 close, fill=t open 원칙).
        exact match만 허용 — 해당 봉이 없으면 0(미체결)."""
        df = bars.get(coin)
        if df is None:
            return 0
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

    def _get_short_stop_price(coin, date):
        """숏 손절가. 롱과 거울이다 — 기준의 최소를 잡고 위로 문턱을 둔다.

        롱 경로(_get_stop_price)는 손대지 않는다. 그쪽을 건드리면 B 가 A 를 재현 못 한다.
        """
        if short_stop_kind == 'none' or short_stop_pct <= 0:
            return None
        df = bars.get(coin)
        if df is None:
            return None
        ci = _get_bar_index(coin, date)
        if ci <= 0:
            return None
        if short_stop_kind == 'entry_price_pct':
            ref = entry_prices.get(coin, 0)
        elif short_stop_kind == 'lowest_close_since_entry_pct':
            start_ci = entry_bar_index.get(coin, -1)
            if start_ci < 0:
                return None
            ref = float(np.min(df['Close'].iloc[start_ci:ci]))
        else:
            return None
        if ref <= 0:
            return None
        return ref * (1.0 + short_stop_pct)

    def _execute_stop_exit(coin, date, stop_price):
        nonlocal capital, trade_count, stop_count
        ci = _get_bar_index(coin, date)
        if ci < 0:
            return False
        if pos_dir < 0:
            high = get_high(bars, coin, ci)
            if high <= 0 or high < stop_price:
                return False
            slip = _slip(coin)
            cur_open = float(bars[coin]['Open'].iloc[ci])
            # 숏 손절은 되사는 주문이라 불리한 쪽이 위다.
            exit_p = max(cur_open, stop_price) * (1 + slip)
            pnl = pos_dir * holdings[coin] * (exit_p - entry_prices[coin])
            tx = holdings[coin] * exit_p * tx_cost
            capital += margins[coin] + pnl - tx
            del holdings[coin]; del entry_prices[coin]; del margins[coin]
            entry_bar_index.pop(coin, None)
            short_cooldown[coin] = short_cooldown_bars
            trade_count += 1
            stop_count += 1
            return True
        low = get_low(bars, coin, ci)
        if low <= 0 or low > stop_price:
            return False
        slip = _slip(coin)
        cur_open = float(bars[coin]['Open'].iloc[ci])
        exit_p = min(cur_open, stop_price) * (1 - slip)
        pnl = pos_dir * holdings[coin] * (exit_p - entry_prices[coin])
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
        if pos_dir < 0:
            # 숏: 자본 = margin - qty*(P-entry), 유지증거금 = qty*P*maint.
            # 자본 <= 유지증거금 이 되는 지점이 위에 있다.
            denom = qty * (1.0 + maint_rate)
            if denom <= 0:
                return None
            return (qty * entry + margin) / denom
        denom = qty * (1.0 - maint_rate)
        if denom <= 0:
            return None
        liq_price = (qty * entry - margin) / denom
        if liq_price <= 0:
            return None
        return liq_price

    def _execute_rebalance(target_weights, date, pv_date=None):
        """Delta 리밸런싱 with 선물 비용."""
        nonlocal capital, trade_count
        # 체결 밴드 = 각 sleeve 의 라이브 규칙 (2026-08-25 정합).
        #   spot: trade/executor_coin.py 는 상대 밴드가 없다. 절대 최소주문 5,000원만 있고
        #         그건 슬리브 규모 대비 0.004%~0.5% 라 성과 영향이 5.8년 최대괴리 0.2% 로
        #         무의미하다(research/bt_spot_min_order.py). → 0.
        #   fut:  trade/auto_trade_binance.py DELTA_THRESHOLD=0.05.
        # 밴드를 축으로 재실험하려면 research/bt_spot_rebal_band.engine.patch 를 적용한다.
        _band = 0.0 if asset_type == 'spot' else 0.05
        # 주문 크기를 정하는 평가액은 신호봉(t-1) 종가로 잰다. 체결가는 t 봉 Open 이라
        # 같은 t 봉 종가로 크기를 정하면 그날 결과를 미리 보는 것이다(2026-10-18 운영자
        # 지시). 선물 엔진은 같은 자리에 이미 pv_date=prev_date 가 있다
        # (backtest_futures_v25.py:1088). 호출부가 pv_date 를 안 주면 종전대로 date.
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
            tmgn = pv * w * margin_factor
            tnotional = tmgn * leverage
            tqty = tnotional / cur
            target_qty[coin] = tqty
            target_margin[coin] = tmgn

        # 매도 (보유 중이지만 target에 없거나 줄어야)
        for coin in list(holdings.keys()):
            cur = _get_fill_price(coin, date)
            if cur <= 0:
                continue
            slip = _slip(coin)
            if coin not in target_qty:
                # 전량 매도
                exit_p = cur * (1 - slip)
                pnl = pos_dir * holdings[coin] * (exit_p - entry_prices[coin])
                tx = holdings[coin] * cur * tx_cost
                capital += margins[coin] + pnl - tx
                del holdings[coin]; del entry_prices[coin]; del margins[coin]
                entry_bar_index.pop(coin, None)
                trade_count += 1
            else:
                delta = target_qty[coin] - holdings[coin]
                if delta < -holdings[coin] * _band:
                    sell_qty = -delta
                    sell_frac = sell_qty / holdings[coin]
                    sell_margin = margins[coin] * sell_frac
                    exit_p = cur * (1 - slip)
                    pnl = pos_dir * sell_qty * (exit_p - entry_prices[coin])
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
            slip = _slip(coin)
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
                if delta > holdings[coin] * _band:
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
        min_bars = max(mom30, mom90, mom60, _vol_bars, sma_period, health_sma_period)
        _excluded = exclude_assets or frozenset()
        for coin in mcap_order:
            if coin in blacklist:
                continue
            if coin in _excluded:
                continue  # drop_top_contributor 시 제외
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
            m_long = calc_mom(c, mom90) if ('mom2' in health_mode or '3mom' in health_mode) else 999
            m_mid = calc_mom(c, mom60) if ('3mom' in health_mode and mom60 > 0) else 999
            if 'vol' in health_mode:
                if vol_mode == 'bar':
                    vol = calc_vol_bars(c, _vol_bars, bars_per_year)
                else:
                    vol = calc_vol_daily(c, bpd, lookback_bars=_vol_bars)
            else:
                vol = 0

            sma_ok = True
            if 'sma' in health_mode and len(c) >= health_sma_period:
                sma_val = float(np.mean(c[-health_sma_period:]))
                sma_ok = c[-1] > sma_val

            if health_mode == 'mom2vol':
                ok = m_short > 0 and m_long > 0 and vol <= vol_threshold
            elif health_mode == '3momvol':
                ok = m_short > 0 and m_mid > 0 and m_long > 0 and vol <= vol_threshold
            elif health_mode == '3mom':
                ok = m_short > 0 and m_mid > 0 and m_long > 0
            elif health_mode == 'mom1vol':
                ok = m_short > 0 and vol <= vol_threshold
            elif health_mode == 'mom1':
                ok = m_short > 0
            elif health_mode == 'mom2':
                ok = m_short > 0 and m_long > 0
            elif health_mode == 'vol':
                ok = vol <= vol_threshold
            elif health_mode == 'sma':
                ok = sma_ok
            elif health_mode == 'smavol':
                ok = sma_ok and vol <= vol_threshold
            else:
                ok = True
            if ok:
                healthy.append(coin)

        # Z-score selection (Mom252 + Sharpe126), V25 옵션
        if selection == 'zscore' and len(healthy) > 1:
            zscore_lb_mom = zscore_mom_bars if zscore_mom_bars > 0 else 252 * bpd
            zscore_lb_sh = zscore_sharpe_bars if zscore_sharpe_bars > 0 else 126 * bpd
            moms, shs = {}, {}
            for coin in healthy:
                df = bars.get(coin)
                if df is None:
                    continue
                ci = df.index.get_indexer([sig_date], method='ffill')[0]
                if ci < 0:
                    continue
                c = df['Close'].values[:ci + 1]
                if len(c) < zscore_lb_mom + 2:
                    continue
                moms[coin] = float(c[-1] / c[-zscore_lb_mom - 1] - 1)
                if len(c) < zscore_lb_sh + 2:
                    continue
                rets = np.diff(c[-zscore_lb_sh - 1:]) / c[-zscore_lb_sh - 1:-1]
                sd = float(rets.std())
                shs[coin] = float(rets.mean() / sd * np.sqrt(bars_per_year)) if sd > 0 else 0.0
            keys = [c for c in healthy if c in moms and c in shs]
            if len(keys) > 1:
                m_arr = np.array([moms[c] for c in keys])
                s_arr = np.array([shs[c] for c in keys])
                m_std = m_arr.std() or 1.0
                s_std = s_arr.std() or 1.0
                z = (m_arr - m_arr.mean()) / m_std + (s_arr - s_arr.mean()) / s_std
                healthy = [k for _, k in sorted(zip(-z, keys))]
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

    def _is_healthy(coin, sig_date):
        """daily health re-check (하나의 코인). _compute_weights 와 동일 헬스 로직."""
        if coin == 'CASH' or coin == 'Cash':
            return True
        df = bars.get(coin)
        if df is None:
            return False
        ci = df.index.get_indexer([sig_date], method='ffill')[0]
        min_bars_local = max(mom30, mom90, mom60, _vol_bars, sma_period, health_sma_period)
        if ci < 0 or ci < min_bars_local:
            return False
        c = df['Close'].values[:ci + 1]
        if health_mode == 'none':
            return True
        m_short = calc_mom(c, mom30) if 'mom' in health_mode else 999
        m_long = calc_mom(c, mom90) if ('mom2' in health_mode or '3mom' in health_mode) else 999
        m_mid = calc_mom(c, mom60) if ('3mom' in health_mode and mom60 > 0) else 999
        if 'vol' in health_mode:
            if vol_mode == 'bar':
                vol = calc_vol_bars(c, _vol_bars, bars_per_year)
            else:
                vol = calc_vol_daily(c, bpd, lookback_bars=_vol_bars)
        else:
            vol = 0
        sma_ok = True
        if 'sma' in health_mode and len(c) >= health_sma_period:
            sma_val = float(np.mean(c[-health_sma_period:]))
            sma_ok = c[-1] > sma_val
        if health_mode == 'mom2vol':
            return m_short > 0 and m_long > 0 and vol <= vol_threshold
        elif health_mode == '3momvol':
            return m_short > 0 and m_mid > 0 and m_long > 0 and vol <= vol_threshold
        elif health_mode == '3mom':
            return m_short > 0 and m_mid > 0 and m_long > 0
        elif health_mode == 'mom1vol':
            return m_short > 0 and vol <= vol_threshold
        elif health_mode == 'mom1':
            return m_short > 0
        elif health_mode == 'mom2':
            return m_short > 0 and m_long > 0
        elif health_mode == 'vol':
            return vol <= vol_threshold
        elif health_mode == 'sma':
            return sma_ok
        elif health_mode == 'smavol':
            return sma_ok and vol <= vol_threshold
        return True

    def _coin_mom2(coin, sig_date):
        """coin 의 (mom_short, mom_long) at sig_date(t-1). 데이터 부족 시 None."""
        df = bars.get(coin)
        if df is None:
            return None
        ci = df.index.get_indexer([sig_date], method='ffill')[0]
        if ci < 0 or ci < max(mom30, mom90):
            return None
        c = df['Close'].values[:ci + 1]
        return calc_mom(c, mom30), calc_mom(c, mom90)

    def _refill_v2_snapshots(sig_date):
        """라이브 _apply_refill_v2_to_state(coin_live_engine.py) 정합 refill.
        각 snapshot 에서 mom_short<0 AND mom_long<0 코인 제거 → fresh healthy
        (mom_short>0 AND mom_long>0, -mom_short then symbol) EW 충당. vol 미포함.
        sig_date=t-1 (look-ahead 방지). snapshots in-place 수정."""
        healthy = []
        # 후보 풀은 그 시점 유니버스다 — 라이브는 그날 받아 온 종목만 본다.
        # bars 전체를 쓰면 나중에 상위로 올라온 코인이 과거에 섞여 생존편향이 된다.
        # (선물 엔진과 같은 기준. 2026-10-18)
        for coin in get_mcap(sig_date):
            if coin == 'BTC' or coin in blacklist:
                continue
            m = _coin_mom2(coin, sig_date)
            if m is None:
                continue
            ms, ml = m
            if ms > 0 and ml > 0:
                healthy.append((coin, ms))
        healthy.sort(key=lambda x: (-x[1], x[0]))
        healthy_sorted = [c for c, _ in healthy]
        for si in range(len(snapshots)):
            snap = snapshots[si]
            sn_coins = sorted([c for c in snap if c.upper() != 'CASH'])
            new_sn = {}
            replaced = 0.0
            for c in sn_coins:
                m = _coin_mom2(c, sig_date)
                failed = (m is not None and m[0] < 0 and m[1] < 0)
                if failed:
                    replaced += float(snap.get(c, 0.0))
                else:
                    new_sn[c] = float(snap.get(c, 0.0))
            cash_w = sum(float(snap.get(k, 0.0)) for k in snap if k.upper() == 'CASH')
            new_sn['CASH'] = cash_w
            if replaced > 0:
                fresh = [c for c in healthy_sorted if c not in new_sn]
                n_failed = len(sn_coins) - sum(1 for k in new_sn if k.upper() != 'CASH')
                n_failed = max(1, n_failed)
                if fresh:
                    picks = fresh[:n_failed]
                    w_per = replaced / len(picks)
                    for c in picks:
                        new_sn[c] = new_sn.get(c, 0.0) + w_per
                else:
                    new_sn['CASH'] = new_sn.get('CASH', 0.0) + replaced
            snapshots[si] = new_sn

    def _merge_snapshots(sig_date_for_health=None):
        combined = {}
        n = len(snapshots)
        for snap in snapshots:
            for t, w in snap.items():
                combined[t] = combined.get(t, 0) + w / n
        # daily health filter overlay (V23 실험)
        if daily_health_exit and sig_date_for_health is not None:
            for t in list(combined.keys()):
                if t in ('CASH', 'Cash'):
                    continue
                if not _is_healthy(t, sig_date_for_health):
                    combined['CASH'] = combined.get('CASH', 0) + combined[t]
                    combined[t] = 0
        total = sum(combined.values())
        if total > 0:
            return {t: w / total for t, w in combined.items() if w > 0}
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
            val = margins[coin] + pos_dir * holdings[coin] * (cur - entry_prices[coin])
            if val > 0:
                w[coin] = val / pv
        cash_w = capital / pv
        if cash_w > 0.001:
            w['CASH'] = cash_w
        return w

    # ═══ 숏 (short_design_spec.md 3·4·5·7절) ═══

    def _short_signal(sig_bar_idx):
        """숏 진입·청산 신호. 전부 t-1 완료봉으로 잰다.

        r = BTC종가 / SMA42, a = min(0.985, 1 - 이격문턱).
        진입은 r <= a 이고(기울기 문을 켜면 SMA42 기울기도 음수여야 한다),
        청산은 r >= a + 여유다. 기울기가 양수로 돌아선 것은 청산 신호가 아니다 —
        진입을 막는 문일 뿐이다(명세 3절).
        """
        c = btc_close[:sig_bar_idx + 1]
        sma_now = calc_sma(c, sma_period)
        if sma_now <= 0 or len(c) == 0:
            return False, False
        r = c[-1] / sma_now
        a = min(0.985, 1.0 - short_entry_discount)
        slope_ok = True
        if short_slope_gate:
            j = sig_bar_idx - short_slope_days
            if j < sma_period:
                slope_ok = False
            else:
                sma_past = calc_sma(btc_close[:j + 1], sma_period)
                slope_ok = sma_past > 0 and (sma_now / sma_past - 1.0) < 0
        return (r <= a and slope_ok), (r >= a + short_exit_margin)

    def _compute_short_weights(sig_date):
        """숏 후보 선정. 롱 헬스의 거울이다 — 두 모멘텀이 음수이고 변동성이 상한 이하.

        변동성 상한은 숏이라는 이유로 다시 안 연다(프로젝트 규칙: 0.05 가 로버스트 최적점,
        양방향 재검증 끝). 손절 쿨다운에 걸린 종목은 뺀다.
        총명목 노출은 short_gross_exposure_cap 으로 묶고, 종목별 상한은
        min(short_per_coin_cap, G_cap/N) 이다(명세 6절). 못 채운 몫은 현금으로 남긴다.
        """
        mcap_order = get_mcap(sig_date)
        min_bars = max(mom30, mom90, _vol_bars, sma_period)
        _excluded = exclude_assets or frozenset()
        cands = []
        for coin in mcap_order:
            if coin in blacklist or coin in _excluded:
                continue
            if short_cooldown.get(coin, 0) > 0:
                continue
            df = bars.get(coin)
            if df is None:
                continue
            ci = df.index.get_indexer([sig_date], method='ffill')[0]
            if ci < 0 or ci < min_bars:
                continue
            c = df['Close'].values[:ci + 1]
            m_short = calc_mom(c, mom30)
            m_long = calc_mom(c, mom90)
            if vol_mode == 'bar':
                vol = calc_vol_bars(c, _vol_bars, bars_per_year)
            else:
                vol = calc_vol_daily(c, bpd, lookback_bars=_vol_bars)
            if short_health_mode == 'inv_mom2vol':
                ok = m_short < 0 and m_long < 0 and vol <= vol_threshold
            elif short_health_mode == 'failed_rally':
                # 길게는 내리막인데 짧게 반등했다 다시 꺾인 것. 반등 실패형.
                ok = m_long < 0 and m_short < 0 and vol <= vol_threshold and c[-1] < float(np.mean(c[-sma_period:]))
            else:
                ok = m_short < 0 and m_long < 0
            if ok:
                cands.append((m_short, coin))
        if not cands:
            return {'CASH': 1.0}
        # 많이 떨어진 순. 동률은 심볼순이라 재현된다.
        cands.sort(key=lambda t: (t[0], t[1]))
        picks = [c for _, c in cands[:short_universe_size]]
        n = len(picks)
        per = min(short_cap, short_per_coin_cap,
                  short_gross_exposure_cap / max(n, 1))
        w = {c: per for c in picks}
        used = per * n
        if used < 1.0:
            w['CASH'] = 1.0 - used
        return w

    def _liquidate_all(date, reason=''):
        """전량 청산 후 현금. 방향 전환은 여기를 반드시 거친다(명세 7절).

        라이브는 전량 청산 → 미체결 취소 → 잔량 0 확인 순인데, 백테스트에는 미체결이
        없으므로 잔량 0 확인이 곧 holdings 를 비우는 것이다.
        """
        nonlocal capital, trade_count
        for coin in list(holdings.keys()):
            cur = _get_fill_price(coin, date)
            if cur <= 0:
                ci = bars[coin].index.get_indexer([date], method='ffill')[0] if coin in bars else -1
                cur = get_close(bars, coin, ci)
            if cur <= 0:
                continue
            slip = _slip(coin)
            exit_p = cur * (1 - pos_dir * slip)
            pnl = pos_dir * holdings[coin] * (exit_p - entry_prices[coin])
            capital += margins[coin] + pnl - holdings[coin] * exit_p * tx_cost
            trade_count += 1
        holdings.clear(); entry_prices.clear(); margins.clear(); entry_bar_index.clear()
        capital = max(capital, 0)

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

        # ── 청산/스탑 체크 ──
        # 롱은 저가로, 숏은 고가로 본다. 롱 경로의 `leverage > 1` 게이트는 그대로 둔다 —
        # 롱 1배는 liq_price 가 0 이라 무해하고, 건드리면 B − A 에 엔진과 무관한 차이가 섞인다.
        # 숏은 1배에서도 청산가가 진입가의 약 2배라 실재하므로 게이트를 안 건다.
        for coin in list(holdings.keys()):
            ci = btc_idx_map.get(date, -1) if coin == 'BTC' else bars[coin].index.get_indexer([date], method='ffill')[0] if coin in bars else -1
            short_side = pos_dir < 0
            touch = get_high(bars, coin, ci) if short_side else get_low(bars, coin, ci)
            if touch <= 0:
                continue

            if short_side:
                liq_price = _get_liq_price(coin) if not is_spot else None
                stop_price = _get_short_stop_price(coin, date)
                hit_liq = liq_price is not None and touch >= liq_price
                hit_stop = stop_price is not None and touch >= stop_price
                stop_first = hit_stop and (not hit_liq or stop_price < liq_price)
            else:
                liq_price = _get_liq_price(coin) if (not is_spot and leverage > 1) else None
                stop_price = _get_stop_price(coin, date) if stop_kind != 'none' and stop_pct > 0 else None
                hit_liq = liq_price is not None and touch <= liq_price
                hit_stop = stop_price is not None and touch <= stop_price
                stop_first = hit_stop and (not hit_liq or stop_price > liq_price)

            if stop_first:
                _execute_stop_exit(coin, date, stop_price)
                continue

            if hit_liq:
                pnl_at_touch = pos_dir * holdings[coin] * (touch - entry_prices[coin])
                eq = margins[coin] + pnl_at_touch
                liq_fee = max(eq, 0) * 0.015
                returned = max(eq - liq_fee, 0)
                capital += returned
                del holdings[coin]; del entry_prices[coin]; del margins[coin]
                entry_bar_index.pop(coin, None)
                if short_side:
                    short_cooldown[coin] = short_cooldown_bars
                liq_count += 1

        # ── 펀딩비 (fut only) ──
        for coin in list(holdings.keys()):
            if is_spot:
                break
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
                        capital -= pos_dir * holdings[coin] * cur * fr
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

        # ── 숏 상태기계 (명세 7절) ──
        # 롱이거나 현금이거나 숏이다. 방향을 바꿀 때는 전량 청산하고 최소 한 봉을
        # 현금으로 보낸 뒤에만 반대로 들어간다. enable_short=False 면 이 블록은
        # sleeve_state 만 따라 적고 아무것도 안 바꾼다(= 구성 B 가 A 와 같은 이유).
        short_mode = False
        if enable_short:
            for _c in list(short_cooldown.keys()):
                short_cooldown[_c] -= 1
                if short_cooldown[_c] <= 0:
                    del short_cooldown[_c]

            s_entry, s_exit = _short_signal(btc_i_prev)
            if canary_on:
                desired = 'LONG'
            elif sleeve_state == 'SHORT':
                desired = 'CASH' if s_exit else 'SHORT'
            else:
                desired = 'SHORT' if s_entry else 'CASH'

            if desired != sleeve_state:
                if holdings:
                    _liquidate_all(date)
                    rebal_count += 1
                pos_dir = 1
                for si in range(n_snapshots):
                    snapshots[si] = {'CASH': 1.0}
                if sleeve_state != 'CASH':
                    # 방향에서 나왔으면 현금을 한 봉 이상 거친다. 현금에서 방향으로
                    # 가는 길은 유예가 없다 — 유예는 뒤집을 때만 건다.
                    cash_until_bar = bar_i + cash_transit_bars
                    sleeve_state = 'CASH'
                elif bar_i > cash_until_bar:
                    sleeve_state = desired
                    if desired == 'SHORT':
                        pos_dir = -1
                        short_entry_bar = bar_i

            if sleeve_state == 'SHORT':
                pos_dir = -1
                short_mode = True
            else:
                pos_dir = 1
            # 숏이 서 있으면 롱 경로를 통째로 끈다. 두 방향이 같은 봉에 서지 않는다.
            if short_mode:
                canary_on = False
                canary_flipped = False

        # ── 숏 대재앙 손절 (명세 8절) ──
        # 종목별 손절(+12%)과 별개로 슬리브를 통째로 닫는 자리다. 한 종목이라도
        # 진입가 대비 이만큼 오르면 그 봉에 전부 닫고 쿨다운 동안 숏을 쉰다.
        if short_mode and holdings and short_catastrophic_stop_pct > 0:
            worst = 0.0
            for coin in list(holdings.keys()):
                ci = bars[coin].index.get_indexer([date], method='ffill')[0] if coin in bars else -1
                hi = get_high(bars, coin, ci)
                ep = entry_prices.get(coin, 0)
                if hi > 0 and ep > 0:
                    worst = max(worst, hi / ep - 1.0)
            if worst >= short_catastrophic_stop_pct:
                hit = list(holdings.keys())
                _liquidate_all(date)
                for coin in hit:
                    short_cooldown[coin] = short_catastrophic_cooldown_bars
                sleeve_state = 'CASH'
                pos_dir = 1
                short_mode = False
                cash_until_bar = bar_i + max(cash_transit_bars, short_catastrophic_cooldown_bars)
                for si in range(n_snapshots):
                    snapshots[si] = {'CASH': 1.0}
                rebal_count += 1

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
                        pnl = pos_dir * holdings[coin] * (cur - entry_prices[coin])
                        capital += margins[coin] + pnl - holdings[coin] * cur * tx_cost
                holdings.clear(); entry_prices.clear(); margins.clear(); entry_bar_index.clear()
                for si in range(n_snapshots):
                    snapshots[si] = {'CASH': 1.0}
                crash_cooldown = crash_cool_bars
                rebal_count += 1

        # ── DD Exit (daily_gate 시 일 1회만; detect=t-1 close, fill=t open) ──
        if dd_lookback > 0 and canary_on and holdings and not canary_flipped and is_daily_bar:
            for coin in list(holdings.keys()):
                df = bars.get(coin)
                if df is None:
                    continue
                ci_p = df.index.get_indexer([prev_date], method='ffill')[0]
                if ci_p < dd_bars:
                    continue
                c = df['Close'].values
                peak = np.max(c[ci_p - dd_bars:ci_p])
                if peak > 0 and (c[ci_p] / peak - 1) <= dd_threshold:
                    cur = _get_fill_price(coin, date)  # t open 체결
                    if cur <= 0:
                        continue
                    slip = _slip(coin)
                    pnl = pos_dir * holdings[coin] * (cur * (1 - pos_dir * slip) - entry_prices[coin])
                    capital += margins[coin] + pnl - holdings[coin] * cur * tx_cost
                    del holdings[coin]; del entry_prices[coin]; del margins[coin]
                    entry_bar_index.pop(coin, None)
                    trade_count += 1

        # ── Blacklist (detect=t-1 close, fill=t open) ──
        if bl_drop < 0 and canary_on and holdings and is_daily_bar:
            for coin in list(holdings.keys()):
                df = bars.get(coin)
                if df is None:
                    continue
                ci_p = df.index.get_indexer([prev_date], method='ffill')[0]
                if ci_p < _bl_lookback:
                    continue
                c = df['Close'].values
                bar_ret = c[ci_p] / c[ci_p - _bl_lookback] - 1
                if bar_ret <= bl_drop:
                    cur = _get_fill_price(coin, date)  # t open 체결
                    if cur <= 0:
                        continue
                    slip = _slip(coin)
                    pnl = pos_dir * holdings[coin] * (cur * (1 - pos_dir * slip) - entry_prices[coin])
                    capital += margins[coin] + pnl - holdings[coin] * cur * tx_cost
                    del holdings[coin]; del entry_prices[coin]; del margins[coin]
                    entry_bar_index.pop(coin, None)
                    blacklist[coin] = bl_cooldown
                    trade_count += 1

        # ── 시그널 → 스냅샷 갱신 ──
        need_rebal = False

        if short_mode:
            # 숏은 앵커에서만 수량을 다시 잡는다(short_rebalance_mode='anchor_fixed_qty').
            # 드리프트로 따라가지 않는다 — 가격이 오르면 저절로 노출이 커지는 쪽이라
            # 그걸 매봉 되맞추면 지는 자리에서 계속 더 파는 동작이 된다(명세 2절 넷째).
            fresh_bar = (short_entry_bar == bar_i)
            anchor_bar = (short_rebalance_mode == 'target_weight'
                          or (snap_interval_bars > 0
                              and (bar_i + phase_offset_bars) % snap_interval_bars == 0))
            if fresh_bar or anchor_bar:
                sw = _compute_short_weights(prev_date)
                if fresh_bar or sw != snapshots[0]:
                    for si in range(n_snapshots):
                        snapshots[si] = dict(sw)
                    need_rebal = True
        elif bar_i <= sma_period + 1:
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
        # ANCHOR_TRADE_MODE=defer 시 snap 갱신만 하고 need_rebal 안 켬 → drift 만 trade trigger
        import os as _osa
        _anchor_defer = _osa.environ.get('ANCHOR_TRADE_MODE', 'on') == 'defer'
        if canary_on and not canary_flipped:
            if snap_interval_bars > 0:
                # 봉 기반 앵커: bar_i % interval로 트리거
                for si in range(n_snapshots):
                    offset = int(si * snap_interval_bars / n_snapshots)
                    if (bar_i + phase_offset_bars) % snap_interval_bars == offset:
                        new_w = _compute_weights(prev_date)
                        if new_w != snapshots[si]:
                            snapshots[si] = new_w
                            if not _anchor_defer:
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
                            if not _anchor_defer:
                                need_rebal = True

        # ── Drift (daily_gate 시 일 1회만) ──
        combined = _merge_snapshots(sig_date_for_health=prev_date)
        if not need_rebal and canary_on and drift_threshold > 0 and holdings and is_daily_bar:
            if _half_turnover(_current_weights(prev_date), combined) >= drift_threshold:
                need_rebal = True
                import os as _os
                # 라이브 정합 refill_v2: drift 발화 시 mom2 음수 슬롯 → fresh healthy 교체
                # (coin_live_engine._apply_refill_v2_to_state 와 동일). DRIFT_HEALTH_MODE='refill'.
                _dhm = _os.environ.get('DRIFT_HEALTH_MODE', 'off')
                if _dhm == 'refill':
                    _refill_v2_snapshots(prev_date)
                    combined = _merge_snapshots(sig_date_for_health=prev_date)
                # 주식식 전체 재선정: drift 발화 시 전 snapshot 을 fresh full selection 으로 교체
                elif _dhm == 'reselect':
                    _fresh = _compute_weights(prev_date)
                    for si in range(len(snapshots)):
                        snapshots[si] = dict(_fresh)
                    combined = _merge_snapshots(sig_date_for_health=prev_date)
                # V25 옵션 B: drift 발화 시 CASH 슬롯 refill (보유 종목 유지)
                elif _os.environ.get('DRIFT_CASH_REFILL', 'off') == 'on':
                    _fresh_full = _compute_weights(prev_date)
                    _fresh_pool = sorted([c for c in _fresh_full.keys() if c not in ('CASH', 'Cash')])
                    if _fresh_pool:
                        for si in range(n_snapshots):
                            sn = snapshots[si]
                            cash_w = sn.get('CASH', 0) + sn.get('Cash', 0)
                            if cash_w <= 0.001:
                                continue
                            already = set(c for c in sn.keys() if c not in ('CASH', 'Cash'))
                            candidates = [c for c in _fresh_pool if c not in already]
                            if not candidates:
                                continue
                            n_slots = max(1, int(round(cash_w / cap)))
                            picks = candidates[:n_slots]
                            w_per = cash_w / len(picks)
                            new_sn = {k: v for k, v in sn.items() if k not in ('CASH', 'Cash')}
                            for c in picks:
                                new_sn[c] = new_sn.get(c, 0) + w_per
                            snapshots[si] = new_sn
                        combined = _merge_snapshots(sig_date_for_health=prev_date)

        if crash_cooldown > 0:
            need_rebal = False

        # ── trace 기록 ──
        if _trace is not None:
            _trace.append({
                'date': date,
                'target': dict(combined),
                'snapshots': [dict(s) for s in snapshots],
                'rebal': need_rebal,
                'stop_kind': stop_kind,
                'stop_pct': stop_pct,
            })

        # ── 리밸런싱 실행 (execution_delay_bars 지원) ──
        # pending queue: 이전에 예약된 리밸런스가 있으면 due bar 에 실행
        if pending_rebalance is not None and bar_i >= pending_rebalance['due_i']:
            _execute_rebalance(pending_rebalance['target'], date, pv_date=prev_date)
            rebal_count += 1
            pending_rebalance = None
        if need_rebal:
            if execution_delay_bars <= 0:
                _execute_rebalance(combined, date, pv_date=prev_date)
                rebal_count += 1
            else:
                # N bar 뒤에 현재 combined target 체결 예약
                pending_rebalance = {'due_i': bar_i + int(execution_delay_bars),
                                      'target': dict(combined)}

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
