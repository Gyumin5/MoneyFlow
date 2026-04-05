#!/usr/bin/env python3
"""선물 백테스트 엔진 v2 — 멀티코인, 시간봉 기반.

v2 수정 (Codex 리뷰 반영):
  1. Look-ahead 수정: t-1 시그널, t 체결 (현물 엔진과 동일)
  2. Universe buffer: 진입 Top N, 퇴출 Top N+5
  3. 월별 시총 순위 변동 반영 (historical_universe.json)
  4. DD Exit (-25%/60d) + Blacklist (-15%/7d) 이식
  5. 펀딩비 올바른 집계 (resample.sum)

Features:
  - 코인별 격리마진 / 수량 고정
  - 실제 바이낸스 펀딩레이트
  - Low/High 장중 청산
  - 시총 기반 슬리피지 차등
  - 1d/4h/1h/15m 지원
"""

import numpy as np, pandas as pd, os, sys, json

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'futures')
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

TICKER_MAP = {
    'BTC-USD': 'BTCUSDT', 'ETH-USD': 'ETHUSDT', 'SOL-USD': 'SOLUSDT',
    'BNB-USD': 'BNBUSDT', 'XRP-USD': 'XRPUSDT', 'DOGE-USD': 'DOGEUSDT',
    'ADA-USD': 'ADAUSDT', 'AVAX-USD': 'AVAXUSDT', 'TRX-USD': 'TRXUSDT',
    'LINK-USD': 'LINKUSDT',
}
REVERSE_MAP = {v: k for k, v in TICKER_MAP.items()}

SLIPPAGE = {
    'BTCUSDT': 0.0002, 'ETHUSDT': 0.0003,
    'SOLUSDT': 0.0004, 'BNBUSDT': 0.0004, 'XRPUSDT': 0.0004,
    'DOGEUSDT': 0.0005, 'ADAUSDT': 0.0005, 'AVAXUSDT': 0.0006,
    'TRXUSDT': 0.0005, 'LINKUSDT': 0.0006,
}

# ─── 월별 시총 순위 로드 ───
def load_monthly_mcap_order():
    """historical_universe.json → 월별 선물 심볼 시총 순위."""
    paths = [
        os.path.join(BASE_DIR, '..', '..', 'backup_20260125', 'historical_universe.json'),
        os.path.join(BASE_DIR, '..', '..', 'data', 'historical_universe.json'),
    ]
    for p in paths:
        if os.path.exists(p):
            with open(p) as f:
                raw = json.load(f)
            result = {}
            for date_str, tickers in raw.items():
                syms = []
                for t in tickers:
                    fut_sym = TICKER_MAP.get(t)
                    if fut_sym:
                        syms.append(fut_sym)
                if syms:
                    result[date_str] = syms
            return result
    return {}

_MCAP_MONTHLY = None
def get_mcap_order(date):
    """해당 월의 시총 순위 반환."""
    global _MCAP_MONTHLY
    if _MCAP_MONTHLY is None:
        _MCAP_MONTHLY = load_monthly_mcap_order()
    month_key = date.strftime('%Y-%m') + '-01'
    if month_key in _MCAP_MONTHLY:
        return _MCAP_MONTHLY[month_key]
    # fallback: 가장 가까운 이전 월
    keys = sorted(k for k in _MCAP_MONTHLY if k <= month_key)
    return _MCAP_MONTHLY[keys[-1]] if keys else list(TICKER_MAP.values())


def load_futures_data(symbols, interval='1h'):
    """바이낸스 선물 OHLCV + 펀딩레이트."""
    data = {}
    for sym in symbols:
        fpath = os.path.join(DATA_DIR, f'{sym}_{interval}.csv')
        if not os.path.exists(fpath):
            fpath = os.path.join(DATA_DIR, f'{sym}_1h.csv')
        if not os.path.exists(fpath):
            continue
        df = pd.read_csv(fpath, parse_dates=['Date'], index_col='Date')
        data[sym] = {'ohlcv': df}
        fpath_fund = os.path.join(DATA_DIR, f'{sym}_funding.csv')
        if os.path.exists(fpath_fund):
            data[sym]['funding'] = pd.read_csv(fpath_fund, parse_dates=['Date'], index_col='Date')['fundingRate']
        else:
            data[sym]['funding'] = pd.Series(dtype=float)
    return data


def resample_ohlcv(df, target):
    """OHLCV를 target interval로 리샘플."""
    if target in ('1h', '15m'):
        return df
    return df.resample(target).agg({
        'Open': 'first', 'High': 'max', 'Low': 'min',
        'Close': 'last', 'Volume': 'sum'
    }).dropna()


def resample_funding(fund_series, target_index, interval):
    """펀딩레이트를 바 주기에 맞게 집계."""
    if interval == '15m':
        # 15m: 8시간마다 발생, 해당 15분 바에만 할당
        return fund_series.reindex(target_index).fillna(0)
    elif interval == '1h':
        return fund_series.reindex(target_index).fillna(0)
    elif interval == '4h':
        # 4h: 8시간마다 → 2개 4h 바에 걸칠 수 있음 → sum
        return fund_series.resample('4h').sum().reindex(target_index).fillna(0)
    else:  # D
        return fund_series.resample('D').sum().reindex(target_index).fillna(0)


class Position:
    """격리마진 선물 포지션."""
    __slots__ = ['symbol', 'qty', 'entry_price', 'margin', 'leverage', 'direction']

    def __init__(self, symbol, qty, entry_price, margin, leverage, direction=1):
        self.symbol = symbol
        self.qty = qty
        self.entry_price = entry_price
        self.margin = margin
        self.leverage = leverage
        self.direction = direction

    def unrealized_pnl(self, cur_price):
        return self.qty * (cur_price - self.entry_price) * self.direction

    def equity(self, cur_price):
        return self.margin + self.unrealized_pnl(cur_price)

    def is_liquidated(self, low_price, high_price, maint_rate=0.004):
        if self.direction == 1:
            eq = self.margin + self.qty * (low_price - self.entry_price)
            maint = self.qty * low_price * maint_rate
        else:
            eq = self.margin + self.qty * (self.entry_price - high_price)
            maint = self.qty * high_price * maint_rate
        return eq <= maint


def run_futures_backtest(
    futures_data, interval='1h', leverage=1.0,
    universe_size=5, universe_buffer=5,
    selection='greedy', cap=1/3,
    tx_cost=0.0004, maint_margin_rate=0.004,
    initial_capital=10000.0,
    start_date='2020-10-01', end_date='2026-03-28',
    dd_lookback=60, dd_threshold=-0.25,
    bl_drop=-0.15, bl_days=7,
):
    """멀티코인 선물 백테스트 v2."""

    btc_raw = futures_data.get('BTCUSDT', {}).get('ohlcv')
    if btc_raw is None:
        return {'Sharpe': 0, 'CAGR': 0, 'MDD': 0, 'Cal': 0, 'Liq': 0, 'Trades': 0}

    bars_per_day = {'D': 1, '4h': 6, '1h': 24, '15m': 96}[interval]
    sma_bars = 50 * bars_per_day

    # 리샘플
    coin_bars = {}
    coin_funding = {}
    for sym, d in futures_data.items():
        cb = resample_ohlcv(d['ohlcv'], interval)
        coin_bars[sym] = cb
        coin_funding[sym] = resample_funding(d['funding'], cb.index, interval)

    btc_bars = coin_bars['BTCUSDT']
    dates = btc_bars.index[(btc_bars.index >= start_date) & (btc_bars.index <= end_date)]
    btc_close = btc_bars['Close'].values

    capital = initial_capital
    positions = {}
    prev_canary = False
    held_coins = set()  # 유니버스 버퍼용
    blacklist = {}  # {sym: remaining_bars}
    _snap_done = {}  # 앵커 중복 방지
    pv_list = []
    liq_count = 0
    trade_count = 0

    mom_bars_30 = 30 * bars_per_day
    mom_bars_90 = 90 * bars_per_day
    vol_bars_90 = 90 * bars_per_day
    dd_bars = dd_lookback * bars_per_day

    for idx in range(sma_bars + 1, len(dates)):  # +1 for t-1 signal
        date = dates[idx]
        prev_date = dates[idx - 1]  # ★ FIX 1: t-1 시그널용

        btc_i = btc_bars.index.get_loc(date) if date in btc_bars.index else -1
        btc_i_prev = btc_bars.index.get_loc(prev_date) if prev_date in btc_bars.index else -1
        if btc_i < sma_bars or btc_i_prev < sma_bars:
            continue

        cur_btc = btc_close[btc_i]
        prev_btc = btc_close[btc_i_prev]
        if np.isnan(cur_btc) or cur_btc <= 0:
            pv = capital + sum(p.equity(cur_btc) for p in positions.values())
            pv_list.append({'Date': date, 'Value': max(pv, 0)})
            continue

        # ★ FIX 1: 카나리는 t-1(prev) 기준, 체결은 t(cur) 기준
        sma_val = np.mean(btc_close[btc_i_prev - sma_bars:btc_i_prev])
        if prev_canary:
            canary_on = not (prev_btc < sma_val * 0.985)
        else:
            canary_on = prev_btc > sma_val * 1.015

        # ── 청산 체크 ──
        for sym in list(positions.keys()):
            pos = positions[sym]
            cb = coin_bars.get(sym)
            if cb is None or date not in cb.index:
                continue
            ci = cb.index.get_loc(date)
            low_p = float(cb['Low'].iloc[ci])
            high_p = float(cb['High'].iloc[ci])
            if pos.is_liquidated(low_p, high_p, maint_margin_rate):
                liq_fee = max(pos.margin * 0.015, 0)
                capital = max(capital - liq_fee, 0)
                held_coins.discard(sym)
                del positions[sym]
                liq_count += 1

        # ── 펀딩비 ──
        for sym, pos in list(positions.items()):
            fr_s = coin_funding.get(sym)
            if fr_s is None or date not in fr_s.index:
                continue
            fr = float(fr_s.loc[date])
            if fr != 0 and not np.isnan(fr):
                cb = coin_bars[sym]
                if date in cb.index:
                    cur_p = float(cb['Close'].loc[date])
                    capital -= pos.qty * cur_p * fr * pos.direction

        # ── Blacklist 감소 ──
        for sym in list(blacklist.keys()):
            blacklist[sym] -= 1
            if blacklist[sym] <= 0:
                del blacklist[sym]

        # ★ FIX 4: DD Exit — 보유 중 60d peak 대비 -25%
        if canary_on and positions and dd_lookback > 0:
            for sym in list(positions.keys()):
                cb = coin_bars.get(sym)
                if cb is None or date not in cb.index:
                    continue
                ci = cb.index.get_loc(date)
                if ci < dd_bars:
                    continue
                c = cb['Close'].values
                peak = np.max(c[ci - dd_bars:ci])
                cur_p = c[ci]
                if peak > 0 and (cur_p / peak - 1) <= dd_threshold:
                    # DD Exit: 매도
                    pos = positions[sym]
                    slip = SLIPPAGE.get(sym, 0.0005)
                    exit_p = cur_p * (1 - slip)
                    pnl = pos.unrealized_pnl(exit_p)
                    tx = pos.qty * cur_p * tx_cost
                    capital += pos.margin + pnl - tx
                    held_coins.discard(sym)
                    del positions[sym]
                    trade_count += 1

        # ★ FIX 4: Blacklist — 보유 중 일간 -15% → 매도 + 7일 제외
        if canary_on and positions and bl_drop < 0:
            for sym in list(positions.keys()):
                cb = coin_bars.get(sym)
                if cb is None or date not in cb.index:
                    continue
                ci = cb.index.get_loc(date)
                if ci < bars_per_day:
                    continue
                c = cb['Close'].values
                daily_ret = c[ci] / c[ci - bars_per_day] - 1
                if daily_ret <= bl_drop:
                    pos = positions[sym]
                    slip = SLIPPAGE.get(sym, 0.0005)
                    exit_p = c[ci] * (1 - slip)
                    pnl = pos.unrealized_pnl(exit_p)
                    tx = pos.qty * c[ci] * tx_cost
                    capital += pos.margin + pnl - tx
                    held_coins.discard(sym)
                    del positions[sym]
                    blacklist[sym] = bl_days * bars_per_day
                    trade_count += 1

        # ── 앵커 체크 (3-snapshot 대신 단순 앵커일 리밸런싱) ──
        snap_days = [1, 10, 19]  # 코인 앵커일
        anchor_rebal = False
        if canary_on and not (canary_on != prev_canary):
            # 일봉: date.day, 시간봉: date에서 일 추출
            day_of_month = date.day if hasattr(date, 'day') else 1
            cur_month_str = date.strftime('%Y-%m')
            for anchor in snap_days:
                anchor_key = f"{cur_month_str}_a{anchor}"
                if day_of_month >= anchor and anchor_key not in _snap_done:
                    _snap_done[anchor_key] = True
                    anchor_rebal = True
                    break

        # ── 카나리 플립 또는 앵커 → 리밸런싱 ──
        flipped = canary_on != prev_canary
        need_rebal = flipped or anchor_rebal

        if need_rebal:
            # 전량 청산
            for sym in list(positions.keys()):
                pos = positions[sym]
                cb = coin_bars.get(sym)
                if cb is None or date not in cb.index:
                    continue
                cur_p = float(cb['Close'].loc[date])
                slip = SLIPPAGE.get(sym, 0.0005)
                exit_p = cur_p * (1 - slip * pos.direction)
                pnl = pos.unrealized_pnl(exit_p)
                tx = pos.qty * cur_p * tx_cost
                capital += pos.margin + pnl - tx
                trade_count += 1
            positions.clear()
            held_coins.clear()

            if canary_on and capital > 100:
                # ★ FIX 3: 월별 시총 순위
                mcap_order = get_mcap_order(date)

                # 헬스체크 (t-1 기준)
                healthy = []
                for sym in mcap_order:
                    if sym in blacklist:
                        continue
                    cb = coin_bars.get(sym)
                    if cb is None or prev_date not in cb.index:
                        continue
                    ci = cb.index.get_loc(prev_date)
                    if ci < max(mom_bars_90, vol_bars_90):
                        continue
                    c = cb['Close'].values
                    mom30 = c[ci] / c[ci - mom_bars_30] - 1
                    mom90 = c[ci] / c[ci - mom_bars_90] - 1
                    daily_step = max(bars_per_day, 1)
                    daily_closes = c[ci - vol_bars_90:ci + 1:daily_step]
                    vol90 = np.std(np.diff(np.log(daily_closes))) if len(daily_closes) > 10 else 999
                    if mom30 > 0 and mom90 > 0 and vol90 <= 0.05:
                        healthy.append(sym)

                # ★ FIX 2: Universe buffer — 보유 중이면 N+buffer까지 유지
                picks = []
                for sym in healthy:
                    rank = mcap_order.index(sym) if sym in mcap_order else 999
                    if rank < universe_size:
                        picks.append(sym)
                    elif sym in held_coins and rank < universe_size + universe_buffer:
                        picks.append(sym)
                picks = picks[:universe_size]  # 최대 N개

                # 그리디 흡수 (t-1 기준)
                if selection == 'greedy' and len(picks) > 1:
                    for i in range(len(picks) - 1, 0, -1):
                        cb_a = coin_bars.get(picks[i - 1])
                        cb_b = coin_bars.get(picks[i])
                        if cb_a is None or cb_b is None:
                            continue
                        if prev_date not in cb_a.index or prev_date not in cb_b.index:
                            continue
                        ci_a = cb_a.index.get_loc(prev_date)
                        ci_b = cb_b.index.get_loc(prev_date)
                        if ci_a < mom_bars_30 or ci_b < mom_bars_30:
                            continue
                        ma = cb_a['Close'].values[ci_a] / cb_a['Close'].values[ci_a - mom_bars_30] - 1
                        mb = cb_b['Close'].values[ci_b] / cb_b['Close'].values[ci_b - mom_bars_30] - 1
                        if ma >= mb:
                            picks.pop(i)

                # 비중 + 진입 (t 가격)
                if picks:
                    w = min(1.0 / len(picks), cap)
                    margin_per = capital * 0.95 * w
                    for sym in picks:
                        cb = coin_bars[sym]
                        if date not in cb.index:
                            continue
                        cur_p = float(cb['Close'].loc[date])
                        slip = SLIPPAGE.get(sym, 0.0005)
                        entry_p = cur_p * (1 + slip)
                        notional = margin_per * leverage
                        qty = notional / entry_p
                        entry_tx = notional * tx_cost
                        if capital >= margin_per + entry_tx:
                            capital -= (margin_per + entry_tx)
                            positions[sym] = Position(sym, qty, entry_p, margin_per, leverage)
                            held_coins.add(sym)
                            trade_count += 1

        # ── 포트폴리오 가치 ──
        pv = capital
        for sym, pos in positions.items():
            cb = coin_bars.get(sym)
            if cb is not None and date in cb.index:
                pv += pos.equity(float(cb['Close'].loc[date]))
        pv_list.append({'Date': date, 'Value': max(pv, 0)})
        prev_canary = canary_on

    # 미청산 정리
    for sym, pos in positions.items():
        cb = coin_bars.get(sym)
        if cb is not None and len(cb) > 0:
            cur_p = float(cb['Close'].iloc[-1])
            tx = pos.qty * cur_p * tx_cost
            capital += pos.margin + pos.unrealized_pnl(cur_p) - tx

    if not pv_list:
        return {'Sharpe': 0, 'CAGR': 0, 'MDD': 0, 'Cal': 0, 'Liq': 0, 'Trades': 0}

    pvdf = pd.DataFrame(pv_list).set_index('Date')
    eq = pvdf['Value']
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if eq.iloc[-1] <= 0 or yrs <= 0:
        return {'Sharpe': 0, 'CAGR': -1, 'MDD': -1, 'Cal': 0, 'Liq': liq_count, 'Trades': trade_count}

    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return {'Sharpe': sh, 'CAGR': cagr, 'MDD': mdd, 'Cal': cal, 'Liq': liq_count, 'Trades': trade_count}


if __name__ == '__main__':
    import time
    t0 = time.time()

    SYMBOLS = ['BTCUSDT', 'ETHUSDT', 'SOLUSDT', 'BNBUSDT', 'XRPUSDT',
               'DOGEUSDT', 'ADAUSDT', 'AVAXUSDT', 'TRXUSDT', 'LINKUSDT']

    for interval in ['D', '4h', '1h', '15m']:
        src = '1h' if interval in ('D', '4h') else interval
        data = load_futures_data(SYMBOLS, src)
        print(f"\n{'='*65}")
        print(f"  {interval} | v2 (LA fix, DD/BL, monthly mcap, fund fix)")
        print(f"{'='*65}")
        print(f"  {'Config':<28s} {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>5s} {'Liq':>3s} {'Tr':>4s}")
        print(f"  {'-'*58}")

        for univ, sel in [(5, 'greedy'), (5, 'mcap'), (10, 'greedy'), (10, 'mcap')]:
            for lev in [1.0, 1.5, 2.0]:
                name = f"Top{univ} {'G' if sel == 'greedy' else 'M'} {lev}x"
                m = run_futures_backtest(
                    data, interval=interval, leverage=lev,
                    universe_size=univ, selection=sel,
                    start_date='2020-10-01')
                liq = f"💀{m['Liq']}" if m['Liq'] > 0 else ""
                print(f"  {name:<28s} {m['Sharpe']:>5.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>5.2f} {liq:>5s} {m['Trades']:>4d}")
        sys.stdout.flush()

    print(f"\n총 소요: {time.time() - t0:.0f}s")
