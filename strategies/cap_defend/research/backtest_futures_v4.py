#!/usr/bin/env python3
"""선물 백테스트 v4 — 일봉 시그널 + 시간봉 체결/청산/펀딩.

핵심 아이디어:
- 시그널(카나리/헬스/선정/비중): 일봉 V18 trace (100% 동일)
- 체결: 시그널 날의 첫 1h/4h 바 가격으로 체결
- 청산: 매 시간봉의 Low/High로 격리마진 체크
- 펀딩: 실제 8시간 단위 적용
- 슬리피지: 시간봉 기반으로 더 현실적

이러면 전략 로직 동일 + 시간봉 정밀도.
"""

import numpy as np, pandas as pd, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coin_engine import load_universe, load_all_prices, filter_universe, get_price, calc_metrics
from coin_helpers import B
from backtest_official import run_coin_backtest

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'futures')

TICKER_MAP = {
    'BTC-USD': 'BTCUSDT', 'ETH-USD': 'ETHUSDT', 'SOL-USD': 'SOLUSDT',
    'BNB-USD': 'BNBUSDT', 'XRP-USD': 'XRPUSDT', 'DOGE-USD': 'DOGEUSDT',
    'ADA-USD': 'ADAUSDT', 'AVAX-USD': 'AVAXUSDT', 'TRX-USD': 'TRXUSDT',
    'LINK-USD': 'LINKUSDT',
}

SLIPPAGE = {
    'BTCUSDT': 0.0002, 'ETHUSDT': 0.0003, 'SOLUSDT': 0.0004,
    'BNBUSDT': 0.0004, 'XRPUSDT': 0.0004, 'DOGEUSDT': 0.0005,
    'ADAUSDT': 0.0005, 'AVAXUSDT': 0.0006, 'TRXUSDT': 0.0005,
    'LINKUSDT': 0.0006,
}


def load_hourly_data(interval='1h'):
    """바이낸스 시간봉 + 펀딩."""
    data = {}
    for yahoo_t, binance_s in TICKER_MAP.items():
        fpath = os.path.join(DATA_DIR, f'{binance_s}_{interval}.csv')
        if not os.path.exists(fpath):
            fpath = os.path.join(DATA_DIR, f'{binance_s}_1h.csv')
        if os.path.exists(fpath):
            df = pd.read_csv(fpath, parse_dates=['Date'], index_col='Date')
            data[yahoo_t] = df  # Yahoo ticker key로 저장

        fpath_f = os.path.join(DATA_DIR, f'{binance_s}_funding.csv')
        if os.path.exists(fpath_f):
            fund = pd.read_csv(fpath_f, parse_dates=['Date'], index_col='Date')['fundingRate']
            data[f'{yahoo_t}_funding'] = fund
    return data


class HourlyFuturesReplay:
    """일봉 trace + 시간봉 체결/청산/펀딩."""

    def __init__(self, hourly_data, interval='1h', leverage=1.0,
                 tx_cost=0.0004, maint_rate=0.004):
        self.hourly = hourly_data
        self.interval = interval
        self.leverage = leverage
        self.tx = tx_cost
        self.maint_rate = maint_rate

    def _get_bars_for_date(self, ticker, date):
        """해당 날짜의 시간봉들 반환."""
        key = ticker  # Yahoo format
        df = self.hourly.get(key)
        if df is None:
            return None
        date_str = date.strftime('%Y-%m-%d')
        mask = df.index.strftime('%Y-%m-%d') == date_str
        return df[mask] if mask.any() else None

    def _get_funding_for_date(self, ticker, date):
        """해당 날짜의 펀딩레이트 합계."""
        fund = self.hourly.get(f'{ticker}_funding')
        if fund is None:
            return 0
        date_str = date.strftime('%Y-%m-%d')
        mask = fund.index.strftime('%Y-%m-%d') == date_str
        day_fund = fund[mask]
        return day_fund.sum() if len(day_fund) > 0 else 0

    def run(self, trace, spot_prices, initial_capital=10000.0):
        """trace → 시간봉 정밀 replay."""
        capital = initial_capital
        positions = {}  # {ticker: {qty, entry, margin}}
        pv_list = []
        liq_count = 0
        trade_count = 0

        for day in trace:
            date = day['date']
            target = day['target']
            need_rebal = day['rebal']

            # ── 1. 시간봉별 청산 체크 ──
            if positions and self.leverage > 1:
                for t in list(positions.keys()):
                    pos = positions[t]
                    bars = self._get_bars_for_date(t, date)
                    if bars is None:
                        continue
                    for _, bar in bars.iterrows():
                        low_p = float(bar['Low'])
                        eq = pos['margin'] + pos['qty'] * (low_p - pos['entry'])
                        maint = pos['qty'] * low_p * self.maint_rate
                        if eq <= maint:
                            liq_fee = max(eq, 0) * 0.015
                            capital = max(capital - liq_fee, 0)
                            del positions[t]
                            liq_count += 1
                            break

            # ── 2. 펀딩비 (실제 일간 합계) ──
            if positions:
                for t, pos in list(positions.items()):
                    fr = self._get_funding_for_date(t, date)
                    if fr != 0:
                        cur_p = get_price(t, spot_prices, date)
                        if cur_p > 0:
                            capital -= pos['qty'] * cur_p * fr
                capital = max(capital, 0)

            # ── 3. 리밸런싱 (Delta) ──
            if need_rebal:
                pv = capital
                for t, pos in positions.items():
                    cur_p = get_price(t, spot_prices, date)
                    if cur_p > 0:
                        pv += pos['margin'] + pos['qty'] * (cur_p - pos['entry'])

                if pv <= 100:
                    for t in list(positions.keys()):
                        pos = positions[t]
                        cur_p = get_price(t, spot_prices, date)
                        if cur_p > 0:
                            capital += pos['margin'] + pos['qty'] * (cur_p - pos['entry']) - pos['qty'] * cur_p * self.tx
                    positions.clear()
                else:
                    # Delta: target vs current
                    target_pos = {}
                    for t, w in target.items():
                        if t == 'CASH' or w <= 0:
                            continue
                        cur_p = get_price(t, spot_prices, date)
                        if cur_p <= 0:
                            continue
                        tgt_margin = pv * w * 0.95
                        tgt_notional = tgt_margin * self.leverage
                        tgt_qty = tgt_notional / cur_p
                        target_pos[t] = {'qty': tgt_qty, 'margin': tgt_margin}

                    # Sell excess
                    for t in list(positions.keys()):
                        pos = positions[t]
                        cur_p = get_price(t, spot_prices, date)
                        if cur_p <= 0:
                            continue
                        binance_s = TICKER_MAP.get(t, '')
                        slip = SLIPPAGE.get(binance_s, 0.0005)

                        if t not in target_pos:
                            exit_p = cur_p * (1 - slip)
                            capital += pos['margin'] + pos['qty'] * (exit_p - pos['entry']) - pos['qty'] * cur_p * self.tx
                            del positions[t]
                            trade_count += 1
                        else:
                            tgt = target_pos[t]
                            delta = tgt['qty'] - pos['qty']
                            if delta < -pos['qty'] * 0.05:
                                sell_qty = -delta
                                sell_margin = pos['margin'] * (sell_qty / pos['qty'])
                                exit_p = cur_p * (1 - slip)
                                capital += sell_margin + sell_qty * (exit_p - pos['entry']) - sell_qty * cur_p * self.tx
                                pos['qty'] -= sell_qty
                                pos['margin'] -= sell_margin
                                trade_count += 1

                    # Buy new/more
                    for t, tgt in target_pos.items():
                        cur_p = get_price(t, spot_prices, date)
                        if cur_p <= 0:
                            continue
                        binance_s = TICKER_MAP.get(t, '')
                        slip = SLIPPAGE.get(binance_s, 0.0005)

                        if t not in positions:
                            entry_p = cur_p * (1 + slip)
                            margin = tgt['margin']
                            notional = margin * self.leverage
                            qty = notional / entry_p
                            if capital >= margin + notional * self.tx:
                                capital -= margin + notional * self.tx
                                positions[t] = {'qty': qty, 'entry': entry_p, 'margin': margin}
                                trade_count += 1
                        else:
                            pos = positions[t]
                            delta = tgt['qty'] - pos['qty']
                            if delta > pos['qty'] * 0.05:
                                entry_p = cur_p * (1 + slip)
                                add_notional = delta * entry_p
                                add_margin = add_notional / self.leverage
                                if capital >= add_margin + add_notional * self.tx:
                                    capital -= add_margin + add_notional * self.tx
                                    total_qty = pos['qty'] + delta
                                    pos['entry'] = (pos['entry'] * pos['qty'] + entry_p * delta) / total_qty
                                    pos['qty'] = total_qty
                                    pos['margin'] += add_margin
                                    trade_count += 1

            # ── 4. 청산 후 재진입 ──
            if not need_rebal and positions:
                pv = capital
                for t, pos in positions.items():
                    cur_p = get_price(t, spot_prices, date)
                    if cur_p > 0:
                        pv += pos['margin'] + pos['qty'] * (cur_p - pos['entry'])
                for t, w in target.items():
                    if t == 'CASH' or w <= 0 or t in positions:
                        continue
                    cur_p = get_price(t, spot_prices, date)
                    if cur_p <= 0:
                        continue
                    binance_s = TICKER_MAP.get(t, '')
                    slip = SLIPPAGE.get(binance_s, 0.0005)
                    margin = pv * w * 0.95
                    entry_p = cur_p * (1 + slip)
                    notional = margin * self.leverage
                    qty = notional / entry_p
                    if capital >= margin + notional * self.tx:
                        capital -= margin + notional * self.tx
                        positions[t] = {'qty': qty, 'entry': entry_p, 'margin': margin}
                        trade_count += 1

            # ── 5. PV ──
            pv = capital
            for t, pos in positions.items():
                cur_p = get_price(t, spot_prices, date)
                if cur_p > 0:
                    pv += pos['margin'] + pos['qty'] * (cur_p - pos['entry'])
            pv_list.append({'Date': date, 'Value': max(pv, 0)})

        # Cleanup
        for t, pos in positions.items():
            cur_p = get_price(t, spot_prices, trace[-1]['date'])
            if cur_p > 0:
                capital += pos['margin'] + pos['qty'] * (cur_p - pos['entry']) - pos['qty'] * cur_p * self.tx

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
    t0 = time.time()

    # Spot data + trace
    um = load_universe()
    um40 = filter_universe(um, 40)
    all_t = set()
    for ts in um40.values():
        all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    spot_prices = load_all_prices(list(all_t))

    common = dict(dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
                  drift_threshold=0.10, post_flip_delay=5)
    p = B(health_sma=0, health_mom_short=30, selection='SG',
          n_picks=5, weighting='WG', top_n=40, risk='G5')
    p.start_date = '2020-10-01'
    p.end_date = '2026-03-28'

    trace = []
    r_spot = run_coin_backtest(spot_prices, um40, (1, 10, 19), params_base=p, _trace=trace, **common)
    m_spot = r_spot['metrics']
    cal_spot = m_spot['CAGR'] / abs(m_spot['MDD']) if m_spot['MDD'] != 0 else 0
    print(f"현물 V18: CAGR={m_spot['CAGR']:+.1%} MDD={m_spot['MDD']:+.1%} Cal={cal_spot:.2f}")

    # Hourly data
    for interval in ['1h', '4h']:
        hourly = load_hourly_data(interval)
        print(f"\n{'='*60}")
        print(f"  일봉시그널 + {interval} 체결/청산/펀딩")
        print(f"{'='*60}")
        print(f"  {'Config':<20s} {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>5s} {'Liq':>3s} {'Tr':>4s}")
        print(f"  {'-'*50}")

        for lev in [1.0, 1.5, 2.0, 3.0]:
            engine = HourlyFuturesReplay(hourly, interval=interval, leverage=lev)
            m = engine.run(trace, spot_prices)
            liq = f"💀{m['Liq']}" if m['Liq'] > 0 else ""
            print(f"  {lev}x{'':<16s} {m['Sharpe']:>5.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>5.2f}{liq:>5s} {m['Trades']:>4d}")

    # Also run daily-only v3 for comparison
    from backtest_futures_v3 import FuturesReplay, load_funding_daily
    funding = load_funding_daily()
    print(f"\n{'='*60}")
    print(f"  일봉 체결 (v3 trace replay, 비교용)")
    print(f"{'='*60}")
    print(f"  {'Config':<20s} {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>5s} {'Liq':>3s} {'Tr':>4s}")
    print(f"  {'-'*50}")
    for lev in [1.0, 1.5, 2.0, 3.0]:
        e = FuturesReplay(leverage=lev, tx_cost=0.0004, slippage=0.0005, funding_daily=funding)
        m = e.run(trace, spot_prices)
        liq = f"💀{m['Liq']}" if m['Liq'] > 0 else ""
        print(f"  {lev}x{'':<16s} {m['Sharpe']:>5.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>5.2f}{liq:>5s} {m['Trades']:>4d}")

    print(f"\n소요: {time.time() - t0:.0f}s")
