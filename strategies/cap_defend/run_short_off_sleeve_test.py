#!/usr/bin/env python3
"""카나리 OFF 구간 숏 sleeve 단독 비교.

목적:
- 현재 롱 전략과 결합하기 전에, OFF 구간 숏 sleeve 자체가 의미가 있는지 먼저 확인
- ON 구간은 전부 CASH, OFF 구간만 숏 포지션 운용

현재 최종 선물 전략의 카나리 정의를 그대로 사용:
- 4h1: SMA 240, hyst 1.5%
- 4h2: SMA 120, hyst 1.5%
- 1h1: SMA 168, hyst 1.5%
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data, SLIPPAGE_MAP
from run_stoploss_test import START, END


FINAL_CANARIES = {
    '4h1': dict(interval='4h', sma_bars=240, canary_hyst=0.015),
    '4h2': dict(interval='4h', sma_bars=120, canary_hyst=0.015),
    '1h1': dict(interval='1h', sma_bars=168, canary_hyst=0.015),
}


SHORT_CASES = [
    dict(
        name='baseline_cash',
        desc='기준선: 항상 현금',
        trigger='all_off',
        weights={},
        leverage=1.0,
    ),
    dict(
        name='all_off_btc_short_100',
        desc='세 카나리 모두 OFF면 BTC 100% 숏',
        trigger='all_off',
        weights={'BTC': 1.0},
        leverage=1.0,
    ),
    dict(
        name='all_off_btc_short_50',
        desc='세 카나리 모두 OFF면 BTC 50% 숏 + CASH 50%',
        trigger='all_off',
        weights={'BTC': 0.5},
        leverage=1.0,
    ),
    dict(
        name='all_off_btc_eth_short',
        desc='세 카나리 모두 OFF면 BTC/ETH 50:50 숏',
        trigger='all_off',
        weights={'BTC': 0.5, 'ETH': 0.5},
        leverage=1.0,
    ),
    dict(
        name='4h_off_btc_short_100',
        desc='4h1/4h2 OFF면 BTC 100% 숏',
        trigger='fourh_off',
        weights={'BTC': 1.0},
        leverage=1.0,
    ),
    dict(
        name='4h_off_btc_short_50',
        desc='4h1/4h2 OFF면 BTC 50% 숏 + CASH 50%',
        trigger='fourh_off',
        weights={'BTC': 0.5},
        leverage=1.0,
    ),
]


class ShortSleeveEngine:
    """OFF 구간 숏 sleeve 비교용 단순 실행 엔진."""

    def __init__(self, bars_1h, funding, leverage=1.0, tx_cost=0.0004,
                 maint_rate=0.004, initial_capital=10000.0):
        self.bars = bars_1h
        self.funding = funding
        self.leverage = leverage
        self.tx_cost = tx_cost
        self.maint_rate = maint_rate
        self.initial_capital = initial_capital

    def run(self, target_series):
        self.capital = self.initial_capital
        self.shorts = {}
        self.entry_prices = {}
        self.margins = {}
        liq_count = 0
        rebal_count = 0
        pv_list = []
        prev_target = {}

        norm_funding = {}
        for coin, fr in self.funding.items():
            norm_funding[coin] = fr.copy()
            norm_funding[coin].index = norm_funding[coin].index.floor('h')

        for date, target in target_series:
            for coin in list(self.shorts.keys()):
                df = self.bars.get(coin)
                if df is None:
                    continue
                ci = df.index.get_indexer([date], method='ffill')[0]
                if ci < 0:
                    continue
                high = float(df['High'].iloc[ci])
                liq_price = self._get_short_liq_price(coin)
                if liq_price is not None and high >= liq_price:
                    eq = self.margins[coin] + self.shorts[coin] * (self.entry_prices[coin] - liq_price)
                    returned = max(eq - max(eq, 0) * 0.015, 0)
                    self.capital += returned
                    del self.shorts[coin]
                    del self.entry_prices[coin]
                    del self.margins[coin]
                    liq_count += 1

            for coin in list(self.shorts.keys()):
                fr_series = norm_funding.get(coin)
                if fr_series is None:
                    continue
                if date in fr_series.index:
                    fr = float(fr_series.loc[date])
                    cur = self._get_price(coin, date)
                    if cur > 0 and fr != 0 and not np.isnan(fr):
                        # 롱과 반대 방향: 양수 funding이면 숏이 받는다
                        self.capital += self.shorts[coin] * cur * fr

            self.capital = max(self.capital, 0)

            need_rebal = (target != prev_target)
            if need_rebal:
                self._execute_rebalance(target, date)
                rebal_count += 1

            pv = self.capital
            for coin in self.shorts:
                cur = self._get_price(coin, date)
                if cur > 0:
                    pv += self.margins[coin] + self.shorts[coin] * (self.entry_prices[coin] - cur)
            pv_list.append({'Date': date, 'Value': max(pv, 0)})
            prev_target = dict(target)

        if not pv_list:
            return {}

        pvdf = pd.DataFrame(pv_list).set_index('Date')
        eq = pvdf['Value']
        eq_daily = eq.resample('D').last().dropna()
        yrs = (eq_daily.index[-1] - eq_daily.index[0]).days / 365.25
        if eq_daily.iloc[-1] <= 0 or yrs <= 0:
            return {'Sharpe': 0, 'CAGR': -1, 'MDD': -1, 'Cal': 0,
                    'Liq': liq_count, 'Rebal': rebal_count, '_equity': eq}
        cagr = (eq_daily.iloc[-1] / eq_daily.iloc[0]) ** (1 / yrs) - 1
        dr = eq_daily.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
        mdd = (eq / eq.cummax() - 1).min()
        cal = cagr / abs(mdd) if mdd != 0 else 0
        return {'Sharpe': sh, 'CAGR': cagr, 'MDD': mdd, 'Cal': cal,
                'Liq': liq_count, 'Rebal': rebal_count, '_equity': eq}

    def _get_price(self, coin, date):
        df = self.bars.get(coin)
        if df is None:
            return 0
        ci = df.index.get_indexer([date], method='ffill')[0]
        return float(df['Close'].iloc[ci]) if ci >= 0 else 0

    def _get_short_liq_price(self, coin):
        qty = self.shorts.get(coin, 0.0)
        entry = self.entry_prices.get(coin, 0.0)
        margin = self.margins.get(coin, 0.0)
        if qty <= 0 or entry <= 0:
            return None
        denom = qty * (1.0 + self.maint_rate)
        if denom <= 0:
            return None
        liq_price = (margin + qty * entry) / denom
        return liq_price if liq_price > 0 else None

    def _execute_rebalance(self, target, date):
        pv = self.capital
        for coin in self.shorts:
            cur = self._get_price(coin, date)
            if cur > 0:
                pv += self.margins[coin] + self.shorts[coin] * (self.entry_prices[coin] - cur)
        if pv <= 0:
            return

        target_qty = {}
        target_margin = {}
        for coin, w in target.items():
            if coin == 'CASH' or w <= 0:
                continue
            cur = self._get_price(coin, date)
            if cur <= 0:
                continue
            tmgn = pv * w * 0.95
            tnotional = tmgn * self.leverage
            target_qty[coin] = tnotional / cur
            target_margin[coin] = tmgn

        for coin in list(self.shorts.keys()):
            cur = self._get_price(coin, date)
            if cur <= 0:
                continue
            slip = SLIPPAGE_MAP.get(coin, 0.0005)
            if coin not in target_qty:
                pnl = self.shorts[coin] * (self.entry_prices[coin] - cur * (1 + slip))
                tx = self.shorts[coin] * cur * self.tx_cost
                self.capital += self.margins[coin] + pnl - tx
                del self.shorts[coin]
                del self.entry_prices[coin]
                del self.margins[coin]
            else:
                delta = target_qty[coin] - self.shorts[coin]
                if delta < -self.shorts[coin] * 0.05:
                    buyback_qty = -delta
                    cover_frac = buyback_qty / self.shorts[coin]
                    cover_margin = self.margins[coin] * cover_frac
                    pnl = buyback_qty * (self.entry_prices[coin] - cur * (1 + slip))
                    tx = buyback_qty * cur * self.tx_cost
                    self.capital += cover_margin + pnl - tx
                    self.shorts[coin] -= buyback_qty
                    self.margins[coin] -= cover_margin

        for coin, tqty in target_qty.items():
            cur = self._get_price(coin, date)
            if cur <= 0:
                continue
            slip = SLIPPAGE_MAP.get(coin, 0.0005)
            if coin not in self.shorts:
                entry_p = cur * (1 - slip)
                margin = target_margin[coin]
                notional = margin * self.leverage
                qty = notional / entry_p
                tx = notional * self.tx_cost
                if self.capital >= margin + tx:
                    self.capital -= margin + tx
                    self.shorts[coin] = qty
                    self.entry_prices[coin] = entry_p
                    self.margins[coin] = margin
            else:
                delta = tqty - self.shorts[coin]
                if delta > self.shorts[coin] * 0.05:
                    entry_p = cur * (1 - slip)
                    add_notional = delta * entry_p
                    add_margin = add_notional / self.leverage
                    tx = add_notional * self.tx_cost
                    if self.capital >= add_margin + tx:
                        self.capital -= add_margin + tx
                        total = self.shorts[coin] + delta
                        self.entry_prices[coin] = (
                            self.entry_prices[coin] * self.shorts[coin] + entry_p * delta
                        ) / total
                        self.shorts[coin] = total
                        self.margins[coin] += add_margin


def build_canary_trace(data, cfg):
    bars, _ = data[cfg['interval']]
    btc = bars['BTC']
    dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    close = btc['Close'].values
    sma_bars = cfg['sma_bars']
    hyst = cfg['canary_hyst']
    prev_canary = False
    out = []
    for date in dates:
        ci = btc.index.get_loc(date)
        if ci < sma_bars:
            canary = False
            ratio = np.nan
        else:
            sma = float(np.mean(close[ci - sma_bars + 1:ci + 1]))
            ratio = float(close[ci] / sma) if sma > 0 else np.nan
            if prev_canary:
                canary = ratio >= (1.0 - hyst)
            else:
                canary = ratio > (1.0 + hyst)
        prev_canary = canary
        out.append((date, canary, ratio))
    return out


def align_canary_series(trace, all_dates_1h):
    idx = 0
    latest_canary = False
    latest_ratio = np.nan
    aligned = []
    for date in all_dates_1h:
        while idx < len(trace) and trace[idx][0] <= date:
            _, latest_canary, latest_ratio = trace[idx]
            idx += 1
        aligned.append((date, latest_canary, latest_ratio))
    return aligned


def build_target_series(all_dates_1h, canaries, case):
    trigger = case['trigger']
    weights = case['weights']
    series = []
    for i, date in enumerate(all_dates_1h):
        c4h1 = canaries['4h1'][i][1]
        c4h2 = canaries['4h2'][i][1]
        c1h1 = canaries['1h1'][i][1]
        if trigger == 'all_off':
            active = (not c4h1) and (not c4h2) and (not c1h1)
        elif trigger == 'fourh_off':
            active = (not c4h1) and (not c4h2)
        else:
            active = False

        if not active or not weights:
            series.append((date, {'CASH': 1.0}))
            continue

        total_w = sum(weights.values())
        target = dict(weights)
        target['CASH'] = max(0.0, 1.0 - total_w)
        series.append((date, target))
    return series


def main():
    t0 = time.time()
    print('Loading data...')
    data = {iv: load_data(iv) for iv in ['4h', '1h']}
    bars_1h, funding_1h = data['1h']
    all_dates_1h = bars_1h['BTC'].index[(bars_1h['BTC'].index >= START) & (bars_1h['BTC'].index <= END)]

    print('Building canary traces...')
    canaries = {}
    for name, cfg in FINAL_CANARIES.items():
        trace = build_canary_trace(data, cfg)
        canaries[name] = align_canary_series(trace, all_dates_1h)

    print('\nResults')
    results = []
    for case in SHORT_CASES:
        target_series = build_target_series(all_dates_1h, canaries, case)
        engine = ShortSleeveEngine(
            bars_1h, funding_1h,
            leverage=case['leverage'],
            tx_cost=0.0004,
            maint_rate=0.004,
            initial_capital=10000.0,
        )
        m = engine.run(target_series)
        row = {
            'name': case['name'],
            'desc': case['desc'],
            'Cal': m.get('Cal', 0),
            'CAGR': m.get('CAGR', 0),
            'MDD': m.get('MDD', 0),
            'Sharpe': m.get('Sharpe', 0),
            'Liq': m.get('Liq', 0),
            'Rebal': m.get('Rebal', 0),
        }
        results.append(row)
        print(
            f"{row['name']:<24} "
            f"Cal={row['Cal']:.2f} "
            f"CAGR={row['CAGR']:+.1%} "
            f"MDD={row['MDD']:+.1%} "
            f"Liq={row['Liq']} Rebal={row['Rebal']}"
        )

    results.sort(key=lambda r: (-r['Cal'], -r['Sharpe'], r['name']))
    print('\nTop candidates')
    for row in results[:5]:
        print(
            f"- {row['name']}: Cal={row['Cal']:.2f}, "
            f"CAGR={row['CAGR']:+.1%}, MDD={row['MDD']:+.1%}, {row['desc']}"
        )
    print(f'\nElapsed: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
