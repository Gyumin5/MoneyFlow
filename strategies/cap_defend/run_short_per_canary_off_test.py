#!/usr/bin/env python3
"""전략별 카나리 OFF 구간 BTC 숏 sleeve 테스트.

원칙:
- 현재 실거래 전략의 카나리 정의를 그대로 사용
- 4h1 OFF, 4h2 OFF, 1h1 OFF를 각각 독립적으로 평가
- 각 전략은 자기 시간축에서만 평가
- 시작점은 BTC short 100%
- 대신 필터 변수는 바로 여러 조합 테스트
"""
import os
import sys
import time

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data, SLIPPAGE_MAP
from run_stoploss_test import START, END


STRATEGIES = {
    '4h1': dict(interval='4h', sma_bars=240, mom_short_bars=10, mom_long_bars=30, canary_hyst=0.015),
    '4h2': dict(interval='4h', sma_bars=120, mom_short_bars=20, mom_long_bars=120, canary_hyst=0.015),
    '1h1': dict(interval='1h', sma_bars=168, mom_short_bars=36, mom_long_bars=720, canary_hyst=0.015),
}


FILTER_CASES = [
    dict(name='off_only', desc='카나리 OFF면 바로 BTC short 100%', below_sma=False, mom_short=False, mom_long=False),
    dict(name='off_and_below_sma', desc='카나리 OFF + BTC<SMA', below_sma=True, mom_short=False, mom_long=False),
    dict(name='off_and_mom_short', desc='카나리 OFF + MomShort<0', below_sma=False, mom_short=True, mom_long=False),
    dict(name='off_and_below_sma_mom_short', desc='카나리 OFF + BTC<SMA + MomShort<0', below_sma=True, mom_short=True, mom_long=False),
    dict(name='off_and_mom_short_mom_long', desc='카나리 OFF + MomShort<0 + MomLong<0', below_sma=False, mom_short=True, mom_long=True),
    dict(name='off_and_below_sma_mom_short_mom_long', desc='카나리 OFF + BTC<SMA + MomShort<0 + MomLong<0', below_sma=True, mom_short=True, mom_long=True),
]


class ShortOnlyEngine:
    def __init__(self, bars, funding, leverage=1.0, tx_cost=0.0004, maint_rate=0.004, initial_capital=10000.0):
        self.bars = bars
        self.funding = funding
        self.leverage = leverage
        self.tx_cost = tx_cost
        self.maint_rate = maint_rate
        self.initial_capital = initial_capital

    def _get_price(self, coin, date):
        df = self.bars.get(coin)
        if df is None:
            return 0.0
        ci = df.index.get_indexer([date], method='ffill')[0]
        return float(df['Close'].iloc[ci]) if ci >= 0 else 0.0

    def _get_short_liq_price(self, coin):
        qty = self.shorts.get(coin, 0.0)
        entry = self.entry_prices.get(coin, 0.0)
        margin = self.margins.get(coin, 0.0)
        if qty <= 0 or entry <= 0:
            return None
        denom = qty * (1.0 + self.maint_rate)
        if denom <= 0:
            return None
        liq = (margin + qty * entry) / denom
        return liq if liq > 0 else None

    def _rebalance(self, target, date):
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
            margin = pv * w * 0.95
            notional = margin * self.leverage
            target_qty[coin] = notional / cur
            target_margin[coin] = margin

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
                entry = cur * (1 - slip)
                margin = target_margin[coin]
                notional = margin * self.leverage
                qty = notional / entry
                tx = notional * self.tx_cost
                if self.capital >= margin + tx:
                    self.capital -= margin + tx
                    self.shorts[coin] = qty
                    self.entry_prices[coin] = entry
                    self.margins[coin] = margin
            else:
                delta = tqty - self.shorts[coin]
                if delta > self.shorts[coin] * 0.05:
                    entry = cur * (1 - slip)
                    add_notional = delta * entry
                    add_margin = add_notional / self.leverage
                    tx = add_notional * self.tx_cost
                    if self.capital >= add_margin + tx:
                        self.capital -= add_margin + tx
                        total = self.shorts[coin] + delta
                        self.entry_prices[coin] = (
                            self.entry_prices[coin] * self.shorts[coin] + entry * delta
                        ) / total
                        self.shorts[coin] = total
                        self.margins[coin] += add_margin

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
                liq = self._get_short_liq_price(coin)
                if liq is not None and high >= liq:
                    eq = self.margins[coin] + self.shorts[coin] * (self.entry_prices[coin] - liq)
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
                        self.capital += self.shorts[coin] * cur * fr

            self.capital = max(self.capital, 0)

            if target != prev_target:
                self._rebalance(target, date)
                rebal_count += 1
            prev_target = dict(target)

            pv = self.capital
            for coin in self.shorts:
                cur = self._get_price(coin, date)
                if cur > 0:
                    pv += self.margins[coin] + self.shorts[coin] * (self.entry_prices[coin] - cur)
            pv_list.append({'Date': date, 'Value': max(pv, 0)})

        if not pv_list:
            return {}
        pvdf = pd.DataFrame(pv_list).set_index('Date')
        eq = pvdf['Value']
        eq_daily = eq.resample('D').last().dropna()
        yrs = (eq_daily.index[-1] - eq_daily.index[0]).days / 365.25
        if eq_daily.iloc[-1] <= 0 or yrs <= 0:
            return {'Sharpe': 0, 'CAGR': -1, 'MDD': -1, 'Cal': 0, 'Liq': liq_count, 'Rebal': rebal_count}
        cagr = (eq_daily.iloc[-1] / eq_daily.iloc[0]) ** (1 / yrs) - 1
        dr = eq_daily.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
        mdd = (eq / eq.cummax() - 1).min()
        cal = cagr / abs(mdd) if mdd != 0 else 0
        return {'Sharpe': sh, 'CAGR': cagr, 'MDD': mdd, 'Cal': cal, 'Liq': liq_count, 'Rebal': rebal_count}


def build_canary_target_series(data, strat_name, filter_case):
    cfg = STRATEGIES[strat_name]
    bars, _ = data[cfg['interval']]
    btc = bars['BTC']
    dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    close = btc['Close'].values
    prev_canary = False
    out = []

    for date in dates:
        ci = btc.index.get_loc(date)
        if ci < cfg['sma_bars']:
            canary = False
            below_sma = False
        else:
            sma = float(np.mean(close[ci - cfg['sma_bars'] + 1:ci + 1]))
            ratio = float(close[ci] / sma) if sma > 0 else np.nan
            if prev_canary:
                canary = ratio >= (1.0 - cfg['canary_hyst'])
            else:
                canary = ratio > (1.0 + cfg['canary_hyst'])
            below_sma = close[ci] < sma if sma > 0 else False

        mom_short_neg = False
        if ci >= cfg['mom_short_bars']:
            mom_short_neg = (close[ci] / close[ci - cfg['mom_short_bars']] - 1.0) < 0

        mom_long_neg = False
        if ci >= cfg['mom_long_bars']:
            mom_long_neg = (close[ci] / close[ci - cfg['mom_long_bars']] - 1.0) < 0

        active = not canary
        if filter_case['below_sma']:
            active = active and below_sma
        if filter_case['mom_short']:
            active = active and mom_short_neg
        if filter_case['mom_long']:
            active = active and mom_long_neg

        out.append((date, {'BTC': 1.0} if active else {'CASH': 1.0}))
        prev_canary = canary

    return out


def main():
    t0 = time.time()
    print('Loading data...')
    data = {iv: load_data(iv) for iv in ['4h', '1h']}

    print('\nResults')
    rows = []
    for strat_name, cfg in STRATEGIES.items():
        bars, funding = data[cfg['interval']]
        engine = ShortOnlyEngine(bars, funding, leverage=1.0, tx_cost=0.0004, maint_rate=0.004, initial_capital=10000.0)
        print(f'\n== {strat_name} ({cfg["interval"]}) ==')
        for filter_case in FILTER_CASES:
            target_series = build_canary_target_series(data, strat_name, filter_case)
            m = engine.run(target_series)
            row = {
                'strategy': strat_name,
                'interval': cfg['interval'],
                'filter': filter_case['name'],
                'desc': filter_case['desc'],
                'Cal': m.get('Cal', 0),
                'CAGR': m.get('CAGR', 0),
                'MDD': m.get('MDD', 0),
                'Sharpe': m.get('Sharpe', 0),
                'Liq': m.get('Liq', 0),
                'Rebal': m.get('Rebal', 0),
            }
            rows.append(row)
            print(
                f"{row['filter']:<34} Cal={row['Cal']:.2f} "
                f"CAGR={row['CAGR']:+.1%} MDD={row['MDD']:+.1%} "
                f"Liq={row['Liq']} Rebal={row['Rebal']}"
            )

    rows.sort(key=lambda r: (-r['Cal'], -r['Sharpe'], r['strategy'], r['filter']))
    print('\nTop candidates')
    for row in rows[:10]:
        print(
            f"- {row['strategy']} / {row['filter']}: "
            f"Cal={row['Cal']:.2f}, CAGR={row['CAGR']:+.1%}, "
            f"MDD={row['MDD']:+.1%}, {row['desc']}"
        )
    print(f'\nElapsed: {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
