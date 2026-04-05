#!/usr/bin/env python3
"""선물 백테스트 v3 — 현물 V18 trace 기반 replay.

현물 run_coin_backtest에서 _trace로 매일 target_weights를 추출,
선물 실행 엔진으로 replay. 전략 로직 100% 동일, 비용만 다름.
"""

import numpy as np, pandas as pd, os, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coin_engine import load_universe, load_all_prices, filter_universe, get_price, calc_metrics
from coin_helpers import B, ANCHOR_DAYS
from backtest_official import run_coin_backtest

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', '..', 'data', 'futures')


def load_funding_daily():
    """BTC 일간 펀딩레이트 합계 로드."""
    fpath = os.path.join(DATA_DIR, 'BTCUSDT_funding.csv')
    if not os.path.exists(fpath):
        return pd.Series(dtype=float)
    fund = pd.read_csv(fpath, parse_dates=['Date'], index_col='Date')['fundingRate']
    return fund.resample('D').sum()


class FuturesReplay:
    """현물 trace를 선물 비용으로 replay."""

    def __init__(self, leverage=1.0, tx_cost=0.0004, slippage=0.0005,
                 funding_daily=None, maint_margin_rate=0.004):
        self.leverage = leverage
        self.tx_cost = tx_cost
        self.slippage = slippage
        self.funding_daily = funding_daily  # pd.Series (일간 펀딩 합계)
        self.maint_rate = maint_margin_rate

    def run(self, trace, prices, initial_capital=10000.0):
        """trace (from run_coin_backtest) → 선물 equity curve."""
        capital = initial_capital
        positions = {}  # {ticker: {'qty': float, 'entry': float, 'margin': float}}
        pv_list = []
        liq_count = 0
        trade_count = 0
        prev_target = {}

        for day in trace:
            date = day['date']
            target = day['target']
            need_rebal = day['rebal']

            # ── 1. 청산 체크 ──
            if self.leverage > 1:
                for t in list(positions.keys()):
                    pos = positions[t]
                    cur_p = get_price(t, prices, date)
                    if cur_p <= 0:
                        continue
                    unrealized = pos['qty'] * (cur_p - pos['entry'])
                    equity = pos['margin'] + unrealized
                    maint_req = pos['qty'] * cur_p * self.maint_rate
                    if equity <= maint_req:
                        liq_fee = max(equity, 0) * 0.015
                        capital = max(capital - liq_fee, 0)
                        del positions[t]
                        liq_count += 1

            # ── 2. 펀딩비 (equity 양수일 때만, 하한 0) ── ★ FIX 3
            if positions and self.funding_daily is not None:
                fr = self.funding_daily.get(date, 0)
                if not np.isnan(fr) and fr != 0:
                    total_fund = 0
                    for t, pos in positions.items():
                        cur_p = get_price(t, prices, date)
                        if cur_p > 0:
                            total_fund += pos['qty'] * cur_p * fr
                    capital -= total_fund
                    if capital < 0:
                        capital = 0  # 펀딩으로 파산 시 0으로 (청산 처리)

            # ── 3. 리밸런싱 (Delta 방식) ── ★ FIX 1, 2
            if need_rebal:
                pv = capital
                for t, pos in positions.items():
                    cur_p = get_price(t, prices, date)
                    if cur_p > 0:
                        pv += pos['margin'] + pos['qty'] * (cur_p - pos['entry'])

                if pv <= 100:
                    # 자본 부족 → 전량 청산
                    for t in list(positions.keys()):
                        pos = positions[t]
                        cur_p = get_price(t, prices, date)
                        if cur_p > 0:
                            pnl = pos['qty'] * (cur_p - pos['entry'])
                            capital += pos['margin'] + pnl - pos['qty'] * cur_p * self.tx_cost
                    positions.clear()
                else:
                    # ★ FIX 2: 레버리지 = 동일 마진으로 더 큰 노출
                    # margin = pv * w * 0.95 (현물과 동일한 자금 투입)
                    # notional = margin × leverage (레버리지만큼 노출 증가)
                    target_positions = {}
                    for t, w in target.items():
                        if t == 'CASH' or w <= 0:
                            continue
                        cur_p = get_price(t, prices, date)
                        if cur_p <= 0:
                            continue
                        target_margin = pv * w * 0.95
                        target_notional = target_margin * self.leverage
                        target_qty = target_notional / cur_p
                        target_positions[t] = {'qty': target_qty, 'margin': target_margin}

                    # ★ FIX 1: Delta 매매 — 변경분만 거래
                    # 먼저 매도 (보유 중이지만 target에 없거나 줄어야 하는 것)
                    for t in list(positions.keys()):
                        pos = positions[t]
                        cur_p = get_price(t, prices, date)
                        if cur_p <= 0:
                            continue
                        if t not in target_positions:
                            # 전량 매도
                            exit_p = cur_p * (1 - self.slippage)
                            pnl = pos['qty'] * (exit_p - pos['entry'])
                            tx = pos['qty'] * cur_p * self.tx_cost
                            capital += pos['margin'] + pnl - tx
                            del positions[t]
                            trade_count += 1
                        else:
                            tgt = target_positions[t]
                            delta_qty = tgt['qty'] - pos['qty']
                            if delta_qty < -pos['qty'] * 0.05:  # 5% 이상 줄어야 매도
                                sell_qty = -delta_qty
                                exit_p = cur_p * (1 - self.slippage)
                                sell_pnl = sell_qty * (exit_p - pos['entry'])
                                sell_margin = pos['margin'] * (sell_qty / pos['qty'])
                                tx = sell_qty * cur_p * self.tx_cost
                                capital += sell_margin + sell_pnl - tx
                                pos['qty'] -= sell_qty
                                pos['margin'] -= sell_margin
                                trade_count += 1

                    # 매수 (target에 있지만 보유 안 하거나 늘려야 하는 것)
                    for t, tgt in target_positions.items():
                        cur_p = get_price(t, prices, date)
                        if cur_p <= 0:
                            continue
                        if t not in positions:
                            # 신규 진입
                            entry_p = cur_p * (1 + self.slippage)
                            margin = tgt['margin']
                            notional = margin * self.leverage
                            qty = notional / entry_p
                            entry_tx = notional * self.tx_cost
                            if capital >= margin + entry_tx:
                                capital -= (margin + entry_tx)
                                positions[t] = {'qty': qty, 'entry': entry_p, 'margin': margin}
                                trade_count += 1
                        else:
                            pos = positions[t]
                            delta_qty = tgt['qty'] - pos['qty']
                            if delta_qty > pos['qty'] * 0.05:  # 5% 이상 늘어야 매수
                                entry_p = cur_p * (1 + self.slippage)
                                add_notional = delta_qty * entry_p
                                add_margin = add_notional / self.leverage
                                add_tx = add_notional * self.tx_cost
                                if capital >= add_margin + add_tx:
                                    capital -= (add_margin + add_tx)
                                    # 평균단가 갱신
                                    total_qty = pos['qty'] + delta_qty
                                    pos['entry'] = (pos['entry'] * pos['qty'] + entry_p * delta_qty) / total_qty
                                    pos['qty'] = total_qty
                                    pos['margin'] += add_margin
                                    trade_count += 1

            # ── 4. 청산 후 재진입 ── ★ FIX 4
            # 청산으로 비어진 포지션이 있고 target에 있으면 재진입 시도
            if not need_rebal and positions:
                pv = capital
                for t, pos in positions.items():
                    cur_p = get_price(t, prices, date)
                    if cur_p > 0:
                        pv += pos['margin'] + pos['qty'] * (cur_p - pos['entry'])
                for t, w in target.items():
                    if t == 'CASH' or w <= 0 or t in positions:
                        continue
                    cur_p = get_price(t, prices, date)
                    if cur_p <= 0:
                        continue
                    margin = pv * w * 0.95
                    entry_p = cur_p * (1 + self.slippage)
                    notional = margin * self.leverage
                    qty = notional / entry_p
                    entry_tx = notional * self.tx_cost
                    if capital >= margin + entry_tx:
                        capital -= (margin + entry_tx)
                        positions[t] = {'qty': qty, 'entry': entry_p, 'margin': margin}
                        trade_count += 1

            # ── 5. 포트폴리오 가치 ──
            pv = capital
            for t, pos in positions.items():
                cur_p = get_price(t, prices, date)
                if cur_p > 0:
                    pv += pos['margin'] + pos['qty'] * (cur_p - pos['entry'])
            pv_list.append({'Date': date, 'Value': max(pv, 0)})

        # 미청산 정리
        for t, pos in positions.items():
            cur_p = get_price(t, prices, date)
            if cur_p > 0:
                tx = pos['qty'] * cur_p * self.tx_cost
                capital += pos['margin'] + pos['qty'] * (cur_p - pos['entry']) - tx

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

    # Load spot data (same as V18)
    um_raw = load_universe()
    um40 = filter_universe(um_raw, 40)
    all_t = set()
    for ts in um40.values():
        all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    prices = load_all_prices(list(all_t))

    # Load funding
    funding = load_funding_daily()

    # Run V18 spot with trace
    common = dict(dd_lookback=60, dd_threshold=-0.25, bl_drop=-0.15, bl_days=7,
                  drift_threshold=0.10, post_flip_delay=5)
    p = B(health_sma=0, health_mom_short=30, selection='SG',
          n_picks=5, weighting='WG', top_n=40, risk='G5')
    p.start_date = '2020-10-01'
    p.end_date = '2026-03-28'

    trace = []
    r_spot = run_coin_backtest(prices, um40, (1, 10, 19), params_base=p, _trace=trace, **common)
    m_spot = r_spot['metrics']
    cal_spot = m_spot['CAGR'] / abs(m_spot['MDD']) if m_spot['MDD'] != 0 else 0

    print(f"현물 V18:        Sh={m_spot['Sharpe']:.2f} CAGR={m_spot['CAGR']:+.1%} MDD={m_spot['MDD']:+.1%} Cal={cal_spot:.2f} Rb={r_spot['rebal_count']}")
    print(f"Trace: {len(trace)} days\n")

    # Replay with different cost models
    print(f"{'Config':<25s} {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>5s} {'Liq':>3s} {'Tr':>4s}")
    print(f"{'-'*55}")

    configs = [
        ('Spot (tx=0.4%)', 1.0, 0.004, 0, None),
        ('Futures 1x (tx+fund)', 1.0, 0.0004, 0.0005, funding),
        ('Futures 1x (tx only)', 1.0, 0.0004, 0.0005, None),
        ('Futures 1.5x', 1.5, 0.0004, 0.0005, funding),
        ('Futures 2x', 2.0, 0.0004, 0.0005, funding),
        ('Futures 3x', 3.0, 0.0004, 0.0005, funding),
        ('Futures 1x no slip', 1.0, 0.0004, 0, funding),
        ('Futures 2x no fund', 2.0, 0.0004, 0.0005, None),
    ]

    for name, lev, tx, slip, fund in configs:
        engine = FuturesReplay(leverage=lev, tx_cost=tx, slippage=slip, funding_daily=fund)
        m = engine.run(trace, prices)
        liq = f"💀{m['Liq']}" if m['Liq'] > 0 else ""
        print(f"{name:<25s} {m['Sharpe']:>5.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>5.2f}{liq:>5s} {m['Trades']:>4d}")

    print(f"\n소요: {time.time() - t0:.0f}s")
