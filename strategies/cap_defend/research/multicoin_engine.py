#!/usr/bin/env python3
"""Multi-coin portfolio simulator (fixed, event-based accounting).

정확한 회계:
- 매 tick t:
  1. 현재 포지션 중 exit 조건 체크 → 매도 (cash += proceeds)
  2. 남은 슬롯에 대해 dip 후보 수집 → n_pick까지 선정 → 매수 (cash -= cost)
  3. equity_t = cash + Σ(position_qty × close_t)
- 중복 계산 없음
- 매 bar 체결은 open/high/low/close 중 명시

사용 방법:
    from multicoin_engine import run_multicoin_portfolio

    eq, events = run_multicoin_portfolio(
        universe=['BTCUSDT','ETHUSDT',...],
        dip_bars=24, dip_threshold=-0.15,
        take_profit=0.08, time_stop_bars=24,
        n_pick=3, select_method='deepest',
        capital=10000.0, leverage=1.0,
        tx_cost=0.003, buy_at='high', sell_at='open',
    )
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')


def load_coin(sym, interval='1h', start='2020-10-01', end='2026-03-30'):
    path = os.path.join(DATA_DIR, f'{sym}_{interval}.csv')
    if not os.path.isfile(path): return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    return df.loc[start:end].copy()


def prep_universe(coins, dip_bars, start='2020-10-01', end='2026-03-30'):
    data = {}
    for c in coins:
        df = load_coin(c, '1h', start, end)
        if df is None or len(df) < 1000:
            continue
        df['dip_pct'] = df['Close'] / df['Close'].shift(dip_bars) - 1.0
        data[c] = df
    return data


def run_multicoin_portfolio(universe, dip_bars, dip_threshold,
                             take_profit, time_stop_bars,
                             n_pick=3, select_method='deepest',
                             capital=10000.0, leverage=1.0,
                             tx_cost=0.003, buy_at='high', sell_at='open',
                             cap_ranks=None, max_pos_capital_pct=None,
                             start='2020-10-01', end='2026-03-30',
                             verbose=False):
    """Run multi-coin C portfolio backtest.

    Args:
        universe: list of coin symbols
        dip_bars: lookback for dip (bars)
        dip_threshold: dip threshold (e.g. -0.15)
        take_profit: TP threshold (e.g. 0.08)
        time_stop_bars: max holding bars
        n_pick: max concurrent positions
        select_method: 'deepest' or 'cap' or 'rank_mix'
        capital: initial capital (KRW/USD agnostic, ratio-based)
        leverage: leverage (1.0 = spot)
        tx_cost: per-side transaction cost (e.g. 0.003 = 30bps)
        buy_at: 'open' | 'high' | 'close'
        sell_at: 'open' | 'low' | 'close'
        cap_ranks: dict coin -> rank (lower = higher cap). default: universe order
        max_pos_capital_pct: if set, cap per-position to this fraction of equity

    Returns:
        equity_series (pd.Series, index=time),
        events (list of dict)
    """
    data = prep_universe(universe, dip_bars, start, end)
    if not data:
        return pd.Series(dtype=float), []

    # 공통 인덱스 (unified bar timestamps)
    idx = None
    for df in data.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    idx = idx.sort_values()
    if len(idx) < 100:
        return pd.Series(dtype=float), []

    if cap_ranks is None:
        cap_ranks = {c: i for i, c in enumerate(universe)}

    cash = capital
    # positions[coin] = {'qty': float, 'entry_px': float, 'entry_ts': ts, 'bars_held': int, 'slot_notional': float}
    positions: dict = {}
    events: list = []
    equity_list = []

    def get_price(row, which):
        cap = which.capitalize()
        return float(row[cap]) if cap in row else float(row['Open'])

    for t_idx, ts in enumerate(idx):
        # 1) Exit check
        to_close = []
        for c, p in positions.items():
            row = data[c].loc[ts]
            sell_px = get_price(row, sell_at)
            pnl_ratio = (sell_px / p['entry_px'] - 1.0)
            if pnl_ratio >= take_profit or p['bars_held'] >= time_stop_bars:
                # close position
                gross = p['qty'] * sell_px
                # leverage-aware: realized_pnl = qty × (sell - entry) × lev (for futures), but spot lev=1
                # 단순화: spot only (leverage=1). 선물이면 engine B 필요
                if leverage == 1.0:
                    proceeds = gross * (1 - tx_cost)
                    cash += proceeds
                else:
                    # margin 반환: entry_notional/lev + realized_pnl − tx
                    margin = p['slot_notional'] / leverage
                    realized = p['qty'] * (sell_px - p['entry_px'])  # leverage-adjusted qty already
                    cash += margin + realized - gross * tx_cost
                to_close.append((c, pnl_ratio, sell_px))
                events.append({
                    'coin': c, 'entry_ts': p['entry_ts'], 'exit_ts': ts,
                    'entry_px': round(p['entry_px'], 6), 'exit_px': round(sell_px, 6),
                    'pnl_pct': round(pnl_ratio * 100, 3),
                    'bars_held': p['bars_held'],
                    'reason': 'TP' if pnl_ratio >= take_profit else 'timeout',
                })

        for c, _, _ in to_close:
            del positions[c]

        # 2) Entry check
        open_slots = n_pick - len(positions)
        if open_slots > 0:
            # 현재 equity (for slot sizing)
            current_equity = cash + sum(p['qty'] * get_price(data[c].loc[ts], 'close') for c, p in positions.items())
            if current_equity <= 0:
                current_equity = cash  # fallback

            candidates = []
            for c, df in data.items():
                if c in positions: continue
                if ts not in df.index: continue
                prev_idx = df.index.get_loc(ts) - 1
                if prev_idx < 0: continue
                prev = df.iloc[prev_idx]
                dip = prev['dip_pct']
                if pd.isna(dip) or dip > dip_threshold: continue
                candidates.append({
                    'coin': c, 'dip': float(dip),
                    'cap_rank': cap_ranks.get(c, 999),
                })

            if candidates:
                if select_method == 'deepest':
                    candidates.sort(key=lambda x: x['dip'])
                elif select_method == 'cap':
                    candidates.sort(key=lambda x: x['cap_rank'])
                elif select_method == 'rank_mix':
                    # cap순위 + dip 깊이 blend
                    candidates.sort(key=lambda x: x['cap_rank'] + 10 * x['dip'])
                else:
                    candidates.sort(key=lambda x: x['dip'])

                picks = candidates[:open_slots]
                for pick in picks:
                    c = pick['coin']
                    row = data[c].loc[ts]
                    buy_px = get_price(row, buy_at)
                    # slot notional target (equal weight). tx 포함해서 cash 여유로 조정.
                    target_slot = current_equity / n_pick
                    if max_pos_capital_pct is not None:
                        target_slot = min(target_slot, current_equity * max_pos_capital_pct)
                    if leverage == 1.0:
                        # spot: 가용 cash에서 (1+tx)로 나눠 notional 산출
                        max_notional = (cash * 0.999) / (1 + tx_cost)
                        slot_notional = min(target_slot, max_notional)
                        if slot_notional <= 0 or buy_px <= 0:
                            continue
                        cost = slot_notional * (1 + tx_cost)
                        qty = slot_notional / buy_px
                        cash -= cost
                    else:
                        # leverage: margin = notional / lev
                        max_notional_by_margin = (cash * 0.999 - target_slot * tx_cost) * leverage
                        slot_notional = max(0.0, min(target_slot, max_notional_by_margin))
                        if slot_notional <= 0 or buy_px <= 0:
                            continue
                        margin = slot_notional / leverage
                        qty = slot_notional / buy_px
                        cost = margin + slot_notional * tx_cost
                        cash -= cost
                    positions[c] = {
                        'qty': qty,
                        'entry_px': buy_px,
                        'entry_ts': ts,
                        'bars_held': 0,
                        'slot_notional': slot_notional,
                    }

        # 3) Update bars_held + compute equity
        for c, p in positions.items():
            p['bars_held'] += 1
        # equity = cash + sum(qty * close)
        pos_val = sum(p['qty'] * float(data[c].loc[ts]['Close']) for c, p in positions.items())
        equity_list.append(cash + pos_val)

    eq_series = pd.Series(equity_list, index=idx)
    return eq_series, events


def metrics(eq, bars_per_year=24*365):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'Sharpe': 0, 'CAGR': 0, 'MDD': 0, 'Cal': 0, 'Final': 0}
    std = rets.std()
    sh = (rets.mean() * bars_per_year) / (std * np.sqrt(bars_per_year)) if std > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else 0.001
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {'Sharpe': round(float(sh), 3), 'CAGR': round(float(cagr), 4),
            'MDD': round(float(mdd), 4), 'Cal': round(float(cal), 3),
            'Final': round(float(eq.iloc[-1]/eq.iloc[0]), 3)}
