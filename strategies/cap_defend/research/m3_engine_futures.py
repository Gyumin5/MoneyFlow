#!/usr/bin/env python3
"""M3 Engine for Futures — leverage support.

m3_engine_final과 동일하되 leverage 파라미터 추가.

Interpretation:
- cap_per_slot: margin 비중 (port_eq 대비). 실제 notional = cap × lev.
- Lot qty: target_notional / entry_px
- Lot value (c_delta 기여): qty × price (notional 기반)
- MTM: qty × (cur - prev) (notional 수익)
- Entry target이 cash (margin)이면 target × lev = notional.

V21 선물은 자체 lev 3x 이미 적용된 상태. 별개로 C가 lev 3x 추가 적용.
port_eq = V21_fut_alone + C_sleeve_delta (both in same currency unit)
"""
from __future__ import annotations
from bisect import bisect_right
from dataclasses import dataclass
from typing import Any
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')
STRAT_DIR = os.path.join(HERE, 'strat_C_v3')
sys.path.insert(0, HERE)

from c_engine_v5 import run_c_v5, load_coin

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','FDUSD','USDD','PYUSD','USDE','LUSD'}


def load_universe_hist():
    with open(os.path.join(ROOT, 'data', 'historical_universe.json')) as f:
        raw = json.load(f)
    out = {}
    for d, tickers in raw.items():
        bare = [t.replace('-USD', '') for t in tickers]
        bare = [c for c in bare if c not in STABLES]
        out[pd.Timestamp(d).normalize()] = bare
    return out


def list_available_futures():
    avail = set()
    for f in os.listdir(DATA_DIR):
        if f.endswith('_1h.csv'):
            c = f.replace('USDT_1h.csv', '')
            avail.add(c)
    return avail


def load_coin_daily(coins):
    out = {}
    for c in coins:
        df = load_coin(c + 'USDT')
        if df is None: continue
        d = df['Close'].resample('D').last().ffill()
        out[c] = d
    return out


def load_v21_futures():
    """V21 선물 daily equity + V21 현물 cash_ratio (근사 재사용)."""
    fut = pd.read_csv(os.path.join(STRAT_DIR, 'v21_futures_daily.csv'),
                      index_col=0, parse_dates=True)
    # cash_ratio는 V21 현물 값 재사용 (근사, same strategy logic)
    spot = pd.read_csv(os.path.join(STRAT_DIR, 'v21_daily.csv'),
                       index_col=0, parse_dates=True)
    # align
    idx = fut.index.intersection(spot.index)
    df = pd.DataFrame({
        'equity': fut.loc[idx, 'equity'],
        'cash_ratio': spot.loc[idx, 'cash_ratio'],
    })
    df['equity'] = df['equity'] / df['equity'].iloc[0]
    df['v21_ret'] = df['equity'].pct_change().fillna(0)
    df['prev_cash'] = df['cash_ratio'].shift(1).fillna(df['cash_ratio'].iloc[0])
    return df


@dataclass
class Lot:
    coin: str
    cap_rank: int
    entry_date: pd.Timestamp
    exit_date: pd.Timestamp
    entry_px: float
    exit_px: float
    pnl_pct: float
    qty: float = 0.0
    last_px: float = 0.0
    entered: bool = False

    @property
    def value(self) -> float:
        return self.qty * self.last_px


def metrics(eq: pd.Series, bpy=252) -> dict:
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'Sharpe':0,'CAGR':0,'MDD':0,'Cal':0,'Final':0}
    std = rets.std()
    sh = (rets.mean()*bpy)/(std*np.sqrt(bpy)) if std > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days/365.25 if days > 0 else 0.001
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return {'Sharpe':round(float(sh),3),'CAGR':round(float(cagr),4),
            'MDD':round(float(mdd),4),'Cal':round(float(cal),3),
            'Final':round(float(eq.iloc[-1]/eq.iloc[0]),3)}


def simulate_fut(events, coin_daily_close, v21_daily, hist_universe,
                 n_pick=1, cap_per_slot=0.333, universe_size=15,
                 tx_cost=0.003, swap_edge_threshold=1, leverage=3.0):
    """
    Futures version with leverage.

    cap_per_slot: margin 비중 (port 대비)
    실제 notional = margin × leverage
    MTM: qty × (cur - prev) where qty는 notional 기반
    """
    idx = pd.DatetimeIndex(v21_daily.index).sort_values().normalize()
    v21 = v21_daily.copy()
    v21.index = idx
    v21 = v21.loc[~v21.index.duplicated(keep='last')].sort_index()

    hist_dates = sorted(pd.Timestamp(d).normalize() for d in hist_universe)
    hist_map = {pd.Timestamp(d).normalize(): coins for d, coins in hist_universe.items()}
    px_map = {c: s.sort_index() for c, s in coin_daily_close.items()}

    def cap_rank(date, coin):
        pos = bisect_right(hist_dates, date.normalize()) - 1
        if pos < 0: return 10**9
        coins = hist_map[hist_dates[pos]]
        try: return coins.index(coin)
        except ValueError: return 10**9

    def close_px(coin, date, fallback):
        ser = px_map.get(coin)
        if ser is None or ser.empty: return float(fallback)
        px = ser.asof(date)
        return float(px) if pd.notna(px) and px > 0 else float(fallback)

    ev = events.copy()
    ev['entry_date'] = pd.to_datetime(ev['entry_ts']).dt.normalize()
    ev['exit_date'] = pd.to_datetime(ev['exit_ts']).dt.normalize()
    ev['cap_rank'] = [cap_rank(d, c) for d, c in zip(ev['entry_date'], ev['coin'])]
    ev = ev[ev['cap_rank'] < universe_size].sort_values(['entry_ts','cap_rank']).reset_index(drop=True)
    by_day = {d: g.to_dict('records') for d, g in ev.groupby('entry_date', sort=False)}

    lots = []
    c_delta = 0.0
    port_vals = []
    stats = {'n_entries':0,'n_natural_exits':0,'n_swaps':0,'n_shrinks':0,
             'n_expands':0,'n_forced_zero':0,'total_tx':0.0,'max_slots':0,
             'n_liquidations':0}

    for date in v21.index:
        v21_eq = float(v21.at[date,'equity'])
        base_cash_ratio = float(v21.at[date,'prev_cash'])

        # MTM + natural exits
        still_open = []
        for lot in lots:
            if lot.exit_date <= date:
                settle_px = float(lot.exit_px)
                c_delta += lot.qty * (settle_px - lot.last_px)
                tx = lot.qty * settle_px * tx_cost
                c_delta -= tx
                stats['total_tx'] += tx
                stats['n_natural_exits'] += 1
                continue
            px = close_px(lot.coin, date, lot.last_px or lot.entry_px)
            # Liquidation check: 3x short margin, -33% 이상 하락 시 청산
            if lot.entered and lot.entry_px > 0:
                raw_pnl_pct = (px / lot.entry_px - 1.0)
                if raw_pnl_pct * leverage <= -1.0 + 0.05:  # margin wipeout 5% buffer
                    # 청산. margin 전액 손실
                    c_delta -= lot.qty * lot.last_px * (1.0 / leverage)  # margin 손실
                    stats['n_liquidations'] += 1
                    continue
            c_delta += lot.qty * (px - lot.last_px)
            lot.last_px = px
            still_open.append(lot)
        lots = still_open

        # Entry
        open_coins = {lot.coin for lot in lots}
        for row in by_day.get(date, []):
            coin = row['coin']
            if coin in open_coins: continue
            new_lot = Lot(coin=coin, cap_rank=int(row['cap_rank']),
                          entry_date=row['entry_date'], exit_date=row['exit_date'],
                          entry_px=float(row['entry_px']), exit_px=float(row['exit_px']),
                          pnl_pct=float(row['pnl_pct']))
            if len(lots) < n_pick:
                lots.append(new_lot)
                open_coins.add(coin)
                continue
            worst = max(lots, key=lambda x: x.cap_rank)
            if worst.cap_rank - new_lot.cap_rank >= swap_edge_threshold:
                if worst.entered and worst.qty > 0:
                    tx = worst.value * tx_cost
                    c_delta -= tx
                    stats['total_tx'] += tx
                lots.remove(worst)
                lots.append(new_lot)
                open_coins.discard(worst.coin)
                open_coins.add(coin)
                stats['n_swaps'] += 1

        # Allocation
        alloc_base_eq = v21_eq + c_delta
        if alloc_base_eq <= 0:
            port_vals.append(max(alloc_base_eq, 0.0001))
            continue
        # margin per slot
        slot_margin = max(cap_per_slot * alloc_base_eq, 0.0)
        slot_notional = slot_margin * leverage  # 실제 notional
        cash_budget_margin = max(base_cash_ratio * alloc_base_eq, 0.0)
        lots.sort(key=lambda x: (x.cap_rank, x.entry_date, x.coin))
        remaining_margin = cash_budget_margin
        rebalanced = []

        for lot in lots:
            cur_px = lot.last_px if lot.entered else close_px(lot.coin, date, lot.entry_px)
            cur_notional = lot.qty * cur_px if lot.entered else 0.0
            # target margin (slot)
            target_margin = min(slot_margin, max(remaining_margin, 0.0))
            target_notional = target_margin * leverage

            if lot.entered:
                if target_notional < cur_notional - 1e-12:
                    sell_val = cur_notional - target_notional
                    tx = sell_val * tx_cost
                    c_delta -= tx
                    stats['total_tx'] += tx
                    stats['n_shrinks'] += 1
                    lot.qty = target_notional / cur_px if target_notional > 0 and cur_px > 0 else 0.0
                    lot.last_px = cur_px
                elif target_notional > cur_notional + 1e-12:
                    buy_val = target_notional - cur_notional
                    tx = buy_val * tx_cost
                    c_delta -= tx
                    stats['total_tx'] += tx
                    stats['n_expands'] += 1
                    if cur_px > 0:
                        lot.qty += buy_val / cur_px
                    lot.last_px = cur_px
            elif target_notional > 0:
                qty = target_notional / lot.entry_px if lot.entry_px > 0 else 0.0
                tx = target_notional * tx_cost
                mark_px = close_px(lot.coin, date, lot.entry_px)
                c_delta -= tx
                c_delta += qty * (mark_px - lot.entry_px)
                stats['total_tx'] += tx
                stats['n_entries'] += 1
                lot.qty = qty
                lot.last_px = mark_px
                lot.entered = True

            if lot.entered and lot.qty > 0:
                rebalanced.append(lot)
            elif lot.entered:
                stats['n_forced_zero'] += 1

            remaining_margin -= target_margin

        lots = rebalanced
        stats['max_slots'] = max(stats['max_slots'], len(lots))
        port_vals.append(v21_eq + c_delta)

    port_equity = pd.Series(port_vals, index=v21.index, name='port_equity')
    port_equity = port_equity / float(port_equity.iloc[0])

    out = {**stats, **metrics(port_equity, bpy=252), 'final_c_delta':round(float(c_delta),4)}
    return port_equity, out


def extract_events(coins, P):
    rows = []
    for c in coins:
        df = load_coin(c + 'USDT')
        if df is None: continue
        _, evs = run_c_v5(df, **P)
        for e in evs:
            e['coin'] = c
            rows.append(e)
    return pd.DataFrame(rows)


if __name__ == '__main__':
    v21_fut = load_v21_futures()
    hist = load_universe_hist()
    avail = list_available_futures()
    coin_daily = load_coin_daily(avail)
    print(f'V21 선물 단독: {metrics(v21_fut["equity"], bpy=252)}')
    print(f'Events base coin count: {len(avail)}')

    P = {'dip_bars':24, 'dip_thr':-0.20, 'tp':0.12, 'tstop':24}
    events = extract_events(avail, P)
    print(f'Events: {len(events)}')

    print('\n=== Sanity: cap=0.333, n_pick 1/2/3, lev=3 ===')
    for n in [1, 2, 3]:
        for uni in [10, 15, 20]:
            port_eq, stats = simulate_fut(events, coin_daily, v21_fut, hist,
                                           n_pick=n, cap_per_slot=0.333,
                                           universe_size=uni, leverage=3.0)
            m = {k: stats[k] for k in ['Sharpe','CAGR','MDD','Cal']}
            print(f'n={n} uni={uni}: {m} entries={stats["n_entries"]} swaps={stats["n_swaps"]} liq={stats["n_liquidations"]}')
