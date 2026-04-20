#!/usr/bin/env python3
"""M3 multi-slot — 사용자 최종 규칙:
- n_pick 여러 개 가능
- 각 slot 개별 cap 할당 (예: cap=1/3 × n_pick=3)
- 시총 상위 순서로 남은 cash에서 차례대로 가져감
- V21 cash 부족 시 강제 청산 대신 뒤 slot부터 자동 축소 (비중 동적 조정)
- Swap: 기존 보유 중인 것보다 더 상위 시총 dip 발동 시 교체
"""
from __future__ import annotations
import os, sys, json
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')
STRAT_DIR = os.path.join(HERE, 'strat_C_v3')
sys.path.insert(0, HERE)
from c_engine_v4 import run_c_v4, load_coin, metrics

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','FDUSD','USDD','PYUSD','USDE','LUSD'}


def load_universe_hist():
    with open(os.path.join(ROOT, 'data', 'historical_universe.json')) as f:
        raw = json.load(f)
    out = {}
    for d, tickers in raw.items():
        bare = [t.replace('-USD', '') for t in tickers]
        bare = [c for c in bare if c not in STABLES]
        out[pd.Timestamp(d)] = bare
    return out


def get_cap_rank(hist, date, coin):
    valid = [d for d in hist.keys() if d <= date]
    if not valid: return 999
    latest = max(valid)
    tops = hist[latest]
    try:
        return tops.index(coin)
    except ValueError:
        return 999


def load_v21():
    v21 = pd.read_csv(os.path.join(STRAT_DIR, 'v21_daily.csv'), index_col=0, parse_dates=True)
    v21['equity'] = v21['equity'] / v21['equity'].iloc[0]
    v21['v21_ret'] = v21['equity'].pct_change().fillna(0)
    return v21


def list_available_futures():
    avail = set()
    for f in os.listdir(DATA_DIR):
        if f.endswith('_1h.csv'):
            c = f.replace('USDT_1h.csv', '')
            avail.add(c)
    return avail


def extract_events(coins, **p):
    rows = []
    for c in coins:
        df = load_coin(c + 'USDT')
        if df is None: continue
        _, evs = run_c_v4(df, **p)
        for e in evs:
            e['coin'] = c
            rows.append(e)
    return pd.DataFrame(rows)


def simulate_multi_slot(v21, events, hist, n_pick, cap_per_slot, universe_size,
                        tx_cost=0.003):
    """
    - 매일 active positions sort by cap_rank (상위 우선)
    - slot 할당: remaining_cash = cash_r, 각 position.slot = min(cap_per_slot, remaining_cash)
    - swap: new dip 발동 시 n_pick 꽉 찼으면 가장 하위 cap_rank 포지션과 교체 (새 것이 더 상위면)
    - 강제 청산 없음 (cash 변동은 자동으로 뒤 slot 축소)
    - tx: 진입/청산/swap에 각 tx_cost
    """
    events = events.copy()
    events['entry_date'] = events['entry_ts'].dt.normalize()
    events['exit_date'] = events['exit_ts'].dt.normalize()
    events['cap_r'] = events.apply(lambda r: get_cap_rank(hist, r['entry_date'], r['coin']), axis=1)
    events = events[events['cap_r'] < universe_size]
    events = events.sort_values('entry_ts').reset_index(drop=True)

    events_by_day = defaultdict(list)
    for _, e in events.iterrows():
        events_by_day[e['entry_date']].append(e.to_dict())

    idx = v21.index
    positions = []
    port_rets = []
    n_swaps = 0
    n_forced = 0

    for date in idx:
        cash_r = v21.loc[date, 'cash_ratio']
        v21_ret = v21.loc[date, 'v21_ret']

        # 1) exit (정상 TP or timeout)
        still_open = []
        day_pnl_contrib = 0.0  # 오늘 실현 pnl 기여 (port_ret에 더함)
        tx_contrib = 0.0
        for p in positions:
            if p['exit_date'] <= date:
                realized = p['pnl_pct'] / 100.0
                day_pnl_contrib += p['slot_alloc'] * realized
                tx_contrib += p['slot_alloc'] * tx_cost  # 청산 tx
            else:
                still_open.append(p)
        positions = still_open

        # 2) Today entries: swap 또는 신규
        today = events_by_day.get(date, [])
        today_sorted = sorted(today, key=lambda x: x['cap_r'])
        open_coins = {p['coin'] for p in positions}
        for ev in today_sorted:
            if ev['coin'] in open_coins: continue
            if len(positions) < n_pick:
                # 신규 진입
                positions.append({
                    'coin': ev['coin'], 'entry_date': date, 'exit_date': ev['exit_date'],
                    'pnl_pct': ev['pnl_pct'], 'cap_r': ev['cap_r'], 'slot_alloc': 0.0,
                })
                tx_contrib += cap_per_slot * tx_cost  # 진입 tx (근사)
            else:
                # swap: 가장 하위 cap_rank와 비교
                worst = max(positions, key=lambda p: p['cap_r'])
                if ev['cap_r'] < worst['cap_r']:
                    # 기존 청산 tx + 신규 진입 tx
                    tx_contrib += worst['slot_alloc'] * tx_cost * 2  # (out+in)
                    positions.remove(worst)
                    positions.append({
                        'coin': ev['coin'], 'entry_date': date, 'exit_date': ev['exit_date'],
                        'pnl_pct': ev['pnl_pct'], 'cap_r': ev['cap_r'], 'slot_alloc': 0.0,
                    })
                    n_swaps += 1
                    open_coins.add(ev['coin'])

        # 3) slot allocation: cap_rank 순서로 cash 할당
        positions.sort(key=lambda p: p['cap_r'])
        remaining = cash_r
        for p in positions:
            alloc = min(cap_per_slot, remaining)
            p['slot_alloc'] = max(0.0, alloc)
            remaining -= alloc
            if remaining <= 0: remaining = 0

        # 4) port ret
        c_w = sum(p['slot_alloc'] for p in positions)
        c_w = min(c_w, cash_r)
        port_ret = (1 - c_w) * v21_ret + day_pnl_contrib - tx_contrib
        port_rets.append(port_ret)

    port_eq = (1 + pd.Series(port_rets, index=idx)).cumprod()
    return port_eq, n_swaps, n_forced


def main():
    v21 = load_v21()
    hist = load_universe_hist()
    avail = list_available_futures()
    print(f'V21 단독: {metrics(v21["equity"], bpy=252)}')

    P = {'dip_bars':24, 'dip_thr':-0.18, 'tp':0.08, 'tstop':48}
    events = extract_events(avail, **P)
    print(f'Events: {len(events)}')

    print('\n=== Multi-slot M3 sweep ===')
    print(f'{"uni":>4} {"n":>3} {"cap/s":>6} {"total":>6} {"Sharpe":>7} {"CAGR":>7} {"MDD":>7} {"Cal":>6} {"swaps":>6}')
    results = []
    # 다양한 n_pick × cap_per_slot 조합
    for uni in [10, 15, 20, 30]:
        for n_pick in [1, 2, 3, 5]:
            # total cap = 1.0 (full cash 활용) 또는 고정
            for total_cap in [0.30, 0.50, 1.00]:
                cap_per_slot = total_cap / n_pick
                port_eq, sw, fc = simulate_multi_slot(v21, events, hist,
                                                       n_pick, cap_per_slot, uni)
                m = metrics(port_eq, bpy=252)
                print(f'{uni:>4} {n_pick:>3} {cap_per_slot:>6.3f} {total_cap:>6.2f} '
                      f'{m["Sharpe"]:>7.3f} {m["CAGR"]:>6.1%} {m["MDD"]:>6.1%} {m["Cal"]:>6.2f} {sw:>6}')
                results.append({'uni': uni, 'n_pick': n_pick, 'cap_per_slot': cap_per_slot,
                                'total_cap': total_cap, **m, 'swaps': sw})

    rdf = pd.DataFrame(results).sort_values('Cal', ascending=False)
    rdf.to_csv(os.path.join(STRAT_DIR, 'm3_multi_slot.csv'), index=False)
    print(f'\nTop 15 by Cal:')
    print(rdf.head(15).to_string(index=False))

    # 2021 제거 (best)
    print('\n=== 2021 제거 (best config) ===')
    best = rdf.iloc[0]
    v21_no = v21[v21.index >= pd.Timestamp('2022-01-01')].copy()
    v21_no['equity'] = v21_no['equity'] / v21_no['equity'].iloc[0]
    v21_no['v21_ret'] = v21_no['equity'].pct_change().fillna(0)
    ev_no = events[events['entry_ts'] >= pd.Timestamp('2022-01-01')]
    port_no, sw, _ = simulate_multi_slot(v21_no, ev_no, hist,
                                          int(best['n_pick']), float(best['cap_per_slot']),
                                          int(best['uni']))
    print(f'  V21 단독 2022+: {metrics(v21_no["equity"], bpy=252)}')
    print(f'  V21+C best 2022+: {metrics(port_no, bpy=252)} swaps={sw}')


if __name__ == '__main__':
    main()
