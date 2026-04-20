#!/usr/bin/env python3
"""M3 strict — 사용자 명시 규칙 적용.

1. V21 100% 기준 고정
2. C는 V21 cash 중 최대 hard_cap (30%) 사용
3. n_pick=1 (한 번에 한 코인만)
4. Swap: 더 상위 시총 dip 발동 시 기존 교체
5. V21 cash < hard_cap 시 C 강제 청산
6. Historical universe (월별)
"""
from __future__ import annotations
import os, sys, json
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')
STRAT_DIR = os.path.join(HERE, 'strat_C_v3')
sys.path.insert(0, HERE)
from c_engine_v4 import run_c_v4, load_coin, metrics

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','FDUSD','USDD','PYUSD','USDE','BUSD','LUSD'}


def load_universe_hist():
    with open(os.path.join(ROOT, 'data', 'historical_universe.json')) as f:
        raw = json.load(f)
    out = {}
    for d, tickers in raw.items():
        bare = [t.replace('-USD', '') for t in tickers]
        bare = [c for c in bare if c not in STABLES]
        out[pd.Timestamp(d)] = bare
    return out


def get_cap_rank_at(hist, date, coin):
    """date 시점에 coin의 시총 순위 (낮을수록 상위). 없으면 999."""
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
    return v21


def list_available_futures():
    avail = set()
    for f in os.listdir(DATA_DIR):
        if f.endswith('_1h.csv'):
            c = f.replace('USDT_1h.csv', '')
            avail.add(c)
    return avail


def extract_all_events(coins, **p):
    """각 coin v4 events 추출."""
    all_e = []
    for c in coins:
        df = load_coin(c + 'USDT')
        if df is None: continue
        _, evs = run_c_v4(df, **p)
        for e in evs:
            e['coin'] = c
            all_e.append(e)
    return pd.DataFrame(all_e)


def simulate_strict(v21, events, hist, hard_cap=0.30, universe_size=15,
                    tx_cost=0.003):
    """사용자 명시 규칙 그대로.
    각 event는 daily 수준으로 처리:
    - entry_date에 진입 시도
    - exit_date에 청산
    - 중간에 V21 cash < hard_cap 이면 강제 청산
    - Swap: 포지션 중 새 entry 후보가 더 상위 cap_rank면 교체
    """
    events = events.copy()
    events['entry_date'] = events['entry_ts'].dt.normalize()
    events['exit_date'] = events['exit_ts'].dt.normalize()
    # entry 시점의 cap_rank 계산
    events['cap_r'] = events.apply(lambda r: get_cap_rank_at(hist, r['entry_date'], r['coin']), axis=1)
    # universe filter: entry 시점 시총 Top universe_size 이내만
    events_in_uni = events[events['cap_r'] < universe_size].copy()
    events_in_uni = events_in_uni.sort_values('entry_ts').reset_index(drop=True)

    idx = v21.index
    # 현재 포지션 (딱 1개)
    current_pos = None  # {'coin', 'entry_date', 'exit_date', 'pnl_pct', 'cap_r'}
    port_rets = []
    n_swaps = 0
    n_forced = 0

    # events를 entry_date 기반으로 dict로
    from collections import defaultdict
    events_by_day = defaultdict(list)
    for _, e in events_in_uni.iterrows():
        events_by_day[e['entry_date']].append(e.to_dict())

    for date in idx:
        cash_r = v21.loc[date, 'cash_ratio']
        v21_ret = v21.loc[date, 'v21_ret'] if 'v21_ret' in v21.columns else v21['equity'].pct_change().fillna(0).loc[date]

        day_exit_pnl = 0.0

        # 1) Force close: V21 cash < hard_cap 되면 C 강제 청산
        if current_pos is not None and cash_r < hard_cap:
            day_exit_pnl -= hard_cap * tx_cost  # 강제 청산 비용만 (unrealized는 손실/이익 0 근사)
            current_pos = None
            n_forced += 1

        # 2) 정상 exit 체크
        if current_pos is not None and current_pos['exit_date'] <= date:
            realized = current_pos['pnl_pct'] / 100.0
            day_exit_pnl += hard_cap * (realized - tx_cost)
            current_pos = None

        # 3) Today entries check
        today = events_by_day.get(date, [])
        # Swap 또는 신규 진입
        if today:
            # 시총 상위 순
            today_sorted = sorted(today, key=lambda x: x['cap_r'])
            best_today = today_sorted[0]
            if current_pos is None:
                # 신규 진입 (cash 충분할 때만)
                if cash_r >= hard_cap:
                    current_pos = {'coin': best_today['coin'],
                                   'entry_date': date,
                                   'exit_date': best_today['exit_date'],
                                   'pnl_pct': best_today['pnl_pct'],
                                   'cap_r': best_today['cap_r']}
                    day_exit_pnl -= hard_cap * tx_cost  # 진입 비용
            else:
                # 현재 있음. swap 체크
                if best_today['cap_r'] < current_pos['cap_r']:
                    # 기존 청산 (unrealized 0 가정 보수적) + 신규 진입
                    day_exit_pnl -= hard_cap * tx_cost  # 기존 청산 tx
                    day_exit_pnl -= hard_cap * tx_cost  # 신규 진입 tx
                    current_pos = {'coin': best_today['coin'],
                                   'entry_date': date,
                                   'exit_date': best_today['exit_date'],
                                   'pnl_pct': best_today['pnl_pct'],
                                   'cap_r': best_today['cap_r']}
                    n_swaps += 1
                # else: skip (기존이 더 상위)

        # 4) Port ret
        c_w = hard_cap if current_pos is not None else 0.0
        c_w = min(c_w, cash_r)  # cash보다 커질 수 없음
        port_ret = (1 - c_w) * v21_ret + day_exit_pnl
        port_rets.append(port_ret)

    port_eq = (1 + pd.Series(port_rets, index=idx)).cumprod()
    return port_eq, n_swaps, n_forced


def main():
    v21 = load_v21()
    v21['v21_ret'] = v21['equity'].pct_change().fillna(0)
    hist = load_universe_hist()
    avail = list_available_futures()
    print(f'V21 단독: {metrics(v21["equity"], bpy=252)}')

    P = {'dip_bars':24, 'dip_thr':-0.18, 'tp':0.08, 'tstop':48}
    events = extract_all_events(avail, **P)
    print(f'Events 추출: {len(events)}')

    print('\n=== M3 Strict (n_pick=1, swap ON, force close precise) ===')
    print(f'{"uni":>4} {"cap":>5} {"Sharpe":>7} {"CAGR":>7} {"MDD":>7} {"Cal":>6} {"swaps":>6} {"forced":>7}')
    results = []
    for uni_size in [5, 7, 10, 15, 20, 30]:
        for cap in [0.10, 0.15, 0.20, 0.30, 0.50]:
            port_eq, sw, fc = simulate_strict(v21, events, hist, cap, uni_size)
            m = metrics(port_eq, bpy=252)
            print(f'{uni_size:>4} {cap:>5.2f} {m["Sharpe"]:>7.3f} {m["CAGR"]:>6.1%} {m["MDD"]:>6.1%} {m["Cal"]:>6.2f} {sw:>6} {fc:>7}')
            results.append({'uni': uni_size, 'cap': cap, **m, 'swaps': sw, 'forced': fc})

    rdf = pd.DataFrame(results).sort_values('Cal', ascending=False)
    rdf.to_csv(os.path.join(STRAT_DIR, 'm3_strict.csv'), index=False)
    print(f'\nTop 10 by Cal:')
    print(rdf.head(10).to_string(index=False))

    # 2021 제거
    print('\n=== 2021 제거 (best) ===')
    best = rdf.iloc[0]
    v21_no = v21[v21.index >= pd.Timestamp('2022-01-01')].copy()
    v21_no['equity'] = v21_no['equity'] / v21_no['equity'].iloc[0]
    v21_no['v21_ret'] = v21_no['equity'].pct_change().fillna(0)
    ev_no = events[events['entry_ts'] >= pd.Timestamp('2022-01-01')]
    port_no, sw_no, fc_no = simulate_strict(v21_no, ev_no, hist, best['cap'], int(best['uni']))
    print(f'  V21 단독 2022+: {metrics(v21_no["equity"], bpy=252)}')
    print(f'  V21+C uni={int(best["uni"])} cap={best["cap"]:.0%} 2022+: {metrics(port_no, bpy=252)} (swaps={sw_no}, forced={fc_no})')


if __name__ == '__main__':
    main()
