#!/usr/bin/env python3
"""M3 with Dynamic Universe — historical_universe.json 기반.

매월 1일 Top N 재선정 (시총 기준)
스테이블코인 제외
바이낸스 선물 상장 코인만
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

STABLES = {'USDT','USDC','BUSD','DAI','TUSD','FDUSD','USDD','PYUSD','USDE'}


def load_universe_history():
    with open(os.path.join(ROOT, 'data', 'historical_universe.json')) as f:
        raw = json.load(f)
    # Date str → list of bare tickers (stablecoin 제외)
    clean = {}
    for d, tickers in raw.items():
        bare = [t.replace('-USD', '') for t in tickers]
        bare = [c for c in bare if c not in STABLES]
        clean[pd.Timestamp(d)] = bare
    return clean


def get_universe_at(hist, date, top_n=10, available_coins=None):
    """해당 date에 유효한 Top N universe 반환.
    hist: {Timestamp: [coin,...]}
    available_coins: 실제 데이터 있는 코인 set (e.g. bina 선물 listed)
    """
    # 가장 최근 (or 같은) hist date 찾기
    valid_dates = [d for d in hist.keys() if d <= date]
    if not valid_dates:
        return []
    latest = max(valid_dates)
    tops = hist[latest]
    if available_coins is not None:
        tops = [c for c in tops if c in available_coins]
    return tops[:top_n]


def load_v21():
    v21 = pd.read_csv(os.path.join(STRAT_DIR, 'v21_daily.csv'), index_col=0, parse_dates=True)
    v21['equity'] = v21['equity'] / v21['equity'].iloc[0]
    return v21


def list_available_futures():
    """바이낸스 선물 데이터 있는 코인 set"""
    avail = set()
    for f in os.listdir(DATA_DIR):
        if f.endswith('_1h.csv'):
            c = f.replace('USDT_1h.csv', '')
            avail.add(c)
    return avail


def main():
    v21 = load_v21()
    hist = load_universe_history()
    avail = list_available_futures()
    print(f'V21 단독: {metrics(v21["equity"], bpy=252)}')
    print(f'바이낸스 선물 available coins: {len(avail)}')

    # 코인별 c_equity 사전 계산 (1h 데이터 있는 것만)
    P = {'dip_bars':24, 'dip_thr':-0.18, 'tp':0.08, 'tstop':48}
    coin_eq_cache = {}
    for c in avail:
        df = load_coin(c + 'USDT')
        if df is None or len(df) < 1000: continue
        eq, _ = run_c_v4(df, **P)
        eq = eq / eq.iloc[0]
        coin_eq_cache[c] = eq.resample('D').last().ffill()
    print(f'C equity 계산 완료: {len(coin_eq_cache)}코인')

    # Daily universe: 매월 1일 기준 Top N
    date_idx = v21.index
    universe_by_date = {}
    for date in date_idx:
        top = get_universe_at(hist, date, top_n=20, available_coins=set(coin_eq_cache.keys()))
        universe_by_date[date] = top

    # Daily C return: 매일 "universe 내 코인의 active 수로 나눈 avg"
    def compute_c_ret_daily(top_n):
        """각 date별 universe(top_n) 내 코인들 active-only 평균 return."""
        c_ret_list = []
        c_active_list = []
        for date in date_idx:
            uni = get_universe_at(hist, date, top_n=top_n, available_coins=set(coin_eq_cache.keys()))
            if not uni:
                c_ret_list.append(0.0)
                c_active_list.append(0.0)
                continue
            # 각 코인의 daily ret
            rets = []
            for c in uni:
                if c not in coin_eq_cache: continue
                eq = coin_eq_cache[c]
                if date not in eq.index: continue
                pos = eq.index.get_loc(date)
                if pos == 0: continue
                r = eq.iloc[pos] / eq.iloc[pos-1] - 1
                if abs(r) > 1e-10:  # active
                    rets.append(r)
            if rets:
                c_ret_list.append(sum(rets) / len(rets))
                c_active_list.append(1.0)
            else:
                c_ret_list.append(0.0)
                c_active_list.append(0.0)
        return pd.Series(c_ret_list, index=date_idx), pd.Series(c_active_list, index=date_idx)

    # Sweep N × cap
    print(f'\n=== Dynamic Historical Universe × Cap (best param) ===')
    print(f'{"N":>3} {"cap":>5} {"Sharpe":>7} {"CAGR":>7} {"MDD":>7} {"Cal":>6} {"active일":>8}')
    all_results = []
    for N in [1, 3, 5, 7, 10, 15, 20]:
        c_ret, c_act = compute_c_ret_daily(N)
        c_eq = (1 + c_ret).cumprod()
        active_days = int(c_act.sum())
        for cap in [0.10, 0.20, 0.30, 0.50]:
            v21_ret = v21['equity'].pct_change().fillna(0)
            c_w = np.minimum(cap, v21['cash_ratio']) * c_act.values
            port_r = (1 - c_w) * v21_ret + c_w * c_ret
            port_eq = (1 + port_r).cumprod()
            m = metrics(port_eq, bpy=252)
            print(f"{N:>3} {cap:>5.2f} {m['Sharpe']:>7.3f} {m['CAGR']:>6.1%} {m['MDD']:>6.1%} {m['Cal']:>6.2f} {active_days:>8d}")
            all_results.append({'N': N, 'cap': cap, **m, 'active_days': active_days})

    rdf = pd.DataFrame(all_results).sort_values('Cal', ascending=False)
    rdf.to_csv(os.path.join(STRAT_DIR, 'm3_hist_universe.csv'), index=False)
    print(f'\nTop 10 by Cal:')
    print(rdf.head(10).to_string(index=False))

    # 2021 제거
    print('\n=== 2021 제거 (best config) ===')
    best = rdf.iloc[0]
    N = int(best['N']); cap = float(best['cap'])
    c_ret, c_act = compute_c_ret_daily(N)
    v21_2 = v21[v21.index >= pd.Timestamp('2022-01-01')].copy()
    v21_2['equity'] = v21_2['equity'] / v21_2['equity'].iloc[0]
    v21_ret2 = v21_2['equity'].pct_change().fillna(0)
    c_ret2 = c_ret.loc[v21_2.index]
    c_act2 = c_act.loc[v21_2.index]
    c_w2 = np.minimum(cap, v21_2['cash_ratio']) * c_act2.values
    port_r2 = (1 - c_w2) * v21_ret2 + c_w2 * c_ret2
    port_eq2 = (1 + port_r2).cumprod()
    print(f'  V21 단독 2022+: {metrics(v21_2["equity"], bpy=252)}')
    print(f'  V21+C N={N} cap={cap:.0%} 2022+: {metrics(port_eq2, bpy=252)}')


if __name__ == '__main__':
    main()
