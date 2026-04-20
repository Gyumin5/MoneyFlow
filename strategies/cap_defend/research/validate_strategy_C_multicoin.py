#!/usr/bin/env python3
"""Strategy C multi-coin portfolio simulator.

각 1h 봉마다 여러 코인 dip 시그널 체크.
- n_pick: 동시 최대 보유 수
- 선정 기준: deepest dip / cap-rank / volume / z-score
- 비중: EW (1/n_pick 씩, 전체 C capital 기준)
- 포지션 가득 차면 새 시그널 skip
- 개별 코인 청산: +TP 또는 time_stop
- 보수 체결: 매수 High / 매도 Open / TX 0.3%
"""
from __future__ import annotations
import os
import numpy as np
import pandas as pd

HERE = os.path.abspath(os.path.dirname(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
DATA_DIR = os.path.join(ROOT, 'data', 'futures')

START = '2020-10-01'
END = '2026-03-30'
TX = 0.003

# Top 10 by market cap (rough order)
UNIVERSE_TOP10 = ['BTCUSDT','ETHUSDT','SOLUSDT','XRPUSDT','BNBUSDT',
                  'DOGEUSDT','ADAUSDT','AVAXUSDT','LINKUSDT','DOTUSDT']
UNIVERSE_TOP5 = UNIVERSE_TOP10[:5]
UNIVERSE_TOP20 = UNIVERSE_TOP10 + ['MATICUSDT','LTCUSDT','BCHUSDT','TRXUSDT','NEARUSDT',
                                    'ATOMUSDT','APTUSDT','ARBUSDT','OPUSDT','SUIUSDT']


def load_coin(sym, interval='1h'):
    path = os.path.join(DATA_DIR, f'{sym}_{interval}.csv')
    if not os.path.isfile(path): return None
    df = pd.read_csv(path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.set_index('Date')
    df = df.loc[START:END].copy()
    return df


def prep_universe(coins, dip_bars):
    data = {}
    for c in coins:
        df = load_coin(c)
        if df is None or len(df) < 1000:
            continue
        df['dip_pct'] = df['Close'] / df['Close'].shift(dip_bars) - 1.0
        data[c] = df
    return data


def run_multicoin(universe, dip_bars, dip_threshold, take_profit, time_stop_bars,
                  n_pick, select_method, lev=1.0, tx=TX, cap_ranks=None):
    """Multi-coin portfolio simulator."""
    data = prep_universe(universe, dip_bars)
    if not data: return None, None
    # 공통 시간 인덱스 (intersect)
    idx = None
    for df in data.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    idx = idx.sort_values()
    if len(idx) < 100: return None, None

    cap_ranks = cap_ranks or {c: i for i, c in enumerate(universe)}
    eq = 10000.0
    equity = []
    # open position per coin: {coin: {'entry_px', 'weight', 'bars_held', 'entry_ts'}}
    positions = {}

    coins_list = list(data.keys())
    events = []

    for i, ts in enumerate(idx):
        # 청산 (bar open 또는 TP)
        to_close = []
        for c, p in positions.items():
            row = data[c].loc[ts]
            sell_px = row['Open']
            pnl = sell_px / p['entry_px'] - 1.0
            if pnl >= take_profit or p['bars_held'] >= time_stop_bars:
                # close this coin
                eq_contrib = p['weight'] * (1 + lev * pnl - tx)
                # 원래 해당 weight만큼 이미 eq에서 차감됐으므로 re-add
                eq += p['capital_allocated'] * (lev * pnl - tx)
                to_close.append((c, pnl, row))
                events.append({'coin': c, 'entry_ts': p['entry_ts'], 'exit_ts': ts,
                              'entry_px': p['entry_px'], 'exit_px': sell_px,
                              'pnl_pct': round(pnl*100, 2),
                              'bars_held': p['bars_held'],
                              'reason': 'TP' if pnl >= take_profit else 'timeout'})
        for c, _, _ in to_close:
            del positions[c]

        # 현재 dip 시그널 체크
        n_open = len(positions)
        if n_open < n_pick:
            candidates = []
            for c, df in data.items():
                if c in positions: continue
                if ts not in df.index: continue
                # prev bar 시그널
                prev_ts_idx = df.index.get_loc(ts) - 1
                if prev_ts_idx < 0: continue
                prev = df.iloc[prev_ts_idx]
                dip = prev['dip_pct']
                if pd.isna(dip) or dip > dip_threshold: continue
                # add candidate
                candidates.append({'coin': c, 'dip': dip, 'cap_rank': cap_ranks.get(c, 999),
                                  'row': df.loc[ts]})
            # 선정
            if candidates:
                if select_method == 'deepest':
                    candidates.sort(key=lambda x: x['dip'])  # 가장 음수 큰 것
                elif select_method == 'cap':
                    candidates.sort(key=lambda x: x['cap_rank'])
                elif select_method == 'zscore':
                    # 여기선 단순하게 deepest와 동일
                    candidates.sort(key=lambda x: x['dip'])
                else:  # default
                    candidates.sort(key=lambda x: x['dip'])

                available_slots = n_pick - n_open
                picks = candidates[:available_slots]
                for pick in picks:
                    c = pick['coin']; row = pick['row']
                    # capital 할당: 균등 분배 가정 (전체 eq / n_pick 슬롯, 현재 n_open+1)
                    # 단순화: 각 슬롯에 (eq / n_pick) 할당
                    slot_cap = eq / n_pick
                    buy_px = row['High']  # 보수 체결
                    positions[c] = {
                        'entry_px': buy_px,
                        'entry_ts': ts,
                        'bars_held': 0,
                        'weight': 1.0 / n_pick,
                        'capital_allocated': slot_cap,
                    }
                    # 진입 비용
                    eq -= slot_cap * tx

        # 포지션 유지: bar 수익 반영
        for c, p in positions.items():
            row = data[c].loc[ts]
            prev_close_idx = data[c].index.get_loc(ts) - 1
            if prev_close_idx < 0: continue
            if p['bars_held'] == 0:
                # 진입 봉: entry_px(High) → Close
                bar_ret = row['Close'] / p['entry_px'] - 1.0
            else:
                prev_close = data[c].iloc[prev_close_idx]['Close']
                bar_ret = row['Close'] / prev_close - 1.0
            eq += p['capital_allocated'] * lev * bar_ret
            # update capital_allocated? 단순화: 초기 할당 유지 (drift 무시)
            p['bars_held'] += 1

        if eq < 0: eq = 0
        equity.append(eq)

    eq_series = pd.Series(equity, index=idx)
    return eq_series, events


def metrics(eq):
    rets = eq.pct_change().dropna()
    if len(rets) == 0 or eq.iloc[-1] <= 0:
        return {'Sharpe': 0, 'CAGR': 0, 'MDD': 0, 'Cal': 0, 'Final': 0}
    bpy = 24*365
    sh = (rets.mean() * bpy) / (rets.std() * np.sqrt(bpy)) if rets.std() > 0 else 0
    days = (eq.index[-1] - eq.index[0]).days
    years = days / 365.25 if days > 0 else 0.001
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/years) - 1
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return {'Sharpe': round(sh, 3), 'CAGR': round(cagr, 4),
            'MDD': round(mdd, 4), 'Cal': round(cal, 3), 'Final': round(eq.iloc[-1]/eq.iloc[0], 3)}


def main():
    print('=== Multi-coin Strategy C Portfolio Simulator ===')
    BEST = {'dip_bars': 24, 'dip_threshold': -0.15, 'take_profit': 0.08, 'time_stop_bars': 24}

    rows = []
    for uni_name, uni in [('Top5', UNIVERSE_TOP5), ('Top10', UNIVERSE_TOP10), ('Top20', UNIVERSE_TOP20)]:
        for n_pick in [1, 2, 3, 5]:
            for method in ['deepest', 'cap']:
                eq, events = run_multicoin(uni, **BEST, n_pick=n_pick, select_method=method)
                if eq is None or len(eq) == 0: continue
                m = metrics(eq)
                m.update({'universe': uni_name, 'n_pick': n_pick, 'method': method,
                         'events': len(events)})
                rows.append(m)
                print(f'  {uni_name} n_pick={n_pick} method={method}: Sharpe={m["Sharpe"]} CAGR={m["CAGR"]:.2%} MDD={m["MDD"]:.2%} Cal={m["Cal"]} events={len(events)}')

    rdf = pd.DataFrame(rows).sort_values('Sharpe', ascending=False)
    print(f'\nTop 10 by Sharpe:')
    cols = ['universe','n_pick','method','events','Sharpe','CAGR','MDD','Cal','Final']
    print(rdf[cols].head(10).to_string(index=False))

    out = os.path.join(HERE, 'strat_C_v3', 'multicoin_sweep.csv')
    rdf.to_csv(out, index=False)
    print(f'\nSaved: {out}')

    # V21 앙상블 평가 (최고 multi-coin 구성으로)
    best_row = rdf.iloc[0]
    print(f'\n=== V21 + 최고 multi-coin C 앙상블 ===')
    print(f'Best C: {best_row["universe"]} n_pick={best_row["n_pick"]} method={best_row["method"]}')
    uni = {'Top5': UNIVERSE_TOP5, 'Top10': UNIVERSE_TOP10, 'Top20': UNIVERSE_TOP20}[best_row['universe']]
    eq_c, _ = run_multicoin(uni, **BEST, n_pick=int(best_row['n_pick']), select_method=best_row['method'])
    eq_c_daily = eq_c.resample('D').last().ffill()

    import sys
    sys.path.insert(0, HERE)
    from phase4_3asset import build_ensemble_full_equity
    spot_top = pd.read_csv(os.path.join(HERE, 'phase3_10x', 'spot_top.csv'))
    ens = spot_top[spot_top['ensemble_tag'] == 'ENS_spot_k3_4b270476'].iloc[0]
    v21_eq = build_ensemble_full_equity(ens)
    if isinstance(v21_eq.index, pd.DatetimeIndex) and v21_eq.index.tz is not None:
        v21_eq = v21_eq.copy(); v21_eq.index = v21_eq.index.tz_localize(None)

    common = eq_c_daily.index.intersection(v21_eq.index)
    c = (eq_c_daily.loc[common] / eq_c_daily.loc[common].iloc[0])
    v = (v21_eq.loc[common] / v21_eq.loc[common].iloc[0])
    c_r = c.pct_change().dropna(); v_r = v.pct_change().dropna()
    cm = c_r.index.intersection(v_r.index)
    c_r = c_r.loc[cm]; v_r = v_r.loc[cm]
    print(f'V21 vs MultiC corr: {c_r.corr(v_r):.4f}')

    for cap in [0.05, 0.10, 0.15, 0.20, 0.30]:
        port_r = (1-cap) * v_r + cap * c_r
        port_eq = (1 + port_r).cumprod()
        rets = port_eq.pct_change().dropna()
        sh = (rets.mean()*252) / (rets.std() * np.sqrt(252)) if rets.std() > 0 else 0
        days = (port_eq.index[-1] - port_eq.index[0]).days
        cagr = (port_eq.iloc[-1] / port_eq.iloc[0]) ** (365.25/days) - 1
        mdd = (port_eq / port_eq.cummax() - 1).min()
        cal = cagr / abs(mdd) if mdd < 0 else 0
        print(f'  C {cap:.0%}: Sharpe={sh:.3f} CAGR={cagr:.2%} MDD={mdd:.2%} Cal={cal:.3f}')


if __name__ == '__main__':
    main()
