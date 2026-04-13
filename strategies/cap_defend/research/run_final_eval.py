#!/usr/bin/env python3
"""최종 평가: (1) 가드 비교 (2) 50:50 SPOT/FUT 포트 리밸 시뮬.

챔피언:
  SPOT: 4h_S240/M30/L720/Sn120 + 4h_S240/M30/L120/Sn60 + V19 (k=3)
  FUT:  4h_S240/M30/L720/Sn60 (k=1)
  FUT-alt: 4h_S240/M30/L720/Sn120 + 4h_S240/M30/L720/Sn60 (k=2)

가드 변형:
  canary_only  — 현 테스트 조건 (DD/BL/Crash 모두 off)
  v19_full     — DD60/-25%, BL-15%/7d, Crash-10%

포트 리밸:
  none, monthly, quarterly, drift8pp, drift12pp
  각 리밸당 0.4% tx cost (round trip)
"""
import csv
import os
import pickle
import sys
import time
import numpy as np
import pandas as pd

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from backtest_futures_full import load_data, run
from run_ensemble import SingleAccountEngine, combine_targets

START = '2020-10-01'
END = '2026-03-28'
OUT_DIR = os.path.join(_here, 'grid_results')
EQ_PKL = os.path.join(OUT_DIR, 'final_eval_equity.pkl')

BASE_NG = dict(canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
               bl_drop=0, bl_days=0, health_mode='mom2vol', n_snapshots=3,
               vol_mode='daily', vol_threshold=0.05)
# V19 full guard: canary + DD60/-25 + BL-15/7 + Crash default(-10%)
BASE_G = dict(canary_hyst=0.015, drift_threshold=0.10,
              dd_threshold=-0.25, dd_lookback=60,
              bl_drop=-0.15, bl_days=7,
              # crash_threshold default -0.10
              health_mode='mom2vol', n_snapshots=3,
              vol_mode='daily', vol_threshold=0.05)

MEMBER_DEFS = {
    '4h_L720_Sn120': dict(interval='4h', sma_bars=240, mom_short_bars=30, mom_long_bars=720, snap_interval_bars=120),
    '4h_L120_Sn60':  dict(interval='4h', sma_bars=240, mom_short_bars=30, mom_long_bars=120, snap_interval_bars=60),
    '4h_L720_Sn60':  dict(interval='4h', sma_bars=240, mom_short_bars=30, mom_long_bars=720, snap_interval_bars=60),
    'V19':           dict(interval='D',  sma_bars=50,  mom_short_bars=30, mom_long_bars=90,  snap_interval_bars=30),
}

CHAMPIONS = {
    'SPOT_k3':      dict(tx=0.004,  members=['4h_L720_Sn120', '4h_L120_Sn60', 'V19']),
    'FUT_k1':       dict(tx=0.0004, members=['4h_L720_Sn60']),
    'FUT_k2':       dict(tx=0.0004, members=['4h_L720_Sn120', '4h_L720_Sn60']),
}


def gen_trace(data, mem_name, tx, guards):
    cfg = dict(MEMBER_DEFS[mem_name])
    cfg.update(guards)
    interval = cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0, tx_cost=tx,
        start_date=START, end_date=END, _trace=trace, **cfg)
    return trace


def eval_ensemble(bars_1h, funding_1h, traces, members, tx):
    w = {m: 1.0/len(members) for m in members}
    btc = bars_1h['BTC']
    all_dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    sub = {m: traces[m] for m in members}
    combined = combine_targets(sub, w, all_dates)
    eng = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=1.0, tx_cost=tx, stop_kind='none',
        leverage_mode='fixed', per_coin_leverage_mode='none',
    )
    return eng.run(combined)


def metrics_from_equity(eq, yrs):
    if len(eq) < 2 or yrs <= 0:
        return dict(Sharpe=0, CAGR=0, MDD=0, Cal=0)
    cagr = (eq.iloc[-1]/eq.iloc[0]) ** (1/yrs) - 1
    dr = eq.pct_change().dropna()
    # daily freq -> annualize
    sh = (dr.mean()/dr.std() * np.sqrt(365)) if dr.std() > 0 else 0
    peak = eq.cummax()
    mdd = (eq/peak - 1).min()
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return dict(Sharpe=float(sh), CAGR=float(cagr), MDD=float(mdd), Cal=float(cal))


def simulate_5050(eq_spot, eq_fut, rule, tx_rt=0.004):
    """50:50 portfolio w/ rebalancing rule.
    rule: 'none' | 'monthly' | 'quarterly' | 'drift_8' | 'drift_12'
    Charges round-trip tx cost on moved notional.
    """
    idx = eq_spot.index.intersection(eq_fut.index)
    es = eq_spot.loc[idx] / eq_spot.loc[idx].iloc[0]
    ef = eq_fut.loc[idx] / eq_fut.loc[idx].iloc[0]
    port = pd.Series(index=idx, dtype=float)
    # state: dollar in each leg, start 0.5 each
    s = 0.5; f = 0.5
    last_rebal_month = idx[0].to_period('M')
    last_rebal_q = idx[0].to_period('Q')
    ds_prev = es.iloc[0]; df_prev = ef.iloc[0]
    port.iloc[0] = s + f
    for i in range(1, len(idx)):
        ret_s = es.iloc[i]/ds_prev - 1
        ret_f = ef.iloc[i]/df_prev - 1
        s *= (1+ret_s); f *= (1+ret_f)
        ds_prev = es.iloc[i]; df_prev = ef.iloc[i]
        tot = s + f
        do_rebal = False
        if rule == 'monthly':
            cm = idx[i].to_period('M')
            if cm != last_rebal_month:
                do_rebal = True; last_rebal_month = cm
        elif rule == 'quarterly':
            cq = idx[i].to_period('Q')
            if cq != last_rebal_q:
                do_rebal = True; last_rebal_q = cq
        elif rule == 'drift_8':
            w_s = s/tot
            if abs(w_s - 0.5) > 0.08: do_rebal = True
        elif rule == 'drift_12':
            w_s = s/tot
            if abs(w_s - 0.5) > 0.12: do_rebal = True
        if do_rebal:
            target_s = 0.5 * tot
            target_f = 0.5 * tot
            moved = abs(s - target_s) + abs(f - target_f)
            cost = moved * (tx_rt / 2)  # 0.2% one-side per leg
            s = target_s - cost/2
            f = target_f - cost/2
        port.iloc[i] = s + f
    yrs = (idx[-1] - idx[0]).days / 365.25
    m = metrics_from_equity(port, yrs)
    return m, port


def main():
    t0 = time.time()
    print('Loading data: D, 4h, 1h ...', flush=True)
    data = {iv: load_data(iv) for iv in ['D', '4h', '1h']}
    bars_1h, funding_1h = data['1h']

    # Part 1: guard comparison — generate traces for all unique members × 2 guard modes
    members_needed = set()
    for ch in CHAMPIONS.values():
        members_needed.update(ch['members'])

    # Also need separate tx for each: SPOT tx 0.4% vs FUT tx 0.04%
    # Since tx is embedded in trace generation, we run per (member, tx) pair
    # But traces are the same regardless of tx (tx is applied at eval). Actually `run` uses tx_cost
    # only to filter drift/rebal decisions — not to adjust targets. Let me keep it consistent:
    # generate with tx=0 (for pure target stream), then tx applied at eval.
    # Simpler: re-gen per (mode tx, guard). Time: each ~15s * 4 members * 2 guards * 2 tx ≈ 4min.

    rows_guard = []
    equity_store = {}  # key: (champ, guard) -> equity series

    for guard_name, guards in [('canary_only', BASE_NG), ('v19_full', BASE_G)]:
        for champ_name, ch in CHAMPIONS.items():
            tx = ch['tx']
            tkey = f'{champ_name}|{guard_name}'
            print(f'\n[{tkey}] generating traces...', flush=True)
            traces = {}
            for m in ch['members']:
                ts = time.time()
                traces[m] = gen_trace(data, m, tx, guards)
                print(f'  trace {m} ({time.time()-ts:.0f}s)', flush=True)
            ts = time.time()
            res = eval_ensemble(bars_1h, funding_1h, traces, ch['members'], tx)
            eq = res.get('_equity')
            # resample to daily for portfolio sim
            eq_d = eq.resample('D').last().dropna() if eq is not None else None
            equity_store[tkey] = eq_d
            print(f'  eval {(time.time()-ts):.0f}s: Sh{res.get("Sharpe",0):.2f} '
                  f'CAGR{res.get("CAGR",0):+.1%} MDD{res.get("MDD",0):+.1%} '
                  f'Cal{res.get("Cal",0):.2f}', flush=True)
            rows_guard.append(dict(
                champion=champ_name, guard=guard_name, tx=tx,
                members='+'.join(ch['members']),
                Sharpe=res.get('Sharpe',0), CAGR=res.get('CAGR',0),
                MDD=res.get('MDD',0), Cal=res.get('Cal',0),
                Liq=res.get('Liq',0), Stops=res.get('Stops',0),
            ))

    # save
    with open(os.path.join(OUT_DIR, 'final_eval_guards.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_guard[0].keys()))
        w.writeheader(); w.writerows(rows_guard)
    with open(EQ_PKL, 'wb') as f:
        pickle.dump(equity_store, f)

    # Part 2: 50:50 portfolio sim — SPOT_k3 × (FUT_k1, FUT_k2) × 2 guard modes × 5 rebal rules
    rows_port = []
    fut_variants = ['FUT_k1', 'FUT_k2']
    rebal_rules = ['none', 'monthly', 'quarterly', 'drift_8', 'drift_12']
    for guard in ['canary_only', 'v19_full']:
        eq_spot = equity_store[f'SPOT_k3|{guard}']
        for fv in fut_variants:
            eq_fut = equity_store[f'{fv}|{guard}']
            for rule in rebal_rules:
                m, port = simulate_5050(eq_spot, eq_fut, rule)
                rows_port.append(dict(
                    guard=guard, fut_variant=fv, rebal=rule,
                    Sharpe=m['Sharpe'], CAGR=m['CAGR'],
                    MDD=m['MDD'], Cal=m['Cal'],
                ))
                print(f'  5050 {guard} {fv} {rule:<11} Sh{m["Sharpe"]:.2f} '
                      f'CAGR{m["CAGR"]:+.1%} MDD{m["MDD"]:+.1%} '
                      f'Cal{m["Cal"]:.2f}', flush=True)
    with open(os.path.join(OUT_DIR, 'final_eval_portfolio.csv'), 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_port[0].keys()))
        w.writeheader(); w.writerows(rows_port)

    print(f'\nDone. {time.time()-t0:.0f}s. Files in {OUT_DIR}/', flush=True)

    # Summary print
    print('\n=== Guard Comparison ===')
    print(f'{"champion":<12} {"guard":<13} {"Sh":>5} {"CAGR":>7} {"MDD":>7} {"Cal":>5}')
    for r in rows_guard:
        print(f'{r["champion"]:<12} {r["guard"]:<13} {r["Sharpe"]:>5.2f} '
              f'{r["CAGR"]:>+7.1%} {r["MDD"]:>+7.1%} {r["Cal"]:>5.2f}')

    print('\n=== 50:50 Portfolio (SPOT_k3 + FUT variant) ===')
    print(f'{"guard":<13} {"fut":<8} {"rebal":<11} {"Sh":>5} {"CAGR":>7} {"MDD":>7} {"Cal":>5}')
    for r in rows_port:
        print(f'{r["guard"]:<13} {r["fut_variant"]:<8} {r["rebal"]:<11} '
              f'{r["Sharpe"]:>5.2f} {r["CAGR"]:>+7.1%} {r["MDD"]:>+7.1%} {r["Cal"]:>5.2f}')


if __name__ == '__main__':
    main()
