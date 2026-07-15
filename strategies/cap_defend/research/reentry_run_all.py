#!/usr/bin/env python3
"""재진입 22개 변형 전체 실행 + 지표 산출 + window rank-sum + 10-anchor.
결과를 JSON(reentry_results.json) + 표(stdout/reentry_tables.txt)로 저장.
"""
import os, sys, json, time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reentry_harness as H
import unified_backtest as ub

START, END = '2020-10-01', '2026-05-31'
WIN_SIZES = [504, 756, 1008]
STRIDES = [63, 126, 252]
PHASES = [int(i * 217 / 10) for i in range(10)]  # 10-anchor phase shifts

CASH = ('CASH', 'Cash')


def risky_set(tw):
    return frozenset(c for c in tw if c not in CASH and tw.get(c, 0) > 1e-9)


def half_turnover(a, b):
    ks = set(a) | set(b)
    return sum(abs(b.get(k, 0) - a.get(k, 0)) for k in ks) / 2.0


def cash_w(tw):
    return sum(tw.get(k, 0) for k in CASH)


def trace_metrics(trace):
    """trace 기반 파생지표: turnover, 현금체류일, 종목교체일수."""
    turnover = 0.0
    prev_exec = {'CASH': 1.0}
    cash_full = 0     # target 거의 전량 현금
    cash_half = 0     # target 현금 >=50%
    swap_days = 0     # rebal 이면서 risky set 변경
    prev_set = frozenset()
    for row in trace:
        tw = row['target']
        cw = cash_w(tw)
        if cw >= 0.999:
            cash_full += 1
        if cw >= 0.5:
            cash_half += 1
        if row['rebal']:
            turnover += half_turnover(prev_exec, tw)
            prev_exec = dict(tw)
            rs = risky_set(tw)
            if rs != prev_set:
                swap_days += 1
            prev_set = rs
    return dict(turnover=turnover, cash_full=cash_full, cash_half=cash_half,
                swap_days=swap_days)


def forward_metrics(bars, stats, btc_close, btc_idx):
    """재진입 이벤트 telemetry: 지연, forward return, canary OFF율, 동시진입."""
    ev = (stats or {}).get('events', [])
    n = len(ev)
    out = dict(n_events=n)
    if n == 0:
        return out
    delays = [e['empty_age'] for e in ev]
    nsel = [e['n_sel'] for e in ev]
    out['delay_p50'] = float(np.percentile(delays, 50))
    out['delay_p90'] = float(np.percentile(delays, 90))
    out['concurrent_mean'] = float(np.mean(nsel))
    out['concurrent_max'] = int(np.max(nsel))
    # forward EW return of picks + canary OFF within N
    def coin_close_arr(coin):
        df = bars.get(coin)
        return df['Close'] if df is not None else None
    fwd = {5: [], 10: [], 21: [], 31: []}
    canoff = {5: [], 10: [], 21: []}
    for e in ev:
        d = e['date']
        picks = e['picks']
        for N in fwd:
            rets = []
            for c in picks:
                s = coin_close_arr(c)
                if s is None:
                    continue
                ci = s.index.get_indexer([d], method='ffill')[0]
                if ci < 0 or ci + N >= len(s):
                    continue
                p0 = float(s.iloc[ci]); pN = float(s.iloc[ci + N])
                if p0 > 0:
                    rets.append(pN / p0 - 1)
            if rets:
                fwd[N].append(float(np.mean(rets)))
        # canary OFF within N: BTC below SMA42*(1-hyst) at any bar
        bi = btc_idx.get(d, -1)
        for N in canoff:
            if bi < 0 or bi + N >= len(btc_close):
                continue
            off = False
            for k in range(1, N + 1):
                j = bi + k
                sma = float(np.mean(btc_close[max(0, j - 42):j])) if j >= 42 else 0
                if sma > 0 and btc_close[j] < sma * (1 - 0.015):
                    off = True; break
            canoff[N].append(1 if off else 0)
    for N in fwd:
        out[f'fwd{N}_med'] = float(np.median(fwd[N])) if fwd[N] else float('nan')
        if N in (5, 10, 21):
            out[f'fwd{N}_p10'] = float(np.percentile(fwd[N], 10)) if fwd[N] else float('nan')
    for N in canoff:
        out[f'canoff{N}'] = float(np.mean(canoff[N])) if canoff[N] else float('nan')
    # 즉시 왕복거래율 proxy: 5봉 내 canary OFF
    out['roundtrip5'] = out.get('canoff5', float('nan'))
    return out


def empty_dur_metrics(stats):
    ds = (stats or {}).get('empty_durations', [])
    if not ds:
        return dict(emptydur_p50=float('nan'), emptydur_p90=float('nan'),
                    emptydur_max=float('nan'), n_episodes=0)
    return dict(emptydur_p50=float(np.percentile(ds, 50)),
                emptydur_p90=float(np.percentile(ds, 90)),
                emptydur_max=float(np.max(ds)), n_episodes=len(ds))


def cagr_of(eq):
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    return (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else float('nan')


def window_rank_sum(eq_dict):
    common = None
    for s in eq_dict.values():
        common = s.index if common is None else common.intersection(s.index)
    common = sorted(common)
    sums = defaultdict(float); wins = defaultdict(int); nwin = 0
    keys = sorted(eq_dict.keys())
    for size in WIN_SIZES:
        for stride in STRIDES:
            for s_idx in range(0, len(common) - size, stride):
                d0, d1 = common[s_idx], common[s_idx + size - 1]
                cals = {}
                ok = True
                for k in keys:
                    seg = eq_dict[k].loc[d0:d1].dropna()
                    if len(seg) < 30:
                        ok = False; break
                    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                    cagr = (seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1 if yrs > 0 else 0
                    peak = seg.cummax(); mdd = float((seg / peak - 1).min())
                    cals[k] = cagr / abs(mdd) if mdd < 0 else 0
                if not ok:
                    continue
                for r, (mk, _) in enumerate(sorted(cals.items(), key=lambda x: -x[1]), 1):
                    sums[mk] += r
                wins[max(cals, key=cals.get)] += 1
                nwin += 1
    return sums, wins, nwin


def main():
    t0 = time.time()
    print('데이터 로드...', flush=True)
    bars, funding = ub.load_data('D')
    btc_close = bars['BTC']['Close'].values
    btc_idx = {d: i for i, d in enumerate(bars['BTC'].index)}

    results = {}
    equities = {}
    print('== 메인 실행 (cost-on + telemetry) ==', flush=True)
    for name, (cfg, groups) in H.VARIANTS.items():
        res, tr, st = H.run_variant(bars, funding, cfg, START, END, want_stats=True)
        eq = res['_equity']
        equities[name] = eq
        m = dict(groups=groups)
        m.update({k: res[k] for k in ['CAGR', 'MDD', 'Cal', 'Sharpe', 'Trades', 'Rebal']})
        m.update(trace_metrics(tr))
        m.update(empty_dur_metrics(st))
        m.update(forward_metrics(bars, st, btc_close, btc_idx))
        # cost-off CAGR (tx_cost=0)
        res0, _, _ = H.run_variant(bars, funding, cfg, START, END,
                                   want_stats=False, want_trace=False, tx_cost=0.0)
        m['CAGR_nocost'] = res0['CAGR']
        results[name] = m
        print(f'  {name:3s} CAGR={m["CAGR"]:+.3f} MDD={m["MDD"]:+.3f} Cal={m["Cal"]:.3f} '
              f'Sh={m["Sharpe"]:.2f} Trades={m["Trades"]} events={m["n_events"]}', flush=True)

    print('== 10-anchor phase robustness (cost-on, Cal) ==', flush=True)
    for name, (cfg, groups) in H.VARIANTS.items():
        cals = []
        for ph in PHASES:
            res, _, _ = H.run_variant(bars, funding, cfg, START, END,
                                      want_stats=False, want_trace=False,
                                      phase_offset_bars=ph)
            cals.append(res['Cal'])
        results[name]['anchor_cal_mean'] = float(np.mean(cals))
        results[name]['anchor_cal_std'] = float(np.std(cals))
        results[name]['anchor_cal_min'] = float(np.min(cals))
        print(f'  {name:3s} anchorCal mean={np.mean(cals):.3f} std={np.std(cals):.3f} '
              f'min={np.min(cals):.3f}', flush=True)

    print('== window rank-sum ==', flush=True)
    sums, wins, nwin = window_rank_sum(equities)
    for name in results:
        results[name]['ranksum'] = sums[name]
        results[name]['winrate'] = wins[name] / nwin if nwin else 0
    print(f'  windows={nwin}', flush=True)

    out = dict(meta=dict(start=START, end=END, nwin=nwin, phases=PHASES,
                         win_sizes=WIN_SIZES, strides=STRIDES,
                         elapsed=time.time() - t0),
               results=results)
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'reentry_results.json'), 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n완료. {time.time()-t0:.0f}s → reentry_results.json', flush=True)


if __name__ == '__main__':
    main()
