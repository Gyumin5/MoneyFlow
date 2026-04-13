#!/usr/bin/env python3
"""후보 단위 자유 조합 sweep — 봉주기 묶음 무시하고 9개 후보를 자유롭게 섞음.

지원 모드:
  --mode all-ktuple : k=2..max-k의 모든 EW 조합 (D+D+4h, 4h+4h+2h 등)
  --mode pairwise   : 모든 후보 쌍의 비율 sweep (10pp step)
  --mode custom     : --mix "name1:30,name2:50,name3:20" 직접 지정

출력 CSV: sorted by Sharpe (desc)
"""
import argparse
import csv
import os
import pickle
import sys
from itertools import combinations

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def equity_to_daily_returns(eq):
    eq = eq.dropna()
    r = eq.pct_change().fillna(0)
    return (1 + r).resample('D').prod() - 1


def metrics(rets, bars_per_year=365):
    if len(rets) < 2:
        return dict(Sharpe=0, CAGR=0, MDD=0, Cal=0)
    eq = (1 + rets).cumprod()
    years = len(rets) / bars_per_year
    cagr = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else 0
    peak = eq.cummax()
    dd = (eq / peak - 1).min()
    std = rets.std()
    sharpe = rets.mean() / std * np.sqrt(bars_per_year) if std > 0 else 0
    cal = cagr / abs(dd) if dd != 0 else 0
    return dict(Sharpe=float(sharpe), CAGR=float(cagr),
                MDD=float(dd), Cal=float(cal))


def composite_returns(daily_rets_dict, weights):
    """weights = {name: weight 0~1}, sum=1.0 (정규화 자동)"""
    total_w = sum(weights.values())
    if total_w == 0:
        return None
    norm = {k: v / total_w for k, v in weights.items()}
    series_list = [daily_rets_dict[k] * w for k, w in norm.items() if k in daily_rets_dict]
    if not series_list:
        return None
    df = pd.concat(series_list, axis=1).fillna(0)
    return df.sum(axis=1)


def get_tf(name):
    return name.split('_')[0]


def composition_label(names):
    """예: D_S50.., 4h_S240.. → '1D+1×4h'"""
    tfs = [get_tf(n) for n in names]
    counts = {}
    for t in tfs:
        counts[t] = counts.get(t, 0) + 1
    return '+'.join(f"{c}{t}" for t, c in sorted(counts.items()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['spot', 'futures'])
    ap.add_argument('--task', default='all-ktuple', choices=['all-ktuple', 'pairwise', 'custom'])
    ap.add_argument('--max-k', type=int, default=4, help='all-ktuple 최대 후보수')
    ap.add_argument('--min-k', type=int, default=2)
    ap.add_argument('--mix', default=None, help="custom: 'name1:30,name2:50,...'")
    ap.add_argument('--top', type=int, default=30)
    ap.add_argument('--traces', default=None)
    ap.add_argument('--out', default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    traces_path = args.traces or os.path.join(
        here, f'grid_results/ensemble_{args.mode}_traces.pkl')
    with open(traces_path, 'rb') as f:
        traces = pickle.load(f)

    names = list(traces.keys())
    print(f"[{args.mode}] {len(names)} candidates loaded")

    # 일봉 수익률 사전 계산
    daily = {n: equity_to_daily_returns(traces[n]['equity']) for n in names}

    if args.task == 'custom':
        if not args.mix:
            print("ERROR: --mix required for custom task")
            return
        weights = {}
        for kv in args.mix.split(','):
            k, v = kv.split(':')
            weights[k.strip()] = float(v)
        rets = composite_returns(daily, weights)
        if rets is None:
            print("ERROR: empty composite")
            return
        m = metrics(rets)
        print(f"\nCustom mix: {weights}")
        print(f"  Sharpe {m['Sharpe']:.2f}  CAGR {m['CAGR']:+.1%}  MDD {m['MDD']:+.1%}  Cal {m['Cal']:.2f}")
        return

    rows = []
    if args.task == 'all-ktuple':
        for k in range(args.min_k, args.max_k + 1):
            for combo in combinations(names, k):
                weights = {n: 1.0 for n in combo}
                rets = composite_returns(daily, weights)
                if rets is None:
                    continue
                m = metrics(rets)
                rows.append(dict(
                    k=k, combo='|'.join(combo),
                    composition=composition_label(combo),
                    **m))
        print(f"Generated {len(rows)} EW k-tuple combos (k={args.min_k}..{args.max_k})")

    elif args.task == 'pairwise':
        for n1, n2 in combinations(names, 2):
            for w1 in range(10, 100, 10):
                weights = {n1: w1, n2: 100 - w1}
                rets = composite_returns(daily, weights)
                if rets is None:
                    continue
                m = metrics(rets)
                rows.append(dict(
                    pair=f"{n1}|{n2}", w1=w1, w2=100 - w1,
                    composition=composition_label([n1, n2]),
                    **m))
        print(f"Generated {len(rows)} pairwise ratio combos")

    out_path = args.out or os.path.join(
        here, f'grid_results/ensemble_{args.mode}_{args.task}.csv')
    rows_sh = sorted(rows, key=lambda r: -r['Sharpe'])
    rows_cal = sorted(rows, key=lambda r: -r['Cal'])

    with open(out_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(rows_sh[0].keys()))
        w.writeheader()
        for r in rows_sh:
            w.writerow({k: f'{v:.4f}' if isinstance(v, float) else v for k, v in r.items()})
    print(f"Saved → {out_path}")

    print(f"\n=== Top {args.top} by Sharpe ===")
    print(f"{'comp':<14} {'Sharpe':>7} {'CAGR':>8} {'MDD':>8} {'Cal':>6}  detail")
    for r in rows_sh[:args.top]:
        det = r.get('combo') or r.get('pair', '')
        wd = f" w={r.get('w1','')}/{r.get('w2','')}" if 'w1' in r else ''
        # 후보 이름 단축
        det_short = det.replace('_S', '_S').replace('|', ' + ')
        print(f"{r['composition']:<14} {r['Sharpe']:>7.2f} {r['CAGR']:>+8.1%} {r['MDD']:>+8.1%} {r['Cal']:>6.2f}  {det_short}{wd}")

    print(f"\n=== Top {args.top} by Calmar ===")
    for r in rows_cal[:args.top]:
        det = r.get('combo') or r.get('pair', '')
        wd = f" w={r.get('w1','')}/{r.get('w2','')}" if 'w1' in r else ''
        det_short = det.replace('|', ' + ')
        print(f"{r['composition']:<14} {r['Sharpe']:>7.2f} {r['CAGR']:>+8.1%} {r['MDD']:>+8.1%} {r['Cal']:>6.2f}  {det_short}{wd}")


if __name__ == '__main__':
    main()
