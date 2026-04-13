#!/usr/bin/env python3
"""앙상블 비율 sweep — trace pickle을 불러 D/4h/2h 비율 조합을 모두 평가.

방식:
  - 각 후보의 일일 수익률 r_i(t) 계산
  - ensemble_ret(t) = sum_i w_i * r_i(t) (분리계좌 가정, 매일 weight 유지)
  - 봉주기별 후보는 EW로 묶어 봉주기 통합 수익률 R_TF(t) 생성
  - TF 비율 sweep: w_D + w_4h + w_2h = 100, 10pp simplex (pairwise 포함)

출력 CSV: w_D, w_4h, w_2h, Sharpe, CAGR, MDD, Cal
"""
import argparse
import csv
import os
import pickle
import sys
from itertools import product

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def equity_to_returns(eq):
    eq = eq.dropna()
    return eq.pct_change().fillna(0)


def metrics_from_returns(rets, bars_per_year=365):
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


def build_tf_returns(traces, candidates_by_tf):
    """봉주기별 EW 수익률 시계열 생성. 일봉 단위로 통일."""
    tf_rets = {}
    for tf, names in candidates_by_tf.items():
        rets_list = []
        for n in names:
            if n not in traces:
                continue
            eq = traces[n]['equity']
            r = equity_to_returns(eq)
            # 일봉 단위로 리샘플 (intraday는 일별 누적수익률로)
            r_daily = (1 + r).resample('D').prod() - 1
            rets_list.append(r_daily)
        if not rets_list:
            continue
        df = pd.concat(rets_list, axis=1).fillna(0)
        tf_rets[tf] = df.mean(axis=1)  # EW
    return tf_rets


def correlation_matrix(traces, names):
    """각 후보 일일 수익률 상관계수 행렬."""
    cols = []
    valid = []
    for n in names:
        if n not in traces:
            continue
        r = equity_to_returns(traces[n]['equity'])
        r_d = (1 + r).resample('D').prod() - 1
        cols.append(r_d)
        valid.append(n)
    df = pd.concat(cols, axis=1).fillna(0)
    df.columns = valid
    return df.corr()


def cash_corr_matrix(traces, names):
    """CASH 시계열 상관계수 (모두 함께 현금화하면 분산효과 없음)."""
    cols = []
    valid = []
    for n in names:
        if n not in traces or 'cash' not in traces[n]:
            continue
        c = traces[n]['cash']
        c_d = c.resample('D').last().ffill()
        cols.append(c_d)
        valid.append(n)
    df = pd.concat(cols, axis=1).fillna(method='ffill').fillna(0)
    df.columns = valid
    return df.corr()


def sweep_triples(tf_rets, step=10):
    """w_D + w_4h + w_2h = 100, step 단위 simplex 전수."""
    rows = []
    tfs = ['D', '4h', '2h']
    if not all(t in tf_rets for t in tfs):
        return rows
    # 공통 인덱스
    df = pd.concat([tf_rets[t] for t in tfs], axis=1).fillna(0)
    df.columns = tfs

    n = 100 // step
    for i, j in product(range(n + 1), repeat=2):
        k = n - i - j
        if k < 0:
            continue
        wD, w4, w2 = i * step / 100, j * step / 100, k * step / 100
        rets = df['D'] * wD + df['4h'] * w4 + df['2h'] * w2
        m = metrics_from_returns(rets)
        rows.append(dict(w_D=int(wD * 100), w_4h=int(w4 * 100), w_2h=int(w2 * 100), **m))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['spot', 'futures'])
    ap.add_argument('--traces', default=None)
    ap.add_argument('--candidates', default=None)
    ap.add_argument('--step', type=int, default=10, help='ratio step pp (default 10)')
    ap.add_argument('--out-prefix', default=None)
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    traces_path = args.traces or os.path.join(
        here, f'grid_results/ensemble_{args.mode}_traces.pkl')
    cand_path = args.candidates or os.path.join(here, 'configs/ensemble_candidates.json')
    out_prefix = args.out_prefix or os.path.join(
        here, f'grid_results/ensemble_{args.mode}')

    import json
    with open(cand_path) as f:
        cfg_all = json.load(f)
    cand_by_tf = {tf: [c['name'] for c in cfg_all[args.mode][tf]]
                  for tf in ['D', '4h', '2h']}

    with open(traces_path, 'rb') as f:
        traces = pickle.load(f)

    print(f"[{args.mode}] traces loaded: {len(traces)}")
    for tf, names in cand_by_tf.items():
        present = [n for n in names if n in traces]
        print(f"  {tf}: {len(present)}/{len(names)}  {present}")

    # 1) 후보 간 상관계수
    all_names = sum(cand_by_tf.values(), [])
    cor = correlation_matrix(traces, all_names)
    cor_path = f'{out_prefix}_corr_returns.csv'
    cor.to_csv(cor_path)
    print(f"\nReturn correlation → {cor_path}")
    print(cor.round(2).to_string())

    cash_cor = cash_corr_matrix(traces, all_names)
    cash_path = f'{out_prefix}_corr_cash.csv'
    cash_cor.to_csv(cash_path)
    print(f"\nCASH correlation → {cash_path}")
    print(cash_cor.round(2).to_string())

    # 2) 봉주기별 EW 통합 수익률
    tf_rets = build_tf_returns(traces, cand_by_tf)
    print(f"\nTF returns built: {list(tf_rets.keys())}")
    for tf, r in tf_rets.items():
        m = metrics_from_returns(r)
        print(f"  {tf} EW: Sh {m['Sharpe']:.2f}  CAGR {m['CAGR']:+.1%}  MDD {m['MDD']:+.1%}  Cal {m['Cal']:.2f}")

    # 2.5) TF간 상관계수
    tf_df = pd.concat(tf_rets, axis=1).fillna(0)
    tf_cor = tf_df.corr()
    print(f"\nTF return correlation:")
    print(tf_cor.round(3).to_string())

    # 3) Triple sweep
    rows = sweep_triples(tf_rets, step=args.step)
    rows.sort(key=lambda r: -r['Sharpe'])
    sweep_path = f'{out_prefix}_triple_sweep.csv'
    with open(sweep_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=['w_D', 'w_4h', 'w_2h', 'Sharpe', 'CAGR', 'MDD', 'Cal'])
        w.writeheader()
        for r in rows:
            w.writerow({k: f'{v:.4f}' if isinstance(v, float) else v for k, v in r.items()})
    print(f"\nTriple sweep → {sweep_path}  ({len(rows)} combos)")
    print("\nTop 15 by Sharpe:")
    print(f"{'wD':>4} {'w4h':>4} {'w2h':>4} {'Sharpe':>7} {'CAGR':>8} {'MDD':>8} {'Cal':>6}")
    for r in rows[:15]:
        print(f"{r['w_D']:>4} {r['w_4h']:>4} {r['w_2h']:>4} {r['Sharpe']:>7.2f} {r['CAGR']:>+8.1%} {r['MDD']:>+8.1%} {r['Cal']:>6.2f}")
    print("\nTop 15 by Calmar:")
    rows_cal = sorted(rows, key=lambda r: -r['Cal'])
    for r in rows_cal[:15]:
        print(f"{r['w_D']:>4} {r['w_4h']:>4} {r['w_2h']:>4} {r['Sharpe']:>7.2f} {r['CAGR']:>+8.1%} {r['MDD']:>+8.1%} {r['Cal']:>6.2f}")

    # 4) 2h 가치 검증: D:4h 비율 유지하며 2h 10/20/30 삽입 (직접 계산 — grid step 무관)
    print("\n=== 2h 추가 가치 검증 (직접 계산, base D:4h 비율 유지) ===")
    print(f"{'wD':>5} {'w4h':>5} {'w2h':>5} | {'Sharpe':>7} {'Cal':>6}  Δ vs base(no-2h)")
    bases = [(50, 50), (60, 40), (40, 60), (70, 30), (30, 70), (80, 20), (20, 80)]
    for d, h in bases:
        # base: D/4h만
        base_rets = tf_df['D'] * (d / 100) + tf_df['4h'] * (h / 100)
        base_m = metrics_from_returns(base_rets)
        print(f"{d:>5} {h:>5} {0:>5} | {base_m['Sharpe']:>7.2f} {base_m['Cal']:>6.2f}  (base)")
        for w2 in [10, 20, 30]:
            scale = (100 - w2) / 100
            wd, wh = d * scale, h * scale
            rets = tf_df['D'] * (wd / 100) + tf_df['4h'] * (wh / 100) + tf_df['2h'] * (w2 / 100)
            m = metrics_from_returns(rets)
            d_sh = m['Sharpe'] - base_m['Sharpe']
            d_cal = m['Cal'] - base_m['Cal']
            print(f"{wd:>5.1f} {wh:>5.1f} {w2:>5} | {m['Sharpe']:>7.2f} {m['Cal']:>6.2f}  ΔSh{d_sh:+.2f} ΔCal{d_cal:+.2f}")


if __name__ == '__main__':
    main()
