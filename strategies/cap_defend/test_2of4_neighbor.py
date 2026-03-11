#!/usr/bin/env python3
"""Parameter neighborhood test for (80,50)+m21,m60 2/4 canary.
   Vary each parameter one at a time to check for overfitting."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())


def P(smas, moms, thr=2):
    return Params(canary='K8', health='H1',
                  vote_smas=tuple(smas), vote_moms=tuple(moms),
                  vote_threshold=thr)

# Baseline: (80,50)+m21,m60 2/4

# ── 1. SMA_long (80) 변경 ─────────────────────────────────────────
T1 = [(f'sma_long={v}', P([v, 50], [21, 60]))
      for v in [60, 70, 80, 90, 100, 110, 120]]

# ── 2. SMA_short (50) 변경 ────────────────────────────────────────
T2 = [(f'sma_short={v}', P([80, v], [21, 60]))
      for v in [20, 30, 40, 50, 60, 70]]

# ── 3. Mom_short (21) 변경 ────────────────────────────────────────
T3 = [(f'mom_short={v}', P([80, 50], [v, 60]))
      for v in [7, 10, 14, 21, 30]]

# ── 4. Mom_long (60) 변경 ─────────────────────────────────────────
T4 = [(f'mom_long={v}', P([80, 50], [21, v]))
      for v in [30, 45, 60, 90, 120]]

# ── 5. Threshold 변경 ────────────────────────────────────────────
T5 = [
    ('2/4 (현재)',  P([80, 50], [21, 60], 2)),
    ('3/4',        P([80, 50], [21, 60], 3)),
    ('1/4',        P([80, 50], [21, 60], 1)),
]

# ── 6. 전체 조합 grid (sma_long × mom_long, 핵심 2개) ──────────
T6 = []
for sl in [60, 80, 100, 120]:
    for ml in [30, 60, 90]:
        T6.append((f'sma{sl}+mom{ml}', P([sl, 50], [21, ml])))


def run_set(strategies, prices, universe, tx=0.004):
    params_list = []
    for _, p in strategies:
        params_list.append(Params(
            canary=p.canary, health=p.health, tx_cost=tx,
            sma_period=p.sma_period,
            vote_smas=p.vote_smas, vote_moms=p.vote_moms,
            vote_threshold=p.vote_threshold,
        ))
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, params_list)
    return results


def print_sweep(title, param_name, strategies, results):
    print(f"\n  {title}")
    print(f"  {'─' * 75}")
    print(f"  {param_name:<15} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Final':>10}  비고")
    print(f"  {'─' * 75}")
    best_sharpe = max(r['metrics']['Sharpe'] for r in results)
    for (name, _), r in zip(strategies, results):
        m = r['metrics']
        marker = " ★" if m['Sharpe'] == best_sharpe else ""
        is_baseline = '80' in name and '50' in name if 'sma' not in name else False
        if '=80' in name or '=50' in name or '=21' in name or '=60' in name or '현재' in name:
            marker += " ←현재"
        print(f"  {name:<15} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {m['Final']:>10,.0f}{marker}")


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()
    r1 = run_set(T1, prices, universe)
    r2 = run_set(T2, prices, universe)
    r3 = run_set(T3, prices, universe)
    r4 = run_set(T4, prices, universe)
    r5 = run_set(T5, prices, universe)
    r6 = run_set(T6, prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    print(f"\n{'=' * 80}")
    print(f"  (80,50)+m21,m60 2/4 — 파라미터 이웃 테스트 (tx=0.4%)")
    print(f"{'=' * 80}")

    print_sweep("1. SMA_long 변경 (50, m21, m60 고정)", "sma_long", T1, r1)
    print_sweep("2. SMA_short 변경 (80, m21, m60 고정)", "sma_short", T2, r2)
    print_sweep("3. Mom_short 변경 (80, 50, m60 고정)", "mom_short", T3, r3)
    print_sweep("4. Mom_long 변경 (80, 50, m21 고정)", "mom_long", T4, r4)
    print_sweep("5. Threshold 변경", "threshold", T5, r5)

    # Grid
    print(f"\n  6. SMA_long × Mom_long 그리드 (sma_short=50, mom_short=21 고정)")
    print(f"  {'─' * 60}")
    print(f"  {'':>15}", end="")
    mom_vals = [30, 60, 90]
    for ml in mom_vals:
        print(f" {'mom'+str(ml):>12}", end="")
    print()
    print(f"  {'─' * 55}")

    idx = 0
    for sl in [60, 80, 100, 120]:
        print(f"  {'sma'+str(sl):<15}", end="")
        for ml in mom_vals:
            m = r6[idx]['metrics']
            marker = "★" if sl == 80 and ml == 60 else " "
            print(f"  {m['Sharpe']:>5.3f}{marker}    ", end="")
            idx += 1
        print()

    # Summary
    print(f"\n{'=' * 80}")
    print(f"  종합 판정")
    print(f"{'=' * 80}")

    for title, results_list in [
        ("SMA_long", r1), ("SMA_short", r2),
        ("Mom_short", r3), ("Mom_long", r4),
    ]:
        sharpes = [r['metrics']['Sharpe'] for r in results_list]
        avg = sum(sharpes) / len(sharpes)
        spread = max(sharpes) - min(sharpes)
        best = max(sharpes)
        worst = min(sharpes)
        print(f"  {title:<12}: 최저 {worst:.3f} → 최고 {best:.3f}"
              f"  편차 {spread:.3f}  {'넓은 고원 ✓' if spread < 0.15 else '좁은 봉우리 ✗'}")


if __name__ == '__main__':
    main()
