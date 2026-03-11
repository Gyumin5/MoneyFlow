#!/usr/bin/env python3
"""SMA crossover robustness: vary BOTH short and long sides comprehensively."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

SHORT_VALS = [3, 5, 7, 10, 15, 20, 25, 30]
LONG_VALS = [40, 45, 50, 55, 60, 65, 70, 75, 80, 90, 100]

STRATEGIES = []
for s in SHORT_VALS:
    for l in LONG_VALS:
        if s >= l:
            continue
        STRATEGIES.append(
            (f'{s}>{l}', Params(canary='K9', health='H1', cross_short=s, cross_long=l))
        )


def run_set(strategies, prices, universe, tx=0.004):
    params_list = []
    for _, p in strategies:
        params_list.append(Params(
            canary=p.canary, health=p.health, tx_cost=tx,
            sma_period=p.sma_period,
            vote_smas=p.vote_smas, vote_moms=p.vote_moms,
            vote_threshold=p.vote_threshold,
            cross_short=p.cross_short, cross_long=p.cross_long,
        ))
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, params_list)
    return results


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")
    print(f"  {len(STRATEGIES)} combinations to test")

    t0 = time.time()
    results = run_set(STRATEGIES, prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # Build lookup
    lookup = {}
    for (name, _), r in zip(STRATEGIES, results):
        lookup[name] = r['metrics']['Sharpe']

    # ── Grid: Short × Long Sharpe ───────────────────────────────
    print(f"\n{'=' * 130}")
    print(f"  SMA Crossover Grid — Sharpe (H1, tx=0.4%)")
    print(f"  행: short SMA, 열: long SMA")
    print(f"{'=' * 130}")

    print(f"\n  {'short↓ long→':>12}", end="")
    for l in LONG_VALS:
        print(f" {l:>6}", end="")
    print(f" {'avg':>7} {'best':>7}")
    print(f"  {'─' * (12 + 7 * len(LONG_VALS) + 16)}")

    best_overall = ('', 0)
    for s in SHORT_VALS:
        print(f"  SMA({s:>3})     ", end="")
        vals = []
        for l in LONG_VALS:
            key = f'{s}>{l}'
            if key in lookup:
                v = lookup[key]
                vals.append(v)
                if v > best_overall[1]:
                    best_overall = (key, v)
                # Highlight top performers
                if v >= 1.45:
                    print(f" {v:>5.3f}★", end="")
                elif v >= 1.35:
                    print(f" {v:>5.3f}·", end="")
                else:
                    print(f" {v:>6.3f}", end="")
            else:
                print(f" {'─':>6}", end="")
        if vals:
            avg = sum(vals) / len(vals)
            best = max(vals)
            print(f" {avg:>7.3f} {best:>7.3f}")
        else:
            print()

    # Column averages
    print(f"  {'─' * (12 + 7 * len(LONG_VALS) + 16)}")
    print(f"  {'avg':>12}", end="")
    for l in LONG_VALS:
        vals = [lookup[f'{s}>{l}'] for s in SHORT_VALS if f'{s}>{l}' in lookup]
        if vals:
            print(f" {sum(vals)/len(vals):>6.3f}", end="")
        else:
            print(f" {'─':>6}", end="")
    print()

    print(f"\n  ★ = Sharpe ≥ 1.45, · = Sharpe ≥ 1.35")
    print(f"  Best overall: {best_overall[0]} → Sharpe {best_overall[1]:.3f}")

    # ── Short SMA robustness (long=65 고정) ─────────────────────
    print(f"\n  Short SMA 강건성 (long=65 고정):")
    for s in SHORT_VALS:
        key = f'{s}>65'
        if key in lookup:
            v = lookup[key]
            bar = '█' * int(v * 20)
            print(f"    SMA({s:>3})>65: {v:.3f} {bar}")
    vals_s = [lookup[f'{s}>65'] for s in SHORT_VALS if f'{s}>65' in lookup]
    spread_s = max(vals_s) - min(vals_s)
    print(f"    spread: {spread_s:.3f} {'넓은 고원 ✓' if spread_s < 0.15 else '좁은 봉우리 ✗'}")

    # ── Long SMA robustness (short=10 고정) ─────────────────────
    print(f"\n  Long SMA 강건성 (short=10 고정):")
    for l in LONG_VALS:
        key = f'10>{l}'
        if key in lookup:
            v = lookup[key]
            bar = '█' * int(v * 20)
            print(f"    10>SMA({l:>3}): {v:.3f} {bar}")
    vals_l = [lookup[f'10>{l}'] for l in LONG_VALS if f'10>{l}' in lookup]
    spread_l = max(vals_l) - min(vals_l)
    print(f"    spread: {spread_l:.3f} {'넓은 고원 ✓' if spread_l < 0.15 else '좁은 봉우리 ✗'}")

    # ── Summary: which short values are robust? ─────────────────
    print(f"\n{'=' * 100}")
    print(f"  Short SMA별 평균 Sharpe (모든 long에 걸친)")
    print(f"{'=' * 100}")
    for s in SHORT_VALS:
        vals = [lookup[f'{s}>{l}'] for l in LONG_VALS if f'{s}>{l}' in lookup]
        if vals:
            avg = sum(vals) / len(vals)
            best = max(vals)
            spread = max(vals) - min(vals)
            bar = '█' * int(avg * 15)
            robust = '✓' if spread < 0.15 else '✗'
            print(f"    SMA({s:>3}): avg {avg:.3f}  best {best:.3f}  spread {spread:.3f} {robust} {bar}")

    print(f"\n  Long SMA별 평균 Sharpe (모든 short에 걸친)")
    for l in LONG_VALS:
        vals = [lookup[f'{s}>{l}'] for s in SHORT_VALS if f'{s}>{l}' in lookup]
        if vals:
            avg = sum(vals) / len(vals)
            best = max(vals)
            spread = max(vals) - min(vals)
            bar = '█' * int(avg * 15)
            robust = '✓' if spread < 0.15 else '✗'
            print(f"    SMA({l:>3}): avg {avg:.3f}  best {best:.3f}  spread {spread:.3f} {robust} {bar}")

    # Top 10
    print(f"\n{'=' * 100}")
    print(f"  Crossover Top 10")
    print(f"{'=' * 100}")
    all_scored = [(name, r['metrics']) for (name, _), r in zip(STRATEGIES, results)]
    all_scored.sort(key=lambda x: x[1]['Sharpe'], reverse=True)
    print(f"\n  {'순위':>3} {'조합':<10} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7}")
    print(f"  {'─' * 40}")
    for i, (name, m) in enumerate(all_scored[:10], 1):
        print(f"  {i:>3}. {name:<10} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%}")


if __name__ == '__main__':
    main()
