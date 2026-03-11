#!/usr/bin/env python3
"""K5+H5 fixed, test all Stage1 survivor combos for S/W/R/G layers.
   Compare tx=0 vs tx=0.4%."""

import os, sys, time, itertools
import multiprocessing as mp
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

# Stage 1 survivors per layer (excluding K/H which are fixed)
LAYER_OPTIONS = {
    'selection':   ['baseline', 'S4', 'S5'],
    'weighting':   ['baseline', 'W1', 'W3', 'W4'],
    'rebalancing': ['baseline', 'R1', 'R2'],
    'risk':        ['baseline', 'G2', 'G3', 'G4', 'G5'],
}

def gen_all_combos():
    """Generate all combinations of S/W/R/G layer options."""
    keys = list(LAYER_OPTIONS.keys())
    options = [LAYER_OPTIONS[k] for k in keys]
    combos = []
    for combo in itertools.product(*options):
        kwargs = {'canary': 'K5', 'health': 'H5'}
        for k, v in zip(keys, combo):
            kwargs[k] = v
        combos.append(Params(**kwargs))
    return combos


def run_all(combos, prices, universe, tx_cost):
    """Run all combos with given tx_cost."""
    params_list = []
    for p in combos:
        p2 = Params(
            canary=p.canary, health=p.health,
            selection=p.selection, weighting=p.weighting,
            rebalancing=p.rebalancing, risk=p.risk,
            tx_cost=tx_cost,
        )
        params_list.append(p2)

    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, params_list)
    return results


def label_short(p):
    """Generate short label showing only non-baseline layers (excluding K5+H5)."""
    parts = []
    for layer in ('selection', 'weighting', 'rebalancing', 'risk'):
        v = getattr(p, layer)
        if v != 'baseline':
            parts.append(v)
    return '+'.join(parts) if parts else '(base S/W/R/G)'


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded\n")

    combos = gen_all_combos()
    print(f"  K5+H5 고정 + S/W/R/G 조합: {len(combos)}개")
    print(f"  S: {LAYER_OPTIONS['selection']}")
    print(f"  W: {LAYER_OPTIONS['weighting']}")
    print(f"  R: {LAYER_OPTIONS['rebalancing']}")
    print(f"  G: {LAYER_OPTIONS['risk']}")

    # Run tx=0 and tx=0.4%
    print(f"\n  Running tx=0 ...")
    t0 = time.time()
    results_0 = run_all(combos, prices, universe, tx_cost=0.0)
    t1 = time.time()
    print(f"  {t1-t0:.1f}s")

    print(f"  Running tx=0.4% ...")
    results_4 = run_all(combos, prices, universe, tx_cost=0.004)
    t2 = time.time()
    print(f"  {t2-t1:.1f}s")

    # Merge and sort
    rows = []
    for i, (r0, r4) in enumerate(zip(results_0, results_4)):
        m0 = r0['metrics']
        m4 = r4['metrics']
        p = combos[i]
        rows.append({
            'label': label_short(p),
            'full_label': 'K5+H5' + ('+' + label_short(p) if label_short(p) != '(base S/W/R/G)' else ''),
            'params': p,
            'n_layers': sum(1 for l in ('selection','weighting','rebalancing','risk')
                          if getattr(p, l) != 'baseline'),
            'Sharpe_0': m0['Sharpe'], 'CAGR_0': m0['CAGR'], 'MDD_0': m0['MDD'],
            'Final_0': m0['Final'], 'Sortino_0': m0['Sortino'],
            'Sharpe_4': m4['Sharpe'], 'CAGR_4': m4['CAGR'], 'MDD_4': m4['MDD'],
            'Final_4': m4['Final'], 'Sortino_4': m4['Sortino'],
            'rebals_0': r0['rebal_count'], 'rebals_4': r4['rebal_count'],
            'cost_drag': m0['CAGR'] - m4['CAGR'],  # CAGR lost to fees
        })

    rows.sort(key=lambda x: -x['Sharpe_4'])

    # ═══════════════════════════════════════════════════════════════
    # 1. FULL TABLE — tx=0 vs tx=0.4%
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 145)
    print("  K5+H5 고정 — S/W/R/G 조합 전구간 (2018~2025.06)")
    print("  tx=0 vs tx=0.4% 비교 (Sharpe(tx=0.4%) 순 정렬)")
    print("=" * 145)
    print(f"\n  {'#':>3} {'S/W/R/G 조합':<24} {'L':>1}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
          f" │{'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
          f" │{'비용손실':>7} {'Rebals':>6}")
    print(f"  {'':>3} {'':>24} {'':>1}"
          f" │{'── tx=0 ──':^33}"
          f" │{'── tx=0.4% ──':^33}"
          f" │{'ΔCAGR':>7} {'':>6}")
    print(f"  {'─' * 140}")

    for i, r in enumerate(rows[:50]):
        print(f"  {i+1:>3} {r['label']:<24} {r['n_layers']:>1}"
              f" │{r['Sharpe_0']:>7.3f} {r['CAGR_0']:>+6.1%} {r['MDD_0']:>6.1%} {r['Final_0']:>10,.0f}"
              f" │{r['Sharpe_4']:>7.3f} {r['CAGR_4']:>+6.1%} {r['MDD_4']:>6.1%} {r['Final_4']:>10,.0f}"
              f" │{r['cost_drag']:>+6.1%} {r['rebals_4']:>6}")

    # ═══════════════════════════════════════════════════════════════
    # 2. S5 vs baseline — 순수 비용 효과 분리
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 145)
    print("  S5 효과 분리: 동일 조합에서 S=baseline vs S=S5 비교")
    print("=" * 145)
    print(f"\n  {'W/R/G 조합':<20}"
          f" │{'S=base Sharpe':>12} {'CAGR':>7} {'비용손실':>7}"
          f" │{'S=S5 Sharpe':>12} {'CAGR':>7} {'비용손실':>7}"
          f" │{'ΔSharpe':>8} {'ΔCAGR(0)':>8} {'ΔCAGR(.4)':>9}")
    print(f"  {'─' * 115}")

    # Group by W/R/G, compare S=baseline vs S=S5
    by_wrg = {}
    for r in rows:
        p = r['params']
        wrg_key = f"{p.weighting}/{p.rebalancing}/{p.risk}"
        s_val = p.selection
        by_wrg.setdefault(wrg_key, {})[s_val] = r

    for wrg_key in sorted(by_wrg.keys()):
        group = by_wrg[wrg_key]
        if 'baseline' in group and 'S5' in group:
            b = group['baseline']
            s5 = group['S5']
            ds = s5['Sharpe_4'] - b['Sharpe_4']
            dc0 = s5['CAGR_0'] - b['CAGR_0']
            dc4 = s5['CAGR_4'] - b['CAGR_4']
            print(f"  {wrg_key:<20}"
                  f" │{b['Sharpe_4']:>12.3f} {b['CAGR_4']:>+6.1%} {b['cost_drag']:>+6.1%}"
                  f" │{s5['Sharpe_4']:>12.3f} {s5['CAGR_4']:>+6.1%} {s5['cost_drag']:>+6.1%}"
                  f" │{ds:>+7.3f} {dc0:>+7.1%} {dc4:>+8.1%}")

    # ═══════════════════════════════════════════════════════════════
    # 3. 비용 영향 Top/Bottom
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 145)
    print("  비용 영향 분석: 거래비용으로 가장 많이/적게 잃는 조합")
    print("=" * 145)

    by_cost = sorted(rows, key=lambda x: x['cost_drag'])
    print(f"\n  비용 손실 가장 적은 10개 (비용에 강한 전략):")
    print(f"  {'#':>3} {'조합':<24} {'CAGR(0)':>8} {'CAGR(.4)':>8} {'비용손실':>8} {'Rebals':>6}")
    print(f"  {'─' * 65}")
    for i, r in enumerate(by_cost[:10]):
        print(f"  {i+1:>3} {r['label']:<24} {r['CAGR_0']:>+7.1%} {r['CAGR_4']:>+7.1%}"
              f" {r['cost_drag']:>+7.1%} {r['rebals_4']:>6}")

    print(f"\n  비용 손실 가장 큰 10개 (비용에 취약한 전략):")
    print(f"  {'#':>3} {'조합':<24} {'CAGR(0)':>8} {'CAGR(.4)':>8} {'비용손실':>8} {'Rebals':>6}")
    print(f"  {'─' * 65}")
    for i, r in enumerate(by_cost[-10:]):
        print(f"  {i+1:>3} {r['label']:<24} {r['CAGR_0']:>+7.1%} {r['CAGR_4']:>+7.1%}"
              f" {r['cost_drag']:>+7.1%} {r['rebals_4']:>6}")

    # ═══════════════════════════════════════════════════════════════
    # 4. BEST PER LAYER COUNT
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 145)
    print("  레이어 수별 최고 전략 (tx=0.4% 기준)")
    print("=" * 145)
    by_n = {}
    for r in rows:
        n = r['n_layers']
        if n not in by_n or r['Sharpe_4'] > by_n[n]['Sharpe_4']:
            by_n[n] = r
    print(f"\n  {'추가L':>4} {'조합':<24} {'Sharpe':>7} {'CAGR':>7} {'MDD':>7} {'Final':>10}"
          f" {'비용손실':>7}")
    print(f"  {'─' * 75}")
    for n in sorted(by_n):
        r = by_n[n]
        print(f"  {n:>4} {r['label']:<24} {r['Sharpe_4']:>7.3f} {r['CAGR_4']:>+6.1%}"
              f" {r['MDD_4']:>6.1%} {r['Final_4']:>10,.0f} {r['cost_drag']:>+6.1%}")

    # ═══════════════════════════════════════════════════════════════
    # 5. tx=0 RANKING (순수 전략 효과)
    # ═══════════════════════════════════════════════════════════════
    rows_by_0 = sorted(rows, key=lambda x: -x['Sharpe_0'])
    print()
    print("=" * 145)
    print("  tx=0 기준 Top 20 (순수 전략 효과, 거래비용 제거)")
    print("=" * 145)
    print(f"\n  {'#':>3} {'S/W/R/G 조합':<24} {'Sharpe':>7} {'CAGR':>7} {'MDD':>7}"
          f" {'Final':>10} {'Sortino':>8} {'Calmar':>7}")
    print(f"  {'─' * 80}")
    for i, r in enumerate(rows_by_0[:20]):
        calmar = r['CAGR_0'] / abs(r['MDD_0']) if r['MDD_0'] != 0 else 0
        print(f"  {i+1:>3} {r['label']:<24} {r['Sharpe_0']:>7.3f} {r['CAGR_0']:>+6.1%}"
              f" {r['MDD_0']:>6.1%} {r['Final_0']:>10,.0f} {r['Sortino_0']:>8.3f}"
              f" {calmar:>7.3f}")

    # ═══════════════════════════════════════════════════════════════
    # 6. RANK COMPARISON: tx=0 vs tx=0.4%
    # ═══════════════════════════════════════════════════════════════
    print()
    print("=" * 145)
    print("  순위 변동: tx=0 → tx=0.4%")
    print("=" * 145)

    rank_0 = {r['label']: i+1 for i, r in enumerate(rows_by_0)}
    rank_4 = {r['label']: i+1 for i, r in enumerate(rows)}

    movers = [(r['label'], rank_0[r['label']], rank_4[r['label']],
               rank_0[r['label']] - rank_4[r['label']])
              for r in rows]
    movers.sort(key=lambda x: -x[3])  # biggest climbers first

    print(f"\n  비용 포함 시 순위 상승 Top 10 (비용에 강해서 올라온 전략):")
    print(f"  {'조합':<24} {'순위(0)':>7} {'순위(.4)':>8} {'변동':>6}")
    print(f"  {'─' * 50}")
    for label, r0, r4, delta in movers[:10]:
        print(f"  {label:<24} {r0:>7} {r4:>8} {delta:>+5}")

    print(f"\n  비용 포함 시 순위 하락 Top 10 (비용에 취약해서 떨어진 전략):")
    print(f"  {'조합':<24} {'순위(0)':>7} {'순위(.4)':>8} {'변동':>6}")
    print(f"  {'─' * 50}")
    for label, r0, r4, delta in movers[-10:]:
        print(f"  {label:<24} {r0:>7} {r4:>8} {delta:>+5}")


if __name__ == '__main__':
    main()
