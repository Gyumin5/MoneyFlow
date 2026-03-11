#!/usr/bin/env python3
"""Test canary hysteresis variants with date-averaging.
   1. Band: ±1%, ±2%, ±3%, ±5%
   2. Grace: 3, 5, 7, 10 days
   3. Consecutive: 3, 5, 7 days
   4. Combinations: band+grace, band+consec
All with PFD5, date-averaged over 10 anchor days."""

import os, sys, time
import multiprocessing as mp
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())
ANCHOR_DAYS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]


def B(**kw):
    base = dict(
        canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1,
        health='HK', health_sma=2, health_mom_short=21,
        health_mom_long=90, vol_cap=0.05,
        post_flip_delay=5,
    )
    base.update(kw)
    return Params(**base)


# ═══════════════════════════════════════════════════════════════════
# Build all params
# ═══════════════════════════════════════════════════════════════════

def build_params():
    all_params = []
    param_map = {}  # (name, anchor_day) -> index

    def add(name, anchor, p):
        idx = len(all_params)
        param_map[(name, anchor)] = idx
        all_params.append(p)

    # Baseline (no hysteresis, with PFD5)
    for d in ANCHOR_DAYS:
        add('BASE', d, B(rebalancing=f'RX{d}'))

    # No PFD5 baseline (for reference)
    for d in ANCHOR_DAYS:
        add('BASE_noPFD', d, B(rebalancing=f'RX{d}', post_flip_delay=0))

    # ── Band variants ──
    for band in [1, 2, 3, 5, 7, 10]:
        for d in ANCHOR_DAYS:
            add(f'B{band}%', d, B(rebalancing=f'RX{d}', canary_band=band))

    # ── Grace period variants ──
    for grace in [3, 5, 7, 10, 14]:
        for d in ANCHOR_DAYS:
            add(f'G{grace}d', d, B(rebalancing=f'RX{d}', canary_grace=grace))

    # ── Consecutive days variants ──
    for consec in [3, 5, 7, 10]:
        for d in ANCHOR_DAYS:
            add(f'C{consec}d', d, B(rebalancing=f'RX{d}', canary_consec=consec))

    # ── Combinations: band + grace ──
    for band, grace in [(2, 3), (2, 5), (3, 3), (3, 5), (5, 3)]:
        for d in ANCHOR_DAYS:
            add(f'B{band}%+G{grace}d', d,
                B(rebalancing=f'RX{d}', canary_band=band, canary_grace=grace))

    # ── Combinations: band + consec ──
    for band, consec in [(2, 3), (2, 5), (3, 3), (3, 5), (5, 3)]:
        for d in ANCHOR_DAYS:
            add(f'B{band}%+C{consec}d', d,
                B(rebalancing=f'RX{d}', canary_band=band, canary_consec=consec))

    # ── Band without PFD5 (to isolate effect) ──
    for band in [2, 3, 5]:
        for d in ANCHOR_DAYS:
            add(f'B{band}%_noPFD', d,
                B(rebalancing=f'RX{d}', canary_band=band, post_flip_delay=0))

    return all_params, param_map


# ═══════════════════════════════════════════════════════════════════
# Process results
# ═══════════════════════════════════════════════════════════════════

def process(results, param_map):
    # Get unique test names
    names = []
    seen = set()
    for (name, _) in param_map:
        if name not in seen:
            seen.add(name)
            names.append(name)

    output = {}
    for name in names:
        metrics = []
        for d in ANCHOR_DAYS:
            key = (name, d)
            if key not in param_map:
                continue
            r = results[param_map[key]]
            metrics.append(r['metrics'])

        if not metrics:
            continue

        avg = {}
        for k in metrics[0]:
            avg[k] = np.mean([m[k] for m in metrics])

        output[name] = {
            'metrics': avg,
            'sharpe_std': np.std([m['Sharpe'] for m in metrics]),
            'yearly': {},  # skip yearly for now
        }

        # Collect yearly from one representative run (anchor=1)
        key = (name, 1)
        if key in param_map:
            output[name]['yearly'] = results[param_map[key]].get('yearly', {})

    return output


# ═══════════════════════════════════════════════════════════════════
# Display
# ═══════════════════════════════════════════════════════════════════

LABELS = {
    'BASE': '베이스라인 (PFD5)',
    'BASE_noPFD': '베이스라인 (PFD없음)',
    'B1%': '밴드 ±1%',
    'B2%': '밴드 ±2%',
    'B3%': '밴드 ±3%',
    'B5%': '밴드 ±5%',
    'B7%': '밴드 ±7%',
    'B10%': '밴드 ±10%',
    'G3d': '유예 3일',
    'G5d': '유예 5일',
    'G7d': '유예 7일',
    'G10d': '유예 10일',
    'G14d': '유예 14일',
    'C3d': '연속 3일',
    'C5d': '연속 5일',
    'C7d': '연속 7일',
    'C10d': '연속 10일',
}


def print_results(output):
    base = output.get('BASE', {}).get('metrics', {})
    base_s = base.get('Sharpe', 0)

    def print_section(title, names):
        print(f"\n  {title}")
        print(f"  {'─' * 85}")
        print(f"  {'전략':<28} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
        print(f"  {'─' * 85}")
        for name in names:
            if name not in output:
                continue
            r = output[name]
            m = r['metrics']
            calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
            delta = m['Sharpe'] - base_s
            label = LABELS.get(name, name)
            marker = ' ★' if delta > 0.05 else ' ↑' if delta > 0 else ''
            print(f"  {label:<28} {m['Sharpe']:>7.3f} {delta:>+6.3f} {r['sharpe_std']:>6.3f}"
                  f" {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}{marker}")

    print(f"\n{'=' * 100}")
    print(f"  카나리아 히스테리시스 테스트 — 날짜 평균 (10 앵커)")
    print(f"{'=' * 100}")

    print_section("0. 베이스라인", ['BASE_noPFD', 'BASE'])

    print_section("1. 밴드 (±N%): ON 유지 쉽게, ON 전환 어렵게",
                  ['B1%', 'B2%', 'B3%', 'B5%', 'B7%', 'B10%'])

    print_section("2. 유예기간: OFF 전환을 N일 지연",
                  ['G3d', 'G5d', 'G7d', 'G10d', 'G14d'])

    print_section("3. 연속확인: 상태변경에 N일 연속 필요",
                  ['C3d', 'C5d', 'C7d', 'C10d'])

    combo_names = [n for n in output if '+' in n]
    if combo_names:
        print_section("4. 조합", sorted(combo_names))

    nopfd_names = [n for n in output if '_noPFD' in n and n != 'BASE_noPFD']
    if nopfd_names:
        print_section("5. 밴드 단독 (PFD5 없음)", sorted(nopfd_names))

    # ── Best summary ──
    print(f"\n{'=' * 100}")
    print(f"  Sharpe 순위 Top 10")
    print(f"{'=' * 100}")
    ranked = sorted(output.items(), key=lambda x: x[1]['metrics']['Sharpe'], reverse=True)
    print(f"\n  {'순위':>3} {'전략':<28} {'Sharpe':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7}")
    print(f"  {'─' * 65}")
    for i, (name, r) in enumerate(ranked[:10], 1):
        m = r['metrics']
        label = LABELS.get(name, name)
        print(f"  {i:>3}. {label:<28} {m['Sharpe']:>7.3f} {r['sharpe_std']:>6.3f}"
              f" {m['CAGR']:>+7.1%} {m['MDD']:>6.1%}")

    # ── Robustness ranking ──
    print(f"\n  σ(Sharpe) 순위 Top 10 (낮을수록 강건)")
    ranked_s = sorted(output.items(), key=lambda x: x[1]['sharpe_std'])
    print(f"\n  {'순위':>3} {'전략':<28} {'σ(S)':>6} {'Sharpe':>7}")
    print(f"  {'─' * 50}")
    for i, (name, r) in enumerate(ranked_s[:10], 1):
        label = LABELS.get(name, name)
        print(f"  {i:>3}. {label:<28} {r['sharpe_std']:>6.3f} {r['metrics']['Sharpe']:>7.3f}")

    # ── Year-by-year for top 3 ──
    years = range(2018, 2026)
    top3 = [name for name, _ in ranked[:3]]
    top3 = ['BASE'] + [n for n in top3 if n != 'BASE']
    print(f"\n  주요 전략 연도별 Sharpe (앵커1일 기준)")
    print(f"  {'전략':<28}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print()
    print(f"  {'─' * 95}")
    for name in top3[:5]:
        if name not in output:
            continue
        ym = output[name]['yearly']
        label = LABELS.get(name, name)
        row = f"  {label:<28}"
        for y in years:
            if y in ym:
                row += f" {ym[y].get('Sharpe', 0):>7.2f}"
            else:
                row += f" {'─':>7}"
        print(row)


def main():
    print("Loading data...")
    prices, universe = load_data()

    all_params, param_map = build_params()
    print(f"  {len(prices)} tickers, {len(all_params)} backtests")

    t0 = time.time()
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, all_params)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    output = process(results, param_map)
    print_results(output)

    print(f"\n  Total: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
