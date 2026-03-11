#!/usr/bin/env python3
"""Test K voting strategies:
   Multiple SMA values from plateau range (50-100) vote together.
   Also test H voting: multiple Mom conditions vote.
   Compare vs single best (SMA60, Mom21+Mom90+Vol5%)."""

import os, sys, time
import multiprocessing as mp

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import Params, load_data, init_pool, run_single

N_WORKERS = min(24, mp.cpu_count())

# Fixed H for K tests
H_BEST = dict(health='HK', health_sma=2, health_mom_short=21,
              health_mom_long=90, vol_cap=0.05)

# ══════════════════════════════════════════════════════════════════
# Part 0: Baselines
# ══════════════════════════════════════════════════════════════════
BASELINES = [
    ('K:SMA(60) 단일',
     Params(canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1, **H_BEST)),
    ('K:SMA(65) 단일',
     Params(canary='K8', vote_smas=(65,), vote_moms=(), vote_threshold=1, **H_BEST)),
    ('K:SMA(70) 단일',
     Params(canary='K8', vote_smas=(70,), vote_moms=(), vote_threshold=1, **H_BEST)),
    ('K:SMA(80) 단일',
     Params(canary='K8', vote_smas=(80,), vote_moms=(), vote_threshold=1, **H_BEST)),
]

# ══════════════════════════════════════════════════════════════════
# Part 1: K SMA 투표 — 다양한 SMA 조합 + 임계치
# ══════════════════════════════════════════════════════════════════
K_VOTE_SMA = []

# 3개 SMA 투표 (majority vote = 2/3)
sma_3_combos = [
    (50, 60, 70), (55, 65, 75), (60, 70, 80),
    (60, 75, 90), (50, 70, 90), (55, 70, 85),
    (60, 80, 100),
]
for combo in sma_3_combos:
    for thresh in [2, 3]:  # majority vs unanimous
        label = f'3v({",".join(str(s) for s in combo)})≥{thresh}'
        K_VOTE_SMA.append(
            (label, Params(canary='K8', vote_smas=combo, vote_moms=(),
                          vote_threshold=thresh, **H_BEST)))

# 5개 SMA 투표 (broader ensemble)
sma_5_combos = [
    (50, 60, 70, 80, 90),
    (55, 65, 75, 85, 95),
    (50, 60, 70, 80, 100),
    (60, 65, 70, 75, 80),  # tight cluster
]
for combo in sma_5_combos:
    for thresh in [2, 3, 4]:
        label = f'5v({",".join(str(s) for s in combo)})≥{thresh}'
        K_VOTE_SMA.append(
            (label, Params(canary='K8', vote_smas=combo, vote_moms=(),
                          vote_threshold=thresh, **H_BEST)))

# 7개 SMA 투표
sma_7_combos = [
    (50, 55, 60, 65, 70, 80, 90),
    (50, 60, 70, 80, 90, 100, 110),
]
for combo in sma_7_combos:
    for thresh in [3, 4, 5]:
        label = f'7v({combo[0]}~{combo[-1]})≥{thresh}'
        K_VOTE_SMA.append(
            (label, Params(canary='K8', vote_smas=combo, vote_moms=(),
                          vote_threshold=thresh, **H_BEST)))

# ══════════════════════════════════════════════════════════════════
# Part 2: K SMA+Mom 혼합 투표
# ══════════════════════════════════════════════════════════════════
K_VOTE_MIX = []
mix_combos = [
    # (smas, moms, threshold)
    ((60, 80), (40,), 2),    # 3 conditions, 2 pass
    ((60, 80), (40,), 3),    # 3 conditions, all pass
    ((50, 70, 90), (30, 45), 3),  # 5 conditions, 3 pass
    ((50, 70, 90), (30, 45), 4),  # 5 conditions, 4 pass
    ((60, 80), (30, 45), 2),      # 4 conditions, 2 pass
    ((60, 80), (30, 45), 3),      # 4 conditions, 3 pass
    ((60, 70, 80), (40,), 2),     # 4 conditions, 2 pass
    ((60, 70, 80), (40,), 3),     # 4 conditions, 3 pass
    ((55, 65, 75, 85), (35, 45), 3),  # 6 conditions, 3 pass
    ((55, 65, 75, 85), (35, 45), 4),  # 6 conditions, 4 pass
]
for smas, moms, thresh in mix_combos:
    n = len(smas) + len(moms)
    s_str = '+'.join(f's{s}' for s in smas)
    m_str = '+'.join(f'm{m}' for m in moms)
    label = f'{n}v({s_str}+{m_str})≥{thresh}'
    K_VOTE_MIX.append(
        (label, Params(canary='K8', vote_smas=smas, vote_moms=moms,
                      vote_threshold=thresh, **H_BEST)))

# ══════════════════════════════════════════════════════════════════
# Part 3: 임계치 그래디언트 — 같은 SMA 세트에서 임계치만 변경
# ══════════════════════════════════════════════════════════════════
K_THRESH = []
base_smas = (50, 55, 60, 65, 70, 75, 80, 85, 90)  # 9 conditions
for thresh in range(1, 10):
    label = f'9v(50~90)≥{thresh}'
    K_THRESH.append(
        (label, Params(canary='K8', vote_smas=base_smas, vote_moms=(),
                      vote_threshold=thresh, **H_BEST)))

ALL = BASELINES + K_VOTE_SMA + K_VOTE_MIX + K_THRESH


def run_set(strategies, prices, universe, tx=0.004):
    params_list = []
    for _, p in strategies:
        params_list.append(Params(
            canary=p.canary, health=p.health, tx_cost=tx,
            sma_period=p.sma_period,
            vote_smas=p.vote_smas, vote_moms=p.vote_moms,
            vote_threshold=p.vote_threshold,
            cross_short=p.cross_short, cross_long=p.cross_long,
            vol_cap=p.vol_cap,
            health_sma=p.health_sma, health_mom_short=p.health_mom_short,
            health_mom_long=p.health_mom_long, health_vol_window=p.health_vol_window,
        ))
    init_pool(prices, universe)
    with mp.Pool(N_WORKERS) as pool:
        results = pool.map(run_single, params_list)
    return results


def print_section(title, strategies, results):
    print(f"\n  {title}")
    print(f"  {'─' * 80}")
    print(f"  {'조건':<40} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 80}")
    best_s = max(r['metrics']['Sharpe'] for r in results) if results else 0
    for (name, _), r in zip(strategies, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        marker = " ★" if m['Sharpe'] == best_s else ""
        print(f"  {name:<40} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}{marker}")


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")
    print(f"  {len(ALL)} K vote configurations to test")

    t0 = time.time()
    results = run_set(ALL, prices, universe)
    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # Split results
    idx = 0
    def get_slice(strategies):
        nonlocal idx
        r = results[idx:idx+len(strategies)]
        idx += len(strategies)
        return r

    r_base = get_slice(BASELINES)
    r_sma = get_slice(K_VOTE_SMA)
    r_mix = get_slice(K_VOTE_MIX)
    r_thresh = get_slice(K_THRESH)

    print(f"\n{'=' * 100}")
    print(f"  K 투표 전략 비교 — H=Mom(21)+Mom(90)+Vol5% 고정")
    print(f"{'=' * 100}")

    print_section("0. 기준 (단일 SMA)", BASELINES, r_base)
    print_section("1. SMA 투표 (3/5/7개 SMA)", K_VOTE_SMA, r_sma)
    print_section("2. SMA+Mom 혼합 투표", K_VOTE_MIX, r_mix)
    print_section("3. 임계치 그래디언트 (9개 SMA, 50~90)", K_THRESH, r_thresh)

    # ── Threshold plateau ──────────────────────────────────────────
    print(f"\n  9-SMA 투표 임계치별 성능:")
    thresh_sharpes = [r['metrics']['Sharpe'] for r in r_thresh]
    for (name, _), r in zip(K_THRESH, r_thresh):
        s = r['metrics']['Sharpe']
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        bar = '█' * int(s * 15)
        print(f"    {name:<22}: {s:.3f} MDD{m['MDD']:>6.1%} Cal{calmar:>5.2f} {bar}")
    spread = max(thresh_sharpes) - min(thresh_sharpes)
    best_t = max(range(len(thresh_sharpes)), key=lambda i: thresh_sharpes[i])
    print(f"    spread: {spread:.3f} {'넓은 고원 ✓' if spread < 0.15 else '편차 있음'}")
    print(f"    최적 임계치: ≥{best_t+1}")

    # ── Vote size comparison ───────────────────────────────────────
    print(f"\n  투표 크기별 최고 성능:")
    single_best = max(r['metrics']['Sharpe'] for r in r_base)
    print(f"    단일 SMA:  {single_best:.3f} (SMA60~80 중 최고)")

    # Group by vote count
    for n_votes in [3, 5, 7]:
        subset = [(name, r) for (name, _), r in zip(K_VOTE_SMA, r_sma)
                  if name.startswith(f'{n_votes}v')]
        if subset:
            best = max(subset, key=lambda x: x[1]['metrics']['Sharpe'])
            avg = sum(x[1]['metrics']['Sharpe'] for x in subset) / len(subset)
            print(f"    {n_votes}개 투표: best {best[1]['metrics']['Sharpe']:.3f} "
                  f"avg {avg:.3f} ({best[0]})")

    mix_best = max(r_mix, key=lambda r: r['metrics']['Sharpe'])
    mix_idx = r_mix.index(mix_best)
    print(f"    혼합투표: best {mix_best['metrics']['Sharpe']:.3f} "
          f"({K_VOTE_MIX[mix_idx][0]})")

    # ── Overall Top 10 ──────────────────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  전체 Top 10 (K 투표)")
    print(f"{'=' * 100}")

    all_scored = []
    for (name, _), r in zip(ALL, results):
        m = r['metrics']
        calmar = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
        all_scored.append((name, m, r, calmar))
    all_scored.sort(key=lambda x: x[1]['Sharpe'], reverse=True)

    print(f"\n  {'순위':>3} {'K 조건':<40} {'Sharpe':>7} {'CAGR':>8} {'MDD':>7} {'Calmar':>7}")
    print(f"  {'─' * 75}")
    for i, (name, m, r, calmar) in enumerate(all_scored[:10], 1):
        print(f"  {i:>3}. {name:<40} {m['Sharpe']:>7.3f} {m['CAGR']:>+7.1%} {m['MDD']:>6.1%} {calmar:>7.2f}")

    # ── Year-by-year for top 5 ─────────────────────────────────────
    years = range(2018, 2026)
    print(f"\n  Top 5 연도별 CAGR")
    print(f"  {'K 조건':<40}", end="")
    for y in years:
        print(f" {y:>7}", end="")
    print(f" {'전체':>8}")
    print(f"  {'─' * 110}")
    for name, m, r, calmar in all_scored[:5]:
        ym = r['yearly']
        row = f"  {name:<40}"
        for y in years:
            if y in ym:
                row += f" {ym[y]['CAGR']:>+6.1%}"
            else:
                row += f" {'─':>7}"
        row += f" {m['CAGR']:>+7.1%}"
        print(row)

    # ── Vote vs Single comparison ──────────────────────────────────
    print(f"\n{'=' * 100}")
    print(f"  결론: 투표 vs 단일")
    print(f"{'=' * 100}")
    top1_name, top1_m, _, top1_c = all_scored[0]
    base_s60 = r_base[0]['metrics']
    base_c = base_s60['CAGR'] / abs(base_s60['MDD']) if base_s60['MDD'] != 0 else 0
    print(f"  SMA(60) 단일:    Sharpe {base_s60['Sharpe']:.3f}  MDD {base_s60['MDD']:>6.1%}  Calmar {base_c:.2f}")
    print(f"  투표 1위:        Sharpe {top1_m['Sharpe']:.3f}  MDD {top1_m['MDD']:>6.1%}  Calmar {top1_c:.2f}  ({top1_name})")
    diff = top1_m['Sharpe'] - base_s60['Sharpe']
    print(f"  Sharpe 차이: {diff:+.3f}")
    if diff > 0.05:
        print(f"  → 투표가 의미있게 개선됨 ✓")
    elif diff > 0:
        print(f"  → 투표가 소폭 개선 (과적합 주의)")
    else:
        print(f"  → 투표가 개선 없음, 단일 SMA가 나음")

    print(f"\n  Completed all in {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
