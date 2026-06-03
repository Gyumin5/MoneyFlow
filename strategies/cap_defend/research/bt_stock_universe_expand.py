"""주식 V25 universe 확장 BT — 국가/레버리지/팩터/테마 ETF 추가 비교.

각 후보: R7 base + 1 ETF (=8 universe). 동일 3-mom (30/72/230) + cap+Cash + multi-snap.
Baseline: R7.

Window rank-sum: 모든 후보 × baseline 동시 비교.
"""
import sys, time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
import bt_stock_coin_v3 as bcv3
from bt_stock_coin_v3 import precompute, CASH_KEY
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_window_rank import _run_multi_eq
from bt_stock_mom_grid import window_rank_sum_multi
from bt_stock_mom3 import run_multi_3mom


R7_BASE = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")

GROUPS = {
    'country': ['EWG', 'EWU', 'EWJ', 'EWA', 'EWC', 'EWI', 'EWQ', 'INDA',
                'FXI', 'MCHI', 'EWY', 'EWZ', 'EWW', 'EZA', 'EWS', 'EWT', 'EWH', 'KWEB'],
    'leverage': ['SSO', 'UPRO', 'QLD', 'TQQQ', 'EET', 'EDC', 'UGL', 'URE', 'SPXL'],
    'factor':   ['USMV', 'QUAL', 'MTUM', 'VLUE', 'SIZE', 'IWN', 'IWO'],
    'theme':    ['ARKK', 'SOXX', 'SMH', 'XLK', 'XLE', 'XLF', 'XLV',
                 'XLY', 'XLP', 'XLU', 'XLI', 'XLB', 'XLRE', 'XLC',
                 'ICLN', 'TAN', 'IBB', 'CIBR', 'BOTZ', 'IYR', 'IGV'],
}


def main():
    t0 = time.time()
    all_candidates = sorted(set([t for grp in GROUPS.values() for t in grp]))
    print(f"R7 base + candidates: {len(all_candidates)}")

    # All tickers to load (R7 + DEF + candidates + EEM canary)
    all_tickers = sorted(set(list(R7_BASE) + list(bcv3.DEF_TICKERS) +
                              ['EEM'] + all_candidates + list(ALL_TICKERS)))
    print(f"Loading {len(all_tickers)} tickers...")
    pm = load_prices(all_tickers, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]

    available = [t for t in all_candidates if t in pdf.columns]
    missing = sorted(set(all_candidates) - set(available))
    print(f"Available: {len(available)}, Missing: {missing}")

    all_periods = sorted(set([30, 42, 72, 230]))
    print("Precompute defaults...")

    sd = pd.Timestamp("2017-01-01"); ed = pd.Timestamp("2026-05-13")

    results = []  # (group, candidate, avg_rank, win%, samples)

    # Build configs: R7 baseline + per-candidate R7+1
    cfg_list = [('R7_base', list(R7_BASE))]
    for grp_name, candidates in GROUPS.items():
        for c in candidates:
            if c not in pdf.columns: continue
            cfg_list.append((f'{grp_name[:2].upper()}_{c}', list(R7_BASE) + [c]))

    print(f"Total configs: {len(cfg_list)}")
    sums_all = defaultdict(float); wins_all = defaultdict(int); n_all = 0

    # 한 anchor 당 모든 cfg 의 equity 계산
    for anchor in range(0, 11):
        eqs = {}
        for tag, universe in cfg_list:
            # universe별 ranked/canary 재계산 (R7+1 = 8 종목)
            bcv3.OFF_R7 = tuple(universe)
            try:
                ranked, mom_off, mom_def, canary = precompute(pdf, all_periods, [42, 63, 126])
            except Exception as e:
                continue
            # 3-mom (30, 72, 230)
            eq = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                               drift_thr=0.05, cash_buf=0.07, ms=30, mid=72, ml=230)
            if eq is not None:
                eqs[tag] = eq
        rs = window_rank_sum_multi(eqs)
        if rs is None: continue
        sums, wins, n = rs
        for k, v in sums.items(): sums_all[k] += v
        for k, v in wins.items(): wins_all[k] += v
        n_all += n
        print(f"  anchor {anchor}: {n_all} windows so far")

    items = sorted(sums_all.items(), key=lambda x: x[1])
    n_cfgs = len(cfg_list)
    print(f"\nTotal windows: {n_all}, total cfgs: {n_cfgs}")
    print(f"\nALL ranking (lower=better, n_cfgs={n_cfgs} → ranks 1~{n_cfgs}):")
    print(f"  {'cfg':<20} {'avg_rank':>9} {'win%':>6} {'vs_R7':>7}")
    r7_rank = sums_all.get('R7_base', 0) / n_all if n_all > 0 else 0
    for k, rs in items:
        marker = ' ← R7' if k == 'R7_base' else ''
        diff = (rs/n_all) - r7_rank
        print(f"  {k:<20} {rs/n_all:>9.3f} {wins_all[k]/n_all*100:>5.1f}% {diff:>+7.3f}{marker}")

    # Group summary
    print(f"\n--- By group avg rank improvement vs R7 ---")
    for grp_name in GROUPS:
        prefix = grp_name[:2].upper() + '_'
        grp_items = [(k, v) for k, v in items if k.startswith(prefix)]
        if grp_items:
            best = grp_items[0]
            worst = grp_items[-1]
            top5 = grp_items[:5]
            print(f"\n  [{grp_name}] {len(grp_items)} candidates, R7={r7_rank:.3f}")
            print(f"    Best: {best[0]} avg={best[1]/n_all:.3f} ({(best[1]/n_all - r7_rank):+.3f})")
            print(f"    Worst: {worst[0]} avg={worst[1]/n_all:.3f} ({(worst[1]/n_all - r7_rank):+.3f})")
            print(f"    Top 5: " + ", ".join([f"{k}({rs/n_all:.2f})" for k, rs in top5]))

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
