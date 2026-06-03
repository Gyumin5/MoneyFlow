"""Stock A/B/C — 윈도우 기반 unified rank-sum (yearly rank 금지 룰).

Window sizes × strides:
- sizes: 504, 756, 1008 days (2y, 3y, 4y)
- strides: 63, 126, 252 days
- 각 window 에서 모드별 Cal 계산 → rank
- 모든 (size, stride, start) 조합의 rank 합 → unified rank-sum
- 낮은 rank-sum = 일관 우위

대상:
A) single-snap n=1 stag=23 + mom30/84 + EW + buf 7% + thr=0.10
B) 현행 V24 multi-snap n=3 stag=23 int=69, NO mom, EW + buf 7%, thr=0.10
C) multi-snap n=3 stag=23 int=69 + mom30/84 + EW + buf 7% + thr=0.10
"""
import sys, time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import precompute, half_t, CASH_KEY, TX
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_single_snap import run_one
from bt_stock_validate import run_multi


WIN_SIZES = [504, 756, 1008]  # 2y, 3y, 4y
STRIDES = [63, 126, 252]


def metrics_window(eq):
    eq = eq.dropna()
    if len(eq) < 30: return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0: return None
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1
    peak = eq.cummax(); mdd = float((eq/peak - 1).min())
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return dict(CAGR=cagr, MDD=mdd, Cal=cal)


def get_eq(pdf, ranked, mom_off, mom_def, canary, mode):
    """Single full-period equity per mode."""
    from bt_stock_coin_v3 import PERIODS
    sd = pd.Timestamp("2017-01-01"); ed = pd.Timestamp("2026-05-13")
    # Use anchor=0 for rank-sum stability; full equity → window slices
    if mode == 'A':
        # need internal eq; modify run_one signature is too much — re-impl
        # Use validate's funcs to extract equity via shim
        pass
    raise NotImplementedError


def run_get_eq(mode, pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor):
    """Return full equity series for mode A/B/C."""
    if mode == 'A':
        return _run_single_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                              30, 84, 0.10, 0.07, 'ew', 23)
    elif mode == 'B':
        return _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                             use_mom=False, drift_thr=0.10, cash_buf=0.07, weight_mode='ew')
    elif mode == 'C':
        return _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                             use_mom=True, drift_thr=0.10, cash_buf=0.07, weight_mode='ew')


def _run_single_eq(pdf, ranked, mom_off, mom_def, canary, start, end, anchor,
                   ms, ml, drift_thr, cash_buf, weight_mode, snap_interval):
    from bt_stock_single_snap import picks_to_target, fresh_pick, select_off
    sim_dates = pdf.index[(pdf.index >= start) & (pdf.index <= end)]
    if len(sim_dates) < 50: return None
    holdings = {CASH_KEY: 1.0}
    cur_picks = []; target = {CASH_KEY: 1.0}
    prev_can = canary.iloc[0] if len(canary) > 0 else False
    equity = []
    for i, d in enumerate(sim_dates):
        if i > 0:
            prev_d = sim_dates[i-1]
            for k in list(holdings.keys()):
                if k == CASH_KEY: continue
                if k in pdf.columns:
                    p_prev = pdf.at[prev_d, k] if prev_d in pdf.index else np.nan
                    p_now = pdf.at[d, k] if d in pdf.index else np.nan
                    if pd.notna(p_prev) and pd.notna(p_now) and p_prev > 0:
                        holdings[k] = holdings[k] * (p_now / p_prev)
        can_now = bool(canary.at[d]) if d in canary.index else prev_can
        if i >= anchor and (i - anchor) % snap_interval == 0:
            if can_now:
                cur_picks = fresh_pick(ranked.at[d], mom_off[ms].loc[d], mom_off[ml].loc[d])
                target = picks_to_target(cur_picks, cash_buf, weight_mode)
            else:
                cur_picks = []
                target = select_off(d, mom_def, cash_buf, weight_mode)
        if can_now != prev_can:
            if can_now:
                cur_picks = fresh_pick(ranked.at[d], mom_off[ms].loc[d], mom_off[ml].loc[d])
                target = picks_to_target(cur_picks, cash_buf, weight_mode)
            else:
                cur_picks = []
                target = select_off(d, mom_def, cash_buf, weight_mode)
            prev_can = can_now
        total = sum(holdings.values())
        if total <= 0:
            holdings = {CASH_KEY: 1.0}; total = 1.0
        cur_w = {k: v/total for k, v in holdings.items()}
        ht = half_t(cur_w, target)
        if ht >= drift_thr:
            if can_now:
                cur_picks = fresh_pick(ranked.at[d], mom_off[ms].loc[d], mom_off[ml].loc[d])
                target = picks_to_target(cur_picks, cash_buf, weight_mode)
            else:
                cur_picks = []
                target = select_off(d, mom_def, cash_buf, weight_mode)
            pv = total * (1 - TX * ht)
            holdings = {k: pv * w for k, w in target.items() if w > 0}
        equity.append(sum(holdings.values()))
        prev_can = can_now
    return pd.Series(equity, index=sim_dates).dropna()


def _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, start, end, anchor,
                  use_mom, drift_thr, cash_buf, weight_mode,
                  snap_int=69, n_snaps=3, ms=30, ml=84):
    from bt_stock_single_snap import picks_to_target, fresh_pick, select_off
    sim_dates = pdf.index[(pdf.index >= start) & (pdf.index <= end)]
    if len(sim_dates) < 50: return None
    stagger = snap_int // n_snaps
    snaps = []
    for k in range(n_snaps):
        snaps.append({'phase': (anchor + k * stagger) % snap_int,
                      'picks': [], 'target': {CASH_KEY: 1.0}})
    holdings = {CASH_KEY: 1.0}
    prev_can = canary.iloc[0] if len(canary) > 0 else False

    def pick_for_snap(d, can_now):
        if can_now:
            if use_mom:
                return fresh_pick(ranked.at[d], mom_off[ms].loc[d], mom_off[ml].loc[d])
            else:
                return [t for t in ranked.at[d] if t != CASH_KEY][:3]
        return []

    def merge_targets():
        agg = {}
        for s in snaps:
            for k, v in s['target'].items():
                agg[k] = agg.get(k, 0.0) + v / n_snaps
        return agg

    equity = []
    for i, d in enumerate(sim_dates):
        if i > 0:
            prev_d = sim_dates[i-1]
            for k in list(holdings.keys()):
                if k == CASH_KEY: continue
                if k in pdf.columns:
                    p_prev = pdf.at[prev_d, k] if prev_d in pdf.index else np.nan
                    p_now = pdf.at[d, k] if d in pdf.index else np.nan
                    if pd.notna(p_prev) and pd.notna(p_now) and p_prev > 0:
                        holdings[k] = holdings[k] * (p_now / p_prev)
        can_now = bool(canary.at[d]) if d in canary.index else prev_can
        for s in snaps:
            if (i - s['phase']) >= 0 and (i - s['phase']) % snap_int == 0:
                if can_now:
                    s['picks'] = pick_for_snap(d, can_now)
                    s['target'] = picks_to_target(s['picks'], cash_buf, weight_mode)
                else:
                    s['picks'] = []
                    s['target'] = select_off(d, mom_def, cash_buf, weight_mode)
        if can_now != prev_can:
            for s in snaps:
                if can_now:
                    s['picks'] = pick_for_snap(d, can_now)
                    s['target'] = picks_to_target(s['picks'], cash_buf, weight_mode)
                else:
                    s['picks'] = []
                    s['target'] = select_off(d, mom_def, cash_buf, weight_mode)
            prev_can = can_now
        target = merge_targets()
        total = sum(holdings.values())
        if total <= 0:
            holdings = {CASH_KEY: 1.0}; total = 1.0
        cur_w = {k: v/total for k, v in holdings.items()}
        ht = half_t(cur_w, target)
        if ht >= drift_thr:
            for s in snaps:
                if can_now:
                    s['picks'] = pick_for_snap(d, can_now)
                    s['target'] = picks_to_target(s['picks'], cash_buf, weight_mode)
                else:
                    s['picks'] = []
                    s['target'] = select_off(d, mom_def, cash_buf, weight_mode)
            target = merge_targets()
            pv = total * (1 - TX * ht)
            holdings = {k: pv * w for k, w in target.items() if w > 0}
        equity.append(sum(holdings.values()))
        prev_can = can_now
    return pd.Series(equity, index=sim_dates).dropna()


def window_rank_sum(eq_dict):
    """eq_dict = {'A': series, 'B': series, 'C': series}.
    For each window (size × stride), compute Cal per mode → rank.
    Return rank sums + win counts.
    """
    # common date range
    common = None
    for s in eq_dict.values():
        if common is None: common = s.index
        else: common = common.intersection(s.index)
    common = sorted(common)
    if len(common) < max(WIN_SIZES) + max(STRIDES):
        return None

    sums = defaultdict(float)
    wins = defaultdict(int)
    total_windows = 0
    mode_keys = sorted(eq_dict.keys())

    for size in WIN_SIZES:
        for stride in STRIDES:
            # generate window starts
            starts = list(range(0, len(common) - size, stride))
            for s_idx in starts:
                d0 = common[s_idx]; d1 = common[s_idx + size - 1]
                cals = {}
                for k in mode_keys:
                    seg = eq_dict[k].loc[d0:d1].dropna()
                    if len(seg) < 30:
                        cals[k] = np.nan; continue
                    yrs = (seg.index[-1]-seg.index[0]).days/365.25
                    if yrs <= 0: cals[k] = np.nan; continue
                    cagr = (seg.iloc[-1]/seg.iloc[0])**(1/yrs)-1
                    peak = seg.cummax(); mdd = float((seg/peak-1).min())
                    cals[k] = cagr/abs(mdd) if mdd < 0 else 0
                if any(np.isnan(v) for v in cals.values()): continue
                # rank (1=best, descending Cal)
                sorted_modes = sorted(cals.items(), key=lambda x: -x[1])
                for r, (mk, _) in enumerate(sorted_modes, 1):
                    sums[mk] += r
                wins[sorted_modes[0][0]] += 1
                total_windows += 1
    return sums, wins, total_windows


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    ranked, mom_off, mom_def, canary = precompute(pdf, [30, 84], [42, 63, 126])

    sd = pd.Timestamp("2017-01-01"); ed = pd.Timestamp("2026-05-13")
    # Avg across 11 anchors → produce 3 averaged equity series
    print("Building equity (10 anchors × 3 modes)...")
    sums_all = defaultdict(float); wins_all = defaultdict(int); n_all = 0

    # rank-sum on each anchor's eq, then aggregate
    per_anchor_results = []
    for anchor in range(0, 11):
        eq_A = _run_single_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                              30, 84, 0.10, 0.07, 'cap', 23)
        eq_B = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                             use_mom=False, drift_thr=0.10, cash_buf=0.07, weight_mode='cap')
        eq_C = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                             use_mom=True, drift_thr=0.10, cash_buf=0.07, weight_mode='cap')
        if eq_A is None or eq_B is None or eq_C is None:
            continue
        rs = window_rank_sum({'A': eq_A, 'B': eq_B, 'C': eq_C})
        if rs is None: continue
        sums, wins, n_win = rs
        per_anchor_results.append((anchor, sums, wins, n_win))
        for k, v in sums.items(): sums_all[k] += v
        for k, v in wins.items(): wins_all[k] += v
        n_all += n_win

    print("=" * 100)
    print("Window-based unified rank-sum (sizes=504/756/1008d × strides=63/126/252d × 11 anchors)")
    print("=" * 100)
    print(f"  total windows: {n_all}")
    print(f"\n  {'mode':<5} {'rank_sum':>10} {'avg_rank':>9} {'win_count':>10} {'win_pct':>8}")
    for k in sorted(sums_all):
        n = n_all
        print(f"  {k:<5} {sums_all[k]:>10.0f} {sums_all[k]/n:>9.3f} {wins_all[k]:>10d} {wins_all[k]/n*100:>7.1f}%")

    print(f"\n낮은 rank_sum = 일관 우위. avg_rank=1.0 이 best, 3.0 이 worst.")
    print(f"\nPer-anchor 분포 (rank-sum):")
    print(f"  {'anchor':<7} {'A':>8} {'B':>8} {'C':>8} {'n_win':>6}")
    for anchor, sums, wins, n_win in per_anchor_results:
        print(f"  {anchor:<7} {sums.get('A',0):>8.0f} {sums.get('B',0):>8.0f} {sums.get('C',0):>8.0f} {n_win:>6}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
