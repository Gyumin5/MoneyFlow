"""3-mom (ms, mid, ml all > 0) health check — window rank-sum.

Uses winners from 2-mom grid:
- 2-mom peaks: (30, 72-84) and (42-45, 210-230)
- 3-mom: keep peak edges (short ms + long ml), insert mid

Grid:
- ms ∈ {21, 30, 42}
- mid ∈ {72, 84, 96, 105, 126}
- ml ∈ {189, 210, 230, 260}
- thr = 0.10

Compare against B (no mom) + top 2-mom (ms=30 ml=84) + top 2-mom (ms=45 ml=210).
"""
import sys, time
import numpy as np
import pandas as pd
from collections import defaultdict

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import precompute, half_t, CASH_KEY, TX
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_single_snap import picks_to_target, select_off
from bt_stock_window_rank import _run_multi_eq
from bt_stock_mom_grid import window_rank_sum_multi


def fresh_pick_3mom(ranked_row, ms_row, mid_row, ml_row):
    picks = []
    for t in ranked_row:
        if t == CASH_KEY: continue
        ms = ms_row.get(t, np.nan)
        mi = mid_row.get(t, np.nan)
        ml = ml_row.get(t, np.nan)
        if all(pd.notna(x) and x > 0 for x in (ms, mi, ml)):
            picks.append(t)
        if len(picks) >= 3: break
    return picks


def run_multi_3mom(pdf, ranked, mom_off, mom_def, canary, start, end, anchor,
                  drift_thr, cash_buf, ms, mid, ml,
                  snap_int=69, n_snaps=3):
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
            return fresh_pick_3mom(ranked.at[d], mom_off[ms].loc[d],
                                   mom_off[mid].loc[d], mom_off[ml].loc[d])
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
                    s['target'] = picks_to_target(s['picks'], cash_buf, 'cap')
                else:
                    s['picks'] = []
                    s['target'] = select_off(d, mom_def, cash_buf, 'cap')
        if can_now != prev_can:
            for s in snaps:
                if can_now:
                    s['picks'] = pick_for_snap(d, can_now)
                    s['target'] = picks_to_target(s['picks'], cash_buf, 'cap')
                else:
                    s['picks'] = []
                    s['target'] = select_off(d, mom_def, cash_buf, 'cap')
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
                    s['target'] = picks_to_target(s['picks'], cash_buf, 'cap')
                else:
                    s['picks'] = []
                    s['target'] = select_off(d, mom_def, cash_buf, 'cap')
            target = merge_targets()
            pv = total * (1 - TX * ht)
            holdings = {k: pv * w for k, w in target.items() if w > 0}
        equity.append(sum(holdings.values()))
        prev_can = can_now
    return pd.Series(equity, index=sim_dates).dropna()


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]

    MS = [21, 30, 42]
    MID = [72, 84, 96, 105, 126]
    ML = [189, 210, 230, 260]
    triples = [(ms, mid, ml) for ms in MS for mid in MID for ml in ML
               if ms < mid < ml]
    print(f"# 3-mom triples: {len(triples)} (+ B + 2-mom anchors)")

    all_periods = sorted(set(MS + MID + ML + [30, 45, 84, 210]))
    ranked, mom_off, mom_def, canary = precompute(pdf, all_periods, [42, 63, 126])

    sd = pd.Timestamp("2017-01-01"); ed = pd.Timestamp("2026-05-13")
    sums_all = defaultdict(float); wins_all = defaultdict(int); n_all = 0

    for anchor in range(0, 11):
        eqs = {}
        eq_B = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                             use_mom=False, drift_thr=0.10, cash_buf=0.07, weight_mode='cap')
        if eq_B is None: continue
        eqs['B'] = eq_B
        # 2-mom anchors
        for tag, ms, ml in [('2m_30_84', 30, 84), ('2m_45_210', 45, 210)]:
            eq = _run_multi_eq(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                               use_mom=True, drift_thr=0.10, cash_buf=0.07,
                               weight_mode='cap', ms=ms, ml=ml)
            if eq is not None: eqs[tag] = eq
        # 3-mom triples
        for ms, mid, ml in triples:
            key = f"3m_{ms:02d}_{mid:03d}_{ml:03d}"
            eq = run_multi_3mom(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                               drift_thr=0.10, cash_buf=0.07, ms=ms, mid=mid, ml=ml)
            if eq is not None: eqs[key] = eq
        rs = window_rank_sum_multi(eqs)
        if rs is None: continue
        sums, wins, n = rs
        for k, v in sums.items(): sums_all[k] += v
        for k, v in wins.items(): wins_all[k] += v
        n_all += n

    items = sorted(sums_all.items(), key=lambda x: x[1])
    print(f"\nTotal windows: {n_all}, total cfgs: {len(triples)+3}")
    print(f"\nALL ranking:")
    print(f"  {'cfg':<22} {'avg_rank':>9} {'win%':>6}")
    for k, rs in items:
        print(f"  {k:<22} {rs/n_all:>9.3f} {wins_all[k]/n_all*100:>5.1f}%")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
