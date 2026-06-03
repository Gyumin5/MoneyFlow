"""Stock 검증 — single-snap winner (Cal 1.17) vs alternatives.

Compares:
- A) single-snap n=1 stag=23 + mom30/84 + EW + buf=7% thr=0.10 (winner)
- B) current V24 multi-snap n=3 stag=23 int=69, NO mom, EW (baseline)
- C) current V24 multi-snap + mom30/84 added

Gates: yearly, anchor offset, fee 2x stress.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import (precompute, half_t, CASH_KEY, TX, PERIODS, N_ANCHORS)
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_single_snap import run_one, picks_to_target, fresh_pick, select_off, DEF_TICKERS


def run_multi(pdf, ranked, mom_off, mom_def, canary, start, end, anchor,
              use_mom, drift_thr, cash_buf, weight_mode,
              snap_int=69, n_snaps=3, ms=30, ml=84):
    """multi-snap stagger BT. use_mom=True 면 fresh_pick (mom30/84), False 면 ranked top3."""
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
                # no mom — just top3 ranked excluding cash
                ps = [t for t in ranked.at[d] if t != CASH_KEY][:3]
                return ps
        return []

    def merge_targets():
        # average n_snaps targets
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
        # snap refresh
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
            # refill: rebuild affected snap targets
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
    eq = pd.Series(equity, index=sim_dates).dropna()
    if len(eq) < 30: return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1 if yrs > 0 else 0
    peak = eq.cummax(); mdd = float((eq/peak - 1).min())
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return dict(CAGR=cagr, MDD=mdd, Cal=cal)


def yearly_breakdown(pdf, ranked, mom_off, mom_def, canary, mode_fn, label):
    """mode_fn(start, end, anchor) -> result dict"""
    out = []
    for yr in range(2017, 2027):
        sd = pd.Timestamp(f"{yr}-01-01"); ed = pd.Timestamp(f"{yr}-12-31")
        if sd > pdf.index[-1] or ed < pdf.index[0]: continue
        rs = []
        for a in range(N_ANCHORS):
            r = mode_fn(sd, ed, a)
            if r: rs.append(r)
        if rs:
            out.append((yr,
                       float(np.mean([r['CAGR'] for r in rs])),
                       float(np.mean([r['MDD'] for r in rs])),
                       float(np.mean([r['Cal'] for r in rs]))))
    return out


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    ranked, mom_off, mom_def, canary = precompute(pdf, [30, 84], [42, 63, 126])

    # 3 modes
    def mode_A(sd, ed, a):
        return run_one(pdf, ranked, mom_off, mom_def, canary, sd, ed, a,
                       30, 84, 0.10, 0.07, 'ew', 23)
    def mode_B(sd, ed, a):  # current V24 multi-snap, no mom, EW
        return run_multi(pdf, ranked, mom_off, mom_def, canary, sd, ed, a,
                         use_mom=False, drift_thr=0.10, cash_buf=0.07, weight_mode='ew')
    def mode_C(sd, ed, a):  # multi-snap + mom
        return run_multi(pdf, ranked, mom_off, mom_def, canary, sd, ed, a,
                         use_mom=True, drift_thr=0.10, cash_buf=0.07, weight_mode='ew')

    # [1] Period averages
    print("=" * 100)
    print("[1] Period averages (10-anchor avg per period)")
    print("=" * 100)
    print(f"  {'mode':<40} {'2017+':>6} {'2018+':>6} {'2020+':>6} {'2021+':>6} {'avg':>6} {'CAGR':>7} {'MDD':>7}")
    for fn, label in [(mode_A, "A) single-snap+mom EW buf7%"),
                      (mode_B, "B) current V24 multi-snap, no mom"),
                      (mode_C, "C) multi-snap+mom EW buf7%")]:
        cls = []; cas = []; mds = []
        for start, end in PERIODS:
            sd = pd.Timestamp(start); ed = pd.Timestamp(end)
            rs = [fn(sd, ed, a) for a in range(N_ANCHORS)]
            rs = [r for r in rs if r]
            if rs:
                cls.append(float(np.mean([r['Cal'] for r in rs])))
                cas.append(float(np.mean([r['CAGR'] for r in rs])))
                mds.append(float(np.mean([r['MDD'] for r in rs])))
        if len(cls) == 4:
            print(f"  {label:<40} {cls[0]:>6.2f} {cls[1]:>6.2f} {cls[2]:>6.2f} {cls[3]:>6.2f} {np.mean(cls):>6.2f} {np.mean(cas)*100:>6.1f}% {np.mean(mds)*100:>7.1f}%")

    # [2] Yearly
    print()
    print("=" * 100)
    print("[2] Yearly breakdown (10-anchor avg per yr)")
    print("=" * 100)
    a_y = yearly_breakdown(pdf, ranked, mom_off, mom_def, canary, mode_A, "A")
    b_y = yearly_breakdown(pdf, ranked, mom_off, mom_def, canary, mode_B, "B")
    c_y = yearly_breakdown(pdf, ranked, mom_off, mom_def, canary, mode_C, "C")
    a_m = {y: c for y, _, _, c in a_y}
    b_m = {y: c for y, _, _, c in b_y}
    c_m = {y: c for y, _, _, c in c_y}
    print(f"  {'yr':<6} {'A_Cal':>6} {'B_Cal':>6} {'C_Cal':>6} {'A-B':>6} {'C-B':>6}")
    a_win_b = 0; c_win_b = 0; a_win_c = 0
    for y in sorted(set(a_m) | set(b_m) | set(c_m)):
        ac = a_m.get(y, np.nan); bc = b_m.get(y, np.nan); cc = c_m.get(y, np.nan)
        ab = ac - bc if pd.notna(ac) and pd.notna(bc) else np.nan
        cb = cc - bc if pd.notna(cc) and pd.notna(bc) else np.nan
        if pd.notna(ab) and ab > 0: a_win_b += 1
        if pd.notna(cb) and cb > 0: c_win_b += 1
        if pd.notna(ac) and pd.notna(cc) and ac > cc: a_win_c += 1
        print(f"  {y:<6} {ac:>6.2f} {bc:>6.2f} {cc:>6.2f} {ab:>+6.2f} {cb:>+6.2f}")
    print(f"  → A>B {a_win_b}/{len(a_m)},  C>B {c_win_b}/{len(c_m)},  A>C {a_win_c}/{min(len(a_m),len(c_m))}")

    # [3] Anchor offset
    print()
    print("=" * 100)
    print("[3] Anchor offset (2018+ period)")
    print("=" * 100)
    sd = pd.Timestamp(PERIODS[1][0]); ed = pd.Timestamp(PERIODS[1][1])
    print(f"  {'anchor':<7} {'A_Cal':>6} {'B_Cal':>6} {'C_Cal':>6} {'A-B':>6} {'C-B':>6}")
    diffs_ab = []; diffs_cb = []
    for a in range(N_ANCHORS):
        ra = mode_A(sd, ed, a); rb = mode_B(sd, ed, a); rc = mode_C(sd, ed, a)
        if ra and rb and rc:
            ab = ra['Cal'] - rb['Cal']; cb = rc['Cal'] - rb['Cal']
            diffs_ab.append(ab); diffs_cb.append(cb)
            print(f"  {a:<7} {ra['Cal']:>6.2f} {rb['Cal']:>6.2f} {rc['Cal']:>6.2f} {ab:>+6.2f} {cb:>+6.2f}")
    print(f"  → A-B avg={np.mean(diffs_ab):+.2f} sigma={np.std(diffs_ab):.2f} pos={sum(1 for d in diffs_ab if d>0)}/{len(diffs_ab)}")
    print(f"  → C-B avg={np.mean(diffs_cb):+.2f} sigma={np.std(diffs_cb):.2f} pos={sum(1 for d in diffs_cb if d>0)}/{len(diffs_cb)}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
