"""N_SNAPS=1 (single snap) + mom + refill on drift BT.

Drift fire 시 fresh select top3 → cap=1/3+Cash slot 또는 EW. multi-snap 불필요.
비교: A (coin V24 multi-snap partial swap) vs B (single snap full reset).
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import (precompute, combine, half_t,
                              CASH_KEY, TX, PERIODS, N_ANCHORS)
from stock_engine import load_prices, ALL_TICKERS

DEF_TICKERS = ("IEF", "BIL", "BNDX", "GLD", "PDBC")
SINGLE_INTERVAL = 23  # single snap 갱신 주기 (월간 anchor 비슷)


def fresh_pick(ranked_row, ms_row, ml_row):
    picks = []
    for t in ranked_row:
        if t == CASH_KEY: continue
        ms = ms_row.get(t, np.nan); ml = ml_row.get(t, np.nan)
        if pd.notna(ms) and pd.notna(ml) and ms > 0 and ml > 0:
            picks.append(t)
        if len(picks) >= 3: break
    return picks


def picks_to_target(picks, cash_buf, mode):
    """mode='cap' (1/3+Cash) or 'ew' (1/n)."""
    if not picks: return {CASH_KEY: 1.0}
    risky = 1.0 - cash_buf
    if mode == 'cap':
        per = risky / 3
        tgt = {t: per for t in picks}
        used = per * len(picks)
        cash = risky - used + cash_buf
    else:  # ew
        per = risky / len(picks)
        tgt = {t: per for t in picks}
        cash = cash_buf
    if cash > 0: tgt[CASH_KEY] = cash
    return tgt


def select_off(d, mom_def, cash_buf, mode):
    scores = []
    for t in DEF_TICKERS:
        r = mom_def[126].at[d, t] if t in mom_def[126].columns else np.nan
        if pd.notna(r) and r > 0: scores.append((t, r))
    scores.sort(key=lambda x: -x[1])
    picks = [t for t, _ in scores[:3]]
    return picks_to_target(picks, cash_buf, mode)


def run_one(pdf, ranked, mom_off, mom_def, canary, start, end, anchor,
            ms, ml, drift_thr, cash_buf, weight_mode, snap_interval):
    sim_dates = pdf.index[(pdf.index >= start) & (pdf.index <= end)]
    if len(sim_dates) < 50: return None
    pall = pdf
    holdings = {CASH_KEY: 1.0}
    cur_picks = []
    target = {CASH_KEY: 1.0}
    prev_can = canary.iloc[0] if len(canary) > 0 else False
    equity = []
    for i, d in enumerate(sim_dates):
        if i > 0:
            prev_d = sim_dates[i-1]
            for k in list(holdings.keys()):
                if k == CASH_KEY: continue
                if k in pall.columns:
                    p_prev = pall.at[prev_d, k] if prev_d in pall.index else np.nan
                    p_now = pall.at[d, k] if d in pall.index else np.nan
                    if pd.notna(p_prev) and pd.notna(p_now) and p_prev > 0:
                        holdings[k] = holdings[k] * (p_now / p_prev)
        can_now = bool(canary.at[d]) if d in canary.index else prev_can
        # snap 갱신 — anchor 부터 snap_interval 마다
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
            # refill: full reselect (single snap 이라 다양성 손실 없음)
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
    eq = pd.Series(equity, index=sim_dates).dropna()
    if len(eq) < 30: return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1 if yrs > 0 else 0
    peak = eq.cummax(); mdd = float((eq/peak - 1).min())
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return dict(CAGR=cagr, MDD=mdd, Cal=cal)


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    ranked, mom_off, mom_def, canary = precompute(pdf, [30, 84], [42, 63, 126])

    print("Single snap (N=1) + mom30/84 + refill on drift")
    print(f"  {'cfg':<35} {'2017+':>6} {'2018+':>6} {'2020+':>6} {'2021+':>6} {'avg':>6} {'CAGR':>7} {'MDD':>7}")
    print("-" * 90)
    for snap_int in (21, 42, 69):
        for thr in (0.05, 0.10):
            for buf in (0.0, 0.07):
                for mode in ('ew', 'cap'):
                    cls = []; cas = []; mds = []
                    for start, end in PERIODS:
                        sd = pd.Timestamp(start); ed = pd.Timestamp(end)
                        rs = []
                        for a in range(N_ANCHORS):
                            r = run_one(pdf, ranked, mom_off, mom_def, canary,
                                       sd, ed, a, 30, 84, thr, buf, mode, snap_int)
                            if r: rs.append(r)
                        if rs:
                            cls.append(float(np.mean([r['Cal'] for r in rs])))
                            cas.append(float(np.mean([r['CAGR'] for r in rs])))
                            mds.append(float(np.mean([r['MDD'] for r in rs])))
                    if len(cls) == 4:
                        avg = np.mean(cls)
                        label = f"int={snap_int} thr={thr:.2f} buf={int(buf*100)}% w={mode}"
                        print(f"  {label:<35} {cls[0]:>6.2f} {cls[1]:>6.2f} {cls[2]:>6.2f} {cls[3]:>6.2f} {avg:>6.2f} {np.mean(cas)*100:>6.1f}% {np.mean(mds)*100:>7.1f}%")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
