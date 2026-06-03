"""v3 framework 안에서 mom 필터 OFF baseline 측정.

비교 위해 동일 BT 구현 (multi-snap N=3, cap=1/3+Cash, orig6m defense)
+ mom 필터 없음 (canary ON 시 top3 Z-score 그대로).
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import (precompute, select_on, select_off, combine, half_t,
                              N_ANCHORS, N_SNAPS, STAGGER, SNAP_INTERVAL, CAP_PER_PICK,
                              CASH_KEY, TX, PERIODS, OFF_R7, DEF_TICKERS)
from stock_engine import load_prices, ALL_TICKERS


def select_no_mom(ranked_row):
    picks = [t for t in ranked_row if t != CASH_KEY][:3]
    if not picks: return {CASH_KEY: 1.0}
    tgt = {t: CAP_PER_PICK for t in picks}
    cash = 1.0 - CAP_PER_PICK * len(picks)
    if cash > 0: tgt[CASH_KEY] = cash
    return tgt


def run_one(prices_df, ranked, mom_def, canary, start, end, anchor, drift_thr, defense_mode='orig6m'):
    sim_dates = prices_df.index[(prices_df.index >= start) & (prices_df.index <= end)]
    if len(sim_dates) < 50: return None
    pall = prices_df
    holdings = {CASH_KEY: 1.0}
    snaps = [{CASH_KEY: 1.0}] * N_SNAPS
    equity = []
    prev_can = canary.iloc[0] if len(canary) > 0 else False
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
        for sidx in range(N_SNAPS):
            offset = anchor + sidx * STAGGER
            if i >= offset and (i - offset) % SNAP_INTERVAL == 0:
                if can_now:
                    snaps[sidx] = select_no_mom(ranked.at[d])
                else:
                    snaps[sidx] = select_off(d, mom_def, defense_mode)
        if can_now != prev_can:
            for sidx in range(N_SNAPS):
                if can_now:
                    snaps[sidx] = select_no_mom(ranked.at[d])
                else:
                    snaps[sidx] = select_off(d, mom_def, defense_mode)
            prev_can = can_now
        combined = combine(snaps)
        total = sum(holdings.values())
        if total <= 0:
            holdings = {CASH_KEY: 1.0}; total = 1.0
        cur_w = {k: v/total for k, v in holdings.items()}
        ht = half_t(cur_w, combined)
        if ht >= drift_thr:
            for sidx in range(N_SNAPS):
                if can_now:
                    snaps[sidx] = select_no_mom(ranked.at[d])
                else:
                    snaps[sidx] = select_off(d, mom_def, defense_mode)
            combined = combine(snaps)
            pv = total * (1 - TX * ht)
            holdings = {k: pv * w for k, w in combined.items() if w > 0}
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
    ranked, _, mom_def, canary = precompute(pdf, [30], [42, 63, 126])
    print("v3 framework 동일 — mom 필터만 OFF, drift=0.10")
    print(f"  {'period':<15} {'CAGR':>8} {'MDD':>8} {'Cal':>6}")
    cls = []
    for start, end in PERIODS:
        sd = pd.Timestamp(start); ed = pd.Timestamp(end)
        rs = []
        for a in range(N_ANCHORS):
            r = run_one(pdf, ranked, mom_def, canary, sd, ed, a, 0.10)
            if r: rs.append(r)
        if rs:
            ca = float(np.mean([r['CAGR'] for r in rs]))
            md = float(np.mean([r['MDD'] for r in rs]))
            cl = float(np.mean([r['Cal'] for r in rs]))
            cls.append(cl)
            print(f"  {start}~{end[:4]} {ca:>+8.1%} {md:>+8.1%} {cl:>6.2f}")
    print(f"  avg Cal = {np.mean(cls):.2f}")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
