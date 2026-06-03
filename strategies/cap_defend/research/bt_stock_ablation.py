"""Ablation: refill (drift trigger) 효과 vs mom 추가 효과 분리.

cells:
  1) mom OFF, thr=0.30 (refill 거의 안 됨) ← pure baseline
  2) mom OFF, thr=0.10 (refill 활성) ← refill 효과
  3) mom ON,  thr=0.30 (mom 만)
  4) mom ON,  thr=0.10 (mom + refill = winner)

4 기간 × 11 anchor.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import (precompute, select_on, select_off, combine, half_t,
                              N_ANCHORS, N_SNAPS, STAGGER, SNAP_INTERVAL, CAP_PER_PICK,
                              CASH_KEY, TX, PERIODS)
from bt_stock_coin_baseline import select_no_mom
from stock_engine import load_prices, ALL_TICKERS


def run_one(prices_df, ranked, mom_off, mom_def, canary, start, end, anchor,
            ms, ml, drift_thr, mom_filter):
    sim_dates = prices_df.index[(prices_df.index >= start) & (prices_df.index <= end)]
    if len(sim_dates) < 50: return None
    pall = prices_df
    holdings = {CASH_KEY: 1.0}
    snaps = [{CASH_KEY: 1.0}] * N_SNAPS
    equity = []
    prev_can = canary.iloc[0] if len(canary) > 0 else False
    rebal_n = 0
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
                    if mom_filter:
                        snaps[sidx] = select_on(ranked.at[d], mom_off[ms].loc[d], mom_off[ml].loc[d])
                    else:
                        snaps[sidx] = select_no_mom(ranked.at[d])
                else:
                    snaps[sidx] = select_off(d, mom_def, 'orig6m')
        if can_now != prev_can:
            for sidx in range(N_SNAPS):
                if can_now:
                    if mom_filter:
                        snaps[sidx] = select_on(ranked.at[d], mom_off[ms].loc[d], mom_off[ml].loc[d])
                    else:
                        snaps[sidx] = select_no_mom(ranked.at[d])
                else:
                    snaps[sidx] = select_off(d, mom_def, 'orig6m')
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
                    if mom_filter:
                        snaps[sidx] = select_on(ranked.at[d], mom_off[ms].loc[d], mom_off[ml].loc[d])
                    else:
                        snaps[sidx] = select_no_mom(ranked.at[d])
                else:
                    snaps[sidx] = select_off(d, mom_def, 'orig6m')
            combined = combine(snaps)
            pv = total * (1 - TX * ht)
            holdings = {k: pv * w for k, w in combined.items() if w > 0}
            rebal_n += 1
        equity.append(sum(holdings.values()))
        prev_can = can_now
    eq = pd.Series(equity, index=sim_dates).dropna()
    if len(eq) < 30: return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1 if yrs > 0 else 0
    peak = eq.cummax(); mdd = float((eq/peak - 1).min())
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return dict(CAGR=cagr, MDD=mdd, Cal=cal, rebal=rebal_n)


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    ranked, mom_off, mom_def, canary = precompute(pdf, [30, 84], [42, 63, 126])

    print("Ablation: refill (drift thr) 효과 vs mom 효과 분리")
    print(f"  {'cfg':<32} {'2017+':>6} {'2018+':>6} {'2020+':>6} {'2021+':>6} {'avg':>6} {'CAGR':>8} {'MDD':>8} {'rebal':>7}")

    configs = [
        ('mom OFF, thr=0.30 (pure base)', False, 0.30),
        ('mom OFF, thr=0.10 (refill만)', False, 0.10),
        ('mom OFF, thr=0.05 (refill 강)', False, 0.05),
        ('mom ON  30/84, thr=0.30 (mom만)', True, 0.30),
        ('mom ON  30/84, thr=0.10 (both)', True, 0.10),
        ('mom ON  30/84, thr=0.05 (both 강)', True, 0.05),
    ]
    for label, mfilter, thr in configs:
        cls = []; cas = []; mds = []; rbs = []
        for start, end in PERIODS:
            sd = pd.Timestamp(start); ed = pd.Timestamp(end)
            rs = []
            for a in range(N_ANCHORS):
                r = run_one(pdf, ranked, mom_off, mom_def, canary,
                           sd, ed, a, 30, 84, thr, mfilter)
                if r: rs.append(r)
            if rs:
                cls.append(float(np.mean([r['Cal'] for r in rs])))
                cas.append(float(np.mean([r['CAGR'] for r in rs])))
                mds.append(float(np.mean([r['MDD'] for r in rs])))
                rbs.append(float(np.mean([r['rebal'] for r in rs])))
        if len(cls) == 4:
            avg = np.mean(cls)
            print(f"  {label:<32} {cls[0]:>6.2f} {cls[1]:>6.2f} {cls[2]:>6.2f} {cls[3]:>6.2f} "
                  f"{avg:>6.2f} {np.mean(cas)*100:>7.1f}% {np.mean(mds)*100:>7.1f}% {np.mean(rbs):>7.1f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
