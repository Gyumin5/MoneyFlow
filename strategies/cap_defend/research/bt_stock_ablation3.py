"""Ablation 3: drift threshold 작게 + 7% cash buffer 적용.

cells: drift_thr × cash_buffer × {EW, cap/Cash} 모두 mom + refill v2 ON 기준.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import (precompute, select_off, combine, half_t,
                              N_ANCHORS, N_SNAPS, STAGGER, SNAP_INTERVAL,
                              CASH_KEY, TX, PERIODS)
from stock_engine import load_prices, ALL_TICKERS


def select_offense_buf(ranked_row, ms_row, ml_row, mom_filter, cash_buf):
    """offense — cash_buf% 는 Cash 로 묶고 나머지 (1-cb) 를 EW 로 픽 N 개에 분배."""
    if mom_filter:
        healthy = []
        for t in ranked_row:
            if t == CASH_KEY: continue
            ms = ms_row.get(t, np.nan); ml = ml_row.get(t, np.nan)
            if pd.notna(ms) and pd.notna(ml) and ms > 0 and ml > 0:
                healthy.append(t)
            if len(healthy) >= 3: break
        picks = healthy
    else:
        picks = [t for t in ranked_row if t != CASH_KEY][:3]
    if not picks:
        return {CASH_KEY: 1.0}
    n = len(picks)
    risky_share = 1.0 - cash_buf
    tgt = {t: risky_share / n for t in picks}
    if cash_buf > 0:
        tgt[CASH_KEY] = cash_buf
    return tgt


def select_off_buf(d, mom_def, mode, cash_buf):
    """defense — cash buffer 동일 적용."""
    if mode == 'orig6m':
        scores = []
        for t in ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC'):
            r = mom_def[126].at[d, t] if t in mom_def[126].columns else np.nan
            if pd.notna(r) and r > 0: scores.append((t, r))
        scores.sort(key=lambda x: -x[1])
        picks = [t for t, _ in scores[:3]]
    else:
        picks = []
    if not picks: return {CASH_KEY: 1.0}
    n = len(picks)
    risky = 1.0 - cash_buf
    tgt = {t: risky/n for t in picks}
    if cash_buf > 0: tgt[CASH_KEY] = cash_buf
    return tgt


def run_one(prices_df, ranked, mom_off, mom_def, canary, start, end, anchor,
            ms, ml, drift_thr, cash_buf):
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
                    snaps[sidx] = select_offense_buf(ranked.at[d], mom_off[ms].loc[d],
                                                     mom_off[ml].loc[d], True, cash_buf)
                else:
                    snaps[sidx] = select_off_buf(d, mom_def, 'orig6m', cash_buf)
        if can_now != prev_can:
            for sidx in range(N_SNAPS):
                if can_now:
                    snaps[sidx] = select_offense_buf(ranked.at[d], mom_off[ms].loc[d],
                                                     mom_off[ml].loc[d], True, cash_buf)
                else:
                    snaps[sidx] = select_off_buf(d, mom_def, 'orig6m', cash_buf)
            prev_can = can_now
        combined = combine(snaps)
        total = sum(holdings.values())
        if total <= 0:
            holdings = {CASH_KEY: 1.0}; total = 1.0
        cur_w = {k: v/total for k, v in holdings.items()}
        ht = half_t(cur_w, combined)
        if ht >= drift_thr:
            # refill v2
            for sidx in range(N_SNAPS):
                if can_now:
                    snaps[sidx] = select_offense_buf(ranked.at[d], mom_off[ms].loc[d],
                                                     mom_off[ml].loc[d], True, cash_buf)
                else:
                    snaps[sidx] = select_off_buf(d, mom_def, 'orig6m', cash_buf)
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

    print(f"mom30/84 + refill v2 + EW. drift_thr × cash_buf grid:")
    print(f"  {'thr':>5} {'cash_buf':>8} {'2017+':>6} {'2018+':>6} {'2020+':>6} {'2021+':>6} {'avg':>6} {'CAGR':>7} {'MDD':>7} {'rebal':>7}")
    print("-" * 90)
    for thr in (0.02, 0.03, 0.05, 0.07, 0.10, 0.15):
        for buf in (0.0, 0.07):
            cls = []; cas = []; mds = []; rbs = []
            for start, end in PERIODS:
                sd = pd.Timestamp(start); ed = pd.Timestamp(end)
                rs = []
                for a in range(N_ANCHORS):
                    r = run_one(pdf, ranked, mom_off, mom_def, canary,
                               sd, ed, a, 30, 84, thr, buf)
                    if r: rs.append(r)
                if rs:
                    cls.append(float(np.mean([r['Cal'] for r in rs])))
                    cas.append(float(np.mean([r['CAGR'] for r in rs])))
                    mds.append(float(np.mean([r['MDD'] for r in rs])))
                    rbs.append(float(np.mean([r['rebal'] for r in rs])))
            if len(cls) == 4:
                avg = np.mean(cls)
                print(f"  {thr:>5.2f} {buf*100:>7.0f}% {cls[0]:>6.2f} {cls[1]:>6.2f} {cls[2]:>6.2f} {cls[3]:>6.2f} {avg:>6.2f} {np.mean(cas)*100:>6.1f}% {np.mean(mds)*100:>7.1f}% {np.mean(rbs):>7.1f}")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
