"""Ablation 2: 진짜 refill v2 (drift-fire snap 재산정) 분리 + cap+Cash slot 분리.

각 cell 토글:
- mom_filter: ON/OFF (canary ON 시 mom_short>0 AND mom_long>0)
- refill_on_drift: drift fire 시 snap 재산정 (mom_filter 가 ON 일 때만 의미)
- cap_cash: cap=1/3+Cash slot (vs EW 1/n)
- drift_thr: 0.10 (default), 0.30 (drift 비활성화 효과)

cells:
1. base — no mom / no refill / EW / thr=0.30
2. drift rebal — no mom / no refill / EW / thr=0.10 (그냥 잦은 리밸)
3. mom only at snap — mom ON / no refill / EW / thr=0.30
4. mom + refill v2 — mom ON / refill / EW / thr=0.10
5. mom + refill + cap/Cash (winner) — mom ON / refill / cap+Cash / thr=0.10
6. cap+Cash only — no mom / no refill / cap+Cash / thr=0.30
7. drift rebal + cap+Cash — no mom / no refill / cap+Cash / thr=0.10
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import (precompute, select_off, combine, half_t,
                              N_ANCHORS, N_SNAPS, STAGGER, SNAP_INTERVAL, CAP_PER_PICK,
                              CASH_KEY, TX, PERIODS)
from stock_engine import load_prices, ALL_TICKERS


def select_offense(ranked_row, ms_row, ml_row, mom_filter, cap_cash):
    """offense snapshot.
    mom_filter: True → mom>0 universe filter
    cap_cash: True → cap=1/3 + Cash slot. False → EW (1/n)
    """
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
    if not picks: return {CASH_KEY: 1.0}
    n = len(picks)
    if cap_cash:
        tgt = {t: CAP_PER_PICK for t in picks}
        cash = 1.0 - CAP_PER_PICK * n
        if cash > 0: tgt[CASH_KEY] = cash
    else:
        tgt = {t: 1.0/n for t in picks}
    return tgt


def run_one(prices_df, ranked, mom_off, mom_def, canary, start, end, anchor,
            ms_period, ml_period, drift_thr, mom_filter, refill_on_drift, cap_cash):
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
        # snap 갱신 — anchor 기반
        for sidx in range(N_SNAPS):
            offset = anchor + sidx * STAGGER
            if i >= offset and (i - offset) % SNAP_INTERVAL == 0:
                if can_now:
                    snaps[sidx] = select_offense(ranked.at[d], mom_off[ms_period].loc[d],
                                                  mom_off[ml_period].loc[d], mom_filter, cap_cash)
                else:
                    snaps[sidx] = select_off(d, mom_def, 'orig6m')
        if can_now != prev_can:
            for sidx in range(N_SNAPS):
                if can_now:
                    snaps[sidx] = select_offense(ranked.at[d], mom_off[ms_period].loc[d],
                                                  mom_off[ml_period].loc[d], mom_filter, cap_cash)
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
            # refill v2 옵션: snap 을 현재 상태로 재산정
            if refill_on_drift:
                for sidx in range(N_SNAPS):
                    if can_now:
                        snaps[sidx] = select_offense(ranked.at[d], mom_off[ms_period].loc[d],
                                                      mom_off[ml_period].loc[d], mom_filter, cap_cash)
                    else:
                        snaps[sidx] = select_off(d, mom_def, 'orig6m')
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
    ranked, mom_off, mom_def, canary = precompute(pdf, [30, 84], [42, 63, 126])

    # cells: (label, mom_filter, refill_on_drift, cap_cash, drift_thr)
    cells = [
        ('A pure base (engine-like)',     False, False, False, 0.30),
        ('B drift rebal only (EW)',       False, False, False, 0.10),
        ('C drift rebal + cap/Cash',      False, False, True,  0.10),
        ('D mom at snap only (EW)',       True,  False, False, 0.30),
        ('E mom + drift rebal (EW)',      True,  False, False, 0.10),
        ('F mom + refill v2 (EW)',        True,  True,  False, 0.10),
        ('G mom + refill v2 + cap/Cash',  True,  True,  True,  0.10),  # winner
        ('H cap/Cash only',               False, False, True,  0.30),
        ('I mom + cap/Cash, no drift',    True,  False, True,  0.30),
    ]
    print(f"{'cell':<35} {'2017+':>6} {'2018+':>6} {'2020+':>6} {'2021+':>6} {'avg':>6} {'CAGR':>7} {'MDD':>7}")
    print("-" * 95)
    rows = []
    for label, mf, rd, cc, thr in cells:
        cls = []; cas = []; mds = []
        for start, end in PERIODS:
            sd = pd.Timestamp(start); ed = pd.Timestamp(end)
            rs = []
            for a in range(N_ANCHORS):
                r = run_one(pdf, ranked, mom_off, mom_def, canary,
                           sd, ed, a, 30, 84, thr, mf, rd, cc)
                if r: rs.append(r)
            if rs:
                cls.append(float(np.mean([r['Cal'] for r in rs])))
                cas.append(float(np.mean([r['CAGR'] for r in rs])))
                mds.append(float(np.mean([r['MDD'] for r in rs])))
        if len(cls) == 4:
            avg = np.mean(cls)
            rows.append((label, avg, cls, cas, mds))
            print(f"{label:<35} {cls[0]:>6.2f} {cls[1]:>6.2f} {cls[2]:>6.2f} {cls[3]:>6.2f} {avg:>6.2f} {np.mean(cas)*100:>6.1f}% {np.mean(mds)*100:>7.1f}%")
    print()
    print("=== avg Cal 순위 ===")
    for r in sorted(rows, key=lambda x: -x[1]):
        print(f"  {r[1]:.2f}  {r[0]}")
    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
