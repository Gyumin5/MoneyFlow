"""주식 sleeve coin-aligned BT v3 — dense mom grid + defense 변형 비교.

ON: mom_s × mom_l dense grid
OFF: defense 변형 4가지
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
from stock_engine import load_prices, ALL_TICKERS


OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
DEF_TICKERS = ("IEF", "BIL", "BNDX", "GLD", "PDBC")
CASH_KEY = 'Cash'
CANARY_SMA = 200
CANARY_HYST = 0.005
TX = 0.001
N_SNAPS = 3
STAGGER = 23
SNAP_INTERVAL = 69
CAP_PER_PICK = 1.0 / 3
N_ANCHORS = 11

PERIODS = [
    ("2017-01-01", "2025-12-31"),
    ("2018-01-01", "2025-12-31"),
    ("2020-01-01", "2025-12-31"),
    ("2021-01-01", "2025-12-31"),
]


def precompute(prices_df, mom_periods, defense_mom_periods):
    p_off = prices_df[list(OFF_R7)]
    p_def = prices_df[list(DEF_TICKERS)]
    mom_off = {n: p_off / p_off.shift(n) - 1 for n in mom_periods}
    mom_def = {n: p_def / p_def.shift(n) - 1 for n in defense_mom_periods}
    # Z-score for offense
    m63 = p_off / p_off.shift(63) - 1
    m126 = p_off / p_off.shift(126) - 1
    m252 = p_off / p_off.shift(252) - 1
    wmom = 0.5*m63 + 0.3*m126 + 0.2*m252
    rets = p_off.pct_change()
    sh126 = rets.rolling(126).mean() / rets.rolling(126).std() * np.sqrt(252)
    z_m = wmom.sub(wmom.mean(axis=1), axis=0).div(wmom.std(axis=1).replace(0, np.nan), axis=0)
    z_s = sh126.sub(sh126.mean(axis=1), axis=0).div(sh126.std(axis=1).replace(0, np.nan), axis=0)
    z = (z_m + z_s).fillna(-1e9)
    ranked = z.apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1)
    # canary
    eem = prices_df['EEM']
    sma200 = eem.rolling(CANARY_SMA).mean()
    dist = eem / sma200 - 1
    canary_raw = (dist > CANARY_HYST).astype(float) - (dist < -CANARY_HYST).astype(float)
    can = []
    prev = False
    for v in canary_raw.fillna(0).values:
        if v == 1: prev = True
        elif v == -1: prev = False
        can.append(prev)
    canary = pd.Series(can, index=eem.index)
    return ranked, mom_off, mom_def, canary


def select_on(ranked_row, ms_row, ml_row):
    healthy = []
    for t in ranked_row:
        if t == CASH_KEY: continue
        ms = ms_row.get(t, np.nan); ml = ml_row.get(t, np.nan)
        if pd.notna(ms) and pd.notna(ml) and ms > 0 and ml > 0:
            healthy.append(t)
        if len(healthy) >= 3: break
    if not healthy: return {CASH_KEY: 1.0}
    tgt = {t: CAP_PER_PICK for t in healthy}
    cash = 1.0 - CAP_PER_PICK * len(healthy)
    if cash > 0: tgt[CASH_KEY] = cash
    return tgt


def select_off(d, mom_def, defense_mode):
    """defense_mode: 'orig6m', 'mom2_63_126', 'mom2_42_126', 'short_63'"""
    if defense_mode == 'cash':
        return {CASH_KEY: 1.0}
    # rank def tickers
    if defense_mode == 'orig6m':
        scores = []
        for t in DEF_TICKERS:
            r = mom_def[126].at[d, t] if t in mom_def[126].columns else np.nan
            if pd.notna(r) and r > 0: scores.append((t, r))
        scores.sort(key=lambda x: -x[1])
        picks = [t for t, _ in scores[:3]]
    elif defense_mode == 'mom2_63_126':
        scores = []
        for t in DEF_TICKERS:
            m63 = mom_def[63].at[d, t] if t in mom_def[63].columns else np.nan
            m126 = mom_def[126].at[d, t] if t in mom_def[126].columns else np.nan
            if pd.notna(m63) and pd.notna(m126) and m63 > 0 and m126 > 0:
                scores.append((t, m126))
        scores.sort(key=lambda x: -x[1])
        picks = [t for t, _ in scores[:3]]
    elif defense_mode == 'mom2_42_126':
        scores = []
        for t in DEF_TICKERS:
            m42 = mom_def[42].at[d, t] if t in mom_def[42].columns else np.nan
            m126 = mom_def[126].at[d, t] if t in mom_def[126].columns else np.nan
            if pd.notna(m42) and pd.notna(m126) and m42 > 0 and m126 > 0:
                scores.append((t, m126))
        scores.sort(key=lambda x: -x[1])
        picks = [t for t, _ in scores[:3]]
    elif defense_mode == 'short_63':
        scores = []
        for t in DEF_TICKERS:
            r = mom_def[63].at[d, t] if t in mom_def[63].columns else np.nan
            if pd.notna(r) and r > 0: scores.append((t, r))
        scores.sort(key=lambda x: -x[1])
        picks = [t for t, _ in scores[:3]]
    else:
        picks = []
    if not picks: return {CASH_KEY: 1.0}
    # cap=1/3 + Cash slot (코인식)
    tgt = {t: CAP_PER_PICK for t in picks}
    cash = 1.0 - CAP_PER_PICK * len(picks)
    if cash > 0: tgt[CASH_KEY] = cash
    return tgt


def combine(snaps):
    out = {}; n = len(snaps)
    for s in snaps:
        for k, v in s.items():
            out[k] = out.get(k, 0) + v/n
    return out


def half_t(cur, tgt):
    keys = set(cur) | set(tgt)
    return sum(abs(cur.get(k, 0) - tgt.get(k, 0)) for k in keys) / 2


def run_one(prices_df, ranked, mom_off, mom_def, canary,
            start, end, anchor, ms, ml, drift_thr, defense_mode):
    sim_dates = prices_df.index[(prices_df.index >= start) & (prices_df.index <= end)]
    if len(sim_dates) < 50: return None
    pall = prices_df
    holdings = {CASH_KEY: 1.0}
    snaps = [{CASH_KEY: 1.0}] * N_SNAPS
    equity = []
    prev_can = canary.iloc[0] if len(canary) > 0 else False
    last_snap_can = [None] * N_SNAPS
    for i, d in enumerate(sim_dates):
        # price step
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
        # snap 갱신
        for sidx in range(N_SNAPS):
            offset = anchor + sidx * STAGGER
            if i >= offset and (i - offset) % SNAP_INTERVAL == 0:
                if can_now:
                    snaps[sidx] = select_on(ranked.at[d], mom_off[ms].loc[d], mom_off[ml].loc[d])
                else:
                    snaps[sidx] = select_off(d, mom_def, defense_mode)
                last_snap_can[sidx] = can_now
        # canary flip
        if can_now != prev_can:
            for sidx in range(N_SNAPS):
                if can_now:
                    snaps[sidx] = select_on(ranked.at[d], mom_off[ms].loc[d], mom_off[ml].loc[d])
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
                    snaps[sidx] = select_on(ranked.at[d], mom_off[ms].loc[d], mom_off[ml].loc[d])
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
    print("데이터 로딩...")
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    print(f"  완료 ({time.time()-t0:.1f}s)")

    MOM_S_GRID = [21, 30, 42, 50, 63]
    MOM_L_GRID = [84, 105, 126, 147, 168, 210, 252]
    DEFENSE_MODES = ['orig6m', 'mom2_63_126', 'mom2_42_126', 'short_63']
    THR = 0.05

    all_off_moms = sorted(set(MOM_S_GRID + MOM_L_GRID))
    all_def_moms = sorted({42, 63, 126})
    print("precompute...")
    ranked, mom_off, mom_def, canary = precompute(pdf, all_off_moms, all_def_moms)
    print(f"  완료 ({time.time()-t0:.1f}s)")

    # 결과 저장 4기간 × cfg → mean
    all_results = {}
    for ms in MOM_S_GRID:
        for ml in MOM_L_GRID:
            if ms >= ml: continue
            for dm in DEFENSE_MODES:
                cls = []
                cas = []
                mds = []
                for start, end in PERIODS:
                    sd = pd.Timestamp(start); ed = pd.Timestamp(end)
                    rs = []
                    for a in range(N_ANCHORS):
                        r = run_one(pdf, ranked, mom_off, mom_def, canary,
                                   sd, ed, a, ms, ml, THR, dm)
                        if r: rs.append(r)
                    if rs:
                        cls.append(float(np.mean([r['Cal'] for r in rs])))
                        cas.append(float(np.mean([r['CAGR'] for r in rs])))
                        mds.append(float(np.mean([r['MDD'] for r in rs])))
                if len(cls) == 4:
                    all_results[(ms, ml, dm)] = (cls, cas, mds)

    # 출력
    print("\n=== 4기간 평균 Cal (높을수록 좋음) ===")
    print(f"  {'mom_s':>5} {'mom_l':>5} {'defense':>14} {'2017+':>7} {'2018+':>7} {'2020+':>7} {'2021+':>7} {'avg':>6}  {'CAGR':>10}  {'MDD':>10}")
    sorted_keys = sorted(all_results.keys(), key=lambda k: -np.mean(all_results[k][0]))
    for k in sorted_keys[:30]:
        cls, cas, mds = all_results[k]
        ms, ml, dm = k
        avg = np.mean(cls)
        cagr_avg = np.mean(cas)
        mdd_avg = np.mean(mds)
        print(f"  {ms:>5} {ml:>5} {dm:>14} "
              f"{cls[0]:>7.2f} {cls[1]:>7.2f} {cls[2]:>7.2f} {cls[3]:>7.2f} {avg:>6.2f}  "
              f"{cagr_avg*100:>9.1f}%  {mdd_avg*100:>9.1f}%")

    # defense 별 best mom 조합
    print("\n=== Defense 모드별 best (avg Cal) ===")
    for dm in DEFENSE_MODES:
        keys_dm = [k for k in all_results if k[2] == dm]
        best_k = max(keys_dm, key=lambda k: np.mean(all_results[k][0]))
        cls, cas, mds = all_results[best_k]
        print(f"  {dm:>14}: mom_s={best_k[0]} mom_l={best_k[1]} avg Cal={np.mean(cls):.2f} "
              f"CAGR={np.mean(cas)*100:.1f}% MDD={np.mean(mds)*100:.1f}%")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
