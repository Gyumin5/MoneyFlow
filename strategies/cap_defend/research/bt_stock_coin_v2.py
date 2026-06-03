"""주식 sleeve coin-aligned BT v2 — multi-snap stagger + multi-mom (벡터화 최적화).

precompute 한 번:
- ranked Z-score per day (DataFrame: index=date, value=list of tickers ranked)
- mom_dict[N] DataFrame (index=date, columns=tickers)
- canary per day (Series)

룰:
- canary: EEM SMA200, hyst 0.5%
- canary OFF → 전액 Cash
- canary ON: universe = mom_short>0 AND mom_long>0 → top min(3, n_healthy) by Z-score → cap=1/3, 부족분 Cash
- multi-snap: N_SNAPS=3, STAGGER=23, SNAP_INTERVAL=69
- drift trigger: 매일 ht 계산, fire 시 즉시 매매
- 11-anchor 평균 × 4 기간
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
from stock_engine import load_prices, ALL_TICKERS


OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
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


def precompute_all(prices_df, mom_periods):
    """전체 기간 한 번에 ranked Z, mom, canary 산출."""
    # mom dataframes
    mom_dfs = {}
    for n in mom_periods:
        ret = prices_df[list(OFF_R7)] / prices_df[list(OFF_R7)].shift(n) - 1
        mom_dfs[n] = ret
    # Z-score: wmom (0.5*m63 + 0.3*m126 + 0.2*m252) + sharpe126
    p = prices_df[list(OFF_R7)]
    m63 = p / p.shift(63) - 1
    m126 = p / p.shift(126) - 1
    m252 = p / p.shift(252) - 1
    wmom = 0.5*m63 + 0.3*m126 + 0.2*m252
    rets = p.pct_change()
    sh126 = rets.rolling(126).mean() / rets.rolling(126).std() * np.sqrt(252)
    # z = (wmom - wmom.row_mean)/row_std + (sh - sh.row_mean)/sh.row_std
    z_m = wmom.sub(wmom.mean(axis=1), axis=0).div(wmom.std(axis=1).replace(0, np.nan), axis=0)
    z_s = sh126.sub(sh126.mean(axis=1), axis=0).div(sh126.std(axis=1).replace(0, np.nan), axis=0)
    z = (z_m + z_s).fillna(-1e9)  # NaN → 최하위 (선정 안 됨)
    # ranked: 각 row 의 z 내림차순으로 ticker 리스트
    ranked = z.apply(lambda row: row.sort_values(ascending=False).index.tolist(), axis=1)
    # canary
    eem = prices_df['EEM']
    sma200 = eem.rolling(CANARY_SMA).mean()
    dist = eem / sma200 - 1
    canary_raw = (dist > CANARY_HYST).astype(float) - (dist < -CANARY_HYST).astype(float)
    # canary_raw: 1=ON, -1=OFF, 0=dead zone (이전 상태 유지)
    can = []
    prev = False
    for v in canary_raw.fillna(0).values:
        if v == 1: prev = True
        elif v == -1: prev = False
        can.append(prev)
    canary = pd.Series(can, index=eem.index)
    return ranked, mom_dfs, canary


def half_turnover(cur, tgt):
    keys = set(cur) | set(tgt)
    return sum(abs(cur.get(k, 0) - tgt.get(k, 0)) for k in keys) / 2


def select_snap(ranked_row, mom_s_row, mom_l_row):
    """canary ON 시 picks 산출."""
    healthy = []
    for t in ranked_row:
        if t == CASH_KEY: continue
        ms = mom_s_row.get(t, np.nan)
        ml = mom_l_row.get(t, np.nan)
        if pd.notna(ms) and pd.notna(ml) and ms > 0 and ml > 0:
            healthy.append(t)
        if len(healthy) >= 3: break
    if not healthy:
        return {CASH_KEY: 1.0}
    tgt = {t: CAP_PER_PICK for t in healthy}
    cash = 1.0 - CAP_PER_PICK * len(healthy)
    if cash > 0:
        tgt[CASH_KEY] = cash
    return tgt


def select_snap_baseline(ranked_row):
    """baseline (mom 필터 없음)."""
    picks = [t for t in ranked_row if t != CASH_KEY][:3]
    if not picks: return {CASH_KEY: 1.0}
    tgt = {t: CAP_PER_PICK for t in picks}
    cash = 1.0 - CAP_PER_PICK * len(picks)
    if cash > 0: tgt[CASH_KEY] = cash
    return tgt


def combine_snaps(snaps):
    out = {}
    n = len(snaps)
    for s in snaps:
        for k, v in s.items():
            out[k] = out.get(k, 0) + v / n
    return out


def run_one(prices_df, ranked, mom_s_df, mom_l_df, canary, start, end,
            anchor, drift_threshold, mom_filter=True):
    sim_dates = prices_df.index[(prices_df.index >= start) & (prices_df.index <= end)]
    if len(sim_dates) < 50: return None
    pdf = prices_df[list(OFF_R7)]
    holdings = {CASH_KEY: 1.0}
    snapshots = [{CASH_KEY: 1.0}] * N_SNAPS
    equity = []
    rebal = 0
    prev_canary = canary.iloc[0] if len(canary) > 0 else False
    for i, d in enumerate(sim_dates):
        # 일일 가격 진행
        if i > 0:
            prev_d = sim_dates[i-1]
            for k in list(holdings.keys()):
                if k == CASH_KEY: continue
                p_prev = pdf.at[prev_d, k] if d in pdf.index and prev_d in pdf.index else np.nan
                p_now = pdf.at[d, k] if d in pdf.index else np.nan
                if pd.notna(p_prev) and pd.notna(p_now) and p_prev > 0:
                    holdings[k] = holdings[k] * (p_now / p_prev)
        # snap 갱신 — 각 snap 의 anchor + sidx*STAGGER 마다
        can_now = bool(canary.at[d]) if d in canary.index else prev_canary
        for sidx in range(N_SNAPS):
            offset = anchor + sidx * STAGGER
            if i >= offset and (i - offset) % SNAP_INTERVAL == 0:
                if can_now:
                    if mom_filter:
                        snapshots[sidx] = select_snap(
                            ranked.at[d], mom_s_df.loc[d], mom_l_df.loc[d])
                    else:
                        snapshots[sidx] = select_snap_baseline(ranked.at[d])
                else:
                    snapshots[sidx] = {CASH_KEY: 1.0}
        # canary flip
        if can_now != prev_canary:
            for sidx in range(N_SNAPS):
                if can_now:
                    if mom_filter:
                        snapshots[sidx] = select_snap(
                            ranked.at[d], mom_s_df.loc[d], mom_l_df.loc[d])
                    else:
                        snapshots[sidx] = select_snap_baseline(ranked.at[d])
                else:
                    snapshots[sidx] = {CASH_KEY: 1.0}
            prev_canary = can_now
        # drift check
        total = sum(holdings.values())
        if total <= 0:
            holdings = {CASH_KEY: 1.0}; total = 1.0
        cur_w = {k: v/total for k, v in holdings.items()}
        combined = combine_snaps(snapshots)
        ht = half_turnover(cur_w, combined)
        if ht >= drift_threshold:
            if can_now:
                for sidx in range(N_SNAPS):
                    if mom_filter:
                        snapshots[sidx] = select_snap(
                            ranked.at[d], mom_s_df.loc[d], mom_l_df.loc[d])
                    else:
                        snapshots[sidx] = select_snap_baseline(ranked.at[d])
            else:
                snapshots = [{CASH_KEY: 1.0}] * N_SNAPS
            combined = combine_snaps(snapshots)
            pv = total * (1 - TX * ht)
            holdings = {k: pv * w for k, w in combined.items() if w > 0}
        equity.append(sum(holdings.values()))
        if can_now != prev_canary:
            prev_canary = can_now
    eq = pd.Series(equity, index=sim_dates).dropna()
    if len(eq) < 30: return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1 if yrs > 0 else 0
    peak = eq.cummax(); mdd = float((eq/peak - 1).min())
    cal = cagr / abs(mdd) if mdd < 0 else 0
    return dict(CAGR=cagr, MDD=mdd, Cal=cal, rebal=rebal)


def main():
    t0 = time.time()
    print("데이터 로딩...")
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    print(f"  완료 ({time.time()-t0:.1f}s). 종목 {len(pdf.columns)} rows {len(pdf)}")

    MOM_S_GRID = [21, 42, 63]
    MOM_L_GRID = [84, 126, 168, 252]
    THR_GRID = [0.03, 0.05, 0.10]
    all_moms = sorted(set(MOM_S_GRID + MOM_L_GRID))

    print("precompute...")
    ranked, mom_dfs, canary = precompute_all(pdf, all_moms)
    print(f"  precompute 완료 ({time.time()-t0:.1f}s)")

    for start, end in PERIODS:
        sd = pd.Timestamp(start); ed = pd.Timestamp(end)
        print(f"\n[{start} ~ {end}]  11-anchor, N=3 stagger=23 interval=69")
        print(f"  {'mom_s':>5} {'mom_l':>5} {'thr':>5} {'CAGR':>8} {'MDD':>8} {'Cal':>6}  Δ")
        # baseline
        rs = []
        for a in range(N_ANCHORS):
            r = run_one(pdf, ranked, mom_dfs[63], mom_dfs[63], canary,
                       sd, ed, anchor=a, drift_threshold=0.10, mom_filter=False)
            if r: rs.append(r)
        if rs:
            ca = float(np.mean([r['CAGR'] for r in rs]))
            md = float(np.mean([r['MDD'] for r in rs]))
            cl = float(np.mean([r['Cal'] for r in rs]))
            print(f"  baseline-thr0.10  {ca:>+8.1%} {md:>+8.1%} {cl:>6.2f}  (engine-like)")
            base_cl = cl
        else:
            base_cl = 0
        # also baseline with thr=0.05, 0.03
        for bt in (0.03, 0.05):
            rs = []
            for a in range(N_ANCHORS):
                r = run_one(pdf, ranked, mom_dfs[63], mom_dfs[63], canary,
                           sd, ed, anchor=a, drift_threshold=bt, mom_filter=False)
                if r: rs.append(r)
            if rs:
                ca = float(np.mean([r['CAGR'] for r in rs]))
                md = float(np.mean([r['MDD'] for r in rs]))
                cl = float(np.mean([r['Cal'] for r in rs]))
                d = cl - base_cl
                print(f"  baseline-thr{bt:.2f}  {ca:>+8.1%} {md:>+8.1%} {cl:>6.2f}  Δ {d:+5.2f}")

        for ms in MOM_S_GRID:
            for ml in MOM_L_GRID:
                if ms >= ml: continue
                for thr in THR_GRID:
                    rs = []
                    for a in range(N_ANCHORS):
                        r = run_one(pdf, ranked, mom_dfs[ms], mom_dfs[ml], canary,
                                   sd, ed, anchor=a, drift_threshold=thr, mom_filter=True)
                        if r: rs.append(r)
                    if not rs: continue
                    ca = float(np.mean([r['CAGR'] for r in rs]))
                    md = float(np.mean([r['MDD'] for r in rs]))
                    cl = float(np.mean([r['Cal'] for r in rs]))
                    d = cl - base_cl
                    flag = "★" if d >= 0.10 else (" " if abs(d) < 0.03 else "")
                    print(f"  {flag}{ms:>4} {ml:>5} {thr:>5.2f} {ca:>+8.1%} {md:>+8.1%} {cl:>6.2f}  Δ {d:+5.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
