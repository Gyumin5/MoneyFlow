"""주식 sleeve 정확한 coin-aligned BT.

룰 (V24 spot/fut 와 정합):
1. 매일 cron:
   - canary (EEM SMA300 hyst 0.5%) 평가
   - canary OFF → defensive top3 (6m ret>0) EW
   - canary ON  → universe filter mom>0 → top min(3, n_healthy) by Z-score
     · 각 픽 cap=1/3 고정 (EW 가 아닌 fixed slot)
     · 남는 슬롯은 Cash
   - 매 cron drift 평가: half_turnover(cur_w, target_w)
   - drift >= threshold → 매도/매수 실행 (체결 가정), refill v2 효과 자동

2. 캐시 키 = 'Cash'. snap interval = 21 거래일 (월간).

mom_period grid: 42, 63, 84, 105, 126
drift_threshold grid: 0.03, 0.05, 0.07, 0.10
기간: 2017+/2018+/2020+/2021+
anchor: 11

각 케이스 단독 sleeve Cal/CAGR/MDD + rebal_count.
"""
import sys, time
from datetime import datetime
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
from stock_engine import load_prices, ALL_TICKERS


OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
DEF = ("IEF", "BIL", "BNDX", "GLD", "PDBC")
CASH_KEY = 'Cash'
SHARPE_LB = 252
CANARY_SMA = 200
CANARY_HYST = 0.005
TX = 0.001

PERIODS = [
    ("2017-01-01", "2025-12-31"),
    ("2018-01-01", "2025-12-31"),
    ("2020-01-01", "2025-12-31"),
    ("2021-01-01", "2025-12-31"),
]
MOM_GRID = [42, 63, 84, 105, 126]
THR_GRID = [0.03, 0.05, 0.07, 0.10]
N_ANCHORS = 11
N_SNAPS = 3
SNAP_DAYS_GAP = 23  # 거래일 stagger (V24 STAGGER)
SNAP_INTERVAL = 69  # 거래일 (V24 SNAP_PERIOD)
CAP_PER_PICK = 1.0 / 3


def half_turnover(cur, tgt):
    keys = set(cur) | set(tgt)
    return sum(abs(cur.get(k, 0) - tgt.get(k, 0)) for k in keys) / 2


def compute_zscore(prices_df, date, momcol='mom126'):
    """Z = z(wmom) + z(sh126)."""
    rows = []
    for t in OFF_R7:
        if t not in prices_df.columns: continue
        s = prices_df[t]
        if date not in s.index: continue
        loc = s.index.get_loc(date)
        if loc < SHARPE_LB: continue
        # wmom = 0.5*mom63 + 0.3*mom126 + 0.2*mom252
        p_now = s.iloc[loc]
        if pd.isna(p_now): continue
        p_63 = s.iloc[loc - 63] if loc >= 63 else np.nan
        p_126 = s.iloc[loc - 126] if loc >= 126 else np.nan
        p_252 = s.iloc[loc - 252] if loc >= 252 else np.nan
        if any(pd.isna([p_63, p_126, p_252])): continue
        wmom = 0.5*(p_now/p_63-1) + 0.3*(p_now/p_126-1) + 0.2*(p_now/p_252-1)
        # sharpe 126
        ret = s.iloc[loc-126:loc].pct_change().dropna()
        if len(ret) < 50: continue
        sh = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
        rows.append({'t': t, 'wmom': wmom, 'sh': sh})
    if not rows: return []
    df = pd.DataFrame(rows).set_index('t')
    m_std = df['wmom'].std()
    s_std = df['sh'].std()
    df['z_m'] = (df['wmom']-df['wmom'].mean())/m_std if m_std > 0 else 0
    df['z_s'] = (df['sh']-df['sh'].mean())/s_std if s_std > 0 else 0
    df['z'] = df['z_m'] + df['z_s']
    return df.sort_values('z', ascending=False).index.tolist()


def get_mom(prices_df, ticker, date, n):
    s = prices_df.get(ticker)
    if s is None or date not in s.index: return np.nan
    loc = s.index.get_loc(date)
    if loc < n: return np.nan
    p_now = s.iloc[loc]; p_n = s.iloc[loc - n]
    if pd.isna(p_now) or pd.isna(p_n) or p_n <= 0: return np.nan
    return p_now / p_n - 1


def canary_eval(eem, date, prev_on):
    if date not in eem.index: return prev_on if prev_on is not None else False
    loc = eem.index.get_loc(date)
    if loc < CANARY_SMA: return False
    sma = eem.iloc[loc - CANARY_SMA + 1: loc + 1].mean()
    cur = eem.iloc[loc]
    if pd.isna(sma) or pd.isna(cur): return prev_on if prev_on is not None else False
    dist = cur / sma - 1
    if dist > CANARY_HYST: return True
    if dist < -CANARY_HYST: return False
    return prev_on if prev_on is not None else (cur > sma)


def compute_target(prices_df, date, mom_period, prev_canary):
    """coin-aligned: filter universe by mom>0, top min(3,n_healthy) by Z, cap=1/3 each."""
    canary_on = canary_eval(prices_df['EEM'], date, prev_canary)
    if not canary_on:
        # defense top3 by 6m ret > 0
        scores = []
        for t in DEF:
            r = get_mom(prices_df, t, date, 126)
            if not np.isnan(r) and r > 0:
                scores.append((t, r))
        scores.sort(key=lambda x: -x[1])
        picks = [t for t, _ in scores[:3]]
        if not picks: return {CASH_KEY: 1.0}, canary_on
        n = len(picks)
        return {**{t: 1.0/n for t in picks}, CASH_KEY: 0.0}, canary_on
    # offense: filter mom>0
    ranked = compute_zscore(prices_df, date)
    healthy = [t for t in ranked if (get_mom(prices_df, t, date, mom_period) or 0) > 0]
    picks = healthy[:3]
    if not picks:
        return {CASH_KEY: 1.0}, canary_on
    tgt = {t: CAP_PER_PICK for t in picks}
    tgt[CASH_KEY] = 1.0 - CAP_PER_PICK * len(picks)
    return tgt, canary_on


def step_returns(prices_df, holdings, date):
    """Update holdings value by daily return."""
    new = {}
    for k, w in holdings.items():
        if k == CASH_KEY:
            new[k] = w
            continue
        s = prices_df.get(k)
        if s is None or date not in s.index:
            new[k] = w; continue
        loc = s.index.get_loc(date)
        if loc == 0:
            new[k] = w; continue
        p_prev = s.iloc[loc-1]; p_now = s.iloc[loc]
        if pd.isna(p_prev) or pd.isna(p_now) or p_prev <= 0:
            new[k] = w; continue
        new[k] = w * (p_now / p_prev)
    return new


def normalize(holdings):
    total = sum(holdings.values())
    if total <= 0: return {CASH_KEY: 1.0}
    return {k: v/total for k, v in holdings.items()}


def run_one(prices_df, start, end, mom_period, drift_threshold, anchor):
    eem = prices_df['EEM']
    dates = eem.index
    mask = (dates >= start) & (dates <= end)
    sim_dates = dates[mask]
    if len(sim_dates) < 50: return None
    # snap_date: 첫 진입은 anchor 일자 offset 후 다음 영업일
    holdings = {CASH_KEY: 1.0}
    target = {CASH_KEY: 1.0}
    prev_canary = None
    equity = []
    rebal_count = 0
    last_target_date = None
    for i, d in enumerate(sim_dates):
        # 자산 진행
        holdings = step_returns(prices_df, holdings, d)
        h_norm = normalize(holdings)
        equity.append(sum(holdings.values()))
        # target 재산정 — 매월 anchor 시점 (대략 21영업일마다)
        if i == 0 or (i - anchor) % 21 == 0:
            target, prev_canary = compute_target(prices_df, d, mom_period, prev_canary)
            last_target_date = d
        else:
            # canary 매일 체크 (flip 시 target 갱신)
            new_can = canary_eval(eem, d, prev_canary)
            if new_can != prev_canary:
                target, prev_canary = compute_target(prices_df, d, mom_period, prev_canary)
                last_target_date = d
        # drift check
        ht = half_turnover(h_norm, target)
        if ht >= drift_threshold:
            # 추가로 target 재산정 (refill 효과: mom 통과한 픽으로 새로 뽑힘)
            target, prev_canary = compute_target(prices_df, d, mom_period, prev_canary)
            # 매매 비용
            tx_cost = TX * ht
            # 적용
            pv = sum(holdings.values()) * (1 - tx_cost)
            holdings = {k: pv * w for k, w in target.items()}
            rebal_count += 1
    eq = pd.Series(equity, index=sim_dates)
    eq = eq.dropna()
    if len(eq) < 30: return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1 if yrs > 0 else 0
    peak = eq.cummax(); mdd = float((eq/peak - 1).min())
    cal = cagr / abs(mdd) if mdd < 0 else 0
    ret = eq.pct_change().dropna()
    sh = ret.mean() / ret.std() * np.sqrt(252) if ret.std() > 0 else 0
    return dict(CAGR=cagr, MDD=mdd, Cal=cal, Sharpe=sh, rebal=rebal_count)


def main():
    t0 = time.time()
    print("데이터 로딩...")
    prices_map = load_prices(ALL_TICKERS, start="2005-01-01")
    prices_df = pd.DataFrame(prices_map)
    # 중복 인덱스 제거 (Date 같은데 시각만 다른 행 — 첫 행만 유지)
    prices_df = prices_df[~prices_df.index.duplicated(keep='first')]
    prices_df = prices_df.sort_index()
    # 일별 종가만 유지 (00:00:00 시각만)
    prices_df = prices_df[prices_df.index.normalize() == prices_df.index]
    print(f"  완료 ({time.time()-t0:.1f}s). 종목 수 {len(prices_df.columns)} rows {len(prices_df)}")

    # baseline (no mom filter): just top3 Z-score, threshold 0.10
    print("\n[baseline] 비교 위해 mom_period=0 (무필터) 로 한 번 실행")

    for start, end in PERIODS:
        sd = pd.Timestamp(start); ed = pd.Timestamp(end)
        print(f"\n[{start} ~ {end}]  11-anchor 평균")
        print(f"  {'mom':>5} {'thr':>6} {'CAGR':>8} {'MDD':>8} {'Cal':>6} {'rebal':>6}")
        # baseline: mom=0 (모두 healthy 로 간주) + thr 0.10
        rs = []
        for a in range(N_ANCHORS):
            r = run_one(prices_df, sd, ed, mom_period=0, drift_threshold=0.10, anchor=a)
            if r: rs.append(r)
        if rs:
            ca = float(np.mean([r['CAGR'] for r in rs]))
            md = float(np.mean([r['MDD'] for r in rs]))
            cl = float(np.mean([r['Cal'] for r in rs]))
            rb = float(np.mean([r['rebal'] for r in rs]))
            print(f"  base    -  {ca:>+8.1%} {md:>+8.1%} {cl:>6.2f} {rb:>6.1f}")
            base_cl = cl
        else:
            base_cl = 0

        for mom in MOM_GRID:
            for thr in THR_GRID:
                rs = []
                for a in range(N_ANCHORS):
                    r = run_one(prices_df, sd, ed, mom_period=mom, drift_threshold=thr, anchor=a)
                    if r: rs.append(r)
                if not rs: continue
                ca = float(np.mean([r['CAGR'] for r in rs]))
                md = float(np.mean([r['MDD'] for r in rs]))
                cl = float(np.mean([r['Cal'] for r in rs]))
                rb = float(np.mean([r['rebal'] for r in rs]))
                d = cl - base_cl
                flag = "★" if d >= 0.10 else (" " if abs(d) < 0.03 else "")
                print(f"  {flag}{mom:>4} {thr:>6.2f} {ca:>+8.1%} {md:>+8.1%} {cl:>6.2f} {rb:>6.1f}  Δ {d:+5.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
