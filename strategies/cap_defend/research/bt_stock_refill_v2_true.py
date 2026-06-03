"""진짜 coin V24 refill v2 BT — partial swap, fixed 3-slot.

룰:
- snapshot 별로 picks 리스트 (최대 3) 유지
- drift fire 시 각 snap 검사:
  · picks 중 mom_short<0 OR mom_long<0 인 픽 = 실패
  · fresh_healthy = Z-score 랭킹 순으로 (mom_short>0 AND mom_long>0) AND (snap.picks 에 없음)
  · 실패 픽을 fresh_healthy 의 top 으로 교체 (랭킹 순)
  · 항상 3 slot 유지 (pool 부족 시 Cash slot)
- weight = cap=1/3 per pick + 부족분 Cash (통일성)
- canary OFF → orig6m defense top3 EW

snap 갱신 (anchor 일):
- 처음 진입: pool 에서 mom 통과한 top3 by Z-score → cap=1/3 each
- snap interval 마다 fresh select (전체 재계산)
- drift fire 사이 = partial refill v2 만
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import (precompute, combine, half_t,
                              N_ANCHORS, N_SNAPS, STAGGER, SNAP_INTERVAL, CAP_PER_PICK,
                              CASH_KEY, TX, PERIODS)
from stock_engine import load_prices, ALL_TICKERS

DEF_TICKERS = ("IEF", "BIL", "BNDX", "GLD", "PDBC")


def fresh_pick_top3(ranked_row, ms_row, ml_row):
    """anchor 일 fresh selection — mom 통과 종목 중 top3 by Z-score."""
    picks = []
    for t in ranked_row:
        if t == CASH_KEY: continue
        ms = ms_row.get(t, np.nan); ml = ml_row.get(t, np.nan)
        if pd.notna(ms) and pd.notna(ml) and ms > 0 and ml > 0:
            picks.append(t)
        if len(picks) >= 3: break
    return picks


def partial_refill(current_picks, ranked_row, ms_row, ml_row):
    """drift fire 시 partial swap — 실패 픽만 교체."""
    # 실패 마크
    failed = []
    kept = []
    for t in current_picks:
        ms = ms_row.get(t, np.nan); ml = ml_row.get(t, np.nan)
        if pd.notna(ms) and pd.notna(ml) and ms > 0 and ml > 0:
            kept.append(t)
        else:
            failed.append(t)
    if not failed:
        return current_picks  # 모두 healthy → no change
    # fresh_healthy = ranked 에서 (mom OK) AND (not in kept) AND (not in failed)
    excluded = set(kept) | set(failed)
    fresh = []
    for t in ranked_row:
        if t == CASH_KEY: continue
        if t in excluded: continue
        ms = ms_row.get(t, np.nan); ml = ml_row.get(t, np.nan)
        if pd.notna(ms) and pd.notna(ml) and ms > 0 and ml > 0:
            fresh.append(t)
        if len(fresh) >= len(failed): break
    return kept + fresh  # 실패 자리 채움. 부족하면 그만큼만


def picks_to_target(picks, cash_buf=0.0):
    """3 slot cap=1/3 + 부족분 Cash + cash_buffer 적용."""
    risky = 1.0 - cash_buf
    # 3 slot 기준 — 픽 < 3 이면 Cash slot 채움
    n_slot = 3
    per_pick = risky / n_slot
    tgt = {t: per_pick for t in picks}
    used = per_pick * len(picks)
    cash = risky - used + cash_buf
    if cash > 0:
        tgt[CASH_KEY] = cash
    return tgt


def select_off_target(d, mom_def, cash_buf):
    """defense top3 by 6m ret > 0 + Cash slot."""
    scores = []
    for t in DEF_TICKERS:
        r = mom_def[126].at[d, t] if t in mom_def[126].columns else np.nan
        if pd.notna(r) and r > 0: scores.append((t, r))
    scores.sort(key=lambda x: -x[1])
    picks = [t for t, _ in scores[:3]]
    return picks_to_target(picks, cash_buf)


def run_one(prices_df, ranked, mom_off, mom_def, canary, start, end, anchor,
            ms, ml, drift_thr, cash_buf):
    sim_dates = prices_df.index[(prices_df.index >= start) & (prices_df.index <= end)]
    if len(sim_dates) < 50: return None
    pall = prices_df
    holdings = {CASH_KEY: 1.0}
    # snap 상태: picks 리스트 (offense) 또는 None (defense/cash)
    snap_picks = [None] * N_SNAPS
    snap_targets = [{CASH_KEY: 1.0}] * N_SNAPS
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
        # snap 갱신 — anchor 일 fresh select
        for sidx in range(N_SNAPS):
            offset = anchor + sidx * STAGGER
            if i >= offset and (i - offset) % SNAP_INTERVAL == 0:
                if can_now:
                    picks = fresh_pick_top3(ranked.at[d], mom_off[ms].loc[d], mom_off[ml].loc[d])
                    snap_picks[sidx] = picks
                    snap_targets[sidx] = picks_to_target(picks, cash_buf)
                else:
                    snap_picks[sidx] = None
                    snap_targets[sidx] = select_off_target(d, mom_def, cash_buf)
        # canary flip
        if can_now != prev_can:
            for sidx in range(N_SNAPS):
                if can_now:
                    picks = fresh_pick_top3(ranked.at[d], mom_off[ms].loc[d], mom_off[ml].loc[d])
                    snap_picks[sidx] = picks
                    snap_targets[sidx] = picks_to_target(picks, cash_buf)
                else:
                    snap_picks[sidx] = None
                    snap_targets[sidx] = select_off_target(d, mom_def, cash_buf)
            prev_can = can_now
        combined = combine(snap_targets)
        total = sum(holdings.values())
        if total <= 0:
            holdings = {CASH_KEY: 1.0}; total = 1.0
        cur_w = {k: v/total for k, v in holdings.items()}
        ht = half_t(cur_w, combined)
        if ht >= drift_thr:
            # partial refill v2 — 각 snap 의 실패 픽만 교체
            for sidx in range(N_SNAPS):
                if can_now and snap_picks[sidx] is not None:
                    new_picks = partial_refill(snap_picks[sidx],
                                                ranked.at[d], mom_off[ms].loc[d], mom_off[ml].loc[d])
                    snap_picks[sidx] = new_picks
                    snap_targets[sidx] = picks_to_target(new_picks, cash_buf)
                elif not can_now:
                    snap_picks[sidx] = None
                    snap_targets[sidx] = select_off_target(d, mom_def, cash_buf)
            combined = combine(snap_targets)
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

    print("TRUE coin V24 refill v2 — partial swap, cap=1/3+Cash slot, 7% buffer")
    print(f"  {'thr':>5} {'2017+':>6} {'2018+':>6} {'2020+':>6} {'2021+':>6} {'avg':>6} {'CAGR':>7} {'MDD':>7}")
    print("-" * 75)
    for thr in (0.02, 0.03, 0.05, 0.07, 0.10):
        for buf in (0.0, 0.07):
            cls = []; cas = []; mds = []
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
            if len(cls) == 4:
                avg = np.mean(cls)
                bufpct = f"{int(buf*100)}%"
                print(f"  {thr:>5.2f} buf={bufpct:<3} {cls[0]:>6.2f} {cls[1]:>6.2f} {cls[2]:>6.2f} {cls[3]:>6.2f} {avg:>6.2f} {np.mean(cas)*100:>6.1f}% {np.mean(mds)*100:>7.1f}%")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
