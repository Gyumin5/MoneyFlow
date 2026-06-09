"""anchor-only(라이브 executor) vs drift-refill(채택 BT) 성과 차이 정량화.

라이브 V25 주식 파라미터: ms=30 mid=72 ml=230, drift_thr=0.05, cash_buf=0.07,
snap_int=69, n_snaps=3, cap weight.

두 모드:
- refill  : 드리프트 발화 시 그날 fresh 3-mom selection 으로 picks 재계산 (현 BT bt_stock_mom3)
- anchor  : 드리프트 발화해도 종목 교체 없이 기존 스냅샷 target 으로만 리밸런싱 (현 라이브 executor)

지표: CAGR, MDD, Calmar, 연 turnover(편도 합), 종목교체 횟수. 11 anchor 평균.
"""
import sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend/research')
from bt_stock_coin_v3 import precompute, half_t, CASH_KEY, TX
from stock_engine import load_prices, ALL_TICKERS
from bt_stock_single_snap import picks_to_target, select_off
from bt_stock_mom3 import fresh_pick_3mom


def run(pdf, ranked, mom_off, mom_def, canary, start, end, anchor,
        drift_thr, cash_buf, ms, mid, ml, snap_int, n_snaps, mode, tx=TX):
    sim_dates = pdf.index[(pdf.index >= start) & (pdf.index <= end)]
    if len(sim_dates) < 50:
        return None
    stagger = snap_int // n_snaps
    snaps = [{'phase': (anchor + k * stagger) % snap_int, 'picks': [], 'target': {CASH_KEY: 1.0}}
             for k in range(n_snaps)]
    holdings = {CASH_KEY: 1.0}
    prev_can = canary.iloc[0] if len(canary) > 0 else False
    turnover = 0.0           # 편도 turnover 누적 (ht 합)
    swaps = 0                # 종목집합 변경 횟수
    prev_picks = None

    def pick_for_snap(d, can_now):
        if can_now:
            return fresh_pick_3mom(ranked.at[d], mom_off[ms].loc[d],
                                   mom_off[mid].loc[d], mom_off[ml].loc[d])
        return []

    def merge_targets():
        agg = {}
        for s in snaps:
            for k, v in s['target'].items():
                agg[k] = agg.get(k, 0.0) + v / n_snaps
        return agg

    def cur_picks_set():
        s = set()
        for sn in snaps:
            s.update(p for p in sn['picks'])
        return frozenset(s)

    equity = []
    for i, d in enumerate(sim_dates):
        if i > 0:
            prev_d = sim_dates[i-1]
            for k in list(holdings.keys()):
                if k == CASH_KEY:
                    continue
                if k in pdf.columns:
                    p_prev = pdf.at[prev_d, k] if prev_d in pdf.index else np.nan
                    p_now = pdf.at[d, k] if d in pdf.index else np.nan
                    if pd.notna(p_prev) and pd.notna(p_now) and p_prev > 0:
                        holdings[k] = holdings[k] * (p_now / p_prev)
        can_now = bool(canary.at[d]) if d in canary.index else prev_can
        # 정규 앵커 갱신 (두 모드 공통)
        for s in snaps:
            if (i - s['phase']) >= 0 and (i - s['phase']) % snap_int == 0:
                if can_now:
                    s['picks'] = pick_for_snap(d, can_now)
                    s['target'] = picks_to_target(s['picks'], cash_buf, 'cap')
                else:
                    s['picks'] = []
                    s['target'] = select_off(d, mom_def, cash_buf, 'cap')
        # 카나리 flip (공통)
        if can_now != prev_can:
            for s in snaps:
                if can_now:
                    s['picks'] = pick_for_snap(d, can_now)
                    s['target'] = picks_to_target(s['picks'], cash_buf, 'cap')
                else:
                    s['picks'] = []
                    s['target'] = select_off(d, mom_def, cash_buf, 'cap')
            prev_can = can_now
        # daily-exit 모드: 매일 3-mom 헬스 재평가 → 탈락 픽 즉시 슬롯 교체 (앵커/드리프트 안 기다림)
        picks_changed_today = False
        if mode == 'daily' and can_now:
            ms_row = mom_off[ms].loc[d]; mid_row = mom_off[mid].loc[d]; ml_row = mom_off[ml].loc[d]

            def _ok_d(t):
                a, b, c = ms_row.get(t, np.nan), mid_row.get(t, np.nan), ml_row.get(t, np.nan)
                return all(pd.notna(x) and x > 0 for x in (a, b, c))
            for s in snaps:
                kept = [t for t in s['picks'] if _ok_d(t)]
                for t in ranked.at[d]:
                    if len(kept) >= 3:
                        break
                    if t == CASH_KEY or t in kept:
                        continue
                    if _ok_d(t):
                        kept.append(t)
                if frozenset(kept) != frozenset(s['picks']):
                    picks_changed_today = True
                    s['picks'] = kept
                    s['target'] = picks_to_target(kept, cash_buf, 'cap')
        target = merge_targets()
        total = sum(holdings.values())
        if total <= 0:
            holdings = {CASH_KEY: 1.0}; total = 1.0
        cur_w = {k: v/total for k, v in holdings.items()}
        ht = half_t(cur_w, target)
        if ht >= drift_thr or picks_changed_today:
            if mode == 'refill':
                # 전량 재선정 (현행 주식): 발화일 fresh top3 로 전 스냅 교체
                for s in snaps:
                    if can_now:
                        s['picks'] = pick_for_snap(d, can_now)
                        s['target'] = picks_to_target(s['picks'], cash_buf, 'cap')
                    else:
                        s['picks'] = []
                        s['target'] = select_off(d, mom_def, cash_buf, 'cap')
                target = merge_targets()
            elif mode == 'slot':
                # 슬롯교체 (코인식): 3-mom 통과 보유는 유지, 탈락 슬롯만 다음 랭킹으로
                ms_row = mom_off[ms].loc[d]; mid_row = mom_off[mid].loc[d]; ml_row = mom_off[ml].loc[d]

                def _ok(t):
                    a, b, c = ms_row.get(t, np.nan), mid_row.get(t, np.nan), ml_row.get(t, np.nan)
                    return all(pd.notna(x) and x > 0 for x in (a, b, c))
                for s in snaps:
                    if can_now:
                        kept = [t for t in s['picks'] if _ok(t)]
                        for t in ranked.at[d]:
                            if len(kept) >= 3:
                                break
                            if t == CASH_KEY or t in kept:
                                continue
                            if _ok(t):
                                kept.append(t)
                        s['picks'] = kept
                        s['target'] = picks_to_target(kept, cash_buf, 'cap')
                    else:
                        s['picks'] = []
                        s['target'] = select_off(d, mom_def, cash_buf, 'cap')
                target = merge_targets()
            # anchor 모드: picks 재계산 없이 기존 target 으로만 리밸런싱
            turnover += ht
            pv = total * (1 - tx * ht)
            holdings = {k: pv * w for k, w in target.items() if w > 0}
        # 종목집합 변경 추적
        ps = cur_picks_set()
        if prev_picks is not None and ps != prev_picks:
            swaps += 1
        prev_picks = ps
        equity.append(sum(holdings.values()))
    eq = pd.Series(equity, index=sim_dates).dropna()
    return eq, turnover, swaps


def metrics(eq):
    if eq is None or len(eq) < 50:
        return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1
    dd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(dd) if dd != 0 else float('nan')
    return cagr, dd, cal, yrs


def main():
    t0 = time.time()
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]

    MS, MID, ML = 30, 72, 230
    DRIFT, BUF = 0.05, 0.07
    SNAP_INT, N_SNAPS = 69, 3
    all_periods = sorted(set([MS, MID, ML, 30, 45, 84, 210]))
    ranked, mom_off, mom_def, canary = precompute(pdf, all_periods, [42, 63, 126])

    sd = pd.Timestamp("2017-01-01"); ed = pd.Timestamp("2026-05-13")

    print(f"# V25 라이브 파라미터: ms={MS} mid={MID} ml={ML} drift={DRIFT} buf={BUF} snap={SNAP_INT} n={N_SNAPS}")
    print(f"# 기간 {sd.date()}~{ed.date()}, 11 anchor 평균. 비용 스트레스 1x/3x/5x (base TX={TX})\n")
    print(f"  {'cost':>5} {'mode':<8} {'CAGR':>7} {'MDD':>7} {'Calmar':>7} {'turnover/yr':>12} {'swaps':>7}")
    MODES = ('anchor', 'refill', 'daily')
    for mult in (1, 3, 5):
        tx = TX * mult
        agg = {m: [] for m in MODES}
        tov = {m: [] for m in MODES}
        swp = {m: [] for m in MODES}
        for anchor in range(0, 11):
            for mode in MODES:
                r = run(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor,
                        DRIFT, BUF, MS, MID, ML, SNAP_INT, N_SNAPS, mode, tx=tx)
                if r is None:
                    continue
                eq, turnover, swaps = r
                m = metrics(eq)
                if m is None:
                    continue
                agg[mode].append(m); tov[mode].append(turnover); swp[mode].append(swaps)
        for mode in MODES:
            ms_ = agg[mode]
            if not ms_:
                print(f"  {mult}x   {mode:<8} (no data)"); continue
            cagr = np.mean([x[0] for x in ms_]); dd = np.mean([x[1] for x in ms_])
            cal = np.mean([x[2] for x in ms_]); yrs = ms_[0][3]
            tvy = np.mean(tov[mode]) / yrs; sw = np.mean(swp[mode])
            print(f"  {mult}x    {mode:<8} {cagr*100:>6.1f}% {dd*100:>6.1f}% {cal:>7.2f} {tvy:>11.2f}x {sw:>7.0f}")
        print()
    print(f"총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
