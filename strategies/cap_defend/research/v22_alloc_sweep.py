#!/usr/bin/env python3
"""V22 자산배분 비교: 주식V17 / 현물V22(V21+C) / 선물V21.

Allocations (stock/spot/fut):
  - 60/40/0
  - 60/35/5
  - 60/30/10
  - 60/20/20

Rebalancing: sleeve relative drift 30% (자산 목표 대비 30% 초과 drift 시 전 자산 복원).
예: 주식 60% 목표, drift >= 18pp (60*30%) 이면 리밸.
"""
from __future__ import annotations
import os, sys, time
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))  # cap_defend
sys.path.insert(0, os.path.join(HERE, "c_tests_v3"))
sys.path.insert(0, os.path.join(HERE, "next_strategies"))

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from m3_engine_final import simulate, load_universe_hist, load_coin_daily, list_available_futures, load_v21 as load_v21_spot
from c_engine_v5 import run_c_v5, load_coin

SLEEVE_DRIFT = 0.30  # 30% 상대 drift
ALLOCATIONS = [
    (0.60, 0.40, 0.00),
    (0.60, 0.35, 0.05),
    (0.60, 0.30, 0.10),
    (0.60, 0.20, 0.20),
]
START = pd.Timestamp("2020-10-01")
END = pd.Timestamp("2026-03-28")


def build_v22_spot_equity() -> pd.Series:
    """V22 현물 (V21 + C champion) daily equity 생성."""
    from c_tests_v3._common3 import filter_bounce_confirm
    avail = sorted(list_available_futures())
    v21_fut_unused = None  # we don't need fut here
    v21_spot_eq = load_v21_spot()

    # C events 재추출 (champion params)
    print("  C events 재추출 (s_dthr12_tp3)...")
    def _one(c):
        df = load_coin(c + "USDT")
        if df is None: return []
        _, evs = run_c_v5(df, dip_bars=24, dip_thr=-0.12, tp=0.03, tstop=24, tx=0.003)
        for e in evs: e["coin"] = c
        return evs
    res = Parallel(n_jobs=24, prefer="threads")(delayed(_one)(c) for c in avail)
    ev = pd.DataFrame([e for b in res for e in b])
    ev = filter_bounce_confirm(ev, 1)
    print(f"  events={len(ev)}")

    hist = load_universe_hist()
    cd = load_coin_daily(avail)

    # V21 spot slice
    mask = (v21_spot_eq.index >= START) & (v21_spot_eq.index <= END)
    v21s = v21_spot_eq[mask].copy()
    v21s['equity'] = v21s['equity'].astype(float) / float(v21s['equity'].iloc[0])
    v21s['v21_ret'] = v21s['equity'].pct_change().fillna(0)
    v21s['prev_cash'] = v21s['cash_ratio'].shift(1).fillna(v21s['cash_ratio'].iloc[0])

    ev_s = ev[(pd.to_datetime(ev['entry_ts']) >= v21s.index[0]) &
               (pd.to_datetime(ev['entry_ts']) <= v21s.index[-1])].copy()
    eq_series, st = simulate(ev_s, cd, v21s.copy(), hist,
                              n_pick=1, cap_per_slot=0.333, universe_size=15,
                              tx_cost=0.003, swap_edge_threshold=1)
    print(f"  V22 spot: Cal={st.get('Cal'):.2f}, CAGR={st.get('CAGR')*100:.1f}%, MDD={st.get('MDD')*100:.1f}%")
    return eq_series


def build_v21_fut_equity() -> pd.Series:
    """V21 선물 3x equity (pre-computed CSV)."""
    path = os.path.join(HERE, "strat_C_v3", "v21_futures_daily.csv")
    df = pd.read_csv(path, parse_dates=['Date']).set_index('Date')
    return df['equity']


def build_stock_equity() -> pd.Series:
    """주식 V17 daily equity. run_bt 로 한 anchor (anchor=1) 만 사용 — 일반화."""
    sys.path.insert(0, "/home/gmoh/mon/251229/strategies/cap_defend")
    from dataclasses import replace
    from stock_engine import SP, load_prices, precompute, _init, run_bt, ALL_TICKERS
    import stock_engine as tsi
    import numpy as np
    OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    DEF = ("IEF", "BIL", "BNDX", "GLD", "PDBC")
    def check_crash_vt(params, ind, date):
        if params.crash == "vt":
            ret = ind.get("VT", pd.DataFrame()).get('ret', pd.Series()).get(date, np.nan)
            try:
                return not np.isnan(ret) and ret <= -params.crash_thresh
            except: return False
        return False

    V17 = SP(offensive=OFF_R7, defensive=DEF, canary_assets=("EEM",),
             canary_sma=200, canary_hyst=0.005, select="zscore3", weight="ew",
             defense="top3", def_mom_period=126, health="none", tx_cost=0.001,
             crash="vt", crash_thresh=0.03, crash_cool=3, sharpe_lookback=252,
             start="2017-01-01", end="2026-03-31", _anchor=6)
    print("  주식 V17 data loading...")
    prices = load_prices(ALL_TICKERS, start="2005-01-01")
    ind = precompute(prices)
    _init(prices, ind)
    tsi.check_crash = check_crash_vt
    df = run_bt(prices, ind, V17)
    if df is None:
        raise RuntimeError("V17 run_bt failed")
    return df['Value']


def align_series(series_map: dict) -> pd.DataFrame:
    """여러 equity series를 공통 trading days로 align.
    각 series를 daily로 reindex (ffill) 후 공통 날짜만."""
    frames = {}
    for name, s in series_map.items():
        s = s.copy()
        s.index = pd.to_datetime(s.index)
        s = s[~s.index.duplicated(keep='last')]
        s = s[(s.index >= START) & (s.index <= END)]
        s = s.resample('D').last().ffill()
        s = s / s.iloc[0]  # normalize
        frames[name] = s
    df = pd.DataFrame(frames).dropna(how='all').ffill().bfill()
    df = df[(df.index >= START) & (df.index <= END)]
    return df


def simulate_portfolio(eq_df: pd.DataFrame, w_stock: float, w_spot: float,
                       w_fut: float, sleeve_drift: float = SLEEVE_DRIFT) -> dict:
    """
    포트폴리오 시뮬 with sleeve drift rebalancing.
    eq_df columns: ['stock', 'spot', 'fut']  (normalized to 1.0 at start)
    반환: {equity, rebal_count}
    """
    tgt = np.array([w_stock, w_spot, w_fut])
    # 현재 비중 (w_k)
    cap = 1.0  # start capital
    cur_val = tgt * cap  # 각 sleeve 가치
    equity = []
    dates = eq_df.index
    prev = {c: eq_df[c].iloc[0] for c in ['stock', 'spot', 'fut']}
    rebals = 0
    for i, d in enumerate(dates):
        # 일일 수익 반영
        for j, c in enumerate(['stock', 'spot', 'fut']):
            p = eq_df[c].iloc[i]
            if tgt[j] > 0 and prev[c] > 0:
                ret = p / prev[c] - 1.0
                cur_val[j] *= (1 + ret)
            prev[c] = p
        total = cur_val.sum()
        equity.append(total)
        # rebalance check: sleeve relative drift
        if total > 0:
            cur_w = cur_val / total
            # drift: 각 sleeve |cur - tgt| / tgt (tgt=0 인 sleeve 는 절대 drift)
            drifted = False
            for j in range(3):
                if tgt[j] > 0:
                    if abs(cur_w[j] - tgt[j]) / tgt[j] >= sleeve_drift:
                        drifted = True; break
                else:
                    # tgt 0 인데 cur_w 가 작은 값이라도 threshold 초과면 리밸
                    if cur_w[j] >= 0.02:  # 2%p
                        drifted = True; break
            if drifted:
                cur_val = tgt * total
                rebals += 1
    eq = pd.Series(equity, index=dates)
    return {'equity': eq, 'rebal_count': rebals}


def compute_metrics(eq: pd.Series) -> dict:
    eq = eq.dropna()
    if len(eq) < 2: return {}
    total_days = (eq.index[-1] - eq.index[0]).days
    years = total_days / 365.25
    cagr = eq.iloc[-1] ** (1/years) - 1 if years > 0 else 0
    roll_max = eq.cummax()
    mdd = (eq / roll_max - 1).min()
    dr = eq.pct_change().dropna()
    ann_vol = dr.std() * np.sqrt(365)  # 365 because we resampled to calendar days
    sharpe = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    down = dr[dr < 0]
    sortino = dr.mean() / down.std() * np.sqrt(365) if len(down) > 1 and down.std() > 0 else sharpe
    calmar = cagr / abs(mdd) if mdd < 0 else 0
    # year-by-year
    yearly = eq.resample('Y').last() / eq.resample('Y').first() - 1
    worst_year = yearly.min()
    best_year = yearly.max()
    # max consecutive loss days
    neg = (dr < 0).astype(int)
    max_loss_streak = 0; cur = 0
    for v in neg.values:
        if v: cur += 1; max_loss_streak = max(max_loss_streak, cur)
        else: cur = 0
    return {
        'Final': round(eq.iloc[-1], 2),
        'CAGR': round(cagr * 100, 2),
        'MDD': round(mdd * 100, 2),
        'Cal': round(calmar, 2),
        'Sharpe': round(sharpe, 2),
        'Sortino': round(sortino, 2),
        'Vol': round(ann_vol * 100, 2),
        'BestYr': round(best_year * 100, 1),
        'WorstYr': round(worst_year * 100, 1),
        'MaxLossStreak': int(max_loss_streak),
    }


def main():
    print("=== V22 자산배분 비교 ===\n")
    t0 = time.time()
    print("1. V22 현물 equity 생성...")
    v22_spot = build_v22_spot_equity()
    print(f"   ({time.time()-t0:.0f}s)\n")

    t1 = time.time()
    print("2. V21 선물 equity 로드...")
    v21_fut = build_v21_fut_equity()
    print(f"   ({time.time()-t1:.0f}s)\n")

    t2 = time.time()
    print("3. 주식 V17 equity 생성...")
    stock = build_stock_equity()
    print(f"   ({time.time()-t2:.0f}s)\n")

    print("4. series align...")
    eq_df = align_series({'stock': stock, 'spot': v22_spot, 'fut': v21_fut})
    print(f"   align: {len(eq_df)} days, {eq_df.index[0].date()} ~ {eq_df.index[-1].date()}")
    print()

    print("5. 각 자산 단독 성과:")
    for c in ['stock', 'spot', 'fut']:
        m = compute_metrics(eq_df[c])
        print(f"   {c:5s}: Final ×{m['Final']:>6.2f} CAGR {m['CAGR']:>6.2f}% MDD {m['MDD']:>6.2f}% Cal {m['Cal']:>4.2f} Sh {m['Sharpe']:>4.2f} Sortino {m['Sortino']:>4.2f} Vol {m['Vol']:>4.1f}%")
    print()

    print("6. 포트폴리오 시뮬 (sleeve drift 30%):")
    rows = []
    for (ws, wsp, wf) in ALLOCATIONS:
        sim = simulate_portfolio(eq_df, ws, wsp, wf, SLEEVE_DRIFT)
        m = compute_metrics(sim['equity'])
        m['Alloc'] = f"{int(ws*100)}/{int(wsp*100)}/{int(wf*100)}"
        m['Rebals'] = sim['rebal_count']
        rows.append(m)
        print(f"   {m['Alloc']:>10s}  Final ×{m['Final']:>6.2f}  CAGR {m['CAGR']:>6.2f}%  MDD {m['MDD']:>7.2f}%  "
              f"Cal {m['Cal']:>5.2f}  Sh {m['Sharpe']:>4.2f}  Sortino {m['Sortino']:>4.2f}  "
              f"Vol {m['Vol']:>5.2f}%  Best/Worst {m['BestYr']:>5.1f}/{m['WorstYr']:>+5.1f}%  "
              f"MaxLoss {m['MaxLossStreak']}d  Rebals {m['Rebals']}")

    df = pd.DataFrame(rows)
    out = os.path.join(HERE, "v22_alloc_sweep.csv")
    df.to_csv(out, index=False)
    print(f"\n저장: {out}")


if __name__ == "__main__":
    main()
