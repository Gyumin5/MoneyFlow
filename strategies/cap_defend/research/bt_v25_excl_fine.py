"""V25 BNB/SOL 제외 — 3단계: fine snap grid + 합성 alloc 비교.

후보:
- spot: sn ∈ {287, 319, 351, 391, 427} × n ∈ {7, 11, 13}
- fut:  sn ∈ {115, 133, 161, 209, 247} × n ∈ {7, 11}
- prime stagger 만. baseline 도 포함.

각 winner 후보 + 합성 60/25/15 (stock 단독 + spot + fut) BT.
"""
from __future__ import annotations
import os, sys, time
from collections import defaultdict
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

START = "2020-10-01"
END = "2026-05-29"
EXCLUDE = ['BNB', 'SOL']

WIN_SIZES = [504, 756, 1008]
STRIDES = [63, 126, 252]


def is_prime(n):
    if n < 2: return False
    if n in (2, 3): return True
    if n % 2 == 0: return False
    for d in range(3, int(n**0.5) + 1, 2):
        if n % d == 0: return False
    return True


def run_spot(sn, n_snap):
    from unified_backtest import run as bt_run, load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    bars, _ = load_data('D')
    m = bt_run(bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=20, mom_long_bars=127,
        vol_threshold=0.05, vol_mode='daily',
        n_snapshots=n_snap, snap_interval_bars=sn,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol',
        exclude_assets=frozenset(EXCLUDE))
    return m.get('_equity') if m else None


def run_fut(sn, n_snap):
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    bars_full, funding = load_data('D')
    k2 = build_K2_signal(bars_full, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    bars = {c: df for c, df in bars_full.items() if c not in EXCLUDE}
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
        tx_cost=0.0006, maint_rate=0.004,
        sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
        canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=n_snap, snap_interval_bars=sn,
        start_date=START, end_date=END)
    return m.get('_equity') if m else None


def run_stock():
    from bt_stock_mom3 import run_multi_3mom
    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    ranked, mom_off, mom_def, canary = precompute(pdf, [30, 72, 230], [42, 63, 126])
    sd = pd.Timestamp(START); ed = pd.Timestamp(END)
    return run_multi_3mom(pdf, ranked, mom_off, mom_def, canary, sd, ed, anchor=0,
                         drift_thr=0.05, cash_buf=0.07, ms=30, mid=72, ml=230,
                         snap_int=69, n_snaps=3)


def metrics(eq):
    if eq is None or len(eq.dropna()) < 30: return None
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1]/eq.iloc[0]) ** (1/yrs) - 1
    peak = eq.cummax(); mdd = (eq/peak - 1).min()
    rets = eq.pct_change().dropna()
    sh = rets.mean()/rets.std()*np.sqrt(252) if rets.std() > 0 else 0
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return cagr*100, mdd*100, sh, cal


def window_rs(eq_dict):
    common = None
    for s in eq_dict.values():
        if s is None: continue
        if common is None: common = s.index
        else: common = common.intersection(s.index)
    if common is None: return None
    common = sorted(common)
    if len(common) < max(WIN_SIZES) + max(STRIDES): return None
    sums = defaultdict(float); wins = defaultdict(int); n = 0
    for size in WIN_SIZES:
        for stride in STRIDES:
            for i in range(0, len(common) - size, stride):
                d0 = common[i]; d1 = common[i + size - 1]
                cals = {}
                for k, s in eq_dict.items():
                    if s is None: cals[k] = np.nan; continue
                    seg = s.loc[d0:d1].dropna()
                    if len(seg) < 30: cals[k] = np.nan; continue
                    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                    if yrs <= 0: cals[k] = np.nan; continue
                    cagr = (seg.iloc[-1]/seg.iloc[0]) ** (1/yrs) - 1
                    peak = seg.cummax(); mdd = float((seg/peak - 1).min())
                    cals[k] = cagr/abs(mdd) if mdd < 0 else 0
                if any(np.isnan(v) for v in cals.values()): continue
                ranked = sorted(cals.items(), key=lambda x: -x[1])
                for r, (mk, _) in enumerate(ranked, 1): sums[mk] += r
                wins[ranked[0][0]] += 1; n += 1
    return sums, wins, n


def gen_pairs(snaps, ns):
    pairs = []
    for sn in snaps:
        for n in ns:
            if sn % n != 0: continue
            st = sn // n
            if not is_prime(st): continue
            pairs.append((sn, n, st))
    return pairs


def build_alloc(eq_st, eq_sp, eq_fu, w_st=0.60, w_sp=0.25, w_fu=0.15):
    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)
    s_st = eq_st.loc[common].pct_change().fillna(0)
    s_sp = eq_sp.loc[common].pct_change().fillna(0)
    s_fu = eq_fu.loc[common].pct_change().fillna(0)
    r = w_st * s_st + w_sp * s_sp + w_fu * s_fu
    return (1 + r).cumprod()


def main():
    t0 = time.time()

    # === fine snap grid ===
    spot_snaps = [217, 287, 319, 351, 391, 427]
    spot_ns = [7, 11, 13]
    fut_snaps = [95, 115, 133, 161, 209, 247]
    fut_ns = [5, 7, 11]

    spot_pairs = gen_pairs(spot_snaps, spot_ns)
    fut_pairs = gen_pairs(fut_snaps, fut_ns)
    print(f"spot pairs: {len(spot_pairs)}, fut pairs: {len(fut_pairs)}")

    # run all
    print("\n[spot fine grid]")
    eq_sp_all = {}
    for sn, n, st in spot_pairs:
        tag = f"sn{sn}_n{n}_st{st}"
        eq = run_spot(sn, n)
        if eq is None: continue
        eq_sp_all[tag] = eq
        m = metrics(eq)
        if m: print(f"  {tag:<20} CAGR {m[0]:5.1f}% MDD {m[1]:+6.1f}% Cal {m[3]:.2f}")
    rs = window_rs(eq_sp_all)
    if rs:
        sums, wins, n_w = rs
        print(f"\n  spot top (n_w={n_w}):")
        for tag, v in sorted(sums.items(), key=lambda x: x[1])[:8]:
            print(f"    {tag:<20} avg_rank={v/n_w:.2f} win={wins[tag]/n_w*100:.1f}%")

    print("\n[fut fine grid]")
    eq_fu_all = {}
    for sn, n, st in fut_pairs:
        tag = f"sn{sn}_n{n}_st{st}"
        eq = run_fut(sn, n)
        if eq is None: continue
        eq_fu_all[tag] = eq
        m = metrics(eq)
        if m: print(f"  {tag:<20} CAGR {m[0]:5.1f}% MDD {m[1]:+6.1f}% Cal {m[3]:.2f}")
    rs = window_rs(eq_fu_all)
    if rs:
        sums, wins, n_w = rs
        print(f"\n  fut top (n_w={n_w}):")
        for tag, v in sorted(sums.items(), key=lambda x: x[1])[:8]:
            print(f"    {tag:<20} avg_rank={v/n_w:.2f} win={wins[tag]/n_w*100:.1f}%")

    # === 합성 alloc 비교 ===
    print("\n[stock V25]"); eq_st = run_stock()
    print(f"  stock CAGR {metrics(eq_st)[0]:.1f}% Cal {metrics(eq_st)[3]:.2f}")

    # baseline + top spot/fut
    sp_base = eq_sp_all.get('sn217_n7_st31')
    sp_top = None; fu_base = eq_fu_all.get('sn95_n5_st19'); fu_top = None
    if rs:
        rs_sp = window_rs(eq_sp_all)
        if rs_sp:
            best_sp = sorted(rs_sp[0].items(), key=lambda x: x[1])[0][0]
            sp_top = eq_sp_all.get(best_sp)
            print(f"\n  spot top tag: {best_sp}")
        rs_fu = window_rs(eq_fu_all)
        if rs_fu:
            best_fu = sorted(rs_fu[0].items(), key=lambda x: x[1])[0][0]
            fu_top = eq_fu_all.get(best_fu)
            print(f"  fut top tag: {best_fu}")

    if sp_base is not None and fu_base is not None:
        alloc_base = build_alloc(eq_st, sp_base, fu_base)
        m = metrics(alloc_base)
        print(f"\n  alloc baseline (sn217+sn95): CAGR {m[0]:.1f}% MDD {m[1]:+.1f}% Sharpe {m[2]:.2f} Cal {m[3]:.2f}")
    if sp_top is not None and fu_top is not None:
        alloc_top = build_alloc(eq_st, sp_top, fu_top)
        m = metrics(alloc_top)
        print(f"  alloc top: CAGR {m[0]:.1f}% MDD {m[1]:+.1f}% Sharpe {m[2]:.2f} Cal {m[3]:.2f}")
    # 추가: base spot + top fut, top spot + base fut
    if sp_base is not None and fu_top is not None:
        m = metrics(build_alloc(eq_st, sp_base, fu_top))
        print(f"  alloc sp_base + fu_top: CAGR {m[0]:.1f}% MDD {m[1]:+.1f}% Cal {m[3]:.2f}")
    if sp_top is not None and fu_base is not None:
        m = metrics(build_alloc(eq_st, sp_top, fu_base))
        print(f"  alloc sp_top + fu_base: CAGR {m[0]:.1f}% MDD {m[1]:+.1f}% Cal {m[3]:.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
