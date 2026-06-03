"""V25 현실적 baseline (BNB,SOL 제외) 자산배분 60/25/15 BT.

stock V25 그대로 + spot/fut 에서 BNB,SOL 제외.
일별 weight 가중합 → 합성 equity → 월별 + 전체 지표.
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

START = "2020-10-01"
END = "2026-05-29"
EXCLUDE = ['BNB', 'SOL']


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


def run_spot(exclude):
    from unified_backtest import run as bt_run, load_data
    bars, _ = load_data('D')
    m = bt_run(bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=20, mom_long_bars=127,
        vol_threshold=0.05, vol_mode='daily',
        n_snapshots=7, snap_interval_bars=217,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol',
        exclude_assets=frozenset(exclude) if exclude else None)
    return m.get('_equity') if m else None


def run_fut(exclude):
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    bars_full, funding = load_data('D')
    k2 = build_K2_signal(bars_full, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    bars = {c: df for c, df in bars_full.items() if c not in exclude} if exclude else bars_full
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
        tx_cost=0.0006, maint_rate=0.004,
        sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
        canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=5, snap_interval_bars=95,
        start_date=START, end_date=END)
    return m.get('_equity') if m else None


def build_alloc(eq_st, eq_sp, eq_fu, w_st=0.60, w_sp=0.25, w_fu=0.15):
    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)
    s_st = eq_st.loc[common].pct_change().fillna(0)
    s_sp = eq_sp.loc[common].pct_change().fillna(0)
    s_fu = eq_fu.loc[common].pct_change().fillna(0)
    r = w_st * s_st + w_sp * s_sp + w_fu * s_fu
    return (1 + r).cumprod()


def metrics(eq):
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1]/eq.iloc[0]) ** (1/yrs) - 1
    peak = eq.cummax(); mdd = (eq/peak - 1).min()
    rets = eq.pct_change().dropna()
    sh = rets.mean()/rets.std()*np.sqrt(252) if rets.std() > 0 else 0
    cal = cagr/abs(mdd) if mdd < 0 else 0
    return cagr*100, mdd*100, sh, cal


def main():
    t0 = time.time()
    print(f"[stock V25 BT]"); eq_st = run_stock()
    print(f"[spot V25 BT — exclude {EXCLUDE}]"); eq_sp_excl = run_spot(EXCLUDE)
    print(f"[fut V25 BT — exclude {EXCLUDE}]"); eq_fu_excl = run_fut(EXCLUDE)
    print(f"[baseline spot/fut full]")
    eq_sp_full = run_spot([])
    eq_fu_full = run_fut([])

    alloc_excl = build_alloc(eq_st, eq_sp_excl, eq_fu_excl)
    alloc_full = build_alloc(eq_st, eq_sp_full, eq_fu_full)

    print("\n========== 자산배분 60/25/15 비교 (2020-10 ~ 2026-05) ==========")
    print(f"  {'cfg':<40} {'CAGR':>8} {'MDD':>8} {'Sharpe':>7} {'Cal':>6}")
    for tag, eq in [
        ('주식 단독 V25',                eq_st),
        ('현물 단독 V25 (full)',         eq_sp_full),
        ('현물 단독 V25 (BNB,SOL 제외)', eq_sp_excl),
        ('선물 단독 V25 (full)',         eq_fu_full),
        ('선물 단독 V25 (BNB,SOL 제외)', eq_fu_excl),
        ('자산배분 60/25/15 (full)',     alloc_full),
        ('자산배분 60/25/15 (BNB,SOL 제외)', alloc_excl),
    ]:
        c, md, sh, cl = metrics(eq)
        print(f"  {tag:<40} {c:>7.1f}% {md:>+7.1f}% {sh:>7.2f} {cl:>6.2f}")

    # 월별 비교 표 출력 (alloc full vs excl)
    print("\n--- 월별 합성 60/25/15 수익률 비교 ---")
    mf = alloc_full.resample('M').last().pct_change() * 100
    me = alloc_excl.resample('M').last().pct_change() * 100
    print(f"  {'월':<10} {'full %':>8} {'-BNB,SOL %':>10} {'Δ':>8}")
    for d in sorted(set(mf.index) | set(me.index)):
        f = mf.get(d); e = me.get(d)
        diff = f - e if not pd.isna(f) and not pd.isna(e) else None
        f_s = f"{f:+.2f}" if not pd.isna(f) else '-'
        e_s = f"{e:+.2f}" if not pd.isna(e) else '-'
        d_s = f"{diff:+.2f}" if diff is not None else '-'
        print(f"  {d.strftime('%Y-%m'):<10} {f_s:>8} {e_s:>10} {d_s:>8}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
