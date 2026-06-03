"""V25 — winner cfg + baseline 각각 × 4가지 exclude 매트릭스.

winner: spot ms=20 ml=127 sn=217 n=7, fut ms=12 ml=118 sn=95 n=5
baseline: spot ms=20 ml=127 sn=217 n=7, fut ms=18 ml=127 sn=95 n=5

exclude 조합:
- none (full universe, BNB+SOL 포함)
- BNB only
- SOL only
- BNB+SOL
"""
from __future__ import annotations
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

START = "2020-10-01"; END = "2026-05-29"


def run_spot(ms, ml, sn, n, exclude):
    from unified_backtest import run as bt_run, load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    bars, _ = load_data('D')
    m = bt_run(bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
        start_date=START, end_date=END,
        sma_bars=42, mom_short_bars=ms, mom_long_bars=ml,
        vol_threshold=0.05, vol_mode='daily',
        n_snapshots=n, snap_interval_bars=sn,
        canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
        universe_size=3, cap=1/3, selection='greedy',
        stop_kind='none', stop_pct=0.0,
        dd_lookback=60, dd_threshold=-99.0,
        bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
        health_mode='mom2vol',
        exclude_assets=frozenset(exclude) if exclude else None)
    return m.get('_equity') if m else None


def run_fut(ms, ml, sn, n, exclude):
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    bars_full, funding = load_data('D')
    k2 = build_K2_signal(bars_full, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    bars = {c: df for c, df in bars_full.items() if c not in exclude} if exclude else bars_full
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
        tx_cost=0.0006, maint_rate=0.004,
        sma_days=42, mom_short_days=ms, mom_long_days=ml, vol_days=90,
        canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=n, snap_interval_bars=sn,
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


def build_alloc(eq_st, eq_sp, eq_fu, w_st=0.60, w_sp=0.25, w_fu=0.15):
    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)
    s_st = eq_st.loc[common].pct_change().fillna(0)
    s_sp = eq_sp.loc[common].pct_change().fillna(0)
    s_fu = eq_fu.loc[common].pct_change().fillna(0)
    r = w_st * s_st + w_sp * s_sp + w_fu * s_fu
    return (1 + r).cumprod()


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


def main():
    t0 = time.time()
    eq_st = run_stock()

    SPOT_BASE = (20, 127, 217, 7)
    FUT_BASE = (18, 127, 95, 5)
    SPOT_WINNER = (20, 127, 217, 7)  # spot 동일 — fine grid 결과 baseline 우위
    FUT_WINNER = (12, 118, 95, 5)

    exclusions = [
        ('none (full)', []),
        ('BNB only',    ['BNB']),
        ('SOL only',    ['SOL']),
        ('BNB+SOL',     ['BNB', 'SOL']),
    ]

    print(f"\n{'cfg':<22} {'exclude':<14} {'spot_Cal':>9} {'fut_Cal':>8} {'alloc_CAGR':>11} {'alloc_MDD':>10} {'alloc_Sh':>9} {'alloc_Cal':>10}")
    for sleeve_tag, sp_cfg, fu_cfg in [
        ('baseline (fut ms18)', SPOT_BASE, FUT_BASE),
        ('winner (fut ms12_ml118)', SPOT_WINNER, FUT_WINNER),
    ]:
        for excl_tag, excl in exclusions:
            eq_sp = run_spot(*sp_cfg, exclude=excl)
            eq_fu = run_fut(*fu_cfg, exclude=excl)
            if eq_sp is None or eq_fu is None: continue
            alloc = build_alloc(eq_st, eq_sp, eq_fu)
            m_sp = metrics(eq_sp); m_fu = metrics(eq_fu); m_al = metrics(alloc)
            print(f"{sleeve_tag:<22} {excl_tag:<14} {m_sp[3]:>8.2f} {m_fu[3]:>7.2f} "
                  f"{m_al[0]:>10.1f}% {m_al[1]:>+9.1f}% {m_al[2]:>8.2f} {m_al[3]:>9.2f}")
        print()

    print(f"총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
