#!/usr/bin/env python3
"""Phase 1 — 3 슬리브 단독 BT 1회 실행. daily equity Series 저장.

각 자산을 V23 라이브 파라미터로 단독 BT → daily equity CSV 출력.
- 주식: stock_engine_snap.run_snapshot, V17 SP 정의 (SNAP=69, N=3, STAGGER=23)
- 코인 spot: run_current_coin_v20_backtest.run_backtest (MEMBERS = V23 D_SMA42 단독)
- 선물: backtest_futures_full.run (V23 sma=42 ms=18 ml=127 sn=95 n=5 drift=0.03 L3)

출력:
- stock_equity.csv (Date, Value)
- spot_equity.csv (Date, Value)
- fut_equity.csv (Date, Value)
"""
import os, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP_DEFEND = os.path.abspath(os.path.join(HERE, '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CAP_DEFEND, '..', '..'))
sys.path.insert(0, CAP_DEFEND)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'trade'))
sys.path.insert(0, PROJECT_ROOT)


def run_stock(start='2017-01-01', end='2025-12-31'):
    from dataclasses import replace
    from stock_engine import SP, load_prices, precompute, _init, ALL_TICKERS
    import stock_engine as tsi
    from stock_engine_snap import run_snapshot
    import numpy as np

    OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    DEF = ("IEF", "BIL", "BNDX", "GLD", "PDBC")

    def check_crash_vt(params, ind, date):
        from stock_engine import get_val
        if params.crash == "vt":
            ret = get_val(ind, "VT", date, "ret")
            return not np.isnan(ret) and ret <= -params.crash_thresh
        return False

    print("  주식 데이터 로딩...")
    t0 = time.time()
    prices = load_prices(ALL_TICKERS, start="2005-01-01")
    ind = precompute(prices)
    _init(prices, ind)
    tsi.check_crash = check_crash_vt
    print(f"  완료 ({time.time() - t0:.1f}s)")

    base = SP(
        offensive=OFF_R7, defensive=DEF,
        canary_assets=("EEM",), canary_sma=200, canary_hyst=0.005,
        select="zscore3", weight="ew", defense="top3",
        def_mom_period=126, health="none",
        tx_cost=0.001, crash="vt", crash_thresh=0.03, crash_cool=3,
        sharpe_lookback=252, start=start, end=end,
    )

    print("  주식 BT 실행 (snap=69, n=3)...")
    t0 = time.time()
    df = run_snapshot(base, snap_days=69, n_snap=3)
    print(f"  완료 ({time.time() - t0:.1f}s)")
    if df is None:
        raise RuntimeError("stock BT failed")
    return df['Value'].copy()


def run_spot(start='2020-10-01', end='2026-05-13'):
    from run_current_coin_v20_backtest import run_backtest
    print("  코인 spot BT 실행 (D_SMA42 sn=217 n=7)...")
    t0 = time.time()
    res = run_backtest(start=start, end=end)
    print(f"  완료 ({time.time() - t0:.1f}s)")
    return res['equity'].copy()


def run_fut(start='2020-10-01', end='2026-05-13'):
    from backtest_futures_full import load_data, run
    print("  선물 BT 실행 (V23, L3)...")
    t0 = time.time()
    bars, funding = load_data('D')
    m = run(
        bars, funding,
        interval='D', leverage=3.0,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03,
        universe_size=3, selection='greedy', cap=1/3,
        tx_cost=0.0006, maint_rate=0.004,
        vol_days=90, vol_threshold=0.05,
        canary_hyst=0.015, health_mode='mom2vol',
        start_date=start, end_date=end,
    )
    print(f"  완료 ({time.time() - t0:.1f}s)")
    return m['_equity'].copy()


def main():
    out = {}

    # 공통 기간: 코인 spot/fut 데이터 가용성에 맞춤. 주식은 더 길게 시작 가능하지만 통합용으로는 매칭 필요
    # 1차로 각자 native 범위 저장. 합집합 → 통합 BT에서 align
    print("=" * 60)
    print("[1/3] 선물")
    print("=" * 60)
    out['fut'] = run_fut()
    out['fut'].to_csv(os.path.join(HERE, 'fut_equity.csv'), header=['Value'])

    print("=" * 60)
    print("[2/3] 코인 spot")
    print("=" * 60)
    out['spot'] = run_spot()
    out['spot'].to_csv(os.path.join(HERE, 'spot_equity.csv'), header=['Value'])

    print("=" * 60)
    print("[3/3] 주식")
    print("=" * 60)
    out['stock'] = run_stock(start='2017-01-01', end='2025-12-31')
    out['stock'].to_csv(os.path.join(HERE, 'stock_equity.csv'), header=['Value'])

    print("\n=" * 60)
    print("완료. 각 슬리브 첫/끝 날짜:")
    for k, s in out.items():
        print(f"  {k}: {s.index[0]} ~ {s.index[-1]}  (n={len(s)})")


if __name__ == '__main__':
    main()
