#!/usr/bin/env python3
"""현재 공식 주식 V17 단독 백테스트.

backtest_official.py 우회 호출 없이 stock_engine만으로 직접 실행한다.
"""

import os
import sys
import time
import argparse
from dataclasses import replace

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from stock_engine import SP, load_prices, precompute, _init, _run_one, get_val, ALL_TICKERS
import stock_engine as tsi


OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
DEF = ("IEF", "BIL", "BNDX", "GLD", "PDBC")
PERIODS_STOCK = [
    ("2017-01-01", "2025-12-31"),
    ("2018-01-01", "2025-12-31"),
    ("2021-01-01", "2025-12-31"),
]


def check_crash_vt(params, ind, date):
    if params.crash == "vt":
        ret = get_val(ind, "VT", date, "ret")
        return not np.isnan(ret) and ret <= -params.crash_thresh
    return False


CURRENT_V17 = SP(
    offensive=OFF_R7,
    defensive=DEF,
    canary_assets=("EEM",),
    canary_sma=200,
    canary_hyst=0.005,
    select="zscore3",
    weight="ew",
    defense="top3",
    def_mom_period=126,
    health="none",
    tx_cost=0.001,
    crash="vt",
    crash_thresh=0.03,
    crash_cool=3,
    sharpe_lookback=252,
)


def main():
    parser = argparse.ArgumentParser(description="Current stock V17 backtest")
    parser.add_argument("--start", default=None)
    parser.add_argument("--end", default=None)
    args = parser.parse_args()

    t0 = time.time()
    print("데이터 로딩...")
    prices = load_prices(ALL_TICKERS, start="2005-01-01")
    ind = precompute(prices)
    _init(prices, ind)
    tsi.check_crash = check_crash_vt
    print(f"  완료 ({time.time() - t0:.1f}s)")

    print("\n" + "=" * 85)
    print("주식 전략 (V17, 11-anchor 평균)")
    print("=" * 85)

    periods = [(args.start, args.end)] if args.start and args.end else PERIODS_STOCK

    for start, end in periods:
        print(f"\n  [{start} ~ {end}]")
        print(f"  {'버전':<8s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Calmar':>7s}")
        print(f"  {'-'*40}")

        sp = replace(CURRENT_V17, start=start, end=end)
        rs = [_run_one(replace(sp, _anchor=a)) for a in range(1, 12)]
        rs = [r for r in rs if r]
        if rs:
            sharpe = np.mean([r["Sharpe"] for r in rs])
            cagr = np.mean([r["CAGR"] for r in rs])
            mdd = np.mean([r["MDD"] for r in rs])
            cal = np.mean([r.get("Calmar", 0) for r in rs])
            print(f"  {'V17':<8s} {sharpe:>7.3f} {cagr:>+8.1%} {mdd:>+8.1%} {cal:>7.2f}")

    print(f"\n총 소요: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
