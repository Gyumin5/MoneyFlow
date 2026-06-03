"""주식 sleeve health filter 비교 BT.

baseline: V17/V24 라이브 — health='none' (현재).
challengers: sma100/150/200, mom21/42/63/126, mom21_63, mom63_vol.

11-anchor 평균 × 3 기간 (2017/2018/2021 ~ 2025).
"""
import os, sys, time
from dataclasses import replace
import numpy as np

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
from stock_engine import SP, load_prices, precompute, _init, _run_one, get_val, ALL_TICKERS
import stock_engine as tsi


OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
DEF = ("IEF", "BIL", "BNDX", "GLD", "PDBC")
PERIODS = [
    ("2017-01-01", "2025-12-31"),
    ("2018-01-01", "2025-12-31"),
    ("2021-01-01", "2025-12-31"),
]


def check_crash_vt(params, ind, date):
    if params.crash == "vt":
        r = get_val(ind, "VT", date, "ret")
        return not np.isnan(r) and r <= -params.crash_thresh
    return False


BASE = SP(
    offensive=OFF_R7, defensive=DEF,
    canary_assets=("EEM",), canary_sma=200, canary_hyst=0.005,
    select="zscore3", weight="ew",
    defense="top3", def_mom_period=126,
    health="none",
    tx_cost=0.001, crash="vt", crash_thresh=0.03, crash_cool=3,
    sharpe_lookback=252,
)

HEALTH_OPTIONS = ['none', 'sma100', 'sma150', 'sma200',
                  'mom21', 'mom42', 'mom63', 'mom126',
                  'mom21_63', 'mom63_vol']


def main():
    t0 = time.time()
    print("데이터 로딩...")
    prices = load_prices(ALL_TICKERS, start="2005-01-01")
    ind = precompute(prices)
    _init(prices, ind)
    tsi.check_crash = check_crash_vt
    print(f"  완료 ({time.time() - t0:.1f}s)")

    for start, end in PERIODS:
        print(f"\n[{start} ~ {end}]  (11-anchor 평균)")
        print(f"  {'health':<12} {'Sharpe':>7} {'CAGR':>8} {'MDD':>8} {'Calmar':>7}  Δ Cal")
        print(f"  {'-'*55}")
        base_cal = None
        for h in HEALTH_OPTIONS:
            sp = replace(BASE, start=start, end=end, health=h)
            rs = [_run_one(replace(sp, _anchor=a)) for a in range(1, 12)]
            rs = [r for r in rs if r]
            if not rs:
                print(f"  {h:<12}  (no data)")
                continue
            sh = float(np.mean([r["Sharpe"] for r in rs]))
            ca = float(np.mean([r["CAGR"] for r in rs]))
            md = float(np.mean([r["MDD"] for r in rs]))
            cl = float(np.mean([r.get("Calmar", 0) for r in rs]))
            if h == 'none':
                base_cal = cl
                delta = 0.0
            else:
                delta = cl - (base_cal or 0)
            flag = "★" if delta >= 0.20 else (" " if abs(delta) < 0.05 else "")
            print(f"  {flag}{h:<11} {sh:>7.3f} {ca:>+8.1%} {md:>+8.1%} {cl:>7.2f}  {delta:>+5.2f}")

    print(f"\n총 소요: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
