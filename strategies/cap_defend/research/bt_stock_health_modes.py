"""주식 sleeve health filter × refill vs filter-after 모드 비교 BT.

두 모드:
  - refill (코인 V24 처럼): 헬스체크 통과 종목만 candidates → top3 by Z-score → 항상 가능한 max-3 picks
                            (현재 stock_engine 의 health=momX 동작)
  - filter_after: 전체 universe 에서 top3 → unhealthy 픽은 drop 후 Cash 슬롯으로 대체
                  → 종종 1~2 picks + Cash 1/3 ~ 2/3 비중

mom 기간: 21, 42, 63, 84, 105, 126, 252
기간:     2017+, 2018+, 2020+, 2021+ ~ 2025-12-31
"""
import os, sys, time
from dataclasses import replace
import numpy as np

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
import stock_engine as se
from stock_engine import (SP, load_prices, precompute, _init, _run_one,
                          get_val, ALL_TICKERS, filter_healthy as orig_filter_healthy,
                          select_offensive as orig_select_offensive)


OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
DEF = ("IEF", "BIL", "BNDX", "GLD", "PDBC")
PERIODS = [
    ("2017-01-01", "2025-12-31"),
    ("2018-01-01", "2025-12-31"),
    ("2020-01-01", "2025-12-31"),
    ("2021-01-01", "2025-12-31"),
]
MOM_PERIODS = [21, 42, 63, 84, 105, 126, 252]


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


_CURRENT_MOM: list = [None]  # closure 변수 (filter_after 픽 mom 체크)
_CURRENT_MOM_SHORT: list = [None]  # combined 모드의 픽 mom 체크


def patched_select_filter_after(params, ind, date, candidates):
    """Mode: filter_after.
    1) select_offensive 로 top N picks 산출 (전체 universe candidates)
    2) mom{_CURRENT_MOM} > 0 통과한 픽만 살리고, 빠진 슬롯은 Cash
    3) 가중치 EW: 살아남은 픽들이 (살아남은/원래픽수) × 1.0 share, Cash 가 나머지
    """
    weights = orig_select_offensive(params, ind, date, candidates)
    if not weights or 'Cash' in weights:
        return weights
    picks = list(weights.keys())
    mom_p = _CURRENT_MOM[0]
    healthy = []
    for t in picks:
        m = get_val(ind, t, date, f'mom{mom_p}')
        if not np.isnan(m) and m > 0:
            healthy.append(t)
    n_total = len(picks)
    if not healthy:
        return {'Cash': 1.0}
    risky_share = len(healthy) / n_total
    cash_share = 1.0 - risky_share
    out = {t: risky_share / len(healthy) for t in healthy}
    if cash_share > 0:
        out['Cash'] = cash_share
    return out


def main():
    t0 = time.time()
    print("데이터 로딩...")
    prices = load_prices(ALL_TICKERS, start="2005-01-01")
    ind = precompute(prices)
    _init(prices, ind)
    se.check_crash = check_crash_vt
    print(f"  완료 ({time.time() - t0:.1f}s)")

    for start, end in PERIODS:
        print(f"\n[{start} ~ {end}]  (11-anchor 평균)")
        print(f"  {'mode':<14} {'mom':>5} {'Sharpe':>7} {'CAGR':>8} {'MDD':>8} {'Calmar':>7}  Δ Cal")
        print(f"  {'-'*65}")

        # baseline (none)
        sp = replace(BASE, start=start, end=end, health='none')
        rs = [_run_one(replace(sp, _anchor=a)) for a in range(1, 12)]
        rs = [r for r in rs if r]
        base_sh = float(np.mean([r["Sharpe"] for r in rs]))
        base_ca = float(np.mean([r["CAGR"] for r in rs]))
        base_md = float(np.mean([r["MDD"] for r in rs]))
        base_cl = float(np.mean([r.get("Calmar", 0) for r in rs]))
        print(f"  {'baseline':<14} {'-':>5} {base_sh:>7.3f} {base_ca:>+8.1%} {base_md:>+8.1%} {base_cl:>7.2f}  +0.00")

        for mp in MOM_PERIODS:
            # refill mode (engine native filter)
            sp_r = replace(BASE, start=start, end=end, health=f'mom{mp}')
            # restore originals
            se.filter_healthy = orig_filter_healthy
            se.select_offensive = orig_select_offensive
            rs_r = [_run_one(replace(sp_r, _anchor=a)) for a in range(1, 12)]
            rs_r = [r for r in rs_r if r]
            if rs_r:
                sh = float(np.mean([r["Sharpe"] for r in rs_r]))
                ca = float(np.mean([r["CAGR"] for r in rs_r]))
                md = float(np.mean([r["MDD"] for r in rs_r]))
                cl = float(np.mean([r.get("Calmar", 0) for r in rs_r]))
                d = cl - base_cl
                flag = "★" if d >= 0.20 else (" " if abs(d) < 0.05 else "")
                print(f"  {flag}{'refill':<13} {mp:>5} {sh:>7.3f} {ca:>+8.1%} {md:>+8.1%} {cl:>7.2f}  {d:>+5.2f}")

            # filter_after mode (monkey patch)
            sp_f = replace(BASE, start=start, end=end, health='none')
            _CURRENT_MOM[0] = mp
            se.select_offensive = patched_select_filter_after
            rs_f = [_run_one(replace(sp_f, _anchor=a)) for a in range(1, 12)]
            rs_f = [r for r in rs_f if r]
            se.select_offensive = orig_select_offensive  # restore
            if rs_f:
                sh = float(np.mean([r["Sharpe"] for r in rs_f]))
                ca = float(np.mean([r["CAGR"] for r in rs_f]))
                md = float(np.mean([r["MDD"] for r in rs_f]))
                cl = float(np.mean([r.get("Calmar", 0) for r in rs_f]))
                d = cl - base_cl
                flag = "★" if d >= 0.20 else (" " if abs(d) < 0.05 else "")
                print(f"  {flag}{'filter_after':<13} {mp:>5} {sh:>7.3f} {ca:>+8.1%} {md:>+8.1%} {cl:>7.2f}  {d:>+5.2f}")

        # combined: refill (mom_long) + filter_after (mom_short)
        print(f"  {'-'*65}")
        for mom_long in (126, 252):
            for mom_short in (21, 42, 63):
                if mom_short >= mom_long: continue
                sp_c = replace(BASE, start=start, end=end, health=f'mom{mom_long}')
                _CURRENT_MOM[0] = mom_short
                se.filter_healthy = orig_filter_healthy
                se.select_offensive = patched_select_filter_after
                rs_c = [_run_one(replace(sp_c, _anchor=a)) for a in range(1, 12)]
                rs_c = [r for r in rs_c if r]
                se.select_offensive = orig_select_offensive
                if rs_c:
                    sh = float(np.mean([r["Sharpe"] for r in rs_c]))
                    ca = float(np.mean([r["CAGR"] for r in rs_c]))
                    md = float(np.mean([r["MDD"] for r in rs_c]))
                    cl = float(np.mean([r.get("Calmar", 0) for r in rs_c]))
                    d = cl - base_cl
                    flag = "★" if d >= 0.20 else (" " if abs(d) < 0.05 else "")
                    label = f"comb{mom_long}/{mom_short}"
                    print(f"  {flag}{label:<13} {'-':>5} {sh:>7.3f} {ca:>+8.1%} {md:>+8.1%} {cl:>7.2f}  {d:>+5.2f}")

    print(f"\n총 소요: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
