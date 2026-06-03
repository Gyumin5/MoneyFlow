"""주식 health filter 확장 그리드 BT — mom 14개 × refill/filter_after × 4기간.

엔진 precompute 가 (21,42,63,126,252) 만 자동 → 누락 momN 컬럼 런타임 보강.
filter_healthy 도 momN 임의 N 지원으로 monkey patch.
"""
import sys, time
from dataclasses import replace
import numpy as np

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
import stock_engine as se
from stock_engine import (SP, load_prices, precompute, _init, _run_one,
                          get_val, ALL_TICKERS, select_offensive as orig_select_offensive)


OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
DEF = ("IEF", "BIL", "BNDX", "GLD", "PDBC")
PERIODS = [
    ("2017-01-01", "2025-12-31"),
    ("2018-01-01", "2025-12-31"),
    ("2020-01-01", "2025-12-31"),
    ("2021-01-01", "2025-12-31"),
]
MOM_PERIODS = [7, 14, 21, 30, 42, 63, 84, 105, 126, 147, 168, 189, 210, 252]


def check_crash_vt(params, ind, date):
    if params.crash == "vt":
        r = get_val(ind, "VT", date, "ret")
        return not np.isnan(r) and r <= -params.crash_thresh
    return False


def augment_mom(ind):
    """precompute 미생성 mom 컬럼 추가."""
    for t, df in ind.items():
        for n in MOM_PERIODS:
            col = f'mom{n}'
            if col not in df.columns:
                df[col] = df['price'] / df['price'].shift(n) - 1
    return ind


def patched_filter_healthy(params, ind, date, tickers):
    """momN (임의 N) 지원."""
    if params.health == 'none':
        return list(tickers)
    if params.health.startswith('mom') and params.health[3:].isdigit():
        n = int(params.health[3:])
        out = []
        for t in tickers:
            m = get_val(ind, t, date, f'mom{n}')
            if not np.isnan(m) and m > 0:
                out.append(t)
        return out
    # fallback to orig
    return orig_filter_healthy(params, ind, date, tickers)


_CURRENT_MOM: list = [None]


def patched_select_filter_after(params, ind, date, candidates):
    """Mode: filter_after — top3 후 mom < 0 픽 Cash 대체."""
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


BASE = SP(
    offensive=OFF_R7, defensive=DEF,
    canary_assets=("EEM",), canary_sma=200, canary_hyst=0.005,
    select="zscore3", weight="ew",
    defense="top3", def_mom_period=126,
    health="none",
    tx_cost=0.001, crash="vt", crash_thresh=0.03, crash_cool=3,
    sharpe_lookback=252,
)


orig_filter_healthy = se.filter_healthy


def main():
    t0 = time.time()
    print("데이터 로딩 + mom 컬럼 augment...")
    prices = load_prices(ALL_TICKERS, start="2005-01-01")
    ind = precompute(prices)
    augment_mom(ind)
    _init(prices, ind)
    se.check_crash = check_crash_vt
    print(f"  완료 ({time.time() - t0:.1f}s)")

    for start, end in PERIODS:
        print(f"\n[{start} ~ {end}]  (11-anchor 평균)")
        print(f"  {'mode':<14} {'mom':>5} {'Sharpe':>7} {'CAGR':>8} {'MDD':>8} {'Calmar':>7}  Δ Cal")
        print(f"  {'-'*65}")

        # baseline
        sp = replace(BASE, start=start, end=end, health='none')
        se.filter_healthy = orig_filter_healthy
        se.select_offensive = orig_select_offensive
        rs = [_run_one(replace(sp, _anchor=a)) for a in range(1, 12)]
        rs = [r for r in rs if r]
        bsh = float(np.mean([r["Sharpe"] for r in rs]))
        bca = float(np.mean([r["CAGR"] for r in rs]))
        bmd = float(np.mean([r["MDD"] for r in rs]))
        bcl = float(np.mean([r.get("Calmar", 0) for r in rs]))
        print(f"  {'baseline':<14} {'-':>5} {bsh:>7.3f} {bca:>+8.1%} {bmd:>+8.1%} {bcl:>7.2f}  +0.00")

        for mp in MOM_PERIODS:
            # refill
            se.filter_healthy = patched_filter_healthy
            se.select_offensive = orig_select_offensive
            sp_r = replace(BASE, start=start, end=end, health=f'mom{mp}')
            rs_r = [_run_one(replace(sp_r, _anchor=a)) for a in range(1, 12)]
            rs_r = [r for r in rs_r if r]
            if rs_r:
                sh = float(np.mean([r["Sharpe"] for r in rs_r]))
                ca = float(np.mean([r["CAGR"] for r in rs_r]))
                md = float(np.mean([r["MDD"] for r in rs_r]))
                cl = float(np.mean([r.get("Calmar", 0) for r in rs_r]))
                d = cl - bcl
                flag = "★" if d >= 0.10 else (" " if abs(d) < 0.03 else "")
                print(f"  {flag}{'refill':<13} {mp:>5} {sh:>7.3f} {ca:>+8.1%} {md:>+8.1%} {cl:>7.2f}  {d:>+5.2f}")

            # filter_after
            se.filter_healthy = orig_filter_healthy
            se.select_offensive = patched_select_filter_after
            sp_f = replace(BASE, start=start, end=end, health='none')
            _CURRENT_MOM[0] = mp
            rs_f = [_run_one(replace(sp_f, _anchor=a)) for a in range(1, 12)]
            rs_f = [r for r in rs_f if r]
            if rs_f:
                sh = float(np.mean([r["Sharpe"] for r in rs_f]))
                ca = float(np.mean([r["CAGR"] for r in rs_f]))
                md = float(np.mean([r["MDD"] for r in rs_f]))
                cl = float(np.mean([r.get("Calmar", 0) for r in rs_f]))
                d = cl - bcl
                flag = "★" if d >= 0.10 else (" " if abs(d) < 0.03 else "")
                print(f"  {flag}{'filter_after':<13} {mp:>5} {sh:>7.3f} {ca:>+8.1%} {md:>+8.1%} {cl:>7.2f}  {d:>+5.2f}")

    print(f"\n총 소요: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
