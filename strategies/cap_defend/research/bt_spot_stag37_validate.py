"""SPOT V24 stag=31 (현행) vs stag=37 (candidate) — 4종 검증 게이트.

1. yearly 분해 — 우위가 단일 연도 과집중인지
2. anchor (start-date offset) 분해 — 특정 위상 의존인지
3. drawdown timing — MDD/회복기간/최악월/연속손실
4. fee/slippage 2x stress — Cal 우위 유지 여부
"""
from __future__ import annotations
import sys, os, time, importlib.util
import numpy as np
import pandas as pd

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
spec = importlib.util.spec_from_file_location("bt_cross", "/tmp/bt_fut_cross.py")
assert spec and spec.loader
bt_cross = importlib.util.module_from_spec(spec); spec.loader.exec_module(bt_cross)
os.environ['DRIFT_HEALTH_MODE'] = 'refill'

bars0, funding0 = bt_cross.load_data('D')

CUR  = dict(name="stag=31 (현행)", n_snap=7, snap_int=217, thr=0.10)
CAND = dict(name="stag=37 (후보)", n_snap=5, snap_int=185, thr=0.10)


def run_bt(cfg, tx=0.0006, start='2020-10-01', end='2026-05-13'):
    return bt_cross.run(bars0, funding0, interval='D', leverage=1.0,
        sma_days=42, mom_short_days=18, mom_long_days=127,
        n_snapshots=cfg['n_snap'], snap_interval_bars=cfg['snap_int'],
        drift_threshold=cfg['thr'],
        universe_size=3, selection='greedy', cap=1/3,
        tx_cost=tx, maint_rate=0.004, vol_days=90, vol_threshold=0.05,
        canary_hyst=0.015, health_mode='mom2vol',
        start_date=start, end_date=end)


def calc_metrics(eq):
    eq = eq.dropna()
    if len(eq) < 30: return None
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0: return None
    cagr = (eq.iloc[-1]/eq.iloc[0])**(1/yrs) - 1
    peak = eq.cummax(); dd = eq/peak - 1
    mdd = float(dd.min())
    cal = cagr / abs(mdd) if mdd < 0 else 0
    # recovery period
    in_dd = dd < -0.05
    max_recov = 0; cur = 0
    for v in in_dd.values:
        if v: cur += 1; max_recov = max(max_recov, cur)
        else: cur = 0
    # worst month return
    monthly = eq.resample('M').last().pct_change().dropna()
    worst_m = float(monthly.min()) if len(monthly) > 0 else 0
    # consec neg months
    neg = (monthly < 0).astype(int); ms = 0; mcur = 0
    for v in neg.values:
        if v: mcur += 1; ms = max(ms, mcur)
        else: mcur = 0
    # turnover proxy
    return dict(CAGR=cagr, MDD=mdd, Cal=cal, recov_days=max_recov,
                worst_m=worst_m, consec_neg_m=ms)


def yearly(eq):
    eq = eq.dropna()
    out = []
    for yr, grp in eq.groupby(eq.index.year):
        if len(grp) < 30: continue
        peak = grp.cummax(); mdd = float((grp/peak - 1).min())
        ret = grp.iloc[-1]/grp.iloc[0] - 1
        cal = ret / abs(mdd) if mdd < 0 else 0
        out.append((yr, ret, mdd, cal))
    return out


def main():
    t0 = time.time()
    print("=" * 100)
    print("SPOT V24 검증: stag=31 (n=7 int=217) vs stag=37 (n=5 int=185), thr=0.10")
    print("=" * 100)

    # GATE 1: 기본 메트릭
    print("\n[1] 기본 메트릭 (tx 0.06%)")
    print(f"  {'config':<25} {'CAGR':>7} {'MDD':>7} {'Cal':>5} {'rec_d':>5} {'worst_m':>8} {'neg_m':>5}")
    for cfg in (CUR, CAND):
        res = run_bt(cfg)
        m = calc_metrics(res['_equity'])
        print(f"  {cfg['name']:<25} {m['CAGR']*100:>6.1f}% {m['MDD']*100:>6.1f}% {m['Cal']:>5.2f} {m['recov_days']:>5d} {m['worst_m']*100:>7.1f}% {m['consec_neg_m']:>5d}")

    # GATE 2: yearly 분해
    print("\n[2] Yearly 분해 (Return / MDD / Cal)")
    eq_cur  = run_bt(CUR)['_equity']
    eq_cand = run_bt(CAND)['_equity']
    y_cur  = {y: (r, m, c) for y, r, m, c in yearly(eq_cur)}
    y_cand = {y: (r, m, c) for y, r, m, c in yearly(eq_cand)}
    print(f"  {'year':<6} {'CUR_ret':>8} {'CAND_ret':>8} {'CUR_mdd':>8} {'CAND_mdd':>8} {'CUR_Cal':>7} {'CAND_Cal':>8} {'우위':>6}")
    cand_wins = 0; cur_wins = 0
    for y in sorted(set(y_cur) | set(y_cand)):
        rc, mc, cc = y_cur.get(y, (np.nan,)*3)
        rd, md, cd = y_cand.get(y, (np.nan,)*3)
        winner = "CAND" if (cd > cc) else "CUR" if (cc > cd) else "tie"
        if winner == "CAND": cand_wins += 1
        elif winner == "CUR": cur_wins += 1
        print(f"  {y:<6} {rc*100:>7.1f}% {rd*100:>7.1f}% {mc*100:>7.1f}% {md*100:>7.1f}% {cc:>7.2f} {cd:>8.2f} {winner:>6}")
    print(f"  → 연도별 CAND 우위 {cand_wins}회 / CUR 우위 {cur_wins}회")

    # GATE 3: anchor 분해 (start offset)
    print("\n[3] Anchor (start-date offset) 분해")
    print(f"  {'offset_days':<12} {'CUR_Cal':>8} {'CAND_Cal':>9} {'CAND-CUR':>9}")
    diffs = []
    for offset in (0, 7, 14, 21, 28, 35, 42, 56, 70):
        start = (pd.Timestamp('2020-10-01') + pd.Timedelta(days=offset)).strftime('%Y-%m-%d')
        try:
            m_c = calc_metrics(run_bt(CUR, start=start)['_equity'])
            m_d = calc_metrics(run_bt(CAND, start=start)['_equity'])
            if m_c and m_d:
                d = m_d['Cal'] - m_c['Cal']
                diffs.append(d)
                print(f"  +{offset:<10d} {m_c['Cal']:>8.2f} {m_d['Cal']:>9.2f} {d:>+9.2f}")
        except Exception as e:
            print(f"  +{offset}: ERR {e}")
    print(f"  → CAND-CUR avg={np.mean(diffs):+.2f} sigma={np.std(diffs):.2f} pos={sum(1 for d in diffs if d>0)}/{len(diffs)}")

    # GATE 4: fee/slippage 2x stress
    print("\n[4] Fee/slippage 2x stress (tx 0.12%)")
    print(f"  {'config':<25} {'CAGR':>7} {'MDD':>7} {'Cal':>5}")
    for cfg in (CUR, CAND):
        m = calc_metrics(run_bt(cfg, tx=0.0012)['_equity'])
        print(f"  {cfg['name']:<25} {m['CAGR']*100:>6.1f}% {m['MDD']*100:>6.1f}% {m['Cal']:>5.2f}")

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
