#!/usr/bin/env python3
"""Phase 4 — 주식 BT 확장 (2026-05) + 주식≥현물≥선물 제약 grid.

변경:
- 주식 end: 2025-12-31 → 2026-05-13 (yfinance 2026-05 까지 가용 확인)
- 비율 제약: w_stock ≥ w_spot ≥ w_fut (5pp stride)
- 전체 grid + plateau + yearly ranksum + composite 재계산
"""
import os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP_DEFEND = os.path.abspath(os.path.join(HERE, '..', '..'))
PROJECT_ROOT = os.path.abspath(os.path.join(CAP_DEFEND, '..', '..'))
sys.path.insert(0, CAP_DEFEND)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'trade'))
sys.path.insert(0, PROJECT_ROOT)


def run_stock_extended():
    from dataclasses import replace
    from stock_engine import SP, load_prices, precompute, _init, ALL_TICKERS
    import stock_engine as tsi
    from stock_engine_snap import run_snapshot

    OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    DEF = ("IEF", "BIL", "BNDX", "GLD", "PDBC")

    def check_crash_vt(params, ind, date):
        from stock_engine import get_val
        if params.crash == "vt":
            ret = get_val(ind, "VT", date, "ret")
            return not np.isnan(ret) and ret <= -params.crash_thresh
        return False

    print("  주식 데이터 로딩 (end=2026-05-13)...")
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
        sharpe_lookback=252, start='2017-01-01', end='2026-05-13',
    )

    print("  주식 BT 실행 (snap=69 n=3, 2017~2026-05)...")
    t0 = time.time()
    df = run_snapshot(base, snap_days=69, n_snap=3)
    print(f"  완료 ({time.time() - t0:.1f}s, n={len(df)})")
    return df['Value'].copy()


def load_curves():
    # 주식: 새로 실행 (확장 기간)
    stock_path = os.path.join(HERE, 'stock_equity_ext.csv')
    if not os.path.exists(stock_path):
        s = run_stock_extended()
        s.to_csv(stock_path, header=['Value'])
    stock = pd.read_csv(stock_path, index_col=0, parse_dates=True)
    spot = pd.read_csv(os.path.join(HERE, 'spot_equity.csv'), index_col=0, parse_dates=True)
    fut = pd.read_csv(os.path.join(HERE, 'fut_equity.csv'), index_col=0, parse_dates=True)

    # normalize to date
    stock.index = pd.to_datetime(stock.index).normalize()
    spot.index = pd.to_datetime(spot.index, utc=True).tz_convert(None).normalize()
    fut.index = pd.to_datetime(fut.index).normalize()
    stock = stock.groupby(stock.index).last()
    spot = spot.groupby(spot.index).last()
    fut = fut.groupby(fut.index).last()

    idx = stock.index.intersection(spot.index).intersection(fut.index)
    print(f"  공통 dates: {idx[0]} ~ {idx[-1]} (n={len(idx)})")
    s_r = stock.loc[idx, 'Value'].pct_change().fillna(0)
    c_r = spot.loc[idx, 'Value'].pct_change().fillna(0)
    f_r = fut.loc[idx, 'Value'].pct_change().fillna(0)
    return pd.DataFrame({'stock': s_r, 'spot': c_r, 'fut': f_r})


def simulate(returns, w_s, w_c, w_f, trigger, threshold, tx_cost=0.0005):
    tgt = np.array([w_s, w_c, w_f])
    cur = tgt.copy()
    eq = []
    rebal_count = 0
    R = returns[['stock', 'spot', 'fut']].values
    for t in range(len(R)):
        r = R[t]
        cur = cur * (1.0 + r)
        pv = cur.sum()
        if pv <= 0:
            eq.append(pv)
            continue
        cur_w = cur / pv
        diff = cur_w - tgt
        if trigger == 'T1':
            m = np.sum(np.abs(diff)) / 2
        elif trigger == 'T2':
            m = np.max(np.abs(diff))
        else:
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.where(tgt > 0, np.abs(diff) / tgt, 0)
            m = np.max(rel)
        if m >= threshold:
            turnover = np.sum(np.abs(cur_w - tgt)) / 2
            pv_new = pv * (1 - tx_cost * 2 * turnover)
            cur = tgt * pv_new
            pv = pv_new
            rebal_count += 1
        eq.append(pv)
    return pd.Series(eq, index=returns.index), rebal_count


def metrics(eq):
    if eq.iloc[-1] <= 0:
        return dict(Cal=-999, CAGR=-1, MDD=-1, Sharpe=-99)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return dict(Cal=float(cal), CAGR=float(cagr), MDD=float(mdd), Sharpe=float(sh))


def yearly_cals(eq):
    out = {}
    for y, g in eq.groupby(eq.index.year):
        if len(g) < 20:
            continue
        ret = g.iloc[-1] / g.iloc[0] - 1
        mdd = (g / g.cummax() - 1).min()
        cal = ret / abs(mdd) if mdd != 0 else 0
        out[y] = float(cal)
    return out


def main():
    print("슬리브 equity 로딩 (확장 기간)...")
    returns = load_curves()

    # 비율 grid: 5pp stride, 합=100, 제약 ws >= wc >= wf
    ratios = []
    for ws in range(0, 101, 5):
        for wc in range(0, 101 - ws, 5):
            wf = 100 - ws - wc
            if wf < 0: continue
            if ws >= wc and wc >= wf:
                ratios.append((ws/100, wc/100, wf/100))
    print(f"  비율 조합 (제약 ws≥wc≥wf): {len(ratios)}")

    triggers = {
        'T1': [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
        'T2': [0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15],
        'T3': [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
    }

    total = sum(len(v) for v in triggers.values()) * len(ratios)
    print(f"  총 sim: {total}")

    rows = []
    yr_rows = []
    done = 0
    for trig, thrs in triggers.items():
        for thr in thrs:
            for (ws, wc, wf) in ratios:
                eq, rebal = simulate(returns, ws, wc, wf, trig, thr)
                m = metrics(eq)
                rows.append(dict(
                    trigger=trig, thr=thr,
                    w_stock=ws, w_spot=wc, w_fut=wf,
                    rebal=rebal,
                    **m,
                ))
                yc = yearly_cals(eq)
                for y, c in yc.items():
                    yr_rows.append(dict(
                        trigger=trig, thr=thr,
                        w_stock=ws, w_spot=wc, w_fut=wf,
                        year=y, y_Cal=c,
                    ))
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{total}")

    df = pd.DataFrame(rows)
    yr = pd.DataFrame(yr_rows)
    df.to_csv(os.path.join(HERE, 'ext_grid.csv'), index=False)
    yr.to_csv(os.path.join(HERE, 'ext_yearly.csv'), index=False)

    # composite + plateau
    key_cols = ['trigger', 'thr', 'w_stock', 'w_spot', 'w_fut']
    df['cfg'] = df[key_cols].astype(str).agg('|'.join, axis=1)
    yr['cfg'] = yr[key_cols].astype(str).agg('|'.join, axis=1)
    yr['rank'] = yr.groupby('year')['y_Cal'].rank(ascending=False, method='min')
    agg = yr.groupby('cfg').agg(
        ranksum=('rank', 'sum'),
        min_yr=('y_Cal', 'min'),
        n_years=('rank', 'count'),
    ).reset_index()

    # plateau: neighbor min Cal
    lookup = {}
    for _, r in df.iterrows():
        lookup[(r.trigger, round(r.thr, 4), round(r.w_stock, 4), round(r.w_spot, 4), round(r.w_fut, 4))] = r.Cal
    thr_steps = {t: sorted(df[df.trigger == t].thr.unique()) for t in triggers}

    def neigh_min(row):
        ws_pp = int(round(row.w_stock * 100))
        wc_pp = int(round(row.w_spot * 100))
        wf_pp = int(round(row.w_fut * 100))
        cals = []
        for dws in (-5, 0, 5):
            for dwc in (-5, 0, 5):
                if (dws, dwc) == (0, 0): continue
                nws = ws_pp + dws
                nwc = wc_pp + dwc
                nwf = 100 - nws - nwc
                # 제약 안 지키는 이웃은 평가에서 제외 (제약 grid 밖)
                if 0 <= nws and 0 <= nwc and 0 <= nwf and nws >= nwc and nwc >= nwf:
                    k = (row.trigger, round(row.thr, 4),
                         round(nws / 100, 4), round(nwc / 100, 4), round(nwf / 100, 4))
                    if k in lookup:
                        cals.append(lookup[k])
        thrs = thr_steps[row.trigger]
        if row.thr in thrs:
            i = thrs.index(row.thr)
            for j in (i - 1, i + 1):
                if 0 <= j < len(thrs):
                    k = (row.trigger, round(thrs[j], 4),
                         round(row.w_stock, 4), round(row.w_spot, 4), round(row.w_fut, 4))
                    if k in lookup:
                        cals.append(lookup[k])
        return min(cals) if cals else np.nan

    print("plateau 계산 중...")
    df['neigh_min'] = df.apply(neigh_min, axis=1)
    df['drop_pct'] = (df.Cal - df.neigh_min) / df.Cal * 100

    full = df.merge(agg, on='cfg')
    full['z_cal'] = (full.Cal - full.Cal.mean()) / full.Cal.std()
    full['z_sharpe'] = (full.Sharpe - full.Sharpe.mean()) / full.Sharpe.std()
    full['z_rs'] = -(full.ranksum - full.ranksum.mean()) / full.ranksum.std()
    full['z_minyr'] = (full.min_yr - full.min_yr.mean()) / full.min_yr.std()
    full['composite'] = full.z_cal + full.z_sharpe + full.z_rs + full.z_minyr

    # plateau-robust (drop ≤ 15%)
    robust = full[(full.drop_pct <= 15) & (full.Cal > 0)].copy()
    print(f"\n  plateau-robust: {len(robust)} / {len(full)}")
    robust = robust.sort_values('composite', ascending=False)

    cols = ['trigger', 'thr', 'w_stock', 'w_spot', 'w_fut', 'rebal',
            'composite', 'Cal', 'CAGR', 'MDD', 'Sharpe', 'ranksum', 'min_yr', 'drop_pct']
    print("\n=== TOP 20 (plateau-robust + composite, 제약 ws≥wc≥wf) ===")
    print(robust.head(20)[cols].to_string(index=False))
    robust.to_csv(os.path.join(HERE, 'ext_final_composite.csv'), index=False)

    # 현재 운영
    cur = full[(full.w_stock == 0.7) & (full.w_spot == 0.15) & (full.w_fut == 0.15) &
               (full.trigger == 'T1') & (full.thr == 0.15)]
    if len(cur) > 0:
        print("\n=== 현재 운영 70/15/15 T1 15pp (확장 기간) ===")
        print(cur[cols].to_string(index=False))


if __name__ == '__main__':
    main()
