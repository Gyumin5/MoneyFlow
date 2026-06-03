#!/usr/bin/env python3
"""Phase 2 — 자산배분 + 리밸 트리거 grid search.

입력: phase_1 의 3 슬리브 daily equity (stock/spot/fut)
출력:
- alloc_grid_results.csv (모든 조합 metrics)
- alloc_top_per_trigger.csv (T1/T2/T3 별 top 20)
- alloc_yearly_ranksum.csv (연도별 ranksum)
- alloc_plateau.csv (best 후보 ±1step plateau 검사)

방식:
- 슬리브 daily return 으로 portfolio 합성. 자산배분 비중 W_s/W_c/W_f (합=100, 5pp grid).
- 리밸 트리거 (T1/T2/T3) 발생 시 portfolio 재조정. 리밸비용 5bp.
- portfolio 일일 시계열 → Cal/CAGR/MDD/Sharpe + yearly metrics.
"""
import os, sys, itertools
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))


def load_curves():
    stock = pd.read_csv(os.path.join(HERE, 'stock_equity.csv'), index_col=0, parse_dates=True)
    spot = pd.read_csv(os.path.join(HERE, 'spot_equity.csv'), index_col=0, parse_dates=True)
    fut = pd.read_csv(os.path.join(HERE, 'fut_equity.csv'), index_col=0, parse_dates=True)
    # tz strip
    for s in (stock, spot, fut):
        if s.index.tz is not None:
            s.index = s.index.tz_localize(None)
    # daily resample / align — 공통 dates 만 사용 (intersection)
    idx = stock.index.intersection(spot.index).intersection(fut.index)
    if len(idx) == 0:
        # 코인 4h vs 주식 D mismatch — D 로 다운샘플
        stock.index = pd.to_datetime(stock.index).normalize()
        spot.index = pd.to_datetime(spot.index).normalize()
        fut.index = pd.to_datetime(fut.index).normalize()
        stock = stock.groupby(stock.index).last()
        spot = spot.groupby(spot.index).last()
        fut = fut.groupby(fut.index).last()
        idx = stock.index.intersection(spot.index).intersection(fut.index)
    s_r = stock.loc[idx, 'Value'].pct_change().fillna(0)
    c_r = spot.loc[idx, 'Value'].pct_change().fillna(0)
    f_r = fut.loc[idx, 'Value'].pct_change().fillna(0)
    return pd.DataFrame({'stock': s_r, 'spot': c_r, 'fut': f_r})


def simulate(returns: pd.DataFrame, w_stock, w_spot, w_fut,
             trigger: str, threshold: float, tx_cost: float = 0.0005):
    """Portfolio sim with allocation rebal trigger.

    trigger:
      'T1' — half_turnover sum|cur-tgt|/2 >= threshold (pp, e.g. 0.10 == 10pp)
      'T2' — max(|cur_i - tgt_i|) >= threshold (pp absolute)
      'T3' — max(|cur_i - tgt_i|/tgt_i) >= threshold (relative fraction)
    threshold 단위: T1/T2 는 비중 단위 (0.10 = 10pp), T3 도 0.10 = 10%
    """
    tgt = np.array([w_stock, w_spot, w_fut])
    cur = tgt.copy()
    pv = 1.0
    eq = []
    rebal_count = 0
    cols = ['stock', 'spot', 'fut']
    R = returns[cols].values
    n = len(R)
    for t in range(n):
        r = R[t]
        # 각 슬리브 비중에 일일 수익 적용
        cur = cur * (1.0 + r)
        pv = cur.sum()
        if pv <= 0:
            eq.append(pv)
            continue
        cur_w = cur / pv
        # 트리거 평가
        diff = cur_w - tgt
        if trigger == 'T1':
            metric = np.sum(np.abs(diff)) / 2
        elif trigger == 'T2':
            metric = np.max(np.abs(diff))
        elif trigger == 'T3':
            with np.errstate(divide='ignore', invalid='ignore'):
                rel = np.where(tgt > 0, np.abs(diff) / tgt, 0)
            metric = np.max(rel)
        else:
            metric = 0
        if metric >= threshold:
            # rebal: tgt 비중으로 복원, 거래비용 차감
            turnover = np.sum(np.abs(cur_w - tgt)) / 2
            pv_new = pv * (1 - tx_cost * 2 * turnover)
            cur = tgt * pv_new
            pv = pv_new
            rebal_count += 1
        eq.append(pv)
    return pd.Series(eq, index=returns.index), rebal_count


def metrics(eq: pd.Series):
    if eq.iloc[-1] <= 0:
        return dict(Cal=-999, CAGR=-1, MDD=-1, Sharpe=-99)
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return dict(Cal=float(cal), CAGR=float(cagr), MDD=float(mdd), Sharpe=float(sh))


def yearly_metrics(eq: pd.Series):
    out = {}
    for y, g in eq.groupby(eq.index.year):
        if len(g) < 20:
            continue
        ret = g.iloc[-1] / g.iloc[0] - 1
        dr = g.pct_change().dropna()
        sh = dr.mean() / dr.std() * np.sqrt(252) if dr.std() > 0 else 0
        mdd = (g / g.cummax() - 1).min()
        cal = ret / abs(mdd) if mdd != 0 else 0
        out[y] = dict(Cal=float(cal), Ret=float(ret), MDD=float(mdd), Sharpe=float(sh))
    return out


def main():
    print("슬리브 equity 로딩...")
    returns = load_curves()
    print(f"  공통 dates: {returns.index[0]} ~ {returns.index[-1]} (n={len(returns)})")

    # 비율 grid: 5pp stride, 합=100
    ratios = []
    for ws in range(0, 101, 5):
        for wc in range(0, 101 - ws, 5):
            wf = 100 - ws - wc
            if wf < 0 or wf > 100:
                continue
            ratios.append((ws/100, wc/100, wf/100))
    print(f"  비율 조합: {len(ratios)}")

    triggers = {
        'T1': [0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.18, 0.20, 0.25],
        'T2': [0.02, 0.03, 0.05, 0.07, 0.10, 0.12, 0.15],
        'T3': [0.05, 0.10, 0.15, 0.20, 0.30, 0.40, 0.50],
    }

    rows = []
    yearly_rows = []
    total = sum(len(v) for v in triggers.values()) * len(ratios)
    print(f"  총 sim: {total}")
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
                yr = yearly_metrics(eq)
                for y, ym in yr.items():
                    yearly_rows.append(dict(
                        trigger=trig, thr=thr,
                        w_stock=ws, w_spot=wc, w_fut=wf,
                        year=y, **{f'y_{k}': v for k, v in ym.items()},
                    ))
                done += 1
                if done % 500 == 0:
                    print(f"  {done}/{total}")

    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(HERE, 'alloc_grid_results.csv'), index=False)
    print(f"\n  저장: alloc_grid_results.csv ({len(df)})")

    # Top per trigger
    top_rows = []
    for trig in triggers:
        sub = df[df.trigger == trig].sort_values('Cal', ascending=False).head(20)
        top_rows.append(sub)
    pd.concat(top_rows).to_csv(os.path.join(HERE, 'alloc_top_per_trigger.csv'), index=False)

    # Yearly rank sum 검증
    ydf = pd.DataFrame(yearly_rows)
    ydf.to_csv(os.path.join(HERE, 'alloc_yearly.csv'), index=False)

    # ranksum across years per config
    cfg_keys = ['trigger', 'thr', 'w_stock', 'w_spot', 'w_fut']
    ydf['cfg'] = ydf[cfg_keys].astype(str).agg('|'.join, axis=1)
    # 연도별 Cal rank (높을수록 좋음)
    ydf['rank_in_year'] = ydf.groupby('year')['y_Cal'].rank(ascending=False, method='min')
    ranksum = ydf.groupby(['cfg'])['rank_in_year'].agg(['sum', 'mean', 'count']).reset_index()
    ranksum.columns = ['cfg', 'ranksum', 'avg_rank', 'n_years']
    ranksum = ranksum.sort_values('ranksum').head(50)
    # split cfg back
    splits = ranksum['cfg'].str.split('|', expand=True)
    splits.columns = cfg_keys
    out_rs = pd.concat([splits, ranksum.drop('cfg', axis=1).reset_index(drop=True)], axis=1)
    out_rs.to_csv(os.path.join(HERE, 'alloc_ranksum.csv'), index=False)

    print("\n=== Top 10 by Cal (전기간) ===")
    print(df.sort_values('Cal', ascending=False).head(10).to_string(index=False))
    print("\n=== Top 10 by yearly ranksum ===")
    print(out_rs.head(10).to_string(index=False))


if __name__ == '__main__':
    main()
