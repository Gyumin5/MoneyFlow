"""
ETF Dynamic Selection × Rebalancing Timing Full Cross Test
===========================================================
Tests top dynamic selection methods across all rebalancing timings.
If a method beats baseline at some timing, it's worth considering.

Usage:
    python3 strategies/cap_defend/backtest_etf_dynamic_timing.py
"""

import os, sys, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = './data'

ALL_ETF_TICKERS = [
    'SPY','QQQ','EFA','EEM','VT','VEA','VNQ','VGK',
    'QUAL','MTUM','IQLT','IMTM','IWD','SCZ',
    'IEF','TLT','BND','AGG','BNDX','BIL','SHY','LQD','TIP',
    'GLD','DBC','PDBC','DBMF','KMLM',
    'VWO','HYG','RWX',
]

START_DATES = ['2019-01-01','2020-01-01','2021-01-01','2022-01-01','2023-01-01']
END_DATE = '2025-12-31'

OFFENSIVE = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','QUAL','MTUM','IQLT','IMTM']
DEFENSIVE = ['IEF','BIL','BNDX','GLD','PDBC']
CANARY = ['VT', 'EEM']


# ===========================================================================
# Data & Indicators
# ===========================================================================

def load_data():
    data_dict = {}
    for ticker in ALL_ETF_TICKERS:
        fp = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(fp): continue
        try:
            df = pd.read_csv(fp, parse_dates=['Date'])
            df = df.drop_duplicates(subset=['Date'], keep='first').set_index('Date')
            col = 'Adj Close' if 'Adj Close' in df else ('Adj_Close' if 'Adj_Close' in df else 'Close')
            if col in df: data_dict[ticker] = df[col]
        except: pass
    idx = pd.date_range(start='2017-01-01', end=END_DATE, freq='D')
    data = pd.DataFrame(data_dict).reindex(idx).ffill()
    print(f"Loaded {len(data_dict)} ETFs, {data.index[0].date()} ~ {data.index[-1].date()}")
    return data


def precompute(data):
    ind = {}
    for col in data.columns:
        p = data[col]
        dr = p.pct_change()
        d = pd.DataFrame({'price': p})
        d['sma200'] = p.rolling(200).mean()
        d['mom63']  = p / p.shift(63) - 1
        d['mom126'] = p / p.shift(126) - 1
        d['mom252'] = p / p.shift(252) - 1
        d['mom_w']  = 0.5*d['mom63'] + 0.3*d['mom126'] + 0.2*d['mom252']
        d['vol90']  = dr.rolling(90).std()
        rm = dr.rolling(126).mean()
        rs = dr.rolling(126).std()
        d['sharpe126'] = (rm / rs.replace(0, np.nan)) * np.sqrt(252)
        ind[col] = d
    return ind


def v(ind, ticker, col, date):
    if ticker not in ind: return np.nan
    try: return ind[ticker][col].loc[date]
    except: return np.nan


def has(ind, ticker, date):
    return pd.notna(v(ind, ticker, 'mom252', date))


# ===========================================================================
# Rebalancing date generators
# ===========================================================================

def gen_month_end(start, end, di):
    return set(pd.date_range(start=start, end=end, freq='M'))

def gen_month_start(start, end, di):
    result = set()
    for d in pd.date_range(start=start, end=end, freq='MS'):
        mask = di >= d
        if mask.any(): result.add(di[mask][0])
    return result

def gen_fixed_day(start, end, di, day=15):
    result = set()
    cur = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while cur <= end_ts:
        try: target = cur.replace(day=min(day, 28))
        except: target = cur.replace(day=28)
        mask = di >= target
        if mask.any():
            nearest = di[mask][0]
            if (nearest - target).days < 10: result.add(nearest)
        cur = cur + pd.DateOffset(months=1)
    return result


# ===========================================================================
# Dynamic Selection Methods
# ===========================================================================

def sel_fixed_top3(scores, **kw):
    return scores.head(3).index.tolist()

def sel_relative_threshold(scores, alpha=0.8, min_n=2, max_n=6, **kw):
    if scores.empty: return []
    pos = scores[scores > 0]
    if pos.empty: return []
    cutoff = pos.iloc[0] * alpha
    picked = pos[pos >= cutoff].iloc[:max_n]
    if len(picked) < min_n:
        picked = scores.head(min_n)
        picked = picked[picked > 0]
    return picked.index.tolist()

def sel_zscore(scores, z_cutoff=0.5, min_n=1, max_n=5, **kw):
    if scores.empty: return []
    mu, sigma = scores.mean(), scores.std()
    if sigma == 0 or pd.isna(sigma):
        return scores.head(min_n).index.tolist()
    z = (scores - mu) / sigma
    picked = scores[z >= z_cutoff]
    if len(picked) > max_n: picked = picked.iloc[:max_n]
    if len(picked) < min_n: picked = scores.head(min_n)
    return picked.index.tolist()

def sel_largest_gap(scores, min_n=2, max_n=6, **kw):
    if len(scores) < 2: return scores.index.tolist()
    vals = scores.values
    best_cut, best_gap = min_n, -1
    for i in range(min_n - 1, min(max_n, len(vals) - 1)):
        gap = vals[i] - vals[i + 1]
        if gap > best_gap: best_gap, best_cut = gap, i + 1
    return scores.head(best_cut).index.tolist()

def sel_kmeans(scores, k=3, min_n=1, max_n=6, **kw):
    if len(scores) < k: return scores.index.tolist()
    vals = scores.values.reshape(-1, 1)
    centroids = np.quantile(vals.flatten(), np.linspace(0, 1, k))
    for _ in range(20):
        labels = np.argmin(np.abs(vals - centroids), axis=1)
        new_c = np.array([vals[labels == i].mean() if (labels == i).any() else centroids[i] for i in range(k)])
        if np.allclose(centroids, new_c): break
        centroids = new_c
    labels = np.argmin(np.abs(vals - centroids), axis=1)
    picked = scores[labels == labels[0]]
    if len(picked) > max_n: picked = picked.iloc[:max_n]
    if len(picked) < min_n: picked = scores.head(min_n)
    return picked.index.tolist()

def sel_topn_tolerance(scores, n=3, epsilon=0.03, max_n=6, **kw):
    if len(scores) < n: return scores.index.tolist()
    cutoff = scores.iloc[n - 1] - epsilon
    picked = scores[scores >= cutoff]
    if len(picked) > max_n: picked = picked.iloc[:max_n]
    return picked.index.tolist()

def sel_softmax(scores, beta=5.0, min_weight=0.05, min_n=2, max_n=6, **kw):
    if scores.empty: return []
    s = scores.values
    exp_s = np.exp(beta * (s - s.max()))
    probs = exp_s / exp_s.sum()
    idx = np.where(probs >= min_weight)[0]
    if len(idx) < min_n: idx = np.arange(min(min_n, len(scores)))
    if len(idx) > max_n: idx = idx[:max_n]
    return scores.index[idx].tolist()


# ===========================================================================
# Strategy builder
# ===========================================================================

def make_strategy(sel_method, sel_params, combine_mode='union'):
    def strategy(d, ind):
        risk_on = all(has(ind,c,d) and v(ind,c,'price',d) > v(ind,c,'sma200',d) for c in CANARY)
        if not risk_on:
            best_t, best_r = 'BIL', -999
            for t in DEFENSIVE:
                r = v(ind, t, 'mom126', d)
                if pd.notna(r) and r > best_r: best_r, best_t = r, t
            return ({'BIL': 1.0}, 'Cash') if best_r < 0 else ({best_t: 1.0}, 'Off')

        rows = [(t, v(ind,t,'mom_w',d), v(ind,t,'sharpe126',d))
                for t in OFFENSIVE if has(ind,t,d)]
        rows = [(t,m,q) for t,m,q in rows if pd.notna(m) and pd.notna(q)]
        if not rows: return {'BIL': 1.0}, 'NoData'
        df = pd.DataFrame(rows, columns=['T','M','Q']).set_index('T')

        if combine_mode == 'union':
            mp = sel_method(df['M'].sort_values(ascending=False), **sel_params)
            qp = sel_method(df['Q'].sort_values(ascending=False), **sel_params)
            picks = list(dict.fromkeys(mp + qp))
        elif combine_mode == 'combined':
            ms, qs = df['M'].std(), df['Q'].std()
            df['C'] = ((df['M']-df['M'].mean())/(ms if ms>0 else 1)*0.5 +
                        (df['Q']-df['Q'].mean())/(qs if qs>0 else 1)*0.5)
            picks = sel_method(df['C'].sort_values(ascending=False), **sel_params)
        else:
            picks = df.nlargest(3,'M').index.tolist()

        if not picks: return {'BIL': 1.0}, 'NoPick'
        return {t: 1.0/len(picks) for t in picks}, f'On({len(picks)})'
    return strategy


# ===========================================================================
# Backtest engine
# ===========================================================================

def run_bt(data, ind, strat_func, start, end, rebal_dates, capital=10000, tx=0.001):
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.isin(data.index)]
    cash, hold, hist, rebals, prev_st = capital, {}, [], 0, None

    for today in dates:
        row = data.loc[today]
        pv = cash + sum(u*(row.get(t,0) if pd.notna(row.get(t,0)) else 0) for t,u in hold.items())
        hist.append({'Date': today, 'Value': pv})
        tgt, st = strat_func(today, ind)
        flip = prev_st is not None and st != prev_st
        prev_st = st
        if today in rebal_dates or flip:
            rebals += 1
            amt = pv * (1 - tx)
            cash, hold = amt, {}
            for t, w in tgt.items():
                p = row.get(t,0) if pd.notna(row.get(t,0)) else 0
                if p > 0:
                    a = amt * w
                    hold[t] = a / p
                    cash -= a
    result = pd.DataFrame(hist).set_index('Date')
    result.attrs['rebals'] = rebals
    return result


def metrics(vals):
    if len(vals) < 2: return {'cagr':0,'mdd':0,'sharpe':0,'sortino':0,'calmar':0,'final':0}
    days = (vals.index[-1] - vals.index[0]).days
    if days <= 0: return {'cagr':0,'mdd':0,'sharpe':0,'sortino':0,'calmar':0,'final':0}
    cagr = (vals.iloc[-1]/vals.iloc[0])**(365.25/days) - 1
    mdd = (vals/vals.cummax()-1).min()
    dr = vals.pct_change().dropna()
    sharpe = (dr.mean()/dr.std())*np.sqrt(252) if dr.std()>0 else 0
    ds = dr[dr<0]
    sortino = (dr.mean()/ds.std())*np.sqrt(252) if len(ds)>0 and ds.std()>0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    return {'cagr':cagr,'mdd':mdd,'sharpe':sharpe,'sortino':sortino,'calmar':calmar,'final':vals.iloc[-1]}


# ===========================================================================
# Main
# ===========================================================================

def main():
    data = load_data()
    print("Pre-computing indicators...")
    ind = precompute(data)
    di = data.index

    # --- Selection methods (best params from previous test + baseline) ---
    strategies = {
        'Baseline Top3∪EW': make_strategy(sel_fixed_top3, {}, 'union'),
        'RelThresh a=0.9 COM': make_strategy(sel_relative_threshold, {'alpha':0.9,'min_n':2,'max_n':6}, 'combined'),
        'RelThresh a=0.8 COM': make_strategy(sel_relative_threshold, {'alpha':0.8,'min_n':2,'max_n':6}, 'combined'),
        'RelThresh a=0.7 UNI': make_strategy(sel_relative_threshold, {'alpha':0.7,'min_n':2,'max_n':6}, 'union'),
        'ZScore z=0.0 COM': make_strategy(sel_zscore, {'z_cutoff':0.0,'min_n':1,'max_n':5}, 'combined'),
        'ZScore z=0.3 COM': make_strategy(sel_zscore, {'z_cutoff':0.3,'min_n':1,'max_n':5}, 'combined'),
        'ZScore z=0.5 UNI': make_strategy(sel_zscore, {'z_cutoff':0.5,'min_n':1,'max_n':5}, 'union'),
        'ZScore z=1.0 UNI': make_strategy(sel_zscore, {'z_cutoff':1.0,'min_n':1,'max_n':5}, 'union'),
        'Gap min=2 max=6 UNI': make_strategy(sel_largest_gap, {'min_n':2,'max_n':6}, 'union'),
        'Gap min=2 max=6 COM': make_strategy(sel_largest_gap, {'min_n':2,'max_n':6}, 'combined'),
        'Gap min=1 max=4 COM': make_strategy(sel_largest_gap, {'min_n':1,'max_n':4}, 'combined'),
        'KMeans k=2 COM': make_strategy(sel_kmeans, {'k':2,'min_n':1,'max_n':6}, 'combined'),
        'KMeans k=3 COM': make_strategy(sel_kmeans, {'k':3,'min_n':1,'max_n':6}, 'combined'),
        'TopN+Tol n=3 e=0.02 COM': make_strategy(sel_topn_tolerance, {'n':3,'epsilon':0.02,'max_n':6}, 'combined'),
        'TopN+Tol n=3 e=0.05 UNI': make_strategy(sel_topn_tolerance, {'n':3,'epsilon':0.05,'max_n':6}, 'union'),
        'TopN+Tol n=3 e=0.10 UNI': make_strategy(sel_topn_tolerance, {'n':3,'epsilon':0.10,'max_n':6}, 'union'),
        'TopN+Tol n=2 e=0.05 COM': make_strategy(sel_topn_tolerance, {'n':2,'epsilon':0.05,'max_n':6}, 'combined'),
        'Softmax b=5 mw=0.10 COM': make_strategy(sel_softmax, {'beta':5.0,'min_weight':0.10,'min_n':2,'max_n':6}, 'combined'),
        'Softmax b=8 mw=0.10 COM': make_strategy(sel_softmax, {'beta':8.0,'min_weight':0.10,'min_n':2,'max_n':6}, 'combined'),
        'Softmax b=3 mw=0.05 UNI': make_strategy(sel_softmax, {'beta':3.0,'min_weight':0.05,'min_n':2,'max_n':6}, 'union'),
    }

    # --- Timings ---
    timings = {
        'MonthEnd':   lambda s, e: gen_month_end(s, e, di),
        'MonthStart': lambda s, e: gen_month_start(s, e, di),
        'Day5':       lambda s, e: gen_fixed_day(s, e, di, 5),
        'Day10':      lambda s, e: gen_fixed_day(s, e, di, 10),
        'Day15':      lambda s, e: gen_fixed_day(s, e, di, 15),
        'Day20':      lambda s, e: gen_fixed_day(s, e, di, 20),
        'Day25':      lambda s, e: gen_fixed_day(s, e, di, 25),
    }

    total = len(strategies) * len(timings) * len(START_DATES)
    print(f"\nRunning {len(strategies)} strategies x {len(timings)} timings x {len(START_DATES)} periods = {total} backtests...\n")

    rows = []
    n = 0
    for sname, sfunc in strategies.items():
        for tname, tgen in timings.items():
            for start in START_DATES:
                n += 1
                try:
                    rebal_dates = tgen(start, END_DATE)
                    res = run_bt(data, ind, sfunc, start, END_DATE, rebal_dates)
                    m = metrics(res['Value'])
                    rows.append({
                        'Strategy': sname, 'Timing': tname, 'Start': start,
                        'CAGR': m['cagr'], 'MDD': m['mdd'], 'Sharpe': m['sharpe'],
                        'Sortino': m['sortino'], 'Calmar': m['calmar'], 'Final': m['final'],
                    })
                except:
                    rows.append({
                        'Strategy': sname, 'Timing': tname, 'Start': start,
                        'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'Calmar': 0, 'Final': 10000,
                    })
            if n % 100 == 0:
                print(f"  Progress: {n}/{total} ({n*100//total}%)")

    df = pd.DataFrame(rows)

    # ===== PART 1: Strategy × Timing average (across all start dates) =====
    pivot = df.groupby(['Strategy','Timing']).agg({'Sharpe':'mean','CAGR':'mean','MDD':'mean','Calmar':'mean'}).reset_index()

    # Best timing per strategy
    best_per_strat = pivot.loc[pivot.groupby('Strategy')['Sharpe'].idxmax()]
    best_per_strat = best_per_strat.sort_values('Sharpe', ascending=False).reset_index(drop=True)

    print(f"\n{'='*130}")
    print(f"  BEST TIMING PER STRATEGY (각 전략의 최적 타이밍)")
    print(f"{'='*130}")
    print(f"{'#':>3} {'Strategy':<30} {'Best Timing':<12} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Calmar':>8}")
    print(f"{'-'*130}")

    bl_sharpe = best_per_strat[best_per_strat['Strategy']=='Baseline Top3∪EW']['Sharpe'].values[0]

    for i, (_, r) in enumerate(best_per_strat.iterrows(), 1):
        mark = ' <-- BASE' if 'Baseline' in r['Strategy'] else (' *' if r['Sharpe'] >= bl_sharpe else '')
        print(f"{i:>3} {r['Strategy']:<30} {r['Timing']:<12} {r['CAGR']:>7.1%} {r['MDD']:>7.1%} "
              f"{r['Sharpe']:>8.2f} {r['Calmar']:>8.2f}{mark}")
    print(f"{'='*130}")

    # ===== PART 2: Full Strategy × Timing matrix (Sharpe) =====
    sharpe_matrix = pivot.pivot(index='Strategy', columns='Timing', values='Sharpe')
    timing_order = ['MonthEnd','MonthStart','Day5','Day10','Day15','Day20','Day25']
    sharpe_matrix = sharpe_matrix.reindex(columns=[t for t in timing_order if t in sharpe_matrix.columns])

    # Sort by max Sharpe across timings
    sharpe_matrix['Max'] = sharpe_matrix.max(axis=1)
    sharpe_matrix['Min'] = sharpe_matrix.min(axis=1)
    sharpe_matrix['Spread'] = sharpe_matrix['Max'] - sharpe_matrix['Min']
    sharpe_matrix = sharpe_matrix.sort_values('Max', ascending=False)

    print(f"\n{'='*150}")
    print(f"  SHARPE MATRIX (Strategy × Timing) — 굵은 숫자가 해당 타이밍 최고")
    print(f"{'='*150}")

    header = f"{'Strategy':<30}"
    for t in timing_order:
        if t in sharpe_matrix.columns:
            header += f" {t:>10}"
    header += f" {'Max':>7} {'Min':>7} {'Spread':>7}"
    print(header)
    print(f"{'-'*150}")

    for name, row in sharpe_matrix.iterrows():
        line = f"{name:<30}"
        for t in timing_order:
            if t in sharpe_matrix.columns:
                val = row[t]
                line += f" {val:>10.2f}"
        line += f" {row['Max']:>7.2f} {row['Min']:>7.2f} {row['Spread']:>7.2f}"
        if 'Baseline' in name:
            line += ' <BASE'
        print(line)

    print(f"{'='*150}")

    # ===== PART 3: CAGR matrix =====
    cagr_matrix = pivot.pivot(index='Strategy', columns='Timing', values='CAGR')
    cagr_matrix = cagr_matrix.reindex(columns=[t for t in timing_order if t in cagr_matrix.columns])
    cagr_matrix['Max'] = cagr_matrix.max(axis=1)
    cagr_matrix = cagr_matrix.sort_values('Max', ascending=False)

    print(f"\n{'='*150}")
    print(f"  CAGR MATRIX (Strategy × Timing)")
    print(f"{'='*150}")

    header = f"{'Strategy':<30}"
    for t in timing_order:
        if t in cagr_matrix.columns:
            header += f" {t:>10}"
    header += f" {'Max':>8}"
    print(header)
    print(f"{'-'*150}")

    for name, row in cagr_matrix.iterrows():
        line = f"{name:<30}"
        for t in timing_order:
            if t in cagr_matrix.columns:
                val = row[t]
                line += f" {val:>9.1%}"
        line += f" {row['Max']:>7.1%}"
        if 'Baseline' in name:
            line += ' <BASE'
        print(line)

    print(f"{'='*150}")

    # ===== PART 4: Any method that beats baseline at ANY timing? =====
    print(f"\n{'='*130}")
    print(f"  BASELINE 대비 분석 — 어떤 타이밍에서든 이기는 전략이 있는가?")
    print(f"{'='*130}")

    bl_by_timing = pivot[pivot['Strategy']=='Baseline Top3∪EW'].set_index('Timing')

    wins = {}
    for sname in strategies:
        if 'Baseline' in sname: continue
        s_by_timing = pivot[pivot['Strategy']==sname].set_index('Timing')
        win_count = 0
        win_timings = []
        for t in timing_order:
            if t in s_by_timing.index and t in bl_by_timing.index:
                if s_by_timing.loc[t, 'Sharpe'] > bl_by_timing.loc[t, 'Sharpe']:
                    win_count += 1
                    win_timings.append(t)
        wins[sname] = (win_count, win_timings)

    for sname, (cnt, tlist) in sorted(wins.items(), key=lambda x: -x[1][0]):
        if cnt > 0:
            print(f"  {sname:<30} Baseline에 승리: {cnt}/{len(timing_order)} 타이밍 ({', '.join(tlist)})")

    if not any(c > 0 for c, _ in wins.values()):
        print(f"  → 모든 타이밍에서 Baseline이 승리")

    # ===== PART 5: Timing stability (which strategy is least sensitive?) =====
    print(f"\n{'='*130}")
    print(f"  타이밍 민감도 (Spread가 작을수록 안정적)")
    print(f"{'='*130}")

    sm = sharpe_matrix[['Spread','Max','Min']].sort_values('Spread')
    print(f"  {'Strategy':<30} {'Spread':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*60}")
    for name, row in sm.iterrows():
        mark = ' <BASE' if 'Baseline' in name else ''
        print(f"  {name:<30} {row['Spread']:>8.2f} {row['Min']:>8.2f} {row['Max']:>8.2f}{mark}")

    print(f"{'='*130}")


if __name__ == '__main__':
    main()
