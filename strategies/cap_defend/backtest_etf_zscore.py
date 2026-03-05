"""
ETF Z-Score Dynamic Selection — 18+ variants × 7 timings
=========================================================
Deep exploration of Z-Score based stock selection methods.

Usage:
    python3 strategies/cap_defend/backtest_etf_zscore.py
"""

import os, sys, warnings
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from scipy import stats as sp_stats

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

        # Time-series Z components (rolling 60-day stats)
        d['mom_w_rm60'] = d['mom_w'].rolling(60).mean()
        d['mom_w_rs60'] = d['mom_w'].rolling(60).std()
        d['sharpe_rm60'] = d['sharpe126'].rolling(60).mean()
        d['sharpe_rs60'] = d['sharpe126'].rolling(60).std()

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
# Z-Score Strategy Variants
# ===========================================================================

def get_scores(d, ind):
    """Get raw Mom and Qual scores for all offensive tickers."""
    rows = []
    for t in OFFENSIVE:
        if not has(ind, t, d): continue
        mom = v(ind, t, 'mom_w', d)
        qual = v(ind, t, 'sharpe126', d)
        vol = v(ind, t, 'vol90', d)
        # Time-series Z components
        mom_rm = v(ind, t, 'mom_w_rm60', d)
        mom_rs = v(ind, t, 'mom_w_rs60', d)
        qual_rm = v(ind, t, 'sharpe_rm60', d)
        qual_rs = v(ind, t, 'sharpe_rs60', d)
        if pd.notna(mom) and pd.notna(qual):
            rows.append({
                'T': t, 'M': mom, 'Q': qual, 'V': vol if pd.notna(vol) else 0.01,
                'M_rm': mom_rm, 'M_rs': mom_rs, 'Q_rm': qual_rm, 'Q_rs': qual_rs,
            })
    if not rows: return None
    return pd.DataFrame(rows).set_index('T')


def cross_z(series):
    """Standard cross-sectional Z-score."""
    mu, sigma = series.mean(), series.std()
    if sigma == 0 or pd.isna(sigma): return pd.Series(0, index=series.index)
    return (series - mu) / sigma


def mad_z(series):
    """Modified Z using median and MAD (robust to outliers)."""
    med = series.median()
    mad = (series - med).abs().median()
    if mad == 0: return pd.Series(0, index=series.index)
    return (series - med) / (1.4826 * mad)


def winsorized_z(series, pct=0.1):
    """Winsorized Z-score (clip extremes at 10/90 percentile)."""
    lo, hi = series.quantile(pct), series.quantile(1 - pct)
    clipped = series.clip(lo, hi)
    mu, sigma = clipped.mean(), clipped.std()
    if sigma == 0 or pd.isna(sigma): return pd.Series(0, index=series.index)
    return (clipped - mu) / sigma


def rank_z(series):
    """Rank-based Z (inverse normal transform of ranks)."""
    n = len(series)
    if n < 2: return pd.Series(0, index=series.index)
    ranks = series.rank()
    # Blom's formula: Phi^-1((r - 3/8) / (n + 1/4))
    probs = (ranks - 0.375) / (n + 0.25)
    probs = probs.clip(0.001, 0.999)
    return pd.Series(sp_stats.norm.ppf(probs), index=series.index)


def make_zscore_strategy(
    z_cutoff=0.0,
    mom_weight=0.5,
    qual_weight=0.5,
    vol_weight=0.0,
    z_method='cross',        # cross, mad, winsorized, rank, timeseries
    abs_mom_filter=False,     # require Mom > 0
    min_n=1, max_n=7,
    weight_mode='ew',         # ew, z_prop, softmax
    softmax_temp=1.0,
    two_stage=False,          # 2-stage filter
    two_stage_first=-0.5,
    adaptive=False,           # adaptive cutoff based on market vol
):
    def strategy(d, ind):
        # Canary check
        risk_on = all(has(ind,c,d) and v(ind,c,'price',d) > v(ind,c,'sma200',d) for c in CANARY)
        if not risk_on:
            best_t, best_r = 'BIL', -999
            for t in DEFENSIVE:
                r = v(ind, t, 'mom126', d)
                if pd.notna(r) and r > best_r: best_r, best_t = r, t
            return ({'BIL': 1.0}, 'Cash') if best_r < 0 else ({best_t: 1.0}, 'Off')

        df = get_scores(d, ind)
        if df is None or df.empty: return {'BIL': 1.0}, 'NoData'

        # Compute Z-scores based on method
        if z_method == 'cross':
            zm = cross_z(df['M'])
            zq = cross_z(df['Q'])
            zv = cross_z(df['V']) if vol_weight != 0 else pd.Series(0, index=df.index)
        elif z_method == 'mad':
            zm = mad_z(df['M'])
            zq = mad_z(df['Q'])
            zv = mad_z(df['V']) if vol_weight != 0 else pd.Series(0, index=df.index)
        elif z_method == 'winsorized':
            zm = winsorized_z(df['M'])
            zq = winsorized_z(df['Q'])
            zv = winsorized_z(df['V']) if vol_weight != 0 else pd.Series(0, index=df.index)
        elif z_method == 'rank':
            zm = rank_z(df['M'])
            zq = rank_z(df['Q'])
            zv = rank_z(df['V']) if vol_weight != 0 else pd.Series(0, index=df.index)
        elif z_method == 'timeseries':
            # Each ticker's Z vs its own 60-day history
            zm = pd.Series(index=df.index, dtype=float)
            zq = pd.Series(index=df.index, dtype=float)
            for t in df.index:
                rs_m = df.loc[t, 'M_rs']
                rs_q = df.loc[t, 'Q_rs']
                zm[t] = (df.loc[t,'M'] - df.loc[t,'M_rm']) / rs_m if pd.notna(rs_m) and rs_m > 0 else 0
                zq[t] = (df.loc[t,'Q'] - df.loc[t,'Q_rm']) / rs_q if pd.notna(rs_q) and rs_q > 0 else 0
            zv = pd.Series(0, index=df.index)
        else:
            zm = cross_z(df['M'])
            zq = cross_z(df['Q'])
            zv = pd.Series(0, index=df.index)

        # Combined Z score
        z_total = mom_weight * zm + qual_weight * zq - vol_weight * zv

        # Adaptive cutoff
        cutoff = z_cutoff
        if adaptive:
            spy_vol = v(ind, 'SPY', 'vol90', d)
            if pd.notna(spy_vol):
                if spy_vol > 0.02:    # High vol (~VIX > 30)
                    cutoff = z_cutoff + 0.5
                elif spy_vol > 0.015: # Medium vol (~VIX > 20)
                    cutoff = z_cutoff + 0.3

        # Selection
        if two_stage:
            # Stage 1: remove bottom
            stage1 = z_total[z_total >= two_stage_first]
            if stage1.empty: stage1 = z_total
            # Stage 2: top half of survivors
            median_z = stage1.median()
            picks_idx = stage1[stage1 >= median_z].index
        else:
            picks_idx = z_total[z_total >= cutoff].index

        # Absolute momentum filter
        if abs_mom_filter:
            picks_idx = [t for t in picks_idx if df.loc[t, 'M'] > 0]

        # Min/max enforcement
        picks_idx = list(picks_idx)
        if len(picks_idx) < min_n:
            # Force top min_n by z_total
            top = z_total.sort_values(ascending=False).head(min_n).index.tolist()
            picks_idx = top
        if len(picks_idx) > max_n:
            # Keep top max_n by z_total
            z_sub = z_total[picks_idx].sort_values(ascending=False)
            picks_idx = z_sub.head(max_n).index.tolist()

        if not picks_idx:
            return {'BIL': 1.0}, 'NoPick'

        # Weighting
        if weight_mode == 'z_prop' and len(picks_idx) > 1:
            zvals = z_total[picks_idx]
            # Shift to positive
            zvals_pos = zvals - zvals.min() + 0.1
            tot = zvals_pos.sum()
            port = {t: float(zvals_pos[t] / tot) for t in picks_idx}
        elif weight_mode == 'softmax' and len(picks_idx) > 1:
            zvals = z_total[picks_idx].values
            exp_z = np.exp(zvals / softmax_temp)
            w = exp_z / exp_z.sum()
            port = {picks_idx[i]: float(w[i]) for i in range(len(picks_idx))}
        else:
            port = {t: 1.0 / len(picks_idx) for t in picks_idx}

        return port, f'On({len(picks_idx)})'
    return strategy


# ===========================================================================
# Build all variants
# ===========================================================================

def build_variants():
    V = {}

    # --- Baseline (fixed top3 union) ---
    def baseline(d, ind):
        risk_on = all(has(ind,c,d) and v(ind,c,'price',d) > v(ind,c,'sma200',d) for c in CANARY)
        if not risk_on:
            best_t, best_r = 'BIL', -999
            for t in DEFENSIVE:
                r = v(ind, t, 'mom126', d)
                if pd.notna(r) and r > best_r: best_r, best_t = r, t
            return ({'BIL': 1.0}, 'Cash') if best_r < 0 else ({best_t: 1.0}, 'Off')
        df = get_scores(d, ind)
        if df is None: return {'BIL': 1.0}, 'NoData'
        picks = list(set(df.nlargest(3,'M').index.tolist() + df.nlargest(3,'Q').index.tolist()))
        return {t: 1.0/len(picks) for t in picks}, 'On'
    V['00.Baseline Top3∪EW'] = baseline

    # --- 1. Cutoff variations (cross-sectional, 50:50, EW) ---
    for z in [-0.5, -0.3, 0.0, 0.3, 0.5, 0.7, 1.0]:
        V[f'01.Cross z≥{z}'] = make_zscore_strategy(z_cutoff=z)

    # --- 2. Mom:Qual weight variations ---
    for mw, qw, label in [(0.7,0.3,'70:30'), (0.6,0.4,'60:40'), (0.4,0.6,'40:60'), (0.3,0.7,'30:70')]:
        V[f'02.MQ={label} z≥0'] = make_zscore_strategy(mom_weight=mw, qual_weight=qw, z_cutoff=0.0)

    # --- 3. Z-Score calculation methods ---
    V['03.MAD-Z z≥0'] = make_zscore_strategy(z_method='mad', z_cutoff=0.0)
    V['03.Winsor-Z z≥0'] = make_zscore_strategy(z_method='winsorized', z_cutoff=0.0)
    V['03.Rank-Z z≥0'] = make_zscore_strategy(z_method='rank', z_cutoff=0.0)
    V['03.TimeSeries-Z z≥1'] = make_zscore_strategy(z_method='timeseries', z_cutoff=1.0)
    V['03.TimeSeries-Z z≥0.5'] = make_zscore_strategy(z_method='timeseries', z_cutoff=0.5)

    # --- 4. Absolute momentum filter ---
    V['04.Cross z≥0 +AbsMom'] = make_zscore_strategy(z_cutoff=0.0, abs_mom_filter=True)
    V['04.Cross z≥-0.3 +AbsMom'] = make_zscore_strategy(z_cutoff=-0.3, abs_mom_filter=True)

    # --- 5. Min/Max constraints ---
    V['05.Cross z≥0.3 min2 max5'] = make_zscore_strategy(z_cutoff=0.3, min_n=2, max_n=5)
    V['05.Cross z≥0 min2 max4'] = make_zscore_strategy(z_cutoff=0.0, min_n=2, max_n=4)
    V['05.Cross z≥0 min3 max6'] = make_zscore_strategy(z_cutoff=0.0, min_n=3, max_n=6)

    # --- 6. Weight modes ---
    V['06.Cross z≥0 Z-Prop'] = make_zscore_strategy(z_cutoff=0.0, weight_mode='z_prop')
    V['06.Cross z≥0 Softmax T=0.5'] = make_zscore_strategy(z_cutoff=0.0, weight_mode='softmax', softmax_temp=0.5)
    V['06.Cross z≥0 Softmax T=1.0'] = make_zscore_strategy(z_cutoff=0.0, weight_mode='softmax', softmax_temp=1.0)
    V['06.Cross z≥-0.3 Softmax T=0.5'] = make_zscore_strategy(z_cutoff=-0.3, weight_mode='softmax', softmax_temp=0.5)

    # --- 7. Two-stage filter ---
    V['07.2Stage (-0.5→top50%)'] = make_zscore_strategy(two_stage=True, two_stage_first=-0.5)
    V['07.2Stage (-0.3→top50%)'] = make_zscore_strategy(two_stage=True, two_stage_first=-0.3)
    V['07.2Stage (0.0→top50%)'] = make_zscore_strategy(two_stage=True, two_stage_first=0.0)

    # --- 8. Adaptive cutoff ---
    V['08.Adaptive z≥0 (vol)'] = make_zscore_strategy(z_cutoff=0.0, adaptive=True)
    V['08.Adaptive z≥-0.3 (vol)'] = make_zscore_strategy(z_cutoff=-0.3, adaptive=True)

    # --- 9. 3-Factor (Mom+Qual+Vol) ---
    V['09.3Factor MQV 40:40:20 z≥0'] = make_zscore_strategy(mom_weight=0.4, qual_weight=0.4, vol_weight=0.2, z_cutoff=0.0)
    V['09.3Factor MQV 40:40:20 z≥0.3'] = make_zscore_strategy(mom_weight=0.4, qual_weight=0.4, vol_weight=0.2, z_cutoff=0.3)
    V['09.3Factor MQV 30:30:40 z≥0'] = make_zscore_strategy(mom_weight=0.3, qual_weight=0.3, vol_weight=0.4, z_cutoff=0.0)

    # --- 10. Defensive / Aggressive profiles ---
    V['10.Defensive Q80 z≥0.5'] = make_zscore_strategy(mom_weight=0.2, qual_weight=0.8, z_cutoff=0.5, min_n=1, max_n=4)
    V['10.Aggressive M80 z≥-0.3'] = make_zscore_strategy(mom_weight=0.8, qual_weight=0.2, z_cutoff=-0.3)

    # --- 11. Best combos from above ---
    V['11.MAD z≥0 Z-Prop'] = make_zscore_strategy(z_method='mad', z_cutoff=0.0, weight_mode='z_prop')
    V['11.Rank z≥0 MQ=60:40'] = make_zscore_strategy(z_method='rank', z_cutoff=0.0, mom_weight=0.6, qual_weight=0.4)
    V['11.Cross z≥0 MQ=60:40 +AbsMom'] = make_zscore_strategy(z_cutoff=0.0, mom_weight=0.6, qual_weight=0.4, abs_mom_filter=True)
    V['11.Winsor z≥0 MQ=70:30'] = make_zscore_strategy(z_method='winsorized', z_cutoff=0.0, mom_weight=0.7, qual_weight=0.3)
    V['11.3F MQV 40:40:20 +AbsMom z≥0'] = make_zscore_strategy(mom_weight=0.4, qual_weight=0.4, vol_weight=0.2, z_cutoff=0.0, abs_mom_filter=True)

    return V


# ===========================================================================
# Backtest & Metrics
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
                    a = amt * w; hold[t] = a / p; cash -= a
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

    variants = build_variants()

    timings = {
        'MonthEnd':   lambda s, e: gen_month_end(s, e, di),
        'MonthStart': lambda s, e: gen_month_start(s, e, di),
        'Day5':       lambda s, e: gen_fixed_day(s, e, di, 5),
        'Day10':      lambda s, e: gen_fixed_day(s, e, di, 10),
        'Day15':      lambda s, e: gen_fixed_day(s, e, di, 15),
        'Day20':      lambda s, e: gen_fixed_day(s, e, di, 20),
        'Day25':      lambda s, e: gen_fixed_day(s, e, di, 25),
    }

    total = len(variants) * len(timings) * len(START_DATES)
    print(f"\nRunning {len(variants)} variants x {len(timings)} timings x {len(START_DATES)} periods = {total} backtests...\n")

    rows = []
    n = 0
    for sname, sfunc in variants.items():
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
                except Exception as e:
                    rows.append({
                        'Strategy': sname, 'Timing': tname, 'Start': start,
                        'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'Calmar': 0, 'Final': 10000,
                    })
            if n % 200 == 0:
                print(f"  Progress: {n}/{total} ({n*100//total}%)")

    df = pd.DataFrame(rows)

    # ===== PART 1: Best timing per strategy (avg across periods) =====
    pivot = df.groupby(['Strategy','Timing']).agg({
        'Sharpe':'mean','CAGR':'mean','MDD':'mean','Sortino':'mean','Calmar':'mean'
    }).reset_index()

    best_per_strat = pivot.loc[pivot.groupby('Strategy')['Sharpe'].idxmax()]
    best_per_strat = best_per_strat.sort_values('Sharpe', ascending=False).reset_index(drop=True)

    bl_sharpe = best_per_strat[best_per_strat['Strategy'].str.contains('Baseline')]['Sharpe'].values[0]

    print(f"\n{'='*140}")
    print(f"  Z-SCORE VARIANTS — BEST TIMING PER STRATEGY (Avg across {len(START_DATES)} periods)")
    print(f"{'='*140}")
    print(f"{'#':>3} {'Strategy':<38} {'Timing':<12} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8}")
    print(f"{'-'*140}")

    for i, (_, r) in enumerate(best_per_strat.iterrows(), 1):
        mark = ' <BASE' if 'Baseline' in r['Strategy'] else (' *' if r['Sharpe'] > bl_sharpe else '')
        print(f"{i:>3} {r['Strategy']:<38} {r['Timing']:<12} {r['CAGR']:>7.1%} {r['MDD']:>7.1%} "
              f"{r['Sharpe']:>8.2f} {r['Sortino']:>8.2f} {r['Calmar']:>8.2f}{mark}")
    print(f"{'='*140}")

    # ===== PART 2: Average across ALL timings (timing-robust ranking) =====
    avg_all = df.groupby('Strategy').agg({
        'Sharpe':'mean','CAGR':'mean','MDD':'mean','Sortino':'mean','Calmar':'mean'
    }).reset_index()
    avg_all = avg_all.sort_values('Sharpe', ascending=False).reset_index(drop=True)

    bl_avg = avg_all[avg_all['Strategy'].str.contains('Baseline')]['Sharpe'].values[0]

    print(f"\n{'='*140}")
    print(f"  TIMING-ROBUST RANKING (Avg across ALL timings × ALL periods)")
    print(f"{'='*140}")
    print(f"{'#':>3} {'Strategy':<38} {'AvgCAGR':>9} {'AvgMDD':>9} {'AvgSharpe':>10} {'AvgSortino':>10} {'AvgCalmar':>10}")
    print(f"{'-'*140}")

    for i, (_, r) in enumerate(avg_all.iterrows(), 1):
        mark = ' <BASE' if 'Baseline' in r['Strategy'] else (' *' if r['Sharpe'] > bl_avg else '')
        print(f"{i:>3} {r['Strategy']:<38} {r['CAGR']:>8.1%} {r['MDD']:>8.1%} "
              f"{r['Sharpe']:>10.2f} {r['Sortino']:>10.2f} {r['Calmar']:>10.2f}{mark}")
    print(f"{'='*140}")

    # ===== PART 3: Top 10 Sharpe matrix =====
    top10_names = avg_all.head(10)['Strategy'].tolist()
    if not any('Baseline' in s for s in top10_names):
        top10_names.append('00.Baseline Top3∪EW')

    timing_order = ['MonthEnd','MonthStart','Day5','Day10','Day15','Day20','Day25']

    # Sharpe matrix for top strategies
    top_pivot = pivot[pivot['Strategy'].isin(top10_names)]
    sharpe_mat = top_pivot.pivot_table(index='Strategy', columns='Timing', values='Sharpe')
    sharpe_mat = sharpe_mat.reindex(columns=[t for t in timing_order if t in sharpe_mat.columns])
    sharpe_mat['Avg'] = sharpe_mat.mean(axis=1)
    sharpe_mat['Spread'] = sharpe_mat[timing_order].max(axis=1) - sharpe_mat[timing_order].min(axis=1)
    sharpe_mat = sharpe_mat.sort_values('Avg', ascending=False)

    print(f"\n{'='*150}")
    print(f"  TOP 10 SHARPE MATRIX (Strategy × Timing)")
    print(f"{'='*150}")
    header = f"{'Strategy':<38}"
    for t in timing_order: header += f" {t:>10}"
    header += f" {'Avg':>7} {'Spread':>7}"
    print(header)
    print(f"{'-'*150}")

    for name, row in sharpe_mat.iterrows():
        line = f"{name:<38}"
        for t in timing_order:
            if t in sharpe_mat.columns: line += f" {row[t]:>10.2f}"
        line += f" {row['Avg']:>7.2f} {row['Spread']:>7.2f}"
        if 'Baseline' in name: line += ' <BASE'
        print(line)
    print(f"{'='*150}")

    # ===== PART 4: Category summary =====
    print(f"\n{'='*140}")
    print(f"  CATEGORY SUMMARY (각 카테고리별 최고 전략)")
    print(f"{'='*140}")

    categories = [
        ('01.', 'Cutoff Variations'),
        ('02.', 'MQ Weight Ratio'),
        ('03.', 'Z Calculation Method'),
        ('04.', 'Abs Mom Filter'),
        ('05.', 'Min/Max Constraints'),
        ('06.', 'Weight Mode'),
        ('07.', 'Two-Stage Filter'),
        ('08.', 'Adaptive Cutoff'),
        ('09.', '3-Factor (MQV)'),
        ('10.', 'Defensive/Aggressive'),
        ('11.', 'Best Combos'),
    ]

    print(f"  {'Category':<24} {'Best Variant':<38} {'AvgSharpe':>10} {'AvgCAGR':>9} {'AvgMDD':>9} {'Spread':>8}")
    print(f"  {'-'*105}")

    for prefix, cat_name in categories:
        cat = avg_all[avg_all['Strategy'].str.startswith(prefix)]
        if cat.empty: continue
        best_name = cat.iloc[0]['Strategy']
        # Get spread for best
        if best_name in sharpe_mat.index:
            spread = sharpe_mat.loc[best_name, 'Spread']
        else:
            s_timing = pivot[pivot['Strategy']==best_name]
            spread = s_timing['Sharpe'].max() - s_timing['Sharpe'].min()
        print(f"  {cat_name:<24} {best_name:<38} {cat.iloc[0]['Sharpe']:>10.2f} "
              f"{cat.iloc[0]['CAGR']:>8.1%} {cat.iloc[0]['MDD']:>8.1%} {spread:>8.2f}")

    print(f"\n  Baseline Avg Sharpe: {bl_avg:.2f}")
    beat_count = len(avg_all[avg_all['Sharpe'] > bl_avg])
    print(f"  Variants beating baseline (all-timing avg): {beat_count}/{len(avg_all)}")
    print(f"{'='*140}")

    # ===== PART 5: Wins against baseline at each timing =====
    print(f"\n{'='*140}")
    print(f"  BASELINE 대비 — 모든 타이밍에서 이기는 전략")
    print(f"{'='*140}")

    bl_by_timing = pivot[pivot['Strategy']=='00.Baseline Top3∪EW'].set_index('Timing')

    all_win = []
    for sname in variants:
        if 'Baseline' in sname: continue
        s_by_timing = pivot[pivot['Strategy']==sname].set_index('Timing')
        wins = sum(1 for t in timing_order
                   if t in s_by_timing.index and t in bl_by_timing.index
                   and s_by_timing.loc[t,'Sharpe'] > bl_by_timing.loc[t,'Sharpe'])
        if wins >= 6:
            avg_s = avg_all[avg_all['Strategy']==sname]['Sharpe'].values[0]
            all_win.append((sname, wins, avg_s))

    all_win.sort(key=lambda x: -x[2])
    for sname, wins, avg_s in all_win:
        print(f"  {sname:<38} {wins}/7 타이밍 승리, Avg Sharpe {avg_s:.2f}")

    print(f"{'='*140}")


if __name__ == '__main__':
    main()
