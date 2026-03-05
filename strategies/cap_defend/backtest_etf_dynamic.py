"""
ETF Dynamic Selection Backtest — 6 methods × parameter variants × combination modes
===================================================================================
Tests dynamic stock selection methods vs fixed top-3 baseline.

Methods:
  1. Relative Threshold (alpha)
  2. Z-Score Cutoff
  3. Largest Gap
  4. K-Means Clustering
  5. Top-N + Tolerance
  6. Softmax Cutoff

Combination modes: Union, Intersection, Combined Score
Weighting: Equal Weight, Inverse Volatility, Softmax Weight

Usage:
    python3 strategies/cap_defend/backtest_etf_dynamic.py
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
# Data
# ===========================================================================

def download_data():
    import requests
    from requests.adapters import HTTPAdapter
    from urllib3.util.retry import Retry
    os.makedirs(DATA_DIR, exist_ok=True)
    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    retries = Retry(total=5, backoff_factor=1.0, status_forcelist=[500,502,503,504])
    session.mount('https://', HTTPAdapter(max_retries=retries))
    for ticker in ALL_ETF_TICKERS:
        fp = os.path.join(DATA_DIR, f"{ticker}.csv")
        if os.path.exists(fp): continue
        try:
            end_ts = int(datetime.now(timezone.utc).timestamp())
            start_ts = int(datetime(2014,1,1,tzinfo=timezone.utc).timestamp())
            url = f"https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"
            params = {"period1":start_ts,"period2":end_ts,"interval":"1d","includeAdjustedClose":"true"}
            resp = session.get(url, params=params, timeout=30)
            if resp.status_code == 200:
                res = resp.json()['chart']['result'][0]
                df = pd.DataFrame({'Date': pd.to_datetime(res['timestamp'],unit='s').date,
                                   'Adj_Close': res['indicators']['adjclose'][0]['adjclose']})
                df.dropna().drop_duplicates('Date').to_csv(fp, index=False)
                print(f"  Downloaded {ticker}")
        except: pass
    print("Download complete.")


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
    val = v(ind, ticker, 'mom252', date)
    return pd.notna(val)


# ===========================================================================
# Dynamic Selection Methods
# ===========================================================================

def sel_fixed_top(scores, n=3, **kw):
    """Baseline: fixed top-N."""
    return scores.head(n).index.tolist()


def sel_relative_threshold(scores, alpha=0.8, min_n=2, max_n=6, **kw):
    """Method 1: Select all within alpha * max_score."""
    if scores.empty: return []
    pos = scores[scores > 0]
    if pos.empty: return []
    cutoff = pos.iloc[0] * alpha  # scores already sorted desc
    picked = pos[pos >= cutoff]
    picked = picked.iloc[:max_n]
    if len(picked) < min_n:
        picked = scores.head(min_n)
        picked = picked[picked > 0]
    return picked.index.tolist()


def sel_zscore(scores, z_cutoff=0.5, min_n=1, max_n=5, **kw):
    """Method 2: Z-score cutoff."""
    if scores.empty: return []
    mu, sigma = scores.mean(), scores.std()
    if sigma == 0 or pd.isna(sigma):
        return scores.head(min_n).index.tolist()
    z = (scores - mu) / sigma
    picked = scores[z >= z_cutoff]
    if len(picked) > max_n: picked = picked.iloc[:max_n]
    if len(picked) < min_n: picked = scores.head(min_n)
    picked = picked[picked > -np.inf]
    return picked.index.tolist()


def sel_largest_gap(scores, min_n=2, max_n=6, **kw):
    """Method 3: Cut at largest gap."""
    if len(scores) < 2: return scores.index.tolist()
    vals = scores.values
    best_cut = min_n
    best_gap = -1
    for i in range(min_n - 1, min(max_n, len(vals) - 1)):
        gap = vals[i] - vals[i + 1]
        if gap > best_gap:
            best_gap = gap
            best_cut = i + 1
    return scores.head(best_cut).index.tolist()


def sel_kmeans(scores, k=3, min_n=1, max_n=6, **kw):
    """Method 4: 1D K-Means clustering, take top cluster."""
    if len(scores) < k:
        return scores.index.tolist()
    vals = scores.values.reshape(-1, 1)
    # Simple 1D k-means without sklearn
    # Use quantile-based initialization
    centroids = np.quantile(vals.flatten(), np.linspace(0, 1, k))
    for _ in range(20):
        labels = np.argmin(np.abs(vals - centroids), axis=1)
        new_c = np.array([vals[labels == i].mean() if (labels == i).any() else centroids[i]
                          for i in range(k)])
        if np.allclose(centroids, new_c): break
        centroids = new_c
    labels = np.argmin(np.abs(vals - centroids), axis=1)
    top_cluster = labels[0]  # first item (highest score) determines top cluster
    picked = scores[labels == top_cluster]
    if len(picked) > max_n: picked = picked.iloc[:max_n]
    if len(picked) < min_n: picked = scores.head(min_n)
    return picked.index.tolist()


def sel_topn_tolerance(scores, n=3, epsilon=0.03, max_n=6, **kw):
    """Method 5: Top-N + tolerance margin."""
    if len(scores) < n:
        return scores.index.tolist()
    cutoff = scores.iloc[n - 1] - epsilon
    picked = scores[scores >= cutoff]
    if len(picked) > max_n: picked = picked.iloc[:max_n]
    return picked.index.tolist()


def sel_softmax(scores, beta=5.0, min_weight=0.05, min_n=2, max_n=6, **kw):
    """Method 6: Softmax probability cutoff."""
    if scores.empty: return []
    s = scores.values
    exp_s = np.exp(beta * (s - s.max()))
    probs = exp_s / exp_s.sum()
    picked_idx = np.where(probs >= min_weight)[0]
    if len(picked_idx) < min_n:
        picked_idx = np.arange(min(min_n, len(scores)))
    if len(picked_idx) > max_n:
        picked_idx = picked_idx[:max_n]
    return scores.index[picked_idx].tolist()


# ===========================================================================
# Weighting Methods
# ===========================================================================

def weight_equal(picks, ind, d):
    """1/N equal weight."""
    if not picks: return {}
    return {t: 1.0 / len(picks) for t in picks}


def weight_ivw(picks, ind, d):
    """Inverse volatility weight."""
    if not picks: return {}
    vols = {}
    for t in picks:
        vol = v(ind, t, 'vol90', d)
        vols[t] = max(vol, 0.001) if pd.notna(vol) else 0.001
    inv = {t: 1.0 / vol for t, vol in vols.items()}
    tot = sum(inv.values())
    return {t: val / tot for t, val in inv.items()} if tot > 0 else weight_equal(picks, ind, d)


def weight_softmax(picks, ind, d, score_map=None, beta=3.0):
    """Softmax score-proportional weight."""
    if not picks or score_map is None: return weight_equal(picks, ind, d)
    s = np.array([score_map.get(t, 0) for t in picks])
    exp_s = np.exp(beta * (s - s.max()))
    w = exp_s / exp_s.sum()
    return {t: float(w[i]) for i, t in enumerate(picks)}


# ===========================================================================
# Strategy builder
# ===========================================================================

def make_strategy(sel_method, sel_params, combine_mode, weight_mode, weight_params=None):
    """
    Build a strategy function from selection/combination/weighting choices.

    combine_mode: 'union', 'intersect', 'combined'
    weight_mode: 'ew', 'ivw', 'softmax'
    """
    if weight_params is None:
        weight_params = {}

    def strategy(d, ind):
        # Canary check
        risk_on = all(
            has(ind, c, d) and v(ind, c, 'price', d) > v(ind, c, 'sma200', d)
            for c in CANARY
        )

        if not risk_on:
            # Defensive mode (unchanged from baseline)
            best_t, best_r = 'BIL', -999
            for t in DEFENSIVE:
                r = v(ind, t, 'mom126', d)
                if pd.notna(r) and r > best_r:
                    best_r, best_t = r, t
            if best_r < 0:
                return {'BIL': 1.0}, 'Cash'
            return {best_t: 1.0}, 'Off'

        # Offensive mode — compute scores
        rows = []
        for t in OFFENSIVE:
            if not has(ind, t, d): continue
            mom = v(ind, t, 'mom_w', d)
            qual = v(ind, t, 'sharpe126', d)
            if pd.notna(mom) and pd.notna(qual):
                rows.append((t, mom, qual))

        if not rows:
            return {'BIL': 1.0}, 'NoData'

        df = pd.DataFrame(rows, columns=['T', 'M', 'Q']).set_index('T')

        if combine_mode == 'union':
            mom_sorted = df['M'].sort_values(ascending=False)
            qual_sorted = df['Q'].sort_values(ascending=False)
            mom_picks = sel_method(mom_sorted, **sel_params)
            qual_picks = sel_method(qual_sorted, **sel_params)
            picks = list(dict.fromkeys(mom_picks + qual_picks))  # preserve order, deduplicate

        elif combine_mode == 'intersect':
            mom_sorted = df['M'].sort_values(ascending=False)
            qual_sorted = df['Q'].sort_values(ascending=False)
            mom_picks = set(sel_method(mom_sorted, **sel_params))
            qual_picks = set(sel_method(qual_sorted, **sel_params))
            picks = list(mom_picks & qual_picks)
            if not picks:
                # Fallback: take highest combined score
                df['C'] = (df['M'] - df['M'].mean()) / (df['M'].std() + 1e-9) + \
                           (df['Q'] - df['Q'].mean()) / (df['Q'].std() + 1e-9)
                picks = [df['C'].idxmax()]

        elif combine_mode == 'combined':
            # Z-normalize and combine
            m_std = df['M'].std()
            q_std = df['Q'].std()
            df['C'] = ((df['M'] - df['M'].mean()) / (m_std if m_std > 0 else 1) * 0.5 +
                        (df['Q'] - df['Q'].mean()) / (q_std if q_std > 0 else 1) * 0.5)
            combined_sorted = df['C'].sort_values(ascending=False)
            picks = sel_method(combined_sorted, **sel_params)
        else:
            picks = df.nlargest(3, 'M').index.tolist()

        if not picks:
            return {'BIL': 1.0}, 'NoPick'

        # Weighting
        if weight_mode == 'ivw':
            port = weight_ivw(picks, ind, d)
        elif weight_mode == 'softmax':
            # Build score map from combined or mom scores
            if combine_mode == 'combined':
                score_map = df['C'].to_dict()
            else:
                score_map = (df['M'].rank(pct=True) + df['Q'].rank(pct=True)).to_dict()
            port = weight_softmax(picks, ind, d, score_map=score_map,
                                  beta=weight_params.get('beta', 3.0))
        else:
            port = weight_equal(picks, ind, d)

        return port, f'On({len(picks)})'

    return strategy


# ===========================================================================
# Define all variants
# ===========================================================================

def build_all_variants():
    variants = {}

    # --- Baseline ---
    def baseline(d, ind):
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
        picks = list(set(df.nlargest(3,'M').index.tolist() + df.nlargest(3,'Q').index.tolist()))
        return {t: 1.0/len(picks) for t in picks}, 'On'

    variants['00.Baseline (Top3∪EW)'] = baseline

    # --- Method 1: Relative Threshold ---
    for alpha in [0.7, 0.8, 0.9]:
        for cm in ['union', 'combined']:
            name = f"01.RelThresh a={alpha} {cm[:3].upper()}"
            variants[name] = make_strategy(sel_relative_threshold,
                                            {'alpha': alpha, 'min_n': 2, 'max_n': 6},
                                            cm, 'ew')

    # --- Method 2: Z-Score ---
    for z in [0.0, 0.3, 0.5, 0.7, 1.0]:
        for cm in ['union', 'combined']:
            name = f"02.ZScore z={z} {cm[:3].upper()}"
            variants[name] = make_strategy(sel_zscore,
                                            {'z_cutoff': z, 'min_n': 1, 'max_n': 5},
                                            cm, 'ew')

    # --- Method 3: Largest Gap ---
    for min_n in [1, 2, 3]:
        for max_n in [4, 6]:
            for cm in ['union', 'combined']:
                name = f"03.Gap min={min_n} max={max_n} {cm[:3].upper()}"
                variants[name] = make_strategy(sel_largest_gap,
                                                {'min_n': min_n, 'max_n': max_n},
                                                cm, 'ew')

    # --- Method 4: K-Means ---
    for k in [2, 3]:
        for cm in ['union', 'combined']:
            name = f"04.KMeans k={k} {cm[:3].upper()}"
            variants[name] = make_strategy(sel_kmeans,
                                            {'k': k, 'min_n': 1, 'max_n': 6},
                                            cm, 'ew')

    # --- Method 5: Top-N + Tolerance ---
    for n in [2, 3, 4]:
        for eps in [0.02, 0.05, 0.10]:
            for cm in ['union', 'combined']:
                name = f"05.TopN+Tol n={n} e={eps} {cm[:3].upper()}"
                variants[name] = make_strategy(sel_topn_tolerance,
                                                {'n': n, 'epsilon': eps, 'max_n': 6},
                                                cm, 'ew')

    # --- Method 6: Softmax Cutoff ---
    for beta in [3.0, 5.0, 8.0]:
        for mw in [0.05, 0.10]:
            for cm in ['union', 'combined']:
                name = f"06.Softmax b={beta} mw={mw} {cm[:3].upper()}"
                variants[name] = make_strategy(sel_softmax,
                                                {'beta': beta, 'min_weight': mw, 'min_n': 2, 'max_n': 6},
                                                cm, 'ew')

    # --- Intersection variants (best params from each method) ---
    variants['07.ZScore z=0.5 INT'] = make_strategy(
        sel_zscore, {'z_cutoff': 0.5, 'min_n': 1, 'max_n': 5}, 'intersect', 'ew')
    variants['07.Gap min=2 max=6 INT'] = make_strategy(
        sel_largest_gap, {'min_n': 2, 'max_n': 6}, 'intersect', 'ew')
    variants['07.TopN+Tol n=3 e=0.05 INT'] = make_strategy(
        sel_topn_tolerance, {'n': 3, 'epsilon': 0.05, 'max_n': 6}, 'intersect', 'ew')

    # --- IVW weighting variants (top methods) ---
    for cm in ['union', 'combined']:
        variants[f'08.Gap min=2 max=6 {cm[:3].upper()} IVW'] = make_strategy(
            sel_largest_gap, {'min_n': 2, 'max_n': 6}, cm, 'ivw')
        variants[f'08.ZScore z=0.5 {cm[:3].upper()} IVW'] = make_strategy(
            sel_zscore, {'z_cutoff': 0.5, 'min_n': 1, 'max_n': 5}, cm, 'ivw')
        variants[f'08.TopN+Tol n=3 e=0.05 {cm[:3].upper()} IVW'] = make_strategy(
            sel_topn_tolerance, {'n': 3, 'epsilon': 0.05, 'max_n': 6}, cm, 'ivw')

    # --- Softmax weighting variants ---
    variants['09.Combined Softmax-W b=3'] = make_strategy(
        sel_zscore, {'z_cutoff': 0.3, 'min_n': 2, 'max_n': 5}, 'combined', 'softmax', {'beta': 3.0})
    variants['09.Combined Softmax-W b=5'] = make_strategy(
        sel_zscore, {'z_cutoff': 0.3, 'min_n': 2, 'max_n': 5}, 'combined', 'softmax', {'beta': 5.0})
    variants['09.Union Softmax-W b=3'] = make_strategy(
        sel_largest_gap, {'min_n': 2, 'max_n': 6}, 'union', 'softmax', {'beta': 3.0})

    # --- Gemini recommended combos ---
    # Combo 1: Defense (Intersect + IVW)
    variants['10.Combo1 Defense (INT+IVW)'] = make_strategy(
        sel_relative_threshold, {'alpha': 0.8, 'min_n': 1, 'max_n': 5}, 'intersect', 'ivw')
    # Combo 2: Balance (Z-Score Union + EW)
    variants['10.Combo2 Balance (Z∪EW)'] = make_strategy(
        sel_zscore, {'z_cutoff': 0.5, 'min_n': 2, 'max_n': 5}, 'union', 'ew')
    # Combo 3: Aggressive (Softmax combined)
    variants['10.Combo3 Aggro (SM+COM)'] = make_strategy(
        sel_softmax, {'beta': 5.0, 'min_weight': 0.10, 'min_n': 1, 'max_n': 4},
        'combined', 'softmax', {'beta': 5.0})

    return variants


# ===========================================================================
# Backtest engine
# ===========================================================================

def run_bt(data, ind, strat_func, start, end, capital=10000, tx=0.001):
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.isin(data.index)]
    month_ends = set(pd.date_range(start=start, end=end, freq='M'))

    cash = capital
    hold = {}
    hist = []
    rebals = 0
    prev_st = None

    for today in dates:
        row = data.loc[today]
        pv = cash + sum(u * (row.get(t, 0) if pd.notna(row.get(t, 0)) else 0) for t, u in hold.items())
        hist.append({'Date': today, 'Value': pv})

        tgt, st = strat_func(today, ind)
        flip = prev_st is not None and st != prev_st
        prev_st = st

        if today in month_ends or flip:
            rebals += 1
            amt = pv * (1 - tx)
            cash = amt
            hold = {}
            for t, w in tgt.items():
                p = row.get(t, 0) if pd.notna(row.get(t, 0)) else 0
                if p > 0:
                    a = amt * w
                    hold[t] = a / p
                    cash -= a

    df = pd.DataFrame(hist).set_index('Date')
    df.attrs['rebals'] = rebals
    return df


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
    if '--download' in sys.argv:
        download_data()
        return

    data = load_data()
    print("Pre-computing indicators...")
    ind = precompute(data)

    variants = build_all_variants()
    total = len(variants) * len(START_DATES)
    print(f"\nRunning {len(variants)} variants x {len(START_DATES)} periods = {total} backtests...\n")

    rows = []
    n = 0
    for name, func in variants.items():
        for start in START_DATES:
            n += 1
            try:
                res = run_bt(data, ind, func, start, END_DATE)
                m = metrics(res['Value'])
                rows.append({
                    'Strategy': name, 'Start': start,
                    'CAGR': m['cagr'], 'MDD': m['mdd'], 'Sharpe': m['sharpe'],
                    'Sortino': m['sortino'], 'Calmar': m['calmar'],
                    'Final': m['final'], 'Rebals': res.attrs.get('rebals', 0),
                })
            except Exception as e:
                rows.append({
                    'Strategy': name, 'Start': start,
                    'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'Calmar': 0,
                    'Final': 10000, 'Rebals': 0,
                })
            if n % 50 == 0:
                print(f"  Progress: {n}/{total} ({n*100//total}%)")

    df = pd.DataFrame(rows)

    # --- Stability Ranking ---
    stab = df.groupby('Strategy').agg({
        'CAGR': 'mean', 'MDD': 'mean', 'Sharpe': 'mean',
        'Sortino': 'mean', 'Calmar': 'mean', 'Rebals': 'mean'
    }).reset_index()
    stab['Score'] = stab['Sharpe'] * 0.4 + stab['Calmar'] * 0.3 + stab['CAGR'] * 2
    stab = stab.sort_values('Score', ascending=False)

    print(f"\n{'='*130}")
    print(f"  DYNAMIC SELECTION — STABILITY RANKING (Avg across {len(START_DATES)} start dates)")
    print(f"{'='*130}")
    print(f"{'#':>3} {'Strategy':<40} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'Reb':>5} {'Score':>7}")
    print(f"{'-'*130}")

    baseline_score = stab[stab['Strategy'].str.contains('Baseline')]['Score'].values
    bl = baseline_score[0] if len(baseline_score) > 0 else 0

    for i, (_, r) in enumerate(stab.iterrows(), 1):
        mark = ''
        if 'Baseline' in r['Strategy']:
            mark = ' <-- BASE'
        elif r['Score'] > bl:
            mark = ' *'
        print(f"{i:>3} {r['Strategy']:<40} {r['CAGR']:>7.1%} {r['MDD']:>7.1%} "
              f"{r['Sharpe']:>8.2f} {r['Sortino']:>8.2f} {r['Calmar']:>8.2f} "
              f"{r['Rebals']:>5.0f} {r['Score']:>7.2f}{mark}")

        if i == 20:
            print(f"  ... ({len(stab) - 20} more below baseline cutoff) ...")
            # Still print baseline if not shown
            remaining = stab.iloc[20:]
            bl_rows = remaining[remaining['Strategy'].str.contains('Baseline')]
            for _, br in bl_rows.iterrows():
                bl_idx = stab.index.get_loc(br.name) + 1
                print(f"{bl_idx:>3} {br['Strategy']:<40} {br['CAGR']:>7.1%} {br['MDD']:>7.1%} "
                      f"{br['Sharpe']:>8.2f} {br['Sortino']:>8.2f} {br['Calmar']:>8.2f} "
                      f"{br['Rebals']:>5.0f} {br['Score']:>7.2f} <-- BASE")
            break

    print(f"{'='*130}")

    # --- Top 10 detail by period ---
    top10_names = stab.head(10)['Strategy'].tolist()
    if 'Baseline' not in ' '.join(top10_names):
        top10_names.append('00.Baseline (Top3∪EW)')

    print(f"\n{'='*130}")
    print(f"  TOP 10 + BASELINE — Detail by Period")
    print(f"{'='*130}")

    for start in START_DATES:
        period = df[df['Start'] == start]
        subset = period[period['Strategy'].isin(top10_names)].sort_values('Sharpe', ascending=False)

        print(f"\n  --- {start} ~ {END_DATE} ---")
        print(f"  {'Strategy':<40} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8}")
        for _, r in subset.iterrows():
            mark = ' <BASE' if 'Baseline' in r['Strategy'] else ''
            print(f"  {r['Strategy']:<40} {r['CAGR']:>7.1%} {r['MDD']:>7.1%} "
                  f"{r['Sharpe']:>8.2f} {r['Sortino']:>8.2f} {r['Calmar']:>8.2f}{mark}")

    # --- Method category summary ---
    print(f"\n{'='*130}")
    print(f"  METHOD CATEGORY SUMMARY")
    print(f"{'='*130}")

    categories = {
        '01.RelThresh': 'Relative Threshold',
        '02.ZScore': 'Z-Score',
        '03.Gap': 'Largest Gap',
        '04.KMeans': 'K-Means',
        '05.TopN': 'Top-N + Tolerance',
        '06.Softmax': 'Softmax Cutoff',
        '07.': 'Intersection',
        '08.': 'IVW Weighting',
        '09.': 'Softmax Weighting',
        '10.': 'Gemini Combos',
    }

    cat_rows = []
    for prefix, cat_name in categories.items():
        cat_strats = stab[stab['Strategy'].str.startswith(prefix)]
        if cat_strats.empty: continue
        best = cat_strats.iloc[0]
        cat_rows.append({
            'Category': cat_name,
            'Best Variant': best['Strategy'],
            'CAGR': best['CAGR'], 'MDD': best['MDD'],
            'Sharpe': best['Sharpe'], 'Score': best['Score'],
        })

    print(f"  {'Category':<22} {'Best Variant':<40} {'CAGR':>7} {'MDD':>7} {'Sharpe':>7} {'Score':>7}")
    print(f"  {'-'*110}")
    for r in cat_rows:
        print(f"  {r['Category']:<22} {r['Best Variant']:<40} {r['CAGR']:>6.1%} {r['MDD']:>6.1%} "
              f"{r['Sharpe']:>7.2f} {r['Score']:>7.2f}")

    print(f"\n  Baseline Score: {bl:.2f}")
    beat_count = len(stab[stab['Score'] > bl])
    print(f"  Variants beating baseline: {beat_count}/{len(stab)}")
    print(f"{'='*130}")


if __name__ == '__main__':
    main()
