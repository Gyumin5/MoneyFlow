"""
ETF Strategy Backtest — 12 strategies compared across multiple periods
=====================================================================
Strategies: Cap Defend Stock, HAA, BAA, VAA, DAA, GTAA-5, GEM,
            All Weather, Trend Following, Adaptive Momentum, Dual Canary

Usage:
    python3 strategies/cap_defend/backtest_etf.py
    python3 strategies/cap_defend/backtest_etf.py --download
"""

import os, sys, json, warnings
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

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def download_etf_data():
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
        if os.path.exists(fp):
            continue
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
        except:
            pass
    print("Download complete.")


def load_etf_data():
    buffer_start = '2017-01-01'
    data_dict = {}
    for ticker in ALL_ETF_TICKERS:
        fp = os.path.join(DATA_DIR, f"{ticker}.csv")
        if not os.path.exists(fp):
            continue
        try:
            df = pd.read_csv(fp, parse_dates=['Date'])
            df = df.drop_duplicates(subset=['Date'], keep='first').set_index('Date')
            col = 'Adj Close' if 'Adj Close' in df else ('Adj_Close' if 'Adj_Close' in df else 'Close')
            if col in df:
                data_dict[ticker] = df[col]
        except:
            pass

    idx = pd.date_range(start=buffer_start, end=END_DATE, freq='D')
    data = pd.DataFrame(data_dict).reindex(idx).ffill()
    print(f"Loaded {len(data_dict)} ETFs, range: {data.index[0].date()} ~ {data.index[-1].date()}")
    return data


# ---------------------------------------------------------------------------
# Pre-compute indicators
# ---------------------------------------------------------------------------

def precompute(data):
    ind = {}
    for col in data.columns:
        p = data[col]
        dr = p.pct_change()

        d = pd.DataFrame({'price': p})
        d['sma200'] = p.rolling(200).mean()
        d['sma210'] = p.rolling(210).mean()

        d['mom21']  = p / p.shift(21) - 1
        d['mom63']  = p / p.shift(63) - 1
        d['mom126'] = p / p.shift(126) - 1
        d['mom252'] = p / p.shift(252) - 1

        # 13612W
        d['m13612w'] = (12*(p/p.shift(21)-1) + 4*(p/p.shift(63)-1)
                        + 2*(p/p.shift(126)-1) + (p/p.shift(252)-1))

        # Average momentum (HAA style)
        d['mom_avg'] = (d['mom21'] + d['mom63'] + d['mom126'] + d['mom252']) / 4

        # Weighted momentum (Cap Defend style)
        d['mom_w'] = 0.5*d['mom63'] + 0.3*d['mom126'] + 0.2*d['mom252']

        # Volatility
        d['vol90'] = dr.rolling(90).std()

        # Sharpe 126d
        rm = dr.rolling(126).mean()
        rs = dr.rolling(126).std()
        d['sharpe126'] = (rm / rs.replace(0, np.nan)) * np.sqrt(252)

        ind[col] = d
    return ind


def v(ind, ticker, col, date):
    """Safe indicator lookup."""
    if ticker not in ind:
        return np.nan
    try:
        return ind[ticker][col].loc[date]
    except:
        return np.nan


def has(ind, ticker, date, min_col='mom252'):
    """Check if ticker has valid data at date."""
    val = v(ind, ticker, min_col, date)
    return pd.notna(val)


# ---------------------------------------------------------------------------
# 12 Strategy functions  →  (portfolio_dict, status_str)
# ---------------------------------------------------------------------------

def s01_cap_defend(d, ind):
    """Current Cap Defend stock strategy."""
    canary = ['VT', 'EEM']
    off = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','QUAL','MTUM','IQLT','IMTM']
    defe = ['IEF','BIL','BNDX','GLD','PDBC']

    risk_on = all(
        has(ind, c, d) and v(ind, c, 'price', d) > v(ind, c, 'sma200', d)
        for c in canary
    )

    if risk_on:
        rows = [(t, v(ind,t,'mom_w',d), v(ind,t,'sharpe126',d))
                for t in off if has(ind,t,d)]
        rows = [(t,m,q) for t,m,q in rows if pd.notna(m) and pd.notna(q)]
        if not rows:
            return {'BIL': 1.0}, 'NoData'
        df = pd.DataFrame(rows, columns=['T','M','Q']).set_index('T')
        picks = list(set(df.nlargest(3,'M').index.tolist() + df.nlargest(3,'Q').index.tolist()))
        return {t: 1.0/len(picks) for t in picks}, 'On'
    else:
        best_t, best_r = 'BIL', -999
        for t in defe:
            r = v(ind, t, 'mom126', d)
            if pd.notna(r) and r > best_r:
                best_r, best_t = r, t
        if best_r < 0:
            return {'BIL': 1.0}, 'Cash'
        return {best_t: 1.0}, 'Off'


def s02_haa(d, ind):
    """HAA — TIP canary, top 4 of 8 risky, BIL/IEF defense."""
    risky = ['SPY','VEA','VWO','AGG','VNQ','GLD','DBC','TLT']
    defe = ['BIL', 'IEF']

    tip_mom = v(ind, 'TIP', 'mom_avg', d)
    canary_bad = pd.isna(tip_mom) or tip_mom < 0

    if canary_bad:
        best_t = max(defe, key=lambda t: v(ind,t,'mom21',d) if pd.notna(v(ind,t,'mom21',d)) else -999)
        return {best_t: 1.0}, 'Off'

    scores = [(t, v(ind,t,'mom_avg',d)) for t in risky if has(ind,t,d)]
    scores = [(t,m) for t,m in scores if pd.notna(m)]
    scores.sort(key=lambda x: x[1], reverse=True)

    top4 = [(t,m) for t,m in scores[:4] if m > 0]
    n_def = 4 - len(top4)
    port = {t: 0.25 for t,_ in top4}
    if n_def > 0:
        best_d = max(defe, key=lambda t: v(ind,t,'mom21',d) if pd.notna(v(ind,t,'mom21',d)) else -999)
        port[best_d] = port.get(best_d, 0) + n_def * 0.25
    return port if port else {'BIL': 1.0}, 'On' if top4 else 'Mixed'


def s03_baa_bal(d, ind):
    """BAA Balanced — 4-canary, top 6 offensive, top 3 defensive."""
    canary = ['SPY','EFA','EEM','AGG']
    off = ['SPY','QQQ','EFA','EEM','VNQ','GLD','DBC','TLT','VEA','VWO','AGG','HYG']
    defe = ['TIP','DBC','BIL','IEF','TLT','LQD','AGG']

    bad = any(pd.isna(v(ind,c,'m13612w',d)) or v(ind,c,'m13612w',d) < 0 for c in canary)

    pool = defe if bad else off
    top_n = 3 if bad else 6
    scores = [(t, v(ind,t,'m13612w',d)) for t in pool if has(ind,t,d)]
    scores = [(t,m) for t,m in scores if pd.notna(m)]
    scores.sort(key=lambda x: x[1], reverse=True)
    picks = [t for t,_ in scores[:top_n]]
    if not picks:
        return {'BIL': 1.0}, 'BIL'
    return {t: 1.0/len(picks) for t in picks}, 'Def' if bad else 'Off'


def s04_baa_agg(d, ind):
    """BAA Aggressive — same canary, top 1 / top 1."""
    canary = ['SPY','EFA','EEM','AGG']
    off = ['QQQ','EFA','EEM','AGG','VNQ','GLD','DBC','TLT','VEA','VWO','HYG','SPY']
    defe = ['TIP','DBC','BIL','IEF','TLT','LQD','AGG']

    bad = any(pd.isna(v(ind,c,'m13612w',d)) or v(ind,c,'m13612w',d) < 0 for c in canary)
    pool = defe if bad else off

    scores = [(t, v(ind,t,'m13612w',d)) for t in pool if has(ind,t,d)]
    scores = [(t,m) for t,m in scores if pd.notna(m)]
    scores.sort(key=lambda x: x[1], reverse=True)
    if scores:
        return {scores[0][0]: 1.0}, 'Def' if bad else 'Off'
    return {'BIL': 1.0}, 'BIL'


def s05_vaa_g4(d, ind):
    """VAA G4 — all 4 must be positive → top 1, else defensive top 1."""
    off = ['SPY','VEA','VWO','BND']
    defe = ['SHY','IEF','LQD']

    scores = [(t, v(ind,t,'m13612w',d)) for t in off if has(ind,t,d)]
    scores = [(t,m) for t,m in scores if pd.notna(m)]
    all_pos = len(scores) == len(off) and all(m > 0 for _,m in scores)

    if all_pos:
        scores.sort(key=lambda x: x[1], reverse=True)
        return {scores[0][0]: 1.0}, f'Off({scores[0][0]})'

    ds = [(t, v(ind,t,'m13612w',d)) for t in defe if has(ind,t,d)]
    ds = [(t,m) for t,m in ds if pd.notna(m)]
    ds.sort(key=lambda x: x[1], reverse=True)
    if ds:
        return {ds[0][0]: 1.0}, f'Def({ds[0][0]})'
    return {'BIL': 1.0}, 'BIL'


def s06_daa(d, ind):
    """DAA — VWO+BND canary, proportional defense."""
    canary = ['VWO', 'BND']
    off = ['SPY','VEA','VWO','BND','VNQ','GLD','DBC']
    defe = ['SHY','IEF','LQD']

    bad = sum(1 for c in canary
              if not has(ind,c,d) or pd.isna(v(ind,c,'m13612w',d)) or v(ind,c,'m13612w',d) < 0)
    cash_frac = bad / len(canary)
    risky_frac = 1 - cash_frac

    port = {}
    if risky_frac > 0:
        os_ = [(t, v(ind,t,'m13612w',d)) for t in off if has(ind,t,d)]
        os_ = [(t,m) for t,m in os_ if pd.notna(m)]
        os_.sort(key=lambda x: x[1], reverse=True)
        n = max(1, len(os_) // 2)
        picks = os_[:n]
        for t,_ in picks:
            port[t] = risky_frac / len(picks)

    if cash_frac > 0:
        ds = [(t, v(ind,t,'m13612w',d)) for t in defe if has(ind,t,d)]
        ds = [(t,m) for t,m in ds if pd.notna(m)]
        ds.sort(key=lambda x: x[1], reverse=True)
        dt = ds[0][0] if ds else 'BIL'
        port[dt] = port.get(dt, 0) + cash_frac

    return port if port else {'BIL': 1.0}, f'D{bad}'


def s07_gtaa5(d, ind):
    """GTAA-5 — each asset vs 10-month SMA, cash if below."""
    assets = ['SPY','EFA','BND','VNQ','DBC']
    port = {}
    for t in assets:
        w = 1.0 / len(assets)
        price = v(ind, t, 'price', d)
        sma = v(ind, t, 'sma210', d)
        if pd.notna(price) and pd.notna(sma) and price > sma:
            port[t] = w
        else:
            port['BIL'] = port.get('BIL', 0) + w
    return port, f'{sum(1 for t in port if t!="BIL")}/5'


def s08_gem(d, ind):
    """GEM — SPY vs EFA 12m return, absolute check, AGG defense."""
    spy_r = v(ind, 'SPY', 'mom252', d)
    efa_r = v(ind, 'EFA', 'mom252', d)
    if pd.isna(spy_r) or pd.isna(efa_r):
        return {'BIL': 1.0}, 'NoData'

    if spy_r > efa_r and spy_r > 0:
        return {'SPY': 1.0}, 'SPY'
    elif efa_r > spy_r and efa_r > 0:
        return {'EFA': 1.0}, 'EFA'
    else:
        return {'AGG': 1.0}, 'AGG'


def s09_all_weather(d, ind):
    """All Weather — static allocation."""
    alloc = {'SPY':0.30, 'TLT':0.40, 'IEF':0.15, 'GLD':0.075, 'DBC':0.075}
    port = {}
    for t, w in alloc.items():
        if t in ind and pd.notna(v(ind,t,'price',d)):
            port[t] = w
        else:
            port['BIL'] = port.get('BIL', 0) + w
    return port, 'Static'


def s10_trend(d, ind):
    """Trend Following — multi-asset vs 200d SMA, BIL if below."""
    assets = ['SPY','EFA','EEM','TLT','GLD','DBC','VNQ']
    port = {}
    for t in assets:
        w = 1.0 / len(assets)
        price = v(ind, t, 'price', d)
        sma = v(ind, t, 'sma200', d)
        if pd.notna(price) and pd.notna(sma) and price > sma:
            port[t] = w
        else:
            port['BIL'] = port.get('BIL', 0) + w
    return port, f'{sum(1 for t in port if t!="BIL")}/7'


def s11_adaptive(d, ind):
    """Adaptive Momentum — top 5 by avg momentum, positive only, IVW."""
    universe = ['SPY','QQQ','EFA','EEM','VNQ','GLD','TLT','IEF','DBC','VEA']
    scores = [(t, v(ind,t,'mom_avg',d)) for t in universe if has(ind,t,d)]
    scores = [(t,m) for t,m in scores if pd.notna(m) and m > 0]
    scores.sort(key=lambda x: x[1], reverse=True)
    top = scores[:5]

    if not top:
        return {'BIL': 1.0}, 'NoPos'

    vols = {t: max(v(ind,t,'vol90',d), 0.001) if pd.notna(v(ind,t,'vol90',d)) else 0.001
            for t,_ in top}
    inv = {t: 1.0/vol for t, vol in vols.items()}
    tot = sum(inv.values())
    return {t: val/tot for t, val in inv.items()}, f'Top{len(top)}'


def s12_dual_canary(d, ind):
    """Dual Canary — TIP+VWO, top 6 offensive 13612W + IVW, top 3 defensive."""
    off = ['SPY','QQQ','EFA','EEM','VT','VEA','VNQ','GLD','DBC','VGK','QUAL','MTUM']
    defe = ['BIL','IEF','TLT','TIP','GLD','DBMF']

    bad = 0
    for c in ['TIP', 'VWO']:
        m = v(ind, c, 'mom_avg', d)
        if pd.isna(m) or m < 0:
            bad += 1

    if bad == 0:  # 100% offensive
        scores = [(t, v(ind,t,'m13612w',d)) for t in off if has(ind,t,d)]
        scores = [(t,m) for t,m in scores if pd.notna(m)]
        scores.sort(key=lambda x: x[1], reverse=True)
        picks = [t for t,_ in scores[:6]]
        if not picks:
            return {'BIL': 1.0}, 'NoData'
        vols = {t: max(v(ind,t,'vol90',d), 0.001) if pd.notna(v(ind,t,'vol90',d)) else 0.001
                for t in picks}
        inv = {t: 1.0/vol for t, vol in vols.items()}
        tot = sum(inv.values())
        return {t: val/tot for t, val in inv.items()}, 'On'

    elif bad == 1:  # 50/50
        os_ = [(t, v(ind,t,'m13612w',d)) for t in off if has(ind,t,d)]
        os_ = [(t,m) for t,m in os_ if pd.notna(m)]
        os_.sort(key=lambda x: x[1], reverse=True)
        op = [t for t,_ in os_[:3]]

        ds = [(t, v(ind,t,'m13612w',d) if has(ind,t,d) else v(ind,t,'mom21',d))
              for t in defe]
        ds = [(t,m) for t,m in ds if pd.notna(m)]
        ds.sort(key=lambda x: x[1], reverse=True)
        dp = [t for t,_ in ds[:3]]

        port = {}
        for t in op: port[t] = 0.5/max(len(op),1)
        for t in dp: port[t] = port.get(t,0) + 0.5/max(len(dp),1)
        return port if port else {'BIL':1.0}, 'Mix'

    else:  # 100% defensive
        ds = [(t, v(ind,t,'m13612w',d) if has(ind,t,d) else v(ind,t,'mom21',d))
              for t in defe]
        ds = [(t,m) for t,m in ds if pd.notna(m)]
        ds.sort(key=lambda x: x[1], reverse=True)
        picks = [t for t,_ in ds[:3]]
        if not picks:
            return {'BIL': 1.0}, 'BIL'
        return {t: 1.0/len(picks) for t in picks}, 'Off'


# ---------------------------------------------------------------------------
# Backtest engine
# ---------------------------------------------------------------------------

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
        pv = cash + sum(u * (row.get(t,0) if pd.notna(row.get(t,0)) else 0) for t,u in hold.items())
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


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def metrics(vals):
    if len(vals) < 2: return {'cagr':0,'mdd':0,'sharpe':0,'sortino':0,'wr':0,'final':0}
    days = (vals.index[-1] - vals.index[0]).days
    if days <= 0: return {'cagr':0,'mdd':0,'sharpe':0,'sortino':0,'wr':0,'final':0}
    cagr = (vals.iloc[-1]/vals.iloc[0])**(365.25/days) - 1
    mdd = (vals/vals.cummax()-1).min()
    dr = vals.pct_change().dropna()
    sharpe = (dr.mean()/dr.std())*np.sqrt(252) if dr.std()>0 else 0
    ds = dr[dr<0]
    sortino = (dr.mean()/ds.std())*np.sqrt(252) if len(ds)>0 and ds.std()>0 else 0
    wr = (dr>0).sum()/len(dr) if len(dr)>0 else 0
    return {'cagr':cagr,'mdd':mdd,'sharpe':sharpe,'sortino':sortino,'wr':wr,'final':vals.iloc[-1]}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

STRATEGIES = {
    'Cap Defend Stock':  s01_cap_defend,
    'HAA (TIP Canary)':  s02_haa,
    'BAA Balanced':      s03_baa_bal,
    'BAA Aggressive':    s04_baa_agg,
    'VAA G4':            s05_vaa_g4,
    'DAA':               s06_daa,
    'GTAA-5 (Faber)':    s07_gtaa5,
    'GEM (Dual Mom)':    s08_gem,
    'All Weather':       s09_all_weather,
    'Trend Following':   s10_trend,
    'Adaptive Momentum': s11_adaptive,
    'Dual Canary':       s12_dual_canary,
}

def main():
    if '--download' in sys.argv:
        download_etf_data()
        return

    data = load_etf_data()
    print("Pre-computing indicators...")
    ind = precompute(data)

    total = len(STRATEGIES) * len(START_DATES)
    print(f"\nRunning {len(STRATEGIES)} strategies x {len(START_DATES)} periods = {total} backtests...\n")

    rows = []
    n = 0
    for name, func in STRATEGIES.items():
        for start in START_DATES:
            n += 1
            res = run_bt(data, ind, func, start, END_DATE)
            m = metrics(res['Value'])
            rows.append({
                'Strategy': name, 'Start': start, 'Final': m['final'],
                'CAGR': m['cagr'], 'MDD': m['mdd'], 'Sharpe': m['sharpe'],
                'Sortino': m['sortino'], 'WinRate': m['wr'],
                'Rebals': res.attrs.get('rebals', 0),
            })
            if n % 10 == 0:
                print(f"  Progress: {n}/{total}")

    df = pd.DataFrame(rows)

    # Per-period tables
    for start in START_DATES:
        period = df[df['Start'] == start].sort_values('Sharpe', ascending=False)
        bm_spy = metrics(data['SPY'].loc[start:END_DATE].dropna())
        bm_btc_f = os.path.join(DATA_DIR, 'BTC-USD.csv')
        bm_btc = {'cagr':0,'mdd':0,'sharpe':0}

        print(f"\n{'='*100}")
        print(f"  ETF STRATEGIES — START: {start} ~ {END_DATE}")
        print(f"{'='*100}")
        print(f"{'#':>3} {'Strategy':<24} {'Final($)':>11} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>8} {'WR':>7} {'Reb':>5}")
        print(f"{'-'*100}")

        for i, (_, r) in enumerate(period.iterrows(), 1):
            mark = ' ***' if i <= 3 else ''
            print(f"{i:>3} {r['Strategy']:<24} {r['Final']:>11,.0f} {r['CAGR']:>7.1%} {r['MDD']:>7.1%} "
                  f"{r['Sharpe']:>8.2f} {r['Sortino']:>8.2f} {r['WinRate']:>6.1%} {r['Rebals']:>5}{mark}")

        print(f"{'-'*100}")
        print(f"    {'SPY B&H':<24} {'':>11} {bm_spy['cagr']:>7.1%} {bm_spy['mdd']:>7.1%} {bm_spy['sharpe']:>8.2f}")

    # Stability ranking
    stab = df.groupby('Strategy').agg({
        'CAGR':'mean','MDD':'mean','Sharpe':'mean','Sortino':'mean','Rebals':'mean'
    }).reset_index()
    stab['Score'] = stab['Sharpe']*0.5 + (1+stab['MDD'])*2 + stab['CAGR']
    stab = stab.sort_values('Score', ascending=False)

    print(f"\n{'='*100}")
    print(f"  STABILITY RANKING — Average across all start dates")
    print(f"{'='*100}")
    print(f"{'#':>3} {'Strategy':<24} {'AvgCAGR':>9} {'AvgMDD':>9} {'AvgSharpe':>10} {'AvgSortino':>11} {'Score':>7}")
    print(f"{'-'*100}")
    for i, (_, r) in enumerate(stab.iterrows(), 1):
        mark = ' <-- TOP' if i <= 5 else ''
        print(f"{i:>3} {r['Strategy']:<24} {r['CAGR']:>8.1%} {r['MDD']:>8.1%} {r['Sharpe']:>10.2f} "
              f"{r['Sortino']:>11.2f} {r['Score']:>7.2f}{mark}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
