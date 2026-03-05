"""
Crypto Strategy Backtest — 12 strategies compared across multiple periods
=========================================================================
Pre-computed indicators for speed (~5min total vs ~2hrs naive)

Usage:
    python3 strategies/cap_defend/backtest_crypto.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings('ignore')

DATA_DIR = './data'
UNIVERSE_FILE = './data/historical_universe.json'

STABLECOINS = {'USDT','USDC','BUSD','DAI','UST','TUSD','PAX','GUSD',
               'FRAX','LUSD','MIM','USDN','FDUSD','USDS','PYUSD','USDE'}

START_DATES = ['2019-01-01','2020-01-01','2021-01-01','2022-01-01','2023-01-01']
END_DATE = '2025-12-31'


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_universe():
    if os.path.exists(UNIVERSE_FILE):
        with open(UNIVERSE_FILE) as f:
            return json.load(f)
    return {}


def collect_tickers(hist_universe):
    tickers = {'BTC-USD'}
    for symbols in hist_universe.values():
        for s in symbols:
            t = s if s.endswith('-USD') else f"{s}-USD"
            sym = t.replace('-USD', '')
            if sym not in STABLECOINS:
                tickers.add(t)
    return tickers


def load_data(tickers):
    buffer_start = '2017-01-01'
    data_dict = {}
    for ticker in tickers:
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
    print(f"Loaded {len(data_dict)} tickers, range: {data.index[0].date()} ~ {data.index[-1].date()}")
    return data


# ---------------------------------------------------------------------------
# Pre-compute ALL indicators (vectorized, fast)
# ---------------------------------------------------------------------------

def precompute(data):
    """Pre-compute indicators for all tickers. Returns dict of DataFrames."""
    ind = {}
    for col in data.columns:
        p = data[col]
        dr = p.pct_change()

        d = pd.DataFrame({'price': p})
        d['sma30'] = p.rolling(30).mean()
        d['sma50'] = p.rolling(50).mean()
        d['sma70'] = p.rolling(70).mean()

        d['mom7']   = p / p.shift(7) - 1
        d['mom21']  = p / p.shift(21) - 1
        d['mom42']  = p / p.shift(42) - 1
        d['mom63']  = p / p.shift(63) - 1
        d['mom126'] = p / p.shift(126) - 1
        d['mom252'] = p / p.shift(252) - 1

        d['vol60']  = dr.rolling(60).std()
        d['vol90']  = dr.rolling(90).std()

        # Rolling Sharpe (vectorized)
        for w in [126, 252]:
            rm = dr.rolling(w).mean()
            rs = dr.rolling(w).std()
            d[f'sharpe{w}'] = (rm / rs.replace(0, np.nan)) * np.sqrt(252)

        # 13612W momentum
        d['m13612w'] = (12*(p/p.shift(21)-1) + 4*(p/p.shift(63)-1)
                        + 2*(p/p.shift(126)-1) + (p/p.shift(252)-1))

        # Risk-adjusted momentum: return / volatility
        d['ram126'] = d['mom126'] / d['vol90'].replace(0, np.nan)

        # Combined mom*invvol
        d['mom_invvol'] = d['mom126'] / d['vol90'].replace(0, np.nan)

        # Health checks (pre-computed booleans)
        d['h_base']   = (p > d['sma30']) & (d['mom21'] > 0) & (d['vol90'] <= 0.10)
        d['h_loose']  = (p > d['sma30']) & (d['mom21'] > 0) & (d['vol90'] <= 0.15)
        d['h_strict'] = (p > d['sma50']) & (d['mom42'] > 0) & (d['vol60'] <= 0.08)

        ind[col] = d
    return ind


def get_universe(date, hist_universe, top_n=50):
    key = date.strftime("%Y-%m") + "-01"
    symbols = hist_universe.get(key, [])
    if not symbols:
        avail = sorted([k for k in hist_universe if k <= key], reverse=True)
        if avail:
            symbols = hist_universe[avail[0]]
    final = []
    for s in symbols:
        t = s if s.endswith('-USD') else f"{s}-USD"
        if t.replace('-USD','') not in STABLECOINS:
            final.append(t)
            if len(final) >= top_n:
                break
    return final


def v(ind, ticker, col, date):
    if ticker not in ind: return np.nan
    try: return ind[ticker][col].loc[date]
    except: return np.nan


def ivw(ind, picks, date, vol_col='vol90'):
    """Inverse volatility weighting."""
    vols = {}
    for t in picks:
        vol = v(ind, t, vol_col, date)
        vols[t] = vol if pd.notna(vol) and vol > 0 else 0.001
    inv = {t: 1.0/vol for t, vol in vols.items()}
    tot = sum(inv.values())
    return {t: val/tot for t, val in inv.items()} if tot > 0 else {t: 1.0/len(picks) for t in picks}


def ew(picks):
    """Equal weighting."""
    return {t: 1.0/len(picks) for t in picks} if picks else {}


# ---------------------------------------------------------------------------
# 12 Crypto Strategy functions → (portfolio_dict, status_str)
# ---------------------------------------------------------------------------

def c01_baseline(d, ind, hu):
    """Current: BTC MA50, health_base, Sharpe(126+252), IVW Top5."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c02_no_health(d, ind, hu):
    """No health filter — pure Sharpe scoring."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    scores = {}
    for c in univ:
        if c not in ind: continue
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c03_risk_adj_mom(d, ind, hu):
    """Score = Return(126) / Vol(90) instead of Sharpe."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        ram = v(ind,c,'ram126',d)
        if pd.notna(ram): scores[c] = ram
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c04_score_13612w(d, ind, hu):
    """13612W momentum scoring (from Keller papers)."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        m = v(ind,c,'m13612w',d)
        if pd.notna(m): scores[c] = m
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c05_equal_weight(d, ind, hu):
    """Same as baseline but equal weight instead of IVW."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ew(top), 'On'


def c06_aggressive(d, ind, hu):
    """VolCap 15%, Top 10, BTC MA30 canary."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma30',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_loose',d)) and v(ind,c,'h_loose',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:10]
    return ivw(ind, top, d), 'On'


def c07_conservative(d, ind, hu):
    """VolCap 6%, Top 3, BTC MA70 canary."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma70',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_strict',d)) and v(ind,c,'h_strict',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:3]
    return ivw(ind, top, d), 'On'


def c08_dual_tf(d, ind, hu):
    """Dual Timeframe: short(1m+3m) + long(6m+12m) blend."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        m21  = v(ind,c,'mom21',d)
        m63  = v(ind,c,'mom63',d)
        m126 = v(ind,c,'mom126',d)
        m252 = v(ind,c,'mom252',d)
        if all(pd.notna(x) for x in [m21,m63,m126,m252]):
            short = m21 + m63
            long_ = m126 + m252
            scores[c] = 0.5*short + 0.5*long_
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c09_mean_revert(d, ind, hu):
    """Mean reversion: buy short-term dips in long-term uptrends."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    candidates = []
    for c in univ:
        if c not in ind: continue
        m7   = v(ind,c,'mom7',d)
        m126 = v(ind,c,'mom126',d)
        vol  = v(ind,c,'vol90',d)
        if pd.isna(m7) or pd.isna(m126) or pd.isna(vol): continue
        if vol > 0.10: continue
        # Long-term uptrend + short-term dip
        if m126 > 0 and m7 < 0:
            candidates.append((c, -m7))  # bigger dip = higher score

    if not candidates:
        # Fallback to baseline when no mean-reversion opportunities
        return c01_baseline(d, ind, hu)

    candidates.sort(key=lambda x: x[1], reverse=True)
    top = [t for t,_ in candidates[:5]]
    return ivw(ind, top, d), 'MR'


def c10_no_canary(d, ind, hu):
    """No BTC canary — always invest if health OK."""
    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c11_strict_health(d, ind, hu):
    """Strict health: SMA50, Mom42, Vol60<=8%."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_strict',d)) and v(ind,c,'h_strict',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        s1, s2 = v(ind,c,'sharpe126',d), v(ind,c,'sharpe252',d)
        if pd.notna(s1) and pd.notna(s2): scores[c] = s1+s2
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


def c12_mom_vol_combo(d, ind, hu):
    """Score = Momentum(126) * InverseVol = momentum quality."""
    bp = v(ind,'BTC-USD','price',d); bs = v(ind,'BTC-USD','sma50',d)
    if pd.isna(bp) or pd.isna(bs) or bp <= bs:
        return {}, 'Off'

    univ = get_universe(d, hu)
    healthy = [c for c in univ if c in ind
               and pd.notna(v(ind,c,'h_base',d)) and v(ind,c,'h_base',d)]
    if not healthy: return {}, 'NoH'

    scores = {}
    for c in healthy:
        mv = v(ind,c,'mom_invvol',d)
        if pd.notna(mv): scores[c] = mv
    if not scores: return {}, 'NoS'

    top = sorted(scores, key=scores.get, reverse=True)[:5]
    return ivw(ind, top, d), 'On'


# ---------------------------------------------------------------------------
# Backtest engine (with health ejection)
# ---------------------------------------------------------------------------

def run_bt(data, ind, strat_func, hu, start, end, capital=10000, tx=0.001,
           health_col='h_base', turnover_th=0.30):
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.isin(data.index)]
    month_ends = set(pd.date_range(start=start, end=end, freq='M'))

    cash = capital
    hold = {}
    hist = []
    rebals = 0

    for today in dates:
        row = data.loc[today]
        pv = cash + sum(u * (row.get(t,0) if pd.notna(row.get(t,0)) else 0) for t,u in hold.items())
        hist.append({'Date': today, 'Value': pv})

        tgt, st = strat_func(today, ind, hu)

        # Monthly trigger
        is_monthly = today in month_ends

        # Turnover trigger
        is_turn = False
        if hold and pv > 0:
            cur_w = {}
            for t, u in hold.items():
                p = row.get(t, 0) if pd.notna(row.get(t, 0)) else 0
                if p > 0: cur_w[t] = (u*p)/pv
            all_t = set(cur_w) | set(tgt)
            turn = sum(abs(tgt.get(t,0)-cur_w.get(t,0)) for t in all_t)/2
            is_turn = turn > turnover_th

        # Health ejection trigger
        is_eject = False
        if hold and health_col:
            for held in hold:
                if held in ind:
                    hv = v(ind, held, health_col, today)
                    if pd.notna(hv) and not hv:
                        is_eject = True
                        break

        if is_monthly or is_turn or is_eject:
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
    'Baseline (현재)':      (c01_baseline,    'h_base', 0.30),
    'No Health Filter':    (c02_no_health,   None,     0.30),
    'RiskAdj Momentum':    (c03_risk_adj_mom,'h_base', 0.30),
    '13612W Score':        (c04_score_13612w,'h_base', 0.30),
    'Equal Weight':        (c05_equal_weight,'h_base', 0.30),
    'Aggressive':          (c06_aggressive,  'h_loose',0.30),
    'Conservative':        (c07_conservative,'h_strict',0.30),
    'Dual Timeframe':      (c08_dual_tf,     'h_base', 0.30),
    'Mean Reversion':      (c09_mean_revert, 'h_base', 0.30),
    'No Canary':           (c10_no_canary,   'h_base', 0.30),
    'Strict Health':       (c11_strict_health,'h_strict',0.30),
    'Mom*InvVol Score':    (c12_mom_vol_combo,'h_base', 0.30),
}


def main():
    hu = load_universe()
    if not hu:
        print("ERROR: No historical_universe.json found")
        sys.exit(1)

    tickers = collect_tickers(hu)
    print(f"Total coin tickers: {len(tickers)}")
    data = load_data(tickers)

    print("Pre-computing indicators (this may take 30-60s)...")
    ind = precompute(data)
    print(f"Pre-computed indicators for {len(ind)} tickers.")

    total = len(STRATEGIES) * len(START_DATES)
    print(f"\nRunning {len(STRATEGIES)} strategies x {len(START_DATES)} periods = {total} backtests...\n")

    rows = []
    n = 0
    for name, (func, hcol, to_th) in STRATEGIES.items():
        for start in START_DATES:
            n += 1
            res = run_bt(data, ind, func, hu, start, END_DATE,
                         health_col=hcol, turnover_th=to_th)
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

        bm_btc = metrics(data['BTC-USD'].loc[start:END_DATE].dropna()) if 'BTC-USD' in data else {}

        print(f"\n{'='*100}")
        print(f"  CRYPTO STRATEGIES — START: {start} ~ {END_DATE}")
        print(f"{'='*100}")
        print(f"{'#':>3} {'Strategy':<22} {'Final($)':>11} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>8} {'WR':>7} {'Reb':>5}")
        print(f"{'-'*100}")

        for i, (_, r) in enumerate(period.iterrows(), 1):
            mark = ' ***' if i <= 3 else ''
            print(f"{i:>3} {r['Strategy']:<22} {r['Final']:>11,.0f} {r['CAGR']:>7.1%} {r['MDD']:>7.1%} "
                  f"{r['Sharpe']:>8.2f} {r['Sortino']:>8.2f} {r['WinRate']:>6.1%} {r['Rebals']:>5}{mark}")

        print(f"{'-'*100}")
        if bm_btc:
            print(f"    {'BTC B&H':<22} {'':>11} {bm_btc.get('cagr',0):>7.1%} {bm_btc.get('mdd',0):>7.1%} {bm_btc.get('sharpe',0):>8.2f}")

    # Stability ranking
    stab = df.groupby('Strategy').agg({
        'CAGR':'mean','MDD':'mean','Sharpe':'mean','Sortino':'mean','Rebals':'mean'
    }).reset_index()
    stab['Score'] = stab['Sharpe']*0.5 + (1+stab['MDD'])*2 + stab['CAGR']
    stab = stab.sort_values('Score', ascending=False)

    print(f"\n{'='*100}")
    print(f"  CRYPTO STABILITY RANKING — Average across all start dates")
    print(f"{'='*100}")
    print(f"{'#':>3} {'Strategy':<22} {'AvgCAGR':>9} {'AvgMDD':>9} {'AvgSharpe':>10} {'AvgSortino':>11} {'Score':>7}")
    print(f"{'-'*100}")
    for i, (_, r) in enumerate(stab.iterrows(), 1):
        mark = ' <-- TOP' if i <= 5 else ''
        print(f"{i:>3} {r['Strategy']:<22} {r['CAGR']:>8.1%} {r['MDD']:>8.1%} {r['Sharpe']:>10.2f} "
              f"{r['Sortino']:>11.2f} {r['Score']:>7.2f}{mark}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
