"""
ETF Signal Flip Guard Test
===========================
Tests various guards against high-frequency flipping at MA200 boundary.

Guards:
  0. No guard (current) — flip triggers immediate rebalancing
  1. Cooldown N days — after flip, ignore flips for N days
  2. Hysteresis band — enter at MA+X%, exit at MA-X%
  3. Consecutive N days — require N consecutive days before flipping
  4. Flip disabled — only rebalance on scheduled dates
  5. Cooldown + Hysteresis combo

Usage:
    python3 strategies/cap_defend/backtest_etf_flip_guard.py
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


def get_signal(d, ind):
    """Get raw risk-on/off signal and portfolio."""
    risk_on = all(
        has(ind, c, d) and v(ind, c, 'price', d) > v(ind, c, 'sma200', d)
        for c in CANARY
    )
    if risk_on:
        rows = [(t, v(ind,t,'mom_w',d), v(ind,t,'sharpe126',d))
                for t in OFFENSIVE if has(ind,t,d)]
        rows = [(t,m,q) for t,m,q in rows if pd.notna(m) and pd.notna(q)]
        if not rows: return {'BIL': 1.0}, 'Cash', False
        df = pd.DataFrame(rows, columns=['T','M','Q']).set_index('T')
        picks = list(set(df.nlargest(3,'M').index.tolist() + df.nlargest(3,'Q').index.tolist()))
        return {t: 1.0/len(picks) for t in picks}, 'On', True
    else:
        best_t, best_r = 'BIL', -999
        for t in DEFENSIVE:
            r = v(ind, t, 'mom126', d)
            if pd.notna(r) and r > best_r: best_r, best_t = r, t
        if best_r < 0: return {'BIL': 1.0}, 'Cash', False
        return {best_t: 1.0}, 'Off', False


def get_signal_hysteresis(d, ind, prev_risk_on, band_pct):
    """Hysteresis: enter at MA*(1+band), exit at MA*(1-band)."""
    if prev_risk_on is None:
        # First day — use standard check
        return get_signal(d, ind)

    # Check with hysteresis
    all_on = True
    for c in CANARY:
        if not has(ind, c, d):
            all_on = False
            break
        price = v(ind, c, 'price', d)
        sma = v(ind, c, 'sma200', d)
        if pd.isna(price) or pd.isna(sma):
            all_on = False
            break

        if prev_risk_on:
            # Currently ON — need price < sma*(1-band) to turn OFF
            if price < sma * (1 - band_pct):
                all_on = False
                break
        else:
            # Currently OFF — need price > sma*(1+band) to turn ON
            if price <= sma * (1 + band_pct):
                all_on = False
                break

    if all_on:
        rows = [(t, v(ind,t,'mom_w',d), v(ind,t,'sharpe126',d))
                for t in OFFENSIVE if has(ind,t,d)]
        rows = [(t,m,q) for t,m,q in rows if pd.notna(m) and pd.notna(q)]
        if not rows: return {'BIL': 1.0}, 'Cash', False
        df = pd.DataFrame(rows, columns=['T','M','Q']).set_index('T')
        picks = list(set(df.nlargest(3,'M').index.tolist() + df.nlargest(3,'Q').index.tolist()))
        return {t: 1.0/len(picks) for t in picks}, 'On', True
    else:
        best_t, best_r = 'BIL', -999
        for t in DEFENSIVE:
            r = v(ind, t, 'mom126', d)
            if pd.notna(r) and r > best_r: best_r, best_t = r, t
        if best_r < 0: return {'BIL': 1.0}, 'Cash', False
        return {best_t: 1.0}, 'Off', False


def run_bt(data, ind, start, end, guard_type='none', guard_param=None, capital=10000, tx=0.001):
    """
    guard_type:
      'none'        — current behavior, flip = immediate rebalance
      'cooldown'    — after flip, no flips for guard_param days
      'hysteresis'  — band of guard_param% around MA200
      'consecutive' — require guard_param consecutive days
      'no_flip'     — only rebalance on scheduled dates
      'cool+hyst'   — cooldown + hysteresis combined
    """
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.isin(data.index)]
    month_ends = set(pd.date_range(start=start, end=end, freq='M'))

    cash = capital
    hold = {}
    hist = []
    rebals = 0
    flip_count = 0
    prev_st = None
    prev_risk_on = None

    # Guard state
    cooldown_until = None
    consec_count = 0
    consec_target_st = None

    for today in dates:
        row = data.loc[today]
        pv = cash + sum(u*(row.get(t,0) if pd.notna(row.get(t,0)) else 0) for t,u in hold.items())
        hist.append({'Date': today, 'Value': pv})

        # Get signal based on guard type
        if guard_type in ('hysteresis', 'cool+hyst'):
            band = guard_param if guard_type == 'hysteresis' else guard_param[1]
            tgt, st, risk_on = get_signal_hysteresis(today, ind, prev_risk_on, band)
            prev_risk_on = risk_on
        else:
            tgt, st, risk_on = get_signal(today, ind)

        # Determine if flip happened
        raw_flip = prev_st is not None and st != prev_st

        # Apply guard
        do_flip = False
        if guard_type == 'none':
            do_flip = raw_flip

        elif guard_type == 'cooldown':
            if raw_flip:
                if cooldown_until is None or today >= cooldown_until:
                    do_flip = True
                    cooldown_until = today + pd.Timedelta(days=guard_param)
                # else: in cooldown, ignore flip

        elif guard_type == 'hysteresis':
            do_flip = raw_flip  # hysteresis already handles the band

        elif guard_type == 'consecutive':
            if raw_flip:
                if st == consec_target_st:
                    consec_count += 1
                else:
                    consec_target_st = st
                    consec_count = 1
                if consec_count >= guard_param:
                    do_flip = True
                    consec_count = 0
                    consec_target_st = None
            else:
                consec_count = 0
                consec_target_st = None

        elif guard_type == 'no_flip':
            do_flip = False  # never flip, only scheduled

        elif guard_type == 'cool+hyst':
            if raw_flip:
                cd = guard_param[0]
                if cooldown_until is None or today >= cooldown_until:
                    do_flip = True
                    cooldown_until = today + pd.Timedelta(days=cd)

        if do_flip:
            flip_count += 1

        if prev_st is not None:
            prev_st_save = prev_st
        prev_st = st

        # Rebalance on schedule or valid flip
        if today in month_ends or do_flip:
            rebals += 1
            amt = pv * (1 - tx)
            cash, hold = amt, {}
            for t, w in tgt.items():
                p = row.get(t,0) if pd.notna(row.get(t,0)) else 0
                if p > 0:
                    a = amt * w; hold[t] = a / p; cash -= a

    result = pd.DataFrame(hist).set_index('Date')
    result.attrs['rebals'] = rebals
    result.attrs['flips'] = flip_count
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


def main():
    data = load_data()
    print("Pre-computing indicators...")
    ind = precompute(data)

    # --- Define all guard variants ---
    guards = {}

    # 0. No guard (current)
    guards['00.No Guard (현재)'] = ('none', None)

    # 1. Cooldown
    for cd in [3, 5, 7, 10, 14, 21, 30]:
        guards[f'01.Cooldown {cd}d'] = ('cooldown', cd)

    # 2. Hysteresis band
    for band in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        guards[f'02.Hysteresis {band:.1%}'] = ('hysteresis', band)

    # 3. Consecutive days
    for n in [2, 3, 5, 7]:
        guards[f'03.Consecutive {n}d'] = ('consecutive', n)

    # 4. No flip at all
    guards['04.No Flip (schedule only)'] = ('no_flip', None)

    # 5. Cooldown + Hysteresis combos
    for cd, band in [(5, 0.01), (5, 0.02), (7, 0.01), (7, 0.02), (10, 0.02), (14, 0.02)]:
        guards[f'05.CD{cd}d+Hyst{band:.1%}'] = ('cool+hyst', (cd, band))

    total = len(guards) * len(START_DATES)
    print(f"\nRunning {len(guards)} guard variants x {len(START_DATES)} periods = {total} backtests...\n")

    rows = []
    n = 0
    for name, (gtype, gparam) in guards.items():
        for start in START_DATES:
            n += 1
            res = run_bt(data, ind, start, END_DATE, guard_type=gtype, guard_param=gparam)
            m = metrics(res['Value'])
            rows.append({
                'Guard': name, 'Start': start,
                'CAGR': m['cagr'], 'MDD': m['mdd'], 'Sharpe': m['sharpe'],
                'Sortino': m['sortino'], 'Calmar': m['calmar'],
                'Final': m['final'],
                'Rebals': res.attrs.get('rebals', 0),
                'Flips': res.attrs.get('flips', 0),
            })
        if n % 20 == 0:
            print(f"  Progress: {n}/{total}")

    df = pd.DataFrame(rows)

    # === Stability ranking ===
    stab = df.groupby('Guard').agg({
        'CAGR': 'mean', 'MDD': 'mean', 'Sharpe': 'mean',
        'Sortino': 'mean', 'Calmar': 'mean',
        'Rebals': 'mean', 'Flips': 'mean',
    }).reset_index()
    stab = stab.sort_values('Sharpe', ascending=False).reset_index(drop=True)

    bl_sharpe = stab[stab['Guard'].str.contains('현재')]['Sharpe'].values[0]

    print(f"\n{'='*150}")
    print(f"  SIGNAL FLIP GUARD — STABILITY RANKING (Avg across {len(START_DATES)} periods)")
    print(f"{'='*150}")
    print(f"{'#':>3} {'Guard':<30} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>8} {'Calmar':>8} {'Rebals':>7} {'Flips':>7}")
    print(f"{'-'*150}")

    for i, (_, r) in enumerate(stab.iterrows(), 1):
        mark = ''
        if '현재' in r['Guard']: mark = ' <-- 현재'
        elif r['Sharpe'] > bl_sharpe: mark = ' *'
        print(f"{i:>3} {r['Guard']:<30} {r['CAGR']:>7.1%} {r['MDD']:>7.1%} "
              f"{r['Sharpe']:>8.2f} {r['Sortino']:>8.2f} {r['Calmar']:>8.2f} "
              f"{r['Rebals']:>7.0f} {r['Flips']:>7.1f}{mark}")

    print(f"{'='*150}")

    # === Per-period detail for top 10 ===
    top10 = stab.head(10)['Guard'].tolist()
    if not any('현재' in g for g in top10):
        top10.append('00.No Guard (현재)')

    print(f"\n{'='*150}")
    print(f"  PER-PERIOD DETAIL — Top 10 + 현재")
    print(f"{'='*150}")

    for start in START_DATES:
        period = df[(df['Start'] == start) & (df['Guard'].isin(top10))]
        period = period.sort_values('Sharpe', ascending=False)
        print(f"\n  --- {start} ---")
        print(f"  {'Guard':<30} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Final($)':>10} {'Reb':>5} {'Flips':>6}")
        for _, r in period.iterrows():
            mark = ' <--' if '현재' in r['Guard'] else ''
            print(f"  {r['Guard']:<30} {r['CAGR']:>7.1%} {r['MDD']:>7.1%} "
                  f"{r['Sharpe']:>8.2f} {r['Final']:>10,.0f} {r['Rebals']:>5} {r['Flips']:>6}{mark}")

    # === Category summary ===
    print(f"\n{'='*150}")
    print(f"  CATEGORY SUMMARY")
    print(f"{'='*150}")

    categories = [
        ('00.', 'No Guard (현재)'),
        ('01.', 'Cooldown'),
        ('02.', 'Hysteresis'),
        ('03.', 'Consecutive'),
        ('04.', 'No Flip'),
        ('05.', 'Cooldown+Hysteresis'),
    ]

    print(f"  {'Category':<22} {'Best Variant':<30} {'Sharpe':>8} {'CAGR':>8} {'MDD':>8} {'Flips':>7}")
    print(f"  {'-'*90}")
    for prefix, cat_name in categories:
        cat = stab[stab['Guard'].str.startswith(prefix)]
        if cat.empty: continue
        best = cat.iloc[0]
        print(f"  {cat_name:<22} {best['Guard']:<30} {best['Sharpe']:>8.2f} "
              f"{best['CAGR']:>7.1%} {best['MDD']:>7.1%} {best['Flips']:>7.1f}")

    # === Flip frequency analysis ===
    print(f"\n{'='*150}")
    print(f"  FLIP 빈도 분석 — 가드 없을 때 얼마나 자주 flip이 발생하는가?")
    print(f"{'='*150}")

    no_guard = df[df['Guard'].str.contains('현재')]
    for _, r in no_guard.iterrows():
        years = (pd.Timestamp(END_DATE) - pd.Timestamp(r['Start'])).days / 365.25
        print(f"  {r['Start']}: {r['Flips']:.0f} flips in {years:.1f} years "
              f"= {r['Flips']/years:.1f} flips/year, {r['Rebals']:.0f} total rebals")

    avg_flips = no_guard['Flips'].mean()
    avg_years = ((pd.Timestamp(END_DATE) - pd.Timestamp('2021-01-01')).days / 365.25)
    print(f"\n  평균 {avg_flips:.0f} flips per period")
    print(f"{'='*150}")


if __name__ == '__main__':
    main()
