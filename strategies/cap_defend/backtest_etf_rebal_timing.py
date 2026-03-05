"""
ETF Rebalancing Timing Sensitivity Test
========================================
Tests whether the fixed top-3 selection is sensitive to rebalancing date.

Timing variants:
  - Month-end (current): last trading day of month
  - Month-start: first trading day of month
  - Mid-month: ~15th trading day
  - Various fixed days: 1st, 5th, 10th, 15th, 20th, 25th
  - Weekly (every Monday)
  - Bi-weekly (every other Monday)
  - Every 2 weeks offset
  - Every 3 weeks

Also tests: how often does 3rd vs 4th place actually flip?

Usage:
    python3 strategies/cap_defend/backtest_etf_rebal_timing.py
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
# Data & Indicators (reuse from backtest_etf)
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
    val = v(ind, ticker, 'mom252', date)
    return pd.notna(val)


# ===========================================================================
# Rebalancing date generators
# ===========================================================================

def gen_month_end(start, end, data_index):
    """Last trading day of each month (current behavior)."""
    dates = pd.date_range(start=start, end=end, freq='M')
    return set(dates)


def gen_month_start(start, end, data_index):
    """First trading day of each month."""
    dates = pd.date_range(start=start, end=end, freq='MS')
    # Snap to nearest trading day
    result = set()
    for d in dates:
        # Find first available date on or after d
        mask = data_index >= d
        if mask.any():
            result.add(data_index[mask][0])
    return result


def gen_fixed_day(start, end, data_index, day=15):
    """Fixed day of each month (snapped to nearest trading day)."""
    result = set()
    current = pd.Timestamp(start)
    end_ts = pd.Timestamp(end)
    while current <= end_ts:
        try:
            target = current.replace(day=min(day, 28))
        except:
            target = current.replace(day=28)
        # Find nearest trading day on or after target
        mask = data_index >= target
        if mask.any():
            nearest = data_index[mask][0]
            # But only if it's in the same month (or close)
            if (nearest - target).days < 10:
                result.add(nearest)
        current = current + pd.DateOffset(months=1)
    return result


def gen_weekly(start, end, data_index, weekday=0):
    """Every week on given weekday (0=Mon)."""
    result = set()
    for d in data_index:
        if d >= pd.Timestamp(start) and d <= pd.Timestamp(end):
            if d.weekday() == weekday:
                result.add(d)
    return result


def gen_biweekly(start, end, data_index, offset=0):
    """Every 2 weeks."""
    mondays = [d for d in data_index
               if d >= pd.Timestamp(start) and d <= pd.Timestamp(end) and d.weekday() == 0]
    return set(mondays[offset::2])


def gen_every_n_weeks(start, end, data_index, n=3):
    """Every N weeks."""
    mondays = [d for d in data_index
               if d >= pd.Timestamp(start) and d <= pd.Timestamp(end) and d.weekday() == 0]
    return set(mondays[::n])


# ===========================================================================
# Strategy (Cap Defend Stock — identical logic, parameterized rebal dates)
# ===========================================================================

def run_bt(data, ind, start, end, rebal_dates, capital=10000, tx=0.001):
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.isin(data.index)]

    cash = capital
    hold = {}
    hist = []
    rebals = 0
    prev_st = None
    pick_history = []  # Track what gets picked

    for today in dates:
        row = data.loc[today]
        pv = cash + sum(u * (row.get(t, 0) if pd.notna(row.get(t, 0)) else 0) for t, u in hold.items())
        hist.append({'Date': today, 'Value': pv})

        # Strategy signal
        risk_on = all(
            has(ind, c, today) and v(ind, c, 'price', today) > v(ind, c, 'sma200', today)
            for c in CANARY
        )

        if risk_on:
            rows = [(t, v(ind,t,'mom_w',today), v(ind,t,'sharpe126',today))
                    for t in OFFENSIVE if has(ind,t,today)]
            rows = [(t,m,q) for t,m,q in rows if pd.notna(m) and pd.notna(q)]
            if not rows:
                tgt, st = {'BIL': 1.0}, 'NoData'
            else:
                df = pd.DataFrame(rows, columns=['T','M','Q']).set_index('T')
                top_m = df.nlargest(3, 'M').index.tolist()
                top_q = df.nlargest(3, 'Q').index.tolist()
                picks = list(set(top_m + top_q))
                tgt = {t: 1.0/len(picks) for t in picks}
                st = 'On'

                if today in rebal_dates:
                    # Record the scores for analysis
                    mom_sorted = df['M'].sort_values(ascending=False)
                    qual_sorted = df['Q'].sort_values(ascending=False)
                    pick_history.append({
                        'Date': today,
                        'Mom_Picks': top_m,
                        'Qual_Picks': top_q,
                        'All_Picks': sorted(picks),
                        'Mom_3rd': mom_sorted.iloc[2] if len(mom_sorted) > 2 else np.nan,
                        'Mom_4th': mom_sorted.iloc[3] if len(mom_sorted) > 3 else np.nan,
                        'Mom_Gap34': (mom_sorted.iloc[2] - mom_sorted.iloc[3]) if len(mom_sorted) > 3 else np.nan,
                        'Qual_3rd': qual_sorted.iloc[2] if len(qual_sorted) > 2 else np.nan,
                        'Qual_4th': qual_sorted.iloc[3] if len(qual_sorted) > 3 else np.nan,
                        'Qual_Gap34': (qual_sorted.iloc[2] - qual_sorted.iloc[3]) if len(qual_sorted) > 3 else np.nan,
                    })
        else:
            best_t, best_r = 'BIL', -999
            for t in DEFENSIVE:
                r = v(ind, t, 'mom126', today)
                if pd.notna(r) and r > best_r:
                    best_r, best_t = r, t
            if best_r < 0:
                tgt, st = {'BIL': 1.0}, 'Cash'
            else:
                tgt, st = {best_t: 1.0}, 'Off'

        flip = prev_st is not None and st != prev_st
        prev_st = st

        if today in rebal_dates or flip:
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

    result = pd.DataFrame(hist).set_index('Date')
    result.attrs['rebals'] = rebals
    result.attrs['picks'] = pick_history
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
    data_index = data.index

    # Define all timing variants
    timing_variants = {}

    # Monthly variants
    timing_variants['Month-End (현재)'] = lambda s, e: gen_month_end(s, e, data_index)
    timing_variants['Month-Start'] = lambda s, e: gen_month_start(s, e, data_index)
    for day in [5, 10, 15, 20, 25]:
        timing_variants[f'Monthly {day}일'] = lambda s, e, d=day: gen_fixed_day(s, e, data_index, d)

    # Weekly / bi-weekly
    timing_variants['Weekly (매주 월)'] = lambda s, e: gen_weekly(s, e, data_index, 0)
    timing_variants['Weekly (매주 수)'] = lambda s, e: gen_weekly(s, e, data_index, 2)
    timing_variants['Weekly (매주 금)'] = lambda s, e: gen_weekly(s, e, data_index, 4)
    timing_variants['Bi-weekly A'] = lambda s, e: gen_biweekly(s, e, data_index, 0)
    timing_variants['Bi-weekly B'] = lambda s, e: gen_biweekly(s, e, data_index, 1)
    timing_variants['Every 3 weeks'] = lambda s, e: gen_every_n_weeks(s, e, data_index, 3)

    total = len(timing_variants) * len(START_DATES)
    print(f"\nRunning {len(timing_variants)} timing variants x {len(START_DATES)} periods = {total} backtests...\n")

    rows = []
    all_picks = {}  # For gap analysis
    n = 0

    for name, gen_func in timing_variants.items():
        for start in START_DATES:
            n += 1
            rebal_dates = gen_func(start, END_DATE)
            res = run_bt(data, ind, start, END_DATE, rebal_dates)
            m = metrics(res['Value'])
            rows.append({
                'Timing': name, 'Start': start,
                'CAGR': m['cagr'], 'MDD': m['mdd'], 'Sharpe': m['sharpe'],
                'Sortino': m['sortino'], 'Calmar': m['calmar'],
                'Final': m['final'], 'Rebals': res.attrs.get('rebals', 0),
            })
            # Store picks for gap analysis
            key = f"{name}|{start}"
            all_picks[key] = res.attrs.get('picks', [])

        if n % 10 == 0:
            print(f"  Progress: {n}/{total}")

    df = pd.DataFrame(rows)

    # ===== PART 1: Timing comparison =====
    stab = df.groupby('Timing').agg({
        'CAGR': ['mean', 'std'],
        'MDD': ['mean', 'std'],
        'Sharpe': ['mean', 'std'],
        'Sortino': 'mean',
        'Calmar': 'mean',
        'Rebals': 'mean',
        'Final': 'mean',
    })
    stab.columns = ['CAGR_avg','CAGR_std','MDD_avg','MDD_std','Sharpe_avg','Sharpe_std',
                     'Sortino_avg','Calmar_avg','Rebals_avg','Final_avg']
    stab = stab.sort_values('Sharpe_avg', ascending=False).reset_index()

    print(f"\n{'='*140}")
    print(f"  REBALANCING TIMING SENSITIVITY — Average across {len(START_DATES)} start dates")
    print(f"{'='*140}")
    print(f"{'#':>3} {'Timing':<22} {'CAGR':>8} {'±':>6} {'MDD':>8} {'±':>6} {'Sharpe':>8} {'±':>6} "
          f"{'Sortino':>8} {'Calmar':>8} {'Reb':>6}")
    print(f"{'-'*140}")

    for i, (_, r) in enumerate(stab.iterrows(), 1):
        mark = ' <-- 현재' if '현재' in r['Timing'] else ''
        print(f"{i:>3} {r['Timing']:<22} {r['CAGR_avg']:>7.1%} {r['CAGR_std']:>5.1%} "
              f"{r['MDD_avg']:>7.1%} {r['MDD_std']:>5.1%} "
              f"{r['Sharpe_avg']:>8.2f} {r['Sharpe_std']:>5.2f} "
              f"{r['Sortino_avg']:>8.2f} {r['Calmar_avg']:>8.2f} "
              f"{r['Rebals_avg']:>6.0f}{mark}")

    print(f"{'='*140}")

    # Spread analysis
    sharpe_vals = stab['Sharpe_avg']
    cagr_vals = stab['CAGR_avg']
    print(f"\n  Sharpe Range: {sharpe_vals.min():.2f} ~ {sharpe_vals.max():.2f} (spread: {sharpe_vals.max()-sharpe_vals.min():.2f})")
    print(f"  CAGR Range:   {cagr_vals.min():.1%} ~ {cagr_vals.max():.1%} (spread: {(cagr_vals.max()-cagr_vals.min()):.1%})")

    # ===== PART 2: Per-period detail =====
    print(f"\n{'='*140}")
    print(f"  PER-PERIOD DETAIL — Monthly variants only")
    print(f"{'='*140}")

    monthly_timings = [t for t in timing_variants.keys() if 'Month' in t or 'Monthly' in t]

    for start in START_DATES:
        period = df[(df['Start'] == start) & (df['Timing'].isin(monthly_timings))]
        period = period.sort_values('Sharpe', ascending=False)
        print(f"\n  --- {start} ---")
        print(f"  {'Timing':<22} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Final($)':>10} {'Reb':>5}")
        for _, r in period.iterrows():
            mark = ' <--' if '현재' in r['Timing'] else ''
            print(f"  {r['Timing']:<22} {r['CAGR']:>7.1%} {r['MDD']:>7.1%} "
                  f"{r['Sharpe']:>8.2f} {r['Final']:>10,.0f} {r['Rebals']:>5}{mark}")

    # ===== PART 3: 3rd vs 4th gap analysis =====
    print(f"\n{'='*140}")
    print(f"  3위 vs 4위 점수 갭 분석 (리밸런싱 시점마다)")
    print(f"{'='*140}")

    # Use month-end picks for analysis
    me_picks = []
    for key, picks in all_picks.items():
        if 'Month-End' in key:
            me_picks.extend(picks)

    if me_picks:
        gaps_df = pd.DataFrame(me_picks)

        mom_gaps = gaps_df['Mom_Gap34'].dropna()
        qual_gaps = gaps_df['Qual_Gap34'].dropna()

        print(f"\n  Momentum 3위-4위 갭:")
        print(f"    평균: {mom_gaps.mean():.4f}")
        print(f"    중앙값: {mom_gaps.median():.4f}")
        print(f"    최소: {mom_gaps.min():.4f}")
        print(f"    최대: {mom_gaps.max():.4f}")
        print(f"    갭 < 0.02 (아슬아슬): {(mom_gaps < 0.02).sum()}/{len(mom_gaps)} ({(mom_gaps < 0.02).mean():.0%})")
        print(f"    갭 < 0.05: {(mom_gaps < 0.05).sum()}/{len(mom_gaps)} ({(mom_gaps < 0.05).mean():.0%})")
        print(f"    갭 < 0.10: {(mom_gaps < 0.10).sum()}/{len(mom_gaps)} ({(mom_gaps < 0.10).mean():.0%})")

        print(f"\n  Quality(Sharpe) 3위-4위 갭:")
        print(f"    평균: {qual_gaps.mean():.4f}")
        print(f"    중앙값: {qual_gaps.median():.4f}")
        print(f"    최소: {qual_gaps.min():.4f}")
        print(f"    최대: {qual_gaps.max():.4f}")
        print(f"    갭 < 0.10: {(qual_gaps < 0.10).sum()}/{len(qual_gaps)} ({(qual_gaps < 0.10).mean():.0%})")
        print(f"    갭 < 0.20: {(qual_gaps < 0.20).sum()}/{len(qual_gaps)} ({(qual_gaps < 0.20).mean():.0%})")
        print(f"    갭 < 0.50: {(qual_gaps < 0.50).sum()}/{len(qual_gaps)} ({(qual_gaps < 0.50).mean():.0%})")

    # ===== PART 4: Pick overlap between timings =====
    print(f"\n{'='*140}")
    print(f"  리밸런싱 타이밍별 종목 선정 일치율")
    print(f"{'='*140}")

    # Compare month-end vs other monthly timings for same start date
    for start in START_DATES[:2]:  # Just first 2 periods
        me_key = f"Month-End (현재)|{start}"
        me_p = all_picks.get(me_key, [])
        if not me_p: continue

        me_sets = {p['Date']: set(p['All_Picks']) for p in me_p}

        print(f"\n  --- {start} --- (Month-End 기준 비교)")

        for other_name in ['Month-Start', 'Monthly 5일', 'Monthly 15일', 'Monthly 20일']:
            other_key = f"{other_name}|{start}"
            other_p = all_picks.get(other_key, [])
            if not other_p: continue

            # Find closest matching dates
            other_sets = {p['Date']: set(p['All_Picks']) for p in other_p}

            # Compare picks month by month
            matches = 0
            total_comp = 0
            partial_matches = 0

            for me_date, me_set in me_sets.items():
                # Find closest other date (within 20 days)
                best_other = None
                best_dist = 999
                for o_date in other_sets:
                    dist = abs((me_date - o_date).days)
                    if dist < best_dist and dist < 20:
                        best_dist = dist
                        best_other = o_date

                if best_other is not None:
                    total_comp += 1
                    o_set = other_sets[best_other]
                    if me_set == o_set:
                        matches += 1
                    elif len(me_set & o_set) >= len(me_set) - 1:
                        partial_matches += 1

            if total_comp > 0:
                print(f"    vs {other_name:<16}: 완전일치 {matches}/{total_comp} ({matches/total_comp:.0%}) | "
                      f"1개 차이 {partial_matches}/{total_comp} ({partial_matches/total_comp:.0%}) | "
                      f"일치+유사 {(matches+partial_matches)/total_comp:.0%}")

    print(f"\n{'='*140}")


if __name__ == '__main__':
    main()
