"""
ETF Signal Flip Guard × Rebalancing Timing Cross Test
======================================================
Tests all guard variants across 7 rebalancing timings to verify
results are not biased toward month-end.

Usage:
    python3 strategies/cap_defend/backtest_etf_flip_guard_timing.py
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
# Signal functions
# ===========================================================================

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
        return get_signal(d, ind)

    all_on = True
    for c in CANARY:
        if not has(ind, c, d):
            all_on = False; break
        price = v(ind, c, 'price', d)
        sma = v(ind, c, 'sma200', d)
        if pd.isna(price) or pd.isna(sma):
            all_on = False; break
        if prev_risk_on:
            if price < sma * (1 - band_pct):
                all_on = False; break
        else:
            if price <= sma * (1 + band_pct):
                all_on = False; break

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


# ===========================================================================
# Backtest engine
# ===========================================================================

def run_bt(data, ind, start, end, rebal_dates, guard_type='none', guard_param=None,
           capital=10000, tx=0.001):
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.isin(data.index)]

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

        elif guard_type == 'hysteresis':
            do_flip = raw_flip

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
            do_flip = False

        elif guard_type == 'cool+hyst':
            if raw_flip:
                cd = guard_param[0]
                if cooldown_until is None or today >= cooldown_until:
                    do_flip = True
                    cooldown_until = today + pd.Timedelta(days=cd)

        if do_flip:
            flip_count += 1

        prev_st = st

        # Rebalance on schedule or valid flip
        if today in rebal_dates or do_flip:
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


# ===========================================================================
# Main
# ===========================================================================

def main():
    data = load_data()
    print("Pre-computing indicators...")
    ind = precompute(data)

    # Trading days index for rebal date generators
    di = data.dropna(how='all').index

    # --- Guard variants (top performers + baseline + representative from each category) ---
    guards = {}

    # Current (no guard)
    guards['00.No Guard (현재)'] = ('none', None)

    # Cooldown — best was 30d; also test 14d, 7d
    for cd in [7, 14, 21, 30]:
        guards[f'01.Cooldown {cd}d'] = ('cooldown', cd)

    # Hysteresis — top performers: 2%, 1%, 1.5%
    for band in [0.005, 0.01, 0.015, 0.02, 0.03, 0.05]:
        guards[f'02.Hysteresis {band:.1%}'] = ('hysteresis', band)

    # Consecutive — all were equivalent (flip=0), test 2d and 3d
    for n in [2, 3, 5]:
        guards[f'03.Consecutive {n}d'] = ('consecutive', n)

    # No flip
    guards['04.No Flip'] = ('no_flip', None)

    # Cooldown+Hysteresis — best was CD5d+Hyst2%
    for cd, band in [(5, 0.01), (5, 0.02), (7, 0.02), (10, 0.02)]:
        guards[f'05.CD{cd}d+Hyst{band:.1%}'] = ('cool+hyst', (cd, band))

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

    total = len(guards) * len(timings) * len(START_DATES)
    print(f"\nRunning {len(guards)} guards x {len(timings)} timings x {len(START_DATES)} periods = {total} backtests...\n")

    rows = []
    n = 0
    for gname, (gtype, gparam) in guards.items():
        for tname, tgen in timings.items():
            for start in START_DATES:
                n += 1
                try:
                    rebal_dates = tgen(start, END_DATE)
                    res = run_bt(data, ind, start, END_DATE, rebal_dates,
                                 guard_type=gtype, guard_param=gparam)
                    m = metrics(res['Value'])
                    rows.append({
                        'Guard': gname, 'Timing': tname, 'Start': start,
                        'CAGR': m['cagr'], 'MDD': m['mdd'], 'Sharpe': m['sharpe'],
                        'Sortino': m['sortino'], 'Calmar': m['calmar'],
                        'Final': m['final'],
                        'Rebals': res.attrs.get('rebals', 0),
                        'Flips': res.attrs.get('flips', 0),
                    })
                except:
                    rows.append({
                        'Guard': gname, 'Timing': tname, 'Start': start,
                        'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0,
                        'Calmar': 0, 'Final': 10000,
                        'Rebals': 0, 'Flips': 0,
                    })
            if n % 100 == 0:
                print(f"  Progress: {n}/{total} ({n*100//total}%)")

    df = pd.DataFrame(rows)

    # ===========================================================================
    # PART 1: Guard × Timing Sharpe matrix (averaged across all start dates)
    # ===========================================================================
    pivot = df.groupby(['Guard','Timing']).agg({
        'Sharpe':'mean','CAGR':'mean','MDD':'mean','Calmar':'mean',
        'Rebals':'mean','Flips':'mean'
    }).reset_index()

    sharpe_matrix = pivot.pivot(index='Guard', columns='Timing', values='Sharpe')
    timing_order = ['MonthEnd','MonthStart','Day5','Day10','Day15','Day20','Day25']
    sharpe_matrix = sharpe_matrix.reindex(columns=[t for t in timing_order if t in sharpe_matrix.columns])

    sharpe_matrix['Avg'] = sharpe_matrix[timing_order].mean(axis=1)
    sharpe_matrix['Max'] = sharpe_matrix[timing_order].max(axis=1)
    sharpe_matrix['Min'] = sharpe_matrix[timing_order].min(axis=1)
    sharpe_matrix['Spread'] = sharpe_matrix['Max'] - sharpe_matrix['Min']
    sharpe_matrix = sharpe_matrix.sort_values('Avg', ascending=False)

    bl_avg = sharpe_matrix.loc['00.No Guard (현재)', 'Avg'] if '00.No Guard (현재)' in sharpe_matrix.index else 0

    print(f"\n{'='*160}")
    print(f"  SHARPE MATRIX — Guard × Timing (Avg across {len(START_DATES)} periods)")
    print(f"{'='*160}")
    hdr = f"{'#':>3} {'Guard':<28}"
    for t in timing_order:
        hdr += f" {t:>9}"
    hdr += f" {'|':>2} {'Avg':>6} {'Min':>6} {'Max':>6} {'Spread':>6}"
    print(hdr)
    print(f"{'-'*160}")

    for i, (gname, row) in enumerate(sharpe_matrix.iterrows(), 1):
        line = f"{i:>3} {gname:<28}"
        for t in timing_order:
            val = row.get(t, 0)
            line += f" {val:>9.2f}"
        mark = ' <--' if '현재' in str(gname) else (' *' if row['Avg'] > bl_avg else '')
        line += f" {'|':>2} {row['Avg']:>6.2f} {row['Min']:>6.2f} {row['Max']:>6.2f} {row['Spread']:>6.2f}{mark}"
        print(line)
    print(f"{'='*160}")

    # ===========================================================================
    # PART 2: CAGR matrix
    # ===========================================================================
    cagr_matrix = pivot.pivot(index='Guard', columns='Timing', values='CAGR')
    cagr_matrix = cagr_matrix.reindex(columns=[t for t in timing_order if t in cagr_matrix.columns])
    cagr_matrix['Avg'] = cagr_matrix[timing_order].mean(axis=1)
    cagr_matrix = cagr_matrix.reindex(sharpe_matrix.index)  # same sort order

    print(f"\n{'='*160}")
    print(f"  CAGR MATRIX — Guard × Timing")
    print(f"{'='*160}")
    hdr = f"{'#':>3} {'Guard':<28}"
    for t in timing_order:
        hdr += f" {t:>9}"
    hdr += f" {'|':>2} {'Avg':>7}"
    print(hdr)
    print(f"{'-'*160}")

    for i, (gname, row) in enumerate(cagr_matrix.iterrows(), 1):
        line = f"{i:>3} {gname:<28}"
        for t in timing_order:
            val = row.get(t, 0)
            line += f" {val:>8.1%}"
        mark = ' <--' if '현재' in str(gname) else ''
        line += f" {'|':>2} {row['Avg']:>6.1%}{mark}"
        print(line)
    print(f"{'='*160}")

    # ===========================================================================
    # PART 3: Flip count matrix
    # ===========================================================================
    flip_matrix = pivot.pivot(index='Guard', columns='Timing', values='Flips')
    flip_matrix = flip_matrix.reindex(columns=[t for t in timing_order if t in flip_matrix.columns])
    flip_matrix['Avg'] = flip_matrix[timing_order].mean(axis=1)
    flip_matrix = flip_matrix.reindex(sharpe_matrix.index)

    print(f"\n{'='*160}")
    print(f"  FLIP COUNT MATRIX — Guard × Timing (평균 flip 횟수)")
    print(f"{'='*160}")
    hdr = f"{'#':>3} {'Guard':<28}"
    for t in timing_order:
        hdr += f" {t:>9}"
    hdr += f" {'|':>2} {'Avg':>6}"
    print(hdr)
    print(f"{'-'*160}")

    for i, (gname, row) in enumerate(flip_matrix.iterrows(), 1):
        line = f"{i:>3} {gname:<28}"
        for t in timing_order:
            val = row.get(t, 0)
            line += f" {val:>9.1f}"
        mark = ' <--' if '현재' in str(gname) else ''
        line += f" {'|':>2} {val:>6.1f}{mark}"
        print(line)
    print(f"{'='*160}")

    # ===========================================================================
    # PART 4: Guards that beat baseline at ALL timings
    # ===========================================================================
    bl_by_timing = pivot[pivot['Guard']=='00.No Guard (현재)'].set_index('Timing')

    print(f"\n{'='*120}")
    print(f"  모든 타이밍에서 현재(No Guard)를 이기는 가드")
    print(f"{'='*120}")

    wins = {}
    for gname in guards:
        if '현재' in gname: continue
        g_by_timing = pivot[pivot['Guard']==gname].set_index('Timing')
        win_count = 0
        win_timings = []
        for t in timing_order:
            if t in g_by_timing.index and t in bl_by_timing.index:
                if g_by_timing.loc[t, 'Sharpe'] > bl_by_timing.loc[t, 'Sharpe']:
                    win_count += 1
                    win_timings.append(t)
        wins[gname] = (win_count, win_timings)

    # Sort by wins desc
    sorted_wins = sorted(wins.items(), key=lambda x: (-x[1][0], x[0]))
    for gname, (cnt, tlist) in sorted_wins:
        if cnt > 0:
            avg_sharpe = sharpe_matrix.loc[gname, 'Avg'] if gname in sharpe_matrix.index else 0
            spread = sharpe_matrix.loc[gname, 'Spread'] if gname in sharpe_matrix.index else 0
            all_mark = ' *** ALL ***' if cnt == len(timing_order) else ''
            print(f"  {gname:<30} {cnt}/{len(timing_order)} 타이밍 승리  "
                  f"Avg Sharpe={avg_sharpe:.2f}  Spread={spread:.2f}  ({', '.join(tlist)}){all_mark}")

    # No winners?
    all_winners = [g for g, (c, _) in sorted_wins if c == len(timing_order)]
    if not all_winners:
        print(f"\n  전체 타이밍 승리 가드 없음. 최다 승리:")
        for gname, (cnt, tlist) in sorted_wins[:5]:
            if cnt > 0:
                print(f"    {gname}: {cnt}/{len(timing_order)} ({', '.join(tlist)})")

    print(f"{'='*120}")

    # ===========================================================================
    # PART 5: Best guard per timing
    # ===========================================================================
    print(f"\n{'='*120}")
    print(f"  타이밍별 최고 가드")
    print(f"{'='*120}")
    print(f"  {'Timing':<12} {'Best Guard':<30} {'Sharpe':>8} {'CAGR':>8} {'Flips':>7}  {'vs 현재':>8}")
    print(f"  {'-'*90}")
    for t in timing_order:
        t_data = pivot[pivot['Timing']==t].sort_values('Sharpe', ascending=False)
        if t_data.empty: continue
        best = t_data.iloc[0]
        bl = t_data[t_data['Guard'].str.contains('현재')]
        bl_s = bl.iloc[0]['Sharpe'] if not bl.empty else 0
        diff = best['Sharpe'] - bl_s
        print(f"  {t:<12} {best['Guard']:<30} {best['Sharpe']:>8.2f} {best['CAGR']:>7.1%} "
              f"{best['Flips']:>7.1f}  {diff:>+7.2f}")
    print(f"{'='*120}")

    # ===========================================================================
    # PART 6: Timing stability — which guard is least sensitive to timing?
    # ===========================================================================
    print(f"\n{'='*120}")
    print(f"  타이밍 안정성 순위 (Spread가 낮을수록 안정적)")
    print(f"{'='*120}")
    stability = sharpe_matrix[['Avg','Min','Max','Spread']].sort_values('Spread')
    print(f"  {'#':>3} {'Guard':<30} {'Avg Sharpe':>10} {'Min':>8} {'Max':>8} {'Spread':>8}")
    print(f"  {'-'*80}")
    for i, (gname, row) in enumerate(stability.iterrows(), 1):
        mark = ' <--' if '현재' in str(gname) else ''
        print(f"  {i:>3} {gname:<30} {row['Avg']:>10.2f} {row['Min']:>8.2f} {row['Max']:>8.2f} {row['Spread']:>8.2f}{mark}")
    print(f"{'='*120}")


if __name__ == '__main__':
    main()
