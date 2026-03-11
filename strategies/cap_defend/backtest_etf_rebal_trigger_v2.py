"""
ETF Rebalancing Trigger V2 — Robustness Test
=============================================
Tests improved triggers that handle unstable recommendations:
  0. None — monthly rebal + signal flip only
  1. T25%+C3d (current) — same rec 3 consecutive days + turnover >= 25%
  2. Sustained T25%+C3d — turnover >= 25% for 3 consecutive days (rec can change)
  3. Majority Vote 5d/3 — ticker in rec 3+ of last 5 days, turnover >= 25%
  4. T25%+C3d + Timeout 7d — current + force after 7 days of high turnover
  5. Score EMA3 + T25% — smooth scores with 3d EMA, then turnover >= 25% immediate
"""

import os, warnings
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

START_DATES = ['2017-06-01','2018-01-01','2019-01-01','2020-01-01',
               '2021-01-01','2022-01-01','2023-01-01']
END_DATE = '2025-12-31'

OFFENSIVE = ['SPY','QQQ','EFA','EEM','VT','VEA','GLD','PDBC','QUAL','MTUM','IQLT','IMTM']
DEFENSIVE = ['IEF','BIL','BNDX','GLD','PDBC']
CANARY = ['VT', 'EEM']
HYST_BAND = 0.01


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
    idx = pd.date_range(start='2016-01-01', end=END_DATE, freq='D')
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
        rm = dr.rolling(126).mean()
        rs = dr.rolling(126).std()
        d['sharpe126'] = (rm / rs.replace(0, np.nan)) * np.sqrt(252)
        # EMA smoothed scores
        d['mom_w_ema3'] = d['mom_w'].ewm(span=3, adjust=False).mean()
        d['sharpe126_ema3'] = d['sharpe126'].ewm(span=3, adjust=False).mean()
        ind[col] = d
    return ind


def check_risk_on(today, ind, prev_risk_on):
    vt_d, eem_d = ind.get('VT'), ind.get('EEM')
    if vt_d is None or eem_d is None: return False
    try:
        vt_p = vt_d.loc[today, 'price']; vt_sma = vt_d.loc[today, 'sma200']
        eem_p = eem_d.loc[today, 'price']; eem_sma = eem_d.loc[today, 'sma200']
    except: return prev_risk_on if prev_risk_on is not None else False
    if any(pd.isna([vt_p, vt_sma, eem_p, eem_sma])): return prev_risk_on if prev_risk_on is not None else False

    if prev_risk_on is None:
        return vt_p > vt_sma and eem_p > eem_sma
    elif prev_risk_on:
        return not (vt_p < vt_sma * (1 - HYST_BAND) or eem_p < eem_sma * (1 - HYST_BAND))
    else:
        return vt_p > vt_sma * (1 + HYST_BAND) and eem_p > eem_sma * (1 + HYST_BAND)


def get_picks(today, ind, use_ema=False):
    """Get offensive picks. Returns (set of tickers, mode)"""
    mom_col = 'mom_w_ema3' if use_ema else 'mom_w'
    sharpe_col = 'sharpe126_ema3' if use_ema else 'sharpe126'
    
    scores = []
    for t in OFFENSIVE:
        d = ind.get(t)
        if d is None: continue
        try:
            m = d.loc[today, mom_col]; q = d.loc[today, sharpe_col]
            if pd.notna(m) and pd.notna(q): scores.append((t, m, q))
        except: continue
    if len(scores) < 6: return set(), 'offense'
    df = pd.DataFrame(scores, columns=['T','Mom','Qual']).set_index('T')
    top_m = set(df.nlargest(3, 'Mom').index)
    top_q = set(df.nlargest(3, 'Qual').index)
    return top_m | top_q, 'offense'


def get_def_pick(today, ind):
    best_t, best_r = None, -999
    for t in DEFENSIVE:
        d = ind.get(t)
        if d is None: continue
        try:
            r = d.loc[today, 'mom126']
            if pd.notna(r) and r > best_r: best_t, best_r = t, r
        except: continue
    if best_r < 0: return set(), 'cash'
    return {best_t}, 'defense'


def calc_turnover(old_picks, new_picks):
    if not old_picks or not new_picks: return 0
    all_t = old_picks | new_picks
    old_w = {t: 1.0/len(old_picks) if t in old_picks else 0 for t in all_t}
    new_w = {t: 1.0/len(new_picks) if t in new_picks else 0 for t in all_t}
    return round(sum(abs(new_w[t] - old_w[t]) for t in all_t) / 2, 4)


def gen_month_end(s, e, data_idx):
    dr = pd.date_range(s, e, freq='D')
    dr = dr[dr.isin(data_idx)]
    return set(dr.to_series().groupby(dr.to_series().dt.to_period('M')).last())

def gen_month_start(s, e, data_idx):
    dr = pd.date_range(s, e, freq='D')
    dr = dr[dr.isin(data_idx)]
    return set(dr.to_series().groupby(dr.to_series().dt.to_period('M')).first())

def gen_fixed_day(s, e, data_idx, day):
    dr = pd.date_range(s, e, freq='D')
    dr = dr[dr.isin(data_idx)]
    result = set()
    for _, grp in dr.to_series().groupby(dr.to_series().dt.to_period('M')):
        candidates = grp[grp.dt.day >= day]
        if len(candidates) > 0: result.add(candidates.iloc[0])
    return result


def run_bt(data, ind, start, end, rebal_dates, trigger_type='none', trigger_param=None):
    capital = 10000
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.isin(data.index)]

    cash = capital
    hold = {}
    hist = []
    rebals = 0
    flip_count = 0
    trigger_count = 0
    prev_risk_on = None
    current_picks = set()

    # Trigger state
    # For type 1 (T25%+C3d): same rec consecutive
    pending_rec = None
    confirm_counter = 0
    # For type 2 (Sustained): turnover high consecutive
    sustained_counter = 0
    # For type 3 (Majority vote): rolling window of recs
    rec_history = []
    # For type 4 (Timeout): high turnover day counter
    timeout_counter = 0
    high_turnover_started = False

    for today in dates:
        row = data.loc[today]
        pv = cash + sum(u*(row.get(t,0) if pd.notna(row.get(t,0)) else 0) for t,u in hold.items())
        hist.append({'Date': today, 'Value': pv})

        risk_on = check_risk_on(today, ind, prev_risk_on)

        do_flip = False
        if prev_risk_on is not None and risk_on != prev_risk_on:
            do_flip = True
            flip_count += 1
        prev_risk_on = risk_on

        # Get today's recommended picks
        use_ema = (trigger_type == 'score_ema')
        if risk_on:
            new_picks, _ = get_picks(today, ind, use_ema=use_ema)
        else:
            new_picks, _ = get_def_pick(today, ind)
        
        if not new_picks:
            continue

        # Check trigger
        do_trigger = False
        turnover = calc_turnover(current_picks, new_picks)

        if trigger_type == 'none':
            pass

        elif trigger_type == 'current_c3d':
            # T25%+C3d: same rec list for 3 consecutive days
            if turnover >= 0.25 and new_picks != current_picks:
                if new_picks == pending_rec:
                    confirm_counter += 1
                else:
                    pending_rec = new_picks.copy()
                    confirm_counter = 1
                if confirm_counter >= 3:
                    do_trigger = True
                    confirm_counter = 0
                    pending_rec = None
            else:
                confirm_counter = 0
                pending_rec = None

        elif trigger_type == 'sustained_c3d':
            # Sustained: turnover >= 25% for 3 consecutive days (rec can change)
            if turnover >= 0.25:
                sustained_counter += 1
                if sustained_counter >= 3:
                    do_trigger = True
                    sustained_counter = 0
            else:
                sustained_counter = 0

        elif trigger_type == 'majority_vote':
            # Majority vote: ticker in rec 3+ of last 5 days
            window, min_votes = trigger_param  # (5, 3)
            rec_history.append(new_picks.copy())
            if len(rec_history) > window:
                rec_history.pop(0)
            
            if len(rec_history) >= window and current_picks:
                # Count how many days each ticker appeared
                from collections import Counter
                ticker_counts = Counter()
                for recs in rec_history:
                    for t in recs:
                        ticker_counts[t] += 1
                # Build consensus picks: tickers with >= min_votes appearances
                consensus = {t for t, c in ticker_counts.items() if c >= min_votes}
                # Keep same size as typical pick count (top by vote count)
                if len(consensus) > 6:
                    consensus = set(sorted(consensus, key=lambda t: -ticker_counts[t])[:6])
                if consensus:
                    consensus_turnover = calc_turnover(current_picks, consensus)
                    if consensus_turnover >= 0.25:
                        do_trigger = True
                        new_picks = consensus  # Use consensus picks

        elif trigger_type == 'timeout':
            # T25%+C3d + Timeout escape
            timeout_days = trigger_param  # 7
            if turnover >= 0.25 and new_picks != current_picks:
                if new_picks == pending_rec:
                    confirm_counter += 1
                else:
                    pending_rec = new_picks.copy()
                    confirm_counter = 1
                if confirm_counter >= 3:
                    do_trigger = True
                    confirm_counter = 0
                    pending_rec = None
                    high_turnover_started = False
                    timeout_counter = 0
            else:
                if turnover < 0.25:
                    confirm_counter = 0
                    pending_rec = None
            
            # Timeout tracking
            if turnover >= 0.25 and not do_trigger:
                if not high_turnover_started:
                    high_turnover_started = True
                    timeout_counter = 1
                else:
                    timeout_counter += 1
                if timeout_counter >= timeout_days:
                    do_trigger = True
                    timeout_counter = 0
                    high_turnover_started = False
                    confirm_counter = 0
                    pending_rec = None
            elif turnover < 0.25:
                high_turnover_started = False
                timeout_counter = 0

        elif trigger_type == 'score_ema':
            # Score EMA + immediate T25%
            if turnover >= 0.25 and new_picks != current_picks:
                do_trigger = True

        if do_trigger:
            trigger_count += 1

        should_rebal = today in rebal_dates or do_flip or do_trigger
        if should_rebal:
            rebals += 1
            target = new_picks if new_picks else current_picks
            if target:
                # Sell all
                for t, u in hold.items():
                    price = row.get(t, 0)
                    if pd.notna(price) and price > 0:
                        cash += u * price * (1 - 0.001)
                hold = {}
                # Buy equal weight
                w = 1.0 / len(target)
                for t in target:
                    price = row.get(t, 0)
                    if pd.notna(price) and price > 0:
                        alloc = cash * w
                        hold[t] = alloc * (1 - 0.001) / price
                cash -= sum(hold[t] * row.get(t, 0) for t in hold if pd.notna(row.get(t, 0)))
                cash = max(cash, 0)
                current_picks = target.copy()

    return pd.DataFrame(hist).set_index('Date')


def metrics(series):
    if len(series) < 2: return {'cagr':0,'mdd':0,'sharpe':0,'sortino':0,'calmar':0,'final':0}
    vals = series.values
    days = (series.index[-1] - series.index[0]).days
    if days < 30: return {'cagr':0,'mdd':0,'sharpe':0,'sortino':0,'calmar':0,'final':vals[-1]}
    
    cagr = (vals[-1]/vals[0])**(365.25/days) - 1
    peak = np.maximum.accumulate(vals)
    dd = (vals - peak) / peak
    mdd = dd.min()
    
    dr = pd.Series(vals).pct_change().dropna()
    sharpe = (dr.mean() / dr.std()) * np.sqrt(252) if dr.std() > 0 else 0
    downside = dr[dr<0].std()
    sortino = (dr.mean() / downside) * np.sqrt(252) if downside > 0 else 0
    calmar = cagr / abs(mdd) if mdd != 0 else 0
    
    return {'cagr': cagr, 'mdd': mdd, 'sharpe': sharpe, 'sortino': sortino, 'calmar': calmar, 'final': vals[-1]}


def main():
    data = load_data()
    di = data.index
    ind = precompute(data)

    triggers = {}
    triggers['00.None'] = ('none', None)
    triggers['01.T25%+C3d (current)'] = ('current_c3d', None)
    triggers['02.Sustained T25%+C3d'] = ('sustained_c3d', None)
    triggers['03.MajVote 5d/3'] = ('majority_vote', (5, 3))
    triggers['04.MajVote 5d/4'] = ('majority_vote', (5, 4))
    triggers['05.T25%+C3d+TO7d'] = ('timeout', 7)
    triggers['06.T25%+C3d+TO5d'] = ('timeout', 5)
    triggers['07.ScoreEMA3+T25%'] = ('score_ema', None)

    timings = {
        'MonthEnd':   lambda s, e: gen_month_end(s, e, di),
        'MonthStart': lambda s, e: gen_month_start(s, e, di),
        'Day5':       lambda s, e: gen_fixed_day(s, e, di, 5),
        'Day10':      lambda s, e: gen_fixed_day(s, e, di, 10),
        'Day15':      lambda s, e: gen_fixed_day(s, e, di, 15),
        'Day20':      lambda s, e: gen_fixed_day(s, e, di, 20),
        'Day25':      lambda s, e: gen_fixed_day(s, e, di, 25),
    }

    total = len(triggers) * len(timings) * len(START_DATES)
    print(f"\nRunning {len(triggers)} triggers x {len(timings)} timings x {len(START_DATES)} periods = {total} backtests...\n")

    rows = []
    n = 0
    for trig_name, (ttype, tparam) in triggers.items():
        for timing_name, tgen in timings.items():
            for start in START_DATES:
                n += 1
                try:
                    rebal_dates = tgen(start, END_DATE)
                    res = run_bt(data, ind, start, END_DATE, rebal_dates,
                                 trigger_type=ttype, trigger_param=tparam)
                    m = metrics(res['Value'])
                    rows.append({
                        'Trigger': trig_name, 'Timing': timing_name, 'Start': start,
                        'CAGR': m['cagr'], 'MDD': m['mdd'], 'Sharpe': m['sharpe'],
                        'Sortino': m['sortino'], 'Calmar': m['calmar'], 'Final': m['final'],
                    })
                except Exception as e:
                    pass
                if n % 50 == 0: print(f"  {n}/{total}...")

    df = pd.DataFrame(rows)
    print(f"\n{'='*80}")
    print(f"RESULTS: {len(df)} backtests completed")
    print(f"{'='*80}\n")

    # Overall ranking by trigger
    agg = df.groupby('Trigger').agg(
        AvgSharpe=('Sharpe','mean'),
        AvgCAGR=('CAGR','mean'),
        AvgMDD=('MDD','mean'),
        AvgSortino=('Sortino','mean'),
        MedianSharpe=('Sharpe','median'),
    ).sort_values('AvgSharpe', ascending=False)
    
    print("=== OVERALL RANKING (by Avg Sharpe) ===")
    print(agg.to_string(float_format=lambda x: f"{x:.3f}"))
    
    # Timing stability: spread (max - min avg sharpe across timings)
    print("\n=== TIMING STABILITY ===")
    timing_agg = df.groupby(['Trigger','Timing'])['Sharpe'].mean().unstack()
    stability = timing_agg.max(axis=1) - timing_agg.min(axis=1)
    stability = stability.sort_values()
    print("Spread (lower = more stable):")
    for trig, spread in stability.items():
        avg = agg.loc[trig, 'AvgSharpe']
        print(f"  {trig:30s}  Spread={spread:.3f}  AvgSharpe={avg:.3f}")

    # How many timings beat baseline (None)?
    baseline_by_timing = df[df['Trigger']=='00.None'].groupby('Timing')['Sharpe'].mean()
    print("\n=== TIMINGS BEATING BASELINE ===")
    for trig in triggers:
        trig_by_timing = df[df['Trigger']==trig].groupby('Timing')['Sharpe'].mean()
        beats = sum(trig_by_timing.get(t, 0) > baseline_by_timing.get(t, 0) for t in baseline_by_timing.index)
        avg = agg.loc[trig, 'AvgSharpe'] if trig in agg.index else 0
        print(f"  {trig:30s}  {beats}/7 timings beat baseline  AvgSharpe={avg:.3f}")

    # Per-timing detail
    print("\n=== PER-TIMING AVG SHARPE ===")
    print(timing_agg.to_string(float_format=lambda x: f"{x:.3f}"))

    df.to_csv('rebal_trigger_v2_results.csv', index=False)
    print("\nResults saved to rebal_trigger_v2_results.csv")


if __name__ == '__main__':
    main()
