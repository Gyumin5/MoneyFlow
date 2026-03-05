"""
ETF Intra-month Rebalancing Trigger Test
=========================================
Signal flip(1% hysteresis) 외에, Risk-On 상태에서 종목 변경 시
리밸런싱을 트리거하는 다양한 전략을 테스트.

Triggers:
  0. None — 정기 리밸런싱 + signal flip만 (현재)
  1. Any Change — 종목 1개라도 바뀌면 즉시 리밸런싱
  2. Score Buffer — 신규 종목이 기존 종목보다 X% 이상 높을 때만
  3. Momentum Decay — 보유 종목 단기 모멘텀 급락 시
  4. Turnover Threshold — 포트폴리오 턴오버 N% 이상일 때
  5. Confirmation Days — 종목 변경이 D일 연속 유지될 때
  6. Combos — 4+5 등 조합

Usage:
    python3 strategies/cap_defend/backtest_etf_rebal_trigger.py
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
HYST_BAND = 0.01  # 1% hysteresis 고정


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
        d['mom21']  = p / p.shift(21) - 1
        d['mom63']  = p / p.shift(63) - 1
        d['mom126'] = p / p.shift(126) - 1
        d['mom252'] = p / p.shift(252) - 1
        d['mom_w']  = 0.5*d['mom63'] + 0.3*d['mom126'] + 0.2*d['mom252']
        rm = dr.rolling(126).mean()
        rs = dr.rolling(126).std()
        d['sharpe126'] = (rm / rs.replace(0, np.nan)) * np.sqrt(252)
        # trailing high (20d)
        d['high20'] = p.rolling(20).max()
        ind[col] = d
    return ind


def v(ind, ticker, col, date):
    if ticker not in ind: return np.nan
    try: return ind[ticker][col].loc[date]
    except: return np.nan

def has(ind, ticker, date):
    return pd.notna(v(ind, ticker, 'mom252', date))


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


def get_picks(d, ind):
    """Get current top picks (attack mode)."""
    rows = [(t, v(ind,t,'mom_w',d), v(ind,t,'sharpe126',d))
            for t in OFFENSIVE if has(ind,t,d)]
    rows = [(t,m,q) for t,m,q in rows if pd.notna(m) and pd.notna(q)]
    if not rows: return set(), {}
    df = pd.DataFrame(rows, columns=['T','M','Q']).set_index('T')
    mom_top3 = set(df.nlargest(3,'M').index.tolist())
    qual_top3 = set(df.nlargest(3,'Q').index.tolist())
    picks = mom_top3 | qual_top3
    # Score map for buffer comparison
    scores = {}
    for t in df.index:
        scores[t] = {'mom_w': df.loc[t,'M'], 'sharpe126': df.loc[t,'Q'],
                      'mom_rank': 0, 'qual_rank': 0}
    mom_ranked = df.sort_values('M', ascending=False).index.tolist()
    qual_ranked = df.sort_values('Q', ascending=False).index.tolist()
    for i, t in enumerate(mom_ranked): scores[t]['mom_rank'] = i+1
    for i, t in enumerate(qual_ranked): scores[t]['qual_rank'] = i+1
    return picks, scores


def check_risk_on(d, ind, prev_risk_on):
    """1% hysteresis check."""
    for c in CANARY:
        if not has(ind, c, d): return False
        price = v(ind, c, 'price', d)
        sma = v(ind, c, 'sma200', d)
        if pd.isna(price) or pd.isna(sma): return False
        if prev_risk_on is None:
            if price <= sma: return False
        elif prev_risk_on:
            if price < sma * (1 - HYST_BAND): return False
        else:
            if price <= sma * (1 + HYST_BAND): return False
    return True


def build_defense(d, ind):
    best_t, best_r = 'BIL', -999
    for t in DEFENSIVE:
        r = v(ind, t, 'mom126', d)
        if pd.notna(r) and r > best_r: best_r, best_t = r, t
    if best_r < 0: return {'BIL': 1.0}
    return {best_t: 1.0}


def run_bt(data, ind, start, end, rebal_dates, trigger_type='none', trigger_param=None,
           capital=10000, tx=0.001):
    """
    trigger_type:
      'none'         — 정기 + signal flip만
      'any_change'   — 종목 1개라도 바뀌면
      'score_buf'    — 신규>기존 * (1+buf) 일 때, param=buf_pct
      'mom_decay'    — 보유종목 mom21 < threshold, param=threshold
      'trailing_stop'— 보유종목 20d high 대비 -X%, param=stop_pct (음수)
      'turnover'     — 턴오버 >= threshold, param=threshold (0~1)
      'confirm'      — 종목변경 D일 연속, param=D
      'turn+confirm' — 턴오버+연속, param=(turnover_thr, confirm_days)
    """
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.isin(data.index)]

    cash = capital
    hold = {}  # {ticker: units}
    hist = []
    rebals = 0
    flip_count = 0
    trigger_count = 0
    prev_risk_on = None
    current_picks = set()

    # Confirmation state
    pending_change = None  # set of new picks that differ
    confirm_counter = 0

    for today in dates:
        row = data.loc[today]
        pv = cash + sum(u*(row.get(t,0) if pd.notna(row.get(t,0)) else 0) for t,u in hold.items())
        hist.append({'Date': today, 'Value': pv})

        risk_on = check_risk_on(today, ind, prev_risk_on)

        # Signal flip
        do_flip = False
        if prev_risk_on is not None and risk_on != prev_risk_on:
            do_flip = True
            flip_count += 1
        prev_risk_on = risk_on

        # Check intra-month trigger (only in risk-on mode, not on flip/scheduled day)
        do_trigger = False
        if risk_on and not do_flip and today not in rebal_dates:
            new_picks, scores = get_picks(today, ind)

            if trigger_type == 'any_change':
                if current_picks and new_picks != current_picks:
                    do_trigger = True

            elif trigger_type == 'score_buf':
                buf = trigger_param
                if current_picks and new_picks != current_picks:
                    new_in = new_picks - current_picks
                    old_out = current_picks - new_picks
                    # Check if any new entrant beats any departing by buf%
                    if new_in and old_out:
                        for nt in new_in:
                            for ot in old_out:
                                ns_m = scores.get(nt, {}).get('mom_w', -999)
                                os_m = scores.get(ot, {}).get('mom_w', -999)
                                ns_q = scores.get(nt, {}).get('sharpe126', -999)
                                os_q = scores.get(ot, {}).get('sharpe126', -999)
                                if (pd.notna(ns_m) and pd.notna(os_m) and ns_m > os_m * (1+buf)) or \
                                   (pd.notna(ns_q) and pd.notna(os_q) and ns_q > os_q * (1+buf)):
                                    do_trigger = True
                                    break
                            if do_trigger: break

            elif trigger_type == 'mom_decay':
                threshold = trigger_param
                if current_picks:
                    for t in current_picks:
                        m21 = v(ind, t, 'mom21', today)
                        if pd.notna(m21) and m21 < threshold:
                            do_trigger = True
                            break

            elif trigger_type == 'trailing_stop':
                stop_pct = trigger_param  # e.g. -0.08
                if current_picks:
                    for t in current_picks:
                        price = v(ind, t, 'price', today)
                        high20 = v(ind, t, 'high20', today)
                        if pd.notna(price) and pd.notna(high20) and high20 > 0:
                            drawdown = price / high20 - 1
                            if drawdown < stop_pct:
                                do_trigger = True
                                break

            elif trigger_type == 'turnover':
                thr = trigger_param
                if current_picks and new_picks:
                    # Calculate weight-based turnover
                    all_tickers = current_picks | new_picks
                    cur_w = {t: 1.0/len(current_picks) if t in current_picks else 0 for t in all_tickers}
                    new_w = {t: 1.0/len(new_picks) if t in new_picks else 0 for t in all_tickers}
                    turnover = sum(abs(new_w[t] - cur_w[t]) for t in all_tickers) / 2
                    if turnover >= thr:
                        do_trigger = True

            elif trigger_type == 'confirm':
                days = trigger_param
                if current_picks and new_picks != current_picks:
                    if new_picks == pending_change:
                        confirm_counter += 1
                    else:
                        pending_change = new_picks.copy()
                        confirm_counter = 1
                    if confirm_counter >= days:
                        do_trigger = True
                        confirm_counter = 0
                        pending_change = None
                else:
                    confirm_counter = 0
                    pending_change = None

            elif trigger_type == 'turn+confirm':
                thr, days = trigger_param
                if current_picks and new_picks:
                    all_tickers = current_picks | new_picks
                    cur_w = {t: 1.0/len(current_picks) if t in current_picks else 0 for t in all_tickers}
                    new_w = {t: 1.0/len(new_picks) if t in new_picks else 0 for t in all_tickers}
                    turnover = sum(abs(new_w[t] - cur_w[t]) for t in all_tickers) / 2
                    if turnover >= thr and new_picks != current_picks:
                        if new_picks == pending_change:
                            confirm_counter += 1
                        else:
                            pending_change = new_picks.copy()
                            confirm_counter = 1
                        if confirm_counter >= days:
                            do_trigger = True
                            confirm_counter = 0
                            pending_change = None
                    else:
                        confirm_counter = 0
                        pending_change = None

        if do_trigger:
            trigger_count += 1

        # Rebalance
        should_rebal = today in rebal_dates or do_flip or do_trigger
        if should_rebal:
            rebals += 1
            if risk_on:
                picks, _ = get_picks(today, ind)
                if picks:
                    tgt = {t: 1.0/len(picks) for t in picks}
                    current_picks = picks
                else:
                    tgt = {'BIL': 1.0}
                    current_picks = set()
            else:
                tgt = build_defense(today, ind)
                current_picks = set()

            amt = pv * (1 - tx)
            cash, hold = amt, {}
            for t, w in tgt.items():
                p = row.get(t,0) if pd.notna(row.get(t,0)) else 0
                if p > 0:
                    a = amt * w; hold[t] = a / p; cash -= a

        # Update picks on scheduled rebalance even without trigger
        elif today in rebal_dates or do_flip:
            pass  # already handled above
        elif risk_on and not current_picks:
            # First day in risk-on, set picks
            picks, _ = get_picks(today, ind)
            current_picks = picks

    result = pd.DataFrame(hist).set_index('Date')
    result.attrs['rebals'] = rebals
    result.attrs['flips'] = flip_count
    result.attrs['triggers'] = trigger_count
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
    di = data.dropna(how='all').index

    # --- Trigger variants ---
    triggers = {}

    # 0. None (현재: 정기+flip만)
    triggers['00.None (현재)'] = ('none', None)

    # 1. Any Change
    triggers['01.Any Change'] = ('any_change', None)

    # 2. Score Buffer
    for buf in [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]:
        triggers[f'02.ScoreBuf {buf:.0%}'] = ('score_buf', buf)

    # 3. Momentum Decay
    for thr in [0, -0.02, -0.05, -0.08]:
        triggers[f'03.MomDecay {thr:.0%}'] = ('mom_decay', thr)

    # 3b. Trailing Stop
    for stop in [-0.05, -0.08, -0.10, -0.15]:
        triggers[f'03b.TrailStop {stop:.0%}'] = ('trailing_stop', stop)

    # 4. Turnover Threshold
    for thr in [0.10, 0.15, 0.20, 0.25, 0.30]:
        triggers[f'04.Turnover {thr:.0%}'] = ('turnover', thr)

    # 5. Confirmation Days
    for d in [2, 3, 5, 7, 10]:
        triggers[f'05.Confirm {d}d'] = ('confirm', d)

    # 6. Combos: Turnover + Confirm
    for thr, d in [(0.15, 2), (0.15, 3), (0.20, 2), (0.20, 3), (0.25, 3), (0.30, 3)]:
        triggers[f'06.T{thr:.0%}+C{d}d'] = ('turn+confirm', (thr, d))

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
                        'Rebals': res.attrs.get('rebals', 0),
                        'Flips': res.attrs.get('flips', 0),
                        'Triggers': res.attrs.get('triggers', 0),
                    })
                except Exception as e:
                    rows.append({
                        'Trigger': trig_name, 'Timing': timing_name, 'Start': start,
                        'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0,
                        'Calmar': 0, 'Final': capital, 'Rebals': 0, 'Flips': 0, 'Triggers': 0,
                    })
            if n % 200 == 0:
                print(f"  Progress: {n}/{total} ({n*100//total}%)")

    df = pd.DataFrame(rows)

    # ===========================================================================
    # PART 1: Sharpe matrix — Trigger × Timing
    # ===========================================================================
    pivot = df.groupby(['Trigger','Timing']).agg({
        'Sharpe':'mean','CAGR':'mean','MDD':'mean',
        'Rebals':'mean','Flips':'mean','Triggers':'mean'
    }).reset_index()

    timing_order = ['MonthEnd','MonthStart','Day5','Day10','Day15','Day20','Day25']
    sharpe_matrix = pivot.pivot(index='Trigger', columns='Timing', values='Sharpe')
    sharpe_matrix = sharpe_matrix.reindex(columns=[t for t in timing_order if t in sharpe_matrix.columns])
    sharpe_matrix['Avg'] = sharpe_matrix[timing_order].mean(axis=1)
    sharpe_matrix['Min'] = sharpe_matrix[timing_order].min(axis=1)
    sharpe_matrix['Max'] = sharpe_matrix[timing_order].max(axis=1)
    sharpe_matrix['Spread'] = sharpe_matrix['Max'] - sharpe_matrix['Min']

    # Add trigger count
    avg_trigs = pivot.groupby('Trigger')['Triggers'].mean()
    sharpe_matrix['Trigs'] = avg_trigs

    sharpe_matrix = sharpe_matrix.sort_values('Avg', ascending=False)

    bl_avg = sharpe_matrix.loc['00.None (현재)', 'Avg'] if '00.None (현재)' in sharpe_matrix.index else 0

    print(f"\n{'='*175}")
    print(f"  SHARPE MATRIX — Trigger × Timing (Avg across {len(START_DATES)} periods)")
    print(f"{'='*175}")
    hdr = f"{'#':>3} {'Trigger':<25}"
    for t in timing_order:
        hdr += f" {t:>9}"
    hdr += f" {'|':>2} {'Avg':>6} {'Min':>6} {'Max':>6} {'Sprd':>6} {'Trigs':>6}"
    print(hdr)
    print(f"{'-'*175}")

    for i, (tname, row) in enumerate(sharpe_matrix.iterrows(), 1):
        line = f"{i:>3} {tname:<25}"
        for t in timing_order:
            val = row.get(t, 0)
            line += f" {val:>9.3f}"
        mark = ' <--' if '현재' in str(tname) else (' *' if row['Avg'] > bl_avg else '')
        line += f" {'|':>2} {row['Avg']:>6.3f} {row['Min']:>6.3f} {row['Max']:>6.3f} {row['Spread']:>6.3f} {row['Trigs']:>6.1f}{mark}"
        print(line)
    print(f"{'='*175}")

    # ===========================================================================
    # PART 2: CAGR matrix
    # ===========================================================================
    cagr_matrix = pivot.pivot(index='Trigger', columns='Timing', values='CAGR')
    cagr_matrix = cagr_matrix.reindex(columns=[t for t in timing_order if t in cagr_matrix.columns])
    cagr_matrix['Avg'] = cagr_matrix[timing_order].mean(axis=1)
    cagr_matrix = cagr_matrix.reindex(sharpe_matrix.index)

    print(f"\n{'='*145}")
    print(f"  CAGR MATRIX — Trigger × Timing")
    print(f"{'='*145}")
    hdr = f"{'#':>3} {'Trigger':<25}"
    for t in timing_order:
        hdr += f" {t:>9}"
    hdr += f" {'|':>2} {'Avg':>7}"
    print(hdr)
    print(f"{'-'*145}")

    for i, (tname, row) in enumerate(cagr_matrix.iterrows(), 1):
        line = f"{i:>3} {tname:<25}"
        for t in timing_order:
            val = row.get(t, 0)
            line += f" {val:>8.1%}"
        mark = ' <--' if '현재' in str(tname) else ''
        line += f" {'|':>2} {row['Avg']:>6.1%}{mark}"
        print(line)
    print(f"{'='*145}")

    # ===========================================================================
    # PART 3: Winners — which triggers beat baseline at all timings?
    # ===========================================================================
    bl_by_timing = pivot[pivot['Trigger']=='00.None (현재)'].set_index('Timing')

    print(f"\n{'='*120}")
    print(f"  현재(None) 대비 승리 타이밍 수")
    print(f"{'='*120}")

    wins = {}
    for tname in triggers:
        if '현재' in tname: continue
        t_by_timing = pivot[pivot['Trigger']==tname].set_index('Timing')
        win_count = 0
        win_timings = []
        for t in timing_order:
            if t in t_by_timing.index and t in bl_by_timing.index:
                if t_by_timing.loc[t, 'Sharpe'] > bl_by_timing.loc[t, 'Sharpe']:
                    win_count += 1
                    win_timings.append(t)
        wins[tname] = (win_count, win_timings)

    sorted_wins = sorted(wins.items(), key=lambda x: (-x[1][0], x[0]))
    for tname, (cnt, tlist) in sorted_wins:
        if cnt > 0:
            avg_s = sharpe_matrix.loc[tname, 'Avg'] if tname in sharpe_matrix.index else 0
            spread = sharpe_matrix.loc[tname, 'Spread'] if tname in sharpe_matrix.index else 0
            trigs = sharpe_matrix.loc[tname, 'Trigs'] if tname in sharpe_matrix.index else 0
            all_mark = ' *** ALL ***' if cnt == len(timing_order) else ''
            print(f"  {tname:<25} {cnt}/{len(timing_order)}  Avg={avg_s:.3f}  Sprd={spread:.3f}  Trigs={trigs:.0f}  ({', '.join(tlist)}){all_mark}")

    print(f"{'='*120}")

    # ===========================================================================
    # PART 4: Category summary
    # ===========================================================================
    print(f"\n{'='*120}")
    print(f"  카테고리별 최고 성과")
    print(f"{'='*120}")

    categories = [
        ('00.', 'None (현재)'),
        ('01.', 'Any Change'),
        ('02.', 'Score Buffer'),
        ('03.', 'Momentum Decay'),
        ('03b.', 'Trailing Stop'),
        ('04.', 'Turnover'),
        ('05.', 'Confirmation'),
        ('06.', 'Combo (Turn+Confirm)'),
    ]

    print(f"  {'Category':<22} {'Best Variant':<25} {'Avg Sharpe':>10} {'Avg CAGR':>9} {'Spread':>8} {'Trigs':>6}")
    print(f"  {'-'*90}")
    for prefix, cat_name in categories:
        cat = sharpe_matrix[sharpe_matrix.index.str.startswith(prefix)]
        if cat.empty: continue
        best = cat.iloc[0]
        best_name = cat.index[0]
        print(f"  {cat_name:<22} {best_name:<25} {best['Avg']:>10.3f} "
              f"{cagr_matrix.loc[best_name, 'Avg']:>8.1%} {best['Spread']:>8.3f} {best['Trigs']:>6.1f}")

    print(f"{'='*120}")

    # ===========================================================================
    # PART 5: Timing stability ranking
    # ===========================================================================
    print(f"\n{'='*110}")
    print(f"  타이밍 안정성 순위 (Spread 낮을수록 안정)")
    print(f"{'='*110}")
    stability = sharpe_matrix[['Avg','Min','Max','Spread','Trigs']].sort_values('Spread')
    print(f"  {'#':>3} {'Trigger':<25} {'Avg':>8} {'Min':>8} {'Max':>8} {'Spread':>8} {'Trigs':>7}")
    print(f"  {'-'*80}")
    for i, (tname, row) in enumerate(stability.head(15).iterrows(), 1):
        mark = ' <--' if '현재' in str(tname) else ''
        print(f"  {i:>3} {tname:<25} {row['Avg']:>8.3f} {row['Min']:>8.3f} {row['Max']:>8.3f} "
              f"{row['Spread']:>8.3f} {row['Trigs']:>7.1f}{mark}")
    print(f"{'='*110}")


if __name__ == '__main__':
    main()
