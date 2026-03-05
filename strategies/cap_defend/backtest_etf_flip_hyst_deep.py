"""
Hysteresis Band Deep Test
=========================
- 밴드폭 0.25% ~ 5.0% 세밀하게 테스트
- 시작점 2017~2023 (ETF 데이터 최대 활용)
- 7개 타이밍 교차

Usage:
    python3 strategies/cap_defend/backtest_etf_flip_hyst_deep.py
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

# 2017년부터 시작 — 더 긴 기간 테스트
START_DATES = ['2017-06-01','2018-01-01','2019-01-01','2020-01-01',
               '2021-01-01','2022-01-01','2023-01-01']
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
    idx = pd.date_range(start='2016-01-01', end=END_DATE, freq='D')
    data = pd.DataFrame(data_dict).reindex(idx).ffill()
    # Check earliest data availability for canary
    for c in CANARY:
        if c in data:
            first_valid = data[c].first_valid_index()
            print(f"  {c} data from {first_valid.date() if first_valid else 'N/A'}")
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


def build_portfolio(d, ind, risk_on):
    if risk_on:
        rows = [(t, v(ind,t,'mom_w',d), v(ind,t,'sharpe126',d))
                for t in OFFENSIVE if has(ind,t,d)]
        rows = [(t,m,q) for t,m,q in rows if pd.notna(m) and pd.notna(q)]
        if not rows: return {'BIL': 1.0}
        df = pd.DataFrame(rows, columns=['T','M','Q']).set_index('T')
        picks = list(set(df.nlargest(3,'M').index.tolist() + df.nlargest(3,'Q').index.tolist()))
        return {t: 1.0/len(picks) for t in picks}
    else:
        best_t, best_r = 'BIL', -999
        for t in DEFENSIVE:
            r = v(ind, t, 'mom126', d)
            if pd.notna(r) and r > best_r: best_r, best_t = r, t
        if best_r < 0: return {'BIL': 1.0}
        return {best_t: 1.0}


def check_risk_on(d, ind, prev_risk_on, band_pct):
    """Check risk on/off with hysteresis band. band_pct=0 means no hysteresis."""
    for c in CANARY:
        if not has(ind, c, d): return False
        price = v(ind, c, 'price', d)
        sma = v(ind, c, 'sma200', d)
        if pd.isna(price) or pd.isna(sma): return False

        if band_pct == 0:
            if price <= sma: return False
        elif prev_risk_on is None:
            if price <= sma: return False
        elif prev_risk_on:
            if price < sma * (1 - band_pct): return False
        else:
            if price <= sma * (1 + band_pct): return False
    return True


def run_bt(data, ind, start, end, rebal_dates, band_pct=0, capital=10000, tx=0.001):
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.isin(data.index)]

    cash = capital
    hold = {}
    hist = []
    rebals = 0
    flip_count = 0
    prev_risk_on = None

    for today in dates:
        row = data.loc[today]
        pv = cash + sum(u*(row.get(t,0) if pd.notna(row.get(t,0)) else 0) for t,u in hold.items())
        hist.append({'Date': today, 'Value': pv})

        risk_on = check_risk_on(today, ind, prev_risk_on, band_pct)

        # Detect flip
        if prev_risk_on is not None and risk_on != prev_risk_on:
            flip_count += 1
            do_flip = True
        else:
            do_flip = False

        prev_risk_on = risk_on

        if today in rebal_dates or do_flip:
            rebals += 1
            tgt = build_portfolio(today, ind, risk_on)
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
    di = data.dropna(how='all').index

    # --- 세밀한 밴드폭 테스트 ---
    bands = [0, 0.0025, 0.005, 0.0075, 0.01, 0.0125, 0.015, 0.0175,
             0.02, 0.025, 0.03, 0.04, 0.05]
    band_names = {b: f'{b:.2%}' if b > 0 else 'No Guard' for b in bands}

    timings = {
        'MonthEnd':   lambda s, e: gen_month_end(s, e, di),
        'MonthStart': lambda s, e: gen_month_start(s, e, di),
        'Day5':       lambda s, e: gen_fixed_day(s, e, di, 5),
        'Day10':      lambda s, e: gen_fixed_day(s, e, di, 10),
        'Day15':      lambda s, e: gen_fixed_day(s, e, di, 15),
        'Day20':      lambda s, e: gen_fixed_day(s, e, di, 20),
        'Day25':      lambda s, e: gen_fixed_day(s, e, di, 25),
    }

    total = len(bands) * len(timings) * len(START_DATES)
    print(f"\nRunning {len(bands)} bands x {len(timings)} timings x {len(START_DATES)} periods = {total} backtests...\n")

    rows = []
    n = 0
    for band in bands:
        for tname, tgen in timings.items():
            for start in START_DATES:
                n += 1
                rebal_dates = tgen(start, END_DATE)
                res = run_bt(data, ind, start, END_DATE, rebal_dates, band_pct=band)
                m = metrics(res['Value'])
                rows.append({
                    'Band': band_names[band], 'BandPct': band,
                    'Timing': tname, 'Start': start,
                    'CAGR': m['cagr'], 'MDD': m['mdd'], 'Sharpe': m['sharpe'],
                    'Sortino': m['sortino'], 'Calmar': m['calmar'], 'Final': m['final'],
                    'Rebals': res.attrs.get('rebals', 0),
                    'Flips': res.attrs.get('flips', 0),
                })
            if n % 100 == 0:
                print(f"  Progress: {n}/{total} ({n*100//total}%)")

    df = pd.DataFrame(rows)

    # ===========================================================================
    # PART 1: Band × Timing Sharpe matrix (avg across all start dates)
    # ===========================================================================
    pivot = df.groupby(['Band','BandPct','Timing']).agg({
        'Sharpe':'mean','CAGR':'mean','MDD':'mean','Flips':'mean','Rebals':'mean'
    }).reset_index()

    timing_order = ['MonthEnd','MonthStart','Day5','Day10','Day15','Day20','Day25']

    sharpe_matrix = pivot.pivot(index='Band', columns='Timing', values='Sharpe')
    sharpe_matrix = sharpe_matrix.reindex(columns=[t for t in timing_order if t in sharpe_matrix.columns])
    sharpe_matrix['Avg'] = sharpe_matrix[timing_order].mean(axis=1)
    sharpe_matrix['Max'] = sharpe_matrix[timing_order].max(axis=1)
    sharpe_matrix['Min'] = sharpe_matrix[timing_order].min(axis=1)
    sharpe_matrix['Spread'] = sharpe_matrix['Max'] - sharpe_matrix['Min']

    # Add avg flips
    avg_flips = pivot.groupby('Band')['Flips'].mean()
    sharpe_matrix['Flips'] = avg_flips

    # Sort by band pct
    band_order = [band_names[b] for b in bands]
    sharpe_matrix = sharpe_matrix.reindex([b for b in band_order if b in sharpe_matrix.index])

    print(f"\n{'='*170}")
    print(f"  SHARPE MATRIX — Band × Timing (Avg across {len(START_DATES)} periods, start from 2017)")
    print(f"{'='*170}")
    hdr = f"{'Band':<10}"
    for t in timing_order:
        hdr += f" {t:>9}"
    hdr += f" {'|':>2} {'Avg':>6} {'Min':>6} {'Max':>6} {'Spread':>6} {'Flips':>6}"
    print(hdr)
    print(f"{'-'*170}")

    best_avg = sharpe_matrix['Avg'].max()
    for bname, row in sharpe_matrix.iterrows():
        line = f"{bname:<10}"
        for t in timing_order:
            val = row.get(t, 0)
            line += f" {val:>9.3f}"
        mark = ' <-- 현재' if bname == 'No Guard' else (' ***' if row['Avg'] == best_avg else '')
        line += f" {'|':>2} {row['Avg']:>6.3f} {row['Min']:>6.3f} {row['Max']:>6.3f} {row['Spread']:>6.3f} {row['Flips']:>6.1f}{mark}"
        print(line)
    print(f"{'='*170}")

    # ===========================================================================
    # PART 2: CAGR matrix
    # ===========================================================================
    cagr_matrix = pivot.pivot(index='Band', columns='Timing', values='CAGR')
    cagr_matrix = cagr_matrix.reindex(columns=[t for t in timing_order if t in cagr_matrix.columns])
    cagr_matrix['Avg'] = cagr_matrix[timing_order].mean(axis=1)
    cagr_matrix = cagr_matrix.reindex([b for b in band_order if b in cagr_matrix.index])

    print(f"\n{'='*170}")
    print(f"  CAGR MATRIX — Band × Timing")
    print(f"{'='*170}")
    hdr = f"{'Band':<10}"
    for t in timing_order:
        hdr += f" {t:>9}"
    hdr += f" {'|':>2} {'Avg':>7}"
    print(hdr)
    print(f"{'-'*170}")

    for bname, row in cagr_matrix.iterrows():
        line = f"{bname:<10}"
        for t in timing_order:
            val = row.get(t, 0)
            line += f" {val:>8.1%}"
        mark = ' <-- 현재' if bname == 'No Guard' else ''
        line += f" {'|':>2} {row['Avg']:>6.1%}{mark}"
        print(line)
    print(f"{'='*170}")

    # ===========================================================================
    # PART 3: Per-start-date breakdown (avg across timings)
    # ===========================================================================
    by_start = df.groupby(['Band','BandPct','Start']).agg({
        'Sharpe':'mean','CAGR':'mean','MDD':'mean','Flips':'mean'
    }).reset_index()

    print(f"\n{'='*140}")
    print(f"  시작점별 Avg Sharpe (모든 타이밍 평균)")
    print(f"{'='*140}")
    hdr = f"{'Band':<10}"
    for s in START_DATES:
        hdr += f" {s[:7]:>10}"
    hdr += f" {'|':>2} {'Avg':>6} {'Min':>6} {'Max':>6}"
    print(hdr)
    print(f"{'-'*140}")

    for band in bands:
        bname = band_names[band]
        line = f"{bname:<10}"
        vals = []
        for s in START_DATES:
            row = by_start[(by_start['BandPct']==band) & (by_start['Start']==s)]
            if not row.empty:
                v_val = row.iloc[0]['Sharpe']
                vals.append(v_val)
                line += f" {v_val:>10.3f}"
            else:
                line += f" {'N/A':>10}"
        mark = ' <-- 현재' if bname == 'No Guard' else ''
        if vals:
            line += f" {'|':>2} {np.mean(vals):>6.3f} {min(vals):>6.3f} {max(vals):>6.3f}{mark}"
        print(line)
    print(f"{'='*140}")

    # ===========================================================================
    # PART 4: 최적 밴드폭 vs 현재 — 각 타이밍에서 몇 % 밴드가 최적인가
    # ===========================================================================
    print(f"\n{'='*100}")
    print(f"  타이밍별 최적 밴드폭")
    print(f"{'='*100}")
    print(f"  {'Timing':<12} {'Best Band':>10} {'Sharpe':>8} {'vs NoGuard':>10} {'Flips':>7}")
    print(f"  {'-'*60}")
    for t in timing_order:
        t_data = pivot[pivot['Timing']==t].sort_values('Sharpe', ascending=False)
        if t_data.empty: continue
        best = t_data.iloc[0]
        ng = t_data[t_data['Band']=='No Guard']
        ng_s = ng.iloc[0]['Sharpe'] if not ng.empty else 0
        diff = best['Sharpe'] - ng_s
        print(f"  {t:<12} {best['Band']:>10} {best['Sharpe']:>8.3f} {diff:>+10.3f} {best['Flips']:>7.1f}")
    print(f"{'='*100}")

    # ===========================================================================
    # PART 5: 0.75% ~ 1.25% 구간 상세 비교
    # ===========================================================================
    focus = [b for b in bands if 0.005 <= b <= 0.02]
    focus_data = pivot[pivot['BandPct'].isin(focus)]
    focus_avg = focus_data.groupby(['Band','BandPct']).agg({'Sharpe':'mean','CAGR':'mean','Flips':'mean'}).reset_index()
    focus_avg = focus_avg.sort_values('Sharpe', ascending=False)

    ng_data = pivot[pivot['BandPct']==0]
    ng_avg = ng_data['Sharpe'].mean()

    print(f"\n{'='*100}")
    print(f"  0.50% ~ 2.00% 밴드 상세 비교 (전체 타이밍 평균)")
    print(f"{'='*100}")
    print(f"  {'Band':>8} {'Avg Sharpe':>11} {'vs NoGuard':>11} {'Avg CAGR':>10} {'Avg Flips':>10}")
    print(f"  {'-'*60}")
    print(f"  {'NoGuard':>8} {ng_avg:>11.3f} {'baseline':>11} {ng_data['CAGR'].mean():>9.1%} {ng_data['Flips'].mean():>10.1f}")
    print(f"  {'-'*60}")
    for _, r in focus_avg.iterrows():
        diff = r['Sharpe'] - ng_avg
        print(f"  {r['Band']:>8} {r['Sharpe']:>11.3f} {diff:>+11.3f} {r['CAGR']:>9.1%} {r['Flips']:>10.1f}")
    print(f"{'='*100}")


if __name__ == '__main__':
    main()
