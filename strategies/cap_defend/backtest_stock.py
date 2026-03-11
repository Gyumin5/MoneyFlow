#!/usr/bin/env python3
"""
Stock ETF Strategy Backtest
============================
Dual Canary (VT+EEM > SMA200) → 공격/수비 전환
공격: 12종 가중모멘텀 Top3 + Sharpe Top3 합집합, 균등배분
수비: 5종 중 6M수익률 최고 1개 (음수면 현금)
리밸런싱: 월간 (매월 첫 거래일)

Usage:
  python3 backtest_stock.py                    # 기본 실행
  python3 backtest_stock.py --start 2017-01-01 # 시작일 변경
"""

import os, sys, argparse
import numpy as np
import pandas as pd
import yfinance as yf

# ─── Constants ───────────────────────────────────────────────────
OFFENSIVE = ['SPY', 'QQQ', 'EFA', 'EEM', 'VT', 'VEA', 'GLD', 'PDBC',
             'QUAL', 'MTUM', 'IQLT', 'IMTM']
DEFENSIVE = ['IEF', 'BIL', 'BNDX', 'GLD', 'PDBC']
CANARY = ['VT', 'EEM']
ALL_TICKERS = list(set(OFFENSIVE + DEFENSIVE + CANARY))
CANARY_MA = 200
TX_COST = 0.001
HYSTERESIS_BAND = 0.01  # 1%


# ─── Data ────────────────────────────────────────────────────────
def load_prices(start='2014-01-01'):
    """Download ETF prices, cache locally."""
    cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'stock_cache')
    os.makedirs(cache_dir, exist_ok=True)
    prices = {}
    for t in ALL_TICKERS:
        cache_file = os.path.join(cache_dir, f'{t}.csv')
        if os.path.exists(cache_file):
            try:
                s = pd.read_csv(cache_file, index_col=0, parse_dates=True).squeeze()
                if isinstance(s, pd.Series) and isinstance(s.index, pd.DatetimeIndex) and len(s) > 100:
                    age = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(cache_file),
                            unit='s')).days
                    if age < 1:
                        prices[t] = s
                        continue
            except Exception:
                pass
        try:
            df = yf.download(t, start=start, progress=False, auto_adjust=True)
            if df is not None and len(df) > 0:
                if isinstance(df.columns, pd.MultiIndex):
                    close = df['Close'][t]
                else:
                    close = df['Close']
                close.to_csv(cache_file)
                prices[t] = close
        except Exception as e:
            print(f"  ⚠️ {t}: {e}")
    return prices


# ─── Indicators ──────────────────────────────────────────────────
def calc_ret(s, n):
    if len(s) < n + 1: return np.nan
    return s.iloc[-1] / s.iloc[-n-1] - 1

def calc_weighted_mom(s):
    """Weighted momentum: 50% 3M + 30% 6M + 20% 12M"""
    if len(s) < 253: return -np.inf
    return 0.5*calc_ret(s,63) + 0.3*calc_ret(s,126) + 0.2*calc_ret(s,252)

def calc_sharpe(s, window=126):
    if len(s) < window: return -np.inf
    r = s.pct_change().iloc[-window:]
    return r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0


# ─── Strategy ────────────────────────────────────────────────────
def get_signal(prices, date, prev_risk_on=None, hysteresis=0.0):
    """
    Returns (weights_dict, risk_on_bool, mode_str)
    hysteresis: 0.01 = 1% band
    """
    def to(t):
        s = prices.get(t)
        if s is None: return pd.Series(dtype=float)
        return s[s.index <= date]

    vt, eem = to('VT'), to('EEM')
    if len(vt) < CANARY_MA or len(eem) < CANARY_MA:
        return {'Cash':1.0}, False, 'No Data'

    vt_sma = vt.rolling(CANARY_MA).mean().iloc[-1]
    eem_sma = eem.rolling(CANARY_MA).mean().iloc[-1]

    if hysteresis > 0 and prev_risk_on is not None:
        if prev_risk_on:
            # Stay ON unless below lower band
            risk_on = not (vt.iloc[-1] < vt_sma*(1-hysteresis) or
                           eem.iloc[-1] < eem_sma*(1-hysteresis))
        else:
            # Turn ON only above upper band
            risk_on = (vt.iloc[-1] > vt_sma*(1+hysteresis) and
                       eem.iloc[-1] > eem_sma*(1+hysteresis))
    else:
        risk_on = (vt.iloc[-1] > vt_sma and eem.iloc[-1] > eem_sma)

    if risk_on:
        scores = []
        for t in OFFENSIVE:
            p = to(t)
            if len(p) >= 253:
                scores.append((t, calc_weighted_mom(p), calc_sharpe(p, 126)))
        if not scores:
            return {'Cash':1.0}, True, 'No Offensive'

        df = pd.DataFrame(scores, columns=['T','Mom','Sh']).set_index('T')
        top_m = df.nlargest(3, 'Mom').index.tolist()
        top_s = df.nlargest(3, 'Sh').index.tolist()
        picks = list(set(top_m + top_s))
        return {t:1.0/len(picks) for t in picks}, True, f'Offense({len(picks)})'
    else:
        results = []
        for t in DEFENSIVE:
            p = to(t)
            r = calc_ret(p, 126) if len(p) > 126 else np.nan
            if pd.notna(r): results.append((t, r))
        if not results:
            return {'Cash':1.0}, False, 'No Defensive'
        best = max(results, key=lambda x: x[1])
        if best[1] < 0:
            return {'Cash':1.0}, False, 'Defense(Cash)'
        return {best[0]:1.0}, False, f'Defense({best[0]})'


# ─── Backtest Engine ─────────────────────────────────────────────
def run_backtest(prices, start_date='2017-01-01', end_date='2026-12-31',
                 initial=10000.0, hysteresis=0.0):
    """Run stock strategy backtest."""
    spy = prices.get('SPY')
    if spy is None: return None

    dates = spy.index[(spy.index >= start_date) & (spy.index <= end_date)]
    if len(dates) < 2: return None

    holdings = {}  # {ticker: shares}
    cash = initial
    prev_month = None
    prev_risk_on = None
    history = []
    rebal_count = 0
    flip_count = 0

    for date in dates:
        cur_month = date.strftime('%Y-%m')
        imc = (prev_month is not None and cur_month != prev_month)
        is_first = (prev_month is None)

        # Portfolio value
        pv = cash
        for t, shares in holdings.items():
            p = prices.get(t)
            if p is not None:
                ps = p[p.index <= date]
                if len(ps) > 0:
                    pv += shares * ps.iloc[-1]

        # Monthly rebalance
        if is_first or imc:
            weights, risk_on, mode = get_signal(prices, date, prev_risk_on, hysteresis)

            if prev_risk_on is not None and prev_risk_on != risk_on:
                flip_count += 1

            # Sell all
            if holdings:
                sell_cash = 0
                for t, shares in holdings.items():
                    p = prices.get(t)
                    if p is not None:
                        ps = p[p.index <= date]
                        if len(ps) > 0:
                            sell_cash += shares * ps.iloc[-1]
                cash = (sell_cash + cash) * (1 - TX_COST)
                holdings = {}
            pv = cash

            # Buy new
            for t, w in weights.items():
                if t == 'Cash': continue
                p = prices.get(t)
                if p is None: continue
                ps = p[p.index <= date]
                if len(ps) == 0: continue
                price = ps.iloc[-1]
                if price > 0:
                    alloc = pv * w
                    holdings[t] = alloc / price
                    cash -= alloc

            prev_risk_on = risk_on
            rebal_count += 1

        # Recalc PV
        pv = cash
        for t, shares in holdings.items():
            p = prices.get(t)
            if p is not None:
                ps = p[p.index <= date]
                if len(ps) > 0:
                    pv += shares * ps.iloc[-1]

        history.append({'Date': date, 'Value': pv, 'Mode': mode if (is_first or imc) else ''})
        prev_month = cur_month

    df = pd.DataFrame(history).set_index('Date')
    df.attrs['rebal_count'] = rebal_count
    df.attrs['flip_count'] = flip_count
    return df


# ─── Metrics ─────────────────────────────────────────────────────
def compute_metrics(df, col='Value'):
    v = df[col]
    if len(v) < 2: return {}
    days = (v.index[-1] - v.index[0]).days
    years = days / 365.25
    cagr = (v.iloc[-1]/v.iloc[0])**(1/years)-1 if years > 0 else 0
    peak = v.cummax()
    mdd = (v/peak-1).min()
    dr = v.pct_change().dropna()
    sharpe = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    down = dr[dr<0]
    sortino = dr.mean()/down.std()*np.sqrt(252) if len(down)>1 and down.std()>0 else sharpe
    calmar = cagr/abs(mdd) if mdd!=0 else 0
    win_rate = (dr > 0).sum() / len(dr) if len(dr) > 0 else 0
    return {'CAGR':cagr, 'MDD':mdd, 'Sharpe':sharpe, 'Sortino':sortino,
            'Calmar':calmar, 'Final':v.iloc[-1], 'Years':years, 'WinRate':win_rate}

def yearly_metrics(df, col='Value'):
    out = {}
    for y in range(df.index[0].year, df.index[-1].year+1):
        mask = df.index.year == y
        if mask.sum() < 10: continue
        vals = df[mask][col]
        ret = vals.iloc[-1]/vals.iloc[0]-1
        pk = vals.cummax()
        mdd = (vals/pk-1).min()
        out[y] = {'ret':ret, 'mdd':mdd}
    return out


# ─── Main ────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--start', default='2017-01-01')
    parser.add_argument('--end', default='2026-12-31')
    args = parser.parse_args()

    print("=" * 85)
    print("  주식 ETF 전략 백테스트 (V11: Dual Canary + Mom/Sharpe + Defensive)")
    print("=" * 85)

    print("\n📈 데이터 로딩...")
    prices = load_prices()
    print(f"  {len(prices)} ETFs loaded")

    # Run variants
    variants = {
        'Base (No Hyst)': 0.0,
        'Hyst 0.5%': 0.005,
        'Hyst 1.0%': 0.01,
        'Hyst 1.5%': 0.015,
        'Hyst 2.0%': 0.02,
        'Hyst 3.0%': 0.03,
    }

    results = {}
    for name, hyst in variants.items():
        df = run_backtest(prices, args.start, args.end, hysteresis=hyst)
        if df is None:
            print(f"  ⚠️ {name} 실패")
            continue
        m = compute_metrics(df)
        results[name] = {'metrics': m, 'history': df, 'yearly': yearly_metrics(df)}

    # Summary
    print(f"\n{'='*90}")
    print(f"  주식 전략 Hysteresis 비교 ({args.start} ~)")
    print(f"{'='*90}")
    print(f"  {'Variant':<18} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'Flips':>6} {'Rebals':>7} {'Final($)':>10}")
    print(f"  {'─'*86}")
    for name in variants:
        if name not in results: continue
        m = results[name]['metrics']
        df = results[name]['history']
        print(f"  {name:<18} {m['CAGR']:>+7.1%} {m['MDD']:>7.1%} {m['Sharpe']:>8.3f} {m['Sortino']:>9.3f}"
              f" {m['Calmar']:>8.2f} {df.attrs.get('flip_count',0):>6} {df.attrs.get('rebal_count',0):>7} ${m['Final']:>9,.0f}")

    # Yearly for base and best
    base_name = 'Base (No Hyst)'
    best_name = max(results, key=lambda k: results[k]['metrics']['Sharpe'])

    if base_name in results:
        print(f"\n{'='*90}")
        print(f"  연도별 수익률: {base_name} vs {best_name}")
        print(f"{'='*90}")
        all_years = set()
        for r in results.values():
            all_years.update(r['yearly'].keys())

        print(f"  {'Year':>6}  {'Base Ret':>10} {'Base MDD':>10}  {'Best Ret':>10} {'Best MDD':>10}  {'Winner':>8}")
        print(f"  {'─'*62}")
        for y in sorted(all_years):
            by = results[base_name]['yearly'].get(y)
            hy = results[best_name]['yearly'].get(y) if best_name != base_name else None
            br = f"{by['ret']:>+9.1%}" if by else f"{'N/A':>10}"
            bm = f"{by['mdd']:>9.1%}" if by else f"{'N/A':>10}"
            hr = f"{hy['ret']:>+9.1%}" if hy else f"{'N/A':>10}"
            hm = f"{hy['mdd']:>9.1%}" if hy else f"{'N/A':>10}"
            winner = ""
            if by and hy:
                winner = best_name.split('(')[0].strip() if hy['ret'] > by['ret'] else 'Base'
            print(f"  {y:>6}  {br} {bm}  {hr} {hm}  {winner:>8}")

    # Benchmark: SPY Buy & Hold
    spy = prices.get('SPY')
    if spy is not None:
        s = spy[(spy.index >= args.start) & (spy.index <= args.end)]
        if len(s) > 1:
            yrs = (s.index[-1]-s.index[0]).days/365.25
            spy_cagr = (s.iloc[-1]/s.iloc[0])**(1/yrs)-1
            pk = s.cummax(); spy_mdd = (s/pk-1).min()
            dr = s.pct_change().dropna()
            spy_sh = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
            print(f"\n  벤치마크:")
            print(f"  {'SPY B&H':<18} CAGR {spy_cagr:+.1%}  MDD {spy_mdd:.1%}  Sharpe {spy_sh:.3f}")
            if base_name in results:
                m = results[base_name]['metrics']
                print(f"  {'Strategy':<18} CAGR {m['CAGR']:+.1%}  MDD {m['MDD']:.1%}  Sharpe {m['Sharpe']:.3f}")

    # Mode distribution
    if base_name in results:
        df = results[base_name]['history']
        modes = df[df['Mode'] != '']['Mode']
        off_count = modes.str.contains('Offense').sum()
        def_count = modes.str.contains('Defense').sum()
        total = off_count + def_count
        print(f"\n  모드 분포: 공격 {off_count}회 ({off_count/total*100:.0f}%) / 수비 {def_count}회 ({def_count/total*100:.0f}%)")


if __name__ == '__main__':
    main()
