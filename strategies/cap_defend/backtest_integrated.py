#!/usr/bin/env python3
"""
Integrated Stock(60%) + Coin(40%) Backtest — V12 vs V14
=======================================================
주식 리밸런싱 시점에 주식/코인 간 60:40도 재조정.
주식: VT+EEM Dual Canary → 공격(Mom+Sharpe Top3+3 EW) / 수비(6M best 1)
코인: strategy_engine 로직 재현 (canary, health, selection, weighting, risk)
"""

import os, sys
import numpy as np
import pandas as pd
import yfinance as yf

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    Params, load_data as load_coin_data, _close_to, get_sma,
    calc_ret as engine_calc_ret, get_vol, get_price,
    resolve_canary, get_healthy_coins, select_coins,
    compute_weights, apply_risk, get_universe_for_date,
    check_coin_dd_exit,
)

# ─── Constants ───────────────────────────────────────────────────
OFFENSIVE = ['SPY', 'QQQ', 'EFA', 'EEM', 'VT', 'VEA', 'GLD', 'PDBC',
             'QUAL', 'MTUM', 'IQLT', 'IMTM']
DEFENSIVE = ['IEF', 'BIL', 'BNDX', 'GLD', 'PDBC']
CANARY = ['VT', 'EEM']
ALL_STOCK_TICKERS = list(set(OFFENSIVE + DEFENSIVE + CANARY))
STOCK_RATIO, COIN_RATIO = 0.60, 0.40
CASH_BUFFER = 0.02
START_DATE = '2018-01-01'
END_DATE = '2025-06-30'
TX_STOCK = 0.001
TX_COIN = 0.004


# ─── Stock Data ──────────────────────────────────────────────────
def load_stock_prices(start='2016-01-01'):
    cache_dir = os.path.join(os.path.dirname(__file__), 'data', 'stock_cache')
    os.makedirs(cache_dir, exist_ok=True)
    prices = {}
    for t in ALL_STOCK_TICKERS:
        cache_file = os.path.join(cache_dir, f'{t}.csv')
        if os.path.exists(cache_file):
            try:
                s = pd.read_csv(cache_file, index_col=0, parse_dates=True).squeeze()
                if isinstance(s, pd.Series) and isinstance(s.index, pd.DatetimeIndex) and len(s) > 100:
                    age_days = (pd.Timestamp.now() - pd.Timestamp(os.path.getmtime(cache_file),
                                unit='s')).days
                    if age_days < 1:
                        prices[t] = s
                        continue
            except Exception:
                pass
        try:
            df = yf.download(t, start=start, progress=False, auto_adjust=True)
            if df is not None and len(df) > 0:
                # Handle MultiIndex columns from yfinance v1.0+
                if isinstance(df.columns, pd.MultiIndex):
                    close = df['Close'][t]
                else:
                    close = df['Close']
                close.to_csv(cache_file)
                prices[t] = close
        except Exception as e:
            print(f"  ⚠️ {t}: {e}")
    return prices


# ─── Stock Strategy ──────────────────────────────────────────────
def _calc_ret(s, n):
    if len(s) < n + 1: return np.nan
    return s.iloc[-1] / s.iloc[-n-1] - 1

def _calc_wmom(s):
    if len(s) < 253: return -np.inf
    return 0.5*_calc_ret(s,63) + 0.3*_calc_ret(s,126) + 0.2*_calc_ret(s,252)

def _calc_sharpe(s, w=126):
    if len(s) < w: return -np.inf
    r = s.pct_change().iloc[-w:]
    return r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0

def stock_signal(stock_prices, date):
    """Returns stock weights dict."""
    def get_to(t):
        s = stock_prices.get(t)
        if s is None: return pd.Series(dtype=float)
        return s[s.index <= date]

    vt, eem = get_to('VT'), get_to('EEM')
    if len(vt)<200 or len(eem)<200:
        return {'Cash':1.0}

    risk_on = (vt.iloc[-1] > vt.rolling(200).mean().iloc[-1] and
               eem.iloc[-1] > eem.rolling(200).mean().iloc[-1])

    if risk_on:
        scores = []
        for t in OFFENSIVE:
            p = get_to(t)
            if len(p)>=253:
                scores.append((t, _calc_wmom(p), _calc_sharpe(p,126)))
        if not scores: return {'Cash':1.0}
        df = pd.DataFrame(scores, columns=['T','Mom','Sh']).set_index('T')
        picks = list(set(df.nlargest(3,'Mom').index.tolist() +
                         df.nlargest(3,'Sh').index.tolist()))
        return {t:1.0/len(picks) for t in picks}
    else:
        results = []
        for t in DEFENSIVE:
            p = get_to(t)
            r = _calc_ret(p,126) if len(p)>126 else np.nan
            if pd.notna(r): results.append((t,r))
        if not results: return {'Cash':1.0}
        best = max(results, key=lambda x:x[1])
        if best[1]<0: return {'Cash':1.0}
        return {best[0]:1.0}


# ─── Stock Portfolio Helpers ─────────────────────────────────────
def stock_value(holdings, cash, stock_prices, date):
    v = cash
    for t, shares in holdings.items():
        p = stock_prices.get(t)
        if p is not None:
            ps = p[p.index <= date]
            if len(ps) > 0:
                v += shares * ps.iloc[-1]
    return v

def stock_rebalance(stock_prices, date, target_value, weights):
    """Rebalance stock to target_value with given weights. Returns (holdings, cash)."""
    holdings = {}
    cash = target_value
    for t, w in weights.items():
        if t == 'Cash': continue
        p = stock_prices.get(t)
        if p is None: continue
        ps = p[p.index <= date]
        if len(ps) == 0: continue
        price = ps.iloc[-1]
        if price > 0:
            alloc = target_value * w
            holdings[t] = alloc / price
            cash -= alloc
    return holdings, cash


# ─── Coin Portfolio Helpers ──────────────────────────────────────
def coin_value(holdings, cash, coin_prices, date):
    v = cash
    for t, units in holdings.items():
        v += units * get_price(t, coin_prices, date)
    return v

def coin_rebalance(coin_prices, date, target_value, weights, tx_cost):
    """Rebalance coin portfolio to target_value with given weights."""
    holdings = {}
    cash = target_value
    for t, w in weights.items():
        if t == 'CASH': continue
        p = get_price(t, coin_prices, date)
        if p > 0:
            alloc = target_value * w
            units = alloc * (1 - tx_cost) / p
            holdings[t] = units
            cash -= alloc
    return holdings, max(cash, 0)


# ─── Integrated Backtest ─────────────────────────────────────────
def run_integrated(stock_prices, coin_prices, universe_map, coin_params,
                   initial_capital=10000.0):
    """
    Run daily simulation with 60:40 rebalancing at stock rebal dates.
    """
    btc = coin_prices.get('BTC-USD')
    if btc is None: return None

    spy = stock_prices.get('SPY')
    if spy is None: return None

    # Common trading dates
    stock_dates = set(spy.index)
    coin_dates = set(btc.index)
    common = sorted(stock_dates & coin_dates)
    common = [d for d in common if START_DATE <= str(d.date()) <= END_DATE]
    if len(common) < 2: return None

    # Initialize
    s_holdings = {}
    s_cash = initial_capital * STOCK_RATIO
    c_holdings = {}
    c_cash = initial_capital * COIN_RATIO

    # Coin state (mirrors strategy_engine)
    state = {
        'prev_canary': False, 'canary_off_days': 0,
        'health_fail_streak': {}, 'prev_picks': [],
        'scaled_months': 2, 'month_start_value': initial_capital * COIN_RATIO,
        'high_watermark': initial_capital * COIN_RATIO,
        'crash_cooldown': 0, 'coin_cooldowns': {},
        'recent_port_vals': [], 'prev_month': None,
        'catastrophic_triggered': False, 'risk_force_rebal': False,
        'canary_on_date': None, 'post_flip_refreshed': False,
        'blacklist': {}, 'dd_exit_count': 0,
    }

    prev_month = None
    history = []
    rebal_count = 0

    from strategy_engine import should_rebalance, execute_rebalance, _port_val

    for i, date in enumerate(common):
        cur_month = date.strftime('%Y-%m')
        imc = (prev_month is not None and cur_month != prev_month)
        is_first = (prev_month is None)

        # ── Daily coin state updates ──
        c_pv = coin_value(c_holdings, c_cash, coin_prices, date)
        state['current_port_val'] = c_pv
        state['high_watermark'] = max(state['high_watermark'], c_pv)
        state['recent_port_vals'].append(c_pv)
        if len(state['recent_port_vals']) > 60:
            state['recent_port_vals'] = state['recent_port_vals'][-60:]

        if imc:
            state['month_start_value'] = c_pv
            state['catastrophic_triggered'] = False
            if state['prev_canary']:
                state['scaled_months'] = state.get('scaled_months', 2) + 1

        # Blacklist daily update
        if coin_params.bl_threshold < 0:
            bl = state['blacklist']
            for t in list(bl.keys()):
                bl[t] -= 1
                if bl[t] <= 0: del bl[t]
            from strategy_engine import _close_to as ct
            for t in get_universe_for_date(universe_map, date):
                if t not in bl:
                    c = ct(t, coin_prices, date)
                    if len(c) >= 2 and (c.iloc[-1]/c.iloc[-2]-1) <= coin_params.bl_threshold:
                        bl[t] = coin_params.bl_days

        # DD Exit daily check
        if coin_params.dd_exit_lookback > 0 and c_holdings:
            dd_exits = [t for t in list(c_holdings.keys())
                        if check_coin_dd_exit(t, coin_prices, date,
                                              coin_params.dd_exit_lookback,
                                              coin_params.dd_exit_threshold)]
            if dd_exits:
                for t in dd_exits:
                    p = get_price(t, coin_prices, date)
                    units = c_holdings.pop(t, 0)
                    if units > 0:
                        c_cash += units * p * (1 - coin_params.tx_cost)
                state['dd_exit_count'] += len(dd_exits)
                c_pv = coin_value(c_holdings, c_cash, coin_prices, date)
                state['current_port_val'] = c_pv

        # Coin canary
        canary_on = resolve_canary(coin_prices, date, coin_params, state)
        canary_flipped = (canary_on != state['prev_canary'])

        if canary_on and canary_flipped:
            state['scaled_months'] = 0
            state['canary_on_date'] = date
            state['post_flip_refreshed'] = False
        elif not canary_on and canary_flipped:
            state['canary_on_date'] = None

        state['is_first_day'] = (i == 0)
        state['is_month_change'] = imc
        state['canary_flipped'] = canary_flipped
        state['canary_on'] = canary_on

        universe = get_universe_for_date(universe_map, date)
        if coin_params.bl_threshold < 0:
            universe = [t for t in universe if t not in state['blacklist']]
        state['current_universe'] = universe

        if canary_on:
            healthy = get_healthy_coins(coin_prices, universe, date, coin_params, state)
            state['healthy_count'] = len(healthy)
            state['current_healthy_set'] = set(healthy)
        else:
            healthy = []
            state['healthy_count'] = 0
            state['current_healthy_set'] = set()

        picks = (select_coins(healthy, coin_prices, date, coin_params, state)
                 if canary_on and healthy else [])

        if picks:
            c_weights = compute_weights(picks, coin_prices, date, coin_params, state)
        else:
            c_weights = {'CASH': 1.0}

        c_weights = apply_risk(c_weights, coin_prices, date, coin_params, state)

        # Coin rebal decision
        do_coin_rebal = should_rebalance(c_weights, c_holdings, c_cash,
                                          coin_prices, date, coin_params, state)

        if not do_coin_rebal and coin_params.post_flip_delay > 0 and canary_on:
            flip_date = state.get('canary_on_date')
            if flip_date and not state.get('post_flip_refreshed', False):
                days_since = (date - flip_date).days
                if days_since >= coin_params.post_flip_delay:
                    state['post_flip_refreshed'] = True
                    do_coin_rebal = True

        if coin_params.rebalancing in ('R2','R6','R7','R8','R9') and state.get('catastrophic_triggered'):
            c_weights = {'CASH': 1.0}
            picks = []

        # ── 60:40 Rebalancing at month change ──
        if is_first or imc:
            s_val = stock_value(s_holdings, s_cash, stock_prices, date)
            c_pv = coin_value(c_holdings, c_cash, coin_prices, date)
            total_val = s_val + c_pv

            # Target: 60% stock, 40% coin
            s_target = total_val * STOCK_RATIO
            c_target = total_val * COIN_RATIO

            # Stock rebalance
            s_weights = stock_signal(stock_prices, date)
            # Sell all stock, get cash
            if s_holdings:
                s_cash = s_val * (1 - TX_STOCK)
                s_holdings = {}
            else:
                s_cash = s_val
            # Buy new stock
            s_holdings, s_cash = stock_rebalance(stock_prices, date, s_target, s_weights)

            # Coin: force rebalance with 60:40 capital reallocation
            c_pv_now = coin_value(c_holdings, c_cash, coin_prices, date)
            if abs(c_pv_now - c_target) > 1.0:  # meaningful difference
                # Scale: sell all coins to cash, then rebuy with c_target
                sell_val = 0
                for t, units in c_holdings.items():
                    sell_val += units * get_price(t, coin_prices, date) * (1 - coin_params.tx_cost)
                c_cash = sell_val + c_cash
                c_holdings = {}

                # Now allocate c_target
                excess = c_cash - c_target
                c_cash = c_target  # coin part gets exactly 40%

                # Rebuy coins with target weights
                c_holdings, c_cash = coin_rebalance(coin_prices, date, c_target,
                                                     c_weights, coin_params.tx_cost)
                state['prev_picks'] = picks[:]
                do_coin_rebal = False  # already rebalanced
            rebal_count += 1

        # Coin-only rebalance (mid-month triggers: DD exit, crash, flip)
        if do_coin_rebal:
            c_pv = coin_value(c_holdings, c_cash, coin_prices, date)
            # Sell all
            sell_val = 0
            for t, units in c_holdings.items():
                sell_val += units * get_price(t, coin_prices, date) * (1 - coin_params.tx_cost)
            c_cash = sell_val + c_cash
            c_holdings = {}
            # Rebuy
            c_holdings, c_cash = coin_rebalance(coin_prices, date, c_pv,
                                                 c_weights, coin_params.tx_cost)
            state['prev_picks'] = picks[:]
            rebal_count += 1

        # Daily values
        s_val = stock_value(s_holdings, s_cash, stock_prices, date)
        c_val = coin_value(c_holdings, c_cash, coin_prices, date)

        history.append({
            'Date': date,
            'Total': s_val + c_val,
            'Stock': s_val,
            'Coin': c_val,
        })

        state['prev_canary'] = canary_on
        state['prev_month'] = cur_month
        prev_month = cur_month

    result = pd.DataFrame(history).set_index('Date')
    result.attrs['rebal_count'] = rebal_count
    result.attrs['dd_exit_count'] = state.get('dd_exit_count', 0)
    return result


# ─── Metrics ─────────────────────────────────────────────────────
def compute_metrics(df, col='Total'):
    values = df[col]
    if len(values)<2: return {}
    days = (values.index[-1]-values.index[0]).days
    years = days/365.25
    cagr = (values.iloc[-1]/values.iloc[0])**(1/years)-1 if years>0 else 0
    peak = values.cummax()
    mdd = (values/peak-1).min()
    dr = values.pct_change().dropna()
    sharpe = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
    down = dr[dr<0]
    sortino = dr.mean()/down.std()*np.sqrt(252) if len(down)>1 and down.std()>0 else sharpe
    calmar = cagr/abs(mdd) if mdd!=0 else 0
    return {'CAGR':cagr,'MDD':mdd,'Sharpe':sharpe,'Sortino':sortino,
            'Calmar':calmar,'Final':values.iloc[-1],'Years':years}

def yearly_metrics(df, col='Total'):
    out = {}
    for y in range(df.index[0].year, df.index[-1].year+1):
        mask = df.index.year==y
        if mask.sum()<10: continue
        vals = df[mask][col]
        ret = vals.iloc[-1]/vals.iloc[0]-1
        peak = vals.cummax()
        mdd = (vals/peak-1).min()
        out[y] = {'ret':ret,'mdd':mdd}
    return out


# ─── Coin Params ─────────────────────────────────────────────────
def make_v12():
    return Params(
        canary='K8', vote_smas=(50,), vote_moms=(), vote_threshold=1,
        health='baseline', health_sma=30, health_mom_short=21, health_mom_long=0,
        vol_cap=0.10, selection='S6', weighting='W6', risk='baseline',
        top_n=50, dd_exit_lookback=0, bl_threshold=0.0,
        start_date=START_DATE, end_date=END_DATE,
    )

def make_v14():
    return Params(
        canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1,
        health='HK', health_sma=0, health_mom_short=21, health_mom_long=90,
        vol_cap=0.05, selection='baseline', weighting='baseline', risk='G5',
        top_n=40, dd_exit_lookback=60, dd_exit_threshold=-0.25,
        bl_threshold=-0.15, bl_days=7,
        start_date=START_DATE, end_date=END_DATE,
    )


# ─── Main ────────────────────────────────────────────────────────
def main():
    print("=" * 85)
    print("  주식(60%) + 코인(40%) 통합 백테스트: V12 vs V14")
    print("  월간 리밸런싱 시 60:40 비율 재조정 포함")
    print("=" * 85)

    print("\n📈 주식 데이터 로딩...")
    stock_prices = load_stock_prices()
    print(f"  {len(stock_prices)} ETFs loaded")

    print("\n🪙 코인 데이터 로딩...")
    coin_prices, coin_universe = load_coin_data(top_n=50)
    print(f"  {len(coin_prices)} tickers loaded")

    versions = {'V12': make_v12(), 'V14': make_v14()}
    results = {}

    for name, params in versions.items():
        print(f"\n{'─'*50}")
        print(f"  {name} 통합 시뮬레이션...")
        hist = run_integrated(stock_prices, coin_prices, coin_universe, params)
        if hist is None:
            print(f"  ⚠️ {name} 실패")
            continue

        m = compute_metrics(hist, 'Total')
        sm = compute_metrics(hist, 'Stock')
        cm = compute_metrics(hist, 'Coin')

        results[name] = {
            'total': m, 'stock': sm, 'coin': cm,
            'history': hist, 'yearly': yearly_metrics(hist),
            'rebal_count': hist.attrs.get('rebal_count', 0),
            'dd_exit_count': hist.attrs.get('dd_exit_count', 0),
        }
        print(f"  주식(60%): CAGR {sm['CAGR']:+.1%}  MDD {sm['MDD']:.1%}  Sharpe {sm['Sharpe']:.3f}")
        print(f"  코인(40%): CAGR {cm['CAGR']:+.1%}  MDD {cm['MDD']:.1%}  Sharpe {cm['Sharpe']:.3f}")
        print(f"  통합:      CAGR {m['CAGR']:+.1%}  MDD {m['MDD']:.1%}  Sharpe {m['Sharpe']:.3f}  Final ${m['Final']:,.0f}")

    if len(results) < 2:
        print("결과 부족"); return

    # Summary
    print(f"\n{'='*90}")
    print(f"  통합 포트폴리오 (Stock 60% + Coin 40%) — 월간 60:40 재조정")
    print(f"{'='*90}")
    print(f"  {'':>8} {'CAGR':>8} {'MDD':>8} {'Sharpe':>8} {'Sortino':>9} {'Calmar':>8} {'Rebals':>7} {'Final($)':>10}")
    print(f"  {'─'*70}")
    for n in ['V12','V14']:
        m = results[n]['total']
        rc = results[n]['rebal_count']
        print(f"  {n:>8} {m['CAGR']:>+7.1%} {m['MDD']:>7.1%} {m['Sharpe']:>8.3f} {m['Sortino']:>9.3f} {m['Calmar']:>8.2f} {rc:>7} ${m['Final']:>9,.0f}")

    t12 = results['V12']['total']
    t14 = results['V14']['total']
    print(f"\n  V14 vs V12 개선:")
    print(f"    Sharpe: {t14['Sharpe']-t12['Sharpe']:+.3f}")
    print(f"    MDD:    {t14['MDD']-t12['MDD']:+.1%} ({'개선' if t14['MDD']>t12['MDD'] else '악화'})")
    print(f"    CAGR:   {t14['CAGR']-t12['CAGR']:+.1%}")

    # Component breakdown
    print(f"\n{'='*90}")
    print(f"  자산별 분해")
    print(f"{'='*90}")
    print(f"  {'':>15} {'V12 CAGR':>10} {'V12 MDD':>10} {'V14 CAGR':>10} {'V14 MDD':>10}")
    print(f"  {'─'*55}")
    for label, key in [('주식(60%)', 'stock'), ('코인(40%)', 'coin'), ('통합', 'total')]:
        print(f"  {label:>15} {results['V12'][key]['CAGR']:>+9.1%} {results['V12'][key]['MDD']:>9.1%}"
              f" {results['V14'][key]['CAGR']:>+9.1%} {results['V14'][key]['MDD']:>9.1%}")

    # Yearly
    all_years = set()
    for r in results.values():
        all_years.update(r['yearly'].keys())

    print(f"\n{'='*90}")
    print(f"  연도별 통합 수익률")
    print(f"{'='*90}")
    print(f"  {'Year':>6}  {'V12 Return':>12} {'V12 MDD':>10}  {'V14 Return':>12} {'V14 MDD':>10}  {'Best':>6}")
    print(f"  {'─'*65}")
    for y in sorted(all_years):
        v12y = results['V12']['yearly'].get(y)
        v14y = results['V14']['yearly'].get(y)
        v12r = f"{v12y['ret']:>+10.1%}" if v12y else f"{'N/A':>12}"
        v12m = f"{v12y['mdd']:>9.1%}" if v12y else f"{'N/A':>10}"
        v14r = f"{v14y['ret']:>+10.1%}" if v14y else f"{'N/A':>12}"
        v14m = f"{v14y['mdd']:>9.1%}" if v14y else f"{'N/A':>10}"
        best = ""
        if v12y and v14y:
            best = "V14" if v14y['ret'] > v12y['ret'] else "V12"
        print(f"  {y:>6}  {v12r} {v12m}  {v14r} {v14m}  {best:>6}")

    # Benchmark
    print(f"\n{'='*90}")
    print(f"  벤치마크 비교")
    print(f"{'='*90}")
    spy = stock_prices.get('SPY')
    if spy is not None:
        s = spy[spy.index >= START_DATE]
        s = s[s.index <= END_DATE]
        if len(s)>1:
            yrs = (s.index[-1]-s.index[0]).days/365.25
            cagr = (s.iloc[-1]/s.iloc[0])**(1/yrs)-1
            pk = s.cummax(); mdd = (s/pk-1).min()
            dr = s.pct_change().dropna()
            sh = dr.mean()/dr.std()*np.sqrt(252) if dr.std()>0 else 0
            print(f"  {'SPY B&H':>15}: CAGR {cagr:+.1%}  MDD {mdd:.1%}  Sharpe {sh:.3f}")

    print(f"  {'V12 통합':>15}: CAGR {t12['CAGR']:+.1%}  MDD {t12['MDD']:.1%}  Sharpe {t12['Sharpe']:.3f}")
    print(f"  {'V14 통합':>15}: CAGR {t14['CAGR']:+.1%}  MDD {t14['MDD']:.1%}  Sharpe {t14['Sharpe']:.3f}")


if __name__ == '__main__':
    main()
