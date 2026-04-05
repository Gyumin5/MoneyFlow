#!/usr/bin/env python3
"""자산배분 비율 x 선물전략 전수 테스트.

모든 선물 전략 후보에 대해 다양한 주식/현물/선물 비율을 테스트.
"""

import os, sys, time
import numpy as np, pandas as pd
from dataclasses import replace
from itertools import combinations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from coin_engine import (
    load_universe, load_all_prices, filter_universe, DEFENSE_TICKERS,
)
from coin_helpers import B
from backtest_official import run_coin_backtest, COIN_VERSIONS, STOCK_VERSIONS
from stock_engine import (
    SP, load_prices as load_stock_prices, precompute as stock_precompute,
    _init as stock_init, run_bt, ALL_TICKERS,
)
import stock_engine as tsi
from backtest_official import check_crash_vt
from backtest_futures_full import load_data
from futures_live_config import CURRENT_STRATEGIES
from futures_ensemble_engine import SingleAccountEngine, combine_targets


ENGINE_KWARGS = dict(
    leverage=5.0,
    stop_kind="prev_close_pct",
    stop_pct=0.15,
    stop_gate="cash_guard",
    stop_gate_cash_threshold=0.34,
    per_coin_leverage_mode="cap_mom_blend_543_cash",
    leverage_floor=3.0,
    leverage_mid=4.0,
    leverage_ceiling=5.0,
    leverage_cash_threshold=0.34,
    leverage_partial_cash_threshold=0.0,
    leverage_count_floor_max=2,
    leverage_count_mid_max=4,
    leverage_canary_floor_gap=0.015,
    leverage_canary_mid_gap=0.04,
    leverage_canary_high_gap=0.08,
    leverage_canary_sma_bars=1200,
    leverage_mom_lookback_bars=24 * 30,
    leverage_vol_lookback_bars=24 * 90,
)

FIXED_BASE = dict(
    canary_hyst=0.015, drift_threshold=0.0,
    dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0, n_snapshots=3,
)

# ═══ 선물 전략 정의 ═══

# 현행 실거래
STRAT_LIVE = {
    "live_1h1": dict(
        interval="1h", sma_bars=168, mom_short_bars=36, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.80,
        snap_interval_bars=27, **FIXED_BASE),
    "live_4h1": dict(
        interval="4h", sma_bars=240, mom_short_bars=10, mom_long_bars=30,
        health_mode="mom1vol", vol_mode="daily", vol_threshold=0.05,
        snap_interval_bars=120, **FIXED_BASE),
    "live_4h2": dict(
        interval="4h", sma_bars=120, mom_short_bars=20, mom_long_bars=120,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
        snap_interval_bars=21, **FIXED_BASE),
}

# d0.05 후보 4전략
STRAT_CAND4 = {
    "4h_d005": dict(
        interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="daily", vol_threshold=0.05,
        snap_interval_bars=60, **FIXED_BASE),
    "2h_b60_S240": dict(
        interval="2h", sma_bars=240, mom_short_bars=20, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
        snap_interval_bars=120, **FIXED_BASE),
    "2h_b60_S120": dict(
        interval="2h", sma_bars=120, mom_short_bars=20, mom_long_bars=720,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
        snap_interval_bars=120, **FIXED_BASE),
    "4h_b60_M20_120": dict(
        interval="4h", sma_bars=240, mom_short_bars=20, mom_long_bars=120,
        health_mode="mom2vol", vol_mode="bar", vol_threshold=0.60,
        snap_interval_bars=21, **FIXED_BASE),
}

# 개별 전략 (단독 테스트용)
INDIVIDUAL_STRATS = {
    "solo_4h1": {"live_4h1": STRAT_LIVE["live_4h1"]},
    "solo_4h2": {"live_4h2": STRAT_LIVE["live_4h2"]},
    "solo_1h1": {"live_1h1": STRAT_LIVE["live_1h1"]},
    "solo_4h_d005": {"4h_d005": STRAT_CAND4["4h_d005"]},
}

# 2전략 조합들
STRAT_2_COMBOS = {}
all_strats = {**STRAT_LIVE, **STRAT_CAND4}
strat_names = list(all_strats.keys())
for a, b in combinations(strat_names, 2):
    key = f"{a}+{b}"
    STRAT_2_COMBOS[key] = {a: all_strats[a], b: all_strats[b]}

# 3전략 (현행 + 주요 변형)
STRAT_3_COMBOS = {
    "live(4h1+4h2+1h1)": STRAT_LIVE,
    "4h1+4h2+4h_d005": {k: all_strats[k] for k in ["live_4h1", "live_4h2", "4h_d005"]},
    "4h1+1h1+4h_d005": {k: all_strats[k] for k in ["live_4h1", "live_1h1", "4h_d005"]},
    "4h2+1h1+4h_d005": {k: all_strats[k] for k in ["live_4h2", "live_1h1", "4h_d005"]},
    "4h1+2h_S240+2h_S120": {k: all_strats[k] for k in ["live_4h1", "2h_b60_S240", "2h_b60_S120"]},
    "4h_d005+2h_S240+2h_S120": {k: all_strats[k] for k in ["4h_d005", "2h_b60_S240", "2h_b60_S120"]},
    "4h_d005+2h_S240+4h_M20_120": {k: all_strats[k] for k in ["4h_d005", "2h_b60_S240", "4h_b60_M20_120"]},
}

# 모든 선물 전략 세트
ALL_FUT_SETS = {
    **{f"1s:{k}": v for k, v in INDIVIDUAL_STRATS.items()},
    "3s:live(현행)": STRAT_LIVE,
    "4s:d005후보": STRAT_CAND4,
    **{f"3s:{k}": v for k, v in STRAT_3_COMBOS.items()},
}

# 비율 조합
RATIOS = [
    (70, 20, 10),
    (60, 30, 10),
    (60, 25, 15),
    (60, 20, 20),
    (55, 30, 15),
    (50, 35, 15),
    (50, 30, 20),
    (50, 25, 25),
    (45, 30, 25),
    (40, 30, 30),
    (40, 40, 20),
]

START = '2020-10-01'
END = '2026-03-28'


def generate_trace(data, cfg, start, end):
    from backtest_futures_full import run
    run_cfg = dict(cfg)
    interval = run_cfg.pop("interval")
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=start, end_date=end, _trace=trace, **run_cfg)
    return trace


def get_coin_equity(prices, um, start, end):
    cfg = COIN_VERSIONS['V18']
    top_n = cfg['params'].get('top_n', 40)
    p = B(**cfg['params'])
    p.start_date = start; p.end_date = end
    r = run_coin_backtest(prices, um.get(top_n, um[40]), (1, 10, 19),
                          dd_lookback=cfg['dd_lookback'], dd_threshold=cfg['dd_threshold'],
                          bl_drop=cfg['bl_drop'], bl_days=cfg['bl_days'],
                          drift_threshold=cfg['drift_threshold'],
                          post_flip_delay=cfg['post_flip_delay'],
                          params_base=p, defense=cfg.get('defense', False))
    return r['equity_curve']


def get_stock_equity(start, end):
    base = STOCK_VERSIONS['V17']
    sp = replace(base, start=start, end=end,
                 _n_tranches=4, tranche_days=(1, 8, 15, 22))
    from stock_engine import run_bt as srun_bt
    dfs = []
    for day in sp.tranche_days:
        tp = SP(**{k: v for k, v in sp.__dict__.items()})
        tp._anchor = day
        tp._n_tranches = 1
        tp.capital = sp.capital / len(sp.tranche_days)
        df = srun_bt(tsi._g_prices, tsi._g_ind, tp)
        if df is not None:
            dfs.append(df)
    if not dfs:
        return None
    merged = dfs[0].copy()
    for df in dfs[1:]:
        common = merged.index.intersection(df.index)
        merged.loc[common, 'Value'] += df.loc[common, 'Value']
    return merged['Value']


def get_futures_equity(data, strategies, start, end):
    combo = {k: 1.0 / len(strategies) for k in strategies}
    intervals = set(cfg['interval'] for cfg in strategies.values())
    bars_1h, funding_1h = data['1h']
    all_dates = bars_1h['BTC'].index[
        (bars_1h['BTC'].index >= start) & (bars_1h['BTC'].index <= end)]
    traces = {name: generate_trace(data, cfg, start, end)
              for name, cfg in strategies.items()}
    combined = combine_targets({k: traces[k] for k in combo}, combo, all_dates)
    engine = SingleAccountEngine(bars_1h, funding_1h, **ENGINE_KWARGS)
    m = engine.run(combined)
    eq = m['_equity']
    eq_daily = eq.resample('D').last().dropna()
    return eq_daily, m


def monthly_rebalance(equity_curves, weights, start, end):
    returns = {}
    for name, eq in equity_curves.items():
        eq = eq.sort_index()
        ret = eq.pct_change().fillna(0)
        returns[name] = ret
    common = None
    for name, ret in returns.items():
        if common is None:
            common = ret.index
        else:
            common = common.intersection(ret.index)
    common = common.sort_values()
    common = common[(common >= start) & (common <= end)]
    if len(common) < 2:
        return None
    capital = 10000.0
    allocations = {name: capital * w for name, w in weights.items()}
    portfolio_values = []
    prev_month = None
    for date in common:
        cur_month = date.strftime('%Y-%m')
        if prev_month is not None and cur_month != prev_month:
            total = sum(allocations.values())
            allocations = {name: total * w for name, w in weights.items()}
        for name in allocations:
            r = returns[name].get(date, 0)
            allocations[name] *= (1 + r)
        total = sum(allocations.values())
        portfolio_values.append({'Date': date, 'Value': total})
        prev_month = cur_month
    df = pd.DataFrame(portfolio_values).set_index('Date')
    eq = df['Value']
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    if yrs <= 0 or eq.iloc[-1] <= 0:
        return None
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1/yrs) - 1
    dr = eq.pct_change().dropna()
    sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
    mdd = (eq / eq.cummax() - 1).min()
    cal = cagr / abs(mdd) if mdd != 0 else 0
    return {'Sharpe': sh, 'CAGR': cagr, 'MDD': mdd, 'Calmar': cal}


def main():
    t0 = time.time()
    print("=" * 100)
    print("자산배분 비율 x 선물전략 전수 테스트")
    print(f"기간: {START} ~ {END}")
    print("=" * 100)

    # === Load data ===
    print("\n데이터 로딩...")
    um_raw = load_universe()
    coin_um = {40: filter_universe(um_raw, 40), 50: filter_universe(um_raw, 50)}
    all_t = set()
    for fm in coin_um.values():
        for ts in fm.values(): all_t.update(ts)
    all_t.update(['BTC-USD', 'ETH-USD'])
    all_t.update(DEFENSE_TICKERS)
    coin_prices = load_all_prices(list(all_t))

    stock_prices = load_stock_prices(ALL_TICKERS, start='2005-01-01')
    stock_ind = stock_precompute(stock_prices)
    stock_init(stock_prices, stock_ind)
    tsi.check_crash = check_crash_vt

    fut_data = {iv: load_data(iv) for iv in ["4h", "2h", "1h"]}
    print(f"  완료 ({time.time()-t0:.1f}s)")

    # === Equity curves ===
    print("\n주식/현물 equity curve 생성...")
    coin_eq = get_coin_equity(coin_prices, coin_um, START, END)
    stock_eq = get_stock_equity(START, END)
    print(f"  완료 ({time.time()-t0:.1f}s)")

    # === 선물 전략별 equity curve ===
    print("\n선물 전략 equity curve 생성...")
    fut_equities = {}
    fut_metrics = {}
    for name, strats in ALL_FUT_SETS.items():
        eq, m = get_futures_equity(fut_data, strats, START, END)
        fut_equities[name] = eq
        fut_metrics[name] = m
        print(f"  {name:<40s} Sh={m['Sharpe']:.2f} CAGR={m['CAGR']:+.0%} MDD={m['MDD']:+.0%} Cal={m['Cal']:.2f} Liq={m['Liq']}")
    print(f"  완료 ({time.time()-t0:.1f}s)")

    # === Phase 1: 선물 전략 단독 순위 ===
    print("\n" + "=" * 100)
    print("Phase 1: 선물 전략 단독 성과 (5x 레버리지)")
    print("=" * 100)
    ranked = sorted(fut_metrics.items(), key=lambda x: -x[1]['Cal'])
    print(f"\n  {'#':>3s} {'전략':<40s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s} {'Liq':>4s}")
    print(f"  {'-'*75}")
    for i, (name, m) in enumerate(ranked, 1):
        print(f"  {i:>3d} {name:<40s} {m['Sharpe']:>7.2f} {m['CAGR']:>+8.0%} {m['MDD']:>+8.0%} {m['Cal']:>7.2f} {m['Liq']:>4d}")

    # === Phase 2: 비율별 포트폴리오 (전 선물 전략) ===
    print("\n" + "=" * 100)
    print("Phase 2: 비율 x 선물전략 전수 조합")
    print("=" * 100)

    results = []
    for ratio in RATIOS:
        s_pct, c_pct, f_pct = ratio
        s_w, c_w, f_w = s_pct/100, c_pct/100, f_pct/100
        for fut_name, fut_eq in fut_equities.items():
            curves = {'stock': stock_eq, 'coin': coin_eq, 'futures': fut_eq}
            weights = {'stock': s_w, 'coin': c_w, 'futures': f_w}
            m = monthly_rebalance(curves, weights, START, END)
            if m:
                results.append({
                    'ratio': f"{s_pct}/{c_pct}/{f_pct}",
                    'futures': fut_name,
                    **m
                })

    # baseline (60/40 without futures)
    bl_m = monthly_rebalance({'stock': stock_eq, 'coin': coin_eq},
                              {'stock': 0.6, 'coin': 0.4}, START, END)

    df_results = pd.DataFrame(results)

    # 비율별 최고 Cal
    print(f"\n  비율별 최고 Calmar 전략:")
    print(f"\n  {'비율':<12s} {'선물전략':<40s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s}")
    print(f"  {'-'*80}")
    print(f"  {'60/40/0':<12s} {'(선물 없음, baseline)':<40s} {bl_m['Sharpe']:>7.3f} {bl_m['CAGR']:>+8.1%} {bl_m['MDD']:>+8.1%} {bl_m['Calmar']:>7.2f}")
    print(f"  {'-'*80}")

    for ratio in RATIOS:
        rstr = f"{ratio[0]}/{ratio[1]}/{ratio[2]}"
        subset = df_results[df_results['ratio'] == rstr].sort_values('Calmar', ascending=False)
        if len(subset) > 0:
            best = subset.iloc[0]
            print(f"  {rstr:<12s} {best['futures']:<40s} {best['Sharpe']:>7.3f} {best['CAGR']:>+8.1%} {best['MDD']:>+8.1%} {best['Calmar']:>7.2f}")

    # 비율별 최고 Sharpe
    print(f"\n  비율별 최고 Sharpe 전략:")
    print(f"\n  {'비율':<12s} {'선물전략':<40s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s}")
    print(f"  {'-'*80}")
    print(f"  {'60/40/0':<12s} {'(선물 없음, baseline)':<40s} {bl_m['Sharpe']:>7.3f} {bl_m['CAGR']:>+8.1%} {bl_m['MDD']:>+8.1%} {bl_m['Calmar']:>7.2f}")
    print(f"  {'-'*80}")

    for ratio in RATIOS:
        rstr = f"{ratio[0]}/{ratio[1]}/{ratio[2]}"
        subset = df_results[df_results['ratio'] == rstr].sort_values('Sharpe', ascending=False)
        if len(subset) > 0:
            best = subset.iloc[0]
            print(f"  {rstr:<12s} {best['futures']:<40s} {best['Sharpe']:>7.3f} {best['CAGR']:>+8.1%} {best['MDD']:>+8.1%} {best['Calmar']:>7.2f}")

    # === Phase 3: 전체 결과 Top 20 (Calmar 순) ===
    print("\n" + "=" * 100)
    print("Phase 3: 전체 Top 20 (Calmar)")
    print("=" * 100)

    top20_cal = df_results.sort_values('Calmar', ascending=False).head(20)
    print(f"\n  {'#':>3s} {'비율':<12s} {'선물전략':<40s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s}")
    print(f"  {'-'*82}")
    for i, (_, row) in enumerate(top20_cal.iterrows(), 1):
        print(f"  {i:>3d} {row['ratio']:<12s} {row['futures']:<40s} {row['Sharpe']:>7.3f} {row['CAGR']:>+8.1%} {row['MDD']:>+8.1%} {row['Calmar']:>7.2f}")

    # Top 20 Sharpe
    print(f"\n  전체 Top 20 (Sharpe):")
    top20_sh = df_results.sort_values('Sharpe', ascending=False).head(20)
    print(f"\n  {'#':>3s} {'비율':<12s} {'선물전략':<40s} {'Sharpe':>7s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s}")
    print(f"  {'-'*82}")
    for i, (_, row) in enumerate(top20_sh.iterrows(), 1):
        print(f"  {i:>3d} {row['ratio']:<12s} {row['futures']:<40s} {row['Sharpe']:>7.3f} {row['CAGR']:>+8.1%} {row['MDD']:>+8.1%} {row['Calmar']:>7.2f}")

    # === Phase 4: 현행 vs 후보 직접 비교 (모든 비율) ===
    print("\n" + "=" * 100)
    print("Phase 4: 현행 vs 후보(4s:d005) 직접 비교")
    print("=" * 100)

    print(f"\n  {'비율':<12s} {'|':>1s} {'현행 Cal':>9s} {'현행 Sh':>9s} {'현행 MDD':>9s} {'|':>1s} {'후보 Cal':>9s} {'후보 Sh':>9s} {'후보 MDD':>9s} {'|':>1s} {'Cal차':>7s}")
    print(f"  {'-'*90}")

    for ratio in RATIOS:
        rstr = f"{ratio[0]}/{ratio[1]}/{ratio[2]}"
        s_w, c_w, f_w = ratio[0]/100, ratio[1]/100, ratio[2]/100

        live_m = monthly_rebalance(
            {'stock': stock_eq, 'coin': coin_eq, 'futures': fut_equities['3s:live(현행)']},
            {'stock': s_w, 'coin': c_w, 'futures': f_w}, START, END)
        cand_m = monthly_rebalance(
            {'stock': stock_eq, 'coin': coin_eq, 'futures': fut_equities['4s:d005후보']},
            {'stock': s_w, 'coin': c_w, 'futures': f_w}, START, END)

        if live_m and cand_m:
            diff = cand_m['Calmar'] - live_m['Calmar']
            print(f"  {rstr:<12s} | {live_m['Calmar']:>9.2f} {live_m['Sharpe']:>9.3f} {live_m['MDD']:>+9.1%} | "
                  f"{cand_m['Calmar']:>9.2f} {cand_m['Sharpe']:>9.3f} {cand_m['MDD']:>+9.1%} | {diff:>+7.2f}")

    # === Phase 5: 서브기간 Top 5 ===
    print("\n" + "=" * 100)
    print("Phase 5: 서브기간 안정성 (Top 5 Calmar 조합)")
    print("=" * 100)

    mid = '2023-07-01'
    top5 = df_results.sort_values('Calmar', ascending=False).head(5)

    for _, row in top5.iterrows():
        rparts = row['ratio'].split('/')
        s_w, c_w, f_w = int(rparts[0])/100, int(rparts[1])/100, int(rparts[2])/100
        fut_eq = fut_equities[row['futures']]
        curves = {'stock': stock_eq, 'coin': coin_eq, 'futures': fut_eq}
        weights = {'stock': s_w, 'coin': c_w, 'futures': f_w}

        m_front = monthly_rebalance(curves, weights, START, mid)
        m_back = monthly_rebalance(curves, weights, mid, END)

        if m_front and m_back:
            ratio_imbal = max(m_front['Calmar'], m_back['Calmar']) / min(m_front['Calmar'], m_back['Calmar']) if min(m_front['Calmar'], m_back['Calmar']) > 0 else 99
            print(f"\n  {row['ratio']} + {row['futures']}")
            print(f"    전체: Cal={row['Calmar']:.2f} Sh={row['Sharpe']:.3f} CAGR={row['CAGR']:+.1%} MDD={row['MDD']:+.1%}")
            print(f"    전반: Cal={m_front['Calmar']:.2f} Sh={m_front['Sharpe']:.3f} CAGR={m_front['CAGR']:+.1%} MDD={m_front['MDD']:+.1%}")
            print(f"    후반: Cal={m_back['Calmar']:.2f} Sh={m_back['Sharpe']:.3f} CAGR={m_back['CAGR']:+.1%} MDD={m_back['MDD']:+.1%}")
            print(f"    비율: {ratio_imbal:.2f}x")

    print(f"\n총 소요: {time.time()-t0:.1f}s")
    print(f"총 조합 수: {len(df_results)}")


if __name__ == '__main__':
    main()
