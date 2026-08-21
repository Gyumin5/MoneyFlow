"""선물 L 적용 시점 비교 — 진입 시점 고정(현행) vs 매일 재조정 (2026-08-21).

가설: 목표 L(min(BTC_cap, 코인별 K2))이 매일 재계산되는데 라이브·BT 는 진입 시점 L 을 유지하고
리밸런싱 하는 날에만 갱신한다. 목표 L 이 바뀐 날 바로 명목을 맞추면(매일 재조정) 상승장에선
노출을 더 빨리 키우고 조정장에선 더 빨리 줄여서 위험조정 성과가 나아질 수 있다.
반대 가설: L 플립이 잦으면 명목 조정 체결비용·슬리피지가 누적돼 순손해.

축: 그것 하나만. 선정·헬스·카나리·스냅샷·드리프트·비용 전부 동일.
변형은 정본 엔진 backtest_futures_v25.py 의 env 토글 DAILY_LEV=on 으로 구현했다.
결론(기각) 후 SSoT 오염 방지를 위해 엔진 패치는 되돌렸다 — 재현하려면 먼저
  git apply strategies/cap_defend/research/bt_fut_daily_lev.engine.patch
를 적용해야 한다. 적용 없이 돌리면 두 팔이 모두 현행이 되어 차이 0 이 나온다(가짜 동률).
증거금은 고정하고 노출만 바꾸며 델타 명목에 tx+슬리피지를 물린다.

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_fut_daily_lev.py
출력: state/<job>/ 로 리다이렉트해서 쓸 것 (원시 로그는 리포트가 아니다).
"""
from __future__ import annotations
import os
import sys
import time
from collections import defaultdict

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP)
sys.path.insert(0, HERE)

START = "2020-10-01"
END = "2026-08-20"
WIN_SIZES = [504, 756, 1008]
STRIDES = [63, 126, 252]


def run_fut(daily_lev: bool, sn=95, n_snap=5):
    """라이브 V25 선물 sleeve 설정 그대로. daily_lev 만 바꾼다."""
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    os.environ['DAILY_LEV'] = 'on' if daily_lev else 'off'
    bars, funding = load_data('D')
    k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1 / 3,
                tx_cost=0.0006, maint_rate=0.004,
                sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
                canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
                health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                n_snapshots=n_snap, snap_interval_bars=sn,
                start_date=START, end_date=END)
    os.environ['DAILY_LEV'] = 'off'
    return m


def run_spot_live():
    """현물 sleeve 라이브 V24 설정 (전 유니버스)."""
    from unified_backtest import run as bt_run, load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    bars, _ = load_data('D')
    m = bt_run(bars, _, interval='D', asset_type='spot', leverage=1.0, tx_cost=0.004,
               start_date=START, end_date=END,
               sma_bars=42, mom_short_bars=20, mom_long_bars=127,
               vol_threshold=0.05, vol_mode='daily',
               n_snapshots=7, snap_interval_bars=217,
               canary_hyst=0.015, drift_threshold=0.10, post_flip_delay=5,
               universe_size=3, cap=1 / 3, selection='greedy',
               stop_kind='none', stop_pct=0.0,
               dd_lookback=60, dd_threshold=-99.0,
               bl_drop=-99.0, bl_days=7, crash_threshold=-99.0,
               health_mode='mom2vol')
    return m.get('_equity')


def run_stock_live():
    """주식 sleeve 라이브 V25 설정."""
    from bt_stock_mom3 import run_multi_3mom
    from bt_stock_coin_v3 import precompute
    from stock_engine import load_prices, ALL_TICKERS
    import bt_stock_coin_v3 as bcv3
    bcv3.OFF_R7 = ("SPY", "QQQ", "VEA", "EEM", "GLD", "PDBC", "VNQ")
    pm = load_prices(ALL_TICKERS, start="2005-01-01")
    pdf = pd.DataFrame(pm)
    pdf = pdf[~pdf.index.duplicated(keep='first')].sort_index()
    pdf = pdf[pdf.index.normalize() == pdf.index]
    ranked, mom_off, mom_def, canary = precompute(pdf, [30, 72, 230], [42, 63, 126])
    return run_multi_3mom(pdf, ranked, mom_off, mom_def, canary,
                          pd.Timestamp(START), pd.Timestamp(END), anchor=0,
                          drift_thr=0.05, cash_buf=0.07, ms=30, mid=72, ml=230,
                          snap_int=69, n_snaps=3)


def build_alloc(eq_st, eq_sp, eq_fu, w_st=0.60, w_sp=0.25, w_fu=0.15):
    common = eq_st.index.intersection(eq_sp.index).intersection(eq_fu.index)
    r = (w_st * eq_st.loc[common].pct_change().fillna(0)
         + w_sp * eq_sp.loc[common].pct_change().fillna(0)
         + w_fu * eq_fu.loc[common].pct_change().fillna(0))
    return (1 + r).cumprod()


def metrics(eq):
    eq = eq.dropna()
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    cagr = (eq.iloc[-1] / eq.iloc[0]) ** (1 / yrs) - 1
    mdd = float((eq / eq.cummax() - 1).min())
    rets = eq.pct_change().dropna()
    sh = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    return dict(CAGR=cagr * 100, MDD=mdd * 100, Sharpe=sh,
                Cal=cagr / abs(mdd) if mdd < 0 else 0)


def window_rs(eq_dict):
    """윈도우 rank-sum (홀드아웃 금지 규약). 낮은 avg_rank 가 우세."""
    common = None
    for s in eq_dict.values():
        common = s.index if common is None else common.intersection(s.index)
    common = sorted(common)
    sums = defaultdict(float)
    wins = defaultdict(int)
    n = 0
    for size in WIN_SIZES:
        for stride in STRIDES:
            for i in range(0, len(common) - size, stride):
                d0, d1 = common[i], common[i + size - 1]
                cals = {}
                for k, s in eq_dict.items():
                    seg = s.loc[d0:d1].dropna()
                    if len(seg) < 30:
                        cals[k] = np.nan
                        continue
                    yrs = (seg.index[-1] - seg.index[0]).days / 365.25
                    if yrs <= 0:
                        cals[k] = np.nan
                        continue
                    cagr = (seg.iloc[-1] / seg.iloc[0]) ** (1 / yrs) - 1
                    mdd = float((seg / seg.cummax() - 1).min())
                    cals[k] = cagr / abs(mdd) if mdd < 0 else 0
                if any(np.isnan(v) for v in cals.values()):
                    continue
                ranked = sorted(cals.items(), key=lambda x: -x[1])
                for r, (mk, _) in enumerate(ranked, 1):
                    sums[mk] += r
                wins[ranked[0][0]] += 1
                n += 1
    return sums, wins, n


def lev_flip_stats():
    """목표 L 이 실제로 얼마나 자주 바뀌는가 — 변형의 체결부담 상한 파악용."""
    from backtest_futures_v25 import build_K2_signal
    from unified_backtest import load_data
    bars, _ = load_data('D')
    k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    rows = []
    for coin, s in k2.items():
        s = s.loc[(s.index >= START) & (s.index <= END)]
        if len(s) < 100:
            continue
        flips = int((s.diff().fillna(0) != 0).sum())
        rows.append((coin, len(s), flips, flips / len(s) * 100, float(s.mean())))
    rows.sort(key=lambda r: -r[3])
    return rows


def main():
    t0 = time.time()
    print(f"기간 {START} ~ {END} | 축: L 적용 시점만")

    print("\n[목표 L 변경 빈도 (전 유니버스, 보유 여부 무관)]")
    print(f"  {'coin':<6s} {'bars':>6s} {'flips':>6s} {'flip%':>7s} {'meanL':>6s}")
    fl = lev_flip_stats()
    for coin, nb, flips, pct, mean_l in fl[:12]:
        print(f"  {coin:<6s} {nb:>6d} {flips:>6d} {pct:>6.1f}% {mean_l:>6.2f}")
    if fl:
        print(f"  ... 전체 {len(fl)}종 평균 flip% = {np.mean([r[3] for r in fl]):.1f}%")

    res = {}
    for tag, dl in [('entry_fixed(현행)', False), ('daily_relever', True)]:
        t1 = time.time()
        m = run_fut(dl)
        eq = m.get('_equity')
        res[tag] = dict(m=m, eq=eq)
        print(f"\n[{tag}] {time.time()-t1:.0f}s  bars={len(eq)} "
              f"{eq.index[0].date()}~{eq.index[-1].date()}")
        print(f"  Cal {m['Cal']:.2f} | CAGR {m['CAGR']:+.1%} | MDD {m['MDD']:+.1%} | "
              f"Sharpe {m['Sharpe']:.2f}")
        print(f"  Trades {m['Trades']} | Rebal {m['Rebal']} | Liq {m['Liq']} | "
              f"Relev {m.get('Relev', 0)} | 최종 {eq.iloc[-1]/eq.iloc[0]:.2f}x")

    eqs = {k: v['eq'] for k, v in res.items()}

    print("\n[연도별 Calmar / 수익률]")
    print(f"  {'year':<6s} {'fixed CAGR':>11s} {'daily CAGR':>11s} {'fixed MDD':>10s} {'daily MDD':>10s}")
    for year in sorted({d.year for d in eqs['entry_fixed(현행)'].index}):
        line = [f"  {year:<6d}"]
        cells = {}
        for k, s in eqs.items():
            seg = s[s.index.year == year].dropna()
            if len(seg) < 30:
                cells[k] = (np.nan, np.nan)
                continue
            r = seg.iloc[-1] / seg.iloc[0] - 1
            mdd = float((seg / seg.cummax() - 1).min())
            cells[k] = (r, mdd)
        line.append(f"{cells['entry_fixed(현행)'][0]:>+10.1%}")
        line.append(f"{cells['daily_relever'][0]:>+10.1%}")
        line.append(f"{cells['entry_fixed(현행)'][1]:>+9.1%}")
        line.append(f"{cells['daily_relever'][1]:>+9.1%}")
        print(' '.join(line))

    print("\n[윈도우 rank-sum]")
    sums, wins, n = window_rs(eqs)
    for k in sorted(sums, key=lambda k: sums[k]):
        print(f"  {k:<20s} avg_rank {sums[k]/n:.3f}  wins {wins[k]:>4d}/{n} "
              f"({wins[k]/n*100:.0f}%)")
    print(f"  windows n={n} (sizes {WIN_SIZES} × strides {STRIDES})")

    print("\n[비용 stress — tx 5x (0.30%)]")
    for tag, dl in [('entry_fixed(현행)', False), ('daily_relever', True)]:
        from backtest_futures_v25 import run as fbt_run, build_K2_signal
        from unified_backtest import load_data
        os.environ['DAILY_LEV'] = 'on' if dl else 'off'
        bars, funding = load_data('D')
        k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                             btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                             l_min=2.0, l_mid=3.0, l_max=4.0)
        m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1 / 3,
                    tx_cost=0.003, maint_rate=0.004,
                    sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
                    canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
                    health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                    n_snapshots=5, snap_interval_bars=95,
                    start_date=START, end_date=END)
        os.environ['DAILY_LEV'] = 'off'
        print(f"  {tag:<20s} Cal {m['Cal']:.2f} | CAGR {m['CAGR']:+.1%} | "
              f"MDD {m['MDD']:+.1%} | Sharpe {m['Sharpe']:.2f} | Trades {m['Trades']}")

    # 왜 차이가 작은가 — 축이 작동할 여지 자체를 잰다
    m_on = res['daily_relever']['m']
    print("\n[변형이 작동할 여지]")
    print(f"  전체 봉 {len(eqs['daily_relever'])} | 리밸일 {m_on['Rebal']} "
          f"| 무리밸 보유봉 {m_on.get('RelevBars', 0)} "
          f"| 기회(코인-일) {m_on.get('RelevOpp', 0)} | 실제 재조정 {m_on['Relev']}")
    print("  → 리밸(드리프트 3pp)이 잦아 무리밸 보유일이 적다. 리밸일엔 현행도 오늘 L 로 목표를 잡으므로")
    print("    두 안이 갈라질 수 있는 날 자체가 좁다.")

    print("\n[드리프트 완화 regime probe — 축이 작동하는 조건에서의 우열 (라이브 채택안 아님)]")
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    for thr in [0.10, 0.20]:
        for tag, dl in [('fixed', False), ('daily', True)]:
            os.environ['DAILY_LEV'] = 'on' if dl else 'off'
            bars, funding = load_data('D')
            k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                                 btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                                 l_min=2.0, l_mid=3.0, l_max=4.0)
            m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1 / 3,
                        tx_cost=0.0006, maint_rate=0.004,
                        sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
                        canary_hyst=0.015, drift_threshold=thr, post_flip_delay=5,
                        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                        n_snapshots=5, snap_interval_bars=95,
                        start_date=START, end_date=END)
            os.environ['DAILY_LEV'] = 'off'
            print(f"  drift={thr:.2f} {tag:<6s} Cal {m['Cal']:.2f} | CAGR {m['CAGR']:+.1%} | "
                  f"MDD {m['MDD']:+.1%} | Sharpe {m['Sharpe']:.2f} | Relev {m['Relev']:>4d} "
                  f"| 기회 {m.get('RelevOpp', 0):>5d}")

    # equity 저장 (후속 alloc 합성용)
    out = os.environ.get('DAILY_LEV_OUT', '')
    if out:
        pd.DataFrame({k: v for k, v in eqs.items()}).to_csv(out)
        print(f"\nequity 저장: {out}")

    if os.environ.get('WITH_ALLOC', 'off') == 'on':
        print("\n[자산배분 60/25/15 합성 — 선물 sleeve 만 교체]")
        eq_sp = run_spot_live()
        eq_st = run_stock_live()
        rows = {}
        for tag, eq_fu in eqs.items():
            alloc = build_alloc(eq_st, eq_sp, eq_fu)
            rows[tag] = (alloc, metrics(alloc))
        for tag, (alloc, mm) in rows.items():
            print(f"  {tag:<20s} Cal {mm['Cal']:.2f} | CAGR {mm['CAGR']:+.1f}% | "
                  f"MDD {mm['MDD']:+.1f}% | Sharpe {mm['Sharpe']:.2f}")
        sums, wins, n = window_rs({k: v[0] for k, v in rows.items()})
        for k in sorted(sums, key=lambda k: sums[k]):
            print(f"  {k:<20s} alloc avg_rank {sums[k]/n:.3f}  wins {wins[k]}/{n}")

    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
