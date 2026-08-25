"""선물 리밸런싱 체결 밴드 비교 — BT 5% (현행) vs 라이브 1% (2026-08-25).

계기: 라이브(auto_trade_binance.DELTA_THRESHOLD=0.01)는 목표 명목 대비 1% 만 벌어져도
체결하는데, 채택 BT(backtest_futures_v25._execute_rebalance)는 수량 ±5% 밴드를 벗어날 때만
체결한다. 드리프트가 사실상 매일 발화하는 구조라(L4 에서 ±1.5%) 이 밴드가 실제 회전수를
정하는 값이고, 라이브가 BT 가정보다 자주 돈다 = 실비용이 BT 보다 크다.

가설: 밴드를 좁히면 목표 추종이 정확해져 위험조정 성과가 오르거나(추종 이득),
반대로 체결비용·슬리피지만 늘어 순손해다(비용 손실). 어느 쪽인지 재고, 라이브를 BT 에
맞출지(1%→5%) BT 를 라이브에 맞출지 판단 근거를 만든다.

축: 그것 하나만. 선정·헬스·카나리·스냅샷·드리프트·레버리지·비용 전부 동일.
변형은 정본 엔진 backtest_futures_v25.py 의 env 토글 REBAL_BAND 로 구현했다(기본 0.05=현행).
결론 후 SSoT 오염 방지를 위해 엔진 패치를 되돌리면, 재현하려면 먼저
  git apply strategies/cap_defend/research/bt_fut_rebal_band.engine.patch
를 적용해야 한다. 적용 없이 돌리면 전 팔이 0.05 로 동작해 차이 0 이 나온다(가짜 동률).

실행: cd /home/gmoh/mon/251229/strategies/cap_defend/research && python3 bt_fut_rebal_band.py
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
BANDS = [0.01, 0.02, 0.03, 0.05, 0.10]
BASE_TX = 0.0006
STRESS_TX = 0.003  # 5x


def _tag(band: float) -> str:
    return f"band {band*100:.0f}%" + ("(현행BT)" if band == 0.05 else
                                      "(라이브)" if band == 0.01 else "")


def run_fut(band: float, tx_cost: float = BASE_TX):
    """라이브 V25 선물 sleeve 설정 그대로. 체결 밴드만 바꾼다."""
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    os.environ['REBAL_BAND'] = f'{band}'
    bars, funding = load_data('D')
    k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1 / 3,
                tx_cost=tx_cost, maint_rate=0.004,
                sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
                canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
                health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                n_snapshots=5, snap_interval_bars=95,
                start_date=START, end_date=END)
    os.environ['REBAL_BAND'] = '0.05'
    return m


def run_spot_live():
    """현물 sleeve 라이브 V24 설정 (밴드 축과 무관 — 합성용 고정 팔)."""
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
    """주식 sleeve 라이브 V25 설정 (합성용 고정 팔)."""
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


def main():
    t0 = time.time()
    print(f"기간 {START} ~ {END} | 축: 체결 밴드만 (drift 0.03 고정)")
    print(f"밴드 후보 {[f'{b*100:.0f}%' for b in BANDS]} | tx {BASE_TX} (stress {STRESS_TX})")

    res = {}
    print(f"\n[sleeve 단독 — tx {BASE_TX*100:.2f}%]")
    print(f"  {'arm':<16s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} "
          f"{'Trades':>7s} {'Rebal':>6s} {'Liq':>4s} {'최종':>9s} {'초':>5s}")
    for band in BANDS:
        t1 = time.time()
        m = run_fut(band)
        eq = m.get('_equity')
        res[_tag(band)] = dict(m=m, eq=eq, band=band)
        print(f"  {_tag(band):<16s} {m['Cal']:>6.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} "
              f"{m['Sharpe']:>7.2f} {m['Trades']:>7d} {m['Rebal']:>6d} {m['Liq']:>4d} "
              f"{eq.iloc[-1]/eq.iloc[0]:>8.1f}x {time.time()-t1:>5.0f}")

    eqs = {k: v['eq'] for k, v in res.items()}
    yrs = (list(eqs.values())[0].index[-1] - list(eqs.values())[0].index[0]).days / 365.25
    print(f"\n[회전 부담] 기간 {yrs:.1f}년")
    base_tr = res[_tag(0.05)]['m']['Trades']
    for k, v in res.items():
        tr = v['m']['Trades']
        print(f"  {k:<16s} 체결 {tr:>5d} ({tr/yrs:>6.1f}회/년, 현행 대비 {tr/base_tr:>5.2f}x)")

    print("\n[연도별 수익률 / MDD]")
    keys = list(eqs)
    print("  year   " + "".join(f"{k:>22s}" for k in keys))
    for year in sorted({d.year for d in list(eqs.values())[0].index}):
        cells = []
        for k in keys:
            seg = eqs[k][eqs[k].index.year == year].dropna()
            if len(seg) < 30:
                cells.append(f"{'-':>22s}")
                continue
            r = seg.iloc[-1] / seg.iloc[0] - 1
            mdd = float((seg / seg.cummax() - 1).min())
            cells.append(f"{r:>+11.1%}/{mdd:>+9.1%}")
        print(f"  {year:<6d} " + "".join(cells))

    print("\n[윈도우 rank-sum — sleeve]")
    sums, wins, n = window_rs(eqs)
    for k in sorted(sums, key=lambda k: sums[k]):
        print(f"  {k:<16s} avg_rank {sums[k]/n:.3f}  wins {wins[k]:>4d}/{n} "
              f"({wins[k]/n*100:.0f}%)")
    print(f"  windows n={n} (sizes {WIN_SIZES} × strides {STRIDES})")

    print(f"\n[비용 stress — tx {STRESS_TX*100:.2f}% (5x)]")
    stress_eqs = {}
    print(f"  {'arm':<16s} {'Cal':>6s} {'CAGR':>8s} {'MDD':>8s} {'Sharpe':>7s} {'Trades':>7s}")
    for band in BANDS:
        m = run_fut(band, tx_cost=STRESS_TX)
        stress_eqs[_tag(band)] = m.get('_equity')
        print(f"  {_tag(band):<16s} {m['Cal']:>6.2f} {m['CAGR']:>+7.1%} {m['MDD']:>+7.1%} "
              f"{m['Sharpe']:>7.2f} {m['Trades']:>7d}")
    print("  [윈도우 rank-sum — stress]")
    sums, wins, n = window_rs(stress_eqs)
    for k in sorted(sums, key=lambda k: sums[k]):
        print(f"    {k:<16s} avg_rank {sums[k]/n:.3f}  wins {wins[k]:>4d}/{n}")

    if os.environ.get('WITH_ALLOC', 'on') == 'on':
        print("\n[자산배분 60/25/15 합성 — 선물 sleeve 만 교체]")
        try:
            eq_sp = run_spot_live()
            eq_st = run_stock_live()
            rows = {}
            for tag, eq_fu in eqs.items():
                alloc = build_alloc(eq_st, eq_sp, eq_fu)
                rows[tag] = (alloc, metrics(alloc))
            for tag, (_a, mm) in rows.items():
                print(f"  {tag:<16s} Cal {mm['Cal']:.2f} | CAGR {mm['CAGR']:+.1f}% | "
                      f"MDD {mm['MDD']:+.1f}% | Sharpe {mm['Sharpe']:.2f}")
            sums, wins, n = window_rs({k: v[0] for k, v in rows.items()})
            for k in sorted(sums, key=lambda k: sums[k]):
                print(f"    {k:<16s} alloc avg_rank {sums[k]/n:.3f}  wins {wins[k]}/{n}")
        except Exception as ex:
            print(f"  합성 실패(무시하고 sleeve 결론 유지): {ex}")

    out = os.environ.get('BAND_OUT', '')
    if out:
        pd.DataFrame(eqs).to_csv(out)
        print(f"\nequity 저장: {out}")

    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
