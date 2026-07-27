"""V25 선물 sleeve 드로다운 분포 — "이 정도 하락이 히스토리상 얼마나 잦았나" 정량화.

read-only 분석. 라이브 코드 무수정. run() V25 defaults + build_K2_signal(동적 L).
현재 참조: 선물 계좌 equity 68k→63k ≈ -7% (계좌레벨), HYPE 단일 -10% coin / -20% leveraged.
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pandas as pd


def main():
    t0 = time.time()
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    bars, funding = load_data('D')
    k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    m = fbt_run(bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
                tx_cost=0.0006, maint_rate=0.004,
                sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
                canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
                health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                n_snapshots=5, snap_interval_bars=95,
                start_date='2020-10-01', end_date='2026-05-13')
    if not m:
        print("NO DATA"); return
    eq = m['_equity'].dropna()
    print(f"기간 {eq.index[0].date()} ~ {eq.index[-1].date()} ({len(eq)}일), "
          f"CAGR={m['CAGR']:+.1%} MDD={m['MDD']:+.1%} Cal={m['Cal']:.2f}")

    peak = eq.cummax()
    dd = eq / peak - 1.0  # 일별 드로다운 (from running peak)

    # ── 드로다운 에피소드 추출: peak 갱신으로 리셋, 각 에피소드의 최저점 기록 ──
    episodes = []  # (start_date, trough_date, end_date_or_None, trough_dd, dur_days, recov_days)
    in_dd = False
    ep_start = None; trough = 0.0; trough_date = None
    for d, v in dd.items():
        if v < -1e-9:
            if not in_dd:
                in_dd = True; ep_start = d; trough = v; trough_date = d
            elif v < trough:
                trough = v; trough_date = d
        else:
            if in_dd:
                episodes.append([ep_start, trough_date, d, trough,
                                 (trough_date - ep_start).days, (d - trough_date).days])
                in_dd = False
    if in_dd:  # 마지막이 미회복 상태로 종료
        episodes.append([ep_start, trough_date, None, trough,
                         (trough_date - ep_start).days, None])

    ep = pd.DataFrame(episodes, columns=['start', 'trough_date', 'recovered', 'trough_dd', 'to_trough_d', 'recov_d'])

    print(f"\n총 드로다운 에피소드 수: {len(ep)}")
    print("깊이별 (최저점이 해당 % 이상 하락한 에피소드 수):")
    for thr in [0.03, 0.05, 0.07, 0.10, 0.15, 0.20, 0.30]:
        sub = ep[ep['trough_dd'] <= -thr]
        n = len(sub)
        med_recov = sub['recov_d'].dropna().median() if n else float('nan')
        print(f"  ≤ -{thr*100:>4.0f}%: {n:>3d}건  (중앙 회복일 {med_recov if pd.isna(med_recov) else int(med_recov)})")

    # time-in-drawdown
    frac_in_dd = float((dd < -0.05).mean())
    frac_in_dd7 = float((dd < -0.07).mean())
    print(f"\n계좌가 -5% 이상 물려있던 날 비율: {frac_in_dd:.1%}")
    print(f"계좌가 -7% 이상 물려있던 날 비율: {frac_in_dd7:.1%}")

    # 현재 참조선(-7% 계좌레벨) 관점
    n7 = len(ep[ep['trough_dd'] <= -0.07])
    yrs = (eq.index[-1] - eq.index[0]).days / 365.25
    print(f"\n현재 계좌 ~-7% 수준 = -7% 이상 에피소드 {n7}건 / {yrs:.1f}년 → 연 {n7/yrs:.1f}회꼴")

    # 최근 대표 에피소드 몇 개 (깊은 것 top)
    top = ep.sort_values('trough_dd').head(8)
    print("\n가장 깊었던 드로다운 top8 (trough_dd / 저점날 / 회복일수):")
    for _, r in top.iterrows():
        rec = 'ONGOING' if r['recovered'] is None else f"{int(r['recov_d'])}d"
        print(f"  {r['trough_dd']:+.1%}  {pd.Timestamp(r['trough_date']).date()}  회복 {rec}")

    print(f"\n소요 {time.time()-t0:.0f}s")


if __name__ == '__main__':
    main()
