"""선물 V25 유지증거금률(maint_rate) 민감도 실험 — ai-debate(2026-07-03) 권고 실행.

질문: 현재 BT 는 전 종목 단일 maint_rate=0.004 로 단순화. 실제 바이낸스는 코인/notional별
0.4~0.65% 로 다름(알트 쪽이 더 높음 = 더 이른 청산). 0.4% vs 보수적 0.65% flat 을 같은
데이터로 비교해 CAGR/Cal/MDD/청산건수가 유의미하게 갈리는지 확인.

판정 기준(ai-debate arbiter): MDD 절대변화 2~3pp 이상 또는 청산 발생/레버리지 상한 판단이
바뀌면 tier-aware 정식 구현, 아니면 known limitation 으로 문서화.
"""
from __future__ import annotations
import os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
CAP = os.path.dirname(HERE)
sys.path.insert(0, CAP); sys.path.insert(0, HERE)

START = "2020-10-01"
END = "2026-05-29"


def run_fut(maint_rate):
    from backtest_futures_v25 import run as fbt_run, build_K2_signal
    from unified_backtest import load_data
    os.environ['DRIFT_CASH_REFILL'] = 'off'
    bars, funding = load_data('D')
    k2 = build_K2_signal(bars, btc_cap_sma_period=42, btc_cap_thr_mid=1.015,
                         btc_cap_thr_max=1.05, k2_sma_period=7, k2_hyst=0.025,
                         l_min=2.0, l_mid=3.0, l_max=4.0)
    m = fbt_run(
        bars, funding, interval='D', leverage=k2, universe_size=3, cap=1/3,
        tx_cost=0.0006, maint_rate=maint_rate,
        sma_days=42, mom_short_days=18, mom_long_days=127, vol_days=90,
        canary_hyst=0.015, drift_threshold=0.03, post_flip_delay=5,
        health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
        n_snapshots=5, snap_interval_bars=95,
        start_date=START, end_date=END,
    )
    return m


if __name__ == '__main__':
    rates = [
        ('current_flat_0.4%', 0.004),
        ('conservative_flat_0.5%', 0.005),
        ('conservative_flat_0.65%', 0.0065),
    ]
    print(f"{'설정':<26s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>7s} {'Sharpe':>7s} {'Liq':>4s} {'Rebal':>6s}")
    print('-' * 70)
    results = {}
    for label, mr in rates:
        m = run_fut(mr)
        results[label] = m
        liq = f"💀{m.get('Liq', 0)}" if m.get('Liq', 0) > 0 else "0"
        print(f"{label:<26s} {m.get('CAGR', 0):>+8.1%} {m.get('MDD', 0):>+8.1%} "
              f"{m.get('Cal', 0):>7.2f} {m.get('Sharpe', 0):>7.2f} {liq:>4s} {m.get('Rebal', 0):>6d}")

    base = results['current_flat_0.4%']
    worst = results['conservative_flat_0.65%']
    mdd_delta_pp = (worst.get('MDD', 0) - base.get('MDD', 0)) * 100
    cagr_delta_pp = (worst.get('CAGR', 0) - base.get('CAGR', 0)) * 100
    print('\n--- 판정 ---')
    print(f"MDD 변화: {mdd_delta_pp:+.2f}pp | CAGR 변화: {cagr_delta_pp:+.2f}pp | "
          f"청산 0.4%={base.get('Liq',0)} vs 0.65%={worst.get('Liq',0)}")
    if abs(mdd_delta_pp) >= 2.0 or worst.get('Liq', 0) > base.get('Liq', 0):
        print("=> 유의미한 차이. tier-aware 정식 구현 권장.")
    else:
        print("=> 미미한 차이. known limitation 으로 문서화, 정식 구현 보류 권장.")
