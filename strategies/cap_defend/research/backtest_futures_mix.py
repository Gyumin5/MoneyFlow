#!/usr/bin/env python3
"""선물 혼합 전략 시뮬레이션.

여러 간격(D/4h/1h) 전략을 1/N으로 배분하여 독립 실행 후 equity curve 합산.
단일 계좌 제약: 동일 배수, 겹치는 종목은 비중 합산.

Usage:
  python3 backtest_futures_mix.py
"""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from backtest_futures_full import load_data, run

START = '2020-10-01'
END = '2026-03-28'


def run_and_get_equity(bars, funding, interval, leverage, config, initial_capital):
    """전략 실행 후 일별 equity curve 반환."""
    from backtest_futures_full import run as _run
    # run()은 내부에서 pv_list를 반환하지 않음 → 메트릭만 반환
    # equity curve를 얻으려면 엔진을 수정해야 하지만,
    # 간단히: 독립 실행 후 메트릭 반환 (혼합은 메트릭 수준에서 근사)
    m = _run(bars, funding, interval=interval, leverage=leverage,
             start_date=START, end_date=END, initial_capital=initial_capital,
             **config)
    return m


def mix_metrics(results, weights=None):
    """여러 전략 결과의 가중 메트릭 근사.
    독립 포트폴리오의 CAGR 합산 (equity curve 없이 근사)."""
    n = len(results)
    if weights is None:
        weights = [1.0/n] * n

    # 가중 평균 (근사)
    wt_cagr = sum(w * r['CAGR'] for w, r in zip(weights, results))
    wt_sh = sum(w * r['Sharpe'] for w, r in zip(weights, results))
    # MDD는 단순 가중평균이 아님 — 최대값으로 보수적 추정
    worst_mdd = min(r['MDD'] for r in results)
    avg_mdd = sum(w * r['MDD'] for w, r in zip(weights, results))
    total_liq = sum(r.get('Liq', 0) for r in results)
    total_rb = sum(r.get('Rebal', 0) for r in results)

    cal = wt_cagr / abs(avg_mdd) if avg_mdd != 0 else 0
    return {
        'Sharpe': wt_sh,
        'CAGR': wt_cagr,
        'MDD_avg': avg_mdd,
        'MDD_worst': worst_mdd,
        'Cal': cal,
        'Liq': total_liq,
        'Rebal': total_rb,
    }


# ─── 전략 설정 ───
CONFIGS = {
    'D_best': {
        'interval': 'D',
        'config': dict(sma_days=40, mom_short_days=15, mom_long_days=90,
                       canary_hyst=0.0, drift_threshold=0.0,
                       dd_threshold=0, dd_lookback=0, bl_drop=0, daily_gate=False),
    },
    'D_v18': {
        'interval': 'D',
        'config': dict(sma_days=50, mom_short_days=30, mom_long_days=90,
                       canary_hyst=0.015, drift_threshold=0.0,
                       dd_threshold=0, dd_lookback=0, bl_drop=0, daily_gate=False),
    },
    '4h_best': {
        'interval': '4h',
        'config': dict(sma_days=40, mom_short_days=30, mom_long_days=60,
                       canary_hyst=0.0, drift_threshold=0.0,
                       dd_threshold=0, dd_lookback=0, bl_drop=0, daily_gate=False),
    },
    '4h_safe': {
        'interval': '4h',
        'config': dict(sma_days=40, mom_short_days=21, mom_long_days=90,
                       canary_hyst=0.015, drift_threshold=0.0,
                       dd_threshold=-0.25, dd_lookback=60, bl_drop=-0.15, daily_gate=False),
    },
    '1h_best': {
        'interval': '1h',
        'config': dict(sma_days=40, mom_short_days=21, mom_long_days=60,
                       canary_hyst=0.0, drift_threshold=0.0,
                       dd_threshold=-0.25, dd_lookback=60, bl_drop=0, daily_gate=False),
    },
}

# ─── 혼합 조합 ───
MIX_COMBOS = [
    # (이름, [(전략키, 비중), ...])
    ('D_best only', [('D_best', 1.0)]),
    ('4h_best only', [('4h_best', 1.0)]),
    ('1h_best only', [('1h_best', 1.0)]),
    ('D+4h 50:50', [('D_best', 0.5), ('4h_best', 0.5)]),
    ('D+4h+1h 1/3', [('D_best', 1/3), ('4h_best', 1/3), ('1h_best', 1/3)]),
    ('D+4h 70:30', [('D_best', 0.7), ('4h_best', 0.3)]),
    ('D_v18+4h_safe', [('D_v18', 0.5), ('4h_safe', 0.5)]),
    ('D_best+4h_safe', [('D_best', 0.5), ('4h_safe', 0.5)]),
]


if __name__ == '__main__':
    t0 = time.time()

    # 데이터 로드
    data = {}
    for key, spec in CONFIGS.items():
        iv = spec['interval']
        if iv not in data:
            print(f"Loading {iv} data...")
            data[iv] = load_data(iv)

    # 각 레버리지별로 모든 전략 실행
    for lev in [1.0, 1.5, 2.0, 3.0]:
        print(f"\n{'='*80}")
        print(f"  Leverage {lev}x")
        print(f"{'='*80}")

        # 개별 전략 실행
        strat_results = {}
        for key, spec in CONFIGS.items():
            iv = spec['interval']
            bars, funding = data[iv]
            m = run(bars, funding, interval=iv, leverage=lev,
                    start_date=START, end_date=END,
                    initial_capital=10000.0, **spec['config'])
            strat_results[key] = m
            liq = f" Liq{m['Liq']}" if m.get('Liq', 0) > 0 else ""
            print(f"  {key:<15s} Sh={m['Sharpe']:.2f} CAGR={m['CAGR']:+.1%}"
                  f" MDD={m['MDD']:+.1%} Cal={m['Cal']:.2f} Rb={m['Rebal']}{liq}")

        # 혼합 결과
        print(f"\n  {'Mix':<20s} {'Sh':>5s} {'CAGR':>8s} {'MDD_avg':>8s} {'MDD_w':>7s} {'Cal':>5s} {'Liq':>3s} {'Rb':>5s}")
        print(f"  {'-'*65}")
        for name, components in MIX_COMBOS:
            results = []
            weights = []
            for skey, w in components:
                results.append(strat_results[skey])
                weights.append(w)
            mx = mix_metrics(results, weights)
            liq = f"💀{mx['Liq']}" if mx['Liq'] > 0 else ""
            print(f"  {name:<20s} {mx['Sharpe']:>5.2f} {mx['CAGR']:>+8.1%}"
                  f" {mx['MDD_avg']:>+8.1%} {mx['MDD_worst']:>+7.1%}"
                  f" {mx['Cal']:>5.2f} {liq:>3s} {mx['Rebal']:>5d}")

    print(f"\n총 소요: {time.time()-t0:.0f}s")
