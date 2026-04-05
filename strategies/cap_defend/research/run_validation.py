#!/usr/bin/env python3
"""최종 전략 검증 — Walk-forward + 수수료/펀딩비 스트레스.

Top 4 전략에 대해:
a. Walk-forward (학습 24개월 → 검증 6개월 롤링)
b. 수수료/슬리피지 스트레스 (1.5x, 2x, 3x 비용)
c. 펀딩비 스트레스 (1.5x, 2x)
"""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np, pandas as pd
from backtest_futures_full import load_data, run
from run_ensemble import SingleAccountEngine, combine_targets, STRATEGIES

START = '2020-10-01'
END = '2026-03-28'

# Top 4 앙상블 정의
TOP4 = {
    '4h1+1h1+1h3': {'4h1': 1/3, '1h1': 1/3, '1h3': 1/3},
    '4h1+1h1': {'4h1': 0.5, '1h1': 0.5},
    '4h1+4h2+1h1': {'4h1': 1/3, '4h2': 1/3, '1h1': 1/3},
    '4h1+4h2+1h1+1h3': {'4h1': 0.25, '4h2': 0.25, '1h1': 0.25, '1h3': 0.25},
}

NEEDED_STRATS = {'4h1', '4h2', '1h1', '1h3'}


def generate_trace(bars, funding, interval, config, start, end):
    """단일 전략 trace 생성."""
    trace = []
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=start, end_date=end, _trace=trace, **config)
    return trace


def run_ensemble_single(traces, weights, all_dates, bars_1h, funding_1h,
                        leverage=2.0, tx_cost=0.0004, funding_mult=1.0):
    """앙상블 실행 (비용 파라미터 조절 가능)."""
    combined = combine_targets(traces, weights, all_dates)

    # 펀딩비 배율 적용
    adj_funding = {}
    for coin, fr in funding_1h.items():
        adj_funding[coin] = fr * funding_mult

    engine = SingleAccountEngine(bars_1h, adj_funding, leverage=leverage, tx_cost=tx_cost)
    return engine.run(combined)


def stress_test(name, traces, weights, all_dates, bars_1h, funding_1h):
    """수수료 + 펀딩비 스트레스 테스트."""
    print(f"\n  ── {name}: 스트레스 테스트 (2x) ──")
    print(f"  {'Condition':<25s} {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s}")
    print(f"  {'-'*55}")

    configs = [
        ('Base (tx=0.04%)', 0.0004, 1.0),
        ('tx 1.5x (0.06%)', 0.0006, 1.0),
        ('tx 2x (0.08%)', 0.0008, 1.0),
        ('tx 3x (0.12%)', 0.0012, 1.0),
        ('Funding 1.5x', 0.0004, 1.5),
        ('Funding 2x', 0.0004, 2.0),
        ('tx 2x + Fund 1.5x', 0.0008, 1.5),
        ('tx 2x + Fund 2x', 0.0008, 2.0),
    ]

    for cond_name, tx, fund_mult in configs:
        m = run_ensemble_single(traces, weights, all_dates, bars_1h, funding_1h,
                                leverage=2.0, tx_cost=tx, funding_mult=fund_mult)
        if m:
            print(f"  {cond_name:<25s} {m['Sharpe']:>5.2f} {m['CAGR']:>+8.1%}"
                  f" {m['MDD']:>+8.1%} {m['Cal']:>6.2f}")


def walk_forward(name, strat_keys, weights_template, data, bars_1h, funding_1h, all_dates_1h):
    """Walk-forward 테스트.
    학습 24개월 → 검증 6개월 롤링.
    파라미터는 고정 (이미 선택됨), equity curve 연결."""
    print(f"\n  ── {name}: Walk-forward (2x) ──")

    # 6개월 단위 구간 생성
    start_dt = pd.Timestamp(START)
    end_dt = pd.Timestamp(END)
    train_months = 24
    test_months = 6

    wf_equity = []
    cursor = start_dt + pd.DateOffset(months=train_months)

    while cursor < end_dt:
        test_end = min(cursor + pd.DateOffset(months=test_months), end_dt)
        train_start = cursor - pd.DateOffset(months=train_months)

        # 검증 구간 실행 (파라미터 고정 — 학습 구간에서 이미 선택된 것으로 가정)
        traces = {}
        for key in strat_keys:
            spec = STRATEGIES[key]
            iv = spec['interval']
            bars, funding = data[iv]
            traces[key] = generate_trace(bars, funding, iv, spec['config'],
                                         str(train_start.date()), str(test_end.date()))

        test_dates = all_dates_1h[(all_dates_1h >= str(cursor.date())) &
                                   (all_dates_1h <= str(test_end.date()))]
        if len(test_dates) < 100:
            cursor = test_end
            continue

        combined = combine_targets(traces, weights_template, test_dates)
        engine = SingleAccountEngine(bars_1h, funding_1h, leverage=2.0)
        m = engine.run(combined)

        if m and '_equity' in m:
            eq = m['_equity']
            # 검증 구간만 추출
            test_eq = eq[(eq.index >= str(cursor.date())) & (eq.index <= str(test_end.date()))]
            if len(test_eq) > 0:
                # 정규화 (시작=1)
                norm = test_eq / test_eq.iloc[0]
                wf_equity.append((cursor.date(), test_end.date(), norm, m))

        cursor = test_end

    if not wf_equity:
        print("  No walk-forward results")
        return

    # 구간별 성과
    print(f"  {'Period':<25s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s}")
    print(f"  {'-'*50}")
    for start_d, end_d, norm, m in wf_equity:
        yrs = (end_d - start_d).days / 365.25
        if yrs > 0 and norm.iloc[-1] > 0:
            cagr = (norm.iloc[-1] / norm.iloc[0]) ** (1/yrs) - 1
            mdd = (norm / norm.cummax() - 1).min()
            cal = cagr / abs(mdd) if mdd != 0 else 0
            print(f"  {str(start_d)}~{str(end_d)} {cagr:>+8.1%} {mdd:>+8.1%} {cal:>6.2f}")

    # 전체 이어붙인 equity
    full_eq_parts = []
    scale = 1.0
    for _, _, norm, _ in wf_equity:
        scaled = norm * scale
        full_eq_parts.append(scaled)
        scale = scaled.iloc[-1]

    if full_eq_parts:
        full_eq = pd.concat(full_eq_parts)
        full_eq = full_eq[~full_eq.index.duplicated(keep='first')]
        total_yrs = (full_eq.index[-1] - full_eq.index[0]).days / 365.25
        if total_yrs > 0 and full_eq.iloc[-1] > 0:
            total_cagr = (full_eq.iloc[-1] / full_eq.iloc[0]) ** (1/total_yrs) - 1
            total_mdd = (full_eq / full_eq.cummax() - 1).min()
            dr = full_eq.resample('D').last().dropna().pct_change().dropna()
            total_sh = dr.mean() / dr.std() * np.sqrt(365) if dr.std() > 0 else 0
            total_cal = total_cagr / abs(total_mdd) if total_mdd != 0 else 0
            print(f"\n  WF 전체: Sh={total_sh:.2f} CAGR={total_cagr:+.1%} MDD={total_mdd:+.1%} Cal={total_cal:.2f}")


if __name__ == '__main__':
    t0 = time.time()

    # 데이터 로드
    data = {}
    for iv in ['D', '4h', '1h']:
        print(f"Loading {iv}...")
        data[iv] = load_data(iv)

    bars_1h, funding_1h = data['1h']
    btc_1h = bars_1h['BTC']
    all_dates_1h = btc_1h.index[(btc_1h.index >= START) & (btc_1h.index <= END)]

    # 각 전략 trace 생성 (full period)
    print("\nGenerating traces...")
    traces = {}
    for key in NEEDED_STRATS:
        spec = STRATEGIES[key]
        iv = spec['interval']
        bars, funding = data[iv]
        traces[key] = generate_trace(bars, funding, iv, spec['config'], START, END)
        print(f"  {key}: {len(traces[key])} bars")

    # Top 4 각각에 대해 검증
    for name, weights in TOP4.items():
        print(f"\n{'='*70}")
        print(f"  {name}")
        print(f"{'='*70}")

        sel_traces = {k: traces[k] for k in weights if k in traces}

        # a. 스트레스 테스트
        stress_test(name, sel_traces, weights, all_dates_1h, bars_1h, funding_1h)

        # b. Walk-forward
        strat_keys = list(weights.keys())
        walk_forward(name, strat_keys, weights, data, bars_1h, funding_1h, all_dates_1h)

    print(f"\n총 소요: {time.time()-t0:.0f}s")
