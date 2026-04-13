#!/usr/bin/env python3
"""풀엔진으로 상위 후보 전략 테스트.

간이엔진(backtest_spot_barfreq.py) 그리드에서 선별한 상위권 후보를
풀엔진(backtest_futures_full.py + SingleAccountEngine)으로 재검증.

비교 대상: 현재 d005 (4전략 앙상블, cap_mom_blend_543_cash 5x동적)
"""
import os, sys, time
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest_futures_full import load_data, run
from futures_ensemble_engine import SingleAccountEngine, combine_targets
from futures_live_config import CURRENT_STRATEGIES, CURRENT_LIVE_COMBO, START, END


def generate_trace(data, cfg):
    run_cfg = dict(cfg)
    interval = run_cfg.pop("interval")
    bars, funding = data[interval]
    trace = []
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=START, end_date=END, _trace=trace, **run_cfg)
    return trace


def run_engine(data, combined, **engine_kw):
    bars_1h, funding_1h = data['1h']
    engine = SingleAccountEngine(bars_1h, funding_1h, **engine_kw)
    return engine.run(combined)


# ─── 후보 전략 정의 ──────────────────────────────────────────────
# 간이엔진 그리드 상위권에서 선별

CANDIDATE_SINGLES = {
    "4h_Ms80Ml720Sn120": dict(
        interval="4h", sma_bars=240, mom_short_bars=80, mom_long_bars=720,
        canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode="mom2vol", vol_mode="daily",
        vol_threshold=0.05, n_snapshots=3, snap_interval_bars=120,
    ),
    "4h_Ms40Ml720Sn120": dict(
        interval="4h", sma_bars=240, mom_short_bars=40, mom_long_bars=720,
        canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode="mom2vol", vol_mode="daily",
        vol_threshold=0.05, n_snapshots=3, snap_interval_bars=120,
    ),
    "4h_Ms30Ml720Sn120": dict(
        interval="4h", sma_bars=240, mom_short_bars=30, mom_long_bars=720,
        canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode="mom2vol", vol_mode="daily",
        vol_threshold=0.05, n_snapshots=3, snap_interval_bars=120,
    ),
}

# D 전략 후보 (앙상블용)
D_CANDIDATES = {
    "D_Ms80Ml90Sn180": dict(
        interval="D", sma_bars=50, mom_short_bars=80, mom_long_bars=90,
        canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode="mom2vol", vol_mode="daily",
        vol_threshold=0.05, n_snapshots=3, snap_interval_bars=180,
    ),
    "D_Ms20Ml90Sn180": dict(
        interval="D", sma_bars=50, mom_short_bars=20, mom_long_bars=90,
        canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode="mom2vol", vol_mode="daily",
        vol_threshold=0.05, n_snapshots=3, snap_interval_bars=180,
    ),
    "D_Ms80Ml240Sn180": dict(
        interval="D", sma_bars=50, mom_short_bars=80, mom_long_bars=240,
        canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
        bl_drop=0, bl_days=0, health_mode="mom2vol", vol_mode="daily",
        vol_threshold=0.05, n_snapshots=3, snap_interval_bars=180,
    ),
}

# 4h 후보 (앙상블용 추가)
H4_EXTRA = {
    "4h_Ms30Ml720Sn120": CANDIDATE_SINGLES["4h_Ms30Ml720Sn120"],
}

# 앙상블 조합 (현물 상위권 기반)
CANDIDATE_ENSEMBLES = {
    "D_M80L90 + 4h_M80L720": {
        "D_Ms80Ml90Sn180": 0.5,
        "4h_Ms80Ml720Sn120": 0.5,
    },
    "D_M20L90 + D_M80L240 + 4h_M80L720": {
        "D_Ms20Ml90Sn180": 1/3,
        "D_Ms80Ml240Sn180": 1/3,
        "4h_Ms80Ml720Sn120": 1/3,
    },
    "D_M20L90 + 4h_M80L720 + 4h_M30L720": {
        "D_Ms20Ml90Sn180": 1/3,
        "4h_Ms80Ml720Sn120": 1/3,
        "4h_Ms30Ml720Sn120": 1/3,
    },
    "D_M80L90 + D_M20L90 + 4h_M80L720": {
        "D_Ms80Ml90Sn180": 1/3,
        "D_Ms20Ml90Sn180": 1/3,
        "4h_Ms80Ml720Sn120": 1/3,
    },
}

# d005 동일 실행 파라미터
ENGINE_PARAMS_D005 = dict(
    leverage=5.0,
    stop_kind="prev_close_pct", stop_pct=0.15,
    stop_gate="cash_guard", stop_gate_cash_threshold=0.34,
    per_coin_leverage_mode="cap_mom_blend_543_cash",
    leverage_floor=3.0, leverage_mid=4.0, leverage_ceiling=5.0,
    leverage_cash_threshold=0.34,
    leverage_count_floor_max=2, leverage_count_mid_max=4,
    leverage_canary_floor_gap=0.015, leverage_canary_mid_gap=0.04,
    leverage_canary_high_gap=0.08, leverage_canary_sma_bars=1200,
    leverage_mom_lookback_bars=24*30, leverage_vol_lookback_bars=24*90,
)

# 고정 배수 실행 파라미터 (스탑+청산 있음)
def engine_fixed(lev):
    return dict(
        leverage=float(lev),
        stop_kind="prev_close_pct", stop_pct=0.15,
        stop_gate="cash_guard", stop_gate_cash_threshold=0.34,
        maint_rate=0.004,
    )


def fmt(m):
    cal = m.get('Cal', 0)
    return (f"Sh {m['Sharpe']:.3f} Cal {cal:.2f} CAGR {m['CAGR']:+.1%} "
            f"MDD {m['MDD']:+.1%} Liq {m.get('Liq', 0)} Stop {m.get('Stops', 0)} "
            f"Rebal {m.get('Rebal', 0)}")


def main():
    t_start = time.time()

    # ─── 데이터 로드 ──────────────────────
    print("Loading data...")
    intervals_needed = {'1h', '2h', '4h', 'D'}
    data = {iv: load_data(iv) for iv in intervals_needed}
    bars_1h, funding_1h = data['1h']
    all_dates = bars_1h['BTC'].index[
        (bars_1h['BTC'].index >= START) & (bars_1h['BTC'].index <= END)]
    print(f"Data loaded in {time.time() - t_start:.1f}s")

    # ─── 모든 전략의 trace 생성 ──────────────────
    print("\nGenerating traces...")
    all_cfgs = {}
    all_cfgs.update(CANDIDATE_SINGLES)
    all_cfgs.update(D_CANDIDATES)
    all_cfgs.update(H4_EXTRA)
    all_cfgs.update(CURRENT_STRATEGIES)

    traces = {}
    for name, cfg in all_cfgs.items():
        t0 = time.time()
        traces[name] = generate_trace(data, cfg)
        print(f"  {name}: {len(traces[name])} entries ({time.time()-t0:.1f}s)")

    # ─── 1. 현재 d005 baseline ──────────────────
    print("\n" + "="*80)
    print("=== Baseline: Current d005 ===")
    print("="*80)
    combined_d005 = combine_targets(
        {k: traces[k] for k in CURRENT_LIVE_COMBO},
        CURRENT_LIVE_COMBO, all_dates)
    m_base = run_engine(data, combined_d005, **ENGINE_PARAMS_D005)
    print(f"  d005 (4strat, 543 dynamic): {fmt(m_base)}")

    # ─── 2. 싱글 전략 + d005 실행층 ──────────────────
    print("\n" + "="*80)
    print("=== Singles with d005 execution (cap_mom_blend_543) ===")
    print("="*80)
    for name in CANDIDATE_SINGLES:
        combined = combine_targets({name: traces[name]}, {name: 1.0}, all_dates)
        m = run_engine(data, combined, **ENGINE_PARAMS_D005)
        print(f"  {name:30s} {fmt(m)}")

    # ─── 3. 싱글 전략 + 고정 배수 ──────────────────
    print("\n" + "="*80)
    print("=== Singles with fixed leverage (stop+liq) ===")
    print("="*80)
    for lev in [3, 4, 5]:
        for name in CANDIDATE_SINGLES:
            combined = combine_targets({name: traces[name]}, {name: 1.0}, all_dates)
            m = run_engine(data, combined, **engine_fixed(lev))
            print(f"  {name:30s} {lev}x: {fmt(m)}")
        print()

    # ─── 4. 앙상블 + d005 실행층 ──────────────────
    print("\n" + "="*80)
    print("=== Ensembles with d005 execution (cap_mom_blend_543) ===")
    print("="*80)
    for ens_name, ens_weights in CANDIDATE_ENSEMBLES.items():
        combined = combine_targets(
            {k: traces[k] for k in ens_weights},
            ens_weights, all_dates)
        m = run_engine(data, combined, **ENGINE_PARAMS_D005)
        print(f"  {ens_name:45s} {fmt(m)}")

    # ─── 5. 앙상블 + 고정 배수 ──────────────────
    print("\n" + "="*80)
    print("=== Ensembles with fixed leverage (stop+liq) ===")
    print("="*80)
    for lev in [3, 4, 5]:
        for ens_name, ens_weights in CANDIDATE_ENSEMBLES.items():
            combined = combine_targets(
                {k: traces[k] for k in ens_weights},
                ens_weights, all_dates)
            m = run_engine(data, combined, **engine_fixed(lev))
            print(f"  {ens_name:45s} {lev}x: {fmt(m)}")
        print()

    print(f"\nTotal elapsed: {time.time() - t_start:.1f}s")
    print("Done.")


if __name__ == '__main__':
    main()
