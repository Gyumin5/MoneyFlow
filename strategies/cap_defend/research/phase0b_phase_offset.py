"""Phase 0b — SN 유지 + 멤버별 phase offset 로 비동기화.

AI 라운드2 합의: SN=[120,30,120] 고정, 각 멤버의 bar_counter 초기 위상만
다르게 주어 "실행 시점 de-crowding" 달성. Codex 경고: 단일 best tuple 채택 금지.
여러 offset tuple 의 median/p25 를 base (0,0,0) 와 비교, ≈ base 면 실전 jitter
후보 채택.

출력:
  phase0b_results.csv
  phase0b_report.md 초안
"""
from __future__ import annotations
import os, sys, time
import pandas as pd
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from backtest_futures_full import load_data, run as bf_run
from futures_ensemble_engine import SingleAccountEngine, combine_targets

OUT = HERE
START = '2020-10-01'
END = '2026-03-28'

# V21 member base (SN 고정)
MEMBER_BASE = {
    'M1_S240_SN120': dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                          canary_hyst=0.015, drift_threshold=0.0,
                          dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
                          health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                          n_snapshots=3, snap_interval_bars=120),
    'M2_S240_SN30':  dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=480,
                          canary_hyst=0.015, drift_threshold=0.0,
                          dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
                          health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                          n_snapshots=3, snap_interval_bars=30),
    'M3_S120_SN120': dict(interval='4h', sma_bars=120, mom_short_bars=20, mom_long_bars=720,
                          canary_hyst=0.015, drift_threshold=0.0,
                          dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
                          health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                          n_snapshots=3, snap_interval_bars=120),
}

# phase offset tuples (M1, M2, M3).
# baseline (0,0,0) + 9 test tuples (coprime with SN, varied magnitudes)
OFFSET_TUPLES = [
    ('base_000',         (0, 0, 0)),
    ('offset_0_7_13',    (0, 7, 13)),
    ('offset_0_11_41',   (0, 11, 41)),   # coprime with 30, 120
    ('offset_0_13_29',   (0, 13, 29)),
    ('offset_0_17_41',   (0, 17, 41)),
    ('offset_0_19_53',   (0, 19, 53)),
    ('offset_0_23_59',   (0, 23, 59)),
    ('offset_0_5_17',    (0, 5, 17)),
    ('offset_7_13_29',   (7, 13, 29)),
    ('offset_11_19_41',  (11, 19, 41)),
]


def generate_trace(data, cfg, phase_off):
    bars, funding = data[cfg['interval']]
    trace = []
    run_cfg = dict(cfg)
    interval = run_cfg.pop('interval')
    bf_run(bars, funding, interval=interval, leverage=1.0,
           start_date=START, end_date=END,
           phase_offset_bars=phase_off,
           _trace=trace, **run_cfg)
    return trace


def run_tuple(data, tup, label):
    traces = {}
    for (name, base), po in zip(MEMBER_BASE.items(), tup):
        traces[name] = generate_trace(data, base, po)

    bars_1h, funding_1h = data['1h']
    all_dates = bars_1h['BTC'].index[(bars_1h['BTC'].index >= START) &
                                       (bars_1h['BTC'].index <= END)]
    weights = {k: 1/3 for k in MEMBER_BASE}
    combined = combine_targets(traces, weights, all_dates)

    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=3.0,
        stop_kind='none', stop_pct=0.0,
        stop_gate='none', stop_gate_cash_threshold=0.0,
        per_coin_leverage_mode='fixed',
        leverage_floor=3.0, leverage_mid=3.0, leverage_ceiling=3.0,
    )
    m = engine.run(combined)
    m['label'] = label
    m['phase_M1'] = tup[0]
    m['phase_M2'] = tup[1]
    m['phase_M3'] = tup[2]
    return m


def main():
    t0 = time.time()
    print('[P0b] Loading data...')
    data = {iv: load_data(iv) for iv in ['4h', '1h']}
    print(f'  done ({time.time()-t0:.0f}s)')

    results = []
    for label, tup in OFFSET_TUPLES:
        t1 = time.time()
        print(f'\n[P0b] Running {label} offsets={tup}')
        m = run_tuple(data, tup, label)
        m['elapsed_s'] = round(time.time() - t1, 1)
        results.append(m)
        print(f'  Cal={m.get("Cal", 0):.2f} CAGR={m.get("CAGR", 0):+.2%} '
              f'MDD={m.get("MDD", 0):+.2%} Rebal={m.get("Rebal", 0)} '
              f'Liq={m.get("Liq", 0)} ({m["elapsed_s"]}s)')

    df = pd.DataFrame(results)
    df.to_csv(os.path.join(OUT, 'phase0b_results.csv'), index=False)

    cols = ['label', 'phase_M1', 'phase_M2', 'phase_M3', 'Cal', 'CAGR', 'MDD',
            'Sharpe', 'Rebal', 'Liq', 'elapsed_s']
    print('\n[P0b] Summary')
    print(df[[c for c in cols if c in df.columns]].to_string(index=False))

    # 분포 통계
    base = df[df['label'] == 'base_000'].iloc[0]
    others = df[df['label'] != 'base_000']
    print('\n[P0b] Offset tuples 분포 (base_000 제외)')
    for k in ['Cal', 'CAGR', 'MDD', 'Rebal', 'Liq']:
        if k not in df.columns:
            continue
        v = others[k].values
        print(f'  {k}: base={base[k]:.4f} median={np.median(v):.4f} '
              f'p25={np.percentile(v, 25):.4f} p75={np.percentile(v, 75):.4f} '
              f'min={v.min():.4f} max={v.max():.4f}')

    # baseline 대비 delta 분석
    print('\n[P0b] Delta vs base_000 (%)')
    for _, row in others.iterrows():
        d_cal = (row['Cal'] - base['Cal']) / abs(base['Cal']) * 100
        d_cagr = (row['CAGR'] - base['CAGR']) / abs(base['CAGR']) * 100
        d_mdd = (row['MDD'] - base['MDD']) / abs(base['MDD']) * 100
        d_reb = row.get('Rebal', 0) - base.get('Rebal', 0)
        d_liq = row.get('Liq', 0) - base.get('Liq', 0)
        print(f"  {row['label']}: Cal {d_cal:+.1f}% CAGR {d_cagr:+.1f}% "
              f"MDD {d_mdd:+.1f}% Rebal{d_reb:+d} Liq{d_liq:+d}")

    print(f'\n[P0b] Total {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
