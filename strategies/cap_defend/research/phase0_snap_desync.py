"""Phase 0 — SN 교체 백테.

V21 futures ensemble (ENS_fut_L3_k3_12652d57) 3멤버의 snap_interval_bars 후보 4개:
  A [120, 30, 120]  baseline (현재 실거래)
  B [123, 33, 123]  최소 이동 + 2 멤버 동기 잔존
  C [33, 123, 129]  권고 (p=[11,41,43] 서로소)
  D [39, 111, 141]  대안 (p=[13,37,47])

출력:
  phase0_results.csv (config × 지표 + turnover)
  phase0_report_draft.md (표 + 요약)
"""
from __future__ import annotations
import os, sys, time
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
PARENT = os.path.dirname(HERE)
sys.path.insert(0, PARENT)

from backtest_futures_full import load_data, run as bf_run
from futures_ensemble_engine import SingleAccountEngine, combine_targets


OUT = HERE
os.makedirs(OUT, exist_ok=True)

START = '2020-10-01'
END = '2026-03-28'

# V21 member base params (auto_trade_binance.py 동일)
MEMBER_BASE = {
    'M1_S240_SN120': dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=720,
                          canary_hyst=0.015, drift_threshold=0.0,
                          dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
                          health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                          n_snapshots=3),
    'M2_S240_SN30':  dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=480,
                          canary_hyst=0.015, drift_threshold=0.0,
                          dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
                          health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                          n_snapshots=3),
    'M3_S120_SN120': dict(interval='4h', sma_bars=120, mom_short_bars=20, mom_long_bars=720,
                          canary_hyst=0.015, drift_threshold=0.0,
                          dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0,
                          health_mode='mom2vol', vol_mode='daily', vol_threshold=0.05,
                          n_snapshots=3),
}

# SN 후보: (M1=S240/SN120, M2=S240/SN30, M3=S120/SN120) 순서
# 멤버 매핑 원칙: 구 SN120→신 SN120근처, 구 SN30→신 SN30근처, 성격 보존
CANDIDATES = {
    'A_baseline_120_30_120':   (120, 30, 120),
    'B_min_shift_123_33_123':  (123, 33, 123),   # 2 멤버 SN=123 동기 잔존 (ablation)
    'C_recommend_123_33_129':  (123, 33, 129),   # 권고: p=[41,11,43] 모두 서로소
    'D_alt_111_39_141':        (111, 39, 141),   # 대안: p=[37,13,47]
}


def generate_trace(data, cfg):
    bars, funding = data[cfg['interval']]
    trace = []
    run_cfg = dict(cfg)
    interval = run_cfg.pop('interval')
    bf_run(bars, funding, interval=interval, leverage=1.0,
           start_date=START, end_date=END,
           _trace=trace, **run_cfg)
    return trace


def run_candidate(data, sn_tuple, label):
    """4 config 중 1개 실행. 3 멤버 EW 1/3 ensemble, L3 고정 3x, 가드 없음."""
    sn1, sn2, sn3 = sn_tuple
    traces = {}
    for (name, base), sn in zip(MEMBER_BASE.items(), [sn1, sn2, sn3]):
        cfg = dict(base)
        cfg['snap_interval_bars'] = sn
        traces[name] = generate_trace(data, cfg)

    bars_1h, funding_1h = data['1h']
    all_dates = bars_1h['BTC'].index[(bars_1h['BTC'].index >= START) &
                                       (bars_1h['BTC'].index <= END)]
    weights = {k: 1/3 for k in MEMBER_BASE}
    combined = combine_targets(traces, weights, all_dates)

    # V21 L3 고정 3x, no guards (auto_trade_binance.py 기준)
    engine = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=3.0,
        stop_kind='none',
        stop_pct=0.0,
        stop_gate='none',
        stop_gate_cash_threshold=0.0,
        per_coin_leverage_mode='fixed',
        leverage_floor=3.0, leverage_mid=3.0, leverage_ceiling=3.0,
    )
    m = engine.run(combined)
    m['label'] = label
    m['SN_M1'] = sn1
    m['SN_M2'] = sn2
    m['SN_M3'] = sn3
    return m


def main():
    t0 = time.time()
    print('[P0] Loading data...')
    intervals = ['4h', '1h']
    data = {iv: load_data(iv) for iv in intervals}
    print(f'  load done ({time.time()-t0:.0f}s)')

    results = []
    for label, sn in CANDIDATES.items():
        print(f'\n[P0] Running {label} SN={sn} ...')
        t1 = time.time()
        m = run_candidate(data, sn, label)
        m['elapsed_s'] = round(time.time() - t1, 1)
        results.append(m)
        print(f'  done ({m["elapsed_s"]}s) Cal={m.get("Cal", 0):.2f} '
              f'CAGR={m.get("CAGR", 0):+.2%} MDD={m.get("MDD", 0):+.2%} '
              f'Rebal={m.get("Rebal", 0)} Stops={m.get("Stops", 0)} '
              f'Liq={m.get("Liq", 0)}')

    df = pd.DataFrame(results)
    out_csv = os.path.join(OUT, 'phase0_results.csv')
    df.to_csv(out_csv, index=False)
    print(f'\n[P0] saved {out_csv}')

    cols = ['label', 'SN_M1', 'SN_M2', 'SN_M3', 'Cal', 'CAGR', 'MDD',
            'Sharpe', 'Rebal', 'Liq', 'elapsed_s']
    cols_present = [c for c in cols if c in df.columns]
    print('\n[P0] Summary')
    print(df[cols_present].to_string(index=False))

    # 변화량 (A 대비)
    if 'A_baseline_120_30_120' in df['label'].values:
        base = df[df['label'] == 'A_baseline_120_30_120'].iloc[0]
        print('\n[P0] Delta vs A baseline (%)')
        for _, row in df.iterrows():
            if row['label'] == 'A_baseline_120_30_120':
                continue
            d_cal = (row['Cal'] - base['Cal']) / abs(base['Cal']) * 100
            d_cagr = (row['CAGR'] - base['CAGR']) / abs(base['CAGR']) * 100
            d_mdd = (row['MDD'] - base['MDD']) / abs(base['MDD']) * 100
            d_reb = row.get('Rebal', 0) - base.get('Rebal', 0)
            print(f"  {row['label']}: Cal {d_cal:+.1f}% CAGR {d_cagr:+.1f}% "
                  f"MDD {d_mdd:+.1f}% Rebal{d_reb:+d}")

    print(f'\n[P0] Total elapsed {time.time()-t0:.0f}s')


if __name__ == '__main__':
    main()
