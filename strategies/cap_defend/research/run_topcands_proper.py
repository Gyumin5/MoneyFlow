#!/usr/bin/env python3
"""Top 후보들을 SingleAccountEngine(proper merged target-weight)으로 재검증.

기존 sweep_candidate_mixes.py / ensemble_cross_interval.py는 각 전략을 독립
백테스트 후 equity 산술평균 → Engine A 방식 → 과대평가.

이 스크립트: backtest_futures_full.run으로 각 멤버의 target-weight trace 생성,
SingleAccountEngine로 merged single-account 평가 → 실매매와 동일한 엔진 B 방식.

모드:
  SPOT-like: leverage=1.0, tx=0.004, stop=none, no funding adjust
  FUT-3x:    leverage=3.0, tx=0.0004  (그리드 기본)
  FUT-5x+G:  leverage=5.0 (capmom 543 cash), stop prev_close15+cash_guard34, tx=0.0004

데이터는 모두 선물 bars/funding. leverage=1.0 땐 funding PnL이 캔슬될 정도로 작음.

출력: 각 모드별 k=2~5 EW 조합 rank-sum Top 10.
"""
import csv
import json
import os
import sys
import time
from itertools import combinations

_here = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _here)
sys.path.insert(0, os.path.dirname(_here))

from backtest_futures_full import load_data, run
from run_ensemble import SingleAccountEngine, combine_targets

START = '2020-10-01'
END = '2026-03-28'
OUT_DIR = os.path.join(_here, 'grid_results')
os.makedirs(OUT_DIR, exist_ok=True)

BASE = dict(canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0,
            bl_drop=0, bl_days=0, health_mode='mom2vol', n_snapshots=3,
            vol_mode='daily', vol_threshold=0.05)

# 후보 정의 (ensemble_candidates.json + d005 멤버)
CANDS_SPOT = {
    'D_S50_M20_L90_Sn120':  dict(interval='D',  sma_bars=50,  mom_short_bars=20, mom_long_bars=90,  snap_interval_bars=120, **BASE),
    'D_S50_M80_L90_Sn120':  dict(interval='D',  sma_bars=50,  mom_short_bars=80, mom_long_bars=90,  snap_interval_bars=120, **BASE),
    'D_S50_M20_L90_Sn60':   dict(interval='D',  sma_bars=50,  mom_short_bars=20, mom_long_bars=90,  snap_interval_bars=60,  **BASE),
    '4h_S240_M30_L720_Sn120': dict(interval='4h', sma_bars=240, mom_short_bars=30, mom_long_bars=720, snap_interval_bars=120, **BASE),
    '4h_S240_M30_L120_Sn60':  dict(interval='4h', sma_bars=240, mom_short_bars=30, mom_long_bars=120, snap_interval_bars=60,  **BASE),
    '4h_S720_M30_L720_Sn60':  dict(interval='4h', sma_bars=720, mom_short_bars=30, mom_long_bars=720, snap_interval_bars=60,  **BASE),
    '2h_S480_M10_L120_Sn120': dict(interval='2h', sma_bars=480, mom_short_bars=10, mom_long_bars=120, snap_interval_bars=120, **BASE),
    '2h_S480_M10_L240_Sn120': dict(interval='2h', sma_bars=480, mom_short_bars=10, mom_long_bars=240, snap_interval_bars=120, **BASE),
    '2h_S1200_M10_L240_Sn120':dict(interval='2h', sma_bars=1200,mom_short_bars=10, mom_long_bars=240, snap_interval_bars=120, **BASE),
    'V19_D_S50_M30_L90_Sn30':dict(interval='D',  sma_bars=50,  mom_short_bars=30, mom_long_bars=90,  snap_interval_bars=30,  **BASE),
}

CANDS_FUT = {
    'D_S50_M20_L90_Sn120':   dict(interval='D',  sma_bars=50,  mom_short_bars=20, mom_long_bars=90,  snap_interval_bars=120, **BASE),
    'D_S50_M20_L90_Sn90':    dict(interval='D',  sma_bars=50,  mom_short_bars=20, mom_long_bars=90,  snap_interval_bars=90,  **BASE),
    'D_S50_M80_L240_Sn120':  dict(interval='D',  sma_bars=50,  mom_short_bars=80, mom_long_bars=240, snap_interval_bars=120, **BASE),
    '4h_S240_M30_L720_Sn120':dict(interval='4h', sma_bars=240, mom_short_bars=30, mom_long_bars=720, snap_interval_bars=120, **BASE),
    '4h_S240_M30_L720_Sn60': dict(interval='4h', sma_bars=240, mom_short_bars=30, mom_long_bars=720, snap_interval_bars=60,  **BASE),
    '4h_S240_M30_L90_Sn60':  dict(interval='4h', sma_bars=240, mom_short_bars=30, mom_long_bars=90,  snap_interval_bars=60,  **BASE),
    '2h_S480_M10_L120_Sn120':dict(interval='2h', sma_bars=480, mom_short_bars=10, mom_long_bars=120, snap_interval_bars=120, **BASE),
    '2h_S1200_M10_L720_Sn90':dict(interval='2h', sma_bars=1200,mom_short_bars=10, mom_long_bars=720, snap_interval_bars=90,  **BASE),
    '2h_S480_M20_L60_Sn180': dict(interval='2h', sma_bars=480, mom_short_bars=20, mom_long_bars=60,  snap_interval_bars=180, **BASE),
    'V19_D_S50_M30_L90_Sn30':dict(interval='D',  sma_bars=50,  mom_short_bars=30, mom_long_bars=90,  snap_interval_bars=30,  **BASE),
}

# 참조: d005 4멤버, V19 equivalent
D005_MEMBERS = {
    'd005_4h_orig': dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=720, snap_interval_bars=60,  vol_mode='daily', vol_threshold=0.05,
                         canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0, health_mode='mom2vol', n_snapshots=3),
    'd005_2h_S240': dict(interval='2h', sma_bars=240, mom_short_bars=20, mom_long_bars=720, snap_interval_bars=120, vol_mode='bar',   vol_threshold=0.60,
                         canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0, health_mode='mom2vol', n_snapshots=3),
    'd005_2h_S120': dict(interval='2h', sma_bars=120, mom_short_bars=20, mom_long_bars=720, snap_interval_bars=120, vol_mode='bar',   vol_threshold=0.60,
                         canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0, health_mode='mom2vol', n_snapshots=3),
    'd005_4h_M20':  dict(interval='4h', sma_bars=240, mom_short_bars=20, mom_long_bars=120, snap_interval_bars=21,  vol_mode='bar',   vol_threshold=0.60,
                         canary_hyst=0.015, drift_threshold=0, dd_threshold=0, dd_lookback=0, bl_drop=0, bl_days=0, health_mode='mom2vol', n_snapshots=3),
}

MODES = {
    'SPOT_1x':   dict(leverage=1.0, tx_cost=0.004, stop_kind='none',
                      leverage_mode='fixed', per_coin_leverage_mode='none'),
    'FUT_1x':    dict(leverage=1.0, tx_cost=0.0004, stop_kind='none',
                      leverage_mode='fixed', per_coin_leverage_mode='none'),
}


def gen_trace(data, cfg):
    cfg = dict(cfg)
    interval = cfg.pop('interval')
    bars, funding = data[interval]
    trace = []
    # tx cost는 엔진에서 계산 — trace 생성 시엔 무관 (target weight만 찍음)
    run(bars, funding, interval=interval, leverage=1.0,
        start_date=START, end_date=END, _trace=trace, **cfg)
    return trace


def eval_ensemble(bars_1h, funding_1h, traces, members, mode_cfg):
    weights = {k: 1.0/len(members) for k in members}
    btc = bars_1h['BTC']
    all_dates = btc.index[(btc.index >= START) & (btc.index <= END)]
    sub = {k: traces[k] for k in members}
    combined = combine_targets(sub, weights, all_dates)
    g = dict(mode_cfg)
    eng = SingleAccountEngine(
        bars_1h, funding_1h,
        leverage=g.get('leverage', 1.0),
        tx_cost=g.get('tx_cost', 0.0004),
        stop_kind=g.get('stop_kind', 'none'),
        stop_pct=g.get('stop_pct', 0.0),
        stop_gate=g.get('stop_gate', 'always'),
        stop_gate_cash_threshold=g.get('stop_gate_cash_threshold', 0.0),
        leverage_mode=g.get('leverage_mode', 'fixed'),
        per_coin_leverage_mode=g.get('per_coin_leverage_mode', 'none'),
        leverage_floor=3.0, leverage_mid=4.0, leverage_ceiling=5.0,
        leverage_cash_threshold=0.34,
        leverage_partial_cash_threshold=0.0,
        leverage_count_floor_max=2, leverage_count_mid_max=4,
        leverage_canary_floor_gap=0.015,
        leverage_canary_mid_gap=0.04,
        leverage_canary_high_gap=0.08,
        leverage_canary_sma_bars=1200,
        leverage_mom_lookback_bars=24*30,
        leverage_vol_lookback_bars=24*90,
    )
    return eng.run(combined)


def rank_sum(rows, keys=('Sharpe', 'CAGR', 'Cal')):
    """각 지표별 내림차순 rank 합. 낮을수록 좋음."""
    ranks = {}
    for key in keys:
        sorted_rows = sorted(rows, key=lambda r: -r[key])
        for i, r in enumerate(sorted_rows):
            ranks.setdefault(r['combo_id'], []).append(i+1)
    for r in rows:
        r['rank_sum'] = sum(ranks[r['combo_id']])
    return sorted(rows, key=lambda r: r['rank_sum'])


def main():
    t0 = time.time()
    print('Loading data: D, 4h, 2h, 1h ...', flush=True)
    data = {iv: load_data(iv) for iv in ['D', '4h', '2h', '1h']}
    bars_1h, funding_1h = data['1h']

    # Dedup candidates across SPOT/FUT/d005
    all_cfgs = {}
    for name, cfg in {**CANDS_SPOT, **CANDS_FUT, **D005_MEMBERS}.items():
        key = (cfg['interval'], cfg['sma_bars'], cfg['mom_short_bars'],
               cfg['mom_long_bars'], cfg['snap_interval_bars'],
               cfg.get('vol_mode'), cfg.get('vol_threshold'))
        if key not in all_cfgs:
            all_cfgs[key] = (name, cfg)
    print(f'Unique traces: {len(all_cfgs)}', flush=True)

    name_to_key = {}
    for name, cfg in {**CANDS_SPOT, **CANDS_FUT, **D005_MEMBERS}.items():
        key = (cfg['interval'], cfg['sma_bars'], cfg['mom_short_bars'],
               cfg['mom_long_bars'], cfg['snap_interval_bars'],
               cfg.get('vol_mode'), cfg.get('vol_threshold'))
        name_to_key[name] = key

    # Generate traces
    traces = {}
    for i, (key, (name, cfg)) in enumerate(all_cfgs.items(), 1):
        ts = time.time()
        traces[key] = gen_trace(data, cfg)
        print(f'  [{i}/{len(all_cfgs)}] {name} ({cfg["interval"]}) {time.time()-ts:.0f}s', flush=True)

    # Build ensemble combos per mode
    SPOT_NAMES = list(CANDS_SPOT.keys())
    FUT_NAMES = list(CANDS_FUT.keys())

    all_results = {}
    for mode_name, mode_cfg in MODES.items():
        cand_names = SPOT_NAMES if mode_name == 'SPOT_1x' else FUT_NAMES
        print(f'\n=== Mode: {mode_name} ({len(cand_names)} cands) ===', flush=True)

        rows = []
        # k=1 singles
        for name in cand_names:
            ts = time.time()
            m = eval_ensemble(bars_1h, funding_1h, {name: traces[name_to_key[name]]}, [name], mode_cfg)
            dt = time.time() - ts
            rows.append(dict(
                combo_id=f'k1|{name}', k=1, members=name,
                Sharpe=m.get('Sharpe', 0), CAGR=m.get('CAGR', 0),
                MDD=m.get('MDD', 0), Cal=m.get('Cal', 0),
                Liq=m.get('Liq', 0), Stops=m.get('Stops', 0),
            ))
            print(f'  k1 {name:<25} Sh{m.get("Sharpe",0):.2f} Cal{m.get("Cal",0):.2f} '
                  f'CAGR{m.get("CAGR",0):+.1%} MDD{m.get("MDD",0):+.1%} ({dt:.0f}s)', flush=True)

        # k=1 incremental save
        partial_csv = os.path.join(OUT_DIR, f'topcands_proper_{mode_name}_k1_partial.csv')
        with open(partial_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        print(f'  partial saved (k=1): {partial_csv}', flush=True)

        # k=2..4 combos (k=5 생략 — over-engineering)
        for k in [2, 3, 4]:
            combos = list(combinations(cand_names, k))
            print(f'  k={k}: {len(combos)} combos', flush=True)
            for j, members in enumerate(combos):
                # rename to use trace names
                trace_dict = {n: traces[name_to_key[n]] for n in members}
                ts = time.time()
                m = eval_ensemble(bars_1h, funding_1h, trace_dict, list(members), mode_cfg)
                dt = time.time() - ts
                rows.append(dict(
                    combo_id=f'k{k}|{"+".join(members)}', k=k, members='+'.join(members),
                    Sharpe=m.get('Sharpe', 0), CAGR=m.get('CAGR', 0),
                    MDD=m.get('MDD', 0), Cal=m.get('Cal', 0),
                    Liq=m.get('Liq', 0), Stops=m.get('Stops', 0),
                ))
                if (j+1) % 20 == 0:
                    print(f'    [{j+1}/{len(combos)}] last {dt:.1f}s', flush=True)

            # k 레벨별 incremental save
            partial_csv = os.path.join(OUT_DIR, f'topcands_proper_{mode_name}_k{k}_partial.csv')
            with open(partial_csv, 'w', newline='') as f:
                w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                w.writeheader()
                w.writerows(rows)
            print(f'  partial saved (k<={k}): {partial_csv}', flush=True)

        # Add d005 reference
        d005_mems = list(D005_MEMBERS.keys())
        trace_dict = {n: traces[name_to_key[n]] for n in d005_mems}
        ts = time.time()
        m = eval_ensemble(bars_1h, funding_1h, trace_dict, d005_mems, mode_cfg)
        rows.append(dict(
            combo_id=f'REF|d005_B0', k=4, members='+'.join(d005_mems),
            Sharpe=m.get('Sharpe', 0), CAGR=m.get('CAGR', 0),
            MDD=m.get('MDD', 0), Cal=m.get('Cal', 0),
            Liq=m.get('Liq', 0), Stops=m.get('Stops', 0),
        ))
        print(f'  d005_B0 ref: Sh{m.get("Sharpe",0):.2f} Cal{m.get("Cal",0):.2f} CAGR{m.get("CAGR",0):+.1%}', flush=True)

        ranked = rank_sum(rows)
        all_results[mode_name] = ranked

        # Save CSV
        out_csv = os.path.join(OUT_DIR, f'topcands_proper_{mode_name}.csv')
        with open(out_csv, 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(ranked[0].keys()))
            w.writeheader()
            w.writerows(ranked)
        print(f'  Saved: {out_csv}', flush=True)

        # Top 15 rank-sum
        print(f'\n=== {mode_name} Rank-Sum Top 15 ===')
        print(f'{"rk":>4} {"k":>2} {"Sh":>5} {"Cal":>5} {"CAGR":>7} {"MDD":>7}  members')
        for r in ranked[:15]:
            print(f'{r["rank_sum"]:>4} {r["k"]:>2} {r["Sharpe"]:>5.2f} {r["Cal"]:>5.2f} '
                  f'{r["CAGR"]:>+7.1%} {r["MDD"]:>+7.1%}  {r["members"][:60]}')

    print(f'\nTotal elapsed: {time.time()-t0:.0f}s', flush=True)


if __name__ == '__main__':
    main()
