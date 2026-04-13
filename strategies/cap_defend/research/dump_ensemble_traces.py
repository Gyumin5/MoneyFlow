#!/usr/bin/env python3
"""앙상블 후보 전략의 equity/cash/picks trace를 pickle로 캐시.

입력: configs/ensemble_candidates.json
출력: grid_results/ensemble_{mode}_traces.pkl

각 후보를 backtest_spot_barfreq.run_backtest로 실행 (record_weights=True)
trace = {name: {'equity': Series, 'cash': Series, 'picks': Series, 'metrics': dict, 'meta': dict}}
"""
import argparse
import json
import os
import pickle
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from coin_engine import Params
from backtest_spot_barfreq import load_binance_prices, run_backtest, BARS_PER_DAY


def make_params(cfg, start, end, mode):
    bpd = BARS_PER_DAY[cfg['interval']]
    tx = 0.004 if mode == 'spot' else 0.0004
    return Params(
        canary='K8', vote_smas=(cfg['sma'],), vote_moms=(), vote_threshold=1,
        canary_band=1.5, health='HK', health_sma=0,
        health_mom_short=cfg['ms'], health_mom_long=cfg['ml'],
        health_vol_window=90 * bpd, vol_cap=0.05,
        sma_period=cfg['sma'], selection='baseline', n_picks=5,
        weighting='baseline', top_n=40, risk='baseline', tx_cost=tx,
        start_date=start, end_date=end,
        dd_exit_lookback=0, dd_exit_threshold=0,
        bl_threshold=0, bl_days=0, drift_threshold=0,
    )


def run_one(cfg, mode, start, end, prices, fm, funding):
    iv = cfg['interval']
    bpd = BARS_PER_DAY[iv]
    leverage = 1.0 if mode == 'spot' else 3.0
    tx = 0.004 if mode == 'spot' else 0.0004
    params = make_params(cfg, start, end, mode)
    t0 = time.time()
    result = run_backtest(
        prices, fm, params, bars_per_day=bpd,
        snap_interval_bars=cfg['snap'], n_snap=3,
        mode=mode, leverage=leverage, tx_cost=tx,
        funding_data=funding if mode == 'futures' else None,
        record_weights=True,
    )
    elapsed = time.time() - t0
    return cfg['name'], {
        'equity': result['equity'],
        'cash': result['cash'],
        'picks': result['picks'],
        'metrics': result['metrics'],
        'rebal': result['rebal_count'],
        'meta': dict(cfg, mode=mode, elapsed=elapsed),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', required=True, choices=['spot', 'futures'])
    ap.add_argument('--candidates', default=None,
                    help='JSON config (default: configs/ensemble_candidates.json)')
    ap.add_argument('--out', default=None,
                    help='pickle output path (default: grid_results/ensemble_{mode}_traces.pkl)')
    ap.add_argument('--start', default='2020-10-01')
    ap.add_argument('--end', default='2026-03-31')
    args = ap.parse_args()

    here = os.path.dirname(os.path.abspath(__file__))
    cand_path = args.candidates or os.path.join(here, 'configs/ensemble_candidates.json')
    out_path = args.out or os.path.join(here, f'grid_results/ensemble_{args.mode}_traces.pkl')

    with open(cand_path) as f:
        cfg_all = json.load(f)
    candidates = []
    for tf in ['D', '4h', '2h']:
        candidates.extend(cfg_all[args.mode][tf])

    print(f"[{args.mode}] {len(candidates)} candidates")
    for c in candidates:
        print(f"  - {c['name']}")

    # 인터벌 한 번씩만 로드되도록 인터벌 그룹화 (병렬은 인터벌 내부에서)
    by_iv = {}
    for c in candidates:
        by_iv.setdefault(c['interval'], []).append(c)

    traces = {}
    t0 = time.time()
    for iv, cs in by_iv.items():
        print(f"\n=== {iv} ({len(cs)}개) ===")
        # 인터벌당 데이터 1회 로드 후 후보들을 직렬 실행 (단일 백테스트가 수초)
        prices, fm, funding = load_binance_prices(iv, top_n=40)
        for c in cs:
            name, tr = run_one(c, args.mode, args.start, args.end, prices, fm, funding)
            traces[name] = tr
            m = tr['metrics']
            cal = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
            print(f"  {name:35s} Sh {m['Sharpe']:.2f}  CAGR {m['CAGR']:+.1%}  MDD {m['MDD']:+.1%}  Cal {cal:.2f}  ({tr['meta']['elapsed']:.0f}s)")

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'wb') as f:
        pickle.dump(traces, f)
    print(f"\nSaved: {out_path}  ({time.time()-t0:.0f}s total)")


if __name__ == '__main__':
    main()
