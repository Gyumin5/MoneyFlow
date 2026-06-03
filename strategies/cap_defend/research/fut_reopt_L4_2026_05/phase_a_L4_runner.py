#!/usr/bin/env python3
"""Phase A — L4 fut grid search.

L3 와 동일 구조 (phase_a_runner.py) 의 L4 버전.

axes:
  SMA: 30, 42, 55, 70
  MS: 12, 18, 25, 35
  ML: 90, 127, 160, 200
  (n_snap, snap_int): prime stagger 12 combos
  drift: 0.02, 0.03, 0.05, 0.08

fixed:
  leverage=4.0, universe=3, cap=1/3, tx=0.0006, maint=0.004
  vol_days=90, vol_threshold=0.05, canary_hyst=0.015
  health=mom2vol
  start=2020-10-01, end=2026-05-13

총 = 4 × 4 × 4 × 12 × 4 = 3,072 configs
"""
import os, sys, time, itertools, multiprocessing as mp

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
sys.path.insert(0, '/home/gmoh/mon/251229/trade')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(OUT_DIR, 'phase_a_L4_results.csv')

SMA_VALUES = [30, 42, 55, 70]
MS_VALUES = [12, 18, 25, 35]
ML_VALUES = [90, 127, 160, 200]
SNAP_COMBOS = [
    (3, 57), (3, 87), (3, 111),
    (5, 95), (5, 115), (5, 145), (5, 185),
    (7, 133), (7, 161), (7, 217),
    (9, 171), (9, 207),
]
DRIFT_VALUES = [0.02, 0.03, 0.05, 0.08]

_BARS = None
_FUNDING = None


def _init_worker():
    global _BARS, _FUNDING
    from backtest_futures_full import load_data
    _BARS, _FUNDING = load_data('D')


def _one(cfg):
    from backtest_futures_full import run
    try:
        m = run(_BARS, _FUNDING,
                interval='D', leverage=4.0,
                universe_size=3, selection='greedy', cap=1/3,
                tx_cost=0.0006, maint_rate=0.004,
                sma_days=cfg['sma'],
                mom_short_days=cfg['ms'],
                mom_long_days=cfg['ml'],
                vol_days=90, vol_threshold=0.05,
                canary_hyst=0.015,
                drift_threshold=cfg['drift'],
                n_snapshots=cfg['n_snap'],
                snap_interval_bars=cfg['snap_int'],
                health_mode='mom2vol',
                start_date='2020-10-01', end_date='2026-05-13')
        if not m:
            return {**cfg, 'Sharpe': None}
        return {**cfg, 'Sharpe': m['Sharpe'], 'CAGR': m['CAGR'], 'MDD': m['MDD'],
                'Cal': m['Cal'], 'Rebal': m['Rebal'], 'Liq': m['Liq']}
    except Exception as e:
        return {**cfg, 'Sharpe': None, 'error': str(e)[:80]}


def main():
    n_workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    configs = []
    for sma, ms, ml, snap, drift in itertools.product(SMA_VALUES, MS_VALUES, ML_VALUES, SNAP_COMBOS, DRIFT_VALUES):
        configs.append(dict(sma=sma, ms=ms, ml=ml,
                            n_snap=snap[0], snap_int=snap[1], drift=drift))
    print(f"total configs: {len(configs)}, workers={n_workers}", flush=True)

    t0 = time.time()
    with mp.Pool(n_workers, initializer=_init_worker) as pool:
        results = []
        for i, r in enumerate(pool.imap_unordered(_one, configs, chunksize=4), 1):
            results.append(r)
            if i % 100 == 0 or i == len(configs):
                el = time.time() - t0
                eta = el / i * (len(configs) - i)
                print(f"  {i}/{len(configs)}  elapsed={el:.0f}s  ETA={eta:.0f}s", flush=True)

    import pandas as pd
    df = pd.DataFrame(results)
    df.to_csv(RESULTS_CSV, index=False)
    print(f"saved: {RESULTS_CSV} ({len(df)} rows) total={time.time()-t0:.0f}s")
    valid = df[df['Sharpe'].notna()]
    if len(valid) > 0:
        top = valid.nlargest(20, 'Cal')
        print("\ntop 20 by Cal:")
        print(top[['sma','ms','ml','n_snap','snap_int','drift','Cal','CAGR','MDD','Sharpe','Liq','Rebal']].to_string(index=False))


if __name__ == '__main__':
    main()
