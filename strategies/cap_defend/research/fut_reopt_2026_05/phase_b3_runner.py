#!/usr/bin/env python3
"""Phase B3 — ml 상방 boundary 확장 + 다른 peak interior fine. n_snap=3, tx=0.0006."""
import os, sys, time, itertools, multiprocessing as mp
sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_CSV = os.path.join(OUT_DIR, 'phase_b3_results.csv')

SMA_VALUES = [36, 40, 44]
MS_VALUES  = [17, 20, 24]
ML_VALUES  = [122, 141, 165, 190]
SNAP_INTS  = [87, 93, 111, 123]  # n_snap=3 × prime {29,31,37,41}
DRIFT_VALUES = [0.020, 0.030, 0.045]
TX_COST = 0.0006

_BARS = None; _FUNDING = None; _END = None
def _init_worker(end_date):
    global _BARS, _FUNDING, _END
    from backtest_futures_full import load_data
    _BARS, _FUNDING = load_data('D')
    _END = end_date

def _one(cfg):
    from backtest_futures_full import run
    try:
        m = run(_BARS, _FUNDING,
                interval='D', leverage=3.0,
                universe_size=3, selection='greedy', cap=1/3,
                tx_cost=TX_COST, maint_rate=0.004,
                sma_days=cfg['sma'], mom_short_days=cfg['ms'], mom_long_days=cfg['ml'],
                vol_days=90, vol_threshold=0.05,
                canary_hyst=0.015, drift_threshold=cfg['drift'],
                n_snapshots=3, snap_interval_bars=cfg['snap_int'],
                health_mode='mom2vol',
                start_date='2020-10-01', end_date=_END)
        if not m:
            return {**cfg, 'Sharpe': None}
        return {**cfg, 'Sharpe': m['Sharpe'], 'CAGR': m['CAGR'], 'MDD': m['MDD'],
                'Cal': m['Cal'], 'Rebal': m['Rebal'], 'Liq': m['Liq']}
    except Exception as e:
        return {**cfg, 'Sharpe': None, 'error': str(e)[:80]}

def main():
    end_date = sys.argv[1] if len(sys.argv) > 1 else '2026-05-13'
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 24
    configs = []
    for sma, ms, ml, snap_int, drift in itertools.product(SMA_VALUES, MS_VALUES, ML_VALUES, SNAP_INTS, DRIFT_VALUES):
        configs.append(dict(sma=sma, ms=ms, ml=ml, snap_int=snap_int, drift=drift))
    print(f"total configs: {len(configs)}, workers={n_workers}", flush=True)
    t0 = time.time()
    with mp.Pool(n_workers, initializer=_init_worker, initargs=(end_date,)) as pool:
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
    print(f"saved: {RESULTS_CSV}  ({len(df)} rows)  total={time.time()-t0:.0f}s")
    valid = df[df['Sharpe'].notna()]
    if len(valid) > 0:
        top = valid.nlargest(20, 'Cal')
        cols = ['sma','ms','ml','snap_int','drift','Cal','CAGR','MDD','Sharpe','Rebal']
        print("\ntop 20 by Cal:")
        print(top[cols].to_string(index=False))

if __name__ == '__main__':
    main()
