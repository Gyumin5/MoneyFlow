#!/usr/bin/env python3
"""Phase A — fut L3 coarse grid (펀딩 fix 후).

axes:
  SMA: 20, 40, 60, 80
  mom_short (MS): 20, 40, 60, 80, 100
  mom_long (ML): 30, 90, 150, 210
  (n_snap, snap_interval_bars): prime stagger 18 combos
  drift_threshold: 0.05, 0.10, 0.15

fixed:
  interval=D, leverage=3.0
  universe_size=3, cap=1/3
  canary_hyst=0.015, vol_threshold=0.05, vol_days=90
  health_mode=mom2vol
  start=2020-10-01, end=2026-03-28 (또는 funding 갱신 후 ~2026-05-13)

총 = 4 × 5 × 4 × 18 × 3 = 4,320 config
"""
import os, sys, json, time, itertools, multiprocessing as mp
from functools import partial

sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(OUT_DIR, exist_ok=True)
RESULTS_CSV = os.path.join(OUT_DIR, 'phase_a_results.csv')

# 축 정의
SMA_VALUES = [20, 40, 60, 80]
MS_VALUES  = [20, 40, 60, 80, 100]
ML_VALUES  = [30, 90, 150, 210]
SNAP_COMBOS = [
    # (n_snap, snap_interval_bars)  prime stagger
    (3, 57), (3, 69), (3, 87), (3, 93), (3, 111), (3, 129),
    (5, 95), (5, 115), (5, 145), (5, 155), (5, 185), (5, 215),
    (7, 133), (7, 161), (7, 203), (7, 217), (7, 259), (7, 301),
]
DRIFT_VALUES = [0.05, 0.10, 0.15]

# 글로벌 데이터 (워커에서 lazy load)
_BARS = None
_FUNDING = None

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
                tx_cost=0.0004, maint_rate=0.004,
                sma_days=cfg['sma'],
                mom_short_days=cfg['ms'],
                mom_long_days=cfg['ml'],
                vol_days=90, vol_threshold=0.05,
                canary_hyst=0.015,
                drift_threshold=cfg['drift'],
                n_snapshots=cfg['n_snap'],
                snap_interval_bars=cfg['snap_int'],
                health_mode='mom2vol',
                start_date='2020-10-01', end_date=_END)
        if not m:
            return {**cfg, 'Sharpe': None}
        return {**cfg, 'Sharpe': m['Sharpe'], 'CAGR': m['CAGR'], 'MDD': m['MDD'],
                'Cal': m['Cal'], 'Rebal': m['Rebal'], 'Liq': m['Liq']}
    except Exception as e:
        return {**cfg, 'Sharpe': None, 'error': str(e)[:80]}

def main():
    end_date = sys.argv[1] if len(sys.argv) > 1 else '2026-03-28'
    n_workers = int(sys.argv[2]) if len(sys.argv) > 2 else 24

    configs = []
    for sma, ms, ml, snap, drift in itertools.product(SMA_VALUES, MS_VALUES, ML_VALUES, SNAP_COMBOS, DRIFT_VALUES):
        configs.append(dict(sma=sma, ms=ms, ml=ml,
                            n_snap=snap[0], snap_int=snap[1], drift=drift))
    print(f"total configs: {len(configs)}, workers={n_workers}, end={end_date}", flush=True)

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
        top = valid.nlargest(10, 'Cal')
        print("\ntop 10 by Cal:")
        print(top[['sma','ms','ml','n_snap','snap_int','drift','Cal','CAGR','MDD','Sharpe','Rebal']].to_string(index=False))

if __name__ == '__main__':
    main()
