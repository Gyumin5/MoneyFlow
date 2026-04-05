#!/usr/bin/env python3
"""1D 타임프레임 Phase 1: 1x 브루트포스 스윕.

full_sweep + daily_vol_sweep과 동일한 파라미터 그리드.
1D는 bpd=1이므로 snap_interval_bars, sma_bars 등이 일(day) 단위.
"""

import os, sys, time, itertools
from multiprocessing import Pool, cpu_count

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from backtest_futures_full import load_data, run
from futures_live_config import START, END

FIXED_BASE = dict(
    canary_hyst=0.015,
    drift_threshold=0.0,
    dd_threshold=0,
    dd_lookback=0,
    bl_drop=0,
    bl_days=0,
    n_snapshots=3,
)


def build_configs():
    """full_sweep + daily_vol_sweep 합산 그리드 (1D용 조정)."""
    configs = []

    # 1D에서는 파라미터가 "일" 단위
    # 4h SMA 240봉 = 40일, 1h SMA 168봉 = 7일
    # 1D에 맞게 일 단위로 재해석
    sma_list = [10, 20, 40, 60, 120, 200]
    mom_short_list = [5, 10, 20, 30, 60]
    mom_long_list = [20, 60, 120, 240, 720]
    snap_list = [5, 10, 15, 21, 30]

    # mom2vol + bar
    for sma in sma_list:
        for ms in mom_short_list:
            for ml in mom_long_list:
                for vt in [0.60, 0.80]:
                    for snap in snap_list:
                        cfg = dict(
                            sma_bars=sma, mom_short_bars=ms, mom_long_bars=ml,
                            health_mode='mom2vol', vol_mode='bar', vol_threshold=vt,
                            snap_interval_bars=snap, **FIXED_BASE,
                        )
                        configs.append(cfg)

    # mom2vol + daily
    for sma in sma_list:
        for ms in mom_short_list:
            for ml in mom_long_list:
                for vt in [0.03, 0.05, 0.08]:
                    for snap in snap_list:
                        cfg = dict(
                            sma_bars=sma, mom_short_bars=ms, mom_long_bars=ml,
                            health_mode='mom2vol', vol_mode='daily', vol_threshold=vt,
                            snap_interval_bars=snap, **FIXED_BASE,
                        )
                        configs.append(cfg)

    # mom1vol + bar
    for sma in sma_list:
        for ms in mom_short_list:
            for vt in [0.60, 0.80]:
                for snap in snap_list:
                    cfg = dict(
                        sma_bars=sma, mom_short_bars=ms, mom_long_bars=30,
                        health_mode='mom1vol', vol_mode='bar', vol_threshold=vt,
                        snap_interval_bars=snap, **FIXED_BASE,
                    )
                    configs.append(cfg)

    # mom1vol + daily
    for sma in sma_list:
        for ms in mom_short_list:
            for vt in [0.03, 0.05, 0.08]:
                for snap in snap_list:
                    cfg = dict(
                        sma_bars=sma, mom_short_bars=ms, mom_long_bars=30,
                        health_mode='mom1vol', vol_mode='daily', vol_threshold=vt,
                        snap_interval_bars=snap, **FIXED_BASE,
                    )
                    configs.append(cfg)

    return configs


def cfg_label(cfg):
    sma = cfg['sma_bars']
    ms = cfg['mom_short_bars']
    ml = cfg['mom_long_bars']
    hm = cfg['health_mode']
    vm = cfg.get('vol_mode', 'bar')
    vt = cfg['vol_threshold']
    sn = cfg['snap_interval_bars']
    vstr = f"d{vt}" if vm == 'daily' else f"b{vt:.0%}"
    return f"D(S{sma},M{ms}/{ml},{hm},{vstr},sn{sn})"


_DATA = {}

def _init_worker():
    global _DATA
    _DATA['D'] = load_data('D')


def _run_one(cfg):
    try:
        bars, funding = _DATA['D']
        m = run(bars, funding, interval='D', leverage=1.0,
                start_date=START, end_date=END, **cfg)
        return (cfg, m)
    except Exception as e:
        return (cfg, {'CAGR': -999, 'MDD': -999, 'Cal': -999, 'error': str(e)})


def main():
    t0 = time.time()
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "1d_sweep_results.txt")
    outf = open(outpath, 'w')

    def log(msg=""):
        print(msg, flush=True)
        outf.write(msg + "\n")
        outf.flush()

    configs = build_configs()
    n_configs = len(configs)
    n_workers = max(1, min(cpu_count() - 1, 24))

    log(f"1D 브루트포스 스윕: {n_configs}개")
    log(f"Workers: {n_workers}")
    log(f"  sma: [10, 20, 40, 60, 120, 200]")
    log(f"  mom_short: [5, 10, 20, 30, 60]")
    log(f"  mom_long: [20, 60, 120, 240, 720]")
    log(f"  snap: [5, 10, 15, 21, 30]")
    log(f"  health: mom2vol(bar/daily) + mom1vol(bar/daily)")
    log()

    log("=" * 110)
    log("Phase 1: 1x 스윕")
    log("=" * 110)
    log("Loading data...")

    results = []
    done = 0

    with Pool(n_workers, initializer=_init_worker) as pool:
        for result in pool.imap_unordered(_run_one, configs, chunksize=8):
            cfg, m = result
            if 'error' not in m:
                results.append((cfg, m))
            done += 1
            if done % 500 == 0 or done == n_configs:
                log(f"  {done}/{n_configs} ({done/n_configs:.0%}, {time.time()-t0:.0f}s)")

    log(f"\n1x 스윕 완료: {time.time()-t0:.0f}s")

    # Sort by Cal
    results.sort(key=lambda x: x[1]['Cal'], reverse=True)

    log(f"\n  --- D Top 30 (Cal 순) ---")
    log(f"    {'#':>3s} {'설정':<65s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s}")
    log(f"  {'-'*95}")
    for i, (cfg, m) in enumerate(results[:30], 1):
        label = cfg_label(cfg)
        log(f"    {i:>3d} {label:<65s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f}")

    # Top 30 by CAGR
    results_cagr = sorted(results, key=lambda x: x[1]['CAGR'], reverse=True)
    log(f"\n  --- D Top 30 (CAGR 순) ---")
    log(f"    {'#':>3s} {'설정':<65s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s}")
    log(f"  {'-'*95}")
    for i, (cfg, m) in enumerate(results_cagr[:30], 1):
        label = cfg_label(cfg)
        log(f"    {i:>3d} {label:<65s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f}")

    log(f"\n총 소요: {time.time()-t0:.1f}s")
    log(f"결과: {outpath}")
    outf.close()


if __name__ == "__main__":
    main()
