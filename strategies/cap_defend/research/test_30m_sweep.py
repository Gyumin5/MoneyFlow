#!/usr/bin/env python3
"""30m 타임프레임 Phase 1: 1x 브루트포스 스윕.

30m은 bpd=48. 파라미터는 봉 단위.
SMA 240봉(30m) = 5일, SMA 1200봉 = 25일 등.
"""

import os, sys, time
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
    configs = []

    # 30m 봉 기준 파라미터
    # SMA: 기존 1h의 2배 스케일 (168h→336, 240h→480 등)
    sma_list = [120, 240, 336, 480, 720]
    mom_short_list = [10, 20, 48, 72, 96]
    mom_long_list = [60, 120, 240, 720, 1440]
    snap_list = [15, 30, 60, 120]

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
                for vt in [0.03, 0.05]:
                    for snap in snap_list:
                        cfg = dict(
                            sma_bars=sma, mom_short_bars=ms, mom_long_bars=ml,
                            health_mode='mom2vol', vol_mode='daily', vol_threshold=vt,
                            snap_interval_bars=snap, **FIXED_BASE,
                        )
                        configs.append(cfg)

    # mom1vol + daily
    for sma in sma_list:
        for ms in mom_short_list:
            for vt in [0.03, 0.05]:
                for snap in snap_list:
                    cfg = dict(
                        sma_bars=sma, mom_short_bars=ms, mom_long_bars=60,
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
    return f"30m(S{sma},M{ms}/{ml},{hm},{vstr},sn{sn})"


_DATA = {}

def _init_worker():
    global _DATA
    _DATA['30m'] = load_data('30m')


def _run_one(cfg):
    try:
        bars, funding = _DATA['30m']
        m = run(bars, funding, interval='30m', leverage=1.0,
                start_date=START, end_date=END, **cfg)
        return (cfg, m)
    except Exception as e:
        return (cfg, {'CAGR': -999, 'MDD': -999, 'Cal': -999, 'error': str(e)})


def main():
    t0 = time.time()
    outpath = os.path.join(os.path.dirname(os.path.abspath(__file__)), "30m_sweep_results.txt")
    outf = open(outpath, 'w')

    def log(msg=""):
        print(msg, flush=True)
        outf.write(msg + "\n")
        outf.flush()

    configs = build_configs()
    n_configs = len(configs)
    n_workers = max(1, min(cpu_count() - 1, 24))

    log(f"30m 브루트포스 스윕: {n_configs}개")
    log(f"Workers: {n_workers}")
    log(f"  sma: [120, 240, 336, 480, 720]")
    log(f"  mom_short: [10, 20, 48, 72, 96]")
    log(f"  mom_long: [60, 120, 240, 720, 1440]")
    log(f"  snap: [15, 30, 60, 120]")
    log(f"  health: mom2vol(bar+daily) + mom1vol(daily)")
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

    results.sort(key=lambda x: x[1]['Cal'], reverse=True)

    log(f"\n  --- 30m Top 30 (Cal 순) ---")
    log(f"    {'#':>3s} {'설정':<65s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s}")
    log(f"  {'-'*95}")
    for i, (cfg, m) in enumerate(results[:30], 1):
        label = cfg_label(cfg)
        log(f"    {i:>3d} {label:<65s} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%} {m['Cal']:>6.2f}")

    results_cagr = sorted(results, key=lambda x: x[1]['CAGR'], reverse=True)
    log(f"\n  --- 30m Top 30 (CAGR 순) ---")
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
