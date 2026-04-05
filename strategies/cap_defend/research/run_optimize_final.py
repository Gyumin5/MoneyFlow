#!/usr/bin/env python3
"""최종 최적화 — 순수 봉 기반, DD/BL/Crash OFF, 과적합 검증 포함.

순서: D → 4h → 1h
내부 병렬: multiprocessing으로 CPU 코어 활용
"""
import sys, os, time, json
from multiprocessing import Pool, cpu_count
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
from backtest_futures_full import load_data, run

START = '2020-10-01'
END = '2026-03-28'
# 서브기간 (과적합 검증용)
PERIODS = [
    ('Full', '2020-10-01', '2026-03-28'),
    ('Bull', '2020-10-01', '2021-12-31'),
    ('Bear', '2022-01-01', '2023-01-31'),
    ('Recovery', '2023-02-01', '2026-03-28'),
]


def build_final_grid(interval):
    """간격별 최종 그리드."""
    if interval == 'D':
        sma_list = [10, 20, 30, 40, 50, 60]
        snap_list = [10, 20, 30]
    elif interval == '4h':
        sma_list = [30, 60, 120, 240, 300]
        snap_list = [18, 30, 60, 120]
    else:  # 1h
        sma_list = [50, 100, 200, 500, 1000, 1200]
        snap_list = [24, 72, 168, 336]

    mom_short_list = [5, 10, 20, 50, 100, 200]
    mom_long_mult = [0, 3, 6]  # 0=단일, 3=short*3, 6=short*6
    health_list = ['mom2vol', 'mom1']
    vol_list = ['daily', 'bar']

    grid = []
    for sma in sma_list:
        for ms in mom_short_list:
            if ms > sma * 3:  # 모멘텀이 SMA의 3배 초과면 skip
                continue
            for ml_mult in mom_long_mult:
                ml = ms * ml_mult if ml_mult > 0 else 0
                if ml_mult == 0:
                    hmode_options = ['mom1vol', 'mom1']  # 단일 mom
                else:
                    hmode_options = ['mom2vol']  # 이중 mom은 mom2vol만 (mom1은 중복)

                for hmode in hmode_options:
                    if 'mom2' in hmode and ml == 0:
                        continue  # mom2 모드인데 long mom 없으면 skip
                    for vm in vol_list:
                        if 'vol' not in hmode and vm == 'bar':
                            continue  # vol 안 쓰는 헬스모드면 vol_mode 변경 무의미
                        for snap in snap_list:
                            grid.append(dict(
                                sma_bars=sma,
                                mom_short_bars=ms,
                                mom_long_bars=ml if ml > 0 else ms * 3,  # 기본값
                                canary_hyst=0.015,
                                drift_threshold=0.0,
                                dd_threshold=0, dd_lookback=0,
                                bl_drop=0, bl_days=0,
                                crash_threshold=-0.10,
                                daily_gate=False,
                                health_mode=hmode,
                                vol_mode=vm,
                                vol_threshold=0.05 if vm == 'daily' else 0.80,
                                n_snapshots=3,
                                snap_interval_bars=snap,
                            ))
    return grid


# 병렬 실행용 글로벌 변수
_g_bars = None
_g_funding = None
_g_interval = None

def _init_worker(bars, funding, interval):
    global _g_bars, _g_funding, _g_interval
    _g_bars = bars
    _g_funding = funding
    _g_interval = interval

def _run_single(cfg):
    """단일 config 실행 (worker)."""
    m = run(_g_bars, _g_funding, interval=_g_interval, leverage=1.0,
            start_date=START, end_date=END, **cfg)
    if m and m.get('CAGR', 0) != 0:
        return {**cfg, **m}
    return None


def _run_subperiod(args):
    """서브기간 실행 (worker)."""
    cfg, period_name, start, end = args
    m = run(_g_bars, _g_funding, interval=_g_interval, leverage=1.0,
            start_date=start, end_date=end, **cfg)
    if m and m.get('CAGR', 0) != 0:
        return (period_name, m)
    return (period_name, None)


def print_top(results, n=20):
    """상위 N개 출력."""
    results.sort(key=lambda x: x.get('Cal', 0), reverse=True)
    print(f"  {'SMA':>4s} {'Mom':>8s} {'Health':>7s} {'Vol':>4s} {'Snap':>4s}"
          f" | {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Rb':>5s}")
    print(f"  {'-'*65}")
    for r in results[:n]:
        ms = r['mom_short_bars']
        hm = r['health_mode']
        if 'mom2' in hm:
            mom = f"{ms}/{r['mom_long_bars']}"
        else:
            mom = str(ms)
        vm = 'D' if r['vol_mode'] == 'daily' else 'B'
        print(f"  {r['sma_bars']:>4d} {mom:>8s} {hm:>7s} {vm:>4s} {r['snap_interval_bars']:>4d}"
              f" | {r['Sharpe']:>5.2f} {r['CAGR']:>+8.1%} {r['MDD']:>+8.1%} {r['Cal']:>6.2f} {r['Rebal']:>5d}")


def select_diverse_top5(results):
    """과적합 필터링 후 서로 다른 전략 5개 선정."""
    sorted_r = sorted(results, key=lambda x: x.get('Cal', 0), reverse=True)
    selected = []
    for r in sorted_r:
        if len(selected) >= 5:
            break
        # 이미 선택된 것과 SMA/Mom이 유사하면 skip
        is_diverse = True
        for s in selected:
            sma_sim = abs(r['sma_bars'] - s['sma_bars']) <= max(r['sma_bars'], s['sma_bars']) * 0.3
            mom_sim = abs(r['mom_short_bars'] - s['mom_short_bars']) <= max(r['mom_short_bars'], s['mom_short_bars']) * 0.3
            if sma_sim and mom_sim:
                is_diverse = False
                break
        if is_diverse:
            selected.append(r)
    return selected


if __name__ == '__main__':
    t_total = time.time()
    n_workers = max(1, cpu_count() - 1)
    print(f"CPU: {cpu_count()} cores, workers: {n_workers}\n")

    intervals = sys.argv[1:] if len(sys.argv) > 1 else ['D', '4h', '1h']

    for interval in intervals:
        t1 = time.time()
        grid = build_final_grid(interval)
        bars, funding = load_data(interval)
        print(f"\n{'='*85}")
        print(f"  {interval} | {len(bars)} coins | {len(grid)} configs | {n_workers} workers")
        print(f"{'='*85}")

        # 병렬 sweep
        with Pool(n_workers, initializer=_init_worker, initargs=(bars, funding, interval)) as pool:
            raw = pool.map(_run_single, grid, chunksize=10)
        results = [r for r in raw if r is not None]
        elapsed = time.time() - t1
        print(f"\n  {interval}: {len(results)} 유효 / {len(grid)} ({elapsed:.0f}s)")

        # Top 20
        print(f"\n  ── {interval} Top 20 by Calmar (1x) ──")
        print_top(results, 20)

        # Diverse Top 5
        top5 = select_diverse_top5(results)
        print(f"\n  ── {interval} Diverse Top 5 ──")

        for i, cfg in enumerate(top5):
            ms = cfg['mom_short_bars']
            hm = cfg['health_mode']
            mom_str = f"{ms}/{cfg['mom_long_bars']}" if 'mom2' in hm else str(ms)
            vm = 'daily' if cfg['vol_mode'] == 'daily' else 'bar'
            print(f"\n  #{i+1}: SMA={cfg['sma_bars']}b Mom={mom_str}b {hm} Vol={vm} Snap={cfg['snap_interval_bars']}b")

            # Config for re-runs
            cp = {k: cfg[k] for k in cfg if k not in ('Sharpe','CAGR','MDD','Cal','Liq','Trades','Rebal')}

            # 레버리지 테스트
            print(f"    {'Lev':>4s} {'Sh':>6s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Rb':>5s}")
            for lev in [1.0, 1.5, 2.0, 3.0]:
                m = run(bars, funding, interval=interval, leverage=lev,
                        start_date=START, end_date=END, **cp)
                if not m: continue
                liq = f' Liq{m["Liq"]}' if m.get('Liq', 0) > 0 else ''
                print(f"    {lev:>4.1f} {m['Sharpe']:>6.2f} {m['CAGR']:>+8.1%} {m['MDD']:>+8.1%}"
                      f" {m['Cal']:>6.2f} {m['Rebal']:>5d}{liq}")

            # 서브기간 일관성 (1x)
            print(f"    서브기간:")
            for pname, pstart, pend in PERIODS:
                m = run(bars, funding, interval=interval, leverage=1.0,
                        start_date=pstart, end_date=pend, **cp)
                if m and m.get('CAGR', 0) != 0:
                    print(f"      {pname:<10s} Sh={m['Sharpe']:>5.2f} CAGR={m['CAGR']:>+7.1%}"
                          f" MDD={m['MDD']:>+7.1%} Cal={m['Cal']:>5.2f}")
                else:
                    print(f"      {pname:<10s} N/A")

            # 인접 파라미터 plateau
            print(f"    Plateau (SMA ±30%, Mom ±50%):")
            sma_base = cfg['sma_bars']
            mom_base = cfg['mom_short_bars']
            for sma_f in [0.7, 1.0, 1.3]:
                for mom_f in [0.5, 1.0, 1.5]:
                    if sma_f == 1.0 and mom_f == 1.0:
                        continue
                    adj = {**cp, 'sma_bars': int(sma_base * sma_f), 'mom_short_bars': int(mom_base * mom_f)}
                    if 'mom2' in hm:
                        adj['mom_long_bars'] = int(adj['mom_short_bars'] * cfg['mom_long_bars'] / mom_base) if mom_base > 0 else adj['mom_short_bars'] * 3
                    m = run(bars, funding, interval=interval, leverage=1.0,
                            start_date=START, end_date=END, **adj)
                    if m and m.get('Cal', 0) > 0:
                        tag = f"SMA{int(sma_base*sma_f)} Mom{int(mom_base*mom_f)}"
                        print(f"      {tag:<16s} Cal={m['Cal']:>5.2f} CAGR={m['CAGR']:>+7.1%}")

        sys.stdout.flush()
        print(f"\n  {interval} 소요: {time.time()-t1:.0f}s")

    print(f"\n총 소요: {time.time()-t_total:.0f}s")
