#!/usr/bin/env python3
"""D + 4h + 1h 전체 최적화 (unbuffered output)."""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_futures_full import load_data, run
from backtest_futures_optimize import build_grid_reduced, print_top

START = '2020-10-01'
END = '2026-03-28'

def run_interval(interval, grid):
    t0 = time.time()
    bars, funding = load_data(interval)
    if 'BTC' not in bars:
        print(f"  {interval}: BTC 데이터 없음")
        return []

    results = []
    for i, cfg in enumerate(grid):
        m = run(bars, funding, interval=interval, leverage=1.0,
                start_date=START, end_date=END, **cfg)
        if m and m.get('CAGR', 0) != 0:
            results.append({**cfg, **m})
        if (i+1) % 100 == 0:
            print(f"  {interval}: {i+1}/{len(grid)} ({time.time()-t0:.0f}s)")

    elapsed = time.time() - t0
    print(f"\n{interval}: {len(results)} 유효 / {len(grid)} ({elapsed:.0f}s)")
    print(f"\n── {interval} Top 20 by Calmar (1x) ──")
    print_top(results, 20, 'Cal')

    # Top 3 leverage
    top3 = sorted(results, key=lambda x: x.get('Cal', 0), reverse=True)[:3]
    levs = [1.0, 1.5, 2.0, 3.0] if interval != '1h' else [1.0, 2.0]
    print(f"\n── {interval} Top 3 × leverage ──")
    for i, cfg in enumerate(top3):
        cp = {k: cfg[k] for k in ['sma_days','mom_short_days','mom_long_days',
                                    'canary_hyst','drift_threshold','dd_threshold',
                                    'dd_lookback','bl_drop','daily_gate']
              if k in cfg}
        dd = 'off' if cfg['dd_threshold']==0 else f"{cfg['dd_threshold']:.0%}"
        bl = 'off' if cfg.get('bl_drop', -0.15)==0 else f"{cfg.get('bl_drop', -0.15):.0%}"
        gate = 'Y' if cfg['daily_gate'] else 'N'
        print(f"\n  #{i+1}: SMA{cfg['sma_days']}d Mom{cfg['mom_short_days']}/{cfg['mom_long_days']}d"
              f" Hyst{cfg['canary_hyst']:.1%} DD={dd} BL={bl} Gate={gate}")
        for lev in levs:
            m = run(bars, funding, interval=interval, leverage=lev,
                    start_date=START, end_date=END, **cp)
            if not m: continue
            liq = f' Liq{m["Liq"]}' if m['Liq'] > 0 else ''
            print(f"    {lev}x: Sh={m['Sharpe']:.2f} CAGR={m['CAGR']:+.1%}"
                  f" MDD={m['MDD']:+.1%} Cal={m['Cal']:.2f} Rb={m['Rebal']}{liq}")

    return results


if __name__ == '__main__':
    t_total = time.time()
    grid = build_grid_reduced()
    print(f"그리드: {len(grid)} 조합\n")

    # D만 단독 실행 (빠름)
    if len(sys.argv) > 1:
        interval = sys.argv[1]
        run_interval(interval, grid)
    else:
        for interval in ['D', '4h', '1h']:
            print(f"\n{'='*80}")
            run_interval(interval, grid)

    print(f"\n총 소요: {time.time()-t_total:.0f}s")
