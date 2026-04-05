#!/usr/bin/env python3
"""확장된 최적화 v2 — 38코인, 봉 단위, 짧은 윈도우 포함, 다양한 헬스 모드."""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_futures_full import load_data, run

START = '2020-10-01'
END = '2026-03-28'


def build_grid_v2():
    """확장 그리드: 짧은 윈도우 + 단일/이중 모멘텀 + 다양한 헬스."""
    grid = []
    for sma in [5, 10, 20, 30, 40, 50, 60]:
        for mom_s in [5, 10, 15, 21, 30, 45]:
            for mom_l_and_mode in [
                (0, 'mom1vol'),    # 단일 모멘텀 + vol
                (0, 'mom1'),       # 단일 모멘텀 only
                (60, 'mom2vol'),   # 이중 모멘텀 + vol
                (90, 'mom2vol'),   # 이중 모멘텀 + vol (long)
            ]:
                mom_l, hmode = mom_l_and_mode
                if mom_l > 0 and mom_l <= mom_s:
                    continue
                for hyst in [0.0, 0.015]:
                    for dd_th in [0, -0.25]:
                        for bl in [0, -0.15]:
                            grid.append(dict(
                                sma_days=sma,
                                mom_short_days=mom_s,
                                mom_long_days=mom_l if mom_l > 0 else 90,
                                canary_hyst=hyst,
                                drift_threshold=0.0,
                                dd_threshold=dd_th,
                                dd_lookback=60 if dd_th != 0 else 0,
                                bl_drop=bl,
                                daily_gate=False,
                                health_mode=hmode,
                            ))
    return grid


def print_top_v2(results, n=20):
    """상위 N개 결과 출력."""
    results.sort(key=lambda x: x.get('Cal', 0), reverse=True)
    print(f"  {'SMA':>3s} {'Mom':>7s} {'Health':>7s} {'Hyst':>5s} {'DD':>5s} {'BL':>5s}"
          f" | {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Rb':>5s}")
    print(f"  {'-'*75}")
    for r in results[:n]:
        mom = f"{r['mom_short_days']}"
        if r['health_mode'] in ('mom2vol', 'mom2'):
            mom += f"/{r['mom_long_days']}"
        hmode = r['health_mode']
        hyst = f"{r['canary_hyst']:.1%}" if r['canary_hyst'] > 0 else "off"
        dd = f"{r['dd_threshold']:.0%}" if r['dd_threshold'] != 0 else "off"
        bl = f"{r.get('bl_drop', -0.15):.0%}" if r.get('bl_drop', -0.15) != 0 else "off"
        print(f"  {r['sma_days']:>3d} {mom:>7s} {hmode:>7s} {hyst:>5s} {dd:>5s} {bl:>5s}"
              f" | {r['Sharpe']:>5.2f} {r['CAGR']:>+8.1%} {r['MDD']:>+8.1%} {r['Cal']:>6.2f} {r['Rebal']:>5d}")


if __name__ == '__main__':
    t0 = time.time()
    grid = build_grid_v2()
    print(f"v2 그리드: {len(grid)} 조합 (38코인, 짧은 윈도우 포함)\n")

    intervals = sys.argv[1:] if len(sys.argv) > 1 else ['D', '4h']

    for interval in intervals:
        t1 = time.time()
        bars, funding = load_data(interval)
        print(f"\n{'='*85}")
        print(f"  {interval} | {len(bars)} coins | {len(grid)} configs")
        print(f"{'='*85}")

        results = []
        for i, cfg in enumerate(grid):
            m = run(bars, funding, interval=interval, leverage=1.0,
                    start_date=START, end_date=END, **cfg)
            if m and m.get('CAGR', 0) != 0:
                results.append({**cfg, **m})
            if (i+1) % 100 == 0:
                print(f"  {interval}: {i+1}/{len(grid)} ({time.time()-t1:.0f}s)")

        print(f"\n  {interval}: {len(results)} 유효 / {len(grid)} ({time.time()-t1:.0f}s)")
        print(f"\n  ── {interval} Top 20 by Calmar (1x) ──")
        print_top_v2(results, 20)

        # Top 3 with leverage
        top3 = sorted(results, key=lambda x: x.get('Cal', 0), reverse=True)[:3]
        levs = [1.0, 1.5, 2.0, 3.0]
        print(f"\n  ── {interval} Top 3 × leverage ──")
        for i, cfg in enumerate(top3):
            cp = {k: cfg[k] for k in cfg if k not in ('Sharpe','CAGR','MDD','Cal','Liq','Trades','Rebal')}
            hm = cfg['health_mode']
            mom_str = f"{cfg['mom_short_days']}"
            if hm in ('mom2vol', 'mom2'):
                mom_str += f"/{cfg['mom_long_days']}"
            dd = 'off' if cfg['dd_threshold']==0 else f"{cfg['dd_threshold']:.0%}"
            bl = 'off' if cfg.get('bl_drop',0)==0 else f"{cfg.get('bl_drop'):.0%}"
            hyst_str = 'off' if cfg['canary_hyst']==0 else f"{cfg['canary_hyst']:.1%}"
            print(f"\n  #{i+1}: SMA{cfg['sma_days']}d Mom{mom_str} {hm} Hyst={hyst_str} DD={dd} BL={bl}")
            for lev in levs:
                m = run(bars, funding, interval=interval, leverage=lev,
                        start_date=START, end_date=END, **cp)
                if not m: continue
                liq = f' Liq{m["Liq"]}' if m.get('Liq',0) > 0 else ''
                print(f"    {lev}x: Sh={m['Sharpe']:.2f} CAGR={m['CAGR']:+.1%}"
                      f" MDD={m['MDD']:+.1%} Cal={m['Cal']:.2f} Rb={m['Rebal']}{liq}")

        sys.stdout.flush()

    print(f"\n총 소요: {time.time()-t0:.0f}s")
