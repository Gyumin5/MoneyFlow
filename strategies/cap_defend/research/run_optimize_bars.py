#!/usr/bin/env python3
"""봉 단위 최적화 — 간격별 고유 전략 탐색.

모든 윈도우를 봉 개수로 지정. 1D의 50봉 ≈ 1h의 50봉 (2일).
각 간격에 맞는 전략을 독립적으로 찾음.
"""
import sys, os, time
sys.stdout.reconfigure(line_buffering=True)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from backtest_futures_full import load_data, run

START = '2020-10-01'
END = '2026-03-28'


def build_bar_grid():
    """봉 단위 그리드. 모든 간격에 동일 적용."""
    grid = []
    for sma_b in [10, 20, 50, 100, 200, 500]:
        for mom_s_b in [5, 10, 20, 50, 100, 200]:
            if mom_s_b >= sma_b * 2:  # 모멘텀이 SMA의 2배 이상이면 skip
                continue
            for mom_l_and_mode in [
                (0, 'mom1vol'),     # 단일 모멘텀 + vol
                (0, 'mom1'),        # 단일 모멘텀 only
                (None, 'mom2vol'),  # 이중 모멘텀(long=short*3) + vol
            ]:
                mom_l_raw, hmode = mom_l_and_mode
                # 이중 모멘텀: long = short * 3 (자동 계산)
                mom_l_b = mom_s_b * 3 if mom_l_raw is None else mom_l_raw

                for hyst in [0.0, 0.015]:
                    for dd_b in [0, sma_b * 3]:  # DD lookback = SMA의 3배 또는 off
                        for bl_b in [0, sma_b]:    # BL cooldown = SMA와 동일 또는 off
                            for n_snap in [3, 6]:
                                grid.append(dict(
                                    sma_bars=sma_b,
                                    mom_short_bars=mom_s_b,
                                    mom_long_bars=mom_l_b,
                                    canary_hyst=hyst,
                                    drift_threshold=0.0,
                                    dd_threshold=-0.25 if dd_b > 0 else 0,
                                    dd_bars_override=dd_b,
                                    dd_lookback=0,
                                    bl_drop=-0.15 if bl_b > 0 else 0,
                                    bl_bars_override=bl_b,
                                    bl_days=0,
                                    daily_gate=False,
                                    health_mode=hmode,
                                    n_snapshots=n_snap,
                                ))
    return grid


def print_top_bars(results, n=20):
    """상위 N개 결과 출력."""
    results.sort(key=lambda x: x.get('Cal', 0), reverse=True)
    print(f"  {'SMA':>4s} {'Mom':>8s} {'Health':>7s} {'Hy':>3s} {'DD':>4s} {'BL':>4s} {'Sn':>2s}"
          f" | {'Sh':>5s} {'CAGR':>8s} {'MDD':>8s} {'Cal':>6s} {'Rb':>5s}")
    print(f"  {'-'*75}")
    for r in results[:n]:
        sma = r['sma_bars']
        mom_s = r['mom_short_bars']
        hm = r['health_mode']
        if 'mom2' in hm:
            mom = f"{mom_s}/{r['mom_long_bars']}"
        else:
            mom = f"{mom_s}"
        hy = f"{r['canary_hyst']:.0%}" if r['canary_hyst'] > 0 else "-"
        dd = f"{r['dd_bars_override']}" if r['dd_bars_override'] > 0 else "-"
        bl = f"{r['bl_bars_override']}" if r['bl_bars_override'] > 0 else "-"
        ns = r['n_snapshots']
        print(f"  {sma:>4d} {mom:>8s} {hm:>7s} {hy:>3s} {dd:>4s} {bl:>4s} {ns:>2d}"
              f" | {r['Sharpe']:>5.2f} {r['CAGR']:>+8.1%} {r['MDD']:>+8.1%} {r['Cal']:>6.2f} {r['Rebal']:>5d}")


if __name__ == '__main__':
    t0 = time.time()
    grid = build_bar_grid()
    print(f"봉 단위 그리드: {len(grid)} 조합\n")

    intervals = sys.argv[1:] if len(sys.argv) > 1 else ['D', '4h', '1h']

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
        print_top_bars(results, 20)

        # Top 3 with leverage
        top3 = sorted(results, key=lambda x: x.get('Cal', 0), reverse=True)[:3]
        print(f"\n  ── {interval} Top 3 × leverage ──")
        for i, cfg in enumerate(top3):
            cp = {k: cfg[k] for k in cfg if k not in ('Sharpe','CAGR','MDD','Cal','Liq','Trades','Rebal')}
            hm = cfg['health_mode']
            ms = cfg['mom_short_bars']
            mom_str = f"{ms}/{cfg['mom_long_bars']}" if 'mom2' in hm else str(ms)
            dd_str = str(cfg['dd_bars_override']) if cfg['dd_bars_override']>0 else 'off'
            bl_str = str(cfg['bl_bars_override']) if cfg['bl_bars_override']>0 else 'off'
            hy_str = 'off' if cfg['canary_hyst']==0 else f"{cfg['canary_hyst']:.1%}"
            print(f"\n  #{i+1}: SMA={cfg['sma_bars']}b Mom={mom_str}b {hm}"
                  f" Hy={hy_str} DD={dd_str}b BL={bl_str}b Snap={cfg['n_snapshots']}")
            for lev in [1.0, 1.5, 2.0, 3.0]:
                m = run(bars, funding, interval=interval, leverage=lev,
                        start_date=START, end_date=END, **cp)
                if not m: continue
                liq = f' Liq{m["Liq"]}' if m.get('Liq',0) > 0 else ''
                print(f"    {lev}x: Sh={m['Sharpe']:.2f} CAGR={m['CAGR']:+.1%}"
                      f" MDD={m['MDD']:+.1%} Cal={m['Cal']:.2f} Rb={m['Rebal']}{liq}")

        sys.stdout.flush()

    print(f"\n총 소요: {time.time()-t0:.0f}s")
