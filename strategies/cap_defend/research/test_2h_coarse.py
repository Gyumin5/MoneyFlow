#!/usr/bin/env python3
"""2h봉 Coarse Grid — 캐스케이드 STEP 3.

독립적 넓은 파라미터 공간 탐색. 현물 + 선물(5x).
짧은 구간(극단기)부터 D수준 장기까지 전 범위.
멀티프로세싱 + 체크포인트 지원.
"""
import os, sys, time, json, gc
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from backtest_spot_barfreq import load_binance_prices, run_backtest
from coin_engine import Params

# ─── 고정 조건 ──────────────────────────────────────────────
N_PICKS = 3
TOP_N = 40
N_SNAP = 3
SELECTION = 'baseline'
WEIGHTING = 'baseline'

# ─── 2h봉 Coarse Grid (바 단위, 12봉/일) ───────────────────
# 로그스케일 진행, 봉 단위 기준
SMA_LIST = [24, 60, 120, 240, 360, 480, 720, 960, 1440, 2400]  # 2일~200일
MS_LIST = [1, 3, 6, 12, 20, 24, 60, 120, 240, 480, 960]         # 2h~80일
ML_LIST = [36, 120, 240, 480, 720, 1440, 2160, 2880, 4320]      # 3일~360일
SNAP_LIST = [36, 84, 168, 336, 720, 1080, 2160]                 # 3일~180일
VC_LIST = [0.05, 0.07]

MODES = [
    ('spot', 1.0, 0.004, 'Spot'),
]

BPD = 12
PERIOD = ('2020-10-01', '2026-03-31')

_g_prices = {}
_g_um = {}
_g_funding = {}


def _init_worker(prices, um, funding):
    global _g_prices, _g_um, _g_funding
    _g_prices = prices
    _g_um = um
    _g_funding = funding


def build_combos():
    combos = []
    for sma in SMA_LIST:
        for ms in MS_LIST:
            for ml in ML_LIST:
                if ml <= ms:
                    continue
                for snap in SNAP_LIST:
                    for vc in VC_LIST:
                        combos.append((sma, ms, ml, snap, vc))
    return combos


def _eval_single(args):
    sma, ms, ml, snap, vc, mode, leverage, tx_cost, start, end = args
    fund = _g_funding if mode == 'futures' else None
    p = Params(
        canary='K8', vote_threshold=1,
        sma_period=sma, canary_band=1.5, vote_smas=(sma,),
        health='HK', health_sma=0,
        health_mom_short=ms, health_mom_long=ml,
        health_vol_window=90, vol_cap=vc,
        n_picks=N_PICKS, top_n=TOP_N,
        selection=SELECTION, weighting=WEIGHTING,
        start_date=start, end_date=end,
    )
    r = run_backtest(_g_prices, _g_um, p,
                     bars_per_day=BPD, snap_interval_bars=max(snap, N_SNAP),
                     mode=mode, leverage=leverage, tx_cost=tx_cost,
                     n_snap=N_SNAP, funding_data=fund)
    m = r['metrics']
    cal = m['CAGR'] / abs(m['MDD']) if m['MDD'] != 0 else 0
    return {
        'params': {'SMA': sma, 'Ms': ms, 'Ml': ml, 'Snap': snap, 'VC': vc},
        'Sharpe': m['Sharpe'], 'Cal': cal,
        'CAGR': m['CAGR'], 'MDD': m['MDD'],
        'Rebal': r['rebal_count'],
    }


def marginal_analysis(results, param_name):
    groups = {}
    for r in results:
        key = r['params'][param_name]
        groups.setdefault(key, []).append(r['Sharpe'])
    print(f"\n  Marginal {param_name}:")
    for k in sorted(groups.keys()):
        vals = groups[k]
        print(f"    {param_name}={k:>6}: n={len(vals):>4}  "
              f"Sh mean={np.mean(vals):.3f}  med={np.median(vals):.3f}  "
              f"std={np.std(vals):.3f}  max={np.max(vals):.3f}")


def print_top20(results, label):
    n = len(results)
    sh_rank = sorted(range(n), key=lambda j: -results[j]['Sharpe'])
    cal_rank = sorted(range(n), key=lambda j: -results[j]['Cal'])
    cagr_rank = sorted(range(n), key=lambda j: -results[j]['CAGR'])

    ranks = {}
    for rank, j in enumerate(sh_rank): ranks.setdefault(j, []).append(rank+1)
    for rank, j in enumerate(cal_rank): ranks[j].append(rank+1)
    for rank, j in enumerate(cagr_rank): ranks[j].append(rank+1)

    ranked = sorted(range(n), key=lambda j: sum(ranks[j]))

    print(f"\n{'='*70}")
    print(f"  {label} Top 20 (순위합)")
    print(f"{'='*70}")
    print(f"{'Rk':>3} {'SMA':>5} {'Ms':>4} {'Ml':>5} {'Sn':>4} {'VC':>5} | "
          f"{'Sh':>6} {'Cal':>5} {'CAGR':>7} {'MDD':>7} {'RkS':>5}")
    for ri, j in enumerate(ranked[:20], 1):
        r = results[j]
        p = r['params']
        rsum = sum(ranks[j])
        print(f"{ri:>3} {p['SMA']:>5} {p['Ms']:>4} {p['Ml']:>5} {p['Snap']:>4} {p['VC']:>5.3f} | "
              f"{r['Sharpe']:>6.3f} {r['Cal']:>5.2f} "
              f"{r['CAGR']:>+7.1%} {r['MDD']:>+7.1%} {rsum:>5}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=24)
    args = parser.parse_args()

    t0 = time.time()
    combos = build_combos()
    print(f"2h Coarse Grid: {len(combos)} 조합 x {len(MODES)} 모드 ({args.workers} workers)")

    print("데이터 로딩...")
    prices, um, funding = load_binance_prices('2h', top_n=TOP_N)
    print(f"  2h: {len(prices)} coins, 완료 ({time.time()-t0:.1f}s)\n")

    for mode, leverage, tx_cost, label in MODES:
        start, end = PERIOD
        json_path = f'research/grid_results/2h_coarse_{label.lower()}.json'

        done_keys = set()
        all_results = []
        if os.path.exists(json_path):
            with open(json_path) as f:
                all_results = json.load(f)
            done_keys = {(r['params']['SMA'], r['params']['Ms'], r['params']['Ml'],
                          r['params']['Snap'], r['params']['VC']) for r in all_results}
            print(f"  체크포인트 복원: {len(all_results)}/{len(combos)}")

        remaining = [(sma, ms, ml, snap, vc, mode, leverage, tx_cost, start, end)
                     for sma, ms, ml, snap, vc in combos
                     if (sma, ms, ml, snap, vc) not in done_keys]

        if not remaining:
            print(f"  {label}: 이미 완료됨")
        else:
            print(f"\n{'='*90}")
            print(f"  {label} ({start}~{end}) — {len(remaining)} remaining")
            print(f"{'='*90}")

            t1 = time.time()
            with Pool(args.workers, initializer=_init_worker,
                      initargs=(prices, um, funding),
                      maxtasksperchild=50) as pool:
                for i, r in enumerate(pool.imap_unordered(_eval_single, remaining), 1):
                    all_results.append(r)
                    if i % 100 == 0 or i == len(remaining):
                        elapsed = time.time() - t1
                        eta = elapsed / i * (len(remaining) - i)
                        p = r['params']
                        print(f"  [{i}/{len(remaining)}] "
                              f"SMA{p['SMA']} Ms{p['Ms']} Ml{p['Ml']} "
                              f"Sn{p['Snap']} VC{p['VC']:.3f} → "
                              f"Sh={r['Sharpe']:.3f} Cal={r['Cal']:.2f} "
                              f"({elapsed:.0f}s, ETA {eta:.0f}s)",
                              flush=True)
                        tmp = json_path + '.tmp'
                        with open(tmp, 'w') as f:
                            json.dump(all_results, f, indent=2)
                        os.replace(tmp, json_path)
                        gc.collect()

            with open(json_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            print(f"\n  저장: {json_path} ({time.time()-t1:.1f}s)")

        print_top20(all_results, label)

        print(f"\n{'='*70}")
        print(f"  {label} Marginal 분포")
        print(f"{'='*70}")
        for pname in ['SMA', 'Ms', 'Ml', 'Snap', 'VC']:
            marginal_analysis(all_results, pname)

    print(f"\n총 소요: {time.time()-t0:.1f}s")


if __name__ == '__main__':
    main()
