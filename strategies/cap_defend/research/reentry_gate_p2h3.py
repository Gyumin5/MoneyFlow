#!/usr/bin/env python3
"""P2/H3 채택 게이트 (탐색 아님, 확인용):
  1) 서브기간 일관성 — F0/P2/H3 를 여러 시작점(레짐)에서 돌려 P2/H3 가 F0 대비 안 나빠지나.
  2) H vol_cap plateau — vthr 5/7/10/12% 단조·평탄성 (뾰족한 최적점이면 기각 신호).
  3) P2 sizing 이웃 — full(A2) / half(P1) / prop(P2) 를 같은 기간에서 대조.
데이터 2019-09~2026-05 (2018 시작 불가). 실거래 로직 무수정, read-only.
"""
import os, sys, json, time
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import reentry_harness as H
import unified_backtest as ub

END = '2026-05-31'
# 데이터 시작 2019-09-08 → warmup 후 첫 시작 2019-11. 레짐별 시작점.
SUBSTARTS = ['2019-11-01', '2021-01-01', '2022-01-01', '2023-01-01', '2024-01-01']

MET = ['CAGR', 'MDD', 'Cal', 'Sharpe', 'Trades', 'Rebal']


def cfg_H(vthr):
    return {'cadence': 0, 'K': 1, 'min_healthy': 1, 'sizing': 'full',
            'health': {'hmode': 'mom2vol', 'vthr': vthr, 'ms_bars': 0}}


def run(bars, funding, cfg, start):
    res, _, _ = H.run_variant(bars, funding, cfg, start, END,
                              want_stats=False, want_trace=False)
    return {k: res[k] for k in MET}


def main():
    t0 = time.time()
    print('데이터 로드...', flush=True)
    bars, funding = ub.load_data('D')

    F0 = H.VARIANTS['F0'][0]
    P2 = H.VARIANTS['P2'][0]
    H3 = H.VARIANTS['H3'][0]
    A2 = H.VARIANTS['A2'][0]
    P1 = H.VARIANTS['P1'][0]

    out = {'meta': {'end': END, 'substarts': SUBSTARTS}, 'subperiod': {}, 'h_plateau': {}, 'p2_sizing': {}}

    # 1) 서브기간 일관성
    print('== 1) 서브기간 일관성 (F0/P2/H3) ==', flush=True)
    for start in SUBSTARTS:
        row = {}
        for name, cfg in [('F0', F0), ('P2', P2), ('H3', H3)]:
            m = run(bars, funding, cfg, start)
            row[name] = m
            print(f'  {start} {name}: Cal={m["Cal"]:.3f} CAGR={m["CAGR"]:+.3f} '
                  f'MDD={m["MDD"]:+.3f} Sh={m["Sharpe"]:.2f}', flush=True)
        # P2/H3 가 F0 대비 Cal 열위인지 표시
        for v in ('P2', 'H3'):
            row[f'{v}_vs_F0_Cal'] = row[v]['Cal'] - row['F0']['Cal']
        out['subperiod'][start] = row

    # 2) H vol_cap plateau (전체기간, 시작 2019-11)
    print('== 2) H vol_cap plateau (vthr 5/7/10/12%) ==', flush=True)
    for vthr in [0.05, 0.07, 0.10, 0.12]:
        m = run(bars, funding, cfg_H(vthr), SUBSTARTS[0])
        out['h_plateau'][f'{vthr:.2f}'] = m
        print(f'  vthr={vthr:.2f}: Cal={m["Cal"]:.3f} CAGR={m["CAGR"]:+.3f} '
              f'MDD={m["MDD"]:+.3f} Trades={m["Trades"]}', flush=True)
    # F0 baseline 참조
    mf = run(bars, funding, F0, SUBSTARTS[0])
    out['h_plateau']['F0_ref'] = mf
    print(f'  F0_ref : Cal={mf["Cal"]:.3f} CAGR={mf["CAGR"]:+.3f} MDD={mf["MDD"]:+.3f}', flush=True)

    # 3) P2 sizing 이웃 (full=A2 / half=P1 / prop=P2), 전체기간
    print('== 3) P2 sizing 이웃 (A2 full / P1 half / P2 prop) ==', flush=True)
    for name, cfg in [('A2_full', A2), ('P1_half', P1), ('P2_prop', P2), ('F0_ref', F0)]:
        m = run(bars, funding, cfg, SUBSTARTS[0])
        out['p2_sizing'][name] = m
        print(f'  {name}: Cal={m["Cal"]:.3f} CAGR={m["CAGR"]:+.3f} MDD={m["MDD"]:+.3f} Trades={m["Trades"]}', flush=True)

    out['meta']['elapsed'] = time.time() - t0
    p = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'reentry_gate_p2h3.json')
    with open(p, 'w') as f:
        json.dump(out, f, indent=2, default=str)
    print(f'\n완료 {time.time()-t0:.0f}s → reentry_gate_p2h3.json', flush=True)


if __name__ == '__main__':
    main()
