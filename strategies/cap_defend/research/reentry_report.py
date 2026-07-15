#!/usr/bin/env python3
"""reentry_results.json → 비교표 렌더링 (비용 반영 후 기준)."""
import json, os, sys

HERE = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(HERE, 'reentry_results.json')) as f:
    D = json.load(f)
R = D['results']
meta = D['meta']
ORDER = ['F0', 'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'D1', 'C1',
         'S1', 'S2', 'S3', 'L1', 'P1', 'P2', 'R1', 'W1', 'H1', 'H2', 'H3', 'H4']
f0 = R['F0']


def g(name, k, d=float('nan')):
    v = R[name].get(k, d)
    try:
        return float(v)
    except Exception:
        return d


print(f"기간 {meta['start']}~{meta['end']} | windows={meta['nwin']} "
      f"(sizes {meta['win_sizes']} × strides {meta['strides']}) | "
      f"phases={len(meta['phases'])} | elapsed={float(meta['elapsed']):.0f}s")
print()
print("== 표1. 22개 변형 핵심지표 (비용 반영 후) ==")
hdr = (f"{'Var':<4}{'CAGR':>8}{'MDD':>8}{'Cal':>7}{'Sharpe':>8}{'Turnov':>8}"
       f"{'Trds':>6}{'CashHf':>7}{'ReDly50':>8}{'RankSum':>9}{'Win%':>6}"
       f"{'aCalMean':>9}{'aCalStd':>8}{'n_ev':>5}")
print(hdr)
print('-' * len(hdr))
for name in ORDER:
    r = R[name]
    print(f"{name:<4}{g(name,'CAGR'):>+8.3f}{g(name,'MDD'):>+8.3f}{g(name,'Cal'):>7.3f}"
          f"{g(name,'Sharpe'):>8.2f}{g(name,'turnover'):>8.1f}{int(g(name,'Trades')):>6d}"
          f"{int(g(name,'cash_half')):>7d}{g(name,'delay_p50'):>8.1f}"
          f"{g(name,'ranksum'):>9.0f}{100*g(name,'winrate'):>6.1f}"
          f"{g(name,'anchor_cal_mean'):>9.3f}{g(name,'anchor_cal_std'):>8.3f}"
          f"{int(g(name,'n_events')):>5d}")

print()
print("== 표2. F0 대비 델타 (비용반영 후) ==")
hdr2 = (f"{'Var':<4}{'dCAGR':>8}{'dMDD':>8}{'dCal':>8}{'dSharpe':>8}"
        f"{'dTurn':>8}{'dRankSum':>9}{'CAGR_noC':>9}")
print(hdr2); print('-' * len(hdr2))
for name in ORDER:
    if name == 'F0':
        continue
    print(f"{name:<4}{g(name,'CAGR')-g('F0','CAGR'):>+8.3f}"
          f"{g(name,'MDD')-g('F0','MDD'):>+8.3f}"
          f"{g(name,'Cal')-g('F0','Cal'):>+8.3f}"
          f"{g(name,'Sharpe')-g('F0','Sharpe'):>+8.2f}"
          f"{g(name,'turnover')-g('F0','turnover'):>+8.1f}"
          f"{g(name,'ranksum')-g('F0','ranksum'):>+9.0f}"
          f"{g(name,'CAGR_nocost'):>+9.3f}")

print()
print("== 표3. 재진입 telemetry (n<20 = 탐색용) ==")
hdr3 = (f"{'Var':<4}{'n_ev':>5}{'ReDly50':>8}{'ReDly90':>8}{'Conc.m':>7}"
        f"{'fwd21med':>9}{'fwd21p10':>9}{'canoff5':>8}{'canoff21':>9}"
        f"{'emptyd90':>9}{'swapDay':>8}")
print(hdr3); print('-' * len(hdr3))
for name in ORDER:
    r = R[name]
    ne = int(g(name, 'n_events'))
    print(f"{name:<4}{ne:>5d}{g(name,'delay_p50'):>8.1f}{g(name,'delay_p90'):>8.1f}"
          f"{g(name,'concurrent_mean'):>7.2f}{g(name,'fwd21_med'):>+9.3f}"
          f"{g(name,'fwd21_p10'):>+9.3f}{g(name,'canoff5'):>8.2f}"
          f"{g(name,'canoff21'):>9.2f}{g(name,'emptydur_p90'):>9.1f}"
          f"{int(g(name,'swap_days')):>8d}")

print()
print("== 그룹별 tradeoff (F0 기준 rank / Cal / dCAGR) ==")
GROUPS = {
    'G0': ['F0'], 'G1': ['A1','A2','A3','A4','A5','A6','D1','C1'],
    'G2': ['F0','A2','A7'], 'G3': ['A2','S1','S2','S3'],
    'G4': ['A1','A2','A3','L1'], 'G5': ['A2','P1','P2'],
    'G6': ['A2','R1','W1'], 'G7': ['H1','H2','H3','H4'],
}
for gk, members in GROUPS.items():
    print(f"[{gk}] " + " ".join(members))
    for m in members:
        print(f"   {m:<4} Cal={g(m,'Cal'):.3f} RankSum={g(m,'ranksum'):.0f} "
              f"Win%={100*g(m,'winrate'):.1f} dCAGR={g(m,'CAGR')-g('F0','CAGR'):+.3f} "
              f"dMDD={g(m,'MDD')-g('F0','MDD'):+.3f} aCalStd={g(m,'anchor_cal_std'):.3f}")
