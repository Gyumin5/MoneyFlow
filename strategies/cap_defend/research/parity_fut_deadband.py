"""카나리 경계(죽은 띠) 구간에서도 라이브와 백테스트가 같은 답을 내는지.

최근 3일만 보면 비트코인이 SMA 에서 멀어 카나리가 확실히 켜진 상태다. 히스테리시스는
경계에서만 의미가 있으므로 그 구간을 따로 골라 본다. 이력 전체를 한 번 비교하고,
그 중 |close/SMA42 - 1| < 1.5% 인 신호일만 따로 센다.
"""
import os
import sys

sys.path.insert(0, os.path.join('/home/gmoh/mon/251229', 'trade', 'ops'))
import signal_parity_daily as sp   # noqa: E402
import unified_backtest as ub      # noqa: E402

bars, funding = ub.load_data('D')
universe = [c for c in sp.REF_UNIVERSE if c in bars]
end_date = str(bars['BTC'].index[-1])[:10]

bt = sp._run_bt(bars, funding, universe, end_date)
live = sp._run_live(bars, universe, list(bt.keys()))

common = sorted(d for d in bt if d in live)
mismatch = [d for d in common if bt[d] != live[d]]

btc = bars['BTC']['Close']
sma = btc.rolling(42).mean()
ratio = (btc / sma) - 1.0
hyst = sp.BT_FUT_PARAMS['canary_hyst']

band = [d for d in common if d in ratio.index and abs(ratio.loc[d]) < hyst]
band_mismatch = [d for d in band if d in mismatch]

print(f'비교한 신호일 {len(common)}일 ({str(common[0])[:10]} ~ {str(common[-1])[:10]})')
print(f'불일치 {len(mismatch)}일')
print(f'그 중 카나리 죽은 띠(|종가/SMA42 - 1| < {hyst:.1%}) 안의 신호일 {len(band)}일, 불일치 {len(band_mismatch)}일')
if band:
    print(f'  죽은 띠 처음 {str(band[0])[:10]} · 마지막 {str(band[-1])[:10]}')
    edge = band[len(band) // 2]
    print(f'  가운데 표본 {str(edge)[:10]}  비율 {ratio.loc[edge]:+.4%}  목표 {live[edge]}')
for d in mismatch[:10]:
    print(f'  다름 {str(d)[:10]}  백테스트 {bt[d]}  라이브 {live[d]}')

# 카나리가 꺼졌다 켜지는 전이일도 따로 본다 — 이어받기가 끊기면 여기서 먼저 갈린다.
flips = []
prev_cash = None
for d in common:
    is_cash = set(live[d].keys()) == {'CASH'}
    if prev_cash is not None and is_cash != prev_cash:
        flips.append(d)
    prev_cash = is_cash
flip_mismatch = [d for d in flips if d in mismatch]
print(f'전액현금 ↔ 보유 전이 {len(flips)}회, 그 중 불일치 {len(flip_mismatch)}회')

sys.exit(1 if mismatch else 0)
