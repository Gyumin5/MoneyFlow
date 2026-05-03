"""V23 shadow ledger — 오늘 시점 V23 가상 운영 보유 산출.

가정: V23 이 처음부터 줄곧 운영되었다고 가정하고 BT engine 으로 forward 시뮬레이션.
산출: 오늘 시점 combined target weights.

단발 실행. 매일 자동화는 별도.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                 'strategies', 'cap_defend'))
from datetime import datetime
import pandas as pd

import unified_backtest as ub
import stock_engine as ts
import stock_engine_snap as tss

START = '2020-10-01'
END = datetime.now().strftime('%Y-%m-%d')
print(f'== V23 shadow ledger (오늘 시점 가상 보유) ==')
print(f'기간: {START} ~ {END}')

# ─── 데이터 로드 ───
print('\n코인 데이터 (D) 로드...')
bars_D, funding = ub.load_data('D')

# ─── spot V23 ───
print('\n[spot] V23: D_SMA42 sn=217 n=7 drift=0.10')
spot_trace = []
ub.run(bars_D, funding, interval='D', asset_type='spot', leverage=1.0,
       sma_days=42, mom_short_days=20, mom_long_days=127,
       vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=7,
       universe_size=3, cap=1/3, tx_cost=0.004,
       health_mode='mom2vol', vol_mode='daily', drift_threshold=0.10,
       snap_interval_bars=217, start_date=START, end_date=END,
       _trace=spot_trace)
print(f'  bars={len(spot_trace)}')
if spot_trace:
    last = spot_trace[-1]
    print(f'  마지막 봉: {last["date"]}')
    print(f'  combined target: {dict(sorted(last["target"].items(), key=lambda x: -x[1]))}')

# ─── fut V23 ───
print('\n[fut] V23: D_SMA42 sn=57 n=3 drift=0.05 L3')
fut_trace = []
ub.run(bars_D, funding, interval='D', asset_type='fut', leverage=3.0,
       sma_days=42, mom_short_days=18, mom_long_days=127,
       vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=3,
       universe_size=3, cap=1/3, tx_cost=0.0004, maint_rate=0.004,
       health_mode='mom2vol', vol_mode='daily', drift_threshold=0.05,
       snap_interval_bars=57, start_date=START, end_date=END,
       _trace=fut_trace)
print(f'  bars={len(fut_trace)}')
if fut_trace:
    last = fut_trace[-1]
    print(f'  마지막 봉: {last["date"]}')
    print(f'  combined target: {dict(sorted(last["target"].items(), key=lambda x: -x[1]))}')

# ─── stock V23 ───
print('\n[stock] V23: SNAP=69 STAGGER=23 N=3')
OFF = ('SPY', 'QQQ', 'VEA', 'EEM', 'EWJ', 'GLD', 'PDBC')
DEF = ('IEF', 'BIL', 'BNDX', 'GLD', 'PDBC')
CAN = ('EEM',)
ts._g_prices = ts.load_prices(list(set(OFF + DEF + CAN)), start='2014-01-01')
ts._g_ind = ts.precompute(ts._g_prices)
sp = ts.SP(offensive=OFF, defensive=DEF, canary_assets=CAN,
           canary_sma=300, canary_hyst=0.020, canary_type='sma',
           health='none', defense='top2', defense_sma=100, def_mom_period=126,
           select='zscore3', n_mom=3, n_sh=3, sharpe_lookback=126,
           weight='ew', crash='none',
           tx_cost=0.001, start=START, end=END, capital=10000.0)
stock_trace = []
tss.run_snapshot_ensemble(ts._g_prices, ts._g_ind, sp,
                           snap_days=69, n_snap=3, _trace=stock_trace)
print(f'  bars={len(stock_trace)}')
if stock_trace:
    last = stock_trace[-1]
    print(f'  마지막 봉: {last["Date"]}')
    print(f'  combined target: {dict(sorted(last["target"].items(), key=lambda x: -x[1]))}')

print('\n== 완료 ==')
