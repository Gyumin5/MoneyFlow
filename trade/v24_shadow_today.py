"""현물 V24 shadow ledger — 오늘 시점 V24 가상 보유 산출.

가정: V24 현물이 처음부터 줄곧 운영되었다고 가정하고 BT 엔진으로 forward 시뮬레이션.
산출: 오늘 시점 combined target weights.

단발 실행이다. 어느 crontab 에도 없다 — 매일 도는 신호 감시는 이 파일이 아니라
`trade/ops/signal_parity_daily.py`(선물·주식)가 한다.

현물만 다룬다 (2026-09-30).
- 선물은 V25 라 이 파일이 부르는 `unified_backtest.run` 으로 표현할 수 없다. 그 엔진은
  스칼라 leverage 하나와 ISOLATED 청산 모델만 받는데, 라이브 V25 는 코인별 동적 배수
  min(BTC_cap, K2) 와 CROSS 다. 선물 BT 정본은
  `strategies/cap_defend/backtest_futures_v25.py` 다.
- 주식은 V25 라 여기 있던 옛 블록(EWJ 유니버스·카나리 SMA300·zscore3 균등)이 라이브와
  다른 전략이었다. 주식 정본은 `strategies/cap_defend/stock_strategy_v25.py` 이고
  라이브 대조는 `trade/ops/signal_parity_daily.py` 가 매일 한다.
둘 다 지웠다. 낡은 파라미터로 도는 블록은 "V24 를 봤다" 고 믿게 만드는 쪽이 더 나쁘다.

라이브 정합 (2026-06-06): coin_live_engine 은 drift 발화 시 refill v2 (mom2 음수 코인 →
fresh healthy 교체) 를 수행한다. shadow BT 도 동일하게 refill 모드로 돌려 BT-of-record 가
라이브와 1:1 일치하게 한다.
검증(research/bt_spot_refill_vs_anchor.py): drift=0.10 에서 refill ≡ anchor-only
(5.4yr 종목교체 0일, CAGR/MDD/Cal 동일) → 채택 성과 불변.
"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..',
                                 'strategies', 'cap_defend'))
os.environ['DRIFT_HEALTH_MODE'] = 'refill'
from datetime import datetime

import unified_backtest as ub

START = '2020-10-01'
END = datetime.now().strftime('%Y-%m-%d')
print('== 현물 V24 shadow ledger (오늘 시점 가상 보유) ==')
print(f'기간: {START} ~ {END}')

# ─── 데이터 로드 ───
print('\n코인 데이터 (D) 로드...')
bars_D, funding = ub.load_data('D')

# ─── spot V24 ───
print('\n[spot] V24: D_SMA42 sn=217 n=7 drift=0.10')
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

print('\n== 완료 ==')
