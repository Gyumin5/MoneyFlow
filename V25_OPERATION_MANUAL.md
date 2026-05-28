# V25 운영 매뉴얼 (2026-05-28 도입)

V24 → V25 마이그레이션. 선물 sleeve 만 변경 (K2 동적 per-coin L + CROSS 마진). 코인 spot / 주식 / 자산배분 V24 그대로 유지.

## 결정 근거 (요약)

5.6yr BT (2020-10 ~ 2026-05, look-ahead 차단 shift(1) lag) 검증.
- BTC cap × per-coin SMA grid + window rank-sum plateau search
- K2 (SMA=7 h=2.5%) 25 cfg × 윈도우 rank-sum 1위, 인접 ±1 cfg 도 Top 10 안 = plateau center 단단함
- adversarial stress (flip-then-crash, 단일 -70% gap, 동시 -55%) 청산 0
- robustness sweep — SMA42/ms18/ml127 plateau 확인 (J 81 cfg rank 1)
- vs J (mom-based) Cal 7.45: K2 8.12 (+0.67) + MDD 7.4pp 개선

AI 합의 (codex + gemini + claude, 2 round)
- A/C/G kill-switch BT 발동 0 (false alarm 우려) → 라이브 미포함
- B 안 (J Lmax=4 CROSS no-stop) 권고. K2 가 J 보다 안정성 우월로 최종 채택.

## 파라미터 (선물 sleeve 만)

| 항목 | V24 | V25 |
|---|---|---|
| Sleeve | D_SMA42 sn=95 n=5 drift=0.03 | 동일 (V24 유지) |
| 레버리지 | 고정 L3 ISO | 동적 per-coin: min(BTC_cap, K2) |
| Lmin/Lmid/Lmax | 3/3/3 | 2/3/4 |
| BTC cap | 없음 | BTC/SMA42 > 1.05 → L4, > 1.015 → L3, else L2 |
| per-coin K2 | 없음 | close/SMA7 > 1.075 → L4, > 1.025 → L3, else L2 |
| 마진모드 | ISOLATED | CROSSED |
| 거래비용 | 0.04% | 0.04% |
| 유지증거금 BT | 0.4% | 0.4% (BTC/ETH 정확, 알트 보수적) |
| cron | 09:05 KST 1d × 1 | 동일 |
| 디버그 로그 | 기본 | DEBUG_LEVERAGE=True, DEBUG_MARGIN=True |

## BT 성능

### 단독 sleeve (5.6yr, alloc 100%)

| sleeve | Cal | CAGR | MDD | Sharpe |
|---|---|---|---|---|
| V24 L3 ISO | 6.44 | 283% | -43.9% | 1.86 |
| J Lmax=4 CROSS | 7.87 | 360% | -45.7% | 1.90 |
| K2 Lmax=4 CROSS | 8.12 | 312% | -38.3% | 1.90 |

### alloc 60/25/15 (주식/spot/fut)

| trigger | fut sleeve | Cal | CAGR | MDD | Sharpe |
|---|---|---|---|---|---|
| 일별 rebal | V24 L3 | 3.08 | 54.5% | -17.7% | 1.93 |
| 일별 rebal | J L4 | 3.34 | 60.8% | -18.2% | 1.95 |
| 일별 rebal | K2 L4 | 3.47 | 56.0% | -16.1% | 1.95 |
| T1+T3U (라이브) | V24 L3 | 4.78 | 102% | -21.4% | 1.00 |
| T1+T3U (라이브) | J L4 | 5.01 | 114% | -22.8% | 1.06 |
| T1+T3U (라이브) | K2 L4 | 5.72 | 106% | -18.6% | 1.02 |

## 동적 L 결정 로직

매일 09:05 cron에서:

```python
btc_close = bars['BTC']['Close'].values  # 1D 봉
prev_close = btc_close[:-1]  # 진행중 봉 제외 (look-ahead 차단)
btc_sma42 = mean(prev_close[-42:])
btc_ratio = prev_close[-1] / btc_sma42

if btc_ratio > 1.05:
    btc_cap = 4  # Lmax
elif btc_ratio > 1.015:
    btc_cap = 3  # Lmid
else:
    btc_cap = 2  # Lmin

for coin in selected_universe:
    coin_close = bars[coin]['Close'].values
    coin_prev = coin_close[:-1]
    coin_sma7 = mean(coin_prev[-7:])
    coin_ratio = coin_prev[-1] / coin_sma7
    
    if coin_ratio > 1.075:
        k2_lev = 4
    elif coin_ratio > 1.025:
        k2_lev = 3
    else:
        k2_lev = 2
    
    final_L[coin] = min(btc_cap, k2_lev)
```

## 마진모드 (V25 = 항상 CROSSED)

V25 가정: 모든 universe 이미 CROSSED. 라이브 전환 시 사용자 사전 수동 변경 완료.

코드 동작 (매 cron):
1. `verify_position_mode_oneway` — 계정 one-way mode 확인 (hedge 면 ABORT)
2. 각 코인 verify_margin_type — CROSSED 가 아니면 ABORT (set 안 함, 비정상 상황 알림)
3. `set_leverage` → `verify_leverage` (매 cron 동적)
4. all-or-nothing: 한 코인이라도 verify 실패 → 매매 전체 ABORT (다음 cron 재시도)
5. 모든 검증 통과 → `force_cancel_all_orders` → `execute_rebalance`

### V25 첫 cron 실행 전 사용자 수동 작업 (필수)

라이브 V24 → V25 전환 시 사용자가 사전 수행:

**Pre-flight 체크리스트** (V25 cron 활성화 전, 수동):
- [ ] 현 V24 ISOLATED 포지션 모두 close (Binance UI 또는 trade/auto_trade_binance.py --close-all)
- [ ] 모든 미체결 주문 cancel
- [ ] 모든 universe (BTC, ETH, SOL, BNB, XRP, DOGE, ADA, AVAX 등) 마진모드 → CROSSED 수동 변경 (Binance UI: 포지션 없을 때만 가능)
- [ ] 계정 position mode = one-way (hedge X) 확인 (Binance UI 설정)
- [ ] V25 코드 + futures_live_config.py 서버 배포 완료
- [ ] cron 활성화 → 첫 09:05 cron fire

**fail-safe 동작** (코드):
- 어떤 코인 verify_margin_type 실패 = 마진모드가 CROSSED 아님 → ABORT + 알림 (set 안 함)
  · 이때 사용자가 Binance UI 에서 수동 변경 후 다음 cron 재시도
- one-way mode 위반 → ABORT + 알림
- data['D'] 누락 → ABORT + 알림
- set_leverage / verify_leverage 실패 → ABORT + 알림
- 안전: V25 fail 시 매매 안 함, V24 포지션 (있다면) 유지, 알림으로 사용자 통지

## V24 → V25 회귀 검증 (필수, 라이브 전환 전)

backtest_futures_v25.py 가 V24 reference 결과를 재현하는지 확인:

```bash
# V25 BT 엔진을 V24 spec 으로 실행 (leverage=3.0 scalar, no mom_stop, no catastrophic)
python3 -c "
import sys; sys.path.insert(0, '/home/gmoh/mon/251229/strategies/cap_defend')
from backtest_futures_v25 import load_data, run
import os; os.environ['DRIFT_HEALTH_MODE'] = 'refill'
bars, fund = load_data('D')
r = run(bars, fund, interval='D', leverage=3.0,
    sma_days=42, mom_short_days=18, mom_long_days=127,
    n_snapshots=5, snap_interval_bars=95, drift_threshold=0.03,
    universe_size=3, selection='greedy', cap=1/3,
    tx_cost=0.0006, maint_rate=0.004, vol_days=90, vol_threshold=0.05,
    canary_hyst=0.015, health_mode='mom2vol',
    start_date='2020-10-01', end_date='2026-05-13')
print(f'V25 engine, L=3 scalar: CAGR={r[\"CAGR\"]*100:.1f}%, MDD={r[\"MDD\"]*100:.1f}%, Cal={r[\"Cal\"]:.2f}')
"
```

기대: V24 reference (backtest_futures_full.py L3) 결과와 ±5% 이내. Cal 6.44 부근.

차이 ≥5% 면 V25 엔진에 bug 있음 → 라이브 전환 보류.

## 코드 변경 요약 (V24 → V25)

### 라이브 엔진
- `trade/auto_trade_binance.py`:
  - LEVERAGE_FLOOR/MID/CEILING = 2/3/4 (V24 3/3/3)
  - K2_SMA_PERIOD=7, K2_HYST=0.025
  - BTC_CAP_SMA_PERIOD=42, BTC_CAP_THR_MID=1.015, BTC_CAP_THR_MAX=1.05
  - MARGIN_TYPE='CROSSED' (V24 ISOLATED)
  - DEBUG_LEVERAGE=True, DEBUG_MARGIN=True
  - SCHEMA_VERSION='V25'
  - 신규 함수: `_calc_btc_cap_lev(data_d)`, `_calc_percoin_k2_lev(coin, data_d)`
  - `get_coin_leverage_map(target, data_1h, data_d)` 시그니처 변경 (data_d 추가)
  - `set_leverage / set_margin_type` 로그 강화

### 설정
- `strategies/cap_defend/futures_live_config.py`:
  - LEVERAGE_MIN=2, LEVERAGE_MID=3, LEVERAGE_MAX=4
  - K2_SMA_PERIOD=7, K2_HYST=0.025
  - BTC_CAP_SMA_PERIOD=42, BTC_CAP_THR_MID=1.015, BTC_CAP_THR_MAX=1.05
  - MARGIN_TYPE='CROSSED'

### 백테스트
- `strategies/cap_defend/backtest_futures_v25.py` (신규):
  - `run()` leverage scalar/Series/dict 지원
  - CROSS 마진 worst-case 청산 모델
  - `build_K2_signal(bars, ...)` 라이브 spec 동일 시그널 헬퍼
- `strategies/cap_defend/backtest_futures_full.py` (V24 reference 유지, deprecated 표시 X)

### 권고
- `strategies/cap_defend/recommend_personal.py`:
  - STRATEGY_VERSION='V25'
  - VERSION_HISTORY 에 V25 entry 추가
  - 선물 sleeve 설명 V25 갱신

### 매뉴얼
- `V24_OPERATION_MANUAL.md`: deprecated 표시 (V25 도입)
- `V25_OPERATION_MANUAL.md`: 본 문서 (신규)
- `CLAUDE.md`: 선물 전략 규칙 V25 갱신

## 라이브 1주차 monitoring 체크리스트

매일 (cron 09:05 + 09:15 daily report):
- [ ] set_leverage 호출 결과 (디버그 로그)
- [ ] set_margin_type 호출 결과 (ISOLATED 잔존 여부)
- [ ] 동적 L 분포 (BTC_cap / K2_per_coin / final_L)
- [ ] drift fire 빈도 (V24 대비 변동 X 정상)
- [ ] PnL 변화 (V24 대비 +/- 비교)

이상 신호 즉시 알람:
- Binance API liq price vs BT 내부 추정 차이 > 5%p
- mark-index basis 급등 (> 0.5%)
- 잔고 vs BT 예상 괴리 > 5%
- set_margin_type 반복 실패 (ISOLATED 강제 유지)
- 청산 1건 발생

## 롤백 절차 (필요 시)

V25 → V24 복귀:
1. `trade/auto_trade_binance.py`: SCHEMA_VERSION='V24', LEVERAGE_FLOOR/MID/CEILING=3/3/3, MARGIN_TYPE='ISOLATED' (또는 set_margin_type 호출 인자 'ISOLATED')
2. `strategies/cap_defend/futures_live_config.py`: V24 spec 복원
3. 모든 포지션 close → set_margin_type ISOLATED → set_leverage 3 → cron 재시작
4. recommend_personal.py STRATEGY_VERSION='V24'
5. 서버 동일 절차로 적용

## 참고

- 채택 BT 결과: `/tmp/dyn_L_K2_plateau.out`, `/tmp/alloc_v24_j_k2_v2.out`
- ai-debate 결정 로그: `/home/gmoh/.claude/state/ai-debate/run-20260528T*`
- 단독 sleeve trace 검증: 카나리 OFF 시 CASH (대부분 5.6yr 기간), ON 시 동적 L 적용
