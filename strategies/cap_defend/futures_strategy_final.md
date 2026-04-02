# 바이낸스 선물 최종 전략 설명

기준일: 2026-04-01  
적용 대상: 바이낸스 USDT-M 선물 자동매매

## 개요

최종 채택 전략은 `1h_09 + 4h_01 + 4h_09` 3전략 앙상블이다.

실행층은 고정 배수가 아니라:
- 종목별 동적 레버리지 `cap_mom_blend_543_cash`
- 스탑 `prev_close 15%`
- 스탑 게이트 `cash_guard(34%)`

를 사용한다.

즉 구조는:
1. 1h / 4h 신호 전략 3개가 목표 비중을 생성
2. 동일 가중으로 합산
3. 종목별로 `5x / 4x / 3x` 레버리지를 다르게 부여
4. 위험 상태(`CASH >= 34%`)일 때만 스탑을 활성화

중요:
- 이 전략의 트랜치 검증과 실거래 반영은 `snap_interval_bars` 기반의 bar-based tranche를 기준으로 했다.
- [backtest_futures_full.py](./backtest_futures_full.py)에는 구방식 달력 앵커 `1/10/19일` fallback도 남아 있지만, 최종 채택 전략은 그 fallback을 사용하지 않는다.

## 최종 신호 전략

### 1h_09
- interval: `1h`
- `sma_bars = 168`
- `mom_short_bars = 36`
- `mom_long_bars = 720`
- `health_mode = mom2vol`
- `vol_mode = bar`
- `vol_threshold = 0.80`
- `n_snapshots = 3`
- `snap_interval_bars = 27`

의미:
- 1시간봉 기반의 빠른 전략
- 3트랜치 구조를 유지하면서도 `24`시간 주기에 과도하게 정렬되지 않도록 `27`을 선택

선택 이유:
- `24`보다 `21/27/33`이 더 좋았고,
- 그중 `27`은 성능이 높으면서 하루 주기와 덜 맞물려 일반화에 유리하다고 판단했다.

### 4h_01
- interval: `4h`
- `sma_bars = 240`
- `mom_short_bars = 10`
- `mom_long_bars = 30`
- `health_mode = mom1vol`
- `vol_mode = daily`
- `vol_threshold = 0.05`
- `n_snapshots = 3`
- `snap_interval_bars = 120`

의미:
- 장기 추세 코어 역할
- 백테스트에서 `120`이 가장 안정적으로 우수했다

### 4h_09
- interval: `4h`
- `sma_bars = 120`
- `mom_short_bars = 20`
- `mom_long_bars = 120`
- `health_mode = mom2vol`
- `vol_mode = bar`
- `vol_threshold = 0.60`
- `n_snapshots = 3`
- `snap_interval_bars = 21`

의미:
- 중기 회전 전략
- 최고점은 `18`이었지만, 매일 같은 시간대 정렬을 줄이기 위해 `21`을 선택

선택 이유:
- `21`은 `18` 대비 성능 손실이 작았고,
- 3트랜치 구조를 유지하면서 `24`와 덜 정렬된다.

## 실행층

### 종목별 동적 레버리지
- mode: `cap_mom_blend_543_cash`
- ceiling: `5x`
- mid: `4x`
- floor: `3x`

규칙:
- 같은 시점에도 종목별로 다른 배수를 줄 수 있다
- cap rank(유니버스 순서)와 모멘텀을 함께 반영해 더 강하고 상대적으로 덜 불안정한 코인에 높은 배수를 준다
- `CASH >= 34%`이면 한 단계씩 배수를 낮춘다

해석:
- 최상위 종목: 기본 `5x`
- 그다음 상위 2개: 기본 `4x`
- 나머지: 기본 `3x`
- 단, 방어 상태에서는 `5 -> 4`, `4 -> 3` 식으로 자동 하향

### 스탑
- kind: `prev_close_pct`
- `stop_pct = 15%`
- gate: `cash_guard`
- `cash_guard threshold = 34%`

동작:
- 목표 포트폴리오의 `CASH` 비중이 `34%` 이상일 때만 스탑 활성화
- 각 포지션에 대해 직전 완성 1시간봉 종가 대비 `-15%`에 reduce-only `STOP_MARKET` 주문을 건다
- 매 `--trade` 실행마다 기존 스탑을 취소하고 다시 건다

선택 이유:
- always-on 스탑은 수익 훼손이 컸다
- 조건부 스탑은 특히 고배수/동적 레버리지 구조에서 위험조정 성과를 개선했다

## 최종 비교 결과

동일 실행층 `capmom 543 + prev_close15 + cash_guard(34%)` 기준:

기존 실거래 조합 `4h1 + 4h2 + 1h1`
- Cal `4.30`
- CAGR `+250.3%`
- MDD `-58.2%`
- Liq `7`
- Stops `11`
- Rebal `1347`

최종 후보 `1h_09 + 4h_01 + 4h_09`
- Cal `4.98`
- CAGR `+221.1%`
- MDD `-44.4%`
- Liq `4`
- Stops `16`
- Rebal `2660`

해석:
- 수익률(CAGR)은 다소 낮아졌지만,
- MDD와 청산이 줄고 Calmar가 크게 개선됐다
- 실거래 운영 기준으로는 새 전략이 더 적합하다고 판단했다

## 운영 메모

- 카나리 `OFF -> ON` 플립 시에는 해당 전략의 3트랜치를 모두 즉시 재평가한다
- 그 이후에는 `snap_interval_bars`와 3트랜치 offset에 따라 순차 갱신한다
- 상태 파일을 리셋하면 다음 실행에서 새로운 기준으로 다시 판단한다
- 현재 구현은 매 실거래 실행마다:
  - 목표 비중 계산
  - 리밸런싱 필요 여부 판단
  - 필요 시 주문 실행
  - 스탑 주문 취소/재등록
  순서로 동작한다

## 관련 파일

- 실거래 코드: `trade/auto_trade_binance.py`
- 스탑 연구 요약: `strategies/cap_defend/stoploss_summary.md`
- 최종 비교 스크립트: `strategies/cap_defend/run_final_signal_compare.py`
- 트랜치 robustness 결과: `strategies/cap_defend/snap_robustness_results.csv`
- 연구 히스토리: `strategies/cap_defend/futures_research_history.md`
