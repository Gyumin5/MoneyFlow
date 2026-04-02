# 바이낸스 선물 전략 연구 히스토리

기준일: 2026-04-02

이 문서는 바이낸스 선물 전략이 어떤 실험 과정을 거쳐 현재 실거래 버전으로 수렴했는지 정리한 기록이다.

## 한 줄 요약

최종 채택 전략은 `1h_09 + 4h_01 + 4h_09` 앙상블이며, 트랜치는 달력 앵커가 아니라 `snap_interval_bars` 기반의 bar-based 방식으로 검증하고 실거래에 반영했다.

## 연구 단계

### 1. 선물 백테스트 엔진 정비

- 초기 선물 엔진은 일봉/시간봉 혼합 실험 단계였고, 인덱싱, 레버리지 적용, 격리마진, look-ahead 성격의 버그를 수정했다.
- 이 단계의 중심 파일은 [backtest_futures_full.py](./backtest_futures_full.py)다.
- 이 엔진은 두 가지 트랜치 방식을 모두 지원한다.
  - 구방식: 달력 앵커 `1/10/19일`
  - 신방식: `snap_interval_bars` 기반 bar-based tranche

### 2. 단일 신호 전략 탐색

- `1h`, `4h`, `D` 시간축 후보를 폭넓게 비교했다.
- 이후 실전성 기준에서 `1h/4h` 조합이 더 유리하다고 판단해, `1h` 후보군과 `4h` 후보군을 따로 스크리닝했다.
- 관련 러너:
  - [run_signal_screen.py](./run_signal_screen.py)

### 3. 앙상블 조합 탐색

- 스크리닝된 후보를 대상으로 `2개 / 3개 / 4개` 조합을 비교했다.
- 이 단계에서 상위 조합으로 다음 후보들이 살아남았다.
  - `1h_09 + 4h_01 + 4h_09`
  - `1h_09 + 1h_05 + 4h_01 + 4h_08`
  - `4h_01 + 4h_09`
- 관련 러너:
  - [run_signal_combo_search.py](./run_signal_combo_search.py)

### 4. 실행층 검증

- 신호 전략과 별개로 실행층도 따로 비교했다.
- 고정 배수보다 종목별 동적 `5/4/3x`가 더 우수했고,
- 그중 `coin_capmom_543_cash`가 가장 좋았다.
- 여기에 조건부 스탑 `prev_close15 + cash_guard(34%)`를 결합한 구성이 최종 실행층으로 채택됐다.

### 5. 트랜치 간격 미세조정

- 최종 후보 조합을 대상으로 `snap_interval_bars`를 다시 튜닝했다.
- 이 단계는 명확히 bar-based tranche 검증이며, 달력 앵커 `1/10/19`를 최종 전략으로 채택한 것이 아니다.
- 관련 러너:
  - [run_snap_finetune.py](./run_snap_finetune.py)

결과:
- `1h_09`: `24`보다 `21/27/33`이 유리
- `4h_01`: `120` 유지가 최선
- `4h_09`: `18`이 최고점이었으나, 시간대 정렬 완화를 위해 `21` 채택

최종 선택:
- `1h_09 snap=27`
- `4h_01 snap=120`
- `4h_09 snap=21`

### 6. 기존 실거래 조합과 최종 비교

동일 실행층 기준으로 기존 `4h1 + 4h2 + 1h1`과 새 조합을 비교했다.

기존:
- Cal `4.30`
- CAGR `+250.3%`
- MDD `-58.2%`
- Liq `7`

최종:
- Cal `4.98`
- CAGR `+221.1%`
- MDD `-44.4%`
- Liq `4`

해석:
- CAGR은 다소 낮아졌지만,
- MDD와 청산 리스크가 줄고 Calmar가 크게 개선돼 실거래 기준으로 교체 가치가 충분하다고 판단했다.

## 달력 앵커와 bar-based의 관계

헷갈리기 쉬운 지점은 [backtest_futures_full.py](./backtest_futures_full.py)에 구방식과 신방식이 함께 남아 있다는 점이다.

- 파일 기본 fallback:
  - `n_snapshots <= 3`일 때 `snap_days = [1, 10, 19]`
- 실제 최종 전략 경로:
  - `snap_interval_bars`를 명시해서 bar-based tranche 사용

즉 `backtest_futures_full.py`만 보면 `1/10/19일` 앵커가 기본처럼 보일 수 있지만, 최종 실거래 전략은 그 fallback을 쓰지 않는다.

## 실거래 반영 결과

실거래 코드는 [auto_trade_binance.py](../../trade/auto_trade_binance.py)에 반영했다.

주요 운영 보강:
- 주문 재시도
- 포지션/PV 조회 재시도
- algo conditional stop 취소/재등록
- `ISOLATED` 마진 사용
- dust 포지션 숨김
- 전량 청산 시 raw 수량 사용
- state bool 정규화
- 리포트 시 실시간 전략 상태 재계산

## 관련 파일

- 엔진: [backtest_futures_full.py](./backtest_futures_full.py)
- 최종 전략 문서: [futures_strategy_final.md](./futures_strategy_final.md)
- 실행 가이드: [futures_backtest_howto.md](./futures_backtest_howto.md)
- 스탑 연구 요약: [stoploss_summary.md](./stoploss_summary.md)
- 조합 탐색: [run_signal_combo_search.py](./run_signal_combo_search.py)
- 트랜치 미세조정: [run_snap_finetune.py](./run_snap_finetune.py)
- 실거래 코드: [auto_trade_binance.py](../../trade/auto_trade_binance.py)
