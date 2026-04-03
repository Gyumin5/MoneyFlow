# 바이낸스 선물 백테스트 실행 가이드

기준일: 2026-04-02

이 문서는 현재 공개 레포 기준으로 바이낸스 선물 전략 백테스트를 어떻게 재현하는지 정리한 실행 가이드다.

## 먼저 알아둘 점

- 최종 실거래 전략은 달력 앵커 `1/10/19일`이 아니라 `snap_interval_bars` 기반의 bar-based tranche를 사용한다.
- 엔진 파일 [backtest_futures_full.py](./backtest_futures_full.py)는 두 방식을 모두 지원한다.
  - fallback: 달력 앵커 `1/10/19일`
  - 최종 채택 경로: `snap_interval_bars`
- 월별 시총 기반 유니버스 재현에는 [../../data/historical_universe.json](../../data/historical_universe.json)이 필요하다.
- 이 파일이 없으면 엔진은 동일한 월별 시총 히스토리를 재현할 수 없고, 결과가 달라질 수 있다.
- OHLCV 원본은 `1h` 기준으로 관리하고, `4h`는 항상 `1h`에서 리샘플링한다.

즉 최종 전략 재현을 하려면 `snap_interval_bars`가 들어간 러너를 봐야 한다.

## 표준 데이터 갱신

선물 데이터는 다음 명령으로 갱신한다.

```bash
python3 strategies/cap_defend/refresh_backtest_data.py --target futures
```

최신성은 다음 명령으로 확인한다.

```bash
python3 strategies/cap_defend/check_data_freshness.py
```

## 파일 역할

### 엔진

- [backtest_futures_full.py](./backtest_futures_full.py)
  - 선물 백테스트 엔진 본체
  - 카나리, health, 트랜치, 스탑, 동적 레버리지까지 포함
- [../../data/historical_universe.json](../../data/historical_universe.json)
  - 월별 시총 히스토리 유니버스 입력
  - `get_mcap()`가 읽는 재현성 핵심 파일
- `data/futures/*_1h.csv`
  - 선물 OHLCV 원본 데이터
  - `4h`는 이 파일에서 다시 만든다

### 조합 탐색

- [run_signal_screen.py](./run_signal_screen.py)
  - `1h`, `4h` 후보 전략군 단독 스크리닝
- [run_signal_combo_search.py](./run_signal_combo_search.py)
  - 상위 후보의 `2개/3개/4개` 조합 비교

### 트랜치 미세조정

- [run_snap_finetune.py](./run_snap_finetune.py)
  - 최상위 조합에 대해 `snap_interval_bars` 미세조정
- [run_snap_robustness.py](./run_snap_robustness.py)
  - `21/27/33` 같은 비정렬 구간 robustness 확인

### 최종 비교

- [run_final_signal_compare.py](./run_final_signal_compare.py)
  - 이전 실거래 조합 vs 현재 실거래 조합 직접 비교

### 실거래 코드

- [auto_trade_binance.py](../../trade/auto_trade_binance.py)
  - 최종 실거래 반영 코드
  - 여기 있는 전략 파라미터가 최종 채택안이다

## 최종 전략 재현 순서

### 1. 최종 전략 문서 확인

먼저 [futures_strategy_final.md](./futures_strategy_final.md)를 읽는다.

핵심 파라미터:
- `1h_09 snap=27`
- `4h_01 snap=120`
- `4h_09 snap=21`
- 실행층: `capmom 5/4/3x`
- 스탑: `prev_close 15% + cash_guard(34%)`

### 2. 최종 직접 비교 실행

가장 빠르게 “현재 실거래 전략이 왜 채택됐는지” 재현하려면 이 스크립트를 실행하면 된다.

```bash
python3 strategies/cap_defend/run_final_signal_compare.py
```

이 스크립트는:
- 이전 실거래 조합 `4h1 + 4h2 + 1h1`
- 현재 실거래 조합 `1h_09 + 4h_01 + 4h_09`
를 같은 실행층에서 직접 비교한다.
- 이때 월별 시총 유니버스는 `data/historical_universe.json` 기준으로 고정된다.

## 탐색 과정을 다시 밟고 싶을 때

### 1. 후보 단독 스크리닝

```bash
python3 strategies/cap_defend/run_signal_screen.py --stage 1h --workers 8
python3 strategies/cap_defend/run_signal_screen.py --stage 4h --workers 8
```

출력:
- `signal_screen_1h.csv`
- `signal_screen_4h.csv`

### 2. 조합 탐색

```bash
python3 strategies/cap_defend/run_signal_combo_search.py --workers 8
```

출력:
- `signal_combo_search.csv`

### 3. 트랜치 간격 미세조정

```bash
python3 strategies/cap_defend/run_snap_finetune.py --workers 8
```

출력:
- `snap_finetune_results.csv`

### 4. robustness 확인

```bash
python3 strategies/cap_defend/run_snap_robustness.py --workers 8
```

출력:
- `snap_robustness_results.csv`

## 코드 읽을 때 주의할 점

### 1. `backtest_futures_full.py`만 보면 헷갈릴 수 있음

이 파일에는 여전히 다음 fallback이 남아 있다.
- `snap_days = [1, 10, 19][:n_snapshots]`

하지만 이것만 보고 “최종 선물 전략도 달력 앵커”라고 이해하면 틀린다.

최종 전략은:
- `snap_interval_bars`를 명시적으로 주고
- bar index 기준으로 트랜치를 갱신한다

### 2. 실거래 코드가 최종 truth source

최종적으로 지금 뭐가 실거래에 들어가 있는지는 [auto_trade_binance.py](../../trade/auto_trade_binance.py)를 보면 된다.

즉:
- 연구 과정: `run_*` + `backtest_futures_full.py`
- 최종 채택안: `futures_strategy_final.md`
- 실제 라이브 반영값: `auto_trade_binance.py`

## 추천 읽기 순서

다른 사람이 처음 repo를 읽는다면 이 순서를 추천한다.

1. [futures_research_history.md](./futures_research_history.md)
2. [futures_strategy_final.md](./futures_strategy_final.md)
3. [futures_backtest_howto.md](./futures_backtest_howto.md)
4. [run_final_signal_compare.py](./run_final_signal_compare.py)
5. [backtest_futures_full.py](./backtest_futures_full.py)
6. [auto_trade_binance.py](../../trade/auto_trade_binance.py)

## 한 줄 요약

최종 선물 전략을 재현하려면:
- `futures_strategy_final.md`로 파라미터를 보고
- `refresh_backtest_data.py --target futures`로 데이터를 갱신하고
- `run_final_signal_compare.py`로 결과를 확인하고
- 필요하면 `run_signal_combo_search.py`, `run_snap_finetune.py`로 탐색 과정을 다시 밟으면 된다.
