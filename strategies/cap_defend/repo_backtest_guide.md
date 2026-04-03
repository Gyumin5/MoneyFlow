# 백테스트 재현 가이드

이 문서는 현재 공개 레포 기준으로 주식, 현물 코인, 선물 전략을 가장 짧은 경로로 재현하는 방법을 정리한다.

## 읽는 순서

1. 현물 코인:
   - [coin_backtest_howto.md](./coin_backtest_howto.md)
2. 주식:
   - [stock_backtest_howto.md](./stock_backtest_howto.md)
3. 선물:
   - [futures_backtest_howto.md](./futures_backtest_howto.md)

## 전략별 실행 명령

### 현물 코인

```bash
python3 strategies/cap_defend/backtest_official.py --coin-only --version v18
```

### 주식

```bash
python3 strategies/cap_defend/backtest_official.py --stock-only --version v17
```

### 선물

데이터 갱신:

```bash
python3 strategies/cap_defend/download_futures_data.py
```

최종 비교:

```bash
python3 strategies/cap_defend/run_final_signal_compare.py
```

## 데이터 기준

세 전략 모두 같은 원칙으로 관리한다.

- 실행 전에 데이터 최신성을 먼저 확인
- 전략 확정 직전에는 데이터 스냅샷을 함께 보관
- 재현에 필요한 핵심 입력 파일은 git에 포함

전략별 기준은 다음과 같다.

- 현물 코인:
  - 로컬 CSV + `data/historical_universe.json`
- 주식:
  - `stock_cache` 우선, 없으면 Yahoo fallback
- 선물:
  - `1h` 원본 + funding
  - `4h`는 `1h`에서 리샘플

## 최소 확인 체크리스트

실행 전에 아래만 확인하면 된다.

- 현물:
  - 주요 CSV 마지막 날짜
  - `historical_universe.json` 존재 여부
- 주식:
  - `stock_cache` 마지막 날짜
- 선물:
  - 주요 `1h` / `funding` 마지막 시점

한 번에 확인하려면:

```bash
python3 strategies/cap_defend/check_data_freshness.py
```

## 현재 공식 버전

- 현물: `V18`
- 주식: `V17`
- 선물: `1h_09 + 4h_01 + 4h_09`
