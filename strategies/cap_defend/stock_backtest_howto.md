# 주식 백테스트 가이드

이 문서는 현재 공개 레포 기준으로 주식 전략 백테스트를 재현하는 최소 절차를 정리한다.

## 대상 전략

- 공식 전략: `V17`
- 엔진:
  - [backtest_official.py](./backtest_official.py)
  - [stock_engine.py](./stock_engine.py)

## 데이터 소스

주식 전략은 다음 우선순위로 가격을 읽는다.

1. `strategies/cap_defend/data/stock_cache/*.csv`
2. 메인 `data/*.csv`
3. `yfinance` 다운로드

즉, 현물/선물과 달리 완전 로컬 고정 구조가 아니라 캐시/로컬 데이터 우선 + 필요 시 Yahoo fallback 구조다.

## 기본 실행

주식 전략만 돌리려면:

```bash
python3 strategies/cap_defend/backtest_official.py --stock-only --version v17
```

데이터를 먼저 갱신하려면:

```bash
python3 strategies/cap_defend/refresh_backtest_data.py --target stock
```

출력에는 기본적으로 다음 구간이 포함된다.

- `2017-01-01 ~ 2025-12-31`
- `2018-01-01 ~ 2025-12-31`
- `2021-01-01 ~ 2025-12-31`

## 데이터 최신성 확인

가장 중요한 것은 `stock_cache`의 마지막 날짜다.

현재 표준 확인 순서는 다음과 같다.

```bash
python3 strategies/cap_defend/refresh_backtest_data.py --target stock
python3 strategies/cap_defend/check_data_freshness.py
```

최신성 확인은 통합 스크립트를 쓰면 된다.

```bash
python3 strategies/cap_defend/check_data_freshness.py
```

검증 포인트:

- 핵심 ETF 캐시 파일이 존재하는지
- 마지막 날짜가 크게 어긋나지 않는지
- 캐시가 비어 있거나 오래되면 실행 시 Yahoo fallback이 발생할 수 있다는 점

## 재현성 주의점

주식 전략은 캐시가 없으면 Yahoo에서 데이터를 다시 받는다. 따라서 완전 동일한 재현이 중요하다면 캐시 파일을 유지한 상태에서 실행하는 것이 좋다.

권장:

- 평소: `stock_cache` 최신성 확인
- 전략 확정 직전:
  - `strategies/cap_defend/data/stock_cache/*.csv`
    를 스냅샷으로 보관

## 정리

재현에 필요한 최소 세트는 다음과 같다.

- `backtest_official.py`
- `stock_engine.py`
- `strategies/cap_defend/data/stock_cache/*.csv` 또는 안정적인 Yahoo 접근 환경
