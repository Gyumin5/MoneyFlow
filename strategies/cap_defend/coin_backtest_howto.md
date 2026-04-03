# 현물 코인 백테스트 가이드

이 문서는 현재 공개 레포 기준으로 현물 코인 전략 백테스트를 재현하는 최소 절차를 정리한다.

## 대상 전략

- 공식 전략: `V18`
- 엔진:
  - [backtest_official.py](./backtest_official.py)
  - [coin_engine.py](./coin_engine.py)
  - [coin_helpers.py](./coin_helpers.py)
  - [coin_dd_exit.py](./coin_dd_exit.py)

## 핵심 데이터

- 가격 데이터:
  - `data/*.csv`
- 월별 유니버스 히스토리:
  - [../../data/historical_universe.json](../../data/historical_universe.json)

현물 코인 전략은 로컬 CSV와 월별 시총 히스토리를 기준으로 동작한다. 즉, 백테스트 재현성은 로컬 데이터 상태에 직접적으로 의존한다.

## 기본 실행

현물 코인 전략만 돌리려면:

```bash
python3 strategies/cap_defend/backtest_official.py --coin-only --version v18
```

데이터를 먼저 갱신하려면:

```bash
python3 strategies/cap_defend/refresh_backtest_data.py --target coin
```

출력에는 기본적으로 다음 구간이 포함된다.

- `2018-01-01 ~ 2025-06-30`
- `2019-01-01 ~ 2025-06-30`
- `2021-01-01 ~ 2025-06-30`

## 데이터 최신성 확인

현물 코인은 현재 표준 갱신 진입점을 다음으로 통일했다.

```bash
python3 strategies/cap_defend/refresh_backtest_data.py --target coin
python3 strategies/cap_defend/check_data_freshness.py
```

최신성 확인은 통합 스크립트를 쓰면 된다.

```bash
python3 strategies/cap_defend/check_data_freshness.py
```

검증 포인트:

- 주요 코인 CSV의 마지막 날짜가 동일한지
- `historical_universe.json`이 존재하는지

## 월별 유니버스 히스토리

`historical_universe.json`은 현재 공개 레포에서 재현용 canonical 입력으로 관리한다.

- 정상 백테스트/재현 workflow에서는 이 파일을 그대로 사용한다.
- 일반적인 데이터 갱신은 가격 CSV를 갱신하고, 월별 시총 히스토리는 git에 포함된 파일을 기준으로 고정한다.

## 백업 원칙

현물 코인은 로컬 CSV가 기준이므로, 중요한 전략 확정 시점에는 데이터 스냅샷을 같이 보관하는 것이 좋다.

권장:

- 평소: 최신성만 확인
- 전략 확정 직전:
  - `data/*.csv`
  - `data/historical_universe.json`
    을 함께 백업

## 정리

재현에 필요한 최소 세트는 다음과 같다.

- `backtest_official.py`
- `coin_engine.py`
- `coin_helpers.py`
- `coin_dd_exit.py`
- `data/*.csv`
- `data/historical_universe.json`
