# Cap Defend 전략 디렉터리 안내

이 디렉터리는 주식, 현물 코인, 선물 전략의 백테스트/실거래 관련 코드를 함께 담고 있다.

처음 보는 경우에는 모든 파일을 읽기보다 아래 순서대로 보는 것이 가장 빠르다.

## 1. 바로 재현하는 공식 경로

### 통합 가이드

- [repo_backtest_guide.md](./repo_backtest_guide.md)

### 현물 코인

- 설명:
  - [coin_backtest_howto.md](./coin_backtest_howto.md)
- 실행:
  - [run_current_coin_backtest.py](./run_current_coin_backtest.py)
- 엔진:
  - [coin_engine.py](./coin_engine.py)
  - [coin_helpers.py](./coin_helpers.py)
  - [coin_dd_exit.py](./coin_dd_exit.py)

### 주식

- 설명:
  - [stock_backtest_howto.md](./stock_backtest_howto.md)
- 실행:
  - [run_current_stock_backtest.py](./run_current_stock_backtest.py)
- 엔진:
  - [stock_engine.py](./stock_engine.py)

### 선물

- 설명:
  - [futures_backtest_howto.md](./futures_backtest_howto.md)
  - [futures_strategy_final.md](./futures_strategy_final.md)
- 실행:
  - [download_futures_data.py](./download_futures_data.py)
  - [run_current_futures_backtest.py](./run_current_futures_backtest.py)
- 엔진:
  - [backtest_futures_full.py](./backtest_futures_full.py)
  - [futures_ensemble_engine.py](./futures_ensemble_engine.py)
  - [futures_live_config.py](./futures_live_config.py)

## 2. 공통 유틸리티

- 데이터 최신성 확인:
  - [check_data_freshness.py](./check_data_freshness.py)
- 데이터 갱신 진입점:
  - [refresh_backtest_data.py](./refresh_backtest_data.py)

## 3. 연구 / 탐색 파일

[research/](./research/) 디렉터리에 모아두었다. 공식 재현에는 필요 없다.

자세한 목록은 [research/README.md](./research/README.md) 참조.

## 4. 운영 관련

- 추천/리포트 생성:
  - [recommend.py](./recommend.py)
  - [recommend_personal.py](./recommend_personal.py)
- 전략 설명 UI:
  - [strategy.html](./strategy.html) — 전략 요약 페이지
  - [strategy_guide.html](./strategy_guide.html) — 상세 전략 가이드
- 보조 스크립트:
  - [daily_history.py](./daily_history.py)
  - [serve.py](./serve.py)

## 한 줄 요약

바로 백테스트하려면:

1. `check_data_freshness.py`
2. `refresh_backtest_data.py`
3. 각 전략의 공식 실행 스크립트

만 보면 된다.
