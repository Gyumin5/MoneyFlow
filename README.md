# MoneyFlow

이 저장소는 현재 운영 중인 세 가지 전략의 백테스트/실거래 코드를 포함한다.

- 주식: `V17`
- 현물 코인: `V18`
- 바이낸스 선물: 현재 실거래 조합

처음 받았을 때는 아래 순서만 따르면 된다.

## 0. 의존성 설치

```bash
pip install -r requirements.txt
```

백테스트만 돌릴 경우 `numpy`, `pandas`, `yfinance`, `requests`면 충분하다.
`python-binance`, `pyupbit`, `flask`는 실거래/서버 운영용이다.

## 1. 먼저 볼 문서

- 전체 안내:
  - [strategies/cap_defend/README.md](./strategies/cap_defend/README.md)
- 통합 재현 가이드:
  - [strategies/cap_defend/repo_backtest_guide.md](./strategies/cap_defend/repo_backtest_guide.md)

## 2. 데이터 최신성 확인

```bash
python3 strategies/cap_defend/check_data_freshness.py
```

## 3. 데이터 갱신

### 주식

```bash
python3 strategies/cap_defend/refresh_backtest_data.py --target stock
```

### 현물 코인

```bash
python3 strategies/cap_defend/refresh_backtest_data.py --target coin
```

### 선물

```bash
python3 strategies/cap_defend/refresh_backtest_data.py --target futures
```

## 4. 단독 백테스트 실행

### 주식

```bash
python3 strategies/cap_defend/run_current_stock_backtest.py
```

### 현물 코인

```bash
python3 strategies/cap_defend/run_current_coin_backtest.py
```

### 선물

```bash
python3 strategies/cap_defend/run_current_futures_backtest.py
```

## 5. 현재 공식 전략

- 주식:
  - `V17`
- 현물 코인:
  - `V18`
- 선물:
  - `1h_09 + 4h_01 + 4h_09`
  - `capmom 5/4/3x`
  - `prev_close15 + cash_guard(34%)`

## 6. 주의

- `data/historical_universe.json`은 재현성 핵심 입력이다.
- 선물은 `1h` 원본을 기준으로 하고 `4h`는 항상 `1h`에서 리샘플링한다.
- 주식은 `stock_cache` 우선, 없으면 Yahoo fallback 구조다.

## 한 줄 요약

처음 받으면:

1. `check_data_freshness.py`
2. `refresh_backtest_data.py`
3. `run_current_*_backtest.py`

이 순서로 보면 된다.
