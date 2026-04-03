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

## 6. 저장소 구조

```
MoneyFlow/
├── README.md                     진입점 — 설치, 실행 순서 안내
├── requirements.txt              pip 의존성
├── data/
│   └── historical_universe.json  월별 시총 유니버스 (재현 핵심 입력)
├── config/
│   └── upbit.example.py          API 키 템플릿
├── scripts/
│   ├── run_recommend.sh          서버 크론용 추천 실행
│   └── run_trade.sh              서버 크론용 매매 실행
│
├── strategies/cap_defend/        ★ 전략 핵심
│   ├── README.md                 디렉터리 안내
│   ├── repo_backtest_guide.md    통합 백테스트 재현 가이드
│   ├── *_backtest_howto.md       전략별 백테스트 설명 (코인/주식/선물)
│   ├── futures_strategy_final.md 선물 최종 전략 명세
│   │
│   ├── run_current_*_backtest.py 공식 백테스트 진입점 (3개)
│   ├── check_data_freshness.py   데이터 최신성 확인
│   ├── refresh_backtest_data.py  데이터 갱신
│   │
│   ├── coin_engine.py            현물 코인 엔진
│   ├── stock_engine.py           주식 엔진
│   ├── backtest_official.py      V12~V17 공식 백테스트
│   ├── backtest_futures_full.py  선물 백테스트 엔진
│   ├── futures_ensemble_engine.py 선물 앙상블 실행 엔진
│   ├── futures_live_config.py    현재 실거래 전략 설정
│   │
│   ├── recommend*.py             추천 HTML 생성
│   ├── strategy*.html            전략 설명 UI
│   │
│   └── research/                 연구/실험 (재현에 불필요, 30 py + 3 md)
│
└── trade/                        ★ 실거래 코드 (서버 배포용)
    ├── auto_trade_upbit.py       코인 현물 자동매매 (업빗)
    ├── auto_trade_binance.py     선물 자동매매 (바이낸스)
    ├── auto_trade_kis.py         주식 자동매매 (한국투자증권)
    ├── executor_*.py             새 아키텍처 executor (코인/주식)
    ├── api_server.py             Flask API 서버
    └── schema.py                 상태 파일 스키마
```

## 7. 주의

- `data/historical_universe.json`은 재현성 핵심 입력이다.
- 선물은 `1h` 원본을 기준으로 하고 `4h`는 항상 `1h`에서 리샘플링한다.
- 주식은 `stock_cache` 우선, 없으면 Yahoo fallback 구조다.

## 한 줄 요약

처음 받으면:

1. `check_data_freshness.py`
2. `refresh_backtest_data.py`
3. `run_current_*_backtest.py`

이 순서로 보면 된다.
