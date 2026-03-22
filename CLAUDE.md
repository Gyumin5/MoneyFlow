# Cap Defend 프로젝트 규칙

## 백테스트 필수 규칙

### Look-Ahead Bias 금지

- **시그널: t-1일 종가 기준, 체결: t일 가격** — 반드시 분리
- 실매매: 09:20에 전일 종가 판단 → 당일 매매
- 코인: `sig_date = prev_date` (prev_date가 None이면 첫날 예외)
- 주식: `sig_date = prev_trading_date`
- crash/DD/blacklist/카나리 모두 동일 규칙

### 일간 리밸런싱

- 코인/주식 모두 **매일** 루프를 돌며 체크 (월간이 아님)
- 카나리, DD exit, blacklist, crash는 매일 체크
- 종목 교체(snapshot)는 앵커일(1/10/19)에만
- 월간 리밸런싱 = "매일 체크하되 앵커일에만 종목 변경"

### 엔진 선택

- **코인: 구 엔진(3-snapshot 합성)** — 실매매(auto_trade_upbit.py)와 동일
- V15 엔진(독립 트랜치)은 실매매와 불일치 → 사용 금지
- 3-snapshot: 비중을 합산(merge)하여 단일 포트폴리오로 리밸런싱

### 파라미터 키

- 코인 엔진(coin_engine.py)이 인식하는 키만 사용:
  - 카나리: K1, K8 등 (K8: vote_smas 기반)
  - 헬스: H1, HK 등
  - 선정: baseline(시총순), S1~S10
  - 비중: baseline(EW), WC(20%Cap), W1(rank decay), W2(InvVol)
  - 리스크: G5(crash breaker)
- `selection='mcap'`, `weighting='ew'` 같은 비표준 키 → 엔진 fallback → 의도와 다른 결과
- B() 기본값 주의: sma_period=150, canary_band=0 → V14+ 에서는 sma_period=60, canary_band=1.0 명시 필요

### 거래비용

- 코인: 0.4% 편도 (수수료 + 슬리피지), 업비트 실제 수수료 ~0.05%
- 주식: 0.1% 편도

### 과적합 방지

- 10-anchor 평균 사용 (코인: (1,10,19) ~ (10,19,28))
- σ(Sharpe) 확인 — 0.1 이하가 robust
- 파라미터 인접값에서 성과 유사해야 (broad plateau)
- 단일 기간 최적이 아닌 다기간(2018~/2019~/2021~) 일관성 확인

## 코드 수정 시 규칙

### recommend는 항상 두 개 수정

- `recommend.py` + `recommend_personal.py` 동시 작업
- 하나만 수정하면 안 됨

### 서버 배포 순서

1. 로컬 수정 + 테스트
2. `scp`로 서버 배포
3. 서버에서 실행 확인
4. API 서버 변경 시 재시작 필요
5. git commit + push

### 서버 파일 매핑

- `strategies/cap_defend/recommend.py` → `/home/ubuntu/recommend.py`
- `strategies/cap_defend/recommend_personal.py` → `/home/ubuntu/recommend_personal.py`
- `trade/auto_trade_upbit.py` → `/home/ubuntu/auto_trade_upbit.py`
- `trade/api_server.py` → `/home/ubuntu/trade_api_server.py`

### 모니터 주의사항

- 모니터(auto_trade --monitor)는 run_trade.sh의 flock으로 중복 방지
- 모니터 내부에 별도 flock 넣으면 자기 차단 → 절대 금지
- 모니터 카나리/crash 비교는 USD 기준 (Upbit KRW-BTC / KRW-USDT)
- DD exit은 CSV 60일 rolling peak (KRW 유지)

## 전략 변경 시 확인사항

- 백테스트 코드(`backtest_official.py`)와 실매매 코드(`auto_trade_upbit.py`) 동기화
- recommend HTML에 변경 반영
- V17_OPERATION_MANUAL.md 업데이트
- 메모리(MEMORY.md) 업데이트
