# V17 Cap Defend 운영 매뉴얼

## 전략 개요

### 자산 배분

- 주식 60% / 코인 40%
- 매월 1회 비중 조정

### 주식 전략 (V17)

| 항목     | 설정                                                |
| -------- | --------------------------------------------------- |
| 유니버스 | SPY, QQQ, VEA, EEM, GLD, PDBC, VNQ (7종)            |
| 카나리아 | EEM > SMA(200), 0.5% Hysteresis                     |
| 선택     | Z-score Top 3 (12M momentum + Sharpe 252d)          |
| 배분     | Equal Weight (33%씩)                                |
| 방어     | Top 3 by 6M return from (IEF, BIL, BNDX, GLD, PDBC) |
| Crash    | VT -3% daily → 3일 현금                             |

**성과** (2017~, 11-anchor 평균): Sharpe 1.037, CAGR +12.1%, MDD -13.2%, Calmar 0.94

### 코인 전략 (V17)

| 항목        | 설정                                          |
| ----------- | --------------------------------------------- |
| 카나리아    | BTC > SMA(60), 1% Hysteresis                  |
| 헬스        | Mom(30)>0, Mom(90)>0, Vol(90)≤5%              |
| 유니버스    | CoinGecko Top 40 + Upbit KRW + 거래대금 10억+ |
| 선택        | 시총순 Top 5                                  |
| 배분        | Equal Weight + 20% Cap (나머지 현금)          |
| Cash Buffer | 동적 (trade_state.json)                       |
| Crash       | BTC -10% → 3일 현금                           |
| DD Exit     | 60일 고점 -25% → 해당 코인 매도               |
| Blacklist   | -15% 일간 → 7일 제외                          |
| 3트랜치     | Day 1/11/21, 각 독립 리밸런싱                 |
| PFD5        | 카나리아 플립 후 5일 추가 리밸런싱            |

---

## 수동매매 운영 규칙 (주식)

### 매일 확인 (HTML 페이지)

```
🚨 Crash/카나리 플립 → 즉시 거래
🟢 정상 → 패스
```

- VT Crash (-3%): 연 0.6회 발동 (보유 중일 때만, 카나리 OFF면 이미 방어중이라 무관)
- 카나리 플립: 연 2-4회

### 매월 1회 (월초 근처, ±며칠 상관없음)

```
1. 주식/코인 비중 60:40 맞춤
2. 주식 추천 3종목 확인
3. 전량 매도 → 3종목 33%씩 재매수 (풀 EW 리밸런싱)
   (종목이 안 바뀌었어도 비중 맞추기 위해 실행)
```

---

## 서버 구성

### 서버 정보

- Oracle Cloud VM: `152.69.225.8`
- User: `ubuntu`
- SSH: `ssh -i ~/.ssh/id_rsa ubuntu@152.69.225.8`

### 파일 구조

```
/home/ubuntu/
├── auto_trade_upbit.py      # 코인 매매 엔진 (--trade / --monitor / --force)
├── recommend.py             # 일반 HTML 리포트 (V17)
├── recommend_personal.py    # 개인 HTML 리포트 (업비트 잔고 연동, V17)
├── trade_api_server.py      # Flask API (Force Trade, Cash Buffer)
├── serve.py                 # 정적 파일 서버 (8080)
├── trade_state.json         # 코인 트랜치/카나리아/pending 상태
├── signal_state.json        # 주식 카나리아/보유종목
├── watchdog_serve.sh        # 서버 생존 감시
├── run_trade.sh             # flock 래퍼 (중복 방지)
├── run_recommend.sh         # flock 래퍼
├── config.py                # API 키 (git 밖, private repo에 관리)
├── data/                    # Yahoo 가격 CSV 캐시
├── portfolio_result.html    # 일반 리포트
└── portfolio_result_gmoh.html  # 개인 리포트
```

### cron 스케줄 (KST)

```
09:15  recommend general + personal  (HTML 생성)
09:20  auto_trade --trade            (코인 매매 판단 + 실행)
:05,:35 auto_trade --monitor         (코인 긴급 탈출 + pending 복구)
*/5    watchdog                      (서버 생존 체크)
```

---

## 코인 매매 흐름

### --trade 모드 (09:20, 1일 1회)

1. 데이터 수집 (CoinGecko, Yahoo, Upbit)
2. 신호 계산 (카나리아, 헬스, 선택, DD, Crash, Blacklist)
3. 3트랜치 트리거 판단
4. 트리거 있으면 → 분할 매매 실행
5. 캐싱 저장 (monitor용, USD 기준)
6. trade_state.json 저장
7. HOLD이어도 캐시 갱신

### --monitor 모드 (30분마다, 24시간)

1. BTC USD 가격 조회 (Upbit KRW-BTC / KRW-USDT, Binance fallback)
2. 긴급 탈출 체크
   - Crash: BTC USD vs 전일종가(USD) -10%
   - 카나리아 OFF: BTC USD < SMA60(USD) × 0.99
   - DD Exit: 보유코인 KRW 60일 rolling peak -25% (CSV 직접 조회)
3. pending 복구 (미완료 매매 재시도)
4. 없으면 즉시 종료 (~5초)

### 트리거 종류

| 트리거        | 발동 조건                   | 모드            |
| ------------- | --------------------------- | --------------- |
| 앵커일        | Day 1/11/21 도달            | trade           |
| 카나리아 플립 | BTC SMA60 돌파/이탈         | trade + monitor |
| Crash         | BTC -10%                    | trade + monitor |
| DD Exit       | 보유코인 -25% from 60d peak | trade + monitor |
| 헬스 실패     | 보유코인 건강 탈락          | trade           |
| PFD5          | 플립 후 5일                 | trade           |
| Buffer 변경   | cash_buffer API 변경        | trade           |

---

## V16 → V17 변경 사항 (2026-03-20)

### 주식 전략

- 선정: Z-score Top 4 (Sharpe 63d) → **Top 3 (Sharpe 252d)**
- Crash Breaker 추가: **VT daily -3% → 3일 현금**

### 코인 모니터

- flock 자기 차단 버그 수정
- 카나리/Crash 비교를 **USD 기준**으로 전환 (환율 왜곡 방지)
- HOLD 시에도 캐시 갱신 (stale 방지)
- DD Exit: **CSV 60일 rolling peak** 직접 조회 (누적 최고가 문제 해결)
- coin_peaks 캐시 제거
- cron: monitor **:05/:35** (trade 09:20과 15분 간격)
- API 서버: run_trade.sh 경유 (flock 일관화)
