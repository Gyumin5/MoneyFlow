# V16 Cap Defend 운영 매뉴얼

## 전략 개요

### 자산 배분

- 주식 60% / 코인 40%
- 업비트에 코인 배분만 입금 (현재 ~2.58억원)

### 코인 전략 (V16)

| 항목        | 설정                                          |
| ----------- | --------------------------------------------- |
| 카나리아    | BTC > SMA(60), 1% Hysteresis                  |
| 헬스        | Mom(30)>0, Mom(90)>0, Vol(90)≤5%              |
| 유니버스    | CoinGecko Top 40 + Upbit KRW + 거래대금 10억+ |
| 선택        | 시총순 Top 5                                  |
| 배분        | Equal Weight + 20% Cap (나머지 현금)          |
| Cash Buffer | 동적 (trade_state.json, 현재 80%)             |
| Crash       | BTC -10% → 3일 현금                           |
| DD Exit     | 60일 고점 -25% → 해당 코인 매도               |
| Blacklist   | -15% 일간 → 7일 제외                          |
| 3트랜치     | Day 1/11/21, 각 독립 리밸런싱                 |
| PFD5        | 카나리아 플립 후 5일 추가 리밸런싱            |

### 비중 계산 예시 (Cash Buffer 80%)

```
코인 비중 = 20% Cap × (1 - 0.80) = 4%
3트랜치 합산 = 4% × 3/3 = 4%
실제 투자 = 총자산 × 4%
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
├── auto_trade_upbit.py      # 매매 엔진 (--trade / --monitor / --force)
├── recommend.py             # 일반 HTML 리포트
├── recommend_personal.py    # 개인 HTML 리포트 (업비트 잔고 연동)
├── trade_api_server.py      # Flask API (Force Trade, Cash Buffer)
├── serve.py                 # 정적 파일 서버 (8080)
├── trade_state.json         # 트랜치/카나리아/pending 상태
├── signal_state.json        # 주식 카나리아/보유종목
├── watchdog_serve.sh        # 서버 생존 감시
├── run_trade.sh             # flock 래퍼 (중복 방지)
├── run_recommend.sh         # flock 래퍼
├── config.py                # API 키 (git 밖, private repo에 관리)
├── portfolio_result.html    # 일반 리포트
└── portfolio_result_gmoh.html  # 개인 리포트
```

### cron 스케줄 (KST)

```
09:15  recommend general + personal  (HTML 생성)
09:20  auto_trade --trade            (매매 판단 + 실행)
:20,:50 auto_trade --monitor         (긴급 탈출 + pending 복구)
*/5    watchdog                      (서버 생존 체크)
```

### 자동 복구

- **watchdog**: 5분마다 serve.py + trade_api_server.py 체크, 죽으면 재시작
- **@reboot**: 서버 재부팅 시 자동 시작
- **logrotate**: 주 1회, 4주 보관

---

## 매매 흐름

### --trade 모드 (09:20, 1일 1회)

1. 데이터 수집 (CoinGecko, Yahoo, Upbit)
2. 신호 계산 (카나리아, 헬스, 선택, DD, Crash, Blacklist)
3. 3트랜치 트리거 판단
4. 트리거 있으면 → 분할 매매 실행
5. 캐싱 저장 (monitor용)
6. trade_state.json 저장

### --monitor 모드 (30분마다, 24시간)

1. 긴급 탈출 체크 (pyupbit 현재가만)
   - Crash: BTC 전일 대비 -10%
   - 카나리아 OFF: BTC < SMA60×0.99
   - DD Exit: 보유코인 < peak×0.75
2. coin_peaks 신고가 갱신
3. pending 복구 (미완료 매매 재시도)
4. 없으면 즉시 종료 (~5초)

### 트리거 종류

| 트리거        | 발동 조건               | 모드            |
| ------------- | ----------------------- | --------------- |
| 앵커일        | Day 1/11/21 도달        | trade           |
| 카나리아 플립 | BTC SMA60 돌파/이탈     | trade           |
| Crash         | BTC -10%                | trade + monitor |
| DD Exit       | 보유코인 -25% from peak | trade + monitor |
| 헬스 실패     | 보유코인 건강 탈락      | trade           |
| 초기 진입     | trade_state 없음        | trade           |
| Force         | 수동 버튼               | trade           |
| Buffer 변경   | cash_buffer 변경        | trade (다음날)  |
| Pending       | 미완료 매매 잔존        | trade + monitor |

---

## 분할 매매

### 일반 매매 (앵커일/버퍼변경 등)

- 주문 전 호가 확인 → 슬리피지 0.3% 이내 금액(safe_amount) 계산
- safe_amount만큼 시장가 주문 → 7초 대기 → 반복
- 3분 타임아웃 → 미완료는 pending
- 매수 전 미체결 주문 취소
- 매도 먼저 (시총 낮은→높은), 매수 나중 (시총 높은→낮은)

### 긴급 매도 (Crash/카나리아OFF)

- 코인별 빠른 분할 (1초 간격, 2~5회)
- 매 회차마다 실제 잔고 재조회
- pending 전부 삭제 + 미체결 취소
- 재진입 안 함 (다음날 09:20에만)

### pending 관리

- 미완료 매매: {코인: {side, target_krw, filled_krw}}
- 완료 기준: 잔여 < 5만원
- monitor가 30분마다 복구 시도
- 긴급 시: pending 전부 삭제

---

## 사용자 조작

### Force Trade (즉시 매매)

1. `http://152.69.225.8:8080/portfolio_result_gmoh.html` 접속
2. "Force Trade" 버튼 클릭
3. PIN 입력 (서버 환경변수 TRADE_PIN)
4. 현재 신호 기준으로 즉시 매매

### Cash Buffer 변경

1. HTML 드롭다운에서 선택 (80%/60%/40%/20%/2%)
2. PIN 입력
3. trade_state.json 업데이트 + buffer_changed 플래그
4. 다음날 09:20에 자동 반영, 또는 Force Trade로 즉시

### 점진적 투자 확대

```
Day 1:  buffer 80% → 투자 4%  (약 1천만)
Day 3:  buffer 60% → 투자 8%  (약 2천만) + Force
Day 7:  buffer 40% → 투자 12% (약 3천만) + Force
Day 14: buffer 20% → 투자 16% (약 4천만) + Force
Day 21: buffer 2%  → 투자 20% (약 5천만) + Force
```

---

## 보안

| 항목                      | 보호                                              |
| ------------------------- | ------------------------------------------------- |
| Force Trade / Buffer 변경 | 4자리 PIN (환경변수 TRADE_PIN)                    |
| API 키                    | config.py (git 밖, private repo moneyflow-config) |
| 동시 실행                 | flock (trade/monitor/API 공유 락)                 |
| git 히스토리              | filter-repo로 비밀번호 제거 완료                  |

---

## 상태 파일

### trade_state.json

```json
{
  "tranches": {
    "1": {
      "last_anchor_month": "2026-03",
      "picks": ["TRX"],
      "weights": { "TRX": 0.04 }
    },
    "11": {
      "last_anchor_month": "2026-03",
      "picks": ["TRX"],
      "weights": { "TRX": 0.04 }
    },
    "21": {
      "last_anchor_month": "2026-03",
      "picks": ["TRX"],
      "weights": { "TRX": 0.04 }
    }
  },
  "pending_trades": {},
  "coin_risk_on": true,
  "cash_buffer": 0.8,
  "btc_prev_close": 109517000,
  "btc_sma60": 109204934,
  "coin_peaks": { "TRX": 449 }
}
```

### signal_state.json

```json
{
  "risk_on": true,
  "signal_flipped": false,
  "coin_risk_on": true,
  "stock_holdings": ["VEA", "EEM", "GLD"]
}
```

---

## 백테스트 성과 (V16, 10-anchor 평균)

| 지표   | 값     |
| ------ | ------ |
| Sharpe | 1.451  |
| CAGR   | +64.5% |
| MDD    | -33.2% |
| Calmar | 1.94   |

### 주식 60% + 코인 40% 합산

| 지표       | 값           |
| ---------- | ------------ |
| Sharpe     | 1.684        |
| CAGR       | +51.5%       |
| MDD        | -30.5%       |
| 7.5년 성과 | 1억 → 22.5억 |

---

## 트러블슈팅

| 증상                          | 원인                   | 해결                               |
| ----------------------------- | ---------------------- | ---------------------------------- |
| Force Trade "잘못된 비밀번호" | API 서버 환경변수 누락 | watchdog 5분 대기 또는 수동 재시작 |
| 매매 안 됨                    | 트리거 없음            | Force Trade 또는 앵커일 대기       |
| HTML 안 바뀜                  | recommend 미실행       | 09:15 cron 확인                    |
| API 서버 접속 불가            | 프로세스 죽음          | watchdog 5분 내 복구               |
| trade_state 손상              | 파일 삭제/오류         | 초기 진입으로 자동 복구            |
