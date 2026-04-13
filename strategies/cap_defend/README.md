# Cap Defend 전략 디렉터리

주식/현물코인/선물 전략의 백테스트 엔진, 실거래 설정, 운영 도구를 포함한다.
현재 운영 기준으로 유지하는 전용 백테스트 진입점은 주식 V17, 코인 V20, 선물 d005 세 개다.

## 바로 재현하기

### 통합 가이드

- [repo_backtest_guide.md](./repo_backtest_guide.md) — 전체 재현 절차

### 데이터 준비

```bash
python3 check_data_freshness.py          # 최신성 확인
python3 refresh_backtest_data.py --target stock    # 주식 갱신
python3 refresh_backtest_data.py --target coin     # 코인 갱신
python3 refresh_backtest_data.py --target futures  # 선물 갱신
```

### 백테스트 실행

```bash
python3 run_current_stock_backtest.py     # 주식 V17
python3 run_current_coin_v20_backtest.py  # 코인 V20
python3 run_current_futures_backtest.py   # 선물 d005
```

---

## 파일 분류

### 엔진 (핵심 로직)

| 파일 | 설명 |
|------|------|
| `stock_engine.py` | 주식 백테스트 엔진 (delta 기반) |
| `backtest_futures_full.py` | 선물 백테스트 엔진 (D/4h/2h/1h/30m/15m) |
| `futures_ensemble_engine.py` | 선물 앙상블 실행 엔진 (다전략 합산) |
| `futures_live_config.py` | 현재 실거래 선물 설정 (d005 4전략) |

### 백테스트 진입점

| 파일 | 설명 |
|------|------|
| `run_current_coin_v20_backtest.py` | 코인 V20 전용 백테스트 |
| `run_current_stock_backtest.py` | 주식 V17 백테스트 |
| `run_current_futures_backtest.py` | 선물 d005 백테스트 |

### 데이터 관리

| 파일 | 설명 |
|------|------|
| `check_data_freshness.py` | 가격 데이터 최신성 확인 |
| `refresh_backtest_data.py` | Yahoo/바이낸스 데이터 갱신 |
| `download_futures_data.py` | 바이낸스 선물 봉 데이터 다운로드 |

### 운영/리포트

| 파일 | 설명 |
|------|------|
| `recommend.py` | 공개 추천 HTML 생성 |
| `recommend_personal.py` | 개인 대시보드 (3자산 배분 모니터 + 텔레그램 알림) |
| `serve.py` | 정적 파일 서버 (port 8080) |
| `strategy.html` | 전략 요약 웹페이지 |
| `strategy_guide.html` | 상세 전략 가이드 웹페이지 |

### 문서

| 파일 | 설명 |
|------|------|
| `STRATEGY_EVOLUTION.md` | V12→V20 전략 진화 기록 (변경 근거, 폐기 아이디어) |
| `repo_backtest_guide.md` | 통합 백테스트 재현 가이드 |
| `stock_backtest_howto.md` | 주식 백테스트 상세 설명 |
| `futures_backtest_howto.md` | 선물 백테스트 상세 설명 |
| `futures_strategy_final.md` | 선물 최종 전략 명세 |

### 레거시

| 파일 | 설명 |
|------|------|
| [`legacy/`](./legacy/) | V18 계열 코인/통합 백테스트와 과거 유틸 모음 |

### 연구 파일

[research/](./research/) — 최적화 스크립트, 실험 결과 파일. 공식 재현에는 불필요.

---

## 전략 아키텍처

### 코인 엔진 흐름 (V20)

```
멤버1 D_SMA50:
  1. BTC vs SMA50 카나리
  2. Mom30/90 + Vol90d 필터
  3. Top5 + 33% cap
  4. 30봉 기준 3-snapshot stagger
  5. 극단갭 -15% 코인 제외

멤버2 4h_SMA240:
  1. BTC vs SMA240 카나리
  2. Mom30/120 + Vol90d 필터
  3. Top5 + 33% cap
  4. 60봉 기준 3-snapshot stagger
  5. 극단갭 -10% 코인 제외

최종:
  1. 두 멤버 50:50 EW 합산
  2. Cash buffer 2%
  3. 4h 기준 현물 리밸런싱
```

### 주식 엔진 흐름 (V17)

```
매일 루프:
  1. VT crash 체크 (-3% → 현금화)
  2. EEM 카나리 (SMA200, 0.5% hyst)
  3. 앵커일(1/8/15/22) → Z-score 선정, 방어 전환
  4. delta 기반 리밸런싱
```

### 선물 엔진 흐름 (d005)

```
2h/4h 주기:
  1. 4전략 각각 시그널 계산 (SMA + Mom + Vol)
  2. EW 합산 → 단일 비중
  3. cap_mom_blend → 레버리지 결정 (3/4/5x)
  4. prev_close 스탑 + cash_guard 체크
  5. 포지션 조정
```

### 자산배분 (V20, 비율은 V19 확정치 유지)

```
매일 09:15 cron:
  1. 주식(한투) + 현물코인(업비트) + 선물(바이낸스) 잔고 조회
  2. 실제 비중 계산 → 목표(60/25/15) 대비 편차
  3. 최대 편차 >= 8%p → 텔레그램 알림 (리밸런싱 필요)
  4. 대시보드에 현재 비중 + 리밸런싱 금액 표시
```

---

## 주의사항

- `run_current_coin_v20_backtest.py`는 V20 라이브 로직을 재현하기 위한 전용 러너다
- `data/historical_universe.json`은 V20 백테스트에서 월별 Top40 유니버스 입력으로 사용한다
- 선물은 `1h` 원본 기준, `4h`/`2h`는 리샘플링
- 현금 키: 백테스트 `CASH`, 실매매/리포트 `Cash` — 혼동 주의
- 상태파일(`trade_state.json`, `signal_state.json`)은 전략 상태이므로 함부로 삭제하지 않는다
