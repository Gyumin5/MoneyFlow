# Cap Defend 전략 디렉터리

주식/현물코인/선물 전략의 백테스트 엔진, 실거래 설정, 운영 도구를 포함한다.
현재 운영 = 주식 V25 / 코인현물 V24 / 선물 V25. 전략 정의 정본은 루트 [`CLAUDE.md`](../../CLAUDE.md).

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
python3 run_current_stock_backtest.py       # 주식 (전략 순수함수 stock_strategy_v25.py)
python3 run_current_coin_v20_backtest.py    # 코인현물 — 라이브 MEMBERS import → 현재 V24 정의로 실행 (파일명만 v20)
python3 backtest_futures_v25.py             # 선물 V25 (CROSS 청산 + 동적 L)
```

코인현물 V24 정합 엔진은 `unified_backtest.py`(asset_type='spot'), 선물 V25 는 `backtest_futures_v25.py`.

---

## 파일 분류

### 엔진 (핵심 로직)

| 파일 | 설명 |
|------|------|
| `unified_backtest.py` | 코인 V24 spot/fut 백테스트 엔진 (라이브 정합) |
| `backtest_futures_v25.py` | 선물 V25 BT 엔진 (CROSS 청산 + 동적 L, build_K2_signal) |
| `stock_strategy_v25.py` | 주식 V25 순수 전략 함수 (compute_offense/defense/eem_canary) |
| `stock_engine.py` | 주식 백테스트 엔진 (delta 기반) |
| `futures_ensemble_engine.py` | 선물 앙상블 실행 엔진 |
| `futures_live_config.py` | 선물 V25 실거래 설정 (동적 L, K2/BTC_CAP, MARGIN_TYPE=CROSSED) |

### 백테스트 진입점

| 파일 | 설명 |
|------|------|
| `run_current_coin_v20_backtest.py` | 코인현물 백테스트 (라이브 MEMBERS import → 현재 V24) |
| `backtest_futures_v25.py` | 선물 V25 백테스트 |
| `run_current_stock_backtest.py` | 주식 백테스트 |

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
| `STRATEGY_EVOLUTION.md` | V12→V25 전략 진화 기록 (변경 근거, 폐기 아이디어) |
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

### 코인현물 엔진 흐름 (V24)

```
D_SMA42 단일 sleeve, 매일 09:05:
  1. BTC vs SMA42 ±1.5% dead-zone 카나리 (stateful)
  2. mom2vol 헬스 (Vol90d cap 5%)
  3. Top3 + 1/3 cap (greedy 흡수)
  4. snap_interval=217봉 × n=7 snapshot merge → 단일 target
  5. drift(half_turnover ≥ 0.10) 발화 시 refill v2, 아니면 앵커일에만 교체
  6. Cash buffer 1%
```

### 주식 엔진 흐름 (V25)

```
평일 23:35 (executor_stock.py 직접 계산):
  1. 가격일자 가드 (야후 지연 → KIS 일봉 보강 → 공통일자 정렬 ≤3영업일 → 초과 SKIP)
  2. EEM 카나리 (SMA200, 0.5% hyst dead-zone)
  3. Z-score(가중Mom + Sharpe126) 랭킹 → 3-mom(30/72/230) 필터 → Top3 cap 1/3 + 7% Cash
  4. 3트랜치 snap merge (SNAP_PERIOD=69, STAGGER=23, N_SNAPS=3)
  5. drift(≥0.05) 발화일 OR 앵커일 → fresh 3-mom 재선정 교체, 아니면 보유 유지
  6. delta 기반 리밸런싱 (가드 없음)
```

### 선물 엔진 흐름 (V25)

```
D_SMA42 sleeve, 매일 09:05:
  1. BTC vs SMA42 ±1.5% 카나리
  2. mom2vol 헬스 → Top3 + 1/3 cap
  3. snap_interval=95봉 × n=5 snapshot merge, drift ≥ 0.03
  4. 동적 per-coin L = min(BTC_cap, K2_percoin), Lmin=2/mid=3/max=4
  5. 마진모드 CROSSED, 스탑 없음, 캐시가드 없음
```

### 자산배분 (V24/V25, 60/25/15)

```
매일 09:15 cron (recommend):
  1. 주식(한투) + 현물코인(업비트) + 선물(바이낸스) 잔고 조회
  2. 실제 비중 계산 → 목표(60/25/15, per-sleeve buffer 7/1/1%) 대비 편차
  3. T1(half_turnover ≥ 20pp) OR T3U_can(sleeve 상대미달 ≥20% AND 카나리 ON) → 텔레그램 알림
  4. 자동 송금·자동 cap 폐지 — 알림만, 사용자 수동 송금
```

---

## 주의사항

- `run_current_coin_v20_backtest.py`는 라이브 로직(현재 V24)을 재현하기 위한 전용 러너다 (파일명만 v20)
- `data/historical_universe.json`은 코인 백테스트에서 월별 Top40 유니버스 입력으로 사용한다 (생존편향 방지)
- 현금 키: 백테스트 `CASH`, 실매매/리포트 `Cash` — 혼동 주의
- 상태파일(`trade_state.json`, `kis_trade_state.json`, `signal_state.json`)은 전략 상태이므로 함부로 삭제하지 않는다
