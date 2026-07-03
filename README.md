# MoneyFlow — 현재 운영: 주식 V25 / 현물코인 V24 / 선물 V25

3자산 포트폴리오 자동매매 시스템. 주식 + 현물코인 + 바이낸스 선물을 통합 관리한다.

> 전략은 이름이 아니라 구현이다. 버전 표기가 stale일 수 있으므로 실제 파라미터·실행순서를 기준으로 판단한다 (자세한 원칙은 [`CLAUDE.md`](./CLAUDE.md)).

## 전략 요약

| 자산 | 버전 | 거래소 | 핵심 전략 | 단독 백테스트 |
|------|------|--------|-----------|--------------|
| 주식 | V25 (2026-05-29) | 한국투자증권 | R7 ETF, EEM SMA200±0.5% 카나리, Z-score(가중Mom+Sharpe126) → 3-mom 필터 → Top3 cap 1/3 + 7% Cash, 3트랜치 | window rank-sum 채택 (C안 avg_rank 1.246) |
| 현물코인 | V24 (2026-04-30) | 업비트 | D_SMA42 단일, n_snap=7 snap_int=217 drift=0.10, BTC SMA42±1.5% 카나리 | Cal 4.63, CAGR +82%, MDD -18% (5.4yr) |
| 선물 | V25 (2026-05-28) | 바이낸스 | D_SMA42 sleeve, 동적 per-coin L=min(BTC_cap,K2), CROSSED 마진 | Cal 8.12, CAGR 312%, MDD -38.3% (5.6yr) |

### 자산배분

- 주식 60% / 업비트(현물) 25% / 바이낸스(선물) 15%
- per-sleeve cash buffer: 주식 7% / spot 1% / fut 1% (총 cash ≈ 5%)
- 자산간 리밸런싱: 자동 송금·자동 cap 폐지. 트리거 발화 시 텔레그램 알림만 전송, 사용자가 수동 송금.
  - T1: half_turnover(sum|cur_w − tgt_w|/2) ≥ 20pp → 전체 리셋
  - T3U_can: sleeve 상대 미달 ≥ 20% AND 해당 sleeve 카나리 ON → 탑업
- 3자산 합성 백테스트(60/25/15 + T1+T3U): Cal 5.72 / CAGR 106% / MDD -18.6%

---

## 빠른 시작

### 1. 의존성 설치

```bash
pip install -r requirements.txt
```

백테스트만: `numpy`, `pandas`, `yfinance`, `requests`
실거래 추가: `python-binance`, `pyupbit`, `flask`, `pykis`

### 2. 데이터 확인 및 갱신

```bash
python3 strategies/cap_defend/check_data_freshness.py
python3 strategies/cap_defend/refresh_backtest_data.py --target stock
python3 strategies/cap_defend/refresh_backtest_data.py --target coin
python3 strategies/cap_defend/refresh_backtest_data.py --target futures
```

### 3. 백테스트 실행

```bash
# 주식 V25 — 전략 순수함수는 stock_strategy_v25.py
python3 strategies/cap_defend/run_current_stock_backtest.py

# 코인현물 V24 — unified_backtest.py(asset_type='spot')가 현행 엔진.
#   run_current_coin_v20_backtest.py 는 coin_live_engine.py 의 라이브 MEMBERS 를 import 하므로
#   파일명은 v20 이지만 현재 V24 정의로 실행됨.
python3 strategies/cap_defend/run_current_coin_v20_backtest.py

# 선물 V25 — CROSS 청산모델 + 동적 L
python3 strategies/cap_defend/backtest_futures_v25.py
```

### 4. 연구 파이프라인

```bash
# research 디렉터리에 그리드/robustness/ablation 스크립트가 있음
python3 strategies/cap_defend/research/run_subperiod_ranksum.py  # window rank-sum
python3 strategies/cap_defend/research/run_block_bootstrap.py    # block bootstrap stress
```

과거 V21 채택 근거·연구 기록은 [`V21_HISTORY.md`](./V21_HISTORY.md) 참조 (역사적 문서).

---

## 전략 상세

### 주식 V25

```
유니버스: SPY, QQQ, VEA, EEM, GLD, PDBC, VNQ (R7 — R7B의 EWJ를 VNQ로 교체)
카나리:   EEM SMA200 + 0.5% hysteresis (dead-zone: 진입 >SMA×1.005, 퇴출 <SMA×0.995)
선정:     Z-score(가중Mom + Sharpe126) 랭킹 → 3-mom(30/72/230) 필터 → Top3
          · 가중Mom = 0.5×ret63 + 0.3×ret126 + 0.2×ret252
비중:     Top3 cap=1/3 + Cash 7%
방어:     IEF, BIL, BNDX, GLD, PDBC (고정 방어자산)
트랜치:   3개 (SNAP_PERIOD=69, STAGGER=23, N_SNAPS=3)
드리프트: threshold 0.05. 종목교체 = 앵커일(69일) OR drift 5pp 발화일 (fresh 3-mom 재선정)
가드:     없음 (EEM 카나리 + 3-mom 필터 + 분산 = 유일 방어)
가격가드: 야후 지연 시 KIS 일봉 1차 보강 → 공통최신일 정렬(≤3영업일) 폴백 → 초과 SKIP
거래비용: 0.1% (편도, BT)
Cron:     평일 23:35 KST (executor_stock.py 가 직접 전략 계산)
```

### 현물코인 V24 (D_SMA42 단일)

```
sleeve:   D_SMA42 단일 (ENSEMBLE_WEIGHTS = {'D_SMA42': 1.0})
스냅:     snap_interval_bars=217, n_snapshots=7 (217 = 7×31, prime stagger 31)
카나리:   BTC SMA42 ±1.5% dead-zone (진입 >SMA×1.015, 퇴출 <SMA×0.985)
헬스:     mom2vol (vol_cap 5%, vol_lookback 90d)
유니버스: CoinGecko Top40 ∩ Binance spot ∩ Upbit KRW ∩ 253d+ ∩ 거래대금 조건
비중:     universe_size=3, cap=1/3
드리프트: drift_threshold=0.10 (자본금 기준 half-turnover). 발화 시 refill v2
가드:     없음. Upbit warning/delisting 코인은 즉시 제외
거래비용: 0.04% (BT), 실매매는 Upbit 실수수료
Cron:     매일 09:05 KST (일 1회)
```

### 선물 V25 (D_SMA42 + 동적 per-coin L + CROSS)

```
sleeve:   D_SMA42 단일 (주식/코인과 동일 sleeve 정의)
스냅:     snap_interval_bars=95, n_snapshots=5, drift_threshold=0.03
카나리:   canary_hyst=0.015, universe_size=3, cap=1/3, health=mom2vol
레버리지: 동적 per-coin L = min(BTC_cap, K2_percoin), Lmin=2 / Lmid=3 / Lmax=4
          · BTC_cap: BTC/SMA42 >1.05→L4, >1.015→L3, else L2
          · K2_percoin: close/SMA7 >1.075→L4, >1.025→L3, else L2
마진모드: CROSSED (V24 ISOLATED 에서 전환)
스탑:     없음 (STOP_PCT=0). 캐시가드 없음. 가드 없음 (분산 + 동적 L 이 유일 방어)
거래비용: 0.04% (바이낸스 maker)
Cron:     매일 09:05 KST (일 1회)
```

---

## 저장소 구조

```
MoneyFlow/
├── README.md                          ← 지금 보고 있는 문서
├── CLAUDE.md                          프로젝트 규칙 (AI 어시스턴트용, 전략 정의 정본)
├── requirements.txt                   pip 의존성
├── history.md                         결정 로그 (append-only ADR)
│
├── data/                              가격 데이터
│   ├── *.csv                          Yahoo Finance 일봉 (코인+주식+ETF)
│   ├── futures/                       바이낸스 선물 봉 데이터
│   ├── historical_universe.json       월별 시총 유니버스 (생존편향 방지)
│   └── universe_cache.json            유니버스 캐시
│
├── config/                            설정
│   ├── settings.py                    공통 설정
│   └── upbit.example.py               API 키 템플릿
│
├── strategies/cap_defend/             ★ 전략 핵심 디렉터리
│   │
│   │── 엔진 / 전략 모듈 ──────────────
│   ├── unified_backtest.py            V24 spot/fut 백테스트 엔진
│   ├── backtest_futures_v25.py        선물 V25 BT 엔진 (CROSS 청산 + 동적 L)
│   ├── stock_strategy_v25.py          주식 V25 순수 전략 함수
│   ├── futures_ensemble_engine.py     선물 앙상블 실행 엔진
│   ├── futures_live_config.py         선물 V25 실거래 설정
│   ├── stock_engine.py                주식 백테스트 엔진
│   │
│   │── 백테스트 진입점 ───────────────
│   ├── run_current_stock_backtest.py  주식 백테스트 실행
│   ├── run_current_coin_v20_backtest.py 코인 백테스트 (라이브 MEMBERS import → V24)
│   ├── run_current_futures_backtest.py 선물 백테스트 실행
│   │
│   │── 데이터 관리 ───────────────────
│   ├── check_data_freshness.py        데이터 최신성 확인
│   ├── refresh_backtest_data.py       데이터 갱신 (stock/coin/futures)
│   ├── download_futures_data.py       선물 봉 데이터 다운로드
│   │
│   │── 운영/리포트 ───────────────────
│   ├── recommend.py                   공개 추천 HTML 생성
│   ├── recommend_personal.py          개인 대시보드 (자산배분 + 신호 + 알림)
│   │
│   │── 문서 ──────────────────────────
│   ├── README.md                      디렉터리 안내
│   ├── STRATEGY_EVOLUTION.md          V12→V25 전략 진화 기록
│   ├── repo_backtest_guide.md         통합 백테스트 재현 가이드
│   ├── stock_backtest_howto.md        주식 백테스트 설명
│   ├── futures_backtest_howto.md      선물 백테스트 설명
│   ├── futures_strategy_final.md      선물 최종 전략 명세
│   │
│   ├── legacy/                        레거시 엔진/백테스트 (참조용)
│   └── research/                      연구/실험 파일 (재현에 불필요)
│
├── trade/                             ★ 실거래 코드 (서버 배포)
│   ├── coin_live_engine.py            V24 코인 앙상블 엔진
│   ├── executor_coin.py               코인 executor (V24, 업비트, 09:05)
│   ├── executor_stock.py              주식 executor (V25, 한국투자증권, 23:35)
│   ├── auto_trade_binance.py          선물 자동매매 (V25, 바이낸스, 09:05)
│   ├── v24_shadow_today.py            코인 shadow 검증 (unified_backtest 호출)
│   ├── api_server.py                  Flask API 서버 (자산 조회)
│   ├── ops/                           서버 크론 스크립트 / 운영 유틸
│   ├── config.py                      서버 설정 (API 키, 비밀번호 등)
│   └── config.example.py              설정 템플릿
│
├── V25_OPERATION_MANUAL.md            현행 운영 매뉴얼 (선물 V25 중심)
└── SERVER_OPS.md                      서버 매핑/cron/헬스체크/복구 (운영 정본)
```

---

## 서버 운영

### 서버 정보

- Oracle Cloud VM (IP/접속 정보는 개인 운영 매뉴얼 참조, 공개 금지)
- 포트: serve.py (8080), api_server (5000)

### Cron 스케줄

| 시간 (KST) | 작업 |
|------|------|
| 09:05 매일 | `executor_coin.py` — 코인현물 V24 (1D, 일 1회) |
| 09:05 매일 | `auto_trade_binance.py --trade` — 선물 V25 (1D, 일 1회) |
| 09:15 매일 | `recommend.py` + `recommend_personal.py` — HTML 생성 + 자산배분 트리거 체크 + 텔레그램 |
| 23:35 평일 | `executor_stock.py` — 주식 V25 자동매매 (직접 전략 계산) |
| */5 | `watchdog_serve.sh` — 서버 생존 체크 |

상세 매핑·복구 절차는 [`SERVER_OPS.md`](./SERVER_OPS.md) 참조.

### 텔레그램 알림

- 봇 토큰/chat_id는 개인 설정 파일/환경변수에서 로드
- 리밸런싱 체결, 자산간 송금 트리거(T1/T3U), 카나리 플립 시 알림

---

## 핵심 설계 원칙

1. Look-Ahead Bias 금지: 시그널은 t-1 종가, 체결은 t일 가격
2. 매일 루프: 월간 전략이라도 매일 상태 점검 (가드 없음 — canary + drift 만). 종목교체는 앵커일에만 (예외: V25 주식은 drift 5pp 발화일에도 교체)
3. 과적합 방지: window rank-sum(홀드아웃 대신), 파라미터 plateau 확인, 다기간 일관성
4. 상태 관리: `trade_state.json`, `signal_state.json`, `kis_trade_state.json` 은 전략 상태 (단순 캐시 아님)
5. Single Source of Truth: 전략 변경 시 백테스트/실매매/추천/매뉴얼/CLAUDE.md 동시 수정

---

## 버전 이력

| 버전 | 날짜 | 주요 변경 |
|------|------|-----------|
| **V25** | **2026-05-29** | **주식: R7(VNQ 교체), EEM SMA200±0.5%, Z-score+3-mom 필터, 3트랜치 cap+Cash (현재 운영)** |
| **V25** | **2026-05-28** | **선물: 동적 per-coin L(BTC_cap+K2), CROSSED 마진, snap=95 n=5 (현재 운영)** |
| **V24** | **2026-04-30** | **모든 자산 1D 단일 + drift trigger. 코인현물 D_SMA42 n=7 snap=217 (현물 현재 운영)** |
| V22 | 2026-04-27 | 코인/선물 1D+4h 2멤버 EW, 주식 snap 3트랜치 stagger |
| V21 | 2026-04-17 | 코인 D봉 3멤버 1/3 EW, 선물 L3 3전략 고정 3x |
| V20 | 2026-04-13 | 코인 D_SMA50 + 4h_SMA240 50:50 EW |
| V17 | 2026-03 | 주식 Z-score Top3(Sh252) + VT Crash |
| V15 | 2026-03 | 주식 R7(+VNQ), Zscore4(Sh63) |
| V12 | 2026-01 | 초기 버전 |

전체 변경 근거와 폐기된 아이디어는 [`strategies/cap_defend/STRATEGY_EVOLUTION.md`](strategies/cap_defend/STRATEGY_EVOLUTION.md),
운영 절차/롤백은 [`V25_OPERATION_MANUAL.md`](./V25_OPERATION_MANUAL.md) 참조.

## 한 줄 요약

처음 받으면: `check_data_freshness.py` → `refresh_backtest_data.py` → 각 자산 백테스트 순서로 실행.

## 문서 인덱스

- [`CLAUDE.md`](./CLAUDE.md) — 전략 정의 정본 + 프로젝트 규칙 (변경 시 최우선 동기화 대상)
- [`V25_OPERATION_MANUAL.md`](./V25_OPERATION_MANUAL.md) — 현행 운영 스펙, 전환 절차, 롤백
- [`SERVER_OPS.md`](./SERVER_OPS.md) — 서버 파일 매핑, cron, 헬스체크, 복구 절차
- [`strategies/cap_defend/STRATEGY_EVOLUTION.md`](./strategies/cap_defend/STRATEGY_EVOLUTION.md) — V12~V25 진화 요약
- [`strategies/cap_defend/repo_backtest_guide.md`](./strategies/cap_defend/repo_backtest_guide.md) — 통합 백테스트 재현 가이드
- [`history.md`](./history.md) — 결정 로그 (append-only ADR)
- [`V21_HISTORY.md`](./V21_HISTORY.md) / [`V21_OPERATION_MANUAL.md`](./V21_OPERATION_MANUAL.md) — 역사적 문서 (V21 시절)
