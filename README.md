# MoneyFlow — V19

3자산 포트폴리오 자동매매 시스템. 주식 + 현물코인 + 바이낸스 선물을 통합 관리한다.

## 전략 요약

| 자산 | 버전 | 거래소 | 핵심 전략 | 백테스트 성과 |
|------|------|--------|-----------|--------------|
| 주식 | V17 | 한국투자증권 | 7종 ETF, EEM 카나리, Z-score Top3, 4트랜치 | Sharpe 1.26, CAGR +13.3%, MDD -11.4% |
| 현물코인 | V18 | 업비트 | BTC SMA50 카나리, 시총순 Top5, Greedy Absorption | Sharpe 1.60, CAGR +67.0%, MDD -26.8% |
| 선물 | d005 | 바이낸스 | 4전략 앙상블, 5x 동적레버리지 | Sharpe 2.08, CAGR +227%, MDD -34% |

### 자산배분

- **주식 60% / 현물코인 25% / 선물 15%**
- 밴드 리밸런싱: 편차 ±8%p 초과 시만 전체 복원 (매일 자동체크 + 텔레그램 알림)
- 통합 포트폴리오: Sharpe 2.12, CAGR +39%, MDD -12.2%

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
python3 strategies/cap_defend/run_current_stock_backtest.py     # 주식 V17
python3 strategies/cap_defend/run_current_coin_backtest.py      # 코인 V18
python3 strategies/cap_defend/run_current_futures_backtest.py   # 선물 d005
```

---

## 전략 상세

### 주식 V17

```
유니버스: SPY, QQQ, VEA, EEM, GLD, PDBC, VNQ (7종 ETF)
카나리:   EEM > SMA(200), 0.5% hysteresis
선정:     Z-score Top 3 (12M momentum + Sharpe 252d)
비중:     EW (33%씩)
방어:     Top 3 by 6M return (IEF, BIL, BNDX, GLD, PDBC)
Crash:    VT -3% daily → 최소 3일 + VT>SMA10 회복 시 재진입
트랜치:   4개 (Day 1/8/15/22, 타이밍 리스크 분산)
거래비용: 0.2% (보수적)
```

### 현물코인 V18

```
유니버스: 시총 Top 40 (월초 historical_universe.json 기준)
카나리:   BTC > SMA(50), 1.5% hysteresis
헬스:     Mom(30)>0 AND Mom(90)>0 AND Vol(90)<=5%
선정:     시총순 Top 5 → Greedy Absorption
비중:     EW + 33% Cap
Crash:    BTC daily -10% → 3일 현금
DD Exit:  60일 peak -25% → 매도
Blacklist: -15% daily → 7일 제외
스냅샷:   3개 (Day 1/11/21), PFD5, Drift 10%
거래비용: 0.4% (편도)
```

### 선물 d005

```
앙상블:   4전략 EW (25%씩)
  1) 4h_d005:  SMA240, Mom20/720, daily vol 5%, snap60
  2) 2h_S240:  SMA240, Mom20/720, bar vol 60%, snap120
  3) 2h_S120:  SMA120, Mom20/720, bar vol 60%, snap120
  4) 4h_M20:   SMA240, Mom20/120, bar vol 60%, snap21
공통:     canary_hyst=0.015, n_snapshots=3, health=mom2vol
레버리지: cap_mom_blend_543_cash (3/4/5x 동적)
스탑:     prev_close -15%, cash_guard(CASH>=34%)
거래비용: 0.04% (바이낸스 maker)
```

---

## 저장소 구조

```
MoneyFlow/
├── README.md                          ← 지금 보고 있는 문서
├── CLAUDE.md                          프로젝트 규칙 (AI 어시스턴트용)
├── requirements.txt                   pip 의존성
├── history.md                         작업 이력
│
├── data/                              가격 데이터
│   ├── *.csv                          Yahoo Finance 일봉 (코인+주식+ETF)
│   ├── futures/                       바이낸스 선물 봉 데이터 (1h/4h/15m/30m)
│   ├── historical_universe.json       월별 시총 유니버스 (재현 핵심)
│   └── universe_cache.json            유니버스 캐시
│
├── config/                            설정
│   ├── settings.py                    공통 설정
│   └── upbit.example.py               API 키 템플릿
│
├── scripts/                           서버 크론 스크립트
│   ├── run_recommend.sh               추천 HTML 생성
│   └── run_trade.sh                   자동매매 실행
│
├── strategies/cap_defend/             ★ 전략 핵심 디렉터리
│   │
│   │── 엔진 ──────────────────────
│   ├── coin_engine.py                 현물 코인 백테스트 엔진
│   ├── coin_helpers.py                코인 헬퍼 (파라미터, merge 등)
│   ├── coin_dd_exit.py                코인 DD exit 로직
│   ├── stock_engine.py                주식 백테스트 엔진
│   ├── backtest_futures_full.py       선물 백테스트 엔진
│   ├── futures_ensemble_engine.py     선물 앙상블 실행 엔진
│   ├── futures_live_config.py         현재 실거래 선물 설정 (d005)
│   │
│   │── 백테스트 진입점 ───────────
│   ├── backtest_official.py           V12~V18 코인+주식 공식 백테스트
│   ├── run_current_coin_backtest.py   코인 V18 백테스트 실행
│   ├── run_current_stock_backtest.py  주식 V17 백테스트 실행
│   ├── run_current_futures_backtest.py 선물 d005 백테스트 실행
│   │
│   │── 데이터 관리 ───────────────
│   ├── check_data_freshness.py        데이터 최신성 확인
│   ├── refresh_backtest_data.py       데이터 갱신 (stock/coin/futures)
│   ├── download_futures_data.py       선물 봉 데이터 다운로드
│   │
│   │── 운영/리포트 ───────────────
│   ├── recommend.py                   공개 추천 HTML 생성
│   ├── recommend_personal.py          개인 대시보드 (자산배분 + 신호 + 알림)
│   ├── daily_history.py               히스토리 빌더
│   ├── serve.py                       정적 파일 서버 (port 8080)
│   ├── strategy.html                  전략 요약 페이지
│   ├── strategy_guide.html            상세 전략 가이드
│   │
│   │── 문서 ──────────────────────
│   ├── README.md                      디렉터리 안내
│   ├── repo_backtest_guide.md         통합 백테스트 재현 가이드
│   ├── coin_backtest_howto.md         코인 백테스트 설명
│   ├── stock_backtest_howto.md        주식 백테스트 설명
│   ├── futures_backtest_howto.md      선물 백테스트 설명
│   ├── futures_strategy_final.md      선물 최종 전략 명세
│   │
│   └── research/                      연구/실험 파일 (재현에 불필요)
│       ├── README.md                  연구 파일 목록
│       ├── test_*.py                  실험 스크립트 (~25개)
│       ├── run_*.py                   최적화/분석 스크립트 (~10개)
│       └── *_results.*                실험 결과 (~30개)
│
├── trade/                             ★ 실거래 코드 (서버 배포)
│   ├── auto_trade_upbit.py            코인 현물 자동매매 (업비트)
│   ├── auto_trade_binance.py          선물 자동매매 (바이낸스)
│   ├── auto_trade_kis.py              주식 자동매매 (한국투자증권)
│   ├── executor_coin.py               코인 executor (새 아키텍처)
│   ├── executor_stock.py              주식 executor (새 아키텍처)
│   ├── api_server.py                  Flask API 서버 (자산 조회)
│   ├── schema.py                      상태 파일 스키마
│   ├── config.py                      서버 설정 (API 키, 비밀번호 등)
│   └── config.example.py              설정 템플릿
│
├── backup_20260125/                   V12 백업 (레거시)
└── V17_OPERATION_MANUAL.md            운영 매뉴얼
```

---

## 서버 운영

### 서버 정보

- Oracle Cloud VM: `[REDACTED_SERVER]`
- 포트: serve.py (8080), api_server (5000)

### Cron 스케줄

| 시간 | 작업 |
|------|------|
| 09:15 | `run_recommend.sh` — 추천 HTML 생성 + 텔레그램 알림 |
| 09:20 | `run_trade.sh` — 코인/주식 자동매매 |
| **:05, **:35 | `auto_trade_upbit.py --monitor` — 코인 모니터 |
| */5 | watchdog |
| */2h 또는 */4h | `auto_trade_binance.py` — 선물 자동매매 |

### 텔레그램 알림

- 봇: `[REDACTED_BOT]`
- Crash/카나리 플립, 자산배분 밴드 초과 시 자동 알림

---

## 핵심 설계 원칙

1. **Look-Ahead Bias 금지**: 시그널은 t-1 종가, 체결은 t일 가격
2. **매일 루프**: 월간 전략이라도 매일 상태 점검 (Crash/DD/Blacklist)
3. **과적합 방지**: 10-anchor 평균, 파라미터 plateau 확인, 다기간 일관성
4. **상태 관리**: `trade_state.json`, `signal_state.json`은 전략 상태 (단순 캐시 아님)
5. **Single Source of Truth**: 전략 변경 시 백테스트/실매매/추천/매뉴얼 동시 수정

---

## 버전 이력

| 버전 | 날짜 | 주요 변경 |
|------|------|-----------|
| V19 | 2026-04 | 선물 d005 4전략 확정 + 자산배분 60/25/15 + 밴드 8pp |
| V18 | 2026-03 | 코인: SMA50+1.5%hyst, Greedy Absorption, EW+33%Cap |
| V17 | 2026-03 | 주식: Z-score Top3(Sh252) + VT Crash |
| V16 | 2026-03 | 코인: Mom30 |
| V15 | 2026-03 | 주식: R7(+VNQ), Zscore4(Sh63) |
| V14 | 2026-02 | 코인: SMA60+hyst, DD+BL+Crash |
| V12 | 2026-01 | 초기 버전 |

## 한 줄 요약

처음 받으면: `check_data_freshness.py` → `refresh_backtest_data.py` → `run_current_*_backtest.py` 순서로 실행.
