# Cap Defend 프로젝트 규칙

## 전략 정의 원칙

### 전략은 이름이 아니라 구현이다

- `V22` 같은 버전명보다 실제 구현이 우선이다.
- 아래 중 하나라도 다르면 같은 전략으로 취급하면 안 된다:
  - 시그널 시점
  - 체결 시점
  - 앵커일
  - 카나리/헬스/선정/비중/Crash/DD/Blacklist 규칙
  - 상태파일(`trade_state.json`, `signal_state.json`) 사용 방식
  - 거래비용, 슬리피지, 현금버퍼, 최소주문 처리
- docstring, 로그 배너, HTML 문구, 클래스명, 매뉴얼의 버전 표기가 stale일 수 있다.
- 버전 표기보다 실제 파라미터와 실행 순서를 기준으로 판단한다.

### Single Source of Truth

- 코인 전략 변경 시 최소 동기화 대상 (V22):
  - `trade/coin_live_engine.py` (앙상블 엔진)
  - `trade/executor_coin.py` (실매매 executor)
  - `strategies/cap_defend/backtest_spot_barfreq.py` (V22 백테스트)
  - `strategies/cap_defend/backtest_official.py` (legacy 참조만)
  - `strategies/cap_defend/recommend.py`
  - `strategies/cap_defend/recommend_personal.py`
  - `V22_OPERATION_MANUAL.md`
  - `CLAUDE.md`
  - 상태파일: `trade_state.json`
- 주식 전략 변경 시 최소 동기화 대상:
  - `trade/executor_stock.py`
  - `strategies/cap_defend/recommend.py`
  - `strategies/cap_defend/recommend_personal.py`
  - `signal_state.json` / `kis_trade_state.json` 사용 규칙
  - `V22_OPERATION_MANUAL.md`
  - `CLAUDE.md`
- 선물 전략 변경 시 최소 동기화 대상:
  - `strategies/cap_defend/futures_live_config.py`
  - `strategies/cap_defend/futures_ensemble_engine.py`
  - `strategies/cap_defend/backtest_futures_full.py`
  - `trade/auto_trade_binance.py` (구현 예정)
  - `strategies/cap_defend/recommend_personal.py`
  - `CLAUDE.md`
- 앵커일, 버퍼, 상태키, 모니터 기준통화가 다르면 "같은 전략"이라고 쓰지 않는다.
- 현재 저장소에는 `1/10/19`와 `1/11/21` 표기가 혼재할 수 있으므로, 변경 시 관련 파일을 반드시 함께 정리한다.

## 코인 현물 전략 규칙 (V22, 재정의 2026-04-27)

V22 = 단일 sleeve combo. spot/fut 모두 1D + 4h 2멤버 EW 앙상블. C 슬리브 폐기.

### 코인 현물 V22 앙상블

- 2멤버 EW 50/50, snap 60일 동기 (live engine: trade/coin_live_engine.py)
- Member 1 (D_SMA42):   interval=D,  SMA42,  Mom20/127, snap=60봉×3,  canary hyst 1.5%
- Member 2 (H4_SMA240): interval=4h, SMA240, Mom12/180, snap=360봉×3, canary hyst 1.5%
- 공통: health=mom2vol (vol_cap 5%, vol_lookback 90d), universe_size=3, cap=1/3
- 가드 없음: gap/exclusion 가드 전면 제거. 앙상블 분산이 유일 방어.
- Upbit warning/delisting 코인은 universe 단계에서 즉시 제외.
- TX: 0.04% (백테스트), 실매매는 Upbit 수수료
- 단독 sleeve 성과 (BT 6.30yr, 2019-09~2025-12): CAGR 51.1%, MDD -21.2%, Cal 2.41, Sharpe 1.32

### 실행 파라미터

- Executor: `trade/executor_coin.py`
- Engine: `trade/coin_live_engine.py`
- State: `trade_state.json`
  - V22: `members/last_target_snapshot/last_upbit_status/last_warning_coins_alerted/rebalancing_needed`
- Cron: `5 9,13,17,21,1,5 * * *` (KST 4h 봉 닫힘 시각, 선물과 동기). bar-idempotency 로 4h/D봉 닫힘 시만 리밸.
- 알림: Upbit 유의 변경 알림 제거. 리밸런싱 시점만 텔레그램 전송.

### 아키텍처

- `cle.compute_live_targets(state, session, ...)` → EngineResult (combined_target, member_targets, freshness)
- `ec.run_once(dry_run)` → 엔진 호출 + delta 매매 + state 저장

### V21 → V22 마이그레이션 (2026-04-27)

- V21 (D 3멤버 EW) → V22 (1D + 4h 2멤버 EW): interval 다양성 + plateau-center cfg 채택
- C 슬리브 (champion mean-reversion) 코드 전면 삭제 (Upbit 등가 검증 탈락)
- KLINE_LIMITS '1h' 제거, '4h' (1500 bars) 추가
- BINANCE_INTERVAL_MAP '1h' 제거, '4h' 추가
- cron 4h 동기로 변경 (선물과 동일: 9/13/17/21/1/5시 :05)
- AI 합의 (Gemini + Codex) 후 plateau-center SMA42 채택 (peak SMA44 → drift 방어 우선)

## 선물 전략 규칙 (V22, 재정의 2026-04-27)

### 앙상블 구성

- 1D + 4h 2멤버 EW 50/50, snap 90일 동기
- D_SMA42:   interval=D,  SMA42,  Mom18/127, snap=90봉×3
- 4h_SMA240: interval=4h, SMA240, Mom12/180, snap=540봉×3 (=90일 동기)
- 공통: canary_hyst=0.015, n_snapshots=3, universe_size=3, cap=1/3
- 단독 sleeve 성과 (BT 6.30yr, L3): CAGR 187.9%, MDD -47.0%, Cal 4.00, Sharpe 0.33

### 실행 파라미터

- 레버리지: 고정 3배 (floor=mid=ceiling=3, 동적 레버리지 비활성)
- 스탑: 없음 (STOP_PCT=0, sync_stop_orders는 잔존 스탑 정리만 수행)
- 캐시 가드: 없음 (STOP_GATE_CASH_THRESHOLD=0)
- 가드 전면 제거 (2026-04-21): stop/cash_guard 로직 본체 삭제, 앙상블 분산이 유일 방어
- 거래비용: 0.04% (바이낸스 maker)
- 유지증거금: 0.4%

### 자산배분 (확정 2026-04-13)

- 주식 60% / 업비트 35% / 바이낸스 5%
- 밴드 리밸런싱: 8pp drift 초과 시만 (매일 cron 체크, 텔레그램 알림)
- 자산간 강제 이동(카나리 레짐 전환) 없음 — 각 자산 내부 방어에 맡김
- 리밸 시 전 자산 목표 비중으로 복원

## 백테스트 필수 규칙

### Look-Ahead Bias 금지

- **시그널: t-1일 종가 기준, 체결: t일 가격**. 반드시 분리한다.
- 실매매: 09:20에 전일 종가 판단 후 당일 매매.
- 코인: `sig_date = prev_date` (`prev_date is None`일 때만 첫날 예외).
- 주식: `sig_date = prev_trading_date`.
- crash, DD, blacklist, 카나리, 헬스, 선정 모두 같은 `sig_date`를 써야 한다.
- 모니터는 예외적으로 장중 현재가를 보되, 비교 기준은 반드시 "저장된 전일 완료봉"과 "저장된 SMA"여야 한다.
- `iloc[-1]`이 진행 중 봉일 수 있는 곳에서는 전일 기준이면 `iloc[-2]`를 명시한다.

### 일간 리밸런싱

- 코인/주식 모두 **매일** 루프를 돈다. 월간 전략이라도 월 1회만 돌리면 안 된다.
- 카나리, DD exit, blacklist, crash, pending 복구는 매일 체크한다.
- 종목 교체(snapshot)는 앵커일에만 한다.
- 월간 리밸런싱의 의미는 "매일 상태를 점검하되, 종목 교체는 앵커일에만"이다.
- 앵커일은 전략 파라미터다. 변경 시 백테스트, 실매매, 히스토리 생성, 매뉴얼, HTML 문구를 동시에 수정한다.

### 엔진 선택

- **코인: 구 엔진(3-snapshot 합성)** 기준을 유지한다.
- V15 독립 트랜치 엔진은 실매매와 불일치하므로 사용 금지.
- 3-snapshot은 독립 포트폴리오 3개를 따로 체결하는 것이 아니라, 비중을 합산(merge)해 단일 포트폴리오로 리밸런싱한다.
- 이벤트 처리 순서도 전략 일부다. 순서를 바꾸면 다른 전략이다.
- 코인 기본 순서는 다음과 같이 유지한다:
  - crash/cooldown
  - canary flip
  - blacklist 갱신
  - DD/헬스 제거
  - PFD/앵커 갱신
  - drift/재진입
  - execute_rebalance

### 파라미터 키와 기본값

- 코인 엔진(`coin_engine.py`)이 인식하는 키만 사용한다.
- 카나리: `K1`, `K8` 등.
- 헬스: `H1`, `HK` 등.
- 선정: `baseline`, `S1`~`S10`.
- 비중: `baseline`, `WC`, `W1`, `W2`.
- 리스크: `G5`.
- `selection='mcap'`, `weighting='ew'` 같은 비표준 키는 엔진 fallback을 일으켜 의도와 다른 결과가 나온다.
- `B()` 기본값은 중립이 아니다.
- `B()` 기본값 함정:
  - `sma_period=150`
  - `canary_band=0`
  - `health_sma=2`
  - `health_mom_short=21`
- V14+ 계열은 필요한 값을 모두 명시적으로 덮어쓴다.
- 주식 `SP()`도 마찬가지다. `canary_assets`, `select`, `weight`, `sharpe_lookback`, `crash`, `crash_thresh`, `crash_cool`은 핵심값을 항상 명시한다.
- 파라미터를 바꾼 뒤에는 실제 `Params/SP` 객체에 어떤 값이 들어갔는지 확인한다.

### 현금 키 규칙

- 백테스트/엔진 계열 코드는 현금 키로 주로 `CASH`를 쓴다.
- 실매매/리포트 계열 코드는 현금 키로 주로 `Cash`를 쓴다.
- 딕셔너리 비교, merge, 동기화 코드에서는 `Cash`와 `CASH`를 혼동하지 않는다.
- 백테스트 코드를 실매매 쪽으로 옮길 때 가장 먼저 현금 키를 확인한다.

### 거래비용

- 코인: 0.4% 편도 기준 유지.
- 주식: 0.1% 편도 기준 유지.
- 거래비용을 바꾸면 성과표만이 아니라 턴오버 해석도 같이 바뀐다.
- 실매매 최소주문, 분할매매, 슬리피지 제약이 백테스트와 크게 다르면 성과를 직접 비교하지 않는다.

### 상태/캐시/기준통화

- `trade_state.json`, `signal_state.json`은 단순 캐시가 아니라 전략 상태다.
- hysteresis dead zone에서는 이전 상태(`coin_risk_on`)를 반드시 참조한다.
- `recommend_personal.py`는 단순 리포트가 아니라 `signal_state.json`과 `trade_state.json`을 갱신하는 stateful 코드다.
- 상태파일 삭제는 단순 초기화가 아니다.
- 상태파일이 없으면 다음과 같은 동작 변화가 생긴다:
  - hysteresis dead zone fallback
  - 첫 실행 시 전 트랜치를 현재 신호로 초기화
  - flip/PFD 문맥 손실
- 상태파일 저장은 항상 원자적 저장(`tmp` + `os.replace`)으로 한다.
- HOLD일에도 monitor 캐시는 갱신해야 한다. stale cache를 남기면 monitor가 잘못 동작한다.
- 모니터 카나리/crash 비교는 USD 기준을 유지한다.
- DD exit은 누적 peak 캐시를 쓰지 말고, 동일 소스 시계열의 **rolling 60일 peak**를 직접 조회한다.
- `coin_peaks` 같은 장기 캐시는 stale 오발동 위험이 크므로 전략 로직에 다시 쓰지 않는다.
- 긴급청산 후에는 `pending_trades`, `tranches`, `coin_risk_on`, peak 관련 캐시를 함께 정리한다.

### 데이터 정합성

- 마지막 데이터 날짜가 목표 날짜와 맞지 않는 자산은 제외한다.
- 코인 Yahoo 종가와 Upbit KRW 종가가 심하게 어긋나면 해당 자산을 제외한다.
- 주식은 가급적 adjusted close 기준을 유지한다.
- `get_price()`의 `ffill`은 편의 기능이지 공짜 체결이 아니다. 비거래/누락 데이터 구간에서 체결 해석을 조심한다.
- 백테스트, 리포트, 모니터가 서로 다른 기준통화나 다른 가격 소스를 쓰면 결과 해석을 분리한다.

### 과적합 방지

- 10-anchor 평균 사용: 코인 `(1,10,19)`~`(10,19,28)`.
- 평균만 보지 말고 anchor 간 분산도 본다.
- `sigma(Sharpe)`는 낮을수록 좋다. 기존 경험상 0.1 이하를 robust 후보로 본다.
- 파라미터 인접값에서 성과가 유사해야 한다.
- 단일 기간 최적이 아니라 다기간(`2018~`, `2019~`, `2021~`) 일관성을 확인한다.
- 성과 개선이 turnover/비용 증가로 설명되면 채택하지 않는다.

## 실매매 운영 규칙

### 주문/리밸런싱

- 매매 전 미체결 주문을 먼저 정리하고 잔고를 본다.
- 리밸런싱은 **매도 먼저, 매수 나중** 순서를 유지한다.
- 최소주문 금액 미만은 강제로 맞추지 않는다.
- 부분 미체결은 `pending_trades`에 저장하고 monitor가 복구한다.
- `--force`는 강제 재실행이지, 앵커를 소모하는 이벤트가 아니다.
- `target_amount`와 `cash_buffer`는 배분 규모를 바꾸는 값이지, 신호 로직을 바꾸는 값이 아니다.

### 모니터 주의사항

- `auto_trade --monitor`의 중복 방지는 `run_trade.sh`의 flock이 담당한다.
- 모니터 내부에 별도 flock을 넣으면 자기 차단이 발생하므로 금지.
- monitor는 긴급 탈출 + pending 복구용이다. 월간 리밸런싱 엔진을 복제하지 않는다.
- cash buffer 변경은 `trade_state['buffer_changed']`로 기록하고, 다음 `--trade` 실행에서 처리한다.
- monitor는 buffer 변경만으로 매매하지 않는다.

## 코드 수정 시 규칙

### recommend는 항상 두 개 수정

- `strategies/cap_defend/recommend.py`와 `strategies/cap_defend/recommend_personal.py`를 동시에 본다.
- UI 문구만 같은 것이 아니라, 신호 계산식, 표시 컬럼, 설명 문구, 버퍼 반영 여부까지 같이 점검한다.

### 상태 스키마 변경 시

- `trade_state.json` 키를 추가/변경하면 아래를 함께 점검한다:
  - `trade/executor_coin.py`
  - `strategies/cap_defend/recommend_personal.py`
  - `trade/api_server.py`
  - 운영 매뉴얼
- 핵심 상태키 예시:
  - `coin_risk_on`
  - `tranches`
  - `last_flip_date`
  - `pfd_done`
  - `pending_trades`
  - `cash_buffer`
  - `buffer_changed`
  - `btc_sma60_usd`
  - `btc_prev_close_usd`
- 가능한 한 하위호환 fallback을 남긴다.

### 서버 배포 순서

1. 로컬 수정 + 테스트
2. `scp`로 서버 배포
3. 서버에서 실행 확인
4. API 서버 변경 시 재시작
5. git commit + push

### 서버 파일 매핑

운영 매핑·cron·헬스체크·복구 절차는 `SERVER_OPS.md` 참조 (단일 source of truth).
요약:

- `strategies/cap_defend/recommend.py` → `~/recommend.py`
- `strategies/cap_defend/recommend_personal.py` → `~/recommend_personal.py`
- `trade/executor_coin.py` → `~/executor_coin.py`
- `trade/executor_stock.py` → `~/executor_stock.py`
- `trade/auto_trade_binance.py` → `~/auto_trade_binance.py`
- `trade/coin_live_engine.py` → `~/coin_live_engine.py`
- `trade/ops/serve.py` → `~/serve.py`
- `trade/ops/trade_api_server.py` → `~/trade_api_server.py`
- `trade/ops/watchdog_serve.sh` → `~/watchdog_serve.sh`
- `trade/ops/run_executor.sh` → `~/run_executor.sh`
- `trade/ops/run_recommend.sh` → `~/run_recommend.sh`
- `trade/ops/daily_history.py` → `~/daily_history.py`
- `trade/ops/crontab.txt` → `crontab -l` 사본

## 데이터 품질

- `historical_universe.json` 류의 월초 시점 시총 데이터는 생존편향 방지에 필수다.
- 가격 CSV는 코인/주식 모두 기준 컬럼(`Adj_Close` 또는 전략 정의된 close)을 명확히 고정한다.
- Yahoo, Upbit, Binance fallback은 각각 역할이 다르다:
  - 백테스트/리포트 기준 시계열
  - 실매매 체결가/현재가
  - 모니터 응급 fallback
- 서로 다른 소스를 혼용하면 반드시 문서에 남긴다.

## 전략 연구 방법론

### 기본 원칙

- 한 번에 한 가지 가설만 바꾼다.
- baseline을 먼저 고정하고, 한 실험에서 바뀐 축을 명확히 기록한다.
- "성능이 좋아 보인다"가 아니라 "왜 좋아져야 하는지"를 먼저 적고 시작한다.
- 규칙 추가 전에는 반드시 기존 실패 사례를 재현 가능한지 확인한다.

### 검증 절차

- In-sample에서 후보를 좁힌 뒤, 최근 구간은 holdout처럼 따로 본다.
- 단일 전체기간 성과보다 서브기간 일관성을 우선한다.
- anchor 평균과 anchor 분산을 함께 확인한다.
- 파라미터 그리드에서 최고점 하나보다 plateau 존재 여부를 본다.
- 새 규칙은 반드시 ablation으로 검증한다.
  - 규칙 ON
  - 규칙 OFF
  - 관련 파라미터 ±인접값
- CAGR, Sharpe만 보지 말고 MDD, Calmar, turnover, rebal 횟수, cash 체류시간을 함께 본다.
- 개선 폭이 거래비용 반영 후에도 유지되는지 확인한다.

### 채택 기준

- 최근 구간만 좋아지고 과거 구간이 망가지면 기각.
- anchor 하나에서만 유난히 좋으면 기각.
- 실매매 엔진으로 이식 불가능하거나 stateful 로직을 재현 못 하면 기각.
- 설명 문구가 길어지고 예외처리가 계속 붙는 규칙은 우선 의심한다.

## 새로운 전략 아이디어 테스트 절차

1. 가설을 한 줄로 적는다.
2. baseline 버전과 바꾸려는 축을 명시한다.
3. 실험 코드는 먼저 백테스트 전용으로 넣고, 실매매 코드는 바로 건드리지 않는다.
4. 최소 3개 시작시점과 10-anchor 평균으로 1차 검증한다.
5. 인접 파라미터와 ablation으로 2차 검증한다.
6. 실제 거래 로그 관점에서 최근 몇 개월 구간을 수동 점검한다.
7. 실매매 엔진으로 같은 상태전이와 같은 이벤트 순서를 재현할 수 있는지 확인한다.
8. 채택 결정 후에만 `trade/coin_live_engine.py`, `trade/executor_coin.py`, `recommend*.py`, 매뉴얼, `CLAUDE.md`를 동기화한다.
9. 서버 반영 전 dry-run 또는 소액/모의 shadow 기간을 둔다.
10. 반영 직후 1주일은 HTML, 로그, state 파일, pending, monitor 동작을 집중 점검한다.

## 전략 변경 시 체크리스트

- [ ] 백테스트 코드 업데이트 (`backtest_spot_barfreq.py` V22 / `legacy/backtest_official.py` legacy 참조)
- [ ] `trade/coin_live_engine.py` + `trade/executor_coin.py` 동기화
- [ ] `recommend.py` + `recommend_personal.py` 동시 수정
- [ ] 앵커일 정의 일치 확인
- [ ] 카나리 hysteresis와 `coin_risk_on` state 참조 일치 확인
- [ ] health/selection/weighting/risk 키가 엔진 인식값과 정확히 일치하는지 확인
- [ ] Crash/DD/Blacklist/cooldown 값 확인
- [ ] DD Exit 기준통화와 peak 계산 방식 확인
- [ ] `trade_state.json` 스키마 영향 확인
- [ ] `V22_OPERATION_MANUAL.md` 업데이트
- [ ] 메모리(`MEMORY.md`) 업데이트
- [ ] `CLAUDE.md`에 새 교훈 반영
- [ ] 서버 배포 + 실행 확인

## 백테스트 ↔ 실매매 차이점

- 백테스트: 무조건 체결. 실매매: 슬리피지, 미체결, 분할매매, 최소주문 제약 존재.
- 백테스트: tx 0.4%로 단순화. 실매매: 거래소 수수료 + 유동성 제약 + 분할체결.
- 백테스트: 엔진 내부 `CASH`. 실매매/리포트: `Cash`.
- 백테스트 앵커와 실매매 트랜치일이 다르면 성과를 직접 1:1 비교하지 않는다.
- `V22_OPERATION_MANUAL.md`는 운영 기준 문서이고, 실제 truth는 코드와 상태 전이까지 포함한다.
