# Cap Defend 프로젝트 규칙

## 전략 정의 원칙

### 전략은 이름이 아니라 구현이다

- `V22` 같은 버전명보다 실제 구현이 우선이다.
- 아래 중 하나라도 다르면 같은 전략으로 취급하면 안 된다:
  - 시그널 시점
  - 체결 시점
  - 앵커일
  - 카나리/헬스/선정/비중 규칙
  - 상태파일(`trade_state.json`, `signal_state.json`) 사용 방식
  - 거래비용, 슬리피지, 현금버퍼, 최소주문 처리
- docstring, 로그 배너, HTML 문구, 클래스명, 매뉴얼의 버전 표기가 stale일 수 있다.
- 버전 표기보다 실제 파라미터와 실행 순서를 기준으로 판단한다.

### Single Source of Truth

- 코인 전략 변경 시 최소 동기화 대상 (V24):
  - `trade/coin_live_engine.py` (앙상블 엔진)
  - `trade/executor_coin.py` (실매매 executor)
  - `strategies/cap_defend/unified_backtest.py` (V24 spot/fut BT 엔진, asset_type='spot'; 호출 `trade/v24_shadow_today.py`)
  - `strategies/cap_defend/recommend.py`
  - `strategies/cap_defend/recommend_personal.py`
  - `V24_OPERATION_MANUAL.md`
  - `CLAUDE.md`
  - 상태파일: `trade_state.json`
- 주식 전략 변경 시 최소 동기화 대상:
  - `trade/executor_stock.py`
  - `strategies/cap_defend/recommend.py`
  - `strategies/cap_defend/recommend_personal.py`
  - `signal_state.json` / `kis_trade_state.json` 사용 규칙
  - `V24_OPERATION_MANUAL.md`
  - `CLAUDE.md`
- 선물 전략 변경 시 최소 동기화 대상 (V25):
  - `strategies/cap_defend/futures_live_config.py`
  - `strategies/cap_defend/backtest_futures_v25.py` (V25 BT 엔진, CROSS + 동적 L)
  - `strategies/cap_defend/futures_ensemble_engine.py`
  - `strategies/cap_defend/backtest_futures_full.py`
  - `trade/auto_trade_binance.py` (구현 예정)
  - `strategies/cap_defend/recommend_personal.py`
  - `CLAUDE.md`
- 앵커일, 버퍼, 상태키, 모니터 기준통화가 다르면 "같은 전략"이라고 쓰지 않는다.
- 현재 저장소에는 `1/10/19`와 `1/11/21` 표기가 혼재할 수 있으므로, 변경 시 관련 파일을 반드시 함께 정리한다.

## 코인 현물 전략 규칙 (V24, 재정의 2026-04-30)

V24 = 모든 자산 1D 단일 + drift trigger. 4h 멤버 제거, cron 1일 1회로 단순화.

### 코인 현물 V24

- D_SMA42 단일 (ENSEMBLE_WEIGHTS = {'D_SMA42': 1.0})
- snap_interval_bars=217, n_snapshots=7 (217 = 7 × 31, prime stagger=31)
- canary_hyst=0.015, universe_size=3, cap=1/3, health=mom2vol (vol_cap 5%, vol_lookback 90d)
- drift_threshold=0.10 (자본금 기준 half-turnover)
- drift 발화 시 refill v2 (라이브 `_apply_refill_v2_to_state`): mom2 음수(ms<0&ml<0) 코인 → fresh healthy 교체. BT 정합 = `unified_backtest.py` DRIFT_HEALTH_MODE='refill' (v24_shadow_today.py 가 설정). 검증(2026-06-06, research/bt_spot_refill_vs_anchor.py): drift=0.10 에서 refill ≡ anchor-only (5.4yr 종목교체 0일, CAGR/MDD/Cal 동일) — 발화일 보유코인이 항상 모멘텀 양수. 즉 현물 종목교체는 사실상 앵커(217)에서만. refill 은 미래 모멘텀급락+drift 동시 발생 대비 dormant 방어.
- 가드 없음. Upbit warning/delisting 코인은 universe 에서 즉시 제외.
- TX: 0.04% (BT), 실매매는 Upbit 수수료
- 단독 sleeve (BT 5.4yr): CAGR +82%, MDD -18%, Cal 4.63

### 실행 파라미터

- Executor: `trade/executor_coin.py` (cur_w 자본금 비중 산출 → engine 주입)
- Engine: `trade/coin_live_engine.py` (SCHEMA_VERSION='V24', drift_fire 평가)
- State: `trade_state.json` — `members/last_target_snapshot/rebalancing_needed/schema_version`
- Cron: `5 9 * * *` (선물과 동기, D봉 닫힘 직후 1일 1회)
- 알림: 리밸런싱 시점만 텔레그램.

### drift trigger

- half_turnover(cur_w, tgt_w) = sum(|tgt - cur|) / 2
- need_rebal = is_daily_bar AND (snap_fire OR (canary_on AND ht >= threshold))
- cur_w spot: Upbit 잔고 → KRW 환산 비중 (Cash 키 = 'Cash')

## 주식 전략 규칙 (V25, 도입 2026-05-29)

### 구성

- Universe: R7 = SPY, QQQ, VEA, EEM, GLD, PDBC, VNQ (R7B 의 EWJ 를 VNQ 로 교체)
- Defense: IEF, BIL, BNDX, GLD, PDBC
- EEM canary SMA200 + 0.5% hysteresis
- 선정: Z-score (Mom + Sharpe126) 랭킹 → 3-mom (30/72/230) 필터 → cap=1/3+Cash
  · Z-rank Mom = 가중 0.5×ret63 + 0.3×ret126 + 0.2×ret252 (채택 BT bt_stock_coin_v3.precompute 정의). 2026-06-06 정정: 라이브 calc_weighted_mom 이 순수252(V15 잔재)였어 채택BT와 종목선정 16.8% 어긋남 → 가중으로 수정, replay-diff 100% 정합. 함수명 믿지 말고 구현 확인.
- multi-snap n=3 stag=23 int=69 (prime stagger 23 그대로 유지)
- drift threshold 0.05 (V24 0.10 에서 강화)
- 종목 교체 = 앵커일(69일) OR drift 5pp 발화일 (V25 주식 한정, 2026-06-06). drift 발화 시 그날 fresh selection(Z-score 랭킹→3-mom 필터→top3 cap)으로 snap picks 리필 후 merge→체결. last_rebal_date 보존(앵커 cadence 유지). 채택 BT(bt_stock_mom3) 정합. executor 플래그 STOCK_DRIFT_REFILL=True (False 면 anchor-only 복귀). V24 코인/선물의 "앵커일에만 교체" 원칙과 구분.
- 가드 없음. EEM canary + 3-mom 필터 + 분산 = 유일 방어
- 5.4yr BT (window rank-sum 1518 × 11 anchors): C 안 (multi+mom cap+Cash) avg_rank 1.246 vs B (현행) 2.471

### 통일 아키텍처 (V25 도입)

- 전략 모듈: `strategies/cap_defend/stock_strategy_v25.py` (순수 함수)
  · compute_offense / compute_defense / eem_canary / compute_strategy
- executor_stock.py 가 23:35 KST cron 에서 직접 전략 계산 (signal_state.json 의존 제거, fallback 으로만 유지)
- recommend / recommend_personal 은 같은 모듈을 호출해 Daily Report 표시 (single source of truth)
- 신호 fallback: signal_state.json V25 + 24h freshness 검증 통과 시만 사용. V24 fallback 은 첫 3 영업일 금지.

### 가드 (V25 도입 시)

- 가격 기준일 hard check: 모든 ticker last_date 일치 안 하면 SKIP
- yfinance T-1 확정 종가만 사용 (max_close_date = KST 어제, 장중 partial row 차단)
- Cold-start SKIP: prev_risk_on + canary_risk_on 둘 다 None + EEM 결측 → SKIP
- EEM 결측 시 risk_on 자동 변경 안 함 (prev_risk_on 유지)
- prev_risk_on=False 명시 보존 (or 연산 대신 None check)

### V24 → V25 마이그레이션 (2026-05-29)

- 주식 sleeve: Z-score Top3 EW → Z-score 랭킹 + 3-mom 필터 + cap=1/3+Cash
- universe: R7B (EWJ) → R7 (VNQ) — BT 와 일치
- drift threshold 0.10 → 0.05
- 코드 변경:
  · `strategies/cap_defend/stock_strategy_v25.py` (신설 — 순수 전략 함수)
  · `trade/executor_stock.py` (_fetch_strategy_prices + _compute_signal_v25 + 가드 5종)
  · `strategies/cap_defend/recommend.py` (모듈 호출)
  · `strategies/cap_defend/recommend_personal.py` (모듈 호출 + universe R7 통일)
- 채택 근거: window rank-sum 5 게이트 통과 (thr/window/plateau/cost stress 5x/multi-anchor)
- AI 토론 조건부 GO: critical 4건 수정 (prev_risk_on, T-1 date, EEM 결측, signal freshness) 모두 적용

## 선물 전략 규칙 (V25, 도입 2026-05-28)

### 구성

- Sleeve: D_SMA42 단일 (V24 와 동일) — snap_interval_bars=95, n_snapshots=5, drift_threshold=0.03
- canary_hyst=0.015, universe_size=3, cap=1/3, health_mode=mom2vol
- 동적 per-coin L (V25 신규): 각 코인 L = min(BTC_cap, K2_per_coin)
  · BTC cap: BTC/SMA42 ratio > 1.05 → L4, > 1.015 → L3, else L2
  · K2 per-coin: close/SMA7 ratio > 1.075 → L4, > 1.025 → L3, else L2
- 마진모드: CROSS (V24 ISOLATED → V25 CROSS)
- 단독 sleeve BT (backtest_futures_v25.py, 5.6yr, tx 0.0006): CAGR 312%, MDD -38.3%, Cal 8.12 (V24 4.05 대비 +4.07)
- alloc 60/25/15 + T1+T3U: Cal 5.72 / MDD -18.6% / CAGR 106%

### 실행 파라미터 (V25)

- 레버리지: 동적 (Lmin=2, Lmid=3, Lmax=4). per-coin min(BTC_cap, K2)
- 스탑: 없음 (STOP_PCT=0)
- 캐시 가드: 없음 (STOP_GATE_CASH_THRESHOLD=0)
- 가드 없음 — 분산 + per-coin L 다이내믹이 유일 방어
- 거래비용: 0.04% (바이낸스 maker)
- 유지증거금: BTC/ETH 0.4%, 알트 0.5~0.65% (실 tier 기준, BT 0.4% 보수적)
- 마진모드: CROSSED (auto_trade_binance.py MARGIN_TYPE = 'CROSSED')
- 4h fetch 제거, 1d 1회 cron (`5 9 * * *`)
- 디버그 로그: DEBUG_LEVERAGE=True, DEBUG_MARGIN=True (V25 검증 기간)

### V24 → V25 마이그레이션 (2026-05-28)

- 선물 sleeve: D_SMA42 + L3 ISO → D_SMA42 + 동적 L4 max + CROSS
- 코인 spot / 주식 sleeve: V24 유지 (변경 없음)
- 자산배분 60/25/15: V24 그대로 유지
- 코드 변경:
  · `strategies/cap_defend/futures_live_config.py` (LEVERAGE_MIN/MID/MAX, K2_*, BTC_CAP_*, MARGIN_TYPE)
  · `trade/auto_trade_binance.py` (LEVERAGE_FLOOR/MID/CEILING=2/3/4, SCHEMA_VERSION='V25', _calc_btc_cap_lev/_calc_percoin_k2_lev, set_margin_type 기본 CROSSED)
  · `strategies/cap_defend/backtest_futures_v25.py` (신규 — CROSS 청산 모델 + build_K2_signal 헬퍼)
  · `strategies/cap_defend/recommend_personal.py` (STRATEGY_VERSION='V25', 선물 sleeve 설명)
- 채택 근거: K2 (SMA=7 h=2.5%) plateau center 검증 (25 cfg window rank-sum 1위). J (mom-based) Cal 7.45 대비 K2 8.12 + MDD 7pp 개선.

### V22 → V24 마이그레이션 (2026-04-30) [이전 — 참고]

- 모든 자산 1D 단일 + drift trigger (4h 멤버 제거)
- 주식: SNAP_PERIOD=126→69, STAGGER=42→23, N_SNAPS=3 유지
- 코인 spot: 1D+4h EW → D_SMA42 단일 sn=217 n=7 drift=0.10
- 선물 sleeve: 1D+4h EW → D_SMA42 단일 sn=57 n=3 drift=0.05 (V24 도입 후 sn=95 n=5 drift=0.03 갱신)
- spot n_snap=7 vs fut n_snap=5 (gcd=1, 서로소)
- stagger: stock=23, spot=31, fut=19 (모두 distinct prime)
- cron 4h x 6 → 1d x 1 (5 9 * * *)
- 마이그레이션 스크립트: `trade/migrate_v22_to_v24.py`
- 운영 매뉴얼: `V24_OPERATION_MANUAL.md` (V25 도입 후 deprecated, `V25_OPERATION_MANUAL.md` 참조)

### V22 → V24 마이그레이션 (2026-04-30)

- 모든 자산 1D 단일 + drift trigger (4h 멤버 제거)
- 주식: SNAP_PERIOD=126→69, STAGGER=42→23, N_SNAPS=3 유지
- 코인 spot: 1D+4h EW → D_SMA42 단일 sn=217 n=7 drift=0.10
- 선물: 1D+4h EW → D_SMA42 단일 sn=57 n=3 drift=0.05
- spot n_snap=7 vs fut n_snap=3 (gcd=1, 서로소)
- stagger: stock=23, spot=31, fut=19 (모두 distinct prime)
- cron 4h x 6 → 1d x 1 (5 9 * * *)
- 마이그레이션 스크립트: `trade/migrate_v22_to_v24.py`
- 운영 매뉴얼: `V24_OPERATION_MANUAL.md`

### 자산배분 (V24 갱신 2026-05-26, B 안 채택 — per-sleeve buffer + 수동 송금)

- 주식 계좌 65% / 업비트 20% / 바이낸스 15% (stock 계좌 안에 cash buffer 포함)
  - 이전: 60/25/15 (M 안, 2026-05-22) → 신규 60/25/15 (B 안, 2026-05-26)
  - stock 실투자 = 60 × 0.93 = 60.45% of total (≈ 기존 60%)
  - total cash ≈ 4.9% (60×0.07 + 20×0.01 + 15×0.01)
  - 변경 근거: alloc_buf5 + per_sleeve_buffer BT — 60/25/15 + stock 7% + 코인 1% 가 BT 60/20/15/5 와 동일 (Cal 3.88, MDD -18.4%, CAGR 71.2%; 60/25/15 baseline 2.58 대비 +1.30 Cal)
- Per-sleeve cash buffer (V24 B 안):
  - stock 계좌 (KIS): 7% cash (계좌 내부 cash buffer, 송금 즉시 활용)
  - spot 계좌 (Upbit): 1% KRW
  - fut 계좌 (Binance): 1% USDT
  - 총 cash ≈ 5% of total
- 리밸 트리거 (OR 조합):
  - T1: half_turnover (sum|cur_w - tgt_w|/2) ≥ 20pp
  - T3U_can: max((tgt - cur_w)/tgt) ≥ 20% AND 해당 sleeve canary ON
    - stock sleeve canary = EEM Risk-On (live: SMA300 hyst 2%)
    - spot sleeve canary = BTC > SMA42*(1+1.5%)
    - fut sleeve canary = binance_state.json strat.canary_on
- 자산배분 자동 rebal 폐지 (B 안 핵심):
  - alloc_transit phantom buffer 시스템 (cap_ratio 자동 적용) 영구 비활성화
  - 트리거 ON 시 텔레그램 알림만 (송금 제안 메시지 + 수동 권장)
  - 사용자가 직접 KIS ↔ Upbit/Binance 송금 수행
  - 각 sleeve executor 는 자기 계좌 내부에서만 자동매매 (sleeve 내부 신호 — V24 stock SP / spot D_SMA42 / fut D_SMA42)
- 알림 rate-limit: 같은 reason 24h 내 1회만 전송
- 폐지된 시스템 (자료 보존):
  - alloc_transit phantom buffer 옵션 D (2026-05-23~2026-05-26)
  - 자동 cap_ratio + cap_defend 매도 트리거
- BT 결과 (5.13yr, V24 sleeve, 알림 시점 100% 송금 가정):
  - alloc 60/20/15 + buffer (6/1/1): CAGR 71.2%, Cal 3.88, MDD -18.4%, 누적 14.99x, fire 16
  - 라이브 trigger E (T1=13 OR T3U=20% state) 그대로 유지
  - 트리거 변경 (T3U=25 state) Cal +0.04 향상은 라이브 안정성 손해로 미채택

## 백테스트 필수 규칙

### Look-Ahead Bias 금지

- **시그널: t-1일 종가 기준, 체결: t일 가격**. 반드시 분리한다.
- 실매매: 09:20에 전일 종가 판단 후 당일 매매.
- 코인: `sig_date = prev_date` (`prev_date is None`일 때만 첫날 예외).
- 주식: `sig_date = prev_trading_date`.
- 카나리, 헬스, 선정 모두 같은 `sig_date`를 써야 한다.
- 모니터는 예외적으로 장중 현재가를 보되, 비교 기준은 반드시 "저장된 전일 완료봉"과 "저장된 SMA"여야 한다.
- `iloc[-1]`이 진행 중 봉일 수 있는 곳에서는 전일 기준이면 `iloc[-2]`를 명시한다.

### 일간 리밸런싱

- 코인/주식 모두 **매일** 루프를 돈다. 월간 전략이라도 월 1회만 돌리면 안 된다.
- 카나리, pending 복구는 매일 체크한다.
- 종목 교체(snapshot)는 앵커일에만 한다. (예외: V25 주식 sleeve 는 drift 5pp 발화일에도 fresh selection 으로 교체 — 위 "주식 전략 규칙 V25" 참조. 코인/선물 V24 는 앵커일에만.)
- 월간 리밸런싱의 의미는 "매일 상태를 점검하되, 종목 교체는 앵커일(+V25 주식은 drift 발화일)에만"이다.
- 앵커일은 전략 파라미터다. 변경 시 백테스트, 실매매, 히스토리 생성, 매뉴얼, HTML 문구를 동시에 수정한다.

### 엔진 선택

- **코인: 구 엔진(3-snapshot 합성)** 기준을 유지한다.
- V15 독립 트랜치 엔진은 실매매와 불일치하므로 사용 금지.
- 3-snapshot은 독립 포트폴리오 3개를 따로 체결하는 것이 아니라, 비중을 합산(merge)해 단일 포트폴리오로 리밸런싱한다.
- 이벤트 처리 순서도 전략 일부다. 순서를 바꾸면 다른 전략이다.
- 코인 기본 순서는 다음과 같이 유지한다:
  - crash/cooldown
  - canary flip
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
- `coin_peaks` 같은 장기 캐시는 stale 오발동 위험이 크므로 전략 로직에 쓰지 않는다.
- 긴급청산 후에는 `pending_trades`, `tranches`, `coin_risk_on` 캐시를 함께 정리한다.

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

- [ ] 백테스트 코드 업데이트 (`unified_backtest.py` V24 spot/fut, 호출 `v24_shadow_today.py` / `legacy/backtest_official.py` legacy 참조)
- [ ] `trade/coin_live_engine.py` + `trade/executor_coin.py` 동기화
- [ ] `recommend.py` + `recommend_personal.py` 동시 수정
- [ ] 앵커일 정의 일치 확인
- [ ] 카나리 hysteresis와 `coin_risk_on` state 참조 일치 확인
- [ ] health/selection/weighting/risk 키가 엔진 인식값과 정확히 일치하는지 확인
- [ ] cooldown / drift threshold 값 확인
- [ ] `trade_state.json` 스키마 영향 확인
- [ ] `V24_OPERATION_MANUAL.md` 업데이트
- [ ] 메모리(`MEMORY.md`) 업데이트
- [ ] `CLAUDE.md`에 새 교훈 반영
- [ ] 서버 배포 + 실행 확인

## 백테스트 ↔ 실매매 차이점

- 백테스트: 무조건 체결. 실매매: 슬리피지, 미체결, 분할매매, 최소주문 제약 존재.
- 백테스트: tx 0.4%로 단순화. 실매매: 거래소 수수료 + 유동성 제약 + 분할체결.
- 백테스트: 엔진 내부 `CASH`. 실매매/리포트: `Cash`.
- 백테스트 앵커와 실매매 트랜치일이 다르면 성과를 직접 1:1 비교하지 않는다.
- `V24_OPERATION_MANUAL.md`는 운영 기준 문서이고, 실제 truth는 코드와 상태 전이까지 포함한다.

## BT↔라이브 선정 정합 (통제 패리티 증명, 2026-06-06)

코인 현물·선물 라이브 선정함수와 채택 BT 선정함수에 동일 유니버스·동일 OHLCV·
동일 BTC 카나리를 주입해 일별 picks/weights 100% 일치를 증명. read-only 하니스.

- 현물: `coin_live_engine.compute_member_target` vs `unified_backtest.run(asset_type='spot')`
  → 2005일 picks/weights 100% 일치 (`research/parity_spot.py`).
- 선물: `auto_trade_binance.compute_strategy_target` vs `unified_backtest.run(asset_type='fut')`
  → 2004일 picks/weights 100% 일치 (`research/parity_fut.py`).
- 결론: 선정 로직(후보순→mom2vol헬스→top3→greedy흡수→cap→카나리 hyst→snap merge)이
  라이브=BT 완전 동일. 주식 같은 점수정의(순수252 vs 가중) 버그 없음 확인.

하니스 설계 핵심 (재현 시 주의):
- BT `run()` 루프는 `bar_i = sma_period+1` 부터 시작 → `trace[0].date != all_dates[0]`.
  정렬은 각 trace 항목의 date 직전 `all_dates` 항목(=prev_date=signal)으로 키잉. (43일 어긋남 주의.)
- snap 머신 무력화: `n_snapshots=1, snap_interval_bars=1` → combined = 매봉 fresh 선정.
- 가드 OFF (라이브 무가드 정합): `dd_lookback=0, bl_drop=0, crash_threshold=-10`.
- 선물 라이브는 `index[-2]`=signal (index[-1]=진행중봉) → bslice 를 d+1 까지 줘 정렬.

남은 구조적 차이 (버그 아님, 의도/미세):
- 유니버스 소스: 라이브=CoinGecko 실시간 top40, BT=`historical_universe.json` point-in-time(생존편향 방지).
  동일 엔진에서 fixed vs point-in-time mcap 비교 시 picks 가 5.69%(114/2005일) 갈림 = 의도된 BT↔라이브 갭(체결 슬리피지급). 선정 로직이 아닌 입력 차이.
- 선물 종가 자름: 라이브=`[:-1]`, BT=`ffill[:ci+1]`. 정상(무결측) 일봉 동일, 결측/stale 봉만 1봉 차. 통일은 dry-run+승인 후 별도 (DEFER).
- 카나리 BTC: 라이브 spot override, BT perp. SMA42 경계일만 영향. 통일 DEFER.
- 운영 stale: `BTCUSDT_1h.csv` 등 1h 입력이 밀리면 shadow 검증 약화 → 데이터 신선도 모니터 필요.
