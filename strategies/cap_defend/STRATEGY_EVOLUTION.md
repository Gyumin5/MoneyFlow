# 전략 진화 (V12 → V25)

이 문서는 Cap Defend 전략의 버전별 변경점과 결정 근거를 정리한다. 자세한 백테스트 결과는 각 버전의 백테스트 코드 또는 [research/](./research/)의 결과 파일을 참조한다. V21 개발 중 실험은 [`../../history/V21_HISTORY.md`](../../history/V21_HISTORY.md) 별도 문서 참조.

현재 운영: 주식 V25 / 코인현물 V24 / 선물 V25.

---

## 한눈에 보기

| 버전 | 시점 | 자산군 | 핵심 변경 | 비고 |
|---|---|---|---|---|
| V12 | 2026-01 | 코인+주식 | 초기 합본 (단순 모멘텀 + 카나리) | 코인 과적합, 주식 단일트랜치 |
| V14 | 2026-02 | 코인 | SMA60+hyst, DD+BL+Crash 가드 추가 | 카나리 SMA 짧고 휩쏘 잦음 |
| V15 | 2026-03 초 | 주식 | 유니버스 R7(+VNQ), Z-score 4트랜치(Sh63) | 단기 Sharpe 노이즈 |
| V16 | 2026-03 중 | 코인 | Mom30 도입 | 단일 모멘텀 과적합 |
| V17 | 2026-03 말 | 주식 | Z-score Top3(Sh252) + VT Crash | V25 로 대체 |
| V18 | 2026-03 말 | 코인 | SMA50+1.5%hyst, Greedy Absorption, EW+33%Cap | 단일 D봉 한계 |
| V19 | 2026-04 초 | 선물+자산배분 | 선물 d005 4전략 EW + 60/25/15 배분 + 8pp 밴드 | 동적 3~5x, stop -15% |
| V20 | 2026-04-13 | 코인 | D_SMA50 + 4h_SMA240 50:50 EW 라이브 앙상블 | 2멤버 D+4h |
| V21 | 2026-04-17 | 코인+선물+배분 | 코인 D봉 3멤버 1/3 EW. 선물 L3 4h 3전략 고정 3x | V24/V25 로 대체 |
| V22 | 2026-04-27 | 전체 | 코인/선물 1D+4h 2멤버 EW, 주식 snap 3트랜치 stagger | V24 로 대체 |
| **V24** | **2026-04-30** | **전체** | **모든 자산 1D 단일 + drift trigger. 코인현물 D_SMA42 n=7 snap=217 drift=0.10, 카나리 SMA42±1.5%** | **(현물 현재 운영)** |
| **V25** | **2026-05-28** | **선물** | **동적 per-coin L=min(BTC_cap,K2), CROSSED 마진, snap=95 n=5 drift=0.03** | **(선물 현재 운영)** |
| **V25** | **2026-05-29** | **주식** | **R7(VNQ 교체), EEM SMA200±0.5%, Z-score+3-mom(30/72/230) 필터 → Top3 cap+7%Cash, 3트랜치, drift 0.05** | **(주식 현재 운영)** |

---

## V12 (2026-01) — 초기 단순화

- 코인: 단일 SMA 카나리 + 모멘텀 Top N, 월간 1회 리밸런싱
- 주식: SPY/QQQ + GLD 단순 비중, 단일 트랜치
- 결과: 횡보장에서 휩쏘, 베어장에서 보호 부족

## V14 (2026-02) — 코인 가드 강화

- 코인 카나리 SMA60 + hysteresis
- 추가 가드: DD exit (60d, 25%), Blacklist (-15% 7d), Crash cooldown
- 결과: MDD 개선되었으나 카나리 hyst 0% → 휩쏘 여전

## V15 (2026-03 초) — 주식 R7 + 4트랜치

- 주식 유니버스 7종 ETF (SPY, QQQ, VEA, EEM, GLD, PDBC, VNQ)
- Z-score 선정 (12M mom + Sharpe 63d)
- 4트랜치 (Day 1/8/15/22) — 타이밍 리스크 분산
- 결과: Sharpe 일관성 부족 (Sh63 단기 노이즈)

## V16 (2026-03 중) — 코인 Mom30

- 코인 모멘텀 윈도우 Mom30
- 결과: 단일 윈도우 과적합 의심, plateau 불충분

## V17 (2026-03 말) — 주식 확정

- Z-score Top 3 (12M mom + Sharpe 252d로 확장)
- VT crash filter (-3% daily → 최소 3일 + VT>SMA10 회복)
- EW 33% per slot, 4트랜치
- 거래비용 0.2% 보수
- 백테스트: Sharpe 1.255, CAGR +13.3%, MDD -11.4%, σ(Sh) 0.019
- 채택 사유: 10-anchor 평균 일관, plateau 넓음, MDD 우수

## V18 (2026-03 말) — 코인 단일 D봉 확정 (이후 V19 잠시 사용)

- 카나리 SMA50 + 1.5% hyst
- Greedy Absorption (cap 33%, 초과분 다음 순위로 흡수)
- EW + 33% cap, Top 5
- DD 60d/-25%, Blacklist -15%/7d, gap exclusion
- backtest_official.py로 V12~V19 비교 가능
- 한계: 단일 D봉 → 진입 타이밍 한 점, 4h 단위 시장 변동 무시

## V19 (2026-04 초) — 선물 도입 + 자산배분 확정

- 선물 d005 4전략 EW (25%씩):
  - 4h_d005 (SMA240, Mom20/720, daily vol 5%, snap60)
  - 2h_S240 (SMA240, Mom20/720, bar vol 60%, snap120)
  - 2h_S120 (SMA120, Mom20/720, bar vol 60%, snap120)
  - 4h_M20  (SMA240, Mom20/120, bar vol 60%, snap21)
- 레버리지: cap_mom_blend_543_cash (3/4/5x 동적, CASH≥34% 시 floor 3x)
- 스탑: prev_close -15%, cash_guard
- 자산배분: 주식 60% / 현물 25% / 선물 15%, 8pp drift band
- 5.5년 백테스트: Sharpe 2.08, CAGR +227%, MDD -34%, Cal 6.69 (선물 단독)
- 통합 포트: Sharpe 2.12, CAGR +39%, MDD -12.2%, Cal 3.21
- PFD 제거 (post_flip_delay 5→0, 포트폴리오 레벨에서 무차별 확인)

## V20 (2026-04-13) — 코인 멀티 인터벌 앙상블

배경: V19까지의 코인 엔진은 단일 D봉 + 월간 앵커. 4월 그리드서치(D/4h/2h/1h 1620조합)에서 D와 4h가 사실상 동률로 1위 (Sharpe ~1.85), 2h/1h는 노이즈로 열위 확인. 두 봉 주기를 앙상블로 결합하면 이벤트 탈동기화로 MDD 추가 개선 가능.

변경:

- 단일 엔진 → 라이브 앙상블 엔진(`trade/coin_live_engine.py`)
- 멤버 1: D_SMA50 (SMA50, Mom30/90, snap 30봉 × 3 stagger, gap-15%/excl 30일)
- 멤버 2: 4h_SMA240 (SMA240, Mom30/120, snap 60봉 × 3 stagger, gap-10%/excl 10일)
- 공통: 카나리 BTC vs SMA + 1.5% hyst, mom2vol(vol_cap 5%), Top5/cap 33%
- 50:50 EW 합산, Cash buffer 2%
- 월간 앵커 1/11/21 폐기 → 봉 단위 stagger
- DD/BL 폐기 → gap threshold + exclusion days
- 상태 스키마 변경: tranches → members, last_flip_date → bar_counter/snap_id
- Upbit warning/delisting delta 알림 (set 비교, 스팸 방지)
- 실행: cron 매시간 :05, bar-idempotency

V19 호환: 표현 불가. backtest_official.py(legacy)는 V12~V19 재현용으로 유지, V20은 `run_current_coin_v20_backtest.py` 전용.

## V21 (2026-04-17) — 10x 그리드 재설계 + 선물 L3 고정 + 가드 제거

배경: V20 이후 dense grid(연속 SMA 값)에서 과적합 의심. 엄격 10배수 그리드로 Phase-1~4 재실행. True blind holdout(train 2020.10~2023.12 / holdout 2024.01~2026.04)으로 선택편향 정량화. L2/L3/L4 sub-period rank-sum 비교. AI 3자(Claude+Gemini+Codex) 검토.

변경:

- 코인 V20 → V21: 2멤버(D+4h) → **3멤버 D봉 1/3씩 EW** (ENS_spot_k3_4b270476)
  - D_SMA50 / D_SMA150 / D_SMA100, 모두 Mom20 계열, daily vol 5%, snap 90봉
  - 4h 로직 완전 제거
  - Cron 매시간 → 일 1회 09:05 KST
- 선물 V19 d005 4전략 → **V21 L3 3전략** (ENS_fut_L3_k3_12652d57)
  - 4h_S240_SN120 / 4h_S240_SN30 / 4h_S120_SN120 (전부 4h봉)
  - 고정 3배 레버리지 (동적 3~5x 폐기), 가드 없음 (stop_kind=none, cash_guard 제거)
  - `sync_stop_orders()`에 `STOP_PCT<=0` early return 추가 (Codex 지적 버그 수정)
  - Cron 매시간 → 4h마다 6회
- 자산배분 V19 60/25/15 → V20 60/35/5 → **V21 60/40/0 sleeve r30**
  - 선물 0%에서 시작 (수동 이동 대기)
  - 밴드 abs 8%p → sleeve r30 (weight × 30%, 최소 2%p)
  - `recommend_personal`이 밴드 초과 시 텔레그램 알림, 자동 리밸 없음

채택 근거 요약:
- Phase-2 plateau 통과율: 10x 49% vs dense 26% (진짜 plateau)
- 3-anchor OOS Cal_mean: 10x 60/35/5 abs15 2.91 vs dense 2.50 (+16%)
- Holdout Cal: 전 후보 1.0 초과 (BTC buy&hold 0.53 대비 2.5~3.3배)
- Sub-period rank-sum: L3가 상위 3위 독점 (60/35/5, 60/30/10, 60/25/15 L3)
- AI 3자 합의: 60/30/10 L3 sleeve 추천 (Cal 3.41 / CAGR 43.2% / MDD -12.7%)
- 현물 앙상블 AI 3자 만장일치: k3_4b270476 (SMA50+100+150 D봉)

남은 우려 (기록):
- Holdout(2024~2026)이 상승장 위주 → 진짜 bear OOS 부재
- Holdout 보고 최종 1개 선택 시 data reuse (blindness 훼손)
- 가드 없음 tail risk (코로나 빔/루나 같은 전방위 붕괴)
- 포트폴리오 레벨 시스템 서킷브레이커 미도입

상세: [`../../history/V21_HISTORY.md`](../../history/V21_HISTORY.md)

---

## 자산배분 결정 흐름 (V19 → V21)

```
2026-04-05 — 4전략 ablation + dynamic 방법론 비교 (4,928조합)
  → InvVol/카나리/밴드 후보 모두 검토
  → 결론: 단순 EW + 8pp drift band가 가장 robust
  → 카나리 레짐 전환(자산간 강제 이동) 기각 — 사용자 선호 (자산 내부 방어에 맡김)

2026-04-06 — V12~V19 전 버전 portfolio backtest로 검증
  → V19 + 60/25/15 배분 채택
  → PFD ablation: 포트폴리오 레벨 무차별 → 제거

2026-04-13 — 코인 V20으로 교체 (배분 비율은 유지)

2026-04-17 — V21 전환:
  - 10x 그리드 재설계 (phase1_10x~phase4_10x)
  - True blind holdout 검증
  - Leverage L2/L3/L4 sub-period ranksum: L3 1~3위 독점
  - 현물 앙상블 k3_4b270476 고정 (AI 3자 만장일치)
  - 선물 L3 12652d57 고정 3x, 가드 없음
  - 배분 60/40/0 sleeve r30, 리밸런싱 수동
```

---

## Strategy C (2026-04-20) — Dip-buy 보조 슬리브 (V21 동일계정)

V21 (추세추종 롱) 이 놓치는 단기 급락 반등을 잡아 V21 성과를 보조.
V21 우선순위 유지 + V21이 안 쓴 cash에서만 3x 레버리지(선물)/1x(현물) dip-buy 동작.

### 시그널 (1h bar, 공통)
- `dip_pct = Close_t / Close_{t-dip_bars} - 1`
- `dip_sig = dip_pct ≤ dip_thr` 성립 시 t+1 bar open 롱 진입
- 청산: TP 도달 또는 tstop 시간 경과 (stop-loss 없음)
- 유니버스: 시총 Top 15, n_pick=1, swap_edge=1

### 현물 C (1x, cap=0.333)

| 파라미터 | 값 |
|---|---|
| dip_bars | 24 |
| dip_thr | -0.20 |
| tp | 0.04 |
| tstop | 24 |
| cap_per_slot | 0.333 (V21 현금 여유 33%까지 진입) |

성과 (V21 현물 단독 대비):
- 전구간: Cal 3.10 → 3.96 (+28%), MDD -19% → -16%
- Holdout 2024+: Cal 1.75 → 2.32 (+33%)

### 선물 C (3x, cap=0.12 최종)

| 파라미터 | 값 |
|---|---|
| dip_bars | 24 |
| dip_thr | -0.18 |
| tp | 0.08 |
| tstop | 48 |
| cap_per_slot | 0.12 |
| leverage | 3.0 (V21 선물과 동일) |

cap 0.12 선정 이유: cap 0.03 ~ 0.333 전범위 테스트 결과 전구간 Cal 3.98 (최고)이며 Train MDD -37.7%로 V21 단독(-43%)보다 **완충 효과**. 0.15 이상부터 MDD 악화.

성과 (V21 선물 단독 대비):
- 전구간: Cal 2.96 → 3.98 (+34%), MDD -48% → -38%
- Holdout 2024+: Cal 1.29 → 1.75 (+36%)
- corr(C_contrib, V21) = -0.48 ~ -0.17 (음상관, 독립 알파)

### 검증 완료 (22개 테스트)
- Walk-forward 5 splits 전원 Cal > 1.6 유지
- Cross-anchor 5 start dates 전원 Cal > 1.9
- Bootstrap holdout: CAGR p5 +0.2%, p50 83%, MDD worst -77%
- Top N event 제거 (~20): Cal > 1.9 유지 (희소성 의존 없음)
- BTC regime 분해: StrongBear 장세에서 C가 V21 보완 (fut +201pp ann ret)
- Intrabar MAE 근사 liquidation: Full Cal -2.5% 감소 (미미)
- Funding fee 반영: 영향 ≈ 0 (C 보유 짧음)
- Portfolio Circuit Breaker: cap 0.12에선 불필요 (자체 방어)
- Parameter 재조정 시도 (n_pick, dip_bars) 전부 과적합으로 기각

### 실전 투입 사항
- Shadow 무의미 (발동 빈도 연 17회, 2-4주 shadow는 통계 무의미)
- 바로 cap 0.12로 소액 실전 투입 가능
- 잔여 실전 리스크: 주문 충돌, 부분체결, 호가 유동성 (shadow/실전에서만 확인)
- 3~6개월 안정 운용 후 cap 상향 검토

### 구현
- 엔진: `strategies/cap_defend/research/m3_engine_final.py` (현물), `m3_engine_futures.py` (선물)
- 신호: `strategies/cap_defend/research/c_engine_v5.py`
- 검증: `strategies/cap_defend/research/c_tests_v2/`
- 실매매: V22 에서 현물만 채택 (선물 C 는 2022 bear 악화로 보류)

---

## V22 (2026-04-21) — 현물 C 슬리브 실전 투입 (champion 재튜닝)

V21 대비 변경: 현물에 Strategy C 슬리브 추가. 선물/주식은 V21 그대로.

### 추가 검증 경과 (c_tests_v3/)
Phase A/B/C/C2/C3 로 dip_thr/가드/tp/tstop/universe 그리드 재탐색. 기존 C 파라미터 대비 champion 확인:

| 자산 | champion 파라미터 | Holdout Cal | Δ vs V21+C |
|---|---|---|---|
| 현물 | dip_thr -0.12, tp 0.03, tstop 24, A2_bounce_w1, cap 0.333 | 3.24 | +40% |
| 선물 | dip_thr -0.14, tp 0.10, G3(A2+B2), cap 0.30 | 4.06 | +132% |

2022 bear / 2025 Q1 adverse 구간 추가 검증 결과:
- 현물 champion: 전 구간 Cal/MDD 개선 (2022 bear Cal -0.03 → +1.57).
- 선물 champion: Holdout 은 개선이나 2022 H1 MDD 가드의 역효과로 -26% → -41% 악화.
→ 선물은 보류, 현물만 투입.

### 현물 V22 C 슬리브 (champion, 실전)

| 파라미터 | 값 |
|---|---|
| dip_bars | 24 |
| dip_thr | -0.12 |
| tp | 0.03 |
| tstop | 24 |
| cap_per_slot | 0.15 (실전 초기, 백테 champion 0.333) |
| 가드 | A2_bounce_w1 (시그널 봉 양봉 후 다음 봉 Open 진입) |
| 실매매 | V21 우선 + V21 안 쓴 cash 에서 C 동작 |

### V22 아키텍처 (intent/merge/finalize 3단계)
- `compute_c_intent(state, bars_1h, universe, now) → CIntent` 주문 X
- `apply_c_to_target(v21_target, c_position, c_intent, total_pv) → merged_target`
- `finalize_c_state(state, intent, fill_result)` 체결 후 state 갱신
- `handle_c_only(...)` V21 skip 경로에서 C 단독 체결

V21 trade path 에서도 merged target 을 execute_delta 에 사용해 C 포지션을 stray 로 매도하지 않도록 보호. (Codex 2차 리뷰 critical 반영)

### 실전 배포 상태 (2026-04-21)
- cron: `5 * * * *` (매시간 :05, 기존 1회/일에서 확장)
- 서버 배포: `trade/coin_live_engine.py`, `trade/executor_coin.py`
- 초기 관측: 5회 cron 정상, dip 조건 미충족 (hold)
- 실전 cap 상향 로드맵: 0.15 → 0.20 → 0.25 → 0.333 (1~3개월 간격)

### 구현
- 실매매: `trade/coin_live_engine.py` (C_SLEEVE_CFG, CIntent, compute_c_intent, apply_c_to_target, fetch_c_bars)
- 실매매: `trade/executor_coin.py` (handle_c_only, finalize_c_state, _market_buy_krw/_market_sell_coin)
- 매뉴얼: V22 운영 매뉴얼 (2026-08-25 삭제 — 내용은 git 이력에 있고 현행 매뉴얼은 `../../OPERATION_MANUAL.md`)

### V22 미채택 (추가 검증 후 보류)
- 선물 C 업그레이드 (f_dthr14 + G3): 2022 bear MDD 악화
- dip_thr-only 완화 (선물): 22건 검증 범위 밖
- BTC regime 가드 (SMA200): V22 목표 단순성과 충돌

---

## V24 (2026-04-30) — 모든 자산 1D 단일 + drift 트리거

V22 의 4h 멤버를 전부 걷어내고 자산군마다 D_SMA42 하나만 남겼다. 대신 종목 교체가 앵커일에만
일어나던 구조에 "목표에서 얼마나 벌어졌나"(drift) 를 트리거로 더했다.

- 코인현물: D_SMA42 단일, snap_interval=217 (=7×31), n_snap=7, drift 0.10, 카나리 BTC SMA42±1.5%,
  헬스 mom2vol(vol_cap 5%, 90d), universe 3종 cap 1/3.
- 선물(당시): D_SMA42 단일, snap=57 n=3 drift=0.05 → V25 에서 95/5/0.03 으로 갱신.
- 주식: SNAP_PERIOD 126→69, STAGGER 42→23, N_SNAPS 3 유지.
- 스태거는 자산군마다 서로 다른 소수(주식 23 / 현물 31 / 선물 19), 스냅 개수는 서로소로 두어
  같은 날 세 자산이 동시에 갈아타지 않게 했다.
- cron 4h×6 → 1d×1 (09:05 KST).
- drift 정의: `half_turnover = Σ|tgt − cur| / 2`, 자본금 기준. `need_rebal = is_daily_bar AND (snap_fire OR (canary_on AND ht ≥ threshold))`.
- 현물 drift 발화 시 refill v2 (모멘텀 둘 다 음수인 코인만 교체). 2026-06-06 검증 결과 drift 0.10 에서는
  refill 이 앵커 전용과 사실상 동일(5.4yr 종목교체 0일) — 미래의 모멘텀 급락 대비 방어로만 남겼다.
- 단독 sleeve BT(5.8yr, 2020-10-01~2026-08-24, 라이브 체결규칙 정합): 현물 Cal 4.35 / CAGR +77.6% / MDD -17.8%.
  옛 표기 4.63/+82% 는 BT 에만 있던 5% 체결 밴드 기준 — 2026-08-25 에 밴드 0 으로 정합하며 정정.

## V25 선물 (2026-05-28) — 동적 per-coin 레버리지 + CROSS

고정 L3 ISOLATED 를 코인별 동적 레버리지와 CROSS 마진으로 바꿨다.

- L = min(BTC_cap, K2_per_coin), Lmin/Lmid/Lmax = 2/3/4.
  · BTC_cap: BTC/SMA42 > 1.05 → 4, > 1.015 → 3, else 2 (시장 전체 상태)
  · K2_per_coin: close/SMA7 > 1.075 → 4, > 1.025 → 3, else 2 (개별 코인 단기 강도)
- 마진 CROSSED, 스냅 95/5, drift 0.03, 스탑·캐시가드 없음.
- 채택 근거: K2(SMA 7, 문턱 2.5%) 가 25개 설정 window rank-sum 1위이자 plateau 중심.
  모멘텀 기반 대안(J) Cal 7.45 대비 K2 8.12, MDD 7pp 개선.
- 단독 sleeve BT(5.6yr): Cal 8.12 / CAGR 312% / MDD -38.3%. 자산배분 60/25/15 합성 Cal 5.72.

## V25 주식 (2026-05-29) — Z-score + 3-mom 필터 + cap/Cash

- 유니버스 R7 (R7B 의 EWJ 를 VNQ 로 교체 — 채택 BT 와 일치시킴).
- 선정: Z-score(가중Mom + Sharpe126) 랭킹 → 3-mom(30/72/230) 필터 → Top3 cap 1/3 + Cash.
  · 가중Mom = 0.5×ret63 + 0.3×ret126 + 0.2×ret252.
- 카나리 EEM SMA200 ±0.5%, 드리프트 0.05, 스냅 69/23/3.
- 종목 교체는 앵커일 OR drift 발화일(그날 fresh 재선정). 코인·선물의 "앵커일에만" 원칙과 다른 유일한 곳.
- 아키텍처 통일: 순수 전략 함수 `stock_strategy_v25.py` 를 executor 와 recommend 가 함께 호출(단일 출처).
  signal_state.json 은 fallback 으로만 남았다.
- 채택 근거: window rank-sum 5게이트 통과(C안 avg_rank 1.246 vs 현행 2.471).

## V25 이후 (2026-06 ~ 2026-08) — 파라미터가 아니라 정합·가드의 시간

버전이 올라가지 않은 기간이지만 실제로 손을 댄 곳이 많다. 전략 정의는 그대로 두고
"라이브가 채택 백테스트와 같은 일을 하는가" 를 맞추는 작업이었다. 항목별 근거는 `../../history.md`.

- 주식 선정 점수 정합(06-06): 라이브가 순수 252일 모멘텀(V15 잔재)을 쓰고 있어 채택 BT 의 가중 모멘텀과
  종목이 16.8% 어긋났다 → 가중으로 수정, replay 100% 일치. 함수 이름이 아니라 구현을 봐야 한다는 사례.
- 코인 현물·선물 선정 패리티 증명(06-06): 같은 입력을 주입해 라이브와 BT 의 일별 종목·비중이
  2,000여 일 100% 일치함을 확인(`research/parity_spot.py`, `parity_fut.py`).
- 선물 드리프트 정의 정합(07-17): cur_w 를 (진입마진+PnL)/equity 로 통일.
- 헬스 vol_cap 0.05 양방향 재검증(07-21): 완화도 타이트화도 모두 열위 — 재실험 대상 아님.
- 선물 전량청산 차단 사고 수정(08-15), 마진 preflight 범위 수정과 상시 무결성 검사 분리(08-21~22).
- 체결 밴드 정합(08-25): 라이브 ±1% → ±5% 로 채택 BT 와 일치. 회전 연 264회 → 156회.

---

## 폐기된 아이디어와 사유

| 아이디어 | 시점 | 폐기 사유 |
|---|---|---|
| DD entry filter | 2026-03 | 과적합, sharp peak |
| 카나리 레짐 전환 (자산간 이동) | 2026-04 | 사용자 선호 + 백테스트 차이 미미 |
| 2h/1h 봉 추가 멤버 | 2026-04 | 노이즈, 동일 universe/canary로 직교성 약함 |
| TLT 방어 추가 | 2026-04 | V19 대비 한계효용 낮음 |
| Post-Flip Delay (PFD) | 2026-04 | 포트폴리오 레벨 무차별 |
| 단일 D봉 코인 (V18 유지) | 2026-04-13 | 4h 결합으로 이벤트 탈동기화 이득 (V24 에서 다시 단일 D봉으로 회귀) |
| 선물 유지증거금 tier 반영 | 2026-06 | 동적 L4 + CROSS 에서 0.4~0.65% 차이는 청산위험 dead zone |
| 자산간 T3O(과대 트림) 트리거 | 2026-06 | CAGR 희생이 채택바 미달, 임계 올려도 개선 평탄 |
| 헬스 vol_th 재조정(0.03~0.04 / 0.06+) | 2026-07-21 | 타이트화는 유니버스 축소로 CAGR 붕괴, 완화는 고변동 알트 편입으로 Cal 붕괴 |
| 카나리 재진입 분할진입(스냅별 stagger) | 2026-07-23 | 현물 Cal 4.55→3.29, 선물 7.58→4.01, MDD 도 악화 |
| 선물 매일 레버리지 재조정 | 2026-08-21 | sleeve rank-sum·비용 스트레스 열위, 작동 조건에선 MDD 악화 |
| 앵커 갱신을 drift 사건구동으로 | 2026-08-21 | 드리프트는 승자 확대로 발화 → 그때 재선정하면 승자를 조기 절단 |
| 체결 밴드 1~3% 로 좁히기 | 2026-08-25 | 수익 동률·낙폭 악화·체결 1.7배, 비용 스트레스 전 구간 열위 |

---

## 채택 기준 (모든 버전 공통)

1. 10-anchor 평균 + σ(Sharpe) 낮음 (0.1 이하 robust)
2. 파라미터 plateau 존재 (인접값 성과 유사)
3. 다기간(2018~/2019~/2021~) 일관성
4. 거래비용 반영 후에도 개선 유지
5. 실매매 엔진으로 상태전이 재현 가능
