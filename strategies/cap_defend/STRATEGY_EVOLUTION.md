# 전략 진화 (V12 → V21)

이 문서는 Cap Defend 전략의 버전별 변경점과 결정 근거를 정리한다. 자세한 백테스트 결과는 각 버전의 백테스트 코드 또는 [research/](./research/)의 결과 파일을 참조한다. V21 개발 중 실험은 [`../../V21_HISTORY.md`](../../V21_HISTORY.md) 별도 문서 참조.

---

## 한눈에 보기

| 버전 | 시점 | 자산군 | 핵심 변경 | 비고 |
|---|---|---|---|---|
| V12 | 2026-01 | 코인+주식 | 초기 합본 (단순 모멘텀 + 카나리) | 코인 과적합, 주식 단일트랜치 |
| V14 | 2026-02 | 코인 | SMA60+hyst, DD+BL+Crash 가드 추가 | 카나리 SMA 짧고 휩쏘 잦음 |
| V15 | 2026-03 초 | 주식 | 유니버스 R7(+VNQ), Z-score 4트랜치(Sh63) | 단기 Sharpe 노이즈 |
| V16 | 2026-03 중 | 코인 | Mom30 도입 | 단일 모멘텀 과적합 |
| V17 | 2026-03 말 | 주식 | Z-score Top3(Sh252) + VT Crash → 확정 | (주식 현재 운영) |
| V18 | 2026-03 말 | 코인 | SMA50+1.5%hyst, Greedy Absorption, EW+33%Cap | 단일 D봉 한계 |
| V19 | 2026-04 초 | 선물+자산배분 | 선물 d005 4전략 EW + 60/25/15 배분 + 8pp 밴드 | 동적 3~5x, stop -15% |
| V20 | 2026-04-13 | 코인 | D_SMA50 + 4h_SMA240 50:50 EW 라이브 앙상블 | 2멤버 D+4h |
| **V21** | **2026-04-17** | **코인+선물+배분** | **코인: D봉 3멤버 1/3 EW (SMA50/150/100). 선물: L3 4h 3전략 고정 3x, 가드 없음. 배분: 60/40/0 sleeve r30** | **(현재 운영)** |

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

상세: [`../../V21_HISTORY.md`](../../V21_HISTORY.md)

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
- 매뉴얼: `V22_OPERATION_MANUAL.md`

### V22 미채택 (추가 검증 후 보류)
- 선물 C 업그레이드 (f_dthr14 + G3): 2022 bear MDD 악화
- dip_thr-only 완화 (선물): 22건 검증 범위 밖
- BTC regime 가드 (SMA200): V22 목표 단순성과 충돌

---

## 폐기된 아이디어와 사유

| 아이디어 | 시점 | 폐기 사유 |
|---|---|---|
| DD entry filter | 2026-03 | 과적합, sharp peak |
| 카나리 레짐 전환 (자산간 이동) | 2026-04 | 사용자 선호 + 백테스트 차이 미미 |
| 2h/1h 봉 추가 멤버 | 2026-04 | 노이즈, 동일 universe/canary로 직교성 약함 |
| TLT 방어 추가 | 2026-04 | V19 대비 한계효용 낮음 |
| Post-Flip Delay (PFD) | 2026-04 | 포트폴리오 레벨 무차별 |
| 단일 D봉 코인 (V18 유지) | 2026-04-13 | 4h 결합으로 이벤트 탈동기화 이득 |

---

## 채택 기준 (모든 버전 공통)

1. 10-anchor 평균 + σ(Sharpe) 낮음 (0.1 이하 robust)
2. 파라미터 plateau 존재 (인접값 성과 유사)
3. 다기간(2018~/2019~/2021~) 일관성
4. 거래비용 반영 후에도 개선 유지
5. 실매매 엔진으로 상태전이 재현 가능
