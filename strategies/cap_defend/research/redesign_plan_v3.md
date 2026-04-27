# V21 재설계 통합 플랜 v3 (주식 + 선물 + 현물) — FINAL

작성: 2026-04-24
합의: Gemini + Codex 라운드2 만장일치 → 라운드11 Codex 최종 Go (Gemini r8+ 쿼터 소진)

## 0. 배경

- 선물 phase0: SN 교체 / phase offset 기각, base Cal 3.10 은 phase=0 lucky anchor overfit
- 현물 phase0c: 유사 overfit 의심
- 주식 V17 (tx 0.25%) baseline Cal 0.41 — V21 style 3-snapshot ensemble 로 재설계
- V21 현재 운영: 코인 현물 + 선물 L3 모두 가드 제거, 앙상블 분산이 유일 방어

## 1. 사용자 확정 방침 (변경 없음)

1. 홀드아웃 없음 (yearly consistency 로 대체)
2. start date sweep 없음 (yearly 가 커버)
3. univ=3 V21 스펙 통일
4. Ensemble: 동일 snap_interval + 동일 interval 내에서만 조합
5. 4h+D 혼합 ensemble 배제
6. 멤버간 phase stagger 초기 미사용 (V21 n_snapshots=3 내부 분산)
7. Yearly 모든 해 (2017~2025 주식 / 2020~2025 코인)
8. 2022 bear 포함 필수
9. 주식 universe V17 실운영 고정 (SPY/QQQ/VEA/EEM/GLD/PDBC/VNQ + IEF/BIL/BNDX/GLD/PDBC)
10. 특정시점/홀드아웃 금지, 전범위 적용 가능한 범용 robustness 테스트만 (라운드9 신규)

## 2. 공통 Workflow (3 자산 동일)

### Step 0 — (주식 전용) iter_refine v2 raw 생성
- v17_snap_iter_v2.py: V17 실운영 universe 로 iter_refine 방식 그리드
- 초기 324 → zoom 확장 최대 6 iter → 수렴 후 v17_snap_v2_out/raw_combined.csv
- 코인은 이미 iter_refine/raw_combined.csv 존재 (재생성 불필요)

### Step 1 — 기존 grid V21 스펙 재평가
- 조건: univ=3, cap=1/3, n_snapshots=3, canary_hyst=0.015, health=mom2vol, stop=none, phase_offset=0
- 입력:
  - 선물: iter_refine/raw_combined.csv (fut 72,815)
  - 현물: iter_refine/raw_combined.csv (spot 48,903)
  - 주식: v17_snap_v2_out/raw_combined.csv
- 출력: redesign_univ3_raw_{fut,spot,stock}.csv
- ETA: ~12h (코인) / 주식은 Step 0 출력 그대로 사용 가능

### Step 2 — Top 500 k=1 추출
- Cal 기준 자산별 top 500
- 출력: redesign_top500_{fut,spot,stock}_k1.csv

### Step 3 — Phase robustness (snap_interval 상대비율)
- phase_ratio = {0.00, 0.07, 0.13, 0.31, 0.49}
- phase_offset_bars(i) = round(phase_ratio[i] × snap_interval_bars), 중복 제거
- Top 500 × 5 phase = 2500 backtest / 자산
- 지표: med_Cal, p20_Cal, Cal_CV (phase 축)
- 필터 (복합):
  - med_Cal ≥ top500 Cal median (자산별)
  - p20_Cal ≥ 자산별 p30
  - Cal_CV ≤ 자산별 p50

### Step 3.5 — Snap cadence nudge (soft gate)
- 대상: Step 3 survivor
- nudge points: prime×3 3~5개 거리순 확장
  - snap=30 → 33/39/51/57/69
  - snap=90 → 57/69/87/93/111
  - snap=120 → 111/123/129
- nice number (30/60/90/120 등 3배수이지만 prime×3 아닌 것) 에 nice_penalty +5 rank
- snap_CV > 자산별 p70 이면 penalty +10
- 지표: snap_nudge_Cal_CV

### Step 3.6 — Parameter plateau check (라운드9 신규)
- Top 25 rank cfg 대상
- axis-wise perturbation: coin(sma/ms/ml/snap), stock(snap/canary_sma/def_mom) × {-20%, -10%, +10%, +20%}
- 지표: neighborhood_Cal_med, neighborhood_p10_Cal, neighborhood_worst_Cal, neighborhood_Cal_CV
- Soft rank: R_plateau_CV (낮을수록 좋음), R_plateau_p10 (높을수록 좋음)
- 의도: 단일 파라미터 peak (lucky) 배제, plateau 위 확인

### Step 4 — Yearly consistency (전 년도)
- 기간: 주식 2017~2025 (9년), 코인 2020~2025 (6년)
- 지표:
  - yearly_med_Cal, worst_year_Cal, positive_years_ratio
  - yearly_CAGR, yearly_MDD (bootstrap 입력)
  - yearly_rank_mean / yearly_rank_worst (Borda 연간 rank, 라운드9 신규)
- 2022 Hard Gate: yearly_Cal_2022 ≥ 자산별 p40 (코인) / 0.3 (주식) 필수
  - yearly_2022 컬럼 결측 시 FAIL

### Step 4.5 — Yearly block permutation diagnostic (라운드9 보조)
- yearly CAGR N=200 random permutation (fixed seed=42)
- bootstrap_Cal_p10, bootstrap_Cal_med, bootstrap_MDD_p10
- rank 축 미포함 (Codex r10: CAGR permutation 은 total CAGR 에 영향 없어 진단력 제한)
- report 진단 표시만

### Step 5 — Rank-Sum 최종 랭킹
- 랭크 축 (라운드9 확장, 라운드10 R_bootstrap 제거):
  - R_phase — phase_med_Cal (높을수록)
  - R_phase_CV — phase_CV (낮을수록)
  - R_snap_CV — snap_nudge_CV (낮을수록)
  - R_yearly — yearly_med_Cal (높을수록)
  - R_adverse — worst_year_Cal (높을수록, 2022 포함)
  - R_positive — pos_ratio (높을수록)
  - R_yearly_rank — yearly_rank_mean (낮을수록, Borda)
  - R_turnover — phase_rebal_med (낮을수록)
  - R_plateau_CV — neighborhood_Cal_CV (낮을수록)
  - R_plateau_p10 — neighborhood_p10_Cal (높을수록)
  - R_tx_stress — tx_2x_ratio (Step 7 2-pass 재실행 시만)
- Nice number snap: +5 penalty
- snap_CV p70 초과: +10 penalty
- Composite rank_sum = Σ R_* + penalty
- Hard gate (2022) 통과 후 rank_sum 낮은 순

### Step 6 — Ensemble 탐색 (k=2/3)
- 후보 풀: Step 5 top 20~30
- 필터 (고정 threshold, 라운드10 확정):
  - 동일 snap_interval + 동일 interval 버킷
  - corr(daily returns) < 0.7
  - bad_day_overlap < 0.8
  - joint_2022_loss ≥ -0.20 (코인) / -0.10 (주식)
  - joint_max_DD ≥ -0.50
- 채택 기준:
  - k=1 대비 Cal / MDD / Sh 중 2개 이상 strict improvement
  - improve_pass = True + gates_pass = True
- 최종 report 는 status=ok AND gates_pass AND improve_pass 3중 AND

### Step 7 — Final stress test
- 5 시나리오 (3자산 공용):
  - baseline, tx_1.5x, tx_2.0x, delay_1bar, drop_top
- drop_top: trace 기반 period-weighted 누적 weight 1위 자산 exclude (2-pass)
- 주식 drop_top 엔진 정식 지원 (라운드9 업그레이드)
- redesign_stock_drop1.py: 엔진 drop_top 과 Top 3 결과 비교 검증용 유지

### Step 8 — 리포트
- CSV: 자산별 rank csv 20~30개
- Report (md): 자산별 8~12개
  - Primary 3 (stress 전 시나리오 pass 후보 중 rank-sum top 3)
  - Backup 5~9
  - Ensemble 후보 (gates+improve 3중 필터 통과)
  - Rejected-but-interesting (stress 열화 큰 고성과)

## 3. 스크립트 구조

공통 helper
- redesign_common.py: parse_cfg, run_bt (with_equity/exec_delay/drop_top), status_resume_keys, is_prime_x3_snap, NICE_SNAPS
- redesign_stock_adapter.py: V17 universe SP + run_snapshot_ensemble + metrics 변환

파이프라인 스크립트
- Step 0: v17_snap_iter_v2.py (주식만)
- Step 1: redesign_rebuild_univ3.py (fut+spot)
- Step 2: redesign_extract_top500.py
- Step 3: redesign_rerank_phase.py + redesign_filter_phase.py
- Step 3.5: redesign_snap_nudge.py
- Step 3.6: redesign_plateau.py
- Step 4: redesign_yearly.py
- Step 4.5: redesign_bootstrap.py
- Step 5: redesign_analyze.py (rank-sum + ensemble 후보 메타)
- Step 6: redesign_ensemble_bt.py (tail co-crash + improve_pass)
- Step 7: redesign_stress.py (5 시나리오 3자산 공통)
- Step 7 보조: redesign_stock_drop1.py (엔진 drop_top 과 비교)
- Step 8: redesign_report.py

Driver
- run_full_pipeline.sh: v17 완료 대기 → 주식 Step 2~8 (3-pass analyze) → rebuild → 선물/현물 Step 2~8

엔진 수정 완료 (2026-04-24)
- unified_backtest.py
  - phase_offset_bars (backward-compat)
  - _equity in result
  - execution_delay_bars (pending_rebalance queue)
  - exclude_assets (drop_top 2-pass)
  - _trace 인자 (매 리밸 기록)
- stock_engine_snap.py (라운드9)
  - execution_delay_bars / exclude_assets / _trace 3개 인자
  - compute_target_with_canary: offensive+defensive 모두 exclude 사전 필터 (dataclasses.replace)
  - canary_assets 은 보호 (신호 유지)
- redesign_common.run_bt
  - stock drop_top 경로: trace period-weighted (weight × duration days)
  - coin drop_top: unified_backtest _trace 기반 cumulative weight

## 4. Rank 2-pass / 3-pass 절차

- pass 1: rank1 = analyze 첫 실행 (plateau/bootstrap/stress 없이)
- pass 2: plateau + bootstrap 완료 → analyze 재실행 = rank2
- pass 3: stress 완료 → analyze 재실행 = rank3 (R_tx_stress 포함)
- 최종 rank = rank3

Driver 순서
rank1 → plateau → bootstrap → rank2 → ensemble_bt → stress → rank3 → report

## 5. Smoke test 검증 (2026-04-24)

엔진 레벨
- fut baseline Cal 9.43, exec_delay=1 Cal 11.24, drop_top Cal 6.82 (BTC dropped)
- stock baseline Cal 0.59, delay_1 Cal 0.53, drop_top Cal 0.43 (GLD dropped)
- stock excl IEF Cal 0.58 (defensive 차순위 대체), excl SPY Cal 0.65

파이프라인 레벨 (mini 5 cfg × 3자산)
- Step 2~8 + 3.6/4.5 전 스크립트 exit 0
- 빈 plateau/bootstrap csv EmptyDataError 가드 추가
- report md 정상 생성, 3중 필터 동작

## 6. AI 합의 이력

- 라운드 1~2: Gemini + Codex 만장일치 (v2 → v3 6개 쟁점)
- 라운드 3~7: 엔진 wiring + AI 지적 15건 반영
- 라운드 8: Gemini + Codex 4개 Conditional 조건 반영 후 Go
- 라운드 9: 스코프 확장 (A/B1/B2/B3/B4 + contribution) Codex 합의
- 라운드 10: 구현 검증, 3개 critical fix 지적
- 라운드 11: 3 fix 반영 후 Codex 최종 Go (Gemini 쿼터 소진)

## 7. 실행 ETA

- Step 0 주식: ~3~5h (iter_refine v2 수렴까지)
- Step 1 코인: ~12h (rebuild_univ3)
- Step 2~8 per 자산: ~2~4h
- 총: 약 1~2일 (병렬 불가 시 순차 18~24h)

## 8. 방침 외 제약

- Gemini 쿼터 소진 → 라운드 8 이후 Codex 단독 판정 (실무상 충분, 라운드 1~7 이중 검증 누적)
- 주식 stock_engine_snap 엔진 레벨 drop_top 정식 지원 (이전 옵션 c 폐기)
- bootstrap 은 rank 축 아닌 진단용
