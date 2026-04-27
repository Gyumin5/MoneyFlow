# V21 재설계 플랜 v1 (2026-04-24)

## 배경
- 선물 P0 결과: SN 교체 / phase offset 모두 기각. 단 phase0c 에서 base Cal 3.10 이 과적합 (lucky anchor) 확인
- 현물 phase0c 진행 중 (A축 CV ~0.13)
- 사용자 지시: 기존 그리드 재사용 + robust metric 재랭킹, k=1/2/3 다 탐색, 앙상블 강제 X
- 홀드아웃 불필요 (이미 robust 검증으로 보완), 대신 연도별 상위권 유지 테스트

## Workflow (선물 / 현물 동일)

### Step 1: 기존 그리드 top 200 추출
- 선물: research/iter_refine, research/grid_results, phase1_v2_sweep 등
- 현물: research/grid_results, research/sweep_results 등
- k 차원을 n_picks (universe_size) 로 취급. k=1/2/3 별로 top 200 씩
- 총 200 × 3 = 600 configs per 자산

### Step 2: Robust 재백테 (각 config 에 대해)
축:
- A. phase offset sweep: [0, 7, 13, 19, 31, 47, 59, 83]  (8 값)
- B. start date sweep: [2020-10-01, 2021-04-01, 2022-01-01] (3 값)
- C. yearly windows (1년 단위): 2021, 2022, 2023, 2024, 2025 (5 년)

총 per config: 8 + 3 + 5 = 16 백테
비용: 600 × 16 × 45s ≈ 120시간 (선물) / 600 × 16 × 95s ≈ 253시간 (현물)

너무 김. 축소:
- top 200 → top 50 (각 k 별)
- phase sweep 8 → 5 값
- start 3 → 2 값 (2020-10, 2022-01)
- yearly 5 → 3 (2022, 2023, 2024)
- per config: 5 + 2 + 3 = 10 백테
- 비용: 150 × 10 × 45s ≈ 19h (선물) / 150 × 10 × 95s ≈ 40h (현물)

여전히 김. 더 축소 필요.

**초기 pass (Tier 1)**:
- top 200 → A phase sweep 4 값만 [0, 13, 31, 59] 우선
- 4 × 200 × 3 (k) = 2400 backtests / 자산
- 선물 2400 × 45s = 30h → 하루치. 현물 2400 × 95s = 63h → 3일

아직 김. 병렬 활용하면 28 cores → 이론 1/28. 선물 ~1시간, 현물 ~2시간 가능.

**최종 설계**:
- Tier 1: top 200 × A축 4 값 phase sweep, 병렬 실행
- Tier 1 CV 통과한 candidates (약 100개 예상) 에 대해 Tier 2: B (start 2값) + C (yearly 3값)

### Step 3: Metric 계산 (각 후보)
Robust backtest 결과 (Tier 1+2 종합) 로:
- med_Cal (phase sweep Cal median)
- med_CAGR
- med_Cal_x_CAGR = med_Cal × med_CAGR
- p5_Cal (worst case)
- yearly_consistency: 3 yearly windows 의 Cal 중 "top 25% in each year" 빈도

### Step 4: 필터링 & 랭킹
- 필터: CV (A축) < 0.15 AND yearly_consistency ≥ 2/3 년 top 25%
- 랭킹:
  - R1 = rank by med_Cal (하위 순위 = 낮은 번호)
  - R2 = rank by med_CAGR
  - R3 = rank by med_Cal_x_CAGR
  - R4 = rank by rank_sum (R1 + R2 + R3)
- 상위 전략 합집합:
  - Top 10 by R1 ∪ Top 10 by R2 ∪ Top 10 by R3 ∪ Top 10 by R4
- 기대 최종 선정: 20~30 unique configs per 자산

### Step 5: 검토 & 채택
- 자동 CSV + report
- 사용자가 직접 값 검토 → CV threshold 조정 or 추가 기준 적용
- Ensemble 조합 탐색 (선택): 선정된 후보들 중 상관 낮은 k=2 / k=3 조합 자동 생성
- Forward walk 불필요 (yearly consistency 가 이미 equivalent)

## 스크립트 구조 (예정)

- `redesign_rerank_fut.py`: 선물 Tier 1+2 실행
- `redesign_rerank_spot.py`: 현물 Tier 1+2 실행
- `redesign_analyze.py`: CV / yearly / rank union 계산 및 리포트

## 주의
- Phase 0 에서 phase_offset_bars 인자 추가함 (선물 backtest_futures_full.py, 현물 run_current_coin_v20_backtest.py 의 initial_phases)
- yearly window 은 기간 짧아 Cal 변동성 큼 — median/중앙값 기준 비교
- top 200 selection 의 metric 이 이미 Cal (phase=0 기반) 이라 선정 편향 있음. Tier 1 에서 많이 걸러질 가능성 — 남은 candidates 로 재시도 or top 을 넓혀 500~1000 까지 확장 고려

## 진행 순서
1. 이 플랜에 대해 AI 토론 (gemini+codex), 합의 될 때까지 반복
2. 합의 후 기존 그리드 결과 경로 확인 (선물/현물 각각)
3. Tier 1 스크립트 작성 + 병렬 실행
4. 중간 결과 진행 모니터링
5. Tier 2 실행
6. 최종 리포트 + 사용자 검토

## 2026-04-24 사용자 확정 방침 (AI 토론 전 선결정)

사전 재추출 (AI 토론과 독립, 병행 실행 중)
- iter_refine raw_combined.csv 121,718 configs 를 V21 스펙 univ=3 으로 전량 재평가
- 실행: redesign_rebuild_univ3.py (PID 688217, 24 worker, ETA ~12h)
- 완료 후 redesign_top200_{fut,spot}.csv 를 univ=3 기준으로 덮어쓰기

universe_size
- V21 스펙 univ=3 고정. iter_refine 의 univ=5 는 재추출로 대체

Ensemble 구조 제약
- 동일 snap_interval_bars + 동일 interval (4h or D) 내에서만 k=2/3 조합 생성
- 4h+D 혼합 ensemble 배제 (snap 비교 불가, 관리·비용·해석 모두 불리)
- 멤버간 phase stagger 는 초기엔 안 함 (V21 내부 n_snapshots=3 이미 phase 분산)

Robustness 축 2개 병행
- Tier 1 phase sweep: 동일 snap 에서 anchor day 이동 {0, 13, 31, 59} (AI 합의 후 확정)
- Step 1.7 snap prime×3 nudge: Tier 1 생존 후보 근방에서 snap={21,33,39,51,57,69,87,93,...} 로 cadence robustness 검증
  - Tier 1 phase sweep = anchor day 이동 (cadence 동일)
  - Step 1.7 snap nudge = cadence 자체 교체 (nice number → prime×3 로 달력 decouple)
- 둘 다 통과한 것만 Tier 2 로 보냄 (CV filter 보다 엄격)

Tier 2 기간 robustness
- start sweep: [2020-10-01, 2022-01-01]
- yearly windows: {2022, 2023, 2024} — 2022 bear 포함 필수

phase_offset_bars 구현 현황
- backtest_futures_full.py: 기존 구현 존재
- unified_backtest.py: 2026-04-24 추가 (기본 0 = backward-compat)
- run_current_coin_v20_backtest.py: initial_phases (V21 3멤버 고정 구조 용)
- 재랭크 스크립트는 unified_backtest.run 사용 (iter_refine 순위 재현성 확보됨)

엔진 통일
- redesign_rerank_{fut,spot}.py 는 unified_backtest.run(asset_type=...) 사용
- backtest_futures_full.py 는 별도 엔진 — 결과 2배 이상 차이 (2026-04-24 smoke test 확인). 사용 안 함
