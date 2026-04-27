# V21 재설계 통합 플랜 v2 (주식 + 선물 + 현물)

작성: 2026-04-24
대상: cap_defend 3개 자산 트랙 (주식 V17, 코인 현물 V21, 코인 선물 V21 L3)
목적: 기존 grid 재평가 + robustness 기준 재랭킹 → plateau 있고 일반화된 전략 후보 선정

## 0. 배경

- 선물 phase0 결과: SN 교체 + phase offset 둘 다 기각. base Cal 3.10 은 phase=0 lucky anchor overfit 확인
- 현물 phase0c 에서 유사한 overfit 의심 (진행중)
- 주식 V17 (tx 0.25%) baseline Cal 0.41 — 기존 iter_refine 으로 부족. V21 style 3-snapshot ensemble 로 재설계
- 사용자 공통 지시:
  1. 홀드아웃 불필요 (연도별 consistency 로 대체)
  2. start date sweep 불필요 (yearly window 가 자동 커버)
  3. 기존 grid 결과 재사용 후 robust 재랭킹
  4. k=1/2/3 모두 탐색, 앙상블 강제 X
  5. universe_size=3 (V21 스펙 통일)
  6. Ensemble 은 동일 snap_interval + 동일 interval 내 조합 (관리·비용·해석)
  7. 4h+D 혼합 ensemble 배제
  8. 멤버간 phase stagger 는 초기엔 안 함 (V21 내부 n_snapshots=3 이미 분산)
  9. 로버스트 축 2개: Tier 1 phase sweep (anchor day) + snap prime×3 nudge (cadence)
 10. Yearly window 은 모든 해 (2017~2025 주식 / 2020~2025 코인)
 11. 2022 bear 포함 필수
 12. 주식 universe 는 현재 V17 실운영 기준 (SPY/QQQ/VEA/EEM/GLD/PDBC/VNQ)

## 1. 공통 Workflow (3개 자산 동일)

### Step 1 — 기존 grid 재평가 (V21 스펙 univ=3)
- 입력: iter_refine/raw_combined.csv (선물/현물) 또는 v17 iter output (주식)
- 재평가 조건 V21 스펙 통일:
  - universe_size=3, cap=1/3, n_snapshots=3
  - canary_hyst=0.015, health_mode=mom2vol
  - stop_kind=none, stop_pct=0 (V21 가드 제거)
  - phase_offset_bars=0 (기준점)
- 출력: redesign_univ3_raw_{stock,fut,spot}.csv

### Step 2 — Top 200 k=1 추출
- Cal 기준 정렬 top 200
- 기존 ensemble csv (선물/현물 grid_results/topcands_proper_*_k{2,3}_partial) 는 k=2/3 top 200 보조 소스로 유지

### Step 3 — Tier 1 Phase robustness
- Phase sweep: phase_offset_bars {0, 7, 13, 31, 59} (총 5값 — AI 합의로 확정)
- Top 200 × 5 phase = 1000 backtest / 자산
- 각 config 지표:
  - med_Cal (phase 축)
  - p5_Cal (phase 축 worst 20 percentile)
  - Cal_CV (phase 축 분산 계수)
- 필터: Cal_CV < 자산별 분위수 p25 (절대 threshold 대신 상대 분위수 — 자산별 변동성 차이 보정)

### Step 4 — Yearly consistency (모든 해)
- 대상: Tier 1 survivor
- 기간:
  - 주식 2017~2025 (9년)
  - 코인 현물/선물 2020~2025 (6년)
- 각 년도 Cal 계산
- 지표:
  - yearly_med_Cal (년도별 중앙값)
  - yearly_p25_Cal (worst 25%)
  - yearly_top25_count (각 년도 자산 내 top 25% 유지한 년수)
  - yearly_rank_mean (년도별 Cal 순위 평균 — 낮을수록 좋음)
- 2022 bear 포함 필수, 2020 코로나빔/2023 AI 랠리 도 자동 포함

### Step 5 — Snap prime×3 nudge (cadence robustness)
- 대상: Tier 1 + Yearly survivor
- 원본 snap 값 ± 10% 범위 내 prime×3 nudge 점 3~5개 선택:
  - 예: snap=90 이면 → {87, 93, 99, 111} (prime×3 = 3×29, 3×31, 3×33, 3×37)
  - 예: snap=30 이면 → {21, 27, 33, 39}
- 각 nudge point 백테스트, Cal 편차 측정
- 필터: snap nudge Cal 분산이 원본 대비 ±10% 이내 통과

### Step 6 — 최종 랭킹
- Tier 1 + Yearly + Snap nudge 모두 통과한 configs 대상
- 지표 집계:
  - composite_Cal = med_Cal × yearly_top25_count / years_total
  - robustness_score = (phase_CV + snap_nudge_CV + yearly_CV) 평균 역수
  - CAGR_adjusted = yearly_med_CAGR
- 랭킹 4개:
  - R1 = rank by composite_Cal (높은 순)
  - R2 = rank by yearly_rank_mean (낮은 순 = 안정적)
  - R3 = rank by robustness_score (높은 순)
  - R4 = rank_sum (R1+R2+R3)
- Top 10 by each R1~R4 ∪ → unique 20~30 candidates per asset

### Step 7 — Ensemble 탐색 (k=2/3)
- Step 6 top 후보들 중 동일 snap_interval + 동일 interval 버킷에서 조합 생성
- pairwise corr (returns 기준) 계산, corr < 0.7 pair/triple 만 backtest
- Ensemble Cal / yearly_consistency 계산
- 단일 k=1 best 대비 improvement 가 robustness 기준 유의한 것만 채택

### Step 8 — 리포트 + 사용자 검토
- 자산별 final_candidates.csv + report.md
- 사용자가 직접 검토 → V21 대체/병행 shadow/기각 결정

## 2. 자산별 Step 1 구체 범위

### 주식 (V17 재최적화)
- 입력: v17_snap_iter 재실행 필요 (현재 UNIVERSE_B 로 돌고있어 폐기)
- Universe 교체: SPY/QQQ/VEA/EEM/GLD/PDBC/VNQ (공격) + IEF/BIL/BNDX/GLD/PDBC (방어)
- Canary: EEM
- 탐색 축: snap_days, canary_sma, canary_hyst, canary_type, select, def_mom_period, health
- 방법: iter_refine zoom (top 30~50 peak 주변, max_iter=6)
- 출력 경로: v17_snap_v2_out/ (기존 결과 보존)
- tx=0.0025

### 선물 (V21 L3 재최적화)
- 입력: iter_refine/raw_combined.csv (fut 72,815 configs)
- 재평가 엔진: unified_backtest.run(asset_type='fut')
- 진행 중: redesign_rebuild_univ3.py (PID 688217, ETA 12.7h)
- tx=0.0004

### 현물 (V21 D봉 재최적화)
- 입력: iter_refine/raw_combined.csv (spot 48,903 configs)
- 재평가 엔진: unified_backtest.run(asset_type='spot')
- 진행 중: 선물과 같은 rebuild 프로세스 (동일 CSV 에 포함)
- tx=0.004

## 3. 규모 추정

- Step 1 univ=3 재평가: ~12h (121k configs, 24 worker)
- Step 1 주식: iter_refine 재실행 2~3h
- Step 2 top 200 추출: <1분
- Step 3 Tier 1 phase sweep: 3 자산 × 200 × 5 = 3000 backtest, ~1~2h
- Step 4 yearly consistency: survivor × 9년 (주식) or 6년 (코인), ~2~4h
- Step 5 snap nudge: survivor × 5 nudge point, ~1h
- Step 6 랭킹: <1분
- Step 7 ensemble: 조합 수 × backtest, ~2~4h

총 ~2일 내 완료

## 4. 주의 / 미확정 사항 (AI 토론 대상)

### 확정된 제약
- 홀드아웃 없음 (yearly consistency 대체)
- start date sweep 없음 (yearly 가 커버)
- 4h+D 혼합 ensemble 없음
- univ=3, V21 스펙 통일
- 동일 snap_interval + 동일 interval 내에서만 ensemble 조합

### AI 검토 요청 포인트
1. Phase sweep 값 개수/범위 (현재 5값 {0,7,13,31,59}). 프라임 기반으로 anchor decorrelation 충분한지? 더 많이 필요한가?
2. Cal_CV 필터 threshold: 절대값 (0.15 등) vs 자산별 분위수 (p25). 어느 쪽 robust 한가?
3. Yearly consistency 지표 4개 중 중복 있나? (med / p25 / top25_count / rank_mean)
4. Snap prime×3 nudge 범위 ±10% 이 적절한가? 더 넓히면 cadence robust 더 확실하지만 후보 탈락률 높아짐
5. Ensemble corr threshold 0.7 적절? 낮추면 diversification 확보, 높이면 후보 수 유지
6. k=1/k=2/k=3 순위를 섞어서 최종 합집합 낼 때 가중치 동일? 아니면 k=1 우선?
7. Final candidates 20~30개가 적절? 채택 decision 단계에서 너무 많으면 사용자 검토 부담
8. Forward walk 없이 yearly consistency 만으로 충분한가? 2022 bear 가 하나뿐이라 bear 후보 검증이 약한 건 아닌가?
9. Composite_Cal 지표 정의 (med_Cal × yearly_top25_count / years_total) 가 합리적? 다른 지표 조합 권장?
10. Robustness_score 역수 평균 방식이 phase/snap/yearly 축 서로 다른 scale 을 제대로 통합하나? 정규화 필요?
