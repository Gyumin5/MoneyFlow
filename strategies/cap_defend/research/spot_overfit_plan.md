# V21 현물 과적합 검증 플랜 (2026-04-24)

## 배경

- V21 선물 phase0c 결과 (`phase0c_report.md`): 3축 (공통 phase / 시작일 / rolling) 모두 CV 임계 초과 → base Cal 3.10 은 lucky anchor 의 과적합
- V21 현물 (코인 D봉 3멤버 SMA50/100/150 EW, snap 90봉) 도 동일 구조 → 동일 위험 가능성 높음
- 선물과 독립적으로 검증해야 함 (선물 결과 가 현물 적용 안 되는 경우 있을 수 있음)

## 목적

1. 현물 V21 backtest Cal 3.10 이 phase / start / rolling 민감한지 확인
2. 선물과 비교해 패턴 유사/상이 파악
3. 과적합 확정 시 V21 전체 (현물+선물) 동시 재설계 대상

## 검증 축 (선물 phase0c 와 동일 구조)

A. **공통 phase sweep** (k, k, k), k ∈ {0, 7, 13, 19, 31, 47, 59, 83, 113}
   - 3 멤버 동일 시프트, 파라미터 그대로
   - 기간 2018-01-01 ~ 2026-03-28 (현물 가용 기간 전체)

B. **시작일 sweep** (phase 0,0,0, 동일 파라미터)
   - start_date ∈ {2018-01-01, 2018-04-01, 2018-07-01, ..., 2019-01-01} 약 5~7 포인트

C. **Rolling sub-period**
   - 2018-2020, 2019-2021, 2020-2022, 2021-2023, 2022-2024, 2023-2025, 2024-2026
   - 2년 윈도우

판정:
- 각 축 CV < 0.10 → plateau (V21 현물 robust)
- CV 0.10~0.20 → 경계, 해석 주의
- CV > 0.20 → overfit 확정

## 현물 V21 구성 (재현용)

- 엔진: `trade/coin_live_engine.py` 앙상블 ENS_spot_k3_4b270476
- Member 1 (D_SMA50):  D봉, SMA50, Mom20/90, snap 90봉×3, hyst 1.5%
- Member 2 (D_SMA150): D봉, SMA150, Mom20/60, snap 90봉×3, hyst 1.5%
- Member 3 (D_SMA100): D봉, SMA100, Mom20/120, snap 90봉×3, hyst 1.5%
- 공통: health mom2vol, vol_cap 5% (90일), universe_size=3, cap=1/3
- TX 0.04% (Upbit 실수수료 기준)
- 가드 없음 (2026-04-21 제거)

## 구현 계획

### 준비
1. 현물 백테 엔진 entry point 파악
   - multicoin_engine.py 또는 coin_live_engine 직접 호출
   - 기존 `run_current_coin_v20_backtest.py` 같은 wrapper 참고
2. `phase_offset_bars` 인자 현물 엔진에 추가 (선물 backtest_futures_full.py 와 동일 패턴)

### 실행
3. `phase0c_spot_overfit.py` 작성 (선물 phase0c_overfit_check.py 구조 재사용)
4. 백그라운드 실행 → `spot_overfit_results.csv`
5. 자동 분석 → `spot_overfit_report.md`

### 판정 후
6. CV > 0.2 확정되면 현물도 overfit 결론 → V21 재설계 범위에 포함
7. AI 라운드 (gemini+codex): 선물/현물 공통 재설계 방향 토론
8. 설계 합의 후 구현/검증

## 시간 예상

- 엔진 파악 + phase 인자 추가: 15분
- 스크립트 작성: 10분
- 백테 실행 (21 configs × D봉 백테 ~20s each): ~7분
- 분석 + 리포트: 5분
- AI 토론: 10분
총 ~50분

## 주의사항

- 현물 D봉 데이터는 선물 4h 보다 훨씬 짧음 → phase_offset 단위도 다름 (일 단위)
- snap_interval_bars=90 (현물 D봉 기준 90일) vs 선물 120 (4h봉 120개 = 20일). 스케일 차이 인지
- 현물 기간은 2018년부터 가용 (선물은 2020-10)
- V22 C 슬리브는 2026-04-22 비활성화 → 순수 V21 구조로 검증
