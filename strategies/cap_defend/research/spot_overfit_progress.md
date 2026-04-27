# V21 현물 과적합 검증 Progress (라이브)

플랜: `spot_overfit_plan.md`
시작: 2026-04-24
구조: 선물 phase0c 와 동일한 3축 검증

## 현재 상태

- [ ] 1. 현물 백테 엔진 entry point 파악
- [ ] 2. phase_offset_bars 현물 엔진 추가
- [ ] 3. phase0c_spot_overfit.py 작성
- [ ] 4. 백그라운드 실행 (21 configs)
- [ ] 5. spot_overfit_results.csv / spot_overfit_report.md
- [ ] 6. 과적합 판정 (CV)
- [ ] 7. AI 토론 (선물+현물 공통 재설계 방향)
- [ ] 8. 재설계 합의 → 구현

## 병렬 트랙 (2026-04-24 사용자 추가 지시)

- [ ] D봉 선물 전략 historical 재조회
  - 4h 봉 대신 1d 봉으로 하면 더 좋은지
  - research/ 과거 백테 결과 찾아서 비교

## 기록

- 진행 전환점마다 이 파일에 추가
- 재개 힌트: 이 파일 + spot_overfit_plan.md + phase0c_report.md 읽기

## 재개 힌트

- 엔진 위치: `trade/coin_live_engine.py` (V21 run 함수 있음)
- 과거 백테 스크립트: `research/run_current_coin_v20_backtest.py` (V20 기준이지만 참고)
- multicoin_engine.py 도 후보 (research/ 내 독립 엔진)
- 선물 D봉 과거 실험: `research/test_d_vs_4h_sweep.py` 확인
