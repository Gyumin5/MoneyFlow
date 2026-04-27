# V21 선물 로버스트니스 Progress (라이브)

플랜: `futures_jitter_plan.md`
시작: 2026-04-24
구조: 각 Phase 마다 AI 토론 (gemini+codex) → 구현 → 백테 → 리포트 → 다음

## 현재 상태

- [x] Phase 0  SN 교체 백테 — **완료 (2026-04-24)**. 결론: V21 SN=[120,30,120] phase(0,0,0) 영구 고정
  - [x] 0.1 AI 라운드1 — 합의 [123,33,129]
  - [x] 0.2 phase0_snap_desync.py
  - [x] 0.3 백테 실행 (4 config, phase0_results.csv)
  - [x] 0.4 phase0_report.md — Cal -30% (C 권고), 통과기준 실패
  - [x] 0.5 AI 라운드2 — [123,33,129] 기각. Option 4 (phase offset) 로 pivot
  - [x] 0.6 backtest_futures_full.py 에 phase_offset_bars 인자 추가
  - [x] 0.7 phase0b_phase_offset.py (10 tuple)
  - [x] 0.8 phase0b 백테 (phase0b_results.csv)
  - [x] 0.9 phase0b_report.md — median Cal -16%, 유일 통과 (0,13,29) 는 체리피킹
  - [x] 0.10 AI 라운드3 — blanket phase jitter 도 기각. 실행층으로만 de-crowding (phase0_ai_round3.md)

## Phase 0 최종 결론
- V21 선물 snap 구조 동결
- backtest_futures_full.py phase_offset_bars 인자는 진단용 유지 (default=0)
- execution de-crowding → auto_trade_binance.py 엔지니어링 방어벽:
  - cron 지연 U(0, 180s) 확장
  - 주문 TWAP 30~120s
  - 백테 불필요 (AI 만장일치). 필요시 dry-run / execution stress test
- MEMORY: project_p0_snap_desync_0424.md
- [ ] Phase 1  Jitter sensitivity
- [ ] Phase 2  Slippage stress
- [ ] Phase 3  MC jitter
- (Phase 4 취소 2026-04-24)

## 2026-04-24 사용자 지시
- P4 는 계획에서 제외
- P0 부터 AI 검토 후 자율적으로 완료까지 진행

---

## Phase 0 진행 로그

### 0.1 AI 토론 — 기본설계 검증
status: pending

질문:
- SN 을 3p (p 소수) 으로 바꾸는 접근이 V21 앙상블 취지 (분산 방어) 를 훼손하지 않는지
- [33, 123, 129] 대 [33, 123, 123] 대 [33, 111, 129] 중 선호
- 기존 백테 재현성 확보 방법 (state 호환)
- 서버 반영 시 위험 (bar_counter reset 필요 여부)

### 0.2 스크립트
경로: strategies/cap_defend/research/phase0_snap_desync.py
상태: 미작성

### 0.3 백테 결과
파일: phase0_results.csv
상태: 미실행

### 0.4 리포트
파일: phase0_report.md
상태: 미작성

### 0.5 최종 AI 리뷰
status: pending

---

## 중요 기록

- V17 주식 phase1a_v2 (36k configs) 백그라운드 실행 중 (PID 264586). CPU 간섭 주의.
- live 변경 금지 until Phase 3 완료 + 크로스체크.
- 각 단계 AI 질의/답변은 phase{N}_ai_round{M}.md 에 원문 저장.

## 재개 힌트 (세션 로스트 복구용)

- 이 파일 + futures_jitter_plan.md 읽기
- 현재 상태 [ ] 체크박스 보고 가장 위의 미완료 항목으로 이동
- AI 토론 결과는 phase*_ai_round*.md 에 있음
- 백그라운드 프로세스는 pgrep -af v17_snap|phase0_|phase1_|phase2_|phase3_|phase4_
- 중단된 백테 resume: 각 스크립트에 checkpoint CSV 가 있음 (*_partial.csv)
