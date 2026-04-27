# V21 재설계 Progress (라이브)

플랜: `redesign_plan_v3.md` (2026-04-24 AI 만장일치 합의 최신)
시작: 2026-04-24
자산: 주식 V17 + 코인 현물 V21 + 코인 선물 V21 L3

## 상태 (2026-04-24 오후)

- [x] 0. 준비 작업
  - [x] Step 1 초판: top 200 추출 (univ=5 기반 — v3 Top 500 재추출로 대체됨)
  - [x] Step 2 스크립트 초판 (redesign_rerank_{fut,spot}.py, unified_backtest.run phase_offset 연동)
  - [x] Step 3 V17 주식 iter_refine 백그라운드 (PID 669470)
  - [x] AI 라운드1 + 라운드2 합의 완료 (플랜 v3)
- [ ] 1. univ=3 재평가 (진행중, PID 688217, ETA ~12h)
- [ ] 2. Top 500 k=1 추출 (재평가 후)
- [ ] 3. 주식 iter_refine v2 재실행 (universe V17 실운영 교체)
- [ ] 4. Step 3 Phase sweep 스크립트 업데이트 (snap-relative phase)
- [ ] 5. Step 3.5 Snap nudge 스크립트 신규
- [ ] 6. Step 4 Yearly consistency 스크립트 신규
- [ ] 7. Step 5 rank-sum analyzer 신규
- [ ] 8. Step 6 ensemble 탐색 (corr + tail co-crash)
- [ ] 9. Step 7 final stress (TX/delay/drop-1)
- [ ] 10. Step 8 report + 사용자 검토
- [ ] 11. AI 코드 리뷰 (스크립트 완성 후)

## 백그라운드 프로세스

- redesign_rebuild_univ3.py (PID 688217, 24 worker, ETA ~12h)
- v17_snap_iter.py (PID 669470, V17 주식 UNIVERSE_B — v3 에서 폐기 예정)

## 재개 힌트

- 주요 문서: redesign_plan_v3.md + redesign_progress.md
- 백그라운드: `pgrep -af 'redesign_rebuild|redesign_rerank|v17_snap_iter'`
- 로그: v17_snap_out/rebuild_univ3_*.log, v17_snap_out/iter_bg_*.log
- AI 합의 출력: ~/.claude/ai-out/20260424-13*.md
