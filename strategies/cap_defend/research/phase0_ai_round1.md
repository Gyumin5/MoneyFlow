# Phase 0 AI Round 1 — SN 교체 기본설계 검토

질의 일시: 2026-04-24 08:23
질의자: Claude (main)
응답자: Gemini + Codex 병렬
원문: `/tmp/gemini_snap_review.txt`, `/home/gmoh/.claude/ai-out/20260424-082319-codex.md`

## 요약

### Q1. 분산방어 취지 훼손 여부
- Gemini: 훼손 X. 동기화는 설계 결함. 비동기화는 V21 방어 철학 강화.
- Codex: 방향 OK. 다만 "완전 무동기" 아닌 "거의 비동기" (gcd=3 공통). 과장 금지.
- Claude: 동의. "같은 봉에 3 멤버가 동일 tranche refresh" 가 구조적 집중위험, 비동기화 = 설계 강화.

### Q2. 후보 선호
- Gemini: [33, 123, 129] (이유: 원본 가장 근접 + 서로소 p)
- Codex:  [33, 123, 129] (이유: cadence 10/40/40 → 11/41/43 최소 변화)
- Claude: 동의. [33, 123, 123] 은 두 멤버 완전 동기 잔존으로 기각, [39, 111, 141] 은 원본에서 더 멀어 성격 변화 큼.
- 합의: **[33, 123, 129]**

### Q3. 백테 reproducibility — state 이월 vs 초기화
- Gemini: 초기화. bar_counter=0 에서 전체 재백테. 이월하면 과거 동기/미래 비동기 섞여 분석 불가.
- Codex: 이월 권장. 더 나은 방식은 `phase_seed = old_bar_counter % new_SN` 명시 저장.
- Claude 판단 (쟁점):
  - **백테**: bar_counter=0 에서 재실행 (Gemini 방향). 구 SN 과 신 SN 을 deterministic 동일 조건으로 비교.
  - **Live**: bar_counter + snapshots 이월, SN 만 교체 (Codex 방향). 전환 자연스럽게.
  - 이 두 결정이 충돌 X (백테 = 기댓값 측정, live = 평균 수렴). 합의 가능.

### Q4. 서버 bar_counter reset 영향
- Gemini: si=0 동시 refresh 트리거 → 1회 리밸 주문 발생. V21 delta only 로 꼬임 없음, 슬리피지 소량.
- Codex: reset 위험. 첫 bar 에서 offset=0 tranche 3개 동시 refresh, snapshots 재계산 몰림. 특히 snapshots 까지 비우면 one-shot 포지션 점프.
- Claude 판단: 합의된 live 전환 방식 = 이월 (snapshots + bar_counter 유지, SN 교체). Reset 불필요. 반영 시 리밸 1회는 발생할 수 있지만 Gemini 말처럼 delta only 로 안전.

### Q5. 숨은 함정
공통:
- turnover 증가 → Binance 최소주문 금액/수량 제한, 자투리 주문 누락
- 부분체결/API rate limit 누적 위험 (거래 빈도↑)
Codex 추가:
- 멤버 순서 매핑 주의: SN30→33, SN120→123, SN120→129 순으로 매핑. 바꾸면 멤버 성격 변질.
- 3p 구조 유지 필수 (그냥 prime 11/41/43 쓰면 tranche offset 불균등)
- 성과보다 turnover/fee/worst loss 먼저 검증

## 라운드1 결론 (합의)

1. 후보: **[33, 123, 129]** 채택
2. 백테 기준: bar_counter=0 (deterministic), SN 4 config 비교
   - A. [120, 30, 120] baseline
   - B. [123, 33, 123]  (1 멤버 동기 잔존, ablation)
   - C. [33, 123, 129]  (권고)
   - D. [39, 111, 141]  (대안)
3. Live 반영 방식 (P0 백테 통과 후): snapshots + bar_counter 유지, SN 교체. 멤버 순서 매핑 주의.
4. 성과 지표 외 turnover/fee/worst-day loss 필수 확인
5. gcd=3 공통 배수 잔존 — 완전 비동기 아님, "거의 비동기" 로 정확히 표현

라운드2 불필요. 스크립트 작성 단계 진입.

## 다음 단계

- 0.2 phase0_snap_desync.py 작성 — 4 config × V21 baseline 백테
- 0.3 실행 → phase0_results.csv
- 0.4 phase0_report.md 작성
- 0.5 AI 리뷰 라운드2
