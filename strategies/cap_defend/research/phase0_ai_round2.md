# Phase 0 AI Round 2 — 결과 리뷰 + 방향 재결정

질의: 2026-04-24 08:30
응답: Gemini + Codex
원문: `/tmp/gemini_p0_r2.txt`, `/tmp/codex_p0_r2.txt`

## 핵심 합의

양쪽 독립적으로 동일 결론:

1. **[123,33,129] 기각** — SN 자체를 바꾸는 비동기화는 V21 alpha 훼손
2. **Option 4 (SN 유지 + bar_counter phase offset)** 가 유일한 다음 검증 가치
3. **목적 재정의**: 애초 문제는 "집단 동시 실행 → 시장충격". 해결은 신호단(SN)이 아닌 실행단 분산이어야 함

## 질문별 정리

### Q1 기각 판단
- Gemini: 기각 맞음. 보유기간(holding period) / 신호 평가주기 바꿔 기존 alpha 훼손
- Codex: 기각 맞음. "비동기화가 좋다" 가 아닌 "SN 자체 변경 = alpha 깎음" 결과
- 결론: 기각 확정

### Q2 D > C 원인
- Gemini: 파라미터 공간의 우연적 조화 (harmonic resonance). 기대 효과 아님
- Codex: D cadence 덜 공격적 (Rebal 861→799). long-leg 둘 더 멀리 벌어짐 (41/43 유사 vs 37/47 분리)
- Claude 종합: 구조적 패턴 있지만 체리피킹 위험. 실전 근거 아님

### Q3 Option 4 (phase offset) 리스크/가능성
- Gemini: 압도적 장점. alpha 100% 보존 + 실행시점만 찢기. "정공법"
- Codex: 깨끗한 접근. baseline alpha 덜 훼손. "timing jitter robustness test" 로 봐야
- 공통 경고: 단일 best offset 채택 = 과최적화. 여러 offset tuple 돌려 median 봐야
- Claude 종합: 동의. 다중 offset tuple 로 분포 검증

### Q4 추가 후보
- Gemini: phase offset 만 검증. SN 자체 건드리는 추가 테스트 즉각 중단
- Codex 우선순위:
  1. SN=[120,30,120] 유지 + phase offset grid
  2. fast leg 만 소폭 변경 [120,39,120] / [120,33,120]
  3. 실행층 (TWAP/지연)
- 결론: Phase 0b 로 phase offset grid 먼저

### Q5 전제 재검토
- Gemini: SN 비동기는 "완전히 잘못된 도구". 시장충격 방지는 실행단 과제. 이미 3-snapshot tranche 로 내부 분산 존재
- Codex: 전제 바꾸는 게 맞다. 문제 "동시 주문", 나쁜 해결 "신호 cadence 변경", 맞는 해결 "실행층 분산 (offset, 지연, 분할, 밴드)"
- 공통: 동기화된 tranche refresh 가 V21 alpha 의 일부. 강세장 집중매수 / 공통 tranche 리셋 / trend capture 가 함께 작동
- 결론: 목적 = "alpha 유지한 execution de-crowding". 비동기화 자체가 목표 아님

## 라운드2 결론

### 채택 방침
- Phase 0 (SN 교체) 완전 기각. V21 SN=[120, 30, 120] 유지
- Phase 0b 로 진행: SN 고정 + 멤버별 bar_counter phase offset grid 테스트
- 판단 기준: offset tuple 10개 median/p25 Cal 이 baseline 의 ±5% 이내면 "실행 jitter 사용 가능"으로 결론
- 단일 best tuple 로 live 고정 금지. 실전 반영 시 매일 랜덤 jitter 하는 방식이 올바름

### 최종 목표 재정의
- "실행 de-crowding" = 같은 전략이 다른 계정/사용자 간 정확히 같은 시점에 주문하지 않도록 분산
- 구현 방법 (priority):
  1. phase offset 매일 랜덤 (같은 alpha, 다른 실행시점)
  2. cron 시작 지연 랜덤화 (이미 일부 적용 중)
  3. 주문층 분할 / TWAP (Phase 2 slippage stress test 로 정당화)

### 다음 단계
- 0.6 phase0b_phase_offset.py 작성 완료
- 0.7 10 tuple 백테 실행
- 0.8 phase0b_report.md 작성 (median/p25/max spread 분석)
- 0.9 AI 리뷰 라운드3 (결론 검증)
- 0.10 P0 마감 → P1 진입
