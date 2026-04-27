# Phase 0 AI Round 3 — 최종 판단

질의: 2026-04-24 08:49
응답: Gemini + Codex
원문: `/tmp/gemini_p0_r3.txt`, `/tmp/codex_p0_r3.txt`

## 완벽 합의

양 AI 독립적으로 동일 결론:

1. **P0 완전 마감** — SN 교체도 phase offset jitter 도 모두 alpha 훼손
2. **V21 SN=[120,30,120] phase (0,0,0) 영구 고정**
3. **Execution de-crowding = 실행층 문제**. 신호단 손대지 않음
4. **P1 진입 OK**. 추가 P0 실험 불필요

## 질문별

### Q1 판단 동의
- Gemini: 완벽 동의. "현재 (0,0,0) 동기화가 핵심 alpha (시장 사이클과의 공명) 이든 과최적화된 최적점이든, 인위적 찢기는 edge 훼손". 아키텍처 결정.
- Codex: 동의. "de-crowding 아이디어는 맞아도 신호단에 넣으면 alpha tax 크다"
- 결론: Reject phase jitter blanket

### Q2 (0,13,29) 이유
- Gemini: 체리피킹. M1=0 유지 + M2/M3 우연히 특정 급락구간 비껴감.
- Codex: lucky offset 가능. 10 tuple 중 1 통과 = robust plateau 아님.
- 결론: 단일 tuple 채택 금지 확정

### Q3 M1 민감도
- Gemini: M1 은 "거시적 앵커". 장기 추세 추종이 거대 레짐 전환 포착. M1 phase 변화 = 사이클 진입/탈출 지연 치명적.
- Codex: fast leg = satellite, M1 = anchor. 느린 멤버 한 번 틀리면 영향 오래감.
- 결론: 구조적 이해 확립 — M1 보호 필수

### Q4 실행층 완화
- Gemini: 백테 불가능, 불필요. Low-freq 4h 전략에서 180s 지연 / 수분 TWAP 의 시뮬레이션은 tick-level orderbook 필요 = 오버엔지니어링. 실매매 코드에 엔지니어링 방어벽으로 구현만.
- Codex: 백테보다 execution stress test. Slippage + fill delay + partial fill grid, burst volatility 체결실패율, retry/cancel 안정성. 실행 시뮬레이션 or 드라이런 방향.
- 결론: 백테 기반 정당화 불필요. auto_trade_binance.py 수정만.

### Q5 P1 전 놓친 대안
- Gemini: 없음. P1 진입.
- Codex: member별 phase sensitivity heatmap 진단용 (선택). 목적은 M1 anchor 확인.
- 결론: 즉시 P1 진입.

## P0 최종 결론 (아카이브용)

1. **V21 SN=[120,30,120] 영구 고정**. 신호단 구조 동결.
2. **phase_offset_bars 기능은 backtest_futures_full.py 에 남겨둠** (디버그/진단용, default=0 = 기존 동작)
3. **Execution de-crowding 은 실행층으로만**:
   - cron 지연 랜덤화 확대 (현재 5~16s → 0~180s U 분포)
   - 주문 분할 TWAP (30~120s)
   - participation cap, min-notional batching
   - 백테 아닌 dry-run / execution stress test 로 검증
4. **P1 진입 가능**: 개별 변수 jitter sensitivity (SMA/mom_long/hyst/vol_threshold) — 이건 alpha 변수 plateau 확인 목적이라 P0 와 다른 범주

## 미해결 리스크 (문서화)
- 3 멤버 완벽 동기화 시 "최대 레버리지 동시 진입/청산" tail risk — 60/35/5 자산배분 관점에서 감당 가능 여부는 별도 검증 필요 (Phase 2 slippage stress 에서 일부 답 나올 예정)
- M1 (S240/SN120) 이 anchor 라는 구조적 이해 → P1 에서 SMA240 변수 jitter 할 때 M1 에 특히 민감할 것. 결과 해석 시 참고
