# 현재 작업: 코인+주식 자동매매 버그 수정

## 수정 목록

### 치명적

- [ ] 1. monitor DD Exit: 전량청산 → 해당 코인만 매도
- [ ] 2. 앵커일 문서 수정: 1/10/19 → 1/11/21 (코드가 맞음)

### High

- [ ] 3. Crash cooldown 상태 저장 (trade_state.json에 기록)
- [ ] 4. rebalancing_needed: monitor에서 pending 완료 시 false로
- [ ] 5. Risk-Off 앵커 마킹: HOLD에서 덮어쓰기 방지
- [ ] 6. 전량 매도 실패 → pending에 저장

### Medium

- [ ] 7. 헬스체크: 트랜치 목표 코인도 검사 범위에 포함
- [ ] 8. 매도/매수 우선순위: 시총순 정렬 보장

## 진행 상태

- AI 검토 요청 중
