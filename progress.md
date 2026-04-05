# 자산배분 + 선물 전략 전환 (2026-04-05)

## 완료

- [x] d005 4전략 앙상블 확정 (4h_d005, 2h_S240, 2h_S120, 4h_M20, EW 25%)
- [x] 자산배분 확정: 주식 60% / 현물코인 25% / 선물 15%, 밴드 8pp
- [x] auto_trade_binance.py 4전략으로 교체 + 2h 데이터 수집 추가
- [x] futures_live_config.py 업데이트
- [x] 서버 배포 + dry-run 성공 (PV=$4824.85, 4전략 모두 정상)
- [x] CLAUDE.md 선물 규칙 섹션 추가
- [x] memory 업데이트
- [x] recommend_personal.py: 자산배분 60/25/15 + 8pp 밴드 모니터 + 텔레그램 알림
- [x] recommend.py: 자산배분 비율 동기화
- [x] 서버 배포 + 실행 테스트 성공 (자산배분 체크 + 알림 전송 확인)

## 다음 작업

- [ ] git commit
