# 현재 작업: 아키텍처 리팩토링 (executor 통합)

## 1단계: signal_state에 execution plan 추가

- [ ] recommend: stock ideal_picks + today_anchors 출력
- [ ] recommend: coin ideal_picks + today_anchors 출력
- [ ] 기존 키 유지 (backward compat)

## 2단계: executor가 plan 읽기 (shadow mode)

- [ ] auto_trade_kis: plan 기반 target vs legacy target diff 로그
- [ ] auto_trade_upbit: plan 기반 target vs legacy target diff 로그

## 3단계: feature flag 전환

- [ ] auto_trade_kis: plan 기반 매매
- [ ] auto_trade_upbit: plan 기반 매매

## 4단계: legacy 제거

- [ ] 기존 trade 전용 로직 제거
- [ ] trade/monitor 통합
