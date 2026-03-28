# 현재 작업: V17 카나리 업데이트 완료

## 완료

- [x] 카나리 재최적화: SMA(50) + 1.5% hyst 확정
  - 1차 넓은 그리드 + 2차 세밀 그리드 + 멀티SMA 투표 + anti-whipsaw
  - 10-anchor, 3기간, plateau 확인, anchor MDD 10/10 개선
- [x] 코드 업데이트:
  - coin_helpers.py: B() vote_smas=(50,)
  - recommend_personal.py: COIN_CANARY_MA_PERIOD=50, HYST=0.015
  - recommend.py: CANARY_SMA_PERIOD=50, HYST=0.015
  - backtest_official.py: V17 정의
  - MEMORY.md 업데이트

## TODO

- [ ] strategy_guide.html 업데이트
- [ ] 서버 배포 (recommend 2개)
- [ ] git commit
- [ ] 방어자산: 유니버스 확장 후 재검토 (보류)
