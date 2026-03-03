# Cap Defend — Capital Defense Strategy

주식+코인 통합 자산배분 전략. 카나리아 시그널로 위험을 감지하고, 모멘텀+퀄리티 기반으로 종목을 선정한다.

## 전략 구조 (Stock 60% + Coin 40%)

### 주식

- **카나리아 시그널**: VT, EEM이 각각 200일 이동평균 위이면 Risk-On
- **공격 모드**: 12개 글로벌 ETF에서 모멘텀(3/6/12개월 가중) + 퀄리티(Sharpe) 상위 3+3 선정
- **수비 모드**: 5개 방어 자산 중 6개월 수익률 최고 1개 선정 (음수면 현금)

### 코인

- **유니버스**: CoinGecko 시총 Top 100 → 업비트 KRW 상장 + 유동성(30일 평균 10억↑) + 히스토리(253일↑) 필터 → 최대 50개
- **카나리아 시그널**: BTC > 50일 이동평균이면 Risk-On
- **헬스 필터**: 현재가 > SMA30 & 21일 수익률 > 0 & 90일 변동성 ≤ 10%
- **종목 선정**: 헬스 통과 코인 중 Sharpe(126일+252일) 상위 5개
- **비중 배분**: 역변동성 가중 (Inverse Volatility Weighting)

## 리밸런싱

- 턴오버 30% 이상 시 리밸런싱 실행
- 헬스체크 탈락 코인은 턴오버 무관하게 즉시 매도
- 현금 버퍼 2% 유지

## 데이터

- Yahoo Finance 우선, CoinGecko 폴백
- 가격 품질 검증: Yahoo 종가(USD→KRW) vs 업비트 종가 10% 이상 괴리 시 제외
- 10회 재시도 + 로컬 캐시 + Stale Data 폴백

## 파일 구조

```
strategies/cap_defend/
├── README.md              # 이 문서
├── recommend.py           # 추천 리포트 (공개용)
├── recommend_personal.py  # 추천 리포트 (개인 자산 연동)
├── check_canary.py        # 카나리아 시그널 확인
└── report.html            # 백테스트 결과 리포트
```
