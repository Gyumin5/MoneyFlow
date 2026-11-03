# 주식 백테스트 가이드

현행 주식 전략은 V25 다. 먼저 알아야 할 것은 하나다 — V25 를 한 명령으로 재현하는
백테스트 진입점이 이 저장소에 없다.

## V25 를 확인하는 두 자리

- 라이브가 오늘 무엇을 고르나 — `stock_strategy_v25.py` 순수함수.
  `recommend.py` 와 `trade/executor_stock.py` 가 같은 함수를 부르므로 이게 신호의 정본이다.
- 채택 근거 계산 — `research/bt_stock_mom3.py`. 3-mom 그리드를 창 순위합으로 줄세우는
  스크립트이고 `bt_stock_window_rank.py` · `bt_stock_coin_v3.py` 를 import 한다.
  단일 설정 러너가 아니라서 "V25 의 성과표" 한 장을 그대로 뱉지는 않는다.

## run_current_stock_backtest.py 는 V17 이다

```bash
python3 strategies/cap_defend/run_current_stock_backtest.py
```

이 스크립트는 `CURRENT_V17` 설정으로 돌고 화면에도 "주식 전략 (V17, 11-anchor 평균)" 으로
찍는다. 기준 출력은 이렇다.

```
[2017-01-01 ~ 2025-12-31]  Sharpe 0.977  CAGR +11.4%  MDD -13.3%  Calmar 0.88
[2018-01-01 ~ 2025-12-31]  Sharpe 0.849  CAGR +10.1%  MDD -13.3%  Calmar 0.77
[2021-01-01 ~ 2025-12-31]  Sharpe 1.017  CAGR +12.6%  MDD -13.2%  Calmar 0.98
```

이 값을 현행 성과로 인용하지 않는다. V17 과 V25 가 다른 자리는 다섯이다.

- 급락 가드 `crash="vt"` 가 켜져 있다 (V25 는 가드 없음)
- `sharpe_lookback=252` (V25 는 126)
- 3-mom(30/72/230) 필터가 없다
- 3트랜치(SNAP_PERIOD=69, STAGGER=23, N_SNAPS=3)와 drift 0.05 가 없다
- 비중이 `weight="ew"` 균등이다 (V25 는 cap 1/3 + 7% Cash)

값만 바꿔서 V25 로 만들 수 있는 구조가 아니다 — 이 스크립트가 쓰는 엔진 `stock_engine.py` 의
설정 클래스 `SP` 에 3-mom 필터·트랜치·drift·cap 비중 자체가 없다.

## V25 전략 정의

- 유니버스 R7: SPY, QQQ, VEA, EEM, GLD, PDBC, VNQ
- EEM 카나리: SMA200 ± 0.5% 히스테리시스 (dead-zone)
- 선정: Z-score(가중Mom 0.5×ret63 + 0.3×ret126 + 0.2×ret252, Sharpe126) 랭킹
  → 3-mom(30/72/230) 필터 → Top3 cap 1/3 + 7% Cash
- 트랜치 3개 (SNAP_PERIOD=69, STAGGER=23, N_SNAPS=3), drift 0.05
- 종목 교체 = 앵커일 OR drift 5pp 발화일
- 가드 없음 (EEM 카나리 + 3-mom 필터 + 분산이 유일 방어)

## 데이터

주식 가격은 이 순서로 읽는다.

1. `strategies/cap_defend/data/stock_cache/*.csv`
2. 메인 `data/*.csv`
3. `yfinance` 내려받기

현물·선물과 달리 완전 로컬 고정이 아니라 캐시 우선 + 야후 fallback 이다. 캐시가 비었거나
오래되면 실행 중에 야후를 치므로, 같은 수치를 다시 받으려면 캐시를 유지한 채 돌린다.

갱신과 최신성 확인:

```bash
python3 strategies/cap_defend/refresh_backtest_data.py --target stock
python3 strategies/cap_defend/check_data_freshness.py
```

볼 것은 핵심 ETF 캐시 파일의 존재와 마지막 날짜다. 전략을 확정하기 직전이면
`strategies/cap_defend/data/stock_cache/*.csv` 를 스냅샷으로 따로 보관한다.
