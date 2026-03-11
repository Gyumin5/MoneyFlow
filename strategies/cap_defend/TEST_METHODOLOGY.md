# Cap Defend 코인 전략 — 테스트 방법론 & 전략 카탈로그

## 고정 파라미터 (Fixed)

| 항목          | 값                      | 비고                              |
| ------------- | ----------------------- | --------------------------------- |
| 유니버스      | 시총 Top 50             | historical_universe.json          |
| 편입 수       | Top 5                   | 시총순 기본                       |
| 체크 빈도     | 매일                    | 카나리아+헬스 매일 평가           |
| 거래비용      | 0.4% (편도)             | 차이분만 거래 (delta rebalancing) |
| 초기자본      | $10,000                 |                                   |
| 백테스트 기간 | 2018-01-01 ~ 2025-06-30 |                                   |
| 벤치마크      | BTC B&H                 |                                   |

## 베이스라인 (Baseline)

| 구성요소   | 설정                                       |
| ---------- | ------------------------------------------ |
| 카나리아   | BTC > SMA150                               |
| 헬스체크   | price > SMA30 AND mom21 > 0 AND vol90 ≤ 5% |
| 코인선택   | 시총순 Top 5                               |
| 가중방식   | 균등배분 (1/5)                             |
| 리밸런싱   | 월간 (매월 1일)                            |
| 리스크관리 | 없음                                       |

**베이스라인 성과 (tx=0.004):** Sharpe 1.297, MDD -48.1%, CAGR +71.1%

---

## Stage 1: Solo Tests (30개 단독 테스트)

각 전략을 베이스라인에서 해당 레이어만 교체하여 독립 평가.
합격 기준: Sharpe ≥ baseline × 0.95 (즉 ≥ 1.232) 또는 MDD 개선 ≥ 3pp

### A. 카나리아 (Canary) — K1~K5

| ID  | 이름            | 설명                                                                   | 기대효과                   |
| --- | --------------- | ---------------------------------------------------------------------- | -------------------------- |
| K1  | Hysteresis Band | 진입: BTC > SMA150 × 1.02, 청산: BTC < SMA150 × 0.98                   | Whipsaw 감소, MDD 개선     |
| K2  | Asymmetric MA   | 진입: BTC > SMA150, 청산: BTC < SMA200 (느린 청산)                     | 불황에 늦게 나가 수익 보존 |
| K3  | Fast Re-entry   | 기본 SMA150 + BTC가 SMA150 아래 떨어진 후 7일 이내 복귀 시 즉시 재진입 | Flash crash 후 빠른 복귀   |
| K4  | Dual Canary     | BTC > SMA150 AND ETH > SMA150                                          | 확실한 강세장만 진입       |
| K5  | 2-of-3 Vote     | BTC > SMA150, BTC > SMA50, BTC mom21 > 0 중 2개 이상 충족              | 노이즈 필터링              |

### B. 헬스체크 (Health Check) — H1~H5

| ID  | 이름                   | 설명                                               | 기대효과                  |
| --- | ---------------------- | -------------------------------------------------- | ------------------------- |
| H1  | Dual Momentum          | mom21 > 0 AND mom90 > 0 (중기 모멘텀 추가)         | 단기 펌핑 필터, 승률 개선 |
| H2  | Dual MA                | price > SMA30 AND price > SMA90                    | 장기 추세 확인            |
| H3  | Persistence Filter     | 3일 연속 헬스 실패 시에만 탈락 (1~2일 실패는 무시) | 노이즈 탈락 방지          |
| H4  | Relative Strength      | (coin/BTC) > SMA30 of ratio — BTC 대비 상대강도    | BTC 언더퍼폼 코인 배제    |
| H5  | Vol Acceleration Block | vol30 > vol90 × 1.5이면 탈락 (변동성 폭발 회피)    | 급등/급락 코인 회피       |

### C. 코인선택 (Selection) — S1~S5

| ID  | 이름                     | 설명                                                 | 기대효과               |
| --- | ------------------------ | ---------------------------------------------------- | ---------------------- |
| S1  | Cap-Constrained Momentum | 시총 Top15 중 mom21 상위 5개                         | 우량주 내 모멘텀       |
| S2  | Blend Rank               | rank = 0.5 × cap_rank + 0.5 × mom21_rank → Top 5     | 시총+모멘텀 균형       |
| S3  | 3+2 Bucket               | 시총 1~3위 고정 + 나머지 중 mom21 Top 2              | Core 안정 + Alpha 추구 |
| S4  | Core-Satellite           | BTC+ETH 고정 + 나머지 3개 시총순                     | 대형주 뼈대            |
| S5  | Incumbent Carry          | 기존 보유 코인에 +2 rank 보너스 → 불필요한 교체 억제 | 턴오버 감소            |

### D. 가중방식 (Weighting) — W1~W5

| ID  | 이름           | 설명                                                           | 기대효과           |
| --- | -------------- | -------------------------------------------------------------- | ------------------ |
| W1  | Rank-Decay     | 1위 30%, 2위 25%, 3위 20%, 4위 15%, 5위 10%                    | 상위 집중          |
| W2  | Shrunk Inv-Vol | inv_vol × 0.7 + equal × 0.3 (역변동성+균등 혼합)               | 안정적 분산        |
| W3  | Momentum Tilt  | 기본 20% + mom21 상위 2개 +5%, 하위 2개 -5%                    | 추세추종 강화      |
| W4  | Breadth-Scaled | 건강한 코인 수 비례 전체 익스포저 조절 (5개면 100%, 3개면 60%) | 약세장 자동 축소   |
| W5  | BTC Fill       | 5개 미만 시 부족분을 BTC로 채움 (빈 슬롯 → BTC)                | 현금 대신 BTC 보유 |

### E. 리밸런싱 (Rebalancing) — R1~R5

| ID  | 이름              | 설명                                        | 기대효과           |
| --- | ----------------- | ------------------------------------------- | ------------------ |
| R1  | Recovery Refresh  | 카나리아 OFF→ON 전환 시 즉시 리밸런싱       | 상승장 진입 가속   |
| R2  | Catastrophic Exit | 포트폴리오 MTD -15% 시 전량 현금화          | 월중 급락 방어     |
| R3  | Ultra-High TO     | 턴오버 > 50% 시에만 리밸런싱 (낮은 TO 무시) | 불필요한 거래 억제 |
| R4  | Banded Weight     | 각 코인 비중이 ±5pp 벗어나야 리밸런싱       | 소규모 변동 무시   |
| R5  | Anchor Day        | 매월 15일 리밸런싱 (월초 대신)              | 월초 군집효과 회피 |

### F. 리스크관리 (Risk Management) — G1~G5

| ID  | 이름            | 설명                                                          | 기대효과           |
| --- | --------------- | ------------------------------------------------------------- | ------------------ |
| G1  | Soft DD Overlay | 포트폴리오 고점 대비 -20% 시 익스포저 50%로 축소              | 점진적 방어        |
| G2  | Vol Target      | 포트폴리오 30일 변동성 > 연 80%이면 비중 축소 (target/actual) | 변동성 자동 조절   |
| G3  | Breadth Ladder  | 건강 코인 ≤2개면 전체 비중 50%, 0개면 현금화                  | 시장 폭 기반 방어  |
| G4  | Rank Floor      | 시총 50위 밖으로 밀린 보유 코인 즉시 매도                     | 소형주 리스크 차단 |
| G5  | Crash Breaker   | BTC 일간 -10% 시 다음 3일 현금 유지                           | Flash crash 방어   |

---

## Stage 2: Pairwise Interaction Screen (~375 테스트)

Stage 1 통과 전략들의 **레이어 간 시너지**를 검증.

### 상호작용 점수 (Interaction Score)

```
I(A,B) = Score(A+B) - Score(A+base) - Score(base+B) + Score(base)
```

- I > 0: 시너지 (함께 쓰면 더 좋음)
- I ≈ 0: 독립적 (각자 효과만 합산)
- I < 0: 간섭 (함께 쓰면 오히려 나빠짐)

### 테스트 조합

서로 다른 레이어 간 조합만 테스트 (같은 레이어 내 조합은 무의미):

- K × H: 5 × 5 = 25
- K × S: 5 × 5 = 25
- K × W: 5 × 5 = 25
- K × R: 5 × 5 = 25
- K × G: 5 × 5 = 25
- H × S: 5 × 5 = 25
- ... (총 15 레이어쌍 × 25 = 375)

합격 기준: I(A,B) > 0 AND combined Sharpe > baseline

---

## Stage 3: Reduced Grid + Walk-Forward Optimization (64~128 조합)

Stage 2에서 시너지가 확인된 조합을 중심으로 Grid Search.

### Walk-Forward Analysis (WFO) 설계

```
전체 기간: 2018-01 ~ 2025-06

Walk-Forward Windows (expanding train + 6mo test):
  Window 1: Train 2018-01~2019-12 → Test 2020-01~2020-06
  Window 2: Train 2018-01~2020-06 → Test 2020-07~2020-12
  Window 3: Train 2018-01~2020-12 → Test 2021-01~2021-06
  Window 4: Train 2018-01~2021-06 → Test 2021-07~2021-12
  Window 5: Train 2018-01~2021-12 → Test 2022-01~2022-06
  Window 6: Train 2018-01~2022-06 → Test 2022-07~2022-12
  Window 7: Train 2018-01~2022-12 → Test 2023-01~2023-06
  Window 8: Train 2018-01~2023-06 → Test 2023-07~2023-12
  Window 9: Train 2018-01~2023-12 → Test 2024-01~2024-06
  Window 10: Train 2018-01~2024-06 → Test 2024-07~2024-12

2025-01~2025-06: 최종 Lockbox (Stage 4에서만 사용)
```

### 합격 기준

1. **OOS Sharpe Ratio**: WFO 10개 윈도우의 OOS Sharpe 중앙값 ≥ 1.0
2. **OOS Degradation**: IS Sharpe 대비 OOS Sharpe 하락 ≤ 30%
3. **Consistency**: OOS 윈도우 중 Sharpe > 0인 비율 ≥ 70%
4. **Parameter Plateau**: 인접 파라미터에서 Sharpe 변화 ≤ 15%

---

## Stage 4: Lockbox Validation (3~5개 최종 후보)

Stage 3 통과 전략을 2025-01 ~ 2025-06 데이터에 적용.

### 최종 선발 기준

1. Lockbox Sharpe > 0.5
2. Lockbox MDD < -40%
3. 전체 기간(2018-2025) Sharpe > baseline
4. 실제 운용 가능성 (일 1회 체크, 거래소 API 호환)

---

## 과적합 방지 (Anti-Overfitting)

1. **Walk-Forward**: IS에서 최적화, OOS에서 검증 (위 참조)
2. **Parameter Plateau**: 특정 값에서만 좋고 주변에서 급감하면 폐기
3. **상관관계 낮은 지표 결합**: 모멘텀+거래량+매크로 등 본질이 다른 지표 결합
4. **Lockbox 봉인**: 2025 데이터는 Stage 4까지 절대 사용 금지
5. **복잡도 페널티**: 동일 성과라면 파라미터 수가 적은 전략 선택

---

## 실행 계획 (Execution Plan)

### 병렬 처리

```python
# Stage 1: 30개 독립 테스트 → ProcessPoolExecutor(max_workers=CPU)
# Stage 2: 375개 쌍 → 동일 병렬
# Stage 3: 64~128개 조합 × 10 WFO 윈도우 → 병렬
# Stage 4: 3~5개 → 순차 (최종 확인)
```

### 자동화 파이프라인

1. `strategy_engine.py` — 통합 백테스트 엔진 (Params 기반)
2. `auto_test.py` — Stage 1→2→3→4 자동 실행 & 결과 저장
3. 각 Stage 결과는 CSV + 요약 리포트로 저장

### 예상 소요

- Stage 1 (30 solo): ~5분 (병렬)
- Stage 2 (375 pair): ~30분 (병렬)
- Stage 3 (128 × 10 WFO): ~2시간 (병렬)
- Stage 4 (5 lockbox): ~1분

---

## 이전 테스트 결과 요약

### 리밸런싱 빈도 테스트 (tx=0.004)

| 빈도 | Sharpe | MDD    | CAGR   |
| ---- | ------ | ------ | ------ |
| 월간 | 1.297  | -48.1% | +71.1% |
| 격주 | 1.104  | -60.4% | +52.6% |
| 주간 | 1.136  | -47.9% | +55.6% |
| 3일  | 1.165  | -60.6% | +57.8% |
| 일간 | 0.905  | -59.1% | +38.8% |

**결론**: 월간 리밸런싱이 모든 지표에서 최우수. tx=0에서도 월간 > 일간.

### 헬스체크 마진 테스트

| 마진          | Sharpe     | 비고           |
| ------------- | ---------- | -------------- |
| 0% (baseline) | 1.297      | 최고           |
| 1%            | ≤ baseline | 모든 변형 열등 |
| 2%            | ≤ baseline |                |
| 3%            | ≤ baseline |                |
| 5%            | ≤ baseline |                |

**결론**: 마진/히스테리시스는 효과 없음. Pick 변경 -3%에 불과.

### 헬스체크 플립률 (일간)

| 지표     | 플립률 | 비고        |
| -------- | ------ | ----------- |
| Mom21    | 5.03%  | 가장 불안정 |
| SMA30    | 4.60%  |             |
| Vol90≤5% | 0.28%  | 매우 안정   |

### 턴오버 원인 분석

| 원인          | 비율 |
| ------------- | ---- |
| 헬스체크 플립 | 58%  |
| 순위 변동     | 42%  |

---

## 알려진 버그

### execute_rebalance() p<=0 버그

```python
# 현재 (버그):
if p <= 0:
    continue  # 보유 코인이 new_holdings에서 누락됨

# 수정:
if p <= 0:
    if t in holdings:
        new_holdings[t] = holdings[t]  # 기존 보유량 유지
    continue
```

가격 데이터 누락 시 보유 코인이 포트폴리오에서 증발하는 문제. 수정 필요.

---

## Codex/Gemini/Claude 합의 우선순위

Codex 추천 단축 리스트: **K3, H3, S1, W5, R1, G3**

3-AI 공통 권고:

1. 레이어별 독립 테스트 먼저 (greedy 아님)
2. Pairwise interaction으로 시너지 탐색
3. Walk-forward로 과적합 검증
4. 2025 lockbox로 최종 확인
