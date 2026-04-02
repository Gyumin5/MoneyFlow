# 바이낸스 선물 숏 전략 연구 히스토리

기준일: 2026-04-02

이 문서는 현재 실거래 중인 선물 롱 전략에 숏 sleeve를 추가할 수 있는지 검토하면서 수행한 실험들을 정리한 기록이다.

## 연구 원칙

- 현재 실거래 카나리 정의를 그대로 사용한다.
  - `4h1`: `SMA240 + hyst 1.5%`
  - `4h2`: `SMA120 + hyst 1.5%`
  - `1h1`: `SMA168 + hyst 1.5%`
- 시간축은 각 전략의 원래 봉을 그대로 사용한다.
  - `4h1`, `4h2`: `4h`
  - `1h1`: `1h`
- 처음 단계에서는 롱 전략과 바로 합치지 않고, 각 카나리 `OFF` 구간에서 숏 sleeve 자체가 의미가 있는지 독립적으로 본다.
- 공개 전략을 참고하더라도, 구현은 일봉이 아니라 현재 시간축에 맞춘 `bar-based` 환산으로 한다.

## 실험 흐름

### 1. OFF sleeve 단독 개념 점검

스크립트:
- [run_short_off_sleeve_test.py](./run_short_off_sleeve_test.py)

목적:
- 숏 sleeve 자체가 전혀 의미가 없는지 먼저 확인

핵심:
- `4h OFF`
- `all OFF`
- `BTC short 50/100%`
- `BTC/ETH short`

요약:
- `4h OFF -> BTC short 100%`는 그나마 의미가 있었음
- `all OFF` 기준 숏은 전반적으로 별로였음

대표 결과:
- `4h_off_btc_short_100`
  - Cal `2.23`
  - CAGR `+43.7%`
  - MDD `-19.6%`
- `4h_off_btc_short_50`
  - Cal `1.97`
  - CAGR `+21.0%`
  - MDD `-10.7%`

주의:
- 이 단계는 최종 구조에 맞춘 실험이라기보다 “숏 sleeve라는 아이디어가 아예 무의미한가”를 본 참고 실험이다.

### 2. CASH 100% 결합형 예비 실험

스크립트:
- [run_short_cash_gate_test.py](./run_short_cash_gate_test.py)

목적:
- 현재 최종 롱 전략에 숏을 바로 덧붙일 수 있는지 예비 확인

핵심:
- 합산 목표가 `CASH 100%`일 때만 숏
- `BTC short 50/100%`
- `BTC < SMA`, `Mom36 < 0`, `Mom720 < 0` 등 약세 필터

해석:
- 결합형 아이디어를 빠르게 점검하는 용도였다.
- 이후 사용자 요구가 “각 카나리별 독립 숏 연구”로 정리되면서, 최종 기준 실험으로는 사용하지 않았다.

### 3. 카나리별 독립 BTC 숏 테스트

스크립트:
- [run_short_per_canary_off_test.py](./run_short_per_canary_off_test.py)

목적:
- `4h1 OFF`, `4h2 OFF`, `1h1 OFF` 각각을 독립적인 숏 트리거로 볼 때 어떤 필터가 나은지 확인

변수:
- `OFF만`
- `OFF + BTC<SMA`
- `OFF + MomShort<0`
- `OFF + BTC<SMA + MomShort<0`
- `OFF + MomShort<0 + MomLong<0`
- `OFF + BTC<SMA + MomShort<0 + MomLong<0`

결과:

#### 4h1 OFF

그나마 의미 있는 유일한 독립 트리거였다.

상위 결과:
- `OFF + BTC<SMA`
  - Cal `0.18`
  - CAGR `+7.2%`
  - MDD `-40.5%`
- `OFF만`
  - Cal `0.17`
  - CAGR `+7.7%`
  - MDD `-44.9%`

해석:
- 단순하게 써도 아주 나쁘진 않았음
- 모멘텀 필터를 더 붙일수록 대체로 더 나빠짐

#### 4h2 OFF

전반적으로 좋지 않았다.

최고 수준도:
- Cal `-0.14`
- CAGR `-8.7%`
- MDD `-62.0%`

#### 1h1 OFF

전반적으로 좋지 않았다.

최고 수준도:
- Cal `-0.22`
- CAGR `-15.9%`
- MDD `-72.0%`

1차 결론:
- 단순 `BTC short 100%` 숏 트리거로는 `4h1 OFF`만 의미가 있었고
- `4h2 OFF`, `1h1 OFF`는 그대로 쓰기 어려웠다.

### 4. 동일 유니버스 숏 선택 실험

스크립트:
- [run_short_per_canary_off_universe_test.py](./run_short_per_canary_off_universe_test.py)

목적:
- 롱과 비슷한 유니버스를 써서 숏 종목을 동적으로 고르면 더 나아지는지 확인

변수:
- 선택 방식
  - `cap_top`
  - `cap_bottom`
  - `mom_short weakest`
  - `mom_long weakest`
  - `mom_blend weakest`
- 선택 개수
  - `1 / 3 / 5`
- 약세 health
  - 없음
  - `mom_short < 0`
  - `mom_short < 0 and mom_long < 0`

결과 요약:
- `4h1`에서 가장 나았던 것은 결국 `cap_top_1_none`
  - Cal `0.17`
  - CAGR `+7.7%`
  - MDD `-44.9%`
- `cap_bottom`이나 `mom weakest` 류는 훨씬 더 나빴다
- `4h2`, `1h1`는 여전히 별로였다

해석:
- “약한 알트 숏”이나 “하위 시총 숏”은 오히려 성능을 망가뜨렸다.
- 숏 쪽은 유동성 큰 코인 위주가 더 맞을 가능성이 드러났다.

### 5. 4h1 OFF 메이저 바스켓 숏

스크립트:
- [run_short_4h1_major_basket_test.py](./run_short_4h1_major_basket_test.py)

목적:
- `4h1 OFF`를 전제로, 고정 메이저 바스켓 분산 숏이 더 나은지 확인

비교 바스켓:
- `BTC`
- `BTC/ETH`
- `BTC/ETH/SOL`
- `BTC/ETH/XRP`

필터:
- 없음
- `BTC < SMA240`

결과:
- `BTC/ETH + BTC<SMA240`
  - Cal `0.33`
  - CAGR `+11.8%`
  - MDD `-35.5%`
- `BTC/ETH`
  - Cal `0.28`
  - CAGR `+11.6%`
  - MDD `-40.9%`
- `BTC` 단일보다 `BTC/ETH` 바스켓이 더 나았음
- `SOL`, `XRP`까지 늘리면 다시 나빠짐

해석:
- 숏은 무조건 단일 BTC보다 “메이저 2종 바스켓”이 더 좋을 수 있었다.
- 다만 고정 바스켓은 실전형 설계로는 한계가 있었다.

### 6. 공개 전략 벤치마크, 고정 메이저 바스켓

스크립트:
- [run_short_public_benchmark_test.py](./run_short_public_benchmark_test.py)

목적:
- ad-hoc 필터 대신 공개된 추세추종 숏 아이디어를 그대로 benchmark로 비교

룰:
- `sma200`
- `tsmom252`
- `donchian50`
- `sma200_and_tsmom252`

주의:
- 이름은 일봉 개념이지만 실제 구현은 각 전략 시간축의 `bar`로 환산했다.
  - `4h`: `200일 = 200*6 bars`
  - `1h`: `200일 = 200*24 bars`

바스켓:
- `BTC`
- `BTC/ETH`
- `BTC/ETH/XRP`

대표 결과:
- `4h1 / BTC-ETH / SMA200 + TSMOM252`
  - Cal `0.86`
  - CAGR `+16.3%`
  - MDD `-19.0%`
- `4h1 / BTC / TSMOM252`
  - Cal `0.69`
- `4h2 / BTC / TSMOM252`
  - Cal `0.59`

해석:
- 공개 전략 benchmark가 앞선 ad-hoc 룰보다 확실히 나았다.
- 특히 `4h1`과 `4h2`는 추세 기반 숏 룰이 유효할 가능성이 보였다.

### 7. 공개 전략 벤치마크, 동적 유니버스

스크립트:
- [run_short_public_dynamic_universe_test.py](./run_short_public_dynamic_universe_test.py)

목적:
- 고정 메이저 바스켓이 아니라, 매 시점 동적 시총 상위 유니버스를 사용한 공개 전략 테스트

유니버스:
- `top3`
- `top5`
- `top8`

룰:
- `sma200`
- `tsmom252`
- `donchian50`
- `sma200_and_tsmom252`

상위 결과:

- `1h1 / top3 / donchian50`
  - Cal `0.53`
  - CAGR `+8.3%`
  - MDD `-15.9%`

- `4h1 / top5 / sma200`
  - Cal `0.41`
  - CAGR `+13.7%`
  - MDD `-33.1%`

- `4h1 / top8 / sma200`
  - Cal `0.34`
  - CAGR `+14.5%`
  - MDD `-43.0%`

- `1h1 / top5 / donchian50`
  - Cal `0.33`
  - CAGR `+9.0%`
  - MDD `-27.4%`

- `4h1 / top3 / donchian50`
  - Cal `0.31`
  - CAGR `+9.5%`
  - MDD `-30.3%`

- `4h2 / top3 / donchian50`
  - Cal `0.30`
  - CAGR `+8.8%`
  - MDD `-28.9%`

해석:
- 동적 유니버스로 바꾸자 그림이 다시 달라졌다.
- `4h1`은 `SMA200` 계열이 더 강했고,
- `4h2`, `1h1`은 `Donchian50`이 상대적으로 나았다.
- 즉 숏 룰은 전략별로 다르게 가져갈 가능성이 생겼다.

### 8. 공개 아이디어 확장판: MACD / RSI / 저항 / 볼밴

스크립트:
- [run_short_public_extended_dynamic_test.py](./run_short_public_extended_dynamic_test.py)

목적:
- 공개형 숏 아이디어를 더 넓게 확인
- 기존 `SMA / TSMOM / Donchian` 추세형과 비교

룰:
- `ema50_200_macd`
  - `EMA50d < EMA200d` and `MACD hist < 0`
- `bear_rsi60`
  - 하락장 + `RSI14 > 60`
- `bear_rsi65`
  - 하락장 + `RSI14 > 65`
- `bear_dyn_resist`
  - 하락장 + `EMA10/SMA20` 저항 근접
- `bear_bb_upper`
  - 하락장 + 볼린저밴드 상단

주의:
- 먼저 `top3 dynamic universe`로 좁혀 빠르게 비교했다.

대표 결과:

#### 4h1

- `ema50_200_macd`
  - Cal `-0.13`
  - CAGR `-7.1%`
  - MDD `-55.8%`
- `bear_rsi60`
  - Cal `-0.13`
  - CAGR `-6.2%`
  - MDD `-47.7%`
- `bear_rsi65`
  - Cal `-0.14`
  - CAGR `-5.5%`
  - MDD `-38.9%`

#### 4h2

- `ema50_200_macd`
  - Cal `-0.13`
  - CAGR `-6.5%`
  - MDD `-51.2%`
- `bear_rsi65`
  - Cal `-0.20`
  - CAGR `-6.6%`
  - MDD `-33.5%`

#### 1h1

- `bear_rsi65`
  - Cal `-0.24`
  - CAGR `-13.7%`
  - MDD `-57.0%`
- `ema50_200_macd`
  - Cal `-0.27`
  - CAGR `-19.6%`
  - MDD `-72.5%`

해석:
- `MACD`, `RSI`, `EMA/SMA 저항`, `볼밴 상단` 계열은 전부 음수였다.
- 즉 현재 OFF 게이트 구조에선 “하락장 반등 숏”보다 “단순 추세 숏”이 더 잘 맞는다.

## 현재까지의 핵심 결론

### 버릴 것

- `4h2 OFF -> BTC short 100%` 단순 규칙
- `1h1 OFF -> BTC short 100%` 단순 규칙
- 하위 시총 숏
- 약한 알트 숏

### 참고 가치가 있는 것

- `4h1 OFF -> BTC/ETH 바스켓 숏`
- `4h1 OFF -> BTC/ETH + BTC<SMA240`

### 현재 가장 유망한 공개 전략형 후보

- `4h1 / top5 / sma200`
- `4h2 / top3 / donchian50`
- `1h1 / top3 / donchian50`

즉 단순히 “모든 OFF에서 BTC를 숏”하는 구조보다,
각 전략의 OFF 구간에 대해 서로 다른 bar-based 공개 전략형 숏 sleeve를 붙이는 쪽이 더 유망해 보인다.
그리고 현재까지는 `MACD/RSI/볼밴/저항` 계열보다 `SMA/Donchian/TSMOM` 계열이 우세하다.

## 아직 안 한 것

- 현재 롱 전략과의 실제 결합형 백테스트
- 현물(일봉) 롱 + 선물 숏 오버레이 백테스트
- 숏 sleeve에 stop-loss까지 포함한 검증

현재 단계의 의미는:
- “숏 sleeve가 붙을 가치가 있는가?”
- “붙는다면 어떤 전략별 OFF 구간에서 어떤 공개 규칙이 나은가?”

를 찾는 탐색 단계다.

## 관련 파일

- OFF sleeve 개념 실험: [run_short_off_sleeve_test.py](./run_short_off_sleeve_test.py)
- CASH gate 결합 예비 실험: [run_short_cash_gate_test.py](./run_short_cash_gate_test.py)
- 카나리별 독립 BTC 숏: [run_short_per_canary_off_test.py](./run_short_per_canary_off_test.py)
- 카나리별 동적 유니버스 숏: [run_short_per_canary_off_universe_test.py](./run_short_per_canary_off_universe_test.py)
- 4h1 메이저 바스켓 숏: [run_short_4h1_major_basket_test.py](./run_short_4h1_major_basket_test.py)
- 공개 전략 benchmark: [run_short_public_benchmark_test.py](./run_short_public_benchmark_test.py)
- 공개 전략 dynamic universe: [run_short_public_dynamic_universe_test.py](./run_short_public_dynamic_universe_test.py)
