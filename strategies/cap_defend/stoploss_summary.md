# 스탑로스 연구 요약

기간: 2020-10-01 ~ 2026-03-28  
대상 기본 전략: `4h1 + 4h2 + 1h1` 앙상블  
기본 선택/비중: 시총순 Top5 -> Greedy Absorption, EW + Cap 33%  
공통 설정: DD Exit OFF, Blacklist OFF

## 1차 스탑로스 스캔

파일: [stoploss_results.csv](/home/gmoh/mon/251229/strategies/cap_defend/stoploss_results.csv)

비교 범위:
- 배수: `2x`, `3x`, `4x`, `5x`
- 스탑:
  - `none`
  - `prev_close_pct`
  - `highest_close_since_entry_pct`
  - `highest_high_since_entry_pct`
  - `rolling_high_close_pct`
  - `rolling_high_high_pct`
- 스탑 폭: `3%, 5%, 7%, 10%, 12%, 15%`
- 롤링 룩백: `3, 6, 12, 24`

핵심 결론:
- always-on 스탑로스는 전반적으로 `none`을 이기지 못했다.
- 가장 덜 나쁜 단순 스탑은 `prev_close 15%`.
- 촘촘한 스탑은 청산은 줄여도 `Stops`가 폭증해 CAGR/Calmar를 크게 훼손했다.

주요 baseline:
- `2x none`: Cal `3.39`, MDD `-34.8%`, Liq `2`
- `3x none`: Cal `3.63`, MDD `-49.7%`, Liq `5`
- `4x none`: Cal `3.58`, MDD `-59.4%`, Liq `14`
- `5x none`: Cal `3.91`, MDD `-69.0%`, Liq `26`

단순 스탑 최선:
- `2x prev_close 15%`: Cal `3.14`, Liq `0`
- `3x prev_close 15%`: Cal `3.30`, Liq `1`
- `4x prev_close 15%`: Cal `3.54`, Liq `9`
- `5x prev_close 15%`: Cal `3.72`, Liq `21`

## 모델 보정

같은 봉에서 stop 과 liquidation 이 모두 걸릴 때:
- 롱 기준 `stop_price > liq_price` 이면 스탑 우선
- 그렇지 않으면 청산 우선

이 보정으로 저배수에서 "스탑이 더 위인데도 청산으로 잡히는" 과보수 문제를 줄였다.

## 2차: 재진입 쿨다운

파일: [stoploss_cooldown_results.csv](/home/gmoh/mon/251229/strategies/cap_defend/stoploss_cooldown_results.csv)

비교 범위:
- 상위 스탑 후보
- 쿨다운: `0, 6, 12, 24, 48, 72, 168` bars

결론:
- 쿨다운은 대부분 의미 있는 개선을 만들지 못했다.
- 일부 rolling 후보에서 소폭 개선은 있었지만 전체 최고 조합은 바뀌지 않았다.

## 3차: 긴 롤링 룩백

파일: [stoploss_long_lookback_results.csv](/home/gmoh/mon/251229/strategies/cap_defend/stoploss_long_lookback_results.csv)

비교 범위:
- rolling 계열 상위 후보
- 룩백: base + `48, 72, 96, 168, 336`

결론:
- 긴 룩백도 뚜렷한 개선이 없었다.
- 최선은 기존 짧은/중간 룩백 (`12`, `24`) 근처에 머물렀다.

## 4차: 조건부 스탑 + ATR 스탑

파일: [conditional_atr_results.csv](/home/gmoh/mon/251229/strategies/cap_defend/conditional_atr_results.csv)

비교 범위:
- `none`
- `prev_close 15%`
- `rolling_high_high 15%, lb=24`
- ATR 기반 2종
- 게이트:
  - `always`
  - `target_exit_only`
  - `cash_guard`
  - `target_exit_or_cash_guard`

기본 cash guard threshold:
- `34%`

핵심 결론:
- 조건부 스탑은 유효했다.
- ATR 스탑은 이번 전략/파라미터에선 탈락.
- 최선은 `prev_close 15% + cash_guard(34%)`.

주요 결과:
- `3x none`: Cal `3.63`, MDD `-49.7%`, Liq `5`
- `3x prev_close15 + cash_guard(34%)`: Cal `3.62`, MDD `-49.4%`, Liq `1`
- `4x none`: Cal `3.58`, MDD `-59.4%`, Liq `14`
- `4x prev_close15 + cash_guard(34%)`: Cal `3.72`, MDD `-59.4%`, Liq `11`
- `5x none`: Cal `3.91`, MDD `-69.0%`, Liq `26`
- `5x prev_close15 + cash_guard(34%)`: Cal `4.01`, MDD `-69.0%`, Liq `24`

해석:
- `3x`에서는 수익성은 거의 유지하면서 청산 리스크를 크게 줄였다.
- `4x`, `5x`에서는 성과도 baseline 보다 개선됐다.

## cash_guard threshold 해석

`cash_guard`는 목표 포트폴리오의 `CASH` 비중이 임계치 이상일 때만 스탑을 켜는 방식이다.

전략의 target 구조상 `CASH` 비중은 연속적이라기보다 대체로:
- `0%`
- `34%` 전후
- `100%`

로 나타나는 경우가 많다.

따라서:
- `cash > 0`는 스탑이 너무 자주 켜질 수 있음
- `cash >= 34%`는 partial defense 상태부터 켜는 효과

실험상 `cash >= 34%`가 가장 균형이 좋았다.

## 5차: Top5 EW 비교

파일: [top5ew_compare_results.csv](/home/gmoh/mon/251229/strategies/cap_defend/top5ew_compare_results.csv)

비교 대상:
- 기존 Greedy 대신 `selection='baseline'`, `cap=1.0`
- 즉 시총순 Top5 Equal Weight
- 케이스:
  - `none`
  - `prev_close 15% + cash_guard(20%)`
  - `prev_close 15% + cash_guard(34%)`
  - `prev_close 15% + cash_guard(40%)`

결론:
- Top5 EW는 Greedy Absorption 보다 전반적으로 열세.
- 특히 고배수에서 MDD/청산이 크게 증가했다.
- 상대적으로는 `4x + cash_guard(20%)`가 제일 나았지만, 전체 추천 축으로는 부적합.

주요 결과:
- `2x none`: Cal `2.17`, MDD `-50.8%`, Liq `3`
- `3x none`: Cal `2.41`, MDD `-66.9%`, Liq `9`
- `4x prev_close15 + cash20`: Cal `1.74`, MDD `-81.3%`, Liq `13`
- `5x none`: Cal `1.29`, MDD `-89.2%`, Liq `39`

## 현재 최종 추천안

1. 보수형
- `3x + prev_close 15% + cash_guard(34%)`
- Cal `3.62`, MDD `-49.4%`, Liq `1`

2. 균형형
- `4x + prev_close 15% + cash_guard(34%)`
- Cal `3.72`, MDD `-59.4%`, Liq `11`

3. 공격형
- `5x + prev_close 15% + cash_guard(34%)`
- Cal `4.01`, MDD `-69.0%`, Liq `24`

## 6차: 동적 레버리지

파일: [dynamic_leverage_results.csv](/home/gmoh/mon/251229/strategies/cap_defend/dynamic_leverage_results.csv)

비교 대상:
- 고정 `3x`, `4x`, `5x`
- 계정 단위 5/4/3:
  - `cash_based_543`
  - `count_based_543`
  - `canary_based_543`
  - `mixed_score_543`
- 종목 단위 5/4/3:
  - `rank_543_cash`
  - `momrank_543_cash`
  - `lowvol_543_cash`
  - `cap_mom_blend_543_cash`
- stop:
  - `none`
  - `prev_close 15% + cash_guard(34%)`

핵심 결론:
- 계정 단위 동적 레버리지보다 종목 단위 동적 레버리지가 더 좋았다.
- 특히 `mom` 기반과 `cap+mom` 기반이 고정 `5x`보다 우수했다.
- low-vol 기반은 상대적으로 약했다.

Top 5:
1. `coin_capmom_543_cash + prev_close15 + cash_guard(34%)`
- Cal `4.30`, CAGR `+250.3%`, MDD `-58.2%`, Liq `7`, Stops `11`

2. `coin_mom_543_cash + prev_close15 + cash_guard(34%)`
- Cal `4.29`, CAGR `+249.9%`, MDD `-58.2%`, Liq `6`, Stops `11`

3. `coin_capmom_543_cash + none`
- Cal `4.14`, CAGR `+241.6%`, MDD `-58.4%`, Liq `12`

4. `coin_mom_543_cash + none`
- Cal `4.13`, CAGR `+241.2%`, MDD `-58.4%`, Liq `11`

5. `fixed_5x + prev_close15 + cash_guard(34%)`
- Cal `4.01`, CAGR `+276.3%`, MDD `-69.0%`, Liq `24`

해석:
- 고정 `5x`는 CAGR은 더 높지만 MDD와 청산이 너무 크다.
- `coin_mom` / `coin_capmom`은 MDD를 약 `11%p` 줄이고, 청산도 크게 줄이면서 Cal을 더 높였다.
- 현재까지의 전체 실험에서 가장 유망한 업그레이드는
  `종목별 5/4/3 동적 레버리지 + 조건부 스탑`이다.

## 제외 권고

- always-on 스탑
- 촘촘한 퍼센트 스탑 (`3~10%`)
- 긴 롤링 룩백 추가 탐색
- 재진입 쿨다운
- ATR 스탑
- Top5 EW

## 다음 후보 작업

## 7차: 최종 신호 조합 재탐색

파일:
- [signal_screen_1h.csv](/home/gmoh/mon/251229/strategies/cap_defend/signal_screen_1h.csv)
- [signal_screen_4h.csv](/home/gmoh/mon/251229/strategies/cap_defend/signal_screen_4h.csv)
- [signal_combo_search.csv](/home/gmoh/mon/251229/strategies/cap_defend/signal_combo_search.csv)
- [snap_finetune_results.csv](/home/gmoh/mon/251229/strategies/cap_defend/snap_finetune_results.csv)
- [snap_robustness_results.csv](/home/gmoh/mon/251229/strategies/cap_defend/snap_robustness_results.csv)

비교 범위:
- 기존 `4h1 + 4h2 + 1h1`를 고정 정답으로 두지 않고
- 1h 후보군 / 4h 후보군을 다시 스크리닝
- `2개 / 3개 / 4개` 조합 탐색
- 실행층은 고정:
  - `coin_capmom_543_cash`
  - `prev_close 15%`
  - `cash_guard(34%)`

상위 조합:
1. `1h_09 + 1h_05 + 4h_01 + 4h_08`
2. `1h_09 + 4h_01 + 4h_09`
3. `4h_01 + 4h_09`

트랜치 미세조정 결과:
- `1h_09`는 `24`보다 `21/27/33`이 우수
- `4h_01`은 `120` 유지가 최선
- `4h_09`는 `18`이 최고였으나 시간대 정렬 완화를 위해 `21` 채택

최종 채택:
- `1h_09(snap=27)`
- `4h_01(snap=120)`
- `4h_09(snap=21)`

동일 실행층에서 기존 전략과 직접 비교:
- 기존 `4h1 + 4h2 + 1h1`
  - Cal `4.30`
  - CAGR `+250.3%`
  - MDD `-58.2%`
  - Liq `7`
- 최종 `1h_09 + 4h_01 + 4h_09`
  - Cal `4.98`
  - CAGR `+221.1%`
  - MDD `-44.4%`
  - Liq `4`

해석:
- CAGR은 다소 낮아졌지만
- MDD와 청산 횟수를 줄이면서 Calmar를 크게 개선
- 실거래 운영 기준으로는 최종 교체 가치가 충분하다고 판단

최종 전략 설명:
- [futures_strategy_final.md](/home/gmoh/mon/251229/strategies/cap_defend/futures_strategy_final.md)

- 실매매 코드에 조건부 스탑 반영 여부 결정
- 실거래에서 stop order 체결 정책 구체화
- 필요 시 3x / 4x / 5x 실전 배치 규칙 분리
