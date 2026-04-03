# Cap Defend 전략 디렉터리 안내

이 디렉터리는 주식, 현물 코인, 선물 전략의 백테스트/실거래 관련 코드를 함께 담고 있다.

처음 보는 경우에는 모든 파일을 읽기보다 아래 순서대로 보는 것이 가장 빠르다.

## 1. 바로 재현하는 공식 경로

### 통합 가이드

- [repo_backtest_guide.md](./repo_backtest_guide.md)

### 현물 코인

- 설명:
  - [coin_backtest_howto.md](./coin_backtest_howto.md)
- 실행:
  - [backtest_official.py](./backtest_official.py)
- 엔진:
  - [coin_engine.py](./coin_engine.py)
  - [coin_helpers.py](./coin_helpers.py)
  - [coin_dd_exit.py](./coin_dd_exit.py)

### 주식

- 설명:
  - [stock_backtest_howto.md](./stock_backtest_howto.md)
- 실행:
  - [backtest_official.py](./backtest_official.py)
- 엔진:
  - [stock_engine.py](./stock_engine.py)

### 선물

- 설명:
  - [futures_backtest_howto.md](./futures_backtest_howto.md)
  - [futures_strategy_final.md](./futures_strategy_final.md)
  - [futures_research_history.md](./futures_research_history.md)
- 실행:
  - [download_futures_data.py](./download_futures_data.py)
  - [run_final_signal_compare.py](./run_final_signal_compare.py)
- 엔진:
  - [backtest_futures_full.py](./backtest_futures_full.py)

참고:
- `run_final_signal_compare.py`의 `current_live`가 현재 실제 실거래 조합이다.
- `old_live`는 과거 비교 기준이다.

## 2. 공통 유틸리티

- 데이터 최신성 확인:
  - [check_data_freshness.py](./check_data_freshness.py)
- 데이터 갱신 진입점:
  - [refresh_backtest_data.py](./refresh_backtest_data.py)

## 3. 연구 / 탐색 파일

아래 파일들은 현재 공식 실행 경로가 아니라, 전략 탐색이나 중간 연구 기록용이다.

### 선물 조합 탐색

- [run_signal_screen.py](./run_signal_screen.py)
- [run_signal_combo_search.py](./run_signal_combo_search.py)
- [run_snap_finetune.py](./run_snap_finetune.py)
- [run_snap_robustness.py](./run_snap_robustness.py)
- [run_ensemble.py](./run_ensemble.py)

### 선물 리스크 / 스탑 / 레버리지 연구

- [run_stoploss_test.py](./run_stoploss_test.py)
- [run_stoploss_cooldown_test.py](./run_stoploss_cooldown_test.py)
- [run_stoploss_long_lookback_test.py](./run_stoploss_long_lookback_test.py)
- [run_dynamic_leverage_test.py](./run_dynamic_leverage_test.py)
- [run_conditional_atr_test.py](./run_conditional_atr_test.py)
- [run_pfd_tuning_test.py](./run_pfd_tuning_test.py)
- [run_top5ew_compare.py](./run_top5ew_compare.py)

### 선물 숏 연구

- [futures_short_research_history.md](./futures_short_research_history.md)
- [run_short_off_sleeve_test.py](./run_short_off_sleeve_test.py)
- [run_short_cash_gate_test.py](./run_short_cash_gate_test.py)
- [run_short_per_canary_off_test.py](./run_short_per_canary_off_test.py)
- [run_short_per_canary_off_universe_test.py](./run_short_per_canary_off_universe_test.py)
- [run_short_4h1_major_basket_test.py](./run_short_4h1_major_basket_test.py)
- [run_short_public_benchmark_test.py](./run_short_public_benchmark_test.py)
- [run_short_public_dynamic_universe_test.py](./run_short_public_dynamic_universe_test.py)
- [run_short_public_extended_dynamic_test.py](./run_short_public_extended_dynamic_test.py)
- [run_short_spot_on_all_futures_off_overlay_test.py](./run_short_spot_on_all_futures_off_overlay_test.py)
- [run_futures_spot_on_all_off_overlay_combo_test.py](./run_futures_spot_on_all_off_overlay_combo_test.py)
- [run_futures_spot_on_all_off_overlay_sweep.py](./run_futures_spot_on_all_off_overlay_sweep.py)

### 기타 과거 연구 / 통합 실험

- [backtest_coin_tranche.py](./backtest_coin_tranche.py)
- [backtest_stock_variants.py](./backtest_stock_variants.py)
- [backtest_integrated_60_40.py](./backtest_integrated_60_40.py)
- [backtest_rebal_sweep.py](./backtest_rebal_sweep.py)
- [backtest_rwa_defense.py](./backtest_rwa_defense.py)
- [backtest_rwa_gold.py](./backtest_rwa_gold.py)
- [backtest_rwa_gold_2.py](./backtest_rwa_gold_2.py)
- [backtest_rwa_v2.py](./backtest_rwa_v2.py)

## 4. 운영 관련

- 추천/리포트 생성:
  - [recommend.py](./recommend.py)
  - [recommend_personal.py](./recommend_personal.py)
- 보조 스크립트:
  - [daily_history.py](./daily_history.py)
  - [serve.py](./serve.py)

## 한 줄 요약

바로 백테스트하려면:

1. [repo_backtest_guide.md](./repo_backtest_guide.md)
2. `refresh_backtest_data.py`
3. `check_data_freshness.py`
4. 각 전략의 공식 실행 스크립트

만 보면 된다.
