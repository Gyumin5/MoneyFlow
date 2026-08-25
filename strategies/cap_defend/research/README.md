# 연구 / 탐색 파일

이 디렉터리는 전략 탐색, 파라미터 튜닝, 강건성 테스트 등 연구 과정에서 사용된 스크립트를 모아둔 곳이다.

**공식 백테스트 재현에는 이 디렉터리가 필요 없다.** 공식 경로는 상위 디렉터리의 `run_current_*_backtest.py`만 사용한다.

이 스크립트들은 작성 당시의 엔진 버전을 기준으로 만들어졌으므로, 엔진 구조가 변경되면 바로 실행되지 않을 수 있다.

## 파일 분류

### 선물 조합 탐색

- `run_signal_screen.py` — 1h/4h 후보 전략군 단독 스크리닝
- `run_signal_combo_search.py` — 상위 후보의 2/3/4개 조합 비교
- `run_snap_finetune.py` — snap_interval_bars 미세조정
- `run_snap_robustness.py` — 비정렬 구간 robustness 확인
- `run_ensemble.py` — 단일 계정 앙상블 백테스트 (원본)

### 선물 리스크 / 스탑 / 레버리지 연구

- `run_stoploss_test.py` — 스탑로스 배치 테스트
- `run_stoploss_cooldown_test.py` — 스탑 후 쿨다운 테스트
- `run_stoploss_long_lookback_test.py` — 장기 룩백 스탑
- `run_dynamic_leverage_test.py` — 동적 레버리지 테스트
- `run_conditional_atr_test.py` — 조건부 ATR 스탑
- `run_pfd_tuning_test.py` — PFD 튜닝
- `run_top5ew_compare.py` — Top5 EW 비교

### 선물 숏 연구

- `futures_short_research_history.md` — 숏 전략 연구 기록
- `run_short_off_sleeve_test.py`
- `run_short_cash_gate_test.py`
- `run_short_per_canary_off_test.py`
- `run_short_per_canary_off_universe_test.py`
- `run_short_4h1_major_basket_test.py`
- `run_short_public_benchmark_test.py`
- `run_short_public_dynamic_universe_test.py`
- `run_short_public_extended_dynamic_test.py`
- `run_short_spot_on_all_futures_off_overlay_test.py`
- `run_futures_spot_on_all_off_overlay_combo_test.py`
- `run_futures_spot_on_all_off_overlay_sweep.py`

### 기타 과거 연구 / 통합 실험

- `backtest_coin_tranche.py` — 코인 트랜치 실험
- `backtest_stock_variants.py` — 주식 변형 실험
- `backtest_integrated_60_40.py` — 60:40 통합 백테스트
- `backtest_rebal_sweep.py` — 리밸런싱 스윕
- `backtest_rwa_defense.py` / `backtest_rwa_gold.py` / `backtest_rwa_gold_2.py` / `backtest_rwa_v2.py` — RWA 방어 전략

### 현행(V24/V25) 검증 스크립트

여기 있는 대부분은 V21 이전 탐색이라 바로 안 돌아간다. 아래는 현행 엔진 기준으로 쓰인 것들이다.

- `parity_spot.py` / `parity_fut.py` — 라이브 선정함수와 채택 BT 에 같은 입력을 넣어 일별 종목·비중이
  일치하는지 증명하는 read-only 하니스 (2026-06-06, 2,000여 일 100% 일치)
- `bt_spot_refill_vs_anchor.py` — 현물 drift 발화 시 refill 이 앵커 전용과 같은 결과인지 확인
- `bt_fut_rebal_band.py` (+ `.engine.patch`) — 선물 체결 밴드 스윕. 정본 엔진은 env 토글로만 바꾸고
  측정 후 원복한다. 패치는 재현용으로만 보관

변형 실험은 정본 엔진(`unified_backtest.py`, `backtest_futures_v25.py`)에 코드를 남기지 않는다.
env 토글로 재고 끝나면 되돌리며(diff 0), 재현 패치만 `*.engine.patch` 로 남긴다.

### 연구 결과 요약 (읽을 가치가 남은 것)

- `futures_research_history.md` — 선물 연구 히스토리
- `futures_short_research_history.md` — 숏 전략 연구 기록 (숏은 채택된 적 없다)
- `stoploss_summary.md` — 스탑로스 연구 요약 (현행 세 sleeve 는 스탑이 없다. 그 근거)
- `T3O_TRIGGER_FINDINGS.md` — 자산간 T3O 트리거. 기각됨
- `reentry_variants_registry.md` — 빈 스냅 재진입 22개 변형. 탐색 종료

V21 시절 재설계 플랜·phase0 리포트류는 2026-08-25 에 삭제했다(전부 V22→V24→V25 로 대체됨).
결론은 `../STRATEGY_EVOLUTION.md` 와 `../../../history/V21_HISTORY.md`, 원문은 git 이력에 있다.
