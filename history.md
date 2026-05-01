## [2026-04-30] V23 도입 (모든 자산 1D 단일 + drift trigger)
tags: V23, 마이그레이션, 코인, 선물, 주식, drift
- 결정: 모든 자산 V23 통일. spot D_SMA42 sn=217 n=7 drift=0.10, fut D_SMA42 sn=57 n=3 drift=0.05 L3, stock SNAP_PERIOD=69 STAGGER=23 N_SNAPS=3. 4h 멤버 제거, cron 4h x 6 → 1d x 1 (5 9 * * *)
- 근거: 5.4yr BT (rank-sum across alloc, 평탄도 300 BT, 차별화·서로소 2,220 portfolio, C 후보 직도입 검증) 통과 + AI 합의 (gemini+codex). spot/fut n_snap 서로소 (7/3 gcd=1), stagger 모두 distinct prime (23/31/19). drift trigger: half_turnover(cur_w, tgt_w) >= threshold, cur_w 자본금 기준
- 코드: trade/{coin_live_engine,executor_coin,executor_stock,auto_trade_binance,migrate_v22_to_v23}.py + strategies/cap_defend/{futures_live_config,recommend,recommend_personal}.py + trade/ops/crontab.txt + V23_OPERATION_MANUAL.md
- codex 검토 반영 (조건부 NO-GO → GO): (1) 매뉴얼 마이그레이션 절차 보정 (cron 활성화는 수동 dry-run 검증 통과 후 마지막), (2) DRIFT_ENABLED / DRIFT_ENABLED_FUT 런타임 플래그 추가 (코드 수정 없이 snap-only fallback), (3) 첫 주 tripwire 추가 (spot >2/day or 8/wk → 0.12~0.15 / fut >1/day or 4/wk → 0.07~0.10)
- 되돌릴 조건: drift whipsaw 시 DRIFT_ENABLED=False 토글 → snap-only 운영. yearly Cal 급락 시 migrate --rollback 으로 V22 복귀
- 서버 배포: 사용자 승인 후 Phase F 진행 예정

## [2026-04-28] V22 일괄 적용 정합성 정리
tags: V22, 운영, 문서정리
- 결정: 모든 자산 (주식/코인/선물) V22 단일 표기. 이전 버전 표기 제거 (progress/history active 영역, MEMORY, recommend docstring 등)
- 근거: 사용자 명시 — 운영은 모두 V22, 옛 버전 표기는 혼선만 야기
- 되돌릴 조건: 새로운 버전(V23) 적용 시점에 동일 절차로 갱신
- archive: 03/04-early 월별로 history/ 하위 이동 (보관)

## [2026-04-27] V22 마이그레이션 (코인/선물/주식 동기)
tags: V22, 마이그레이션, 코인, 선물, 주식
- 결정: 코인·선물·주식 모두 V22 통일. 코인/선물 = 1D + 4h 2멤버 EW (snap 60일/90일 동기), 주식 = snap-based 3-tranche stagger (SNAP_PERIOD=126, STAGGER=42, N_SNAPS=3)
- 근거: AI 합의 (Gemini+Codex) — interval 다양성, plateau-center cfg 채택, 가드 분산방어 일원화. 자산배분 60/40/0 수동 유지 (선물 0%)
- 되돌릴 조건: 다음 라운드 백테스트에서 단일 interval 회귀가 robust 하다고 입증되면

## [2026-04-22] V22 코인 C 슬리브 비활성
tags: V22, 코인, C슬리브
- 결정: V22 C 슬리브(champion mean-reversion) cap=0 으로 비활성. 코드/문서 유지, 단독 sleeve 운용 안 함
- 근거: Upbit 1h 등가 백테스트 탈락 — 단독 Cal 3.10(가드없음 D ensemble) 대비 C+D 등가 Cal 1.46, MDD -19% → -44%. champion 이 Binance 1h 기반이라 Upbit 등가 검증 불가
- 되돌릴 조건: Upbit-aware champion 재탐색 결과 등가 sleeve 가 robust 하면 cap_per_slot 복원

## [2026-04-21] V22 가드 전면 제거
tags: V22, 가드, 분산방어
- 결정: stop/cash_guard 본체 코드 삭제, 앙상블 분산이 유일 방어. rebalancing_needed 플래그로 통일
- 근거: 가드 ON/OFF ablation 에서 ON 시 OOS Cal 손실 일관 관찰. 분산방어가 더 robust
- 되돌릴 조건: 시스템 서킷브레이커(-20%) 도입 검토 시 별도 layer 로 추가
