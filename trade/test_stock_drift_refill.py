"""executor_stock.py V25 drift-refill 단위 검증.

채택 BT(bt_stock_mom3) 정합성 + 이벤트/상태기계 가드 확인.
KIS API 불필요 (refill_snaps_fresh + 드리프트 게이트 산술만 검증).
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def half_t(cur_w, tgt):
    keys = set(cur_w) | set(tgt)
    return sum(abs(cur_w.get(k, 0.0) - tgt.get(k, 0.0)) for k in keys) / 2


def build_target_buf(merged, buf):
    tb = {k: float(w) * (1.0 - buf) for k, w in merged.items()}
    tb['Cash'] = tb.get('Cash', 0.0) + max(0.0, 1.0 - sum(tb.values()))
    return tb


def main():
    import executor_stock as ex

    N = ex.N_SNAPS
    BUF = ex.CASH_BUFFER_DEFAULT
    cap = 1.0 / 3.0

    def fresh_snaps(picks):
        w = {t: cap for t in picks}
        cs = 1.0 - cap * len(picks)
        if cs > 1e-9:
            w['Cash'] = cs
        snaps = {}
        for i in range(N):
            snaps[str(i)] = {'picks': list(picks), 'weights': dict(w),
                             'last_rebal_date': '2026-04-30'}
        return snaps

    fails = []

    # --- T1: refill 이 snap picks 를 fresh 로 교체, last_rebal_date 보존 ---
    state = {'snapshots': fresh_snaps(['EEM', 'PDBC', 'QQQ'])}
    signal = {'stock': {'risk_on': True,
                        'offense_picks': ['EEM', 'VEA', 'QQQ'],
                        'offense_weights': {'EEM': cap, 'VEA': cap, 'QQQ': cap}}}
    changed = ex.refill_snaps_fresh(signal, state)
    merged = ex.merge_tranches(state)
    if not changed:
        fails.append('T1: changed=False (교체 감지 실패)')
    if set(k for k in merged if k != 'Cash') != {'EEM', 'VEA', 'QQQ'}:
        fails.append(f'T1: merged picks 불일치 {merged}')
    if any(state['snapshots'][str(i)]['last_rebal_date'] != '2026-04-30' for i in range(N)):
        fails.append('T1: last_rebal_date 변경됨 (앵커 cadence 훼손)')

    # --- T2: picks 동일 시 no-op (changed=False) ---
    state2 = {'snapshots': fresh_snaps(['EEM', 'PDBC', 'QQQ'])}
    sig2 = {'stock': {'risk_on': True,
                      'offense_picks': ['EEM', 'PDBC', 'QQQ'],
                      'offense_weights': {'EEM': cap, 'PDBC': cap, 'QQQ': cap}}}
    if ex.refill_snaps_fresh(sig2, state2):
        fails.append('T2: 동일 picks 인데 changed=True')

    # --- T3: 드리프트 게이트 — old target 기준 ht (BT 순서) ---
    # 보유가 옛 target(EEM/PDBC/QQQ)과 거의 같으면 ht<5pp → 발화 안 함
    old_merged = ex.merge_tranches({'snapshots': fresh_snaps(['EEM', 'PDBC', 'QQQ'])})
    tb_old = build_target_buf(old_merged, BUF)
    cur_same = dict(tb_old)  # 보유=옛 target
    ht_same = half_t(cur_same, tb_old)
    if ht_same >= 0.05:
        fails.append(f'T3a: 보유=옛target 인데 ht={ht_same:.3f}≥5pp (오발화)')
    # PDBC 를 통째로 다른 종목이 대체할 만큼 보유가 쏠리면 ht≥5pp
    cur_drift = dict(tb_old)
    # PDBC 비중을 VEA 로 옮겨 옛 target 대비 큰 편차 생성
    moved = cur_drift.pop('PDBC', 0.0)
    cur_drift['VEA'] = cur_drift.get('VEA', 0.0) + moved
    ht_drift = half_t(cur_drift, tb_old)
    if ht_drift < 0.05:
        fails.append(f'T3b: PDBC→VEA 쏠림인데 ht={ht_drift:.3f}<5pp (미발화)')

    # --- T4: defense (risk_off) 경로도 리필 ---
    state4 = {'snapshots': fresh_snaps(['EEM', 'PDBC', 'QQQ'])}
    sig4 = {'stock': {'risk_on': False,
                      'defense_picks': ['IEF', 'BIL'],
                      'defense_weights': {'IEF': 0.5, 'BIL': 0.5}}}
    ex.refill_snaps_fresh(sig4, state4)
    m4 = ex.merge_tranches(state4)
    if set(k for k in m4 if k != 'Cash') != {'IEF', 'BIL'}:
        fails.append(f'T4: defense 리필 실패 {m4}')

    # --- T5: 빈 signal 은 no-op (picks/weights 둘다 없음) ---
    state5 = {'snapshots': fresh_snaps(['EEM', 'PDBC', 'QQQ'])}
    if ex.refill_snaps_fresh({'stock': {'risk_on': True,
                                        'offense_picks': [], 'offense_weights': {}}}, state5):
        fails.append('T5: 빈 signal 인데 changed=True')
    if set(k for k in ex.merge_tranches(state5) if k != 'Cash') != {'EEM', 'PDBC', 'QQQ'}:
        fails.append('T5: 빈 signal 인데 기존 picks 훼손')

    # --- T6: flag OFF 면 anchor-only (refill 함수 자체는 존재하나 호출부 가드) ---
    if not hasattr(ex, 'STOCK_DRIFT_REFILL'):
        fails.append('T6: STOCK_DRIFT_REFILL 플래그 없음')

    # --- T7: 카나리 OFF flip → 모든 snapshot 즉시 방어로 교체 (앵커 안 기다림) ---
    state7 = {'prev_risk_on': True, 'snapshots': fresh_snaps(['EEM', 'PDBC', 'QQQ'])}
    sig7 = {'stock': {'risk_on': False,
                      'offense_picks': ['EEM', 'QQQ', 'VEA'],
                      'offense_weights': {'EEM': cap, 'QQQ': cap, 'VEA': cap},
                      'defense_picks': ['IEF', 'BIL', 'GLD'],
                      'defense_weights': {'IEF': cap, 'BIL': cap, 'GLD': cap}}}
    flipped = ex.check_canary_flip(sig7, state7)
    m7 = ex.merge_tranches(state7)
    if not flipped:
        fails.append('T7: 카나리 flip 미감지')
    if set(k for k in m7 if k != 'Cash') != {'IEF', 'BIL', 'GLD'}:
        fails.append(f'T7: 카나리 OFF 인데 방어 전환 안 됨 {m7}')
    if not state7.get('rebalancing_needed'):
        fails.append('T7: rebalancing_needed 미설정')
    if any(state7['snapshots'][str(i)]['last_rebal_date'] != '2026-04-30' for i in range(N)):
        fails.append('T7: 카나리 flip 이 last_rebal_date 변경 (앵커 cadence 훼손)')

    # --- T8: 카나리 변화 없으면 flip=False, snapshot 보존 ---
    state8 = {'prev_risk_on': True, 'snapshots': fresh_snaps(['EEM', 'PDBC', 'QQQ'])}
    sig8 = {'stock': {'risk_on': True,
                      'offense_picks': ['EEM', 'QQQ', 'VEA'],
                      'offense_weights': {'EEM': cap, 'QQQ': cap, 'VEA': cap},
                      'defense_picks': ['IEF', 'BIL', 'GLD'],
                      'defense_weights': {'IEF': cap, 'BIL': cap, 'GLD': cap}}}
    if ex.check_canary_flip(sig8, state8):
        fails.append('T8: 카나리 동일한데 flip=True')
    if set(k for k in ex.merge_tranches(state8) if k != 'Cash') != {'EEM', 'PDBC', 'QQQ'}:
        fails.append('T8: 카나리 동일한데 snapshot 변경됨')

    print(f"# drift-refill 단위 검증 (N_SNAPS={N}, buf={BUF}, cap={cap:.3f})")
    print(f"  T3 ht: same={ht_same:.4f}  drift={ht_drift:.4f}")
    print(f"  merged after refill: {merged}")
    if fails:
        print("FAIL:")
        for f in fails:
            print("  -", f)
        sys.exit(1)
    print("ALL PASS (T1~T8)")


if __name__ == '__main__':
    main()
