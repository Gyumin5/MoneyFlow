"""V17 V21-style snapshot ensemble backtester (v2 — Codex 리뷰 반영).

수정사항:
1. resolve_canary() 재사용 — hysteresis (prev_risk_on) 정확 재현. snap별 독립 state.
2. 리밸 = "목표 비중 일괄 체결" (매도 → 매수, tx 양방향)
3. Crash 당일 재매수 금지 (crash_cooldown 중엔 snap rebal skip)
4. 스케줄: 월단위 anchor (monthly_anchor_mode=True, 기본) or calendar-day
5. DD/health_daily_exit/flip_rebal: snapshot 모드 비활성화 (명시)
6. canary_type/canary_band/canary_extra: resolve_canary() 가 처리 → 자동 지원
7. select_offensive 는 n_mom 파라미터로 top N 제어
8. Phase-3 검증: monthly_anchor_mode=True + n_snap=1 로 run_bt 재현성 비교
"""
from __future__ import annotations
import os, sys
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import replace as _dc_replace
from stock_engine import (
    SP, get_price, get_val, resolve_canary,
    filter_healthy, select_offensive, select_defensive,
)


def compute_target_with_canary(params: SP, ind, sig_date, prev_risk_on,
                                exclude_assets=None) -> tuple:
    """sig_date 기준 target weights + 갱신된 risk_on 상태 반환.
    hysteresis (prev_risk_on) 유지로 V17 정확 재현.
    exclude_assets: drop_top_contributor 용. offensive/defensive 후보에서 제외.
    """
    risk_on = resolve_canary(params, ind, sig_date, prev_risk_on)
    if risk_on is None:
        risk_on = False
    excl = frozenset(exclude_assets or ())

    if risk_on:
        candidates = [t for t in params.offensive if t not in excl]
        if not candidates:
            return {'Cash': 1.0}, risk_on
        healthy = filter_healthy(params, ind, sig_date, candidates)
        if not healthy:
            return {'Cash': 1.0}, risk_on
        weights = select_offensive(params, ind, sig_date, healthy)
        if not weights:
            return {'Cash': 1.0}, risk_on
        # excl 재검증 (select_offensive 가 다른 경로로 포함했을 경우 대비)
        weights = {k: v for k, v in weights.items() if k not in excl}
        if not weights:
            return {'Cash': 1.0}, risk_on
        return _normalize_weights(weights), risk_on
    else:
        # defensive 도 exclude → 후보에서 먼저 빼고 select_defensive 호출 (Codex r10 fix)
        # canary_assets 은 그대로 유지 (신호 보호)
        if excl:
            defensive_filtered = tuple(t for t in params.defensive if t not in excl)
            if not defensive_filtered:
                return {'Cash': 1.0}, risk_on
            params_def = _dc_replace(params, defensive=defensive_filtered)
        else:
            params_def = params
        weights = select_defensive(params_def, ind, sig_date)
        if not weights:
            return {'Cash': 1.0}, risk_on
        # 안전망: select 결과에서 excl 재필터
        weights = {k: v for k, v in weights.items() if k not in excl}
        if not weights:
            return {'Cash': 1.0}, risk_on
        return _normalize_weights(weights), risk_on


def _normalize_weights(w: dict) -> dict:
    s = sum(w.values())
    if s <= 0: return {'Cash': 1.0}
    return {k: v/s for k, v in w.items()}


def _ensemble_avg(snap_targets: list) -> dict:
    """3 snap 의 target 을 EW 평균해 combined target 반환."""
    n = len(snap_targets)
    combined: dict[str, float] = {}
    for snap in snap_targets:
        for t, w in snap.items():
            combined[t] = combined.get(t, 0.0) + w / n
    return _normalize_weights(combined)


def _rebal_to_target(holdings, cash, pv, target_w, ind, date, tx_cost):
    """현재 holdings/cash → target_w 로 일괄 리밸.
    매도 → 매수 순서. tx_cost 양방향 적용.
    반환: (new_holdings, new_cash)
    """
    if pv <= 0:
        return holdings, cash

    # 1) 매도: target_w 에 없는 or 축소되는 포지션
    target_notional = {k: pv * w for k, w in target_w.items() if k != 'Cash'}
    for t in list(holdings.keys()):
        p = get_price(ind, t, date)
        if np.isnan(p): continue
        cur_notional = holdings[t] * p
        tgt_not = target_notional.get(t, 0.0)
        if tgt_not < cur_notional - 1e-6:
            sell_notional = cur_notional - tgt_not
            sell_shares = min(holdings[t], sell_notional / p)
            holdings[t] -= sell_shares
            cash += sell_shares * p * (1 - tx_cost)
            if holdings[t] < 1e-9:
                holdings.pop(t, None)

    # 2) 매수: target_w 에서 확대되는 포지션 — 가용 cash 비율로 분배
    needs = []
    for t, tgt_not in target_notional.items():
        p = get_price(ind, t, date)
        if np.isnan(p) or p <= 0: continue
        cur_notional = holdings.get(t, 0.0) * p
        if tgt_not > cur_notional + 1e-6:
            needs.append((t, tgt_not - cur_notional, p))
    total_buy = sum(n for _, n, _ in needs)
    if total_buy > 0 and cash > 0:
        # tx 고려: cash * (1 - tx_cost) 가 실제 매수 가능액
        avail = cash / (1 + tx_cost)
        scale = min(1.0, avail / total_buy)
        for t, need_not, p in needs:
            buy_not = need_not * scale
            if buy_not < 1.0: continue
            buy_shares = buy_not / p
            cost = buy_shares * p * (1 + tx_cost)
            if cost > cash: break
            holdings[t] = holdings.get(t, 0.0) + buy_shares
            cash -= cost

    return holdings, cash


def _snap_anchor_day(snap_idx, stagger):
    """snap_idx 의 월 anchor day. stagger 10 이면 snap0=1, snap1=11, snap2=21."""
    return 1 + snap_idx * stagger


def _should_rebal_scheduled(days, snap_next_sched):
    """Schedule-based trigger. 실제 fire day 가 holiday 로 밀려도 다음 schedule 은 원래 시점 고정.
    주말/holiday 무관하게 rebal 횟수 phase offset 에 대해 일정."""
    return days >= snap_next_sched


def run_snapshot_ensemble(prices_dict, ind, params: SP,
                           snap_days: int = 30, n_snap: int = 3,
                           monthly_anchor_mode: bool = False,
                           phase_offset: int = 0,
                           execution_delay_bars: int = 0,
                           exclude_assets=None,
                           _trace: list | None = None) -> pd.DataFrame | None:
    """V21 스타일 snapshot 단일 계좌 백테스트 (v4 — event-based rebal).

    Rebal trigger:
      1) snap target 업데이트 (snap 일정 도달)
      2) canary flip (combined risk_on 이 이전 대비 바뀐 경우)
    Drift 기반 rebal 없음 (사용자 지정).

    - calendar mode (기본): days_since_start 기반. snap i 는 (days - i*stagger) % snap_days == 0.
    - 각 snap 독립 prev_risk_on state 유지 (hysteresis).
    - Rebal 시 delta 체결 (tx 양방향).
    """
    stagger = max(1, snap_days // n_snap)
    spy = ind.get('SPY')
    if spy is None: return None
    dates = spy.index[(spy.index >= params.start) & (spy.index <= params.end)]
    if len(dates) < 2: return None

    excl = frozenset(exclude_assets or ())
    pending_rebal = None  # execution_delay_bars 용 {'due_i': int, 'target': dict}
    snap_targets = [{'Cash': 1.0}] * n_snap
    snap_prev_risk = [None] * n_snap
    snap_rebaled_this_month = [False] * n_snap  # monthly_anchor_mode 용
    # calendar mode: next scheduled calendar-day (days_since_start) per snap.
    # holiday 로 actual fire 가 밀려도 다음 schedule 은 원래 slot 유지 (drift 없음).
    snap_next_sched = [phase_offset + i * stagger for i in range(n_snap)]

    holdings: dict[str, float] = {}
    cash = params.capital
    history = []
    rebal_count = 0
    prev_trading_date = None
    prev_month = None
    start_date = dates[0]

    for bar_i, date in enumerate(dates):
        cur_month = date.strftime('%Y-%m')
        is_month_change = (prev_month is not None and cur_month != prev_month)
        sig_date = prev_trading_date if prev_trading_date is not None else date

        # monthly_anchor_mode: 월 바뀌면 모든 snap 의 이번월 rebal 플래그 reset
        if monthly_anchor_mode and is_month_change:
            snap_rebaled_this_month = [False] * n_snap

        # Portfolio value
        pv = cash
        for t, shares in holdings.items():
            p = get_price(ind, t, date)
            if not np.isnan(p):
                pv += shares * p

        # Crash breaker: 본 엔진에서 제거 (사용자 결정 — 실거래 일치 단순화).
        # 방어는 canary + select_defensive 만으로 처리.

        # Canary flip 체크: 현재 sig_date 기준 risk_on 을 각 snap 의 prev 와 비교
        # flip 발생한 snap 은 즉시 target 재계산 + rebal trigger
        canary_flip_triggered = False
        for i in range(n_snap):
            if snap_prev_risk[i] is None: continue
            cur_risk = resolve_canary(params, ind, sig_date, snap_prev_risk[i])
            if cur_risk is not None and cur_risk != snap_prev_risk[i]:
                # flip
                new_target, new_risk = compute_target_with_canary(
                    params, ind, sig_date, snap_prev_risk[i], exclude_assets=excl)
                snap_targets[i] = new_target
                snap_prev_risk[i] = new_risk
                canary_flip_triggered = True

        # Snap target 정규 갱신 (스냅샷 주기 도달)
        any_snap_updated = canary_flip_triggered
        days_since_start = (date - start_date).days
        for i in range(n_snap):
            if monthly_anchor_mode:
                anchor = _snap_anchor_day(i, stagger)
                should_rebal = (not snap_rebaled_this_month[i]) and date.day >= anchor
            else:
                should_rebal = _should_rebal_scheduled(
                    days_since_start, snap_next_sched[i])

            if should_rebal:
                new_target, new_risk = compute_target_with_canary(
                    params, ind, sig_date, snap_prev_risk[i], exclude_assets=excl)
                snap_targets[i] = new_target
                snap_prev_risk[i] = new_risk
                any_snap_updated = True
                if monthly_anchor_mode:
                    snap_rebaled_this_month[i] = True
                else:
                    # Advance schedule by exactly snap_days (fire date 와 독립).
                    # 여러 holiday 연속으로 한 슬롯 이상 밀렸으면 따라잡을 때까지 전진.
                    while snap_next_sched[i] <= days_since_start:
                        snap_next_sched[i] += snap_days

        # Snap 이 하나라도 갱신 → combined target 체결 (즉시 or delay)
        do_execute = None  # 실제 체결할 target dict
        if any_snap_updated:
            combined = _ensemble_avg(snap_targets)
            if execution_delay_bars <= 0:
                do_execute = combined
            else:
                pending_rebal = {'due_i': bar_i + int(execution_delay_bars),
                                 'target': combined}

        # pending_rebal 만료 체크 (delay 체결)
        if pending_rebal is not None and bar_i >= pending_rebal['due_i']:
            do_execute = pending_rebal['target']
            pending_rebal = None

        if do_execute is not None:
            pv = cash
            for t, shares in holdings.items():
                p = get_price(ind, t, date)
                if not np.isnan(p):
                    pv += shares * p
            holdings, cash = _rebal_to_target(holdings, cash, pv, do_execute,
                                                ind, date, params.tx_cost)
            rebal_count += 1
            if _trace is not None:
                _trace.append({'Date': date, 'target': dict(do_execute),
                               'rebal': True})
            pv = cash
            for t, shares in holdings.items():
                p = get_price(ind, t, date)
                if not np.isnan(p):
                    pv += shares * p

        history.append({'Date': date, 'Value': pv})
        prev_trading_date = date
        prev_month = cur_month

    if not history:
        return None
    df = pd.DataFrame(history).set_index('Date')
    df.attrs['rebal_count'] = rebal_count
    df.attrs['flip_count'] = 0
    return df


def run_snapshot(params, snap_days, n_snap=3, monthly_anchor_mode=False):
    """Convenience wrapper. Assumes _g_prices/_g_ind set via stock_engine._init."""
    import stock_engine as tsi
    return run_snapshot_ensemble(tsi._g_prices, tsi._g_ind, params,
                                  snap_days=snap_days, n_snap=n_snap,
                                  monthly_anchor_mode=monthly_anchor_mode)
