#!/usr/bin/env python3
"""Test turnover filter on snapshot averaging (Method B).
Turnover filter: skip regular rebalancing if half-turnover < threshold.
Force-execute: canary flip, PFD5, first day.
All tests use 10-date averaging."""

import os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(__file__))
from strategy_engine import (
    Params, load_data,
    get_universe_for_date, resolve_canary, get_healthy_coins,
    select_coins, compute_weights, apply_risk,
    execute_rebalance, _port_val, get_price,
    calc_metrics, calc_yearly_metrics
)

ANCHOR_DAYS = [1, 4, 7, 10, 13, 16, 19, 22, 25, 28]


def B(**kw):
    base = dict(
        canary='K8', vote_smas=(60,), vote_moms=(), vote_threshold=1,
        health='HK', health_sma=2, health_mom_short=21,
        health_mom_long=90, vol_cap=0.05,
    )
    base.update(kw)
    return Params(**base)


def compute_signal_weights(prices, universe_map, date, params, state):
    canary_on = state.get('canary_on', False)
    if not canary_on:
        return {'CASH': 1.0}
    universe = get_universe_for_date(universe_map, date)
    healthy = get_healthy_coins(prices, universe, date, params, state)
    state['healthy_count'] = len(healthy)
    state['current_healthy_set'] = set(healthy)
    if not healthy:
        return {'CASH': 1.0}
    picks = select_coins(healthy, prices, date, params, state)
    if not picks:
        return {'CASH': 1.0}
    weights = compute_weights(picks, prices, date, params, state)
    return weights


def merge_snapshots(snapshots):
    if not snapshots:
        return {'CASH': 1.0}
    combined = {}
    n = len(snapshots)
    for snap in snapshots:
        for t, w in snap.items():
            combined[t] = combined.get(t, 0) + w / n
    total = sum(combined.values())
    if total > 0:
        return {t: w / total for t, w in combined.items()}
    return {'CASH': 1.0}


def calc_current_weights(holdings, cash, prices, date):
    """Calculate current portfolio weights from holdings."""
    pv = _port_val(holdings, cash, prices, date)
    if pv <= 0:
        return {'CASH': 1.0}
    weights = {}
    for t, units in holdings.items():
        p = get_price(t, prices, date)
        val = units * p
        if val > 0:
            weights[t] = val / pv
    cash_w = cash / pv
    if cash_w > 0.001:
        weights['CASH'] = cash_w
    return weights


def calc_half_turnover(current_w, target_w):
    """Calculate half-turnover: sum(|target - current|) / 2."""
    all_keys = set(current_w.keys()) | set(target_w.keys())
    l1 = sum(abs(target_w.get(k, 0) - current_w.get(k, 0)) for k in all_keys)
    return l1 / 2


def run_snapshot_with_turnover(prices, universe_map, snapshot_days,
                                turnover_threshold=0.0,
                                mode='gate',
                                post_flip_delay=5, params_base=None):
    """Run snapshot backtest with turnover filter.

    Args:
        turnover_threshold: half-turnover threshold (0.0 = disabled)
        mode: 'gate' = skip scheduled rebal if turnover < threshold
              'drift' = rebal any day if price drift causes turnover > threshold
        post_flip_delay: days after OFF→ON flip for refresh
    """
    if params_base is None:
        params_base = B()

    btc = prices.get('BTC-USD')
    if btc is None:
        return _empty()

    all_dates = btc.index[(btc.index >= params_base.start_date) &
                          (btc.index <= params_base.end_date)]
    if len(all_dates) == 0:
        return _empty()

    holdings = {}
    cash = params_base.initial_capital
    state = {
        'prev_canary': False, 'canary_off_days': 0,
        'health_fail_streak': {}, 'prev_picks': [],
        'scaled_months': 2, 'month_start_value': params_base.initial_capital,
        'high_watermark': params_base.initial_capital,
        'crash_cooldown': 0, 'coin_cooldowns': {},
        'recent_port_vals': [], 'prev_month': None,
        'catastrophic_triggered': False, 'risk_force_rebal': False,
        'canary_on_date': None, 'post_flip_refreshed': False,
    }

    n_snap = len(snapshot_days)
    snapshots = [{'CASH': 1.0} for _ in range(n_snap)]
    snap_done = {}

    portfolio_values = []
    rebal_count = 0
    skip_count = 0
    drift_rebal_count = 0  # extra rebals triggered by drift

    for i, date in enumerate(all_dates):
        cur_month = date.strftime('%Y-%m')
        imc = (state['prev_month'] is not None and cur_month != state['prev_month'])

        pv = _port_val(holdings, cash, prices, date)
        state['current_port_val'] = pv
        state['high_watermark'] = max(state['high_watermark'], pv)
        state['recent_port_vals'].append(pv)
        if len(state['recent_port_vals']) > 60:
            state['recent_port_vals'] = state['recent_port_vals'][-60:]

        if imc:
            state['month_start_value'] = pv
            state['catastrophic_triggered'] = False

        canary_on = resolve_canary(prices, date, params_base, state)
        canary_flipped = (canary_on != state['prev_canary'])

        if canary_on and canary_flipped:
            state['canary_on_date'] = date
            state['post_flip_refreshed'] = False
        elif not canary_on and canary_flipped:
            state['canary_on_date'] = None

        state['canary_on'] = canary_on
        state['is_month_change'] = imc

        need_rebal = False
        force_rebal = False

        # First day
        if i == 0:
            for si in range(n_snap):
                snapshots[si] = compute_signal_weights(prices, universe_map, date, params_base, state)
            need_rebal = True
            force_rebal = True

        # Canary flip
        elif canary_flipped:
            if canary_on:
                for si in range(n_snap):
                    snapshots[si] = compute_signal_weights(prices, universe_map, date, params_base, state)
            else:
                for si in range(n_snap):
                    snapshots[si] = {'CASH': 1.0}
            need_rebal = True
            force_rebal = True

        # PFD refresh
        elif post_flip_delay > 0 and canary_on:
            flip_date = state.get('canary_on_date')
            if flip_date and not state.get('post_flip_refreshed', False):
                days_since = (date - flip_date).days
                if days_since >= post_flip_delay:
                    state['post_flip_refreshed'] = True
                    for si in range(n_snap):
                        snapshots[si] = compute_signal_weights(prices, universe_map, date, params_base, state)
                    need_rebal = True
                    force_rebal = True

        # Regular monthly snapshots
        if canary_on and not canary_flipped:
            for si, anchor in enumerate(snapshot_days):
                key = f"{cur_month}_snap{si}"
                if date.day >= anchor and key not in snap_done:
                    snap_done[key] = True
                    new_w = compute_signal_weights(prices, universe_map, date, params_base, state)
                    if new_w != snapshots[si]:
                        snapshots[si] = new_w
                        need_rebal = True

        combined = merge_snapshots(snapshots)

        # ── Mode: Gate ──
        # Skip scheduled rebal if turnover too small
        if mode == 'gate':
            if need_rebal and not force_rebal and turnover_threshold > 0:
                current_w = calc_current_weights(holdings, cash, prices, date)
                ht = calc_half_turnover(current_w, combined)
                if ht < turnover_threshold:
                    need_rebal = False
                    skip_count += 1

        # ── Mode: Drift ──
        # On scheduled days: always rebal (no gate).
        # On non-scheduled days: check if price drift exceeds threshold.
        elif mode == 'drift':
            if not need_rebal and canary_on and turnover_threshold > 0:
                current_w = calc_current_weights(holdings, cash, prices, date)
                ht = calc_half_turnover(current_w, combined)
                if ht >= turnover_threshold:
                    need_rebal = True
                    drift_rebal_count += 1

        # ── Mode: Combined ──
        # Gate on scheduled days (skip if small) + Drift on non-scheduled days (add if big).
        # Uses gate_threshold for gate, turnover_threshold for drift.
        elif mode == 'combined':
            gate_th = state.get('_gate_threshold', turnover_threshold)
            drift_th = turnover_threshold
            if need_rebal and not force_rebal and gate_th > 0:
                current_w = calc_current_weights(holdings, cash, prices, date)
                ht = calc_half_turnover(current_w, combined)
                if ht < gate_th:
                    need_rebal = False
                    skip_count += 1
            if not need_rebal and canary_on and drift_th > 0:
                current_w = calc_current_weights(holdings, cash, prices, date)
                ht = calc_half_turnover(current_w, combined)
                if ht >= drift_th:
                    need_rebal = True
                    drift_rebal_count += 1

        if need_rebal:
            holdings, cash = execute_rebalance(holdings, cash, combined, prices,
                                               date, params_base.tx_cost)
            rebal_count += 1

        pv = _port_val(holdings, cash, prices, date)
        portfolio_values.append({'Date': date, 'Value': pv})
        state['prev_canary'] = canary_on
        state['prev_month'] = cur_month

    if not portfolio_values:
        return _empty()

    pvdf = pd.DataFrame(portfolio_values).set_index('Date')
    m = calc_metrics(pvdf)
    ym = calc_yearly_metrics(pvdf)

    return {
        'metrics': m,
        'yearly': ym,
        'rebal_count': rebal_count,
        'skip_count': skip_count,
        'drift_rebal_count': drift_rebal_count,
        'pv': pvdf,
    }


def _empty():
    return {'metrics': {'CAGR': 0, 'MDD': 0, 'Sharpe': 0, 'Sortino': 0, 'Final': 0},
            'yearly': {}, 'rebal_count': 0, 'skip_count': 0, 'pv': pd.DataFrame()}


def main():
    print("Loading data...")
    prices, universe = load_data()
    print(f"  {len(prices)} tickers loaded")

    t0 = time.time()

    # Turnover thresholds to test (half-turnover %)
    thresholds = [0, 0.02, 0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

    # Collect results: {threshold: [metrics_per_anchor]}
    all_results = {th: [] for th in thresholds}
    all_rebals = {th: [] for th in thresholds}
    all_skips = {th: [] for th in thresholds}

    for di, base_d in enumerate(ANCHOR_DAYS):
        snap_days = [(base_d - 1 + i * 9) % 28 + 1 for i in range(3)]
        print(f"  Anchor {base_d:>2}: ", end="", flush=True)

        for th in thresholds:
            r = run_snapshot_with_turnover(
                prices, universe, snap_days,
                turnover_threshold=th,
                post_flip_delay=5, params_base=B()
            )
            all_results[th].append(r['metrics'])
            all_rebals[th].append(r['rebal_count'])
            all_skips[th].append(r.get('skip_count', 0))

        sharpes = [all_results[th][-1]['Sharpe'] for th in thresholds]
        print(f"T0={sharpes[0]:.3f} T5={sharpes[2]:.3f} T10={sharpes[4]:.3f} T20={sharpes[6]:.3f}")

    print(f"\n  Completed in {time.time()-t0:.1f}s")

    # Compute averages
    def avg_metrics(results):
        avg = {}
        for key in results[0]:
            avg[key] = np.mean([r[key] for r in results])
        std_s = np.std([r['Sharpe'] for r in results])
        return avg, std_s

    print(f"\n{'=' * 100}")
    print(f"  턴오버 필터 테스트 — 스냅샷 평균(B) + PFD5, 날짜 평균 (10 앵커)")
    print(f"{'=' * 100}")

    base_sharpe = avg_metrics(all_results[0])[0]['Sharpe']

    print(f"\n  {'임계값':>8} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7}"
          f" {'Calmar':>7} {'리밸':>5} {'스킵':>5}")
    print(f"  {'─' * 75}")

    for th in thresholds:
        avg, std = avg_metrics(all_results[th])
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        delta = avg['Sharpe'] - base_sharpe
        avg_rebal = np.mean(all_rebals[th])
        avg_skip = np.mean(all_skips[th])
        label = f"{th*100:.0f}%" if th > 0 else "없음"
        marker = ' ★' if delta > 0.02 else ' ↑' if delta > 0 else ''
        print(f"  {label:>8} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
              f" {avg_rebal:>5.0f} {avg_skip:>5.0f}{marker}")

    # Per-anchor detail for key thresholds
    print(f"\n  앵커별 Sharpe (주요 임계값):")
    key_ths = [0, 0.05, 0.10, 0.15, 0.20]
    header = f"  {'앵커':>5}"
    for th in key_ths:
        label = f"T{th*100:.0f}%"
        header += f" {label:>7}"
    print(header)
    print(f"  {'─' * 45}")
    for i, d in enumerate(ANCHOR_DAYS):
        row = f"  {d:>5}"
        for th in key_ths:
            row += f" {all_results[th][i]['Sharpe']:>7.3f}"
        print(row)

    # ═══════════════════════════════════════════════════════════════════
    # Part 2: Drift mode — daily price drift monitoring
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  드리프트 감시 모드 — 매일 비중 이탈 확인, 임계값 초과 시 추가 매매")
    print(f"{'=' * 100}")

    drift_thresholds = [0.05, 0.08, 0.10, 0.15, 0.20, 0.30]

    drift_results = {th: [] for th in drift_thresholds}
    drift_rebals = {th: [] for th in drift_thresholds}
    drift_extra = {th: [] for th in drift_thresholds}

    for di, base_d in enumerate(ANCHOR_DAYS):
        snap_days = [(base_d - 1 + i * 9) % 28 + 1 for i in range(3)]
        print(f"  Anchor {base_d:>2}: ", end="", flush=True)

        for th in drift_thresholds:
            r = run_snapshot_with_turnover(
                prices, universe, snap_days,
                turnover_threshold=th,
                mode='drift',
                post_flip_delay=5, params_base=B()
            )
            drift_results[th].append(r['metrics'])
            drift_rebals[th].append(r['rebal_count'])
            drift_extra[th].append(r.get('drift_rebal_count', 0))

        sharpes = [drift_results[th][-1]['Sharpe'] for th in drift_thresholds]
        print(f"D5={sharpes[0]:.3f} D10={sharpes[2]:.3f} D20={sharpes[4]:.3f}")

    print(f"\n  {'임계값':>8} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7}"
          f" {'Calmar':>7} {'리밸':>5} {'드리프트':>7}")
    print(f"  {'─' * 80}")

    # Show baseline (no drift, T0) first
    print(f"  {'기준':>8} {base_sharpe:>7.3f} {'+0.000':>7} {avg_metrics(all_results[0])[1]:>6.3f}"
          f" {avg_metrics(all_results[0])[0]['CAGR']:>+7.1%}"
          f" {avg_metrics(all_results[0])[0]['MDD']:>6.1%}"
          f" {avg_metrics(all_results[0])[0]['CAGR']/abs(avg_metrics(all_results[0])[0]['MDD']):>7.2f}"
          f" {np.mean(all_rebals[0]):>5.0f} {'0':>7}")

    for th in drift_thresholds:
        avg, std = avg_metrics(drift_results[th])
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        delta = avg['Sharpe'] - base_sharpe
        avg_rebal = np.mean(drift_rebals[th])
        avg_drift = np.mean(drift_extra[th])
        label = f"D{th*100:.0f}%"
        marker = ' ★' if delta > 0.02 else ' ↑' if delta > 0 else ''
        print(f"  {label:>8} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
              f" {avg_rebal:>5.0f} {avg_drift:>7.0f}{marker}")

    # ═══════════════════════════════════════════════════════════════════
    # Part 3: Drift mode — PFD5 on vs off
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  드리프트 + PFD5 on/off 비교 — PFD5가 드리프트 감시와 중복되는지 확인")
    print(f"{'=' * 100}")

    drift_pfd_thresholds = [0.05, 0.08, 0.10, 0.15, 0.20]
    pfd_variants = [
        ('PFD5', 5),
        ('PFD0', 0),
    ]

    pfd_results = {}  # (pfd_label, th) -> [metrics]
    pfd_rebals = {}
    pfd_drift_counts = {}

    for pfd_label, pfd_val in pfd_variants:
        for th in drift_pfd_thresholds:
            key = (pfd_label, th)
            pfd_results[key] = []
            pfd_rebals[key] = []
            pfd_drift_counts[key] = []

    for base_d in ANCHOR_DAYS:
        snap_days = [(base_d - 1 + i * 9) % 28 + 1 for i in range(3)]
        print(f"  Anchor {base_d:>2}: ", end="", flush=True)

        for pfd_label, pfd_val in pfd_variants:
            for th in drift_pfd_thresholds:
                r = run_snapshot_with_turnover(
                    prices, universe, snap_days,
                    turnover_threshold=th,
                    mode='drift',
                    post_flip_delay=pfd_val, params_base=B()
                )
                key = (pfd_label, th)
                pfd_results[key].append(r['metrics'])
                pfd_rebals[key].append(r['rebal_count'])
                pfd_drift_counts[key].append(r.get('drift_rebal_count', 0))

        # Show one sample
        s5 = pfd_results[('PFD5', 0.10)][-1]['Sharpe']
        s0 = pfd_results[('PFD0', 0.10)][-1]['Sharpe']
        print(f"D10%+PFD5={s5:.3f}  D10%+PFD0={s0:.3f}")

    print(f"\n  {'전략':>16} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7}"
          f" {'Calmar':>7} {'리밸':>5} {'드리프트':>7}")
    print(f"  {'─' * 85}")

    for pfd_label, pfd_val in pfd_variants:
        for th in drift_pfd_thresholds:
            key = (pfd_label, th)
            avg, std = avg_metrics(pfd_results[key])
            calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
            delta = avg['Sharpe'] - base_sharpe
            avg_rebal = np.mean(pfd_rebals[key])
            avg_drift = np.mean(pfd_drift_counts[key])
            label = f"D{th*100:.0f}%+{pfd_label}"
            marker = ' ★' if delta > 0.02 else ' ↑' if delta > 0 else ''
            print(f"  {label:>16} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
                  f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
                  f" {avg_rebal:>5.0f} {avg_drift:>7.0f}{marker}")
        print()  # separator between PFD variants

    # ═══════════════════════════════════════════════════════════════════
    # Part 4: Combined — Gate + Drift 동시 적용
    # ═══════════════════════════════════════════════════════════════════
    print(f"\n{'=' * 100}")
    print(f"  Gate + Drift 동시 적용 — 정기일 소액 스킵 + 비정기일 큰 드리프트 매매")
    print(f"{'=' * 100}")

    # Test combos: (gate_threshold, drift_threshold)
    combos = [
        (0.05, 0.10),
        (0.05, 0.15),
        (0.05, 0.20),
        (0.08, 0.10),
        (0.08, 0.15),
        (0.08, 0.20),
        (0.10, 0.15),
        (0.10, 0.20),
    ]

    combo_results = {c: [] for c in combos}
    combo_rebals = {c: [] for c in combos}
    combo_skips = {c: [] for c in combos}
    combo_drifts = {c: [] for c in combos}

    for base_d in ANCHOR_DAYS:
        snap_days = [(base_d - 1 + i * 9) % 28 + 1 for i in range(3)]
        print(f"  Anchor {base_d:>2}: ", end="", flush=True)

        for gate_th, drift_th in combos:
            # Pass gate_threshold via state hack
            p = B()
            r = run_snapshot_with_turnover(
                prices, universe, snap_days,
                turnover_threshold=drift_th,
                mode='combined',
                post_flip_delay=5, params_base=p
            )
            # Re-run with correct gate threshold by injecting into state
            # Actually need to pass gate_th properly — use a wrapper
            r = _run_combined(prices, universe, snap_days, gate_th, drift_th)
            key = (gate_th, drift_th)
            combo_results[key].append(r['metrics'])
            combo_rebals[key].append(r['rebal_count'])
            combo_skips[key].append(r.get('skip_count', 0))
            combo_drifts[key].append(r.get('drift_rebal_count', 0))

        s1 = combo_results[(0.05, 0.10)][-1]['Sharpe']
        s2 = combo_results[(0.08, 0.15)][-1]['Sharpe']
        print(f"G5+D10={s1:.3f}  G8+D15={s2:.3f}")

    print(f"\n  {'전략':>16} {'Sharpe':>7} {'Δ':>7} {'σ(S)':>6} {'CAGR':>8} {'MDD':>7}"
          f" {'Calmar':>7} {'리밸':>5} {'스킵':>5} {'드리프트':>7}")
    print(f"  {'─' * 95}")

    for gate_th, drift_th in combos:
        key = (gate_th, drift_th)
        avg, std = avg_metrics(combo_results[key])
        calmar = avg['CAGR'] / abs(avg['MDD']) if avg['MDD'] != 0 else 0
        delta = avg['Sharpe'] - base_sharpe
        avg_rebal = np.mean(combo_rebals[key])
        avg_skip = np.mean(combo_skips[key])
        avg_drift = np.mean(combo_drifts[key])
        label = f"G{gate_th*100:.0f}%+D{drift_th*100:.0f}%"
        marker = ' ★' if delta > 0.02 else ' ↑' if delta > 0 else ''
        print(f"  {label:>16} {avg['Sharpe']:>7.3f} {delta:>+6.3f} {std:>6.3f}"
              f" {avg['CAGR']:>+7.1%} {avg['MDD']:>6.1%} {calmar:>7.2f}"
              f" {avg_rebal:>5.0f} {avg_skip:>5.0f} {avg_drift:>7.0f}{marker}")

    print(f"\n  Total: {time.time()-t0:.1f}s")


def _run_combined(prices, universe, snap_days, gate_th, drift_th, pfd=5):
    """Helper: run combined mode with separate gate/drift thresholds."""
    params = B()
    btc = prices.get('BTC-USD')
    if btc is None:
        return _empty()

    all_dates = btc.index[(btc.index >= params.start_date) &
                          (btc.index <= params.end_date)]
    if len(all_dates) == 0:
        return _empty()

    holdings = {}
    cash = params.initial_capital
    state = {
        'prev_canary': False, 'canary_off_days': 0,
        'health_fail_streak': {}, 'prev_picks': [],
        'scaled_months': 2, 'month_start_value': params.initial_capital,
        'high_watermark': params.initial_capital,
        'crash_cooldown': 0, 'coin_cooldowns': {},
        'recent_port_vals': [], 'prev_month': None,
        'catastrophic_triggered': False, 'risk_force_rebal': False,
        'canary_on_date': None, 'post_flip_refreshed': False,
    }

    n_snap = len(snap_days)
    snapshots = [{'CASH': 1.0} for _ in range(n_snap)]
    snap_done = {}
    portfolio_values = []
    rebal_count = 0
    skip_count = 0
    drift_rebal_count = 0

    for i, date in enumerate(all_dates):
        cur_month = date.strftime('%Y-%m')
        imc = (state['prev_month'] is not None and cur_month != state['prev_month'])

        pv = _port_val(holdings, cash, prices, date)
        state['current_port_val'] = pv
        state['high_watermark'] = max(state['high_watermark'], pv)
        state['recent_port_vals'].append(pv)
        if len(state['recent_port_vals']) > 60:
            state['recent_port_vals'] = state['recent_port_vals'][-60:]

        if imc:
            state['month_start_value'] = pv
            state['catastrophic_triggered'] = False

        canary_on = resolve_canary(prices, date, params, state)
        canary_flipped = (canary_on != state['prev_canary'])

        if canary_on and canary_flipped:
            state['canary_on_date'] = date
            state['post_flip_refreshed'] = False
        elif not canary_on and canary_flipped:
            state['canary_on_date'] = None

        state['canary_on'] = canary_on
        state['is_month_change'] = imc

        need_rebal = False
        force_rebal = False

        if i == 0:
            for si in range(n_snap):
                snapshots[si] = compute_signal_weights(prices, universe, date, params, state)
            need_rebal = True
            force_rebal = True
        elif canary_flipped:
            if canary_on:
                for si in range(n_snap):
                    snapshots[si] = compute_signal_weights(prices, universe, date, params, state)
            else:
                for si in range(n_snap):
                    snapshots[si] = {'CASH': 1.0}
            need_rebal = True
            force_rebal = True
        elif pfd > 0 and canary_on:
            flip_date = state.get('canary_on_date')
            if flip_date and not state.get('post_flip_refreshed', False):
                days_since = (date - flip_date).days
                if days_since >= pfd:
                    state['post_flip_refreshed'] = True
                    for si in range(n_snap):
                        snapshots[si] = compute_signal_weights(prices, universe, date, params, state)
                    need_rebal = True
                    force_rebal = True

        if canary_on and not canary_flipped:
            for si, anchor in enumerate(snap_days):
                key = f"{cur_month}_snap{si}"
                if date.day >= anchor and key not in snap_done:
                    snap_done[key] = True
                    new_w = compute_signal_weights(prices, universe, date, params, state)
                    if new_w != snapshots[si]:
                        snapshots[si] = new_w
                        need_rebal = True

        combined = merge_snapshots(snapshots)

        # Gate: skip small scheduled rebalances
        if need_rebal and not force_rebal and gate_th > 0:
            current_w = calc_current_weights(holdings, cash, prices, date)
            ht = calc_half_turnover(current_w, combined)
            if ht < gate_th:
                need_rebal = False
                skip_count += 1

        # Drift: add rebal on non-scheduled days if big drift
        if not need_rebal and canary_on and drift_th > 0:
            current_w = calc_current_weights(holdings, cash, prices, date)
            ht = calc_half_turnover(current_w, combined)
            if ht >= drift_th:
                need_rebal = True
                drift_rebal_count += 1

        if need_rebal:
            holdings, cash = execute_rebalance(holdings, cash, combined, prices,
                                               date, params.tx_cost)
            rebal_count += 1

        pv = _port_val(holdings, cash, prices, date)
        portfolio_values.append({'Date': date, 'Value': pv})
        state['prev_canary'] = canary_on
        state['prev_month'] = cur_month

    if not portfolio_values:
        return _empty()

    pvdf = pd.DataFrame(portfolio_values).set_index('Date')
    m = calc_metrics(pvdf)
    ym = calc_yearly_metrics(pvdf)

    return {
        'metrics': m, 'yearly': ym,
        'rebal_count': rebal_count,
        'skip_count': skip_count,
        'drift_rebal_count': drift_rebal_count,
        'pv': pvdf,
    }


if __name__ == '__main__':
    main()
