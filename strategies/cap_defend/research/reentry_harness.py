#!/usr/bin/env python3
"""빈 스냅 트랜치 동적 재진입 — 백테스트 전용 하니스 (F0 + 21 변형).

절대 규칙:
- 실매매/라이브 코드 및 프로덕션 unified_backtest.py 를 수정하지 않는다.
- 이 하니스는 unified_backtest.run() 소스를 런타임에 슬라이스해 _run_base 로
  재구성하고, 재진입 훅(reentry_cfg is not None 일 때만 동작)을 주입한다.
  → reentry_cfg=None (F0) 은 엔진 run() 과 소스 동일 = 결과 동일 (검증됨).
- look-ahead 금지: 시그널 t-1 종가(prev_date), 체결 t(date). 재진입도 동일 sig_date.

재진입 상태머신 (트랜치별):
- empty_age[si]: 연속 empty 봉 수 (target risky <= 1e-6). non-empty 시 0.
- reentered[si]: 자기 앵커 이후 off-anchor 재진입 여부 (freeze/A7 경계).
- sizing_state[si]: P1 half-step 추적 (0=none,1=half,2=full).
- reentry_cooldown: L1 글로벌 쿨다운.
- K 제한 우선순위: empty_age desc → phase offset (성과점수 금지).

변형은 reentry_cfg dict 로 파라미터화. VARIANTS 참조.
"""
import os, sys, re
import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
_CD = os.path.join(_HERE, '..')
sys.path.insert(0, _CD)
import unified_backtest as ub

# ═══════════════════════════════════════════════════════════════════
# 1) 엔진 run() 소스 슬라이스 + 재진입 훅 주입 → _run_base 생성
# ═══════════════════════════════════════════════════════════════════
_ENGINE_SRC = os.path.join(_CD, 'unified_backtest.py')
with open(_ENGINE_SRC) as _f:
    _lines = _f.readlines()
# def run( ... ) 부터 다음 top-level def/블록 전까지 추출
_start = next(i for i, l in enumerate(_lines) if l.startswith('def run('))
_end = next(i for i in range(_start + 1, len(_lines))
            if _lines[i].startswith('# ═══ 메인 ═══'))
_run_src = ''.join(_lines[_start:_end])

# --- 변형 1: 함수명 + 시그니처 확장 ---
_run_src = _run_src.replace('def run(', 'def _run_base(', 1)
_run_src = _run_src.replace(
    '        _trace=None):',
    '        _trace=None, reentry_cfg=None, _restats=None):', 1)

# --- 변형 2: 메인 루프 직전에 재진입 state + 헬퍼 주입 ---
_INIT_BLOCK = r'''
    # ══ [REENTRY] 상태머신 초기화 (reentry_cfg 없으면 완전 무효) ══
    _re_empty_age = [0] * n_snapshots
    _re_reentered = [False] * n_snapshots
    _re_sizing = [0] * n_snapshots  # P1: 0 none, 1 half, 2 full
    _re_cooldown = 0                # L1 글로벌
    _re_hc_hist = []                # bar별 healthy_count (default health)
    _re_can_hist = []               # bar별 canary_on
    _re_empty_run = [0] * n_snapshots  # 진행중 empty episode 길이 (durations 집계용)

    def _re_offset(si):
        return int(si * snap_interval_bars / n_snapshots)

    def _re_risky(sn):
        return sum(v for k, v in sn.items() if str(k).upper() != 'CASH')

    def _re_healthy_list(sig_date, hmode, vthr, ms_bars):
        """변형 health 로 healthy 코인 리스트 (greedy/cap 전, gate/count 용)."""
        mcap_order = get_mcap(sig_date)
        out = []
        _ms = ms_bars if ms_bars > 0 else mom30
        min_bars = max(_ms, mom90, mom60, _vol_bars, sma_period, health_sma_period)
        _excluded = exclude_assets or frozenset()
        for coin in mcap_order:
            if coin in blacklist or coin in _excluded:
                continue
            df = bars.get(coin)
            if df is None:
                continue
            ci = df.index.get_indexer([sig_date], method='ffill')[0]
            if ci < 0 or ci < min_bars:
                continue
            c = df['Close'].values[:ci + 1]
            if hmode == 'none':
                out.append(coin); continue
            m_short = calc_mom(c, _ms) if 'mom' in hmode else 999
            m_long = calc_mom(c, mom90) if ('mom2' in hmode or '3mom' in hmode) else 999
            if 'vol' in hmode:
                if vol_mode == 'bar':
                    vol = calc_vol_bars(c, _vol_bars, bars_per_year)
                else:
                    vol = calc_vol_daily(c, bpd, lookback_bars=_vol_bars)
            else:
                vol = 0
            if hmode == 'mom2vol':
                ok = m_short > 0 and m_long > 0 and vol <= vthr
            elif hmode == 'mom1vol':
                ok = m_short > 0 and vol <= vthr
            elif hmode == 'mlvol':      # H1: ml>0 & vol만 (ms 제거)
                ok = m_long > 0 and vol <= vthr
            elif hmode == 'mom2':
                ok = m_short > 0 and m_long > 0
            elif hmode == 'vol':
                ok = vol <= vthr
            else:
                ok = m_short > 0 and m_long > 0 and vol <= vthr
            if ok:
                out.append(coin)
        return out

    def _re_target_v(sig_date, hmode, vthr, ms_bars):
        """H 변형용 재진입 target (health 변형 + top3 cap + greedy). 기본은
        엔진 _compute_weights 를 그대로 쓰므로 이 함수는 H 계열에서만 호출."""
        healthy = _re_healthy_list(sig_date, hmode, vthr, ms_bars)
        picks = healthy[:universe_size]
        if selection == 'greedy' and len(picks) > 1:
            for i in range(len(picks) - 1, 0, -1):
                df_a = bars.get(picks[i - 1]); df_b = bars.get(picks[i])
                if df_a is None or df_b is None:
                    continue
                ci_a = df_a.index.get_indexer([sig_date], method='ffill')[0]
                ci_b = df_b.index.get_indexer([sig_date], method='ffill')[0]
                ma = calc_mom(df_a['Close'].values[:ci_a + 1], mom30)
                mb = calc_mom(df_b['Close'].values[:ci_b + 1], mom30)
                if ma >= mb:
                    picks.pop(i)
        if not picks:
            return {'CASH': 1.0}
        w = min(1.0 / len(picks), cap)
        weights = {coin: w for coin in picks}
        total = sum(weights.values())
        if total < 0.999:
            weights['CASH'] = 1.0 - total
        return weights

    def _re_apply_sizing(tgt, mode, si, hcount):
        risky = {k: v for k, v in tgt.items() if str(k).upper() != 'CASH'}
        if not risky:
            return None, 0
        if mode == 'full':
            return dict(tgt), 2
        if mode == 'half':
            st = _re_sizing[si]
            frac = 1.0 if st >= 1 else 0.5
            new_st = 2 if st >= 1 else 1
            new_sn = {c: v * frac for c, v in risky.items()}
            new_sn['CASH'] = max(0.0, 1.0 - sum(new_sn.values()))
            return new_sn, new_st
        if mode == 'prop':
            frac = min(3, max(1, hcount)) / 3.0
            new_sn = {c: v * frac for c, v in risky.items()}
            new_sn['CASH'] = max(0.0, 1.0 - sum(new_sn.values()))
            return new_sn, 2
        return dict(tgt), 2

'''
_run_src = _run_src.replace('    # ═══ 메인 루프 ═══',
                            _INIT_BLOCK + '    # ═══ 메인 루프 ═══', 1)

# --- 변형 3: Drift 직전에 재진입 훅 주입 ---
_REENTRY_BLOCK = r'''
        # ══ [REENTRY] 빈 스냅 트랜치 동적 재진입 (reentry_cfg 있을 때만) ══
        if reentry_cfg is not None:
            _cfg = reentry_cfg
            # empty_age / reentered 갱신
            for si in range(n_snapshots):
                _is_empty = _re_risky(snapshots[si]) <= 1e-6
                if _is_empty:
                    _re_empty_age[si] += 1
                    _re_empty_run[si] += 1
                else:
                    if _re_empty_run[si] > 0 and _restats is not None:
                        _restats.setdefault('empty_durations', []).append(_re_empty_run[si])
                    _re_empty_age[si] = 0
                    _re_empty_run[si] = 0
                # freeze 경계: 자기 앵커 봉이면 reentered/sizing 리셋
                if snap_interval_bars > 0 and canary_on and not canary_flipped:
                    if (bar_i + phase_offset_bars) % snap_interval_bars == _re_offset(si):
                        _re_reentered[si] = False
                        _re_sizing[si] = 0
            # health telemetry
            _hcfg = _cfg.get('health') or {}
            _hmode = _hcfg.get('hmode', health_mode)
            _vthr = _hcfg.get('vthr', vol_threshold)
            _ms_bars = _hcfg.get('ms_bars', 0)
            if reentry_cfg is not None and _restats is not None:
                _hc_now = len(_re_healthy_list(prev_date, health_mode, vol_threshold, 0))
                _re_hc_hist.append(_hc_now)
                _re_can_hist.append(canary_on)

            _can_ok = canary_on and not canary_flipped and crash_cooldown <= 0 and is_daily_bar
            if _can_ok:
                if _re_cooldown > 0:
                    _re_cooldown -= 1
                # 후보 트랜치 수집
                _target_range = _cfg.get('target', 'empty')
                _cad = _cfg.get('cadence', 0)
                _cands = []
                _all_empty = all(_re_risky(snapshots[j]) <= 1e-6 for j in range(n_snapshots))
                for si in range(n_snapshots):
                    _sn = snapshots[si]
                    _risky_w = _re_risky(_sn)
                    _cash_w = _sn.get('CASH', 0) + _sn.get('Cash', 0)
                    if _target_range == 'partial50':      # R1
                        _is_cand = _cash_w >= 0.5 - 1e-9
                    elif _cfg.get('sizing') == 'half':     # P1: empty OR half-filled 업그레이드
                        _is_cand = (_risky_w <= 1e-6) or (_re_sizing[si] == 1)
                    else:
                        _is_cand = _risky_w <= 1e-6
                    if not _is_cand:
                        continue
                    if _target_range == 'all_empty' and not _all_empty:  # W1
                        continue
                    if _cad > 0 and (_re_empty_age[si] % _cad != 0 or _re_empty_age[si] == 0):
                        continue
                    _cands.append(si)
                # 안정성/쿨다운 게이트
                if _cands:
                    if _cfg.get('cooldown', 0) > 0 and _re_cooldown > 0:
                        _cands = []
                if _cands:
                    _hc = len(_re_healthy_list(prev_date, _hmode, _vthr, _ms_bars))
                    if _hc < _cfg.get('min_healthy', 1):
                        _cands = []
                if _cands and _cfg.get('hstable', 0) > 0:
                    _need = _cfg['hstable']
                    _hist = _re_hc_hist[-_need:] if _restats is not None else None
                    if _hist is None or len(_hist) < _need or any(h < 1 for h in _hist):
                        _cands = []
                if _cands and _cfg.get('cstable', 0) > 0:
                    _need = _cfg['cstable']
                    _hist = _re_can_hist[-_need:] if _restats is not None else None
                    if _hist is None or len(_hist) < _need or not all(_hist):
                        _cands = []
                # C1: 잠재 target drift 게이트 (empty cur=CASH 대비 ht>=drift_threshold)
                if _cands and _cfg.get('c1_drift'):
                    _pt = _compute_weights(prev_date)
                    _ht = _half_turnover({'CASH': 1.0}, _pt)
                    if _ht < drift_threshold:
                        _cands = []
                # K 제한: empty_age desc → phase offset asc
                if _cands:
                    _cands.sort(key=lambda si: (-_re_empty_age[si], _re_offset(si)))
                    _K = _cfg.get('K', None)
                    _sel = _cands if _K is None else _cands[:_K]
                    # 재진입 target 계산 (default=엔진 _compute_weights, H=변형)
                    if _hcfg:
                        _tgt = _re_target_v(prev_date, _hmode, _vthr, _ms_bars)
                    else:
                        _tgt = _compute_weights(prev_date)
                    _tgt_risky = [c for c in _tgt if str(c).upper() != 'CASH']
                    if _tgt_risky:
                        _hcount = len(_re_healthy_list(prev_date, _hmode, _vthr, _ms_bars))
                        _fired = False
                        for si in _sel:
                            _new_sn, _new_st = _re_apply_sizing(_tgt, _cfg.get('sizing', 'full'), si, _hcount)
                            if _new_sn is None:
                                continue
                            snapshots[si] = _new_sn
                            _re_reentered[si] = True
                            _re_sizing[si] = _new_st
                            need_rebal = True
                            _fired = True
                            if _restats is not None:
                                _restats.setdefault('events', []).append({
                                    'date': date, 'si': si,
                                    'empty_age': _re_empty_age[si],
                                    'picks': list(_tgt_risky),
                                    'n_sel': len(_sel),
                                })
                        if _fired and _cfg.get('cooldown', 0) > 0:
                            _re_cooldown = _cfg['cooldown']
            # A7: 재진입 트랜치 다음 앵커 전까지 daily 재평가
            if _cfg.get('a7_reeval'):
                for si in range(n_snapshots):
                    if not _re_reentered[si]:
                        continue
                    _is_anchor = (snap_interval_bars > 0 and
                                  (bar_i + phase_offset_bars) % snap_interval_bars == _re_offset(si))
                    if _is_anchor:
                        continue
                    _nw = _compute_weights(prev_date)
                    if _nw != snapshots[si]:
                        snapshots[si] = _nw
                        need_rebal = True

'''
_run_src = _run_src.replace('        # ── Drift (daily_gate 시 일 1회만) ──',
                            _REENTRY_BLOCK + '        # ── Drift (daily_gate 시 일 1회만) ──', 1)

# --- exec 네임스페이스 (엔진 모듈 헬퍼 바인딩) ---
_ns = {
    'np': np, 'pd': pd, 'os': os,
    'get_mcap': ub.get_mcap, 'calc_sma': ub.calc_sma, 'calc_mom': ub.calc_mom,
    'calc_vol_daily': ub.calc_vol_daily, 'calc_vol_bars': ub.calc_vol_bars,
    'get_close': ub.get_close, 'get_low': ub.get_low, 'get_high': ub.get_high,
    'SLIPPAGE_MAP': ub.SLIPPAGE_MAP,
}
exec(compile(_run_src, '<reentry_harness:_run_base>', 'exec'), _ns)
_run_base = _ns['_run_base']


# ═══════════════════════════════════════════════════════════════════
# 2) V24 spot 고정 파라미터 + 변형 정의
# ═══════════════════════════════════════════════════════════════════
SPOT_KW = dict(
    interval='D', asset_type='spot', leverage=1.0,
    sma_days=42, mom_short_days=20, mom_long_days=127,
    vol_days=90, vol_threshold=0.05, canary_hyst=0.015, n_snapshots=7,
    universe_size=3, cap=1/3, tx_cost=0.004,
    health_mode='mom2vol', vol_mode='daily', drift_threshold=0.10,
    snap_interval_bars=217,
)

# 변형 레지스트리: reentry_cfg (None=F0). group 태그.
VARIANTS = {
    'F0': (None, ['G0']),
    'A1': (dict(cadence=0, K=None, min_healthy=1, sizing='full'), ['G1', 'G4']),
    'A2': (dict(cadence=0, K=1, min_healthy=1, sizing='full'), ['G1', 'G4']),
    'A3': (dict(cadence=0, K=2, min_healthy=1, sizing='full'), ['G1', 'G4']),
    'A4': (dict(cadence=5, K=1, min_healthy=1, sizing='full'), ['G1']),
    'A5': (dict(cadence=10, K=1, min_healthy=1, sizing='full'), ['G1']),
    'A6': (dict(cadence=21, K=1, min_healthy=1, sizing='full'), ['G1']),
    'A7': (dict(cadence=0, K=1, min_healthy=1, sizing='full', a7_reeval=True), ['G2']),
    'D1': (dict(cadence=31, K=1, min_healthy=1, sizing='full'), ['G1']),
    'C1': (dict(cadence=0, K=1, min_healthy=1, sizing='full', c1_drift=True), ['G1']),
    'S1': (dict(cadence=0, K=1, min_healthy=2, sizing='full'), ['G3']),
    'S2': (dict(cadence=0, K=1, min_healthy=1, sizing='full', hstable=3), ['G3']),
    'S3': (dict(cadence=0, K=1, min_healthy=1, sizing='full', cstable=3), ['G3']),
    'L1': (dict(cadence=0, K=None, min_healthy=1, sizing='full', cooldown=5), ['G4']),
    'P1': (dict(cadence=0, K=1, min_healthy=1, sizing='half'), ['G5']),
    'P2': (dict(cadence=0, K=1, min_healthy=1, sizing='prop'), ['G5']),
    'R1': (dict(cadence=0, K=1, min_healthy=1, sizing='full', target='partial50'), ['G6']),
    'W1': (dict(cadence=0, K=1, min_healthy=1, sizing='full', target='all_empty'), ['G6']),
    'H1': (dict(cadence=0, K=1, min_healthy=1, sizing='full',
               health=dict(hmode='mlvol', vthr=0.05, ms_bars=0)), ['G7']),
    'H2': (dict(cadence=0, K=1, min_healthy=1, sizing='full',
               health=dict(hmode='mom2vol', vthr=0.07, ms_bars=0)), ['G7']),
    'H3': (dict(cadence=0, K=1, min_healthy=1, sizing='full',
               health=dict(hmode='mom2vol', vthr=0.10, ms_bars=0)), ['G7']),
    'H4': (dict(cadence=0, K=1, min_healthy=1, sizing='full',
               health=dict(hmode='mom2vol', vthr=0.05, ms_bars=10)), ['G7']),
}


def run_variant(bars, funding, cfg, start, end, want_stats=True,
                want_trace=True, **overrides):
    """단일 변형 실행 → (result_dict, trace, restats)."""
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'  # 라이브 정합 (F0 baseline)
    trace = [] if want_trace else None
    restats = {} if want_stats else None
    kw = dict(SPOT_KW)
    kw.update(start_date=start, end_date=end)
    kw.update(overrides)
    res = _run_base(bars, funding, _trace=trace, reentry_cfg=cfg,
                    _restats=restats, **kw)
    return res, trace, restats


if __name__ == '__main__':
    # F0 parity 셀프체크
    START, END = '2020-10-01', '2026-05-31'
    print('데이터 로드...')
    bars, funding = ub.load_data('D')
    print('F0 (harness, reentry OFF) 실행...')
    r0, t0, _ = run_variant(bars, funding, None, START, END, want_stats=False)
    print('현행 엔진 ub.run(spot) 실행...')
    os.environ['DRIFT_HEALTH_MODE'] = 'refill'
    tr_eng = []
    kw = dict(SPOT_KW); kw.update(start_date=START, end_date=END)
    r_eng = ub.run(bars, funding, _trace=tr_eng, **kw)
    e0 = r0['_equity']; ee = r_eng['_equity']
    print(f'\n== F0 parity ==')
    print(f'harness bars={len(e0)}  engine bars={len(ee)}')
    print(f'CAGR  h={r0["CAGR"]:.6f}  e={r_eng["CAGR"]:.6f}')
    print(f'MDD   h={r0["MDD"]:.6f}  e={r_eng["MDD"]:.6f}')
    print(f'Cal   h={r0["Cal"]:.6f}  e={r_eng["Cal"]:.6f}')
    print(f'Trades h={r0["Trades"]}  e={r_eng["Trades"]}  Rebal h={r0["Rebal"]} e={r_eng["Rebal"]}')
    aligned = e0.reindex(ee.index)
    maxdiff = float((aligned - ee).abs().max())
    print(f'equity max abs diff = {maxdiff:.10e}')
    print('PARITY OK' if maxdiff < 1e-6 and len(e0) == len(ee) else 'PARITY FAIL')
